"""
The numerical backend: a cycle-accurate, bit-exact pure-Python model of a generated module.

The backend's :func:`generate` returns a :class:`NumericalModel` -- an opaque, serializable handle wrapping the
compiled kernel (it hides the LIR, which is not part of the public API). It already describes the kernel's typed I/O
ports (``inputs``/``outputs``), module name, and float format. Calling :meth:`NumericalModel.elaborate` produces a
:class:`NumericalSimulator`: the runnable, per-clock state machine. Splitting the two keeps the serializable artifact
pure data and lets the simulator be an ordinary (non-pickled) object.

:class:`NumericalSimulator` mirrors the generated ZISC RTL: it holds the same fetch PC and register files
(``regs``/``bregs``) and advances exactly one ``posedge clk`` per :meth:`NumericalSimulator.tick`, driving ``next_pc``
with the same sequencer the Verilog emits (reset / out_valid / in_ready / terminator redirect, back-pressure included).
Every operator evaluates the same ZKF bits as the hardware, so it reproduces a transaction bit-for-bit AND
cycle-for-cycle: the same inputs reach ``out_valid`` on the same cycle and present the same output bits. The persistent
slot registers simply live in ``regs``/``bregs`` and carry across transactions; :meth:`NumericalSimulator.reset`
reloads the reset snapshot.

The timing is read off the shared LIR cycle helpers, which are in the fetch-PC frame: ``operand_read_cycle(S)`` and
``landing_cycle(commit)`` are the literal ``pc`` values at which an operand is sampled and a result becomes readable in
the array. So the simulator needs no separate clock frame and no growing timeline. The only mutable state beyond the
register files is ``_pending``: the in-flight operator results -- the compact stand-in for the RTL operator pipeline. A
result is computed when its operands are sampled (at its read PC) but only becomes readable at its
landing PC; like the hardware, the register file is written at the landing, not at read time. Inputs, by contrast,
carry no latency, so :meth:`set_inputs` writes the input lanes directly. A loop re-fires the same PCs on every revisit,
and ``_pending`` only ever holds the handful of results in flight, so an arbitrarily deep loop runs in bounded memory.

The convenience :meth:`NumericalSimulator.run` drives ``tick`` over one whole transaction (inputs -> outputs); a caller
wanting the cycle count counts its own ``tick`` invocations, and a cosimulator ticks the simulator in lockstep with the
DUT.
"""

from dataclasses import dataclass
from typing import assert_never

from .._value import FloatValue
from .._lir import BoolInputLoad, FloatConstRef, FloatInputLoad, FloatOperand
from .._lir import RegRef, ScheduledOp
from .._lir import BoolRegRef, Lir
from .._lir import BoolConstRef, BoolOperand, Branch, Jump, Ret
from .._lir import install_landing, landing_cycle, operand_read_cycle
from .._lir import scalar_type_of
from .._operators import *
from .._type import FloatFormat, LogicalPort

type ModelInput = FloatValue | float | bool
type ModelOutput = FloatValue | bool
type _Dst = RegRef | BoolRegRef
type _Value = FloatValue | bool


def _coerce_input(value: ModelInput, fmt: FloatFormat, index: int) -> FloatValue:
    if isinstance(value, FloatValue):
        if value.fmt != fmt:
            raise ValueError(f"input {index} has {value.fmt}, expected {fmt}")
        return value
    if type(value) is float:
        return FloatValue.from_float(fmt, value)
    raise TypeError(f"input {index} must be FloatValue or float, got {type(value).__name__}")


def _coerce_bool_input(value: ModelInput, index: int) -> bool:
    if type(value) is bool:
        return value
    raise TypeError(f"input {index} must be bool, got {type(value).__name__}")


def _coerce_inputs(lir: Lir, inputs: tuple[ModelInput, ...]) -> tuple[list[FloatValue], list[bool]]:
    """Split the flat positional inputs into the float and boolean banks, in port order, coercing/validating each."""
    if len(inputs) != len(lir.inputs):
        raise ValueError(f"expected {len(lir.inputs)} inputs, got {len(inputs)}")
    fmt = lir.float_format
    float_values: list[FloatValue] = []
    bool_values: list[bool] = []
    for index, (load, raw_input) in enumerate(zip(lir.inputs, inputs, strict=True)):
        match load:
            case FloatInputLoad():
                float_values.append(_coerce_input(raw_input, fmt, index))
            case BoolInputLoad():
                bool_values.append(_coerce_bool_input(raw_input, index))
            case _:
                assert_never(load)
    return float_values, bool_values


def _signature(ports: list[LogicalPort]) -> str:
    return ", ".join(f"{port.name}: {port.scalar_type}" for port in ports)


@dataclass(frozen=True, slots=True)
class _OpEvent:
    """An operator firing scheduled at a read PC: the LIR op and the absolute PC it commits on (for its landings)."""

    op: ScheduledOp
    commit_pc: int


@dataclass(frozen=True, slots=True)
class _Install:
    source: FloatOperand | BoolOperand
    dst: _Dst


class _Kernel:
    """
    Read-only metadata shared by the serializable handle and the runnable simulator: both wrap one compiled kernel and
    describe its module name, float format, and typed I/O ports. ``inputs``/``outputs`` are the kernel's logical ports
    -- the parameters and return values named as the user wrote them, each with its :class:`ScalarType`.
    Subclasses set the ``_lir`` instance attribute; this base is never instantiated directly.
    """

    _lir: Lir

    @property
    def module_name(self) -> str:
        return self._lir.module_name

    @property
    def float_format(self) -> FloatFormat:
        return self._lir.float_format

    @property
    def inputs(self) -> list[LogicalPort]:
        fmt = self._lir.float_format
        return [LogicalPort(load.name, scalar_type_of(load, fmt)) for load in self._lir.inputs]

    @property
    def outputs(self) -> list[LogicalPort]:
        fmt = self._lir.float_format
        return [LogicalPort(wire.name, scalar_type_of(wire, fmt)) for wire in self._lir.outputs]

    @property
    def bit_exact(self) -> bool:
        """
        Whether this kernel's model matches the emitted RTL bit-for-bit (false iff it selects a faithfully-rounded
        operator). A driver comparing the model against the hardware uses it to pick exact vs tolerant comparison.
        """
        return self._lir.bit_exact

    def __str__(self) -> str:
        return (
            f"{type(self).__name__}({self.module_name!r}: "
            f"({_signature(self.inputs)}) -> ({_signature(self.outputs)}))"
        )


class NumericalSimulator(_Kernel):
    """
    The runnable cycle-accurate, bit-exact model of a generated module (see the module docstring). ``regs``/``bregs``
    are the live register files, ``consts`` the float constant pool, ``pc`` the fetch program counter; :meth:`tick`
    advances one clock and :attr:`output_values` reads the result while :attr:`out_valid`. The persistent state is just
    the slot registers within ``regs``/``bregs``, carried across transactions.
    Construct one from a :class:`NumericalModel` via :meth:`NumericalModel.elaborate`.
    """

    def __init__(self, lir: Lir) -> None:
        self._lir = lir
        self.consts: list[FloatValue] = [FloatValue.from_float(lir.float_format, value) for value in lir.float_consts]
        self.regs: dict[int, FloatValue] = {}  # wide register file (Verilog ``regs``)
        self.bregs: dict[int, bool] = {}  # boolean register file (Verilog ``bregs``)
        self.pc = 0
        self._pending: dict[int, list[tuple[_Dst, _Value]]] = {}  # landing PC -> in-flight (dest, value) writes
        self._op_events: dict[int, list[_OpEvent]] = {}  # read PC -> firings sampling their operands there
        self._installs: dict[int, list[_Install]] = {}  # fire PC -> pc-gated installs (readable one PC later)
        self._boundary: list[_Install] = []  # state writebacks gated to the accepted-output boundary edge
        self._terminators: dict[int, Jump | Branch] = {}  # terminator PC -> its redirecting terminator
        self._decode()
        self.reset()

    def reset(self) -> None:
        """Reload every persistent slot register with its reset snapshot and clear the in-flight state, as at rst."""
        fmt = self._lir.float_format
        self.pc = 0
        self.regs = {
            slot.reg.index: FloatValue.from_float(fmt, slot.reset_value) for slot in self._lir.float_state_slots
        }
        self.bregs = {slot.reg.index: slot.reset_value for slot in self._lir.bool_state_slots}
        self._pending = {}

    def set_inputs(self, *inputs: ModelInput) -> None:
        """
        Present the input values (in module-port order) on the input lanes. Inputs carry no latency, so they are
        written into their register lanes directly (the hardware latches them at the accept edge; nothing reads them
        before the first executing step, so writing them when presented is observationally identical).
        """
        float_values, bool_values = _coerce_inputs(self._lir, inputs)
        for float_load, float_value in zip(self._lir.float_inputs, float_values):
            self.regs[float_load.dst.index] = float_value
        for bool_load, bool_value in zip(self._lir.bool_inputs, bool_values):
            self.bregs[bool_load.dst.index] = bool_value

    def tick(self, in_valid: bool, out_ready: bool) -> None:
        """
        Advance one ``posedge clk``: compute ``next_pc`` from the current PC and the handshake (branches reading
        ``bregs``, the present/accept holds), commit the accepted-boundary state writeback (read-first), advance the
        PC, then apply that PC's datapath.
        """
        next_pc = self._next_pc(in_valid, out_ready)
        if self.pc in self._terminators:
            # A block whose terminator redirects earlier than its drained boundary (cross-block overlap) leaves
            # in-flight results still landing past its terminator PC; those landings belong to whichever arm the
            # redirect takes, so re-key the pending writes from the fall-through frame onto the taken successor's
            # frame. For a fall-through arm (and for every fully-drained block) the shift is zero -- a no-op. This
            # dynamic single-arm shift is the per-path instance of the static ``successor_local_cycle`` map that
            # ``_trace_landing`` / ``Lir.write_landing_pcs`` apply to every arm at once; that the two agree is locked by
            # ``test_spilled_result_landings_match_the_numerical_model``.
            shift = next_pc - (self.pc + 1)
            if shift:
                self._pending = {(pc + shift if pc > self.pc else pc): writes for pc, writes in self._pending.items()}
        if self.pc == self._lir.last_pc and out_ready:  # accepted boundary edge: advance persistent state (read-first)
            # The Ret wrap does not re-key _pending (it is excluded from _terminators), so any write still keyed past the
            # boundary here is orphaned -- a silently dead install. The schedule must drain every landing within last_pc.
            assert all(
                k <= self._lir.last_pc for k in self._pending
            ), f"install orphaned past last_pc {self._lir.last_pc}"
            installed = [(inst.dst, self._read(inst.source)) for inst in self._boundary]
            for dst, value in installed:
                self._write(dst, value)
        self.pc = next_pc
        self._apply(next_pc)

    def run(self, *inputs: ModelInput, max_cycles: int = 1_000_000_000) -> list[ModelOutput]:
        """
        Run one whole transaction by driving :meth:`tick`: present ``inputs``, advance to ``out_valid``, read the
        outputs, and accept them (advancing the persistent state). ``max_cycles`` bounds a non-terminating kernel; a
        caller that wants the realized cycle count drives ``tick`` itself and counts the calls.
        """
        elapsed = 0

        def step(in_valid: bool, out_ready: bool) -> None:
            nonlocal elapsed
            if elapsed >= max_cycles:
                raise RuntimeError(f"transaction did not reach out_valid within {max_cycles} cycles")
            self.tick(in_valid, out_ready)
            elapsed += 1

        while not self.in_ready:  # drain any in-flight transaction left by a partial prior drive
            step(False, True)
        self.set_inputs(*inputs)  # present inputs only once idle, so the drained transaction reads its own input lanes
        step(True, False)  # accept: pc 0 -> 1
        while not self.out_valid:
            step(False, False)
        outputs = self.output_values
        step(False, True)  # accept the output: pc -> 0, advancing the persistent state
        return outputs

    @property
    def in_ready(self) -> bool:
        """The fetch PC idles at 0 between transactions, so PC 0 is the accept-ready state."""
        return self.pc == 0

    @property
    def out_valid(self) -> bool:
        """last_pc is the Ret boundary the fetch PC reaches once the outputs are presentable in the array."""
        return self.pc == self._lir.last_pc

    @property
    def output_values(self) -> list[ModelOutput]:
        """The output values in port order, combinational from the register files (meaningful while ``out_valid``)."""
        return [self._read(wire.tap) for wire in self._lir.outputs]

    def _decode(self) -> None:
        lir = self._lir
        for block in lir.blocks:
            base = lir.block_base[block.index]
            block_ops: list[ScheduledOp] = [*block.ops, *block.inline_ops]
            for op in block_ops:
                read_pc = operand_read_cycle(op.operator, base + op.issue_cycle, lir.fetch_lag)
                self._op_events.setdefault(read_pc, []).append(_OpEvent(op, base + op.commit_cycle))
            for copy in block.copies:
                self._installs.setdefault(base + copy.fire_step(lir.fetch_lag), []).append(
                    _Install(copy.source, copy.dst)
                )
            for write in block.bool_writes:
                self._installs.setdefault(base + write.fire_step(lir.fetch_lag), []).append(
                    _Install(write.source, write.dst)
                )
            if not isinstance(block.terminator, Ret):
                self._terminators[lir.term_pc(block)] = block.terminator
        # A non-coalesced wide slot installs by a pc-gated copy -- early (before the boundary, like a phi copy) or at
        # the boundary (gated on the accepted-output edge). A boolean slot always installs at the accepted boundary
        # edge.
        for slot in lir.float_state_slots:
            if not slot.needs_copy:
                continue
            if lir.float_state_install_is_boundary(slot):
                self._boundary.append(_Install(slot.tap, slot.reg))
            else:
                self._installs.setdefault(lir.state_copy_step(slot), []).append(_Install(slot.tap, slot.reg))
        self._boundary += [_Install(slot.live_out, slot.reg) for slot in lir.bool_state_slots if slot.needs_copy]

    def _next_pc(self, in_valid: bool, out_ready: bool) -> int:
        """The RTL next-PC sequencer: hold at present/accept boundaries, otherwise redirect or advance the fetch."""
        pc = self.pc
        if pc == self._lir.last_pc:  # present: hold the result until it is taken
            return 0 if out_ready else pc
        if pc == 0:  # accept: hold until a transaction arrives
            return 1 if in_valid else 0
        terminator = self._terminators.get(pc)
        match terminator:
            case None:
                return pc + 1
            case Branch(cond=cond, if_true=if_true, if_false=if_false):
                return self._lir.block_base[if_true if self.bregs[cond.index] else if_false]
            case Jump(target=target):
                return self._lir.block_base[target]

    def _apply(self, pc: int) -> None:
        # Commit the landings due here BEFORE sampling this PC's reads/installs (read-first within the cycle).
        for dst, value in self._pending.pop(pc, ()):
            self._write(dst, value)
        for event in self._op_events.get(pc, ()):
            results = event.op.operator.evaluate(*[self._read(operand) for operand in event.op.operands])
            # every result of this firing lands at the one bank-independent cycle
            landing = landing_cycle(event.commit_pc, self._lir.fetch_lag)
            for write in event.op.writes:
                result = results[write.port]
                if isinstance(write.dst, RegRef):
                    assert isinstance(result, FloatValue) and isinstance(write.conditioner, FloatSignControl)
                    self._pending.setdefault(landing, []).append((write.dst, write.conditioner.apply_value(result)))
                else:
                    assert isinstance(result, bool) and isinstance(write.conditioner, BoolInversion)
                    self._pending.setdefault(landing, []).append((write.dst, write.conditioner.apply(result)))
        # Installs are a parallel bundle (read every source before enqueueing any destination, so an in-place
        # self-conditioned install ``b <= ~b`` and a swap are read-then-write correct) and land one PC later -- the
        # pc-gated write ``regs[dst] <= src`` at this PC is readable on the next.
        resolved = [(inst.dst, self._read(inst.source)) for inst in self._installs.get(pc, ())]
        for dst, value in resolved:
            self._pending.setdefault(install_landing(pc), []).append((dst, value))

    def _read(self, operand: FloatOperand | BoolOperand) -> _Value:
        if isinstance(operand, FloatOperand):
            float_source = operand.source
            base = (
                self.consts[float_source.index]
                if isinstance(float_source, FloatConstRef)
                else self.regs[float_source.index]
            )
            return operand.sign.apply_value(base)
        bool_source = operand.source
        if isinstance(bool_source, BoolConstRef):
            return operand.inversion.apply(bool_source.value)
        return operand.inversion.apply(self.bregs[bool_source.index])

    def _write(self, dst: _Dst, value: _Value) -> None:
        if isinstance(dst, RegRef):
            assert isinstance(value, FloatValue)
            self.regs[dst.index] = value
        else:
            assert isinstance(value, bool)
            self.bregs[dst.index] = value


class NumericalModel(_Kernel):
    """
    An opaque, serializable handle to a compiled kernel -- the artifact :func:`generate` returns.
    It is picklable, so a generated testbench can embed it.
    :meth:`elaborate` builds a fresh :class:`NumericalSimulator` to actually run it.
    """

    def __init__(self, lir: Lir) -> None:
        self._lir = lir

    def elaborate(self) -> NumericalSimulator:
        return NumericalSimulator(self._lir)


def generate(lir: Lir) -> NumericalModel:
    return NumericalModel(lir)
