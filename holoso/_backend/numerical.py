"""
The numerical backend: a pure-Python functional model of a generated module, bit-exact to its RTL.

``NumericalModel`` is a software interpreter for the scheduled LIR microprogram. It replays the schedule applying the
same per-operator ZKF rounding the hardware does, so one ``__call__`` reproduces one module transaction bit-for-bit:
flat input values in port order to a flat tuple of output values in port order. A stateful module additionally carries
its persistent state registers between calls, exactly as the hardware retains them between initiations; ``reset()``
reloads the snapshot the hardware loads at module reset.
"""

from dataclasses import dataclass, field

from .._value import FloatValue
from .._lir import BoolInputLoad, BoolOutputWire, FloatConstRef, FloatInputLoad, FloatOperand
from .._lir import FloatOutputWire, InlineScheduledOp, PooledScheduledOp, RegRef, ScheduledOp
from .._lir import Lir
from .._lir import BoolConstRef, BoolOperand, Branch, Jump, Ret
from .._lir import operand_read_cycle, result_landing_cycle
from .._operators import *
from .._type import FloatFormat

type ModelInput = FloatValue | float | bool
type ModelOutput = FloatValue | bool


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


def _resident[T](timeline: list[tuple[int, T]], read_cycle: int) -> T:
    """
    The value a register holds when read at hardware-frame ``read_cycle``: the latest write whose landing cycle does
    not exceed it. The timeline is seeded with the value carried in from the previous block (landing 0) and extended in
    landing order as the block's operators commit, so a register reused for several values yields the live one.
    """
    assert timeline[0][0] <= read_cycle, "a register is read before its first value lands"
    value = timeline[0][1]
    for landing, written in timeline:
        if landing <= read_cycle:
            value = written
        else:
            break
    return value


@dataclass(slots=True)
class NumericalModel:
    """
    A bit-exact functional model of a generated module: one ``__call__`` is one module transaction.

    Call it with the input values in module-port order; it returns the output values in module-port order. The result
    matches the generated Verilog bit-for-bit because every operator evaluates the same ZKF bits as the hardware. A
    stateful module carries its persistent state registers between calls (``reset()`` reloads the reset snapshot),
    mirroring how the hardware retains state across initiations and reloads it at reset. The model holds the scheduled
    ``Lir`` and is picklable, so a generated testbench can embed it. For a control-flow kernel one ``__call__`` follows
    the taken path block by block, so the result is correct for any branch outcome.

    TODO: the model reproduces output bits exactly but not yet the per-transaction cycle latency (which is data-
          dependent once branches/loops shortcut the PC). Predicting that latency would enable cycle-accurate
          lockstep cosimulation; today the bench checks output bits at the out_valid handshake instead.
    """

    lir: Lir
    _state: list[FloatValue] = field(init=False)
    _bool_state: list[bool] = field(init=False)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reload every persistent state register with its reset snapshot, as the hardware does at module reset."""
        fmt = self.lir.float_format
        self._state = [FloatValue.from_float(fmt, slot.reset_value) for slot in self.lir.float_state_slots]
        self._bool_state = [slot.reset_value for slot in self.lir.bool_state_slots]

    def __call__(self, *inputs: ModelInput) -> tuple[ModelOutput, ...]:
        lir = self.lir
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
        return self._run_cfg(float_values, bool_values)

    def _run_cfg(self, float_values: list[FloatValue], bool_values: list[bool]) -> tuple[ModelOutput, ...]:
        """
        Execute a control-flow microprogram by following the taken path block by block. Each block evaluates its
        scheduled float ops (in commit order), installs its phi-arm copies and boolean writes, then its terminator
        selects the successor; at Ret the outputs are read and the persistent state is taken from the slot registers.
        """
        lir = self.lir
        fmt = lir.float_format
        consts = [FloatValue.from_float(fmt, const_value) for const_value in lir.float_consts]
        regs: dict[int, FloatValue] = {
            load.dst.index: input_value for load, input_value in zip(lir.float_inputs, float_values)
        }
        for fslot, state_value in zip(lir.float_state_slots, self._state):
            regs[fslot.reg.index] = state_value
        bregs: dict[int, bool] = {
            load.dst.index: input_value for load, input_value in zip(lir.bool_inputs, bool_values)
        }
        for bslot, bool_state_value in zip(lir.bool_state_slots, self._bool_state):
            bregs[bslot.reg.index] = bool_state_value

        def fval(operand: FloatOperand) -> FloatValue:
            source = operand.source
            base = consts[source.index] if isinstance(source, FloatConstRef) else regs[source.index]
            return operand.sign.apply_value(base)

        def bval(operand: BoolOperand) -> bool:
            source = operand.source
            return source.value if isinstance(source, BoolConstRef) else bregs[source.index]

        blocks = {block.index: block for block in lir.blocks}
        current = lir.entry
        for _ in range(1_000_000):  # bounded against a non-terminating kernel; ample for any real loop trip count
            block = blocks[current]
            # Per-register write timeline within this block, in the executing-step (hardware) frame: seeded with the
            # value carried in from the previous block (landing 0) and extended as each operator commits. An operand is
            # resolved at its read cycle, so a register reused for several values within the block yields the one live
            # at the read -- the same read-first resolution the straight-line path applies globally via write_timeline.
            ftl: dict[int, list[tuple[int, FloatValue]]] = {reg: [(0, value)] for reg, value in regs.items()}
            btl: dict[int, list[tuple[int, bool]]] = {reg: [(0, value)] for reg, value in bregs.items()}

            def fread(operand: FloatOperand, read_cycle: int) -> FloatValue:
                source = operand.source
                base = (
                    consts[source.index]
                    if isinstance(source, FloatConstRef)
                    else _resident(ftl[source.index], read_cycle)
                )
                return operand.sign.apply_value(base)

            def bread(operand: BoolOperand, read_cycle: int) -> bool:
                source = operand.source
                return source.value if isinstance(source, BoolConstRef) else _resident(btl[source.index], read_cycle)

            # All operations -- pooled firings and inline ops across both banks -- evaluate in a single
            # commit-sorted pass, so a producer is computed before any consumer (dependency edges guarantee strictly
            # increasing commits along every dependence); the read-cycle resolution above makes the reads
            # register-reuse-correct even when a consumer commits after a later writer of one of its operands.
            scheduled: list[ScheduledOp] = [*block.ops, *block.inline_ops]
            scheduled.sort(
                key=lambda o: (o.commit_cycle, 0 if isinstance(o, PooledScheduledOp) else 1, o.writes[0].dst.index)
            )
            for op in scheduled:
                read = operand_read_cycle(op.operator, op.issue_cycle)
                values = [
                    fread(operand, read) if isinstance(operand, FloatOperand) else bread(operand, read)
                    for operand in op.operands
                ]
                results = op.operator.evaluate(*values)
                # Every tapped output port lands per its destination bank: the wide bank pays the writeback latch,
                # the latch-free boolean bank only the read-first edge -- the same rule the RTL write enables
                # implement. Landings stay strictly increasing per register (interference forbids same-landing
                # sharing), which _resident's timeline resolution relies on.
                for write in op.writes:
                    land = result_landing_cycle(write.dst, op.commit_cycle)
                    result = results[write.port]
                    if isinstance(write.dst, RegRef):
                        assert isinstance(result, FloatValue) and isinstance(write.conditioner, FloatSignControl)
                        timeline = ftl.setdefault(write.dst.index, [])
                        assert not timeline or timeline[-1][0] < land, "wide landings must be strictly increasing"
                        timeline.append((land, write.conditioner.apply_value(result)))
                    else:
                        assert isinstance(result, bool) and isinstance(write.conditioner, BoolInversion)
                        bool_timeline = btl.setdefault(write.dst.index, [])
                        assert (
                            not bool_timeline or bool_timeline[-1][0] < land
                        ), "boolean landings must be strictly increasing"
                        bool_timeline.append((land, write.conditioner.apply(result)))
            # The values resident at the block boundary: the last write to each register (or the carried-in value). The
            # phi-arm copies, the terminator, and the Ret reads all fire at the drained tail, so they read these.
            regs = {reg: timeline[-1][1] for reg, timeline in ftl.items()}
            bregs = {reg: timeline[-1][1] for reg, timeline in btl.items()}
            # Phi-arm installs are a parallel copy bundle (a swap reads both old values), so read every source before
            # writing any destination -- both for the float copies and the boolean writes.
            copied = [fval(copy.source) for copy in block.copies]
            for copy, copy_value in zip(block.copies, copied):
                regs[copy.dst.index] = copy_value
            written = [bval(bool_write.source) for bool_write in block.bool_writes]
            for bool_write, write_value in zip(block.bool_writes, written):
                bregs[bool_write.dst.index] = write_value
            match block.terminator:
                case Jump(target=target):
                    current = target
                case Branch(cond=cond, if_true=if_true, if_false=if_false):
                    current = if_true if bregs[cond.index] else if_false
                case Ret():
                    # Read-first at the boundary: the slot register is read-only in the body, so an output (or any read
                    # of a slot's live-in, e.g. ``return old``) sees the OLD slot value here; the new persistent state
                    # is each slot's live-out -- a distinct value (a phi/operator/input/const), tapped with its sign.
                    def out_value(wire: FloatOutputWire | BoolOutputWire) -> ModelOutput:
                        match wire:
                            case FloatOutputWire():
                                return fval(wire.tap)
                            case BoolOutputWire():
                                return bval(wire.tap)

                    outputs = tuple(out_value(wire) for wire in lir.outputs)
                    # The slot's live-out is sampled on its install step (block-relative), not at the boundary: an
                    # early-installed slot's source register may be reused later in the Ret block, so reading it at the
                    # boundary would see the tenant. For a boundary install this step is the boundary, so it degenerates.
                    base = lir.block_base[block.index]
                    self._state = [fread(slot.tap, lir.state_copy_step(slot) - base) for slot in lir.float_state_slots]
                    self._bool_state = [bval(slot.live_out) for slot in lir.bool_state_slots]
                    return outputs
        raise RuntimeError("control flow did not reach a return; the CFG may not terminate")

    @property
    def input_names(self) -> tuple[str, ...]:
        return tuple(load.name for load in self.lir.inputs)

    @property
    def output_names(self) -> tuple[str, ...]:
        return tuple(wire.name for wire in self.lir.outputs)

    @property
    def float_format(self) -> FloatFormat:
        return self.lir.float_format

    def __str__(self) -> str:
        return (
            f"{type(self).__name__}({self.lir.module_name!r},"
            f" inputs={self.input_names}, outputs={self.output_names},"
            f" float_format={self.float_format})"
        )


def generate(lir: Lir) -> NumericalModel:
    """Build the bit-exact functional model from a finished :class:`Lir`."""
    return NumericalModel(lir)
