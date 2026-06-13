"""
The low-level IR (LIR): the scheduled, bound, register-allocated microprogram for the synthesized ZISC machine.

A :class:`Lir` is controller-agnostic -- it describes which hardware operators issue on which cycle, reading/writing
which typed storage resources, with which folded sign controls.
"""

from dataclasses import dataclass

from .._operators import (
    BoolInversion,
    FloatSignControl,
    HardwareOperator,
    InlineHardwareOperator,
    PooledHardwareOperator,
    PortConditioner,
)
from .._type import BoolType, FloatFormat, FloatType
from ._ports import ControlInputPort, ControlOutputPort, ControlPort, DataInputPort, DataOutputPort, Port

FETCH_STAGES = 3
FETCH_LAG = FETCH_STAGES - 1


# Executing-step (hardware) frame cycle offsets: the single source of truth shared by the LIR cycle helpers below, the
# write timeline, the numerical model, the scheduler's dependency edges, and the register allocator, so a value's
# landing/read/copy/boundary cycle is computed in exactly one place and the consumers cannot drift (the bug this
# centralization prevents). Timing is a property of the REGISTER BANK and of the operation class (instance-backed vs
# inline), never of an individual operator: the wide bank reads through a read latch and writes through a writeback
# latch; the boolean bank is latch-free (results are written directly at their commit step and read directly).
def wide_landing_cycle(commit_cycle: int) -> int:
    """The cycle a wide result is readable in the array: its commit plus the write latch and the read-first edge."""
    return commit_cycle + FETCH_LAG + 2


def bool_landing_cycle(commit_cycle: int) -> int:
    """The cycle a boolean result becomes readable: written directly at its commit step, visible the next step."""
    return commit_cycle + FETCH_LAG + 1


def pooled_wide_read_cycle(issue_cycle: int) -> int:
    """The cycle an instance-backed operator samples a wide operand -- the read latch presents the address early."""
    return issue_cycle + FETCH_LAG - 1


def pooled_bool_read_cycle(issue_cycle: int) -> int:
    """The cycle an instance-backed operator samples a boolean operand: latch-free, directly on its in_valid step."""
    return issue_cycle + FETCH_LAG


def inline_fire_cycle(commit_cycle: int, dst_is_wide: bool) -> int:
    """
    The cycle an inline combinational operation (boolean logic, a float<->bool cast) fires: it is one PC-gated
    statement that reads ALL its operands and writes its destination on this single step -- the commit step for a
    boolean destination, one later for a wide one (aligned with the wide bank's write-latch write enables).
    """
    return commit_cycle + FETCH_LAG + (1 if dst_is_wide else 0)


def operand_read_cycle(operator: HardwareOperator, issue_cycle: int) -> int:
    """
    The hardware-frame cycle on which an operation samples its register operands (an operation reads all its operands
    on one cycle), the single definition shared by the register allocator's interference, the liveness views, and the
    numerical model so none can drift. A pooled instance reads through the wide read latch; no pooled operator reads
    a boolean operand yet, ENFORCED here -- when one appears it is presented latch-free on the in_valid step
    (``pooled_bool_read_cycle``) and this dispatch must grow per-operand granularity, reconciled with
    ``dependency_edge``. An inline operation fires -- and reads -- on its writeback step.
    """
    if isinstance(operator, PooledHardwareOperator):
        assert all(isinstance(ty, FloatType) for ty in operator.signature.operand_types), operator.mnemonic
        return pooled_wide_read_cycle(issue_cycle)
    dst_is_wide = isinstance(operator.signature.result_types[0], FloatType)
    return inline_fire_cycle(issue_cycle + operator.latency, dst_is_wide)


def dependency_edge(producer: HardwareOperator, producer_port: int, consumer: HardwareOperator) -> int:
    """
    The minimum same-block scheduling distance from a producer's commit to a consumer's issue (``issue_consumer >=
    commit_producer + edge``), derived from the landing of the producer's tapped output port's bank and the
    consumer's operand-read timing so the scheduler, the liveness views, and the model share one rule. The clamp
    keeps an inline consumer's commit strictly after its producer's, which the model's commit-ordered evaluation
    relies on. The zero-offset evaluation below is exact because every cycle helper is affine in its cycle argument
    with unit slope, so the difference at zero is the frame-independent spacing; a helper that ever loses that
    affinity breaks this derivation. No pooled operator reads a boolean operand yet, ENFORCED here in lockstep with
    ``operand_read_cycle`` (which charges every pooled consumer the wide read latch): the first bool-reading pooled
    operator must reconcile its presentation -- latch-free on the in_valid step, ``pooled_bool_read_cycle`` -- in
    both helpers at once.
    """
    producer_wide = isinstance(producer.signature.result_types[producer_port], FloatType)
    landing = wide_landing_cycle(0) if producer_wide else bool_landing_cycle(0)
    if isinstance(consumer, PooledHardwareOperator):
        assert producer_wide, f"{consumer.mnemonic}: pooled operators read only wide operands today"
        read = pooled_wide_read_cycle(0)
    else:
        dst_is_wide = isinstance(consumer.signature.result_types[0], FloatType)
        read = inline_fire_cycle(consumer.latency, dst_is_wide)
    return max(landing - read, 1 - consumer.latency)


def copy_step_cycle(install_cycle: int) -> int:
    """The step a non-coalesced slot writeback fires and samples its source."""
    return install_cycle + FETCH_LAG + 1


def boundary_step(makespan: int) -> int:
    """The boundary / initiation-interval step: the last result lands here and outputs are resident here."""
    return makespan + 2 + FETCH_LAG


def residence_rows(defs: list[int], uses: list[int], present: int) -> set[int]:
    """
    Collapse a register's definition and use cycles into the set of cycles on which it holds a live value, by the
    read-first rule: each value resides from its landing through its last use no later than the next definition (a read
    on the next definition's cycle is read-first and still returns the old value), the boundary at latest. Shared by the
    float- and boolean-bank liveness so both banks compute residence in exactly one place.
    """
    writes = sorted(defs)
    reads = sorted(uses)
    rows: set[int] = set()
    for i, start in enumerate(writes):
        nxt = writes[i + 1] if i + 1 < len(writes) else present + 1
        last = max((use for use in reads if start <= use <= nxt), default=start)
        rows.update(range(start, last + 1))
    return rows


@dataclass(frozen=True, slots=True)
class OperatorInstance:
    """
    One physical operator module, e.g. ``u_fadd_326215ea_0`` or ``u_fcmp_7296114c_0``.

    ``operator`` is the fully specified pooled hardware operator it elaborates; ``index`` numbers the copies of that
    operator value. The scheduler pools firings by the hardware-operator instance: equal operators may time-share one
    module, each instance accepting a new firing every ``initiation_interval`` cycles.
    """

    operator: PooledHardwareOperator
    index: int  # 0-based within this concrete operator value

    def __post_init__(self) -> None:
        # Every pooled operator passes through here, so its three hand-synchronized per-port declarations are
        # validated once at the source: HDL port names align with the result types, and the commutation permutation
        # (when declared) is a type-preserving bijection -- a bad future declaration fails here, not in emission.
        result_types = self.operator.signature.result_types
        assert len(self.operator.output_hdl_ports) == len(result_types), self.operator.mnemonic
        permutation = self.operator.swap_output_permutation
        if permutation is not None:
            assert sorted(permutation) == list(range(len(result_types))), self.operator.mnemonic
            assert all(result_types[permutation[p]] == result_types[p] for p in range(len(permutation)))
        # Busy windows are tracked per block, so cross-block soundness needs the instance provably idle by the time
        # any successor block can issue on it. The worst case is a firing committing exactly at its block's makespan
        # (issue = makespan - latency): the boundary redirect fires ``boundary_step(0)`` steps past the makespan, the
        # successor's base begins one step later, and its first issue one step after that -- an issue-to-issue gap of
        # exactly ``latency + boundary_step(0) + 2``. A deeper-throttled operator needs cross-block busy tracking.
        assert self.operator.initiation_interval <= self.operator.latency + boundary_step(0) + 2, (
            f"{self.operator.mnemonic}: initiation_interval {self.operator.initiation_interval} needs cross-block "
            f"busy tracking (max supported is latency + {boundary_step(0) + 2})"
        )


@dataclass(frozen=True, slots=True)
class RegRef:
    """A read/write of wide data register ``index`` in the shared register bank."""

    index: int

    @property
    def stable_label(self) -> str:
        return f"r{self.index}"

    @property
    def is_register(self) -> bool:
        return True


@dataclass(frozen=True, slots=True)
class BoolRegRef:
    """A read/write of boolean register ``index`` in the 1-bit boolean register bank."""

    index: int

    @property
    def stable_label(self) -> str:
        return f"b{self.index}"

    @property
    def is_register(self) -> bool:
        return True


def result_landing_cycle(dst: RegRef | BoolRegRef, commit_cycle: int) -> int:
    """
    The cycle a result lands per its destination bank -- the single dispatch every consumer (liveness, the write
    timeline, the numerical model, the report) routes through, so the per-bank rule cannot drift between them.
    """
    return wide_landing_cycle(commit_cycle) if isinstance(dst, RegRef) else bool_landing_cycle(commit_cycle)


@dataclass(frozen=True, slots=True)
class ConstRef:
    """An immediate constant, ``index`` into one typed LIR constant pool."""

    index: int

    @property
    def stable_label(self) -> str:
        return f"c{self.index}"

    @property
    def is_register(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class FloatConstRef(ConstRef):
    """An immediate floating-point constant, ``index`` into the LIR float constant pool."""


@dataclass(frozen=True, slots=True)
class Operand:
    """An operator input: a register read or a constant immediate."""

    source: RegRef | ConstRef


@dataclass(frozen=True, slots=True)
class FloatOperand(Operand):
    """A float operator input: a wide-register read or float constant immediate, with folded sign control."""

    source: RegRef | FloatConstRef
    sign: FloatSignControl = FloatSignControl()

    @property
    def stable_label(self) -> str:
        return self.sign.decorate(self.source.stable_label)


@dataclass(frozen=True, slots=True)
class InputLoad:
    """An input port sampled into a typed register at in_valid."""

    name: str
    dst: RegRef | BoolRegRef


@dataclass(frozen=True, slots=True)
class FloatInputLoad(InputLoad):
    """A float input port sampled into a wide register at in_valid."""

    dst: RegRef


@dataclass(frozen=True, slots=True)
class FloatStateSlot:
    """
    A persistent float state register: reset to ``reset_value``, holding the slot's live-in (carried over from the
    previous initiation) until it is overwritten, and holding the slot's live-out from ``install_cycle`` onward.

    ``tap`` is the live-out's source tap (register/constant + folded sign), the same primitive an output wire taps; here
    the sink is the slot register rather than a port. When the tap is exactly ``reg`` with an identity sign the live-out
    coalesced onto the slot register (its producing operator wrote it) and the backend emits no copy; otherwise the
    backend latches the tap into ``reg`` at ``install_cycle``: as early as the old live-in is read and the source is
    available, the initiation boundary at the latest. Installing before the boundary lets the source register be reused
    by unrelated operations for the rest of the initiation. A public attribute's observable ``state_<name>`` port is a
    separate output wire tapping the same value, not a property of the slot.
    """

    name: str
    reg: RegRef
    reset_value: float
    tap: FloatOperand
    install_cycle: int  # scheduler-frame cycle the live-out lands in reg (its max, makespan + 1, is the boundary)

    @property
    def needs_copy(self) -> bool:
        return not (self.tap.source == self.reg and self.tap.sign == FloatSignControl())


@dataclass(frozen=True, slots=True)
class BoolInputLoad(InputLoad):
    """A boolean input port sampled into a boolean register at in_valid."""

    dst: BoolRegRef


@dataclass(frozen=True, slots=True)
class BoolConstRef:
    """A boolean immediate (``True``/``False``); the bool bank has no constant pool, the value rides inline."""

    value: bool


type BoolSource = BoolRegRef | BoolConstRef


@dataclass(frozen=True, slots=True)
class BoolOperand:
    """
    A boolean operand: a boolean register read or an immediate True/False, with an optional folded inversion -- the
    1-bit dual of :class:`FloatOperand`'s sign control, free in fabric. An inverted immediate folds to its negated
    value at construction, so a constant operand always carries the identity inversion.
    """

    source: BoolSource
    inversion: BoolInversion = BoolInversion()

    def __post_init__(self) -> None:
        if isinstance(self.source, BoolConstRef) and self.inversion.invert:
            object.__setattr__(self, "source", BoolConstRef(not self.source.value))
            object.__setattr__(self, "inversion", BoolInversion())

    @property
    def stable_label(self) -> str:
        if isinstance(self.source, BoolConstRef):
            return "1" if self.source.value else "0"
        return self.inversion.decorate(self.source.stable_label)


@dataclass(frozen=True, slots=True)
class PortWrite:
    """
    One tapped output port of a firing: the ``port``-th result lands in ``dst`` through its type's conditioner (a
    folded sign control on a wide destination, an optional inversion on a boolean one). Untapped ports of the firing
    simply have no PortWrite -- the module output is left unconnected.
    """

    port: int
    dst: RegRef | BoolRegRef
    conditioner: PortConditioner


@dataclass(frozen=True, slots=True)
class PooledScheduledOp:
    """
    One pooled-instance firing in the software-pipelined schedule: ``inst`` asserts ``in_valid`` on ``issue_cycle``
    (its operands sampled per the bank read discipline), and on ``commit_cycle == issue_cycle + latency`` every
    tapped output port lands in its destination register. The writes are sorted by port and pairwise distinct in
    both port and destination -- members of one firing land simultaneously, so the allocator always gives them
    distinct registers.
    """

    inst: OperatorInstance
    operands: list[FloatOperand | BoolOperand]
    writes: list[PortWrite]
    issue_cycle: int
    latency: int

    @property
    def operator(self) -> PooledHardwareOperator:
        return self.inst.operator

    def __post_init__(self) -> None:
        assert self.writes, "a firing with no tapped output cannot exist (an unused operation has no MIR node)"
        ports = [write.port for write in self.writes]
        assert ports == sorted(set(ports)), f"write ports must be sorted and distinct: {ports}"
        assert len({write.dst for write in self.writes}) == len(self.writes), "write destinations must be distinct"

    @property
    def commit_cycle(self) -> int:
        return self.issue_cycle + self.latency


@dataclass(frozen=True, slots=True)
class InlineScheduledOp:
    """
    One inline-operator firing: a single PC-gated statement that reads its operands and writes its one result on its
    fire step (the commit step for a boolean destination, one later for a wide one).
    """

    operator: InlineHardwareOperator
    operands: list[FloatOperand | BoolOperand]
    write: PortWrite
    issue_cycle: int
    latency: int

    @property
    def writes(self) -> list[PortWrite]:
        return [self.write]

    @property
    def commit_cycle(self) -> int:
        return self.issue_cycle + self.latency


@dataclass(frozen=True, slots=True)
class OutputWire:
    """An output port: a named external sink driven at the present step by a typed source tap."""

    name: str
    tap: FloatOperand | BoolOperand


@dataclass(frozen=True, slots=True)
class FloatOutputWire(OutputWire):
    """A float output port: the named external sink for a float source tap (register/constant + folded sign)."""

    tap: FloatOperand


@dataclass(frozen=True, slots=True)
class BoolOutputWire(OutputWire):
    """A boolean output port: the named external sink for a boolean register or immediate, with a folded inversion."""

    tap: BoolOperand


@dataclass(frozen=True, slots=True)
class FloatCopy:
    """
    A register-to-register move installing a phi arm's value into the merged register at a predecessor's tail: ``dst``
    takes ``source`` on the block-relative ``issue_cycle``. Used when a phi arm is not an operator result that can be
    coalesced directly onto the merged register (e.g. an input, a constant, or a value defined in another block).
    """

    dst: RegRef
    source: FloatOperand
    issue_cycle: int


@dataclass(frozen=True, slots=True)
class BoolWrite:
    """
    A boolean register install of a phi arm (a bool const or another bool register, with the arm's folded inversion)
    on a block-relative cycle.
    """

    dst: BoolRegRef
    source: BoolOperand
    issue_cycle: int


@dataclass(frozen=True, slots=True)
class Jump:
    """Unconditional control transfer to block ``target``."""

    target: int


@dataclass(frozen=True, slots=True)
class Branch:
    """Conditional control transfer on boolean register ``cond``."""

    cond: BoolRegRef
    if_true: int
    if_false: int


@dataclass(frozen=True, slots=True)
class Ret:
    """The sole function exit: outputs and persistent state are resident at the block boundary."""


type Terminator = Jump | Branch | Ret


@dataclass(frozen=True, slots=True)
class LirBlock:
    """
    One basic block of the scheduled microprogram, with block-relative cycles (block start is cycle 0). ``ops``
    (pooled firings), ``inline_ops``, ``copies``, and ``bool_writes`` are the block's datapath events; ``terminator``
    redirects the fetch PC at the block boundary. ``block_makespan`` is the last commit cycle inside the block (0 if
    it has none).
    """

    index: int
    ops: list[PooledScheduledOp]
    inline_ops: list[InlineScheduledOp]
    copies: list[FloatCopy]
    bool_writes: list[BoolWrite]
    terminator: Terminator
    block_makespan: int


@dataclass(frozen=True, slots=True)
class BoolStateSlot:
    """
    A persistent boolean state register: reset to ``reset_value``, holding the slot's live-in throughout the
    transaction and installing its live-out (``live_out``, a boolean register or constant with a folded inversion)
    at the boundary, read-first
    -- so an output or branch that still reads the live-in sees the old value, exactly like a float slot.
    """

    name: str
    reg: BoolRegRef
    reset_value: bool
    live_out: BoolOperand

    @property
    def needs_copy(self) -> bool:
        """
        False only when the live-out already resides in the slot register UNINVERTED (an unwritten slot); a live-out
        under an inversion needs the install copy to apply it, even from the slot's own register.
        """
        return not (
            isinstance(self.live_out.source, BoolRegRef)
            and self.live_out.source == self.reg
            and not self.live_out.inversion.invert
        )


@dataclass(frozen=True, slots=True)
class RegFileLayout:
    """The shared wide data register file resource."""

    width: int
    nreg: int
    nrd: int
    nwr: int
    nload: int


@dataclass(frozen=True, slots=True)
class BoolRegFileLayout:
    """The boolean register bank: ``nreg`` 1-bit registers (branch conditions and boolean state)."""

    nreg: int


@dataclass(frozen=True, slots=True)
class InputProducer:
    """A write to a register that came from an input-load lane ``index`` (in module-port order)."""

    index: int


@dataclass(frozen=True, slots=True)
class OperationProducer:
    """A write to a register that came from operation ``index`` in ``Lir.ops``."""

    index: int


@dataclass(frozen=True, slots=True)
class StateProducer:
    """A state register's live-in: the value it carries over from the previous initiation (or the reset snapshot)."""

    index: int  # index into Lir.float_state_slots


@dataclass(frozen=True, slots=True)
class InlineProducer:
    """A wide-register write that came from an inline firing: ``Lir.blocks[block].inline_ops[index]``."""

    block: int
    index: int


type Producer = InputProducer | OperationProducer | StateProducer | InlineProducer

# The common surface of the two firing classes (operator/operands/writes/issue/commit), as the model consumes it.
type ScheduledOp = PooledScheduledOp | InlineScheduledOp


@dataclass(frozen=True, slots=True)
class Lir:
    module_name: str
    instances: list[OperatorInstance]
    float_consts: list[float]  # constant pool: index -> value
    float_format: FloatFormat
    regfile: RegFileLayout
    inputs: list[FloatInputLoad | BoolInputLoad]  # ordered as the function parameters
    ops: list[PooledScheduledOp]  # the pipelined pooled firings, flattened across blocks with ABSOLUTE issue cycles
    outputs: list[FloatOutputWire | BoolOutputWire]
    float_state_slots: list[FloatStateSlot]  # persistent registers, ordered as the instance attributes
    # Control-flow overlay. A straight-line kernel has a single block ending in Ret; ``blocks[0]`` is the entry,
    # ``block_base[i]`` is block i's absolute start PC, and ``last_pc`` is the out_valid boundary (the single Ret).
    blocks: list[LirBlock]
    block_base: list[int]
    entry: int
    last_pc: int  # LASTPC: the fetch PC at which out_valid asserts (the single Ret block's boundary)
    min_initiation_interval: int  # shortest executable path latency; exact for branch-free kernels, else a lower bound
    bool_regfile: BoolRegFileLayout
    bool_state_slots: list[BoolStateSlot]  # persistent boolean registers (branch conditions, boolean attributes)

    def __post_init__(self) -> None:
        assert self.regfile.width == self.float_format.width

    @property
    def ports(self) -> list[Port]:
        float_type = FloatType(self.float_format)
        bool_type = BoolType()
        ports: list[Port] = [
            ControlInputPort("clk", 1),
            ControlInputPort("rst", 1),
            ControlInputPort("in_valid", 1),
            ControlOutputPort("in_ready", 1),
            ControlOutputPort("out_valid", 1),
            ControlInputPort("out_ready", 1),
        ]
        for load in self.inputs:
            match load:
                case FloatInputLoad():
                    ports.append(DataInputPort(f"in_{load.name}", float_type))
                case BoolInputLoad():
                    ports.append(DataInputPort(f"in_{load.name}", bool_type))
        for wire in self.outputs:
            match wire:
                case FloatOutputWire():
                    ports.append(DataOutputPort(wire.name, float_type))
                case BoolOutputWire():
                    ports.append(DataOutputPort(wire.name, bool_type))
        ports.append(ControlOutputPort("err_pc", self.cyc_width))
        return ports

    @property
    def float_inputs(self) -> list[FloatInputLoad]:
        return [load for load in self.inputs if isinstance(load, FloatInputLoad)]

    @property
    def bool_inputs(self) -> list[BoolInputLoad]:
        return [load for load in self.inputs if isinstance(load, BoolInputLoad)]

    @property
    def float_outputs(self) -> list[FloatOutputWire]:
        return [wire for wire in self.outputs if isinstance(wire, FloatOutputWire)]

    @property
    def bool_outputs(self) -> list[BoolOutputWire]:
        return [wire for wire in self.outputs if isinstance(wire, BoolOutputWire)]

    @property
    def input_ports(self) -> list[DataInputPort]:
        return [port for port in self.ports if isinstance(port, DataInputPort)]

    @property
    def output_ports(self) -> list[DataOutputPort]:
        return [port for port in self.ports if isinstance(port, DataOutputPort)]

    @property
    def control_ports(self) -> list[ControlPort]:
        return [port for port in self.ports if isinstance(port, ControlPort)]

    @property
    def present_step(self) -> int:
        """
        The hardware executing step on which the outputs are valid in the register array: the fetch PC reaches
        ``last_pc`` (the Ret boundary) and the executing step lags it by FETCH_LAG. For a straight-line kernel this is
        ``makespan + 2`` (the last commit plus the write latch); for a CFG it is the Ret block's resident step.
        """
        return self.last_pc - FETCH_LAG

    @property
    def cyc_width(self) -> int:
        """Bit width of the err_pc diagnostic: enough to hold any executing step ``0..present_step``."""
        return max(1, self.present_step.bit_length())

    @property
    def initiation_interval(self) -> int:
        """
        The out_valid boundary PC (``last_pc``). For a straight-line kernel this equals the observable
        in_valid->out_valid latency; with branches the per-path latency varies and is reported by the numerical model,
        while ``min_initiation_interval`` is the statically-known lower bound (exact when branch-free).
        """
        return self.last_pc

    def state_copy_step(self, slot: FloatStateSlot) -> int:
        """
        The fetch-PC value -- equivalently the hardware-frame cycle -- on which a non-coalesced slot's writeback copy
        fires. For a boundary install this is ``initiation_interval`` (LASTPC), where it reduces to the accepted-
        transaction edge. The copy reads its source and lands the new live-out in the slot register on this same step;
        shared by liveness and the emitter so the two cannot drift.
        """
        return copy_step_cycle(slot.install_cycle)

    @property
    def read_set_per_port(self) -> dict[tuple[OperatorInstance, int], list[int]]:
        """
        For each operator read port -- identified by its ``(instance, operand-position)`` pair -- the sorted distinct
        register indices it ever reads across the schedule.

        Constant operands are excluded: they are immediates on the per-operand const-select path, not register reads.
        Ports that never read a register are absent. This drives the sparse per-port read mux: a port that reads a
        single register needs no mux at all, and one that reads several needs a mux spanning only those registers.
        """
        sets: dict[tuple[OperatorInstance, int], set[int]] = {}
        for op in self.ops:
            for pos, operand in enumerate(op.operands):
                if isinstance(operand.source, RegRef):
                    sets.setdefault((op.inst, pos), set()).add(operand.source.index)
        return {port: sorted(regs) for port, regs in sets.items()}

    @property
    def write_set_per_register(self) -> dict[int, list[tuple[OperatorInstance, int]]]:
        """
        For each WIDE register index, the ``(instance, output port)`` lanes that ever write it, in a canonical order.

        This drives the sparse per-register write select: a register written by a single lane needs no write-port
        mux. The input-load writers of registers ``0..nload-1`` are tracked separately via ``lir.float_inputs``
        (they are a distinct, address-free write source folded into the same select).
        """
        return self._write_sets(RegRef)

    @property
    def bool_write_set_per_register(self) -> dict[int, list[tuple[OperatorInstance, int]]]:
        """The boolean-bank counterpart of :attr:`write_set_per_register`, in the same canonical lane order."""
        return self._write_sets(BoolRegRef)

    def _write_sets(self, bank: type[RegRef] | type[BoolRegRef]) -> dict[int, list[tuple[OperatorInstance, int]]]:
        sets: dict[int, list[tuple[OperatorInstance, int]]] = {}
        for op in self.ops:
            for write in op.writes:
                if not isinstance(write.dst, bank):
                    continue
                writers = sets.setdefault(write.dst.index, [])
                if (op.inst, write.port) not in writers:
                    writers.append((op.inst, write.port))
        for writers in sets.values():
            writers.sort(key=lambda lane: (lane[0].operator.instance_stem, lane[0].index, lane[1]))
        return sets

    @property
    def group_by_cycle(self) -> tuple[dict[int, list[PooledScheduledOp]], dict[int, list[PooledScheduledOp]]]:
        """The schedule grouped into per-cycle issues and commits, each canonically ordered."""
        issues: dict[int, list[PooledScheduledOp]] = {}
        commits: dict[int, list[PooledScheduledOp]] = {}
        for op in self.ops:
            issues.setdefault(op.issue_cycle, []).append(op)
            commits.setdefault(op.commit_cycle, []).append(op)
        for group in (issues, commits):
            for ops in group.values():
                ops.sort(
                    key=lambda op: (
                        op.inst.operator.instance_stem,
                        op.inst.index,
                        op.writes[0].dst.index,
                        op.issue_cycle,
                    )
                )
        return issues, commits

    @property
    def reg_liveness(self) -> dict[RegRef, set[int]]:
        """
        Map each wide register to the actual clock cycles on which it holds a live value.

        This is cycle-accurate to the emitted hardware, in the executing-step (hardware) frame. Timing comes from the
        shared helpers: an input lands on cycle 1; an operator result lands on ``result_landing_cycle`` (which for the
        last result is the initiation interval); an operand is read on ``operand_read_cycle``; an output tap on the
        present cycle; and a non-coalesced slot's writeback lands (and reads its source) on ``state_copy_step`` -- the
        present cycle for a boundary copy, earlier for an early install. A slot register additionally stays live
        through the present cycle, since its live-out must reside there for the next initiation. Each row spans a value
        from when it lands in the array through its last read.
        """
        present = self.initiation_interval  # hardware-frame present / boundary step
        defs: dict[RegRef, list[int]] = {}
        uses: dict[RegRef, list[int]] = {}
        for load in self.float_inputs:
            defs.setdefault(load.dst, []).append(1)
        for slot in self.float_state_slots:
            defs.setdefault(slot.reg, []).append(1)  # the live-in is resident in the slot register from the start
            # The live-out must reside in the slot register at the boundary to carry into the next initiation, so the
            # register stays live through the boundary even when nothing reads it again this frame -- installing the
            # new value early is not its death.
            uses.setdefault(slot.reg, []).append(present)
            if slot.needs_copy:  # the non-coalesced live-out lands on the install step and is carried to the next call
                defs.setdefault(slot.reg, []).append(self.state_copy_step(slot))
        for op in self.ops:
            for write in op.writes:
                if isinstance(write.dst, RegRef):
                    defs.setdefault(write.dst, []).append(result_landing_cycle(write.dst, op.commit_cycle))
            read = operand_read_cycle(op.inst.operator, op.issue_cycle)
            for operand in op.operands:
                if isinstance(operand.source, RegRef):
                    uses.setdefault(operand.source, []).append(read)
        for wire in self.float_outputs:
            if isinstance(wire.tap.source, RegRef):
                uses.setdefault(wire.tap.source, []).append(present)
        for slot in self.float_state_slots:  # the live-out tap is read on the install step to persist the slot
            if isinstance(slot.tap.source, RegRef):
                uses.setdefault(slot.tap.source, []).append(self.state_copy_step(slot))
        # Inline ops touch the wide bank too: a float->bool cast reads a float operand and the bool->float cast
        # writes a wide register, whose result lands at the wide bank's ``wide_landing_cycle`` like any wide result
        # (its PC-gated write is aligned with the write-latch write enables). Without these the residence of a cast
        # operand or result would be missing from the tint.
        for block in self.blocks:
            base_pc = self.block_base[block.index]
            for bop in block.inline_ops:
                read = operand_read_cycle(bop.operator, base_pc + bop.issue_cycle)
                for inline_operand in bop.operands:
                    if isinstance(inline_operand.source, RegRef):
                        uses.setdefault(inline_operand.source, []).append(read)
                if isinstance(bop.write.dst, RegRef):
                    defs.setdefault(bop.write.dst, []).append(
                        result_landing_cycle(bop.write.dst, base_pc + bop.commit_cycle)
                    )
            # A phi-arm copy installs its value into the merged register at the predecessor's tail and reads its source
            # there (the wide-bank analog of ``bool_writes`` below). Without this a merged register's residence -- and,
            # under reuse, the source register's last read -- would be missing from the tint.
            for copy in block.copies:
                step = base_pc + copy_step_cycle(copy.issue_cycle)
                defs.setdefault(copy.dst, []).append(step)
                if isinstance(copy.source.source, RegRef):
                    uses.setdefault(copy.source.source, []).append(step)
        return {reg: residence_rows(defs.get(reg, []), uses.get(reg, []), present) for reg in defs.keys() | uses.keys()}

    @property
    def bool_liveness(self) -> dict[BoolRegRef, set[int]]:
        """
        Map each boolean register to the cycles on which it holds a live value, the boolean-bank counterpart of
        :attr:`reg_liveness` in the same executing-step frame. A boolean register is defined when a comparison,
        boolean-logic op, or float->bool cast commits its result, when a boolean phi/state install lands, and -- for a
        persistent slot -- at the live-in resident from cycle 1; it is read by a boolean-logic op or a bool->float cast
        taking it as an operand, by a branch testing it as a condition, by a phi/state install copying it, and at the
        boundary where a slot's live-out must persist for the next initiation.
        """
        present = self.initiation_interval
        defs: dict[BoolRegRef, list[int]] = {}
        uses: dict[BoolRegRef, list[int]] = {}
        for slot in self.bool_state_slots:
            defs.setdefault(slot.reg, []).append(1)  # the live-in is resident from the start
            uses.setdefault(slot.reg, []).append(present)  # ...and the live-out must reside through the boundary
            if slot.needs_copy:
                defs.setdefault(slot.reg, []).append(present)  # the new live-out lands at the boundary (read-first)
                if isinstance(slot.live_out.source, BoolRegRef):
                    uses.setdefault(slot.live_out.source, []).append(present)
        for load in self.bool_inputs:
            defs.setdefault(load.dst, []).append(1)
        # Pooled firings write the boolean bank too (the comparator's tapped flags); their issue cycles are absolute.
        for op in self.ops:
            for write in op.writes:
                if isinstance(write.dst, BoolRegRef):
                    defs.setdefault(write.dst, []).append(result_landing_cycle(write.dst, op.commit_cycle))
        for block in self.blocks:
            base_pc = self.block_base[block.index]
            for bop in block.inline_ops:
                # Boolean operands are read by inline ops on their single fire step (operand_read_cycle).
                read = operand_read_cycle(bop.operator, base_pc + bop.issue_cycle)
                for operand in bop.operands:
                    if isinstance(operand.source, BoolRegRef):  # boolean logic and the bool->float cast read booleans
                        uses.setdefault(operand.source, []).append(read)
                if isinstance(bop.write.dst, BoolRegRef):  # boolean logic and the float->bool cast write booleans
                    defs.setdefault(bop.write.dst, []).append(
                        result_landing_cycle(bop.write.dst, base_pc + bop.commit_cycle)
                    )
            for bwrite in block.bool_writes:
                step = base_pc + copy_step_cycle(bwrite.issue_cycle)
                defs.setdefault(bwrite.dst, []).append(step)
                if isinstance(bwrite.source.source, BoolRegRef):
                    uses.setdefault(bwrite.source.source, []).append(step)
            if isinstance(block.terminator, Branch):  # the next-PC case reads the condition at the block's boundary PC
                uses.setdefault(block.terminator.cond, []).append(base_pc + boundary_step(block.block_makespan))
        for wire in self.bool_outputs:
            if isinstance(wire.tap.source, BoolRegRef):
                uses.setdefault(wire.tap.source, []).append(present)
        return {reg: residence_rows(defs.get(reg, []), uses.get(reg, []), present) for reg in defs.keys() | uses.keys()}

    @property
    def write_timeline(self) -> dict[RegRef, list[tuple[int, Producer]]]:
        """
        Per-register write timeline ``(landing cycle, producer)`` in the hardware/executing-step frame, used to resolve
        a register source at a hardware read cycle. A value is readable from the cycle it lands in the array: inputs and
        state live-ins on cycle 1, an operator result on ``result_landing_cycle``.
        """
        writes: dict[RegRef, list[tuple[int, Producer]]] = {}
        for i, load in enumerate(self.float_inputs):
            writes.setdefault(load.dst, []).append((1, InputProducer(i)))
        # A slot register starts each initiation holding its live-in (the value carried over from the previous one);
        # a coalesced operator may then overwrite it later in the same initiation via its own OperationProducer entry.
        for s, slot in enumerate(self.float_state_slots):
            writes.setdefault(slot.reg, []).append((1, StateProducer(s)))
        for j, op in enumerate(self.ops):
            for write in op.writes:
                if isinstance(write.dst, RegRef):
                    writes.setdefault(write.dst, []).append(
                        (result_landing_cycle(write.dst, op.commit_cycle), OperationProducer(j))
                    )
        # Inline firings write the wide bank too (the bool->float cast); without them a cast-fed operand would
        # resolve to no producer at all.
        for block in self.blocks:
            base_pc = self.block_base[block.index]
            for k, inline_op in enumerate(block.inline_ops):
                if isinstance(inline_op.write.dst, RegRef):
                    writes.setdefault(inline_op.write.dst, []).append(
                        (
                            result_landing_cycle(inline_op.write.dst, base_pc + inline_op.commit_cycle),
                            InlineProducer(block.index, k),
                        )
                    )
        for events in writes.values():
            events.sort(key=lambda event: event[0])
        return writes
