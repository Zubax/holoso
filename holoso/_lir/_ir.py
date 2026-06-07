"""
The low-level IR (LIR): the scheduled, bound, register-allocated microprogram for the synthesized ZISC machine.

A :class:`Lir` is controller-agnostic -- it describes which hardware operators issue on which cycle, reading/writing
which typed storage resources, with which folded sign controls.
"""

from dataclasses import dataclass

from .._hir import RelationalOp
from .._operators import FCmpOperator, FloatHardwareOperator, FloatSignControl, HardwareOperator
from .._type import FloatFormat, FloatType
from ._ports import ControlInputPort, ControlOutputPort, ControlPort, DataInputPort, DataOutputPort, Port

FETCH_STAGES = 3
FETCH_LAG = FETCH_STAGES - 1


# Executing-step (hardware) frame cycle offsets: the single source of truth shared by the LIR cycle helpers below, the
# write timeline, the numerical model, and the register allocator, so a value's landing/read/copy/boundary cycle is
# computed in exactly one place and the four consumers cannot drift (the bug this centralization prevents).
def landing_cycle(commit_cycle: int) -> int:
    """The cycle a result becomes readable in the array: its commit plus the write latch and the read-first edge."""
    return commit_cycle + FETCH_LAG + 2


def read_latch_cycle(issue_cycle: int) -> int:
    """The cycle an operator reads its operands -- the read latch presents the address early."""
    return issue_cycle + FETCH_LAG - 1


def copy_step_cycle(install_cycle: int) -> int:
    """The step a non-coalesced slot writeback fires and samples its source."""
    return install_cycle + FETCH_LAG + 1


def boundary_step(makespan: int) -> int:
    """The boundary / initiation-interval step: the last result lands here and outputs are resident here."""
    return makespan + 2 + FETCH_LAG


@dataclass(frozen=True, slots=True)
class OperatorInstance:
    """
    One physical operator module, e.g. ``u_fadd_326215ea_0`` or ``u_fmul_ilog2_const_7296114c_0``.

    ``operator`` is the fully specified hardware operator it elaborates; ``index`` numbers the copies of that operator
    value. The scheduler pools operations by the hardware-operator instance: equal operators may time-share one module.
    """

    operator: HardwareOperator
    index: int  # 0-based within this concrete operator value


@dataclass(frozen=True, slots=True)
class FloatOperatorInstance(OperatorInstance):
    """One physical floating-point operator module."""

    operator: FloatHardwareOperator
    index: int


@dataclass(frozen=True, slots=True)
class RegRef:
    """A read/write of typed register ``index`` in one register bank."""

    index: int

    @property
    def stable_label(self) -> str:
        return f"r{self.index}"

    @property
    def is_register(self) -> bool:
        return True


@dataclass(frozen=True, slots=True)
class FloatRegRef(RegRef):
    """A read/write of float register ``index`` in the float register bank."""


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
    """A float operator input: a float register read or float constant immediate, with folded sign control."""

    source: FloatRegRef | FloatConstRef
    sign: FloatSignControl = FloatSignControl()

    @property
    def stable_label(self) -> str:
        return self.sign.decorate(self.source.stable_label)


@dataclass(frozen=True, slots=True)
class ScheduledOp:
    """
    One operator firing in the software-pipelined schedule.

    ``inst`` is the bound physical instance, ``issue_cycle`` is the cycle its ``in_valid`` is asserted, and the result
    commits to ``dst`` at ``commit_cycle == issue_cycle + latency``.
    """

    inst: OperatorInstance
    dst: RegRef
    issue_cycle: int
    latency: int

    @property
    def commit_cycle(self) -> int:
        return self.issue_cycle + self.latency


@dataclass(frozen=True, slots=True)
class FloatScheduledOp(ScheduledOp):
    """One floating-point operator firing in the software-pipelined schedule."""

    inst: FloatOperatorInstance
    operands: list[FloatOperand]
    result_sign: FloatSignControl
    dst: FloatRegRef
    issue_cycle: int
    latency: int


@dataclass(frozen=True, slots=True)
class InputLoad:
    """An input port sampled into a typed register at in_valid."""

    name: str
    dst: RegRef


@dataclass(frozen=True, slots=True)
class FloatInputLoad(InputLoad):
    """A float input port sampled into a float register at in_valid."""

    dst: FloatRegRef


@dataclass(frozen=True, slots=True)
class OutputWire:
    """An output port: a named external sink driven at the present step by a typed source tap."""

    name: str
    tap: Operand


@dataclass(frozen=True, slots=True)
class FloatOutputWire(OutputWire):
    """A float output port: the named external sink for a float source tap (register/constant + folded sign)."""

    tap: FloatOperand


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
    reg: FloatRegRef
    reset_value: float
    tap: FloatOperand
    install_cycle: int  # scheduler-frame cycle the live-out lands in reg (its max, makespan + 1, is the boundary)

    @property
    def needs_copy(self) -> bool:
        return not (self.tap.source == self.reg and self.tap.sign == FloatSignControl())


@dataclass(frozen=True, slots=True)
class BoolRegRef:
    """A read/write of boolean register ``index`` in the 1-bit boolean register bank."""

    index: int

    @property
    def stable_label(self) -> str:
        return f"b{self.index}"


@dataclass(frozen=True, slots=True)
class BoolConstRef:
    """A boolean immediate (``True``/``False``); the bool bank has no constant pool, the value rides inline."""

    value: bool


type BoolSource = BoolRegRef | BoolConstRef


@dataclass(frozen=True, slots=True)
class BoolOperand:
    """A boolean operand: a boolean register read or an immediate True/False."""

    source: BoolSource


@dataclass(frozen=True, slots=True)
class FloatCopy:
    """
    A register-to-register move installing a phi arm's value into the merged register at a predecessor's tail: ``dst``
    takes ``source`` on the block-relative ``issue_cycle``. Used when a phi arm is not an operator result that can be
    coalesced directly onto the merged register (e.g. an input, a constant, or a value defined in another block).
    """

    dst: FloatRegRef
    source: FloatOperand
    issue_cycle: int


@dataclass(frozen=True, slots=True)
class BoolWrite:
    """A boolean register install of a phi arm (a bool const or another bool register) on a block-relative cycle."""

    dst: BoolRegRef
    source: BoolOperand
    issue_cycle: int


@dataclass(frozen=True, slots=True)
class BoolScheduledOp:
    """
    A scheduled float comparator firing in a block: it reads float operands (with folded sign controls), and on
    ``commit_cycle == issue_cycle + latency`` its one-hot order flags are reduced by ``relation`` into the boolean
    register ``dst``. All comparisons share one pooled ``holoso_fcmp`` instance: each block holds at most one
    comparison and blocks are mutually exclusive, so the emitter PC-muxes the single comparator's operands.
    """

    operator: FCmpOperator
    operands: list[FloatOperand]
    dst: BoolRegRef
    relation: RelationalOp
    issue_cycle: int
    latency: int

    @property
    def commit_cycle(self) -> int:
        return self.issue_cycle + self.latency


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
    One basic block of the scheduled microprogram, with block-relative cycles (block start is cycle 0). ``float_ops``,
    ``float_copies``, and ``bool_writes`` are the block's datapath events; ``terminator`` redirects the fetch PC at the
    block boundary. ``block_makespan`` is the last commit cycle inside the block (0 if it has no datapath events).
    """

    index: int
    float_ops: list[FloatScheduledOp]
    bool_ops: list[BoolScheduledOp]
    float_copies: list[FloatCopy]
    bool_writes: list[BoolWrite]
    terminator: Terminator
    block_makespan: int


@dataclass(frozen=True, slots=True)
class BoolStateSlot:
    """
    A persistent boolean state register: reset to ``reset_value``, holding the slot's live-in throughout the
    transaction and installing its live-out (``live_out``, a boolean register or constant) at the boundary, read-first
    -- so an output or branch that still reads the live-in sees the old value, exactly like a float slot.
    """

    name: str
    reg: BoolRegRef
    reset_value: bool
    live_out: BoolOperand

    @property
    def needs_copy(self) -> bool:
        """False only when the live-out already resides in the slot register (an unwritten slot); else install it."""
        return not (isinstance(self.live_out.source, BoolRegRef) and self.live_out.source == self.reg)


@dataclass(frozen=True, slots=True)
class RegFileLayout:
    """A typed register file resource."""

    nreg: int
    nrd: int
    nwr: int
    nload: int


@dataclass(frozen=True, slots=True)
class BoolRegFileLayout:
    """The boolean register bank: ``nreg`` 1-bit registers (branch conditions and boolean state)."""

    nreg: int


@dataclass(frozen=True, slots=True)
class FloatRegFileLayout(RegFileLayout):
    """The floating-point register file resource and its scalar format."""

    fmt: FloatFormat
    nreg: int  # number of float registers (N)
    nrd: int  # combinational read ports
    nwr: int  # synchronous write ports
    nload: int  # immediate parallel-load lanes: registers 0..nload-1 are loaded from load_data at in_valid


@dataclass(frozen=True, slots=True)
class InputProducer:
    """A write to a register that came from an input-load lane ``index`` (in module-port order)."""

    index: int


@dataclass(frozen=True, slots=True)
class OperationProducer:
    """A write to a register that came from operation ``index`` in ``Lir.float_ops``."""

    index: int


@dataclass(frozen=True, slots=True)
class StateProducer:
    """A state register's live-in: the value it carries over from the previous initiation (or the reset snapshot)."""

    index: int  # index into Lir.float_state_slots


type FloatProducer = InputProducer | OperationProducer | StateProducer


@dataclass(frozen=True, slots=True)
class Lir:
    module_name: str
    float_instances: list[FloatOperatorInstance]
    float_consts: list[float]  # constant pool: index -> value
    float_regfile: FloatRegFileLayout
    float_inputs: list[FloatInputLoad]  # ordered as the function parameters
    float_ops: list[FloatScheduledOp]  # the pipelined schedule, flattened across blocks with ABSOLUTE issue cycles
    float_outputs: list[FloatOutputWire]
    float_state_slots: list[FloatStateSlot]  # persistent registers, ordered as the instance attributes
    makespan: int  # last absolute commit cycle (0 if no ops)
    op_count: int
    max_chain_len: int  # longest dependency chain in hardware operators (for verification tolerance)
    # Control-flow overlay. A straight-line kernel has a single block ending in Ret; ``blocks[0]`` is the entry,
    # ``block_base[i]`` is block i's absolute start PC, and ``last_pc`` is the out_valid boundary (the single Ret).
    blocks: list[LirBlock]
    block_base: list[int]
    entry: int
    last_pc: int  # LASTPC: the fetch PC at which out_valid asserts (the single Ret block's boundary)
    min_initiation_interval: int  # shortest executable path latency; exact for branch-free kernels, else a lower bound
    bool_regfile: BoolRegFileLayout
    bool_state_slots: list[BoolStateSlot]  # persistent boolean registers (branch conditions, boolean attributes)

    @property
    def ports(self) -> list[Port]:
        scalar_type = FloatType(self.float_regfile.fmt)
        ports: list[Port] = [
            ControlInputPort("clk", 1),
            ControlInputPort("rst", 1),
            ControlInputPort("in_valid", 1),
            ControlOutputPort("in_ready", 1),
            ControlOutputPort("out_valid", 1),
            ControlInputPort("out_ready", 1),
        ]
        ports.extend(DataInputPort(f"in_{load.name}", scalar_type) for load in self.float_inputs)
        ports.extend(DataOutputPort(wire.name, scalar_type) for wire in self.float_outputs)
        ports.append(ControlOutputPort("err_pc", self.cyc_width))
        return ports

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

    def result_landing_cycle(self, op: FloatScheduledOp) -> int:
        """
        Hardware-frame cycle on which an operator result lands in the register array ready to read. For the last result
        this equals the initiation interval. This is the single definition shared by liveness, the write timeline, the
        model, and the register allocator so they cannot drift.
        """
        return landing_cycle(op.commit_cycle)

    def operand_read_cycle(self, op: FloatScheduledOp) -> int:
        """Hardware-frame cycle on which an operator reads its operands (the read latch presents the address early)."""
        return read_latch_cycle(op.issue_cycle)

    def state_copy_step(self, slot: FloatStateSlot) -> int:
        """
        The fetch-PC value -- equivalently the hardware-frame cycle -- on which a non-coalesced slot's writeback copy
        fires. For a boundary install this is ``initiation_interval`` (LASTPC), where it reduces to the accepted-
        transaction edge. The copy reads its source and lands the new live-out in the slot register on this same step;
        shared by liveness and the emitter so the two cannot drift.
        """
        return copy_step_cycle(slot.install_cycle)

    @property
    def has_state(self) -> bool:
        """Whether the module retains persistent state across initiations."""
        return bool(self.float_state_slots) or bool(self.bool_state_slots)

    @property
    def is_control_flow(self) -> bool:
        """
        Whether this kernel took the control-flow build path (a CFG of blocks and/or a boolean register bank) rather
        than the straight-line single-block path. The model and the emitter branch on this to select the CFG execution
        / boundary-install path; both must agree, so the predicate lives here once.
        TODO FIXME: THIS IS TEMPORARY. The straight-line/single-block path will be merged with CFG eventually.
        """
        return len(self.blocks) > 1 or bool(self.bool_state_slots)

    @property
    def read_set_per_port(self) -> dict[tuple[FloatOperatorInstance, int], list[int]]:
        """
        For each operator read port -- identified by its ``(instance, operand-position)`` pair -- the sorted distinct
        register indices it ever reads across the schedule.

        Constant operands are excluded: they are immediates on the per-operand const-select path, not register reads.
        Ports that never read a register are absent. This drives the sparse per-port read mux: a port that reads a
        single register needs no mux at all, and one that reads several needs a mux spanning only those registers.
        """
        sets: dict[tuple[FloatOperatorInstance, int], set[int]] = {}
        for op in self.float_ops:
            for pos, operand in enumerate(op.operands):
                if isinstance(operand.source, FloatRegRef):
                    sets.setdefault((op.inst, pos), set()).add(operand.source.index)
        return {port: sorted(regs) for port, regs in sets.items()}

    @property
    def write_set_per_register(self) -> dict[int, list[FloatOperatorInstance]]:
        """
        For each register index, the operator instances that ever write it (each through its own dedicated write port),
        in a canonical order.

        This drives the sparse per-register write select: a register written by a single instance needs no write-port
        mux. The input-load writers of registers ``0..nload-1`` are tracked separately via ``lir.float_inputs``
        (they are a distinct, address-free write source folded into the same select).
        """
        sets: dict[int, list[FloatOperatorInstance]] = {}
        for op in self.float_ops:
            writers = sets.setdefault(op.dst.index, [])
            if op.inst not in writers:
                writers.append(op.inst)
        for writers in sets.values():
            writers.sort(key=lambda inst: (inst.operator.instance_stem, inst.index))
        return sets

    @property
    def group_by_cycle(self) -> tuple[dict[int, list[FloatScheduledOp]], dict[int, list[FloatScheduledOp]]]:
        """The schedule grouped into per-cycle issues and commits, each canonically ordered."""
        issues: dict[int, list[FloatScheduledOp]] = {}
        commits: dict[int, list[FloatScheduledOp]] = {}
        for op in self.float_ops:
            issues.setdefault(op.issue_cycle, []).append(op)
            commits.setdefault(op.commit_cycle, []).append(op)
        for group in (issues, commits):
            for ops in group.values():
                ops.sort(key=lambda op: (op.inst.operator.instance_stem, op.inst.index, op.dst.index, op.issue_cycle))
        return issues, commits

    @property
    def float_liveness(self) -> dict[FloatRegRef, set[int]]:
        """
        Map each float register to the actual clock cycles on which it holds a live value.

        This is cycle-accurate to the emitted hardware, in the executing-step (hardware) frame. Timing comes from the
        shared helpers: an input lands on cycle 1; an operator result lands on ``result_landing_cycle`` (which for the
        last result is the initiation interval); an operand is read on ``operand_read_cycle``; an output tap on the
        present cycle; and a non-coalesced slot's writeback lands (and reads its source) on ``state_copy_step`` -- the
        present cycle for a boundary copy, earlier for an early install. A slot register additionally stays live
        through the present cycle, since its live-out must reside there for the next initiation. Each row spans a value
        from when it lands in the array through its last read.
        """
        present = self.initiation_interval  # hardware-frame present / boundary step
        defs: dict[FloatRegRef, list[int]] = {}
        uses: dict[FloatRegRef, list[int]] = {}
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
        for op in self.float_ops:
            defs.setdefault(op.dst, []).append(self.result_landing_cycle(op))
            read = self.operand_read_cycle(op)
            for operand in op.operands:
                if isinstance(operand.source, FloatRegRef):
                    uses.setdefault(operand.source, []).append(read)
        for wire in self.float_outputs:
            if isinstance(wire.tap.source, FloatRegRef):
                uses.setdefault(wire.tap.source, []).append(present)
        for slot in self.float_state_slots:  # the live-out tap is read on the install step to persist the slot
            if isinstance(slot.tap.source, FloatRegRef):
                uses.setdefault(slot.tap.source, []).append(self.state_copy_step(slot))
        live: dict[FloatRegRef, set[int]] = {}
        for reg in defs.keys() | uses.keys():
            writes = sorted(defs.get(reg, []))
            reads = sorted(uses.get(reg, []))
            rows: set[int] = set()
            for i, start in enumerate(writes):
                nxt = writes[i + 1] if i + 1 < len(writes) else present + 1
                # Read-first: a read on the next write's cycle still returns this value, so it belongs to this interval
                # (a non-coalesced slot register read by an output at the boundary keeps its live-in residence intact).
                last = max((use for use in reads if start <= use <= nxt), default=start)
                rows.update(range(start, last + 1))
            live[reg] = rows
        return live

    @property
    def float_write_timeline(self) -> dict[FloatRegRef, list[tuple[int, FloatProducer]]]:
        """
        Per-register write timeline ``(landing cycle, producer)`` in the hardware/executing-step frame, used to resolve
        a register source at a hardware read cycle. A value is readable from the cycle it lands in the array: inputs and
        state live-ins on cycle 1, an operator result on ``result_landing_cycle``.
        """
        writes: dict[FloatRegRef, list[tuple[int, FloatProducer]]] = {}
        for i, load in enumerate(self.float_inputs):
            writes.setdefault(load.dst, []).append((1, InputProducer(i)))
        # A slot register starts each initiation holding its live-in (the value carried over from the previous one);
        # a coalesced operator may then overwrite it later in the same initiation via its own OperationProducer entry.
        for s, slot in enumerate(self.float_state_slots):
            writes.setdefault(slot.reg, []).append((1, StateProducer(s)))
        for j, op in enumerate(self.float_ops):
            writes.setdefault(op.dst, []).append((self.result_landing_cycle(op), OperationProducer(j)))
        for events in writes.values():
            events.sort(key=lambda event: event[0])
        return writes
