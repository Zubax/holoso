"""
The low-level IR (LIR): the scheduled, bound, register-allocated microprogram for the synthesized ZISC machine.

A :class:`Lir` is controller-agnostic -- it describes which hardware operators issue on which cycle, reading/writing
which typed storage resources, with which folded sign controls.
"""

from dataclasses import dataclass

from .._operators import FloatHardwareOperator, FloatSignControl, HardwareOperator
from .._type import FloatFormat, FloatType
from ._ports import ControlInputPort, ControlOutputPort, ControlPort, DataInputPort, DataOutputPort, Port

FETCH_STAGES = 3
FETCH_LAG = FETCH_STAGES - 1


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
    previous initiation) until it is overwritten, and holding the slot's live-out at the initiation boundary.

    ``tap`` is the live-out's source tap (register/constant + folded sign), the same primitive an output wire taps; here
    the sink is the slot register rather than a port. When the tap is exactly ``reg`` with an identity sign the live-out
    coalesced onto the slot register (its producing operator wrote it) and the backend emits no boundary copy; otherwise
    the backend latches the tap into ``reg`` at the boundary. A public attribute's observable ``state_<name>`` port is a
    separate output wire tapping the same value, not a property of the slot.
    """

    name: str
    reg: FloatRegRef
    reset_value: float
    tap: FloatOperand

    @property
    def needs_copy(self) -> bool:
        return not (self.tap.source == self.reg and self.tap.sign == FloatSignControl())


@dataclass(frozen=True, slots=True)
class RegFileLayout:
    """A typed register file resource."""

    nreg: int
    nrd: int
    nwr: int
    nload: int


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
    float_ops: list[FloatScheduledOp]  # the pipelined schedule, ordered by (issue_cycle, value ID)
    float_outputs: list[FloatOutputWire]
    float_state_slots: list[FloatStateSlot]  # persistent registers, ordered as the instance attributes
    makespan: int  # last commit cycle (0 if no ops); the in_valid->out_valid latency is makespan + 1
    op_count: int
    max_chain_len: int  # longest dependency chain in hardware operators (for verification tolerance)

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
        The hardware executing step on which the outputs are valid in the register array (NOT the scheduler-frame cycle
        ``makespan + 1`` the allocator and model use). The last operator commits on the makespan cycle; the write latch
        delays the write, so the result lands on the next edge and is presented on the executing step ``makespan + 2``.
        """
        return self.makespan + 2

    @property
    def cyc_width(self) -> int:
        """Bit width of the err_pc diagnostic: enough to hold any executing step ``0..present_step``."""
        return max(1, self.present_step.bit_length())

    @property
    def initiation_interval(self) -> int:
        """
        Observable in_valid->out_valid latency.

        Cycle 0 accepts and loads the inputs; compute cycles 1..makespan run the schedule. With the write latch the
        result is presented on the executing step ``present_step``; the executing step lags the fetch PC by FETCH_LAG,
        so out_valid is asserted FETCH_LAG cycles later: ``present_step + FETCH_LAG``. Zero-op modules still present
        one accept-relative cycle plus the fixed staging.
        """
        return self.present_step + FETCH_LAG

    def result_landing_cycle(self, op: FloatScheduledOp) -> int:
        """
        Hardware-frame cycle on which an operator result lands in the register array ready to read: its commit cycle
        plus the write latch, the read-first edge, and the fetch lag. For the last result this equals the initiation
        interval. This is the single definition shared by liveness and the report so the two cannot drift.
        """
        return op.commit_cycle + FETCH_LAG + 2

    def operand_read_cycle(self, op: FloatScheduledOp) -> int:
        """Hardware-frame cycle on which an operator reads its operands (the read latch presents the address early)."""
        return op.issue_cycle + FETCH_LAG - 1

    @property
    def has_state(self) -> bool:
        """Whether the module retains persistent state across initiations."""
        return bool(self.float_state_slots)

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
        last result is the initiation interval); an operand is read on ``operand_read_cycle``; and an output tap and a
        non-coalesced slot's boundary write both happen on the present cycle (the initiation interval). Each row spans a
        value from when it lands in the array through its last read.
        """
        present = self.initiation_interval  # hardware-frame present / boundary step
        defs: dict[FloatRegRef, list[int]] = {}
        uses: dict[FloatRegRef, list[int]] = {}
        for load in self.float_inputs:
            defs.setdefault(load.dst, []).append(1)
        for slot in self.float_state_slots:
            defs.setdefault(slot.reg, []).append(1)  # the live-in is resident in the slot register from the start
            if slot.needs_copy:  # the non-coalesced live-out lands at the boundary and is carried to the next call
                defs.setdefault(slot.reg, []).append(present)
        for op in self.float_ops:
            defs.setdefault(op.dst, []).append(self.result_landing_cycle(op))
            read = self.operand_read_cycle(op)
            for operand in op.operands:
                if isinstance(operand.source, FloatRegRef):
                    uses.setdefault(operand.source, []).append(read)
        for wire in self.float_outputs:
            if isinstance(wire.tap.source, FloatRegRef):
                uses.setdefault(wire.tap.source, []).append(present)
        for slot in self.float_state_slots:  # the live-out tap is read at the boundary to persist the slot
            if isinstance(slot.tap.source, FloatRegRef):
                uses.setdefault(slot.tap.source, []).append(present)
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
        """Per-register write timeline (commit cycle, producer) used to resolve a register source at a read cycle."""
        writes: dict[FloatRegRef, list[tuple[int, FloatProducer]]] = {}
        for i, load in enumerate(self.float_inputs):
            writes.setdefault(load.dst, []).append((0, InputProducer(i)))
        # A slot register starts each initiation holding its live-in (the value carried over from the previous one);
        # a coalesced operator may then overwrite it later in the same initiation via its own OperationProducer entry.
        for s, slot in enumerate(self.float_state_slots):
            writes.setdefault(slot.reg, []).append((0, StateProducer(s)))
        for j, op in enumerate(self.float_ops):
            writes.setdefault(op.dst, []).append((op.commit_cycle, OperationProducer(j)))
        for events in writes.values():
            events.sort(key=lambda event: event[0])
        return writes
