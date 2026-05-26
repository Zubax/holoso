"""
The low-level IR (LIR): the scheduled, bound, register-allocated microprogram for the synthesized ZISC machine.

A :class:`Lir` is controller-agnostic -- it describes which hardware operators issue on which cycle, reading/writing
which typed storage resources, with which folded sign controls.
"""

from dataclasses import dataclass

from .._operators import FloatHardwareOperator, FloatSignControl, HardwareOperator
from .._type import FloatFormat


@dataclass(frozen=True, slots=True)
class OperatorInstance:
    """
    One physical operator module, e.g. ``u_fadd_e8_m24_0`` or ``u_fmul_ilog2_const_e8_m24_k_m2_0``.

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


@dataclass(frozen=True, slots=True)
class FloatRegRef(RegRef):
    """A read/write of float register ``index`` in the float register bank."""


@dataclass(frozen=True, slots=True)
class ConstRef:
    """An immediate constant, ``index`` into one typed LIR constant pool."""

    index: int


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
    """An output port driven from a typed register or constant immediate."""

    name: str
    source: RegRef | ConstRef


@dataclass(frozen=True, slots=True)
class FloatOutputWire(OutputWire):
    """A float output port driven from a float register or constant immediate, with folded output sign control."""

    source: FloatRegRef | FloatConstRef
    sign: FloatSignControl = FloatSignControl()


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
class Lir:
    module_name: str
    float_instances: list[FloatOperatorInstance]
    float_consts: list[float]  # constant pool: index -> value
    float_regfile: FloatRegFileLayout
    float_inputs: list[FloatInputLoad]  # ordered as the function parameters
    float_ops: list[FloatScheduledOp]  # the pipelined schedule, ordered by (issue_cycle, value ID)
    float_outputs: list[FloatOutputWire]
    makespan: int  # last commit cycle (0 if no ops); the in_valid->out_valid latency is makespan + 1
    op_count: int
    max_chain_len: int  # longest dependency chain in hardware operators (for verification tolerance)

    @property
    def cyc_width(self) -> int:
        """Bit width of the cycle counter (and ``err_cyc``): enough to hold ``0..makespan+1``."""
        return max(1, (self.makespan + 1).bit_length())

    @property
    def initiation_interval(self) -> int:
        """
        Exact in_valid->out_valid latency: the schedule makespan plus one cycle to present.

        Cycle 0 accepts and writes the inputs; compute cycles 1..makespan run the schedule; the last operator commits
        on the makespan cycle; the result lands in the register file on the next edge and is presented on
        cycle makespan+1. Zero-op modules present on cycle 1.
        """
        return self.makespan + 1
