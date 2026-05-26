"""
The low-level IR (LIR): the scheduled, bound, register-allocated microprogram for the synthesized ZISC machine.

A :class:`Lir` is controller-agnostic -- it describes which hardware operators issue on which cycle, reading/writing
which registers, with which folded sign controls.
"""

from dataclasses import dataclass

from ._operators import HardwareOperator, SignControl
from ._type import FloatFormat


@dataclass(frozen=True, slots=True)
class OperatorInstance:
    """
    One physical operator module, e.g. ``u_fadd_0`` or ``u_fmul_ilog2_const_2``.

    ``operator`` is the fully specified hardware operator it elaborates; ``index`` numbers the copies of that operator
    class. The scheduler pools operations by the hardware-operator instance: equal operators may time-share one module.
    """

    operator: HardwareOperator
    index: int  # 0-based within its operator class (contiguous across that class's distinct operators)


@dataclass(frozen=True, slots=True)
class RegRef:
    """A read/write of float register ``index`` in the register bank."""

    index: int


@dataclass(frozen=True, slots=True)
class ConstRef:
    """An immediate constant, ``index`` into the LIR constant pool."""

    index: int


@dataclass(frozen=True, slots=True)
class Operand:
    """An operator input: a register read or a constant immediate, with a folded sign control."""

    source: RegRef | ConstRef
    sign: SignControl = SignControl()


@dataclass(frozen=True, slots=True)
class ScheduledOp:
    """
    One operator firing in the software-pipelined schedule.

    ``inst`` is the bound physical instance, ``issue_cycle`` is the cycle its ``in_valid`` is asserted, and the result
    commits to ``dst`` at ``commit_cycle == issue_cycle + latency``.
    """

    inst: OperatorInstance
    operands: list[Operand]
    result_sign: SignControl
    dst: RegRef
    issue_cycle: int
    latency: int

    @property
    def commit_cycle(self) -> int:
        return self.issue_cycle + self.latency


@dataclass(frozen=True, slots=True)
class InputLoad:
    """An input port sampled into a register at in_valid."""

    name: str
    dst: RegRef


@dataclass(frozen=True, slots=True)
class OutputWire:
    """An output port driven from a register or constant immediate, with a folded output sign control."""

    name: str
    source: RegRef | ConstRef
    sign: SignControl = SignControl()


@dataclass(frozen=True, slots=True)
class FloatRegFileLayout:
    """The floating-point register file resource and its scalar format."""

    fmt: FloatFormat
    nreg: int  # number of float registers (N)
    nrd: int  # combinational read ports
    nwr: int  # synchronous write ports
    nload: int  # immediate parallel-load lanes: registers 0..nload-1 are loaded from load_data at in_valid


@dataclass(frozen=True, slots=True)
class Lir:
    module_name: str
    instances: list[OperatorInstance]
    consts: list[float]  # constant pool: index -> value
    regfile: FloatRegFileLayout
    inputs: list[InputLoad]  # ordered as the function parameters
    ops: list[ScheduledOp]  # the pipelined schedule, ordered by (issue_cycle, value ID)
    outputs: list[OutputWire]
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
