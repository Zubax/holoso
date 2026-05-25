"""
The low-level IR (LIR): the scheduled, bound, register-allocated microprogram for the synthesized ZISC machine.

A :class:`Lir` is controller-agnostic -- it describes which operators issue on which cycle, reading/writing which
registers, with which folded sign-ops.
"""

from dataclasses import dataclass

from .format import FloatFormat
from .operators import Op, Sgnop


@dataclass(frozen=True, slots=True)
class OperatorInstance:
    """
    One physical operator module, e.g. ``u_fadd_0`` or ``u_fmul_ilog2_const_2``.

    ``op`` is the fully-specified operator it elaborates (its parameters baked in); ``index`` numbers the copies of
    that operator class. The scheduler pools ops onto instances by the ``op`` instance: equal ops may time-share one
    instance (at most one issue per instance per cycle, a fully pipelined instance carrying several ops in flight),
    bounded by the per-class instance budget. E.g., all ``fadd`` share, while ``fmul_ilog2_const`` shares by exponent.
    """

    op: Op
    index: int  # 0-based within its operator class (contiguous across that class's distinct ops)


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
    """An operator input: a register read or a constant immediate, with a folded sign-op."""

    source: RegRef | ConstRef
    sgnop: Sgnop


@dataclass(frozen=True, slots=True)
class ScheduledOp:
    """
    One operator firing in the software-pipelined schedule.

    ``inst`` is the bound physical instance (decided by the scheduler), ``issue_cycle`` is the cycle its ``in_valid``
    is asserted (operands read combinationally that cycle), and the result commits to ``dst`` at ``commit_cycle ==
    issue_cycle + latency`` (readable one cycle later, since the register file is read-first). Operators are fully
    pipelined, so one instance may carry several ops in flight; same-kind ops share a latency so two ops on one
    instance never commit on the same cycle.
    """

    inst: OperatorInstance
    a: Operand
    b: Operand | None  # None for a unary operator
    y_sgnop: Sgnop
    dst: RegRef
    issue_cycle: int
    latency: int

    @property
    def operands(self) -> list[Operand]:
        b = self.b
        return [self.a] if b is None else [self.a, b]

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
    """An output port driven from a register or constant immediate, with a folded output sign-op."""

    name: str
    source: RegRef | ConstRef
    sgnop: Sgnop


@dataclass(frozen=True, slots=True)
class RegFileLayout:
    nreg: int  # number of float registers (N)
    nrd: int  # combinational read ports
    nwr: int  # synchronous write ports
    nload: int  # immediate parallel-load lanes: registers 0..nload-1 are loaded from load_data at in_valid


@dataclass(frozen=True, slots=True)
class Lir:
    fmt: FloatFormat
    module_name: str
    instances: tuple[OperatorInstance, ...]
    consts: tuple[float, ...]  # constant pool: index -> value
    regfile: RegFileLayout
    inputs: tuple[InputLoad, ...]  # ordered as the function parameters
    ops: tuple[ScheduledOp, ...]  # the pipelined schedule, ordered by (issue_cycle, value-id)
    outputs: tuple[OutputWire, ...]
    makespan: int  # last commit cycle (0 if no ops); the in_valid->out_valid latency is makespan + 1
    op_count: int
    max_chain_len: int  # longest dependency chain in operators (for verification tolerance)

    @property
    def cyc_width(self) -> int:
        """Bit width of the cycle counter (and ``err_cyc``): enough to hold ``0..makespan+1``."""
        return max(1, (self.makespan + 1).bit_length())
