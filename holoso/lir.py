"""The low-level IR (LIR): the scheduled, bound, register-allocated microprogram for the synthesized ZISC machine.

A :class:`Lir` is controller-agnostic -- it describes which operators run in which step, reading/writing which
registers, with which folded sign-ops. The backend renders it to Verilog; this is the seam where a second controller
could be added.
"""

from __future__ import annotations

from dataclasses import dataclass

from .format import FloatFormat
from .operators import OpKind


@dataclass(frozen=True, slots=True)
class OperatorInstance:
    """One physical operator module, e.g. ``u_fadd_0``.

    ``fadd``/``fmul``/``fdiv`` instances are shared across steps (bound by step position). ``fmul_ilog2_const`` takes
    its exponent ``K`` as an elaboration-time parameter, so each such op gets a dedicated instance carrying its ``k``.
    """

    kind: OpKind
    index: int  # 0-based within its kind
    k: int | None = None  # elaboration-time exponent for FMUL_ILOG2 instances


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
    """An operator input: a register read or a constant immediate, with a folded 2-bit sign-op."""

    source: RegRef | ConstRef
    sgnop: int


@dataclass(frozen=True, slots=True)
class Issue:
    """One operator started in a step: its instance, operands, result sign-op, dest register, and div0 flag."""

    inst: OperatorInstance
    a: Operand
    b: Operand | None  # None for unary FMUL_ILOG2
    y_sgnop: int
    k: int | None  # exponent for FMUL_ILOG2
    dst: RegRef
    has_div0: bool


@dataclass(frozen=True, slots=True)
class Step:
    """One FSM step: a set of operators issued in parallel; the controller waits for all to complete (barrier)."""

    index: int
    issues: tuple[Issue, ...]
    latency: int  # max issued-operator latency -- the barrier wait, used only for the II estimate


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
    sgnop: int


@dataclass(frozen=True, slots=True)
class RegFileLayout:
    nreg: int  # number of float registers (N)
    nrd: int  # combinational read ports
    nwr: int  # synchronous write ports


@dataclass(frozen=True, slots=True)
class Lir:
    fmt: FloatFormat
    module_name: str
    instances: tuple[OperatorInstance, ...]
    consts: tuple[float, ...]  # constant pool: index -> value
    regfile: RegFileLayout
    inputs: tuple[InputLoad, ...]  # ordered as the function parameters
    steps: tuple[Step, ...]
    outputs: tuple[OutputWire, ...]
    op_count: int
    max_chain_len: int  # longest dependency chain in operators (for verification tolerance)
