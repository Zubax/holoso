"""
The operator model -- a class hierarchy whose instances are operators built from the synthesis configuration -- and
the sign-op encoding.
"""

import enum
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from ._type import FloatFormat


class Sgnop(enum.IntFlag):
    """
    Folded sign manipulation applied to an operator operand or output.

    A 2-bit field: bit 0 = negate, bit 1 = absolute value (so ``ABS | NEG`` means ``-|x|``). The integer values match
    ``HOLOSO_FSGNOP_*`` in ``holoso_support.vh`` (NONE=0, NEG=1, ABS=2, ABS|NEG=3), which is why ``IntFlag`` is used:
    ``int(op)`` yields the Verilog encoding while membership tests (``Sgnop.ABS in op``) express the bit semantics.
    """

    NONE = 0
    NEG = 1
    ABS = 2

    def decorate(self, text: str) -> str:
        """Wrap a value's name to show this sign-op: NEG -> ``-x``, ABS -> ``|x|``, ABS|NEG -> ``-|x|``."""
        if Sgnop.ABS in self:
            text = f"|{text}|"
        if Sgnop.NEG in self:
            text = f"-{text}"
        return text


# ----------------------------------------------------------------------------------------------------------------------
# Operator type hierarchy. Each operator is a frozen, equal-by-value instance constructed from the synthesis config.
# The fully-specified instance is itself the resource-sharing key, so equal ops time-share one physical module.
# Nothing downstream branches on operator identity.


class OperatorDef(ABC):
    """Kind-level metadata and naming common to the concrete operators."""

    mnemonic: ClassVar[str]
    arity: ClassVar[int]
    error_ports: ClassVar[list[str]] = []

    @property
    def module_name(self) -> str:
        return f"holoso_{self.mnemonic}"


@dataclass(frozen=True)
class Op(OperatorDef, ABC):
    """
    A fully-specified operator: one concrete module configuration. Frozen-dataclass equality makes the instance
    itself the resource-sharing key, so equal ops time-share one physical module. Every per-operator behavior is a
    method here; the uniform ``*operands`` signatures keep the abstract methods LSP-compatible while concrete bodies
    unpack per their arity. A field-less frozen-dataclass base lets the concrete ops (and the deterministic
    ``dataclasses.astuple`` sort over them) be recognized uniformly.
    """

    @property
    @abstractmethod
    def latency(self) -> int:
        """Exact cycle latency of this fully-specified operator instance."""

    @abstractmethod
    def evaluate(self, *operands: float) -> float: ...

    @abstractmethod
    def render(self, *operands: str) -> str:
        """Human-friendly expression for the report and trace comments (never parsed). Best to keep it compact."""

    @abstractmethod
    def hdl_params(self) -> dict[str, int]:
        """Operator-specific ``#(.NAME(v))`` params; the backend prepends ``WEXP``/``WMAN``."""


class ParameterizedOp(ABC):
    """
    A family of operators needing per-node parameters; a factory producing concrete :class:`Op` instances. It carries
    only config-time values, not operator metadata -- the concrete :class:`Op` it produces owns that.
    """

    @abstractmethod
    def instantiate(self, *params: int) -> Op: ...


@dataclass(frozen=True, slots=True)
class FloatOp(Op, ABC):
    """A fully-specified floating-point operator bound to one ZKF format."""

    fmt: FloatFormat


@dataclass(frozen=True, slots=True)
class FloatParameterizedOp(ParameterizedOp, ABC):
    """A floating-point operator family bound to one ZKF format."""

    fmt: FloatFormat


@dataclass(frozen=True, slots=True)
class FAddOp(FloatOp):
    mnemonic: ClassVar[str] = "fadd"
    arity: ClassVar[int] = 2
    stage_decode: int = 0
    stage_align: int = 0

    def __post_init__(self) -> None:
        if self.stage_decode not in (0, 1):
            raise ValueError(f"stage_decode must be 0 or 1; got {self.stage_decode!r}")
        if self.stage_align not in (0, 1):
            raise ValueError(f"stage_align must be 0 or 1; got {self.stage_align!r}")

    @property
    def latency(self) -> int:
        return 6 + self.stage_decode + self.stage_align

    def evaluate(self, *operands: float) -> float:
        a, b = operands
        return a + b

    def render(self, *operands: str) -> str:
        a, b = operands
        return f"{a}+{b}"

    def hdl_params(self) -> dict[str, int]:
        params: dict[str, int] = {}
        if self.stage_decode:
            params["STAGE_DECODE"] = 1
        if self.stage_align:
            params["STAGE_ALIGN"] = 1
        return params


@dataclass(frozen=True, slots=True)
class FMulOp(FloatOp):
    mnemonic: ClassVar[str] = "fmul"
    arity: ClassVar[int] = 2
    stage_product: int = 0

    def __post_init__(self) -> None:
        if self.stage_product not in (0, 1):
            raise ValueError(f"stage_product must be 0 or 1; got {self.stage_product!r}")

    @property
    def latency(self) -> int:
        return 3 + self.stage_product

    def evaluate(self, *operands: float) -> float:
        a, b = operands
        return a * b

    def render(self, *operands: str) -> str:
        a, b = operands
        return f"{a}×{b}"

    def hdl_params(self) -> dict[str, int]:
        return {"STAGE_PRODUCT": 1} if self.stage_product else {}


@dataclass(frozen=True, slots=True)
class FDivOp(FloatOp):
    mnemonic: ClassVar[str] = "fdiv"
    arity: ClassVar[int] = 2
    error_ports: ClassVar[list[str]] = ["div0"]
    stage_input: int = 0

    def __post_init__(self) -> None:
        if self.stage_input not in (0, 1):
            raise ValueError(f"stage_input must be 0 or 1; got {self.stage_input!r}")

    @property
    def latency(self) -> int:
        w = self.fmt.wman
        return 4 + ((w + 2 + ((w + 2) % 2)) // 2) + self.stage_input

    def evaluate(self, *operands: float) -> float:
        a, b = operands
        return a / b if b else math.copysign(math.inf, a)

    def render(self, *operands: str) -> str:
        a, b = operands
        return f"{a}/{b}"

    def hdl_params(self) -> dict[str, int]:
        return {"STAGE_INPUT": 1} if self.stage_input else {}


@dataclass(frozen=True, slots=True)
class FMulILog2Op(FloatOp):
    """Exact scaling by a power of two, ``a * 2**k``; the concrete op the factory returns."""

    mnemonic: ClassVar[str] = "fmul_ilog2_const"
    arity: ClassVar[int] = 1
    k: int
    stage_decode: int = 0

    def __post_init__(self) -> None:
        # k's range is format-dependent (|k| < 2**(WEXP-1)) and is enforced during lowering, not here.
        if self.stage_decode not in (0, 1):
            raise ValueError(f"stage_decode must be 0 or 1; got {self.stage_decode!r}")

    @property
    def latency(self) -> int:
        return 1 + self.stage_decode

    def evaluate(self, *operands: float) -> float:
        (a,) = operands
        return math.ldexp(a, self.k)

    def render(self, *operands: str) -> str:
        (a,) = operands
        return f"{a}×2^{self.k}"

    def hdl_params(self) -> dict[str, int]:
        params: dict[str, int] = {"K": self.k}
        if self.stage_decode:
            params["STAGE_DECODE"] = 1
        return params


@dataclass(frozen=True, slots=True)
class FMulILog2GenericOp(FloatParameterizedOp):
    """The ilog2 family: a factory whose stage knob is baked into every concrete op it instantiates."""

    stage_decode: int = 0

    def __post_init__(self) -> None:
        if self.stage_decode not in (0, 1):
            raise ValueError(f"stage_decode must be 0 or 1; got {self.stage_decode!r}")

    def instantiate(self, *params: int) -> FMulILog2Op:
        (k,) = params
        return FMulILog2Op(fmt=self.fmt, k=k, stage_decode=self.stage_decode)


# Order is load-bearing: it reproduces the operator-instance numbering the scheduler and backend emit.
ALL_OP_CLASSES: list[type[Op]] = [FAddOp, FMulOp, FDivOp, FMulILog2Op]


@dataclass(frozen=True)
class OpConfig:
    """
    The operator configuration threaded into synthesis; each field fixes one operator's format and parameters.
    Constructed explicitly by the caller (no defaults), held on the pipeline and never hashed.
    """

    fadd: FAddOp
    fmul: FMulOp
    fdiv: FDivOp
    fmul_ilog2: FMulILog2GenericOp

    @property
    def float_format(self) -> FloatFormat:
        formats = {self.fadd.fmt, self.fmul.fmt, self.fdiv.fmt, self.fmul_ilog2.fmt}
        if len(formats) != 1:
            ordered = ", ".join(str(fmt) for fmt in sorted(formats, key=lambda fmt: (fmt.wexp, fmt.wman)))
            raise ValueError(f"all floating-point operators must use the same format; got {ordered}")
        return self.fadd.fmt
