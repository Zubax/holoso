"""
The operator model -- a class hierarchy whose instances are operators built from the synthesis configuration -- and
the sign-op encoding.
"""

import enum
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from .format import FloatFormat


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

    @abstractmethod
    def latency(self, fmt: FloatFormat) -> int: ...

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
    only its config-time knobs, not operator metadata -- the concrete :class:`Op` it produces owns that.
    """

    @abstractmethod
    def instantiate(self, *params: int) -> Op: ...


@dataclass(frozen=True, slots=True)
class FAddOp(Op):
    mnemonic: ClassVar[str] = "fadd"
    arity: ClassVar[int] = 2
    decode: int = 0
    align: int = 0

    def __post_init__(self) -> None:
        if self.decode not in (0, 1):
            raise ValueError(f"decode must be 0 or 1; got {self.decode!r}")
        if self.align not in (0, 1):
            raise ValueError(f"align must be 0 or 1; got {self.align!r}")

    def latency(self, fmt: FloatFormat) -> int:
        return 6 + self.decode + self.align

    def evaluate(self, *operands: float) -> float:
        a, b = operands
        return a + b

    def render(self, *operands: str) -> str:
        a, b = operands
        return f"{a}+{b}"

    def hdl_params(self) -> dict[str, int]:
        params: dict[str, int] = {}
        if self.decode:
            params["STAGE_DECODE"] = 1
        if self.align:
            params["STAGE_ALIGN"] = 1
        return params


@dataclass(frozen=True, slots=True)
class FMulOp(Op):
    mnemonic: ClassVar[str] = "fmul"
    arity: ClassVar[int] = 2
    product: int = 0

    def __post_init__(self) -> None:
        if self.product not in (0, 1):
            raise ValueError(f"product must be 0 or 1; got {self.product!r}")

    def latency(self, fmt: FloatFormat) -> int:
        return 3 + self.product

    def evaluate(self, *operands: float) -> float:
        a, b = operands
        return a * b

    def render(self, *operands: str) -> str:
        a, b = operands
        return f"{a}×{b}"

    def hdl_params(self) -> dict[str, int]:
        return {"STAGE_PRODUCT": 1} if self.product else {}


@dataclass(frozen=True, slots=True)
class FDivOp(Op):
    mnemonic: ClassVar[str] = "fdiv"
    arity: ClassVar[int] = 2
    error_ports: ClassVar[list[str]] = ["div0"]
    input_stage: int = 0

    def __post_init__(self) -> None:
        if self.input_stage not in (0, 1):
            raise ValueError(f"input_stage must be 0 or 1; got {self.input_stage!r}")

    def latency(self, fmt: FloatFormat) -> int:
        w = fmt.wman
        return 4 + ((w + 2 + ((w + 2) % 2)) // 2) + self.input_stage

    def evaluate(self, *operands: float) -> float:
        a, b = operands
        return a / b if b else math.copysign(math.inf, a)

    def render(self, *operands: str) -> str:
        a, b = operands
        return f"{a}/{b}"

    def hdl_params(self) -> dict[str, int]:
        return {"STAGE_INPUT": 1} if self.input_stage else {}


@dataclass(frozen=True, slots=True)
class FMulILog2Op(Op):
    """Exact scaling by a power of two, ``a * 2**k``; the concrete op the factory returns."""

    mnemonic: ClassVar[str] = "fmul_ilog2_const"
    arity: ClassVar[int] = 1
    k: int
    decode: int = 0

    def __post_init__(self) -> None:
        # k's range is format-dependent (|k| < 2**(WEXP-1)) and is enforced during lowering, not here.
        if self.decode not in (0, 1):
            raise ValueError(f"decode must be 0 or 1; got {self.decode!r}")

    def latency(self, fmt: FloatFormat) -> int:
        return 1 + self.decode

    def evaluate(self, *operands: float) -> float:
        (a,) = operands
        return math.ldexp(a, self.k)

    def render(self, *operands: str) -> str:
        (a,) = operands
        return f"{a}×2^{self.k}"

    def hdl_params(self) -> dict[str, int]:
        params: dict[str, int] = {"K": self.k}
        if self.decode:
            params["STAGE_DECODE"] = 1
        return params


@dataclass(frozen=True, slots=True)
class FMulILog2GenericOp(ParameterizedOp):
    """The ilog2 family: a factory whose ``decode`` knob is baked into every concrete op it instantiates."""

    decode: int = 0

    def __post_init__(self) -> None:
        if self.decode not in (0, 1):
            raise ValueError(f"decode must be 0 or 1; got {self.decode!r}")

    def instantiate(self, *params: int) -> FMulILog2Op:
        (k,) = params
        return FMulILog2Op(k=k, decode=self.decode)


# Order is load-bearing: it reproduces the operator-instance numbering the scheduler and backend emit.
ALL_OP_CLASSES: list[type[Op]] = [FAddOp, FMulOp, FDivOp, FMulILog2Op]


@dataclass(frozen=True)
class OpConfig:
    """
    The operator configuration threaded into synthesis; each field fixes one operator's parameters. Constructed
    explicitly by the caller (no defaults), held on the pipeline and never hashed.
    """

    fadd: FAddOp
    fmul: FMulOp
    fdiv: FDivOp
    fmul_ilog2: FMulILog2GenericOp
