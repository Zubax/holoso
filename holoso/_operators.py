"""Hardware operator models and folded sign controls."""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from ._type import FloatFormat

# ----------------------------------------------------------------------------------------------------------------------
# Sign controls and hardware operators: selected, fully configured resources.


@dataclass(frozen=True, slots=True)
class SignControl:
    """A hardware-side sign conditioner: absolute value first, then optional negation."""

    negate: bool = False
    absolute: bool = False

    def then(self, outer: "SignControl") -> "SignControl":
        """Compose two controls where ``self`` is applied first and ``outer`` after."""
        if outer.absolute:
            return SignControl(negate=outer.negate, absolute=True)
        return SignControl(negate=self.negate ^ outer.negate, absolute=self.absolute)

    def apply_float(self, value: float) -> float:
        if self.absolute:
            value = abs(value)
        if self.negate:
            value = -value
        return value

    def decorate(self, text: str) -> str:
        if self.absolute:
            text = f"|{text}|"
        if self.negate:
            text = f"-{text}"
        return text

    @property
    def encoded(self) -> int:
        return (1 if self.negate else 0) | (2 if self.absolute else 0)


@dataclass(frozen=True)
class HardwareOperator(ABC):
    """
    A fully specified hardware operator configuration.

    Frozen-dataclass equality makes an instance the resource-sharing key: equal operators time-share one physical
    module. Each concrete operator owns its timing, reference semantics, notation, and HDL parameters.
    """

    mnemonic: ClassVar[str]
    arity: ClassVar[int]
    error_ports: ClassVar[list[str]] = []

    @property
    def module_name(self) -> str:
        return f"holoso_{self.mnemonic}"

    @property
    @abstractmethod
    def latency(self) -> int:
        """Exact cycle latency of this fully specified operator instance."""

    @abstractmethod
    def evaluate(self, *operands: float) -> float: ...

    @abstractmethod
    def render(self, *operands: str) -> str:
        """Human-friendly expression for the report and trace comments."""

    @abstractmethod
    def hdl_params(self) -> dict[str, int]:
        """Operator-specific ``#(.NAME(v))`` params; the backend prepends ``WEXP``/``WMAN``."""


class ParameterizedHardwareOperator(ABC):
    """
    A family of hardware operators needing per-node parameters.

    It carries only config-time values; the concrete :class:`HardwareOperator` it produces owns the hardware metadata.
    """

    @abstractmethod
    def instantiate(self, *params: int) -> HardwareOperator: ...


@dataclass(frozen=True, slots=True)
class FloatHardwareOperator(HardwareOperator, ABC):
    """A fully specified floating-point operator bound to one ZKF format."""

    fmt: FloatFormat


@dataclass(frozen=True, slots=True)
class FloatParameterizedHardwareOperator(ParameterizedHardwareOperator, ABC):
    """A floating-point operator family bound to one ZKF format."""

    fmt: FloatFormat


@dataclass(frozen=True, slots=True)
class FAddOperator(FloatHardwareOperator):
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
class FMulOperator(FloatHardwareOperator):
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
class FDivOperator(FloatHardwareOperator):
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
class FMulILog2Operator(FloatHardwareOperator):
    """Exact scaling by a power of two, ``a * 2**k``; the concrete operator the family returns."""

    mnemonic: ClassVar[str] = "fmul_ilog2_const"
    arity: ClassVar[int] = 1
    k: int
    stage_decode: int = 0

    def __post_init__(self) -> None:
        # k's range is format-dependent (|k| < 2**(WEXP-1)) and is enforced by HIR-to-MIR lowering.
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
class FMulILog2OperatorFamily(FloatParameterizedHardwareOperator):
    """The ilog2 family: a factory whose stage knob is baked into every concrete operator it instantiates."""

    stage_decode: int = 0

    def __post_init__(self) -> None:
        if self.stage_decode not in (0, 1):
            raise ValueError(f"stage_decode must be 0 or 1; got {self.stage_decode!r}")

    def instantiate(self, *params: int) -> FMulILog2Operator:
        (k,) = params
        return FMulILog2Operator(fmt=self.fmt, k=k, stage_decode=self.stage_decode)


# Order is load-bearing: it reproduces the operator-instance numbering the scheduler and backend emit.
ALL_OPERATOR_CLASSES: list[type[HardwareOperator]] = [
    FAddOperator,
    FMulOperator,
    FDivOperator,
    FMulILog2Operator,
]


@dataclass(frozen=True)
class OpConfig:
    """
    The hardware operator configuration threaded into synthesis.

    Constructed explicitly by the caller (no defaults), held on the pipeline and never hashed. Each field fixes one
    operator's format and parameters.
    """

    fadd: FAddOperator
    fmul: FMulOperator
    fdiv: FDivOperator
    fmul_ilog2: FMulILog2OperatorFamily

    @property
    def float_format(self) -> FloatFormat:
        formats = {self.fadd.fmt, self.fmul.fmt, self.fdiv.fmt, self.fmul_ilog2.fmt}
        if len(formats) != 1:
            ordered = ", ".join(str(fmt) for fmt in sorted(formats, key=lambda fmt: (fmt.wexp, fmt.wman)))
            raise ValueError(f"all floating-point operators must use the same format; got {ordered}")
        return self.fadd.fmt
