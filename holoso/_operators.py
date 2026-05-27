"""Hardware operator models and folded floating-point sign controls."""

import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from ._type import FloatFormat, FloatType, ScalarSignature


def _instance_stem_text(text: str) -> str:
    return re.sub(r"[^0-9a-z_]+", "_", text.lower()).strip("_") or "x"


def _instance_stem_int(value: int) -> str:
    return f"m{-value}" if value < 0 else str(value)


def _hdl_param_stems(params: dict[str, int]) -> list[str]:
    return [f"{_instance_stem_text(name)}_{_instance_stem_int(value)}" for name, value in sorted(params.items())]


@dataclass(frozen=True, slots=True)
class FloatSignControl:
    """A hardware-side floating-point sign conditioner: absolute value first, then optional negation."""

    negate: bool = False
    absolute: bool = False

    def then(self, outer: "FloatSignControl") -> "FloatSignControl":
        """Compose two controls where ``self`` is applied first and ``outer`` after."""
        if outer.absolute:
            return FloatSignControl(negate=outer.negate, absolute=True)
        return FloatSignControl(negate=self.negate ^ outer.negate, absolute=self.absolute)

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
    error_ports: ClassVar[list[str]] = []

    @property
    def module_name(self) -> str:
        return f"holoso_{self.mnemonic}"

    @property
    def instance_stem(self) -> str:
        """
        Verilog-safe physical instance stem, unique for distinct hardware identities of this operator family.

        Override this if the operator's hardware identity is not fully captured by its mnemonic and HDL params.
        """
        return "_".join([_instance_stem_text(self.mnemonic), *_hdl_param_stems(self.hdl_params())])

    @property
    @abstractmethod
    def latency(self) -> int:
        """Exact cycle latency of this fully specified operator instance."""

    @abstractmethod
    def render(self, *operands: str) -> str:
        """Human-friendly expression for the report and trace comments."""

    @abstractmethod
    def hdl_params(self) -> dict[str, int]:
        """Operator-specific ``#(.NAME(v))`` params; the backend prepends ``WEXP``/``WMAN``."""

    @property
    @abstractmethod
    def signature(self) -> ScalarSignature:
        """Concrete operand/result types."""

    @property
    def arity(self) -> int:
        return self.signature.arity


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

    @property
    def instance_stem(self) -> str:
        return "_".join(
            [
                _instance_stem_text(self.mnemonic),
                f"e{self.fmt.wexp}",
                f"m{self.fmt.wman}",
                *_hdl_param_stems(self.hdl_params()),
            ]
        )

    def float_signature(self, arity: int) -> ScalarSignature:
        ty = FloatType(self.fmt)
        return ScalarSignature((ty,) * arity, ty)

    @abstractmethod
    def evaluate(self, *operands: float) -> float: ...


@dataclass(frozen=True, slots=True)
class FloatParameterizedHardwareOperator(ParameterizedHardwareOperator, ABC):
    """A floating-point operator family bound to one ZKF format."""

    fmt: FloatFormat


@dataclass(frozen=True, slots=True)
class FAddOperator(FloatHardwareOperator):
    mnemonic: ClassVar[str] = "fadd"
    stage_decode: int = 0
    stage_align: int = 0
    stage_output: int = 0

    def __post_init__(self) -> None:
        if self.stage_decode not in (0, 1):
            raise ValueError(f"stage_decode must be 0 or 1; got {self.stage_decode!r}")
        if self.stage_align not in (0, 1):
            raise ValueError(f"stage_align must be 0 or 1; got {self.stage_align!r}")
        if self.stage_output not in (0, 1):
            raise ValueError(f"stage_output must be 0 or 1; got {self.stage_output!r}")

    @property
    def latency(self) -> int:
        return 4 + self.stage_decode + self.stage_align + self.stage_output

    @property
    def signature(self) -> ScalarSignature:
        return self.float_signature(2)

    def evaluate(self, *operands: float) -> float:
        a, b = operands
        return a + b

    def render(self, *operands: str) -> str:
        a, b = operands
        return f"{a}+{b}"

    def hdl_params(self) -> dict[str, int]:
        return {"STAGE_DECODE": self.stage_decode, "STAGE_ALIGN": self.stage_align, "STAGE_OUTPUT": self.stage_output}


@dataclass(frozen=True, slots=True)
class FMulOperator(FloatHardwareOperator):
    mnemonic: ClassVar[str] = "fmul"
    stage_input: int = 0
    stage_product: int = 0
    stage_output: int = 0

    def __post_init__(self) -> None:
        if self.stage_input not in (0, 1):
            raise ValueError(f"stage_input must be 0 or 1; got {self.stage_input!r}")
        if self.stage_product not in (0, 1):
            raise ValueError(f"stage_product must be 0 or 1; got {self.stage_product!r}")
        if self.stage_output not in (0, 1):
            raise ValueError(f"stage_output must be 0 or 1; got {self.stage_output!r}")

    @property
    def latency(self) -> int:
        return 1 + self.stage_input + self.stage_product + self.stage_output

    @property
    def signature(self) -> ScalarSignature:
        return self.float_signature(2)

    def evaluate(self, *operands: float) -> float:
        a, b = operands
        return a * b

    def render(self, *operands: str) -> str:
        a, b = operands
        return f"{a}×{b}"

    def hdl_params(self) -> dict[str, int]:
        return {"STAGE_INPUT": self.stage_input, "STAGE_PRODUCT": self.stage_product, "STAGE_OUTPUT": self.stage_output}


@dataclass(frozen=True, slots=True)
class FDivOperator(FloatHardwareOperator):
    mnemonic: ClassVar[str] = "fdiv"
    error_ports: ClassVar[list[str]] = ["div0"]
    stage_input: int = 0
    stage_output: int = 0

    def __post_init__(self) -> None:
        if self.stage_input not in (0, 1):
            raise ValueError(f"stage_input must be 0 or 1; got {self.stage_input!r}")
        if self.stage_output not in (0, 1):
            raise ValueError(f"stage_output must be 0 or 1; got {self.stage_output!r}")

    @property
    def latency(self) -> int:
        w = self.fmt.wman
        return 2 + self.stage_input + ((w + 2 + ((w + 2) % 2)) // 2) + self.stage_output

    @property
    def signature(self) -> ScalarSignature:
        return self.float_signature(2)

    def evaluate(self, *operands: float) -> float:
        a, b = operands
        return a / b if b else math.copysign(math.inf, a)

    def render(self, *operands: str) -> str:
        a, b = operands
        return f"{a}/{b}"

    def hdl_params(self) -> dict[str, int]:
        return {"STAGE_INPUT": self.stage_input, "STAGE_OUTPUT": self.stage_output}


@dataclass(frozen=True, slots=True)
class FMulILog2Operator(FloatHardwareOperator):
    """Exact scaling by a power of two, ``a * 2**k``; the concrete operator the family returns."""

    mnemonic: ClassVar[str] = "fmul_ilog2_const"
    k: int
    stage_decode: int = 0

    def __post_init__(self) -> None:
        limit = 1 << (self.fmt.wexp - 1)
        if abs(self.k) >= limit:
            raise ValueError(f"k must satisfy |k| < {limit} for {self.fmt}; got {self.k!r}")
        if self.stage_decode not in (0, 1):
            raise ValueError(f"stage_decode must be 0 or 1; got {self.stage_decode!r}")

    @property
    def latency(self) -> int:
        return 1 + self.stage_decode

    @property
    def signature(self) -> ScalarSignature:
        return self.float_signature(1)

    def evaluate(self, *operands: float) -> float:
        (a,) = operands
        return math.ldexp(a, self.k)

    def render(self, *operands: str) -> str:
        (a,) = operands
        return f"{a}×2^{self.k}"

    def hdl_params(self) -> dict[str, int]:
        return {"K": self.k, "STAGE_DECODE": self.stage_decode}


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
