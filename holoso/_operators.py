"""Hardware operator models and folded floating-point sign controls."""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from hashlib import blake2s
from typing import ClassVar

from ._value import FloatValue, add_float_values, div_float_values, mul_float_values, mul_ilog2_float_value
from ._type import BoolType, FloatFormat, FloatType, ScalarSignature


def _instance_stem_text(text: str) -> str:
    return re.sub(r"[^0-9a-z_]+", "_", text.lower()).strip("_") or "x"


def _instance_stem_hash(params: dict[str, int]) -> str:
    payload = "\n".join(f"{name}={value}" for name, value in sorted(params.items())).encode("ascii")
    return blake2s(payload, digest_size=4).hexdigest()


def _hashed_instance_stem(mnemonic: str, params: dict[str, int]) -> str:
    return f"{_instance_stem_text(mnemonic)}_{_instance_stem_hash(params)}"


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

    def apply_value(self, value: FloatValue) -> FloatValue:
        return value.apply_sign(negate=self.negate, absolute=self.absolute)

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
    Commutative operators allow port assignment orient each use's operands to shrink the per-port read muxes.
    """

    mnemonic: ClassVar[str]
    error_ports: ClassVar[list[str]] = []
    is_commutative: ClassVar[bool] = False

    @property
    def module_name(self) -> str:
        return f"holoso_{self.mnemonic}"

    @property
    def instance_stem(self) -> str:
        """
        Verilog-safe physical instance stem, compactly identifying this operator family and its HDL params.
        Override this if the operator's hardware identity is not fully captured by its mnemonic and HDL params.
        """
        return _hashed_instance_stem(self.mnemonic, self.hdl_params())

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
        params = {"WEXP": self.fmt.wexp, "WMAN": self.fmt.wman}
        params.update(self.hdl_params())
        return _hashed_instance_stem(self.mnemonic, params)

    def float_signature(self, arity: int) -> ScalarSignature:
        ty = FloatType(self.fmt)
        return ScalarSignature((ty,) * arity, ty)

    def _validated_operands(self, operands: tuple[FloatValue, ...], arity: int) -> tuple[FloatValue, ...]:
        if len(operands) != arity:
            raise ValueError(f"{self.mnemonic} expected {arity} operands, got {len(operands)}")
        for index, operand in enumerate(operands):
            if not isinstance(operand, FloatValue):
                raise TypeError(f"{self.mnemonic} operand {index} must be FloatValue, got {type(operand).__name__}")
            if operand.fmt != self.fmt:
                raise ValueError(f"{self.mnemonic} operand {index} has {operand.fmt}, expected {self.fmt}")
        return operands

    @abstractmethod
    def evaluate(self, *operands: FloatValue) -> FloatValue: ...


@dataclass(frozen=True, slots=True)
class FloatParameterizedHardwareOperator(ParameterizedHardwareOperator, ABC):
    """A floating-point operator family bound to one ZKF format."""

    fmt: FloatFormat


@dataclass(frozen=True, slots=True)
class FAddOperator(FloatHardwareOperator):
    mnemonic: ClassVar[str] = "fadd"
    is_commutative: ClassVar[bool] = True  # signed sum: a+b == b+a bit-for-bit (each operand carries its own sign)
    stage_input: int = 0
    stage_decode: int = 0
    stage_align: int = 0
    stage_normalize: int = 0  # close-cancellation normshift barriers, 0..2 (forwarded to _zkf_normshift.STAGE_SPLIT)
    stage_pack: int = 0
    stage_output: int = 0

    def __post_init__(self) -> None:
        for field in ("stage_input", "stage_decode", "stage_align", "stage_pack", "stage_output"):
            if getattr(self, field) not in (0, 1):
                raise ValueError(f"{field} must be 0 or 1; got {getattr(self, field)!r}")
        if self.stage_normalize not in (0, 1, 2):
            raise ValueError(f"stage_normalize must be 0, 1, or 2; got {self.stage_normalize!r}")

    @property
    def latency(self) -> int:
        return (
            4
            + self.stage_input
            + self.stage_decode
            + self.stage_align
            + self.stage_normalize
            + self.stage_pack
            + self.stage_output
        )

    @property
    def signature(self) -> ScalarSignature:
        return self.float_signature(2)

    def evaluate(self, *operands: FloatValue) -> FloatValue:
        a, b = self._validated_operands(operands, 2)
        return add_float_values(a, b)

    def render(self, *operands: str) -> str:
        a, b = operands
        return f"{a}+{b}"

    def hdl_params(self) -> dict[str, int]:
        return {
            "STAGE_INPUT": self.stage_input,
            "STAGE_DECODE": self.stage_decode,
            "STAGE_ALIGN": self.stage_align,
            "STAGE_NORMALIZE": self.stage_normalize,
            "STAGE_PACK": self.stage_pack,
            "STAGE_OUTPUT": self.stage_output,
        }


@dataclass(frozen=True, slots=True)
class FMulOperator(FloatHardwareOperator):
    mnemonic: ClassVar[str] = "fmul"
    is_commutative: ClassVar[bool] = True  # product: a*b == b*a bit-for-bit
    stage_input: int = 0
    stage_product: int = 0
    stage_pack: int = 0
    stage_output: int = 0

    def __post_init__(self) -> None:
        for field in ("stage_input", "stage_product", "stage_pack", "stage_output"):
            if getattr(self, field) not in (0, 1):
                raise ValueError(f"{field} must be 0 or 1; got {getattr(self, field)!r}")

    @property
    def latency(self) -> int:
        return 1 + self.stage_input + self.stage_product + self.stage_pack + self.stage_output

    @property
    def signature(self) -> ScalarSignature:
        return self.float_signature(2)

    def evaluate(self, *operands: FloatValue) -> FloatValue:
        a, b = self._validated_operands(operands, 2)
        return mul_float_values(a, b)

    def render(self, *operands: str) -> str:
        a, b = operands
        return f"{a}×{b}"

    def hdl_params(self) -> dict[str, int]:
        return {
            "STAGE_INPUT": self.stage_input,
            "STAGE_PRODUCT": self.stage_product,
            "STAGE_PACK": self.stage_pack,
            "STAGE_OUTPUT": self.stage_output,
        }


@dataclass(frozen=True, slots=True)
class FDivOperator(FloatHardwareOperator):
    mnemonic: ClassVar[str] = "fdiv"
    error_ports: ClassVar[list[str]] = ["div0"]
    stage_input: int = 0
    stage_pack: int = 0
    stage_output: int = 0

    def __post_init__(self) -> None:
        for field in ("stage_input", "stage_pack", "stage_output"):
            if getattr(self, field) not in (0, 1):
                raise ValueError(f"{field} must be 0 or 1; got {getattr(self, field)!r}")

    @property
    def latency(self) -> int:
        w = self.fmt.wman
        return 2 + self.stage_input + ((w + 2 + ((w + 2) % 2)) // 2) + self.stage_pack + self.stage_output

    @property
    def signature(self) -> ScalarSignature:
        return self.float_signature(2)

    def evaluate(self, *operands: FloatValue) -> FloatValue:
        a, b = self._validated_operands(operands, 2)
        return div_float_values(a, b)

    def render(self, *operands: str) -> str:
        a, b = operands
        return f"{a}/{b}"

    def hdl_params(self) -> dict[str, int]:
        return {"STAGE_INPUT": self.stage_input, "STAGE_PACK": self.stage_pack, "STAGE_OUTPUT": self.stage_output}


@dataclass(frozen=True, slots=True)
class FMulILog2Operator(FloatHardwareOperator):
    """Exact scaling by a power of two, ``a * 2**k``; the concrete operator the family returns."""

    mnemonic: ClassVar[str] = "fmul_ilog2_const"
    k: int
    stage_input: int = 0
    stage_decode: int = 0

    def __post_init__(self) -> None:
        limit = (1 << self.fmt.wexp) - 2
        if self.k < -limit or self.k >= limit:
            raise ValueError(f"k must satisfy {-limit} <= k < {limit} for {self.fmt}; got {self.k!r}")
        for field in ("stage_input", "stage_decode"):
            if getattr(self, field) not in (0, 1):
                raise ValueError(f"{field} must be 0 or 1; got {getattr(self, field)!r}")

    @property
    def latency(self) -> int:
        return 1 + self.stage_input + self.stage_decode

    @property
    def signature(self) -> ScalarSignature:
        return self.float_signature(1)

    def evaluate(self, *operands: FloatValue) -> FloatValue:
        (a,) = self._validated_operands(operands, 1)
        return mul_ilog2_float_value(a, self.k)

    def render(self, *operands: str) -> str:
        (a,) = operands
        return f"{a}×2^{self.k}"

    def hdl_params(self) -> dict[str, int]:
        return {"K": self.k, "STAGE_INPUT": self.stage_input, "STAGE_DECODE": self.stage_decode}


@dataclass(frozen=True, slots=True)
class FMulILog2OperatorFamily(FloatParameterizedHardwareOperator):
    """The ilog2 family: a factory whose stage knobs are baked into every concrete operator it instantiates."""

    stage_input: int = 0
    stage_decode: int = 0

    def __post_init__(self) -> None:
        for field in ("stage_input", "stage_decode"):
            if getattr(self, field) not in (0, 1):
                raise ValueError(f"{field} must be 0 or 1; got {getattr(self, field)!r}")

    def instantiate(self, *params: int) -> FMulILog2Operator:
        (k,) = params
        return FMulILog2Operator(fmt=self.fmt, k=k, stage_input=self.stage_input, stage_decode=self.stage_decode)


@dataclass(frozen=True, slots=True)
class FCmpOperator(HardwareOperator):
    """
    A floating-point comparator: it produces the three mutually-exclusive one-hot order flags (a>b, a==b, a<b) with
    input sign conditioning. The specific relation (lt/le/...) is selected at the consuming boolean operation by a
    cheap reduction of the one-hot flags, so one comparator instance serves every relation.
    ZKF has no NaN, so for ZKF the ordering is total.
    """

    mnemonic: ClassVar[str] = "fcmp"
    fmt: FloatFormat
    stage_input: int = 0

    def __post_init__(self) -> None:
        if self.stage_input not in (0, 1):
            raise ValueError(f"stage_input must be 0 or 1; got {self.stage_input!r}")

    @property
    def instance_stem(self) -> str:
        params = {"WEXP": self.fmt.wexp, "WMAN": self.fmt.wman}
        params.update(self.hdl_params())
        return _hashed_instance_stem(self.mnemonic, params)

    @property
    def latency(self) -> int:
        return 1 + self.stage_input

    @property
    def signature(self) -> ScalarSignature:
        return ScalarSignature((FloatType(self.fmt), FloatType(self.fmt)), BoolType())

    def render(self, *operands: str) -> str:
        a, b = operands
        return f"cmp({a},{b})"

    def hdl_params(self) -> dict[str, int]:
        return {"STAGE_INPUT": self.stage_input}


@dataclass(frozen=True)
class OpConfig:
    """
    The hardware operator configuration threaded into synthesis. Constructed by the user before synthesis.
    Each field fixes one operator's format and parameters.
    """

    fadd: FAddOperator
    fmul: FMulOperator
    fdiv: FDivOperator
    fmul_ilog2: FMulILog2OperatorFamily
    fcmp: FCmpOperator

    @property
    def float_format(self) -> FloatFormat:
        formats = {self.fadd.fmt, self.fmul.fmt, self.fdiv.fmt, self.fmul_ilog2.fmt, self.fcmp.fmt}
        if len(formats) != 1:
            ordered = ", ".join(str(fmt) for fmt in sorted(formats, key=lambda fmt: (fmt.wexp, fmt.wman)))
            raise ValueError(f"all floating-point operators must use the same format; got {ordered}")
        return self.fadd.fmt
