"""
The float operators. Each delegates its timing and its reference arithmetic to the external ZKF library.
An operator whose format parameterizes it is a float one whatever else it touches, so the casts across the type
boundary live here too.
"""

from abc import ABC
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import ClassVar

import zkf

from .._value import FloatValue, IntValue, RoundMode, ScalarValue
from .._type import BoolType, FloatFormat, FloatType, IntFormat, IntType
from ._common import (
    ComparatorOperator,
    FloatSignControl,
    ImmediateField,
    InlineHardwareOperator,
    ParameterizedHardwareOperator,
    PooledHardwareOperator,
    PortConditioner,
    ScalarSignature,
)

_ROUND_LABEL: dict[RoundMode, str] = {
    RoundMode.NEAREST_EVEN: "round",
    RoundMode.FLOOR: "floor",
    RoundMode.CEIL: "ceil",
    RoundMode.TRUNC: "trunc",
}
"""Rendered into the report and the ROM comments, so it is pinned here rather than taken from zkf's member names."""


@dataclass(frozen=True, slots=True)
class ZkfBackedOperator(PooledHardwareOperator, ABC):
    """The float format parameterizes every ZKF-backed module, including one whose operand or result is an integer."""

    fmt: FloatFormat

    _model: zkf.OperatorModel = field(init=False, compare=False, repr=False)

    @property
    def latency(self) -> int:
        return self._model.latency

    @property
    def initiation_interval(self) -> int:
        return self._model.initiation_interval

    @property
    def params(self) -> dict[str, int]:
        return self._model.params


@dataclass(frozen=True, slots=True)
class FloatHardwareOperator(ZkfBackedOperator, ABC):
    """Float on both sides, which is what makes the narrowed operand validator below sound."""

    @property
    def scalar_type(self) -> FloatType:
        return FloatType(self.fmt)

    def _validated_operands(self, operands: tuple[ScalarValue, ...]) -> tuple[FloatValue, ...]:
        validated: list[FloatValue] = []
        for operand in super()._validated_operands(operands):
            assert isinstance(operand, FloatValue)
            validated.append(operand)
        return tuple(validated)


@dataclass(frozen=True, slots=True)
class FAddOperator(FloatHardwareOperator):
    @dataclass(frozen=True, slots=True)
    class Options:
        stage_input: int = 0  # takes any count of input register stages (extra stages relieve routing congestion)
        stage_decode: int = 0
        stage_align: int = 0
        stage_normalize: int = 0
        stage_pack: int = 0
        stage_output: int = 0

    mnemonic: ClassVar[str] = "fadd"
    operand_hdl_ports: ClassVar[list[str]] = ["a", "b"]
    output_hdl_ports: ClassVar[list[str]] = ["y"]
    swap_output_permutation: ClassVar[tuple[int, ...]] = (0,)  # signed sum: a+b == b+a bit-for-bit
    opt: Options

    def __post_init__(self) -> None:
        model = zkf.AddModel(
            zkf.ZkfFormat(self.fmt.wexp, self.fmt.wman),
            stage_input=self.opt.stage_input,
            stage_decode=self.opt.stage_decode,
            stage_align=self.opt.stage_align,
            stage_normalize=self.opt.stage_normalize,
            stage_pack=self.opt.stage_pack,
            stage_output=self.opt.stage_output,
        )
        object.__setattr__(self, "_model", model)

    @property
    def signature(self) -> ScalarSignature:
        return ScalarSignature((self.scalar_type,) * 2, (self.scalar_type,))

    def evaluate(self, *operands: ScalarValue, immediates: tuple[int, ...] = ()) -> tuple[FloatValue, ...]:
        a, b = self._validated_operands(operands)
        return (a + b,)

    def render(self, *operands: str, immediates: tuple[int, ...] = ()) -> str:
        a, b = operands
        return f"{a}+{b}"


@dataclass(frozen=True, slots=True)
class FMulOperator(FloatHardwareOperator):
    @dataclass(frozen=True, slots=True)
    class Options:
        stage_input: int = 0
        stage_product: int = 0  # splitting the product is rarely useful unless wman exceeds the DSP slice input width
        stage_pack: int = 0
        stage_output: int = 0

    mnemonic: ClassVar[str] = "fmul"
    operand_hdl_ports: ClassVar[list[str]] = ["a", "b"]
    output_hdl_ports: ClassVar[list[str]] = ["y"]
    swap_output_permutation: ClassVar[tuple[int, ...]] = (0,)  # product: a*b == b*a bit-for-bit
    opt: Options
    wmultiplier: int

    def __post_init__(self) -> None:
        model = zkf.MulModel(
            zkf.ZkfFormat(self.fmt.wexp, self.fmt.wman),
            wmultiplier=self.wmultiplier,
            stage_input=self.opt.stage_input,
            stage_product=self.opt.stage_product,
            stage_pack=self.opt.stage_pack,
            stage_output=self.opt.stage_output,
        )
        object.__setattr__(self, "_model", model)

    @property
    def signature(self) -> ScalarSignature:
        return ScalarSignature((self.scalar_type,) * 2, (self.scalar_type,))

    def evaluate(self, *operands: ScalarValue, immediates: tuple[int, ...] = ()) -> tuple[FloatValue, ...]:
        a, b = self._validated_operands(operands)
        return (a * b,)

    def render(self, *operands: str, immediates: tuple[int, ...] = ()) -> str:
        a, b = operands
        return f"{a}×{b}"


@dataclass(frozen=True, slots=True)
class FDivOperator(FloatHardwareOperator):
    @dataclass(frozen=True, slots=True)
    class Options:
        stage_input: int = 0
        stage_pack: int = 0
        stage_output: int = 0

    mnemonic: ClassVar[str] = "fdiv"
    operand_hdl_ports: ClassVar[list[str]] = ["a", "b"]
    output_hdl_ports: ClassVar[list[str]] = ["y"]
    error_ports: ClassVar[list[str]] = ["div0"]
    opt: Options

    def __post_init__(self) -> None:
        model = zkf.DivModel(
            zkf.ZkfFormat(self.fmt.wexp, self.fmt.wman),
            stage_input=self.opt.stage_input,
            stage_pack=self.opt.stage_pack,
            stage_output=self.opt.stage_output,
        )
        object.__setattr__(self, "_model", model)

    @property
    def signature(self) -> ScalarSignature:
        return ScalarSignature((self.scalar_type,) * 2, (self.scalar_type,))

    def evaluate(self, *operands: ScalarValue, immediates: tuple[int, ...] = ()) -> tuple[FloatValue, ...]:
        a, b = self._validated_operands(operands)
        return (a / b,)

    def render(self, *operands: str, immediates: tuple[int, ...] = ()) -> str:
        a, b = operands
        return f"{a}/{b}"


@dataclass(frozen=True, slots=True)
class FMulILog2Operator(FloatHardwareOperator):
    """Exact scaling by a power of two, ``a * 2**k``; the concrete operator the family returns."""

    @dataclass(frozen=True, slots=True)
    class Options:
        stage_input: int = 0
        stage_decode: int = 0

    mnemonic: ClassVar[str] = "fmul_ilog2_const"
    operand_hdl_ports: ClassVar[list[str]] = ["a"]
    output_hdl_ports: ClassVar[list[str]] = ["y"]
    k: int
    opt: Options

    def __post_init__(self) -> None:
        model = zkf.MulIlog2ConstModel(
            zkf.ZkfFormat(self.fmt.wexp, self.fmt.wman),
            k=self.k,
            stage_input=self.opt.stage_input,
            stage_decode=self.opt.stage_decode,
        )
        object.__setattr__(self, "_model", model)

    @property
    def signature(self) -> ScalarSignature:
        return ScalarSignature((self.scalar_type,) * 1, (self.scalar_type,))

    def evaluate(self, *operands: ScalarValue, immediates: tuple[int, ...] = ()) -> tuple[FloatValue, ...]:
        (a,) = self._validated_operands(operands)
        return (a.scale_pow2(self.k),)

    def render(self, *operands: str, immediates: tuple[int, ...] = ()) -> str:
        (a,) = operands
        return f"{a}×2^{self.k}"


@dataclass(frozen=True, slots=True)
class FMulILog2OperatorFamily(ParameterizedHardwareOperator):
    """The ilog2 family: a factory whose stage knobs are baked into every concrete operator it instantiates."""

    fmt: FloatFormat
    opt: FMulILog2Operator.Options

    def instantiate(self, *params: int) -> FMulILog2Operator:
        (k,) = params
        return FMulILog2Operator(fmt=self.fmt, k=k, opt=self.opt)


@dataclass(frozen=True, slots=True)
class FCmpOperator(FloatHardwareOperator, ComparatorOperator):
    """ZKF has no NaN, so the ordering is total."""

    @dataclass(frozen=True, slots=True)
    class Options:
        stage_input: int = 0

    mnemonic: ClassVar[str] = "fcmp"
    opt: Options

    def __post_init__(self) -> None:
        model = zkf.CmpModel(zkf.ZkfFormat(self.fmt.wexp, self.fmt.wman), stage_input=self.opt.stage_input)
        object.__setattr__(self, "_model", model)

    def evaluate(self, *operands: ScalarValue, immediates: tuple[int, ...] = ()) -> tuple[bool, ...]:
        a, b = self._validated_operands(operands)
        ordering = a.compare(b)
        return ordering > 0, ordering == 0, ordering < 0


@dataclass(frozen=True, slots=True)
class FRoundOperator(FloatHardwareOperator):
    """
    Round a float to an integral-valued float. One pooled instance serves all four modes (nearest-even, floor, ceil,
    trunc) via the 2-bit ``round_mode`` immediate, as one comparator serves every relation.
    """

    @dataclass(frozen=True, slots=True)
    class Options:
        """The zkf core is combinational, hence the nonzero default: a pooled operator needs latency >= 1."""

        stage_input: int = 1
        stage_decode: int = 0
        stage_pack: int = 0
        stage_output: int = 0

    mnemonic: ClassVar[str] = "fround"
    operand_hdl_ports: ClassVar[list[str]] = ["a"]
    output_hdl_ports: ClassVar[list[str]] = ["y"]
    immediate_ports: ClassVar[list[ImmediateField]] = [ImmediateField("round_mode", 2)]
    opt: Options

    _EVAL: ClassVar[dict[RoundMode, Callable[[FloatValue], FloatValue]]] = {
        RoundMode.NEAREST_EVEN: FloatValue.round,
        RoundMode.FLOOR: FloatValue.floor,
        RoundMode.CEIL: FloatValue.ceil,
        RoundMode.TRUNC: FloatValue.trunc,
    }

    def __post_init__(self) -> None:
        model = zkf.RoundModel(
            zkf.ZkfFormat(self.fmt.wexp, self.fmt.wman),
            stage_input=self.opt.stage_input,
            stage_decode=self.opt.stage_decode,
            stage_pack=self.opt.stage_pack,
            stage_output=self.opt.stage_output,
        )
        object.__setattr__(self, "_model", model)
        if self.latency < 1:
            raise ValueError("fround needs at least one register stage (a pooled operator must have latency >= 1)")

    @property
    def signature(self) -> ScalarSignature:
        return ScalarSignature((self.scalar_type,) * 1, (self.scalar_type,))

    def evaluate(self, *operands: ScalarValue, immediates: tuple[int, ...] = ()) -> tuple[FloatValue, ...]:
        (a,) = self._validated_operands(operands)
        (mode,) = immediates
        return (self._EVAL[RoundMode(mode)](a),)

    def render(self, *operands: str, immediates: tuple[int, ...] = ()) -> str:
        (a,) = operands
        (mode,) = immediates
        return f"{_ROUND_LABEL[RoundMode(mode)]}({a})"


@dataclass(frozen=True, slots=True)
class FFmaOperator(FloatHardwareOperator):
    """
    Fused multiply-add ``a*b + c``, single-rounded (full-width product rounded once with ``c``). Arity 3; serves the
    explicit ``math.fma`` and the implicit ``a*b+c`` fusion. Not commutative under operand reversal (gives ``c*b+a``).
    """

    @dataclass(frozen=True, slots=True)
    class Options:
        stage_input: int = 0
        stage_product: int = 0
        stage_decode: int = 0
        stage_align: int = 0
        stage_normalize: int = 0
        stage_pack: int = 0
        stage_output: int = 0

    mnemonic: ClassVar[str] = "ffma"
    operand_hdl_ports: ClassVar[list[str]] = ["a", "b", "c"]
    output_hdl_ports: ClassVar[list[str]] = ["y"]
    opt: Options
    wmultiplier: int

    def __post_init__(self) -> None:
        model = zkf.FmaModel(
            zkf.ZkfFormat(self.fmt.wexp, self.fmt.wman),
            wmultiplier=self.wmultiplier,
            stage_input=self.opt.stage_input,
            stage_product=self.opt.stage_product,
            stage_decode=self.opt.stage_decode,
            stage_align=self.opt.stage_align,
            stage_normalize=self.opt.stage_normalize,
            stage_pack=self.opt.stage_pack,
            stage_output=self.opt.stage_output,
        )
        object.__setattr__(self, "_model", model)

    @property
    def signature(self) -> ScalarSignature:
        return ScalarSignature((self.scalar_type,) * 3, (self.scalar_type,))

    def evaluate(self, *operands: ScalarValue, immediates: tuple[int, ...] = ()) -> tuple[FloatValue, ...]:
        a, b, c = self._validated_operands(operands)
        return (FloatValue.fma(a, b, c),)

    def render(self, *operands: str, immediates: tuple[int, ...] = ()) -> str:
        a, b, c = operands
        return f"{a}×{b}+{c}"


@dataclass(frozen=True, slots=True)
class FSortOperator(FloatHardwareOperator):
    """
    A 2-element float sorter emitting the ascending ``(min, max)`` of its operands, with input and per-output sign
    conditioning. ``min(a,b)`` taps port 0 and ``max(a,b)`` port 1; one instance serves both, and a min and a max over
    one operand pair fuse into a single firing (as the comparator's relations do).
    NOT commutative: min/max preserve the selected operand's exact bits, and the sorter breaks a tie toward the second
    operand, so swapping operands can flip the sign of a zero result (a -0 conditioned from a zero magnitude).
    """

    @dataclass(frozen=True, slots=True)
    class Options:
        stage_input: int = 0

    mnemonic: ClassVar[str] = "fsort"
    operand_hdl_ports: ClassVar[list[str]] = ["a", "b"]
    output_hdl_ports: ClassVar[list[str]] = ["min", "max"]
    opt: Options

    def __post_init__(self) -> None:
        model = zkf.SortModel(zkf.ZkfFormat(self.fmt.wexp, self.fmt.wman), stage_input=self.opt.stage_input)
        object.__setattr__(self, "_model", model)

    @property
    def signature(self) -> ScalarSignature:
        return ScalarSignature((self.scalar_type,) * 2, (self.scalar_type,) * 2)

    def evaluate(self, *operands: ScalarValue, immediates: tuple[int, ...] = ()) -> tuple[FloatValue, ...]:
        a, b = self._validated_operands(operands)
        return FloatValue.sort(a, b)

    def render(self, *operands: str, immediates: tuple[int, ...] = ()) -> str:
        a, b = operands
        return f"{self.mnemonic}({a},{b})"

    def render_output(
        self, port: int, conditioner: PortConditioner, *operands: str, immediates: tuple[int, ...] = ()
    ) -> str:
        assert isinstance(conditioner, FloatSignControl)
        a, b = operands
        return conditioner.decorate(f"{self.output_hdl_ports[port]}({a}, {b})")


@dataclass(frozen=True, slots=True)
class FExp2Operator(FloatHardwareOperator):
    @dataclass(frozen=True, slots=True)
    class Options:
        stage_input: int = 0
        stage_reduce: int = 0
        stage_product: int = 0
        stage_pack: int = 0
        stage_output: int = 0

    mnemonic: ClassVar[str] = "fexp2"
    operand_hdl_ports: ClassVar[list[str]] = ["a"]
    output_hdl_ports: ClassVar[list[str]] = ["y"]
    opt: Options
    wmultiplier: int

    def __post_init__(self) -> None:
        model = zkf.Exp2Model(
            zkf.ZkfFormat(self.fmt.wexp, self.fmt.wman),
            wmultiplier=self.wmultiplier,
            stage_input=self.opt.stage_input,
            stage_reduce=self.opt.stage_reduce,
            stage_product=self.opt.stage_product,
            stage_pack=self.opt.stage_pack,
            stage_output=self.opt.stage_output,
        )
        object.__setattr__(self, "_model", model)

    @property
    def signature(self) -> ScalarSignature:
        return ScalarSignature((self.scalar_type,) * 1, (self.scalar_type,))

    def evaluate(self, *operands: ScalarValue, immediates: tuple[int, ...] = ()) -> tuple[FloatValue, ...]:
        (a,) = self._validated_operands(operands)
        return (a.exp2(),)

    def render(self, *operands: str, immediates: tuple[int, ...] = ()) -> str:
        (a,) = operands
        return f"2^{a}"


@dataclass(frozen=True, slots=True)
class FLog2Operator(FloatHardwareOperator):
    @dataclass(frozen=True, slots=True)
    class Options:
        stage_input: int = 0
        stage_decode: int = 0
        stage_product: int = 0
        stage_product_final: int = 0
        stage_normalize: int = 0
        stage_normalize_output: int = 0
        stage_pack: int = 0
        stage_output: int = 0

    mnemonic: ClassVar[str] = "flog2"
    operand_hdl_ports: ClassVar[list[str]] = ["a"]
    output_hdl_ports: ClassVar[list[str]] = ["y"]
    error_ports: ClassVar[list[str]] = ["domain_error", "pole"]
    opt: Options
    wmultiplier: int

    def __post_init__(self) -> None:
        model = zkf.Log2Model(
            zkf.ZkfFormat(self.fmt.wexp, self.fmt.wman),
            wmultiplier=self.wmultiplier,
            stage_input=self.opt.stage_input,
            stage_decode=self.opt.stage_decode,
            stage_product=self.opt.stage_product,
            stage_product_final=self.opt.stage_product_final,
            stage_normalize=self.opt.stage_normalize,
            stage_normalize_output=self.opt.stage_normalize_output,
            stage_pack=self.opt.stage_pack,
            stage_output=self.opt.stage_output,
        )
        object.__setattr__(self, "_model", model)

    @property
    def signature(self) -> ScalarSignature:
        return ScalarSignature((self.scalar_type,) * 1, (self.scalar_type,))

    def evaluate(self, *operands: ScalarValue, immediates: tuple[int, ...] = ()) -> tuple[FloatValue, ...]:
        (a,) = self._validated_operands(operands)
        return (a.log2(),)

    def render(self, *operands: str, immediates: tuple[int, ...] = ()) -> str:
        (a,) = operands
        return f"log2({a})"


@dataclass(frozen=True, slots=True)
class FSincosOperator(FloatHardwareOperator):
    """NOT throughput-1: the core holds one transaction in flight and re-accepts one cycle after retiring."""

    @dataclass(frozen=True, slots=True)
    class Options:
        unroll100: int = 100
        stage_input: int = 0
        stage_product: int = 0
        stage_normalize: int = 0
        stage_pack: int = 0
        stage_output: int = 0

    mnemonic: ClassVar[str] = "fsincos"
    operand_hdl_ports: ClassVar[list[str]] = ["a"]
    output_hdl_ports: ClassVar[list[str]] = ["sin", "cos"]
    opt: Options
    wmultiplier: int

    def __post_init__(self) -> None:
        model = zkf.SincosModel(
            zkf.ZkfFormat(self.fmt.wexp, self.fmt.wman),
            wmultiplier=self.wmultiplier,
            unroll100=self.opt.unroll100,
            stage_input=self.opt.stage_input,
            stage_product=self.opt.stage_product,
            stage_normalize=self.opt.stage_normalize,
            stage_pack=self.opt.stage_pack,
            stage_output=self.opt.stage_output,
        )
        object.__setattr__(self, "_model", model)

    @property
    def signature(self) -> ScalarSignature:
        return ScalarSignature((self.scalar_type,) * 1, (self.scalar_type,) * 2)

    def evaluate(self, *operands: ScalarValue, immediates: tuple[int, ...] = ()) -> tuple[FloatValue, ...]:
        (a,) = self._validated_operands(operands)
        return a.sincos()

    def render(self, *operands: str, immediates: tuple[int, ...] = ()) -> str:
        (a,) = operands
        return f"sincos({a})"

    def render_output(
        self, port: int, conditioner: PortConditioner, *operands: str, immediates: tuple[int, ...] = ()
    ) -> str:
        assert isinstance(conditioner, FloatSignControl)
        return conditioner.decorate(f"{self.output_hdl_ports[port]}({', '.join(operands)})")


@dataclass(frozen=True, slots=True)
class FAtan2Operator(FloatHardwareOperator):
    """NOT throughput-1: the core holds one transaction in flight and re-accepts one cycle after retiring."""

    @dataclass(frozen=True, slots=True)
    class Options:
        """A nearby hypot over the same operands folds into the magnitude port for free."""

        unroll100: int = 100
        stage_input: int = 0
        stage_product: int = 0
        stage_normalize: int = 0
        stage_pack: int = 0
        stage_output: int = 0

    mnemonic: ClassVar[str] = "fatan2"
    operand_hdl_ports: ClassVar[list[str]] = ["a", "b"]
    output_hdl_ports: ClassVar[list[str]] = ["theta", "mag"]
    opt: Options
    wmultiplier: int

    def __post_init__(self) -> None:
        model = zkf.Atan2Model(
            zkf.ZkfFormat(self.fmt.wexp, self.fmt.wman),
            wmultiplier=self.wmultiplier,
            unroll100=self.opt.unroll100,
            stage_input=self.opt.stage_input,
            stage_product=self.opt.stage_product,
            stage_normalize=self.opt.stage_normalize,
            stage_pack=self.opt.stage_pack,
            stage_output=self.opt.stage_output,
        )
        object.__setattr__(self, "_model", model)

    @property
    def signature(self) -> ScalarSignature:
        return ScalarSignature((self.scalar_type,) * 2, (self.scalar_type,) * 2)

    def evaluate(self, *operands: ScalarValue, immediates: tuple[int, ...] = ()) -> tuple[FloatValue, ...]:
        y, x = self._validated_operands(operands)
        return FloatValue.atan2(y, x)

    def render(self, *operands: str, immediates: tuple[int, ...] = ()) -> str:
        y, x = operands
        return f"atan2({y},{x})"

    def render_output(
        self, port: int, conditioner: PortConditioner, *operands: str, immediates: tuple[int, ...] = ()
    ) -> str:
        assert isinstance(conditioner, FloatSignControl)
        return conditioner.decorate(f"{self.output_hdl_ports[port]}({', '.join(operands)})")


@dataclass(frozen=True, slots=True)
class FloatClassificationOperator(InlineHardwareOperator, ABC):
    fmt: FloatFormat

    @property
    def signature(self) -> ScalarSignature:
        return ScalarSignature((FloatType(self.fmt),), (BoolType(),))


@dataclass(frozen=True, slots=True)
class FloatIsFiniteOperator(FloatClassificationOperator):
    mnemonic: ClassVar[str] = "fisfinite"

    def render(self, *operands: str, immediates: tuple[int, ...] = ()) -> str:
        (a,) = operands
        return f"isfinite({a})"

    def verilog_expr(self, *operand_nets: str) -> str:
        (a,) = operand_nets
        return f"holoso_fisfinite({a})"

    def evaluate(self, *operands: ScalarValue, immediates: tuple[int, ...] = ()) -> tuple[bool, ...]:
        (a,) = self._validated_operands(operands)
        assert isinstance(a, FloatValue)
        return (a.fmt.is_finite(a.bits),)


@dataclass(frozen=True, slots=True)
class FloatIsPosInfOperator(FloatClassificationOperator):
    mnemonic: ClassVar[str] = "fisposinf"

    def render(self, *operands: str, immediates: tuple[int, ...] = ()) -> str:
        (a,) = operands
        return f"isposinf({a})"

    def verilog_expr(self, *operand_nets: str) -> str:
        (a,) = operand_nets
        return f"holoso_fisposinf({a})"

    def evaluate(self, *operands: ScalarValue, immediates: tuple[int, ...] = ()) -> tuple[bool, ...]:
        (a,) = self._validated_operands(operands)
        assert isinstance(a, FloatValue)
        return (not a.fmt.is_finite(a.bits) and not a.negative,)


@dataclass(frozen=True, slots=True)
class FloatIsNegInfOperator(FloatClassificationOperator):
    mnemonic: ClassVar[str] = "fisneginf"

    def render(self, *operands: str, immediates: tuple[int, ...] = ()) -> str:
        (a,) = operands
        return f"isneginf({a})"

    def verilog_expr(self, *operand_nets: str) -> str:
        (a,) = operand_nets
        return f"holoso_fisneginf({a})"

    def evaluate(self, *operands: ScalarValue, immediates: tuple[int, ...] = ()) -> tuple[bool, ...]:
        (a,) = self._validated_operands(operands)
        assert isinstance(a, FloatValue)
        return (not a.fmt.is_finite(a.bits) and a.negative,)


@dataclass(frozen=True, slots=True)
class FloatToBoolOperator(InlineHardwareOperator):
    mnemonic: ClassVar[str] = "ftobool"
    fmt: FloatFormat

    @property
    def signature(self) -> ScalarSignature:
        return ScalarSignature((FloatType(self.fmt),), (BoolType(),))

    def render(self, *operands: str, immediates: tuple[int, ...] = ()) -> str:
        (a,) = operands
        return f"bool({a})"

    def verilog_expr(self, *operand_nets: str) -> str:
        (a,) = operand_nets
        return f"holoso_ftobool({a})"

    def evaluate(self, *operands: ScalarValue, immediates: tuple[int, ...] = ()) -> tuple[bool, ...]:
        (a,) = self._validated_operands(operands)
        assert isinstance(a, FloatValue)
        return (a.exponent != 0,)


@dataclass(frozen=True, slots=True)
class BoolToFloatOperator(InlineHardwareOperator):
    mnemonic: ClassVar[str] = "ffrombool"
    fmt: FloatFormat

    @property
    def signature(self) -> ScalarSignature:
        return ScalarSignature((BoolType(),), (FloatType(self.fmt),))

    def render(self, *operands: str, immediates: tuple[int, ...] = ()) -> str:
        (a,) = operands
        return f"float({a})"

    def verilog_expr(self, *operand_nets: str) -> str:
        (a,) = operand_nets
        return f"holoso_ffrombool({a})"

    def evaluate(self, *operands: ScalarValue, immediates: tuple[int, ...] = ()) -> tuple[FloatValue, ...]:
        (a,) = self._validated_operands(operands)
        assert isinstance(a, bool)
        return (FloatValue.from_float(self.fmt, 1.0 if a else 0.0),)


@dataclass(frozen=True, slots=True)
class FFromIntOperator(ZkfBackedOperator):
    """Signed integer to float, nearest with ties to even; a magnitude past the finite range becomes an infinity."""

    ifmt: IntFormat

    @dataclass(frozen=True, slots=True)
    class Options:
        stage_input: int = 0
        stage_normalize: int = 0
        stage_pack: int = 0
        stage_output: int = 0

    mnemonic: ClassVar[str] = "ffromint"
    operand_hdl_ports: ClassVar[list[str]] = ["a"]
    output_hdl_ports: ClassVar[list[str]] = ["y"]
    opt: Options

    def __post_init__(self) -> None:
        model = zkf.FromIntModel(
            zkf.ZkfFormat(self.fmt.wexp, self.fmt.wman),
            wint=self.ifmt.width,
            stage_input=self.opt.stage_input,
            stage_normalize=self.opt.stage_normalize,
            stage_pack=self.opt.stage_pack,
            stage_output=self.opt.stage_output,
        )
        object.__setattr__(self, "_model", model)

    @property
    def signature(self) -> ScalarSignature:
        return ScalarSignature((IntType(self.ifmt),), (FloatType(self.fmt),))

    def evaluate(self, *operands: ScalarValue, immediates: tuple[int, ...] = ()) -> tuple[FloatValue, ...]:
        (a,) = self._validated_operands(operands)
        assert isinstance(a, IntValue)
        return (a.to_float(self.fmt),)

    def render(self, *operands: str, immediates: tuple[int, ...] = ()) -> str:
        (a,) = operands
        return f"float({a})"


@dataclass(frozen=True, slots=True)
class FToIntOperator(ZkfBackedOperator):
    """
    Float to signed integer, saturating at the rails, an infinity reaching one of them. One pooled instance serves
    all four modes through the ``round_mode`` immediate, as ``fround`` does.
    """

    ifmt: IntFormat

    @dataclass(frozen=True, slots=True)
    class Options:
        stage_input: int = 0

    mnemonic: ClassVar[str] = "ftoint"
    operand_hdl_ports: ClassVar[list[str]] = ["a"]
    output_hdl_ports: ClassVar[list[str]] = ["y"]
    immediate_ports: ClassVar[list[ImmediateField]] = [ImmediateField("round_mode", 2)]
    opt: Options

    def __post_init__(self) -> None:
        model = zkf.ToIntModel(
            zkf.ZkfFormat(self.fmt.wexp, self.fmt.wman), wint=self.ifmt.width, stage_input=self.opt.stage_input
        )
        object.__setattr__(self, "_model", model)

    @property
    def signature(self) -> ScalarSignature:
        return ScalarSignature((FloatType(self.fmt),), (IntType(self.ifmt),))

    def evaluate(self, *operands: ScalarValue, immediates: tuple[int, ...] = ()) -> tuple[IntValue, ...]:
        (a,) = self._validated_operands(operands)
        assert isinstance(a, FloatValue)
        (mode,) = immediates
        return (IntValue.from_float(self.ifmt, a, RoundMode(mode)),)

    def render(self, *operands: str, immediates: tuple[int, ...] = ()) -> str:
        (a,) = operands
        (mode,) = immediates
        return f"i{_ROUND_LABEL[RoundMode(mode)]}({a})"


@dataclass(frozen=True, slots=True)
class FMulILog2VarOperator(ZkfBackedOperator):
    """Exact scaling by a power of two: ``a * 2**k``; every ``k`` is legal."""

    ifmt: IntFormat

    @dataclass(frozen=True, slots=True)
    class Options:
        stage_input: int = 0
        stage_decode: int = 0

    mnemonic: ClassVar[str] = "fmul_ilog2"
    operand_hdl_ports: ClassVar[list[str]] = ["a", "k"]
    output_hdl_ports: ClassVar[list[str]] = ["y"]
    opt: Options

    def __post_init__(self) -> None:
        model = zkf.MulIlog2Model(
            zkf.ZkfFormat(self.fmt.wexp, self.fmt.wman),
            wk=self.ifmt.width,
            stage_input=self.opt.stage_input,
            stage_decode=self.opt.stage_decode,
        )
        object.__setattr__(self, "_model", model)

    @property
    def params(self) -> dict[str, int]:
        # The wrapper sizes the exponent port by the machine's integer format, so it spells the core's WK as WINT.
        return {("WINT" if name == "WK" else name): value for name, value in self._model.params.items()}

    @property
    def signature(self) -> ScalarSignature:
        return ScalarSignature((FloatType(self.fmt), IntType(self.ifmt)), (FloatType(self.fmt),))

    def evaluate(self, *operands: ScalarValue, immediates: tuple[int, ...] = ()) -> tuple[FloatValue, ...]:
        a, k = self._validated_operands(operands)
        assert isinstance(a, FloatValue) and isinstance(k, IntValue)
        return (a.scale_pow2(k.value),)

    def render(self, *operands: str, immediates: tuple[int, ...] = ()) -> str:
        a, k = operands
        return f"{a}×2^{k}"
