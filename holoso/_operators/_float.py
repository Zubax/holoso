"""
The float operators and primitives. Every operator is a ZKF block configured from its options, which answers its
timing and evaluates every primitive running on it. An operator whose format parameterizes it is a float one whatever
else it touches, so the casts across the type boundary live here too.
"""

from abc import ABC
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import ClassVar, Self

import zkf

from .._value import FloatValue, IntValue, RoundMode, ScalarValue
from .._type import BoolType, FloatFormat, FloatType, IntFormat, IntType, ScalarType
from ._common import (
    BaseOperatorOptions,
    ComparatorPrimitive,
    HardwareOperator,
    InlinePrimitive,
    agreed,
    float_operand,
    int_operand,
    PooledPrimitive,
    ScalarSignature,
)

_ROUND_LABEL: dict[RoundMode, str] = {
    RoundMode.NEAREST_EVEN: "round",
    RoundMode.FLOOR: "floor",
    RoundMode.CEIL: "ceil",
    RoundMode.TRUNC: "trunc",
}
"""Rendered into the report and the ROM comments, so it is pinned here rather than taken from zkf's member names."""


def _floats(fmt: FloatFormat, *names: str) -> dict[str, ScalarType]:
    return {name: FloatType(fmt) for name in names}


@dataclass(frozen=True, slots=True, kw_only=True)
class FAddOptions(BaseOperatorOptions):
    """The knobs of `zkf.Add`."""

    CoreStages = zkf.Add.CoreStages

    stage_input: int
    stage_core: zkf.Add.CoreStages
    stage_output: int


class FAddOperator(HardwareOperator[zkf.Add]):
    __slots__ = ()
    name = "fadd"

    @classmethod
    def build(cls, fmt: FloatFormat, options: FAddOptions) -> Self:
        block = zkf.Add(
            fmt=fmt,
            stage_input=options.stage_input,
            stage_core=options.stage_core,
            stage_output=options.stage_output,
        )
        return cls(block, options.instances, _floats(fmt, "a", "b"), _floats(fmt, "y"))


@dataclass(frozen=True, slots=True)
class FAddPrimitive(PooledPrimitive[zkf.Add]):
    operator: FAddOperator
    _commutative: ClassVar[bool] = True  # signed sum: a+b == b+a bit-for-bit

    def evaluate(self, *operands: ScalarValue, block: zkf.Block | None = None) -> FloatValue:
        block = self.operator.block if block is None else block
        assert isinstance(block, zkf.Add)
        a, b = map(float_operand, self._validated_operands(operands))
        return block.evaluate(a, b)

    def render(self, *operands: str) -> str:
        a, b = operands
        return f"{a}+{b}"


@dataclass(frozen=True, slots=True, kw_only=True)
class FMulOptions(BaseOperatorOptions):
    """The knobs of `zkf.Mul`."""

    stage_input: int
    stage_core_mul: int
    stage_output: int
    mul_attrs: zkf.Attrs = field(default_factory=dict)


class FMulOperator(HardwareOperator[zkf.Mul]):
    __slots__ = ()
    name = "fmul"

    @classmethod
    def build(cls, fmt: FloatFormat, options: FMulOptions, wmultiplier: int) -> Self:
        block = zkf.Mul(
            fmt=fmt,
            wmultiplier=wmultiplier,
            stage_input=options.stage_input,
            stage_core_mul=options.stage_core_mul,
            stage_output=options.stage_output,
            mul_attrs=options.mul_attrs,
        )
        return cls(block, options.instances, _floats(fmt, "a", "b"), _floats(fmt, "y"))


@dataclass(frozen=True, slots=True)
class FMulPrimitive(PooledPrimitive[zkf.Mul]):
    operator: FMulOperator
    _commutative: ClassVar[bool] = True  # product: a*b == b*a bit-for-bit

    def evaluate(self, *operands: ScalarValue, block: zkf.Block | None = None) -> FloatValue:
        block = self.operator.block if block is None else block
        assert isinstance(block, zkf.Mul)
        a, b = map(float_operand, self._validated_operands(operands))
        return block.evaluate(a, b)

    def render(self, *operands: str) -> str:
        a, b = operands
        return f"{a}×{b}"


@dataclass(frozen=True, slots=True, kw_only=True)
class FDivsqrtOptions(BaseOperatorOptions):
    """The knobs of `zkf.Divsqrt`."""

    CoreStages = zkf.Divsqrt.CoreStages

    stage_input: int
    stage_core: zkf.Divsqrt.CoreStages
    stage_output: int
    fast: bool


class FDivsqrtOperator(HardwareOperator[zkf.Divsqrt]):
    """
    One digit-recurrence pipeline serving the quotient `a / b` and the square root of `a`, chosen per firing on
    `op_sqrt`, at one latency. `error` reports a zero divisor in the first and a negative operand in the second. An
    instance running one of them alone is built for it alone, without the other's datapath.
    """

    __slots__ = ()
    name = "fdivsqrt"
    error_ports = ("error",)

    @classmethod
    def build(cls, fmt: FloatFormat, options: FDivsqrtOptions) -> Self:
        block = zkf.Divsqrt(
            fmt=fmt,
            stage_input=options.stage_input,
            stage_core=options.stage_core,
            stage_output=options.stage_output,
            fast=options.fast,
        )
        return cls(block, options.instances, _floats(fmt, "a", "b"), _floats(fmt, "y"))

    def instance_block(self, primitives: frozenset[PooledPrimitive]) -> zkf.Divsqrt:
        op_sqrt = agreed(primitives, "op_sqrt", bool)
        return self.block if op_sqrt is None else self.block.specialized(op_sqrt=op_sqrt)


@dataclass(frozen=True, slots=True)
class FDivPrimitive(PooledPrimitive[zkf.Divsqrt]):
    operator: FDivsqrtOperator
    immediates: ClassVar[Mapping[str, int]] = {"op_sqrt": False}

    def evaluate(self, *operands: ScalarValue, block: zkf.Block | None = None) -> FloatValue:
        block = self.operator.block if block is None else block
        assert isinstance(block, zkf.Divsqrt)
        a, b = map(float_operand, self._validated_operands(operands))
        return block.evaluate(self.immediates["op_sqrt"], a, b).y

    def render(self, *operands: str) -> str:
        a, b = operands
        return f"{a}/{b}"


@dataclass(frozen=True, slots=True)
class FSqrtPrimitive(PooledPrimitive[zkf.Divsqrt]):
    """Correctly-rounded square root; a negative operand yields -inf and raises `error`."""

    operator: FDivsqrtOperator
    immediates: ClassVar[Mapping[str, int]] = {"op_sqrt": True}

    def evaluate(self, *operands: ScalarValue, block: zkf.Block | None = None) -> FloatValue:
        block = self.operator.block if block is None else block
        assert isinstance(block, zkf.Divsqrt)
        (a,) = map(float_operand, self._validated_operands(operands))
        return block.evaluate(self.immediates["op_sqrt"], a, FloatValue(a.fmt, 0)).y

    def render(self, *operands: str) -> str:
        (a,) = operands
        return f"√{a}"


@dataclass(frozen=True, slots=True, kw_only=True)
class FILog2Options(BaseOperatorOptions):
    """The knobs of `zkf.Ilog2`."""

    stage_input: int
    stage_output: int


class FILog2Operator(HardwareOperator[zkf.Ilog2]):
    __slots__ = ()
    name = "filog2"

    @classmethod
    def build(cls, fmt: FloatFormat, ifmt: IntFormat, options: FILog2Options) -> Self:
        block = zkf.Ilog2(
            fmt=fmt,
            wint=ifmt.width,
            stage_input=options.stage_input,
            stage_output=options.stage_output,
        )
        return cls(block, options.instances, _floats(fmt, "a"), {"y": IntType(ifmt)})


@dataclass(frozen=True, slots=True)
class FILog2Primitive(PooledPrimitive[zkf.Ilog2]):
    """
    The extraction half of the exponent pair, `FMulILog2Primitive` being the scaling half: the two compose with no
    arithmetic between them, sharing ZKF's binade convention (the unbiased exponent, `-bias` for zero and `bias+1` for
    an infinity, so an extremum over the answer needs no special case for them). Sign-invariant: the operand's
    conditioner reaches no logic at all, sign conditioning rewriting only the bit the exponent field does not read.
    """

    operator: FILog2Operator
    unconditioned_operands: ClassVar[frozenset[int]] = frozenset({0})

    def evaluate(self, *operands: ScalarValue, block: zkf.Block | None = None) -> IntValue:
        block = self.operator.block if block is None else block
        assert isinstance(block, zkf.Ilog2)
        (a,) = map(float_operand, self._validated_operands(operands))
        return self._int_result(block.evaluate(a).y)

    def render(self, *operands: str) -> str:
        (a,) = operands
        return f"ilog2({a})"


@dataclass(frozen=True, slots=True, kw_only=True)
class FMulILog2Options(BaseOperatorOptions):
    """The knobs of `zkf.MulIlog2`."""

    CoreStages = zkf.MulIlog2.CoreStages

    stage_input: int
    stage_core: zkf.MulIlog2.CoreStages
    stage_output: int


class FMulILog2Operator(HardwareOperator[zkf.MulIlog2]):
    __slots__ = ()
    name = "fmul_ilog2"

    @classmethod
    def build(cls, fmt: FloatFormat, ifmt: IntFormat, options: FMulILog2Options) -> Self:
        block = zkf.MulIlog2(
            fmt=fmt,
            wk=ifmt.width,
            stage_input=options.stage_input,
            stage_core=options.stage_core,
            stage_output=options.stage_output,
        )
        return cls(block, options.instances, {"a": FloatType(fmt), "k": IntType(ifmt)}, _floats(fmt, "y"))


@dataclass(frozen=True, slots=True)
class FMulILog2Primitive(PooledPrimitive[zkf.MulIlog2]):
    """Exact scaling by a power of two: `a * 2**k`; every `k` is legal."""

    operator: FMulILog2Operator

    def evaluate(self, *operands: ScalarValue, block: zkf.Block | None = None) -> FloatValue:
        block = self.operator.block if block is None else block
        assert isinstance(block, zkf.MulIlog2)
        a, k = self._validated_operands(operands)
        return block.evaluate(float_operand(a), int_operand(k))

    def render(self, *operands: str) -> str:
        a, k = operands
        return f"{a}×2^{k}"


@dataclass(frozen=True, slots=True, kw_only=True)
class FCmpOptions(BaseOperatorOptions):
    """
    The knobs of `zkf.Cmp`. The comparator runs every float relation, min and max, one firing per cycle, so a kernel
    issuing several of them at once shortens by raising `instances`.
    """

    stage_input: int
    stage_output: int


class FCmpOperator(HardwareOperator[zkf.Cmp]):
    """The comparator, which orders its operands on the flags and answers them sorted beside."""

    __slots__ = ()
    name = "fcmp"

    @classmethod
    def build(cls, fmt: FloatFormat, options: FCmpOptions) -> Self:
        block = zkf.Cmp(fmt=fmt, stage_input=options.stage_input, stage_output=options.stage_output)
        flags: dict[str, ScalarType] = {"a_gt_b": BoolType(), "a_eq_b": BoolType(), "a_lt_b": BoolType()}
        return cls(block, options.instances, _floats(fmt, "a", "b"), {**flags, **_floats(fmt, "min", "max")})


@dataclass(frozen=True, slots=True)
class FCmpPrimitive(ComparatorPrimitive[zkf.Cmp]):
    """
    An ordering of the sign-conditioned operands, beside the comparator's ascending `min` and `max` of them. ZKF has
    no NaN, so the ordering is total.
    """

    operator: FCmpOperator

    def evaluate(self, *operands: ScalarValue, block: zkf.Block | None = None) -> bool:
        block = self.operator.block if block is None else block
        assert isinstance(block, zkf.Cmp)
        a, b = map(float_operand, self._validated_operands(operands))
        result = block.evaluate(a, b)
        return self._flag(result.a_gt_b, result.a_eq_b, result.a_lt_b)


@dataclass(frozen=True, slots=True)
class FMinPrimitive(PooledPrimitive[zkf.Cmp]):
    """
    The comparator's `min`, which keeps the selected operand's exact bits: on a tie it answers the second operand, so
    exchanging the operands can flip the sign of a zero, and it has no mirror.
    """

    operator: FCmpOperator
    output_name: ClassVar[str] = "min"

    def evaluate(self, *operands: ScalarValue, block: zkf.Block | None = None) -> FloatValue:
        block = self.operator.block if block is None else block
        assert isinstance(block, zkf.Cmp)
        a, b = map(float_operand, self._validated_operands(operands))
        return block.evaluate(a, b).min

    def render(self, *operands: str) -> str:
        a, b = operands
        return f"min({a},{b})"


@dataclass(frozen=True, slots=True)
class FMaxPrimitive(PooledPrimitive[zkf.Cmp]):
    """The comparator's `max`, which answers the first operand on a tie; see `FMinPrimitive`."""

    operator: FCmpOperator
    output_name: ClassVar[str] = "max"

    def evaluate(self, *operands: ScalarValue, block: zkf.Block | None = None) -> FloatValue:
        block = self.operator.block if block is None else block
        assert isinstance(block, zkf.Cmp)
        a, b = map(float_operand, self._validated_operands(operands))
        return block.evaluate(a, b).max

    def render(self, *operands: str) -> str:
        a, b = operands
        return f"max({a},{b})"


@dataclass(frozen=True, slots=True, kw_only=True)
class FRintOptions(BaseOperatorOptions):
    """The knobs of `zkf.Rint`."""

    CoreStages = zkf.Rint.CoreStages

    stage_input: int
    stage_core: zkf.Rint.CoreStages
    stage_output: int


class FRintOperator(HardwareOperator[zkf.Rint]):
    """
    Rounds a float to an integer and answers it twice at once, as a float of the same format and as a signed integer.
    The rounding is selected on the `round_mode` port by its own encoding.
    """

    __slots__ = ()
    name = "frint"

    @classmethod
    def build(cls, fmt: FloatFormat, ifmt: IntFormat, options: FRintOptions) -> Self:
        block = zkf.Rint(
            fmt=fmt,
            wint=ifmt.width,
            stage_input=options.stage_input,
            stage_core=options.stage_core,
            stage_output=options.stage_output,
        )
        return cls(block, options.instances, _floats(fmt, "a"), {"y_float": FloatType(fmt), "y_int": IntType(ifmt)})

    def instance_block(self, primitives: frozenset[PooledPrimitive]) -> zkf.Rint:
        round_mode = agreed(primitives, "round_mode", RoundMode)
        return self.block if round_mode is None else self.block.specialized(round_mode=round_mode)


@dataclass(frozen=True, slots=True)
class FRintPrimitive(PooledPrimitive[zkf.Rint]):
    """
    One rounding of a float (nearest-even, floor, ceil or trunc) as an integral-valued float. The rounder answers the
    same rounding as a signed integer beside it (`FRintToIntPrimitive`), so a float use and an integer use of one
    rounding of one operand are one firing.
    """

    operator: FRintOperator
    rounding: RoundMode
    output_name: ClassVar[str] = "y_float"

    @property
    def immediates(self) -> Mapping[str, int]:
        return {"round_mode": self.rounding}

    def evaluate(self, *operands: ScalarValue, block: zkf.Block | None = None) -> FloatValue:
        block = self.operator.block if block is None else block
        assert isinstance(block, zkf.Rint)
        (a,) = map(float_operand, self._validated_operands(operands))
        return block.evaluate(a, self.rounding).y_float

    def render(self, *operands: str) -> str:
        (a,) = operands
        return f"{_ROUND_LABEL[self.rounding]}({a})"


@dataclass(frozen=True, slots=True)
class FRintToIntPrimitive(PooledPrimitive[zkf.Rint]):
    """One rounding of a float as a signed integer saturating at the rails, an infinity reaching one of them."""

    operator: FRintOperator
    rounding: RoundMode
    output_name: ClassVar[str] = "y_int"

    @property
    def immediates(self) -> Mapping[str, int]:
        return {"round_mode": self.rounding}

    def evaluate(self, *operands: ScalarValue, block: zkf.Block | None = None) -> IntValue:
        block = self.operator.block if block is None else block
        assert isinstance(block, zkf.Rint)
        (a,) = map(float_operand, self._validated_operands(operands))
        return self._int_result(block.evaluate(a, self.rounding).y_int)

    def render(self, *operands: str) -> str:
        (a,) = operands
        return f"i{_ROUND_LABEL[self.rounding]}({a})"


@dataclass(frozen=True, slots=True, kw_only=True)
class FFmaOptions(BaseOperatorOptions):
    """The knobs of `zkf.Fma`; `stage_core_add` takes `FAddOptions.CoreStages`."""

    stage_input: int
    stage_core_mul: int
    stage_core_add: zkf.Add.CoreStages
    stage_output: int
    mul_attrs: zkf.Attrs = field(default_factory=dict)


class FFmaOperator(HardwareOperator[zkf.Fma]):
    __slots__ = ()
    name = "ffma"

    @classmethod
    def build(cls, fmt: FloatFormat, options: FFmaOptions, wmultiplier: int) -> Self:
        block = zkf.Fma(
            fmt=fmt,
            wmultiplier=wmultiplier,
            stage_input=options.stage_input,
            stage_core_mul=options.stage_core_mul,
            stage_core_add=options.stage_core_add,
            stage_output=options.stage_output,
            mul_attrs=options.mul_attrs,
        )
        return cls(block, options.instances, _floats(fmt, "a", "b", "c"), _floats(fmt, "y"))


@dataclass(frozen=True, slots=True)
class FFmaPrimitive(PooledPrimitive[zkf.Fma]):
    """
    Fused multiply-add `a*b + c`, single-rounded (full-width product rounded once with `c`). Not commutative under
    operand reversal (gives `c*b+a`).
    """

    operator: FFmaOperator

    def evaluate(self, *operands: ScalarValue, block: zkf.Block | None = None) -> FloatValue:
        block = self.operator.block if block is None else block
        assert isinstance(block, zkf.Fma)
        a, b, c = map(float_operand, self._validated_operands(operands))
        return block.evaluate(a, b, c)

    def render(self, *operands: str) -> str:
        a, b, c = operands
        return f"{a}×{b}+{c}"


@dataclass(frozen=True, slots=True, kw_only=True)
class FExp2Options(BaseOperatorOptions):
    """The knobs of `zkf.Exp2`."""

    CoreStages = zkf.Exp2.CoreStages

    stage_input: int
    stage_core_kernel_level: int
    stage_core: zkf.Exp2.CoreStages
    stage_output: int
    fast: bool
    rom_attrs: zkf.Attrs = field(default_factory=dict)
    mul_attrs: zkf.Attrs = field(default_factory=dict)


class FExp2Operator(HardwareOperator[zkf.Exp2]):
    __slots__ = ()
    name = "fexp2"

    @classmethod
    def build(cls, fmt: FloatFormat, options: FExp2Options, wmultiplier: int) -> Self:
        block = zkf.Exp2(
            fmt=fmt,
            wmultiplier=wmultiplier,
            stage_input=options.stage_input,
            stage_core_kernel_level=options.stage_core_kernel_level,
            stage_core=options.stage_core,
            stage_output=options.stage_output,
            fast=options.fast,
            rom_attrs=options.rom_attrs,
            mul_attrs=options.mul_attrs,
        )
        return cls(block, options.instances, _floats(fmt, "x"), _floats(fmt, "y"))


@dataclass(frozen=True, slots=True)
class FExp2Primitive(PooledPrimitive[zkf.Exp2]):
    operator: FExp2Operator

    def evaluate(self, *operands: ScalarValue, block: zkf.Block | None = None) -> FloatValue:
        block = self.operator.block if block is None else block
        assert isinstance(block, zkf.Exp2)
        (a,) = map(float_operand, self._validated_operands(operands))
        return block.evaluate(a)

    def render(self, *operands: str) -> str:
        (a,) = operands
        return f"2^{a}"


@dataclass(frozen=True, slots=True, kw_only=True)
class FLog2Options(BaseOperatorOptions):
    """The knobs of `zkf.Log2`."""

    CoreStages = zkf.Log2.CoreStages

    stage_input: int
    stage_core_kernel_level: int
    stage_core_mul: int
    stage_core: zkf.Log2.CoreStages
    stage_output: int
    fast: bool
    rom_attrs: zkf.Attrs = field(default_factory=dict)
    mul_attrs: zkf.Attrs = field(default_factory=dict)


class FLog2Operator(HardwareOperator[zkf.Log2]):
    __slots__ = ()
    name = "flog2"
    error_ports = ("domain_error", "pole")

    @classmethod
    def build(cls, fmt: FloatFormat, options: FLog2Options, wmultiplier: int) -> Self:
        block = zkf.Log2(
            fmt=fmt,
            wmultiplier=wmultiplier,
            stage_input=options.stage_input,
            stage_core_kernel_level=options.stage_core_kernel_level,
            stage_core_mul=options.stage_core_mul,
            stage_core=options.stage_core,
            stage_output=options.stage_output,
            fast=options.fast,
            rom_attrs=options.rom_attrs,
            mul_attrs=options.mul_attrs,
        )
        return cls(block, options.instances, _floats(fmt, "x"), _floats(fmt, "y"))


@dataclass(frozen=True, slots=True)
class FLog2Primitive(PooledPrimitive[zkf.Log2]):
    operator: FLog2Operator

    def evaluate(self, *operands: ScalarValue, block: zkf.Block | None = None) -> FloatValue:
        block = self.operator.block if block is None else block
        assert isinstance(block, zkf.Log2)
        (a,) = map(float_operand, self._validated_operands(operands))
        return block.evaluate(a).y

    def render(self, *operands: str) -> str:
        (a,) = operands
        return f"log2({a})"


@dataclass(frozen=True, slots=True, kw_only=True)
class FCordicOptions(BaseOperatorOptions):
    """
    The knobs of `zkf.Cordic`. A nearby hypot over the operands of an atan2 folds into the vectoring magnitude for free.
    """

    CoreStages = zkf.Cordic.CoreStages

    stage_input: int
    stage_core_mul: int
    stage_core: zkf.Cordic.CoreStages
    stage_output: int
    fast: bool
    rotation_seed: int = 2
    rom_attrs: zkf.Attrs = field(default_factory=dict)
    mul_attrs: zkf.Attrs = field(default_factory=dict)


class FCordicOperator(HardwareOperator[zkf.Cordic]):
    """
    The CORDIC core, rotating or vectoring as each firing drives `vectoring`: a rotation reads a phase in turns and
    yields its sine and cosine, a vectoring reads `(y, x)` and yields their angle in turns, and their magnitude where a
    primitive asks for it. ZKF times the two by the value of `vectoring`. NOT throughput-1: the core holds one
    transaction in flight at either timing, which is also what keeps the results of the two, each at its own latency,
    leaving an instance in issue order. An instance is built for the drives its primitives make, and with the magnitude
    only where one of them asks for it.
    """

    __slots__ = ()
    name = "fcordic"

    @classmethod
    def build(cls, fmt: FloatFormat, options: FCordicOptions, wmultiplier: int) -> Self:
        block = zkf.Cordic(
            fmt=fmt,
            wmultiplier=wmultiplier,
            stage_input=options.stage_input,
            stage_core_mul=options.stage_core_mul,
            stage_core=options.stage_core,
            stage_output=options.stage_output,
            fast=options.fast,
            rotation_seed=options.rotation_seed,
            rom_attrs=options.rom_attrs,
            mul_attrs=options.mul_attrs,
        )
        return cls(block, options.instances, _floats(fmt, "a", "b"), _floats(fmt, "r0", "r1"))

    def instance_block(self, primitives: frozenset[PooledPrimitive]) -> zkf.Cordic:
        # ZKF's `specialized` starts from the default on every call, so the one call carries both knobs: the magnitude
        # is built where some primitive answers it, rotating, vectoring or both.
        length = any(isinstance(primitive, FHypotPrimitive) for primitive in primitives)
        return self.block.specialized(vectoring=agreed(primitives, "vectoring", bool), length=length)


@dataclass(frozen=True, slots=True)
class FSinPrimitive(PooledPrimitive[zkf.Cordic]):
    """The sine of a phase in turns, the rotation's `r0`."""

    operator: FCordicOperator
    immediates: ClassVar[Mapping[str, int]] = {"vectoring": False}
    output_name: ClassVar[str] = "r0"

    def evaluate(self, *operands: ScalarValue, block: zkf.Block | None = None) -> FloatValue:
        block = self.operator.block if block is None else block
        assert isinstance(block, zkf.Cordic)
        (a,) = map(float_operand, self._validated_operands(operands))
        return block.evaluate(self.immediates["vectoring"], a, FloatValue(a.fmt, 0)).r0

    def render(self, *operands: str) -> str:
        (a,) = operands
        return f"sin({a})"


@dataclass(frozen=True, slots=True)
class FCosPrimitive(PooledPrimitive[zkf.Cordic]):
    """The cosine of a phase in turns, the rotation's `r1`."""

    operator: FCordicOperator
    immediates: ClassVar[Mapping[str, int]] = {"vectoring": False}
    output_name: ClassVar[str] = "r1"

    def evaluate(self, *operands: ScalarValue, block: zkf.Block | None = None) -> FloatValue:
        block = self.operator.block if block is None else block
        assert isinstance(block, zkf.Cordic)
        (a,) = map(float_operand, self._validated_operands(operands))
        return block.evaluate(self.immediates["vectoring"], a, FloatValue(a.fmt, 0)).r1

    def render(self, *operands: str) -> str:
        (a,) = operands
        return f"cos({a})"


@dataclass(frozen=True, slots=True)
class FAtan2Primitive(PooledPrimitive[zkf.Cordic]):
    """The angle of `(y, x)` in turns, the vectoring's `r0`."""

    operator: FCordicOperator
    immediates: ClassVar[Mapping[str, int]] = {"vectoring": True}
    output_name: ClassVar[str] = "r0"

    def evaluate(self, *operands: ScalarValue, block: zkf.Block | None = None) -> FloatValue:
        block = self.operator.block if block is None else block
        assert isinstance(block, zkf.Cordic)
        y, x = map(float_operand, self._validated_operands(operands))
        return block.evaluate(self.immediates["vectoring"], y, x).r0

    def render(self, *operands: str) -> str:
        y, x = operands
        return f"atan2({y},{x})"


@dataclass(frozen=True, slots=True)
class FHypotPrimitive(PooledPrimitive[zkf.Cordic]):
    """
    The magnitude of `(y, x)`, the vectoring's `r1`, which an instance builds only where some primitive bound to it
    answers it. Its operands' signs matter to the angle beside it, so it declines none of them.
    """

    operator: FCordicOperator
    immediates: ClassVar[Mapping[str, int]] = {"vectoring": True}
    output_name: ClassVar[str] = "r1"

    def evaluate(self, *operands: ScalarValue, block: zkf.Block | None = None) -> FloatValue:
        block = self.operator.block if block is None else block
        assert isinstance(block, zkf.Cordic)
        y, x = map(float_operand, self._validated_operands(operands))
        return block.evaluate(self.immediates["vectoring"], y, x).r1

    def render(self, *operands: str) -> str:
        y, x = operands
        return f"hypot({y},{x})"


@dataclass(frozen=True, slots=True)
class _FloatClassificationPrimitive(InlinePrimitive, ABC):
    fmt: FloatFormat

    @property
    def signature(self) -> ScalarSignature:
        return ScalarSignature((FloatType(self.fmt),), BoolType())


@dataclass(frozen=True, slots=True)
class FloatIsFinitePrimitive(_FloatClassificationPrimitive):
    # The exponent field alone decides finiteness; the directional classifiers below read the sign and so declare
    # nothing, which is why this sits here rather than on the shared base.
    unconditioned_operands: ClassVar[frozenset[int]] = frozenset({0})

    def render(self, *operands: str) -> str:
        (a,) = operands
        return f"isfinite({a})"

    def verilog_expr(self, *operand_nets: str) -> str:
        (a,) = operand_nets
        return f"holoso_fisfinite({a})"

    def evaluate(self, *operands: ScalarValue) -> bool:
        (a,) = self._validated_operands(operands)
        assert isinstance(a, FloatValue)
        return a.is_finite


@dataclass(frozen=True, slots=True)
class FloatIsPosInfPrimitive(_FloatClassificationPrimitive):
    def render(self, *operands: str) -> str:
        (a,) = operands
        return f"isposinf({a})"

    def verilog_expr(self, *operand_nets: str) -> str:
        (a,) = operand_nets
        return f"holoso_fisposinf({a})"

    def evaluate(self, *operands: ScalarValue) -> bool:
        (a,) = self._validated_operands(operands)
        assert isinstance(a, FloatValue)
        return a.is_inf and not a.negative


@dataclass(frozen=True, slots=True)
class FloatIsNegInfPrimitive(_FloatClassificationPrimitive):
    def render(self, *operands: str) -> str:
        (a,) = operands
        return f"isneginf({a})"

    def verilog_expr(self, *operand_nets: str) -> str:
        (a,) = operand_nets
        return f"holoso_fisneginf({a})"

    def evaluate(self, *operands: ScalarValue) -> bool:
        (a,) = self._validated_operands(operands)
        assert isinstance(a, FloatValue)
        return a.is_inf and a.negative


@dataclass(frozen=True, slots=True)
class FloatToBoolPrimitive(InlinePrimitive):
    unconditioned_operands: ClassVar[frozenset[int]] = frozenset({0})  # a zero test reads the exponent alone
    fmt: FloatFormat

    @property
    def signature(self) -> ScalarSignature:
        return ScalarSignature((FloatType(self.fmt),), BoolType())

    def render(self, *operands: str) -> str:
        (a,) = operands
        return f"bool({a})"

    def verilog_expr(self, *operand_nets: str) -> str:
        (a,) = operand_nets
        return f"holoso_ftobool({a})"

    def evaluate(self, *operands: ScalarValue) -> bool:
        (a,) = self._validated_operands(operands)
        assert isinstance(a, FloatValue)
        return not a.is_zero


@dataclass(frozen=True, slots=True)
class BoolToFloatPrimitive(InlinePrimitive):
    fmt: FloatFormat

    @property
    def signature(self) -> ScalarSignature:
        return ScalarSignature((BoolType(),), FloatType(self.fmt))

    def render(self, *operands: str) -> str:
        (a,) = operands
        return f"float({a})"

    def verilog_expr(self, *operand_nets: str) -> str:
        (a,) = operand_nets
        return f"holoso_ffrombool({a})"

    def evaluate(self, *operands: ScalarValue) -> FloatValue:
        (a,) = self._validated_operands(operands)
        assert isinstance(a, bool)
        return self.fmt.encode(1.0 if a else 0.0)


@dataclass(frozen=True, slots=True, kw_only=True)
class FFromIntOptions(BaseOperatorOptions):
    """The knobs of `zkf.FromInt`."""

    CoreStages = zkf.FromInt.CoreStages

    stage_input: int
    stage_core: zkf.FromInt.CoreStages
    stage_output: int


class FFromIntOperator(HardwareOperator[zkf.FromInt]):
    __slots__ = ()
    name = "ffromint"

    @classmethod
    def build(cls, fmt: FloatFormat, ifmt: IntFormat, options: FFromIntOptions) -> Self:
        block = zkf.FromInt(
            fmt=fmt,
            wint=ifmt.width,
            stage_input=options.stage_input,
            stage_core=options.stage_core,
            stage_output=options.stage_output,
        )
        return cls(block, options.instances, {"a": IntType(ifmt)}, _floats(fmt, "y"))


@dataclass(frozen=True, slots=True)
class FFromIntPrimitive(PooledPrimitive[zkf.FromInt]):
    """Signed integer to float, nearest with ties to even; a magnitude past the finite range becomes an infinity."""

    operator: FFromIntOperator

    def evaluate(self, *operands: ScalarValue, block: zkf.Block | None = None) -> FloatValue:
        block = self.operator.block if block is None else block
        assert isinstance(block, zkf.FromInt)
        (a,) = map(int_operand, self._validated_operands(operands))
        return block.evaluate(a)

    def render(self, *operands: str) -> str:
        (a,) = operands
        return f"float({a})"
