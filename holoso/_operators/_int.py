"""
The integer operators and primitives. The operators each carry their own closed-form latency, and the pooled
primitives their own reference arithmetic, saturating wherever the operation can leave the format; the inline ones are
native Verilog over the whole wide bank, sound because an integer fills that register exactly (DESIGN.md, Types).
"""

from abc import ABC
from dataclasses import dataclass
from typing import ClassVar, Self

from .._value import IntValue, ScalarValue
from .._type import BoolType, IntFormat, IntType
from ._common import (
    BaseOperatorOptions,
    BoolInversion,
    ComparatorPrimitive,
    HardwareOperator,
    InlinePrimitive,
    OperatorPort,
    PooledPrimitive,
    ScalarSignature,
    comparator_ports,
)


def _int_operator[O: HardwareOperator](
    cls: type[O],
    fmt: IntFormat,
    options: BaseOperatorOptions,
    operands: tuple[str, ...],
    outputs: tuple[str, ...],
    latency: int = 2,
    knobs: dict[str, int] | None = None,
) -> O:
    ints = IntType(fmt)
    return cls.of_one_mode(
        {"W": fmt.width, **(knobs or {}), "LATENCY": latency},
        options.instances,
        tuple(OperatorPort(name, ints) for name in operands),
        tuple(OperatorPort(name, ints) for name in outputs),
        initiation_interval=1,
    )


@dataclass(frozen=True, slots=True)
class IntPrimitive(PooledPrimitive, ABC):
    """
    The dual of FloatPrimitive. Saturation is what the integer type does at its extremes rather than a failure,
    and HIR marks the saturating operations speculatable, so the `saturated` sideband every module raises stays
    unconnected and unmodeled -- an if-converted arm that saturates must not raise the machine's error flag.
    """

    def _validated_operands(self, operands: tuple[ScalarValue, ...]) -> tuple[IntValue, ...]:
        validated: list[IntValue] = []
        for operand in super()._validated_operands(operands):
            assert isinstance(operand, IntValue)
            validated.append(operand)
        return tuple(validated)


@dataclass(frozen=True, slots=True)
class IAddOptions(BaseOperatorOptions): ...


class IAddOperator(HardwareOperator):
    __slots__ = ()
    name = "iadds"

    @classmethod
    def build(cls, fmt: IntFormat, options: IAddOptions) -> Self:
        return _int_operator(cls, fmt, options, ("a", "b"), ("y",))


@dataclass(frozen=True, slots=True)
class IAddPrimitive(IntPrimitive):
    operator: IAddOperator
    swap_output_permutation: ClassVar[tuple[int, ...]] = (0,)

    def evaluate(self, *operands: ScalarValue) -> tuple[IntValue, ...]:
        a, b = self._validated_operands(operands)
        return (a + b,)

    def render(self, *operands: str) -> str:
        a, b = operands
        return f"{a}+{b}"


@dataclass(frozen=True, slots=True)
class ISubOptions(BaseOperatorOptions): ...


class ISubOperator(HardwareOperator):
    __slots__ = ()
    name = "isubs"

    @classmethod
    def build(cls, fmt: IntFormat, options: ISubOptions) -> Self:
        return _int_operator(cls, fmt, options, ("a", "b"), ("y",))


@dataclass(frozen=True, slots=True)
class ISubPrimitive(IntPrimitive):
    """Also serves negation as `0-x`: there is no negation module, and this one saturates `-MIN` correctly."""

    operator: ISubOperator

    def evaluate(self, *operands: ScalarValue) -> tuple[IntValue, ...]:
        a, b = self._validated_operands(operands)
        return (a - b,)

    def render(self, *operands: str) -> str:
        a, b = operands
        return f"{a}-{b}"


@dataclass(frozen=True, slots=True)
class IMulOptions(BaseOperatorOptions):
    stage_product: int = 0
    """Splitting the product is useful when the width exceeds the DSP slice input. See Verilog holoso_imuls."""


class IMulOperator(HardwareOperator):
    __slots__ = ()
    name = "imuls"

    @classmethod
    def build(cls, fmt: IntFormat, options: IMulOptions) -> Self:
        if not 0 <= options.stage_product <= 4:
            raise ValueError(f"imuls stage_product must be in 0..4, got {options.stage_product}")
        knobs = {"STAGE_PRODUCT": options.stage_product}
        return _int_operator(cls, fmt, options, ("a", "b"), ("y",), 2 + options.stage_product, knobs)


@dataclass(frozen=True, slots=True)
class IMulPrimitive(IntPrimitive):
    operator: IMulOperator
    swap_output_permutation: ClassVar[tuple[int, ...]] = (0,)

    def evaluate(self, *operands: ScalarValue) -> tuple[IntValue, ...]:
        a, b = self._validated_operands(operands)
        return (a * b,)

    def render(self, *operands: str) -> str:
        a, b = operands
        return f"{a}×{b}"


@dataclass(frozen=True, slots=True)
class IDivOptions(BaseOperatorOptions): ...


class IDivOperator(HardwareOperator):
    __slots__ = ()
    name = "idivs"
    error_ports = ("div0",)

    @classmethod
    def build(cls, fmt: IntFormat, options: IDivOptions) -> Self:
        latency = 3 + (fmt.width + 1) // 2  # one radix-4 step per two quotient bits
        # Floor, because that is what Python's `//` and `%` mean and HIR has no other division; the core's truncating
        # mode is unreachable from a kernel.
        knobs = {"QUOTIENT_FLOOR": 1}
        return _int_operator(cls, fmt, options, ("num", "den"), ("quo", "rem"), latency, knobs)


@dataclass(frozen=True, slots=True)
class IDivPrimitive(IntPrimitive):
    """Floor division and its remainder together: one firing answers both."""

    operator: IDivOperator

    def evaluate(self, *operands: ScalarValue) -> tuple[IntValue, ...]:
        a, b = self._validated_operands(operands)
        return a.divmod_floor(b)

    def render(self, *operands: str) -> str:
        a, b = operands
        return f"{a}//{b}"

    def render_output(self, result: int, inversion: BoolInversion | None, *operands: str) -> str:
        assert inversion is None
        a, b = operands
        return f"{a}//{b}" if result == 0 else f"{a}%{b}"


@dataclass(frozen=True, slots=True)
class IAbsOptions(BaseOperatorOptions): ...


class IAbsOperator(HardwareOperator):
    __slots__ = ()
    name = "iabss"

    @classmethod
    def build(cls, fmt: IntFormat, options: IAbsOptions) -> Self:
        return _int_operator(cls, fmt, options, ("x",), ("y",))


@dataclass(frozen=True, slots=True)
class IAbsPrimitive(IntPrimitive):
    operator: IAbsOperator

    def evaluate(self, *operands: ScalarValue) -> tuple[IntValue, ...]:
        (a,) = self._validated_operands(operands)
        return (abs(a),)

    def render(self, *operands: str) -> str:
        (a,) = operands
        return f"|{a}|"


@dataclass(frozen=True, slots=True)
class IShlOptions(BaseOperatorOptions): ...


class IShlOperator(HardwareOperator):
    __slots__ = ()
    name = "ishl"

    @classmethod
    def build(cls, fmt: IntFormat, options: IShlOptions) -> Self:
        return _int_operator(cls, fmt, options, ("x", "shamt"), ("shft", "prod"))


@dataclass(frozen=True, slots=True)
class IShlPrimitive(IntPrimitive):
    """
    An arithmetic shift by a signed amount, left when positive. It emits both readings of a left shift at once:
    `shft` lets the high bits fall off the word, while `prod` is the multiplication by a power of two, saturating
    instead. Which one a shift wants is a lowering decision, so the primitive commits to neither.
    """

    operator: IShlOperator
    output_labels: ClassVar[tuple[str, ...]] = ("shft", "prod")

    def evaluate(self, *operands: ScalarValue) -> tuple[IntValue, ...]:
        a, b = self._validated_operands(operands)
        return a.shift_left(b)

    def render(self, *operands: str) -> str:
        a, b = operands
        return f"{a}<<{b}"


@dataclass(frozen=True, slots=True)
class IShrOptions(BaseOperatorOptions): ...


class IShrOperator(HardwareOperator):
    __slots__ = ()
    name = "ishr"

    @classmethod
    def build(cls, fmt: IntFormat, options: IShrOptions) -> Self:
        return _int_operator(cls, fmt, options, ("x", "shamt"), ("shft",))


@dataclass(frozen=True, slots=True)
class IShrPrimitive(IntPrimitive):
    """
    The mirror of IShlPrimitive, right when positive.
    Neither direction can rail, so it emits one raw reading and no saturation.
    """

    operator: IShrOperator

    def evaluate(self, *operands: ScalarValue) -> tuple[IntValue, ...]:
        a, b = self._validated_operands(operands)
        return (a.shift_right(b),)

    def render(self, *operands: str) -> str:
        a, b = operands
        return f"{a}>>{b}"


@dataclass(frozen=True, slots=True)
class IPopcntOptions(BaseOperatorOptions): ...


class IPopcntOperator(HardwareOperator):
    """
    The module answers on a port only as wide as the count needs -- the narrowest unsigned one holding a count of the
    magnitude, which the RTL derives again as `$clog2(W)` -- and the connection widens it with a zero fill, a count
    never being negative.
    """

    __slots__ = ()
    name = "ipopcnt"

    @classmethod
    def build(cls, fmt: IntFormat, options: IPopcntOptions) -> Self:
        count_width = (fmt.width - 1).bit_length()
        assert (1 << (count_width - 1)) <= fmt.width - 1 < (1 << count_width)
        return _int_operator(cls, fmt, options, ("x",), ("y",), knobs={"WY": count_width})


@dataclass(frozen=True, slots=True)
class IPopcntPrimitive(IntPrimitive):
    """
    The population count of the magnitude, as Python's `int.bit_count()`, so a negative operand counts the ones of `-x`.
    The negation that overflows a signed word is exactly the magnitude `2**(W-1)` read unsigned, so unlike
    IAbsPrimitive nothing saturates and the count never reaches the width.
    """

    operator: IPopcntOperator

    def evaluate(self, *operands: ScalarValue) -> tuple[IntValue, ...]:
        (a,) = self._validated_operands(operands)
        count = abs(a.value).bit_count()
        assert 0 <= count < a.fmt.width
        return (IntValue.from_int(a.fmt, count),)

    def render(self, *operands: str) -> str:
        (a,) = operands
        return f"popcnt({a})"


@dataclass(frozen=True, slots=True)
class ICmpOptions(BaseOperatorOptions): ...


class ICmpOperator(HardwareOperator):
    __slots__ = ()
    name = "icmp"

    @classmethod
    def build(cls, fmt: IntFormat, options: ICmpOptions) -> Self:
        params = {"W": fmt.width, "LATENCY": 2}
        return cls.of_one_mode(params, options.instances, *comparator_ports(IntType(fmt)), initiation_interval=1)


@dataclass(frozen=True, slots=True)
class ICmpPrimitive(IntPrimitive, ComparatorPrimitive):
    """Two's complement is totally ordered."""

    operator: ICmpOperator

    def evaluate(self, *operands: ScalarValue) -> tuple[bool, ...]:
        a, b = self._validated_operands(operands)
        ordering = a.compare(b)
        return ordering > 0, ordering == 0, ordering < 0


@dataclass(frozen=True, slots=True)
class IntInlinePrimitive(InlinePrimitive, ABC):
    fmt: IntFormat

    @property
    def scalar_type(self) -> IntType:
        return IntType(self.fmt)

    def _validated_operands(self, operands: tuple[ScalarValue, ...]) -> tuple[IntValue, ...]:
        validated: list[IntValue] = []
        for operand in super()._validated_operands(operands):
            assert isinstance(operand, IntValue)
            validated.append(operand)
        return tuple(validated)


@dataclass(frozen=True, slots=True)
class IntBitwisePrimitive(IntInlinePrimitive, ABC):
    @property
    def signature(self) -> ScalarSignature:
        return ScalarSignature((self.scalar_type,) * 2, (self.scalar_type,))


@dataclass(frozen=True, slots=True)
class IntBwAndPrimitive(IntBitwisePrimitive):
    mnemonic: ClassVar[str] = "ibwand"

    def verilog_expr(self, *operand_nets: str) -> str:
        a, b = operand_nets
        return f"({a}) & ({b})"

    def evaluate(self, *operands: ScalarValue) -> tuple[IntValue, ...]:
        a, b = self._validated_operands(operands)
        return (a & b,)


@dataclass(frozen=True, slots=True)
class IntBwOrPrimitive(IntBitwisePrimitive):
    mnemonic: ClassVar[str] = "ibwor"

    def verilog_expr(self, *operand_nets: str) -> str:
        a, b = operand_nets
        return f"({a}) | ({b})"

    def evaluate(self, *operands: ScalarValue) -> tuple[IntValue, ...]:
        a, b = self._validated_operands(operands)
        return (a | b,)


@dataclass(frozen=True, slots=True)
class IntBwXorPrimitive(IntBitwisePrimitive):
    mnemonic: ClassVar[str] = "ibwxor"

    def verilog_expr(self, *operand_nets: str) -> str:
        a, b = operand_nets
        return f"({a}) ^ ({b})"

    def evaluate(self, *operands: ScalarValue) -> tuple[IntValue, ...]:
        a, b = self._validated_operands(operands)
        return (a ^ b,)


@dataclass(frozen=True, slots=True)
class IntBwNotPrimitive(IntInlinePrimitive):
    mnemonic: ClassVar[str] = "ibwnot"

    @property
    def signature(self) -> ScalarSignature:
        return ScalarSignature((self.scalar_type,), (self.scalar_type,))

    def verilog_expr(self, *operand_nets: str) -> str:
        (a,) = operand_nets
        return f"~({a})"

    def evaluate(self, *operands: ScalarValue) -> tuple[IntValue, ...]:
        (a,) = self._validated_operands(operands)
        return (~a,)


@dataclass(frozen=True, slots=True)
class IntShiftConstPrimitive(IntInlinePrimitive):
    """
    An arithmetic shift by a count fixed at compile time, left when positive; the raw bit shift, so a left shift
    drops what leaves the word rather than saturating as the pooled `holoso_ishl` also offers.
    """

    mnemonic: ClassVar[str] = "ishiftc"
    shamt: int

    def __post_init__(self) -> None:
        super().__post_init__()
        # Zero is the identity, which HIR or MIR folds; a count reaching the word answers a constant or a sign fill, and
        # clamping a width-less count to the word is the lowering's job.
        assert 0 < abs(self.shamt) < self.fmt.width, self.shamt

    @property
    def signature(self) -> ScalarSignature:
        return ScalarSignature((self.scalar_type,), (self.scalar_type,))

    def verilog_expr(self, *operand_nets: str) -> str:
        (a,) = operand_nets
        if self.shamt < 0:
            return f"{{$signed({a}) >>> {-self.shamt}}}"
        return f"({a}) << {self.shamt}"

    def render(self, *operands: str) -> str:
        (a,) = operands
        return f"{a}<<{self.shamt}" if self.shamt > 0 else f"{a}>>{-self.shamt}"

    def evaluate(self, *operands: ScalarValue) -> tuple[IntValue, ...]:
        (a,) = self._validated_operands(operands)
        return (a.shift_left(IntValue.from_int(self.fmt, self.shamt)).shft,)


@dataclass(frozen=True, slots=True)
class IntToBoolPrimitive(InlinePrimitive):
    mnemonic: ClassVar[str] = "itobool"
    fmt: IntFormat

    @property
    def signature(self) -> ScalarSignature:
        return ScalarSignature((IntType(self.fmt),), (BoolType(),))

    def render(self, *operands: str) -> str:
        (a,) = operands
        return f"bool({a})"

    def verilog_expr(self, *operand_nets: str) -> str:
        (a,) = operand_nets
        return f"|({a})"

    def evaluate(self, *operands: ScalarValue) -> tuple[bool, ...]:
        (a,) = self._validated_operands(operands)
        assert isinstance(a, IntValue)
        return (a.value != 0,)


@dataclass(frozen=True, slots=True)
class BoolToIntPrimitive(InlinePrimitive):
    mnemonic: ClassVar[str] = "ifrombool"
    fmt: IntFormat

    @property
    def signature(self) -> ScalarSignature:
        return ScalarSignature((BoolType(),), (IntType(self.fmt),))

    def render(self, *operands: str) -> str:
        (a,) = operands
        return f"int({a})"

    def verilog_expr(self, *operand_nets: str) -> str:
        (a,) = operand_nets
        # Concatenation makes the widening self-determined: an operand net carrying a folded inversion arrives as
        # `~net`, and a bare one spliced into a wide register write would complement the carrier, not the single bit.
        return f"{{1'b0, {a}}}"

    def evaluate(self, *operands: ScalarValue) -> tuple[IntValue, ...]:
        (a,) = self._validated_operands(operands)
        assert isinstance(a, bool)
        return (IntValue.from_int(self.fmt, int(a)),)
