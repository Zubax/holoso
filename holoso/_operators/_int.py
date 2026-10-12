"""
The integer operators and primitives. The operators each carry their own closed-form latency, and the pooled
primitives their own reference arithmetic, saturating wherever the operation can leave the format; the inline ones are
native Verilog over the whole wide bank, sound because an integer fills that register exactly (DESIGN.md, Types).
"""

from abc import ABC
from dataclasses import dataclass
from enum import IntEnum
from typing import ClassVar, Self

from .._value import IntValue, ScalarValue
from .._type import BoolType, IntFormat, IntType
from .._util import Relation
from ._common import (
    BaseOperatorOptions,
    BoolInversion,
    ComparatorPrimitive,
    HardwareOperator,
    InlinePrimitive,
    ModePort,
    OperatorMode,
    OperatorPort,
    PooledPrimitive,
    ScalarSignature,
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
    and HIR marks the saturating operations speculatable, so the `saturated` sideband a saturating module raises stays
    unconnected and unmodeled -- an if-converted arm that saturates must not raise the machine's error flag.
    """

    def _validated_operands(self, operands: tuple[ScalarValue, ...]) -> tuple[IntValue, ...]:
        validated: list[IntValue] = []
        for operand in super()._validated_operands(operands):
            assert isinstance(operand, IntValue)
            validated.append(operand)
        return tuple(validated)


@dataclass(frozen=True, slots=True)
class IAddsOptions(BaseOperatorOptions):
    """
    The adder also runs every integer subtraction, negation and comparison, one firing per cycle, so a kernel issuing
    several of them at once shortens by raising `instances`.
    """

    fast: bool = False
    """Trades area for shorter combinational paths."""


class AddMode(IntEnum):
    """The value driven on `sub`, which is also the `MODE` elaborating the adder for that operation alone."""

    ADD = 0
    SUB = 1


class IAddOperator(HardwareOperator):
    """
    The saturating adder, which also subtracts, chosen per firing on `sub`, at one latency. A subtraction orders `a`
    against `b` on the flags as a side effect, so its mode drives them beside the difference, and a comparison is the
    same code read through the flags alone. Firings of all three contend for the one adder.
    """

    __slots__ = ()
    name = "iadds"
    mode_port = ModePort("sub", 1)

    @classmethod
    def build(cls, fmt: IntFormat, options: IAddsOptions) -> Self:
        ints, flag = IntType(fmt), BoolType()
        operands = (OperatorPort("a", ints), OperatorPort("b", ints))
        outputs = (
            OperatorPort("y", ints),
            OperatorPort("a_gt_b", flag),
            OperatorPort("a_eq_b", flag),
            OperatorPort("a_lt_b", flag),
        )
        add = OperatorMode(int(AddMode.ADD), "LATENCY", 1, 2, (0,))
        subtract = OperatorMode(int(AddMode.SUB), "LATENCY", 1, 2, (0, 1, 2, 3))
        compare = OperatorMode(int(AddMode.SUB), "LATENCY", 1, 2, (1, 2, 3))
        fast = int(options.fast)
        adding = {"W": fmt.width, "MODE": int(AddMode.ADD), "FAST": fast, "LATENCY": 2}
        subtracting = {"W": fmt.width, "MODE": int(AddMode.SUB), "FAST": fast, "LATENCY": 2}
        single_mode_params = {add: adding, subtract: subtracting, compare: subtracting}
        params = {"W": fmt.width, "MODE": 2, "FAST": fast, "LATENCY": 2}  # MODE=2 takes `sub` per firing
        return cls(params, options.instances, operands, outputs, (add, subtract, compare), single_mode_params)

    @property
    def addition(self) -> OperatorMode:
        return self.modes[0]

    @property
    def subtraction(self) -> OperatorMode:
        return self.modes[1]

    @property
    def comparison(self) -> OperatorMode:
        return self.modes[2]


@dataclass(frozen=True, slots=True)
class IAddPrimitive(IntPrimitive):
    operator: IAddOperator
    swap_output_permutation: ClassVar[tuple[int, ...]] = (0,)

    @property
    def mode(self) -> OperatorMode:
        return self.operator.addition

    def evaluate(self, *operands: ScalarValue) -> tuple[IntValue, ...]:
        a, b = self._validated_operands(operands)
        return (a + b,)

    def render(self, *operands: str) -> str:
        a, b = operands
        return f"{a}+{b}"


@dataclass(frozen=True, slots=True)
class ISubPrimitive(IntPrimitive):
    """
    Also serves negation as `0-x`: there is no negation module, and the subtraction saturates `-MIN` correctly. The
    order flags of the operands leave beside the difference, so a comparison of the same two operands taps this firing
    instead of taking one of its own.
    """

    operator: IAddOperator

    @property
    def mode(self) -> OperatorMode:
        return self.operator.subtraction

    def flag_of(self, relation: Relation) -> tuple[int, BoolInversion]:
        """The result carrying `relation`, and the inversion reading it: the comparison's tap, found by its port."""
        tap, inversion = ICmpPrimitive.tap_of(relation)
        return self.mode.outputs.index(self.operator.comparison.outputs[tap]), inversion

    def evaluate(self, *operands: ScalarValue) -> tuple[IntValue | bool, ...]:
        a, b = self._validated_operands(operands)
        ordering = a.compare(b)
        return a - b, ordering > 0, ordering == 0, ordering < 0

    def render(self, *operands: str) -> str:
        a, b = operands
        return f"{a}-{b}"

    def render_output(self, result: int, inversion: BoolInversion | None, *operands: str) -> str:
        if inversion is None:
            return self.render(*operands)
        tap = self.operator.comparison.outputs.index(self.mode.outputs[result])
        return ICmpPrimitive(self.operator).render_output(tap, inversion, *operands)


@dataclass(frozen=True, slots=True)
class ICmpPrimitive(IntPrimitive, ComparatorPrimitive):
    """Two's complement is totally ordered."""

    operator: IAddOperator

    @property
    def mode(self) -> OperatorMode:
        return self.operator.comparison

    def evaluate(self, *operands: ScalarValue) -> tuple[bool, ...]:
        a, b = self._validated_operands(operands)
        ordering = a.compare(b)
        return ordering > 0, ordering == 0, ordering < 0


@dataclass(frozen=True, slots=True)
class IMulsOptions(BaseOperatorOptions):
    stage_product: int = 0
    """Splitting the product is useful when the width exceeds the DSP slice input. See Verilog holoso_imuls."""


class IMulOperator(HardwareOperator):
    __slots__ = ()
    name = "imuls"

    @classmethod
    def build(cls, fmt: IntFormat, options: IMulsOptions) -> Self:
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
class IDivsOptions(BaseOperatorOptions): ...


class IDivOperator(HardwareOperator):
    __slots__ = ()
    name = "idivs"
    error_ports = ("div0",)

    @classmethod
    def build(cls, fmt: IntFormat, options: IDivsOptions) -> Self:
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
class IAbssOptions(BaseOperatorOptions): ...


class IAbsOperator(HardwareOperator):
    __slots__ = ()
    name = "iabss"

    @classmethod
    def build(cls, fmt: IntFormat, options: IAbssOptions) -> Self:
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
class IShftOptions(BaseOperatorOptions): ...


class ShiftMode(IntEnum):
    """The value driven on `right`."""

    LEFT = 0
    RIGHT = 1


class IShftOperator(HardwareOperator):
    """
    One barrel shifter serving the left shift and the right shift, chosen per firing on `right`, at one latency. The
    shift is the raw bit shift either way, so the module raises no saturation sideband.
    """

    __slots__ = ()
    name = "ishft"
    mode_port = ModePort("right", 1)

    @classmethod
    def build(cls, fmt: IntFormat, options: IShftOptions) -> Self:
        ints = IntType(fmt)
        modes = tuple(OperatorMode(int(mode), "LATENCY", 1, 2, (0,)) for mode in ShiftMode)
        operands = (OperatorPort("x", ints), OperatorPort("shamt", ints))
        outputs = (OperatorPort("shft", ints),)
        return cls({"W": fmt.width, "LATENCY": 2}, options.instances, operands, outputs, modes)

    def mode_of_shift(self, mode: ShiftMode) -> OperatorMode:
        return self.mode_of(int(mode))


@dataclass(frozen=True, slots=True)
class IShlPrimitive(IntPrimitive):
    """
    An arithmetic shift by a signed amount, left when positive and right when negative. A left shift lets the high
    bits fall off the word: saturating instead is what a multiplication by a power of two does, on the multiplier.
    """

    operator: IShftOperator

    @property
    def mode(self) -> OperatorMode:
        return self.operator.mode_of_shift(ShiftMode.LEFT)

    def evaluate(self, *operands: ScalarValue) -> tuple[IntValue, ...]:
        a, b = self._validated_operands(operands)
        return (a.shift_left(b),)

    def render(self, *operands: str) -> str:
        a, b = operands
        return f"{a}<<{b}"


@dataclass(frozen=True, slots=True)
class IShrPrimitive(IntPrimitive):
    """
    The mirror of IShlPrimitive on the same operator: right when positive and left when negative, that left shift
    dropping what leaves the word as well.
    """

    operator: IShftOperator

    @property
    def mode(self) -> OperatorMode:
        return self.operator.mode_of_shift(ShiftMode.RIGHT)

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
    An arithmetic shift by a count fixed at compile time, left when positive; the raw bit shift, as the pooled
    `holoso_ishft` computes for a runtime count, so a left shift drops what leaves the word.
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
        return (a.shift_left(IntValue.from_int(self.fmt, self.shamt)),)


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
