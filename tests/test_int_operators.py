"""
The integer operators and their primitives, pooled and inline: their reference semantics, their closed-form timing, and
the one knob among them. The lowering selects these (pinned in `test_int_selection`); here they are driven directly
because only a direct drive can sweep every operand of the narrow widths exhaustively.

The sweeps score `evaluate` against an independent fixed-width oracle (`tests/hdl/hdl_integer_oracle.py`), so the
values are checked rather than merely claimed. What they do NOT check is the hardware: `tests/test_rtl.py` scores each
block's RTL against its `evaluate`, the kernel cosimulations drive the connections the emitter makes, and
`tests/hdl/test_int_inline.py` the inline expressions, where a wide-bank expression that lost its sign extension would
fail while every assertion here passed.
"""

from collections.abc import Callable, Sequence
from dataclasses import fields
from typing import get_args, get_type_hints

import pytest
import zkf

import holoso
from holoso import (
    FFromIntOptions,
    FloatFormat,
    FloatType,
    FloatValue,
    FRintOptions,
    IAbssOptions,
    IAddsOptions,
    IDivsOptions,
    IMulsOptions,
    IntFormat,
    IPopcntOptions,
    IShftOptions,
    OperatorOptions,
    Options,
    UnsupportedConstruct,
)
from holoso._operators import (
    BoolToIntPrimitive,
    FFromIntPrimitive,
    FRintPrimitive,
    FRintToIntPrimitive,
    HardwareOperator,
    IAbsPrimitive,
    IAddPrimitive,
    IFloorDivPrimitive,
    IModPrimitive,
    IMulPrimitive,
    IntBwAndPrimitive,
    IntBwNotPrimitive,
    IntBwOrPrimitive,
    IntBwXorPrimitive,
    IntShiftConstPrimitive,
    IntToBoolPrimitive,
    IPopcntPrimitive,
    IShlPrimitive,
    IShrPrimitive,
    ICmpPrimitive,
    ISubPrimitive,
    PooledPrimitive,
    RoundMode,
)
from holoso._operators._common import BaseOperatorOptions
from holoso._operators._float import FFromIntOperator, FRintOperator
from holoso._operators._int import (
    IAbsOperator,
    IAddOperator,
    IDivOperator,
    IMulOperator,
    IPopcntOperator,
    IShftOperator,
)
from holoso._operators._common import TotalOrdering
from holoso._operators._int._inline import _IntInlinePrimitive
from holoso._util import Relation
from holoso._type import IntType
from holoso._value import IntValue

from ._modelref import build_ops, instantiated_modules, with_instances
from .hdl.hdl_integer_oracle import (
    expected_iadds_add,
    expected_iadds_cmp,
    expected_iadds_sub,
    expected_idivs,
    expected_imuls,
    expected_unary,
    ishl,
    ishr,
    signed,
)
from . import _options as lean

EXHAUSTIVE_WIDTHS = (2, 3, 4, 5, 6)
PRODUCTION_WIDTHS = (24, 33, 44)


def _corners(fmt: IntFormat) -> list[int]:
    return [fmt.min, fmt.min + 1, -3, -2, -1, 0, 1, 2, 3, fmt.max - 1, fmt.max]


def _int_format(primitive: PooledPrimitive | _IntInlinePrimitive) -> IntFormat:
    operand = primitive.signature.operand_types[0]
    assert isinstance(operand, IntType)
    return operand.fmt


def _evaluate(primitive: PooledPrimitive | _IntInlinePrimitive, *operands: int) -> int | bool:
    fmt = _int_format(primitive)
    result = primitive.evaluate(*(IntValue.from_int(fmt, operand) for operand in operands))
    assert isinstance(result, IntValue | bool)
    return result if isinstance(result, bool) else result.value


def _bits(primitives: Sequence[PooledPrimitive], *operand_bits: int) -> dict[str, int]:
    """The answers of one drive's `primitives` by the RTL port each drives, so an oracle dict compares directly."""
    answers: dict[str, int] = {}
    for primitive in primitives:
        result = primitive.evaluate(*(IntValue.from_bits(_int_format(primitive), bits) for bits in operand_bits))
        assert isinstance(result, IntValue | bool)
        answers[primitive.output_name] = int(result) if isinstance(result, bool) else result.bits
    return answers


def _oracle(expected: dict[str, int], primitives: Sequence[PooledPrimitive]) -> dict[str, int]:
    """The answers on the ports the primitives answer."""
    return {primitive.output_name: expected[primitive.output_name] for primitive in primitives}


def _flags(adder: IAddOperator) -> list[ICmpPrimitive]:
    return [ICmpPrimitive(adder, ordering) for ordering in TotalOrdering]


@pytest.mark.parametrize("width", EXHAUSTIVE_WIDTHS)
def test_every_operator_answers_as_the_rtl_does_over_every_operand_pair(width: int) -> None:
    fmt = IntFormat(width)
    adder = IAddOperator.build(fmt, IAddsOptions())
    add, subtract = [IAddPrimitive(adder)], [ISubPrimitive(adder), *_flags(adder)]
    shifter = IShftOperator.build(fmt, IShftOptions())
    shift_left, shift_right = [IShlPrimitive(shifter)], [IShrPrimitive(shifter)]
    divider = IDivOperator.build(fmt, IDivsOptions())
    idiv = [IFloorDivPrimitive(divider), IModPrimitive(divider)]
    unary: list[PooledPrimitive] = [
        IAbsPrimitive(IAbsOperator.build(fmt, IAbssOptions())),
        IPopcntPrimitive(IPopcntOperator.build(fmt, IPopcntOptions())),
    ]
    # Staging is a timing knob, so every multiplier configuration must answer the one product; a word within one slice
    # takes up to three product registers.
    multipliers = [
        IMulPrimitive(IMulOperator.build(fmt, IMulsOptions(stage_core_mul=stage), lean.WMULTIPLIER))
        for stage in range(4)
    ]
    for a in range(1 << width):
        for primitive in unary:
            want = expected_unary(primitive.operator.name, a, width)
            assert _bits([primitive], a) == _oracle(want, [primitive]), (type(primitive).__name__, a)
        for b in range(1 << width):
            # One block adds and subtracts, a subtraction ordering its operands beside, so each operation has its own
            # reference, and so has each direction of the shifter.
            assert _bits(add, a, b) == _oracle(expected_iadds_add(a, b, width), add), (a, b)
            difference, order = expected_iadds_sub(a, b, width), expected_iadds_cmp(a, b, width)
            assert _bits(subtract, a, b) == _oracle({**difference, **order}, subtract), (a, b)
            assert _bits(shift_left, a, b) == {"shft": ishl(a, b, width)}, (a, b)
            assert _bits(shift_right, a, b) == {"shft": ishr(a, b, width)}, (a, b)
            for imul in multipliers:
                assert _bits([imul], a, b) == _oracle(expected_imuls(a, b, width), [imul]), (imul.latency, a, b)
            assert _bits(idiv, a, b) == _oracle(expected_idivs(a, b, width), idiv), (a, b)


@pytest.mark.parametrize("width", EXHAUSTIVE_WIDTHS)
def test_floor_division_obeys_the_division_identity(width: int) -> None:
    # What the oracle comparison cannot show: that the answers are a division at all, not a shared misreading.
    fmt = IntFormat(width)
    divider = IDivOperator.build(fmt, IDivsOptions())
    for num in range(fmt.min, fmt.max + 1):
        for den in range(fmt.min, fmt.max + 1):
            quotient, remainder = _evaluate(IFloorDivPrimitive(divider), num, den), _evaluate(
                IModPrimitive(divider), num, den
            )
            if den == 0 or (num == fmt.min and den == -1):
                continue  # no quotient exists, or none the width holds; the oracle pins what is answered instead
            assert num == den * quotient + remainder
            assert abs(remainder) < abs(den)
            assert not remainder or (remainder < 0) == (den < 0), "the remainder follows the divisor, as floor does"


@pytest.mark.parametrize("width", EXHAUSTIVE_WIDTHS)
def test_comparator_flags_are_one_hot_and_serve_every_relation(width: int) -> None:
    fmt = IntFormat(width)
    adder = IAddOperator.build(fmt, IAddsOptions())
    answers: dict[Relation, Callable[[int, int], bool]] = {
        Relation.GT: lambda a, b: a > b,
        Relation.EQ: lambda a, b: a == b,
        Relation.LT: lambda a, b: a < b,
        Relation.GE: lambda a, b: a >= b,
        Relation.NE: lambda a, b: a != b,
        Relation.LE: lambda a, b: a <= b,
    }
    for a in range(fmt.min, fmt.max + 1):
        for b in range(fmt.min, fmt.max + 1):
            assert sum(bool(_evaluate(flag, a, b)) for flag in _flags(adder)) == 1, "the order flags are one-hot"
            for relation, answer in answers.items():
                flag, inversion = ICmpPrimitive.of(adder, relation)
                assert inversion.apply(bool(_evaluate(flag, a, b))) == answer(a, b), relation


@pytest.mark.parametrize("width", PRODUCTION_WIDTHS)
def test_edge_cases_at_the_production_widths(width: int) -> None:
    # The sweeps stop far below these, and saturation is where a width-dependent slip would hide.
    fmt = IntFormat(width)
    adder = IAddOperator.build(fmt, IAddsOptions())
    iadd = IAddPrimitive(adder)
    isub = [ISubPrimitive(adder), *_flags(adder)]
    imul = IMulPrimitive(IMulOperator.build(fmt, IMulsOptions(), lean.WMULTIPLIER))
    divider = IDivOperator.build(fmt, IDivsOptions())
    idiv = [IFloorDivPrimitive(divider), IModPrimitive(divider)]

    def each(primitives: Sequence[PooledPrimitive], *operands: int) -> list[int | bool]:
        return [_evaluate(primitive, *operands) for primitive in primitives]

    assert _evaluate(IAbsPrimitive(IAbsOperator.build(fmt, IAbssOptions())), fmt.min) == fmt.max
    assert _evaluate(iadd, fmt.min, fmt.min) == fmt.min
    assert _evaluate(iadd, fmt.max, fmt.max) == fmt.max
    assert each(isub, fmt.min, fmt.max) == [fmt.min, False, False, True], "the order survives the overflow"
    assert each(isub, 0, fmt.min) == [fmt.max, True, False, False], "negation via 0-x saturates, not wraps"
    assert _evaluate(imul, fmt.min, fmt.min) == fmt.max
    assert _evaluate(imul, fmt.min, 1) == fmt.min
    assert each(idiv, fmt.min, -1) == [fmt.max, 0]
    assert each(idiv, -7, 2) == [-4, 1], "the quotient floors, as Python's // does"

    for numerator in _corners(fmt):
        assert each(idiv, numerator, 0) == [fmt.min if numerator < 0 else fmt.max, numerator]

    shifter = IShftOperator.build(fmt, IShftOptions())
    left = IShlPrimitive(shifter)
    for count in (0, 1, width - 1, width, width + 1, fmt.max):
        assert _evaluate(left, 0, count) == 0
        assert _evaluate(left, -1, -count) == -1, "sign fill makes -1 a fixed point of every right shift"
    # The shift is raw: the bit walks up into the sign and then off the word.
    assert _evaluate(left, 1, width - 2) == 1 << (width - 2)
    assert _evaluate(left, 1, width - 1) == fmt.min
    assert _evaluate(left, 1, width) == 0
    assert _evaluate(left, fmt.min, fmt.min) == -1, "a count past the word saturates to the word itself"
    assert _evaluate(left, fmt.max, 1) == -2, "the left shift drops what leaves the word rather than clamping"

    right = IShrPrimitive(shifter)
    for count in (0, 1, width - 1, width, width + 1, fmt.max):
        assert _evaluate(right, 0, count) == 0
        assert _evaluate(right, -1, count) == -1, "sign fill makes -1 a fixed point of every right shift"
    # A negative count shifts left, and that shift is raw: the bit walks up into the sign and then off the word.
    assert _evaluate(right, 1, 2 - width) == 1 << (width - 2)
    assert _evaluate(right, 1, 1 - width) == fmt.min
    assert _evaluate(right, 1, -width) == 0
    assert _evaluate(right, fmt.min, fmt.min) == 0, "|MIN| is past the word, so the left it asks for empties it"
    assert _evaluate(right, fmt.max, -1) == -2, "the left shift drops what leaves the word rather than clamping"


def test_each_adder_primitive_drives_its_own_immediate() -> None:
    # The two drive the one `sub` input to opposite values; a relation must tap a flag, not the difference.
    adder = IAddOperator.build(IntFormat(33), IAddsOptions())
    add, subtract = IAddPrimitive(adder), ISubPrimitive(adder)
    assert (add.immediates, subtract.immediates) == ({"sub": False}, {"sub": True})
    assert (add.output_name, subtract.output_name) == ("y", "y")
    flags = {Relation.GT: "a_gt_b", Relation.EQ: "a_eq_b", Relation.LT: "a_lt_b"}
    for relation, flag in flags.items():
        primitive, inversion = ICmpPrimitive.of(adder, relation)
        assert primitive.immediates == {"sub": True}
        assert primitive.output_name == flag and inversion.is_identity


@pytest.mark.parametrize("width", EXHAUSTIVE_WIDTHS)
def test_the_two_shift_directions_mirror_each_other_over_every_operand_pair(width: int) -> None:
    # Each direction must be the other read backwards: the direction bit and the sign of the count say the same thing.
    # MIN has no negation in the format, so it is the one count they legitimately part on.
    fmt = IntFormat(width)
    shifter = IShftOperator.build(fmt, IShftOptions())
    left, right = IShlPrimitive(shifter), IShrPrimitive(shifter)
    assert (left.immediates, right.immediates) == ({"right": False}, {"right": True})
    for a in range(1 << width):
        for b in range(fmt.min + 1, fmt.max + 1):
            mirrored = right.evaluate(IntValue.from_bits(fmt, a), IntValue.from_int(fmt, -b))
            assert isinstance(mirrored, IntValue)
            assert _bits([left], a, fmt.encode(b))["shft"] == mirrored.bits, (a, b)
    for a in range(1 << width):
        assert _bits([right], a, fmt.encode(fmt.min))["shft"] == 0, "a left shift past the word empties the word"


@pytest.mark.parametrize("width", (2, 3, 24, 33, 44))
def test_closed_form_latencies(width: int) -> None:
    fmt = IntFormat(width)
    idiv = IFloorDivPrimitive(IDivOperator.build(fmt, IDivsOptions()))
    assert idiv.latency == 1 + -(-width // 2), "one radix-4 step per two quotient bits, rounded up"
    for primitive in (
        IAddPrimitive(IAddOperator.build(fmt, IAddsOptions())),
        ISubPrimitive(IAddOperator.build(fmt, IAddsOptions())),
        IAbsPrimitive(IAbsOperator.build(fmt, IAbssOptions())),
        IShlPrimitive(IShftOperator.build(fmt, IShftOptions())),
        IShrPrimitive(IShftOperator.build(fmt, IShftOptions())),
        IPopcntPrimitive(IPopcntOperator.build(fmt, IPopcntOptions())),
    ):
        assert primitive.drive.timing == zkf.Timing(0, 1)


@pytest.mark.parametrize("stage_core_mul", range(5))
def test_multiplier_staging_costs_exactly_one_cycle_each(stage_core_mul: int) -> None:
    # The 33-bit word cuts into a two-by-two grid of 18-bit slices, which takes up to four product registers.
    operator = IMulOperator.build(IntFormat(33), IMulsOptions(stage_core_mul=stage_core_mul), lean.WMULTIPLIER)
    primitive = IMulPrimitive(operator)
    assert primitive.drive.timing == zkf.Timing(stage_core_mul, 1)


def test_only_the_divider_reports_an_error_and_only_a_division_by_zero() -> None:
    # Saturation is the integer type's defined behaviour, and the saturating operators are speculatable, so none of
    # them may raise the machine's error flag; MIN // -1 saturates the divider too, and must stay off `div0`.
    fmt = IntFormat(33)
    assert IDivOperator.build(fmt, IDivsOptions()).error_ports == ("div0",)
    for operator in (
        IAddOperator.build(fmt, IAddsOptions()),
        IMulOperator.build(fmt, IMulsOptions(), lean.WMULTIPLIER),
        IAbsOperator.build(fmt, IAbssOptions()),
        IShftOperator.build(fmt, IShftOptions()),
        IPopcntOperator.build(fmt, IPopcntOptions()),
    ):
        assert operator.error_ports == (), operator.name


@pytest.mark.parametrize("width", (2, 3, *PRODUCTION_WIDTHS))
def test_the_population_count_counts_the_magnitude(width: int) -> None:
    # Counting the magnitude is what makes -1 answer 1 rather than filling the word, and what caps the count one
    # short of the width -- MIN counts its single bit because the negation that overflows a signed word is exactly
    # the magnitude.
    fmt = IntFormat(width)
    operator = IPopcntOperator.build(fmt, IPopcntOptions())
    primitive = IPopcntPrimitive(operator)
    assert primitive.signature.result_type == IntType(fmt), "the count is an ordinary machine integer"
    corners = (0, -1, fmt.min, fmt.min + 1, fmt.max)
    assert {value: _evaluate(primitive, value) for value in corners} == {
        0: 0,
        -1: 1,
        fmt.min: 1,
        fmt.min + 1: width - 1,
        fmt.max: width - 1,
    }


def test_multiplier_staging_is_part_of_the_hardware_identity() -> None:
    # The operator is the resource-sharing key: two differently staged multipliers must not pool onto one module.
    fmt = IntFormat(33)
    operators = [IMulOperator.build(fmt, IMulsOptions(stage_core_mul=stage), lean.WMULTIPLIER) for stage in range(5)]
    assert len(set(operators)) == len(operators)
    default = IMulOperator.build(fmt, IMulsOptions(), lean.WMULTIPLIER)
    assert default == IMulOperator.build(fmt, IMulsOptions(stage_core_mul=0), lean.WMULTIPLIER)


def test_the_multiplier_knob_reaches_the_built_machine() -> None:
    # It must arrive carrying the user's staging, the machine's integer format (not the float one) and the machine's
    # multiplier width, which is not the one every other test machine states.
    options = Options(operator=OperatorOptions(imuls=IMulsOptions(stage_core_mul=3)), wint_min=44, wmultiplier=27)
    imul = build_ops(options, 44).imuls
    assert {*imul.operand_ports.values(), *imul.output_ports.values()} == {IntType(IntFormat(44))}
    assert IMulPrimitive(imul).latency == 3
    assert (imul.block.wint, imul.block.stage_core_mul, imul.block.wmultiplier) == (44, 3, 27)
    assert build_ops(Options(operator=OperatorOptions(), wmultiplier=18), 16).imuls == IMulOperator.build(
        IntFormat(16), IMulsOptions(stage_core_mul=0), 18
    )


@pytest.mark.parametrize("width", EXHAUSTIVE_WIDTHS)
def test_inline_bitwise_and_casts_answer_over_every_operand(width: int) -> None:
    # The reference works on the raw bit patterns, so it knows nothing of the primitive's own sign convention. A
    # bitwise combination never leaves the range, so a saturating implementation would answer the rail for `~min`.
    fmt = IntFormat(width)
    mask = (1 << width) - 1
    conjunction, disjunction, exclusive = IntBwAndPrimitive(fmt), IntBwOrPrimitive(fmt), IntBwXorPrimitive(fmt)
    complement, truth = IntBwNotPrimitive(fmt), IntToBoolPrimitive(fmt)
    for a in range(1 << width):
        assert _evaluate(complement, signed(a, width)) == signed(~a & mask, width)
        assert truth.evaluate(IntValue.from_bits(fmt, a)) == (a != 0)
        for b in range(1 << width):
            operands = (signed(a, width), signed(b, width))
            assert _evaluate(conjunction, *operands) == signed(a & b, width)
            assert _evaluate(disjunction, *operands) == signed(a | b, width)
            assert _evaluate(exclusive, *operands) == signed(a ^ b, width)
    assert _evaluate(complement, fmt.min) == fmt.max and _evaluate(complement, fmt.max) == fmt.min

    cast = BoolToIntPrimitive(fmt)
    assert cast.evaluate(True) == IntValue.from_int(fmt, 1)
    assert cast.evaluate(False) == IntValue.from_int(fmt, 0)


@pytest.mark.parametrize("width", EXHAUSTIVE_WIDTHS)
def test_constant_shift_over_every_count_and_operand(width: int) -> None:
    fmt = IntFormat(width)
    for count in (count for count in range(1 - width, width) if count != 0):
        primitive = IntShiftConstPrimitive(fmt, count)
        for a in range(1 << width):
            want = ishl(a, fmt.encode(count), width)
            assert primitive.evaluate(IntValue.from_bits(fmt, a)) == IntValue.from_bits(fmt, want), (count, a)

    assert IntShiftConstPrimitive(fmt, 1).render("r0") == "r0<<1"
    assert IntShiftConstPrimitive(fmt, -1).render("r0") == "r0>>1"
    assert _evaluate(IntShiftConstPrimitive(fmt, 1 - width), -1) == -1, "sign fill survives the widest right shift"
    assert _evaluate(IntShiftConstPrimitive(fmt, width - 1), fmt.min) == 0, "the sign bit shifts off the word"


def test_the_constant_shift_is_the_raw_shift_and_not_the_saturating_one() -> None:
    # The inline shift drops what leaves the word, as the pooled shifter does; the saturating reading is the
    # multiplier's.
    fmt = IntFormat(33)
    assert _evaluate(IntShiftConstPrimitive(fmt, 1), fmt.max) == -2
    assert _evaluate(IShlPrimitive(IShftOperator.build(fmt, IShftOptions())), fmt.max, 1) == -2
    assert _evaluate(IMulPrimitive(IMulOperator.build(fmt, IMulsOptions(), lean.WMULTIPLIER)), fmt.max, 2) == fmt.max


@pytest.mark.parametrize("wint", (4, 17, 44))
def test_the_conversions_saturate_at_the_rails_and_round_trip_the_extremes(wint: int) -> None:
    ffmt, ifmt = FloatFormat(8, 24), IntFormat(wint)
    rint = FRintOperator.build(ffmt, ifmt, lean.frint())
    from_int = FFromIntPrimitive(FFromIntOperator.build(ffmt, ifmt, lean.ffromint()))

    def convert(value: float, mode: RoundMode) -> int:
        result = FRintToIntPrimitive(rint, mode).evaluate(ffmt.encode(value))
        assert isinstance(result, IntValue)
        return result.value

    for mode in RoundMode:
        assert convert(float("inf"), mode) == ifmt.max, "an infinity reaches the rail, it is not an error"
        assert convert(float("-inf"), mode) == ifmt.min
        assert convert(float(ifmt.max) * 4.0, mode) == ifmt.max
        assert convert(float(ifmt.min) * 4.0, mode) == ifmt.min
    assert [convert(2.5, mode) for mode in RoundMode] == [2, 2, 3, 2]
    assert [convert(-2.5, mode) for mode in RoundMode] == [-2, -3, -2, -2]

    # MAX is unrepresentable once the width outgrows the mantissa, so it converts up and saturates coming back.
    for extreme in (ifmt.min, ifmt.max):
        image = from_int.evaluate(IntValue.from_int(ifmt, extreme))
        assert isinstance(image, FloatValue)
        back = FRintToIntPrimitive(rint, RoundMode.NEAREST_EVEN).evaluate(image)
        assert isinstance(back, IntValue) and back.value == extreme


def test_rounding_before_converting_is_not_the_same_as_converting_with_that_mode() -> None:
    # Strength reduction's conversion of a rounding in the rounding's own mode (the integer result of the one rounding)
    # is a rewrite that can change the answer, and the fastmath charter (DESIGN.md, Direction) licenses it anyway.
    # Here 3.5 rounds to +inf, which saturates, while a direct nearest-even conversion answers 4.
    ffmt, ifmt = FloatFormat(2, 4), IntFormat(33)
    rint = FRintOperator.build(ffmt, ifmt, lean.frint())
    rounded = FRintPrimitive(rint, RoundMode.NEAREST_EVEN).evaluate(ffmt.encode(3.5))
    fused = FRintToIntPrimitive(rint, RoundMode.NEAREST_EVEN).evaluate(ffmt.encode(3.5))
    staged = FRintToIntPrimitive(rint, RoundMode.TRUNC).evaluate(rounded)
    assert isinstance(fused, IntValue) and isinstance(staged, IntValue)
    assert fused.value == 4 and staged.value == ifmt.max


def test_the_conversion_knobs_reach_the_built_machine() -> None:
    ops = build_ops(
        Options(
            operator=OperatorOptions(
                ffromint=lean.ffromint(stage_input=1, stage_core=FFromIntOptions.CoreStages.PACK),
                frint=lean.frint(stage_input=2),
            ),
            ffmt=FloatFormat(6, 18),
            wint_min=44,
            wmultiplier=18,
        ),
        44,
    )
    assert FFromIntPrimitive(ops.ffromint).latency == 3
    assert {FRintPrimitive(ops.frint, rounding).latency for rounding in RoundMode} == {3}
    # The latency cannot tell the input stage from the output one, so the blocks are compared whole.
    fmt = FloatFormat(6, 18)
    pack = FFromIntOptions.CoreStages.PACK
    assert ops.ffromint.block == zkf.FromInt(fmt=fmt, wint=44, stage_input=1, stage_core=pack, stage_output=1)
    none = FRintOptions.CoreStages.NONE
    assert ops.frint.block == zkf.Rint(fmt=fmt, wint=44, stage_input=2, stage_core=none, stage_output=1)
    assert list(ops.ffromint.operand_ports.values()) == [IntType(IntFormat(44))]
    assert list(ops.ffromint.output_ports.values()) == [FloatType(FloatFormat(6, 18))]
    assert list(ops.frint.operand_ports.values()) == [FloatType(FloatFormat(6, 18))]
    assert list(ops.frint.output_ports.values()) == [
        FloatType(FloatFormat(6, 18)),
        IntType(IntFormat(44)),
    ]


def _everything_configured() -> Options:
    """Every float operator present, so the catalogue walks its whole surface."""
    return Options(
        operator=OperatorOptions(
            fadd=lean.fadd(),
            fmul=lean.fmul(),
            fdivsqrt=lean.fdivsqrt(),
            fmul_ilog2=lean.fmul_ilog2(),
            filog2=lean.filog2(),
            fcmp=lean.fcmp(),
            frint=lean.frint(),
            ffma=lean.ffma(),
            fexp2=lean.fexp2(),
            flog2=lean.flog2(),
            fcordic=lean.fcordic(),
            ffromint=lean.ffromint(),
        ),
        ffmt=FloatFormat(6, 18),
        wint_min=33,
        wmultiplier=18,
    )


def test_the_catalogue_builds_every_operator_for_the_machines_own_formats() -> None:
    # A conversion operator carries one format per side, so a check keyed on a single format could not see a wrong
    # `ifmt` at all. The catalogue BUILDS each operator from the machine's formats, so the mismatch is unrepresentable
    # rather than merely caught; this walks the whole catalogue and pins that.
    options = _everything_configured()
    ops = build_ops(options, options.wint_min)
    for name in (field.name for field in fields(OperatorOptions)):
        operator: HardwareOperator = getattr(ops, name)
        for port, ty in [*operator.operand_ports.items(), *operator.output_ports.items()]:
            if isinstance(ty, FloatType):
                assert ty.fmt == ops.float_format, (name, port)
            if isinstance(ty, IntType):
                assert ty.fmt == ops.int_format, (name, port)


def test_every_operator_is_publicly_configurable() -> None:
    # An operator the public options do not name is unreachable by configuration, and one whose knobs are not publicly
    # exported cannot be spelled at all.
    for name, annotation in get_type_hints(OperatorOptions).items():
        unwrapped = [arg for arg in get_args(annotation) if arg is not type(None)] or [annotation]
        assert len(unwrapped) == 1, name
        (knob,) = unwrapped
        assert issubclass(knob, BaseOperatorOptions) and knob is not BaseOperatorOptions
        assert getattr(holoso, knob.__name__) is knob
    options = _everything_configured()
    ops = build_ops(options, options.wint_min)
    kinds = {
        kind
        for module in (holoso._operators._float, holoso._operators._int)
        for kind in vars(module).values()
        if isinstance(kind, type) and issubclass(kind, HardwareOperator) and kind is not HardwareOperator
    }
    assert {type(getattr(ops, field.name)) for field in fields(OperatorOptions)} == kinds
    # Asking whether a machine has an operator reads the options field of that name.
    assert all(getattr(ops, field.name).name == field.name for field in fields(OperatorOptions))


def test_the_instance_cap_reaches_the_operator_but_never_the_rtl() -> None:
    # The count is a machine-level budget, so it rides operator identity (two configurations are different operators)
    # but must not reach the block, whose name is the module every instance of the operator renders.
    narrow = IMulOperator.build(IntFormat(32), IMulsOptions(), lean.WMULTIPLIER)
    wide = IMulOperator.build(IntFormat(32), IMulsOptions(instances=4), lean.WMULTIPLIER)
    assert wide.instances == 4 and narrow != wide
    options = _everything_configured()
    ops = build_ops(options, options.wint_min)
    widened = build_ops(with_instances(options, 4), options.wint_min)
    for name in (field.name for field in fields(OperatorOptions)):
        operator: HardwareOperator = getattr(ops, name)
        assert getattr(widened, name).instances == 4 and operator.instances == 1, name
        assert getattr(widened, name).block.name == operator.block.name, name


def _add(a: float, b: float) -> float:
    return a + b


def test_a_float_only_build_configures_an_integer_operator_without_instantiating_it() -> None:
    options = Options(operator=OperatorOptions(fadd=lean.fadd()), ffmt=FloatFormat(6, 18), wint_min=44, wmultiplier=18)
    ops = build_ops(options, options.wint_min)
    assert {*ops.imuls.operand_ports.values(), *ops.imuls.output_ports.values()} == {IntType(IntFormat(44))}
    for conversion in ("ffromint", "frint"):  # a conversion is optional, as every float operator is
        with pytest.raises(UnsupportedConstruct, match="not configured"):
            getattr(ops, conversion)
    result = holoso.synthesize(_add, options, name="ImulUnused")
    assert instantiated_modules(result) == {"add"}, "an available operator no kernel reaches costs no fabric"
