"""
The integer operators and their primitives, pooled and inline: their reference semantics, their closed-form timing, and
the one knob among them. The lowering selects these (pinned in `test_int_selection`); here they are driven directly
because only a direct drive can sweep every operand of the narrow widths exhaustively.

The sweeps score `evaluate` against the very oracle the HDL benches score the RTL against, so the values are
checked rather than merely claimed. What they do NOT check is the configuration the hardware is built in: the
latencies, the RTL parameters and the port names are pinned elsewhere -- by the elaboration probe in
`test_backend.py` and by the benches, which take all three from the operator itself. A wrong `QUOTIENT_FLOOR`
would leave every assertion here passing and fail there, as would a wide-bank expression that lost its sign
extension (`tests/hdl/test_int_inline.py`).
"""

from collections.abc import Callable
from dataclasses import fields
from typing import get_args, get_type_hints

import pytest

import holoso
from holoso import (
    FFromIntOptions,
    FloatFormat,
    FloatType,
    FloatValue,
    FRoundOptions,
    FToIntOptions,
    IAbsOptions,
    IAddOptions,
    ICmpOptions,
    IDivOptions,
    IMulOptions,
    IntFormat,
    IPopcntOptions,
    IShlOptions,
    IShrOptions,
    ISubOptions,
    OperatorOptions,
    Options,
    UnsupportedConstruct,
)
from holoso._operators import (
    BaseOperatorOptions,
    BoolToIntPrimitive,
    FFromIntOperator,
    FFromIntPrimitive,
    FRoundOperator,
    FRoundPrimitive,
    FToIntOperator,
    FToIntPrimitive,
    HardwareOperator,
    IAbsOperator,
    IAbsPrimitive,
    IAddOperator,
    IAddPrimitive,
    ICmpOperator,
    ICmpPrimitive,
    IDivOperator,
    IDivPrimitive,
    IMulOperator,
    IMulPrimitive,
    IntBwAndPrimitive,
    IntBwNotPrimitive,
    IntBwOrPrimitive,
    IntBwXorPrimitive,
    IntShiftConstPrimitive,
    IntToBoolPrimitive,
    IPopcntOperator,
    IPopcntPrimitive,
    IShlOperator,
    IShlPrimitive,
    IShrOperator,
    IShrPrimitive,
    ISubOperator,
    ISubPrimitive,
    OperatorPort,
    RoundMode,
)
from holoso._operators._int import IntPrimitive, IntInlinePrimitive
from holoso._util import Relation
from holoso._type import IntType
from holoso._value import IntValue

from ._modelref import build_ops
from .hdl.hdl_integer_oracle import expected_idivs, expected_imuls, expected_simple, ishl, signed

EXHAUSTIVE_WIDTHS = (2, 3, 4, 5, 6)
PRODUCTION_WIDTHS = (24, 33, 44)


def _corners(fmt: IntFormat) -> list[int]:
    return [fmt.min, fmt.min + 1, -3, -2, -1, 0, 1, 2, 3, fmt.max - 1, fmt.max]


def _int_format(primitive: IntPrimitive | IntInlinePrimitive) -> IntFormat:
    operand = primitive.signature.operand_types[0]
    assert isinstance(operand, IntType)
    return operand.fmt


def _evaluate(primitive: IntPrimitive | IntInlinePrimitive, *operands: int) -> list[int | bool]:
    fmt = _int_format(primitive)
    values: list[int | bool] = []
    for result in primitive.evaluate(*(IntValue.from_int(fmt, operand) for operand in operands)):
        assert isinstance(result, IntValue | bool)
        values.append(result if isinstance(result, bool) else result.value)
    return values


def _bits(primitive: IntPrimitive, *operand_bits: int) -> dict[str, int]:
    """Keyed by the RTL port names the module drives, so an oracle dict compares directly."""
    fmt = _int_format(primitive)
    results = primitive.evaluate(*(IntValue.from_bits(fmt, bits) for bits in operand_bits))
    return {
        port.name: int(result) if isinstance(result, bool) else result.bits
        for port, result in zip(_outputs(primitive), results, strict=True)
    }


def _oracle(expected: dict[str, int], primitive: IntPrimitive) -> dict[str, int]:
    """The value ports alone: the saturation sidebands are deliberately not modeled."""
    return {port.name: expected[port.name] for port in _outputs(primitive)}


def _outputs(primitive: IntPrimitive) -> list[OperatorPort]:
    return [primitive.operator.output_ports[port] for port in primitive.mode.outputs]


@pytest.mark.parametrize("width", EXHAUSTIVE_WIDTHS)
def test_every_operator_answers_as_the_rtl_does_over_every_operand_pair(width: int) -> None:
    fmt = IntFormat(width)
    binary = [
        IAddPrimitive(IAddOperator.build(fmt, IAddOptions())),
        ISubPrimitive(ISubOperator.build(fmt, ISubOptions())),
        ICmpPrimitive(ICmpOperator.build(fmt, ICmpOptions())),
        IShlPrimitive(IShlOperator.build(fmt, IShlOptions())),
        IShrPrimitive(IShrOperator.build(fmt, IShrOptions())),
    ]
    idiv = IDivPrimitive(IDivOperator.build(fmt, IDivOptions()))
    unary = [
        IAbsPrimitive(IAbsOperator.build(fmt, IAbsOptions())),
        IPopcntPrimitive(IPopcntOperator.build(fmt, IPopcntOptions())),
    ]
    # Staging is a timing knob, so every multiplier configuration must answer the one product.
    multipliers = [IMulPrimitive(IMulOperator.build(fmt, IMulOptions(stage_product=stage))) for stage in range(5)]
    for a in range(1 << width):
        for primitive in unary:
            want = expected_simple(primitive.operator.module_name, a, 0, width)
            assert _bits(primitive, a) == _oracle(want, primitive), (type(primitive).__name__, a)
        for b in range(1 << width):
            for primitive in binary:
                want = expected_simple(primitive.operator.module_name, a, b, width)
                assert _bits(primitive, a, b) == _oracle(want, primitive), (type(primitive).__name__, a, b)
            for imul in multipliers:
                assert _bits(imul, a, b) == _oracle(expected_imuls(a, b, width), imul), (imul.operator.params, a, b)
            assert _bits(idiv, a, b) == _oracle(expected_idivs(a, b, width, True), idiv), (a, b)


@pytest.mark.parametrize("width", EXHAUSTIVE_WIDTHS)
def test_floor_division_obeys_the_division_identity(width: int) -> None:
    # What the oracle comparison cannot show: that the answers are a division at all, not a shared misreading.
    fmt = IntFormat(width)
    primitive = IDivPrimitive(IDivOperator.build(fmt, IDivOptions()))
    for num in range(fmt.min, fmt.max + 1):
        for den in range(fmt.min, fmt.max + 1):
            quotient, remainder = _evaluate(primitive, num, den)
            if den == 0 or (num == fmt.min and den == -1):
                continue  # no quotient exists, or none the width holds; the oracle pins what is answered instead
            assert num == den * quotient + remainder
            assert abs(remainder) < abs(den)
            assert not remainder or (remainder < 0) == (den < 0), "the remainder follows the divisor, as floor does"


@pytest.mark.parametrize("width", EXHAUSTIVE_WIDTHS)
def test_comparator_flags_are_one_hot_and_serve_every_relation(width: int) -> None:
    fmt = IntFormat(width)
    primitive = ICmpPrimitive(ICmpOperator.build(fmt, ICmpOptions()))
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
            flags = _evaluate(primitive, a, b)
            assert sum(flags) == 1, "the order flags are one-hot"
            for relation, answer in answers.items():
                port, inversion = primitive.tap_of(relation)
                assert inversion.apply(bool(flags[port])) == answer(a, b), relation


@pytest.mark.parametrize("width", PRODUCTION_WIDTHS)
def test_edge_cases_at_the_production_widths(width: int) -> None:
    # The sweeps stop far below these, and saturation is where a width-dependent slip would hide.
    fmt = IntFormat(width)
    iadd = IAddPrimitive(IAddOperator.build(fmt, IAddOptions()))
    isub = ISubPrimitive(ISubOperator.build(fmt, ISubOptions()))
    imul = IMulPrimitive(IMulOperator.build(fmt, IMulOptions()))
    idiv = IDivPrimitive(IDivOperator.build(fmt, IDivOptions()))
    assert _evaluate(IAbsPrimitive(IAbsOperator.build(fmt, IAbsOptions())), fmt.min) == [fmt.max]
    assert _evaluate(iadd, fmt.min, fmt.min) == [fmt.min]
    assert _evaluate(iadd, fmt.max, fmt.max) == [fmt.max]
    assert _evaluate(isub, fmt.min, fmt.max) == [fmt.min]
    assert _evaluate(isub, 0, fmt.min) == [fmt.max], "negation via 0-x saturates instead of wrapping"
    assert _evaluate(imul, fmt.min, fmt.min) == [fmt.max]
    assert _evaluate(imul, fmt.min, 1) == [fmt.min]
    assert _evaluate(idiv, fmt.min, -1) == [fmt.max, 0]
    assert _evaluate(idiv, -7, 2) == [-4, 1], "the quotient floors, as Python's // does"

    for numerator in _corners(fmt):
        assert _evaluate(idiv, numerator, 0) == [fmt.min if numerator < 0 else fmt.max, numerator]

    shift = IShlPrimitive(IShlOperator.build(fmt, IShlOptions()))
    for count in (0, 1, width - 1, width, width + 1, fmt.max):
        assert _evaluate(shift, 0, count) == [0, 0]
        assert _evaluate(shift, -1, -count) == [-1, -1], "sign fill makes -1 a fixed point of every right shift"
        assert _evaluate(shift, 1, count)[1] == (1 << count if count < width - 1 else fmt.max)
    assert _evaluate(shift, fmt.min, fmt.min) == [-1, -1], "a count past the word saturates to the word itself"
    assert _evaluate(shift, fmt.max, 1) == [-2, fmt.max], "the raw shift drops the bit the saturating one clamps on"

    right = IShrPrimitive(IShrOperator.build(fmt, IShrOptions()))
    for count in (0, 1, width - 1, width, width + 1, fmt.max):
        assert _evaluate(right, 0, count) == [0]
        assert _evaluate(right, -1, count) == [-1], "sign fill makes -1 a fixed point of every right shift"
    # A negative count shifts left, and that shift is raw: the bit walks up into the sign and then off the word.
    assert _evaluate(right, 1, 2 - width) == [1 << (width - 2)]
    assert _evaluate(right, 1, 1 - width) == [fmt.min]
    assert _evaluate(right, 1, -width) == [0]
    assert _evaluate(right, fmt.min, fmt.min) == [0], "|MIN| is past the word, so the left it asks for empties it"
    assert _evaluate(right, fmt.max, -1) == [-2], "the left shift drops what leaves the word rather than clamping"


@pytest.mark.parametrize("width", EXHAUSTIVE_WIDTHS)
def test_the_two_shifters_mirror_each_other_over_every_operand_pair(width: int) -> None:
    # Each must be the other read backwards, or the pair is not worth two modules. MIN has no negation in the
    # format, so it is the one count they legitimately part on.
    fmt = IntFormat(width)
    left, right = IShlPrimitive(IShlOperator.build(fmt, IShlOptions())), IShrPrimitive(
        IShrOperator.build(fmt, IShrOptions())
    )
    for a in range(1 << width):
        for b in range(fmt.min + 1, fmt.max + 1):
            (mirrored,) = right.evaluate(IntValue.from_bits(fmt, a), IntValue.from_int(fmt, -b))
            assert _bits(left, a, fmt.encode(b))["shft"] == mirrored.bits, (a, b)
    for a in range(1 << width):
        assert _bits(right, a, fmt.encode(fmt.min))["shft"] == 0, "a left shift past the word empties the word"


@pytest.mark.parametrize("width", (2, 3, 24, 33, 44))
def test_closed_form_latencies(width: int) -> None:
    fmt = IntFormat(width)
    idiv = IDivPrimitive(IDivOperator.build(fmt, IDivOptions()))
    assert idiv.latency == 3 + -(-width // 2), "one radix-4 step per two quotient bits, rounded up"
    for primitive in (
        IAddPrimitive(IAddOperator.build(fmt, IAddOptions())),
        ISubPrimitive(ISubOperator.build(fmt, ISubOptions())),
        IAbsPrimitive(IAbsOperator.build(fmt, IAbsOptions())),
        IShlPrimitive(IShlOperator.build(fmt, IShlOptions())),
        IShrPrimitive(IShrOperator.build(fmt, IShrOptions())),
        ICmpPrimitive(ICmpOperator.build(fmt, ICmpOptions())),
        IPopcntPrimitive(IPopcntOperator.build(fmt, IPopcntOptions())),
    ):
        assert primitive.latency == 2
        assert primitive.initiation_interval == 1


@pytest.mark.parametrize("stage_product", range(5))
def test_multiplier_staging_costs_exactly_one_cycle_each(stage_product: int) -> None:
    primitive = IMulPrimitive(IMulOperator.build(IntFormat(33), IMulOptions(stage_product=stage_product)))
    assert primitive.latency == 2 + stage_product
    assert primitive.initiation_interval == 1


def test_only_the_divider_reports_an_error_and_only_a_division_by_zero() -> None:
    # Saturation is the integer type's defined behaviour, and the saturating operators are speculatable, so none of
    # them may raise the machine's error flag; MIN // -1 saturates the divider too, and must stay off `div0`.
    fmt = IntFormat(33)
    assert IDivOperator.build(fmt, IDivOptions()).error_ports == ("div0",)
    for operator in (
        IAddOperator.build(fmt, IAddOptions()),
        ISubOperator.build(fmt, ISubOptions()),
        IMulOperator.build(fmt, IMulOptions()),
        IAbsOperator.build(fmt, IAbsOptions()),
        IShlOperator.build(fmt, IShlOptions()),
        IShrOperator.build(fmt, IShrOptions()),
        ICmpOperator.build(fmt, ICmpOptions()),
        IPopcntOperator.build(fmt, IPopcntOptions()),
    ):
        assert operator.error_ports == (), operator.name


@pytest.mark.parametrize("width", (2, 3, *PRODUCTION_WIDTHS))
def test_the_population_count_counts_the_magnitude_and_answers_on_a_minimal_port(width: int) -> None:
    # Counting the magnitude is what makes -1 answer 1 rather than filling the word, and what caps the count one
    # short of the width -- MIN counts its single bit because the negation that overflows a signed word is exactly
    # the magnitude. WY is pinned because it sizes the RTL port, and the count that just fits it is what makes the
    # width minimal rather than merely sufficient.
    fmt = IntFormat(width)
    operator = IPopcntOperator.build(fmt, IPopcntOptions())
    assert operator.params == {"W": width, "WY": (width - 1).bit_length(), "LATENCY": 2}
    count_width = operator.params["WY"]
    assert width - 1 < (1 << count_width) and width - 1 >= (1 << (count_width - 1))
    primitive = IPopcntPrimitive(operator)
    assert primitive.signature.result_types == (IntType(fmt),), "the count is an ordinary machine integer"
    corners = (0, -1, fmt.min, fmt.min + 1, fmt.max)
    assert {value: _evaluate(primitive, value)[0] for value in corners} == {
        0: 0,
        -1: 1,
        fmt.min: 1,
        fmt.min + 1: width - 1,
        fmt.max: width - 1,
    }


def test_multiplier_staging_is_part_of_the_hardware_identity() -> None:
    # The operator is the resource-sharing key: two differently staged multipliers must not pool onto one module.
    fmt = IntFormat(33)
    operators = [IMulOperator.build(fmt, IMulOptions(stage_product=stage)) for stage in range(5)]
    assert len(set(operators)) == len(operators)
    assert IMulOperator.build(fmt, IMulOptions()) == IMulOperator.build(fmt, IMulOptions(stage_product=0))


def test_the_multiplier_knob_reaches_the_built_machine() -> None:
    # It must arrive carrying the user's staging AND the machine's integer format, not the float one.
    imul = build_ops(Options(OperatorOptions(imul=IMulOptions(stage_product=3)), wint_min=44), 44).imul
    assert {port.scalar_type for port in imul.operand_ports + imul.output_ports} == {IntType(IntFormat(44))}
    assert imul.latencies[0] == 5
    assert imul.params == {"W": 44, "STAGE_PRODUCT": 3, "LATENCY": 5}
    assert build_ops(Options(OperatorOptions()), 16).imul == IMulOperator.build(
        IntFormat(16), IMulOptions(stage_product=0)
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
        assert _evaluate(complement, signed(a, width)) == [signed(~a & mask, width)]
        assert truth.evaluate(IntValue.from_bits(fmt, a)) == (a != 0,)
        for b in range(1 << width):
            operands = (signed(a, width), signed(b, width))
            assert _evaluate(conjunction, *operands) == [signed(a & b, width)]
            assert _evaluate(disjunction, *operands) == [signed(a | b, width)]
            assert _evaluate(exclusive, *operands) == [signed(a ^ b, width)]
    assert _evaluate(complement, fmt.min) == [fmt.max] and _evaluate(complement, fmt.max) == [fmt.min]

    cast = BoolToIntPrimitive(fmt)
    assert cast.evaluate(True) == (IntValue.from_int(fmt, 1),)
    assert cast.evaluate(False) == (IntValue.from_int(fmt, 0),)


@pytest.mark.parametrize("width", EXHAUSTIVE_WIDTHS)
def test_constant_shift_over_every_count_and_operand(width: int) -> None:
    fmt = IntFormat(width)
    for count in (count for count in range(1 - width, width) if count != 0):
        primitive = IntShiftConstPrimitive(fmt, count)
        for a in range(1 << width):
            want = ishl(a, fmt.encode(count), width).shft
            assert primitive.evaluate(IntValue.from_bits(fmt, a)) == (IntValue.from_bits(fmt, want),), (count, a)

    assert IntShiftConstPrimitive(fmt, 1).render("r0") == "r0<<1"
    assert IntShiftConstPrimitive(fmt, -1).render("r0") == "r0>>1"
    assert _evaluate(IntShiftConstPrimitive(fmt, 1 - width), -1) == [-1], "sign fill survives the widest right shift"
    assert _evaluate(IntShiftConstPrimitive(fmt, width - 1), fmt.min) == [0], "the sign bit shifts off the word"


def test_the_constant_shift_is_the_raw_shift_and_not_the_saturating_one() -> None:
    # The inline shift drops what leaves the word; the saturating reading needs the pooled `holoso_ishl`.
    fmt = IntFormat(33)
    assert _evaluate(IntShiftConstPrimitive(fmt, 1), fmt.max) == [-2]
    assert _evaluate(IShlPrimitive(IShlOperator.build(fmt, IShlOptions())), fmt.max, 1) == [-2, fmt.max]


@pytest.mark.parametrize("wint", (4, 17, 44))
def test_the_conversions_saturate_at_the_rails_and_round_trip_the_extremes(wint: int) -> None:
    ffmt, ifmt = FloatFormat(8, 24), IntFormat(wint)
    to_int = FToIntOperator.build(ffmt, ifmt, FToIntOptions())
    from_int = FFromIntPrimitive(FFromIntOperator.build(ffmt, ifmt, FFromIntOptions()))

    def convert(value: float, mode: RoundMode) -> int:
        (result,) = FToIntPrimitive(to_int, mode).evaluate(FloatValue.from_float(ffmt, value))
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
        (image,) = from_int.evaluate(IntValue.from_int(ifmt, extreme))
        assert isinstance(image, FloatValue)
        (back,) = FToIntPrimitive(to_int, RoundMode.NEAREST_EVEN).evaluate(image)
        assert isinstance(back, IntValue) and back.value == extreme


def test_rounding_before_converting_is_not_the_same_as_converting_with_that_mode() -> None:
    # Strength reduction's conversion of a rounding in the rounding's own mode (one ftoint(x, ROUND)) is a rewrite that
    # can change the answer, and the fastmath charter (DESIGN.md, Direction) licenses it anyway. Here 3.5 rounds to
    # +inf, which saturates, while a direct nearest-even conversion answers 4.
    ffmt, ifmt = FloatFormat(2, 4), IntFormat(33)
    fround = FRoundPrimitive(FRoundOperator.build(ffmt, FRoundOptions()), RoundMode.NEAREST_EVEN)
    ftoint = FToIntOperator.build(ffmt, ifmt, FToIntOptions())
    x = FloatValue.from_float(ffmt, 3.5)
    (rounded,) = fround.evaluate(x)
    assert isinstance(rounded, FloatValue)
    (fused,) = FToIntPrimitive(ftoint, RoundMode.NEAREST_EVEN).evaluate(x)
    (staged,) = FToIntPrimitive(ftoint, RoundMode.TRUNC).evaluate(rounded)
    assert isinstance(fused, IntValue) and isinstance(staged, IntValue)
    assert fused.value == 4 and staged.value == ifmt.max


def test_the_conversion_knobs_reach_the_built_machine() -> None:
    ops = build_ops(
        Options(
            OperatorOptions(ffromint=FFromIntOptions(stage_input=1, stage_pack=1), ftoint=FToIntOptions(stage_input=2)),
            ffmt=FloatFormat(6, 18),
            wint_min=44,
        ),
        44,
    )
    assert {ops.ffromint.latency(mode) for mode in ops.ffromint.modes} == {3}
    assert {ops.ftoint.latency(mode) for mode in ops.ftoint.modes} == {6}
    assert ops.ffromint.params == {
        "WEXP": 6,
        "WMAN": 18,
        "WINT": 44,
        "STAGE_INPUT": 1,
        "STAGE_NORMALIZE": 0,
        "STAGE_PACK": 1,
        "STAGE_OUTPUT": 0,
        "LATENCY": 3,
    }
    assert ops.ftoint.params == {"WEXP": 6, "WMAN": 18, "WINT": 44, "STAGE_INPUT": 2, "LATENCY": 6}
    assert [port.scalar_type for port in ops.ffromint.operand_ports] == [IntType(IntFormat(44))]
    assert [port.scalar_type for port in ops.ffromint.output_ports] == [FloatType(FloatFormat(6, 18))]
    assert [port.scalar_type for port in ops.ftoint.operand_ports] == [FloatType(FloatFormat(6, 18))]
    assert [port.scalar_type for port in ops.ftoint.output_ports] == [IntType(IntFormat(44))]


def _everything_configured() -> Options:
    """Every float operator present, so the catalogue walks its whole surface."""
    return Options(
        OperatorOptions(
            fadd=holoso.FAddOptions(),
            fmul=holoso.FMulOptions(),
            fdiv=holoso.FDivOptions(),
            fmul_ilog2=holoso.FMulILog2Options(),
            filog2=holoso.FILog2Options(),
            fcmp=holoso.FCmpOptions(),
            fround=holoso.FRoundOptions(),
            ffma=holoso.FFmaOptions(),
            fsort=holoso.FSortOptions(),
            fsqrt=holoso.FSqrtOptions(),
            fexp2=holoso.FExp2Options(),
            flog2=holoso.FLog2Options(),
            fcordic=holoso.FCordicOptions(),
            ffromint=holoso.FFromIntOptions(),
            ftoint=holoso.FToIntOptions(),
        ),
        ffmt=FloatFormat(6, 18),
        wint_min=33,
    )


def test_the_catalogue_builds_every_operator_for_the_machines_own_formats() -> None:
    # A conversion operator carries one format per side, so a check keyed on a single format could not see a wrong
    # `ifmt` at all. The catalogue BUILDS each operator from the machine's formats, so the mismatch is unrepresentable
    # rather than merely caught; this walks the whole catalogue and pins that.
    options = _everything_configured()
    ops = build_ops(options, options.wint_min)
    for name in (field.name for field in fields(OperatorOptions)):
        operator: HardwareOperator = getattr(ops, name)
        for port in operator.operand_ports + operator.output_ports:
            if isinstance(port.scalar_type, FloatType):
                assert port.scalar_type.fmt == ops.float_format, (name, port)
            if isinstance(port.scalar_type, IntType):
                assert port.scalar_type.fmt == ops.int_format, (name, port)


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
        for kind in vars(holoso._operators).values()
        if isinstance(kind, type) and issubclass(kind, HardwareOperator) and kind is not HardwareOperator
    }
    assert {type(getattr(ops, field.name)) for field in fields(OperatorOptions)} == kinds


def test_the_instance_cap_reaches_the_operator_but_never_the_rtl() -> None:
    # The count is a machine-level budget, so it rides operator identity (two configurations are different operators)
    # but must not become a module parameter, which would fail elaboration against the shipped cores.
    narrow, wide = IMulOperator.build(IntFormat(32), IMulOptions()), IMulOperator.build(
        IntFormat(32), IMulOptions(instances=4)
    )
    assert wide.instances == 4 and narrow != wide
    options = _everything_configured()
    ops = build_ops(options, options.wint_min)
    for name in (field.name for field in fields(OperatorOptions)):
        operator: HardwareOperator = getattr(ops, name)
        assert "INSTANCES" not in {param.upper() for param in operator.params}


def _sum_and_difference(a: int, b: int) -> tuple[int, int]:
    return a + b, a - b


def test_operators_alike_in_every_physical_field_remain_different_kinds() -> None:
    # The adder and the subtractor agree in parameters, ports and timing, so their kind alone keeps an addition from
    # time-sharing the subtractor's module.
    result = holoso.synthesize(_sum_and_difference, Options(OperatorOptions(), wint_min=16), name="SumAndDifference")
    verilog = result.verilog_output.verilog
    assert verilog.count("holoso_iadds #") == 1 and verilog.count("holoso_isubs #") == 1
    sim = result.numerical_model.elaborate()
    for a, b in ((3, 5), (-7, 2), (100, -100)):
        assert [int(value) for value in sim.run(a, b) if isinstance(value, holoso.IntValue)] == [a + b, a - b]


def _add(a: float, b: float) -> float:
    return a + b


def test_a_float_only_build_configures_an_integer_operator_without_instantiating_it() -> None:
    options = Options(OperatorOptions(fadd=holoso.FAddOptions()), ffmt=FloatFormat(6, 18), wint_min=44)
    ops = build_ops(options, options.wint_min)
    assert {port.scalar_type for port in ops.imul.operand_ports + ops.imul.output_ports} == {IntType(IntFormat(44))}
    for conversion in ("ffromint", "ftoint"):  # a conversion is optional, as every float operator is
        with pytest.raises(UnsupportedConstruct, match="not configured"):
            getattr(ops, conversion)
    verilog = holoso.synthesize(_add, options, name="ImulUnused").verilog_output.verilog
    assert "holoso_imuls" not in verilog, "an available operator no kernel reaches costs no fabric"
    assert "holoso_ffromint" not in verilog and "holoso_ftoint" not in verilog
