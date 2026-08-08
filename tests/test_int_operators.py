"""
The integer operators, pooled and inline: their reference semantics, their closed-form timing, and the one knob
among them. No lowering selects these yet, so they are driven directly.

The sweeps score ``evaluate`` against the very oracle the HDL benches score the RTL against, so the values are
checked rather than merely claimed. What they do NOT check is the configuration the hardware is built in: the
latencies, the RTL parameters and the port names are pinned elsewhere -- by the elaboration probe in
``test_backend.py`` and by the benches, which take all three from the operator itself. A wrong ``QUOTIENT_FLOOR``
would leave every assertion here passing and fail there, as would a wide-bank expression that lost its sign
extension (``tests/hdl/test_int_inline.py``).
"""

from collections.abc import Callable
from dataclasses import replace

import pytest

import holoso
from holoso import (
    FFromIntOptions,
    FRoundOptions,
    FToIntOptions,
    FloatFormat,
    FloatType,
    FloatValue,
    IMulOptions,
    IntFormat,
    OperatorOptions,
    Options,
)
from holoso._eel import lower
from holoso._hir import optimize
from holoso._mir import lower as lower_to_mir
from holoso._operators import (
    BoolToIntOperator,
    FFromIntOperator,
    FRoundOperator,
    FToIntOperator,
    IAbsOperator,
    IAddOperator,
    ICmpOperator,
    IDivOperator,
    IMulOperator,
    IShlOperator,
    ISubOperator,
    IntBwAndOperator,
    IntBwNotOperator,
    IntBwOrOperator,
    IntBwXorOperator,
    IntHardwareOperator,
    IntInlineOperator,
    IntShiftConstOperator,
    IntToBoolOperator,
    Relation,
    RoundMode,
)
from holoso._type import IntType
from holoso._value import IntValue

from ._modelref import DEFAULT_IFCONV_MAX_OPS, build_ops
from .hdl.hdl_integer_oracle import expected_idivs, expected_imuls, expected_simple, ishl, signed

EXHAUSTIVE_WIDTHS = (2, 3, 4, 5, 6)
PRODUCTION_WIDTHS = (24, 33, 44)


def _corners(fmt: IntFormat) -> list[int]:
    return [fmt.min, fmt.min + 1, -3, -2, -1, 0, 1, 2, 3, fmt.max - 1, fmt.max]


def _evaluate(operator: IntHardwareOperator | IntInlineOperator, *operands: int) -> list[int | bool]:
    fmt = operator.fmt
    values: list[int | bool] = []
    for result in operator.evaluate(*(IntValue.from_int(fmt, operand) for operand in operands)):
        assert isinstance(result, IntValue | bool)
        values.append(result if isinstance(result, bool) else result.value)
    return values


def _bits(operator: IntHardwareOperator, *operand_bits: int) -> dict[str, int]:
    """Keyed by the RTL port names the module drives, so an oracle dict compares directly."""
    fmt = operator.fmt
    results = operator.evaluate(*(IntValue.from_bits(fmt, bits) for bits in operand_bits))
    return {
        port: int(result) if isinstance(result, bool) else result.bits
        for port, result in zip(operator.output_hdl_ports, results, strict=True)
    }


def _oracle(expected: dict[str, int], operator: IntHardwareOperator) -> dict[str, int]:
    """The value ports alone: the saturation sidebands are deliberately not modeled."""
    return {port: expected[port] for port in operator.output_hdl_ports}


@pytest.mark.parametrize("width", EXHAUSTIVE_WIDTHS)
def test_every_operator_answers_as_the_rtl_does_over_every_operand_pair(width: int) -> None:
    fmt = IntFormat(width)
    binary = [IAddOperator(fmt), ISubOperator(fmt), ICmpOperator(fmt), IShlOperator(fmt)]
    idiv, iabs = IDivOperator(fmt), IAbsOperator(fmt)
    # Staging is a timing knob, so every multiplier configuration must answer the one product.
    multipliers = [IMulOperator(fmt, IMulOptions(stage_product=stage)) for stage in range(5)]
    for a in range(1 << width):
        want = expected_simple(iabs.module_name, a, 0, width)
        assert _bits(iabs, a) == _oracle(want, iabs)
        for b in range(1 << width):
            for operator in binary:
                want = expected_simple(operator.module_name, a, b, width)
                assert _bits(operator, a, b) == _oracle(want, operator), (operator.mnemonic, a, b)
            for imul in multipliers:
                assert _bits(imul, a, b) == _oracle(expected_imuls(a, b, width), imul), (imul.params, a, b)
            assert _bits(idiv, a, b) == _oracle(expected_idivs(a, b, width, True), idiv), (a, b)


@pytest.mark.parametrize("width", EXHAUSTIVE_WIDTHS)
def test_floor_division_obeys_the_division_identity(width: int) -> None:
    # What the oracle comparison cannot show: that the answers are a division at all, not a shared misreading.
    fmt = IntFormat(width)
    operator = IDivOperator(fmt)
    for num in range(fmt.min, fmt.max + 1):
        for den in range(fmt.min, fmt.max + 1):
            quotient, remainder = _evaluate(operator, num, den)
            if den == 0 or (num == fmt.min and den == -1):
                continue  # no quotient exists, or none the width holds; the oracle pins what is answered instead
            assert num == den * quotient + remainder
            assert abs(remainder) < abs(den)
            assert not remainder or (remainder < 0) == (den < 0), "the remainder follows the divisor, as floor does"


@pytest.mark.parametrize("width", EXHAUSTIVE_WIDTHS)
def test_comparator_flags_are_one_hot_and_serve_every_relation(width: int) -> None:
    fmt = IntFormat(width)
    operator = ICmpOperator(fmt)
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
            flags = _evaluate(operator, a, b)
            assert sum(flags) == 1, "the order flags are one-hot"
            for relation, answer in answers.items():
                port, inversion = operator.tap_of(relation)
                assert inversion.apply(bool(flags[port])) == answer(a, b), relation


@pytest.mark.parametrize("width", PRODUCTION_WIDTHS)
def test_edge_cases_at_the_production_widths(width: int) -> None:
    # The sweeps stop far below these, and saturation is where a width-dependent slip would hide.
    fmt = IntFormat(width)
    assert _evaluate(IAbsOperator(fmt), fmt.min) == [fmt.max]
    assert _evaluate(IAddOperator(fmt), fmt.min, fmt.min) == [fmt.min]
    assert _evaluate(IAddOperator(fmt), fmt.max, fmt.max) == [fmt.max]
    assert _evaluate(ISubOperator(fmt), fmt.min, fmt.max) == [fmt.min]
    assert _evaluate(ISubOperator(fmt), 0, fmt.min) == [fmt.max], "negation via 0-x saturates instead of wrapping"
    assert _evaluate(IMulOperator(fmt, IMulOptions()), fmt.min, fmt.min) == [fmt.max]
    assert _evaluate(IMulOperator(fmt, IMulOptions()), fmt.min, 1) == [fmt.min]
    assert _evaluate(IDivOperator(fmt), fmt.min, -1) == [fmt.max, 0]
    assert _evaluate(IDivOperator(fmt), -7, 2) == [-4, 1], "the quotient floors, as Python's // does"

    for numerator in _corners(fmt):
        assert _evaluate(IDivOperator(fmt), numerator, 0) == [fmt.min if numerator < 0 else fmt.max, numerator]

    shift = IShlOperator(fmt)
    for count in (0, 1, width - 1, width, width + 1, fmt.max):
        assert _evaluate(shift, 0, count) == [0, 0]
        assert _evaluate(shift, -1, -count) == [-1, -1], "sign fill makes -1 a fixed point of every right shift"
        assert _evaluate(shift, 1, count)[1] == (1 << count if count < width - 1 else fmt.max)
    assert _evaluate(shift, fmt.min, fmt.min) == [-1, -1], "a count past the word saturates to the word itself"
    assert _evaluate(shift, fmt.max, 1) == [-2, fmt.max], "the raw shift drops the bit the saturating one clamps on"


@pytest.mark.parametrize("width", (2, 3, 24, 33, 44))
def test_closed_form_latencies(width: int) -> None:
    fmt = IntFormat(width)
    assert IDivOperator(fmt).latency == 3 + -(-width // 2), "one radix-4 step per two quotient bits, rounded up"
    for operator in (IAddOperator(fmt), ISubOperator(fmt), IAbsOperator(fmt), IShlOperator(fmt), ICmpOperator(fmt)):
        assert operator.latency == 2
        assert operator.initiation_interval == 1


@pytest.mark.parametrize("stage_product", range(5))
def test_multiplier_staging_costs_exactly_one_cycle_each(stage_product: int) -> None:
    operator = IMulOperator(IntFormat(33), IMulOptions(stage_product=stage_product))
    assert operator.latency == 2 + stage_product
    assert operator.initiation_interval == 1


def test_only_the_divider_reports_an_error_and_only_a_division_by_zero() -> None:
    # Saturation is the integer type's defined behaviour, and the saturating operators are speculatable, so none of
    # them may raise the machine's error flag; MIN // -1 saturates the divider too, and must stay off ``div0``.
    fmt = IntFormat(33)
    assert IDivOperator(fmt).error_ports == ["div0"]
    for operator in (
        IAddOperator(fmt),
        ISubOperator(fmt),
        IMulOperator(fmt, IMulOptions()),
        IAbsOperator(fmt),
        IShlOperator(fmt),
        ICmpOperator(fmt),
    ):
        assert operator.error_ports == [], operator.mnemonic


def test_multiplier_staging_is_part_of_the_hardware_identity() -> None:
    # The operator is the resource-sharing key: two differently staged multipliers must not pool onto one module.
    fmt = IntFormat(33)
    instances = [IMulOperator(fmt, IMulOptions(stage_product=stage)) for stage in range(5)]
    assert len({operator.instance_stem for operator in instances}) == len(instances)
    assert len(set(instances)) == len(instances)
    assert IMulOperator(fmt, IMulOptions()) == IMulOperator(fmt, IMulOptions(stage_product=0))


def test_the_multiplier_knob_reaches_the_built_machine() -> None:
    # It must arrive carrying the user's staging AND the machine's integer format, not the float one.
    imul = build_ops(Options(OperatorOptions(imul=IMulOptions(stage_product=3)), wint_min=44)).imul
    assert imul.fmt == IntFormat(44)
    assert imul.latency == 5
    assert imul.params == {"W": 44, "STAGE_PRODUCT": 3, "LATENCY": 5}
    assert build_ops(Options(OperatorOptions())).imul.opt == IMulOptions(stage_product=0)


@pytest.mark.parametrize("width", EXHAUSTIVE_WIDTHS)
def test_inline_bitwise_and_casts_answer_over_every_operand(width: int) -> None:
    # The reference works on the raw bit patterns, so it knows nothing of the operator's own sign convention. A
    # bitwise combination never leaves the range, so a saturating implementation would answer the rail for ``~min``.
    fmt = IntFormat(width)
    mask = (1 << width) - 1
    conjunction, disjunction, exclusive = IntBwAndOperator(fmt), IntBwOrOperator(fmt), IntBwXorOperator(fmt)
    complement, truth = IntBwNotOperator(fmt), IntToBoolOperator(fmt)
    for a in range(1 << width):
        assert _evaluate(complement, signed(a, width)) == [signed(~a & mask, width)]
        assert truth.evaluate(IntValue.from_bits(fmt, a)) == (a != 0,)
        for b in range(1 << width):
            operands = (signed(a, width), signed(b, width))
            assert _evaluate(conjunction, *operands) == [signed(a & b, width)]
            assert _evaluate(disjunction, *operands) == [signed(a | b, width)]
            assert _evaluate(exclusive, *operands) == [signed(a ^ b, width)]
    assert _evaluate(complement, fmt.min) == [fmt.max] and _evaluate(complement, fmt.max) == [fmt.min]

    cast = BoolToIntOperator(fmt)
    assert cast.evaluate(True) == (IntValue.from_int(fmt, 1),)
    assert cast.evaluate(False) == (IntValue.from_int(fmt, 0),)


@pytest.mark.parametrize("width", EXHAUSTIVE_WIDTHS)
def test_constant_shift_over_every_count_and_operand(width: int) -> None:
    fmt = IntFormat(width)
    for count in (count for count in range(1 - width, width) if count != 0):
        operator = IntShiftConstOperator(fmt, count)
        for a in range(1 << width):
            want = ishl(a, fmt.encode(count), width).shft
            assert operator.evaluate(IntValue.from_bits(fmt, a)) == (IntValue.from_bits(fmt, want),), (count, a)

    assert IntShiftConstOperator(fmt, 1).render("r0") == "r0<<1"
    assert IntShiftConstOperator(fmt, -1).render("r0") == "r0>>1"
    assert _evaluate(IntShiftConstOperator(fmt, 1 - width), -1) == [-1], "sign fill survives the widest right shift"
    assert _evaluate(IntShiftConstOperator(fmt, width - 1), fmt.min) == [0], "the sign bit shifts off the word"


def test_the_constant_shift_serves_only_the_counts_that_are_shifts() -> None:
    # Zero is the identity, which HIR or MIR folds; a count reaching the word answers a constant or a sign fill, and
    # clamping a width-less HIR count down to the word is the lowering's job rather than the operator's.
    fmt = IntFormat(8)
    for count in (0, fmt.width, -fmt.width, fmt.max, fmt.min):
        with pytest.raises(ValueError):
            IntShiftConstOperator(fmt, count)


def test_the_constant_shift_is_the_raw_shift_and_not_the_saturating_one() -> None:
    # The inline shift drops what leaves the word; the saturating reading needs the pooled ``holoso_ishl``.
    fmt = IntFormat(33)
    assert _evaluate(IntShiftConstOperator(fmt, 1), fmt.max) == [-2]
    assert _evaluate(IShlOperator(fmt), fmt.max, 1) == [-2, fmt.max]


@pytest.mark.parametrize("wint", (4, 17, 44))
def test_the_conversions_saturate_at_the_rails_and_round_trip_the_extremes(wint: int) -> None:
    ffmt, ifmt = FloatFormat(8, 24), IntFormat(wint)
    to_int = FToIntOperator(ffmt, ifmt, FToIntOptions())
    from_int = FFromIntOperator(ffmt, ifmt, FFromIntOptions())

    def convert(value: float, mode: RoundMode) -> int:
        (result,) = to_int.evaluate(FloatValue.from_float(ffmt, value), immediates=(int(mode),))
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
        (back,) = to_int.evaluate(image, immediates=(int(RoundMode.NEAREST_EVEN),))
        assert isinstance(back, IntValue) and back.value == extreme


def test_rounding_before_converting_is_not_the_same_as_converting_with_that_mode() -> None:
    # The MIR fusion of FloatToInt(FloatRound(x)) into one ftoint(x, ROUND) is therefore a rewrite that can change
    # the answer, and the fastmath charter licenses it anyway (see TODO.md). Here 3.5 rounds to +inf, which
    # saturates, while a direct nearest-even conversion answers 4.
    ffmt, ifmt = FloatFormat(2, 4), IntFormat(33)
    fround = FRoundOperator(ffmt, FRoundOptions())
    ftoint = FToIntOperator(ffmt, ifmt, FToIntOptions())
    x = FloatValue.from_float(ffmt, 3.5)
    (rounded,) = fround.evaluate(x, immediates=(int(RoundMode.NEAREST_EVEN),))
    assert isinstance(rounded, FloatValue)
    (fused,) = ftoint.evaluate(x, immediates=(int(RoundMode.NEAREST_EVEN),))
    (staged,) = ftoint.evaluate(rounded, immediates=(int(RoundMode.TRUNC),))
    assert isinstance(fused, IntValue) and isinstance(staged, IntValue)
    assert fused.value == 4 and staged.value == ifmt.max


def test_the_conversion_knobs_reach_the_built_machine() -> None:
    ops = build_ops(
        Options(
            OperatorOptions(ffromint=FFromIntOptions(stage_input=1, stage_pack=1), ftoint=FToIntOptions(stage_input=2)),
            ffmt=FloatFormat(6, 18),
            wint_min=44,
        )
    )
    assert ops.ffromint is not None and ops.ftoint is not None
    assert ops.ffromint.latency == 3 and ops.ftoint.latency == 6
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
    assert ops.ffromint.signature.operand_types == (IntType(IntFormat(44)),)
    assert ops.ffromint.signature.result_types == (FloatType(FloatFormat(6, 18)),)
    assert ops.ftoint.signature.operand_types == (FloatType(FloatFormat(6, 18)),)
    assert ops.ftoint.signature.result_types == (IntType(IntFormat(44)),)


def test_the_lowering_checks_every_port_format_and_not_just_the_operator_kind() -> None:
    # A conversion operator carries one format per side, so the check that keyed on the operator's own ``fmt`` could
    # not see a wrong ``ifmt`` at all -- it read the float side and agreed with itself.
    options = Options(OperatorOptions(fadd=holoso.FAddOptions()), ffmt=FloatFormat(6, 18), wint_min=33)
    ops = build_ops(options)
    mismatched = replace(ops, ftoint=FToIntOperator(options.ffmt, IntFormat(17), FToIntOptions()))
    hir = optimize(lower(_add).hir, DEFAULT_IFCONV_MAX_OPS)
    lower_to_mir(hir, ops, options.ffmt, options.ifmt)  # the premise: the matching configuration lowers cleanly
    with pytest.raises(AssertionError, match="ftoint"):
        lower_to_mir(hir, mismatched, options.ffmt, options.ifmt)


def _add(a: float, b: float) -> float:
    return a + b


def test_a_float_only_build_configures_an_integer_operator_without_instantiating_it() -> None:
    options = Options(OperatorOptions(fadd=holoso.FAddOptions()), ffmt=FloatFormat(6, 18), wint_min=44)
    ops = build_ops(options)
    assert ops.imul.fmt == IntFormat(44)
    assert ops.ffromint is None and ops.ftoint is None, "a conversion is optional, as every float operator is"
    verilog = holoso.synthesize(_add, options, name="ImulUnused").verilog_output.verilog
    assert "holoso_imuls" not in verilog, "an available operator no kernel reaches costs no fabric"
    assert "holoso_ffromint" not in verilog and "holoso_ftoint" not in verilog
