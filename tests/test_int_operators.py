"""
The pooled integer operators: their reference semantics, their closed-form timing, and the one knob among them.
No lowering selects these yet, so they are driven directly.

The sweeps score ``evaluate`` against the very oracle the HDL benches score the RTL against, which is what makes the
"matches the RTL bit-for-bit" claim a checked one rather than a stated one. The latencies are pinned against the
hardware separately, by the elaboration probe in ``test_backend.py`` and by the benches.
"""

from collections.abc import Callable

import pytest

import holoso
from holoso import FloatFormat, IMulOptions, IntFormat, OperatorOptions, Options
from holoso._operators import (
    IAbsOperator,
    IAddOperator,
    ICmpOperator,
    IDivOperator,
    IMulOperator,
    IShiftOperator,
    ISubOperator,
    IntHardwareOperator,
    Relation,
)
from holoso._value import IntValue

from ._modelref import build_ops
from .hdl.hdl_integer_oracle import expected_idivs, expected_imuls, expected_simple

EXHAUSTIVE_WIDTHS = (2, 3, 4, 5, 6)
PRODUCTION_WIDTHS = (24, 33, 44)


def _corners(fmt: IntFormat) -> list[int]:
    return [fmt.min, fmt.min + 1, -3, -2, -1, 0, 1, 2, 3, fmt.max - 1, fmt.max]


def _evaluate(operator: IntHardwareOperator, *operands: int) -> list[int | bool]:
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
    binary = [IAddOperator(fmt), ISubOperator(fmt), ICmpOperator(fmt), IShiftOperator(fmt)]
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

    shift = IShiftOperator(fmt)
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
    for operator in (IAddOperator(fmt), ISubOperator(fmt), IAbsOperator(fmt), IShiftOperator(fmt), ICmpOperator(fmt)):
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
        IShiftOperator(fmt),
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
    imul = build_ops(Options(OperatorOptions(imul=IMulOptions(stage_product=3)), ifmt=IntFormat(17))).imul
    assert imul.fmt == IntFormat(17)
    assert imul.latency == 5
    assert imul.params == {"W": 17, "STAGE_PRODUCT": 3, "LATENCY": 5}
    assert build_ops(Options(OperatorOptions())).imul.opt == IMulOptions(stage_product=0)


def _add(a: float, b: float) -> float:
    return a + b


def test_a_float_only_build_configures_an_integer_operator_without_instantiating_it() -> None:
    options = Options(OperatorOptions(fadd=holoso.FAddOptions()), ffmt=FloatFormat(6, 18), ifmt=IntFormat(44))
    assert build_ops(options).imul.fmt == IntFormat(44)
    verilog = holoso.synthesize(_add, options, name="ImulUnused").verilog_output.verilog
    assert "holoso_imuls" not in verilog, "an available operator no kernel reaches costs no fabric"
