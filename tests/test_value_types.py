"""Unit tests for the public value/format types and the shared numeric test helpers."""

import numpy as np
import pytest

from holoso import FloatFormat, FloatValue
from ._modelref import default_tolerance, evaluate_reference, random_legal_bits, spd_matrix, unit_roundoff, within

F32 = FloatFormat(8, 24)
FMT = FloatFormat(6, 18)


def test_codec_known_binary32_values() -> None:
    assert F32.encode(1.0) == 0x3F800000
    assert F32.encode(2.0) == 0x40000000
    assert F32.encode(0.5) == 0x3F000000
    assert F32.encode(-1.0) == 0xBF800000
    assert F32.encode(0.0) == 0
    assert F32.decode(0x3F800000) == 1.0
    assert F32.decode(0) == 0.0


def test_codec_round_trip_within_unit_roundoff() -> None:
    rng = np.random.default_rng(1)
    for fmt in (F32, FMT):
        u = unit_roundoff(fmt)
        for _ in range(500):
            x = float(rng.uniform(-100.0, 100.0))
            y = fmt.decode(fmt.encode(x))
            assert abs(y - x) <= u * abs(x) + 1e-30


def test_codec_exact_powers_and_simple_fractions() -> None:
    for value in (3.0, 0.25, -7.5, 16.0, 0.125):
        assert FMT.decode(FMT.encode(value)) == value


def test_float_value_factories_and_fields() -> None:
    value = FloatValue.from_bits(F32, 0x3F800001)
    assert value.fmt == F32
    assert value.bits == 0x3F800001
    assert value.negative is False
    assert value.exponent == 0x7F
    assert float(FloatValue.from_float(F32, 1.0)) == 1.0

    with pytest.raises(TypeError, match="from_bits"):
        FloatValue(F32, 1.0)
    with pytest.raises(TypeError, match="float"):
        FloatValue.from_float(F32, 1)
    with pytest.raises(TypeError, match="int"):
        FloatValue.from_bits(F32, True)
    with pytest.raises(ValueError, match="fit"):
        FloatValue.from_bits(F32, 1 << F32.width)


def test_is_legal_rejects_subnormal_and_negative_zero() -> None:
    # exp == 0 with nonzero fraction is subnormal; sign bit with zero magnitude is negative zero.
    assert not FMT.is_legal(0b1)
    neg_zero = 1 << (FMT.width - 1)
    assert not FMT.is_legal(neg_zero)
    assert FMT.is_legal(FMT.encode(1.0))


def test_compare_float_values_exact_for_wide_formats() -> None:
    # The model's comparison must be exact, not via a lossy float64 decode: two values differing only in the lowest
    # mantissa bit of a >53-bit mantissa must compare unequal (decode would collapse them).
    fmt = FloatFormat(wexp=8, wman=60)
    bias = (1 << (fmt.wexp - 1)) - 1
    one = FloatValue.from_bits(fmt, bias << fmt.wman)
    one_plus_ulp = FloatValue.from_bits(fmt, (bias << fmt.wman) | 1)
    assert fmt.decode(one.bits) == fmt.decode(one_plus_ulp.bits)  # lossy: float64 cannot tell them apart
    assert one.compare(one_plus_ulp) == -1
    assert one_plus_ulp.compare(one) == 1
    assert one.compare(one) == 0
    # Signs, zero, and infinities form a total order (ZKF has no NaN).
    neg_one = one.apply_sign(negate=True, absolute=False)
    zero = FloatValue.from_float(fmt, 0.0)
    pos_inf = FloatValue.from_float(fmt, float("inf"))
    neg_inf = FloatValue.from_float(fmt, float("-inf"))
    ascending = [neg_inf, neg_one, zero, one, pos_inf]
    for lower_value, higher_value in zip(ascending, ascending[1:]):
        assert lower_value.compare(higher_value) == -1
        assert higher_value.compare(lower_value) == 1


def test_reference_evaluates_and_flattens() -> None:
    def f(a: float, b: float) -> list[float]:
        return [a + b, a * b]

    assert evaluate_reference(f, {"a": 2.0, "b": 3.0}) == [5.0, 6.0]


def test_tolerance_predicate() -> None:
    assert within(1.0, 1.0, 0.0, 0.0)
    assert within(1.001, 1.0, 0.01, 0.0)
    assert not within(1.1, 1.0, 0.01, 0.0)
    assert within(float("inf"), float("inf"), 1.0, 1.0)
    assert not within(float("inf"), 1.0, 1.0, 1.0)


def test_default_tolerance_scales_with_format_and_size() -> None:
    coarse = default_tolerance(FMT, 100)[0]
    fine = default_tolerance(F32, 100)[0]
    assert coarse > fine  # 6/18 has a larger unit roundoff than 8/24
    assert default_tolerance(FMT, 200)[0] > default_tolerance(FMT, 10)[0]


def test_sampling_legal_and_spd() -> None:
    rng = np.random.default_rng(7)
    for _ in range(200):
        bits = random_legal_bits(FMT, rng)
        assert FMT.is_legal(bits) and FMT.is_finite(bits)
    cov = spd_matrix(rng, 3)
    assert np.all(np.linalg.eigvalsh(cov) > 0.0)
