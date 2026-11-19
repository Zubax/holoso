"""Unit tests for the shared numeric test helpers, and the exponent convention Holoso builds on."""

import math

import numpy as np
import zkf

from holoso import FloatFormat, FloatValue
from ._modelref import default_tolerance, evaluate_reference, random_legal_bits, spd_matrix, within

F32 = FloatFormat(8, 24)
FMT = FloatFormat(6, 18)


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
        value = FloatValue(FMT, random_legal_bits(FMT, rng))
        assert value.is_finite and value.canonicalize() == value
    cov = spd_matrix(rng, 3)
    assert np.all(np.linalg.eigvalsh(cov) > 0.0)


def test_ilog2_is_the_unbiased_exponent() -> None:
    # The limit answers are what let an extremum over the exponent need no special case: zero must fall below every
    # finite exponent and an infinity above every one, and neither may depend on the sign or the fraction.
    for fmt in (F32, FMT):
        ilog2 = zkf.Ilog2(fmt=fmt, wint=fmt.wfull)
        bias = fmt.bias
        assert ilog2.evaluate(fmt.encode(0.0)).y == -bias
        assert ilog2.evaluate(fmt.encode(math.inf)).y == bias + 1
        assert ilog2.evaluate(fmt.encode(-math.inf)).y == bias + 1
        for value in (1.0, 1.5, 2.0, 3.0, 0.5, 0.75, 1024.0, 2.0**-20):
            expected = math.floor(math.log2(value))
            assert ilog2.evaluate(fmt.encode(value)).y == expected
            assert ilog2.evaluate(fmt.encode(-value)).y == expected
            assert -bias < expected < bias + 1
