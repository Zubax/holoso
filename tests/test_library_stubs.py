"""
Plain-Python numerical verification of the library stubs: each composite stub is executed directly (no compiler
involved) and compared against the math/numpy function it substitutes. This checks the ALGORITHM (the identity the
stub encodes); the lowering of the same stubs is checked end-to-end in test_extended_operators / test_matrix /
test_cosim.
"""

import math

import numpy as np
import pytest

from holoso._frontend._lib import Intrinsic, Library, resolve
from holoso._frontend._lib._linalg import _transpose, matmul_, outer_, trace_
from holoso._frontend._lib._intrinsics import (
    abs_,
    atan2_,
    ceil_,
    cos_,
    exp2_,
    floor_,
    fma_,
    hypot_,
    isfinite_,
    isinf_,
    isneginf_,
    isposinf_,
    log2_,
    max_,
    min_,
    round_,
    sin_,
    sqrt_,
    trunc_,
)
from holoso._frontend._lib._numpy import (
    acos_,
    acosh_,
    asin_,
    asinh_,
    atan_,
    atanh_,
    cbrt_,
    cosh_,
    degrees_,
    exp_,
    expm1_,
    log10_,
    log1p_,
    log_,
    pow_,
    radians_,
    sign_,
    sinh_,
    tan_,
    tanh_,
)

_INF = float("inf")


def test_registry_resolves_the_expected_externals() -> None:
    intrinsic_externals: list[object] = [math.sqrt, np.sqrt, math.sin, np.cos, math.atan2, np.arctan2]
    intrinsic_externals += [abs, min, max, round, math.fma, np.fmin, np.fmax, np.fix, np.rint, np.around]
    for external in intrinsic_externals:
        assert isinstance(resolve(external), Intrinsic), external
    library_externals: list[object] = [math.cbrt, np.cbrt, math.tan, np.sign, math.exp, np.log10]
    library_externals += [pow, math.pow, np.power, np.float_power]
    library_externals += [math.sinh, np.cosh, math.tanh, math.asinh, np.arcsinh, math.acosh, math.atanh]
    library_externals += [math.expm1, np.log1p, math.degrees, np.rad2deg, math.radians, np.deg2rad]
    library_externals += [np.matmul, np.dot, np.trace, np.outer]  # transpose is structural, not a stub
    for external in library_externals:
        assert isinstance(resolve(external), Library), external
    # ``@`` resolves the matmul stub; ``.T``/np.transpose lower structurally as a leaf permutation.
    assert resolve(np.matmul) == Library(matmul_) and resolve(np.transpose) is None  # type: ignore[arg-type]
    assert resolve(np.dot) == resolve(np.matmul)  # identical on the supported 1-D/2-D non-scalar domain
    # An unregistered callable resolves to nothing; an unhashable shadow does not crash the lookup.
    assert resolve(math.erf) is None and resolve(np.zeros(3)) is None
    assert resolve(np.linalg.inv) is None and resolve(np.inner) is None  # deliberately not implemented yet


def test_intrinsic_stubs_match_their_references() -> None:
    for x in (0.0, -0.0, 0.75, -2.5, 3.0, 100.0, -1e-30):
        assert exp2_(x) == math.exp2(x)
        assert sin_(x) == math.sin(x)
        assert cos_(x) == math.cos(x)
        assert floor_(x) == np.floor(x) and ceil_(x) == np.ceil(x) and trunc_(x) == np.trunc(x)
        assert abs_(x) == abs(x)
        assert round_(x) == np.round(x)
        assert isfinite_(x) and not isinf_(x) and not isposinf_(x) and not isneginf_(x)
    for x in (0.25, 1.0, 4.0, 1e30):
        assert log2_(x) == math.log2(x)
        assert sqrt_(x) == math.sqrt(x)
    assert atan2_(3.0, -4.0) == math.atan2(3.0, -4.0)
    assert hypot_(3.0, 4.0) == 5.0
    assert min_(1.5, -2.0) == -2.0 and max_(1.5, -2.0) == 1.5
    assert fma_(3.0, 4.0, 5.0) == math.fma(3.0, 4.0, 5.0)
    assert floor_(_INF) == _INF and ceil_(-_INF) == -_INF
    assert round_(2.5) == 2.0 and round_(3.5) == 4.0 and round_(-_INF) == -_INF
    assert isposinf_(_INF) and not isneginf_(_INF) and isinf_(-_INF) and not isfinite_(_INF)
    assert exp2_(1e30) == _INF  # saturates like the hardware instead of raising like math.exp2


def test_sign() -> None:
    for x in (1e-300, 0.5, 7.0, _INF):
        assert sign_(x) == 1.0 and sign_(-x) == -1.0
    assert sign_(0.0) == 0.0 and sign_(-0.0) == 0.0
    assert math.isnan(sign_(math.nan))  # r = x in the zero branch reproduces np.sign(nan) = nan exactly


def test_cbrt() -> None:
    for x in (8.0, -27.0, 0.5, -1e-6, 1e18, 3.7):
        assert cbrt_(x) == pytest.approx(math.cbrt(x), rel=1e-12), x
    assert cbrt_(0.0) == 0.0 and cbrt_(-0.0) == 0.0


def test_tan() -> None:
    for x in (0.0, 0.3, -1.2, 2.0, 100.0, math.pi / 2):  # pi/2 is not exact in binary64, so tan() is finite there
        assert tan_(x) == pytest.approx(math.tan(x), rel=1e-12), x


def test_atan() -> None:
    for x in (0.0, 1.0, -1.0, 0.001, -1e6, _INF):
        assert atan_(x) == pytest.approx(math.atan(x), rel=1e-12), x


def test_asin_acos() -> None:
    for x in (0.0, 0.5, -0.5, 0.9, -0.999, 1.0, -1.0):
        assert asin_(x) == pytest.approx(math.asin(x), rel=1e-7, abs=1e-9), x
        assert acos_(x) == pytest.approx(math.acos(x), rel=1e-7, abs=1e-9), x
    with pytest.raises(ValueError):
        asin_(1.5)  # domain violation raises in plain Python (math.sqrt of a negative), like math.asin


def test_exp_log() -> None:
    for x in (0.0, 1.0, -1.0, 10.0, -30.0, 0.001):
        assert exp_(x) == pytest.approx(math.exp(x), rel=1e-12), x
    for x in (0.001, 0.5, 1.0, math.e, 100.0, 1e30):
        assert log_(x) == pytest.approx(math.log(x), rel=1e-12, abs=1e-15), x
        assert log10_(x) == pytest.approx(math.log10(x), rel=1e-12, abs=1e-15), x


def test_pow_rungs_are_exact_including_negative_bases() -> None:
    for b in (2.0, -2.0, 0.5, -1.5, 3.0):
        for e in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0):
            assert pow_(b, e) == math.pow(b, e), (b, e)
    assert pow_(0.0, 0.0) == 1.0
    assert pow_(-2.0, 3.0) == -8.0


def test_pow_general_path() -> None:
    for b, e in ((2.0, 0.5), (3.0, 2.5), (10.0, -1.5), (0.5, 8.0), (1.0, 123.456)):
        assert pow_(b, e) == pytest.approx(math.pow(b, e), rel=1e-12), (b, e)
    with pytest.raises(ValueError):
        pow_(-2.0, 6.0)  # off the rungs, the a>0 identity raises in plain Python where math.pow is 64.0


def test_pow_zero_base() -> None:
    # A zero base short-circuits to r = b, so the stub no longer routes through log2(0) (which raised here before) and
    # matches math.pow for a non-negative exponent: 0**0 == 1, 0**positive == 0. Exponents off the 0..5 rungs (0.5,
    # 7.0, ...) are the ones that used to reach the general path and fail.
    for e in (0.0, 0.5, 1.0, 2.0, 5.0, 7.0, 123.4):
        assert pow_(0.0, e) == math.pow(0.0, e), e


def test_pow_unit_base() -> None:
    # A unit base short-circuits to 1.0, so pow(1, e) == 1 for every e -- including a non-finite one, where the general
    # path's exp2(e * log2(1)) = exp2(e * 0) would otherwise yield nan (inf * 0). Matches IEEE 754 / math.pow.
    for e in (0.0, 0.5, 2.0, 7.0, -3.0, _INF, -_INF, math.nan):
        assert pow_(1.0, e) == 1.0, e


def test_hyperbolic() -> None:
    for x in (-4.0, -1.0, -0.1, 0.0, 0.1, 1.0, 4.0):
        assert sinh_(x) == pytest.approx(math.sinh(x), rel=1e-12, abs=1e-15), x
        assert cosh_(x) == pytest.approx(math.cosh(x), rel=1e-12), x
    for x in (-30.0, -2.0, 0.0, 2.0, 30.0):  # the stable sigmoid form holds tanh in [-1,1] without exp overflow
        assert tanh_(x) == pytest.approx(math.tanh(x), rel=1e-12, abs=1e-15), x


def test_inverse_hyperbolic() -> None:
    # 1e200/1e300 exceed float64's own x*x overflow (~1.3e154), exercising the large-|x| branch that returns ln(2|x|).
    for x in (-1e6, -2.0, 0.0, 2.0, 1e6, 1e200, -1e200, 1e300):  # the sign/abs form also keeps large-negative asinh
        assert asinh_(x) == pytest.approx(math.asinh(x), rel=1e-12, abs=1e-15), x
    for x in (1.0, 1.5, 4.0, 100.0, 1e200, 1e300):
        assert acosh_(x) == pytest.approx(math.acosh(x), rel=1e-12, abs=1e-15), x
    for x in (-0.99, -0.5, 0.0, 0.5, 0.99):
        assert atanh_(x) == pytest.approx(math.atanh(x), rel=1e-12, abs=1e-15), x
    with pytest.raises(ValueError):
        acosh_(0.5)  # domain violation (sqrt of a negative), like math.acosh


def test_expm1_log1p() -> None:
    for x in (-1.0, -0.1, 0.1, 1.0, 10.0):
        assert expm1_(x) == pytest.approx(math.expm1(x), rel=1e-12, abs=1e-15), x
    for x in (-0.5, 0.0, 0.5, 10.0, 100.0):
        assert log1p_(x) == pytest.approx(math.log1p(x), rel=1e-12, abs=1e-15), x


def test_degrees_radians() -> None:
    for x in (-3.14, -1.0, 0.0, 1.0, 90.0):
        assert degrees_(x) == pytest.approx(math.degrees(x), rel=1e-12, abs=1e-15), x
        assert radians_(x) == pytest.approx(math.radians(x), rel=1e-12, abs=1e-15), x


def test_matmul_matches_numpy_in_every_rank_combination() -> None:
    rng = np.random.default_rng(20260710)
    a, b = rng.normal(size=(3, 4)), rng.normal(size=(4, 2))
    u, w = rng.normal(size=4), rng.normal(size=3)
    # The left fold is not BLAS's summation order, so the agreement is up to rounding, not bit-exact.
    assert np.allclose(matmul_(a, b), a @ b)
    assert np.allclose(matmul_(a, u), a @ u)  # a 1-D right operand is a column whose axis is dropped
    assert np.allclose(matmul_(w, a), w @ a)  # a 1-D left operand is a row whose axis is dropped
    assert np.allclose(matmul_(u, u), u @ u)  # both promoted and both dropped: a scalar dot product
    assert np.ndim(matmul_(u, u)) == 0


def test_transpose_helper_matches_numpy() -> None:
    m = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert np.allclose(_transpose(m), m.T)  # the matrix product's column-walk helper, 2-D only by contract


def test_trace_and_outer_match_numpy() -> None:
    rng = np.random.default_rng(20260712)
    s, u, v = rng.normal(size=(4, 4)), rng.normal(size=3), rng.normal(size=5)
    assert trace_(s) == pytest.approx(np.trace(s))
    assert np.allclose(outer_(u, v), np.outer(u, v))


def test_linalg_stubs_reject_the_shapes_they_do_not_support() -> None:
    # The stub raises in plain Python exactly where lowering rejects the kernel; the messages are the diagnostics the
    # frontend surfaces at the user's call site, so they are asserted here rather than only through the compiler.
    scalar, vector, matrix, cube = np.float64(2.0), np.arange(3.0), np.ones((2, 3)), np.ones((2, 2, 2))
    with pytest.raises(ValueError, match="scalar"):
        matmul_(scalar, vector)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="1-D or 2-D"):
        matmul_(cube, vector)
    assert np.allclose(matmul_(matrix, vector), matrix @ vector)  # 2×3 @ 3 agrees, so the mismatch below is genuine
    with pytest.raises(ValueError, match="mismatch"):
        matmul_(matrix, np.arange(2.0))
    # Transpose is structural in the compiler now; _transpose here is only the matrix product's plain
    # column-walk helper, exercised through matmul_ above (its scalar/rank guards live in the caller).
    assert np.allclose(_transpose(matrix), matrix.T)
    with pytest.raises(ValueError, match="square"):
        trace_(matrix)
    with pytest.raises(ValueError, match="matrix"):
        trace_(vector)
    with pytest.raises(ValueError, match="1-D"):
        outer_(matrix, vector)


def test_empty_matrix_product_keeps_its_trailing_dimension() -> None:
    # Regression (A3): the 2-D x 2-D product of an empty-outer operand collapsed to shape (0,), so folded
    # shape queries lied about the rank. The stub must match numpy's shape exactly.
    from holoso._frontend._lib._linalg import matmul_

    empty = np.zeros((0, 3))
    other = np.zeros((3, 2))
    assert matmul_(empty, other).shape == np.matmul(empty, other).shape == (0, 2)
    assert matmul_(other.T, empty.T).shape == np.matmul(other.T, empty.T).shape == (2, 0)


def test_integer_trace_stays_integer() -> None:
    # Regression (C5): the trace accumulator started at 0.0, folding an integer matrix to float and breaking
    # range(np.trace(M)).
    from holoso._frontend._lib._linalg import trace_

    m = np.array([[1, 2], [3, 4]])
    result = trace_(m)
    assert result == np.trace(m) == 5
    assert not isinstance(result, float) and isinstance(int(result), int)
    assert isinstance(result, (int, np.integer))
