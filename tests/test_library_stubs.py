"""
Plain-Python numerical verification of the library stubs: each composite stub is executed directly (no compiler
involved) and compared against the math/numpy function it substitutes at host binary64 precision. This checks the
ALGORITHM (the identity the stub encodes) at rel~1e-12, which no public path can reach: the composite stubs cannot
be configured at FloatFormat(11, 52) (no flog2 tables there), the synthesizable counterparts in
test_extended_operators resolve only ~1e-4..1e-7, and the large-|x| inverse-hyperbolic branches overflow every
synthesizable format. The lowering of the same stubs is checked end-to-end in test_extended_operators /
test_matrix / test_cosim.
"""

import math
from typing import Any

import numpy as np
import pytest
from jaxtyping import Float64

import holoso
from holoso import FAddOptions, OperatorOptions, Options, UnsupportedConstruct
from holoso._eel import lower
from holoso._eel._lib import Array, ScalarFunction, resolve
from holoso._eel._ir import BinaryOp, ScalarType
from holoso._eel._lib._linalg import cross, inv, matmul, norm, transpose

from ._eeloracle import assert_hir_matches_reference
from ._modelref import DEFAULT_UNROLL_MAX_TRIPS, spd_matrix
from holoso._eel._lib._intrinsics import (
    abs_float,
    atan2,
    ceil,
    cos,
    exp2,
    floor,
    fma,
    hypot,
    isfinite,
    isinf,
    isneginf,
    isposinf,
    log2,
    max_float,
    min_float,
    round_,
    sin,
    sqrt,
    trunc,
)
from holoso._eel._lib._pow import pow_, pow_chain_float, pow_chain_int, pow_reciprocal
from holoso._eel._lib._numpy import (
    acos,
    acosh,
    asin,
    asinh,
    atan,
    atanh,
    cbrt,
    cosh,
    degrees,
    exp,
    expm1,
    log10,
    log1p,
    log,
    maximum,
    minimum,
    radians,
    sign_float,
    sinh,
    tan,
    tanh,
)

_INF = float("inf")


def test_registry_resolves_the_expected_externals() -> None:
    for external in (np.transpose, np.ravel, np.dot, np.trace, np.outer, np.cross, np.linalg.norm, np.linalg.inv):
        assert isinstance(resolve(external), Array), external
    assert resolve(np.minimum) == Array(minimum) == resolve(np.fmin)  # type: ignore[arg-type]
    assert resolve(np.maximum) == Array(maximum) == resolve(np.fmax)  # type: ignore[arg-type]
    # An operator is a key like any callee object, so `**` and its every spelling are ONE four-lowering entry.
    power_entry = resolve(BinaryOp.POW)
    assert isinstance(power_entry, ScalarFunction) and len(power_entry.lowerings) == 4
    for power in (pow, math.pow, np.power, np.pow, np.float_power):
        assert resolve(power) == power_entry, power
    assert resolve(np.matmul) == Array(matmul) == resolve(BinaryOp.MATMUL)  # type: ignore[arg-type]
    # A transpose is a non-copying derivation on the host, so its match carries the storage-equivalence flag.
    assert resolve(np.transpose) == Array(transpose, derives=True)  # type: ignore[arg-type]
    assert resolve(np.dot) == Array(matmul)  # type: ignore[arg-type]
    assert resolve(np.zeros(3)) is None  # an unhashable shadow does not crash the lookup


def test_unregistered_calls_refuse_through_public_synthesis() -> None:
    def erf_kernel(x: float) -> float:
        return math.erf(x)

    def inner_kernel(v: Float64[np.ndarray, "2"], w: Float64[np.ndarray, "2"]) -> float:
        return np.inner(v, w)  # type: ignore[no-any-return]

    for kernel, match in (
        (erf_kernel, r"calls to 'math\.erf' are not supported yet"),
        (inner_kernel, r"calls to 'np\.inner' are not supported yet"),
    ):
        with pytest.raises(UnsupportedConstruct, match=match):
            holoso.synthesize(kernel, Options(OperatorOptions(fadd=FAddOptions())), name="kernel")


# Enumerated, so dropping a spelling or a domain fails here rather than going unnoticed.
_FLOAT_ONLY: list[object] = [
    math.sqrt, np.sqrt, math.sin, np.sin, math.cos, np.cos, math.tan, np.tan,
    math.asin, np.arcsin, math.acos, np.arccos, math.atan, np.arctan, math.atan2, np.arctan2,
    math.sinh, np.sinh, math.cosh, np.cosh, math.tanh, np.tanh,
    math.asinh, np.arcsinh, math.acosh, np.arccosh, math.atanh, np.arctanh,
    math.exp, np.exp, math.exp2, np.exp2, math.expm1, np.expm1,
    math.log, np.log, math.log2, np.log2, math.log10, np.log10, math.log1p, np.log1p,
    math.degrees, np.degrees, np.rad2deg, math.radians, np.radians, np.deg2rad,
    math.cbrt, np.cbrt, math.fabs, np.fabs, math.fma, math.hypot, np.hypot,
    math.isfinite, np.isfinite, math.isinf, np.isinf, np.isneginf, np.isposinf,
    np.rint,  # the one rounding spelling whose own answer on an integer is a float
]  # fmt: skip
_INT_AND_FLOAT: list[object] = [
    abs, np.abs, np.absolute, min, max, np.sign,
    round, np.round, np.around, math.floor, np.floor, math.ceil, np.ceil, math.trunc, np.trunc, np.fix,
    pow, math.pow, np.power, np.float_power, BinaryOp.POW,
]  # fmt: skip
_INT_ONLY: list[object] = [int.bit_count, np.bitwise_count]


def test_every_spelling_resolves_with_the_domains_it_serves() -> None:
    """
    White-box registry-completeness sentinel: driving all ~80 spellings through synthesize would be prohibitively
    slow, and dropping a spelling or a domain must fail here rather than go unnoticed.
    """
    for external, served in ((e, [ScalarType.FLOAT]) for e in _FLOAT_ONLY):
        match = resolve(external)
        assert isinstance(match, ScalarFunction), external
        assert match.domains == served, external
    for external in _INT_AND_FLOAT:
        match = resolve(external)
        assert isinstance(match, ScalarFunction), external
        assert match.domains == [ScalarType.INT, ScalarType.FLOAT], external
    for external in _INT_ONLY:
        match = resolve(external)
        assert isinstance(match, ScalarFunction), external
        assert match.domains == [ScalarType.INT], external
    for external in (
        np.transpose,
        np.ravel,
        np.dot,
        np.trace,
        np.outer,
        np.cross,
        np.matmul,
        BinaryOp.MATMUL,
        np.linalg.norm,
        np.linalg.inv,
    ):
        assert isinstance(resolve(external), Array), external
    # The numpy elementwise spellings are array composites; the builtin min/max keep the scalar entries.
    for external in (np.minimum, np.fmin, np.maximum, np.fmax):
        assert isinstance(resolve(external), Array), external
    for member in (np.ndarray.T, np.ndarray.dot, np.ndarray.flatten, np.ndarray.ravel, np.ndarray.transpose):
        assert isinstance(resolve(member), Array), member


def test_intrinsic_stubs_match_their_references() -> None:
    for x in (0.0, -0.0, 0.75, -2.5, 3.0, 100.0, -1e-30):
        assert exp2(x) == math.exp2(x)
        assert sin(x) == math.sin(x)
        assert cos(x) == math.cos(x)
        assert floor(x) == np.floor(x) and ceil(x) == np.ceil(x) and trunc(x) == np.trunc(x)
        assert abs_float(x) == abs(x)
        assert round_(x) == np.round(x)
        assert isfinite(x) and not isinf(x) and not isposinf(x) and not isneginf(x)
    for x in (0.25, 1.0, 4.0, 1e30):
        assert log2(x) == math.log2(x)
        assert sqrt(x) == math.sqrt(x)
    assert atan2(3.0, -4.0) == math.atan2(3.0, -4.0)
    assert hypot(3.0, 4.0) == 5.0
    assert min_float(1.5, -2.0) == -2.0 and max_float(1.5, -2.0) == 1.5
    assert fma(3.0, 4.0, 5.0) == math.fma(3.0, 4.0, 5.0)
    assert floor(_INF) == _INF and ceil(-_INF) == -_INF
    assert round_(2.5) == 2.0 and round_(3.5) == 4.0 and round_(-_INF) == -_INF
    assert isposinf(_INF) and not isneginf(_INF) and isinf(-_INF) and not isfinite(_INF)
    assert exp2(1e30) == _INF  # saturates like the hardware instead of raising like math.exp2


def test_sign() -> None:
    for x in (1e-300, 0.5, 7.0, _INF):
        assert sign_float(x) == 1.0 and sign_float(-x) == -1.0
    assert sign_float(0.0) == 0.0 and sign_float(-0.0) == 0.0
    assert math.isnan(sign_float(math.nan))  # r = x in the zero branch reproduces np.sign(nan) = nan exactly


def test_cbrt() -> None:
    for x in (8.0, -27.0, 0.5, -1e-6, 1e18, 3.7):
        assert cbrt(x) == pytest.approx(math.cbrt(x), rel=1e-12), x
    assert cbrt(0.0) == 0.0 and cbrt(-0.0) == 0.0


def test_tan() -> None:
    for x in (0.0, 0.3, -1.2, 2.0, 100.0, math.pi / 2):  # pi/2 is not exact in binary64, so tan() is finite there
        assert tan(x) == pytest.approx(math.tan(x), rel=1e-12), x


def test_atan() -> None:
    for x in (0.0, 1.0, -1.0, 0.001, -1e6, _INF):
        assert atan(x) == pytest.approx(math.atan(x), rel=1e-12), x


def test_asin_acos() -> None:
    for x in (0.0, 0.5, -0.5, 0.9, -0.999, 1.0, -1.0):
        assert asin(x) == pytest.approx(math.asin(x), rel=1e-7, abs=1e-9), x
        assert acos(x) == pytest.approx(math.acos(x), rel=1e-7, abs=1e-9), x
    with pytest.raises(ValueError):
        asin(1.5)  # domain violation raises in plain Python (math.sqrt of a negative), like math.asin


def test_exp_log() -> None:
    for x in (0.0, 1.0, -1.0, 10.0, -30.0, 0.001):
        assert exp(x) == pytest.approx(math.exp(x), rel=1e-12), x
    for x in (0.001, 0.5, 1.0, math.e, 100.0, 1e30):
        assert log(x) == pytest.approx(math.log(x), rel=1e-12, abs=1e-15), x
        assert log10(x) == pytest.approx(math.log10(x), rel=1e-12, abs=1e-15), x


def test_pow_recovers_the_sign_of_a_negative_base() -> None:
    # The magnitude rides the exp2/log2 identity over `abs(b)`, so an integer exponent gets its sign back from the
    # exponent's parity rather than from a per-exponent ladder: exact wherever the identity itself is exact.
    for b in (2.0, -2.0, 0.5, -0.5):
        for e in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0):
            assert pow_(b, e) == math.pow(b, e), (b, e)
    assert pow_(0.0, 0.0) == 1.0
    assert pow_(-2.0, 3.0) == -8.0
    # A non-integer exponent of a negative base has no real value; the identity's log2 of a negative operand answers it.
    assert math.isnan(pow_(-2.0, 2.5))


def test_pow_general_path() -> None:
    for b, e in ((2.0, 0.5), (3.0, 2.5), (10.0, -1.5), (0.5, 8.0), (1.0, 123.456), (-1.5, 3.0), (-2.0, 6.0)):
        assert pow_(b, e) == pytest.approx(math.pow(b, e), rel=1e-12), (b, e)
    # `|b| == 1` with a non-finite exponent is the IEEE special case the guard chain answers ahead of exp2(inf*0).
    assert pow_(-1.0, _INF) == 1.0 and pow_(-1.0, -_INF) == 1.0 and pow_(1.0, _INF) == 1.0


def test_pow_chain_matches_the_host_where_the_chain_is_exact() -> None:
    # Exact wherever every partial product is representable; the reciprocal is the one rounding it adds.
    for b in (2.0, -2.0, 0.5, -0.5, 1.0, 3.0):
        for n in (0, 1, 2, 3, 5, 8):
            assert pow_chain_float(b, n) == b**n, (b, n)
        for n in (-1, -2, -3):
            assert pow_reciprocal(b, n) == b**n, (b, n)
    assert pow_chain_float(0.0, 0) == 1.0 and pow_chain_float(-7.5, 0) == 1.0
    assert pow_chain_float(0.0, 3) == 0.0
    with pytest.raises(ZeroDivisionError):
        pow_reciprocal(0.0, -1)  # the reciprocal raises exactly as `0.0 ** -1` does
    for b, n in ((7.0, 5), (1.0000001, 4), (-3.25, 7)):
        assert pow_chain_float(b, n) == pytest.approx(b**n, rel=1e-13), (b, n)


def test_pow_int_is_the_exact_integer_power() -> None:
    # Square-and-multiply over exact host integers, so the answer is exact where the float chain would saturate.
    for b in (0, 1, 2, -2, 3, -7, 10):
        for e in (0, 1, 2, 3, 5, 8, 13):
            assert pow_chain_int(b, e) == b**e, (b, e)
    assert pow_chain_int(2, 200) == 2**200  # exact at arbitrary precision, where the float chain would saturate
    assert pow_chain_int(0, 0) == 1


def test_pow_zero_base() -> None:
    # A zero base needs no rung of its own: the general path is exp2(e * log2(0)), and log2(0) is -inf, so a positive
    # exponent gives exp2(-inf) == 0.0 and a negative one exp2(+inf) == inf -- exactly what math.pow answers.
    for e in (0.0, 0.5, 1.0, 2.0, 5.0, 7.0, 123.4):
        assert pow_(0.0, e) == math.pow(0.0, e), e


def test_pow_unit_base() -> None:
    # A unit base short-circuits to 1.0, so pow(1, e) == 1 for every e -- including a non-finite one, where the general
    # path's exp2(e * log2(1)) = exp2(e * 0) would otherwise yield nan (inf * 0). Matches IEEE 754 / math.pow.
    for e in (0.0, 0.5, 2.0, 7.0, -3.0, _INF, -_INF, math.nan):
        assert pow_(1.0, e) == 1.0, e


def test_hyperbolic() -> None:
    for x in (-4.0, -1.0, -0.1, 0.0, 0.1, 1.0, 4.0):
        assert sinh(x) == pytest.approx(math.sinh(x), rel=1e-12, abs=1e-15), x
        assert cosh(x) == pytest.approx(math.cosh(x), rel=1e-12), x
    for x in (-30.0, -2.0, 0.0, 2.0, 30.0):  # the stable sigmoid form holds tanh in [-1,1] without exp overflow
        assert tanh(x) == pytest.approx(math.tanh(x), rel=1e-12, abs=1e-15), x


def test_inverse_hyperbolic() -> None:
    # 1e200/1e300 exceed float64's own x*x overflow (~1.3e154), exercising the large-|x| branch that returns ln(2|x|).
    for x in (-1e6, -2.0, 0.0, 2.0, 1e6, 1e200, -1e200, 1e300):  # the sign/abs form also keeps large-negative asinh
        assert asinh(x) == pytest.approx(math.asinh(x), rel=1e-12, abs=1e-15), x
    for x in (1.0, 1.5, 4.0, 100.0, 1e200, 1e300):
        assert acosh(x) == pytest.approx(math.acosh(x), rel=1e-12, abs=1e-15), x
    for x in (-0.99, -0.5, 0.0, 0.5, 0.99):
        assert atanh(x) == pytest.approx(math.atanh(x), rel=1e-12, abs=1e-15), x
    with pytest.raises(ValueError):
        acosh(0.5)  # domain violation (sqrt of a negative), like math.acosh


def test_expm1_log1p() -> None:
    for x in (-1.0, -0.1, 0.1, 1.0, 10.0):
        assert expm1(x) == pytest.approx(math.expm1(x), rel=1e-12, abs=1e-15), x
    for x in (-0.5, 0.0, 0.5, 10.0, 100.0):
        assert log1p(x) == pytest.approx(math.log1p(x), rel=1e-12, abs=1e-15), x


def test_degrees_radians() -> None:
    for x in (-3.14, -1.0, 0.0, 1.0, 90.0):
        assert degrees(x) == pytest.approx(math.degrees(x), rel=1e-12, abs=1e-15), x
        assert radians(x) == pytest.approx(math.radians(x), rel=1e-12, abs=1e-15), x


def test_inv_stub_matches_numpy() -> None:
    rng = np.random.default_rng(1234)
    cases: list[np.ndarray] = [np.eye(n) for n in range(1, 5)]
    cases.append(np.array([[0.0, 1.0], [1.0, 0.0]]))  # a zero leading pivot: unpivoted elimination would divide by 0
    cases.append(np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]))  # cyclic: pivots at every column
    for n in (2, 3, 4):
        cases.append(rng.uniform(-1.0, 1.0, (n, n)) + np.eye(n) * n)  # diagonally dominant
        cases.append(spd_matrix(rng, n))
    cases.append(np.array([[1.0, 1.0], [1.0, 1.004]]))  # cond ~ 1e3
    cases.append(np.array([[3e5, -1e5], [2e5, 7e5]]))  # large-magnitude elements
    for m in cases:
        original = m.copy()
        assert np.allclose(inv(m), np.linalg.inv(m), rtol=1e-11, atol=1e-13), m
        assert np.array_equal(m, original)  # the argument is never mutated
    promoted = inv(np.array([[2, 0], [0, 4]]))  # an int matrix promotes to a float inverse, as numpy's does
    assert promoted.dtype == np.float64 and np.allclose(promoted, [[0.5, 0.0], [0.0, 0.25]], rtol=1e-15)
    with pytest.raises(ValueError, match="matrix"):
        inv(np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="square"):
        inv(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))


def test_cross_stub_matches_numpy() -> None:
    rng = np.random.default_rng(3141)
    for _ in range(8):
        u, v = rng.uniform(-4.0, 4.0, 3), rng.uniform(-4.0, 4.0, 3)
        assert np.allclose(cross(u, v), np.cross(u, v), rtol=1e-15, atol=0.0)
    # numpy >= 2 removed the 2-vector form, so the stub's 1-element-array answer has no external reference.
    for _ in range(4):
        pq = rng.uniform(-4.0, 4.0, 4)
        got = cross(pq[:2], pq[2:])
        assert got.shape == (1,) and float(got[0]) == float(pq[0] * pq[3] - pq[1] * pq[2])
    ints = cross(np.array([1, 0, 0]), np.array([0, 1, 0]))  # an int pair keeps the int elements, as numpy's does
    assert ints.dtype == np.int64 and np.array_equal(ints, [0, 0, 1])
    with pytest.raises(ValueError, match="1-D"):
        cross(np.array([[1.0, 0.0, 0.0]]), np.array([0.0, 1.0, 0.0]))
    with pytest.raises(ValueError, match="unsupported cross"):
        cross(np.array([1.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    with pytest.raises(ValueError, match="unsupported cross"):
        cross(np.array([1.0, 0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0, 0.0]))


def test_cross_inlining_matches_the_host() -> None:
    def kernel(u: Float64[np.ndarray, "3"], v: Float64[np.ndarray, "3"]) -> Float64[np.ndarray, "3"]:
        return np.cross(u, v)

    rng = np.random.default_rng(2718)
    vectors = [
        {"u_0": 1.0, "u_1": 0.0, "u_2": 0.0, "v_0": 0.0, "v_1": 1.0, "v_2": 0.0},
        *(
            {
                name: float(value)
                for name, value in zip(("u_0", "u_1", "u_2", "v_0", "v_1", "v_2"), rng.uniform(-3, 3, 6))
            }
            for _ in range(6)
        ),
    ]
    assert assert_hir_matches_reference(
        lower(kernel, DEFAULT_UNROLL_MAX_TRIPS).hir, kernel, vectors, label="cross"
    ) == len(vectors)

    def kernel2(u: Float64[np.ndarray, "2"], v: Float64[np.ndarray, "2"]) -> Float64[np.ndarray, "1"]:
        return np.cross(u, v)

    def reference2(u: Float64[np.ndarray, "2"], v: Float64[np.ndarray, "2"]) -> Float64[np.ndarray, "1"]:
        # numpy >= 2 removed the 2-vector form, so the stub's own Python is the reference.
        return cross(u, v)  # type: ignore[no-any-return]

    vectors2 = [
        {"u_0": 1.0, "u_1": 0.0, "v_0": 0.0, "v_1": 1.0},
        {"u_0": 2.0, "u_1": -3.0, "v_0": 0.5, "v_1": 1.25},
        {"u_0": -1.5, "u_1": -1.5, "v_0": -1.5, "v_1": -1.5},
    ]
    assert assert_hir_matches_reference(
        lower(kernel2, DEFAULT_UNROLL_MAX_TRIPS).hir, reference2, vectors2, label="cross2"
    ) == len(vectors2)


def test_norm_stub_matches_numpy() -> None:
    rng = np.random.default_rng(2020)
    for n in (1, 2, 3, 5):
        v = rng.uniform(-4.0, 4.0, n)
        assert np.isclose(norm(v), np.linalg.norm(v), rtol=1e-14, atol=0.0)
        for order in (2, 1, math.inf, -math.inf):
            assert np.isclose(norm(v, order), np.linalg.norm(v, order), rtol=1e-14, atol=0.0), (n, order)
    ints = np.array([10000, 10000, 10000])  # int elements promote to float, so nothing saturates on the way
    assert np.isclose(norm(ints), np.linalg.norm(ints), rtol=1e-14, atol=0.0)
    assert np.isclose(norm(ints, 1), np.linalg.norm(ints, 1), rtol=1e-14, atol=0.0)
    m = rng.uniform(-4.0, 4.0, (2, 3))
    assert np.isclose(norm(m), np.linalg.norm(m), rtol=1e-14, atol=0.0)  # the default answers Frobenius, as numpy's
    with pytest.raises(ValueError, match="unsupported matrix norm"):
        norm(m, 2)
    with pytest.raises(ValueError, match="unsupported vector norm"):
        norm(rng.uniform(-1.0, 1.0, 3), 3)
    with pytest.raises(ValueError, match="1-D or 2-D"):
        norm(np.float64(1.0))  # type: ignore[arg-type]


def test_norm_inlining_matches_the_host() -> None:
    def kernel(v: Float64[np.ndarray, "3"]) -> tuple[float, float, float, float]:
        return (
            np.linalg.norm(v),
            np.linalg.norm(v, 1),
            np.linalg.norm(v, np.inf),
            np.linalg.norm(v, -np.inf),
        )

    rng = np.random.default_rng(1616)
    vectors = [
        {"v_0": 3.0, "v_1": -4.0, "v_2": 12.0},
        {"v_0": 0.0, "v_1": 0.0, "v_2": 0.0},
        *({name: float(value) for name, value in zip(("v_0", "v_1", "v_2"), rng.uniform(-5, 5, 3))} for _ in range(5)),
    ]
    assert assert_hir_matches_reference(
        lower(kernel, DEFAULT_UNROLL_MAX_TRIPS).hir, kernel, vectors, label="norm"
    ) == len(vectors)

    def int_kernel(a: int, b: int) -> float:
        return np.linalg.norm(np.array([a, b]))  # type: ignore[no-any-return]  # int elements answer a float norm

    int_vectors = [{"a": a, "b": b} for a, b in [(3, 4), (0, 0), (-5, 12), (10000, 10000)]]
    assert assert_hir_matches_reference(
        lower(int_kernel, DEFAULT_UNROLL_MAX_TRIPS).hir, int_kernel, int_vectors, label="int_norm"
    ) == len(int_vectors)


def test_minimum_maximum_stubs_match_numpy() -> None:
    rng = np.random.default_rng(4321)
    vec = rng.uniform(-4.0, 4.0, 3)
    mat = rng.uniform(-4.0, 4.0, (2, 3))
    pairs: list[tuple[Any, Any]] = [
        (np.float64(2.5), np.float64(-1.0)),
        (vec, np.float64(0.5)),
        (np.float64(0.5), vec),
        (mat, np.float64(-0.25)),
        (np.float64(-0.25), mat),
        (vec, rng.uniform(-4.0, 4.0, 3)),
        (mat, rng.uniform(-4.0, 4.0, (2, 3))),
    ]
    for stub, reference in ((minimum, np.minimum), (maximum, np.maximum)):
        for a, b in pairs:
            assert np.array_equal(stub(a, b), reference(a, b)), (stub.__name__, a, b)
    ints = np.array([3, -5, 7])
    got = minimum(ints, np.array([1, 9, -7]))
    assert got.dtype == np.int64 and np.array_equal(got, [1, -5, -7])
    # Mixed dtypes: the values match numpy's promoted result even though the stub's dtype may not promote...
    assert np.array_equal(maximum(ints, np.array([2.5, -6.0, 8.0])), np.maximum(ints, [2.5, -6.0, 8.0]))
    # ...until the float's exact-integer range: the stub compares exactly where numpy promotes first, so it keeps
    # the true maximum that promotion rounds away (the documented divergence, pinned in both directions).
    assert maximum(np.array([2**53 + 1]), np.array([float(2**53)]))[0] == 2**53 + 1
    assert np.maximum(np.array([2**53 + 1]), np.array([float(2**53)]))[0] == float(2**53)
    with pytest.raises(ValueError, match="shape mismatch"):
        minimum(vec, rng.uniform(-1.0, 1.0, 4))
    with pytest.raises(ValueError, match="shape mismatch"):
        maximum(vec, mat)
    with pytest.raises(ValueError, match="rows of length"):
        minimum(mat, rng.uniform(-4.0, 4.0, (2, 2)))
    with pytest.raises(ValueError, match="at most 2-D"):
        minimum(rng.uniform(-1.0, 1.0, (2, 2, 2)), np.float64(0.0))  # type: ignore[arg-type]


def test_minimum_maximum_inlining_matches_the_host() -> None:
    """The array-composite route: scalar broadcast on either side over floats, and the int lowering per element."""

    def clamp2(v: Float64[np.ndarray, "2"]) -> Float64[np.ndarray, "2"]:
        return np.minimum(np.maximum(v, -1.0), 1.5)  # type: ignore[no-any-return]

    vectors = [{"v_0": a, "v_1": b} for a, b in [(0.0, 2.0), (-3.0, 1.5), (1.5, -1.0), (7.0, -7.0)]]
    assert assert_hir_matches_reference(
        lower(clamp2, DEFAULT_UNROLL_MAX_TRIPS).hir, clamp2, vectors, label="clamp2"
    ) == len(vectors)

    def clamp22(m: Float64[np.ndarray, "2 2"]) -> Float64[np.ndarray, "2 2"]:
        return np.minimum(np.maximum(m, -1.0), 1.5)  # type: ignore[no-any-return]

    matrix_vectors = [
        {"m_0_0": 0.0, "m_0_1": 2.0, "m_1_0": -3.0, "m_1_1": 1.5},
        {"m_0_0": 7.0, "m_0_1": -7.0, "m_1_0": 0.5, "m_1_1": -0.5},
    ]
    assert assert_hir_matches_reference(
        lower(clamp22, DEFAULT_UNROLL_MAX_TRIPS).hir, clamp22, matrix_vectors, label="clamp22"
    ) == len(matrix_vectors)

    def int_extrema(a: int, b: int) -> int:
        v = np.minimum(np.array([a, b]), np.array([3, -2]))
        w = np.maximum(v, -10)
        return w[0] * 100 + w[1]  # type: ignore[no-any-return]

    int_vectors = [{"a": a, "b": b} for a, b in [(0, 0), (5, -7), (-20, 4), (3, -2)]]
    assert assert_hir_matches_reference(
        lower(int_extrema, DEFAULT_UNROLL_MAX_TRIPS).hir, int_extrema, int_vectors, label="int_extrema"
    ) == len(int_vectors)


def test_composite_stub_inlining_matches_the_host_at_binary64() -> None:
    """
    The one integrated differential at host binary64: the frontend inlines the composite stubs and the HIR evaluator
    reproduces their host values exactly, composing the distinct inlining routes -- tan opens with tuple unpacking,
    cbrt reaches sign_float by name plus the exp2/log2 siblings, sinh is a plain composite. The public counterpart
    (test_extended_operators) traverses the same routes only at synthesizable-format tolerances.
    """

    def kernel(x: float) -> float:
        return math.tan(x) + math.cbrt(x) + math.sinh(x / 4.0)

    vectors = [{"x": 0.5}, {"x": -1.0}, {"x": 8.0}, {"x": 2.0}]
    assert assert_hir_matches_reference(
        lower(kernel, DEFAULT_UNROLL_MAX_TRIPS).hir, kernel, vectors, label="composite_inlining"
    ) == len(vectors)
