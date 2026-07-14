"""
Public-API, black-box behavioral tests for the extended float operators.
Every test drives the compiler only through ``holoso.synthesize(fn, ops).numerical_model.elaborate()``
and asserts on observable output values against an INDEPENDENT reference.
"""

import math
from collections.abc import Callable

import numpy as np
import pytest

import holoso
from holoso import (
    FAddOperator,
    FAtan2Operator,
    FCmpOperator,
    FDivOperator,
    FExp2Operator,
    FloatFormat,
    FloatValue,
    FFmaOperator,
    FLog2Operator,
    FMulILog2OperatorFamily,
    FMulOperator,
    FRoundOperator,
    FSincosOperator,
    FSortOperator,
    OpConfig,
    UnsupportedConstruct,
)

# Bare-name imports so a ``from math import floor`` style kernel resolves through the test module globals.
from math import ceil, floor, log2, trunc

# Aliased imports: the local name is NOT the canonical spelling, so dispatch must resolve by callee-object identity.
from math import floor as aliased_floor
from math import fma as aliased_fma
from math import tan as aliased_tan

FMT = FloatFormat(8, 24)  # binary32: a float64 decode of any in-format value is exact, so math/round is an exact oracle
_POS_INF = float("inf")
_NEG_INF = float("-inf")


def _ops(
    *,
    with_round: bool = True,
    with_fma: bool = True,
    with_sort: bool = True,
    with_exp2: bool = True,
    with_log2: bool = True,
    with_sincos: bool = True,
    with_atan2: bool = True,
) -> OpConfig:
    return OpConfig(
        FAddOperator(FMT),
        FMulOperator(FMT),
        FDivOperator(FMT),
        FMulILog2OperatorFamily(FMT),
        FCmpOperator(FMT),
        fround=FRoundOperator(FMT) if with_round else None,
        ffma=FFmaOperator(FMT) if with_fma else None,
        fsort=FSortOperator(FMT) if with_sort else None,
        fexp2=FExp2Operator(FMT) if with_exp2 else None,
        flog2=FLog2Operator(FMT) if with_log2 else None,
        fsincos=FSincosOperator(FMT) if with_sincos else None,
        fatan2=FAtan2Operator(FMT) if with_atan2 else None,
    )


def _sim(fn: Callable[..., object], name: str) -> holoso.NumericalSimulator:
    return holoso.synthesize(fn, _ops(), name=name).numerical_model.elaborate()


def _bits(value: FloatValue | bool) -> int:
    assert isinstance(value, FloatValue)
    return value.bits


def _round_ref(value: float, mode: int) -> int:
    """Independent reference: round the in-format value with Python's math/round, then re-encode. Exact at binary32."""
    fv = FloatValue.from_float(FMT, value)
    v = float(fv)
    if math.isinf(v):
        return fv.bits
    if mode == 0:
        n = round(v)  # banker's rounding (half to even), matching zkf_round mode 0
    elif mode == 1:
        n = math.floor(v)
    elif mode == 2:
        n = math.ceil(v)
    else:
        n = math.trunc(v)
    return FloatValue.from_float(FMT, float(n)).bits


@pytest.mark.skip(reason="FIR_PARITY_PENDING: tuple return — stage 9 aggregate returns")
def test_float_classification_intrinsics() -> None:
    def kernel(x: float) -> tuple[bool, bool, bool, bool, bool, bool]:
        return (
            math.isfinite(x),
            math.isinf(x),
            bool(np.isfinite(x)),
            bool(np.isinf(x)),
            bool(np.isposinf(x)),
            bool(np.isneginf(x)),
        )

    sim = _sim(kernel, "float_classification")
    for x in (0.0, 1.5, -2.0, float("inf"), float("-inf")):
        got = sim.run(x)
        want = [
            math.isfinite(x),
            math.isinf(x),
            bool(np.isfinite(x)),
            bool(np.isinf(x)),
            bool(np.isposinf(x)),
            bool(np.isneginf(x)),
        ]
        assert got == want, f"x={x}: {got} vs {want}"


@pytest.mark.skip(reason="FIR_PARITY_PENDING: aggregate (tuple) returns — stage 9 (aggregate returns/np.array)")
def test_float_classification_constants_use_target_format() -> None:
    def kernel(x: float) -> tuple[bool, bool, bool, bool, bool, bool, bool, bool]:
        return (
            math.isfinite(1e100),
            math.isinf(1e100),
            bool(np.isposinf(1e100)),
            bool(np.isneginf(-1e100)),
            math.isfinite(_POS_INF),
            math.isinf(_NEG_INF),
            bool(np.isposinf(_POS_INF)),
            bool(np.isneginf(_NEG_INF)),
        )

    assert _sim(kernel, "float_classification_const").run(0.0) == [
        False,
        True,
        True,
        True,
        False,
        True,
        True,
        True,
    ]


# A battery spanning ties (x.5 at both parities), the sub-one band (|x| < 1), already-integral values, and infinities.
_ROUND_VECTORS = [
    0.0,
    0.3,
    -0.3,
    0.5,
    -0.5,
    0.7,
    -0.7,
    1.5,
    -1.5,
    2.5,
    -2.5,
    3.5,
    -3.5,
    4.5,
    -4.5,
    1.0,
    -1.0,
    2.0,
    -2.0,
    100.7,
    -100.7,
    16777215.5,
    -16777215.5,
    float("inf"),
    float("-inf"),
]


@pytest.mark.skip(reason="FIR_PARITY_PENDING: aggregate (tuple) returns — stage 9 (aggregate returns/np.array)")
def test_round_modes_match_reference() -> None:
    def kernel(x: float) -> tuple[float, float, float, float]:
        return (math.floor(x), math.ceil(x), math.trunc(x), round(x))

    sim = _sim(kernel, "round_all_modes")
    modes_by_output = (1, 2, 3, 0)  # floor, ceil, trunc, round
    for value in _ROUND_VECTORS:
        out = sim.run(value)
        for index, mode in enumerate(modes_by_output):
            assert _bits(out[index]) == _round_ref(value, mode), f"value={value} output={index} mode={mode}"


@pytest.mark.skip(reason="FIR_PARITY_PENDING: aggregate (tuple) returns — stage 9 (aggregate returns/np.array)")
def test_round_dispatch_numpy_and_bare_name() -> None:
    # numpy.<name> under an alias, and bare names imported via ``from math import ...`` must both dispatch.
    def kernel(x: float) -> tuple[float, float, float, float, float, float]:
        return (np.floor(x), np.ceil(x), np.trunc(x), floor(x), ceil(x), trunc(x))

    sim = _sim(kernel, "round_dispatch")
    for value in _ROUND_VECTORS:
        out = sim.run(value)
        for index, mode in enumerate((1, 2, 3, 1, 2, 3)):
            assert _bits(out[index]) == _round_ref(value, mode), f"value={value} output={index}"


@pytest.mark.skip(reason="FIR_PARITY_PENDING: aggregate (tuple) returns — stage 9 (aggregate returns/np.array)")
def test_round_sign_folds_into_operand() -> None:
    # The input sign chain folds onto the rounder operand and is applied BEFORE rounding: floor(-x) is the rounder fed
    # -x, NOT a negation of floor(x). Asserts the directional modes against the directly-negated reference.
    def kernel(x: float) -> tuple[float, float, float]:
        return (math.floor(-x), math.ceil(abs(x)), math.trunc(-x))

    sim = _sim(kernel, "round_sign_fold")
    for value in _ROUND_VECTORS:
        out = sim.run(value)
        assert _bits(out[0]) == _round_ref(-value, 1), f"floor(-x) value={value}"
        assert _bits(out[1]) == _round_ref(abs(value), 2), f"ceil(|x|) value={value}"
        assert _bits(out[2]) == _round_ref(-value, 3), f"trunc(-x) value={value}"


def test_round_ndigits_is_rejected() -> None:
    def kernel(x: float) -> float:
        return round(x, 2)

    with pytest.raises(UnsupportedConstruct):
        holoso.synthesize(kernel, _ops(), name="round_ndigits")


def test_round_unconfigured_is_rejected() -> None:
    def kernel(x: float) -> float:
        return math.floor(x)

    with pytest.raises(UnsupportedConstruct):
        holoso.synthesize(kernel, _ops(with_round=False), name="round_unconfigured")


def test_fma_matches_single_rounded_reference() -> None:
    def kernel(a: float, b: float, c: float) -> float:
        return math.fma(a, b, c)

    sim = _sim(kernel, "fma_basic")
    # Hand-computed exact cases: exact cancellation to +0 and an exactly-representable sum.
    assert float(sim.run(2.0, 3.0, -6.0)[0]) == 0.0
    assert float(sim.run(1.5, 2.0, 1.0)[0]) == 4.0
    # Random sweep against the exact fused reference (FloatValue.fma); the HDL bench anchors that reference to the RTL.
    rng = np.random.default_rng(0xF)
    for _ in range(3000):
        a, b, c = (float(np.float32(rng.standard_normal() * 12)) for _ in range(3))
        ref = FloatValue.fma(
            FloatValue.from_float(FMT, a), FloatValue.from_float(FMT, b), FloatValue.from_float(FMT, c)
        )
        assert _bits(sim.run(a, b, c)[0]) == ref.bits, f"a={a} b={b} c={c}"


def test_fma_sign_folds_per_operand() -> None:
    # Each operand's sign chain folds independently: math.fma(-a, |b|, -c) is (-a)*|b| + (-c).
    def kernel(a: float, b: float, c: float) -> float:
        return math.fma(-a, abs(b), -c)

    sim = _sim(kernel, "fma_signs")
    rng = np.random.default_rng(0x5)
    for _ in range(1500):
        a, b, c = (float(np.float32(rng.standard_normal() * 12)) for _ in range(3))
        ref = FloatValue.fma(
            FloatValue.from_float(FMT, -a), FloatValue.from_float(FMT, abs(b)), FloatValue.from_float(FMT, -c)
        )
        assert _bits(sim.run(a, b, c)[0]) == ref.bits, f"a={a} b={b} c={c}"


def test_fma_unconfigured_is_rejected() -> None:
    def kernel(a: float, b: float, c: float) -> float:
        return math.fma(a, b, c)

    with pytest.raises(UnsupportedConstruct):
        holoso.synthesize(kernel, _ops(with_fma=False), name="fma_unconfigured")


@pytest.mark.skip(reason="FIR_PARITY_PENDING: aggregate (tuple) returns — stage 9 (aggregate returns/np.array)")
def test_intrinsic_dispatch_resolves_aliased_imports() -> None:
    # An aliased import binds a non-canonical local name to the real function object; dispatch resolves by callee
    # identity, so ``aliased_floor`` (= math.floor) lowers as floor and ``aliased_fma`` (= math.fma) as fma.
    def kernel(a: float, b: float, c: float) -> tuple[float, float]:
        return (aliased_floor(a), aliased_fma(a, b, c))

    sim = _sim(kernel, "aliased_intrinsics")
    for a, b, c in [(2.7, 3.0, 1.0), (-1.5, 2.0, -0.5), (0.3, -4.0, 2.0), (-100.7, 1.5, 3.5)]:
        out = sim.run(a, b, c)
        assert _bits(out[0]) == _round_ref(a, 1), f"floor alias a={a}"
        assert _bits(out[1]) == FloatValue.fma(_v(a), _v(b), _v(c)).bits, f"fma alias a={a} b={b} c={c}"


@pytest.mark.skipif(hasattr(np, "fma"), reason="np.fma exists on this numpy and correctly dispatches to ffma")
def test_numpy_fma_is_rejected() -> None:
    # ``np.fma`` does not exist on this numpy, so it does not resolve to a real function and must not dispatch to ffma
    # by spelling alone (it would not run as plain Python either); the skip guards the numpy versions that do define it.
    def kernel(a: float, b: float, c: float) -> float:
        return np.fma(a, b, c)  # type: ignore[attr-defined, no-any-return]

    with pytest.raises(UnsupportedConstruct):
        holoso.synthesize(kernel, _ops(), name="numpy_fma")


def _v(x: float) -> FloatValue:
    return FloatValue.from_float(FMT, float(x))


def test_implicit_mul_add_contracts_to_fma_only_with_ffma() -> None:
    # ``a*b + c`` with a single-use product contracts to one fma (single rounding) when ffma is configured, and stays
    # a separate multiply-then-add (double rounding) when it is not. The two genuinely differ on many inputs, so the
    # contraction is observable; the test asserts the exact reference for each configuration and that they diverge.
    def kernel(a: float, b: float, c: float) -> float:
        return a * b + c

    fused = holoso.synthesize(kernel, _ops(with_fma=True), name="contract_on").numerical_model.elaborate()
    separate = holoso.synthesize(kernel, _ops(with_fma=False), name="contract_off").numerical_model.elaborate()
    rng = np.random.default_rng(0x515)
    diverged = 0
    for _ in range(5000):
        a, b, c = (float(np.float32(rng.standard_normal() * 9)) for _ in range(3))
        single = FloatValue.fma(_v(a), _v(b), _v(c)).bits
        double = ((_v(a) * _v(b)) + _v(c)).bits
        assert _bits(fused.run(a, b, c)[0]) == single, f"fused a={a} b={b} c={c}"
        assert _bits(separate.run(a, b, c)[0]) == double, f"separate a={a} b={b} c={c}"
        diverged += single != double
    assert diverged > 0, "expected single- and double-rounded results to differ on some inputs"


@pytest.mark.skip(reason="FIR_PARITY_PENDING: aggregate (tuple) returns — stage 9 (aggregate returns/np.array)")
def test_implicit_fma_not_contracted_when_product_is_shared() -> None:
    # A product used by more than the add (here also returned) must NOT contract -- the rounded product is observed
    # elsewhere, so the add keeps double-rounding semantics even with ffma configured.
    def kernel(a: float, b: float, c: float) -> tuple[float, float]:
        p = a * b
        return p + c, p

    sim = holoso.synthesize(kernel, _ops(with_fma=True), name="shared_product").numerical_model.elaborate()
    rng = np.random.default_rng(0x5AD)
    for _ in range(5000):
        a, b, c = (float(np.float32(rng.standard_normal() * 9)) for _ in range(3))
        product = _v(a) * _v(b)
        assert _bits(sim.run(a, b, c)[0]) == (product + _v(c)).bits, f"add a={a} b={b} c={c}"
        assert _bits(sim.run(a, b, c)[1]) == product.bits, f"product a={a} b={b} c={c}"


def test_implicit_fma_contracts_across_blocks() -> None:
    # The product is computed in the entry block but its only consumer (the add) lives in a conditional arm, so the
    # multiply migrates blocks when it contracts. A division in the other arm keeps the diamond a real branch (division
    # is not speculatable, so if-conversion cannot collapse it to one block), exercising the cross-block path.
    def kernel(a: float, b: float, c: float, cond: bool) -> float:
        p = a * b
        if cond:
            r = p + c
        else:
            r = c / a
        return r

    sim = holoso.synthesize(kernel, _ops(with_fma=True), name="fma_cross_block").numerical_model.elaborate()
    rng = np.random.default_rng(0xB10C)
    diverged = 0
    for _ in range(4000):
        a, b, c = (float(np.float32(rng.standard_normal() * 9 + 1e-3)) for _ in range(3))
        single = FloatValue.fma(_v(a), _v(b), _v(c)).bits
        assert _bits(sim.run(a, b, c, True)[0]) == single, f"taken-arm a={a} b={b} c={c}"
        diverged += single != ((_v(a) * _v(b)) + _v(c)).bits
    assert (
        diverged > 0
    ), "expected the contracted (single-rounded) result to differ from multiply-then-add on the cross-block path"


def test_implicit_fma_distributes_product_sign() -> None:
    # The product's folded sign distributes onto the multiplier operands: negation onto one, absolute onto both. Each
    # variant is its OWN kernel so its product stays single-use (sharing one ``a*b`` across all three would intern to a
    # used-twice product and suppress the contraction).
    def k_neg(a: float, b: float, c: float) -> float:
        return -(a * b) + c

    def k_abs(a: float, b: float, c: float) -> float:
        return abs(a * b) + c

    def k_neg_abs(a: float, b: float, c: float) -> float:
        return -abs(a * b) + c

    cases: list[tuple[Callable[[float, float, float], float], Callable[[float, float, float], int]]] = [
        (k_neg, lambda a, b, c: FloatValue.fma(_v(-a), _v(b), _v(c)).bits),
        (k_abs, lambda a, b, c: FloatValue.fma(_v(abs(a)), _v(abs(b)), _v(c)).bits),
        (k_neg_abs, lambda a, b, c: FloatValue.fma(_v(-abs(a)), _v(abs(b)), _v(c)).bits),
    ]
    rng = np.random.default_rng(0x516)
    for kernel, reference in cases:
        sim = holoso.synthesize(kernel, _ops(with_fma=True), name=kernel.__name__).numerical_model.elaborate()
        for _ in range(2000):
            a, b, c = (float(np.float32(rng.standard_normal() * 9)) for _ in range(3))
            assert _bits(sim.run(a, b, c)[0]) == reference(a, b, c), f"{kernel.__name__} a={a} b={b} c={c}"


# Pairs spanning equal values, both infinities, sign-crossing, zero, and ordinary magnitudes.
_MINMAX_VECTORS = [
    (2.0, 3.0),
    (3.0, 2.0),
    (-1.5, 4.0),
    (4.0, -1.5),
    (5.0, 5.0),
    (0.0, 0.0),
    (-2.0, 2.0),
    (float("inf"), 2.0),
    (2.0, float("inf")),
    (float("-inf"), -3.0),
    (float("inf"), float("-inf")),
    (100.7, -100.7),
    (1e-30, 1e30),
]


@pytest.mark.skip(reason="FIR_PARITY_PENDING: aggregate (tuple) returns — stage 9 (aggregate returns/np.array)")
def test_min_max_match_reference() -> None:
    # min(a,b) and max(a,b) over the same pair fuse into one sorter firing that writes two wide registers at once;
    # both outputs are checked against the bit-preserving reference (FloatValue.sort), which the HDL bench anchors to
    # the RTL. The equal and infinity pairs pin the tie direction and the total-order handling of the extrema.
    def kernel(a: float, b: float) -> tuple[float, float]:
        return (min(a, b), max(a, b))

    sim = _sim(kernel, "min_max_pair")
    for a, b in _MINMAX_VECTORS:
        lo, hi = FloatValue.sort(_v(a), _v(b))
        out = sim.run(a, b)
        assert _bits(out[0]) == lo.bits, f"min a={a} b={b}"
        assert _bits(out[1]) == hi.bits, f"max a={a} b={b}"
    rng = np.random.default_rng(0x504)
    for _ in range(4000):
        a, b = (float(np.float32(rng.standard_normal() * 12)) for _ in range(2))
        lo, hi = FloatValue.sort(_v(a), _v(b))
        out = sim.run(a, b)
        assert _bits(out[0]) == lo.bits and _bits(out[1]) == hi.bits, f"a={a} b={b}"


@pytest.mark.skip(reason="FIR_PARITY_PENDING: aggregate (tuple) returns — stage 9 (aggregate returns/np.array)")
def test_min_max_sign_folds_into_operands() -> None:
    # Each operand's sign chain folds onto its sorter operand and is applied BEFORE the sort: min(-a, |b|) is the
    # sorter fed (-a, |b|). This drives the operand conditioners on a commutative multi-output operator.
    def kernel(a: float, b: float) -> tuple[float, float]:
        return (min(-a, abs(b)), max(abs(a), -b))

    sim = _sim(kernel, "min_max_signs")
    rng = np.random.default_rng(0x510)
    for _ in range(3000):
        a, b = (float(np.float32(rng.standard_normal() * 12)) for _ in range(2))
        lo, _ignore = FloatValue.sort(_v(-a), _v(abs(b)))
        _ignore2, hi = FloatValue.sort(_v(abs(a)), _v(-b))
        out = sim.run(a, b)
        assert _bits(out[0]) == lo.bits, f"min(-a,|b|) a={a} b={b}"
        assert _bits(out[1]) == hi.bits, f"max(|a|,-b) a={a} b={b}"


@pytest.mark.skip(reason="FIR_PARITY_PENDING: aggregate (tuple) returns — stage 9 (aggregate returns/np.array)")
def test_min_max_dispatch_numpy() -> None:
    # numpy.minimum/maximum are the binary elementwise forms and must dispatch by callee identity to the sorter.
    def kernel(a: float, b: float) -> tuple[float, float]:
        return (np.minimum(a, b), np.maximum(a, b))

    sim = _sim(kernel, "min_max_numpy")
    for a, b in _MINMAX_VECTORS:
        lo, hi = FloatValue.sort(_v(a), _v(b))
        out = sim.run(a, b)
        assert _bits(out[0]) == lo.bits and _bits(out[1]) == hi.bits, f"a={a} b={b}"


def test_min_max_wrong_arity_is_rejected() -> None:
    def kernel(a: float, b: float, c: float) -> float:
        return max(a, b, c)  # only the binary form is supported

    with pytest.raises(UnsupportedConstruct):
        holoso.synthesize(kernel, _ops(), name="min_max_arity")


def test_min_max_unconfigured_is_rejected() -> None:
    def kernel(a: float, b: float) -> float:
        return min(a, b)

    with pytest.raises(UnsupportedConstruct):
        holoso.synthesize(kernel, _ops(with_sort=False), name="min_max_unconfigured")


@pytest.mark.skip(reason="FIR_PARITY_PENDING: aggregate (tuple) returns — stage 9 (aggregate returns/np.array)")
def test_min_max_is_not_bit_commutative() -> None:
    # min/max preserve the selected operand's exact bits and break ties toward the second operand, so they are NOT
    # bit-commutative: swapping operands can flip the sign of a zero. Two mirrored mins over the same pair must each
    # keep their source operand order (the operator must not be marked commutative), or out_0's zero sign diverges
    # from the reference. At x=0, sign conditioning makes -0, exposing the tie.
    def kernel(x: float, y: float) -> tuple[float, float]:
        return (min(-x, y), min(y, -x))

    sim = _sim(kernel, "min_max_mirror")
    for x, y in [(0.0, 0.0), (5.0, 5.0), (-3.0, -3.0), (2.0, 7.0), (-4.0, 1.5)]:
        neg_x = _v(x).apply_sign(negate=True, absolute=False)
        ref0 = FloatValue.sort(neg_x, _v(y))[0]
        ref1 = FloatValue.sort(_v(y), neg_x)[0]
        out = sim.run(x, y)
        assert _bits(out[0]) == ref0.bits, f"min(-x,y) x={x} y={y}"
        assert _bits(out[1]) == ref1.bits, f"min(y,-x) x={x} y={y}"


def test_min_max_of_constants_fold() -> None:
    # min/max of two constants fold in the format-agnostic HIR, so a kernel using only constant min/max needs no
    # fsort hardware; synthesizing with fsort unconfigured proves the fold (an unfolded min/max would be rejected).
    def kernel(x: float) -> float:
        return x + min(2.5, 1.5) + max(2.5, 1.5)

    sim = holoso.synthesize(kernel, _ops(with_sort=False), name="min_max_fold").numerical_model.elaborate()
    for x in [0.0, 3.0, -1.5, 100.25]:
        ref = (_v(x) + _v(1.5)) + _v(2.5)
        assert _bits(sim.run(x)[0]) == ref.bits, f"x={x}"


def test_min_max_infinity_constants_fold() -> None:
    def kernel(x: float) -> float:
        return x + min(1e400, 2.0) + max(-1e400, 3.0)

    sim = holoso.synthesize(kernel, _ops(with_sort=False), name="min_max_inf_fold").numerical_model.elaborate()
    for x in [0.0, 3.0, -1.5, 100.25]:
        ref = (_v(x) + _v(2.0)) + _v(3.0)
        assert _bits(sim.run(x)[0]) == ref.bits, f"x={x}"


def test_min_max_nan_constant_is_rejected() -> None:
    def max_selects_finite(x: float) -> float:
        return x + max(2.0, 1e400 - 1e400)

    with pytest.raises(UnsupportedConstruct):
        holoso.synthesize(max_selects_finite, _ops(), name="max_selects_finite")


def _ulp32(value: float) -> float:
    """The binary32 quantum at ``value``'s magnitude, for the coarse native-accuracy guards."""
    if value == 0.0 or not math.isfinite(value):
        return math.ldexp(1.0, -149)
    return math.ldexp(1.0, max(math.frexp(abs(value))[1] - FMT.wman, -149))


_EXP2_VECTORS = [0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 3.5, -3.5, 7.25, -7.25, 10.0, 0.125, -12.5, 100.0, -100.0]
_LOG2_VECTORS = [1.0, 2.0, 4.0, 8.0, 0.5, 0.25, 3.0, 5.0, 1.5, 100.0, 1e-12, 1e12, 16777216.0, 0.1, math.pi]


def test_exp2_matches_model_and_native() -> None:
    def kernel(x: float) -> float:
        return math.exp2(x)

    sim = _sim(kernel, "exp2_basic")
    for x in _EXP2_VECTORS:
        # sim and the reference share FloatValue.exp2 (circular); the native check breaks the circularity.
        out = sim.run(x)[0]
        assert _bits(out) == _v(x).exp2().bits, f"exp2 bit-exact x={x}"
        native = math.exp2(x)
        assert abs(float(out) - native) <= 2 * _ulp32(native), f"exp2 accuracy x={x}"
    rng = np.random.default_rng(0xE2)
    for _ in range(200):
        x = float(np.float32(rng.standard_normal() * 20))
        assert _bits(sim.run(x)[0]) == _v(x).exp2().bits, f"exp2 sweep x={x}"


def test_log2_matches_model_and_native() -> None:
    def kernel(x: float) -> float:
        return math.log2(x)

    sim = _sim(kernel, "log2_basic")
    for x in _LOG2_VECTORS:
        out = sim.run(x)[0]
        assert _bits(out) == _v(x).log2().bits, f"log2 bit-exact x={x}"
        native = math.log2(x)
        assert abs(float(out) - native) <= 2 * _ulp32(native), f"log2 accuracy x={x}"
    rng = np.random.default_rng(0x109)
    for _ in range(200):
        x = float(np.float32(abs(rng.standard_normal()) * 1000 + 1e-9))
        assert _bits(sim.run(x)[0]) == _v(x).log2().bits, f"log2 sweep x={x}"


def test_pow_two_lowers_to_exp2() -> None:
    def k_int_base(x: float) -> float:
        return 2**x

    def k_float_base(x: float) -> float:
        return 2.0**x  # type: ignore[no-any-return]

    for kernel in (k_int_base, k_float_base):
        sim = _sim(kernel, kernel.__name__)
        for x in (0.5, 3.5, -2.5, 7.25):  # fractional: a multiply chain could not produce these
            assert _bits(sim.run(x)[0]) == _v(x).exp2().bits, f"{kernel.__name__} x={x}"


def test_pow_nonconstant_or_nontwo_base_is_rejected() -> None:
    def k_runtime_base(x: float, y: float) -> float:
        return x**y  # type: ignore[no-any-return]

    def k_ten_base(x: float) -> float:
        return 10**x

    for fn in (k_runtime_base, k_ten_base):
        with pytest.raises(UnsupportedConstruct):
            holoso.synthesize(fn, _ops(), name=fn.__name__)


@pytest.mark.skip(reason="FIR_PARITY_PENDING: aggregate (tuple) returns — stage 9 (aggregate returns/np.array)")
def test_exp2_log2_dispatch_numpy_and_bare_name() -> None:
    def kernel(x: float) -> tuple[float, float, float]:
        return (np.exp2(x), np.log2(x), log2(x))

    sim = _sim(kernel, "exp2_log2_dispatch")
    for x in (2.0, 3.0, 100.0):  # positive; dispatch resolves by callee identity, so it is value-independent
        out = sim.run(x)
        assert _bits(out[0]) == _v(x).exp2().bits, f"np.exp2 x={x}"
        assert _bits(out[1]) == _v(x).log2().bits, f"np.log2 x={x}"
        assert _bits(out[2]) == _v(x).log2().bits, f"bare log2 x={x}"


def test_exp2_log2_of_constants_fold() -> None:
    folded = float(math.exp2(1.25) + math.log2(3.5) + np.exp2(-2.0) + log2(8.0))

    def kernel(x: float) -> float:
        return x + (math.exp2(1.25) + math.log2(3.5) + float(np.exp2(-2.0)) + log2(8.0))

    sim = holoso.synthesize(
        kernel, _ops(with_exp2=False, with_log2=False), name="exp2_log2_constants_fold"
    ).numerical_model.elaborate()
    for x in [0.0, 3.0, -1.5, 100.25]:
        ref = _v(x) + _v(folded)
        assert _bits(sim.run(x)[0]) == ref.bits, f"x={x}"


@pytest.mark.skip(reason="FIR_PARITY_PENDING: aggregate (tuple) returns — stage 9 (aggregate returns/np.array)")
def test_exp2_log2_sign_folds_into_operand() -> None:
    def kernel(x: float) -> tuple[float, float]:
        return (math.exp2(-x), math.log2(abs(x)))

    sim = _sim(kernel, "exp2_log2_signs")
    for x in [1.0, -1.0, 2.5, -2.5, 0.5, -0.5, 7.0, -7.0, 0.0]:
        out = sim.run(x)
        assert _bits(out[0]) == _v(-x).exp2().bits, f"exp2(-x) x={x}"
        assert _bits(out[1]) == _v(abs(x)).log2().bits, f"log2(|x|) x={x}"


def test_log2_pole_and_domain_values() -> None:
    # Error cases yield -inf; the pole/domain flags are not modeled here (the HDL bench checks them on the RTL).
    def kernel(x: float) -> float:
        return math.log2(x)

    sim = _sim(kernel, "log2_poles")
    assert float(sim.run(0.0)[0]) == float("-inf")
    for x in [-1.0, -2.5, -1e30]:
        assert float(sim.run(x)[0]) == float("-inf"), f"domain x={x}"
    assert float(sim.run(float("inf"))[0]) == float("inf")


def test_exp2_log2_unconfigured_is_rejected() -> None:
    def exp2_kernel(x: float) -> float:
        return math.exp2(x)

    def log2_kernel(x: float) -> float:
        return math.log2(x)

    for fn, ops in ((exp2_kernel, _ops(with_exp2=False)), (log2_kernel, _ops(with_log2=False))):
        with pytest.raises(UnsupportedConstruct):
            holoso.synthesize(fn, ops, name=fn.__name__)


# The turn<->radian scale constants MIR inserts, encoded in the format exactly as the compiler does.
_INV_TAU = FloatValue.from_float(FMT, 1.0 / (2.0 * math.pi))
_TAU = FloatValue.from_float(FMT, 2.0 * math.pi)

# Angles within a few periods; the turn-scale multiply grows absolute phase error with |x|.
_TRIG_VECTORS = [0.0, 0.25, -0.25, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, math.pi / 2, math.pi, -math.pi, 3.0, -6.0]


def _sincos_ref(x: float) -> tuple[int, int]:
    # Bit-exact reference: turn-native model of the format-scaled operand, mirroring MIR's fmul + fsincos.
    s, c = (_v(x) * _INV_TAU).sincos()
    return s.bits, c.bits


@pytest.mark.skip(reason="FIR_PARITY_PENDING: aggregate (tuple) returns — stage 9 (aggregate returns/np.array)")
def test_sincos_matches_model_and_native() -> None:
    def kernel(x: float) -> tuple[float, float]:
        return (math.sin(x), math.cos(x))

    sim = _sim(kernel, "sincos_basic")
    for x in _TRIG_VECTORS:
        out = sim.run(x)
        sin_bits, cos_bits = _sincos_ref(x)
        assert _bits(out[0]) == sin_bits and _bits(out[1]) == cos_bits, f"sin/cos bit-exact x={x}"
        assert abs(float(out[0]) - math.sin(x)) <= 4 * _ulp32(1.0), f"sin accuracy x={x}"
        assert abs(float(out[1]) - math.cos(x)) <= 4 * _ulp32(1.0), f"cos accuracy x={x}"
    rng = np.random.default_rng(0x51C)
    for _ in range(200):
        x = float(np.float32(rng.standard_normal() * 3))
        out = sim.run(x)
        sin_bits, cos_bits = _sincos_ref(x)
        assert _bits(out[0]) == sin_bits and _bits(out[1]) == cos_bits, f"sincos sweep x={x}"


def test_lone_sin_value() -> None:
    # A lone sin still synthesizes (cos port untapped); this checks its value. The firing count is asserted
    # structurally in test_schedule.
    def sin_only(x: float) -> float:
        return math.sin(x)

    sim = _sim(sin_only, "sin_only")
    for x in _TRIG_VECTORS:
        assert _bits(sim.run(x)[0]) == _sincos_ref(x)[0], f"lone sin x={x}"


@pytest.mark.skip(reason="FIR_PARITY_PENDING: aggregate (tuple) returns — stage 9 (aggregate returns/np.array)")
def test_sincos_sign_folds_into_operand() -> None:
    # sin(-x)/cos(-x) fold the negation onto the scaled operand (CORDIC fed -(x/tau)), so both reuse one firing.
    def kernel(x: float) -> tuple[float, float]:
        return (math.sin(-x), math.cos(-x))

    sim = _sim(kernel, "sincos_signs")
    for x in _TRIG_VECTORS:
        out = sim.run(x)
        sin_bits, cos_bits = _sincos_ref(-x)
        assert _bits(out[0]) == sin_bits and _bits(out[1]) == cos_bits, f"sin/cos(-x) x={x}"


@pytest.mark.skip(reason="FIR_PARITY_PENDING: aggregate (tuple) returns — stage 9 (aggregate returns/np.array)")
def test_sincos_dispatch_numpy() -> None:
    def kernel(x: float) -> tuple[float, float]:
        return (np.sin(x), np.cos(x))

    sim = _sim(kernel, "sincos_numpy")
    for x in (0.5, 1.0, -2.0):
        out = sim.run(x)
        sin_bits, cos_bits = _sincos_ref(x)
        assert _bits(out[0]) == sin_bits and _bits(out[1]) == cos_bits, f"np.sin/cos x={x}"


def test_sincos_unconfigured_is_rejected() -> None:
    def kernel(x: float) -> float:
        return math.sin(x)

    with pytest.raises(UnsupportedConstruct):
        holoso.synthesize(kernel, _ops(with_sincos=False), name="sincos_unconfigured")


_ATAN2_VECTORS = [(1.0, 1.0), (3.0, 4.0), (-3.0, 4.0), (3.0, -4.0), (-3.0, -4.0), (1.0, 0.0), (0.0, 1.0), (2.5, -0.5)]


def _atan2_ref(y: float, x: float) -> tuple[int, int]:
    # theta is scaled from turns to radians by MIR's post-multiply; magnitude (hypot) is units-free and unscaled.
    theta_turns, mag = FloatValue.atan2(_v(y), _v(x))
    return (theta_turns * _TAU).bits, mag.bits


def test_atan2_matches_model_and_native() -> None:
    def kernel(y: float, x: float) -> float:
        return math.atan2(y, x)

    sim = _sim(kernel, "atan2_basic")
    for y, x in _ATAN2_VECTORS:
        out = sim.run(y, x)[0]
        assert _bits(out) == _atan2_ref(y, x)[0], f"atan2 bit-exact y={y} x={x}"
        assert abs(float(out) - math.atan2(y, x)) <= 4 * _ulp32(math.pi), f"atan2 accuracy y={y} x={x}"
    rng = np.random.default_rng(0xA7A)
    for _ in range(200):
        y, x = (float(np.float32(rng.standard_normal() * 8)) for _ in range(2))
        assert _bits(sim.run(y, x)[0]) == _atan2_ref(y, x)[0], f"atan2 sweep y={y} x={x}"


def test_atan2_dispatch_numpy_arctan2() -> None:
    # numpy spells the two-arg arctangent ``arctan2`` (== ``np.atan2`` on numpy>=2.0).
    def kernel(y: float, x: float) -> float:
        return np.arctan2(y, x)  # type: ignore[no-any-return]

    sim = _sim(kernel, "atan2_numpy")
    for y, x in _ATAN2_VECTORS:
        assert _bits(sim.run(y, x)[0]) == _atan2_ref(y, x)[0], f"np.arctan2 y={y} x={x}"


def test_atan2_unconfigured_is_rejected() -> None:
    def kernel(y: float, x: float) -> float:
        return math.atan2(y, x)

    with pytest.raises(UnsupportedConstruct):
        holoso.synthesize(kernel, _ops(with_atan2=False), name="atan2_unconfigured")


@pytest.mark.skip(reason="FIR_PARITY_PENDING: aggregate (tuple) returns — stage 9 (aggregate returns/np.array)")
def test_hypot_fused_with_atan2() -> None:
    # hypot(y, x) beside atan2(y, x) fuses into the atan2 CORDIC's magnitude port (units-free, no scale), exact
    # against the model even at the origin and infinities.
    def kernel(y: float, x: float) -> tuple[float, float]:
        return (math.hypot(y, x), math.atan2(y, x))

    sim = _sim(kernel, "hypot_fused")
    for y, x in [*_ATAN2_VECTORS, (0.0, 0.0), (float("inf"), 2.0)]:
        theta_bits, mag_bits = _atan2_ref(y, x)
        out = sim.run(y, x)
        assert _bits(out[0]) == mag_bits, f"fused hypot y={y} x={x}"
        assert _bits(out[1]) == theta_bits, f"fused atan2 y={y} x={x}"


@pytest.mark.skip(reason="FIR_PARITY_PENDING: aggregate (tuple) returns — stage 9 (aggregate returns/np.array)")
def test_hypot_sign_flipped_still_fuses_with_atan2() -> None:
    # The fusion collapses operand signs, so hypot(-x, y) still fuses into atan2(y, x)'s magnitude port. The magnitude
    # is sign-invariant; bit-exactness against the atan2 model confirms the fused path (the primitive decomposition
    # would only be approximate).
    def kernel(y: float, x: float) -> tuple[float, float]:
        return (math.hypot(-x, y), math.atan2(y, x))

    sim = _sim(kernel, "hypot_sign_flipped")
    for y, x in [*_ATAN2_VECTORS, (0.0, 0.0), (float("inf"), 2.0)]:
        _theta_bits, mag_bits = _atan2_ref(y, x)
        assert _bits(sim.run(y, x)[0]) == mag_bits, f"sign-flipped fused hypot y={y} x={x}"


def test_hypot_lone_decomposition_is_approximate() -> None:
    # A lone hypot (no adjacent atan2) falls back to the primitive decomposition (needs fsort/fexp2/flog2); approximate
    # on ordinary finite nonzero inputs.
    def kernel(y: float, x: float) -> float:
        return math.hypot(y, x)

    sim = _sim(kernel, "hypot_lone")
    assert _bits(sim.run(0.0, 0.0)[0]) == _v(0.0).bits
    assert _bits(sim.run(float("inf"), 2.0)[0]) == _v(float("inf")).bits
    rng = np.random.default_rng(0x4F0)
    for _ in range(200):
        y, x = (float(np.float32(rng.standard_normal() * 8)) for _ in range(2))
        native = math.hypot(y, x)
        if native < 1e-3:
            continue
        assert abs(float(sim.run(y, x)[0]) - native) <= 64 * _ulp32(native), f"lone hypot y={y} x={x}"


def test_hypot_lone_missing_primitive_is_rejected() -> None:
    # The decomposition needs fsort/fexp2/flog2; absent any of them, a lone hypot is a clear configuration error.
    def kernel(y: float, x: float) -> float:
        return math.hypot(y, x)

    for ops in (_ops(with_sort=False), _ops(with_exp2=False), _ops(with_log2=False)):
        with pytest.raises(UnsupportedConstruct):
            holoso.synthesize(kernel, ops, name="hypot_lone_reject")


def _sqrt_ref(x: float) -> int:
    if x == 0.0:
        return _v(0.0).bits
    return _v(x).log2().scale_pow2(-1).exp2().bits


def test_sqrt_matches_decomposition_and_native() -> None:
    def kernel(x: float) -> float:
        return math.sqrt(x)

    sim = _sim(kernel, "sqrt_basic")
    assert _bits(sim.run(0.0)[0]) == _v(0.0).bits
    for x in [0.25, 0.5, 1.0, 2.0, 4.0, 9.0, 100.0, 1e-3, 1e6, math.pi]:
        out = sim.run(x)[0]
        assert _bits(out) == _sqrt_ref(x), f"sqrt bit-exact x={x}"
        native = math.sqrt(x)
        assert abs(float(out) - native) <= 32 * _ulp32(native), f"sqrt accuracy x={x}"
    rng = np.random.default_rng(0x59A)
    for _ in range(200):
        x = float(np.float32(abs(rng.standard_normal()) * 100 + 1e-6))
        assert _bits(sim.run(x)[0]) == _sqrt_ref(x), f"sqrt sweep x={x}"


def test_sqrt_dispatch_numpy() -> None:
    def kernel(x: float) -> float:
        return np.sqrt(x)  # type: ignore[no-any-return]

    sim = _sim(kernel, "sqrt_numpy")
    for x in (2.0, 9.0, 0.25):
        assert _bits(sim.run(x)[0]) == _sqrt_ref(x), f"np.sqrt x={x}"


@pytest.mark.skip(reason="FIR_PARITY_PENDING: aggregate (tuple) returns — stage 9 (aggregate returns/np.array)")
def test_trig_of_constants_fold() -> None:
    # Trig of literal operands folds in the format-agnostic HIR, so a kernel of only constant trig needs no CORDIC:
    # synthesizing with fsincos/fatan2 unconfigured proves the fold.
    def kernel(x: float) -> tuple[float, float, float, float, float]:
        return (math.sin(0.5), math.cos(0.5), math.atan2(1.0, 2.0), math.hypot(3.0, 4.0), math.sqrt(2.0))

    ops = _ops(with_sincos=False, with_atan2=False, with_exp2=False, with_log2=False, with_sort=False)
    sim = holoso.synthesize(kernel, ops, name="trig_fold").numerical_model.elaborate()
    out = sim.run(0.0)
    for index, ref in enumerate(
        (math.sin(0.5), math.cos(0.5), math.atan2(1.0, 2.0), math.hypot(3.0, 4.0), math.sqrt(2.0))
    ):
        assert _bits(out[index]) == _v(ref).bits, f"folded output {index}"


def test_constant_fold_declines_unsupported_transcendental_results() -> None:
    # An overflowing compile-time constant (hypot or a product that yields inf) is refused with a located rejection:
    # the fold cannot represent the infinity, so the frontend rejects rather than leaving a hardware op or crashing on
    # a raw ValueError from math.sin.
    def unfoldable(x: float) -> tuple[float, float]:
        overflow = math.hypot(1.5e308, 1.5e308)
        return math.sin(overflow), math.cos(overflow)

    with pytest.raises(UnsupportedConstruct, match="finite"):
        holoso.synthesize(unfoldable, _ops(), name="inf_fold_unfold")

    def sin_of_inf(x: float) -> float:
        return math.sin(1e300 * 1e300) + x

    with pytest.raises(UnsupportedConstruct, match="finite"):
        holoso.synthesize(sin_of_inf, _ops(), name="inf_fold_decline")


def test_atan2_fold_normalizes_signed_zero() -> None:
    # ZKF has no negative zero, so a folded atan2 over -0.0 must match the datapath's atan2(+0.0), not Python's
    # signed-zero branch cut (+/-pi) the datapath can never produce. Regression: FloatConst previously kept -0.0 and
    # flipped the result to -pi.
    def folded(x: float) -> float:
        z = 0.0
        return math.atan2(0.0, -z) + x  # atan2(0.0, -0.0) is a compile-time constant

    def runtime(y: float, x: float) -> float:
        return math.atan2(y, x)

    fold_bits = _bits(_sim(folded, "atan2_fold_neg_zero").run(0.0)[0])
    runtime_bits = _bits(_sim(runtime, "atan2_runtime_zero").run(0.0, 0.0)[0])
    assert fold_bits == runtime_bits, "folded signed-zero atan2 must match the datapath's atan2(0, 0)"


# --- Composite library stubs: black-box value checks against the substituted math/numpy references. The stubs lower
# --- through the ordinary operator set, so tolerances cover binary32 rounding plus the identity's error amplification.


def test_sign_values() -> None:
    def kernel(x: float) -> float:
        return np.sign(x)  # type: ignore[no-any-return]

    sim = _sim(kernel, "lib_sign")
    for x, want in ((3.5, 1.0), (-1e-30, -1.0), (0.0, 0.0), (_POS_INF, 1.0), (_NEG_INF, -1.0)):
        assert float(sim.run(x)[0]) == want, f"sign({x})"


def test_cbrt_matches_reference() -> None:
    def kernel(x: float) -> float:
        return math.cbrt(x)

    sim = _sim(kernel, "lib_cbrt")
    for x in (8.0, -27.0, 0.5, -1e-6, 1e18, 3.7, -0.125):
        assert float(sim.run(x)[0]) == pytest.approx(math.cbrt(x), rel=4e-6), f"cbrt({x})"
    assert float(sim.run(0.0)[0]) == 0.0


def test_tan_matches_reference() -> None:
    def kernel(x: float) -> float:
        return math.tan(x)

    sim = _sim(kernel, "lib_tan")
    for x in (0.0, 0.3, 1.0, -1.2, 2.0):
        assert float(sim.run(x)[0]) == pytest.approx(math.tan(x), rel=1e-5, abs=1e-6), f"tan({x})"


def test_tan_pole_returns_signed_infinity() -> None:
    # At the format-nearest pi/2, cos rounds to exactly zero; the c==0 guard returns a signed infinity (by sin's sign)
    # rather than dividing. Only the value is asserted here; that the divide is skipped (no div-by-zero error) is a
    # real-branch property the cosim exercises via its err_pc check.
    def kernel(x: float) -> float:
        return math.tan(x)

    sim = _sim(kernel, "lib_tan_pole")
    pole = float(np.float32(math.pi / 2))
    assert float(sim.run(pole)[0]) == _POS_INF
    assert float(sim.run(-pole)[0]) == _NEG_INF
    assert float(sim.run(0.5)[0]) == pytest.approx(math.tan(0.5), rel=1e-5)  # a normal input still divides


@pytest.mark.skip(reason="FIR_PARITY_PENDING: aggregate (tuple) returns — stage 9 (aggregate returns/np.array)")
def test_atan_asin_acos_match_references() -> None:
    def kernel(x: float) -> tuple[float, float, float]:
        return (math.atan(x), math.asin(x), math.acos(x))

    sim = _sim(kernel, "lib_arcs")
    for x in (0.0, 0.5, -0.5, 0.9, -0.999, 1.0, -1.0):
        out = sim.run(x)
        assert float(out[0]) == pytest.approx(math.atan(x), abs=1e-5), f"atan({x})"
        assert float(out[1]) == pytest.approx(math.asin(x), abs=1e-4), f"asin({x})"
        assert float(out[2]) == pytest.approx(math.acos(x), abs=1e-4), f"acos({x})"


@pytest.mark.skip(reason="FIR_PARITY_PENDING: aggregate (tuple) returns — stage 9 (aggregate returns/np.array)")
def test_exp_log_log10_match_references() -> None:
    def kernel(x: float) -> tuple[float, float, float]:
        return (math.exp(x), math.log(x), math.log10(x))

    sim = _sim(kernel, "lib_exp_log")
    for x in (0.001, 0.5, 2.0, math.e, 100.0, 1e30):
        out = sim.run(x)
        assert float(out[0]) == pytest.approx(math.exp(min(x, 80.0)), rel=2e-5) or x > 80.0, f"exp({x})"
        assert float(out[1]) == pytest.approx(math.log(x), rel=1e-5, abs=1e-6), f"log({x})"
        assert float(out[2]) == pytest.approx(math.log10(x), rel=1e-5, abs=1e-6), f"log10({x})"


@pytest.mark.skip(reason="FIR_PARITY_PENDING: tuple return — stage 9 aggregate returns")
def test_composite_dispatch_numpy_spellings() -> None:
    def kernel(x: float) -> tuple[float, float, float, float]:
        return (np.cbrt(x), np.tan(x), np.arcsin(x), np.exp(x))

    sim = _sim(kernel, "lib_np_spellings")
    for x in (0.25, -0.5):
        out = sim.run(x)
        assert float(out[0]) == pytest.approx(math.cbrt(x), rel=4e-6)
        assert float(out[1]) == pytest.approx(math.tan(x), rel=1e-5)
        assert float(out[2]) == pytest.approx(math.asin(x), abs=1e-5)
        assert float(out[3]) == pytest.approx(math.exp(x), rel=1e-5)


def test_composite_dispatch_aliased_import() -> None:
    def kernel(x: float) -> float:
        return aliased_tan(x)

    sim = _sim(kernel, "lib_tan_aliased")
    assert float(sim.run(0.7)[0]) == pytest.approx(math.tan(0.7), rel=1e-5)


def test_composite_unconfigured_operator_is_rejected() -> None:
    # cbrt expands through exp2/log2; without them configured the synthesis must fail, proving the expansion is real.
    def kernel(x: float) -> float:
        return math.cbrt(x)

    with pytest.raises(UnsupportedConstruct):
        holoso.synthesize(kernel, _ops(with_exp2=False), name="lib_cbrt_unconfigured")


def test_pow_with_a_constant_exponent_is_a_multiply_chain() -> None:
    # ``pow(x, 3.0)`` with a compile-time exponent lowers to a multiply chain, so it needs neither exp2 nor log2 --
    # exactly like ``x**3``.
    def kernel(x: float) -> float:
        return pow(x, 3.0)  # type: ignore[no-any-return]

    for ops in (_ops(with_exp2=False), _ops(with_log2=False), _ops(with_exp2=False, with_log2=False)):
        sim = holoso.synthesize(kernel, ops, name="lib_pow_const").numerical_model.elaborate()
        for x in (2.0, 3.0, -1.5):
            assert float(sim.run(x)[0]) == pytest.approx(x**3, rel=1e-6)


def test_pow_static_exponent_matches_star_star_bit_exactly() -> None:
    def with_pow(x: float) -> float:
        return pow(x, 3.0)  # type: ignore[no-any-return]

    def with_star(x: float) -> float:
        return x**3

    sim_pow, sim_star = _sim(with_pow, "lib_pow_static"), _sim(with_star, "lib_pow_star")
    for x in (2.0, -2.0, 0.5, -1.5, 100.0):
        assert _bits(sim_pow.run(x)[0]) == _bits(sim_star.run(x)[0]), f"pow(x,3) vs x**3 x={x}"
    assert float(sim_pow.run(-2.0)[0]) == -8.0


def test_pow_runtime_exponent_rungs_and_general_path() -> None:
    def kernel(b: float, e: float) -> float:
        return math.pow(b, e)

    sim = _sim(kernel, "lib_pow_runtime")
    for b in (2.0, -2.0, 0.5, -1.5):  # a rung-hit runtime exponent is exact even for a negative base
        for e in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0):
            assert float(sim.run(b, e)[0]) == pytest.approx(math.pow(b, e), rel=1e-6), f"rung b={b} e={e}"
    for b, e in ((2.0, 0.5), (3.0, 2.5), (10.0, -1.5), (0.5, 8.5)):  # general path: the a>0 exp2/log2 identity
        assert float(sim.run(b, e)[0]) == pytest.approx(math.pow(b, e), rel=1e-5), f"general b={b} e={e}"


@pytest.mark.skip(reason="FIR_PARITY_PENDING: aggregate (tuple) returns — stage 9 (aggregate returns/np.array)")
def test_pow_dispatch_builtin_and_numpy_spellings() -> None:
    def kernel(b: float, e: float) -> tuple[float, float, float]:
        return (pow(b, e), np.power(b, e), np.float_power(b, e))

    sim = _sim(kernel, "lib_pow_spellings")
    out = sim.run(3.0, 2.0)
    assert float(out[0]) == 9.0 and float(out[1]) == 9.0 and float(out[2]) == 9.0


def test_pow_zero_base_returns_correct_value() -> None:
    # The b==0 short-circuit keeps the value correct (0**0 == 1, 0**positive == 0); only the datapath value is asserted
    # here, not the pole error signal (unmodeled by the numerical backend; the RTL cosim is where signals are checked).
    def kernel(b: float, e: float) -> float:
        return math.pow(b, e)

    sim = _sim(kernel, "lib_pow_zero_base")
    assert float(sim.run(0.0, 0.0)[0]) == 1.0
    for e in (0.5, 1.0, 2.0, 7.0, 20.0):
        assert float(sim.run(0.0, e)[0]) == 0.0, f"pow(0, {e})"


def test_pow_unit_base_returns_one_including_infinite_exponent() -> None:
    # The b==1 short-circuit gives pow(1, e) == 1 for every representable e, avoiding the exp2(inf*0)=nan the general
    # path hits at a non-finite exponent. (A NaN exponent is not ZKF-representable, so only +-inf is exercised here.)
    def kernel(b: float, e: float) -> float:
        return math.pow(b, e)

    sim = _sim(kernel, "lib_pow_unit_base")
    for e in (0.0, 2.5, 7.0, _POS_INF, _NEG_INF):
        assert float(sim.run(1.0, e)[0]) == 1.0, f"pow(1, {e})"


@pytest.mark.skip(reason="FIR_PARITY_PENDING: aggregate (tuple) returns — stage 9 (aggregate returns/np.array)")
def test_hyperbolic_match_references() -> None:
    def kernel(x: float) -> tuple[float, float, float]:
        return (math.sinh(x), math.cosh(x), math.tanh(x))

    sim = _sim(kernel, "lib_hyperbolic")
    # 88, 89 are inside binary32's exp-overflow band [88.7, 89.4]: sinh/cosh are ~2e38 (representable), so folding the
    # /2 into the exponent must keep them finite instead of overflowing exp before the halving.
    for x in (0.0, 0.5, -0.5, 2.0, -2.0, 5.0, -5.0, 88.0, 89.0, -89.0):
        out = sim.run(x)
        assert float(out[0]) == pytest.approx(math.sinh(x), rel=1e-5, abs=1e-5), f"sinh({x})"
        assert float(out[1]) == pytest.approx(math.cosh(x), rel=1e-5), f"cosh({x})"
        assert float(out[2]) == pytest.approx(math.tanh(x), rel=1e-5, abs=1e-5), f"tanh({x})"


@pytest.mark.skip(reason="FIR_PARITY_PENDING: aggregate (tuple) returns — stage 9 (aggregate returns/np.array)")
def test_inverse_hyperbolic_match_references() -> None:
    # Independent inputs: asinh spans all reals, acosh needs w>=1, atanh needs |u|<1.
    def kernel(x: float, w: float, u: float) -> tuple[float, float, float]:
        return (math.asinh(x), math.acosh(w), math.atanh(u))

    sim = _sim(kernel, "lib_inverse_hyperbolic")
    # The large-|x| cases (>= 1e19) are the regression: binary32 x*x overflows to inf there, so without the branch
    # asinh/acosh returned inf across the whole band up to FLT_MAX, where the true values (~44..89) are representable.
    cases = [(0.0, 1.0, 0.0), (2.0, 1.5, 0.5), (-2.0, 2.0, -0.5), (100.0, 100.0, 0.9), (-100.0, 4.0, -0.9)]
    cases += [(1e19, 1e19, 0.5), (1e30, 1e38, -0.5), (-1e30, 3e38, 0.9)]
    for x, w, u in cases:
        out = sim.run(x, w, u)
        assert float(out[0]) == pytest.approx(math.asinh(x), rel=1e-5, abs=1e-4), f"asinh({x})"
        assert float(out[1]) == pytest.approx(math.acosh(w), rel=1e-5, abs=1e-4), f"acosh({w})"
        assert float(out[2]) == pytest.approx(math.atanh(u), rel=1e-5, abs=1e-5), f"atanh({u})"


@pytest.mark.skip(reason="FIR_PARITY_PENDING: aggregate (tuple) returns — stage 9 (aggregate returns/np.array)")
def test_expm1_log1p_degrees_radians_match_references() -> None:
    def kernel(x: float) -> tuple[float, float, float, float]:
        return (math.expm1(x), math.log1p(x * x), math.degrees(x), math.radians(x))

    sim = _sim(kernel, "lib_expm1_log1p_angles")
    for x in (0.0, 0.5, -0.5, 2.0, -2.0):
        out = sim.run(x)
        assert float(out[0]) == pytest.approx(math.expm1(x), rel=1e-5, abs=1e-5), f"expm1({x})"
        assert float(out[1]) == pytest.approx(math.log1p(x * x), rel=1e-5, abs=1e-5), f"log1p({x * x})"
        assert float(out[2]) == pytest.approx(math.degrees(x), rel=1e-5, abs=1e-5), f"degrees({x})"
        assert float(out[3]) == pytest.approx(math.radians(x), rel=1e-5, abs=1e-5), f"radians({x})"


@pytest.mark.skip(reason="FIR_PARITY_PENDING: aggregate (tuple) returns — stage 9 (aggregate returns/np.array)")
def test_new_composite_and_binary_numpy_spellings() -> None:
    def kernel(x: float, y: float) -> tuple[float, float, float, float, float]:
        return (np.sinh(x), np.arcsinh(x), np.fmin(x, y), np.fmax(x, y), np.fix(x))  # type: ignore[return-value]

    sim = _sim(kernel, "lib_new_spellings")
    for x, y in ((1.5, -2.0), (-0.5, 3.0)):
        out = sim.run(x, y)
        assert float(out[0]) == pytest.approx(math.sinh(x), rel=1e-5, abs=1e-5)
        assert float(out[1]) == pytest.approx(math.asinh(x), rel=1e-5, abs=1e-5)
        assert float(out[2]) == min(x, y) and float(out[3]) == max(x, y)
        assert float(out[4]) == math.trunc(x)
