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
    FSortOperator,
    OpConfig,
    UnsupportedConstruct,
)

# Bare-name imports so a ``from math import floor`` style kernel resolves through the test module globals.
from math import ceil, floor, log2, trunc

# Aliased imports: the local name is NOT the canonical spelling, so dispatch must resolve by callee-object identity.
from math import floor as aliased_floor
from math import fma as aliased_fma

FMT = FloatFormat(8, 24)  # binary32: a float64 decode of any in-format value is exact, so math/round is an exact oracle


def _ops(
    *,
    with_round: bool = True,
    with_fma: bool = True,
    with_sort: bool = True,
    with_exp2: bool = True,
    with_log2: bool = True,
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


def test_round_modes_match_reference() -> None:
    def kernel(x: float) -> tuple[float, float, float, float]:
        return (math.floor(x), math.ceil(x), math.trunc(x), round(x))

    sim = _sim(kernel, "round_all_modes")
    modes_by_output = (1, 2, 3, 0)  # floor, ceil, trunc, round
    for value in _ROUND_VECTORS:
        out = sim.run(value)
        for index, mode in enumerate(modes_by_output):
            assert _bits(out[index]) == _round_ref(value, mode), f"value={value} output={index} mode={mode}"


def test_round_dispatch_numpy_and_bare_name() -> None:
    # numpy.<name> under an alias, and bare names imported via ``from math import ...`` must both dispatch.
    def kernel(x: float) -> tuple[float, float, float, float, float, float]:
        return (np.floor(x), np.ceil(x), np.trunc(x), floor(x), ceil(x), trunc(x))

    sim = _sim(kernel, "round_dispatch")
    for value in _ROUND_VECTORS:
        out = sim.run(value)
        for index, mode in enumerate((1, 2, 3, 1, 2, 3)):
            assert _bits(out[index]) == _round_ref(value, mode), f"value={value} output={index}"


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


def test_min_max_nonfinite_constant_is_rejected() -> None:
    # A non-finite constant operand must not be hidden by the fold's selection: even when the finite side would be
    # selected, the non-finite constant must reach the validator and be rejected, as a bare non-finite literal is.
    def min_selects_finite(x: float) -> float:
        return x + min(1e400, 2.0)  # 1e400 overflows to +inf; the fold would select 2.0 but must not drop the inf

    def max_selects_finite(x: float) -> float:
        return x + max(2.0, 1e400 - 1e400)  # 1e400 - 1e400 is NaN; must be rejected, not selected away

    for fn in (min_selects_finite, max_selects_finite):
        with pytest.raises(UnsupportedConstruct):
            holoso.synthesize(fn, _ops(), name=fn.__name__)


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


def test_exp2_log2_dispatch_numpy_and_bare_name() -> None:
    def kernel(x: float) -> tuple[float, float, float]:
        return (np.exp2(x), np.log2(x), log2(x))

    sim = _sim(kernel, "exp2_log2_dispatch")
    for x in (2.0, 3.0, 100.0):  # positive; dispatch resolves by callee identity, so it is value-independent
        out = sim.run(x)
        assert _bits(out[0]) == _v(x).exp2().bits, f"np.exp2 x={x}"
        assert _bits(out[1]) == _v(x).log2().bits, f"np.log2 x={x}"
        assert _bits(out[2]) == _v(x).log2().bits, f"bare log2 x={x}"


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
