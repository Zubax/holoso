"""
The rewrites that share work rather than rewriting one expression in place.

Op counts are asserted on MIR rather than on the emitted Verilog because a pooled operator instantiates one module
per class however many times it fires, so the RTL cannot tell six divisions from four. Values are asserted against
exact references wherever a rule changes the rounding, and against the un-rewritten spelling wherever it must not.
"""

import collections
import math
from collections.abc import Callable

import numpy as np
import pytest

import holoso
from holoso import (
    FAddOptions,
    FDivOptions,
    FFmaOptions,
    FloatFormat,
    FloatValue,
    FMulILog2Options,
    FMulOptions,
    OperatorOptions,
    Options,
)
from holoso._eel import lower
from holoso._mir import MirOperation
from holoso._mir import lower as lower_to_mir

from ._modelref import DEFAULT_UNROLL_MAX_TRIPS, mir_options

FMT = FloatFormat(8, 18)


def _options(*, fma: bool = True, fmt: FloatFormat = FMT) -> Options:
    return Options(
        OperatorOptions(
            fadd=FAddOptions(),
            fmul=FMulOptions(),
            fdiv=FDivOptions(),
            fmul_ilog2=FMulILog2Options(),
            ffma=FFmaOptions() if fma else None,
        ),
        ffmt=fmt,
    )


def _mnemonics(fn: Callable[..., object], options: Options) -> collections.Counter[str]:
    hir = lower(fn, DEFAULT_UNROLL_MAX_TRIPS).hir
    mir = lower_to_mir(hir, mir_options(options))
    counts: collections.Counter[str] = collections.Counter()
    for block in mir.blocks:
        for vid in block.operations:
            node = mir.nodes[vid]
            assert isinstance(node, MirOperation)
            counts[node.operator.mnemonic] += 1
    return counts


def _run(fn: Callable[..., object], options: Options, name: str, *args: float) -> tuple[float, ...]:
    model = holoso.synthesize(fn, options, name=name).numerical_model.elaborate()
    return tuple(float(v) for v in model.run(*args))


def _scaled_sum(a: float, c: float) -> float:
    return a * 8.0 + c


def test_exponent_scaling_contracts_into_an_fma() -> None:
    # A scaler that forfeits the fusion is not the cheaper operator. The ffma-less control is what pins the
    # contraction as the reason the count moved.
    fused = _mnemonics(_scaled_sum, _options(fma=True))
    assert fused["ffma"] == 1 and fused["fmul_ilog2"] == 0 and fused["fadd"] == 0
    separate = _mnemonics(_scaled_sum, _options(fma=False))
    assert separate["fmul_ilog2"] == 1 and separate["fadd"] == 1


def test_exponent_contraction_declines_where_the_format_rounds_the_scale() -> None:
    # FloatFormat(3, 4) encodes 0.125 as 0.25 -- finite and nonzero, so a degradation test would admit it, and the
    # contraction would multiply by twice the scale the scaler applies exactly.
    def kernel(a: float, c: float) -> float:
        return a * 0.125 + c

    narrow = _options(fma=True, fmt=FloatFormat(3, 4))
    assert _mnemonics(kernel, narrow)["fmul_ilog2"] == 1, "the scaler must stand where 2**k is not exact"
    assert _mnemonics(kernel, narrow)["ffma"] == 0


def test_exponent_contraction_declines_past_the_host_range() -> None:
    # A composed exponent is unbounded by design, while `2.0**k` rails past k=1023 in the compiler's own arithmetic.
    def kernel(a: float, c: float) -> float:
        return (a * 2.0**1000) * 2.0**1000 + c

    counts = _mnemonics(kernel, _options(fma=True))
    assert counts["fmul_ilog2"] == 1 and counts["fadd"] == 1


def _twice_read_product(a: float, b: float, c: float, d: float) -> tuple[float, float]:
    p = a * b
    return p + c, p + d


def test_product_read_only_by_additions_is_carried_by_all_of_them() -> None:
    # Nothing observes the rounding that contracting removes, and each fma keeps its own product.
    counts = _mnemonics(_twice_read_product, _options(fma=True))
    assert counts["ffma"] == 2 and counts["fmul"] == 0 and counts["fadd"] == 0
    rng = np.random.default_rng(0xA1)
    fused = holoso.synthesize(_twice_read_product, _options(fma=True), name="absorbed").numerical_model.elaborate()
    for _ in range(200):
        a, b, c, d = (float(np.float32(rng.standard_normal() * 7)) for _ in range(4))
        va, vb = FloatValue.from_float(FMT, a), FloatValue.from_float(FMT, b)
        expected = tuple(FloatValue.fma(va, vb, FloatValue.from_float(FMT, addend)).bits for addend in (c, d))
        got = tuple(v for v in fused.run(a, b, c, d))
        assert all(isinstance(v, FloatValue) for v in got)
        assert tuple(v.bits for v in got if isinstance(v, FloatValue)) == expected, f"a={a} b={b} c={c} d={d}"


def test_product_an_addition_reads_twice_is_not_absorbed() -> None:
    # One fma carries one product, so a sum naming it on both sides absorbs only one of its two uses. Suppressing
    # it on a claim count blind to the multiplicity would leave the addend without a value.
    def kernel(a: float, b: float) -> float:
        p = a * b
        return p + p

    counts = _mnemonics(kernel, _options(fma=True))
    assert counts["fmul"] == 1 and counts["fadd"] == 1 and counts["ffma"] == 0


def test_product_reached_through_a_shared_sign_is_not_absorbed() -> None:
    # A sign op is never lowered, so a second reader of one would ask for a base the contraction had suppressed.
    def kernel(a: float, b: float, c: float, d: float) -> tuple[float, float]:
        n = -(a * b)
        return n + c, n + d

    counts = _mnemonics(kernel, _options(fma=True))
    assert counts["fmul"] == 1 and counts["fadd"] == 2 and counts["ffma"] == 0


def test_product_an_output_also_reads_is_not_absorbed() -> None:
    # The rounded product is observed elsewhere, so the sum keeps its own rounding.
    def kernel(a: float, b: float, c: float) -> tuple[float, float]:
        p = a * b
        return p + c, p

    counts = _mnemonics(kernel, _options(fma=True))
    assert counts["fmul"] == 1 and counts["fadd"] == 1 and counts["ffma"] == 0


def _two_divides(x: float, y: float) -> tuple[float, float]:
    return 1.0 / y, x / y


def test_a_division_is_answered_from_a_reciprocal_already_computed() -> None:
    counts = _mnemonics(_two_divides, _options(fma=False))
    assert counts["fdiv"] == 1 and counts["fmul"] == 1
    for x, y in ((3.0, 4.0), (-1.5, 0.25), (7.0, -2.0)):
        recip, quotient = _run(_two_divides, _options(fma=False), "shared_recip", x, y)
        assert recip == 1.0 / y
        assert quotient == x * recip, "the quotient is the product with the shared reciprocal, not a second divide"


def test_a_reciprocal_of_a_product_uses_the_reciprocals_of_its_factors() -> None:
    def kernel(x: float, m: float) -> tuple[float, float, float]:
        return 1.0 / m, 1.0 / (m * m), x / (m * m)

    counts = _mnemonics(kernel, _options(fma=False))
    assert counts["fdiv"] == 1 and counts["fmul"] == 2


def test_reciprocal_of_a_product_parts_company_at_zero_times_infinity() -> None:
    # The license this rewrite needs, pinned: `1/(p*q)` is `1/0`, hence an infinity, where `(1/p)*(1/q)` is
    # `inf*0`, hence zero. The rewritten kernel answers the latter.
    def kernel(p: float, q: float) -> tuple[float, float, float]:
        return 1.0 / p, 1.0 / q, 1.0 / (p * q)

    _, _, joint = _run(kernel, _options(fma=False), "recip_zero_inf", 0.0, math.inf)
    assert joint == 0.0


def test_divisions_without_a_reciprocal_are_left_alone() -> None:
    # Introducing a reciprocal here would add an operation, so two divisions stay two divisions.
    def kernel(a: float, b: float, y: float) -> tuple[float, float]:
        return a / y, b / y

    assert _mnemonics(kernel, _options(fma=False))["fdiv"] == 2


def test_a_reciprocal_the_round_would_delete_is_not_adopted() -> None:
    # `z` folds to zero, which kills the reciprocal -- but only in a later round. Adopting it on the strength of a
    # liveness that has not settled would trade one division for a division and a multiply.
    def kernel(x: float, y: float) -> float:
        r = 1.0 / y
        z = 0.0 * r
        return x / y + z

    counts = _mnemonics(kernel, _options(fma=False))
    assert counts["fdiv"] == 1 and counts["fmul"] == 0


def _same_form_twice(a: float, b: float) -> tuple[float, float]:
    return 2.0 * a + 3.0 * b, 4.0 * a + 6.0 * b


def _unrelated_forms(a: float, b: float) -> tuple[float, float]:
    return 2.0 * a + 3.0 * b, 4.0 * a + 7.0 * b


def test_a_proportional_sum_is_answered_as_a_scaling_of_the_first() -> None:
    # The baseline differs only in a coefficient that breaks the proportion, which is what makes the count
    # attributable to the sharing.
    shared = _mnemonics(_same_form_twice, _options(fma=False))
    baseline = _mnemonics(_unrelated_forms, _options(fma=False))
    assert sum(shared.values()) < sum(baseline.values())
    assert shared["fadd"] == 1 and baseline["fadd"] == 2
    for a, b in ((1.5, 2.25), (-3.0, 0.5), (0.0, 7.0)):
        first, second = _run(_same_form_twice, _options(fma=False), "proportional", a, b)
        assert (first, second) == (2.0 * a + 3.0 * b, 2.0 * (2.0 * a + 3.0 * b))


def test_a_proportional_sum_with_an_inexact_ratio_takes_the_rounded_ratio() -> None:
    # The ratio need not be dyadic: the rounded third is the license this rewrite carries, not an accident of it.
    def kernel(a: float, b: float) -> tuple[float, float]:
        return 3.0 * a + 9.0 * b, a + 3.0 * b

    assert _mnemonics(kernel, _options(fma=False))["fadd"] == 1
    for a, b in ((1.5, 2.25), (-3.0, 0.5)):
        keeper, derived = _run(kernel, _options(fma=False), "inexact_ratio", a, b)
        assert derived == keeper * (1.0 / 3.0)


def _no_node_retires(a: float, b: float) -> tuple[float, float, float, float]:
    t = 3.0 * a
    u = 3.0 * b
    return t + u, a + b, t, u


def test_a_proportional_sum_that_retires_nothing_is_left_alone() -> None:
    # Both scalings are read elsewhere, so answering one sum from the other retires nothing and costs a multiply.
    counts = _mnemonics(_no_node_retires, _options(fma=False))
    assert counts["fadd"] == 2 and counts["fmul"] == 2


def test_an_infinite_constant_leaves_a_sum_opaque() -> None:
    # An infinity is a value here and no rational names it, so a sum holding one is shared with nothing.
    inf = math.inf

    def kernel(x: float) -> tuple[float, float]:
        return x + inf, 2.0 * x + 2.0 * inf

    assert _mnemonics(kernel, _options(fma=False))["fadd"] == 2


def _equal_scales(x: float, y: float) -> float:
    return 3.0 * x + 3.0 * y


def _opposite_scales(x: float, y: float) -> float:
    return 3.0 * x + (-3.0) * y


def _differing_exponents(x: float, y: float) -> float:
    return 3.0 * x + 6.0 * y


def _pow2_through_a_negation(x: float, y: float) -> float:
    return 2.0 * x + (-2.0) * y


@pytest.mark.parametrize(
    "kernel,expected",
    [
        (_equal_scales, {"fmul": 1, "fadd": 1}),
        (_opposite_scales, {"fmul": 1, "fadd": 1}),
        (_pow2_through_a_negation, {"fmul_ilog2": 1, "fadd": 1}),
    ],
    ids=["equal", "opposite-sign", "power-of-two-through-a-negation"],
)
def test_a_common_significand_is_factored_out_of_a_sum(
    kernel: Callable[[float, float], float], expected: dict[str, int]
) -> None:
    counts = _mnemonics(kernel, _options(fma=False))
    for mnemonic, count in expected.items():
        assert counts[mnemonic] == count, f"{mnemonic}: {counts}"


def test_factoring_declines_where_it_removes_nothing() -> None:
    # Carrying an exponent step inside retires nothing: it swaps a multiply for a scaler and moves the multiply
    # behind the addition. `3x + 6y` is the shape that tells the two apart -- factored it would read
    # fmul + fmul_ilog2 + fadd.
    assert _mnemonics(_differing_exponents, _options(fma=False)) == collections.Counter({"fmul": 2, "fadd": 1})


def test_factoring_declines_where_the_scalings_are_read_elsewhere() -> None:
    # Both negations are single-use but the scalers beneath them are not, so factoring would add an operation.
    def kernel(x: float, y: float) -> tuple[float, float, float]:
        t = 2.0 * x
        s = 2.0 * y
        return -t + -s, t, s

    assert _mnemonics(kernel, _options(fma=False))["fmul_ilog2"] == 2


# Defects the review loop found: each of these crashed or answered differently before its fix.


def test_factoring_leaves_an_addend_the_fold_already_answered() -> None:
    # `known * 0.52` has both operands known, so constant evaluation settles it and no identity may reach past it.
    # Factoring the common scale out would recompute it as `(0.52 + y) * 0.52`.
    narrow = FloatFormat(3, 4)

    def kernel(x: float, y: float) -> float:
        known = x * 0.0 + 0.52
        return known * 0.52 + y * 0.52

    options = Options(OperatorOptions(fadd=FAddOptions(), fmul=FMulOptions()), ffmt=narrow)
    model = holoso.synthesize(kernel, options, name="folded_addend").numerical_model.elaborate()
    (got,) = model.run(3.0, 0.0)
    assert isinstance(got, FloatValue)
    assert got.bits == FloatValue.from_float(narrow, 0.52 * 0.52).bits


def test_a_deeply_nested_product_divisor_builds_its_reciprocal() -> None:
    # Expanding a nested divisor's reciprocal by recursion exceeds the interpreter's limit well inside the
    # unrolling budget, on a kernel shape that is entirely ordinary.
    def kernel(x: float, y: float) -> float:
        product = x
        total = 1.0 / x + 1.0 / y
        for _ in range(990):
            product = product * y
            total = total + 1.0 / product
        return total

    model = holoso.synthesize(kernel, _options(fma=False), name="deep_divisor").numerical_model.elaborate()
    assert float(model.run(1.0, 1.0)[0]) == 992.0


def test_sharing_does_not_adopt_a_reciprocal_the_other_sharing_pass_strands() -> None:
    # Linear sharing retires `same`, the reciprocal's only reader. Running both passes over one settled graph would
    # let `x/y` adopt the reciprocal first and strand it a moment later -- here, a demand for an unconfigured
    # operator.
    def kernel(a: float, b: float, x: float, y: float) -> tuple[float, float, float]:
        reciprocal = 1.0 / y
        keeper = a + b
        same = (a + reciprocal) + (b - reciprocal)
        return keeper, same, x / y

    options = Options(OperatorOptions(fadd=FAddOptions(), fdiv=FDivOptions()), ffmt=FMT)
    model = holoso.synthesize(kernel, options, name="sharing_order").numerical_model.elaborate()
    assert tuple(float(v) for v in model.run(2.0, 3.0, 6.0, 3.0)) == (5.0, 5.0, 2.0)


def test_a_sum_answered_from_a_sum_it_contains_is_left_alone() -> None:
    # The inner `x+x` is the keeper the rewrite would READ, so it is not retired by it, and nothing else is.
    def kernel(x: float) -> float:
        return x + x + x

    assert _mnemonics(kernel, _options(fma=False)) == collections.Counter({"fadd": 2})
    assert _run(kernel, _options(fma=False), "thrice", 1.5) == (4.5,)


def test_a_sign_operand_is_not_credited_as_retired() -> None:
    # A negation is a sideband, not an operator, so dropping its last use retires nothing. Counting it would buy
    # one adder for a multiplier and an inexact third -- the trade the rule refuses.
    def kernel(a: float, b: float) -> tuple[float, float, float, float]:
        t = 3.0 * a
        s = 5.0 * b
        return -9.0 * a + 15.0 * b, -t + s, t, s

    counts = _mnemonics(kernel, _options(fma=False))
    assert counts["fmul"] == 4 and counts["fadd"] == 2
