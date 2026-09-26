"""
The rewrites that share work rather than rewriting one expression in place.

Op counts are asserted on MIR rather than on the emitted Verilog because a pooled operator instantiates one module
per class however many times it fires, so the RTL cannot tell six divisions from four. Values are asserted against
exact references wherever a rule changes the rounding, and against the un-rewritten spelling wherever it must not.
"""

import collections
import dataclasses
import math
from collections.abc import Callable

import numpy as np
import pytest

import holoso
from holoso import (
    FAddOptions,
    FCmpOptions,
    FDivOptions,
    FFmaOptions,
    FloatFormat,
    FloatValue,
    FMulILog2Options,
    FMulOptions,
    OperatorOptions,
    Options,
    UnsupportedConstruct,
)
from holoso._eel import lower
from holoso._mir import MirConst, MirOperation
from holoso._mir import lower as lower_to_mir

from ._modelref import (
    DEFAULT_UNROLL_MAX_TRIPS,
    build_lir,
    default_tolerance,
    mir_options,
    within,
)

FMT = FloatFormat(8, 18)


def _options(*, fma: bool = True, fmt: FloatFormat = FMT) -> Options:
    return Options(
        OperatorOptions(
            fcmp=FCmpOptions(),
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


def _schedule(fn: Callable[..., object], options: Options, name: str) -> tuple[int, int]:
    """The shipped program's schedule."""
    lir = build_lir(lower_to_mir(lower(fn, DEFAULT_UNROLL_MAX_TRIPS).hir, mir_options(options)), name)
    return lir.min_initiation_interval, lir.last_pc


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


def _composed_product_first(a: float, c: float, d: float) -> float:
    return (a * 2.0) * 3.0 + c * d


def _composed_product_last(a: float, c: float, d: float) -> float:
    return c * d + (a * 2.0) * 3.0


def _standing_multiply_reads_a_constant(fn: Callable[..., object]) -> bool:
    hir = lower(fn, DEFAULT_UNROLL_MAX_TRIPS).hir
    mir = lower_to_mir(hir, mir_options(_options(fma=True)))
    (multiply,) = [
        node for node in mir.nodes.values() if isinstance(node, MirOperation) and node.operator.mnemonic == "fmul"
    ]
    return any(isinstance(mir.nodes[operand], MirConst) for operand in multiply.operands)


def test_an_add_offered_two_products_contracts_the_one_computed_first() -> None:
    # The graph's own order of computation decides which product the fma rounds exactly, not the value's name.
    assert not _standing_multiply_reads_a_constant(_composed_product_first)
    assert _standing_multiply_reads_a_constant(_composed_product_last)


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
    # One fma carries one product, so a sum naming it twice absorbs only one use; the absolute value keeps this a sum.
    def kernel(a: float, b: float) -> float:
        p = a * b
        return p + abs(p)

    counts = _mnemonics(kernel, _options(fma=True))
    assert counts["fmul"] == 1 and counts["fadd"] == 1 and counts["ffma"] == 0


def test_product_read_through_a_shared_sign_is_carried_by_every_addition() -> None:
    # A negation two additions share goes with them exactly as the product does.
    def kernel(x: float, y: float, z: float) -> tuple[float, float]:
        p = x * -3.0
        return p + y, p + z

    assert _mnemonics(kernel, _options(fma=True)) == collections.Counter({"ffma": 2})
    assert _schedule(kernel, _options(fma=True), "shared_sign_fma") == (9, 9)
    assert _run(kernel, _options(fma=True), "shared_sign_fma", 2.0, 1.0, 5.0) == (-5.0, -1.0)


def test_product_reached_through_a_sign_something_else_reads_is_not_absorbed() -> None:
    # The negated product is an output too, so the product is observed and every addition keeps its own rounding.
    def kernel(a: float, b: float, c: float, d: float) -> tuple[float, float, float]:
        n = -(a * b)
        return n + c, n + d, n

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


def test_the_sharing_passes_do_not_strand_a_reciprocal_between_them() -> None:
    # The collapse answers `same` as `a`, retiring the reciprocal's only reader. Running both passes over one settled
    # graph would let `x/y` adopt the reciprocal first and strand it a moment later, demanding an absent multiplier.
    def kernel(a: float, x: float, y: float) -> tuple[float, float]:
        reciprocal = 1.0 / y
        same = (a + reciprocal) - reciprocal
        return same, x / y

    options = Options(OperatorOptions(fadd=FAddOptions(), fdiv=FDivOptions()), ffmt=FMT)
    model = holoso.synthesize(kernel, options, name="sharing_order").numerical_model.elaborate()
    assert tuple(float(v) for v in model.run(2.0, 6.0, 3.0)) == (2.0, 2.0)


def test_a_reciprocal_the_round_would_delete_is_not_adopted() -> None:
    # `z` folds to zero, which kills the reciprocal -- but only in a later round. Adopting it on the strength of a
    # liveness that has not settled would trade one division for a division and a multiply.
    def kernel(x: float, y: float) -> float:
        r = 1.0 / y
        z = 0.0 * r
        return x / y + z

    counts = _mnemonics(kernel, _options(fma=False))
    assert counts["fdiv"] == 1 and counts["fmul"] == 0


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
    # behind the addition. `3x + 6y` tells the two apart: factored it would read fmul + fmul_ilog2 + fadd.
    assert _mnemonics(_differing_exponents, _options(fma=False)) == collections.Counter({"fmul": 2, "fadd": 1})


def test_factoring_declines_where_the_scalings_are_read_elsewhere() -> None:
    # Both negations are single-use but the scalers beneath them are not, so factoring would add an operation.
    def kernel(x: float, y: float) -> tuple[float, float, float]:
        t = 2.0 * x
        s = 2.0 * y
        return -t + -s, t, s

    assert _mnemonics(kernel, _options(fma=False))["fmul_ilog2"] == 2


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
    # unrolling budget, on a kernel shape that is entirely ordinary. The allocation's quality is beside the point.
    def kernel(x: float, y: float) -> float:
        product = x
        total = 1.0 / x + 1.0 / y
        for _ in range(990):
            product = product * y
            total = total + 1.0 / product
        return total

    options = dataclasses.replace(_options(fma=False), regalloc_effort=0)
    model = holoso.synthesize(kernel, options, name="deep_divisor").numerical_model.elaborate()
    assert float(model.run(1.0, 1.0)[0]) == 992.0


def test_a_sum_of_one_value_is_a_scaling_of_it() -> None:
    # The single-term form `{x: 3}` is one multiply rather than two additions -- the general case of `x + x -> 2x`.
    def kernel(x: float) -> float:
        return x + x + x

    assert _mnemonics(kernel, _options(fma=False)) == collections.Counter({"fmul": 1})
    assert _run(kernel, _options(fma=False), "thrice", 1.5) == (4.5,)


def test_a_cancelling_difference_is_answered_by_its_remainder() -> None:
    # The difference keeps none of its operands' digits; the form knows the remainder exactly.
    def kernel(x: float) -> float:
        return x - 0.999 * x

    assert _mnemonics(kernel, _options(fma=False)) == collections.Counter({"fmul": 1})
    got = _run(kernel, _options(fma=False), "cancelling", 1.0)[0]
    assert abs(got - 0.001) < 0.001 * 2.0**-16, f"{got} is nowhere near the exact remainder"


def test_a_sum_that_cancels_entirely_needs_no_operation() -> None:
    def restored(x: float, y: float) -> float:
        return (x + y) - y

    def offset(x: float) -> float:
        return (x + 1.0) - x

    def erased(x: float, y: float) -> float:
        # Every term cancels; the float64 oracle's -1.0 at these magnitudes is its own rounding.
        return (x + y) - x - y

    assert _mnemonics(restored, _options(fma=False)) == collections.Counter()
    assert _mnemonics(offset, _options(fma=False)) == collections.Counter()
    assert _mnemonics(erased, _options(fma=False)) == collections.Counter()
    assert _run(restored, _options(fma=False), "restored", 3.25, 1e9) == (3.25,)
    assert _run(offset, _options(fma=False), "offset", 1e9) == (1.0,)
    assert _run(erased, _options(fma=False), "erased", 1e20, 1.0) == (0.0,)


def test_a_form_the_pass_gave_up_on_is_not_read_as_a_scaling_of_itself() -> None:
    # An oversized form reads as `1 * itself`; answering it would multiply by a value defined nowhere.
    def kernel(x: float, y: float) -> float:
        p = x
        total = x
        for _ in range(70):  # past the term limit, so the form gives up and stands for itself
            p = p * y
            total = total + p
        return total

    model = holoso.synthesize(kernel, _options(fma=False), name="opaque_form").numerical_model.elaborate()
    assert float(model.run(1.0, 1.0)[0]) == 71.0


def _relations_over_one_pair(x: float) -> tuple[bool, bool]:
    return 3.0 == x, 3.0 < x  # written constant-first, both of them


def _relations_in_mixed_spelling(x: float) -> tuple[bool, bool]:
    return x == 3.0, 3.0 < x


def test_relations_over_one_pair_keep_the_pair() -> None:
    # Settling the constant on the right mirrors the ordering relation too, so `3.0 < x` and `x > 3.0` fire once.
    options = _options(fma=False)
    assert _schedule(_relations_over_one_pair, options, "relations")[0] == 4
    assert _schedule(_relations_in_mixed_spelling, options, "mixed_relations")[0] == 4


def _mixed_spelling(x: float, z: float) -> tuple[float, float]:
    return 5.0 * z, (x + 5.0 * z) - x


def test_a_product_the_source_wrote_and_one_an_answer_mints_are_one_node() -> None:
    # Without a canonical operand order the answer's product and the written `5.0 * z` would be two nodes.
    assert _mnemonics(_mixed_spelling, _options(fma=False)) == collections.Counter({"fmul": 1})
    assert _run(_mixed_spelling, _options(fma=False), "mixed_spelling", 4.0, 3.0) == (15.0, 15.0)


def _answered_as_a_number(x: float, y: float) -> tuple[float, float]:
    base = 3.0 * x + 5.0 * y
    doubled = 6.0 * x + 10.0 * y
    return doubled - 2.0 * base, 9.0 * x + 15.0 * y  # the first cancels to zero, the second to a scaling


def test_a_sum_answered_as_a_number_reads_nothing() -> None:
    # A number is computed from nothing, so the sums the cancelled one read die with it.
    assert _mnemonics(_answered_as_a_number, _options(fma=False)) == collections.Counter({"fmul": 2, "fadd": 1})
    assert _run(_answered_as_a_number, _options(fma=False), "answered_number", 4.0, 3.0) == (0.0, 81.0)


def _cancels_around_another_sum(x: float, y: float, z: float) -> tuple[float, float]:
    k = 3.0 * x + 6.0 * y
    s2 = 4.0 * x + 8.0 * y
    s1 = (k + z) - k  # cancels to `z`, taking `k`'s only reader with it
    return s1, s2


def test_a_sum_whose_terms_cancel_is_answered_wherever_it_stands() -> None:
    # Two collapses in one block, the second reading a value the first retires.
    assert _mnemonics(_cancels_around_another_sum, _options(fma=False)) == collections.Counter(
        {"fmul_ilog2": 2, "fadd": 1}
    )
    assert _run(_cancels_around_another_sum, _options(fma=False), "cancels_around", 4.0, 3.0, 2.0) == (2.0, 40.0)


def _through_a_free_answer(x: float, y: float, z: float) -> tuple[float, float, float]:
    freed = (x + z) - x  # answers `z` itself, so nothing is left to retire
    survivor = 5.0 * y  # read twice, so it cannot be what pays
    keeper = 3.0 * z + 15.0 * y
    adopter = freed + survivor  # would adopt `keeper` at 1/3, crediting a value that costs nothing
    return keeper, adopter, survivor


def test_an_answer_reads_the_value_another_answer_left() -> None:
    # A collapse whose operand is itself an answered sum reads the answer, not the addition it replaced.
    assert _mnemonics(_through_a_free_answer, _options(fma=False)) == collections.Counter({"fmul": 3, "fadd": 2})
    assert _run(_through_a_free_answer, _options(fma=False), "through_free", 2.0, 3.0, 4.0) == (57.0, 19.0, 15.0)


def _summed_through_a_merge(x: float, y: float, c: bool) -> float:
    if c:
        r = (x + y) - y  # answered as `x`, so the arm's addition goes and the merge takes `x` directly
    else:
        r = y
    return r


def test_a_merge_carries_the_arm_it_reads() -> None:
    # Liveness reaches through a merge to what an answered arm reads, or the arm's addition would stand.
    options = dataclasses.replace(_options(fma=False), ifconv_max_ops=0)  # keep the branch, so the merge is a phi
    assert _mnemonics(_summed_through_a_merge, options) == collections.Counter({"select": 1})
    model = holoso.synthesize(_summed_through_a_merge, options, name="merged_arm").numerical_model.elaborate()
    for c in (True, False):
        assert float(model.run(2.0, 3.0, c)[0]) == _summed_through_a_merge(2.0, 3.0, c), f"c={c}"


def _summed_once_the_merge_folds(x: float, c: bool) -> float:
    s = x + 1e-20 * x
    if c:
        t = s
    else:
        t = x + 1e-20 * x
    return t - x


def test_a_sum_is_answered_over_the_settled_graph() -> None:
    # The arms intern into one only after if-conversion and the merge folds only in the round after that. An answer
    # taken before it folds rounds `s` to `x` alone, and the difference then cancels to nothing where the settled sum
    # still carries `1e-20 * x`.
    counts = _mnemonics(_summed_once_the_merge_folds, _options(fma=False))
    assert counts["fmul"] == 1 and counts["fadd"] == 0
    for c in (True, False):
        (out,) = _run(_summed_once_the_merge_folds, _options(fma=False), "settled_sum", 3.0, c)
        assert out == pytest.approx(3e-20, rel=1e-4), f"c={c}"


def test_a_sign_operand_is_not_credited_as_retired() -> None:
    # A negation is a sideband, not an operator, so dropping its last use retires nothing. Counting it would buy
    # one adder for a multiplier and an inexact third -- the trade the rule refuses.
    def kernel(a: float, b: float) -> tuple[float, float, float, float]:
        t = 3.0 * a
        s = 5.0 * b
        return -9.0 * a + 15.0 * b, -t + s, t, s

    counts = _mnemonics(kernel, _options(fma=False))
    assert counts["fmul"] == 4 and counts["fadd"] == 2


_WIDE = Options(
    OperatorOptions(fadd=FAddOptions(), fmul=FMulOptions(), fmul_ilog2=FMulILog2Options()),
    ffmt=FloatFormat(12, 36),
)
"""
A format whose exponent span far outreaches the host's, so a coefficient the host cannot name is one the
DATAPATH still holds -- which is what makes declining the collapse the correct answer rather than a refusal.
"""


def test_a_collapse_declines_a_coefficient_the_host_cannot_name() -> None:
    # `3 * 2**-1100` underflows `float(Fraction)` to zero; `b + b` beside it is an exact power of two.
    def kernel(x: float) -> float:
        a = x * 2.0**-1000
        b = a * 2.0**-100
        return b + b + b

    assert _mnemonics(kernel, _WIDE) == collections.Counter({"fmul_ilog2": 2, "fadd": 1})
    model = holoso.synthesize(kernel, _WIDE, name="underflowed_collapse").numerical_model.elaborate()
    assert float(model.run(2.0**1000)[0]) == kernel(2.0**1000)


def test_a_collapse_declines_a_subnormal_coefficient() -> None:
    # `3 * 2**-1059` survives `float(Fraction)` as a subnormal, which names no scaling.
    def kernel(x: float) -> float:
        a = x * 2.0**-1000
        b = a * 2.0**-59
        return b + b + b

    assert _mnemonics(kernel, _WIDE) == collections.Counter({"fmul_ilog2": 2, "fadd": 1})
    model = holoso.synthesize(kernel, _WIDE, name="subnormal_collapse").numerical_model.elaborate()
    assert float(model.run(2.0**1000)[0]) == 3.0 * 2.0**-59


def test_a_power_of_two_answer_needs_no_host_float() -> None:
    # `2**1201` names no host float but an exponent needs none, so the sum is one scaler emitted in its final shape.
    def kernel(x: float) -> float:
        a = x * 2.0**600
        b = a * 2.0**600
        return b + b

    assert _mnemonics(kernel, _WIDE) == collections.Counter({"fmul_ilog2": 1})
    model = holoso.synthesize(kernel, _WIDE, name="exact_exponent").numerical_model.elaborate()
    assert float(model.run(2.0**-1000)[0]) == 2.0**201


def test_a_sum_that_cancels_to_a_number_no_host_float_names_stands() -> None:
    # A number the machine must hold has to be a host float, so the cancelled sum keeps its addition; `a` is an
    # output so the scalings are not factored out first.
    def kernel(x: float) -> tuple[float, float]:
        a = (x + 2.0**600) * 2.0**600
        b = x * 2.0**600
        return a, a - b

    assert _mnemonics(kernel, _WIDE)["fadd"] == 2
    model = holoso.synthesize(kernel, _WIDE, name="huge_constant_stands").numerical_model.elaborate()
    scaled, got = model.run(0.0)
    assert isinstance(got, FloatValue) and got == scaled
    fmt = _WIDE.ffmt
    assert got.bits == fmt.encode(2.0**600) + (600 << (fmt.wman - 1)), "2**1200: the exponent field of 2**600 stepped"


def _collapse_retires_nothing(x: float) -> tuple[float, float]:
    scaled = 3.0 * x
    return scaled, scaled + scaled


def test_a_collapse_is_taken_whether_or_not_it_retires_anything() -> None:
    # Every collapse replaces an addition with at most one operation, so nothing is priced: `{x: 6}` is a multiply.
    counts = _mnemonics(_collapse_retires_nothing, _options(fma=False))
    assert counts["fadd"] == 0 and counts["fmul"] == 2


def test_a_collapse_asks_for_the_operator_its_answer_needs() -> None:
    # A sum selects its operator as a written product does: `{x: 2}` the scaler, `{x: 3}` the multiplier, an fma
    # standing in where the multiplier is missing.
    def doubled(x: float) -> float:
        return x + x

    def tripled(x: float) -> float:
        return x + x + x

    def five_minus_two(x: float) -> float:
        return 5.0 * x - 2.0 * x

    no_scaler = Options(OperatorOptions(fadd=FAddOptions(), fmul=FMulOptions()), ffmt=FMT)
    no_multiplier = Options(OperatorOptions(fadd=FAddOptions(), fmul_ilog2=FMulILog2Options()), ffmt=FMT)
    fma_for_the_multiplier = Options(
        OperatorOptions(fadd=FAddOptions(), ffma=FFmaOptions(), fmul_ilog2=FMulILog2Options()), ffmt=FMT
    )
    with pytest.raises(UnsupportedConstruct):
        holoso.synthesize(doubled, no_scaler, name="no_scaler")
    with pytest.raises(UnsupportedConstruct):
        holoso.synthesize(tripled, no_multiplier, name="no_multiplier")
    with pytest.raises(UnsupportedConstruct):
        holoso.synthesize(five_minus_two, fma_for_the_multiplier, name="fma_for_the_multiplier")
    assert _run(doubled, _options(fma=False), "doubled", 1.25) == (2.5,)
    assert _run(tripled, _options(fma=False), "tripled", 1.25) == (3.75,)
    assert _mnemonics(five_minus_two, _options(fma=True)) == collections.Counter({"fmul": 1})
    assert _run(five_minus_two, _options(fma=True), "five_minus_two", 2.0) == (6.0,)


@pytest.mark.parametrize(
    "kernel",
    [
        _answered_as_a_number,
        _mixed_spelling,
    ],
)
def test_an_answer_is_emitted_in_the_shape_the_reducer_leaves_it(kernel: Callable[..., object]) -> None:
    # White-box: only the reducing round's identity over the linear pass's output shows an answer needs no restating.
    from holoso._hir import _dce, _linear, _strength_reduce

    settled = lower(kernel, DEFAULT_UNROLL_MAX_TRIPS).hir
    while (reduced := _dce.eliminate_dead_code(_strength_reduce.run(settled))) != settled:
        settled = reduced
    answered = _dce.eliminate_dead_code(_linear.run(settled))
    assert answered != settled, "the kernel must give the pass something to answer"
    assert _strength_reduce.run(answered) == answered


def test_twin_products_of_opposite_sign_share_one_multiply() -> None:
    # `x*3.0` and `x*-3.0` are one node read through a sideband.
    def kernel(x: float) -> tuple[float, float]:
        return x * 3.0, x * -3.0

    assert _mnemonics(kernel, _options(fma=False)) == collections.Counter({"fmul": 1})
    assert _run(kernel, _options(fma=False), "twins", 2.0) == (6.0, -6.0)


def test_a_difference_of_equal_scalings_is_one_scaling_of_the_difference() -> None:
    # A reader stopping at the subtraction's negation would leave two multiplies.
    def kernel(x: float, y: float) -> float:
        return 3.0 * x - 3.0 * y

    assert _mnemonics(kernel, _options(fma=False)) == collections.Counter({"fmul": 1, "fadd": 1})
    assert _run(kernel, _options(fma=False), "scaled_difference", 2.0, 0.5) == (4.5,)


def test_a_negative_scaling_composes_through_its_sign_before_it_is_factored() -> None:
    # The factoring reads through the negation `t*-2.0` composes into.
    def kernel(x: float, y: float) -> float:
        t = x * 1.5
        return t * -2.0 + 3.0 * y

    assert _mnemonics(kernel, _options(fma=False)) == collections.Counter({"fmul": 1, "fadd": 1})
    assert _run(kernel, _options(fma=False), "composed_negative", 2.0, 1.0) == (-3.0,)


def test_a_written_subnormal_literal_keeps_its_magnitude() -> None:
    # A subnormal the kernel wrote is kept, carried as a significand and an exponent under its peeled sign.
    def kernel(x: float) -> float:
        return x * -3e-310

    fmt = FloatFormat(11, 36)
    options = Options(OperatorOptions(fadd=FAddOptions(), fmul=FMulOptions(), fmul_ilog2=FMulILog2Options()), ffmt=fmt)
    assert _mnemonics(kernel, options) == collections.Counter({"fmul": 1, "fmul_ilog2": 1})
    (got,) = _run(kernel, options, "subnormal_literal", 1e300)
    assert within(got, -3e-10, *default_tolerance(fmt, 2, magnitude=3e-10))


def test_composition_stops_where_the_composed_coefficient_is_unnameable() -> None:
    # The reader stops where composing fails and the exponent beneath is the base; both pairs still factor to one.
    def kernel(x: float, y: float) -> float:
        b = (x * 2.0**600) * 2.0**600
        d = (y * 2.0**600) * 2.0**600
        return b * 3.0 + d * 3.0

    assert _mnemonics(kernel, _WIDE) == collections.Counter({"fmul": 1, "fmul_ilog2": 1, "fadd": 1})
    assert _run(kernel, _WIDE, "unnameable_composition", 2.0**-1000, 2.0**-1000) == (6.0 * 2.0**200,)


def test_a_cancelled_sum_is_refused_over_the_constant_it_denotes() -> None:
    # The terms cancel to `6e38`, which the format cannot hold; a bare constant is no operator's to carry.
    def kernel(x: float, y: float) -> float:
        return (x + 3e38) + (y + 3e38) - (x + y)

    with pytest.raises(UnsupportedConstruct):
        holoso.synthesize(kernel, _options(fma=False), name="cancelled_to_a_rail")


def _shared_inner_scaling(x: float, y: float) -> tuple[float, float]:
    p = x * 3.0
    return p, p * 2.0 + y * 2.0


def test_a_scaling_composes_down_to_the_layer_another_consumer_wants() -> None:
    # `p` is read twice, so the outer scaling composes onto it and stops: factoring the pair still costs one multiply.
    assert _mnemonics(_shared_inner_scaling, _options(fma=False)) == collections.Counter(
        {"fmul": 1, "fmul_ilog2": 1, "fadd": 1}
    )
    assert _run(_shared_inner_scaling, _options(fma=False), "shared_inner", 2.0, 5.0) == (6.0, 22.0)


def test_a_constant_past_the_formats_reach_does_not_name_the_scaler_as_the_remedy() -> None:
    # The scaler splits a constant into a significand and an exponent, so it is the remedy only where the exponent
    # itself lands inside the format; past that it changes nothing and the message must not offer it.
    def kernel(x: float) -> float:
        return x * 1e-40

    narrow = FloatFormat(6, 18)
    without = Options(OperatorOptions(fadd=FAddOptions(), fmul=FMulOptions()), ffmt=narrow)
    with pytest.raises(UnsupportedConstruct):
        holoso.synthesize(kernel, without, name="past_the_reach")
    with pytest.raises(UnsupportedConstruct):
        holoso.synthesize(kernel, dataclasses.replace(without, operator=_with_scaler(without.operator)), name="k2")


def _with_scaler(ops: OperatorOptions) -> OperatorOptions:
    return dataclasses.replace(ops, fmul_ilog2=FMulILog2Options())
