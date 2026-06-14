"""
Public-API, black-box behavioral tests for floating-point edge behavior, the full relational/boolean breadth, the
fmul_ilog2 power-of-two strength reduction, and constant folding / static evaluation (round-3 axes B-F).

Every test drives the compiler ONLY through the public API (``holoso.synthesize(fn, ops).numerical_model.elaborate()``
then the simulator's ``run``) and asserts solely on observable output VALUES -- bits for floats, identity for bools --
never on any internal structure. The references are chosen to be FALSIFIABLE without a tolerance fudge wherever the
hardware must be exact:

  - Axis B (float edge behavior): kernels driven with ``format_edge_bits``-derived inputs (±0, ±smallest-normal,
    ±largest-finite, ±0.5, ±1) passed as exact ``FloatValue`` bit patterns, asserting algebraic identities the hardware
    must honor exactly -- x+(-x)==0, x*1.0==x, x*0.0==+0 (ZKF has no negative zero), abs via a sign-select, -(-x)==x,
    commutativity a+b==b+a and a*b==b*a bit-identical, overflow largest+largest -> +inf staying inf through a further
    op, and 1.0/0.0 emitting the error-path output (+inf). The references use ``FloatFormat.round`` so overflow and
    underflow follow the FORMAT's own rounding, not float64 (largest+largest is +inf in ZKF but finite in float64).
    Associativity is never asserted -- it does not hold in finite precision.

  - Axis C (relational + chained comparisons): all six operators ``< <= > >= == !=`` each in a bool-returning kernel,
    swept across vectors that straddle the exact boundary (equal, just-below, just-above), exact bool results; a chained
    ``lo < x < hi`` (two comparators AND-fused) over the four region boundaries; ``==`` / ``!=`` on bit-equal vs
    bit-different operands.

  - Axis D (fmul_ilog2 strength reduction): multiplication by power-of-two constants (x*2, x*0.5, x*8, x*0.125, x*2**-5)
    lowers to the fmul_ilog2 family and is EXACT (a power-of-two only shifts the exponent, barring overflow/underflow);
    a non-power-of-two constant (x*3) stays an ordinary fmul and is still correct; a power-of-two that pushes a normal
    input to overflow (largest*2 -> inf) and to underflow (smallest_normal*0.25 -> 0).

  - Axis E (boolean connectives + float<->bool casts): full truth tables for ``a and b``, ``a or b``, ``a and b or c``,
    De Morgan equivalence ``not (a and b)`` == ``(not a) or (not b)``; ``float(cond)`` (exactly 0.0/1.0) feeding
    arithmetic; a float compared then cast then multiplied (cross-domain chain); ``bool(x)`` truthiness.

  - Axis F (constant folding / static evaluation): a subexpression that folds to ZERO (``x + (2*3 - 6)`` == x, distinct
    from the existing x+7 fold); a compile-time-constant branch whose dead arm divides by 0.0 and must never execute; a
    read-only attribute folding to its snapshot in a condition; a bounded ``for`` loop fully unrolling a Horner
    polynomial, proved bit-identical to a hand-unrolled straight-line form (reference-free -- same op order, same bits).

Edge inputs are passed as exact ``FloatValue.from_bits`` so the extremes stay exact even where they would overflow a
Python float, and outputs are compared on ``.bits`` so the assertions cannot silently pass on a rounding accident.
"""

import math

import holoso
from holoso import (
    FAddOperator,
    FCmpOperator,
    FDivOperator,
    FloatFormat,
    FloatValue,
    FMulILog2OperatorFamily,
    FMulOperator,
    OpConfig,
)

from ._modelref import default_tolerance, format_edge_bits, within

FMT = FloatFormat(6, 18)


def _ops() -> OpConfig:
    return OpConfig(
        FAddOperator(FMT), FMulOperator(FMT), FDivOperator(FMT), FMulILog2OperatorFamily(FMT), FCmpOperator(FMT)
    )


def _sim(fn, name: str) -> holoso.NumericalSimulator:  # type: ignore[no-untyped-def]
    return holoso.synthesize(fn, _ops(), name=name).numerical_model.elaborate()


def _val(bits: int) -> FloatValue:
    """An exact ZKF input value from a raw bit pattern (so extremes survive even when a Python float cannot hold them)."""
    return FloatValue.from_bits(FMT, bits)


def _round(value: float) -> float:
    """The format-accurate reference: snap a real result to ZKF, so overflow -> inf and underflow follow the packer."""
    return FMT.round(value)


# --------------------------------------------------------------------------------------------------------------------
# Axis B: floating-point edge behavior via kernels + algebraic identities. Inputs are the canonical ZKF edge patterns
# (zero, ±0.5, ±1, ±smallest-normal, ±largest-finite); identities are asserted EXACTLY on the output bits. The
# reference for overflow/underflow is ``FloatFormat.round`` of the exact real, so it follows the format, not float64.
# --------------------------------------------------------------------------------------------------------------------

_EDGES = format_edge_bits(FMT)  # zero, ±0.5, ±1, ±smallest-normal, ±largest-finite (9 patterns)


def _add(a, b):  # type: ignore[no-untyped-def]
    return a + b


def _mul(a, b):  # type: ignore[no-untyped-def]
    return a * b


def _neg_self(x):  # type: ignore[no-untyped-def]
    # x + (-x) must be exactly +0 for every finite x.
    return x + (-x)


def test_additive_inverse_is_zero_over_edges() -> None:
    sim = _sim(_neg_self, "add_inverse")
    for bits in _EDGES:
        out = sim.run(_val(bits))[0]
        # x + (-x) == +0 for every finite edge; the largest-finite case does not overflow (it cancels exactly).
        assert out.bits == 0, f"x=0x{bits:x} ({float(_val(bits))}): x+(-x) bits=0x{out.bits:x} ({float(out)})"


def test_mul_by_one_is_identity_over_edges() -> None:
    sim = _sim(_mul, "mul_one")
    one = FloatValue.from_float(FMT, 1.0)
    for bits in _EDGES:
        out = sim.run(_val(bits), one)[0]
        assert out.bits == bits, f"x*1.0 changed bits: 0x{out.bits:x} vs 0x{bits:x}"


def test_mul_by_zero_is_positive_zero_over_edges() -> None:
    # x * 0.0 == +0 for every finite x, and ZKF has no negative zero, so even (-x)*0 is the canonical +0 (sign bit 0).
    sim = _sim(_mul, "mul_zero")
    zero = FloatValue.from_float(FMT, 0.0)
    for bits in _EDGES:
        out = sim.run(_val(bits), zero)[0]
        assert out.bits == 0 and out.sign == 0, f"x*0.0 not +0: bits=0x{out.bits:x} sign={out.sign}"


def _abs_via_select(x):  # type: ignore[no-untyped-def]
    # abs via a sign-select: -x if x < 0 else x. The magnitude must equal |x| exactly (a pure sign-bit clear in ZKF).
    return -x if x < 0.0 else x


def test_abs_via_select_clears_sign_over_edges() -> None:
    sim = _sim(_abs_via_select, "abs_select")
    for bits in _EDGES:
        x = _val(bits)
        out = sim.run(x)[0]
        want = x.apply_sign(negate=False, absolute=True)  # the magnitude bit pattern (sign bit cleared)
        assert out.bits == want.bits, f"abs(0x{bits:x}) bits=0x{out.bits:x} vs 0x{want.bits:x}"


def _double_neg(x):  # type: ignore[no-untyped-def]
    return -(-x)


def test_double_negation_is_identity_over_edges() -> None:
    sim = _sim(_double_neg, "double_neg")
    for bits in _EDGES:
        out = sim.run(_val(bits))[0]
        assert out.bits == bits, f"-(-x) changed bits: 0x{out.bits:x} vs 0x{bits:x}"


def test_addition_commutes_bit_identical_over_edges() -> None:
    # a + b must equal b + a bit-for-bit: the commutative-port assignment must not change the value.
    sim = _sim(_add, "add_commute")
    for ab in _EDGES:
        for bb in _EDGES:
            forward = sim.run(_val(ab), _val(bb))[0]
            reverse = sim.run(_val(bb), _val(ab))[0]
            assert (
                forward.bits == reverse.bits
            ), f"a+b != b+a: a=0x{ab:x} b=0x{bb:x}: 0x{forward.bits:x} vs 0x{reverse.bits:x}"


def test_multiplication_commutes_bit_identical_over_edges() -> None:
    sim = _sim(_mul, "mul_commute")
    for ab in _EDGES:
        for bb in _EDGES:
            forward = sim.run(_val(ab), _val(bb))[0]
            reverse = sim.run(_val(bb), _val(ab))[0]
            assert (
                forward.bits == reverse.bits
            ), f"a*b != b*a: a=0x{ab:x} b=0x{bb:x}: 0x{forward.bits:x} vs 0x{reverse.bits:x}"


def _largest_finite() -> FloatValue:
    frac_bits = FMT.wman - 1
    max_exp = (1 << FMT.wexp) - 2  # the all-ones exponent is infinity, so the largest finite exponent is one below it
    return _val((max_exp << frac_bits) | ((1 << frac_bits) - 1))


def _overflow_then_mul(x, y):  # type: ignore[no-untyped-def]
    # (x + x) overflows to +inf at the extreme; multiplying by y must keep it inf (inf * finite-positive == inf).
    return (x + x) * y


def test_overflow_to_inf_and_stays_inf() -> None:
    sim = _sim(_overflow_then_mul, "overflow_inf")
    lf = _largest_finite()
    one = FloatValue.from_float(FMT, 1.0)
    out = sim.run(lf, one)[0]
    # largest + largest overflows: the ZKF-accurate reference is round(2*largest) == +inf, and inf*1 stays inf.
    want = _round(_round(2.0 * float(lf)) * 1.0)
    assert math.isinf(want) and want > 0.0  # guard the reference itself
    assert math.isinf(float(out)) and float(out) > 0.0, f"overflow chain not +inf: {float(out)} (bits 0x{out.bits:x})"


def _div(a, b):  # type: ignore[no-untyped-def]
    return a / b


def test_divide_by_zero_emits_inf_error_path() -> None:
    # 1.0 / 0.0 is the error path; its OUTPUT value is +inf (err_pc is not model-observable, so it is not asserted).
    sim = _sim(_div, "div_zero")
    out = sim.run(FloatValue.from_float(FMT, 1.0), FloatValue.from_float(FMT, 0.0))[0]
    assert math.isinf(float(out)) and float(out) > 0.0, f"1.0/0.0 not +inf: {float(out)} (bits 0x{out.bits:x})"


# --------------------------------------------------------------------------------------------------------------------
# Axis C: relational + chained comparisons, full breadth. All six operators each in a bool-returning kernel, swept
# across the exact boundary (equal, just-below, just-above); a chained ``lo < x < hi`` over the four region boundaries;
# == / != on bit-equal vs bit-different operands.
# --------------------------------------------------------------------------------------------------------------------


def _k_lt(x, y):  # type: ignore[no-untyped-def]
    return x < y


def _k_le(x, y):  # type: ignore[no-untyped-def]
    return x <= y


def _k_gt(x, y):  # type: ignore[no-untyped-def]
    return x > y


def _k_ge(x, y):  # type: ignore[no-untyped-def]
    return x >= y


def _k_eq(x, y):  # type: ignore[no-untyped-def]
    return x == y


def _k_ne(x, y):  # type: ignore[no-untyped-def]
    return x != y


def test_all_six_relational_operators_exact_at_boundary() -> None:
    # A representable pivot and its exact neighbours one ULP away (so just-below / just-above straddle the boundary).
    pivot = 2.0
    pivot_bits = FMT.encode(pivot)
    below = float(_val(pivot_bits - 1))  # one ULP below pivot
    above = float(_val(pivot_bits + 1))  # one ULP above pivot
    pairs = [(pivot, pivot), (below, pivot), (above, pivot), (pivot, below), (pivot, above), (-pivot, pivot)]
    for fn, py, name in [
        (_k_lt, lambda a, b: a < b, "lt"),
        (_k_le, lambda a, b: a <= b, "le"),
        (_k_gt, lambda a, b: a > b, "gt"),
        (_k_ge, lambda a, b: a >= b, "ge"),
        (_k_eq, lambda a, b: a == b, "eq"),
        (_k_ne, lambda a, b: a != b, "ne"),
    ]:
        sim = _sim(fn, f"rel_{name}")
        for x, y in pairs:
            got = sim.run(FloatValue.from_float(FMT, x), FloatValue.from_float(FMT, y))[0]
            want = py(x, y)
            assert got is want, f"{name}({x}, {y}) = {got}, want {want}"


def _chained(lo, x, hi):  # type: ignore[no-untyped-def]
    # lo < x < hi lowers to two comparators AND-fused; must match Python's chained-comparison semantics exactly.
    return lo < x < hi


def test_chained_comparison_over_all_region_boundaries() -> None:
    sim = _sim(_chained, "chained_cmp")
    lo, hi = 1.0, 3.0
    # The four region boundaries plus interior/exterior points: below lo, AT lo, between, AT hi, above hi.
    for x in (0.5, 1.0, 2.0, 3.0, 3.5, lo, hi):
        got = sim.run(FloatValue.from_float(FMT, lo), FloatValue.from_float(FMT, x), FloatValue.from_float(FMT, hi))[0]
        assert got is (lo < x < hi), f"{lo} < {x} < {hi} = {got}, want {lo < x < hi}"


def test_equality_bit_equal_vs_bit_different() -> None:
    # == is True iff the operands are numerically equal; bit-different normals must compare unequal, bit-equal equal.
    sim_eq = _sim(_k_eq, "eq_bits")
    sim_ne = _sim(_k_ne, "ne_bits")
    base = FMT.encode(1.5)
    same = _val(base)
    other = _val(base + 1)  # one ULP different -> a different value
    assert sim_eq.run(same, same)[0] is True
    assert sim_eq.run(same, other)[0] is False
    assert sim_ne.run(same, same)[0] is False
    assert sim_ne.run(same, other)[0] is True


# --------------------------------------------------------------------------------------------------------------------
# Axis D: fmul_ilog2 power-of-two strength reduction. Multiplying by a power-of-two constant only shifts the exponent,
# so the result is EXACT (no rounding) -- asserted bit-for-bit against ``FloatFormat.round`` of the exact product. A
# non-power-of-two constant stays an ordinary fmul and is still correct. Power-of-two overflow/underflow edges too.
# --------------------------------------------------------------------------------------------------------------------


def _x_times_2(x):  # type: ignore[no-untyped-def]
    return x * 2.0


def _x_times_half(x):  # type: ignore[no-untyped-def]
    return x * 0.5


def _x_times_8(x):  # type: ignore[no-untyped-def]
    return x * 8.0


def _x_times_eighth(x):  # type: ignore[no-untyped-def]
    return x * 0.125


def _x_times_2_pow_neg5(x):  # type: ignore[no-untyped-def]
    return x * 2.0**-5


def test_power_of_two_strength_reduction_is_exact() -> None:
    # For a power-of-two scale, the result equals round(exact product) bit-for-bit -- no rounding occurs in range.
    cases = [
        (_x_times_2, 2.0, "x2"),
        (_x_times_half, 0.5, "xhalf"),
        (_x_times_8, 8.0, "x8"),
        (_x_times_eighth, 0.125, "xeighth"),
        (_x_times_2_pow_neg5, 2.0**-5, "x2neg5"),
    ]
    for fn, factor, name in cases:
        sim = _sim(fn, f"pow2_{name}")
        for x in (-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0, 17.0):
            got = sim.run(FloatValue.from_float(FMT, x))[0]
            want = FloatValue.from_float(FMT, _round(x * factor))
            assert got.bits == want.bits, f"{name}: {x}*{factor} bits=0x{got.bits:x} vs 0x{want.bits:x}"


def _x_times_3(x):  # type: ignore[no-untyped-def]
    # 3.0 is NOT a power of two, so this stays an ordinary fmul; still must round-correctly.
    return x * 3.0


def test_non_power_of_two_stays_ordinary_fmul_and_correct() -> None:
    sim = _sim(_x_times_3, "mul3")
    for x in (-3.0, -1.0, 0.0, 0.5, 1.0, 3.0, 17.0, 100.0):
        got = sim.run(FloatValue.from_float(FMT, x))[0]
        want = FloatValue.from_float(FMT, _round(x * 3.0))
        assert got.bits == want.bits, f"x*3.0: {x} bits=0x{got.bits:x} vs 0x{want.bits:x}"


def test_power_of_two_overflow_and_underflow_edges() -> None:
    # A power-of-two scale that pushes a normal input out of range: largest * 2 -> +inf; smallest_normal * 0.25 -> +0.
    sim_double = _sim(_x_times_2, "pow2_ovf")
    sim_quarter = _sim(_x_times_eighth, "pow2_unf")  # 0.125 underflows the smallest normal even more deeply
    lf = _largest_finite()
    over = sim_double.run(lf)[0]
    assert math.isinf(float(over)) and float(over) > 0.0, f"largest*2 not +inf: {float(over)}"
    frac_bits = FMT.wman - 1
    smallest_normal = _val(1 << frac_bits)  # exponent 1, zero fraction
    under = sim_quarter.run(smallest_normal)[0]
    # smallest_normal * 0.125 underflows below the half-MIN_NORMAL boundary -> rounds to +0 (the format's own rule).
    assert under.bits == 0, f"smallest_normal*0.125 not +0: bits=0x{under.bits:x} ({float(under)})"


# --------------------------------------------------------------------------------------------------------------------
# Axis E: boolean connectives + float<->bool casts. Full truth tables, De Morgan equivalence, float(cond) feeding
# arithmetic, a compare->cast->multiply cross-domain chain, and bool(x) truthiness.
# --------------------------------------------------------------------------------------------------------------------


def _k_and(a: bool, b: bool):  # type: ignore[no-untyped-def]
    return a and b


def _k_or(a: bool, b: bool):  # type: ignore[no-untyped-def]
    return a or b


def _k_and_or(a: bool, b: bool, c: bool):  # type: ignore[no-untyped-def]
    return a and b or c  # (a and b) or c by Python precedence


def test_boolean_and_or_truth_tables() -> None:
    sim_and = _sim(_k_and, "bool_and")
    sim_or = _sim(_k_or, "bool_or")
    for a in (True, False):
        for b in (True, False):
            assert sim_and.run(a, b)[0] is (a and b), f"and({a},{b})"
            assert sim_or.run(a, b)[0] is (a or b), f"or({a},{b})"


def test_boolean_and_or_compound_truth_table() -> None:
    sim = _sim(_k_and_or, "bool_and_or")
    for a in (True, False):
        for b in (True, False):
            for c in (True, False):
                assert sim.run(a, b, c)[0] is ((a and b) or c), f"and_or({a},{b},{c})"


def _demorgan_lhs(a: bool, b: bool):  # type: ignore[no-untyped-def]
    return not (a and b)


def _demorgan_rhs(a: bool, b: bool):  # type: ignore[no-untyped-def]
    return (not a) or (not b)


def test_de_morgan_equivalence_full_truth_table() -> None:
    sim_lhs = _sim(_demorgan_lhs, "demorgan_lhs")
    sim_rhs = _sim(_demorgan_rhs, "demorgan_rhs")
    for a in (True, False):
        for b in (True, False):
            lhs = sim_lhs.run(a, b)[0]
            rhs = sim_rhs.run(a, b)[0]
            want = not (a and b)
            assert lhs is want and rhs is want, f"de morgan ({a},{b}): lhs={lhs} rhs={rhs} want={want}"


def _float_of_cond(x, y):  # type: ignore[no-untyped-def]
    # float(x > y) must be exactly 0.0 or 1.0; feeding it into arithmetic gives a clean gate.
    return float(x > y) * 10.0 + 1.0


def test_float_of_bool_is_exactly_zero_or_one_feeding_arithmetic() -> None:
    sim = _sim(_float_of_cond, "float_cond")
    for x, y in [(3.0, 1.0), (1.0, 3.0), (2.0, 2.0)]:
        got = sim.run(FloatValue.from_float(FMT, x), FloatValue.from_float(FMT, y))[0]
        want = FloatValue.from_float(FMT, (10.0 if x > y else 0.0) + 1.0)  # exactly 11.0 or 1.0
        assert got.bits == want.bits, f"float({x}>{y})*10+1 bits=0x{got.bits:x} vs 0x{want.bits:x}"


def _cross_domain_chain(x, y):  # type: ignore[no-untyped-def]
    # A float compared, the bool cast to float, then multiplied by a float: a full float->bool->float round trip.
    return float(x > y) * (x + y)


def test_compare_cast_multiply_cross_domain_chain() -> None:
    sim = _sim(_cross_domain_chain, "cross_chain")
    for x, y in [(3.0, 1.0), (1.0, 3.0), (2.0, 2.0), (-1.0, -2.0)]:
        got = sim.run(FloatValue.from_float(FMT, x), FloatValue.from_float(FMT, y))[0]
        gate = 1.0 if x > y else 0.0
        want = FloatValue.from_float(FMT, _round(gate * _round(x + y)))  # the sum rounds, then the (exact) gate scales
        assert got.bits == want.bits, f"cross chain ({x},{y}) bits=0x{got.bits:x} vs 0x{want.bits:x}"


def _bool_of_float(x):  # type: ignore[no-untyped-def]
    # bool(x) truthiness: nonzero -> True, +0 -> False.
    return bool(x)


def test_bool_of_float_truthiness() -> None:
    sim = _sim(_bool_of_float, "bool_float")
    assert sim.run(FloatValue.from_float(FMT, 0.0))[0] is False
    assert sim.run(FloatValue.from_float(FMT, 1.0))[0] is True
    assert sim.run(FloatValue.from_float(FMT, -2.5))[0] is True
    # smallest-normal is the smallest nonzero magnitude -> still truthy.
    frac_bits = FMT.wman - 1
    assert sim.run(_val(1 << frac_bits))[0] is True


# --------------------------------------------------------------------------------------------------------------------
# Axis F: constant folding / static evaluation. A subexpression folding to ZERO (distinct from the existing x+7 fold),
# a compile-time-constant branch dropping a divide-by-zero dead arm, a read-only attribute folding in a condition, and
# a bounded ``for`` loop fully unrolling a Horner polynomial -- proved bit-identical to a hand-unrolled straight-line
# version (reference-free: same op order => same rounding => same bits).
# --------------------------------------------------------------------------------------------------------------------


def _fold_to_zero(x):  # type: ignore[no-untyped-def]
    # The subexpression 2*3 - 6 folds to 0.0 at compile time, so x + 0.0 == x exactly for every representable x.
    return x + (2.0 * 3.0 - 6.0)


def test_constant_subexpression_folds_to_zero() -> None:
    sim = _sim(_fold_to_zero, "fold_zero")
    for bits in _EDGES:
        out = sim.run(_val(bits))[0]
        assert out.bits == bits, f"x + (2*3-6) changed bits: 0x{out.bits:x} vs 0x{bits:x}"


def _dead_arm_divides_by_zero(x):  # type: ignore[no-untyped-def]
    # 3.0 < 2.0 folds to False; the THEN arm (which divides by a compile-time 0.0) must be pruned, never lowered.
    if 3.0 < 2.0:
        r = x / 0.0
    else:
        r = x + 1.0
    return r


def test_constant_condition_drops_divide_by_zero_dead_arm() -> None:
    sim = _sim(_dead_arm_divides_by_zero, "dead_arm")
    for x in (-3.0, 0.0, 1.0, 5.0, 7.0):
        got = sim.run(FloatValue.from_float(FMT, x))[0]
        want = FloatValue.from_float(FMT, _round(x + 1.0))
        assert got.bits == want.bits, f"x+1 (dead arm pruned): {x} bits=0x{got.bits:x} vs 0x{want.bits:x}"


class _AttributeConfig:
    """A read-only attribute snapshotted at synthesis; the condition that reads it must fold to its captured value."""

    def __init__(self, threshold: float) -> None:
        self._threshold = threshold

    def __call__(self, x):  # type: ignore[no-untyped-def]
        if self._threshold > 2.0:  # _threshold == 3.0 is a compile-time snapshot -> the condition folds to True
            r = x + 1.0
        else:
            r = x / 0.0  # would error if ever lowered; a correct result proves this arm was pruned
        return r


def test_read_only_attribute_folds_in_condition() -> None:
    sim = _sim(_AttributeConfig(3.0).__call__, "attr_fold")
    for x in (-2.0, 0.0, 3.0, 9.0):
        got = sim.run(FloatValue.from_float(FMT, x))[0]
        want = FloatValue.from_float(FMT, _round(x + 1.0))
        assert got.bits == want.bits, f"attr-fold x+1: {x} bits=0x{got.bits:x} vs 0x{want.bits:x}"


def _horner_loop(x):  # type: ignore[no-untyped-def]
    # A bounded for-loop that fully unrolls a Horner evaluation of 1*x^4 + 1*x^3 + 1*x^2 + 1*x + 1.
    acc = 0.0
    for _ in range(5):
        acc = acc * x + 1.0
    return acc


def _horner_unrolled(x):  # type: ignore[no-untyped-def]
    # The SAME Horner recurrence written out straight-line: identical op order, so the bits must match exactly.
    acc = 0.0
    acc = acc * x + 1.0
    acc = acc * x + 1.0
    acc = acc * x + 1.0
    acc = acc * x + 1.0
    acc = acc * x + 1.0
    return acc


def test_bounded_for_loop_unrolls_to_straight_line_bit_identical() -> None:
    sim_loop = _sim(_horner_loop, "horner_loop")
    sim_flat = _sim(_horner_unrolled, "horner_flat")
    for x in (-2.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
        looped = sim_loop.run(FloatValue.from_float(FMT, x))[0]
        flat = sim_flat.run(FloatValue.from_float(FMT, x))[0]
        assert (
            looped.bits == flat.bits
        ), f"unrolled loop != straight-line at x={x}: 0x{looped.bits:x} vs 0x{flat.bits:x}"


def test_bounded_for_loop_polynomial_matches_reference() -> None:
    # An independent float64 reference (within tolerance) backs up the reference-free bit-identical check above.
    sim = _sim(_horner_loop, "horner_ref")
    for x in (-2.0, -0.5, 0.5, 1.0, 1.5, 2.0):
        got = float(sim.run(FloatValue.from_float(FMT, x))[0])
        want = (((1.0 * x + 1.0) * x + 1.0) * x + 1.0) * x + 1.0  # the degree-4 polynomial closed form
        rtol, atol = default_tolerance(FMT, op_count=10, magnitude=max(1.0, abs(want)))
        assert within(got, want, rtol, atol), f"horner x={x}: {got} vs {want}"
