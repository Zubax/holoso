"""
Public-API behavioral tests for floating-point edge behavior, the full relational/boolean breadth, fmul_ilog2
power-of-two strength reduction, trivial fast-math float folding, and constant folding / static evaluation.

Every test drives the compiler only through the public API (``holoso.synthesize(fn, ops)`` and the numerical simulator).
Most assertions are on observable output values -- bits for floats, identity for bools -- while explicit operator
selection guards inspect generated Verilog text through the public synthesis result. The references are chosen to be
FALSIFIABLE without a tolerance fudge wherever the hardware must be exact:

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

  - Trivial fast-math folds: algebraic identities, zero folds, sign folds, self-division, and self-subtraction remove
    fadd/fmul/fdiv/fmul_ilog2 hardware where policy permits, including observable sideband and signed-zero deviations.

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
from collections.abc import Callable

import numpy as np
import pytest

import holoso
from holoso import (
    FAddOptions,
    FCmpOptions,
    FDivOptions,
    FMulILog2Options,
    FMulOptions,
    FloatFormat,
    FloatValue,
    OperatorOptions,
    Options,
    SynthesisError,
)
from ._modelref import default_tolerance, format_edge_bits, within

FMT = FloatFormat(6, 18)


def _ops() -> Options:
    return Options(
        OperatorOptions(
            fadd=FAddOptions(),
            fmul=FMulOptions(),
            fdiv=FDivOptions(),
            fmul_ilog2=FMulILog2Options(),
            fcmp=FCmpOptions(),
        ),
        ffmt=FMT,
    )


def _sim(fn: Callable[..., object], name: str) -> holoso.NumericalSimulator:
    return holoso.synthesize(fn, _ops(), name=name).numerical_model.elaborate()


def _val(bits: int) -> FloatValue:
    """
    An exact ZKF input value from a raw bit pattern (so extremes survive even when a Python float cannot hold them).
    """
    return FloatValue.from_bits(FMT, bits)


def _round(value: float) -> float:
    """The format-accurate reference: snap a real result to ZKF, so overflow -> inf and underflow follow the packer."""
    return FMT.round(value)


_EDGES = format_edge_bits(FMT)  # zero, ±0.5, ±1, ±smallest-normal, ±largest-finite (9 patterns)


def _add(a: float, b: float) -> float:
    return a + b


def _mul(a: float, b: float) -> float:
    return a * b


def _neg_self(x: float) -> float:
    return x + (-x)


def test_additive_inverse_is_zero_over_edges() -> None:
    sim = _sim(_neg_self, "add_inverse")
    for bits in _EDGES:
        out = sim.run(_val(bits))[0]
        assert isinstance(out, FloatValue)
        # x + (-x) == +0 for every finite edge; the largest-finite case does not overflow (it cancels exactly).
        assert out.bits == 0, f"x=0x{bits:x} ({float(_val(bits))}): x+(-x) bits=0x{out.bits:x} ({float(out)})"


def test_mul_by_one_is_identity_over_edges() -> None:
    sim = _sim(_mul, "mul_one")
    one = FloatValue.from_float(FMT, 1.0)
    for bits in _EDGES:
        out = sim.run(_val(bits), one)[0]
        assert isinstance(out, FloatValue)
        assert out.bits == bits, f"x*1.0 changed bits: 0x{out.bits:x} vs 0x{bits:x}"


def test_mul_by_zero_is_positive_zero_over_edges() -> None:
    # x * 0.0 == +0 for every finite x, and ZKF has no negative zero, so even (-x)*0 is the canonical +0 (sign bit 0).
    sim = _sim(_mul, "mul_zero")
    zero = FloatValue.from_float(FMT, 0.0)
    for bits in _EDGES:
        out = sim.run(_val(bits), zero)[0]
        assert isinstance(out, FloatValue)
        assert out.bits == 0 and not out.negative, f"x*0.0 not +0: bits=0x{out.bits:x} negative={out.negative}"


def _abs_via_select(x: float) -> float:
    # abs via a sign-select: -x if x < 0 else x. The magnitude must equal |x| exactly (a pure sign-bit clear in ZKF).
    return -x if x < 0.0 else x


def test_abs_via_select_clears_sign_over_edges() -> None:
    sim = _sim(_abs_via_select, "abs_select")
    for bits in _EDGES:
        x = _val(bits)
        out = sim.run(x)[0]
        assert isinstance(out, FloatValue)
        want = x.apply_sign(negate=False, absolute=True)
        assert out.bits == want.bits, f"abs(0x{bits:x}) bits=0x{out.bits:x} vs 0x{want.bits:x}"


def _double_neg(x: float) -> float:
    return -(-x)


def test_double_negation_is_identity_over_edges() -> None:
    sim = _sim(_double_neg, "double_neg")
    for bits in _EDGES:
        out = sim.run(_val(bits))[0]
        assert isinstance(out, FloatValue)
        assert out.bits == bits, f"-(-x) changed bits: 0x{out.bits:x} vs 0x{bits:x}"


def test_addition_commutes_bit_identical_over_edges() -> None:
    # a + b must equal b + a bit-for-bit: the commutative-port assignment must not change the value.
    sim = _sim(_add, "add_commute")
    for ab in _EDGES:
        for bb in _EDGES:
            forward = sim.run(_val(ab), _val(bb))[0]
            reverse = sim.run(_val(bb), _val(ab))[0]
            assert isinstance(forward, FloatValue) and isinstance(reverse, FloatValue)
            assert (
                forward.bits == reverse.bits
            ), f"a+b != b+a: a=0x{ab:x} b=0x{bb:x}: 0x{forward.bits:x} vs 0x{reverse.bits:x}"


def test_multiplication_commutes_bit_identical_over_edges() -> None:
    sim = _sim(_mul, "mul_commute")
    for ab in _EDGES:
        for bb in _EDGES:
            forward = sim.run(_val(ab), _val(bb))[0]
            reverse = sim.run(_val(bb), _val(ab))[0]
            assert isinstance(forward, FloatValue) and isinstance(reverse, FloatValue)
            assert (
                forward.bits == reverse.bits
            ), f"a*b != b*a: a=0x{ab:x} b=0x{bb:x}: 0x{forward.bits:x} vs 0x{reverse.bits:x}"


def _largest_finite() -> FloatValue:
    frac_bits = FMT.wman - 1
    max_exp = (1 << FMT.wexp) - 2  # the all-ones exponent is infinity, so the largest finite exponent is one below it
    return _val((max_exp << frac_bits) | ((1 << frac_bits) - 1))


def _overflow_then_mul(x: float, y: float) -> float:
    # (x + x) overflows to +inf at the extreme; multiplying by y must keep it inf (inf * finite-positive == inf).
    return (x + x) * y


def test_overflow_to_inf_and_stays_inf() -> None:
    sim = _sim(_overflow_then_mul, "overflow_inf")
    lf = _largest_finite()
    one = FloatValue.from_float(FMT, 1.0)
    out = sim.run(lf, one)[0]
    assert isinstance(out, FloatValue)
    # largest + largest overflows: the ZKF-accurate reference is round(2*largest) == +inf, and inf*1 stays inf.
    want = _round(_round(2.0 * float(lf)) * 1.0)
    assert math.isinf(want) and want > 0.0  # guard the reference itself
    assert math.isinf(float(out)) and float(out) > 0.0, f"overflow chain not +inf: {float(out)} (bits 0x{out.bits:x})"


def _div(a: float, b: float) -> float:
    return a / b


def test_divide_by_zero_emits_inf_error_path() -> None:
    # 1.0 / 0.0 is the error path; its OUTPUT value is +inf (err_pc is not model-observable, so it is not asserted).
    sim = _sim(_div, "div_zero")
    out = sim.run(FloatValue.from_float(FMT, 1.0), FloatValue.from_float(FMT, 0.0))[0]
    assert isinstance(out, FloatValue)
    assert math.isinf(float(out)) and float(out) > 0.0, f"1.0/0.0 not +inf: {float(out)} (bits 0x{out.bits:x})"


_INF = math.inf  # a module-level constant, the supported spelling of a compile-time infinity in a kernel


def _inf_minus_inf(x: float) -> float:
    return _INF + -_INF


def _runtime_cancellation(x: float) -> float:
    return x + -x


def test_the_identities_govern_unknown_operands_and_arithmetic_governs_constants() -> None:
    # The two halves of the charter, side by side on the same shapes. Over an operand the compiler cannot see, the
    # identity holds whatever the value turns out to be, so the expression folds away and no hardware is emitted.
    # Over constants nothing is assumed, ordinary arithmetic decides, and an indeterminate form names no number.
    for kernel, refused in (
        (_inf_minus_inf, "the sum"),
        (_inf_div_inf, "the quotient"),
        (_zero_div_zero, "the quotient"),
        (_zero_times_inf, "the product"),
    ):
        with pytest.raises(SynthesisError, match=refused):
            holoso.synthesize(kernel, _ops(), name=kernel.__name__.lstrip("_"))
    # ... while the same three identities over a runtime operand are unaffected, and emit nothing either.
    result = holoso.synthesize(_runtime_cancellation, _ops(), name="runtime_cancellation")
    assert "holoso_fadd #" not in result.verilog_output.verilog
    out = result.numerical_model.elaborate().run(_val(0))[0]
    assert isinstance(out, FloatValue)
    assert out.bits == 0


def _inf_div_inf(x: float) -> float:
    return _INF / _INF


def _zero_div_zero(x: float) -> float:
    return 0.0 / 0.0


def _zero_times_inf(x: float) -> float:
    return 0.0 * _INF


def _self_division(x: float) -> float:
    return x / x


def test_self_division_folds_to_one_over_an_unknown_operand() -> None:
    # x/x == 1 holds whatever the unknown x turns out to be, so no divider is emitted and the answer is 1 even where
    # the datapath would have computed something else -- the divergence the charter accepts. The constant twins of this
    # shape (0.0/0.0, inf/inf) name no number and are refused instead, which the test above pins.
    result = holoso.synthesize(_self_division, _ops(), name="self_division")
    assert "holoso_fdiv #" not in result.verilog_output.verilog
    out = result.numerical_model.elaborate().run(_val(0))[0]  # x == 0 at run time, and the answer is still 1
    assert isinstance(out, FloatValue)
    assert out.bits == FloatValue.from_float(FMT, 1.0).bits


def _k_lt(x: float, y: float) -> bool:
    return x < y


def _k_le(x: float, y: float) -> bool:
    return x <= y


def _k_gt(x: float, y: float) -> bool:
    return x > y


def _k_ge(x: float, y: float) -> bool:
    return x >= y


def _k_eq(x: float, y: float) -> bool:
    return x == y


def _k_ne(x: float, y: float) -> bool:
    return x != y


def test_all_six_relational_operators_exact_at_boundary() -> None:
    # A representable pivot and its exact neighbours one ULP away (so just-below / just-above straddle the boundary).
    pivot = 2.0
    pivot_bits = FMT.encode(pivot)
    below = float(_val(pivot_bits - 1))
    above = float(_val(pivot_bits + 1))
    pairs = [(pivot, pivot), (below, pivot), (above, pivot), (pivot, below), (pivot, above), (-pivot, pivot)]
    cases: list[tuple[Callable[[float, float], bool], Callable[[float, float], bool], str]] = [
        (_k_lt, lambda a, b: a < b, "lt"),
        (_k_le, lambda a, b: a <= b, "le"),
        (_k_gt, lambda a, b: a > b, "gt"),
        (_k_ge, lambda a, b: a >= b, "ge"),
        (_k_eq, lambda a, b: a == b, "eq"),
        (_k_ne, lambda a, b: a != b, "ne"),
    ]
    for fn, py, name in cases:
        sim = _sim(fn, f"rel_{name}")
        for x, y in pairs:
            got = sim.run(FloatValue.from_float(FMT, x), FloatValue.from_float(FMT, y))[0]
            want = py(x, y)
            assert got is want, f"{name}({x}, {y}) = {got}, want {want}"


def _chained(lo: float, x: float, hi: float) -> bool:
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
    other = _val(base + 1)
    assert sim_eq.run(same, same)[0] is True
    assert sim_eq.run(same, other)[0] is False
    assert sim_ne.run(same, same)[0] is False
    assert sim_ne.run(same, other)[0] is True


def _x_times_2(x: float) -> float:
    return x * 2.0


def _x_times_half(x: float) -> float:
    return x * 0.5


def _x_times_8(x: float) -> float:
    return x * 8.0


def _x_times_eighth(x: float) -> float:
    return x * 0.125


def _x_times_2_pow_neg5(x: float) -> float:
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
            assert isinstance(got, FloatValue)
            want = FloatValue.from_float(FMT, _round(x * factor))
            assert got.bits == want.bits, f"{name}: {x}*{factor} bits=0x{got.bits:x} vs 0x{want.bits:x}"


def _x_times_3(x: float) -> float:
    # 3.0 is NOT a power of two, so this stays an ordinary fmul; still must round-correctly.
    return x * 3.0


def test_non_power_of_two_stays_ordinary_fmul_and_correct() -> None:
    sim = _sim(_x_times_3, "mul3")
    for x in (-3.0, -1.0, 0.0, 0.5, 1.0, 3.0, 17.0, 100.0):
        got = sim.run(FloatValue.from_float(FMT, x))[0]
        assert isinstance(got, FloatValue)
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
    assert isinstance(under, FloatValue)
    # smallest_normal * 0.125 underflows below the half-MIN_NORMAL boundary -> rounds to +0 (the format's own rule).
    assert under.bits == 0, f"smallest_normal*0.125 not +0: bits=0x{under.bits:x} ({float(under)})"


def _trivial_float_folds(x: float, y: float) -> tuple[float, ...]:
    return (
        x * 1.0,
        1.0 * x,
        x / 1.0,
        x + 0.0,
        0.0 + x,
        x - 0.0,
        0.0 - x,
        x * 0.0,
        0.0 * x,
        0.0 / y,
        x * -1.0,
        -1.0 * x,
        x / -1.0,
        x / x,
        x - x,
    )


def test_trivial_fast_math_float_folds_are_operator_free_and_bit_exact() -> None:
    result = holoso.synthesize(_trivial_float_folds, _ops(), name="trivial_float_folds")
    verilog = result.verilog_output.verilog
    assert "holoso_fadd #" not in verilog
    assert "holoso_fmul #" not in verilog
    assert "holoso_fdiv #" not in verilog
    assert "holoso_fmul_ilog2_const" not in verilog

    sim = result.numerical_model.elaborate()
    vectors = [
        FloatValue.from_float(FMT, 0.0),
        FloatValue.from_bits(FMT, 1 << (FMT.width - 1)),
        FloatValue.from_float(FMT, 1.5),
        FloatValue.from_float(FMT, -2.0),
        FloatValue.from_float(FMT, float("inf")),
        FloatValue.from_float(FMT, float("-inf")),
    ]
    one = FloatValue.from_float(FMT, 1.0)
    for x in vectors:
        for y in vectors:
            out = sim.run(x, y)
            values: list[FloatValue] = []
            for value in out:
                assert isinstance(value, FloatValue)
                values.append(value)
            negated = x.apply_sign(negate=True, absolute=False).bits
            expected_bits = [
                x.bits,
                x.bits,
                x.bits,
                x.bits,
                x.bits,
                x.bits,
                negated,
                0,
                0,
                0,
                negated,
                negated,
                negated,
                one.bits,
                0,
            ]
            for index, (got, want) in enumerate(zip(values, expected_bits, strict=True)):
                assert got.bits == want, f"fold {index} x=0x{x.bits:x} y=0x{y.bits:x}: 0x{got.bits:x} != 0x{want:x}"


def _dynamic_div(x: float, y: float) -> float:
    return x / y


def test_dynamic_non_identical_division_still_emits_fdiv() -> None:
    result = holoso.synthesize(_dynamic_div, _ops(), name="dynamic_div")
    assert "holoso_fdiv #" in result.verilog_output.verilog


def _div_by_zero_const(x: float) -> float:
    return x / 0.0


def _dead_div_by_zero(x: float) -> float:
    unused = x / 0.0  # noqa: F841 -- never read, so DCE removes it before anything can be diagnosed
    return x + 1.0


def _div_by_zero_in_a_live_arm(x: float) -> float:
    r = x
    if x > 0.0:
        r = x / 0.0
    return r


def _div_by_zero_behind_a_folded_guard(x: float) -> float:
    """
    The kernel writes ``x / w``, never ``x / 0.0``: unrolling is what substitutes the zero. Whether the guard resolves
    before the body is lowered or after it reaches HIR decides only which pass deletes the disabled tap, never whether
    the kernel builds -- refusing what the compiler's own transformation put there is what survivor-based refusal
    exists to prevent.
    """
    total = 0.0
    for w in [1.0, 0.0, 2.0]:  # a disabled tap, spelled as a zero weight
        if w > 0.0:  # the guard the user wrote to avoid exactly the division below
            total = total + x / w
    return total


def _self_division_of_a_failing_expression(x: float) -> float:
    bad = x / 0.0
    return bad / bad


def _failure_discarded_by_a_frontend_shortcut(x: float) -> float:
    return x + math.sqrt(-1.0) ** 0


def test_a_failure_an_identity_deletes_is_not_refused() -> None:
    # ``x/x == 1`` answers the first kernel and ``**0`` the second, leaving the division by zero dead for DCE. Refusal
    # is over the SURVIVORS, so an expression no operation is left reading was never the program's to answer for.
    # Python has no answer for either -- it evaluates what the optimizer deletes and raises there -- which is the
    # charter's own divergence: what the optimizer deletes signals no error.
    three = FloatValue.from_float(FMT, 3.0)
    assert float(_sim(_self_division_of_a_failing_expression, "self_div_of_failure").run(three)[0]) == 1.0
    assert float(_sim(_failure_discarded_by_a_frontend_shortcut, "failure_discarded").run(three)[0]) == 4.0


def test_a_division_by_a_zero_constant_stays_a_division() -> None:
    # A fold answers only where it knows EVERY operand, and an unknown numerator over a zero divisor is not that: the
    # quotient goes to hardware, which asserts div0 on every run. That the compiler could have named the fault and did
    # not is the charter's license at work -- a missed refusal is never a defect -- and the constant twin (0.0/0.0,
    # pinned above) is what a fold does name. What must NOT happen is the reciprocal rewrite: there is no 1/0 to
    # multiply by, so the division has to survive as itself rather than become a multiply or a folded infinity.
    for kernel in (_div_by_zero_const, _div_by_zero_in_a_live_arm):
        verilog = holoso.synthesize(kernel, _ops(), name=kernel.__name__.lstrip("_")).verilog_output.verilog
        assert "holoso_fdiv #" in verilog, kernel
        assert "holoso_fmul #" not in verilog, kernel
    # A value nothing reads is deleted before the sweep sees it, and so is an arm a guard excludes -- whether that
    # guard is resolved by the front end or by HIR. Neither is in the program the sweep is given.
    # 4.0 rather than ``_dead_div_by_zero(3.0)``: Python evaluates the statement the optimizer deletes and raises.
    assert float(_sim(_dead_div_by_zero, "dead_div_zero").run(FloatValue.from_float(FMT, 3.0))[0]) == 4.0
    guarded = _sim(_div_by_zero_behind_a_folded_guard, "div_zero_behind_guard")
    assert float(guarded.run(FloatValue.from_float(FMT, 3.0))[0]) == 4.5


def _k_and(a: bool, b: bool) -> bool:
    return a and b


def _k_or(a: bool, b: bool) -> bool:
    return a or b


def _k_and_or(a: bool, b: bool, c: bool) -> bool:
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


def _demorgan_lhs(a: bool, b: bool) -> bool:
    return not (a and b)


def _demorgan_rhs(a: bool, b: bool) -> bool:
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


def _float_of_cond(x: float, y: float) -> float:
    # float(x > y) must be exactly 0.0 or 1.0; feeding it into arithmetic gives a clean gate.
    return float(x > y) * 10.0 + 1.0


def test_float_of_bool_is_exactly_zero_or_one_feeding_arithmetic() -> None:
    sim = _sim(_float_of_cond, "float_cond")
    for x, y in [(3.0, 1.0), (1.0, 3.0), (2.0, 2.0)]:
        got = sim.run(FloatValue.from_float(FMT, x), FloatValue.from_float(FMT, y))[0]
        assert isinstance(got, FloatValue)
        want = FloatValue.from_float(FMT, (10.0 if x > y else 0.0) + 1.0)
        assert got.bits == want.bits, f"float({x}>{y})*10+1 bits=0x{got.bits:x} vs 0x{want.bits:x}"


def _cross_domain_chain(x: float, y: float) -> float:
    # A float compared, the bool cast to float, then multiplied by a float: a full float->bool->float round trip.
    return float(x > y) * (x + y)


def test_compare_cast_multiply_cross_domain_chain() -> None:
    sim = _sim(_cross_domain_chain, "cross_chain")
    for x, y in [(3.0, 1.0), (1.0, 3.0), (2.0, 2.0), (-1.0, -2.0)]:
        got = sim.run(FloatValue.from_float(FMT, x), FloatValue.from_float(FMT, y))[0]
        assert isinstance(got, FloatValue)
        gate = 1.0 if x > y else 0.0
        want = FloatValue.from_float(FMT, _round(gate * _round(x + y)))  # the sum rounds, then the (exact) gate scales
        assert got.bits == want.bits, f"cross chain ({x},{y}) bits=0x{got.bits:x} vs 0x{want.bits:x}"


def _bool_of_float(x: float) -> bool:
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


def _fold_to_zero(x: float) -> float:
    # The subexpression 2*3 - 6 folds to 0.0 at compile time, so x + 0.0 == x exactly for every representable x.
    return x + (2.0 * 3.0 - 6.0)


def test_constant_subexpression_folds_to_zero() -> None:
    sim = _sim(_fold_to_zero, "fold_zero")
    for bits in _EDGES:
        out = sim.run(_val(bits))[0]
        assert isinstance(out, FloatValue)
        assert out.bits == bits, f"x + (2*3-6) changed bits: 0x{out.bits:x} vs 0x{bits:x}"


def _dead_arm_divides_by_zero(x: float) -> float:
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
        assert isinstance(got, FloatValue)
        want = FloatValue.from_float(FMT, _round(x + 1.0))
        assert got.bits == want.bits, f"x+1 (dead arm pruned): {x} bits=0x{got.bits:x} vs 0x{want.bits:x}"


class _AttributeConfig:
    """A read-only attribute snapshotted at synthesis; the condition that reads it must fold to its captured value."""

    def __init__(self, threshold: float) -> None:
        self._threshold = threshold

    def __call__(self, x: float) -> float:
        if self._threshold > 2.0:  # _threshold == 3.0 is a compile-time snapshot -> the condition folds to True
            r = x + 1.0
        else:
            r = x / 0.0  # would error if ever lowered; a correct result proves this arm was pruned
        return r


def test_read_only_attribute_folds_in_condition() -> None:
    sim = _sim(_AttributeConfig(3.0).__call__, "attr_fold")
    for x in (-2.0, 0.0, 3.0, 9.0):
        got = sim.run(FloatValue.from_float(FMT, x))[0]
        assert isinstance(got, FloatValue)
        want = FloatValue.from_float(FMT, _round(x + 1.0))
        assert got.bits == want.bits, f"attr-fold x+1: {x} bits=0x{got.bits:x} vs 0x{want.bits:x}"


def _horner_loop(x: float) -> float:
    # A bounded for-loop that fully unrolls a Horner evaluation of 1*x^4 + 1*x^3 + 1*x^2 + 1*x + 1.
    acc = 0.0
    for _ in range(5):
        acc = acc * x + 1.0
    return acc


def _horner_unrolled(x: float) -> float:
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
        assert isinstance(looped, FloatValue) and isinstance(flat, FloatValue)
        assert (
            looped.bits == flat.bits
        ), f"unrolled loop != straight-line at x={x}: 0x{looped.bits:x} vs 0x{flat.bits:x}"


def test_bounded_for_loop_polynomial_matches_reference() -> None:
    # An independent float64 reference (within tolerance) backs up the reference-free bit-identical check above.
    sim = _sim(_horner_loop, "horner_ref")
    for x in (-2.0, -0.5, 0.5, 1.0, 1.5, 2.0):
        got = float(sim.run(FloatValue.from_float(FMT, x))[0])
        want = (((1.0 * x + 1.0) * x + 1.0) * x + 1.0) * x + 1.0
        rtol, atol = default_tolerance(FMT, op_count=10, magnitude=max(1.0, abs(want)))
        assert within(got, want, rtol, atol), f"horner x={x}: {got} vs {want}"


_TINY = math.ldexp(1.0, -1022)  # module scope: a kernel resolves module-level constants, not enclosing locals


def _scale_past_the_carrier(c: bool) -> float:
    r = True
    if c:
        r = c
    return (4.0 * float(r)) / _TINY


def test_a_power_of_two_scale_past_the_carrier_folds_to_an_infinity() -> None:
    # Strength reduction can mint a power-of-two scale whose host-precision fold overflows; it folds to the operand's
    # own infinity rather than letting an OverflowError escape as a raw traceback.
    fmt = FloatFormat(11, 53)
    ops = Options(
        OperatorOptions(
            fadd=FAddOptions(),
            fmul=FMulOptions(),
            fdiv=FDivOptions(),
            fmul_ilog2=FMulILog2Options(),
            fcmp=FCmpOptions(),
        ),
        ffmt=fmt,
    )
    sim = holoso.synthesize(_scale_past_the_carrier, ops, name="pow2_carrier_overflow").numerical_model.elaborate()
    for c in (False, True):
        assert math.isinf(float(sim.run(c)[0])), f"c={c}"
