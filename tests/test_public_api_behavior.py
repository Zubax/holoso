"""
Public-API, black-box behavioral tests for if-conversion and select forms, NOT-folding in every consumer position,
phi-arm coalescing, typed ports and multi-output kernels, and persistent boolean state.

Every test drives the compiler ONLY through the public API: `holoso.synthesize(fn, ops) -> SynthesisResult`, then
`result.numerical_model.elaborate() -> NumericalSimulator`, and exercises the simulator (`run` / `reset` /
typed `inputs` / `outputs` metadata). Assertions are on OBSERVABLE behavior only: output values against an
INDEPENDENT reference (the same kernel evaluated in Python float64 within the format tolerance, two mathematically-
equivalent formulations that must agree, or a hand-computed exact value), persistent state across transactions
including a `reset` partway, and the typed-port metadata. No internal LIR / schedule / register / cycle-count
structure is inspected, so these survive a deep refactor of any mid/back-end pass. Complements
test_overlap_behavior.py (the cross-block-overlap surface) and test_arithmetic_behavior.py (numerical behavior and
constant folding).
"""

from collections.abc import Callable

import pytest

import holoso
from holoso import (
    BoolType,
    FAddOptions,
    FCmpOptions,
    FDivOptions,
    FMulILog2Options,
    FMulOptions,
    FloatFormat,
    FloatType,
    OperatorOptions,
    Options,
)

from ._modelref import default_tolerance, within

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


def _close(got: float, want: float, op_count: int = 12) -> bool:
    """Reduced-precision agreement for a kernel of about `op_count` ZKF ops over operands of order unity-to-ten."""
    rtol, atol = default_tolerance(FMT, op_count, magnitude=max(1.0, abs(want)))
    return within(got, want, rtol, atol)


def _abs_via_select(x: float, c: float) -> float:
    # x if c>0 else -x : the negation folds into the select's operand sign conditioner -- one compare, one mux.
    return x if c > 0.0 else -x


def _neg_abs_via_select(x: float, c: float) -> float:
    # The sign rides the OTHER arm, exercising the conditioner on the opposite operand. Both arms are sign chains over
    # the same input, so the whole diamond collapses to one select.
    return -x if c > 0.0 else x


def test_sign_folds_into_select_both_orientations() -> None:
    # Both outputs are sign-selected copies of a representable input, so the comparison: exact `==` on both sides of
    # the boundary and AT the boundary (c == 0.0 takes the else arm, matching Python's `> 0.0`).
    sim_a = _sim(_abs_via_select, "sel_abs")
    sim_b = _sim(_neg_abs_via_select, "sel_neg_abs")
    for x in (-2.0, -0.5, 0.0, 0.5, 2.0):
        for c in (-1.0, 0.0, 0.5, 2.0):  # 0.0 is the exact decision boundary of `c > 0.0`
            assert float(sim_a.run(x, c)[0]) == _abs_via_select(x, c), f"abs x={x} c={c}"
            assert float(sim_b.run(x, c)[0]) == _neg_abs_via_select(x, c), f"neg_abs x={x} c={c}"


def _cmp_select_float_chain(x: float, y: float) -> float:
    # max(x, y) via a pure compare->select diamond, then a float multiply -- a cross-bank chain (boolean comparison
    # feeding a wide select feeding a wide multiply) that stays STRAIGHT-LINE (no branch). Doubling is exact in ZKF.
    return (x if x > y else y) * 2.0


def test_comparison_select_float_chain_stays_straight_line() -> None:
    sim = _sim(_cmp_select_float_chain, "cmp_sel_mul")
    for x in (-2.0, -0.5, 0.5, 2.0):
        for y in (-2.0, 0.5, 2.0):
            # x == y across the boundary too: when equal the select takes the else arm (y), and 2*x == 2*y anyway.
            for xy in ((x, y), (x, x)):
                got = float(sim.run(*xy)[0])
                assert got == _cmp_select_float_chain(*xy), f"xy={xy}: {got}"


def _diff_arith_select(x: float, y: float, c: float) -> float:
    # A select whose two arms are DIFFERENT arithmetic (product vs sum), both speculatable -> if-converts to a select
    # over two distinct sub-DAGs.
    return (x * y) if c > 0.0 else (x + y)


def test_select_arms_different_arithmetic() -> None:
    sim = _sim(_diff_arith_select, "sel_diff_arith")
    for x in (-2.0, 0.5, 3.0):
        for y in (-1.0, 2.0):
            for c in (-1.0, 0.0, 1.5):
                got = float(sim.run(x, y, c)[0])
                want = _diff_arith_select(x, y, c)
                assert _close(got, want, op_count=4), f"x={x} y={y} c={c}: {got} vs {want}"


def _nested_ternary(x: float, y: float) -> float:
    # A nested ternary in expression position: the inner sign-select is one arm of the outer select. All four leaves
    # are sign chains of representable inputs, so every result is exact.
    return (x if x > 0.0 else -x) if y > 0.0 else (y if x > 0.0 else -y)


def test_nested_ternary_select() -> None:
    sim = _sim(_nested_ternary, "nested_tern")
    for x in (-2.0, 0.0, 0.5, 2.0):
        for y in (-2.0, 0.0, 0.5, 2.0):
            assert float(sim.run(x, y)[0]) == _nested_ternary(x, y), f"x={x} y={y}"


def _ifconv_division_form(x: float, y: float, c: float) -> float:
    # IF-CONVERTIBLE formulation: the diamond merges two speculatable arms (add/sub), then ONE division outside it.
    # The select collapses the diamond to straight-line code; the division is not inside an arm, so nothing blocks it.
    m = (x + y) if c > 0.0 else (x - y)
    return m / (y * y + 1.0)  # structurally-nonzero, non-power-of-two divisor


def _real_branch_division_form(x: float, y: float, c: float) -> float:
    # REAL-BRANCH formulation of the SAME math: each arm carries the (unspeculatable) division, so the diamond CANNOT
    # if-convert and stays a genuine branch -- yet the value computed on each path is identical to the if-converted
    # form. The two must agree (a strong cross-check: one compiles to a select + one divide, the other to a branch
    # with a divide per arm). The divisor is structurally nonzero and non-power-of-two so it is not strength-reduced
    # to a multiply (which would be speculatable and re-collapse the branch).
    if c > 0.0:
        r = (x + y) / (y * y + 1.0)
    else:
        r = (x - y) / (y * y + 1.0)
    return r


def test_ifconvertible_and_real_branch_forms_agree() -> None:
    # Same operands and the same single division on each path: the if-converted (select) form and the real-branch form
    # must agree on both polarities and at the boundary. They additionally happen to be bit-identical here, which we
    # assert as a stronger bonus check; if a future heuristic change broke that, the `_close` agreement still stands.
    # The structural difference is compile-time, so a directed set suffices: both branch polarities, the c == 0
    # boundary, both signs of each operand, and a nontrivial (non-power-of-two) divisor via y.
    sim_ifc = _sim(_ifconv_division_form, "ifc_div")
    sim_br = _sim(_real_branch_division_form, "br_div")
    samples = [
        (2.0, 3.0, 1.0),
        (2.0, 3.0, -1.0),
        (2.0, 3.0, 0.0),
        (-1.0, 2.0, 0.5),
        (-2.0, -3.0, 1.0),
        (2.0, -3.0, -0.5),
        (-2.0, 3.0, 0.0),
        (0.0, 0.5, 1.0),
        (1.5, -0.5, -2.0),
    ]
    for x, y, c in samples:
        a = float(sim_ifc.run(x, y, c)[0])
        b = float(sim_br.run(x, y, c)[0])
        want = _real_branch_division_form(x, y, c)
        assert _close(a, want, op_count=6), f"ifc x={x} y={y} c={c}: {a} vs {want}"
        assert _close(b, want, op_count=6), f"branch x={x} y={y} c={c}: {b} vs {want}"
        assert a == b, f"forms disagree x={x} y={y} c={c}: ifc={a} branch={b}"


# `not` never materializes hardware -- it folds into the consumer's sideband on the CONSUMER side.


def _not_in_branch_condition(c: bool, x: float, y: float) -> float:
    if not c:
        r = x + y
    else:
        r = x - y
    return r


def test_not_as_branch_condition() -> None:
    sim = _sim(_not_in_branch_condition, "not_branch")
    for c in (True, False):
        for x, y in [(2.0, 1.0), (-3.0, 4.0)]:
            got = float(sim.run(c, x, y)[0])
            assert got == _not_in_branch_condition(c, x, y), f"c={c} x={x} y={y}: {got}"


def _not_in_boolean_logic(a: bool, b: bool) -> bool:
    return a and not b


def test_not_in_boolean_logic_operand() -> None:
    sim = _sim(_not_in_boolean_logic, "not_and")
    for a in (True, False):
        for b in (True, False):
            got = sim.run(a, b)[0]
            want = _not_in_boolean_logic(a, b)
            assert got is want, f"a={a} b={b}: {got} vs {want}"


def _not_as_output(c: bool) -> bool:
    return not c


def test_not_as_boolean_output() -> None:
    sim = _sim(_not_as_output, "not_out")
    assert [(p.name, p.scalar_type) for p in sim.outputs] == [("out_0", BoolType())]
    assert sim.run(True)[0] is False
    assert sim.run(False)[0] is True


class _BoolNotState:
    """A boolean state slot driven by `not`: `self.flag = not self.flag` (the phi/state-slot inversion)."""

    def __init__(self) -> None:
        self.flag = False

    def __call__(self) -> bool:
        self.flag = not self.flag
        return self.flag


def test_not_drives_boolean_state_slot() -> None:
    sim = _sim(_BoolNotState().__call__, "not_state")
    reference = _BoolNotState()
    for _ in range(8):
        assert sim.run()[0] is reference()


def _not_in_phi_arm(c: bool, d: bool) -> bool:
    return (not d) if c else d


def test_not_in_boolean_phi_arm() -> None:
    sim = _sim(_not_in_phi_arm, "not_phi")
    for c in (True, False):
        for d in (True, False):
            got = sim.run(c, d)[0]
            want = _not_in_phi_arm(c, d)
            assert got is want, f"c={c} d={d}: {got} vs {want}"


def _double_negation(c: bool) -> bool:
    # `not not c` must fold to the identity (two consumer-side inversions cancel).
    return not not c


def test_double_negation_is_identity() -> None:
    sim = _sim(_double_negation, "not_not")
    assert sim.run(True)[0] is True
    assert sim.run(False)[0] is False


def _comparison_both_polarities(x: float, y: float) -> tuple[bool, float]:
    # ONE comparison consumed in BOTH polarities: `x > y` drives a boolean output directly, and `not (x > y)` (the
    # complement) gates a float select. One comparator tap must serve both -- the polarities must stay consistent.
    gt = x > y
    sel = 10.0 if gt else 20.0
    other = 1.0 if not gt else 0.0
    return gt, sel + other


def test_comparison_in_both_polarities() -> None:
    sim = _sim(_comparison_both_polarities, "both_pol")
    for x in (-1.0, 0.0, 1.0, 2.0):
        for y in (-1.0, 0.0, 1.0):
            for xy in ((x, y), (x, x)):  # include x == y, where `x > y` is False and `not` is True
                got = sim.run(*xy)
                want_gt, want_val = _comparison_both_polarities(*xy)
                assert got[0] is want_gt, f"xy={xy}: bool {got[0]} vs {want_gt}"
                assert float(got[1]) == want_val, f"xy={xy}: val {float(got[1])} vs {want_val}"


def _pure_recurrence(x: float, n: float) -> float:
    # A PURE loop-carried recurrence (no division -- the existing in-loop test is division-bearing): a Horner-like
    # multiply-accumulate over a data-dependent trip count. The header phi for `acc` is loop-carried; its live-in
    # overlaps its back-edge arm, so the coalescer must judge it correctly (it keeps a copy where it must, coalesces
    # where it can) without changing the value.
    acc = 0.0
    i = n
    while i > 0.0:
        acc = acc * x + 1.0  # bounded by choosing |x| <= 1 below so a long stream stays in-range
        i = i - 1.0
    return acc


def test_pure_loop_carried_recurrence_matches_reference() -> None:
    sim = _sim(_pure_recurrence, "pure_recur")
    for x, n in [(0.5, 0.0), (0.5, 1.0), (0.5, 4.0), (-0.75, 5.0), (1.0, 3.0), (0.0, 6.0), (-1.0, 4.0)]:
        got = float(sim.run(x, n)[0])
        want = _pure_recurrence(x, n)
        # Each trip is one mul + one add; tolerance scales with the realized trip count.
        assert _close(got, want, op_count=2 * max(1, int(n)) + 2), f"x={x} n={n}: {got} vs {want}"


def _soundness_corner(x: float, n: float) -> float:
    # The coalescing soundness corner: a REAL diamond INSIDE a loop whose arms reuse an input (`x`) that is ALSO the
    # seed of the phi result (`acc` is seeded from x and is the loop-carried header phi), branching ON that phi
    # result (`acc > 0.0`). The else arm divides (structurally-nonzero, non-power-of-two divisor) so the diamond
    # CANNOT if-convert and stays a genuine branch with phi arms and install copies -- the corner where a wrong
    # coalesce would clobber a value (the carry / input `x`) still live in a sibling arm. Output must follow the
    # source semantics, not whatever the allocator chose. Unlike the overlap file's in-loop diamond (whose condition
    # is the loop-invariant `x > 1.0`), the condition here is the loop-carried phi itself.
    acc = x
    i = n
    while i > 0.0:
        acc = (acc + x) if acc > 0.0 else (acc - x) / (x * x + 1.0)
        i = i - 1.0
    return acc


def test_coalescing_soundness_corner_matches_reference() -> None:
    sim = _sim(_soundness_corner, "soundness")
    for x, n in [(1.5, 0.0), (1.5, 3.0), (-1.5, 3.0), (0.5, 5.0), (-0.5, 4.0), (2.0, 2.0), (-2.0, 6.0)]:
        got = float(sim.run(x, n)[0])
        want = _soundness_corner(x, n)
        assert _close(got, want, op_count=2 * max(1, int(n)) + 2), f"x={x} n={n}: {got} vs {want}"


def _mixed_tuple_io(flag: bool, x: float, y: float) -> tuple[bool, float, float]:
    # The two float outputs are exact (sum/difference of representable inputs).
    inside = flag and (x > y)
    return inside, x + y, x - y


def test_mixed_tuple_io_metadata_and_values() -> None:
    sim = _sim(_mixed_tuple_io, "mixed_tuple")
    assert [(p.name, p.scalar_type) for p in sim.inputs] == [
        ("flag", BoolType()),
        ("x", FloatType(FMT)),
        ("y", FloatType(FMT)),
    ]
    assert [(p.name, p.scalar_type) for p in sim.outputs] == [
        ("out_0", BoolType()),
        ("out_1", FloatType(FMT)),
        ("out_2", FloatType(FMT)),
    ]
    for flag in (True, False):
        for x in (-1.0, 0.5, 2.0):
            for y in (0.5, 2.0):
                for xy in ((x, y), (x, x)):  # include x == y, where x > y is False
                    got = sim.run(flag, *xy)
                    inside, total, diff = _mixed_tuple_io(flag, *xy)
                    assert got[0] is inside, f"flag={flag} xy={xy}: bool {got[0]} vs {inside}"
                    assert float(got[1]) == total, f"flag={flag} xy={xy}: {float(got[1])} vs {total}"
                    assert float(got[2]) == diff, f"flag={flag} xy={xy}: {float(got[2])} vs {diff}"


def _mixed_list_io(flag: bool, x: float) -> tuple[float, ...]:
    # A LIST-literal return (vs the tuple above): the same aggregate-flattening path through a list, with a bool cast
    # into a float lane.
    return [x + 1.0, float(flag)]  # type: ignore[return-value]  # a list literal is the point; sequences are one kind


def test_mixed_list_io_metadata_and_values() -> None:
    sim = _sim(_mixed_list_io, "mixed_list")
    assert [(p.name, p.scalar_type) for p in sim.inputs] == [("flag", BoolType()), ("x", FloatType(FMT))]
    assert [(p.name, p.scalar_type) for p in sim.outputs] == [("out_0", FloatType(FMT)), ("out_1", FloatType(FMT))]
    for flag in (True, False):
        for x in (-2.0, 0.0, 3.0):
            got = sim.run(flag, x)
            assert float(got[0]) == x + 1.0, f"flag={flag} x={x}: {float(got[0])}"
            assert float(got[1]) == float(flag), f"flag={flag} x={x}: {float(got[1])}"


def _unused_bool_input(flag: bool, x: float, y: float) -> float:
    # An UNUSED boolean input: it is a real port but the float result must not depend on it at all.
    return x * y + 1.0


def test_unused_bool_input_is_neutral() -> None:
    sim = _sim(_unused_bool_input, "unused_bool")
    assert [(p.name, p.scalar_type) for p in sim.inputs] == [
        ("flag", BoolType()),
        ("x", FloatType(FMT)),
        ("y", FloatType(FMT)),
    ]
    assert [(p.name, p.scalar_type) for p in sim.outputs] == [("out_0", FloatType(FMT))]
    for x in (-2.0, 0.5, 3.0):
        for y in (-1.0, 2.0):
            with_true = float(sim.run(True, x, y)[0])
            with_false = float(sim.run(False, x, y)[0])
            want = _unused_bool_input(True, x, y)
            assert with_true == with_false, f"unused flag perturbed output x={x} y={y}: {with_true} vs {with_false}"
            assert _close(with_true, want, op_count=2), f"x={x} y={y}: {with_true} vs {want}"


class _BoolStateMachine:
    """A boolean state slot updated by a comparison and consumed both as state and to gate the float output."""

    def __init__(self) -> None:
        self.armed = False

    def __call__(self, x: float) -> tuple[float, bool]:
        # latch `armed` once x crosses 1.0, and keep it latched; the float output is gated on the PREVIOUS armed.
        gated = 100.0 if self.armed else x
        if x > 1.0:
            self.armed = True
        return gated, self.armed


def test_boolean_state_slot_stream_and_reset() -> None:
    sim = _sim(_BoolStateMachine().__call__, "bool_state_machine")
    reference = _BoolStateMachine()
    stream = [0.5, 0.9, 1.5, 0.2, 2.0, -1.0, 0.3]  # crosses the latch threshold, then stays latched
    for x in stream:
        got = sim.run(x)
        want_gated, want_armed = reference(x)
        # The gated value is either a passed-through input (rounded to ZKF) or the exact constant 100.0; `_close`
        # covers both. The armed bool is the load-bearing state assertion and stays exact.
        assert _close(float(got[0]), want_gated, op_count=2), f"gated at x={x}: {float(got[0])} vs {want_gated}"
        assert got[1] is want_armed, f"armed at x={x}: {got[1]} vs {want_armed}"
    sim.reset()
    fresh = _BoolStateMachine()
    for x in [0.5, 2.0, 0.1]:  # the latch must be cleared by reset, then re-arm
        got = sim.run(x)
        want_gated, want_armed = fresh(x)
        assert _close(float(got[0]), want_gated, op_count=2), f"post-reset gated at x={x}: {float(got[0])}"
        assert got[1] is want_armed, f"post-reset armed at x={x}: {got[1]} vs {want_armed}"


def _lt_kernel(x: float, y: float) -> bool:
    return x < y


def _gt_swapped_kernel(x: float, y: float) -> bool:
    return y > x


def test_commuted_comparisons_agree_bool_exact() -> None:
    # `x < y` and `y > x` are the same predicate; the comparator's gt/lt flags transpose under operand swap, so a
    # commutative orientation that reorders the operands must keep the result bit-identical. Bool-exact, including the
    # equality boundary where both are False.
    sim_lt = _sim(_lt_kernel, "lt_kernel")
    sim_gt = _sim(_gt_swapped_kernel, "gt_swapped")
    for x in (-1.0, 0.0, 0.5, 1.0, 2.0):
        for y in (-1.0, 0.0, 1.0, 2.0):
            for xy in ((x, y), (x, x)):  # x == y boundary
                a = sim_lt.run(*xy)[0]
                b = sim_gt.run(*xy)[0]
                want = xy[0] < xy[1]
                assert a is want and b is want, f"xy={xy}: lt={a} gt={b} want={want}"


def test_out_of_range_operator_stage_knob_is_rejected_at_synthesis() -> None:
    # Stage knobs are public, documented configuration, so an out-of-range value must surface as a diagnostic through
    # `synthesize` itself rather than deep in operator construction.
    def add(a: float, b: float) -> float:
        return a + b

    options = Options(OperatorOptions(fadd=FAddOptions(stage_decode=7)), ffmt=FMT)
    with pytest.raises(ValueError, match="outside"):
        holoso.synthesize(add, options, name="bad_stage_knob")
