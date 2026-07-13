"""
Public-API black-box tests that PIN early-return behavior across the shapes the FIR frontend must support: a return in
a dynamic branch, multiple return sites that must agree in type, a return inside a while body, a return inside an
inlined callee (including in the callee's own while body), a return in an unrolled ``for`` trip, a state write that
happens before an early return, and a bool-typed early return. The old single-pass frontend rejected these; the FIR
frontend lowers them via a hidden ReturnPlace and one canonical exit, so every path stores the return value and jumps
to the single Ret.

Each test drives the compiler only through ``holoso.synthesize(fn, ops)`` and the elaborated numerical simulator, and
asserts output VALUES against the same kernel evaluated in Python -- exact (``==``) wherever the arithmetic is a sign
flip, a doubling, or small-integer addition that ZKF represents exactly, so the assertions cannot pass on a rounding
accident. Type-disagreeing return sites are asserted to reject with a clean located error, not a raw crash.
"""

from collections.abc import Callable

import pytest

import holoso
from holoso import (
    FAddOperator,
    FCmpOperator,
    FDivOperator,
    FloatFormat,
    FMulILog2OperatorFamily,
    FMulOperator,
    OpConfig,
    UnsupportedConstruct,
)

FMT = FloatFormat(6, 18)


def _ops() -> OpConfig:
    return OpConfig(
        FAddOperator(FMT), FMulOperator(FMT), FDivOperator(FMT), FMulILog2OperatorFamily(FMT), FCmpOperator(FMT)
    )


def _sim(fn: Callable[..., object], name: str) -> holoso.NumericalSimulator:
    return holoso.synthesize(fn, _ops(), name=name).numerical_model.elaborate()


def _exact_over(fn: Callable[..., float], name: str, vectors: list[tuple[float, ...]]) -> None:
    """Every listed vector must reproduce the Python reference bit-exactly (the kernels are chosen exact in ZKF)."""
    sim = _sim(fn, name)
    for vec in vectors:
        got = float(sim.run(*vec)[0])
        want = float(fn(*vec))
        assert got == want, f"{name}{vec}: got {got}, want {want}"


def test_dynamic_arm_early_return_matches_at_and_around_the_boundary() -> None:
    def abs_via_early_return(x: float) -> float:
        if x < 0.0:
            return -x
        return x

    # 0.0 is the exact decision boundary of ``x < 0.0`` (takes the else arm, so +0.0); sign flips are exact in ZKF.
    _exact_over(abs_via_early_return, "abs_early", [(-2.0,), (-0.5,), (0.0,), (0.5,), (2.0,)])


def test_three_return_sites_agree_and_select_the_right_one() -> None:
    def tiered(x: float) -> float:
        if x > 10.0:
            return 100.0
        if x > 0.0:
            return x
        return -x

    _exact_over(tiered, "tiered", [(20.0,), (10.0,), (5.0,), (0.0,), (-3.0,)])


def test_return_inside_while_body_takes_the_first_crossing() -> None:
    def accumulate_until(step: float) -> float:
        acc = 0.0
        while acc < 100.0:
            acc = acc + step
            if acc > 5.0:
                return acc
        return acc

    # step 3 -> 6 (first crossing); step 1 -> 6 after six adds; step 8 -> 8. All sums are exact integers in ZKF.
    _exact_over(accumulate_until, "accum_until", [(3.0,), (1.0,), (8.0,)])


def _clamp_high(v: float) -> float:
    if v > 8.0:
        return 8.0
    return v


def test_inlined_callee_early_return_then_more_work() -> None:
    def kernel(x: float) -> float:
        return _clamp_high(x) + 1.0

    _exact_over(kernel, "inlined_clamp", [(10.0,), (8.0,), (2.0,), (-1.0,)])


def _drain_to_below_two(v: float) -> float:
    acc = v
    while acc > 1.0:
        acc = acc - 1.0
        if acc < 2.0:
            return acc
    return acc


def test_inlined_callee_with_a_return_inside_its_while_body() -> None:
    def kernel(x: float) -> float:
        return _drain_to_below_two(x) * 2.0

    # 5 -> drains 5,4,3,2 then 2<2 false, 1<2 true at acc=1 -> 1*2=2 ; 1.5 -> 0.5<2 -> 1.0 ; 0.5 -> loop skipped -> 1.0
    _exact_over(kernel, "inlined_drain", [(5.0,), (1.5,), (0.5,)])


def test_early_return_in_an_unrolled_for_trip() -> None:
    def ramp_capped(x: float) -> float:
        acc = x
        for _ in range(4):
            acc = acc + 1.0
            if acc > 2.0:
                return acc
        return acc

    _exact_over(ramp_capped, "ramp_capped", [(0.0,), (2.0,), (-5.0,)])


def test_return_in_both_diamond_arms() -> None:
    def spread(x: float, y: float) -> float:
        if x > y:
            return x - y
        else:
            return y - x

    _exact_over(spread, "spread", [(5.0, 1.0), (1.0, 5.0), (3.0, 3.0)])


def test_state_write_before_an_early_return_persists_across_transactions() -> None:
    class Accumulator:
        def __init__(self) -> None:
            self.total = 0.0

        def __call__(self, x: float) -> float:
            self.total = self.total + x  # the state write happens BEFORE the early return
            if self.total > 3.0:
                return self.total
            return 0.0

    sim = _sim(Accumulator().__call__, "accum_state")
    reference = Accumulator()
    # The state must accumulate across the stream regardless of which return site fires each cycle.
    for x in (1.0, 1.0, 2.0, 1.0, -5.0, 4.0):
        got = float(sim.run(x)[0])
        want = float(reference(x))
        assert got == want, f"accum x={x}: got {got}, want {want}"


def test_bool_typed_early_return_agreement() -> None:
    def in_band(x: float) -> bool:
        if x < 0.0:
            return False
        if x > 1.0:
            return False
        return True

    sim = _sim(in_band, "in_band")
    for x in (-1.0, 0.0, 0.5, 1.0, 2.0):
        assert bool(sim.run(x)[0]) == in_band(x), f"in_band x={x}"


def test_return_sites_of_disagreeing_type_are_rejected() -> None:
    # A float site and a bool site cannot share the single return place: this must be a clean located rejection, never
    # a raw crash deep in emission.
    def mixed(x: float) -> float:
        if x > 0.0:
            return x
        return x > 1.0

    with pytest.raises(UnsupportedConstruct, match="irreconcilable kinds"):
        holoso.synthesize(mixed, _ops(), name="mixed_sites")


def test_state_written_on_only_one_early_return_path_carries_across_the_exit() -> None:
    # The state slot is written on the fall-through path but NOT the early-return path, so the canonical exit needs a
    # NON-trivial state-slot phi (old value on the early-return arm, updated value on the other). This early-return x
    # state-merge intersection is the most likely place a slot/phi refactor would silently regress.
    class PartialWrite:
        def __init__(self) -> None:
            self.s = 0.0

        def __call__(self, x: float) -> float:
            if x > 10.0:
                return x  # self.s is NOT written on this path
            self.s = self.s + x
            return self.s

    sim = _sim(PartialWrite().__call__, "partial_write")
    reference = PartialWrite()
    for x in (1.0, 2.0, 20.0, 3.0, 15.0, -4.0):  # the 20 and 15 take the early return and must leave state untouched
        assert float(sim.run(x)[0]) == float(reference(x)), f"partial_write x={x}"


def test_inlined_helper_method_that_writes_self_and_early_returns() -> None:
    # A member method calls a helper METHOD that writes self AND early-returns; the outer method also reads the same
    # attribute. State-slot analysis must span the inline expansion whose callee itself early-returns.
    class BumpThenRead:
        def __init__(self) -> None:
            self.s = 0.0

        def _bump(self, x: float) -> float:
            self.s = self.s + x
            if self.s > 5.0:
                return self.s
            return 0.0

        def __call__(self, x: float) -> float:
            y = self._bump(x)
            return y + self.s

    sim = _sim(BumpThenRead().__call__, "bump_then_read")
    reference = BumpThenRead()
    for x in (2.0, 2.0, 2.0, -5.0, 4.0):
        assert float(sim.run(x)[0]) == float(reference(x)), f"bump_then_read x={x}"


def test_early_return_crosses_both_levels_of_a_nested_loop() -> None:
    def nested(x: float) -> float:
        acc = x
        while acc < 100.0:
            inner = 0.0
            while inner < 10.0:
                inner = inner + 1.0
                acc = acc + 1.0
                if acc > 5.0:
                    return acc  # exits BOTH loops at once
            acc = acc + 0.5
        return acc

    # x=0 -> counts up to 6 across the inner trips; x=3 -> 6; x=100 -> outer never runs, returns 100. All exact.
    _exact_over(nested, "nested_early", [(0.0,), (3.0,), (100.0,)])


def test_void_return_type_updates_state_only_on_the_written_path() -> None:
    # A ``-> None`` method with a bare early return has no out_0 data port; the persistent state is the only
    # observable, written on one path and skipped on the early-return path.
    class VoidBump:
        def __init__(self) -> None:
            self.s = 0.0

        def __call__(self, x: float) -> None:
            if x > 0.0:
                return  # bare early return, no write
            self.s = self.s + x

    sim = _sim(VoidBump().__call__, "void_bump")
    assert [p.name for p in sim.outputs] == ["state_s"]  # no data return port, only the persistent state
    reference = VoidBump()
    for x in (1.0, -2.0, 3.0, -4.0, -1.0):
        got_state = float(sim.run(x)[0])
        reference(x)
        assert got_state == float(reference.s), f"void_bump x={x}"
