"""
Public-API, black-box behavioral tests for register-allocation and merge-resolution regressions.

Every test drives the compiler ONLY through the public API (`holoso.synthesize(fn, options)` and the elaborated
`numerical_model`) and asserts on observable values against a CPython reference or independent literals. These are
the behavior halves of allocator corners whose structural triggers (where one is needed to keep the corner exercised)
live in test_schedule.py; the tests here assert only build success and output/state correctness, so they survive a
deep refactor of the allocator, coalescer, and merge threading.
"""

import itertools
import math
from collections.abc import Callable

import holoso
from holoso import FloatFormat

from ._modelref import default_options

FMT = FloatFormat(6, 18)


def _sim(fn: Callable[..., object], name: str) -> holoso.NumericalSimulator:
    return holoso.synthesize(fn, default_options(FMT), name=name).numerical_model.elaborate()


def _check_float_kernel(fn: Callable[..., tuple[float, ...]], name: str, samples: list[tuple[float, ...]]) -> None:
    simulator = _sim(fn, name)  # crash-before: the install-free oracle admitted an unsound merge -> backstop assert
    for args in samples:
        got = [float(v) for v in simulator.run(*args)]
        ref = [float(v) for v in fn(*args)]
        assert len(got) == len(ref)
        for g, r in zip(got, ref):
            assert abs(g - r) <= 1e-2 * max(1.0, abs(r)), f"{name}{args}: {got} vs {ref}"


def test_back_edge_carried_merge_phi_kernel_builds_and_matches_reference() -> None:
    # Regression (review round 2, Codex): merge threading deletes a merge block's phis after composing the arm each
    # successor phi takes FROM the merge -- but ONLY that arm. A loop-invariant value the loop header carries on its
    # BACK-EDGE arm is a successor-phi arm too, yet from a different predecessor, so composition would not rewrite it;
    # deleting the merge phi would dangle. The guard must refuse such a merge (the deferred self-latch case).
    # Crash-before: optimize() raised KeyError after threading deleted the still-referenced merge phi.
    def loop_invariant_merge(a: float, den: float, c: float) -> float:
        if a > 0.0:
            x = a / den  # a real (non-speculatable) division branch -> a separate merge block holding phi x
        else:
            x = c
        z = 0.0
        while z < 1.0:  # x (the merge phi) is loop-invariant: carried on the loop header's back-edge arm, not rewritten
            z = x
        return z

    simulator = _sim(loop_invariant_merge, "loop_invariant_merge")
    for a, den, c in [(2.0, 2.0, 3.0), (-1.0, 4.0, 5.0), (3.0, 1.0, 0.0)]:  # x >= 1 so the latch loop terminates
        (got,) = simulator.run(a, den, c)
        assert math.isclose(float(got), loop_invariant_merge(a, den, c), rel_tol=1e-6)


def test_phi_coalescing_residual_install_conflict_is_resolved() -> None:
    # Regression: a phi (`a`) coalesces onto input `x`'s register because the install-free oracle sees no overlap,
    # yet `x` stays live in the else block as a sibling phi's identity arm (`z = x`) exactly where `a`'s residual
    # (sign-folded) else-arm install writes that shared register. The final, install-aware interference then flags the
    # class against itself and the coloring backstop aborted the build. The fixpoint must de-coalesce `a` and build.
    # The division keeps the diamond a real branch (un-if-converted), which is what creates the phi merge.
    def k(x: float, b: float, cc: float) -> tuple[float, float, float]:
        if b < cc:
            a = x
            z = 1.0
            d = b
        else:
            a = -(x + 1.0)
            z = x
            d = x / b
        return a, z, d

    _check_float_kernel(k, "coal_c1", [(2.0, 3.0, 5.0), (2.0, 3.0, 1.0), (-4.0, 2.0, 10.0), (1.5, 4.0, 0.5)])


def test_phi_coalescing_conflict_resolved_under_reversed_declaration_order() -> None:
    # The same hazard with the assignments and the return reversed: value ids -- hence the deterministic phi processing
    # order the union-find follows -- change, so a DIFFERENT phi wins the merge onto `x`. The fixpoint must converge
    # regardless of which phi coalesced first; this pins the resolution as order-independent, not an artifact of one id
    # assignment.
    def k(x: float, b: float, cc: float) -> tuple[float, float, float]:
        if b < cc:
            d = b
            z = 1.0
            a = x
        else:
            d = x / b
            z = x
            a = -(x + 1.0)
        return d, z, a

    _check_float_kernel(k, "coal_c2", [(2.0, 3.0, 5.0), (2.0, 3.0, 1.0), (-4.0, 2.0, 10.0), (1.5, 4.0, 0.5)])


def test_phi_coalescing_conflict_resolved_with_swapped_branch_arms() -> None:
    # The mirror: the coalescing identity arm sits in the else block and the sign-folded residual arm in the then block,
    # so the conflict is exercised from the opposite branch polarity. Confirms the de-coalescing is arm-order agnostic.
    def k(x: float, b: float, cc: float) -> tuple[float, float, float]:
        if b < cc:
            a = -(x + 1.0)
            z = x
            d = x / b
        else:
            a = x
            z = 1.0
            d = b
        return a, z, d

    _check_float_kernel(k, "coal_c3", [(2.0, 3.0, 5.0), (2.0, 3.0, 1.0), (-4.0, 2.0, 10.0), (1.5, 4.0, 0.5)])


def test_bool_phi_coalescing_residual_install_conflict_is_resolved() -> None:
    # The boolean-bank twin of the residual-install conflict: phi `a` coalesces onto input `q`'s 1-bit register
    # while `q` stays live as sibling phi `z`'s identity arm (`z = q`) where `a`'s residual (inverted) else-arm
    # install writes the shared register. A boolean phi keeps the diamond a real branch (bool phis are never
    # if-converted). The fixpoint must de-coalesce and build; checked bit-exact across all eight boolean input vectors.
    def k(p: bool, q: bool, r: bool) -> tuple[bool, bool, bool]:
        if p:
            a = q
            z = True
            d = r
        else:
            a = not q
            z = q
            d = q and r
        return a, z, d

    simulator = _sim(k, "coal_bool")  # crash-before: the bool oracle admitted the unsound merge -> backstop assert
    for p, q, r in itertools.product([False, True], repeat=3):
        got = list(simulator.run(p, q, r))
        for v in got:
            assert isinstance(v, bool)
        assert got == list(k(p, q, r)), f"coal_bool({p},{q},{r}): {got} vs {list(k(p, q, r))}"


def test_noop_state_writeback_streams_and_matches_reference() -> None:
    # The behavior half of test_schedule.py test_state_war_backstop_allows_noop_writeback: a no-op writeback (live-out
    # is the live-in value itself) must build and stream correctly -- the state holds its reset value while the output
    # tracks the input exactly (both are copies of representable values).
    class Hold:
        def __init__(self) -> None:
            self.s = 0.0

        def __call__(self, x: float) -> float:
            out = self.s + x
            self.s = self.s
            return out

    simulator = _sim(Hold().__call__, "hold_noop")
    reference = Hold()
    for x in (2.0, -3.0):
        got = simulator.run(x)
        want = reference(x)
        assert float(got[0]) == want, f"x={x}: {float(got[0])} vs {want}"
        assert float(got[1]) == reference.s  # the public slot's state_s port carries the (held) state


def test_write_only_state_slot_matches_reference() -> None:
    # The behavior half of test_schedule.py test_cfg_write_only_state_slot_is_reserved: a state slot written on every
    # arm but never read before the write. The returned value is the assign-and-return leaf of the public `acc`
    # state, so the model exposes it through the `state_acc` port; both arms' values are exact independent literals.
    class WriteOnlyBranch:
        def __init__(self) -> None:
            self.acc = 0.0

        def __call__(self, x: float) -> float:
            if x > 0.0:
                t = x * 2.0
            else:
                t = x * 3.0
            self.acc = -t
            return self.acc

    simulator = _sim(WriteOnlyBranch().__call__, "write_only")
    reference = WriteOnlyBranch()
    assert [p.name for p in simulator.outputs] == ["state_acc"]
    for x, want in [(3.0, -6.0), (-2.0, 6.0), (0.5, -1.0), (-1.5, 4.5)]:
        assert reference(x) == want
        assert float(simulator.run(x)[0]) == want, f"x={x}"


class _InvertedState:
    def __init__(self) -> None:
        self._flip = False

    def step(self, x: float) -> float:
        old = self._flip
        self._flip = not self._flip
        return x if old else -x


def test_bool_state_slot_carries_a_live_out_inversion() -> None:
    # The toggle's live-out is its own live-in inverted: the inversion rides the slot's install (even though the source
    # register IS the slot register), so the model must toggle across transactions and reset must restore the phase.
    simulator = _sim(_InvertedState().step, "toggle")
    reference = _InvertedState()
    for x in (1.0, 2.0, 3.0, 4.0):
        assert float(simulator.run(x)[0]) == reference.step(x)
    simulator.reset()
    fresh = _InvertedState()
    for x in (5.0, 6.0, 7.0):
        assert float(simulator.run(x)[0]) == fresh.step(x)
