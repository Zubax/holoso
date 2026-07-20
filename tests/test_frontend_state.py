"""Frontend tests: component state -- slots, resets, stores, read-only folding, provenance, and state ports."""

import dataclasses
import re
import sys
import types
from pathlib import Path
from typing import cast

import inspect

import numpy as np
import pytest

import holoso
from holoso import FloatFormat, SourceUnavailable, UnsupportedConstruct
from holoso._frontend import lower
from holoso._frontend._fir._ir import LocatedRejection
from holoso._hir import Branch, FloatAdd, FloatConst, FloatDiv, IntType, Operation, optimize, Phi, Select, StateRead

from ._frontend_common import (
    _INEXACT_INTEGER as _INEXACT_INTEGER,
    _BIG_A as _BIG_A,
    _BIG_B as _BIG_B,
    _INT_TABLE as _INT_TABLE,
    _BIG_F as _BIG_F,
    _assert_shape_kernel_matches_python as _assert_shape_kernel_matches_python,
)
from ._modelref import arith_count as _arith_count, default_ops


def _multiply(a: float, b: float) -> float:
    """A module-level callable, so a call to it takes the inlining graft path a nested def would too."""
    return a * b


def test_dead_arm_attr_write_does_not_block_readonly_fold() -> None:
    # Regression (Codex): a write to a read-only boolean attribute inside a statically-dead `if False:` arm must not
    # mark it as assigned -- otherwise the attribute is wrongly treated as runtime and a later guard on it is not
    # folded, spuriously rejecting a return that the fold would have made unreachable.
    class DeadFlagGuard:
        def __init__(self) -> None:
            self._flag = False
            self.y = 0.0

        def __call__(self, x: float) -> float:
            if self._flag:  # _flag is read-only False -> folds away; the return arm is dead
                return x
            if False:
                self._flag = True  # noqa -- dead arm: must not count as assigning _flag
            self.y = x
            return self.y

    hir = lower(DeadFlagGuard().__call__)  # must not raise (the read-only fold removes the return-in-branch)
    assert [slot.name for slot in hir.state_slots] == ["y"]  # only y is state; _flag stays a read-only constant


def test_static_comparison_dead_arm_does_not_block_readonly_fold() -> None:
    # Regression (Codex): a write under a statically-false COMPARISON guard (not just a literal bool) must not mark the
    # attribute as assigned -- the read-only scan folds any attribute-free statically-known condition, as lowering does.
    class StaticCmpDeadFlag:
        def __init__(self) -> None:
            self._flag = False
            self.y = 0.0

        def __call__(self, x: float) -> float:
            if 1.0 < 0.0:
                self._flag = True  # noqa -- dead arm (statically-false comparison): must not assign _flag
            if self._flag:
                return x
            self.y = x
            return self.y

    hir = lower(StaticCmpDeadFlag().__call__)  # must not raise; _flag stays read-only so the return arm folds away
    assert [slot.name for slot in hir.state_slots] == ["y"]


def test_zero_trip_for_write_does_not_mark_attribute_assigned() -> None:
    # Regression (user): a write inside `for _ in range(0)` never executes, so the read-only scan must not count it as
    # an assignment -- otherwise a later guard on the attribute becomes a runtime branch and a return in the (actually
    # dead) arm is wrongly rejected. The scan mirrors the static trip count, as lowering and the state scan do.
    class ZeroForFlag:
        def __init__(self) -> None:
            self._flag = False
            self.y = 0.0

        def __call__(self, x: float) -> float:
            for _ in range(0):
                self._flag = True  # noqa -- zero-trip loop: never runs
            if self._flag:
                return x
            self.y = x
            return self.y

    hir = lower(ZeroForFlag().__call__)
    assert [slot.name for slot in hir.state_slots] == ["y"]  # _flag stays a read-only constant; only y is state


def test_zero_trip_self_attr_range_write_does_not_mark_attribute_assigned() -> None:
    # A read-only integer attribute used as a static range bound must be visible to the read-only assignment scan too.
    # Otherwise the scan treats the zero-trip loop body as reachable and later fails to fold the read-only flag guard.
    class ZeroSelfRangeFlag:
        def __init__(self) -> None:
            self.iterations = 0
            self._flag = False
            self.y = 0.0

        def __call__(self, x: float) -> float:
            for _ in range(self.iterations):
                self._flag = True  # noqa -- zero-trip loop: never runs
            if self._flag:
                return x
            self.y = x
            return self.y

    hir = lower(ZeroSelfRangeFlag().__call__)
    assert [slot.name for slot in hir.state_slots] == ["y"]


def test_constant_boolean_attribute_branch_folds() -> None:
    # Regression (Codex): a branch on a read-only boolean attribute has a compile-time-known condition; only the taken
    # arm lowers, so a write in the dead arm does not become spurious persistent state.
    class Disabled:
        def __init__(self) -> None:
            self.flag = False
            self.y = 0.0

        def __call__(self, x: float) -> float:
            if self.flag:
                self.y = x
            return self.y

    hir = lower(Disabled().__call__)
    assert [slot.name for slot in hir.state_slots] == []  # folded: y never written, no state, no branch
    assert len(optimize(hir).blocks) == 1


def test_numpy_boolean_attribute_branch_folds() -> None:
    # Regression (Codex): a read-only np.bool_ attribute must fold like a Python bool (it is exposed as boolean state
    # elsewhere), so the disabled arm's write does not become spurious state.
    class NpDisabled:
        def __init__(self) -> None:
            self.flag = np.bool_(False)
            self.y = 0.0

        def __call__(self, x: float) -> float:
            if self.flag:
                self.y = x
            return self.y

    hir = lower(NpDisabled().__call__)
    assert [slot.name for slot in hir.state_slots] == []
    assert len(optimize(hir).blocks) == 1


def test_static_integer_comparison_branch_folds() -> None:
    # Regression (Codex): a comparison of static integers (an unrolled loop counter against a bound) is known at
    # compile time and folds to one arm; a write gated by a statically-false guard must not become spurious state, and
    # no dynamic branch is emitted (integers are exact in any ZKF format, so the fold matches the comparator).
    class GuardAlwaysFalse:
        def __init__(self) -> None:
            self.x = 0.0

        def __call__(self, v: float) -> float:
            for i in range(3):
                if i > 5:  # never true over range(3): x must not become state
                    self.x = v
            return self.x

    folded = lower(GuardAlwaysFalse().__call__)
    assert [slot.name for slot in folded.state_slots] == []
    assert len(optimize(folded).blocks) == 1

    class GuardReal:
        def __init__(self) -> None:
            self.acc = 0.0

        def __call__(self, v: float) -> float:
            for i in range(3):
                if i > 0:  # true for i in {1, 2}: acc genuinely accumulates
                    self.acc = self.acc + v
            return self.acc

    real = lower(GuardReal().__call__)
    assert [slot.name for slot in real.state_slots] == ["acc"]
    assert _arith_count(real, FloatAdd) == 2  # one accumulate per folded-true trip (i=1, i=2), none for i=0


def test_static_float_comparison_branch_folds() -> None:
    # Regression (Codex finding 1, fast-math): a comparison of compile-time floats (a literal, a read-only float
    # attribute, or arithmetic of these) folds to one arm so a guarded write under a statically-false condition does
    # not become spurious state. Folding is float64 (fast-math, accepted per DESIGN.md); model and RTL follow the same
    # arm regardless.
    class ConfigGate:
        def __init__(self) -> None:
            self.threshold = 0.0  # read-only config: 0.0 > 1.0 is statically false
            self.gain = 2.0
            self.y = 0.0

        def __call__(self, x: float) -> float:
            if self.threshold > 1.0:
                self.y = x
            if self.gain * 3.0 > 10.0:  # 6.0 > 10.0 is statically false
                self.y = x
            return self.y

    folded = lower(ConfigGate().__call__)
    assert [slot.name for slot in folded.state_slots] == []
    assert len(optimize(folded).blocks) == 1

    class ConfigEnabled:
        def __init__(self) -> None:
            self.threshold = 5.0
            self.y = 0.0

        def __call__(self, x: float) -> float:
            if self.threshold > 1.0:  # 5.0 > 1.0 statically true: the write is taken
                self.y = x
            return self.y

    enabled = lower(ConfigEnabled().__call__)
    assert [slot.name for slot in enabled.state_slots] == ["y"]
    assert len(optimize(enabled).blocks) == 1


def test_dead_assignment_after_return_does_not_suppress_fold() -> None:
    # Regression (Codex finding 3): the read-only-attribute scan stops at a return (like lowering), so an assignment
    # in dead code after a return does not mask the attribute's read-only-ness and the branch on it still folds.
    class DeadAfterReturn:
        def __init__(self) -> None:
            self.flag = False
            self.y = 0.0

        def __call__(self, x: float) -> float:
            if self.flag:  # flag is read-only -> folds to the (empty) else arm
                self.y = x
            result = self.y
            return result
            self.flag = False  # noqa -- dead code: must not count as an assignment of flag

    hir = lower(DeadAfterReturn().__call__)
    assert [slot.name for slot in hir.state_slots] == []
    assert len(optimize(hir).blocks) == 1


def test_public_boolean_state_attribute_is_output() -> None:
    class PublicBool:
        def __init__(self) -> None:
            self.flag = False
            self.y = 0.0

        def __call__(self, a: float, b: float) -> float:
            self.flag = a < b
            self.y = a
            return self.y

    hir = lower(PublicBool().__call__)
    assert [o.name for o in hir.outputs] == ["state_flag", "state_y"]
    assert {slot.name for slot in hir.state_slots} == {"flag", "y"}


def test_attribute_written_only_in_while_is_not_read_only() -> None:
    # Regression: the read-only-attribute scan (_collect_assigned) must descend `while` bodies, not just `if`/`for`.
    # An attribute written only inside a while loop is genuinely runtime-varying state; if the scan misses the write it
    # is misclassified as read-only and a later branch on it folds against the (stale) reset snapshot -- a SILENT
    # MISCOMPILATION that takes a fixed arm for every input. Here ``acc`` becomes 3*x at runtime (reset 0.0), so the
    # guard ``acc > 1.0`` is genuinely dynamic and must emit a real branch, not fold to the reset's (false) arm.
    class WhileWrittenGuard:
        def __init__(self) -> None:
            self.acc = 0.0

        def __call__(self, x: float) -> float:
            c = 3.0
            while c > 0.0:
                self.acc = self.acc + x
                c = c - 1.0
            if self.acc > 1.0:  # acc is runtime state, NOT the read-only reset 0.0: must stay a real branch
                r = 100.0
            else:
                r = -100.0
            return r

    hir = lower(WhileWrittenGuard().__call__)
    assert [slot.name for slot in hir.state_slots] == ["acc"]  # acc is persistent state
    # The acc-guard must be a real runtime branch (plus the while's own exit branch): two branches, not one folded away.
    assert sum(1 for b in hir.blocks if isinstance(b.terminator, Branch)) == 2


def test_loop_carried_attr_in_statically_dead_arm_does_not_crash() -> None:
    # Regression: _loop_assigned must be fold-aware, mirroring lowering. When an attribute's only write inside a while
    # body sits in a statically-dead (constant-folded-away) `if` arm, that write is never reachable, so the attribute
    # is not persistent state. A fold-unaware scan would still list it as loop-carried and crash _lower_while with a
    # KeyError opening a header phi for a value that is not loaded as state.
    class DeadArmCarry:
        def __init__(self) -> None:
            self.acc = 0.0

        def __call__(self, x: float) -> float:
            c = 2.0
            while c > 0.0:
                if 0.5 > 1.0:  # statically false: the only write of acc is unreachable
                    self.acc = self.acc + x
                c = c - 1.0
            return x

    hir = lower(DeadArmCarry().__call__)
    assert [slot.name for slot in hir.state_slots] == []  # the dead write makes acc no state at all


def test_loop_carried_attr_written_only_in_live_folded_arm() -> None:
    # Companion to the above: when the live (folded-true) arm carries the only write, the attribute IS state and the
    # loop lowers without a spurious self-referential header phi for an unwritten value.
    class LiveArmCarry:
        def __init__(self) -> None:
            self.b = 0.5

        def __call__(self, x: float) -> float:
            h = 1.0
            while h > 0.0:
                if 1.5 <= 2.0:  # statically true: this arm's write is the reachable one
                    self.b = self.b + x
                else:
                    self.b = x  # folded-away arm: must not create a phantom carried value
                h = h - 1.0
            return self.b

    hir = lower(LiveArmCarry().__call__)
    assert [slot.name for slot in hir.state_slots] == ["b"]


def _integrator_class() -> type:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    from trapezoidal_leaky_streaming_integrator import TrapezoidalLeakyStreamingIntegrator

    return TrapezoidalLeakyStreamingIntegrator


def test_stateful_method_state_slots_and_dedup() -> None:
    integrator = _integrator_class()(k=2**-22)
    hir = lower(integrator.__call__)
    assert hir.input_names() == ["x", "dt"]  # self is dropped; remaining parameters become inputs
    # `return self.y` is deduped onto the public state port state_y; the private _x_prev gets no port, so the output
    # list alone distinguishes public from private. Both slots reset to 0.
    assert [o.name for o in hir.outputs] == ["state_y"]
    slots = {s.name: s for s in hir.state_slots}
    assert set(slots) == {"y", "_x_prev"}
    assert (
        cast(FloatConst, slots["y"].reset_value).value == 0.0
        and cast(FloatConst, slots["_x_prev"].reset_value).value == 0.0
    )
    assert {n.slot for n in hir.nodes.values() if isinstance(n, StateRead)} == {"y", "_x_prev"}


def test_returned_public_state_alias_is_deduped() -> None:
    # The dedup is by dataflow, not spelling: returning a public attribute through an alias must still collapse onto its
    # state_<attr> port rather than emitting a second positional output for the same value.
    class Aliased:
        def __init__(self) -> None:
            self.y = 0.0

        def __call__(self, x: float) -> float:
            self.y = x
            y = self.y
            return y

    hir = lower(Aliased().__call__)
    assert [o.name for o in hir.outputs] == ["state_y"]


def test_deferred_call_graft_retracts_stale_edges_straight_line() -> None:
    # R8-2 round-8, case (a): a library call (np.dot on the cut operand) defers behind a transient store
    # violation (self.t = u, whose else-arm value overflows the float carrier) and a transiently-rejecting feed
    # (w = 3 << s, whose else-arm shift count is negative), then grafts on a revisit. The graft replaces the
    # block's terminator, but the pre-graft terminator's recorded out-edge used to survive as a phantom edge to
    # the old successor and reach emission, which read the return place undefined and crashed with a bare
    # "escaped analysis" AssertionError. With the stale edge retracted the kernel analyzes cleanly and refuses
    # for the honest reason -- the runtime integer shift -- exactly as its violation-free control does.
    class Probe:
        def __init__(self) -> None:
            self.t = 0.0

        def step(self, flag: bool) -> float:
            if flag:
                u = 1.0
                s = 1
            else:
                u = 2**53 + 1
                s = -1
            self.t = u
            w = 3 << s
            v = w * 0.5
            a = np.array([v, v])
            y = np.dot(a, a)
            return y + self.t  # type: ignore[no-any-return]

    class Control:
        def __init__(self) -> None:
            self.t = 0.0

        def step(self, flag: bool) -> float:
            if flag:
                u = 1.0
                s = 1
            else:
                u = 2.0
                s = 2
            self.t = u
            w = 3 << s
            v = w * 0.5
            a = np.array([v, v])
            y = np.dot(a, a)
            return y + self.t  # type: ignore[no-any-return]

    ops = default_ops(FloatFormat(8, 23))
    with pytest.raises(UnsupportedConstruct, match="integer values are not yet lowerable") as probe:
        holoso.synthesize(Probe().step, ops, name="graft_edge_straight")
    with pytest.raises(UnsupportedConstruct, match="integer values are not yet lowerable") as control:
        holoso.synthesize(Control().step, ops, name="graft_edge_straight_ctrl")
    assert str(probe.value) == str(control.value)


def test_deferred_call_graft_retracts_stale_edges_with_branch() -> None:
    # R8-2 round-8, case (b): the same deferral, now with a branch after the call. The stale out-edge left a
    # phantom successor whose env was derived on the graft-skipping path with the call result unbound; the
    # continuation then joined into that stale env (a bound value joining an unbound one yields maybe-unbound),
    # surfacing a nonsense "local ... may be unbound" rejection at an innocent return line. Retracting the edge
    # and dropping the orphaned env lets the branch analyze cleanly and refuse for the same honest reason as the
    # violation-free control.
    class Probe:
        def __init__(self) -> None:
            self.t = 0.0

        def step(self, flag: bool, pick: bool) -> float:
            if flag:
                u = 1.0
                s = 1
            else:
                u = 2**53 + 1
                s = -1
            self.t = u
            w = 3 << s
            v = w * 0.5
            a = np.array([v, v])
            y = np.dot(a, a)
            if pick:
                return y + 1.0  # type: ignore[no-any-return]
            return self.t

    class Control:
        def __init__(self) -> None:
            self.t = 0.0

        def step(self, flag: bool, pick: bool) -> float:
            if flag:
                u = 1.0
                s = 1
            else:
                u = 2.0
                s = 2
            self.t = u
            w = 3 << s
            v = w * 0.5
            a = np.array([v, v])
            y = np.dot(a, a)
            if pick:
                return y + 1.0  # type: ignore[no-any-return]
            return self.t

    ops = default_ops(FloatFormat(8, 23))
    with pytest.raises(UnsupportedConstruct, match="integer values are not yet lowerable") as probe:
        holoso.synthesize(Probe().step, ops, name="graft_edge_branch")
    with pytest.raises(UnsupportedConstruct, match="integer values are not yet lowerable") as control:
        holoso.synthesize(Control().step, ops, name="graft_edge_branch_ctrl")
    assert str(probe.value) == str(control.value)


def test_deferred_graftable_call_does_not_starve_the_state_fixpoint() -> None:
    # The graft-deferral seam admits false rejections (see the open-defect witnesses below and TODO.md), but a fix
    # for them must not cost accepts that already work. Withholding a deferred graftable call's terminator edges --
    # the round-10 attempt at closing the seam -- starved this kernel's outer state fixpoint: the withheld edge is
    # the loop body's only successor, so the loop never re-flowed, the transiently-inexact `self.t` store never saw
    # its operand promote to runtime float, and a valid kernel that synthesizes to real Verilog was refused with
    # "state attribute 't' ... not exactly representable". Both spellings must keep synthesizing; the control
    # differs only by dropping the graftable call, isolating the call as the trigger.
    class Probe:
        def __init__(self) -> None:
            self.t = 0.0

        def step(self, x: float, run: bool) -> float:
            first = True
            while run:
                self.t = (2**53 + 1) if first else x
                a = np.array([(2**64) if first else x, x])
                np.dot(a, a)
                first = False
                run = False
            return x + self.t

    class Control:  # differs only by dropping the graftable call
        def __init__(self) -> None:
            self.t = 0.0

        def step(self, x: float, run: bool) -> float:
            first = True
            while run:
                self.t = (2**53 + 1) if first else x
                np.array([(2**64) if first else x, x])
                first = False
                run = False
            return x + self.t

    ops = default_ops(FloatFormat(8, 23))
    probe = holoso.synthesize(Probe().step, ops, name="graft_defer_no_starve")
    control = holoso.synthesize(Control().step, ops, name="graft_defer_no_starve_ctrl")
    assert len(probe.verilog_output.verilog) > 0
    assert len(control.verilog_output.verilog) > 0


def test_graftable_call_deferral_false_rejection_witnesses() -> None:
    # EXECUTABLE RECORD of an OPEN defect class -- these outcomes are documented defects, not desired behavior.
    # Each kernel below is honest Python that a correct compiler would synthesize; the analyzer refuses it because
    # a graftable call deferred behind a transiently-pending store violation publishes out-edges carrying its
    # not-yet-computed result, and a downstream read of that result joins to maybe-unbound. These are the residue
    # of the PHANTOM-ENVIRONMENT mechanism, the half the one-edge-deep graft retraction cannot reach; the
    # detectable-contradiction shapes are refused by the post-stabilization gate instead (sibling tests above).
    # Documented in TODO.md; the class is the one the Stage-4 restructure dissolves by residualizing after the
    # fixpoint instead of during it. Their residue is false rejections -- but note the seam as a whole is NOT
    # bounded to those: test_phantom_environment_miscompile_is_still_open pins a silently wrong value.
    #
    # The kernels live here rather than only in prose because prose transcriptions of them have silently rotted:
    # dropping the both-arms read or the wide-int feed makes a shape vanish. When the restructure lands these stop
    # rejecting -- that is the intended outcome, and this test must then flip to asserting synthesis.
    class BothArms:  # a single graftable call whose result is read on BOTH arms of a following branch
        def __init__(self) -> None:
            self.t = 0.0

        def step(self, x: float, flag: bool, pick: bool) -> float:
            if flag:
                u = 1.0
                q = 1.0
            else:
                u = 2**53 + 1
                q = 2**64
            self.t = u
            a = np.array([q, x])
            y = np.dot(a, a)
            if pick:
                return y + 1.0  # type: ignore[no-any-return]
            return y + 2.0  # type: ignore[no-any-return]

    class TwoGraftableCalls:  # a second graftable call grafts while the first is still deferred
        def __init__(self) -> None:
            self.t = 0.0

        def helper(self, x: float) -> float:
            return x

        def step(self, x: float, flag: bool, pick: bool) -> float:
            if flag:
                u = 1.0
                q = 1.0
            else:
                u = 2**53 + 1
                q = 2**64
            self.t = u
            a = np.array([q, x])
            y = np.dot(a, a)
            z = self.helper(x)
            if pick:
                return y + z  # type: ignore[no-any-return]
            return y + z + 1.0  # type: ignore[no-any-return]

    class StarredArguments:  # starred-argument validation refuses before the user call can graft
        def __init__(self) -> None:
            self.t = 0.0

        def step(self, x: float, flag: bool, pick: bool) -> float:
            if flag:
                u = 1.0
                q = 1.0
            else:
                u = 2**53 + 1
                q = 2**64
            self.t = u
            args = np.array([q, x])
            y = _multiply(*args)
            if pick:
                return y + 1.0
            return y + 2.0

    ops = default_ops(FloatFormat(8, 23))
    for label, kernel, blamed in (
        ("both_arms", BothArms().step, "return y + 2.0"),
        ("two_graftable_calls", TwoGraftableCalls().step, "return y + z + 1.0"),
        ("starred_arguments", StarredArguments().step, "return y + 2.0"),
    ):
        with pytest.raises(UnsupportedConstruct, match="may be unbound here") as witness:
            holoso.synthesize(kernel, ops, name=f"graft_defer_{label}")
        # Pin the SITE too: a future change that rejects at a different, also-wrong line would otherwise keep
        # this green. All three currently blame the fall-through return of the branch that reads the result.
        location = witness.value.location
        assert location is not None, label
        assert (location.line or "").strip().startswith(blamed), label


def test_speculated_dead_arm_that_stores_is_refused() -> None:
    # A speculated arm the stabilized facts prove dead used to be emitted, shipping a public `state_s` port whose
    # register carried the dead arm's store as its only functional driver -- inert solely because the sequencer's
    # selector was hard-loaded to zero. The gate refuses it instead, and unconditionally -- narrowing the rule to
    # arms that store was tried twice and both attempts readmitted silent miscompiles (see the siblings). The
    # control differs only in the two constants that arm the deferral, and must still synthesize. Nothing here
    # reads a deferred result, which isolates the executable-marking mechanism from the phantom-environment one
    # -- and np.array is a CONVERSION, never grafted, so this mechanism needs only a deferred call.
    class Probe:
        def __init__(self) -> None:
            self.t = 0.0
            self.s = 0.0

        def step(self, x: float, flag: bool) -> float:
            if flag:
                u = 1.0
                q = 1.0
            else:
                u = 2**53 + 1
                q = 2**64
            self.t = u
            a = np.array([q, x])
            if a.shape[0] > 5:  # statically false -- the array has two elements
                self.s = 7.0
                r = 1.0
            else:
                r = 2.0
            return x + r + self.s

    class Control:  # identical but for the constants, so the deferral never opens
        def __init__(self) -> None:
            self.t = 0.0
            self.s = 0.0

        def step(self, x: float, flag: bool) -> float:
            if flag:
                u = 1.0
                q = 1.0
            else:
                u = 2.0
                q = 3.0
            self.t = u
            a = np.array([q, x])
            if a.shape[0] > 5:
                self.s = 7.0
                r = 1.0
            else:
                r = 2.0
            return x + r + self.s

    ops = default_ops(FloatFormat(8, 23))
    with pytest.raises(UnsupportedConstruct, match="prove unreachable") as refusal:
        holoso.synthesize(Probe().step, ops, name="defer_dead_arm")
    assert refusal.value.location is not None  # located, so the implicated branch is visible
    control = holoso.synthesize(Control().step, ops, name="defer_dead_arm_ctrl")
    assert "state_s" not in [port.name for port in control.ports]  # no deferral, so the dead arm is pruned


def test_speculated_dead_arm_without_a_store_is_refused_too() -> None:
    # The refusal is deliberately unconditional, and this kernel is why it cannot be narrowed to arms that
    # store. An inert arm looks harmless and is not: it poisons the phi at the merge, which keeps a DOWNSTREAM
    # guard residual so that guard's store does the promoting. Refusing here costs an accept that would have
    # been correct; admitting it cost a silent miscompile. See TODO.md for the measured trade.
    class Inert:
        def __init__(self) -> None:
            self.t = 0.0

        def step(self, x: float, flag: bool) -> float:
            if flag:
                u = 1.0
                q = 1.0
            else:
                u = 2**53 + 1
                q = 2**64
            self.t = u
            a = np.array([q, x])
            if a.shape[0] > 5:  # statically false, and inert: no store on the speculated arm
                pass
            return x + 1.0

    with pytest.raises(UnsupportedConstruct, match="prove unreachable"):
        holoso.synthesize(Inert().step, default_ops(FloatFormat(8, 23)), name="defer_dead_arm_inert")


def test_store_after_an_inert_speculated_arm_is_refused_too() -> None:
    # Also refused, and this is the cost side of the unconditional rule: the store here really does run on the
    # taken path regardless. Scoping the check to the arm's exclusive region to spare exactly this shape was
    # tried and silently disabled the whole check inside a loop, where the back-edge puts the dead arm within
    # the live arm's reach and the exclusive region is always empty. The accept is not worth that.
    class MergeStore:
        def __init__(self) -> None:
            self.t = 0.0
            self.s = 0.0

        def step(self, x: float, flag: bool) -> float:
            if flag:
                u = 1.0
                q = 1.0
            else:
                u = 2**53 + 1
                q = 2**64
            self.t = u
            a = np.array([q, x])
            if a.shape[0] > 5:  # statically false and inert
                pass
            self.s = x * 2.0  # on the live path, reached from both arms
            return self.s

    with pytest.raises(UnsupportedConstruct, match="prove unreachable"):
        holoso.synthesize(MergeStore().step, default_ops(FloatFormat(8, 23)), name="defer_merge_store")


def test_speculated_dead_store_does_not_silently_miscompile() -> None:
    # Why emitting a speculated-then-dead arm is unsound rather than merely wasteful, and the sharpest available
    # witness for it. A store on the dead arm promotes `s` from a read-only constant -- folded at binary64, where
    # 1 + 2**-30 is greater than 1 -- into a runtime state slot whose RESET is materialized in the E8M23 carrier,
    # where it rounds to exactly 1.0. The guard `self.s > 1` therefore flipped, and a kernel returning 10.0 in
    # Python returned 20.0 in hardware with no error raised. The same hazard is guarded for arms dead by
    # `if False:` and by a static comparison; this is the third kind, dead by fold-after-marking. Note the
    # divergence is format-dependent -- E11M52 was correct throughout -- so the assertion below must run in a
    # carrier narrower than the reset's binary64 value.
    class Miscompiled:
        def __init__(self) -> None:
            self.t = 0.0
            self.s = 1 + 2**-30

        def step(self, x: float, flag: bool) -> float:
            if flag:
                u = 1.0
                q = 1.0
            else:
                u = 2**53 + 1
                q = 2**64
            self.t = u
            a = np.array([q, x])
            if a.shape[0] > 5:  # statically false; its store must never reach hardware
                self.s = 7.0
            if self.s > 1:
                return 10.0
            return 20.0

    assert Miscompiled().step(2.0, False) == 10.0  # the Python answer the hardware contradicted with 20.0
    with pytest.raises(UnsupportedConstruct, match="prove unreachable") as refusal:
        holoso.synthesize(Miscompiled().step, default_ops(FloatFormat(8, 23)), name="dead_store_miscompile")
    assert refusal.value.location is not None


def test_speculated_dead_arm_store_does_not_strand_the_rank_walk() -> None:
    # The same speculated-arm mechanism reached the plan replay as a bare KeyError rather than a diagnostic:
    # `rank` is built from the blocks an executable edge chain REACHES, while the replay iterates the add-only
    # MARKED set, so a marked-but-unreachable block carrying a store indexed `rank` and died with no message,
    # no location, and no exception type a user could act on. Two arms are needed to strand the block.
    class Probe:
        def __init__(self) -> None:
            self.t = 0.0
            self.s = 0.0

        def step(self, x: float, flag: bool) -> float:
            if flag:
                u = 1.0
                q = 1.0
            else:
                u = 2**53 + 1
                q = 2**64
            self.t = u
            a = np.array([q, x])
            y = np.dot(a, a)  # noqa: F841 -- the deferred call is the trigger; its result is deliberately unused
            if a.shape[0] > 5:  # statically false
                if flag:
                    self.s = 7.0
                else:
                    self.s = 8.0
            return x + self.s

    with pytest.raises(UnsupportedConstruct) as refusal:
        holoso.synthesize(Probe().step, default_ops(FloatFormat(8, 23)), name="dead_arm_rank_walk")
    assert refusal.value.location is not None  # a diagnostic, where a bare KeyError used to escape


def test_speculated_arm_reports_before_the_graph_asserts() -> None:
    # The gate runs before `_validate`, whose asserts describe a graph the gate may already know is inconsistent.
    # Three nested statically-decidable guards behind a deferred graftable call leave an unresolved call on the
    # speculated arm, which tripped "unexpanded call survived" as a BARE AssertionError -- and, because asserts
    # vanish under -O, the debug build crashed where the optimized build explained. Two levels do not reach it.
    class Nested:
        def __init__(self) -> None:
            self.t = 0.0

        def step(self, x: float, flag: bool) -> float:
            if flag:
                u = 1.0
                q = 1.0
            else:
                u = 2**53 + 1
                q = 2**64
            self.t = u
            a = np.array([q, x])
            _y = np.dot(a, a)  # noqa: F841 -- the graftable deferral is the trigger
            if a.shape[0] > 5:
                if a.size > 1:
                    if len(a) > 3:
                        pass
            return x + 1.0

    with pytest.raises(UnsupportedConstruct, match="never runs|prove unreachable") as refusal:
        holoso.synthesize(Nested().step, default_ops(FloatFormat(8, 23)), name="nested_speculation")
    assert refusal.value.location is not None  # a diagnostic, where a bare AssertionError used to escape


def test_speculated_dead_store_inside_a_loop_is_refused() -> None:
    # The miscompile witness with its guard moved inside a `while`. A narrowing that tested only the region
    # reachable EXCLUSIVELY through the speculated arm silently stopped checking anything here: the back-edge
    # puts the dead arm within the live arm's reach, so the difference is empty for every branch in a loop body
    # and the kernel returned 20.0 against Python's 10.0. The check is unconditional so this cannot recur.
    class LoopDeadStore:
        def __init__(self) -> None:
            self.t = 0.0
            self.s = 1 + 2**-30

        def step(self, x: float, flag: bool, run: bool) -> float:
            if flag:
                u = 1.0
                q = 1.0
            else:
                u = 2**53 + 1
                q = 2**64
            while run:
                self.t = u
                a = np.array([q, x])
                if a.shape[0] > 5:  # statically false; its store must never reach hardware
                    self.s = 7.0
                run = False
            if self.s > 1:
                return 10.0
            return 20.0

    assert LoopDeadStore().step(2.0, False, True) == 10.0
    with pytest.raises(UnsupportedConstruct, match="prove unreachable"):
        holoso.synthesize(LoopDeadStore().step, default_ops(FloatFormat(8, 23)), name="loop_dead_store")


def test_inert_speculated_arm_poisoning_a_later_guard_is_refused() -> None:
    # Why the refusal cannot be scoped to arms that store. NEITHER arm here stores; they only assign a local.
    # But the merge phi over them stays residual because the branch was speculated, so the LATER guard on `g`
    # never folds and its store is emitted -- promoting `s` to runtime state, whose reset rounds in the carrier.
    # Narrowing the check to storing arms admitted this and returned 20.0 against Python's 10.0.
    class PhiPoison:
        def __init__(self) -> None:
            self.t = 0.0
            self.s = 1 + 2**-30

        def step(self, x: float, flag: bool) -> float:
            if flag:
                u = 1.0
                q = 1.0
            else:
                u = 2**53 + 1
                q = 2**64
            self.t = u
            a = np.array([q, x])
            if a.shape[0] > 5:  # statically false, and stores nothing
                g = True
            else:
                g = False
            if g:  # the phi stays residual, so this store is emitted
                self.s = 7.0
            if self.s > 1:
                return 10.0
            return 20.0

    assert PhiPoison().step(2.0, False) == 10.0
    with pytest.raises(UnsupportedConstruct, match="prove unreachable"):
        holoso.synthesize(PhiPoison().step, default_ops(FloatFormat(8, 23)), name="phi_poison")


def test_runtime_state_discovered_on_a_dead_round_is_refused() -> None:
    # The third miscompile route, and the only one that is not visible on the final graph at all. W grows
    # monotonically: round 1 speculates through `self.mode` being Known-True and discovers a store to `s`, round 2
    # proves the shape guard false so that store is unreachable, and every per-round check then passes -- but `s`
    # stays in the runtime-state set, so its reset materializes in the carrier instead of folding at binary64 and
    # the guard flips. Accepted, no error, 20.0 against Python's 10.0, with a spurious public state_s port.
    class CrossRound:
        def __init__(self) -> None:
            self.mode = True
            self.t = 0.0
            self.s = 1 + 2**-30

        def step(self, x: float, new_mode: bool) -> float:
            if self.mode:
                u: float = 2**53 + 1
                q: float = 2**64
            else:
                u = x
                q = x
            self.t = u
            a = np.array([q, x])
            if a.shape[0] > 5:
                self.s = 7.0
            self.mode = new_mode
            return 10.0 if self.s > 1.0 else 20.0

    assert CrossRound().step(2.0, False) == 10.0
    with pytest.raises(UnsupportedConstruct, match="earlier analysis round") as refusal:
        holoso.synthesize(CrossRound().step, default_ops(FloatFormat(8, 23)), name="cross_round_stale")
    # The store that promoted the leaf is the only line the user can act on, and it is gone from the stable
    # graph by the time this check runs -- the per-round store map is empty for it, so the promotion origin has
    # to be remembered when the leaf enters the runtime-state set or the refusal renders at line 0, column 0.
    location = refusal.value.location
    assert location is not None
    assert (location.line or "").strip() == "self.s = 7.0"


def test_stale_runtime_state_reports_the_source_earliest_store() -> None:
    # Two components carrying the SAME attribute path both go stale, so the message text -- which names only the
    # path -- cannot tell them apart and the reported LOCATION is the whole diagnostic. Selection must be a
    # property of the source, not of set iteration over identity-hashed leaves.
    class Cell:
        def __init__(self) -> None:
            self.s = 1 + 2**-30

    class TwoCells:
        def __init__(self) -> None:
            self.mode = True
            self.t = 0.0
            self.a = Cell()
            self.b = Cell()

        def step(self, x: float, new_mode: bool) -> float:
            if self.mode:
                u: float = 2**53 + 1
                q: float = 2**64
            else:
                u = x
                q = x
            self.t = u
            span = np.array([q, x])
            if span.shape[0] > 5:
                self.a.s = 7.0
                self.b.s = 9.0
            self.mode = new_mode
            return 10.0 if self.a.s > 1.0 else 20.0

    assert TwoCells().step(2.0, False) == 10.0
    with pytest.raises(UnsupportedConstruct, match="earlier analysis round") as refusal:
        holoso.synthesize(TwoCells().step, default_ops(FloatFormat(8, 23)), name="two_cells_stale")
    location = refusal.value.location
    assert location is not None
    assert (location.line or "").strip() == "self.a.s = 7.0"


def test_cross_round_state_verdicts_are_located_at_their_store() -> None:
    # Not just the stale-leaf refusal: EVERY state verdict drawn after the promoting round has an empty
    # per-round store map to look in, so reset diagnostics rendered at line 0 too. The promotion origin is what
    # they all fall back on.
    class MissingReset:
        def __init__(self) -> None:
            self.mode = True
            self.t = 0.0

        def step(self, x: float, new_mode: bool) -> float:
            if self.mode:
                u: float = 2**53 + 1
                q: float = 2**64
            else:
                u = x
                q = x
            self.t = u
            span = np.array([q, x])
            if span.shape[0] > 4:
                self.s = 7.0  # never initialized in __init__, so there is no reset to reconstruct
            self.mode = new_mode
            return x

    with pytest.raises(UnsupportedConstruct, match="does not exist on the component") as refusal:
        holoso.synthesize(MissingReset().step, default_ops(FloatFormat(8, 23)), name="missing_reset")
    location = refusal.value.location
    assert location is not None
    assert (location.line or "").strip().startswith("self.s = 7.0")


def _assert_store_helpers_tie() -> None:
    """
    Both store-selection tests below only discriminate while `_store_alpha.put` and `_store_beta.put` sit at
    the SAME source position -- a property of the fixture files, not of the code under test. A one-line drift
    in either helper would make them pass for the wrong reason, so it is asserted rather than trusted.
    """
    from tests import _store_alpha, _store_beta

    alpha_source, alpha_line = inspect.getsourcelines(_store_alpha.put)
    beta_source, beta_line = inspect.getsourcelines(_store_beta.put)
    assert alpha_line == beta_line
    assert [line.rstrip() for line in alpha_source] == [line.rstrip() for line in beta_source]


def test_state_port_order_follows_execution_rank_not_frame_identity() -> None:
    # `first_store` keys (source_position, rank), and the execution rank is load-bearing ABI: two components
    # whose stores are reached from ONE unrolled call site tie on source position, and the rank is what puts
    # them in the order they execute. Ordering those ties by frame identity instead silently renames the module
    # ports by filename -- a public ABI change, which nothing else in the suite would notice.
    from tests import _store_alpha, _store_beta

    _assert_store_helpers_tie()

    class Zed:
        def __init__(self) -> None:
            self.s = 0.0

    class Aye:
        def __init__(self) -> None:
            self.s = 0.0

    class Pair:
        def __init__(self) -> None:
            self.z = Zed()
            self.a = Aye()

        def step(self, x: float) -> float:
            for stage in ((_store_beta.put, self.z), (_store_alpha.put, self.a)):
                stage[0](stage[1], x)
            return x

    built = holoso.synthesize(Pair().step, default_ops(FloatFormat(8, 23)), name="pair_ports")
    state_ports = [port.name for port in built.ports if port.name.startswith("state_")]
    assert state_ports == ["state_z__s", "state_a__s"]  # beta executes first; alpha's filename sorts first


def test_a_store_diagnostic_names_the_first_executed_of_two_tied_stores() -> None:
    # Two stores at IDENTICAL source positions in different files, reached through one unrolled call site, tie
    # on source position and can only be separated by something outside it. The per-store selection deliberately
    # keeps `source_position` and therefore first-wins in transfer order, which is execution order: the loop
    # runs beta first, so beta is named. Ordering those ties by frame identity instead would name alpha here,
    # purely because its filename sorts earlier -- a worse answer, and the reason that conversion was reverted.
    # This governs the per-round store map only. The promotion pick orders a SET and so must be total, which
    # means it does break the identical tie by filename; the same helper pair reached on the latch path names
    # alpha. The two paths differ on purpose, and neither is a general "first-executed" guarantee.
    from tests import _store_alpha, _store_beta

    _assert_store_helpers_tie()

    class Tied:
        def __init__(self) -> None:
            self.mode = True
            self.t = 0.0
            self.s = (1.0, 2.0)  # an unsupported reset, so a store to `s` draws the verdict

        def step(self, x: float, flag: bool) -> float:
            # The wide-int prologue defers the first store's verdict, so BOTH stores are recorded before the
            # selection runs -- without it the verdict fires at the first store and the tie never arises.
            if self.mode:
                u: float = 2**53 + 1
                q: float = 2**64
            else:
                u = x
                q = x
            self.t = u
            span = np.array([q, x])
            for put in (_store_beta.put, _store_alpha.put):
                put(self, x + span.shape[0])
            self.mode = flag
            return x

    with pytest.raises(UnsupportedConstruct, match="unsupported reset type") as refusal:
        holoso.synthesize(Tied().step, default_ops(FloatFormat(8, 23)), name="tied_stores")
    assert "in put():" in str(refusal.value)  # both helpers render identically; only the file separates them
    located = refusal.value
    assert isinstance(located, LocatedRejection)
    assert located.origin[0].file.endswith("_store_beta.py")


def test_a_verdict_prefers_a_raise_guarded_store_over_the_promoter() -> None:
    # EXECUTABLE RECORD of the OTHER half of the `_state_origin` trade, NOT desired behavior. Both sources are
    # populated here and they DISAGREE -- this is not the map-empty fallback case. The per-round store map
    # leads, and it holds every store the round transferred, including one in a block the exit cannot be
    # reached from -- here the store before the `raise`, which is source-earlier than the live one and takes
    # the anchor, while the latch holds the live store. Preferring the promotion latch instead names the live
    # store on THIS shape and a dead store on
    # the shape the neighbouring test pins, which is why neither order is a fix and this one is not "safer".
    # Both witnesses exist so that a future round cannot make either half worse against a green suite.
    class RaiseGuarded:
        def __init__(self) -> None:
            self.mode = True
            self.t = 0.0

        def step(self, x: float, boom: bool) -> float:
            if self.mode:
                u: float = 2**53 + 1
                q: float = 2**64
            else:
                u = x
                q = x
            self.t = u
            span = np.array([q, x])
            if boom:
                self.zz = 3.0  # never reached past the raise, yet it is the line the message names
                raise ValueError("stop")
            self.zz = 1.0
            self.mode = False
            return x

    with pytest.raises(UnsupportedConstruct, match="does not exist on the component") as refusal:
        holoso.synthesize(RaiseGuarded().step, default_ops(FloatFormat(8, 23)), name="raise_guarded_anchor")
    location = refusal.value.location
    assert location is not None
    assert (location.line or "").strip().startswith("self.zz = 3.0")  # the trade; the live store is self.zz = 1.0


def test_a_mid_round_verdict_still_anchors_on_a_speculated_store() -> None:
    # EXECUTABLE RECORD of an OPEN residual of the deferral seam, NOT desired behavior. The refusal itself is
    # correct -- a tuple reset is genuinely unsupported -- but the LINE it names is one Python never runs.
    # The verdict is raised before `s` has a promotion-latch entry -- here the analysis aborts before any
    # end-of-round pass could fill one -- so the location comes from whatever stores the worklist has reached,
    # speculated arms included, and the dead arm wins on source order. Both
    # lookup orders behave identically here, which is why the priority is not a fix for this: measured by
    # swapping them and re-running. Deleting the dead arm moves the anchor to `self.s = self.s`.
    class MidRound:
        def __init__(self) -> None:
            self.t = 0.0
            self.s = (1.0, 2.0)

        def step(self, x: float, flag: bool) -> float:
            if flag:
                u = 1.0
                q = 1.0
            else:
                u = 2**53 + 1
                q = 2**64
            self.t = u
            span = np.array([q, x])
            if span.shape[0] > 5:
                self.s = 7.0  # type: ignore[assignment]  # statically false arm; Python never runs it
            self.s = self.s
            return x + 1.0

    with pytest.raises(UnsupportedConstruct, match="unsupported reset type") as refusal:
        holoso.synthesize(MidRound().step, default_ops(FloatFormat(8, 23)), name="mid_round_anchor")
    location = refusal.value.location
    assert location is not None
    assert (location.line or "").strip().startswith("self.s = 7.0")  # WRONG line; must become self.s = self.s


def test_state_verdicts_do_not_anchor_on_a_store_proved_dead() -> None:
    # The leaf's PROMOTION origin is latched at the round that first promoted it, and the state set's
    # monotonicity keeps the leaf, not that store's reachability -- on this shape the latched store sits in the
    # arm the stabilized facts prove dead. A verdict that preferred it would point the user at a line that never
    # runs, so the per-round store map leads and the promotion origin only stands in when that map is empty.
    class TupleReset:
        def __init__(self) -> None:
            self.mode = True
            self.t = 0.0
            self.s = (1.0, 2.0)  # an unsupported reset type, so every round draws a verdict about `s`

        def step(self, x: float, new_mode: bool) -> float:
            if self.mode:
                u: float = 2**53 + 1
                q: float = 2**64
            else:
                u = x
                q = x
            self.t = u
            span = np.array([q, x])
            if span.shape[0] > 5:
                self.s = 7.0  # type: ignore[assignment]  # Python never runs this; the facts prove it dead
            self.s = self.s
            self.mode = new_mode
            return x

    with pytest.raises(UnsupportedConstruct) as refusal:
        holoso.synthesize(TupleReset().step, default_ops(FloatFormat(8, 23)), name="tuple_reset")
    location = refusal.value.location
    assert location is not None
    assert (location.line or "").strip() == "self.s = self.s"


def test_stale_runtime_state_blames_a_store_that_actually_promoted() -> None:
    # An earlier store standing in a block the exit cannot be reached from promotes nothing, so it must not be
    # named: the leaf's provenance comes from the stores that put it in the runtime-state set, not from the
    # source-earliest store anywhere in the round.
    class DeadEarlier:
        def __init__(self) -> None:
            self.mode = True
            self.t = 0.0
            self.s = 1 + 2**-30

        def step(self, x: float, new_mode: bool) -> float:
            if self.mode:
                u: float = 2**53 + 1
                q: float = 2**64
            else:
                u = x
                q = x
            self.t = u
            span = np.array([q, x])
            if span.shape[0] > 5:
                self.s = 3.0
                raise RuntimeError("never reached")
            if span.shape[0] > 4:
                self.s = 7.0
            self.mode = new_mode
            return 10.0 if self.s > 1.0 else 20.0

    with pytest.raises(UnsupportedConstruct, match="earlier analysis round") as refusal:
        holoso.synthesize(DeadEarlier().step, default_ops(FloatFormat(8, 23)), name="dead_earlier_store")
    location = refusal.value.location
    assert location is not None
    assert (location.line or "").strip() == "self.s = 7.0"


def test_self_assignment_defeats_the_runtime_state_check() -> None:
    # EXECUTABLE RECORD of an OPEN defect (TODO.md), NOT desired behavior. The runtime-state check requires every
    # retained leaf to have a store in the final executable graph, which a trivial `self.s = self.s` satisfies --
    # so the check closes only the spelling where NO store survives. Delete that one line and this kernel is
    # correctly refused; keep it and the dead-arm store's promotion of `s` goes through and the guard flips.
    # Pinned rather than fixed: adding a check that reasons about which stores are meaningful is the same move
    # that has failed four times in this seam.
    class SelfAssign:
        def __init__(self) -> None:
            self.mode = True
            self.t = 0.0
            self.s = 1 + 2**-30

        def step(self, x: float, new_mode: bool) -> float:
            if self.mode:
                u: float = 2**53 + 1
                q: float = 2**64
            else:
                u = x
                q = x
            self.t = u
            a = np.array([q, x])
            if a.shape[0] > 5:
                self.s = 7.0  # Python never runs this
            self.s = self.s  # keeps `s` stored in the final graph, so the runtime-state check finds nothing
            self.mode = new_mode
            return 10.0 if self.s > 1.0 else 20.0

    expected = SelfAssign().step(2.0, False)
    assert expected == 10.0
    built = holoso.synthesize(SelfAssign().step, default_ops(FloatFormat(8, 23)), name="self_assign_stale")
    outputs = dict(zip((port.name for port in built.output_ports), built.numerical_model.elaborate().run(2.0, False)))
    assert float(outputs["out_0"]) == 20.0  # WRONG; must become 10.0 when the restructure lands


def test_live_in_poisoning_miscompile_is_still_open() -> None:
    # EXECUTABLE RECORD of an OPEN defect (TODO.md), NOT desired behavior: a SILENT WRONG ANSWER. The W/D
    # accumulator has two halves and only W is guarded. A round-1 speculated arm drives D[mode] down to a
    # residual live-in; round 2 prunes that arm, so no check on the stable graph sees anything wrong, and the
    # trailing store keeps `mode` in first_store so the runtime-state check passes too. The guard on `mode` then
    # reads the poisoned live-in and takes a branch Python never takes. Mirroring the W check does not help:
    # W staleness leaves a residue to detect, D staleness is byte-identical to what the final round derives.
    class DPoison:
        def __init__(self) -> None:
            self.mode = True  # Python: always True -- reset True, and the only executed store writes True
            self.t = 0.0
            self.s = 1 + 2**-30  # rounds to 1.0 in E8M23, exact in E11M52

        def step(self, x: float, flag: bool) -> float:
            if self.mode:  # Known(True) on round 1, so u is a Known inexact int and the array call defers
                u: float = 2**53 + 1
                q: float = 2**64
            else:
                u = x
                q = x
            self.t = u
            a = np.array([q, x])
            if a.shape[0] > 5:  # speculated on round 1, statically false and pruned on round 2
                self.mode = False  # its only effect is to drive D[mode] residual
            if self.mode:
                r = 10.0
            else:
                self.s = 7.0  # Python never runs this; the analyzer emits it
                r = 20.0
            if flag:
                self.mode = True  # keeps `mode` stored, so the runtime-state check finds nothing stale
            return r if self.s > 1.0 else 30.0

    expected = DPoison().step(2.0, False)
    assert expected == 10.0
    narrow = holoso.synthesize(DPoison().step, default_ops(FloatFormat(8, 23)), name="d_poison")
    outputs = dict(zip((port.name for port in narrow.output_ports), narrow.numerical_model.elaborate().run(2.0, False)))
    assert float(outputs["out_0"]) == 30.0  # WRONG; must become 10.0 when the restructure lands
    wide = holoso.synthesize(DPoison().step, default_ops(FloatFormat(11, 52)), name="d_poison_wide")
    wide_outputs = dict(
        zip((port.name for port in wide.output_ports), wide.numerical_model.elaborate().run(2.0, False))
    )
    assert float(wide_outputs["out_0"]) == expected  # correct where the reset is exact, isolating the channel


def test_phantom_environment_miscompile_is_still_open() -> None:
    # EXECUTABLE RECORD of an OPEN defect (TODO.md), NOT desired behavior, and the most serious one left: a
    # SILENT WRONG ANSWER. The deferred inlined helper sets `self.gate`, but the phantom environment left by the
    # graft keeps the stale `False` alive, so `self.gate` settles as a RUNTIME bool instead of Known(True). The
    # else arm is therefore genuinely live as far as the analyzer can tell -- there is no contradiction between
    # recorded reachability and the settled facts, so the post-stabilization gate cannot see this one. Its
    # `self.s = 7.0` promotes `s` to runtime state, the reset 1 + 2**-30 rounds to 1.0 in E8M23, and the guard
    # flips. Closing this is a first-class acceptance criterion for the resolution-totality restructure.
    class OneDiamond:
        def __init__(self) -> None:
            self.t = 0.0
            self.gate = False
            self.s = 1 + 2**-30

        def helper(self, a: float, b: float) -> float:
            self.gate = True
            return a * b

        def step(self, x: float, flag: bool, pick: bool) -> float:
            if flag:
                u = 1.0
                q = 1.0
            else:
                u = 2**53 + 1
                q = 2**64
            self.t = u
            args = np.array([q, x])
            self.helper(*args)
            if pick:
                pad = 1.0
            else:
                pad = 2.0
            if self.gate:
                marker = 0.0
            else:
                self.s = 7.0
                marker = 100.0
            if self.s > 1:
                return 10.0 + pad + marker
            return 20.0 + pad + marker

    arguments = (3.0, False, False)
    expected = OneDiamond().step(*arguments)
    assert expected == 12.0  # plain Python: the helper runs, so gate is True and s keeps its exact reset
    narrow = holoso.synthesize(OneDiamond().step, default_ops(FloatFormat(8, 23)), name="phantom_miscompile")
    assert "state_s" in [port.name for port in narrow.ports]  # the defect: the dead arm's store becomes state
    assert float(narrow.numerical_model.elaborate().run(*arguments)[0]) == 22.0  # WRONG; must become 12.0
    # In a carrier that represents the reset exactly the same structure is value-correct, which isolates the
    # divergence to reset materialization rather than to the datapath.
    wide = holoso.synthesize(OneDiamond().step, default_ops(FloatFormat(11, 52)), name="phantom_miscompile_wide")
    assert float(wide.numerical_model.elaborate().run(*arguments)[0]) == expected


def test_deferred_side_effect_branch_is_refused_not_crashed() -> None:
    # When the deferred call has a state side effect that a following branch tests, a graft leaves the orphaned
    # source block's out-edges standing, so the successor keeps a predecessor that never runs and its phi has no
    # arm for it. A raw RuntimeError used to escape HIR emission here -- not a SynthesisError, not located. The
    # kernel is honest Python returning 10.0, so this refusal is still a defect (TODO.md); pinning it keeps the
    # failure a diagnostic, and when the restructure closes the class this must become synthesis.
    #
    # This is also the only coverage of the seam's SECOND producer of stale reachability: the condition here is
    # a state read whose live-in join settles late, with every fact legitimately bound throughout -- no unbound
    # operand is ever involved, so no check on a condition's operand could have caught it. That is why the gate
    # tests the RESULT (recorded reachability against settled facts) rather than any producer.
    class Probe:
        def __init__(self) -> None:
            self.t = 0.0
            self.gate = False

        def helper(self, a: float, b: float) -> float:
            self.gate = True
            return a * b

        def step(self, x: float, flag: bool) -> float:
            if flag:
                u = 1.0
                q = 1.0
            else:
                u = 2**53 + 1
                q = 2**64
            self.t = u
            args = np.array([q, x])
            self.helper(*args)
            if self.gate:
                z = 10.0
            else:
                z = 20.0
            return z

    assert Probe().step(2.0, False) == 10.0  # the kernel is well-defined Python
    ops = default_ops(FloatFormat(8, 23))
    with pytest.raises(UnsupportedConstruct, match="path that never runs") as refusal:
        holoso.synthesize(Probe().step, ops, name="defer_side_effect_state_condition")
    assert refusal.value.location is not None  # a diagnostic, where a raw RuntimeError used to escape emission


def test_mixed_return_dedupes_public_alias_keeps_distinct_leaf() -> None:
    class Mixed:
        def __init__(self) -> None:
            self.y = 0.0

        def __call__(self, x: float) -> tuple[float, float]:
            self.y = x * 2.0
            a = self.y
            return (a, x)  # a aliases public self.y (deduped to state_y); x is distinct (keeps its positional out_1)

    hir = lower(Mixed().__call__)
    assert [o.name for o in hir.outputs] == ["out_1", "state_y"]


def test_return_value_equal_to_public_state_is_deduped_even_without_aliasing() -> None:
    # Dedup keys on the value, not provenance: returning x while x is also a public slot's live-out collapses onto that
    # slot's port even though the return never names the attribute. This is safe -- state_last carries the very same
    # wire, so the value stays observable; a separate out_0 would only duplicate it.
    class Passthrough:
        def __init__(self) -> None:
            self.last = 0.0

        def __call__(self, x: float) -> float:
            self.last = x
            return x

    hir = lower(Passthrough().__call__)
    assert [o.name for o in hir.outputs] == ["state_last"]


def test_unreachable_state_write_is_ignored() -> None:
    # A state write after the return is unreachable and never lowered; collecting it must not be attempted (it used to
    # crash with a KeyError). The method synthesizes as if the dead line were not there.
    class Dead:
        def __init__(self) -> None:
            self.y = 0.0

        def __call__(self, x: float) -> float:
            return x
            self.y = x  # unreachable

    hir = lower(Dead().__call__)
    assert [o.name for o in hir.outputs] == ["out_0"]
    assert hir.state_slots == []


def test_attribute_written_only_in_dead_code_reads_as_constant() -> None:
    # An attribute whose only assignment is unreachable is not state: a reachable read of it folds to its snapshot
    # constant, so it gets no slot and no out_<attr> port (whether it is state depends on its write being reachable).
    class Stale:
        def __init__(self) -> None:
            self.y = 5.0

        def __call__(self, x: float) -> float:
            r = x + self.y  # y folds to its snapshot 5.0 -- its only write is dead
            return r
            self.y = x  # unreachable

    hir = lower(Stale().__call__)
    assert hir.state_slots == []
    assert [o.name for o in hir.outputs] == ["out_0"]
    assert all(not (isinstance(n, StateRead) and n.slot == "y") for n in hir.nodes.values())


def test_stateful_readonly_attribute_is_folded_constant() -> None:
    integrator = _integrator_class()(k=2**-22)
    hir = optimize(lower(integrator.__call__))
    assert "k" not in {s.name for s in hir.state_slots}
    assert all(not (isinstance(n, StateRead) and n.slot == "k") for n in hir.nodes.values())


def test_stateful_reset_state_is_the_instance_snapshot() -> None:
    # The reset value is whatever the instance holds at synthesis time, including post-construction mutation.
    integrator = _integrator_class()(k=2**-22)
    integrator.y = 1.5
    slots = {s.name: s for s in lower(integrator.__call__).state_slots}
    assert cast(FloatConst, slots["y"].reset_value).value == 1.5


def test_init_method_target_is_lowered_as_a_state_writer() -> None:
    # An __init__ is just a method that assigns self attributes; the frontend lowers it, treating those attributes as
    # the state it writes (public ones are exposed as state ports, private ones stay internal).
    integrator = _integrator_class()(k=2**-22)
    hir = lower(integrator.__init__)
    assert {slot.name for slot in hir.state_slots} == {"k", "y", "_x_prev"}
    assert [o.name for o in hir.outputs] == ["state_k", "state_y"]


def test_class_object_target_is_rejected() -> None:
    with pytest.raises(SourceUnavailable, match="bound method"):
        lower(_integrator_class())


def test_method_without_return_exposes_public_state() -> None:
    class Accumulator:
        def __init__(self) -> None:
            self.total = 0.0

        def update(self, x: float) -> None:
            self.total = self.total + x

    hir = lower(Accumulator().update)
    assert [o.name for o in hir.outputs] == ["state_total"]
    assert {s.name for s in hir.state_slots} == {"total"}


def test_assigning_uninitialized_attribute_is_rejected() -> None:
    class Bad:
        def __init__(self) -> None:
            self.y = 0.0

        def __call__(self, x: float) -> float:
            self.scratch = x
            return self.y

    with pytest.raises(UnsupportedConstruct, match="does not exist on the component"):
        lower(Bad().__call__)


def test_read_only_self_attribute_real_part_folds_through() -> None:
    # ``self.y.real`` on a read-only float attribute is just ``self.y`` (a float is its own real part); the frontend
    # reads it permissively and folds the access, lowering to ``x + self.y``.
    class ReadsReal:
        def __init__(self) -> None:
            self.y = 0.0

        def __call__(self, x: float) -> float:
            return x + self.y.real

    hir = lower(ReadsReal().__call__)
    assert [o.name for o in hir.outputs] == ["out_0"]


def test_stateful_tuple_assignment_to_attributes() -> None:
    # Unpacking into self attributes must register both as persistent state; the swap reads the live-ins first.
    class Rotate:
        def __init__(self) -> None:
            self.x = 1.0
            self.y = 2.0

        def step(self, k: float) -> float:
            self.x, self.y = self.y, self.x + k
            return self.x

    hir = lower(Rotate().step)
    assert {s.name for s in hir.state_slots} == {"x", "y"}
    assert "state_x" in {o.name for o in hir.outputs}


def test_numpy_array_state_decomposes_like_a_list() -> None:
    import numpy.typing as npt

    @dataclasses.dataclass
    class Filt:
        v: npt.NDArray[np.float64]  # shape-less annotation: holoso infers the length from the reset value

        def step(self, a: float) -> None:
            self.v = self.v * a

    hir = lower(Filt(np.array([1.0, 2.0, 3.0])).step)
    assert {s.name for s in hir.state_slots} == {"v_0", "v_1", "v_2"}
    assert [o.name for o in hir.outputs] == ["state_v_0", "state_v_1", "state_v_2"]


def test_jaxtyping_array_field_lowers_and_is_validated() -> None:
    from jaxtyping import Float64

    @dataclasses.dataclass
    class Filt:
        v: Float64[np.ndarray, "3"]

        def step(self, a: float) -> None:
            self.v = self.v * a

    assert {s.name for s in lower(Filt(np.array([1.0, 2.0, 3.0])).step).state_slots} == {"v_0", "v_1", "v_2"}
    with pytest.raises(UnsupportedConstruct, match="declared array type"):
        lower(Filt(np.array([1.0, 2.0, 3.0, 4.0])).step)  # value shape (4,) violates the declared "3"


def test_numpy_integer_array_state_does_not_coerce_to_real() -> None:
    # The reset fixes the slot schema per cell (B1): an integer-reset array slot cannot take float cells.
    @dataclasses.dataclass
    class Filt:
        v: np.ndarray

        def step(self, a: float) -> None:
            self.v = self.v * a

    with pytest.raises(UnsupportedConstruct, match="stores an incompatible type at cell"):
        lower(Filt(np.array([2, 3])).step)


def test_matrix_state_decomposes_row_major() -> None:
    @dataclasses.dataclass
    class Filt:
        m: np.ndarray

        def step(self, a: float) -> None:
            self.m = self.m * a

    hir = lower(Filt(np.array([[1.0, 2.0], [3.0, 4.0]])).step)
    assert [s.name for s in hir.state_slots] == ["m_0_0", "m_0_1", "m_1_0", "m_1_1"]
    assert [cast(FloatConst, s.reset_value).value for s in hir.state_slots] == [1.0, 2.0, 3.0, 4.0]
    assert [o.name for o in hir.outputs] == ["state_m_0_0", "state_m_0_1", "state_m_1_0", "state_m_1_1"]


def test_three_dimensional_array_state_is_rejected() -> None:
    @dataclasses.dataclass
    class Filt:
        m: np.ndarray

        def step(self, a: float) -> None:
            self.m = self.m * a

    with pytest.raises(UnsupportedConstruct, match="3-D"):
        lower(Filt(np.zeros((2, 2, 2))).step)


def test_ekf1_stateful_structure() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    import ekf1_stateful

    filt = ekf1_stateful.Ekf1(
        x=[0.0, 0.0, 0.0], P_urt=[1.0, 0.0, 0.0, 1.0, 0.0, 1.0], R_diag=[1.0, 1.0], Q_diag=np.array([1.0, 1.0, 1.0])
    )
    hir = lower(filt.update)
    assert hir.input_names() == ["dt", "u_shunt", "di_dt"]  # self dropped; keyword-only params become inputs
    assert [o.name for o in hir.outputs] == ["state_x_0", "state_x_1", "state_x_2"] + [
        f"state_P_urt_{i}" for i in range(6)
    ]
    assert {s.name for s in hir.state_slots} == {f"x_{i}" for i in range(3)} | {f"P_urt_{i}" for i in range(6)}
    assert _arith_count(hir, FloatDiv) == 1  # the inlined kernel's single 1/x21


def test_vector_state_decomposes_to_per_element_slots() -> None:
    class Vec:
        def __init__(self) -> None:
            self.v = [1.0, 2.0, 3.0]

        def update(self, a: float) -> None:
            self.v = [self.v[0] + a, self.v[1], self.v[2]]

    hir = lower(Vec().update)
    assert {s.name: cast(FloatConst, s.reset_value).value for s in hir.state_slots} == {
        "v_0": 1.0,
        "v_1": 2.0,
        "v_2": 3.0,
    }
    assert [o.name for o in hir.outputs] == ["state_v_0", "state_v_1", "state_v_2"]


def test_vector_state_shape_mismatch_is_rejected() -> None:
    class Vec:
        def __init__(self) -> None:
            self.v = [0.0, 0.0]

        def update(self, a: float) -> None:
            self.v = [a]

    with pytest.raises(UnsupportedConstruct, match="2-element vector"):
        lower(Vec().update)


def test_vector_state_nested_shape_is_rejected() -> None:
    # A nested aggregate has the right leaf count (2) but the wrong shape: the slot layout is a flat 2-vector, so the
    # next transaction would reconstruct a flat shape that disagrees with the one written this transaction.
    class Vec:
        def __init__(self) -> None:
            self.v = [0.0, 0.0]

        def update(self, a: float, b: float) -> None:
            self.v = [[a, b]]  # type: ignore[list-item]

    with pytest.raises(UnsupportedConstruct, match="incompatible shape"):
        lower(Vec().update)


def test_vector_state_slot_name_collision_is_rejected() -> None:
    # The vector ``v`` decomposes into slot ``v_0``, which would alias the distinct scalar attribute ``v_0``.
    class Vec:
        def __init__(self) -> None:
            self.v = [1.0]
            self.v_0 = 2.0

        def update(self, a: float) -> None:
            self.v = [a]
            self.v_0 = a + 1.0

    with pytest.raises(
        UnsupportedConstruct, match="state slot name collision on 'v_0' between distinct component attributes"
    ):
        lower(Vec().update)


def test_dataclass_instance_is_stateful() -> None:
    @dataclasses.dataclass
    class Acc:
        total: float
        gain: list  # type: ignore[type-arg]

        def step(self, x: float) -> None:
            self.total = self.total + x * self.gain[0]

    hir = lower(Acc(0.0, [2.0]).step)
    assert {s.name for s in hir.state_slots} == {"total"}  # gain is read-only config, not state
    assert [o.name for o in hir.outputs] == ["state_total"]


def test_attribute_written_on_one_arm_becomes_a_phi() -> None:
    # The update lives in only one arm (anti-windup style); its live-out is a phi against the live-in. The condition is
    # a dynamic comparison so a real branch is emitted (a read-only boolean attribute would fold the branch away).
    class Clamp:
        def __init__(self) -> None:
            self.acc = 0.0

        def __call__(self, x: float) -> float:
            if x > 0.0:
                self.acc = x
            return self.acc

    raw = lower(Clamp().__call__)
    slots = {s.name: s for s in raw.state_slots}
    assert isinstance(raw.nodes[slots["acc"].live_out], Phi)
    # The empty-else diamond then if-converts: the merge becomes select(cond, written, live_in) -- a data mux.
    hir = optimize(raw)
    slots = {s.name: s for s in hir.state_slots}
    live_out = hir.nodes[slots["acc"].live_out]
    assert isinstance(live_out, Operation) and isinstance(live_out.operator, Select)


def test_reassigning_the_instance_parameter_self_is_rejected() -> None:
    # ``self`` is the fixed instance the attributes resolve against, not a value: ``self.x`` keeps reading the original
    # instance regardless of any later ``self = ...``, so rebinding it (any form) would silently miscompile -- rejected.
    class _PlainAssign:
        def __init__(self) -> None:
            self.a = 1.0

        def __call__(self, x: float) -> float:
            self = x  # type: ignore[assignment]
            return self.a

    class _Walrus:
        def __init__(self) -> None:
            self.a = 1.0

        def __call__(self, x: float) -> float:
            y = (self := x)  # type: ignore[assignment]
            return self.a + y

    class _Augmented:
        def __init__(self) -> None:
            self.a = 1.0

        def __call__(self, x: float) -> float:
            self += x  # type: ignore[operator, assignment]
            return self.a

    class _ForCounter:
        def __init__(self) -> None:
            self.a = 1.0

        def __call__(self, x: float) -> float:
            for self in range(2):  # type: ignore[assignment]
                pass
            return self.a

    class _Unpack:
        def __init__(self) -> None:
            self.a = 1.0

        def __call__(self, x: float) -> float:
            self, y = x, x  # type: ignore[assignment]
            return self.a + y

    for ctor in (_PlainAssign, _Walrus, _Augmented, _ForCounter, _Unpack):
        with pytest.raises(UnsupportedConstruct, match="instance parameter"):
            lower(ctor().__call__)


def test_writing_a_self_attribute_and_a_plain_local_named_self_are_accepted() -> None:
    # The rejection must not touch a legitimate attribute write (persistent state) or a plain (non-method) function
    # whose local happens to be named ``self`` -- there is no instance there, so ``self`` is an ordinary local.
    class _StateWrite:
        def __init__(self) -> None:
            self.a = 1.0

        def __call__(self, x: float) -> float:
            self.a = self.a + x
            return self.a

    lower(_StateWrite().__call__)  # no exception

    def plain(x: float) -> float:
        self = x  # an ordinary local in a plain function (no instance)
        return self + 1.0

    lower(plain)  # no exception


def test_readonly_scan_stops_at_a_returning_folded_arm() -> None:
    # Regression (review #1): a folded ``if`` whose taken arm returns makes the rest unreachable; the read-only scan
    # must stop there, so an attribute assigned only afterwards is not wrongly counted as written. Here ``gate`` is
    # read-only, so the first guard folds and its return is permitted -- which fails if ``gate`` is mismarked.
    class K:
        def __init__(self) -> None:
            self.gate = True
            self.y = 0.0

        def __call__(self, u: float) -> float:
            if self.gate:
                return u + 1.0
            self.y = u
            if True:
                return self.y
            self.gate = False  # unreachable; must not mark ``gate`` assigned

    assert lower(K().__call__).state_slots == []


def test_float_cast_connective_comparison_condition_folds_without_spurious_state() -> None:
    # Regression (review #2): ``float(X or True) > 0.5`` is the constant True; the guard must fold so the dead else-arm
    # write does NOT become a persistent-state slot (and output port).
    class K:
        def __init__(self) -> None:
            self.y = 0.0
            self.z = 0.0

        def __call__(self, u: float) -> float:
            if float(u > 0.0 or True) > 0.5:
                self.y = u
            else:
                self.z = u  # unreachable
            return self.y

    hir = optimize(lower(K().__call__))
    assert [slot.name for slot in hir.state_slots] == ["y"]
    assert len(hir.blocks) == 1


def test_absorbing_attribute_connective_keeps_a_dead_arm_attribute_read_only() -> None:
    # Regression (review #3): ``self.flag or True`` folds in the read-only scan (attribute opaque, absorbing operand
    # decides it), so ``self.other`` -- written only in the dead else -- stays read-only, and the later guard on it
    # folds rather than leaking ``self.z`` as state.
    class K:
        def __init__(self) -> None:
            self.flag = True
            self.other = True
            self.y = 0.0
            self.z = 0.0

        def __call__(self, u: float) -> float:
            if self.flag or True:
                pass
            else:
                self.other = False  # unreachable
            if self.other:
                self.y = u
            else:
                self.z = u  # unreachable
            return self.y

    hir = lower(K().__call__)
    assert [slot.name for slot in hir.state_slots] == ["y"]


def test_read_only_scan_does_not_misfold_a_reassigned_for_counter() -> None:
    # Regression (review, miscompile): the read-only scan must not bind a static ``for`` counter and then fold a
    # counter-dependent condition against a STALE value -- which would drop ``_flag`` from the assigned set, wrongly
    # treat it as read-only, and fold the later ``if self._flag:`` to a fixed arm, diverging from lowering. The scan
    # leaves the counter unbound (conservative), so the body's writes are recorded and ``_flag`` stays state. The
    # marker loop iterates a FLOAT so the runtime rebind keeps the storage schema (B1).
    class K:
        def __init__(self) -> None:
            self._flag = False
            self.y = 0.0
            self.z = 0.0

        def __call__(self, u: float) -> float:
            for i in (0.0,):
                i = u  # noqa: PLW2901  # the loop counter is reassigned to a runtime value
                if i > 0.0:
                    self._flag = True
            if self._flag:
                self.z = u
            else:
                self.y = u
            return self.y

    slots = {slot.name for slot in lower(K().__call__).state_slots}
    assert "_flag" in slots and "z" in slots


# ---------------------------------------------------------------- reachability scan vs lowering


def test_state_write_only_on_a_folded_away_shape_branch_is_not_state() -> None:
    # The scan runs before the body is lowered, so it cannot fold a shape query and descends both arms, registering the
    # write. Lowering folds the branch away and never touches the attribute, which therefore keeps its reset value for
    # good and is not state. Before ``_prune_untouched_state`` this crashed with a raw KeyError from slot registration.
    from jaxtyping import Float64

    class DeadShapeBranch:
        def __init__(self) -> None:
            self.s = 0.0

        def step(self, x: Float64[np.ndarray, "3"]) -> float:
            if len(x) == 4:  # statically false for a declared 3-vector
                self.s = x[0]
            return x[0]  # type: ignore[no-any-return]

    hir = lower(DeadShapeBranch().step)
    assert [slot.name for slot in hir.state_slots] == []
    assert [o.name for o in hir.outputs] == ["out_0"]

    class DeadAggregateLoop:
        def __init__(self) -> None:
            self.a = 0.0

        def step(self, x: float) -> float:
            for _ in []:  # zero trips, so the write is unreachable
                self.a = x
            return x

    assert [slot.name for slot in lower(DeadAggregateLoop().step).state_slots] == []

    class AlsoRead:
        # When the attribute is also READ on a live path, the read folds to the reset value: the only write sits on
        # a folded-away branch, so the attribute is a frozen constant -- no slot, no port -- exactly as in Python,
        # where the instance attribute never changes.
        def __init__(self) -> None:
            self.s = 0.25

        def step(self, x: Float64[np.ndarray, "3"]) -> float:
            if len(x) == 4:
                self.s = x[0]
            return self.s + x[0]  # type: ignore[no-any-return]

    hir = lower(AlsoRead().step)
    assert [slot.name for slot in hir.state_slots] == []
    assert [o.name for o in hir.outputs] == ["out_0"]
    sim = holoso.synthesize(AlsoRead().step, default_ops(FloatFormat(11, 52)), name="alsoread").numerical_model
    simulator = sim.elaborate()
    reference = AlsoRead()
    inputs = np.array([1.5, 0.0, 0.0])
    for _ in range(3):
        assert float(simulator.run(*inputs.tolist())[0]) == pytest.approx(reference.step(inputs))


def test_state_write_under_an_aggregate_for_is_not_dropped_by_a_stale_counter() -> None:
    # ``for i in <aggregate>`` binds a runtime value, so the target's compile-time binding must be demoted in the
    # reachability scan exactly as lowering demotes it. Otherwise the scan folds ``i == 2.0`` on the leaked counter of
    # the preceding marker loop (a FLOAT, keeping the storage schema across the rebind), walks only one arm, misses
    # the write, and the state slot silently disappears -- the module would return the reset constant forever.
    from jaxtyping import Float64

    class StaleCounter:
        def __init__(self) -> None:
            self.s = 0.0

        def step(self, x: Float64[np.ndarray, "2"]) -> float:
            for i in (0.0, 1.0, 2.0):
                pass
            for i in x:  # i is demoted here; the scan must not keep the leaked value 2.0
                if i == 2.0:
                    pass
                else:
                    self.s = i
            return self.s

    hir = lower(StaleCounter().step)
    assert [slot.name for slot in hir.state_slots] == ["s"]

    sim = holoso.synthesize(StaleCounter().step, default_ops(FloatFormat(11, 52)), name="stale").numerical_model
    simulator = sim.elaborate()
    inputs = np.array([5.0, 7.0])
    assert float(simulator.run(*inputs.tolist())[0]) == pytest.approx(StaleCounter().step(inputs))


def test_state_write_after_a_raise_does_not_poison_the_read_only_scan() -> None:
    # A raise ends the block, so the assignment below it is unreachable and must not mark ``flag`` as written --
    # otherwise ``flag`` stops being a read-only constant, the guard becomes a runtime branch, and the raise in the
    # statically-dead else arm is misreported as sitting on a data-dependent path.
    class GuardedByAReadOnlyFlag:
        def __init__(self) -> None:
            self.flag = True
            self.y = 0.0

        def step(self, a: float) -> float:
            if self.flag:
                self.y = a
            else:
                raise ValueError("flag must be set")
                self.flag = False  # noqa: F841  # unreachable: the raise ends the block

            return self.y

    hir = lower(GuardedByAReadOnlyFlag().step)
    assert [slot.name for slot in hir.state_slots] == ["y"]  # flag stays a read-only constant


def test_an_untouched_state_attribute_is_not_resurrected_by_an_unrelated_branch() -> None:
    # _merge_state must not load the live-in of an attribute NEITHER arm touched: both arms start from the same
    # pre-branch state, so doing so would conjure a register (and a public port) out of a branch that never
    # mentions the attribute, undoing _prune_untouched_state.
    from jaxtyping import Float64

    class DeadWritePlusUnrelatedBranch:
        def __init__(self) -> None:
            self.s = 0.0

        def step(self, x: Float64[np.ndarray, "3"]) -> float:
            if len(x) == 4:  # dead: the write never happens
                self.s = x[0]
            r = x[0]
            if x[1] > 0.0:  # an unrelated dynamic branch, which merges state
                r = x[2]
            return r  # type: ignore[no-any-return]

    hir = lower(DeadWritePlusUnrelatedBranch().step)
    assert [slot.name for slot in hir.state_slots] == []
    assert [o.name for o in hir.outputs] == ["out_0"]


def test_a_scan_never_folds_a_shape_query_against_an_environment_lowering_will_not_have() -> None:
    # The loop-carried scan walks a while body BEFORE its phis exist, so the environment it sees is the preheader's.
    # Were it allowed to resolve a name there, it would fold ``i.ndim == 1`` against the scalar ``i`` bound before the
    # loop, miss the state write in the arm it skipped, open no loop phi, and discard the accumulation entirely --
    # a silent miscompile returning the reset value forever.
    from jaxtyping import Float64

    class AccumulateOverRows:
        def __init__(self) -> None:
            self.s = 0.0

        def step(self, c: bool, m: Float64[np.ndarray, "2 3"]) -> float:
            i = 0.0
            while c:
                for i in m:
                    if i.ndim == 1:
                        self.s = self.s + i[0]
                i = 0.0
                c = False
            return self.s

    rows = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert AccumulateOverRows().step(True, rows) == 5.0  # the kernel is runnable Python
    assert [slot.name for slot in lower(AccumulateOverRows().step).state_slots] == ["s"]

    sim = holoso.synthesize(
        AccumulateOverRows().step, default_ops(FloatFormat(11, 52)), name="accumulate_rows"
    ).numerical_model
    assert float(sim.elaborate().run(True, *rows.flatten().tolist())[0]) == pytest.approx(5.0)


def test_a_state_attribute_read_only_inside_a_while_loop_keeps_its_slot() -> None:
    # A while loop restores the pre-loop state environment on exit, dropping whatever its body loaded, so membership
    # there cannot decide whether an attribute was touched. Pruning on it would drop a slot whose StateRead is still
    # in the HIR, leaving the register allocator to trip over an undeclared slot.
    class ReadOnlyInsideLoop:
        def __init__(self) -> None:
            self.gain = 2.0

        def update(self, x: float) -> float:
            v = [1.0, 2.0]
            if len(v) == 3:  # dead: the scan cannot fold it, so ``gain`` is over-registered as state
                self.gain = x
            acc = 0.0
            while acc < x:
                acc = acc + self.gain
            return acc

    assert ReadOnlyInsideLoop().update(5.0) == 6.0
    hir = lower(ReadOnlyInsideLoop().update)
    slots = {slot.name for slot in hir.state_slots}
    reads = {node.slot for node in hir.nodes.values() if isinstance(node, StateRead)}
    assert reads <= slots  # every StateRead names a declared slot

    sim = holoso.synthesize(
        ReadOnlyInsideLoop().update, default_ops(FloatFormat(11, 52)), name="read_only_in_loop"
    ).numerical_model
    assert float(sim.elaborate().run(5.0)[0]) == pytest.approx(6.0)


def test_only_a_write_lowering_reaches_is_validated() -> None:
    # The scan walks paths lowering folds away, so it validates nothing: a write it cannot turn into state is passed
    # over, and the rejection happens at the write itself, if and when lowering gets there. A dead branch assigning an
    # attribute the instance never had is dead code, exactly as it is in Python.
    from jaxtyping import Float64

    class DeadWriteToAnUninitializedAttribute:
        def __init__(self) -> None:
            self.ok = 0.0

        def step(self, v: Float64[np.ndarray, "2"]) -> float:
            if v.ndim == 2:  # statically false for a vector
                self.never_initialized = 1.0
            return v[0]  # type: ignore[no-any-return]

    assert DeadWriteToAnUninitializedAttribute().step(np.array([3.0, 4.0])) == 3.0  # runnable Python
    assert [slot.name for slot in lower(DeadWriteToAnUninitializedAttribute().step).state_slots] == []

    class ReachableWriteToAnUninitializedAttribute:
        def __init__(self) -> None:
            self.ok = 0.0

        def step(self, x: float) -> float:
            if x > 0.0:  # a runtime arm is lowered, so the write is reached
                self.never_initialized = x
            return x

    with pytest.raises(
        UnsupportedConstruct,
        match="state attribute 'never_initialized' does not exist on the component at compile time",
    ):
        lower(ReachableWriteToAnUninitializedAttribute().step)


def test_an_all_integer_state_selector_stays_a_typed_integer_slot() -> None:
    # An integer reset with only integer stores keeps a typed integer slot: the exact 2**53 + 1 never enters the
    # float bank, the guard compares integer-to-integer, and the kernel is contained at the MIR integer boundary.
    inexact = 2**53 + 1  # the first integer float64 cannot represent

    class Selector:
        def __init__(self) -> None:
            self.selector = inexact
            self.total = 0.0

        def step(self, x: float) -> float:
            if x > 100.0:  # a runtime guard, so `selector` really is persistent state
                self.selector = 0
            if self.selector == 2**53:  # False in Python: the integer slot compares exactly, never rounded
                self.total = self.total + 100.0 * x
            else:
                self.total = self.total + x
            return self.total

    assert Selector().step(1.0) == 1.0  # Python compares the integer exactly
    hir = lower(Selector().step)
    slot = next(s for s in hir.state_slots if s.name == "selector")
    assert isinstance(hir.nodes[slot.live_out].type, IntType)


def test_an_integer_vector_state_reset_keeps_exact_per_element_slots() -> None:
    class ExactVector:
        # 2**53 itself round-trips into the float bank exactly, and so does any small integer.
        def __init__(self) -> None:
            self.taps = [1, 2**53, -3]
            self.y = 0.0

        def step(self, x: float) -> float:
            self.taps = [self.taps[0], self.taps[1], self.taps[2]]  # written, so the vector really is state
            self.y = self.y + self.taps[1] * x
            return self.y

    assert [slot.name for slot in lower(ExactVector().step).state_slots] == ["taps_0", "taps_1", "taps_2", "y"]


def test_state_slot_names_only_collide_among_the_attributes_lowering_keeps() -> None:
    # The scan over-registers `v_0` from a write lowering folds away, and `v_0` is also the first slot of the vector
    # `v`. Checking for the collision before the prune would reject a kernel whose colliding attribute is dead code.
    from jaxtyping import Float64

    class DeadCollider:
        def __init__(self) -> None:
            self.v = np.array([1.0, 2.0])
            self.v_0 = 100.0

        def step(self, x: Float64[np.ndarray, "1"]) -> float:
            if x.ndim == 2:  # dead: the scan cannot fold it, so `v_0` is over-registered
                self.v_0 = x[0]
            self.v = self.v + x[0]
            return self.v[0]  # type: ignore[no-any-return]

    assert [slot.name for slot in lower(DeadCollider().step).state_slots] == ["v_0", "v_1"]

    class LiveCollider:
        def __init__(self) -> None:
            self.v = np.array([1.0, 2.0])
            self.v_0 = 100.0

        def step(self, x: float) -> float:
            self.v_0 = x  # reached, so both attributes really do claim the slot name `v_0`
            self.v = self.v + x
            return self.v[0]  # type: ignore[no-any-return]

    with pytest.raises(
        UnsupportedConstruct, match="state slot name collision on 'v_0' between distinct component attributes"
    ):
        lower(LiveCollider().step)


def test_a_loop_carries_only_attributes_that_are_really_state() -> None:
    # The scan collects self-attribute writes syntactically, so a write it cannot turn into state must not open a
    # loop-header phi for it; otherwise the phi's live-in lookup fails with a bare KeyError.
    from jaxtyping import Float64

    class DeadWriteInLoop:
        def __init__(self) -> None:
            self.ok = 0.0

        def step(self, run: bool, v: Float64[np.ndarray, "3"]) -> float:
            while run:
                if v.ndim == 2:  # dead
                    self.never_initialized = v[1]
                run = False
            return v[0]  # type: ignore[no-any-return]

    assert DeadWriteInLoop().step(True, np.array([2.0, 3.0, 5.0])) == 2.0
    assert [slot.name for slot in lower(DeadWriteInLoop().step).state_slots] == []


def test_a_shape_query_reads_the_reset_value_not_the_state_decomposition() -> None:
    # `.ndim` of a read-only 3-D array attribute is a compile-time integer; only STATE is restricted to 1-D and 2-D.
    from jaxtyping import Float64

    class Cube:
        def __init__(self) -> None:
            self.cube = np.zeros((2, 2, 2))

        def step(self, v: Float64[np.ndarray, "2"]) -> float:
            if v.ndim == 2:
                if self.cube.ndim == 3:
                    return v[1]  # type: ignore[no-any-return]
            return v[0]  # type: ignore[no-any-return]

    assert Cube().step(np.array([2.0, 9.0])) == 2.0
    assert [o.name for o in lower(Cube().step).outputs] == ["out_0"]


def test_a_write_is_validated_only_where_lowering_reaches_it() -> None:
    # The scan walks paths lowering folds away, so it cannot validate. Each attribute below is unrepresentable as
    # state; a dead write to it is dead code, a reachable one is an error. Both halves must hold, in a loop body too.
    from jaxtyping import Float64

    class Descriptor:
        def __init__(self) -> None:
            self.__dict__["p"] = 1.0
            self.y = 0.0

        @property
        def p(self) -> float:
            return 2.0

        @p.setter
        def p(self, value: float) -> None:
            pass

    class DeadDescriptorWriteInLoop(Descriptor):
        def step(self, run: bool, v: Float64[np.ndarray, "2"]) -> float:
            while run:
                if v.ndim == 2:  # dead
                    self.p = v[0]
                run = False
            return v[0]  # type: ignore[no-any-return]

    class LiveDescriptorWriteInLoop(Descriptor):
        def step(self, run: bool, v: Float64[np.ndarray, "2"]) -> float:
            while run:
                self.p = v[0]
                run = False
            return v[0]  # type: ignore[no-any-return]

    class DeadCubeWrite:
        def __init__(self) -> None:
            self.cube = np.zeros((2, 2, 2))
            self.ok = 0.0

        def step(self, v: Float64[np.ndarray, "2"]) -> float:
            if v.ndim == 2:  # dead: a 3-D attribute cannot be state
                self.cube = v
            return v[0]  # type: ignore[no-any-return]

    v = np.array([1.0, 2.0])
    assert DeadDescriptorWriteInLoop().step(True, v) == 1.0
    assert [slot.name for slot in lower(DeadDescriptorWriteInLoop().step).state_slots] == []
    assert [slot.name for slot in lower(DeadCubeWrite().step).state_slots] == []

    with pytest.raises(UnsupportedConstruct, match="descriptor"):
        lower(LiveDescriptorWriteInLoop().step)


def test_an_integer_the_float_datapath_cannot_hold_never_enters_it() -> None:
    # An integer that rounds would read back as another number, so a comparison against the source literal flips.
    # The guard is on the value entering the datapath, not on the spelling: a reset, a literal, or a module constant.
    inexact, colliding = 2**53 + 1, 2**53

    class WrittenFromAModuleConstant:
        def __init__(self) -> None:
            self.selector = 0
            self.total = 0.0

        def step(self, x: float) -> float:
            if x > 100.0:
                self.selector = _INEXACT_INTEGER
            self.total = self.total + (100.0 * x if self.selector == colliding else x)
            return self.total

    reference = WrittenFromAModuleConstant()
    assert [reference.step(v) for v in (1.0, 101.0)] == [1.0, 102.0]  # the integer never equals 2**53 in Python
    hir = lower(WrittenFromAModuleConstant().step)  # the inexact integer stays a typed integer, never a rounded float
    selector = next(slot for slot in hir.state_slots if slot.name == "selector")
    assert isinstance(hir.nodes[selector.live_out].type, IntType)

    class HugeReset:
        def __init__(self) -> None:
            self.counter = 10**400  # beyond the float range entirely

        def step(self, x: float) -> float:
            if x > 0.0:
                self.counter = 0
            return x

    hir = lower(HugeReset().step)  # the huge integer is kept exact as an integer, not overflowed into a float
    (counter,) = hir.state_slots
    assert isinstance(hir.nodes[counter.live_out].type, IntType)

    class ReadOnlyInexact:
        # An inexact integer attribute in a float-add position promotes and rounds, exactly as Python's `int + float`
        # promotes -- accepted fastmath precision loss, not a rejection.
        def __init__(self) -> None:
            self.offset = inexact

        def step(self, x: float) -> float:
            return self.offset + x

    rounded = lower(ReadOnlyInexact().step)
    assert float(2**53) in [n.value for n in rounded.nodes.values() if isinstance(n, FloatConst)]


def test_mixed_int_float_static_comparison_folds_exactly() -> None:
    # Regression (TODO): a static comparison mixing an integer expression with a float must compare exactly, as
    # Python does; a float64 fold of the integer side rounds 2**53 + 1 onto 2**53 and takes the wrong arm silently.
    class WrongArmGuard:
        def __init__(self) -> None:
            self.x = 0.0

        def step(self, v: float) -> float:
            if _BIG_A + _BIG_B == _BIG_F:  # False in Python: the integer sum compares exactly
                self.x = v
            return self.x

    assert WrongArmGuard().step(1.0) == 0.0
    hir = lower(WrongArmGuard().step)
    assert [slot.name for slot in hir.state_slots] == []
    assert len(optimize(hir).blocks) == 1

    class RightArmGuard:
        def __init__(self) -> None:
            self.x = 0.0

        def step(self, v: float) -> float:
            if _BIG_A + _BIG_B > _BIG_F:  # True in Python: the fold must still take the arm, not reject
                self.x = v
            return self.x

    assert RightArmGuard().step(1.0) == 1.0
    assert [slot.name for slot in lower(RightArmGuard().step).state_slots] == ["x"]


def test_read_only_inexact_int_attribute_comparison_folds_exactly() -> None:
    # Regression (TODO): a read-only integer attribute keeps its exact value in a static comparison; the float64
    # fold of the attribute would round it onto the comparand and take the wrong arm silently.
    class Selector:
        def __init__(self) -> None:
            self._sel = 2**53 + 1
            self.y = 0.0

        def step(self, v: float) -> float:
            if self._sel == _BIG_F:  # False in Python
                self.y = v
            return self.y

    assert Selector().step(1.0) == 0.0
    hir = lower(Selector().step)
    assert [slot.name for slot in hir.state_slots] == []
    assert len(optimize(hir).blocks) == 1


def test_np_int_array_element_comparison_follows_numpy_semantics() -> None:
    # Companion pin to the Python-int exactness fix: a numpy scalar operand must NOT be folded exactly, because
    # numpy itself converts an np.int64 to float64 in a comparison -- np.int64(2**53 + 1) == float(2**53) is True in
    # numpy -- so each operand folds under its own source semantics.
    class Selector:
        def __init__(self) -> None:
            self.y = 0.0

        def step(self, v: float) -> float:
            if _INT_TABLE[0, 0] == _BIG_F:  # True under numpy: the element converts to float64 and rounds
                self.y = v
            return self.y

    assert Selector().step(1.0) == 1.0
    hir = lower(Selector().step)
    assert [slot.name for slot in hir.state_slots] == ["y"]
    assert len(optimize(hir).blocks) == 1


def test_read_only_object_attribute_ndim_folds_as_a_constant() -> None:
    # ``self.config.ndim`` reads the stored object's class attribute (1) as a compile-time constant, so ``ndim == 0``
    # folds to False and the kernel is ``-x`` -- matching Python.
    class Config:
        ndim = 1

    class Kernel:
        def __init__(self) -> None:
            self.config = Config()

        def step(self, x: float) -> float:
            return x if self.config.ndim == 0 else -x  # Python: -x, because Config.ndim is 1

    assert Kernel().step(3.0) == -3.0
    model = holoso.synthesize(
        Kernel().step, default_ops(FloatFormat(11, 52)), name="ndim_fold"
    ).numerical_model.elaborate()
    for x in (3.0, -2.0, 0.0):
        assert float(model.run(x)[0]) == -x


def test_a_scan_never_rejects_an_arm_lowering_folds_away() -> None:
    # The scan descends both arms of a shape-dependent branch, so it must not validate what it finds there.
    from jaxtyping import Float64

    class DeadInvalidShapeQuery:
        def __init__(self) -> None:
            self.values = [1.0, 2.0]
            self.total = 0.0

        def step(self, v: Float64[np.ndarray, "2"], x: float) -> float:
            if v.ndim == 2:  # dead
                if self.values.ndim == 1:  # type: ignore[attr-defined]  # a list has no .ndim
                    self.total = x
            return self.total

    assert DeadInvalidShapeQuery().step(np.zeros(2), 1.0) == 0.0
    # `total` is written only under the folded-away arm, so it is a frozen constant -- no slot, no port -- and the
    # live read folds to the reset value; the point is that the kernel LOWERS at all.
    hir = lower(DeadInvalidShapeQuery().step)
    assert [slot.name for slot in hir.state_slots] == []
    assert [o.name for o in hir.outputs] == ["out_0"]
    model = holoso.synthesize(
        DeadInvalidShapeQuery().step, default_ops(FloatFormat(11, 52)), name="deadq"
    ).numerical_model
    assert float(model.elaborate().run(0.0, 0.0, 1.0)[0]) == DeadInvalidShapeQuery().step(np.zeros(2), 1.0) == 0.0


def test_a_nested_reset_sequence_is_shaped_like_the_aggregate_it_denotes() -> None:
    # `len(self.nested[0])` is 3 in Python, so the snapshot's shape must describe every axis, not just the outermost.
    from jaxtyping import Float64

    class NestedRows:
        def __init__(self) -> None:
            self.nested = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

        def step(self, v: Float64[np.ndarray, "3"]) -> float:
            acc = 0.0
            for i in range(len(self.nested[0])):
                acc = acc + v[i]
            return acc

    v = np.array([10.0, 20.0, 30.0])
    assert NestedRows().step(v) == 60.0
    _assert_shape_kernel_matches_python(NestedRows().step, v)


def test_a_ragged_or_empty_reset_sequence_still_has_a_length() -> None:
    from jaxtyping import Float64

    class RaggedRows:
        def __init__(self) -> None:
            self.ragged = [[1.0], [2.0, 3.0]]
            self.empty: list[float] = []

        def step(self, v: Float64[np.ndarray, "3"]) -> float:
            return v[len(self.ragged)] + v[len(self.ragged[1])] + v[len(self.empty)]  # type: ignore[no-any-return]

    v = np.array([10.0, 20.0, 30.0])
    assert RaggedRows().step(v) == 70.0
    _assert_shape_kernel_matches_python(RaggedRows().step, v)


def test_indexing_a_reset_sequence_of_arrays_yields_an_array() -> None:
    from jaxtyping import Float64

    class ArrayRows:
        def __init__(self) -> None:
            self.rows = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]

        def step(self, v: Float64[np.ndarray, "3"]) -> float:
            return v[self.rows[0].ndim]  # type: ignore[no-any-return]  # the element is the ndarray, not a list

    v = np.array([10.0, 20.0, 30.0])
    assert ArrayRows().step(v) == 20.0
    _assert_shape_kernel_matches_python(ArrayRows().step, v)


def test_a_shape_query_on_a_nested_reset_sequence_element_is_rejected() -> None:
    from jaxtyping import Float64

    class NestedNdim:
        def __init__(self) -> None:
            self.nested = [[1.0, 2.0]]

        def step(self, v: Float64[np.ndarray, "3"]) -> float:
            return v[self.nested[0].ndim]  # type: ignore[attr-defined, return-value]  # a list has no .ndim

    with pytest.raises(UnsupportedConstruct, match="a list has no attribute 'ndim'"):
        lower(NestedNdim().step)


def test_subscripting_a_non_container_reset_attribute_is_a_located_rejection() -> None:
    # Navigating the reset snapshot must not index whatever `__getitem__` a stored object happens to carry.
    from jaxtyping import Float64

    class ForeignAttr:
        def __init__(self) -> None:
            self.lookup = {"a": 1}

        def step(self, v: Float64[np.ndarray, "3"]) -> float:
            return v[len(self.lookup[0])]  # type: ignore[arg-type, index, no-any-return]  # a KeyError in Python

    with pytest.raises(UnsupportedConstruct):
        lower(ForeignAttr().step)


def test_a_scan_must_not_fold_a_counter_an_empty_aggregate_never_rebinds() -> None:
    # Lowering runs an empty aggregate's body zero times, so `i` keeps its outer value. A scan that walks the body once
    # and adopts the inner counter would fold `i == 1` away and never see the state write it guards.
    class EmptyAggregateCounter:
        def __init__(self) -> None:
            self.s = 0.0

        def step(self, a: float) -> float:
            for i in range(2):
                pass
            for _unused in []:  # type: ignore[var-annotated]
                for i in range(5):  # noqa: B007  # never runs; must not leak i == 4 into the scan
                    pass
            if i == 1:
                self.s = a
            return self.s

    reference = EmptyAggregateCounter()
    assert reference.step(7.0) == 7.0 and reference.s == 7.0
    assert [slot.name for slot in lower(EmptyAggregateCounter().step).state_slots] == ["s"]
    sim = holoso.synthesize(
        EmptyAggregateCounter().step, default_ops(FloatFormat(11, 52)), name="empty_aggregate_counter"
    ).numerical_model.elaborate()
    assert dict(zip([p.name for p in sim.outputs], [float(x) for x in sim.run(7.0)], strict=True))["state_s"] == 7.0


def test_a_scan_must_not_fold_a_branch_on_a_counter_the_loop_body_rebinds() -> None:
    # The aggregate loop's first trip rebinds `i` to a runtime value, so lowering takes the else arm on the second
    # trip. A scan that keeps `i == 0` static walks only the then arm and misses `self.s`, whose write then has
    # nowhere to land. The marker loop leaks a FLOAT `i` so the runtime rebind keeps the storage schema (B1).
    from jaxtyping import Float64

    class RebindingCounter:
        def __init__(self) -> None:
            self.s = 0.0

        def step(self, v: Float64[np.ndarray, "2"]) -> float:
            for i in (0.0,):
                pass
            for x in v:
                if i == 0:
                    i = x  # noqa: PLW2901
                else:
                    self.s = x
            return self.s

    reference = RebindingCounter()
    assert reference.step(np.array([5.0, 7.0])) == 7.0
    sim = holoso.synthesize(
        RebindingCounter().step, default_ops(FloatFormat(11, 52)), name="rebinding_counter"
    ).numerical_model.elaborate()
    assert (
        dict(zip([p.name for p in sim.outputs], [float(x) for x in sim.run(5.0, 7.0)], strict=True))["state_s"] == 7.0
    )


def test_a_scan_demotes_the_aggregate_target_before_discovering_body_rebinds() -> None:
    # The aggregate loop's target `x` leaks a stale value from an earlier same-named loop. Discovering what the
    # body rebinds must happen with `x` already demoted, or the fold of `if x != 0` hides the `j = x` rebind, `j` is
    # restored stale, and the guarded state write is missed -- tripping `assert attr in self._state_order`. The
    # marker loops leak FLOAT zeros so the runtime rebinds keep the storage schema (B1).
    class LeakedAggregateTarget:
        def __init__(self) -> None:
            self.s = 0.0

        def step(self, a: float) -> float:
            for x in (0.0,):  # noqa: B007  # leaks x == 0.0
                pass
            for j in (0.0,):  # noqa: B007  # leaks j == 0.0
                pass
            for x in [a]:  # noqa: B007  # aggregate: target demoted, body rebinds j
                if x != 0:
                    j = x  # noqa: PLW2901
            if j != 0:
                self.s = j
            return self.s

    reference = LeakedAggregateTarget()
    assert reference.step(5.0) == 5.0 and reference.s == 5.0
    sim = holoso.synthesize(
        LeakedAggregateTarget().step, default_ops(FloatFormat(11, 52)), name="leaked_aggregate_target"
    ).numerical_model.elaborate()
    assert dict(zip([p.name for p in sim.outputs], [float(x) for x in sim.run(5.0)], strict=True))["state_s"] == 5.0


def test_a_tuple_index_of_a_list_reset_attribute_is_a_located_rejection() -> None:
    # `self.rows[0,]` indexes a Python list with a one-tuple, which CPython rejects; the reset-state navigator must
    # not silently reinterpret it as the numpy-style `self.rows[0]`.
    from jaxtyping import Float64

    class TupleIndexedRows:
        def __init__(self) -> None:
            self.rows = [[1.0, 2.0]]

        def step(self, v: Float64[np.ndarray, "3"]) -> float:
            return v[len(self.rows[0,])]  # type: ignore[call-overload,no-any-return]  # a TypeError in Python

    with pytest.raises(UnsupportedConstruct):
        lower(TupleIndexedRows().step)


def test_a_returned_leaf_equal_to_a_public_live_out_rides_the_state_port_across_branches() -> None:
    # Review rounds 6+7: the return leaf and the state live-out read through DIFFERENT places, so a value merged
    # across branches arrives as two distinct-but-identical exit phis (the value must flow through a LOCAL with
    # per-arm stores -- returning self.y itself shares one phi through the Braun cache and never exercised the
    # fix). Dedup keys on the phi's structure. Round 7 extended the identity through pure conversion wrappers:
    # a store-side and a return-side IntToFloat live in different blocks, which the per-block interner keeps
    # distinct.
    class Latch:
        def __init__(self) -> None:
            self.y = 0.0

        def step(self, flag: bool, x: float) -> tuple[float]:
            if flag:
                r = x
                self.y = r
            else:
                r = -x
                self.y = r
            return (r,)

    result = holoso.synthesize(Latch().step, default_ops(FloatFormat(11, 52)), name="dedup_phi")
    assert [p.name for p in result.output_ports] == ["state_y"]
    elaborated = result.numerical_model.elaborate()
    assert float(elaborated.run(True, 3.0)[0]) == 3.0
    assert float(elaborated.run(False, 5.0)[0]) == -5.0

    class Wrapped:
        def __init__(self) -> None:
            self.y = 0.0

        def step(self, flag: bool) -> float:
            n = 1 if flag else 2
            self.y = n
            return n

    wrapped = holoso.synthesize(Wrapped().step, default_ops(FloatFormat(11, 52)), name="dedup_wrap")
    assert [p.name for p in wrapped.output_ports] == ["state_y"]
    assert float(wrapped.numerical_model.elaborate().run(True)[0]) == 1.0


def test_exit_identity_is_depth_bounded_and_catches_nested_merges() -> None:
    # Review round 8: the recursive dedup identity walked the whole exit DAG -- exponential on shared operands
    # (repeated squaring) and a RecursionError on x**1024 (an explicitly permitted power chain). The walk is
    # depth-capped and cycle-guarded, and recursing through phi ARMS dedups nested merges that differ only by
    # inner phi ids.
    def power_chain(x: float) -> float:
        return x**1024

    def repeated_squaring(x: float) -> float:
        for _ in range(30):
            x = x * x
        return x

    for kernel in (power_chain, repeated_squaring):
        lower(kernel)  # the pin is completion itself (previously RecursionError / effectively non-terminating)

    class Nested:
        def __init__(self) -> None:
            self.y = 0.0

        def step(self, a: bool, b: bool, x: float) -> float:
            if a:
                if b:
                    r = x
                    self.y = r
                else:
                    r = -x
                    self.y = r
            else:
                r = x + x
                self.y = r
            return r

    result = holoso.synthesize(Nested().step, default_ops(FloatFormat(11, 52)), name="nested_dedup")
    assert [p.name for p in result.output_ports] == ["state_y"]
    elaborated = result.numerical_model.elaborate()
    assert float(elaborated.run(True, True, 3.0)[0]) == 3.0
    assert float(elaborated.run(True, False, 3.0)[0]) == -3.0
    assert float(elaborated.run(False, True, 3.0)[0]) == 6.0


def test_exit_dedup_handles_deep_and_shared_exit_graphs() -> None:
    # Review round 9: the depth-4 identity cap missed dedup on deeper per-arm-store nesting; the memoized walk
    # shares subtree identities (repeated squaring stays linear) and dedups arbitrarily nested merges, while the
    # depth bound only guards genuinely chain-shaped graphs (x**1024 still lowers).
    class Deep:
        def __init__(self) -> None:
            self.y = 0.0

        def step(self, a: bool, b: bool, c: bool, x: float) -> float:
            r = x
            if a:
                r = -r
            self.y = r
            if b:
                r = -r
                self.y = r
            if c:
                r = -r
                self.y = r
            return r

    result = holoso.synthesize(Deep().step, default_ops(FloatFormat(11, 52)), name="deep_dedup")
    assert [p.name for p in result.output_ports] == ["state_y"]
    elaborated = result.numerical_model.elaborate()
    for a in (True, False):
        for b in (True, False):
            for c in (True, False):
                assert float(elaborated.run(a, b, c, 3.0)[0]) == Deep().step(a, b, c, 3.0)


def test_emission_resets_come_from_the_analysis_snapshot_not_a_live_read() -> None:
    # MISCOMPILE: emission re-read state resets with a raw getattr, so a permitted compile-time evaluation that
    # mutated the component after analysis stabilized moved the RTL reset value (first transaction 11.0 where
    # Python gives 3.0). Emission consumes the analyzer's snapshot now.
    module = types.ModuleType("lazy_poke")
    holder: dict[str, object] = {}

    def module_getattr(name: str) -> float:
        if name != "trigger":
            raise AttributeError(name)
        holder["active"].acc = 9.0  # type: ignore[attr-defined]
        return 0.0

    module.__getattr__ = module_getattr  # type: ignore[method-assign]

    class Accumulator:
        def __init__(self) -> None:
            self.acc = 1.0

        def step(self, x: float) -> float:
            before = self.acc
            ignored = module.trigger
            self.acc = before + x + ignored
            return self.acc

    instance = Accumulator()
    holder["active"] = instance
    Accumulator.step.__globals__["module"] = module
    try:
        model = holoso.synthesize(instance.step, default_ops(FloatFormat(11, 52)), name="snap_reset").numerical_model
        assert float(model.elaborate().run(2.0)[0]) == 3.0
    finally:
        Accumulator.step.__globals__.pop("module", None)


def test_inadmissible_state_reset_is_located_at_the_store() -> None:
    # Regression (E3): the reset/join rejection carried a synthetic 0:0 origin; it must name the store site.
    class BadReset:
        def __init__(self) -> None:
            self.h: object = object()

        def step(self, x: float) -> float:
            self.h = x
            return x

    with pytest.raises(UnsupportedConstruct) as excinfo:
        lower(BadReset().step)
    assert re.search(r"step:[1-9]\d*:", str(excinfo.value)), str(excinfo.value)


def test_an_aliased_slot_descriptor_refuses_instead_of_forking_state() -> None:
    # S2.9 review (pre-existing miscompile): an alias to ANOTHER slot's member descriptor passed the blanket
    # slots exemption, so the alias and its target lowered as independent state where Python shares one storage
    # location (step(3.0) computed 1.0 against Python's 3.0).
    class Base:
        __slots__ = ("value",)

        def __init__(self) -> None:
            self.value = 1.0

    class Aliased(Base):
        alias = Base.value  # type: ignore[misc]

        def step(self, x: float) -> float:
            self.alias = x
            return self.value

    assert Aliased().step(3.0) == 3.0  # runnable Python: one storage location
    with pytest.raises(UnsupportedConstruct, match="descriptors are not supported|descriptor attribute"):
        lower(Aliased().step)


def test_unrolled_component_fanout_keeps_source_state_port_order() -> None:
    # S2.11 review: every trip of a statically unrolled loop shares the storing op's whole origin chain, so the
    # first-store keys tie exactly; the tie then fell back to block-id order, which the unroller hands out in
    # reverse trip order, reversing the state ports against the source text of the iterable.
    class Channel:
        def __init__(self) -> None:
            self.value = 0.0

    class Bank:
        def __init__(self) -> None:
            self.first = Channel()
            self.second = Channel()

        def update_all(self, x: float) -> None:
            for channel in (self.first, self.second):
                channel.value = x

        def step(self, x: float) -> float:
            self.update_all(x)
            return x

    result = holoso.synthesize(Bank().step, default_ops(FloatFormat(6, 18)), name="fanout_order")
    assert [p.name for p in result.output_ports] == ["state_first__value", "state_second__value"]


def test_nan_state_reset_is_a_located_rejection() -> None:
    # S2.11 review: a NaN reset snapshot used to surface as the HIR constant domain's raw UnsupportedConstruct
    # with no location and no rendered prefix; it must refuse at emission, located at the leaf's first store.
    class NanReset:
        def __init__(self) -> None:
            self.gain = float("nan")

        def step(self, x: float) -> float:
            self.gain = self.gain * 0.5
            return x * self.gain

    with pytest.raises(UnsupportedConstruct, match=r"step:\d+:\d+: .*NaN constant") as excinfo:
        lower(NanReset().step)
    location = excinfo.value.location
    assert location is not None and location.filename == __file__
    assert location.line is not None and "self.gain = self.gain * 0.5" in location.line
