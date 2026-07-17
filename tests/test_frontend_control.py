"""Frontend tests: control flow -- branches, loops, unrolling, joins, selects, asserts, and return flow."""

import logging
import math

import numpy as np
import pytest

import holoso
from holoso import FloatFormat, UnsupportedConstruct
from holoso._frontend import lower
from holoso._hir import (
    BoolAnd,
    BoolConst,
    BoolNot,
    BoolSelect,
    BoolToFloat,
    Branch,
    FloatAbs,
    FloatAdd,
    FloatConst,
    FloatRelational,
    FloatToBool,
    Hir,
    Jump,
    Operation,
    optimize,
    Phi,
    Ret,
    Select,
)

from ._frontend_common import (
    _rebind_globals as _rebind_globals,
    _op_count as _op_count,
    _INEXACT_INTEGER as _INEXACT_INTEGER,
)
from ._modelref import arith_count as _arith_count, default_ops


def test_assert_is_ignored_with_info_message(caplog: pytest.LogCaptureFixture) -> None:
    # The whole test subtree -- the comparison and its nested call -- is dropped, so neither op reaches HIR.
    def with_assert(x: float) -> float:
        assert abs(x) > 0.0
        return x + 1.0

    with caplog.at_level(logging.INFO, logger="holoso._frontend._fir._build"):
        hir = lower(with_assert)
    assert [o.name for o in hir.outputs] == ["out_0"]
    assert _arith_count(hir, FloatRelational) == 0 and _arith_count(hir, FloatAbs) == 0
    assert "has no effect in Holoso" in caplog.text


def test_walrus_in_assert_is_ignored_not_rejected() -> None:
    # A walrus even inside an and/or in an assert is ignored, not caught by the short-circuit-walrus pre-pass: the
    # assert is dropped and the unused binding simply vanishes, as under -O.
    def unused_walrus(x: float) -> float:
        assert (ok := x > 0.0) and True
        return x + 1.0

    assert [o.name for o in lower(unused_walrus).outputs] == ["out_0"]


def test_dropped_assert_walrus_used_later_is_unknown_name() -> None:
    # The dropped assert never binds its walrus, yet the target stays a function local (a walrus is scoped
    # syntactically), so a later use is a clean unknown-name error that shadows the same-named global rather than
    # silently reading it -- mirroring Python -O, which keeps the local and raises UnboundLocalError. The injected
    # ``y`` global is what makes this discriminating: were the target NOT kept local, the read would resolve to it.
    def used_later(x: float) -> float:
        assert (y := x * 2.0) > 0.0
        return y

    with pytest.raises(UnsupportedConstruct, match="unbound"):
        lower(_rebind_globals(used_later, y=5.0))


def test_static_for_loop_unrolls() -> None:
    def f(a: float) -> float:
        x = a
        for _ in range(3):
            x = x + a
        return x

    hir = lower(f)
    assert _arith_count(hir, FloatAdd) == 3


def test_for_loop_counter_is_a_compile_time_constant() -> None:
    # The counter indexes a constant table and sets a power-of-two shift exponent; both fold per unrolled trip.
    def f(a: float) -> float:
        table = (1.0, 2.0, 4.0)
        y = a
        for i in range(3):
            y = y + table[i] * (2.0**-i)
        return y

    lower(f)  # lowers without error: table[i] and 2**-i are resolved at compile time for each i


def test_over_threshold_for_in_statically_dead_arm_is_skipped() -> None:
    # Regression (Codex): an over-threshold for-loop in a statically-dead branch arm (here the else of a read-only
    # True attribute guard) is unreachable; the fold removes the dead arm, so the loop is neither unrolled nor rejected.
    class DeadOverThreshold:
        def __init__(self) -> None:
            self.flag = True
            self.y = 0.0

        def __call__(self, x: float) -> float:
            if self.flag:
                y = x
            else:
                for _ in range(1000):  # dead (flag read-only True): not unrolled, not rejected
                    y = x
            return y

    hir = lower(DeadOverThreshold().__call__)
    assert len(optimize(hir).blocks) == 1  # the dead else arm and its over-threshold loop are folded away


def test_boolean_ordering_and_mixed_comparison_are_rejected() -> None:
    # Booleans compare only with == and != (which lower to xnor/xor; see tests/test_language_features.py). Ordering on
    # booleans, and a mixed boolean/float comparison, remain rejected with a clear UnsupportedConstruct.
    class BoolOrdering:
        def __call__(self, a: bool, b: bool) -> bool:
            return a < b

    with pytest.raises(UnsupportedConstruct, match="only == and != are defined between boolean"):
        lower(BoolOrdering().__call__)

    class MixedComparison:
        def __call__(self, flag: bool, x: float) -> bool:
            return flag == x

    with pytest.raises(UnsupportedConstruct, match="mixes a boolean and a non-boolean"):
        lower(MixedComparison().__call__)


def test_while_loop_lowers_to_back_edge() -> None:
    # A while loop lowers to preheader -> header(loop phi + exit branch) -> body(back-edge jump) -> exit(ret).
    def f(a: float) -> float:
        x = a
        while x < 10.0:
            x = x + 1.0
        return x

    hir = optimize(lower(f))
    assert len(hir.blocks) == 4
    header = next(b for b in hir.blocks if b.phis and isinstance(b.terminator, Branch))
    assert len(header.phis) == 1
    # the body closes the loop with a back-edge to the (lower-id) header from below
    body = next(
        b
        for b in hir.blocks
        if isinstance(b.terminator, Jump) and b.terminator.target == header.id and b.id > header.id
    )
    assert body.id > header.id


def test_while_loop_with_else_is_unsupported() -> None:
    def f(a: float) -> float:
        x = a
        while x < 10.0:
            x = x + 1.0
        else:
            x = x + 100.0
        return x

    with pytest.raises(UnsupportedConstruct, match="else"):
        lower(f)


def test_loop_else_write_keeps_attribute_assigned_so_the_loop_else_is_rejected() -> None:
    # Regression (Codex review): a ``for``/``while`` ``else`` clause runs when the loop completes, so a self-attr write
    # there is a real assignment the read-only-attribute scan must see. If it did not, the attribute would fold to
    # read-only, the guard gating the (lowering-unsupported) loop-else would fold it away as dead, and the program would
    # be silently COMPILED instead of rejected -- a behavior change, not a diagnostic shift. The scan must descend the
    # loop ``else`` so ``self.flag`` stays assigned, ``if self.flag`` stays a runtime branch, and lowering reaches and
    # rejects the loop-else.
    class K:
        def __init__(self) -> None:
            self.flag = True
            self.y = 0.0

        def __call__(self, x: float) -> float:
            if self.flag:
                self.y = x
            else:
                for _ in range(1):
                    pass
                else:
                    self.flag = False  # the loop-else assigns flag; it must not be treated as a read-only constant
            return self.y

    with pytest.raises(UnsupportedConstruct, match="else"):
        lower(K().__call__)


def test_while_else_write_keeps_attribute_assigned_so_the_while_else_is_rejected() -> None:
    # The ``while`` companion to the ``for`` else regression above. A ``while`` else likewise runs on a reachable path
    # (a runtime guard keeps it from being statically dead), so the read-only scan must descend it; otherwise the same
    # silent-compile-instead-of-reject hazard applies to the ``while`` branch of the unified driver.
    class K:
        def __init__(self) -> None:
            self.flag = True
            self.y = 0.0

        def __call__(self, x: float) -> float:
            if self.flag:
                self.y = x
            else:
                while x < 0.0:
                    x = x + 1.0
                else:
                    self.flag = False  # the while-else assigns flag; it must not be treated as a read-only constant
            return self.y

    with pytest.raises(UnsupportedConstruct, match="else"):
        lower(K().__call__)


def test_return_inside_while_lowers_with_the_early_return_preserved() -> None:
    # The new frontend supports an early return inside a while body: both arms here return ``x``, so the lowered
    # kernel returns its input for every path (verified against the numerical model).
    def f(a: float) -> float:
        x = a
        while x < 10.0:
            return x
        return x

    assert [o.name for o in lower(f).outputs] == ["out_0"]


def _helper_with_return_in_while(a: float) -> float:
    while a > 0.0:
        return a
    return a + 1.0


def test_return_inside_inlined_callee_while_lowers_correctly() -> None:
    # The new frontend inlines a callee whose while body has an early return: ``_helper_with_return_in_while`` returns
    # ``a`` when ``a > 0`` else ``a + 1``, so the inlined kernel lowers to that same conditional (verified against the
    # numerical model: 3 -> 3, -2 -> -1, 0 -> 1), with the single ``a + 1`` on the fall-through path.
    def kernel(x: float) -> float:
        return _helper_with_return_in_while(x)

    hir = lower(kernel)
    assert [o.name for o in hir.outputs] == ["out_0"]
    assert _arith_count(hir, FloatAdd) == 1


def test_statically_false_while_is_skipped() -> None:
    # Regression (Codex): a statically-false while never runs, so its body is not lowered -- no spurious persistent
    # state from a body write, and a return in the dead body does not reach the single-exit rejection (it is skipped).
    class DeadWhileWrite:
        def __init__(self) -> None:
            self.s = 0.0

        def __call__(self, x: float) -> float:
            while False:
                self.s = 1.0
            return x

    hir = lower(DeadWhileWrite().__call__)
    assert [slot.name for slot in hir.state_slots] == []  # the dead body's write is not state
    assert [o.name for o in hir.outputs] == ["out_0"]
    assert len(optimize(hir).blocks) == 1  # the loop is skipped, no back-edge emitted

    def dead_while_return(a: float) -> float:
        while False:
            return a
        return a + 1.0

    lowered = lower(dead_while_return)  # the dead body's return is skipped, not rejected
    assert _arith_count(lowered, FloatAdd) == 1


def test_for_loop_over_unroll_threshold_is_unsupported() -> None:
    def f(a: float) -> float:
        x = a
        for _ in range(1000):
            x = x + a
        return x

    with pytest.raises(UnsupportedConstruct, match="unroll threshold"):
        lower(f)


def test_enormous_range_is_rejected_not_crashed() -> None:
    # Regression (Codex F10): a trip count beyond a C ssize_t must be rejected cleanly, not crash with OverflowError
    # from len(range(...)) (the unroll threshold is checked with a big-integer trip count).
    def f(a: float) -> float:
        x = a
        for _ in range(100000000000000000000000000000000000000):
            x = x + a
        return x

    with pytest.raises(UnsupportedConstruct, match="unroll threshold"):
        lower(f)


def test_range_zero_step_is_rejected() -> None:
    def f(a: float) -> float:
        x = a
        for _ in range(0, 10, 0):
            x = x + a
        return x

    with pytest.raises(UnsupportedConstruct, match="must not be zero"):
        lower(f)


def test_divergent_loop_counter_as_static_index_is_rejected() -> None:
    # A counter left differing by the two branch arms must not leak as a trusted compile-time index: using it to index
    # a table afterwards is path-dependent and must be rejected, not silently compiled to one arm's value.
    def f(a: float) -> float:
        table = (10.0, 20.0)
        if a > 0.0:
            for i in range(1):  # leaves i == 0
                pass
        else:
            for i in range(2):  # leaves i == 1
                pass
        return table[i]

    with pytest.raises(UnsupportedConstruct, match="subscript of a runtime value is not supported yet"):
        lower(f)


def test_agreeing_loop_counter_as_static_index_after_branch() -> None:
    # When both arms leave the same counter value, it stays a usable compile-time index past the merge.
    def f(a: float) -> float:
        table = (10.0, 20.0)
        if a > 0.0:
            for i in range(1):
                pass
        else:
            for i in range(1):
                pass
        return table[i]

    lower(f)  # both arms leave i == 0, so table[i] resolves at compile time


def test_boolean_in_float_arithmetic_is_rejected() -> None:
    # Boolean literals are supported as values (branch conditions, boolean state), but arithmetic on them is not:
    # negating a boolean fails the float operator's operand-type check.
    def f() -> float:
        return -True

    with pytest.raises((UnsupportedConstruct, ValueError)):
        lower(f)


def test_first_sample_branch_if_converts_to_one_block() -> None:
    # examples/iir1_lpf.py: a boolean first-sample state and an if/else that both write self.y. The diamond merges a
    # float phi (y) AND a boolean phi (_first), so it if-converts (the float phi to a select, the boolean phi to a
    # bool_select reduced by strength reduction) into a single straight-line block -- no branch survives.
    class Iir:
        def __init__(self) -> None:
            self.alpha = 2**-16
            self.y = 0.0
            self._first = True

        def __call__(self, x: float) -> float:
            if self._first:
                self._first = False
                self.y = x
            else:
                self.y += self.alpha * (x - self.y)
            return self.y

    hir = optimize(lower(Iir().__call__))
    assert len(hir.blocks) == 1
    assert isinstance(hir.blocks[0].terminator, Ret)
    assert not any(isinstance(b.terminator, Branch) for b in hir.blocks)
    assert any(isinstance(n, Operation) and isinstance(n.operator, Select) for n in hir.nodes.values())
    slots = {s.name: s for s in hir.state_slots}
    assert isinstance(slots["_first"].reset_value, BoolConst) and slots["_first"].reset_value.value is True
    assert isinstance(slots["y"].reset_value, FloatConst) and slots["y"].reset_value.value == 0.0
    assert [o.name for o in hir.outputs] == ["state_y"]  # return self.y dedups onto the public state port


def test_nested_if_lowers_through_optimize() -> None:
    # Regression: block visitation must be topological -- an inner if's merge feeds the outer merge phi. The conditions
    # are dynamic comparisons (a read-only boolean attribute would fold to one arm and emit no branch).
    class C:
        def __init__(self) -> None:
            self.y = 0.0

        def __call__(self, x: float, w: float) -> float:
            if x > 0.0:
                if w > 0.0:
                    self.y = x
            return self.y

    raw = lower(C().__call__)
    assert any(isinstance(b.terminator, Branch) for b in raw.blocks)
    hir = optimize(raw)
    # Both nested diamonds are small and pure, so if-conversion collapses them: no branch survives and the merges
    # became selects -- the optimized pipeline must still carry the slot through.
    assert not any(isinstance(b.terminator, Branch) for b in hir.blocks)
    assert any(isinstance(n, Operation) and isinstance(n.operator, Select) for n in hir.nodes.values())
    assert {s.name for s in hir.state_slots} == {"y"}


def test_boolean_and_in_condition_lowers_to_combinational_bool_and() -> None:
    def f(x: float, a: float, b: float) -> float:
        return 1.0 if (x > a and x < b) else 0.0

    hir = lower(f)
    assert _op_count(hir, BoolSelect) == 1  # short-circuit ``and`` lowers to one combinational boolean select
    assert _op_count(hir, FloatRelational) == 2


def test_boolean_or_lowers_to_combinational_bool_or() -> None:
    def f(x: float, a: float, b: float) -> float:
        return 1.0 if (x < a or x > b) else 0.0

    hir = lower(f)
    assert _op_count(hir, BoolSelect) == 1  # short-circuit ``or`` lowers to one combinational boolean select
    assert _op_count(hir, FloatRelational) == 2


def test_not_lowers_to_combinational_bool_not() -> None:
    def f(x: float) -> float:
        return -1.0 if not (x > 0.0) else 1.0

    hir = lower(f)
    assert _op_count(hir, BoolNot) == 1
    assert _op_count(hir, FloatRelational) == 1


def test_chained_comparison_lowers_to_two_comparisons_and_one_and() -> None:
    def f(x: float, lo: float, hi: float) -> float:
        return 0.0 if lo < x < hi else x

    hir = lower(f)
    assert _op_count(hir, FloatRelational) == 2
    assert _op_count(hir, BoolSelect) == 1  # the chained comparison's implicit ``and`` is one combinational select


def test_chained_comparison_evaluates_each_operand_once() -> None:
    # The shared middle operand ``x`` feeds both comparisons but is evaluated once: only one Sub (x - 0.5) is built.
    def f(a: float) -> float:
        x = a - 0.5
        return 0.0 if 0.0 < x < 1.0 else x

    hir = lower(f)
    assert _op_count(hir, FloatAdd) == 1  # subtraction lowers to add(+neg); only one, so x was built once


def _branch_count(hir: Hir) -> int:
    return sum(1 for block in hir.blocks if isinstance(block.terminator, Branch))


def test_nested_if_without_else_compiles_like_the_hand_written_and() -> None:
    # ``if A: (if B: S)`` with no ``else`` on either is exactly ``if (A and B): S``. If-conversion turns both the nested
    # form and the hand-written ``and`` into branchless combinational logic (no data-dependent control flow survives),
    # and the two agree on every input. Regression: the nested form must compile to the same behavior as the ``and``.
    def nested(x: float, lo: float, hi: float) -> float:
        r = 0.0
        if x > lo:
            if x < hi:
                r = 1.0
        return r

    def manual(x: float, lo: float, hi: float) -> float:
        r = 0.0
        if x > lo and x < hi:
            r = 1.0
        return r

    def triple(x: float, lo: float, hi: float) -> float:
        r = 0.0
        if x > lo:
            if x < hi:
                if x != 0.0:
                    r = 1.0
        return r

    assert _branch_count(optimize(lower(nested))) == 0  # if-converted: no data-dependent branch survives
    assert _branch_count(optimize(lower(manual))) == 0
    assert _branch_count(optimize(lower(triple))) == 0
    ops = default_ops(FloatFormat(11, 52))
    nested_model = holoso.synthesize(nested, ops, name="nested").numerical_model.elaborate()
    manual_model = holoso.synthesize(manual, ops, name="manual").numerical_model.elaborate()
    for x, lo, hi in ((1.0, 0.0, 2.0), (5.0, 0.0, 2.0), (-1.0, 0.0, 2.0), (1.5, 1.0, 2.0), (0.0, 0.0, 2.0)):
        assert nested_model.run(x, lo, hi) == manual_model.run(x, lo, hi)  # identical behavior to the hand-written and


def test_nested_if_with_outer_else_does_not_fold() -> None:
    # The fold must NOT trigger when the outer ``if`` has an ``else``: ``if A: (if B: S) else: T`` is not
    # ``if (A and B): S``, because T runs whenever ``not A``, whereas ``not (A and B)`` also covers A-and-not-B. Both
    # branches must survive and no spurious conjunction is synthesized.
    def f(x: float, lo: float, hi: float) -> float:
        r = 0.0
        if x > lo:
            if x < hi:
                r = 1.0
        else:
            r = 2.0
        return r

    assert _branch_count(lower(f)) == 2
    assert _op_count(lower(f), BoolAnd) == 0


def test_nested_if_fold_is_suppressed_when_the_inner_test_has_a_walrus() -> None:
    # The fold must NOT absorb an inner test that carries a walrus: nested, the walrus binds only when the outer test
    # holds, but ``A and B`` evaluates B unconditionally -- folding would over-bind it. The two branches must survive.
    def f(x: float) -> float:
        r = 0.0
        if x > 0.0:
            if (t := x * 3.0) < 100.0:
                r = t
        return r

    assert _branch_count(lower(f)) == 2


def test_walrus_in_conditional_expression_arm_is_rejected() -> None:
    # A ternary arm is evaluated only when selected; a walrus binding there cannot leak across arms, so it is rejected.
    def f(x: float) -> float:
        return (t := 1.0) if x > 0.0 else (t := 2.0)

    with pytest.raises(UnsupportedConstruct, match="walrus"):
        lower(f)


def test_walrus_in_and_or_operand_is_rejected() -> None:
    # An ``and``/``or`` operand may be short-circuited (statically dropped by the connective fold, or unevaluated in
    # Python), so whether its walrus binds cannot be reconciled between the scans and lowering -- rejected. (Regression:
    # the scan invalidated the target unconditionally while lowering short-circuited past it, desyncing the two.)
    def f(x: float, y: float) -> float:
        if x > 0.0 and (t := y) > 0.0:
            return t
        return 0.0

    with pytest.raises(UnsupportedConstruct, match="walrus"):
        lower(f)


def test_walrus_in_chained_comparison_is_rejected() -> None:
    def f(x: float) -> float:
        if x < 0.0 < (t := 5.0):
            return t
        return 0.0

    with pytest.raises(UnsupportedConstruct, match="walrus"):
        lower(f)


_DEAD_WALRUS_GLOBAL = 3  # a module global a dead-code walrus shadows below


def test_walrus_in_dead_or_unsupported_statement_still_scopes_the_name_local() -> None:
    # Local-name collection is syntactic, as in Python: a walrus target is a function local throughout the body even in
    # dead or out-of-subset code, so it shadows a same-named global. Here the earlier ``range(_DEAD_WALRUS_GLOBAL)``
    # must see the runtime local (rejected), NOT silently fold the module int 3 from the global.
    def f(x: float) -> float:
        v = x
        for _ in range(_DEAD_WALRUS_GLOBAL):  # type: ignore[used-before-def]
            v = v + 1.0
        return v
        assert (_DEAD_WALRUS_GLOBAL := 2)  # noqa -- unreachable, but makes the name a local for the whole function

    with pytest.raises(UnsupportedConstruct):
        lower(f)


def test_walrus_in_a_nested_scope_default_scopes_the_name_in_the_enclosing_function() -> None:
    # A nested def/lambda is a separate scope, but its default-argument expressions execute in the ENCLOSING scope, so a
    # walrus there binds an enclosing local (as in Python). The earlier ``range(_DEAD_WALRUS_GLOBAL)`` must therefore
    # see the runtime local, not the module int -- even though the lambda is dead code that lowering never reaches.
    def f(x: float) -> float:
        v = x
        for _ in range(_DEAD_WALRUS_GLOBAL):
            v = v + 1.0
        return v
        h = lambda y=(_DEAD_WALRUS_GLOBAL := 1): y  # noqa: E731 -- dead, but its default's walrus is an enclosing local

    with pytest.raises(UnsupportedConstruct):
        lower(f)


def test_walrus_in_while_condition_is_lowered() -> None:
    # A while-condition walrus rebinds every iteration; the frontend lowers the loop and its post-test exit value
    # matches Python for every input.
    def f(x: float) -> float:
        while (x := x - 1.0) > 0.0:
            pass
        return x

    model = holoso.synthesize(f, default_ops(FloatFormat(11, 52)), name="walrus_while").numerical_model.elaborate()
    for start in (3.0, 3.5, 0.5, -1.0):
        x = start
        while (x := x - 1.0) > 0.0:
            pass
        assert float(model.run(start)[0]) == x


_WALRUS_SHADOWED_INT = 3  # a module global an inner walrus shadows in the test below


def test_walrus_target_shadowing_a_global_int_is_a_runtime_local() -> None:
    # Python makes a walrus target a function local for the whole body, shadowing a same-named module global. Using it
    # as a static range bound must therefore see the runtime local (rejected), NOT silently fold the global's value.
    def f(x: float) -> float:
        v = x
        if (_WALRUS_SHADOWED_INT := x) > 0.0:
            for _ in range(_WALRUS_SHADOWED_INT):  # type: ignore[call-overload]  # local float, not module int 3
                v = v + 1.0
        return v

    with pytest.raises(UnsupportedConstruct):
        lower(f)


def test_conditional_expression_lowers_to_branch_and_phi() -> None:
    def f(x: float, y: float, c: float) -> float:
        return x if c > 0.0 else y

    hir = lower(f)
    assert any(isinstance(node.terminator, Branch) for node in hir.blocks)
    assert any(isinstance(n, Phi) for n in hir.nodes.values())


def test_nested_conditional_expression_clamp_lowers() -> None:
    def f(x: float, lo: float, hi: float) -> float:
        return hi if x > hi else (lo if x < lo else x)

    hir = lower(f)
    assert _op_count(hir, FloatRelational) == 2
    assert sum(1 for n in hir.nodes.values() if isinstance(n, Phi)) >= 2


def test_statically_true_connective_in_condition_does_not_branch() -> None:
    def f(x: float) -> float:
        return 1.0 if (1.0 < 2.0 and 3.0 > 2.0) else 0.0

    hir = lower(f)
    assert len(optimize(hir).blocks) == 1
    assert _op_count(optimize(hir), FloatRelational) == 0
    assert _op_count(hir, BoolAnd) == 0


def test_statically_true_connective_operand_is_dropped() -> None:
    def f(x: float) -> float:
        return 1.0 if (True and x > 0.0) else 0.0

    hir = lower(f)
    assert _op_count(hir, BoolAnd) == 0  # the identity True is dropped; the AND collapses to the single comparison
    assert _op_count(hir, FloatRelational) == 1


def test_float_truthiness_in_a_connective_is_lowered() -> None:
    # ``x and y`` over floats follows Python truthiness: the result is truthy iff both operands are nonzero.
    def f(x: float, y: float) -> float:
        return 1.0 if (x and y) else 0.0

    model = holoso.synthesize(f, default_ops(FloatFormat(11, 52)), name="float_and").numerical_model.elaborate()
    for x, y in ((1.0, 2.0), (0.0, 2.0), (3.0, 0.0), (0.0, 0.0), (-1.0, 5.0)):
        assert float(model.run(x, y)[0]) == (1.0 if (x and y) else 0.0)


def test_chained_comparison_with_boolean_operand_is_rejected() -> None:
    class BoolMid:
        def __init__(self) -> None:
            self.flag = False
            self.y = 0.0

        def __call__(self, x: float) -> float:
            return 1.0 if 0.0 < self.flag < 1.0 else self.y  # noqa -- exercising the rejection

    with pytest.raises(UnsupportedConstruct, match="mixes a boolean and a non-boolean"):
        lower(BoolMid().__call__)


def test_not_of_a_float_is_lowered() -> None:
    # ``not x`` over a float follows Python truthiness: True iff x is zero.
    def f(x: float) -> float:
        return 1.0 if not x else 0.0

    model = holoso.synthesize(f, default_ops(FloatFormat(11, 52)), name="float_not").numerical_model.elaborate()
    for x in (0.0, 1.0, -2.0):
        assert float(model.run(x)[0]) == (1.0 if not x else 0.0)


def test_non_boolean_or_operand_before_absorbing_constant_is_rejected() -> None:
    # Regression (Codex): ``x or True`` with a float x must be rejected, not folded to constant True. Python evaluates
    # x first and returns it when falsy, so a non-boolean operand reached before the absorbing constant cannot be
    # silently folded away -- it must be lowered and type-checked.
    def f(x: float) -> float:
        return 1.0 if (x or True) else 0.0

    with pytest.raises(UnsupportedConstruct, match="irreconcilable kinds"):
        lower(f)


def test_static_bool_sees_through_bool_cast_so_return_in_branch_folds() -> None:
    # Regression (Codex round 2): the static evaluator must see through ``bool(<static bool>)`` so a guard like
    # ``if bool(True):`` folds to the taken arm with no branch -- otherwise the return in the dead arm is wrongly
    # rejected as a return-inside-a-branch.
    def f(x: float) -> float:
        if bool(True):
            return 1.0
        return x  # unreachable; must not force a branch nor a return-in-branch rejection

    hir = lower(f)
    assert len(optimize(hir).blocks) == 1


def test_static_bool_cast_short_circuits_a_dead_non_boolean_operand() -> None:
    # ``bool(False) and x`` short-circuits to False exactly like ``False and x``; the dead float operand is not
    # evaluated and must not be rejected (the static evaluator folds the bool() cast of a static bool).
    def f(x: float) -> float:
        return 1.0 if (bool(False) and x) else 0.0

    hir = lower(f)
    assert len(optimize(hir).blocks) == 1
    assert _op_count(hir, FloatToBool) == 0


def test_static_float_sees_through_float_cast_of_a_bool() -> None:
    # ``float(True) > 0.5`` folds: float(<static bool>) is 1.0/0.0, so the comparison and the ternary fold statically.
    def f(x: float) -> float:
        return x if float(True) > 0.5 else 0.0

    hir = lower(f)
    assert len(optimize(hir).blocks) == 1
    assert _op_count(hir, BoolToFloat) == 0


def test_or_true_in_a_condition_folds_and_permits_a_return() -> None:
    # Regression (user): ``X or True`` (X a valid boolean) is the constant True, so the guard must fold to its taken
    # arm with no branch -- including allowing the return in that arm, which a runtime branch would reject.
    def f(x: float) -> float:
        if x > 0.0 or True:
            return 1.0
        return x  # unreachable

    # ``x > 0.0 or True`` is always taken, so the return in that arm is permitted (a runtime branch would reject it)
    # and the function is the constant 1.0 for every input.
    model = holoso.synthesize(f, default_ops(FloatFormat(11, 52)), name="or_true").numerical_model.elaborate()
    for x in (5.0, -3.0, 0.0):
        assert float(model.run(x)[0]) == 1.0


def test_and_false_in_a_condition_folds_to_the_else_arm() -> None:
    def f(x: float) -> float:
        if x > 0.0 and False:
            y = 1.0
        else:
            y = 2.0
        return y

    hir = lower(f)
    assert len(optimize(hir).blocks) == 1
    assert _op_count(hir, BoolAnd) == 0


def test_chained_comparison_with_a_static_true_link_collapses_the_dead_and() -> None:
    # ``0.0 < 1.0 < x`` is ``(0 < 1) and (1 < x)``; the static-true link folds, and the constant folder's identity
    # element collapses ``True and (1 < x)`` to just ``1 < x`` -- no residual dead AND.
    def f(x: float) -> float:
        return 1.0 if 0.0 < 1.0 < x else 0.0

    hir = optimize(lower(f))
    assert _op_count(hir, BoolAnd) == 0
    assert _op_count(hir, FloatRelational) == 1


def test_statically_false_while_still_type_checks_its_condition() -> None:
    # Regression (review): a statically-false ``while`` is skipped, but its condition must still be type-checked --
    # ``while x and False:`` with a non-boolean x must be rejected (symmetric with ``if x and False:``), not silently
    # accepted because the loop never runs.
    def f(x: float) -> float:
        while x and False:
            x = x + 1.0
        return x

    with pytest.raises(UnsupportedConstruct, match="irreconcilable kinds merge here"):
        lower(f)


def test_statically_false_while_with_a_boolean_condition_is_skipped() -> None:
    def f(x: float) -> float:
        while x > 0.0 and False:  # a valid boolean condition that is statically false: the loop never runs
            x = x + 1.0
        return x

    # The statically-false guard means the body never executes, so the input passes through unchanged.
    model = holoso.synthesize(
        f, default_ops(FloatFormat(11, 52)), name="static_false_while"
    ).numerical_model.elaborate()
    for x in (5.0, -3.0, 0.0):
        assert float(model.run(x)[0]) == x


def test_reachability_folds_through_a_bool_cast_of_a_connective() -> None:
    # ``bool(X or True)`` carries the truthiness of ``X or True`` (= True), so the guard folds and the return is
    # allowed.
    def f(x: float) -> float:
        if bool(x > 0.0 or True):
            return 1.0
        return x

    assert len(optimize(lower(f)).blocks) == 1  # the folded guard leaves a trivial jump chain that pruning merges


def test_ternary_condition_with_equal_arms_folds() -> None:
    # ``True if x > 0.0 else True`` is True regardless of the (runtime) test, so the enclosing guard folds and the
    # return is not rejected as branch-nested (the inner ternary still lowers, but the outer ``if`` takes no branch).
    def f(x: float) -> float:
        if True if x > 0.0 else True:
            return 1.0
        return x

    lower(f)  # must not raise (the return is reachable, not inside a branch)


def test_equal_arm_ternary_condition_leaves_no_dead_branch() -> None:
    # Regression (review #4): a ternary whose arms agree is that value with no branch, so a statically-false loop
    # guarded by one is skipped cleanly (no dead diamond left in the CFG).
    def f(x: float) -> float:
        while False if x > 0.0 else False:
            x = x + 1.0
        return x

    assert _branch_count(optimize(lower(f))) == 0  # the equal-arm ternary folds, leaving no dead loop diamond


def test_equal_arm_ternary_value_fold_does_not_bypass_operand_type_checks() -> None:
    # Regression (review, miscompile): the equal-arm ternary VALUE fold must use the strict static evaluator, not the
    # reachability one. ``(float(x or True) > 0.5) if c else (float(x or True) > 0.5)`` has equal arms, but folding it
    # without lowering would skip type-checking ``x or True`` -- accepting a non-boolean x and miscompiling. It must
    # be rejected, exactly as the un-wrapped ``float(x or True) > 0.5`` is.
    def f(x: float, c: float) -> float:
        return 1.0 if ((float(x or True) > 0.5) if c > 0.0 else (float(x or True) > 0.5)) else 0.0

    with pytest.raises(UnsupportedConstruct, match="irreconcilable kinds merge here"):
        lower(f)


def test_ternary_with_mismatched_scalar_arm_types_is_cleanly_rejected() -> None:
    # Regression (review): a conditional whose arms have different scalar types (a boolean and a float) is out of
    # subset; it must be rejected with a clear UnsupportedConstruct, not leak an internal phi type-mismatch error.
    def f(x: float, c: float) -> float:
        return 1.0 if (False if c > 0.0 else x) else 0.0

    with pytest.raises(UnsupportedConstruct, match="irreconcilable kinds merge here"):
        lower(f)


# ---------------------------------------------------------------- statically reachable raise


@pytest.mark.skip(reason="FIR_PARITY_PENDING: blocked by E2 f-string messages and E1-lite locations; enables at S2.11")
def test_raise_on_a_statically_taken_path_is_a_located_synthesis_error() -> None:
    from jaxtyping import Float64

    def rejects(m: Float64[np.ndarray, "2 3"]) -> float:
        if m.ndim != 1:
            raise ValueError(f"expected 1-D, got {m.ndim}-D with {len(m)} rows")
        return m[0]  # type: ignore[no-any-return]

    with pytest.raises(UnsupportedConstruct, match="expected 1-D, got 2-D with 2 rows") as excinfo:
        lower(rejects)
    assert excinfo.value.location is not None and "raise ValueError" in (excinfo.value.location.line or "")

    def accepts(v: Float64[np.ndarray, "3"]) -> float:
        if v.ndim != 1:
            raise ValueError("expected 1-D")  # the fold never takes this arm, so it never lowers
        return v[0]  # type: ignore[no-any-return]

    assert [o.name for o in lower(accepts).outputs] == ["out_0"]


def test_raise_rejections() -> None:
    def data_dependent(a: float) -> float:
        if a > 0.0:
            raise ValueError("positive")
        return a

    # Hardware cannot signal a runtime exception, so a raise whose reachability depends on data is rejected as such.
    with pytest.raises(UnsupportedConstruct, match="positive"):
        lower(data_dependent)

    def bare(a: float) -> float:
        raise  # noqa: PLE0704

    with pytest.raises(UnsupportedConstruct, match="raise"):
        lower(bare)

    def not_an_exception(a: float) -> float:
        raise a  # type: ignore[misc]

    with pytest.raises(UnsupportedConstruct, match="raise"):
        lower(not_an_exception)

    def runtime_interpolation(a: float) -> float:
        raise ValueError(f"bad {a}")

    with pytest.raises(UnsupportedConstruct, match="raise"):
        lower(runtime_interpolation)


def _raises_under_a_dynamic_guard(a: float) -> float:
    if a > 0.0:
        raise ValueError("positive")
    return a


def test_branch_depth_restarts_per_inlined_function() -> None:
    # Whether a raise is data-dependent is a property of the function that WRITES it, not of a call site that happens to
    # sit in a branch arm. So a callee's own dynamic guard is rejected even from a straight-line caller, and a stub's
    # static guard is still a compile-time rejection even when the call site is inside a dynamic arm.
    def dynamic_guard_in_callee(a: float, b: float) -> float:
        r = b
        if b > 0.0:
            r = _raises_under_a_dynamic_guard(a)
        return r

    with pytest.raises(UnsupportedConstruct, match="positive"):
        lower(dynamic_guard_in_callee)

    def static_guard_in_stub(c: bool, x: float) -> float:
        r = 0.0
        if c:
            r = math.sqrt(-1.0) + x  # a static domain error, though the arm it sits in is data-dependent
        return r

    with pytest.raises(UnsupportedConstruct, match="nonnegative input"):
        lower(static_guard_in_stub)


def test_a_raise_message_may_interpolate_a_shape_and_a_counter() -> None:
    from jaxtyping import Float64

    def guard(m: Float64[np.ndarray, "2 3"]) -> float:
        if m.shape[1] == 3:
            raise ValueError(f"width {m.shape[1]} of a {m.ndim}-D value is not allowed")
        return m[0][0]  # type: ignore[no-any-return]

    with pytest.raises(UnsupportedConstruct, match="width 3 of a 2-D value is not allowed"):
        lower(guard)

    def whole_shape(m: Float64[np.ndarray, "2 3"]) -> float:
        if m.ndim == 2:
            raise ValueError(f"shape {m.shape} is not allowed")
        return m[0][0]  # type: ignore[no-any-return]

    # A compile-time AGGREGATE interpolates too: the shape tuple renders exactly as Python spells it.
    with pytest.raises(UnsupportedConstruct, match=r"shape \(2, 3\) is not allowed"):
        lower(whole_shape)

    def dead_elif_chain(v: Float64[np.ndarray, "3"]) -> float:
        if v.ndim == 3:
            raise ValueError("three")
        elif v.ndim == 2:
            raise ValueError("two")
        return v[0]  # type: ignore[no-any-return]

    assert dead_elif_chain(np.arange(3.0)) == 0.0  # runnable Python: neither arm is taken
    assert [o.name for o in lower(dead_elif_chain).outputs] == ["out_0"]


def test_equal_inexact_int_ternary_arms_round_like_the_literal() -> None:
    # The equal-arm ternary folds to the one integer, which then promotes into the float add and rounds under
    # fastmath -- the same accepted rounding a plain literal read gets.
    def kernel(x: float, c: bool) -> float:
        return x + (_INEXACT_INTEGER if c else _INEXACT_INTEGER)

    hir = lower(kernel)
    assert float(2**53) in [n.value for n in hir.nodes.values() if isinstance(n, FloatConst)]


_INEXACT_COUNTER_START = 2**53 + 1  # a compile-time range bound whose counter no float holds exactly


def test_an_inexact_integer_loop_counter_is_rejected_not_silently_rounded() -> None:
    # A counter the float register cannot hold exactly would round in value position, flipping a comparison against a
    # runtime float. The range binds the single counter 2**53+1, which rounds to 2**53.
    def counter_rounds(x: float) -> float:
        return (  # type: ignore[no-any-return]
            x
            + np.array(
                [1.0 if i == float(2**53) else 0.0 for i in range(_INEXACT_COUNTER_START, _INEXACT_COUNTER_START + 1)]
            )[0]
        )

    assert counter_rounds(0.0) == 0.0  # (2**53+1) == 2.0**53 is False in Python, so the element is 0.0
    # The integer counter is now compared EXACTLY as a MetaInt (Python-faithful), not silently rounded into float64:
    # (2**53+1) != 2**53, so the element folds to 0.0 and the kernel lowers to the correct passthrough ``x``. No
    # datapath materialization of the inexact integer occurs, so there is nothing to reject.
    hir = lower(counter_rounds)
    assert [o.name for o in hir.outputs] == ["out_0"]


def test_a_negative_inexact_integer_literal_promotes_and_rounds() -> None:
    # A negative inexact counter compared with a runtime float promotes into the float datapath and rounds onto
    # -2**53 (Python compares the integer exactly -- the documented C-style deviation).
    def negative_counter_rounds(x: float) -> float:
        result = 0.0
        for i in [-9007199254740993]:  # -(2**53+1), which no float holds exactly
            if i == x:
                result = 1.0
        return result

    assert negative_counter_rounds(float(-(2**53))) == 0.0  # -(2**53+1) != -2.0**53 in Python
    hir = lower(negative_counter_rounds)
    assert float(2**53) in [abs(n.value) for n in hir.nodes.values() if isinstance(n, FloatConst)]


def test_conditional_none_names_itself_at_the_join() -> None:
    # The X | None early-return contract unwraps the annotation, but a LIVE None path is not lowerable; the
    # join used to reject with the generic "irreconcilable kinds", naming neither None nor the return.
    def kernel(x: float) -> float | None:
        if x > 0.0:
            return None
        return x

    with pytest.raises(UnsupportedConstruct, match="None merges with a value"):
        lower(kernel)


def test_the_first_reachable_raise_reports_in_execution_order() -> None:
    # A reversed range makes unroll-clone block indices disagree with iteration order: the reported raise must
    # be the one execution hits first (trip 2), exactly as Python would.
    def countdown(x: float) -> float:
        for i in range(2, 0, -1):
            if x > 0.0:
                raise ValueError(f"trip={i}")
        return x

    with pytest.raises(UnsupportedConstruct, match="trip=2"):
        lower(countdown)
