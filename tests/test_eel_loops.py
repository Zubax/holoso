"""
Loops and comprehensions: static unrolling (trip-by-trip, budget-bounded), the residual while with its
loop-carried scalar phis and break/continue exit lanes, and the permanent bans (dynamic trip counts,
aggregate carries, degenerate exit-on-every-path bodies). Positive shapes are oracle-verified against
CPython; every ban is pinned as a located rejection.
"""

from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pytest

from holoso._eel import lower
from holoso._errors import UnsupportedConstruct

from ._eel_corpus import convergence_steps
from ._eeloracle import assert_hir_matches_reference

type _Row = Mapping[str, float | bool | int]


def _oracle(fn: Callable[..., object], vectors: Sequence[_Row]) -> None:
    compared = assert_hir_matches_reference(lower(fn), fn, vectors, label=fn.__name__)
    assert compared == len(vectors)


def _rejects(fn: object, match: str) -> None:
    with pytest.raises(UnsupportedConstruct, match=match):
        lower(fn)


_X_ROWS: list[_Row] = [{"x": 2.0}, {"x": -1.5}, {"x": 0.0}, {"x": 7.25}]


# ---------------------------------------------------------------------- static unrolling


def _range_sum(x: float) -> float:
    acc = x
    for i in range(4):
        acc = acc + float(i)
    return acc


def _range_bounds(x: float) -> float:
    acc = x
    for i in range(1, 4):
        acc = acc * float(i)
    return acc


def _range_step(x: float) -> float:
    acc = x
    for i in range(5, 1, -2):
        acc = acc + float(i)
    return acc


def _sequence_iteration(x: float) -> float:
    acc = 0.0
    for w in (0.5, 0.25, x):
        acc = acc + w
    return acc


def _vector_iteration(x: float) -> float:
    acc = 0.0
    for v in np.array([x, 2.0, 3.0]):
        acc = acc + v
    return acc


def _matrix_row_iteration(x: float) -> float:
    acc = 0.0
    for row in np.array([[x, 1.0], [2.0, 3.0]]):
        acc = acc + row[0] * row[1]
    return acc


def _nested_static(x: float) -> float:
    acc = x
    for i in range(2):
        for j in range(3):
            acc = acc + float(i * 3 + j)
    return acc


def _target_survives(x: float) -> float:
    for w in (1.0, 2.0, x):
        pass
    return w


def _return_inside_static_loop(x: float) -> float:
    for w in (1.0, 2.0, 3.0):
        if w > 1.5:
            return x * w
        x = x + w
    return x


def _static_while(x: float) -> float:
    k = 3
    while k > 0:
        x = x * 2.0
        k = k - 1
    return x


def _static_while_walrus(x: float) -> float:
    k = 6
    while (half := k // 2) > 0:
        x = x + float(half)
        k = k - 2
    return x + float(half)


def _range_len_and_rebuild(x: float) -> float:
    span = range(3)
    ns = list(range(1, 3))
    return x * float(len(span)) + float(ns[0] + ns[1])


def _empty_range(x: float) -> float:
    acc = x
    for i in range(0):
        acc = acc + float(i)
    return acc


def test_static_loops_unroll_and_match_cpython() -> None:
    for fn in (
        _range_sum,
        _range_bounds,
        _range_step,
        _sequence_iteration,
        _vector_iteration,
        _matrix_row_iteration,
        _nested_static,
        _target_survives,
        _return_inside_static_loop,
        _static_while,
        _static_while_walrus,
        _range_len_and_rebuild,
        _empty_range,
    ):
        _oracle(fn, _X_ROWS)


def _zero_trip_target(x: float) -> float:
    for w in range(0):
        pass
    return x + float(w)


def _while_true(x: float) -> float:
    while True:
        x = x + 1.0
    return x


def test_a_zero_trip_target_is_unbound_after_the_loop() -> None:
    _rejects(_zero_trip_target, "the local name 'w' is not bound on every path")


def test_a_non_terminating_static_loop_exhausts_the_budget() -> None:
    _rejects(_while_true, "the graph expansion budget is exhausted while expanding the unrolled loop")


def _huge_static_range(x: float) -> float:
    for _ in range(10**25):
        x = x + 1.0
    return x


def test_a_huge_static_range_is_a_located_budget_rejection() -> None:
    _rejects(_huge_static_range, "budget is exhausted while expanding the range materialization")


# ---------------------------------------------------------------------- the residual while


def _countdown(x: float) -> float:
    while x > 0.0:
        x = x - 1.0
    return x


def _int_entry_promotes(x: float) -> float:
    scale = 1
    while x > 1.0:
        scale = scale / 2.0  # type: ignore[assignment]
        x = x - 1.0
    return float(scale)


def _float_entry_int_back(x: float) -> float:
    y = 1.5
    while x > 0.0:
        y = 1
        x = x - 1.0
    return y


def _header_walrus_zero_trip(x: float) -> float:
    while (last := x) > 0.0:
        x = x - 1.0
    return last


def _header_walrus_rebinds_a_carry(q: float) -> float:
    j = 50.0
    while (j := q * 2.0) < 10.0:
        q = q + 1.0
    return j + q


def _bool_carry(x: float) -> bool:
    odd = False
    while x > 0.0:
        odd = not odd
        x = x - 1.0
    return odd


def _bare_name_condition(x: float, go: bool) -> float:
    while go:
        x = x + 1.0
    return x


def _nested_residual(x: float) -> float:
    total = 0.0
    while x > 0.0:
        y = x
        while y > 1.0:
            y = y * 0.5
            total = total + 1.0
        x = x - 2.0
    return total


def _static_inside_residual(x: float) -> float:
    while x > 1.0:
        for _ in (0, 1):
            x = x * 0.5
    return x


def _residual_inside_static(x: float) -> float:
    for k in (1.0, 2.0):
        while x > k:
            x = x - k
    return x


def _residual_inside_branch(x: float, up: bool) -> float:
    if up:
        while x < 10.0:
            x = x * 2.0 + 1.0
    return x


def _int_carry(x: float) -> int:
    count = 0
    while x > 0.0:
        count = count + 1
        x = x - 1.0
    return count


def _residual_int_back(x: float, n: int) -> float:
    y = 0.5
    while x > 0.0:
        y = n + n
        x = x - 1.0
    return y


def test_residual_whiles_match_cpython() -> None:
    _oracle(_countdown, [{"x": 3.5}, {"x": 0.0}, {"x": -2.0}, {"x": 0.25}])
    _oracle(_int_entry_promotes, [{"x": 3.0}, {"x": 0.5}])
    _oracle(_float_entry_int_back, [{"x": 2.0}, {"x": -1.0}])
    _oracle(_header_walrus_zero_trip, [{"x": 2.5}, {"x": 0.0}, {"x": -3.0}])
    _oracle(_header_walrus_rebinds_a_carry, [{"q": 1.0}, {"q": 6.0}])
    _oracle(_bool_carry, [{"x": 3.0}, {"x": 4.0}, {"x": 0.0}])
    _oracle(_bare_name_condition, [{"x": 1.5, "go": False}])
    _oracle(_nested_residual, [{"x": 7.0}, {"x": 0.5}, {"x": -1.0}])
    _oracle(_static_inside_residual, [{"x": 9.0}, {"x": 1.0}])
    _oracle(_residual_inside_static, [{"x": 8.5}, {"x": 0.75}])
    _oracle(_residual_inside_branch, [{"x": 1.0, "up": True}, {"x": 1.0, "up": False}, {"x": 12.0, "up": True}])
    _oracle(_int_carry, [{"x": 3.5}, {"x": 0.0}, {"x": 1.0}])
    _oracle(_residual_int_back, [{"x": 2.0, "n": 3}, {"x": 0.0, "n": 3}, {"x": 1.0, "n": -4}])


def _body_only_name(x: float) -> float:
    while x > 0.0:
        t = x
        x = x - 1.0
    return t


def _bool_to_int_carry(x: float) -> float:
    b = True
    while x > 0.0:
        b = 1  # type: ignore[assignment]
        x = x - 1.0
    return float(int(b))


def _array_carry(x: float) -> float:
    v = np.zeros(2)
    while x > 0.0:
        v = v + 1.0
        x = x - 1.0
    return v[0]  # type: ignore[no-any-return]


def _array_store_inside_residual(x: float) -> float:
    v = np.zeros(2)
    while x > 0.0:
        v[0] = x
        x = x - 1.0
    return v[0]  # type: ignore[no-any-return]


def _sequence_carry(x: float) -> float:
    pair = (x, 1.0)
    while pair[0] > 0.0:
        pair = (pair[0] - 1.0, pair[1])
    return pair[1]


def _float_truthiness_condition(x: float) -> float:
    while x:
        x = x - 1.0
    return x


def _aggregate_truthiness_condition(x: float) -> float:
    while (x, x):
        x = x - 1.0
    return x


class _Accumulating:
    def __init__(self) -> None:
        self.acc = 0.0

    def step(self, x: float) -> float:
        while x > 0.0:
            self.acc = self.acc + x
            x = x - 1.0
        return self.acc


def test_a_state_write_inside_a_residual_loop_carries_the_slot() -> None:
    _oracle(_Accumulating().step, [{"x": 3.5}, {"x": 0.0}, {"x": 2.0}])


def test_residual_while_gaps_and_bans() -> None:
    for fn, match in [
        (_body_only_name, "the local name 't' is not bound on every path"),
        (_bool_to_int_carry, "the loop rebinds the local 'b' from bool to int across iterations"),
        (_array_carry, "'v' is an array; only bool, int, and float values can be carried"),
        (_array_store_inside_residual, "'v' is an array; only bool, int, and float values can be carried"),
        (_sequence_carry, "'pair' is a sequence; only bool, int, and float values can be carried"),
        (_float_truthiness_condition, "the branch condition must be a bool"),
        (_aggregate_truthiness_condition, "the truthiness of an aggregate is not supported"),
    ]:
        _rejects(fn, match)


# ---------------------------------------------------------------------- dynamic trips and degenerate exits


def _break_in_residual_while(x: float) -> float:
    while x > 0.0:
        break
    return x


def _break_under_residual_condition(x: float) -> float:
    for i in range(3):
        if x > float(i):
            break
        x = x + 1.0
    return x


def _continue_in_static_loop(x: float) -> float:
    for i in range(3):
        continue
    return x


def _dynamic_trip_count(n: int) -> float:
    acc = 0.0
    for i in range(n):
        acc = acc + float(i)
    return acc


def _scalar_iteration(x: float) -> float:
    for v in x:  # type: ignore[attr-defined]
        x = x + v
    return x


def test_dynamic_trips_reject_located() -> None:
    for fn, match in [
        (_dynamic_trip_count, "a range argument must be a compile-time constant int"),
        (_scalar_iteration, "a scalar is not iterable"),
    ]:
        _rejects(fn, match)


def test_a_body_leaving_the_loop_on_every_path_cannot_iterate() -> None:
    # The ruled degenerate family: no path reaches the back edge, so the loop is an `if` in disguise.
    _rejects(_break_in_residual_while, "leaves the loop on every path, so the loop cannot iterate")


# ---------------------------------------------------------------------- loop-internal break and continue


def _continue_in_residual_while(x: float) -> float:
    while x > 0.0:
        if x > 4.0:
            x = x - 4.0
            continue
        x = x - 1.0
    return x


def _while_true_guarded_break(x: float) -> float:
    while True:
        if x > 100.0:
            break
        x = x + 1.0
    return x


def _break_static(x: float) -> float:
    acc = x
    for i in range(5):
        if i == 3:
            break
        acc = acc + float(i)
    return acc


def _continue_static(x: float) -> float:
    acc = x
    for i in range(5):
        if i == 2:
            continue
        acc = acc + float(i)
    return acc


def _break_at_threshold(t: float) -> float:
    acc = 0.0
    for i in range(4):
        if float(i) >= t:
            break
        acc = acc * 2.0 + float(i) + 1.0
    return acc


def _continue_under_residual_condition(x: float) -> float:
    acc = 0.0
    for i in range(4):
        if x > float(i):
            continue
        acc = acc + x + float(i)
    return acc


def _break_then_continue(x: float) -> float:
    acc = 0.0
    for i in range(4):
        if acc > x:
            break
        if float(i) == 2.0:
            continue
        acc = acc + float(i) + 1.0
    return acc


def _unrolled_break_inside_residual_while(x: float, n: float) -> float:
    while n > 0.0:
        for i in range(4):
            if x > float(i):
                break
            x = x + 0.5
        n = n - 1.0
    return x


def _continue_then_break(x: float) -> float:
    marker = -1.0
    for i in range(3):
        if x > float(i):
            continue
        marker = float(i) * 10.0 + 1.0
        break
    return marker


def _continue_then_break_while(x: float) -> float:
    marker = -1.0
    k = 0
    while k < 3:
        k = k + 1
        if x > float(k):
            continue
        marker = float(k) * 10.0 + 1.0
        break
    return marker


def _broadcast_helper(x: float, y: float) -> float:
    for _ in range(1):
        if x > 0.0:
            return x
        if y > 1.0:
            break
    while y > 0.0:
        y = y - 1.0
    return y


def _residual_while_on_broadcast_lane(x: float, y: float) -> float:
    return _broadcast_helper(x, y)


def _post_break_binding_dropped(t: float) -> float:
    for i in range(3):
        if float(i) >= t:
            break
        y = t + float(i)
    return y


def _early_convergence(x: float, tol: float) -> float:
    err = x
    steps = 0.0
    while err > tol:
        err = err * 0.5
        steps = steps + 1.0
        if steps > 6.0:
            break
    return err + steps * 0.001


def _int_rebind_on_continue(c: bool, x: float) -> float:
    y = 0.0
    while x > 0.0:
        x = x - 1.0
        if c:
            y = 1
            continue
        y = 2
    return y


def _continue_only_body(x: float) -> float:
    while x > 0.0:
        x = x - 1.5
        continue
    return x


def _skim_then_break(x: float, n: float) -> float:
    hits = 0.0
    while n > 0.0:
        n = n - 1.0
        if x > n:
            hits = hits + 1.0
            continue
        break
    return hits + n * 0.5


def _mixed_exits(x: float, lim: float) -> float:
    acc = 0.0
    while x > 0.0:
        x = x - 1.0
        if acc > lim:
            return acc * 2.0
        if x > 3.0:
            acc = acc + 2.0
            continue
        if x < 0.5:
            break
        acc = acc + 1.0
    return acc + x


def _flag_on_break(x: float) -> float:
    found = False
    while x > 0.0:
        x = x - 1.0
        if x < 1.0:
            found = True
            break
    return x + (2.0 if found else 0.5)


def _two_breaks(x: float, a: float) -> float:
    tag = 0.0
    while x > 0.0:
        x = x - 1.0
        if x > a:
            tag = 1.0
            break
        if x < 0.5:
            tag = 2.0
            break
    return x + tag * 10.0


def _int_carry_with_break(n: int, lim: int) -> int:
    k = 0
    while k < n:
        k = k + 2
        if k > lim:
            break
    return k


class _AccumulateUntil:
    def __init__(self) -> None:
        self.total = 0.0

    def step(self, x: float, cap: float) -> float:
        while x > 0.0:
            x = x - 1.0
            self.total = self.total + x
            if self.total > cap:
                break
        return self.total * 0.5


def _promoted_carry_with_break(n: float) -> float:
    k = 0
    while n > 0.0:
        n = n - 1.0
        if n < 2.0:
            break
        # The int-to-float rebind IS the promotion trigger, so the loop re-runs with a float phi.
        k = k + 0.5  # type: ignore[assignment]
    return float(k) + n


def _promoted_carry_with_continue(n: float, c: bool) -> float:
    k = 0
    while n > 0.0:
        n = n - 1.0
        if c:
            k = k + 0.5  # type: ignore[assignment]
            continue
        k = k + 1
    return float(k)


def _broadcast_exit_helper(x: float, y: float) -> float:
    for _ in range(1):
        if x > 0.0:
            return x  # the pending lane leaves two sinks open, so the exits below are born broadcast
        if y > 1.0:
            break
    for i in range(3):
        if y > float(i):
            break
        y = y + 1.0
    return y


def _exits_on_a_broadcast_piece(x: float, y: float) -> float:
    return _broadcast_exit_helper(x, y)


class _ElideWithLoopExits:
    def __init__(self) -> None:
        self.level = 0.0

    def step(self, x: float, cap: float) -> float:
        while x > 0.0:
            x = x - 1.0
            self.level = self.level + x
            if self.level > cap:
                return self.level  # an in-loop site whose row matches the public slot's live-out
            if x < 1.0:
                break
        return self.level


def _post_break_binding_dropped_residual(x: float) -> float:
    while x > 0.0:
        x = x - 1.0
        if x > 2.0:
            break
        y = x + 0.25
    return y


def test_loop_exits_match_cpython() -> None:
    _oracle(_break_static, _X_ROWS)
    _oracle(_continue_static, _X_ROWS)
    _oracle(_continue_in_static_loop, _X_ROWS)
    _oracle(_break_under_residual_condition, [{"x": 5.0}, {"x": -0.5}, {"x": 0.0}, {"x": -3.0}])
    _oracle(_break_at_threshold, [{"t": 0.0}, {"t": 1.0}, {"t": 2.0}, {"t": 3.0}, {"t": 4.5}])
    _oracle(_continue_under_residual_condition, [{"x": -1.0}, {"x": 1.5}, {"x": 2.5}, {"x": 5.0}])
    _oracle(_break_then_continue, [{"x": 0.5}, {"x": 2.0}, {"x": 4.0}, {"x": 100.0}])
    # The break lane's env must be a snapshot: the continue-lane fold revives the frame dict in place,
    # which would corrupt a pending lane captured by reference.
    _oracle(_continue_then_break, [{"x": -1.0}, {"x": 0.5}, {"x": 1.5}, {"x": 2.5}, {"x": 5.0}])
    _oracle(_continue_then_break_while, [{"x": -1.0}, {"x": 1.5}, {"x": 2.5}, {"x": 3.5}, {"x": 5.0}])
    _oracle(
        _unrolled_break_inside_residual_while,
        [{"x": 0.2, "n": 2.0}, {"x": -1.0, "n": 3.0}, {"x": 5.0, "n": 1.0}, {"x": 0.0, "n": 0.0}],
    )
    # A pending callee return keeps the loop's lanes from sealing, so the following residual while is
    # broadcast into both open sinks; the emitter must scrub body-only bindings at each loop's exit, or
    # the arm join phis values that do not dominate it.
    _oracle(
        _residual_while_on_broadcast_lane,
        [{"x": -1.0, "y": -1.0}, {"x": 1.0, "y": 5.0}, {"x": -2.0, "y": 3.0}, {"x": -1.0, "y": 2.0}],
    )


def test_residual_loop_exits_match_cpython() -> None:
    _oracle(_continue_in_residual_while, [{"x": 9.5}, {"x": 4.0}, {"x": 0.0}, {"x": -2.0}, {"x": 17.25}])
    _oracle(_early_convergence, [{"x": 8.0, "tol": 1.0}, {"x": 1.0, "tol": 2.0}, {"x": 100.0, "tol": 0.0}])
    # The back-edge int-to-float conversion must reach every back sink: a direct continue bypasses the
    # flat body tail, so a conversion placed only there leaves the continue arm without the converted atom.
    _oracle(_int_rebind_on_continue, [{"c": True, "x": 2.0}, {"c": False, "x": 3.0}, {"c": True, "x": 0.0}])
    _oracle(_continue_only_body, [{"x": 4.0}, {"x": 0.5}, {"x": 0.0}, {"x": -1.0}])
    # The fall path never survives here, yet the continues keep a real back edge: this must stay accepted.
    _oracle(_skim_then_break, [{"x": 5.0, "n": 3.0}, {"x": 0.0, "n": 2.0}, {"x": 1.5, "n": 4.0}])
    _oracle(_mixed_exits, [{"x": 6.0, "lim": 2.0}, {"x": 3.0, "lim": 100.0}, {"x": 0.0, "lim": 1.0}])
    _oracle(_flag_on_break, [{"x": 3.5}, {"x": 0.5}, {"x": 0.0}, {"x": -2.0}])
    _oracle(_two_breaks, [{"x": 5.0, "a": 2.0}, {"x": 1.0, "a": 100.0}, {"x": 0.0, "a": 1.0}, {"x": 3.0, "a": 0.6}])
    _oracle(convergence_steps, [{"x": 100.0, "tol": 1.0}, {"x": 3.0, "tol": 8.0}, {"x": 500.0, "tol": 0.0}])
    _oracle(_int_carry_with_break, [{"n": 7, "lim": 4}, {"n": 3, "lim": 100}, {"n": 0, "lim": 1}])
    # The promote-and-rerun cycle discards a pass that already built exit lanes and terminators.
    _oracle(_promoted_carry_with_break, [{"n": 6.0}, {"n": 1.0}, {"n": 0.0}, {"n": 3.0}])
    _oracle(_promoted_carry_with_continue, [{"n": 4.0, "c": True}, {"n": 4.0, "c": False}, {"n": 0.0, "c": True}])
    # An exit lane born while several sinks are still open, and output-row elision at an in-loop return
    # site that shares the loop with a break.
    _oracle(_exits_on_a_broadcast_piece, [{"x": 1.0, "y": 0.0}, {"x": -1.0, "y": 5.0}, {"x": -1.0, "y": 0.5}])
    _oracle(
        _ElideWithLoopExits().step,
        [{"x": 4.0, "cap": 3.0}, {"x": 2.0, "cap": 100.0}, {"x": 0.0, "cap": 1.0}, {"x": 5.0, "cap": 0.0}],
    )
    _oracle(
        _AccumulateUntil().step,
        [{"x": 3.5, "cap": 10.0}, {"x": 5.0, "cap": 4.0}, {"x": 0.0, "cap": 1.0}, {"x": 2.0, "cap": 0.5}],
    )


def test_a_name_bound_only_before_the_break_drops_at_the_loop_exit() -> None:
    _rejects(_post_break_binding_dropped, "the local name 'y' is not bound on every path reaching this read")
    _rejects(_post_break_binding_dropped_residual, "the local name 'y' is not bound on every path reaching this read")


def test_a_while_true_with_a_guarded_break_stays_a_budget_rejection() -> None:
    # Ruled: the statically-true condition selects unrolling, the pending break lane never ends it, and the
    # budget draws the located rejection; no rotation into a residual while, no special detector.
    _rejects(_while_true_guarded_break, "the graph expansion budget is exhausted while expanding the unrolled loop")


# ---------------------------------------------------------------------- comprehensions


def _comp_over_sequence(x: float) -> float:
    ys = [x * s for s in (1.0, 2.0, 3.0)]
    return ys[0] + ys[1] + ys[2]


def _comp_over_vector(x: float) -> float:
    doubled = [v * 2.0 for v in np.array([x, 1.5])]
    return doubled[0] + doubled[1]  # type: ignore[no-any-return]


def _comp_over_rows(x: float) -> float:
    firsts = [row[0] for row in np.array([[x, 1.0], [2.0, 3.0]])]
    return firsts[0] * firsts[1]  # type: ignore[no-any-return]


def _nested_comp_matrix(x: float) -> float:
    m = np.array([[x * float(i) + float(j) for j in range(2)] for i in range(2)])
    return m[1, 1] - m[0, 1]  # type: ignore[no-any-return]


def _comp_over_range(x: float) -> float:
    weights = [float(k) * x for k in range(1, 4)]
    return weights[0] + weights[2]


def _comp_of_comp(x: float) -> float:
    base = [x + float(i) for i in range(2)]
    shifted = [b * 2.0 for b in base]
    return shifted[0] + shifted[1]


def _conditional_element(x: float) -> float:
    clipped = [v if v > 0.0 else 0.0 for v in (x, -x, 1.0)]
    return clipped[0] + clipped[1] + clipped[2]


def _empty_comp(x: float) -> float:
    tail: tuple[float, ...] = (1.0, 2.0)
    empty = [v for v in tail[2:]]
    return x + float(len(empty))


def _comp_returned(x: float) -> tuple[float, ...]:
    return tuple(v * x for v in (1.0, 2.0))


def test_comprehensions_match_cpython() -> None:
    for fn in (
        _comp_over_sequence,
        _comp_over_vector,
        _comp_over_rows,
        _nested_comp_matrix,
        _comp_over_range,
        _comp_of_comp,
        _conditional_element,
        _empty_comp,
    ):
        _oracle(fn, _X_ROWS)


def test_a_generator_argument_is_still_a_ban() -> None:
    _rejects(_comp_returned, "generator expressions are not supported")


# Just over half the expansion budget, so one body fits and two do not.
_HALF_BUDGET_EXPONENT = 55001


def _float_carry_fits(n: int, x: float) -> float:
    acc = 0.0
    i = 0
    while i < n:
        acc = acc + x**_HALF_BUDGET_EXPONENT
        i = i + 1
    return acc


def _int_carry_promotes_then_fits(n: int, x: float) -> float:
    acc = 0  # the int seed makes the first pass assume an INT carry, so the whole body is built twice
    i = 0
    while i < n:
        acc = acc + x**_HALF_BUDGET_EXPONENT  # type: ignore[assignment]  # the rebind to float IS the promotion
        i = i + 1
    return acc


def test_a_discarded_promote_pass_does_not_charge_the_budget() -> None:
    """
    Regression: the promote fixpoint discards what the previous pass built but kept charging its budget spend, so a
    body over half the budget was refused for the sole reason that its accumulator was seeded ``0`` and not ``0.0``.
    """
    fits = lower(_float_carry_fits)
    promotes = lower(_int_carry_promotes_then_fits)
    assert len(promotes.nodes) == len(fits.nodes)
