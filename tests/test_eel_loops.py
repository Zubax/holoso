"""
Loops and comprehensions: static unrolling (trip-by-trip, budget-bounded), the residual while with its
loop-carried scalar phis and break/continue exit lanes, and the permanent bans (dynamic trip counts,
aggregate carries, degenerate exit-on-every-path bodies). Positive shapes are oracle-verified against
CPython; every ban is pinned as a located rejection.
"""

from collections.abc import Callable, Mapping, Sequence

import dataclasses

import numpy as np
import pytest

import holoso
from holoso import FAddOptions, FCmpOptions, FMulILog2Options, FMulOptions, OperatorOptions, Options
from holoso._eel import lower
from holoso._errors import UnsupportedConstruct

from ._eeloracle import assert_hir_matches_reference
from ._modelref import DEFAULT_UNROLL_MAX_TRIPS
from ._public import strip_locations

type _Row = Mapping[str, float | bool | int]

_MIN_OPTIONS = Options(OperatorOptions())


def _oracle(fn: Callable[..., object], vectors: Sequence[_Row]) -> None:
    compared = assert_hir_matches_reference(lower(fn, DEFAULT_UNROLL_MAX_TRIPS).hir, fn, vectors, label=fn.__name__)
    assert compared == len(vectors)


def _as_int(value: object) -> int:
    assert isinstance(value, holoso.IntValue)
    return int(value)


def _rejects(fn: object, match: str) -> None:
    assert callable(fn)
    with pytest.raises(UnsupportedConstruct, match=match):
        holoso.synthesize(fn, _MIN_OPTIONS, name="k")


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


def _tuple_target(x: float) -> float:
    acc = 0.0
    for a, b in ((x, 2.0), (3.0, x), (0.5, 0.25)):
        acc = acc + a * b
    return acc


def _enumerated_sequence(x: float) -> float:
    acc = 0.0
    for k, w in enumerate((0.5, 0.25, x)):
        acc = acc + float(k) * w
    return acc


def _enumerated_with_start(x: float) -> float:
    acc = 0.0
    for k, w in enumerate((x, 2.0 * x), 1):
        acc = acc + float(k) * w
    for k, w in enumerate((x, -x), start=2):
        acc = acc + float(k) * w
    return acc


def _enumerated_range(x: float) -> float:
    acc = x
    for k, n in enumerate(range(2, 5)):
        acc = acc + float(k * n)
    return acc


def _enumerated_pairs_unspread(x: float) -> float:
    acc = 0.0
    for p in enumerate((x, 2.0)):
        acc = acc + p[1] + float(p[0])
    return acc


def _materialized_enumerate(x: float) -> float:
    pairs = tuple(enumerate((x, 2.0)))
    return pairs[0][1] + pairs[1][1] + float(pairs[1][0])


def _enumerate_inside_residual_while(x: float) -> float:
    while x < 8.0:
        acc = 0.0
        for k, w in enumerate((0.5, 0.25)):
            acc = acc + float(k) * w
        x = x + 1.0 + acc
    return x


def _mutated_while_enumerated(x: float) -> float:
    xs = np.array([x, 2.0, 3.0])
    acc = 0.0
    for k, w in enumerate(xs):
        acc = acc + w
        xs[2] = 9.0
    return acc + float(xs[2])


def _reused_enumerate(x: float) -> float:
    pairs = enumerate((x, 2.0))
    acc = 0.0
    for _, w in pairs:
        acc = acc + w
    for _, w in pairs:
        acc = acc + w
    return acc


def _subscripted_enumerate(x: float) -> float:
    return enumerate((x, 2.0))[0][1]  # type: ignore[index,no-any-return]


def _measured_enumerate(x: float) -> float:
    return x + float(len(enumerate((x, 2.0))))  # type: ignore[arg-type]


def _replayed_enumerate(x: float) -> float:
    pairs = enumerate((1.0, 2.0))
    acc = 0.0
    while x > 0.0:
        for _, w in pairs:
            acc = acc + w
        x = x - 1.0
    return acc


def _sibling_replayed_enumerate(x: float) -> float:
    acc = 0.0
    while ((pairs := enumerate((1.0, 2.0))), x > 0.0)[1]:
        x = x - 1.0
    while x > -3.0:
        for _, w in pairs:
            acc = acc + w
        x = x - 1.0
    return acc


def _joined_enumerate(x: float) -> float:
    pairs = enumerate((1.0, 2.0)) if x > 0.0 else enumerate((3.0, 4.0))
    acc = 0.0
    for _, w in pairs:
        acc = acc + w
    for _, w in pairs:
        acc = acc + w
    return acc


def _tensored_enumerate(x: float) -> float:
    return float(np.array(enumerate((x, 2.0)))[0][1])


def _returned_enumerate(x: float) -> tuple[tuple[int, float], tuple[int, float]]:
    return enumerate((x, 2.0))  # type: ignore[return-value]


class _StoredIterator:
    def __init__(self) -> None:
        self.pairs = ((0, 0.0), (1, 0.0))

    def step(self, load: bool, x: float) -> float:
        if load:
            self.pairs = enumerate((x, 2.0))  # type: ignore[assignment]
            return -1.0
        total = 0.0
        for _, value in self.pairs:
            total = total + value
        return total


def test_enumerate_is_a_one_shot_iterator() -> None:
    # CPython's enumerate is lazy and one-shot: a mid-iteration store would read the eager snapshot stale
    # (conservatively refused, like the borrow on a directly iterated array), a drained iterator yields
    # nothing where the snapshot would yield again, and only iteration consumes it.
    _rejects(_mutated_while_enumerated, "shared")
    _rejects(_reused_enumerate, "consumed once")
    _rejects(_subscripted_enumerate, "not subscriptable")
    _rejects(_measured_enumerate, "requires an aggregate")
    # The residualized body replays every hardware iteration, re-consuming what Python drains once; the
    # sibling spelling escapes one residual loop through a walrus header binding and drains in another at
    # the same depth, so the guard must compare pass identity, not depth.
    _rejects(_replayed_enumerate, "outside this data-dependent loop")
    _rejects(_sibling_replayed_enumerate, "outside this data-dependent loop")
    # No consumer may launder the one-shot into an ordinary re-iterable value: not a branch join, not an
    # array conversion, not the module boundary, and not a persistent-state slot reloaded next transaction.
    _rejects(_joined_enumerate, "cannot merge")
    _rejects(_tensored_enumerate, "requires a sequence or array argument")
    _rejects(_returned_enumerate, "is not a sequence")
    with pytest.raises(UnsupportedConstruct, match="cannot be installed"):
        holoso.synthesize(_StoredIterator().step, _MIN_OPTIONS, name="kernel")


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
        _tuple_target,
        _enumerated_sequence,
        _enumerated_with_start,
        _enumerated_range,
        _enumerated_pairs_unspread,
        _materialized_enumerate,
        _enumerate_inside_residual_while,
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


def test_a_huge_static_range_stop_does_not_fit_the_integer_format() -> None:
    _rejects(_huge_static_range, "does not fit int24; raise wint_min")


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
    _oracle(_nested_residual, [{"x": 7.0}, {"x": 0.5}, {"x": -1.0}])
    _oracle(_static_inside_residual, [{"x": 9.0}, {"x": 1.0}])
    _oracle(_residual_inside_static, [{"x": 8.5}, {"x": 0.75}])
    _oracle(_residual_inside_branch, [{"x": 1.0, "up": True}, {"x": 1.0, "up": False}, {"x": 12.0, "up": True}])
    _oracle(_int_carry, [{"x": 3.5}, {"x": 0.0}, {"x": 1.0}])
    _oracle(_residual_int_back, [{"x": 2.0, "n": 3}, {"x": 0.0, "n": 3}, {"x": 1.0, "n": -4}])


def test_the_public_artifacts_discriminate_unrolling_from_residualization() -> None:
    """
    The unroll-vs-residualize decision is itself public: a residual while keeps loop-carried phis in the
    location-stripped `frontend_ir[-1]` and makes the transaction latency data-dependent (max II `None`),
    while an unrolled loop leaves neither.
    """
    unrolled = holoso.synthesize(_static_while, Options(OperatorOptions(fmul_ilog2=FMulILog2Options())), name="k")
    residual = holoso.synthesize(_countdown, Options(OperatorOptions(fadd=FAddOptions(), fcmp=FCmpOptions())), name="k")
    assert unrolled.initiation_interval[1] is not None
    assert residual.initiation_interval[1] is None
    unrolled_lines = strip_locations(unrolled.frontend_ir[-1]).splitlines()
    residual_lines = strip_locations(residual.frontend_ir[-1]).splitlines()
    assert not any(line.lstrip().startswith("phi %") for line in unrolled_lines)
    assert any(line.lstrip().startswith("phi %") for line in residual_lines)


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


def _dynamic_trip_count(n: int) -> float:
    acc = 0.0
    for i in range(n):
        acc = acc + float(i)
    return acc


def _scalar_iteration(x: float) -> float:
    for v in x:  # type: ignore[attr-defined]
        x = x + v
    return x


def test_a_scalar_is_not_iterable() -> None:
    _rejects(_scalar_iteration, "a scalar is not iterable")


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


# Just over half the expansion budget, so one body fits and two do not.
_HALF_BUDGET_ELEMENTS = 26000


def _float_carry_fits(n: int, x: float) -> float:
    acc = 0.0
    i = 0
    while i < n:
        acc = acc + x * float(len(list(range(_HALF_BUDGET_ELEMENTS))))
        i = i + 1
    return acc


def _int_carry_promotes_then_fits(n: int, x: float) -> float:
    acc = 0  # the int seed makes the first pass assume an INT carry, so the whole body is built twice
    i = 0
    while i < n:
        acc = acc + x * float(len(list(range(_HALF_BUDGET_ELEMENTS))))  # type: ignore[assignment]
        i = i + 1
    return acc


def test_a_discarded_promote_pass_does_not_charge_the_budget() -> None:
    """
    Regression: the promote fixpoint discards what the previous pass built but kept charging its budget spend, so a
    body over half the budget was refused for the sole reason that its accumulator was seeded `0` and not `0.0`.
    The re-run kernel therefore must synthesize, to Verilog byte-identical with its float-seeded twin under the one
    module name.
    """
    options = Options(OperatorOptions(fadd=FAddOptions(), fmul=FMulOptions()))
    fits = holoso.synthesize(_float_carry_fits, options, name="k")
    promotes = holoso.synthesize(_int_carry_promotes_then_fits, options, name="k")
    assert promotes.initiation_interval == fits.initiation_interval
    assert promotes.verilog_output.verilog == fits.verilog_output.verilog


# ---------------------------------------------------------------------- the counted for


def _counted_sum(n: int) -> int:
    s = 0
    for i in range(n):
        s = s + i
    return s


def _counted_bounds(a: int, b: int) -> int:
    s = 0
    for i in range(a, b):
        s = s + i
    return s


def _counted_down(n: int) -> int:
    s = 0
    for i in range(n, 0, -1):
        s = s + i
    return s


def _counted_step(n: int) -> int:
    s = 0
    for i in range(0, n, 3):
        s = s + i
    return s


def _counted_float_body(n: int) -> float:
    acc = 0.0
    for i in range(n):
        acc = acc + float(i)
    return acc


def _counted_carried_out(n: int) -> int:
    i = 99
    for i in range(n):
        pass
    return i


def _counted_target_rebound(n: int) -> int:
    s = 0
    for i in range(n):
        i = i * 2
        s = s + i
    return s


def _counted_break(n: int) -> int:
    s = 0
    for i in range(n):
        if i > 3:
            break
        s = s + i
    return s


def _counted_continue(n: int) -> int:
    s = 0
    for i in range(n):
        if i == 2:
            continue
        s = s + i
    return s


def _counted_nested(n: int) -> int:
    s = 0
    for i in range(n):
        for j in range(i):
            s = s + 1
    return s


def _counted_eval_once(n: int) -> int:
    r = range(n)
    n = 0
    t = 0
    for i in r:
        t = t + 1
    return t + n


def _counted_shadow(n: int) -> int:
    t = 0
    for n in range(n):
        t = t + 1
    return t


def _counted_reused_range(n: int) -> int:
    r = range(n)
    s = 0
    for i in r:
        s = s + i
    for j in r:
        s = s + 1
    return s


def _sum_over(r: range) -> int:
    s = 0
    for i in r:
        s = s + i
    return s


def _counted_helper_transport(n: int) -> int:
    return _sum_over(range(n))


def _counted_prebound_float(n: int) -> float:
    i = 0.5
    for i in range(n):
        pass
    return float(i)


def _first_square_over(n: int) -> int:
    for i in range(n):
        if i * i > 5:
            return i
    return -1


def _counted_return_in_helper(n: int) -> int:
    return _first_square_over(n) + 100


def _counted_rail_overshoot_down() -> tuple[int, int]:
    trips = 0
    last = 0
    for last in range(-8388600, -8388607, -3):
        trips = trips + 1
    return trips, last


class _CountedFromSlot:
    def __init__(self) -> None:
        self.k = 2

    def step(self, n: int) -> int:
        s = 0
        for i in range(self.k, n):
            s = s + i
        return s


def _counted_inside_residual_while(n: int, x: float) -> int:
    s = 0
    while x > 0.0:
        for i in range(n):
            s = s + i
        x = x - 1.0
    return s


def _range_from_helper_arms(c: bool) -> int:
    s = 0
    for i in _pick_range(c):
        s = s + i
    return s


def _pick_range(c: bool) -> range:
    if c:
        return range(4)
    return range(4)


def _nested_range_display() -> int:
    s = 0
    for r in [range(2), range(3)]:
        for i in r:
            s = s + i
    return s


_N_ROWS: list[_Row] = [{"n": 0}, {"n": 1}, {"n": 2}, {"n": 7}]


def test_counted_loops_match_cpython() -> None:
    _oracle(_counted_sum, _N_ROWS)
    _oracle(_counted_bounds, [{"a": 0, "b": 0}, {"a": -2, "b": 3}, {"a": 3, "b": 1}, {"a": 1, "b": 6}])
    _oracle(_counted_down, _N_ROWS)
    _oracle(_counted_step, [{"n": 0}, {"n": 1}, {"n": 3}, {"n": 10}])
    _oracle(_counted_float_body, _N_ROWS)
    _oracle(_counted_carried_out, _N_ROWS)
    _oracle(_counted_target_rebound, _N_ROWS)
    _oracle(_counted_break, [{"n": 0}, {"n": 3}, {"n": 9}])
    _oracle(_counted_continue, [{"n": 2}, {"n": 6}])
    _oracle(_counted_nested, [{"n": 0}, {"n": 1}, {"n": 4}])
    _oracle(_counted_eval_once, [{"n": 0}, {"n": 4}])
    _oracle(_counted_shadow, [{"n": 0}, {"n": 5}])
    _oracle(_counted_reused_range, [{"n": 0}, {"n": 4}])
    _oracle(_counted_helper_transport, _N_ROWS)
    _oracle(_counted_prebound_float, _N_ROWS)
    _oracle(_range_from_helper_arms, [{"c": True}, {"c": False}])
    _oracle(_nested_range_display, [{}])
    _oracle(_counted_return_in_helper, [{"n": 0}, {"n": 2}, {"n": 5}])
    _oracle(_CountedFromSlot().step, [{"n": 0}, {"n": 2}, {"n": 6}])
    _oracle(_counted_inside_residual_while, [{"n": 0, "x": 0.0}, {"n": 3, "x": 2.0}])


def _static_counted_sum() -> int:
    s = 0
    for i in range(4):
        s = s + i
    return s


def test_the_unroll_threshold_selects_the_lowering_of_a_pure_kernel() -> None:
    """
    The same kernel unrolls at or below the threshold (exact II, no phis) and residualizes above it
    (open-ended II, loop phis), with identical values either way; the branch-free body keeps the II
    discriminator sound. Only the lowering is claimed: a counted body is a data-dependent region, so it
    refuses what any residual loop refuses -- an aggregate state install, a runtime-int subscript.
    """
    unrolled = holoso.synthesize(_static_counted_sum, Options(OperatorOptions(), unroll_max_trips=4), name="k")
    counted = holoso.synthesize(_static_counted_sum, Options(OperatorOptions(), unroll_max_trips=3), name="k")
    assert unrolled.initiation_interval[1] is not None
    assert counted.initiation_interval[1] is None
    assert not any(line.lstrip().startswith("phi %") for line in strip_locations(unrolled.frontend_ir[-1]).splitlines())
    assert any(line.lstrip().startswith("phi %") for line in strip_locations(counted.frontend_ir[-1]).splitlines())
    assert [_as_int(v) for v in unrolled.numerical_model.elaborate().run()] == [6]
    assert [_as_int(v) for v in counted.numerical_model.elaborate().run()] == [6]


def _static_counted_target() -> int:
    for w in range(3):
        pass
    return w


def test_a_static_counted_loop_defines_its_target_after_the_loop() -> None:
    forced = holoso.synthesize(_static_counted_target, Options(OperatorOptions(), unroll_max_trips=0), name="k")
    (out,) = forced.numerical_model.elaborate().run()
    assert _as_int(out) == 2


def _counted_rail_overshoot() -> tuple[int, int]:
    trips = 0
    last = 0
    for last in range(8388600, 8388607, 3):
        trips = trips + 1
    return trips, last


def test_a_saturating_counter_overshoot_is_unobservable() -> None:
    """The final advance saturates at the rail the kernel asked for, which must end the loop and never leak."""
    options = Options(OperatorOptions(), wint_min=24, unroll_max_trips=0)  # the bounds are 24-bit; nothing lends it
    for kernel, expected in ((_counted_rail_overshoot, [3, 8388606]), (_counted_rail_overshoot_down, [3, -8388606])):
        forced = holoso.synthesize(kernel, options, name="k")
        assert [_as_int(v) for v in forced.numerical_model.elaborate().run()] == expected


def _unused_huge_range(x: float) -> float:
    r = range(10**25)
    return x + 1.0


def test_an_unconsumed_huge_range_compiles() -> None:
    result = holoso.synthesize(_unused_huge_range, Options(OperatorOptions(fadd=FAddOptions())), name="k")
    (out,) = result.numerical_model.elaborate().run(2.0)
    assert float(out) == 3.0


def _range_join_equal(c: bool) -> int:
    r = range(3) if c else range(3)
    s = 0
    for i in r:
        s = s + i
    return s


def _static_range_consumers() -> int:
    v = np.array(range(3))
    picked = range(5)[3]
    lo, hi = range(2)
    return int(v[2]) + picked + lo + hi + len(range(10**6))


def _returned_static_range() -> tuple[int, ...]:
    return range(3)  # type: ignore[return-value]


def test_static_ranges_decay_wherever_a_sequence_is_demanded() -> None:
    _oracle(_range_join_equal, [{"c": True}, {"c": False}])
    # `len(range(10**6))` folds to a value no 16-bit word holds, so this kernel names the width it needs.
    wide = dataclasses.replace(_MIN_OPTIONS, wint_min=24)
    result = holoso.synthesize(_static_range_consumers, wide, name="k")
    (out,) = result.numerical_model.elaborate().run()
    assert _as_int(out) == _static_range_consumers()
    returned = holoso.synthesize(_returned_static_range, _MIN_OPTIONS, name="k")
    assert [_as_int(v) for v in returned.numerical_model.elaborate().run()] == [0, 1, 2]


def _runtime_step(n: int, m: int) -> int:
    s = 0
    for i in range(0, n, m):
        s = s + i
    return s


def _float_range_bound(x: float) -> float:
    for i in range(x):  # type: ignore[call-overload]
        x = x + 1.0
    return x


def _zero_static_step(n: int) -> int:
    s = 0
    for i in range(0, n, 0):
        s = s + i
    return s


def _runtime_range_len(n: int) -> int:
    return len(range(n))


def _overflowing_range_len() -> int:
    return len(range(10**25))


def _runtime_range_list(n: int) -> int:
    v = list(range(n))
    return v[0]


def _range_arithmetic(n: int) -> int:
    return range(n) + 1  # type: ignore[operator]


def _range_condition(n: int) -> int:
    if range(n):
        return 1
    return 0


def _range_attr(n: int) -> int:
    return range(n).start


def _returned_runtime_range(n: int) -> tuple[int, ...]:
    return range(n)  # type: ignore[return-value]


def _unconditional_break(n: int) -> int:
    for i in range(n):
        break
    return n


def _unbound_target_after_runtime_loop(n: int) -> int:
    for i in range(n):
        pass
    return i


def _aggregate_target_runtime_loop(n: int) -> int:
    i = [0]
    for i in range(n):  # type: ignore[assignment]
        pass
    return n


def _range_join_unequal(c: bool, n: int) -> int:
    r = range(3) if c else range(n)
    s = 0
    for i in r:
        s = s + i
    return s


class _RangeState:
    def __init__(self) -> None:
        self.x = (0, 1)

    def step(self, n: int) -> int:
        self.x = range(n)  # type: ignore[assignment]
        return n


def _range_into_an_array_operation(x: float) -> float:
    v = np.array([x, 2.0])
    return float(v @ range(2))


def _range_through_a_residual_loop(x: float) -> range:
    while x > 0.0:
        if x > 2.0:
            return range(3)
        x = x - 1.0
    return range(3)


def _range_crossing_a_residual_frame(x: float) -> int:
    s = 0
    for i in _range_through_a_residual_loop(x):
        s = s + i
    return s


def test_counted_loop_gaps_and_bans() -> None:
    for fn, match in [
        (_runtime_step, "the range step must be a compile-time constant int"),
        (_float_range_bound, "a range argument must be an int, not a float"),
        (_zero_static_step, r"range\(\) rejects its arguments"),
        (_runtime_range_len, r"len\(\) of a range with a runtime bound is not supported"),
        (_overflowing_range_len, r"len\(\) of this range overflows, exactly as it does in CPython"),
        (_runtime_range_list, "a range with a runtime bound can only drive a for loop"),
        (_range_arithmetic, "a range cannot be used as a scalar here"),
        (_range_condition, "a range cannot be used as a scalar here"),
        (_range_attr, "a range has no supported attribute 'start'"),
        (_returned_runtime_range, "a range with a runtime bound can only drive a for loop"),
        (_unconditional_break, "leaves the loop on every path, so the loop cannot iterate"),
        (_unbound_target_after_runtime_loop, "the local name 'i' is not bound on every path"),
        (_aggregate_target_runtime_loop, "'i' is a sequence; only bool, int, and float values can be carried"),
        (_range_join_unequal, "holds branch values the compiler cannot merge"),
        (_RangeState().step, "a range with a runtime bound can only drive a for loop"),
        (_range_crossing_a_residual_frame, "a range returned across a data-dependent region is not supported"),
        (_range_into_an_array_operation, "an array operation on a Python list/tuple is not supported"),
    ]:
        _rejects(fn, match)
