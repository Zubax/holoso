"""
Loops and comprehensions: static unrolling (trip-by-trip, budget-bounded), the residual while with its
loop-carried scalar phis, and the staged gaps (break/continue, dynamic trip counts, aggregate carries).
Positive shapes are oracle-verified against CPython; every staged gap is pinned as a located rejection.
"""

from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pytest

from holoso._eel import lower
from holoso._errors import UnsupportedConstruct

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


# ---------------------------------------------------------------------- staged exits and dynamic trips


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


def test_staged_exits_and_dynamic_trips_reject_located() -> None:
    for fn, match in [
        (_break_in_residual_while, "`break` is not supported yet"),
        (_break_under_residual_condition, "`break` is not supported yet"),
        (_continue_in_static_loop, "`continue` is not supported yet"),
        (_dynamic_trip_count, "a range argument must be a compile-time constant int"),
        (_scalar_iteration, "a scalar is not iterable"),
    ]:
        _rejects(fn, match)


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
