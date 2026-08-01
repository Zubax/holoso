"""
The mutation/ownership model, one test per reachable event; the borrow overlay and comprehension embedding
become executable with loops (M7), which owns the exhaustive DoD.

Recorded conservatism (a ruling, not a defect): a chained element read of a 2-D array (``m[0][1]``) derives
the row on the way, so the array is shared from then on and later stores through it reject -- the spellings
that stay admitted are multi-axis reads (``m[0, 1]``) and chained STORE targets, whose prefix is exempt.
Allocation states are global monotone facts, so a sharing event in one residual arm conservatively blocks a
store in the sibling arm as well.
"""

from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pytest
from jaxtyping import Float64

from holoso._eel import lower
from holoso._errors import UnsupportedConstruct

from ._eeloracle import assert_hir_matches_reference

type _Row = Mapping[str, float | bool | int]

_BUF = np.array([0.0, 0.0])


def _oracle(fn: Callable[..., object], vectors: Sequence[_Row]) -> None:
    compared = assert_hir_matches_reference(lower(fn), fn, vectors, label=fn.__name__)
    assert compared == len(vectors)


def _rejects(fn: object, match: str) -> None:
    with pytest.raises(UnsupportedConstruct, match=match):
        lower(fn)


# ---------------------------------------------------------------------- blessed idioms, oracle-verified


def _zeros_seed(x: float) -> tuple[float, float, float]:
    acc = np.zeros(3)
    acc[0] = x
    acc[1] = acc[0] + 1.0
    acc[2] += 2.5
    return acc[0], acc[1], acc[2]


def _int_accumulator(n: int) -> tuple[int, int]:
    hist = np.array([0, 10])
    hist[0] = n
    hist[1] += n
    return hist[0], hist[1]


def _multi_axis_accumulate(x: float) -> float:
    m = np.zeros((2, 2))
    m[0, 0] = x
    m[1, 1] = m[0, 0] + 1.0
    m[0][1] = 3.0
    return m[1, 1] + m[0, 1]  # type: ignore[no-any-return]


def _delay_line(x: float) -> tuple[float, float]:
    line = np.array([1.0, 2.0])
    line = np.array((*line[1:], x))
    return line[0], line[1]


def _whole_array_accumulate(x: float) -> float:
    acc = np.zeros(2)
    acc += np.array([x, x])
    acc *= 2.0
    acc /= 4.0
    acc -= 0.5
    return acc[1]  # type: ignore[no-any-return]


def _array_copy_is_writable(x: float) -> float:
    a = np.array([x, 2.0])
    b = np.array(a)
    b[0] = 9.0
    return a[0] + b[0]  # type: ignore[no-any-return]


def _len_is_not_a_handle(x: float) -> float:
    a = np.zeros(2)
    n = len(a)
    a[0] = float(n) + x
    return a[0]  # type: ignore[no-any-return]


def _fresh_argument_is_mutable(x: float) -> float:
    return _bump(np.array([x, 0.0]))


def _bump(v: Float64[np.ndarray, "2"]) -> float:
    v[0] += 1.0
    return v[0]  # type: ignore[no-any-return]


def _tuple_mid_path(x: float) -> float:
    c = (np.zeros(1), np.zeros(1))
    c[0][0] = x
    return c[0][0]  # type: ignore[no-any-return]


def _int_leaf_promotes_into_float_array(n: int) -> float:
    m = np.zeros(2)
    m[0] = n
    return m[0]  # type: ignore[no-any-return]


def test_blessed_idioms_admit_and_match_the_host() -> None:
    for fn in (
        _zeros_seed,
        _multi_axis_accumulate,
        _delay_line,
        _whole_array_accumulate,
        _array_copy_is_writable,
        _len_is_not_a_handle,
        _fresh_argument_is_mutable,
        _tuple_mid_path,
    ):
        _oracle(fn, [{"x": 2.0}, {"x": -0.5}])
    _oracle(_int_accumulator, [{"n": 3}, {"n": -7}])
    _oracle(_int_leaf_promotes_into_float_array, [{"n": 5}])


def _augmented_element_via_the_chained_spelling(v: float) -> float:
    m = np.zeros((2, 2))
    m[0][0] += v
    return m[0][0]  # type: ignore[no-any-return]


def test_augmented_element_stores_stay_admitted() -> None:
    _oracle(_augmented_element_via_the_chained_spelling, [{"v": 2.0}])


# ---------------------------------------------------------------------- one rejection per second-handle event


def _alias_by_name(x: float) -> float:
    a = np.zeros(1)
    b = a
    a[0] = x
    return b[0]  # type: ignore[no-any-return]


def _alias_multi_target(x: float) -> float:
    a = b = np.zeros(1)  # noqa: F841
    a[0] = x
    return a[0]  # type: ignore[no-any-return]


def _alias_walrus_embedding(x: float) -> float:
    z = ((t := np.array([x])), 2.0)
    t[0] = 5.0
    return z[1] + t[0]  # type: ignore[no-any-return]


def _alias_walrus_argument(x: float) -> float:
    r = _bump_array((t := np.array([x])))
    return r + t[0]  # type: ignore[no-any-return]


def _bump_array(buf: Float64[np.ndarray, "1"]) -> float:
    buf[0] = 5.0
    return buf[0]  # type: ignore[no-any-return]


def _callee_mutates_a_named_argument(x: float) -> float:
    buf = np.array([x])
    return _bump_array(buf)


def _unpack_extraction(x: float) -> float:
    pair = (np.zeros(1), np.zeros(1))
    r, s = pair
    r[0] = x
    return s[0]  # type: ignore[no-any-return]


def _index_extraction(x: float) -> float:
    m = np.ones((2, 2))
    row = m[0]  # noqa: F841
    m[1, 1] = x
    return m[1, 1]  # type: ignore[no-any-return]


def _root_shared_by_backup(w: float) -> float:
    grid = np.zeros((2, 2))
    backup = grid  # noqa: F841
    grid[1, 1] = w
    return grid[1, 1]  # type: ignore[no-any-return]


def _embedding_into_a_display(x: float) -> float:
    row = np.zeros(1)
    c = (row, 2.0)  # noqa: F841
    row[0] = x
    return row[0]  # type: ignore[no-any-return]


def _sequence_slice_shares_descendants(x: float) -> float:
    nested = (np.zeros(1), np.zeros(1))
    view = nested[0:1]  # noqa: F841
    nested[0][0] = x
    return nested[0][0]  # type: ignore[no-any-return]


def _asarray_shares(x: float) -> float:
    a = np.array([x, 2.0])
    b = np.asarray(a)  # noqa: F841
    a[0] = 9.0
    return a[0]  # type: ignore[no-any-return]


def _transpose_attribute_shares(x: float) -> float:
    v = np.array([x, 2.0])
    w = v.T  # noqa: F841
    v[0] = 9.0
    return v[0]  # type: ignore[no-any-return]


def _flatten_capture_shares(x: float) -> float:
    v = np.array([x, 2.0])
    f = v.flatten  # noqa: F841
    v[0] = 9.0
    return v[0]  # type: ignore[no-any-return]


def _tensor_slice_shares(x: float) -> float:
    v = np.array([x, 2.0, 3.0])
    head = v[0:2]  # noqa: F841
    v[0] = 9.0
    return v[0]  # type: ignore[no-any-return]


def _join_of_distinct_allocations(c: bool, x: float) -> float:
    xs = np.array([x]) if c else np.array([2.0])
    xs[0] = 5.0  # CPython would write through one of the two originals
    return xs[0]  # type: ignore[no-any-return]


def test_every_reachable_sharing_event_blocks_the_store() -> None:
    for fn in (
        _alias_by_name,
        _alias_multi_target,
        _alias_walrus_embedding,
        _alias_walrus_argument,
        _unpack_extraction,
        _index_extraction,
        _root_shared_by_backup,
        _embedding_into_a_display,
        _sequence_slice_shares_descendants,
        _asarray_shares,
        _transpose_attribute_shares,
        _flatten_capture_shares,
        _tensor_slice_shares,
        _join_of_distinct_allocations,
    ):
        _rejects(fn, "it is shared .*rebind a fresh value instead")


def test_a_callee_mutating_a_named_argument_rejects_in_the_chain() -> None:
    _rejects(_callee_mutates_a_named_argument, r"in _bump_array\(\): cannot store into buf\[0\]: it is shared")


# ---------------------------------------------------------------------- escaped roots


def _module_global_store(x: float) -> float:
    _BUF[0] += x
    return _BUF[0]  # type: ignore[no-any-return]


def _captured_alias_store(x: float) -> float:
    b = _BUF
    b[0] = x
    return b[0]  # type: ignore[no-any-return]


def _parameter_store(v: Float64[np.ndarray, "2"]) -> float:
    v[0] = 1.0
    return v[0]  # type: ignore[no-any-return]


def _make_closure_kernel() -> Callable[[float], float]:
    cell = np.zeros(1)

    def kernel(x: float) -> float:
        cell[0] = x
        return cell[0]  # type: ignore[no-any-return]

    return kernel


def _helper_with_default(x: float, buf: object = _BUF) -> float:
    buf[0] = x  # type: ignore[index]
    return buf[0]  # type: ignore[index, no-any-return]


def _default_store(x: float) -> float:
    return _helper_with_default(x)


def test_external_roots_are_never_mutable() -> None:
    _rejects(_module_global_store, "cannot mutate '_BUF': it was captured from outside the kernel")
    _rejects(_captured_alias_store, "it arrived from outside the kernel")
    _rejects(_parameter_store, "it arrived from outside the kernel")
    _rejects(_make_closure_kernel(), "cannot mutate 'cell': it was captured from outside the kernel")
    _rejects(_default_store, r"in _helper_with_default\(\): cannot store into buf\[0\]: it arrived from outside")


def test_numpy_slice_writes_reject_at_desugar() -> None:
    def slice_write(x: float) -> float:
        m = np.ones((2, 2))
        m[0, :] = x
        return m[0][0]  # type: ignore[no-any-return]

    _rejects(slice_write, "slice assignment is not supported")


# ---------------------------------------------------------------------- structural store rules


def _sequence_terminal(x: float) -> float:
    t = (0.0, 1.0)
    t[0] = x  # type: ignore[index]
    return t[0]


def _list_spelled_terminal(x: float) -> float:
    l = [0.0, 1.0]  # noqa: E741
    l[0] = x
    return l[0]


def _broadcast_store(x: float) -> float:
    m = np.ones((2, 2))
    m[0] = x  # numpy would broadcast into the row
    return m[0][0]  # type: ignore[no-any-return]


def _aggregate_rhs(x: float) -> float:
    m = np.zeros(2)
    m[0] = (x,)
    return m[0]  # type: ignore[no-any-return]


def _scalar_item_assignment(x: float) -> float:
    y = 1.0
    y[0] = x  # type: ignore[index]
    return y[0]  # type: ignore[index, no-any-return]


def _dynamic_store_index(n: int, x: float) -> float:
    m = np.zeros(2)
    m[n] = x
    return m[0]  # type: ignore[no-any-return]


def _store_out_of_bounds(x: float) -> float:
    m = np.zeros(1)
    m[2] = x
    return m[0]  # type: ignore[no-any-return]


def _float_into_int_array(x: float) -> float:
    a = np.array([0, 1])
    a[0] = x
    return float(a[0])


def _int_in_place_family_change(n: int) -> float:
    a = np.array([n, 2 * n])
    a /= 2
    return float(a[0])


def _tensor_aug_with_sequence(x: float) -> float:
    acc = np.zeros(2)
    acc += (x, x)
    return acc[0]  # type: ignore[no-any-return]


def _sequence_concat(x: float) -> float:
    t = (x,) + (2.0,)
    return t[1]


def _sequence_repeat(x: float) -> float:
    t = (x,) * 2
    return t[1]


def _sequence_aug(x: float) -> float:
    t = (x,)
    t += (2.0,)  # type: ignore[assignment]
    return t[0]


def _attribute_store(x: float) -> float:
    v = np.zeros(2)
    v.shape = (2, 1)
    return x


def test_structural_store_rules() -> None:
    for fn, match in [
        (_sequence_terminal, r"cannot store into t\[0\]: sequences are immutable"),
        (_list_spelled_terminal, r"cannot store into l\[0\]: sequences are immutable"),
        (_broadcast_store, "a store into an array must write one scalar element"),
        (_aggregate_rhs, "storing an aggregate into a container is not supported"),
        (_scalar_item_assignment, "a scalar does not support item assignment"),
        (_dynamic_store_index, "a subscript index must be a compile-time constant int"),
        (_store_out_of_bounds, "index 2 is out of bounds for an axis of length 1"),
        (_float_into_int_array, "storing a float into an integer array truncates on the host"),
        (_int_in_place_family_change, "cannot change the array's element family from int to float"),
        (_tensor_aug_with_sequence, "cannot mix an array with a Python list/tuple"),
        (_sequence_concat, r"the operator `\+` is not supported on a sequence"),
        (_sequence_repeat, r"the operator `\*` is not supported on a sequence"),
        (_sequence_aug, r"the operator `\+=` is not supported on a sequence"),
        (_attribute_store, "attribute stores are not supported yet"),
    ]:
        _rejects(fn, match)


# ---------------------------------------------------------------------- the recorded chained-read conservatism


def _chained_read_poisons_the_array(v: float) -> float:
    m = np.zeros((2, 2))
    m[1][1] = m[0][1] + v
    return m[1][1]  # type: ignore[no-any-return]


def test_chained_reads_poison_later_stores_by_ruling() -> None:
    _rejects(_chained_read_poisons_the_array, "it is shared")


# ---------------------------------------------------------------------- store admission under residual branches


def _store_under_a_residual_branch(c: bool, x: float) -> float:
    acc = np.zeros(2)
    if c:
        acc[0] = x
    else:
        acc[0] = -x
    acc[1] = acc[0] * 2.0
    return acc[0] + acc[1]  # type: ignore[no-any-return]


def test_stores_join_leafwise_under_residual_branches() -> None:
    _oracle(_store_under_a_residual_branch, [{"c": True, "x": 2.0}, {"c": False, "x": 2.0}])


def _sequence_store_index(x: float) -> float:
    m = np.zeros((2, 2))
    idx = [0, 1]
    m[idx] = x
    return m[0, 1]  # type: ignore[no-any-return]


def test_a_sequence_store_index_rejects() -> None:
    _rejects(_sequence_store_index, "a sequence index on an array is not supported")
