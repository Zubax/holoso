"""
Aggregate semantics, driven black-box through the differential oracle wherever the behavior is observable
and through located-rejection pins where it is not.
"""

import math
import types
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pytest
from jaxtyping import Float32, Float64, Int64

import holoso
from holoso import FAddOptions, OperatorOptions, Options
from holoso._eel import lower
from holoso._errors import UnsupportedConstruct

from ._eeloracle import assert_hir_matches_reference
from ._modelref import DEFAULT_UNROLL_MAX_TRIPS

_MIN_OPTIONS = Options(OperatorOptions(fadd=FAddOptions()))

type _Row = Mapping[str, float | bool | int]

_TABLE = (2.0, 4.0, 8.0)
_GRID = [[1.0, 2.0], [3.0, 4.0]]
_ARRAY = np.array([[1.0, 2.0], [3.0, 4.0]])
_WITH_NAN = (1.0, float("nan"))


def _oracle(fn: Callable[..., object], vectors: Sequence[_Row]) -> None:
    compared = assert_hir_matches_reference(lower(fn, DEFAULT_UNROLL_MAX_TRIPS).hir, fn, vectors, label=fn.__name__)
    assert compared == len(vectors)


def _rejects(fn: object, match: str) -> None:
    with pytest.raises(UnsupportedConstruct, match=match):
        lower(fn, DEFAULT_UNROLL_MAX_TRIPS)


# ---------------------------------------------------------------------- constant subscripts follow CPython


def _subscripts(x: float) -> tuple[float, float, float, float]:
    nested = ((1.0, 2.0), (3.0, x))
    return _TABLE[-1], _GRID[1][0], _ARRAY[0][1], nested[1][1]


def _slices(x: float) -> tuple[float, float, float, int]:
    head = _TABLE[:2]
    clamped = _TABLE[1:99]
    rows = _ARRAY[0:1]
    return head[0], clamped[-1], rows[0][1] * x, len(_TABLE[2:2])


def _multi_axis(x: float) -> tuple[float, float, float, float]:
    column = _ARRAY[:, 1]
    row = _ARRAY[1, :]
    block = _ARRAY[0:2, 1:2]
    return _ARRAY[1, 1] * x, column[0], row[0], block[1][0]


def test_constant_subscripts() -> None:
    for fn in (_subscripts, _slices, _multi_axis):
        _oracle(fn, [{"x": 2.0}, {"x": -0.5}])


def _index_out_of_bounds(x: float) -> float:
    return _TABLE[3] * x  # type: ignore[misc, no-any-return]


def _index_dynamic(n: int) -> float:
    return _TABLE[n]


def _index_bool(x: float) -> float:
    return _TABLE[True] * x


def _index_scalar(x: float) -> float:
    return x[0]  # type: ignore[index, no-any-return]


def _slice_scalar(x: float) -> float:
    return x[0:1]  # type: ignore[index, no-any-return]


def _empty_tensor_slice(x: float) -> float:
    return len(_ARRAY[:0]) * x


def _axis_bound_behind_empty_slice(x: float) -> float:
    return len(_ARRAY[:0, 99]) * x  # the empty leading slice must not mask the axis-1 bounds fault


def _multi_axis_on_sequence(x: float) -> float:
    return _GRID[0, 0] * x  # type: ignore[call-overload, no-any-return]


def _multi_axis_rank(x: float) -> float:
    return float(_ARRAY[0, 0, 0]) * x


def test_subscript_rejections() -> None:
    for fn, match in [
        (_index_out_of_bounds, "index 3 is out of bounds for a sequence of length 3"),
        (_index_dynamic, "a subscript index must be a compile-time constant int"),
        (_index_bool, "a subscript index must be an int, not a bool"),
        (_index_scalar, "a scalar is not subscriptable"),
        (_slice_scalar, "a scalar cannot be sliced"),
        (_empty_tensor_slice, "the slice selects no elements"),
        (_axis_bound_behind_empty_slice, "index 99 is out of bounds for an axis of length 2"),
        (_multi_axis_on_sequence, "too many indices: a multi-axis subscript works only on an array"),
        (_multi_axis_rank, "must name every axis: the array is 2-D, got 3"),
    ]:
        _rejects(fn, match)


# ---------------------------------------------------------------------- unpacking and splats


def _unpacks(x: float) -> tuple[float, float, float, float]:
    a, b, c = _TABLE
    r0, r1 = _ARRAY
    p, q = r0[0], r1[1]
    return a + b, c * x, p, q


def test_unpacking() -> None:
    _oracle(_unpacks, [{"x": 1.5}])


def _unpack_too_many(x: float) -> float:
    a, b = _TABLE  # type: ignore[misc]
    return a + b + x  # type: ignore[no-any-return]


def _unpack_too_few(x: float) -> float:
    a, b, c, d = _TABLE  # type: ignore[misc]
    return a + b + c + d + x  # type: ignore[no-any-return]


def _unpack_scalar(x: float) -> float:
    a, b = x  # type: ignore[misc]
    return a + b  # type: ignore[has-type, no-any-return]


def test_unpack_arity_uses_cpython_wording() -> None:
    _rejects(_unpack_too_many, r"too many values to unpack \(expected 2\)")
    _rejects(_unpack_too_few, r"not enough values to unpack \(expected 4, got 3\)")
    _rejects(_unpack_scalar, "cannot unpack a scalar")


def _splats(x: float) -> tuple[float, float, float, float, float, float]:
    seed = (x, 2.0)
    spliced = (0.5, *seed, *_TABLE[:1])
    y = math.hypot(*seed)
    return spliced[0], spliced[1], spliced[2], spliced[3], y, min(*seed)


def test_splats_in_displays_and_calls() -> None:
    _oracle(_splats, [{"x": 3.0}, {"x": -4.0}])


def _splat_scalar(x: float) -> float:
    return math.hypot(*x)  # type: ignore[misc]


def test_splatting_a_scalar_rejects() -> None:
    _rejects(_splat_scalar, "cannot unpack a scalar")


# ---------------------------------------------------------------------- factories and conversions


def _eye_factory(x: float) -> tuple[float, float]:
    identity = np.eye(2)
    return identity[0][0] * x + identity[0][1], identity[1][1]


def test_np_eye_folds_to_a_fresh_identity() -> None:
    _oracle(_eye_factory, [{"x": 2.0}, {"x": -0.5}])


def _factory_empty(x: float) -> float:
    return np.zeros(0)[0] * x  # type: ignore[no-any-return]


def _factory_residual_argument(n: int) -> float:
    return np.zeros(n)[0]  # type: ignore[no-any-return]


def _factory_keyword(x: float) -> float:
    return np.zeros(3, dtype=float)[0] * x  # type: ignore[no-any-return]


def test_factory_rejections() -> None:
    _rejects(_factory_empty, r"np.zeros\(\) must build a non-empty 1-D or 2-D numeric array")
    _rejects(_factory_residual_argument, r"the arguments of np.zeros\(\) must be compile-time constants")
    _rejects(_factory_keyword, r"np.zeros\(\) takes no keyword arguments")


def _convert_scalar(x: float) -> float:
    return np.array(x)  # type: ignore[return-value]


def _convert_empty(x: float) -> float:
    return np.array([])[0] * x  # type: ignore[no-any-return]


def _convert_ragged(x: float) -> float:
    return np.array([[1.0], [2.0, x]])[0][0]  # type: ignore[no-any-return]


def _convert_deep(x: float) -> float:
    return np.array([[[x]]])[0][0][0]  # type: ignore[no-any-return]


def _convert_bool_mix(x: float) -> float:
    return np.array([True, x])[1]  # type: ignore[no-any-return]


def _list_of_scalar(x: float) -> float:
    return list(x)[0]  # type: ignore[call-overload, no-any-return]


def test_conversion_rejections() -> None:
    for fn, match in [
        (_convert_scalar, r"np.array\(\) requires a sequence or array argument, not a scalar"),
        (_convert_empty, r"np.array\(\) of an empty sequence is not supported"),
        (_convert_ragged, r"np.array\(\) requires rectangular rows of equal nonzero length"),
        (_convert_deep, r"np.array\(\) supports only 1-D and 2-D rectangular constructions"),
        (_convert_bool_mix, "an array must hold numbers, not booleans"),
        (_list_of_scalar, r"list\(\) requires an aggregate argument"),
    ]:
        _rejects(fn, match)


# ---------------------------------------------------------------------- shape queries and tensor methods


def _shape_queries(x: float) -> tuple[int, int, int, int, int, float]:
    rows, cols = _ARRAY.shape
    flat = _ARRAY.flatten()
    vector = np.array([x, 2.0])
    return rows + cols, _ARRAY.ndim, len(_ARRAY), x.ndim, len(x.shape), vector.T[0] + flat[3]  # type: ignore[attr-defined]


def test_shape_queries_and_scalar_rank_zero() -> None:
    # A plain Python float has no .ndim on the host, so the oracle cannot drive it; pin by public evaluation.
    result = holoso.synthesize(_shape_queries, _MIN_OPTIONS, name="kernel")
    assert [p.name for p in result.output_ports] == ["out_0", "out_1", "out_2", "out_3", "out_4", "out_5"]
    assert [float(v) for v in result.numerical_model.elaborate().run(1.5)] == [4.0, 2.0, 2.0, 0.0, 0.0, 1.5 + 4.0]


def _sequence_shape(x: float) -> float:
    return _TABLE.shape[0] * x  # type: ignore[attr-defined, no-any-return]


def _sequence_append(x: float) -> float:
    row = [x]
    appender = row.append  # noqa: F841 -- the attribute read itself is the pinned construct
    return row[0]


def _scalar_attribute(x: float) -> float:
    return x.imag


def _array_unknown_attribute(x: float) -> float:
    return _ARRAY.strides[0] * x


def _float_bit_count(x: float) -> int:
    return x.bit_count()  # type: ignore[attr-defined, no-any-return]


def _bool_bit_count(b: bool) -> int:
    return b.bit_count()


def _bit_count_with_argument(x: int) -> int:
    return x.bit_count(3)  # type: ignore[call-arg]


def _bit_count_not_called(x: int) -> int:
    return x.bit_count  # type: ignore[return-value]


def test_attribute_rejections() -> None:
    for fn, match in [
        (_sequence_shape, r"`.shape` on a Python sequence is not supported; build a numpy array"),
        (_sequence_append, "a sequence has no supported attribute 'append'"),
        (_scalar_attribute, "a scalar has no supported attribute 'imag'"),
        (_array_unknown_attribute, "an array has no supported attribute 'strides'"),
        (_float_bit_count, "a scalar has no supported attribute 'bit_count'"),
        (_bool_bit_count, r"\.bit_count\(\) takes int operands, got bool"),
        (_bit_count_with_argument, r"\.bit_count\(\) takes 0 argument\(s\), got 1"),
        (_bit_count_not_called, "the returned value is not a int scalar"),
    ]:
        _rejects(fn, match)


# ---------------------------------------------------------------------- the bans


def _truthy_list(x: float) -> float:
    values = [x]
    if values:
        x = x + 1.0
    return x


def _not_tuple(x: float) -> bool:
    return not (x, x)


def _aggregate_equality(x: float) -> bool:
    return [x] == [x]


def _aggregate_ordering(x: float) -> bool:
    return (x,) < (x, x)


def _tensor_scalar_compare(x: float) -> bool:
    return bool(np.zeros(2) == x)  # numpy would build a mask; the subset bans it


def test_truthiness_and_comparison_bans() -> None:
    for fn, match in [
        (_truthy_list, "the truthiness of an aggregate is not supported"),
        (_not_tuple, "the truthiness of an aggregate is not supported"),
        (_aggregate_equality, "aggregate comparison is not supported"),
        (_aggregate_ordering, "aggregate comparison is not supported"),
        (_tensor_scalar_compare, "aggregate comparison is not supported"),
    ]:
        _rejects(fn, match)


# ---------------------------------------------------------------------- elementwise arithmetic and kind mixing


def _shape_mismatch(x: float) -> float:
    return (np.zeros(2) + np.zeros(3))[0] * x  # type: ignore[no-any-return]


def _vector_matrix_mismatch(x: float) -> float:
    return (np.array([x, x]) + _ARRAY)[0][0]  # type: ignore[no-any-return]


def _tensor_pow(x: float) -> float:
    return (np.full(2, 3.0) ** 2)[0] * x  # type: ignore[no-any-return]


def _tensor_list_mix(x: float) -> float:
    return (np.zeros(2) + [x, x])[0]  # type: ignore[no-any-return]


def _sequence_sub(x: float) -> float:
    return ((x, x) - (x, x))[0]  # type: ignore[operator, no-any-return]


def _list_of_array_demotes(v: Float64[np.ndarray, "2"], s: float) -> float:
    # list(arr) produces a Python sequence (as in Python), so arithmetic on the result is rejected even though the
    # argument was an array -- guards against the builtin accidentally keeping array semantics.
    return (list(v) * s)[0]  # type: ignore[operator, no-any-return]


def _bool_tensor_arithmetic(x: float) -> float:
    return (np.array([True, False]) + np.array([True, True]))[0] * x  # type: ignore[no-any-return]


def test_elementwise_rejections() -> None:
    for fn, match in [
        (_shape_mismatch, r"array shapes \(2,\) and \(3,\) do not match"),
        (_vector_matrix_mismatch, r"array shapes \(2,\) and \(2, 2\) do not match"),
        (_tensor_pow, r"the operator `\*\*` is not supported on arrays yet"),
        (_tensor_list_mix, "cannot mix an array with a Python list/tuple"),
        (_sequence_sub, "the operator `-` is not supported on a sequence"),
        (_list_of_array_demotes, "not supported on a sequence"),
        (_bool_tensor_arithmetic, "an array must hold numbers, not booleans"),
    ]:
        _rejects(fn, match)


# ---------------------------------------------------------------------- shaped parameters and returns


def _matrix_param(m: Float64[np.ndarray, "2 2"]) -> float:
    return m[0][0] + m[1][1]  # type: ignore[no-any-return]


def test_matrix_parameter_decomposes_row_major() -> None:
    result = holoso.synthesize(_matrix_param, _MIN_OPTIONS, name="kernel")
    assert [p.name for p in result.input_ports] == ["in_m_0_0", "in_m_0_1", "in_m_1_0", "in_m_1_1"]
    _oracle(_matrix_param, [{"m_0_0": 1.0, "m_0_1": 2.0, "m_1_0": 3.0, "m_1_1": 4.0}])


def _colliding_params(v: Float64[np.ndarray, "2"], v_0: float) -> float:
    return v[0] - v_0  # type: ignore[no-any-return]


def _shapeless_param(v: np.ndarray) -> float:
    return float(v[0])


def _int_array_param(v: Float64[np.ndarray, "0"]) -> float:  # type: ignore[unused-ignore]
    return float(v[0])


def test_parameter_rejections() -> None:
    _rejects(_colliding_params, "the decomposed parameter names collide on 'v_0'")
    _rejects(_shapeless_param, "the annotation of parameter 'v' is not supported yet")
    _rejects(_int_array_param, "array dimensions must be at least 1")


def _tuple_of_tensor(x: float) -> tuple[Float64[np.ndarray, "2"], float]:
    return np.array([x, 2.0]), x + 1.0


def test_aggregate_returns_flatten_row_major() -> None:
    result = holoso.synthesize(_tuple_of_tensor, _MIN_OPTIONS, name="kernel")
    assert [p.name for p in result.output_ports] == ["out_0_0", "out_0_1", "out_1"]
    assert [float(v) for v in result.numerical_model.elaborate().run(3.0)] == [3.0, 2.0, 4.0]


def _return_shape_mismatch(x: float) -> Float64[np.ndarray, "3"]:
    return np.array([x, x])


def _return_kind_mismatch(x: float) -> list[float]:
    return (x, x)  # type: ignore[return-value]


def _return_leaf_mismatch(x: float) -> tuple[bool, bool]:
    return (x, x)  # type: ignore[return-value]


def _return_nan_leaf(x: float) -> tuple[float, float]:
    return x, _WITH_NAN[1]


def test_return_conformance_rejections() -> None:
    for fn, match in [
        (_return_shape_mismatch, r"has shape \(2,\) where the annotation declares \(3,\)"),
        (_return_kind_mismatch, "list annotations are not supported; annotate a tuple"),
        (_return_leaf_mismatch, "type float where the annotation declares bool"),
        (_return_nan_leaf, "is NaN, which the compiler cannot represent"),
    ]:
        _rejects(fn, match)


def _dead_nan_element(x: float) -> float:
    return _WITH_NAN[0] * x


def test_a_dead_nan_element_does_not_reject() -> None:
    _oracle(_dead_nan_element, [{"x": 2.0}])


# ---------------------------------------------------------------------- instance attribute reads


class _Config:
    __slots__ = ("slotted",)

    def __init__(self) -> None:
        self.slotted = 1.0


class _Base:
    gain = 2.0
    deep = False

    @property
    def doubled(self) -> float:
        return self.gain * 2.0

    @staticmethod
    def offset() -> float:
        return 0.5

    @classmethod
    def is_deep(cls) -> bool:
        return cls.deep

    def method(self) -> float:
        return 1.0


class _Derived(_Base):
    deep = True


_DERIVED = _Derived()
_SLOTTED = _Config()
_NAMESPACE = types.SimpleNamespace(value=3.0, table=(1.0, 2.0))


def _instance_reads(x: float) -> float:
    return _NAMESPACE.value * x + _DERIVED.gain + _DERIVED.doubled + _DERIVED.offset() + _NAMESPACE.table[1]  # type: ignore[no-any-return]


def test_instance_reads_resolve_through_the_mro_without_live_calls() -> None:
    _oracle(_instance_reads, [{"x": 2.0}])


def _reads_slot_descriptor(x: float) -> float:
    return _SLOTTED.slotted * x


def _calls_instance_method(x: float) -> float:
    return _DERIVED.method() * x


def _calls_classmethod(x: float) -> float:
    # The classmethod reads ``cls.deep``, overridden in ``_Derived``, so a wrong receiver folds the other arm.
    if _DERIVED.is_deep():
        return x + 8.0
    return x - 5.0


def _missing_attribute(x: float) -> float:
    return _NAMESPACE.absent * x  # type: ignore[no-any-return]


def test_instance_and_class_methods_inline_like_helpers() -> None:
    _oracle(_calls_instance_method, [{"x": 2.5}])
    _oracle(_calls_classmethod, [{"x": 1.0}])


def test_instance_read_rejections() -> None:
    for fn, match in [
        (_reads_slot_descriptor, "_Config defines __slots__, so 'slotted' has no instance __dict__ entry to read"),
        (_missing_attribute, "'_NAMESPACE' has no attribute 'absent'"),
    ]:
        _rejects(fn, match)


# ---------------------------------------------------------------------- kinds are provenance


def _rectangular_list_is_still_a_sequence(x: float) -> float:
    grid = [[x, 1.0], [2.0, 3.0]]
    return grid.ndim * x  # type: ignore[attr-defined, no-any-return]


def test_a_rectangular_homogeneous_list_is_a_sequence_not_an_array() -> None:
    _rejects(_rectangular_list_is_still_a_sequence, r"`.ndim` on a Python sequence is not supported; build a numpy")


def _sequence_flatten(x: float) -> float:
    return [1.0, 2.0].flatten()[0] * x  # type: ignore[attr-defined, no-any-return]


def test_a_registered_ndarray_method_on_a_sequence_is_rejected() -> None:
    # `.ndim`/`.shape` above hit the hand-listed arm; a method registered on ndarray hits the resolve() arm.
    _rejects(_sequence_flatten, r"`.flatten` on a Python sequence is not supported; build a numpy")


def _ragged_chained(a: float, b: float) -> float:
    # Chained m[i][j] on a (even ragged) list stays valid plain list indexing, where multi-axis m[i, j] rejects.
    m = [[a, b], [a]]
    return m[0][1]


def test_chained_indexing_on_a_ragged_list_stays_valid() -> None:
    _oracle(_ragged_chained, [{"a": 1.5, "b": -2.0}])


def _branch_kind_mismatch(c: bool, a: float, b: float) -> float:
    # Only bool, int, and float values join branches, so a value that is an array in one arm and a sequence in the
    # other cannot merge at all.
    if c:
        v = np.array([a, b])
    else:
        v = [a, b]  # type: ignore[assignment]
    return v[0]  # type: ignore[no-any-return]


def test_a_branch_kind_mismatch_cannot_join() -> None:
    _rejects(_branch_kind_mismatch, "aggregates join only when every arm agrees")


# ---------------------------------------------------------------------- review round-1 regression pins


_INT8_TABLE = np.array([120, 7], dtype=np.int8)
_NP_NAN = np.float64("nan")


def _int_dtype_array(x: float) -> float:
    return float(_INT8_TABLE[0]) * x


def test_an_integer_array_capture_folds_width_less() -> None:
    _oracle(_int_dtype_array, [{"x": 2.0}])


def _np_nan_element_alive(x: float) -> float:
    return np.array([_NP_NAN, 2.0])[0] * x  # type: ignore[no-any-return]


def _np_nan_element_dead(x: float) -> float:
    return np.array([_NP_NAN, 2.0])[1] * x  # type: ignore[no-any-return]


def test_a_numpy_nan_element_is_judged_at_its_use() -> None:
    _rejects(_np_nan_element_alive, "is NaN, which the compiler cannot represent")
    _oracle(_np_nan_element_dead, [{"x": 1.5}])


_ROW_INDEX = (1, 0)
_FANCY = [1, 0]


def _named_tuple_index(x: float) -> float:
    return _ARRAY[_ROW_INDEX] * x  # type: ignore[no-any-return]


def _fancy_literal(x: float) -> float:
    return float(_ARRAY[[1, 0]].ndim) * x


def _fancy_named(x: float) -> float:
    return float(_ARRAY[_FANCY].ndim) * x


def test_sequence_indices_on_arrays_reject() -> None:
    # A list index means advanced indexing on the host.
    for fn in (_named_tuple_index, _fancy_literal, _fancy_named):
        _rejects(fn, "a sequence index on an array is not supported")


def _list_container_annotation(v: Float64[list, "2"]) -> float:  # type: ignore[type-arg]
    return v[0]  # type: ignore[no-any-return]


def test_a_non_ndarray_shaped_annotation_rejects() -> None:
    # A jaxtyping list container keeps Python sequence semantics; decomposing it as an array would make
    # `v * 2` mean elementwise where the host repeats.
    _rejects(_list_container_annotation, "only numpy array containers are supported")


def _doubling_splats(x: float) -> float:
    t0 = (x, x)
    t1 = (*t0, *t0)
    t2 = (*t1, *t1)
    t3 = (*t2, *t2)
    t4 = (*t3, *t3)
    t5 = (*t4, *t4)
    t6 = (*t5, *t5)
    t7 = (*t6, *t6)
    t8 = (*t7, *t7)
    t9 = (*t8, *t8)
    t10 = (*t9, *t9)
    t11 = (*t10, *t10)
    t12 = (*t11, *t11)
    t13 = (*t12, *t12)
    t14 = (*t13, *t13)
    t15 = (*t14, *t14)
    t16 = (*t15, *t15)
    t17 = (*t16, *t16)
    return t17[0]


def test_splat_doubling_exhausts_the_budget_instead_of_hanging() -> None:
    _rejects(_doubling_splats, "graph expansion budget is exhausted")


# ---------------------------------------------------------------------- dtype widths are not modeled


_NP_INT = np.int8(120)
_F32 = np.array([0.5, 0.25], dtype=np.float32)
_F32_SCALAR = np.float32(1.5)
_HUGE = 18446744073709551616


def _np_int_scalar(x: int) -> int:
    return int(_NP_INT + x)


def _narrow_float_reads(x: float) -> float:
    return float(_F32[0]) * x + float(_F32_SCALAR)


def _narrow_param(v: Float32[np.ndarray, "2"]) -> float:
    return float(v[0] + v[1])


def _int_param(v: Int64[np.ndarray, "2"]) -> int:
    return v[0] + v[1]  # type: ignore[no-any-return]


def _int_filled_factory(x: float) -> float:
    return np.full(2, 1)[0] + x  # type: ignore[no-any-return]


def _huge_int_construction(x: float) -> float:
    a = np.array([_HUGE, x])
    return float(a[0])


def test_dtype_widths_fold_into_the_width_less_model() -> None:
    _oracle(_np_int_scalar, [{"x": 3}])
    _oracle(_narrow_float_reads, [{"x": 2.0}])
    _oracle(_narrow_param, [{"v_0": 1.5, "v_1": 2.5}])
    _oracle(_int_param, [{"v_0": 3, "v_1": 4}])
    _oracle(_int_filled_factory, [{"x": 2.5}])
    _oracle(_huge_int_construction, [{"x": 1.0}])


def _factory_budget(x: float) -> float:
    big = np.zeros(200_000)
    return big[0] + x  # type: ignore[no-any-return]


def test_factories_charge_the_budget() -> None:
    _rejects(_factory_budget, "graph expansion budget is exhausted while expanding the array factory")


class _StoresABoundMethod:
    def __init__(self) -> None:
        self.slot = 0

    def __call__(self, x: int) -> int:
        self.slot = x.bit_count  # type: ignore[assignment]
        return 0


def _stores_a_bound_method_into_an_array(x: int) -> float:
    v = np.asarray((1.0, 2.0))
    v[0] = x.bit_count
    return float(v[0])


def test_a_bound_method_is_named_by_its_receiver_wherever_it_is_refused() -> None:
    """Both store sites report the kind; a scalar receiver must never be called an array method."""
    _rejects(_StoresABoundMethod().__call__, "a bound scalar method cannot be stored")
    _rejects(_stores_a_bound_method_into_an_array, "a bound scalar method cannot be stored")
