"""
The structural fact domain: layouts, canonical leaf order, normalize/materialize round trips, and layout joins.
White-box by necessity -- this is the pure value library under the analyzer; behavioral coverage rides the kernel
suites once the spine consumes it.
"""

import dataclasses

import numpy as np
import pytest

from holoso._frontend._fir._fact import (
    ArrayDType,
    ArrayIndex,
    ArrayLayout,
    AggregateFact,
    ContainerFlavor,
    Known,
    LayoutMismatch,
    ListIndex,
    ListLayout,
    RecordField,
    RecordLayout,
    Residual,
    StructuralLayout,
    TupleIndex,
    TupleLayout,
    aggregate_of,
    join_layouts,
    leaf_count,
    leaf_paths,
    materialize_static,
    normalize_static,
)
from holoso._frontend._fir._value import SemType, StaticSeq, admit


def _normalized(obj: object) -> AggregateFact:
    admitted = admit(obj)
    assert admitted is not None
    fact = normalize_static(admitted)
    assert isinstance(fact, AggregateFact)
    return fact


def test_normalize_materialize_round_trips_exactly() -> None:
    @dataclasses.dataclass(frozen=True)
    class Gains:
        kp: float
        taps: tuple[float, float]

    table = np.array([[1.0, 2.0], [3.0, 4.0]])
    table.setflags(write=False)
    for obj in ((1.0, (2, True)), [1.0, [2.0]], table, Gains(kp=0.5, taps=(1.0, 2.0))):
        fact = _normalized(obj)
        rebuilt = materialize_static(fact)
        assert rebuilt is not None
        assert normalize_static(rebuilt) == fact
        assert len(fact.leaves) == leaf_count(fact.layout) == len(leaf_paths(fact.layout))


def test_canonical_leaf_order_is_row_major_and_declaration_order() -> None:
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
    matrix.setflags(write=False)
    fact = _normalized(matrix)
    values = [leaf.value.value for leaf in fact.leaves if isinstance(leaf, Known)]  # type: ignore[union-attr]
    assert values == [1.0, 2.0, 3.0, 4.0]
    assert leaf_paths(fact.layout) == (
        (ArrayIndex((0, 0)),),
        (ArrayIndex((0, 1)),),
        (ArrayIndex((1, 0)),),
        (ArrayIndex((1, 1)),),
    )

    @dataclasses.dataclass(frozen=True)
    class Pair:
        second: float
        first: float

    record = _normalized(Pair(second=2.0, first=1.0))
    assert [segment for (segment,) in leaf_paths(record.layout)] == [RecordField("second"), RecordField("first")]


def test_nested_construction_flattens_in_element_order() -> None:
    inner = aggregate_of((Residual(SemType.FLOAT), Known(admit(2.0))), is_list=False)  # type: ignore[arg-type]
    outer = aggregate_of((Known(admit(1.0)), inner), is_list=True)  # type: ignore[arg-type]
    assert isinstance(outer.layout, ListLayout)
    assert leaf_count(outer.layout) == 3
    assert outer.child(1) == inner
    assert leaf_paths(outer.layout)[1] == (ListIndex(1), TupleIndex(0))


def test_structural_static_values_never_circulate_inside_known() -> None:
    fact = _normalized(((1.0, 2.0), 3.0))
    assert all(not isinstance(leaf.value, StaticSeq) for leaf in fact.leaves if isinstance(leaf, Known))


def test_flavor_mixed_join_degrades_to_structural_and_keeps_arity() -> None:
    as_tuple = _normalized((1.0, 2.0)).layout
    as_list = _normalized([1.0, 2.0]).layout
    joined = join_layouts(as_tuple, as_list)
    assert isinstance(joined, StructuralLayout)
    assert joined.flavors == frozenset({ContainerFlavor.TUPLE, ContainerFlavor.LIST})
    assert leaf_count(joined) == 2
    with pytest.raises(LayoutMismatch, match="arities"):
        join_layouts(as_tuple, _normalized((1.0, 2.0, 3.0)).layout)


def test_array_joins_promote_dtype_and_reject_shape_or_bool_mixes() -> None:
    ints = ArrayLayout((2,), ArrayDType.INT64)
    floats = ArrayLayout((2,), ArrayDType.FLOAT64)
    bools = ArrayLayout((2,), ArrayDType.BOOL)
    assert join_layouts(ints, floats) == floats
    assert join_layouts(bools, bools) == bools
    with pytest.raises(LayoutMismatch, match="boolean"):
        join_layouts(bools, floats)
    with pytest.raises(LayoutMismatch, match="shapes"):
        join_layouts(floats, ArrayLayout((3,), ArrayDType.FLOAT64))
    vector_join = join_layouts(floats, TupleLayout((None, None)))
    assert isinstance(vector_join, StructuralLayout) and ContainerFlavor.ARRAY in vector_join.flavors


def test_record_joins_require_class_identity() -> None:
    @dataclasses.dataclass(frozen=True)
    class A:
        x: float

    @dataclasses.dataclass(frozen=True)
    class B:
        x: float

    layout_a = _normalized(A(x=1.0)).layout
    layout_b = _normalized(B(x=1.0)).layout
    assert join_layouts(layout_a, layout_a) == layout_a
    with pytest.raises(LayoutMismatch, match="classes"):
        join_layouts(layout_a, layout_b)


def test_structural_layout_never_materializes() -> None:
    joined = join_layouts(_normalized((1.0,)).layout, _normalized([1.0]).layout)
    assert isinstance(joined, StructuralLayout)
    fact = AggregateFact(joined, (Known(admit(1.0)),))  # type: ignore[arg-type]
    assert materialize_static(fact) is None  # even fully Known: the runtime container flavor is path-dependent


def test_scalar_and_aggregate_never_join() -> None:
    with pytest.raises(LayoutMismatch, match="scalar"):
        join_layouts(None, _normalized((1.0,)).layout)
