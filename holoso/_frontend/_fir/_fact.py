"""
The analyzer's fact domain: scalar boundness states plus one canonical structural aggregate. An aggregate fact is
a typed recursive layout (which container flavors, shapes, dtypes, record identities) over one FLAT, canonically
ordered tuple of atomic leaf facts -- flat because per-leaf SSA, ABI naming, state decomposition, and return
flattening all consume exactly that order, and one walker over the layout defines it everywhere. A structural
static value (StaticSeq/StaticArray/StaticRecord) never circulates inside Known: it normalizes to AggregateFact at
every admission boundary and reconstructs concretely only when every leaf is Known.

A CFG join of two aggregates first reconciles the LAYOUTS (identical flavors recurse; a tuple arm meeting a list
arm of the same arity degrades to a StructuralLayout carrying the flavor set, keeping only flavor-independent
behavior; ndarray dtypes promote int -> float leafwise), then joins the leaves positionally under the one
result layout. Persistent state deliberately does NOT use this relation: a slot's reset container flavor and
geometry are load-bearing (the next transaction reconstructs that Python object), so state stores demand identical
layout instead of a structural degrade.

Canonical leaf order: tuple/list/structural element order; ndarray C row-major; record fields declaration-order.
"""

import enum
import math
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from ._value import (
    MetaInt,
    NpFloat,
    NpInt,
    SemType,
    StaticArray,
    NpBool,
    StaticBool,
    StaticFloat,
    StaticRecord,
    StaticSeq,
    StaticValue,
    admit,
)


class LayoutMismatch(ValueError):
    """Two layouts that cannot merge (or a store into a slot of a different geometry); the caller locates it."""


class ArrayDType(enum.Enum):
    BOOL = enum.auto()
    INT = enum.auto()
    FLOAT = enum.auto()


class ContainerFlavor(enum.Enum):
    TUPLE = enum.auto()
    LIST = enum.auto()
    ARRAY = enum.auto()


@dataclass(frozen=True, slots=True)
class TupleLayout:
    items: tuple["ValueLayout", ...]


@dataclass(frozen=True, slots=True)
class ListLayout:
    items: tuple["ValueLayout", ...]


@dataclass(frozen=True, slots=True)
class ArrayLayout:
    shape: tuple[int, ...]
    dtype: ArrayDType


@dataclass(frozen=True, slots=True, eq=False)
class RecordLayout:
    """Record identity is the CLASS identity: a metaclass-defined __eq__ must never conflate distinct classes."""

    klass: type
    fields: tuple[tuple[str, "ValueLayout"], ...]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RecordLayout):
            return NotImplemented
        return self.klass is other.klass and self.fields == other.fields

    def __hash__(self) -> int:
        return hash((id(self.klass), self.fields))


@dataclass(frozen=True, slots=True)
class StructuralLayout:
    """
    A CFG-join result whose runtime container flavor is path-dependent. Only flavor-independent structural behavior
    (length, fixed indexing, iteration, unpacking) remains available; flavor-specific semantics (array arithmetic,
    list concatenation, shape queries) reject rather than guess.
    """

    flavors: frozenset[ContainerFlavor]
    items: tuple["ValueLayout", ...]


ATOM = None  # a scalar leaf position marks itself with None: the layout tree's only non-aggregate node

type AggregateLayout = TupleLayout | ListLayout | ArrayLayout | RecordLayout | StructuralLayout
type ValueLayout = None | AggregateLayout


@dataclass(frozen=True, slots=True)
class TupleIndex:
    value: int


@dataclass(frozen=True, slots=True)
class ListIndex:
    value: int


@dataclass(frozen=True, slots=True)
class ArrayIndex:
    coordinates: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RecordField:
    name: str


@dataclass(frozen=True, slots=True)
class StructuralIndex:
    value: int


type PathSegment = TupleIndex | ListIndex | ArrayIndex | RecordField | StructuralIndex
type LeafPath = tuple[PathSegment, ...]


def _array_coordinates(shape: tuple[int, ...]) -> Iterator[tuple[int, ...]]:
    yield from (tuple(map(int, coordinates)) for coordinates in np.ndindex(*shape))


def leaf_paths(layout: ValueLayout) -> tuple[LeafPath, ...]:
    """Every scalar leaf's typed path, in the canonical order the flat leaf tuple follows."""
    if layout is ATOM:
        return ((),)
    paths: list[LeafPath] = []
    match layout:
        case TupleLayout(items=items):
            for index, item in enumerate(items):
                paths += [(TupleIndex(index), *tail) for tail in leaf_paths(item)]
        case ListLayout(items=items):
            for index, item in enumerate(items):
                paths += [(ListIndex(index), *tail) for tail in leaf_paths(item)]
        case StructuralLayout(items=items):
            for index, item in enumerate(items):
                paths += [(StructuralIndex(index), *tail) for tail in leaf_paths(item)]
        case ArrayLayout(shape=shape):
            paths += [(ArrayIndex(coordinates),) for coordinates in _array_coordinates(shape)]
        case RecordLayout(fields=fields):
            for name, item in fields:
                paths += [(RecordField(name), *tail) for tail in leaf_paths(item)]
    return tuple(paths)


def leaf_count(layout: ValueLayout) -> int:
    if layout is ATOM:
        return 1
    match layout:
        case TupleLayout(items=items) | ListLayout(items=items) | StructuralLayout(items=items):
            return sum(leaf_count(item) for item in items)
        case ArrayLayout(shape=shape):
            return math.prod(shape)  # exact for arbitrary Python ints (np.prod would wrap at 64 bits)
        case RecordLayout(fields=fields):
            return sum(leaf_count(item) for _, item in fields)
    raise AssertionError(layout)


def child_layouts(layout: AggregateLayout) -> tuple[ValueLayout, ...]:
    """The direct children in canonical order: a 2-D array's children are its rows (one structural axis at a time)."""
    match layout:
        case TupleLayout(items=items) | ListLayout(items=items) | StructuralLayout(items=items):
            return items
        case ArrayLayout(shape=shape, dtype=dtype):
            if len(shape) == 1:
                return (ATOM,) * shape[0]
            return (ArrayLayout(shape[1:], dtype),) * shape[0]
        case RecordLayout(fields=fields):
            return tuple(item for _, item in fields)
    raise AssertionError(layout)


def outer_arity(layout: AggregateLayout) -> int:
    return len(child_layouts(layout))


def child_slice(layout: AggregateLayout, index: int) -> tuple[ValueLayout, int, int]:
    """The child layout at ``index`` plus its [start, stop) window into the parent's flat leaf tuple."""
    children = child_layouts(layout)
    assert 0 <= index < len(children)
    start = sum(leaf_count(child) for child in children[:index])
    return children[index], start, start + leaf_count(children[index])


# ---------------------------------------- facts ----------------------------------------


@dataclass(frozen=True, slots=True)
class Unbound:
    pass


@dataclass(frozen=True, slots=True)
class Known:
    value: StaticValue


@dataclass(frozen=True, slots=True)
class Residual:
    type: SemType


@dataclass(frozen=True, slots=True, eq=False)
class Reference:
    """
    An identity-keyed reference fact: a callable, module, class, stateful component, or any other object outside
    the value domain. References are a separate SORT from values -- never data, never foldable, no generic escape
    back into Python -- so every place a reference may act (a call target, a namespace lookup, a state receiver,
    an isinstance classinfo, an inert dtype argument) is an explicit arm, and everything else refuses by type.
    Equality and hash key on the REFERENT's identity: value-based forms would call the referent's own ``==`` (an
    ndarray poisons enclosing comparisons) and are partial under hashing (an unhashable referent raises).
    """

    obj: object

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Reference):
            return NotImplemented
        return self.obj is other.obj

    def __hash__(self) -> int:
        return hash(id(self.obj))


type AtomicFact = Known | Residual | Reference


@dataclass(frozen=True, slots=True)
class AggregateFact:
    layout: AggregateLayout
    leaves: tuple[AtomicFact, ...]

    def __post_init__(self) -> None:
        assert len(self.leaves) == leaf_count(self.layout)
        assert not any(isinstance(leaf, Known) and isinstance(leaf.value, _STRUCTURAL) for leaf in self.leaves)

    def child(self, index: int) -> "BoundFact":
        layout, start, stop = child_slice(self.layout, index)
        if layout is ATOM:
            return self.leaves[start]
        return AggregateFact(layout, self.leaves[start:stop])


type BoundFact = Known | Residual | Reference | AggregateFact


@dataclass(frozen=True, slots=True)
class MaybeUnbound:
    """Joined bound-and-unbound, always at the root: reading it is a located rejection (Python may raise)."""

    inner: BoundFact


type Fact = Unbound | BoundFact | MaybeUnbound

_STRUCTURAL = (StaticSeq, StaticArray, StaticRecord)


def _array_dtype(array: np.ndarray) -> ArrayDType:
    if array.dtype == np.bool_:
        return ArrayDType.BOOL
    if np.issubdtype(array.dtype, np.integer):
        return ArrayDType.INT
    assert np.issubdtype(array.dtype, np.floating)
    return ArrayDType.FLOAT


def normalize_static(value: StaticValue) -> BoundFact:
    """A static value as a fact: structural values flatten to AggregateFact, scalars stay Known."""
    match value:
        case StaticSeq(items=items, is_list=is_list):
            children = tuple(normalize_static(item) for item in items)
            layout: AggregateLayout = (ListLayout if is_list else TupleLayout)(
                tuple(_layout_of(child) for child in children)
            )
            return AggregateFact(layout, tuple(_flat_leaves(children)))
        case StaticArray(array=array):
            leaves = []
            for element in array.flatten():
                admitted = admit(element)
                assert admitted is not None and not isinstance(admitted, _STRUCTURAL)
                leaves.append(Known(admitted))
            return AggregateFact(ArrayLayout(tuple(map(int, array.shape)), _array_dtype(array)), tuple(leaves))
        case StaticRecord(klass=klass, field_values=field_values):
            children = tuple(normalize_static(item) for _, item in field_values)
            fields = tuple((name, _layout_of(child)) for (name, _), child in zip(field_values, children))
            return AggregateFact(RecordLayout(klass, fields), tuple(_flat_leaves(children)))
        case _:
            return Known(value)


def _layout_of(fact: BoundFact) -> ValueLayout:
    return fact.layout if isinstance(fact, AggregateFact) else ATOM


def _flat_leaves(children: tuple[BoundFact, ...]) -> Iterator[AtomicFact]:
    for child in children:
        if isinstance(child, AggregateFact):
            yield from child.leaves
        else:
            yield child


def aggregate_of(children: tuple[BoundFact, ...], is_list: bool) -> AggregateFact:
    """A tuple/list construction over already-normalized child facts."""
    layout = (ListLayout if is_list else TupleLayout)(tuple(_layout_of(child) for child in children))
    return AggregateFact(layout, tuple(_flat_leaves(children)))


def record_of(klass: type, children: tuple[tuple[str, BoundFact], ...]) -> AggregateFact:
    """A record construction over already-normalized child facts, one per field in declaration order."""
    layout = RecordLayout(klass, tuple((name, _layout_of(child)) for name, child in children))
    return AggregateFact(layout, tuple(_flat_leaves(tuple(child for _, child in children))))


def numpy_kinded(leaf: Known, dtype: ArrayDType) -> Known:
    """
    A Known scalar re-kinded onto an array dtype's numpy scalar kind, exactly as numpy element extraction
    would yield it (np.float64/np.int64/np.bool_). The caller guarantees representability: integers fit the
    signed 64-bit range, and the bool kind receives only booleans -- so the conversion is total.
    """
    concrete = _concrete_scalar(leaf)
    match dtype:
        case ArrayDType.FLOAT:
            converted: object = np.float64(concrete)  # type: ignore[arg-type]  # a bool casts to 0.0/1.0
        case ArrayDType.INT:
            assert isinstance(concrete, (int, np.integer)) and not isinstance(concrete, bool)
            converted = np.int64(concrete)
        case ArrayDType.BOOL:
            assert isinstance(concrete, (bool, np.bool_))
            converted = np.bool_(concrete)
    admitted = admit(converted)
    assert admitted is not None and not isinstance(admitted, _STRUCTURAL)
    return Known(admitted)


def materialize_static(fact: BoundFact) -> StaticValue | None:
    """
    The concrete static value of a fully-Known fact, or None when any leaf is residual or the container flavor is
    path-dependent (a StructuralLayout cannot truthfully pick a concrete container). The exact-layout round trip
    ``normalize_static(materialize_static(f)) == f`` holds whenever the result is not None.
    """
    match fact:
        case Known(value=value):
            return value
        case Residual():
            return None
        case AggregateFact(layout=layout, leaves=leaves):
            if not all(isinstance(leaf, Known) for leaf in leaves):
                return None
            match layout:
                case TupleLayout() | ListLayout():
                    children = [materialize_static(fact.child(i)) for i in range(outer_arity(layout))]
                    if any(child is None for child in children):
                        return None
                    return StaticSeq(
                        tuple(child for child in children if child is not None),
                        is_list=isinstance(layout, ListLayout),
                    )
                case ArrayLayout(shape=shape, dtype=dtype):
                    numpy_dtype = {
                        ArrayDType.BOOL: np.bool_,
                        ArrayDType.INT: np.int64,
                        ArrayDType.FLOAT: np.float64,
                    }[dtype]
                    values = [_concrete_scalar(leaf) for leaf in leaves]
                    array: np.ndarray = np.array(values, dtype=numpy_dtype).reshape(shape)
                    array.setflags(write=False)
                    return StaticArray(array)
                case RecordLayout(klass=klass, fields=fields):
                    children = [materialize_static(fact.child(i)) for i in range(outer_arity(layout))]
                    if any(child is None for child in children):
                        return None
                    named = tuple((name, child) for (name, _), child in zip(fields, children) if child is not None)
                    return StaticRecord(klass, named)
                case StructuralLayout():
                    return None
    raise AssertionError(fact)


def _concrete_scalar(leaf: AtomicFact) -> object:
    assert isinstance(leaf, Known)
    match leaf.value:
        case StaticBool(value=value) | NpBool(value=value):
            return value
        case MetaInt(value=value) | NpInt(value=value):
            return value
        case StaticFloat(value=value) | NpFloat(value=value):
            return value
    raise AssertionError(leaf.value)


# ---------------------------------------- joins ----------------------------------------


def join_layouts(a: ValueLayout, b: ValueLayout) -> ValueLayout:
    """
    The common layout of a CFG join, or a LayoutMismatch. Identical flavors recurse; positional containers of
    unlike flavor but equal arity degrade to a StructuralLayout (their union of flavors); ndarray dtypes promote
    int64 with float64 leafwise; records require the same class identity; everything else is a mismatch.
    """
    if a == b:
        return a
    if a is ATOM or b is ATOM:
        raise LayoutMismatch("a scalar merges with an aggregate here")
    match a, b:
        case (ArrayLayout(shape=sa, dtype=da), ArrayLayout(shape=sb, dtype=db)):
            if sa != sb:
                raise LayoutMismatch(f"arrays of shapes {sa} and {sb} merge here")
            if ArrayDType.BOOL in (da, db):
                raise LayoutMismatch("a boolean array merges with a numeric array here")
            return ArrayLayout(sa, ArrayDType.FLOAT if da is not db else da)
        case (RecordLayout() as ra, RecordLayout() as rb):
            if ra.klass is not rb.klass:
                raise LayoutMismatch(
                    f"records of different classes ({ra.klass.__name__} and {rb.klass.__name__}) merge here"
                )
            if tuple(name for name, _ in ra.fields) != tuple(name for name, _ in rb.fields):
                raise LayoutMismatch(f"records of class {ra.klass.__name__} with different fields merge here")
            fields = tuple((name, join_layouts(la, lb)) for (name, la), (_, lb) in zip(ra.fields, rb.fields))
            return RecordLayout(ra.klass, fields)
        case (
            (TupleLayout() | ListLayout() | StructuralLayout()) as pa,
            (TupleLayout() | ListLayout() | StructuralLayout()) as pb,
        ):
            items_a, items_b = child_layouts(pa), child_layouts(pb)
            if len(items_a) != len(items_b):
                raise LayoutMismatch(f"positional containers of arities {len(items_a)} and {len(items_b)} merge here")
            items = tuple(join_layouts(la, lb) for la, lb in zip(items_a, items_b))
            flavors = _flavors(pa) | _flavors(pb)
            if flavors == {ContainerFlavor.TUPLE}:
                return TupleLayout(items)
            if flavors == {ContainerFlavor.LIST}:
                return ListLayout(items)
            return StructuralLayout(frozenset(flavors), items)
        case ((ArrayLayout() as array), (TupleLayout() | ListLayout() | StructuralLayout()) as positional) | (
            (TupleLayout() | ListLayout() | StructuralLayout()) as positional,
            (ArrayLayout() as array),
        ):
            if len(array.shape) != 1 or array.dtype is ArrayDType.BOOL:
                raise LayoutMismatch("an array merges with a positional container here")
            items_b = child_layouts(positional)
            if array.shape[0] != len(items_b) or any(item is not ATOM for item in items_b):
                raise LayoutMismatch("an array merges with a positional container of a different shape here")
            return StructuralLayout(frozenset({ContainerFlavor.ARRAY}) | _flavors(positional), items_b)
    raise LayoutMismatch("values of irreconcilable shapes merge here")


def _flavors(layout: AggregateLayout) -> frozenset[ContainerFlavor]:
    match layout:
        case TupleLayout():
            return frozenset({ContainerFlavor.TUPLE})
        case ListLayout():
            return frozenset({ContainerFlavor.LIST})
        case ArrayLayout():
            return frozenset({ContainerFlavor.ARRAY})
        case StructuralLayout(flavors=flavors):
            return flavors
    raise AssertionError(layout)
