"""Compile-time lowering values: scalar wires and ordered aggregates, plus the persistent-attribute shape."""

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from math import prod

from .._hir import Const
from .._util import ValueId
from ._ast_support import Path


class Value(ABC):
    """
    A compile-time lowering value: a single scalar HIR wire or an ordered aggregate of values. Aggregates (vectors,
    matrices, tuples) are pure frontend bookkeeping over scalar registers -- per DESIGN.md they never exist as hardware
    aggregates -- so they never enter HIR; only their scalar leaves do.
    """

    @abstractmethod
    def walk(self, path: Path) -> Iterator[tuple[Path, ValueId]]:
        """Leaves are yielded row-major, ``path`` extended by the aggregate index at each level."""

    def leaves(self) -> list[ValueId]:
        return [vid for _, vid in self.walk([])]

    def flatten(self) -> "Aggregate":
        """The leaves flatten in row-major order; ``.flatten()`` is a numpy operation, so the result is an array."""
        return Aggregate(tuple(Scalar(vid) for vid in self.leaves()), array=True)

    def output_leaves(self) -> list[tuple[Path, ValueId]]:
        return list(self.walk([]))


@dataclass(frozen=True, slots=True)
class Scalar(Value):
    id: ValueId

    def walk(self, path: Path) -> Iterator[tuple[Path, ValueId]]:
        yield list(path), self.id

    def output_leaves(self) -> list[tuple[Path, ValueId]]:
        # A bare scalar return is out_0 (leaf position 0), not the empty-path "out", to match the multi-output and
        # reference orderings; walking a lone scalar would otherwise yield the empty path.
        return [([0], self.id)]


@dataclass(frozen=True, slots=True)
class Aggregate(Value):
    """
    An ordered aggregate of values with an explicit semantic marker: ``array`` is True for a numpy array (a shaped
    parameter, an ndarray constant/state, ``np.array(...)``, or a result of an array operation) and False for a plain
    Python list/tuple. The two spellings differ in Python -- list ``+`` concatenates, list ``-``/``@``/``.T`` are
    errors -- so the numpy-only operations (arithmetic, the matrix product, transpose, ``.flatten()``, multi-axis
    indexing) are rejected on a Python sequence, while the structural operations valid on both (indexing, slicing,
    unpacking, building, returning) apply regardless of the flag.
    """

    items: tuple[Value, ...]
    array: bool

    def walk(self, path: Path) -> Iterator[tuple[Path, ValueId]]:
        for index, item in enumerate(self.items):
            yield from item.walk([*path, index])


def array_shape(value: Value) -> tuple[int, ...] | None:
    """
    The numpy-style shape of a value when it forms a rectangular array: ``()`` for a scalar, ``(n,)`` for a flat
    aggregate, ``(m, n)`` for an aggregate of equal-shape aggregates, and so on. A ragged or empty aggregate is not an
    array and yields None; array-typed operations (matmul, transpose, shape validation) gate on this while purely
    structural ones (tuple returns, unpacking) do not.
    """
    match value:
        case Scalar():
            return ()
        case Aggregate(items=items):
            if not items:
                return None
            shapes = {array_shape(item) for item in items}
            if len(shapes) != 1 or (inner := next(iter(shapes))) is None:
                return None
            return (len(items), *inner)
    raise TypeError(f"unexpected value {value!r}")


def shape_name(shape: tuple[int, ...]) -> str:
    """A shape for a diagnostic: ``a scalar``, ``a 3-element vector``, ``a 2×3 matrix``, ``a 2×2×2 array``."""
    match shape:
        case ():
            return "a scalar"
        case (n,):
            return f"a {n}-element vector"
        case (_, _):
            return f"a {'×'.join(str(dim) for dim in shape)} matrix"
        case _:
            return f"a {'×'.join(str(dim) for dim in shape)} array"


def array_of(shape: tuple[int, ...], leaves: Sequence[Value], *, array: bool) -> Value:
    """Rebuild a value of the given shape from its row-major scalar leaves, stamping every level with ``array``."""
    if shape == ():
        (leaf,) = leaves
        return leaf
    assert leaves and len(leaves) % shape[0] == 0
    stride = len(leaves) // shape[0]
    return Aggregate(
        tuple(array_of(shape[1:], leaves[stride * i : stride * (i + 1)], array=array) for i in range(shape[0])),
        array=array,
    )


@dataclass(frozen=True, slots=True)
class StateAttr:
    """
    The scalar-slot decomposition of one instance attribute, derived from the reset snapshot: a scalar occupies a single
    bare-named slot, a vector or matrix one indexed slot per element in row-major order. It is the single source of an
    attribute's shape -- its slot names, its typed reset values, and the array shape an assigned value must have. The
    element type lives in the typed ``resets`` (a :class:`BoolConst` reset marks a boolean attribute, a scalar only
    since boolean aggregates are not supported), so no separate type flag is carried.
    """

    shape: tuple[int, ...]
    slots: list[str]
    resets: list[Const]
    array: bool  # True when the reset snapshot is a numpy array, False for a Python list/tuple (or a scalar)

    def __post_init__(self) -> None:
        count = prod(self.shape)  # prod(()) == 1: a scalar occupies exactly one slot
        assert len(self.slots) == count and len(self.resets) == count

    def accepts(self, value: Value) -> bool:
        """
        Whether an assigned value matches this shape exactly. Checking the full shape -- not merely the leaf count --
        keeps the assigned value consistent with the per-element slot layout that the next transaction reconstructs
        from the reset snapshot.
        """
        return array_shape(value) == self.shape

    def compose(self, scalars: tuple[Scalar, ...]) -> Value:
        return array_of(self.shape, scalars, array=self.array)
