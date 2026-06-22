"""Compile-time lowering values: scalar wires and ordered aggregates, plus the persistent-attribute shape."""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass

from .._hir import Const
from .._util import ValueId
from ._ast_support import _Path


class _Value(ABC):
    """
    A compile-time lowering value: a single scalar HIR wire or an ordered aggregate of values. Aggregates (vectors,
    matrices, tuples) are pure frontend bookkeeping over scalar registers -- per DESIGN.md they never exist as hardware
    aggregates -- so they never enter HIR; only their scalar leaves do.
    """

    @abstractmethod
    def walk(self, path: _Path) -> Iterator[tuple[_Path, ValueId]]:
        """Yield ``(path, scalar)`` leaves row-major, extending ``path`` by the aggregate index at each level."""

    def leaves(self) -> list[ValueId]:
        return [vid for _, vid in self.walk([])]

    def flatten(self) -> "_Aggregate":
        """Collapse to a flat aggregate of all scalar leaves in row-major order (the ``.flatten()`` method)."""
        return _Aggregate(tuple(_Scalar(vid) for vid in self.leaves()))

    def output_leaves(self) -> list[tuple[_Path, ValueId]]:
        """The (path, scalar) pairs naming this returned value's output ports; an aggregate uses its indexed paths."""
        return list(self.walk([]))


@dataclass(frozen=True, slots=True)
class _Scalar(_Value):
    id: ValueId

    def walk(self, path: _Path) -> Iterator[tuple[_Path, ValueId]]:
        yield list(path), self.id

    def output_leaves(self) -> list[tuple[_Path, ValueId]]:
        # A bare scalar return is out_0 (leaf position 0), not the empty-path "out", to match the multi-output and
        # reference orderings; walking a lone scalar would otherwise yield the empty path.
        return [([0], self.id)]


@dataclass(frozen=True, slots=True)
class _Aggregate(_Value):
    items: tuple[_Value, ...]

    def walk(self, path: _Path) -> Iterator[tuple[_Path, ValueId]]:
        for index, item in enumerate(self.items):
            yield from item.walk([*path, index])


@dataclass(frozen=True, slots=True)
class _StateAttr:
    """
    The scalar-slot decomposition of one instance attribute, derived from the reset snapshot: a scalar occupies a single
    bare-named slot, a vector one indexed slot per element. It is the single source of an attribute's shape -- its slot
    names, its typed reset values, and whether an assigned value must be a scalar or a same-length flat aggregate. The
    element type lives in the typed ``resets`` (a :class:`BoolConst` reset marks a boolean attribute, a scalar only since
    boolean vectors are not supported), so no separate type flag is carried.
    """

    is_vector: bool
    slots: list[str]
    resets: list[Const]

    def accepts(self, value: _Value) -> bool:
        """
        Whether an assigned value matches this shape: a scalar attribute accepts only a scalar, a vector only a flat
        aggregate of the same length. Checking the full shape -- not merely the leaf count -- keeps the assigned value
        consistent with the per-element slot layout that the next transaction reconstructs from the reset snapshot.
        """
        if not self.is_vector:
            return isinstance(value, _Scalar)
        return (
            isinstance(value, _Aggregate)
            and len(value.items) == len(self.slots)
            and all(isinstance(item, _Scalar) for item in value.items)
        )

    def compose(self, scalars: tuple[_Scalar, ...]) -> _Value:
        """A scalar attribute is its single wire; a vector attribute is the aggregate of its per-element wires."""
        return _Aggregate(scalars) if self.is_vector else scalars[0]
