"""
Shared auxiliary entities used across the IR layers.
"""

import operator
import re
from collections.abc import Callable, Mapping
from enum import Enum

type ValueId = int
"""An SSA value identifier, unique within one IR graph."""

type BlockId = int
"""A basic-block identifier, unique within one control-flow graph."""

VERILOG_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def reverse_postorder_of(entry: BlockId, successors: Mapping[BlockId, list[BlockId]]) -> list[BlockId]:
    """
    Predecessors before successors, a back-edge target before its body; iterative, since an unrolled loop chains
    thousands of blocks.
    """
    order: list[BlockId] = []
    visited = {entry}
    stack: list[tuple[BlockId, int]] = [(entry, 0)]
    while stack:
        bid, index = stack[-1]
        arms = successors[bid]
        if index < len(arms):
            stack[-1] = (bid, index + 1)
            if arms[index] not in visited:
                visited.add(arms[index])
                stack.append((arms[index], 0))
        else:
            stack.pop()
            order.append(bid)
    order.reverse()
    return order


class Relation(Enum):
    """
    One order relation between two scalars, shared by the semantic comparisons and the comparators that serve them.
    The value is the symbol a tapped flag renders as.
    """

    LT = "<"
    LE = "≤"
    GT = ">"
    GE = "≥"
    EQ = "="
    NE = "≠"

    def __repr__(self) -> str:
        return self.name

    def holds(self, a: int | float, b: int | float) -> bool:
        return _HOLDS[self](a, b)

    @property
    def mirror(self) -> "Relation":
        """The relation that means the same with the operands exchanged."""
        return _MIRROR.get(self, self)


_HOLDS: dict[Relation, Callable[[int | float, int | float], bool]] = {
    Relation.LT: operator.lt,
    Relation.LE: operator.le,
    Relation.GT: operator.gt,
    Relation.GE: operator.ge,
    Relation.EQ: operator.eq,
    Relation.NE: operator.ne,
}
_MIRROR = {Relation.LT: Relation.GT, Relation.GT: Relation.LT, Relation.LE: Relation.GE, Relation.GE: Relation.LE}
