"""
Shared auxiliary entities used across the IR layers.
"""

from collections.abc import Mapping

type ValueId = int
"""An SSA value identifier, unique within one IR graph."""

type BlockId = int
"""A basic-block identifier, unique within one control-flow graph."""


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
