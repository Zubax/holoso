"""
A subtraction orders its operands on the adder's flags as a side effect, so a comparison of the same two operands
needs no firing of its own.
"""

import logging
from dataclasses import dataclass

from .._hir import Hir, IntComparison, IntSub, Operation
from .._util import ValueId

_logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FusedComparison:
    subtraction: ValueId
    mirrored: bool
    """Whether the comparison reads the subtraction's operands in the opposite order."""


def plan_fusions(hir: Hir) -> dict[ValueId, FusedComparison]:
    """
    Map each integer comparison to a same-block subtraction of the same two operands, in either order, so MIR can tap
    the subtraction's flags (the two fuse into one adder firing) rather than fire a comparison. Block-local, like the
    LIR firing fusion it feeds.
    """
    plans: dict[ValueId, FusedComparison] = {}
    for block in hir.blocks:
        subtractions: dict[tuple[ValueId, ...], ValueId] = {}
        for vid in block.operations:
            if isinstance(node := hir.nodes[vid], Operation) and isinstance(node.operator, IntSub):
                subtractions.setdefault(node.operands, vid)
        for vid in block.operations:
            if isinstance(node := hir.nodes[vid], Operation) and isinstance(node.operator, IntComparison):
                a, b = node.operands
                if (a, b) in subtractions:
                    plans[vid] = FusedComparison(subtractions[(a, b)], mirrored=False)
                elif (b, a) in subtractions:
                    plans[vid] = FusedComparison(subtractions[(b, a)], mirrored=True)
    if plans:
        _logger.info("Comparison: %d fused onto an adjacent subtraction's flags", len(plans))
    return plans
