"""
An integer subtraction orders its operands on the adder's flags as a side effect, and a float min or max on the
comparator's, so a comparison of the same two operands needs no firing of its own.
"""

import logging
from dataclasses import dataclass

from .._hir import FloatComparison, FloatMax, FloatMin, Hir, IntComparison, IntSub, Operation, Operator
from .._operators import PortConditioner
from .._util import ValueId
from ._signs import collapse_conditioner

_logger = logging.getLogger(__name__)

_ORDERING_OPERATORS: dict[type[Operator], tuple[type[Operator], ...]] = {
    IntComparison: (IntSub,),
    FloatComparison: (FloatMin, FloatMax),
}
"""For each kind of comparison, the operations whose firing orders their operands as that comparison would."""


@dataclass(frozen=True, slots=True)
class FusedComparison:
    host: ValueId
    """The operation whose firing orders the compared operands."""
    mirrored: bool
    """Whether the comparison reads the host's operands in the opposite order."""


def _conditioned_operands(hir: Hir, node: Operation) -> tuple[tuple[ValueId, PortConditioner], ...]:
    """The operands as the ports will read them, so that every spelling of one sign chain is one operand."""
    return tuple(collapse_conditioner(hir.nodes, operand) for operand in node.operands)


def plan_comparison_fusions(hir: Hir) -> dict[ValueId, FusedComparison]:
    """
    Map each comparison to a same-block operation that orders the same two operands, in either order, so MIR can tap
    that operation's flags (the two fuse into one firing) rather than fire a comparison. Block-local, like the LIR
    firing fusion it feeds.
    """
    plans: dict[ValueId, FusedComparison] = {}
    for block in hir.blocks:
        operations = [(vid, node) for vid in block.operations if isinstance(node := hir.nodes[vid], Operation)]
        for comparison, ordering in _ORDERING_OPERATORS.items():
            hosts: dict[tuple[tuple[ValueId, PortConditioner], ...], ValueId] = {}
            for vid, node in operations:
                if isinstance(node.operator, ordering):
                    hosts.setdefault(_conditioned_operands(hir, node), vid)
            for vid, node in operations:
                if isinstance(node.operator, comparison):
                    a, b = _conditioned_operands(hir, node)
                    if (a, b) in hosts:
                        plans[vid] = FusedComparison(hosts[(a, b)], mirrored=False)
                    elif (b, a) in hosts:
                        plans[vid] = FusedComparison(hosts[(b, a)], mirrored=True)
    if plans:
        _logger.info("Comparison: %d fused onto the flags of an operation ordering the same operands", len(plans))
    return plans
