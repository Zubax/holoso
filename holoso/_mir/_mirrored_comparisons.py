"""
An integer subtraction orders its operands on the adder's flags as a side effect, and a float min or max on the
comparator's, so a comparison of the same two operands needs no firing of its own. A comparison reading them in the
host's order is the host's primitive over the same operands and fuses with it as any two taps of one firing do; one
reading them the other way round must be given the host's order, and asks for the mirrored relation.
"""

import logging

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


def _conditioned_operands(hir: Hir, node: Operation) -> tuple[tuple[ValueId, PortConditioner], ...]:
    """The operands as the ports will read them, so that every spelling of one sign chain is one operand."""
    return tuple(collapse_conditioner(hir.nodes, operand) for operand in node.operands)


def plan_mirrored_comparisons(hir: Hir) -> dict[ValueId, ValueId]:
    """
    Map each comparison that reads a same-block ordering operation's operands in the opposite order (and not in the
    same order, which fuses by itself) to that host, whose operand order it then takes. Block-local, like the LIR
    firing fusion it feeds.
    """
    plans: dict[ValueId, ValueId] = {}
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
                    if (a, b) not in hosts and (b, a) in hosts:
                        plans[vid] = hosts[(b, a)]
    if plans:
        _logger.info("Comparison: %d mirrored onto the operand order of an operation ordering them", len(plans))
    return plans
