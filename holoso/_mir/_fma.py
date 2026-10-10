"""The fused multiply-add on a machine that has no operator for it."""

import logging

from .._hir import FloatAdd, FloatFma, FloatMul, Hir, HirBuilder, Node, Operation, copy_node, rebuild
from .._operators import FFmaOperator, OpConfig
from .._util import ValueId

_logger = logging.getLogger(__name__)


def expand_fmas(hir: Hir, ops: OpConfig) -> Hir:
    """Runs ahead of the optimizer, which then reads the product and the sum as it would read them written apart."""
    if ops.serves(FFmaOperator):
        return hir
    fused = sum(isinstance(node, Operation) and isinstance(node.operator, FloatFma) for node in hir.nodes.values())
    if not fused:
        return hir

    def build_value(builder: HirBuilder, vid: ValueId, node: Node, remap: dict[ValueId, ValueId]) -> ValueId:
        match node:
            case Operation(operator=FloatFma(), operands=(a, b, c)):
                product = builder.operation(FloatMul(), [remap[a], remap[b]])
                return builder.operation(FloatAdd(), [remap[c], product])
            case _:
                return copy_node(builder, node, remap)

    _logger.info("FMA expansion: %d fused multiply-add(s) computed as a product and a sum, lacking ffma", fused)
    return rebuild(hir, build_value)
