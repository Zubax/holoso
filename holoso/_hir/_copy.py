"""Shared HIR rewrite helper."""

from typing import assert_never

from ._const import Const
from ._ir import HirBuilder, InPort, Node, Operation, ValueId


def copy_node(builder: HirBuilder, node: Node, remap: dict[ValueId, ValueId]) -> ValueId:
    """Rebuild ``node`` into ``builder`` with operands remapped."""
    match node:
        case InPort(name=name, type=type):
            return builder.input(name, type)
        case Const():
            return builder.const_node(node)
        case Operation(operator=operator, operands=operands):
            return builder.operation(operator, [remap[operand] for operand in operands])
        case _ as unreachable:
            assert_never(unreachable)
