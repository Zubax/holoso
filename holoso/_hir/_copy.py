"""Shared HIR rewrite helper."""

from typing import assert_never

from ._ir import Const, HirBuilder, InPort, Node, Operation, ValueId


def copy_node(builder: HirBuilder, node: Node, remap: dict[ValueId, ValueId]) -> ValueId:
    """Rebuild ``node`` into ``builder`` with operands remapped."""
    match node:
        case InPort(name=name):
            return builder.input(name)
        case Const(value=value):
            return builder.const(value)
        case Operation(operator=operator, operands=operands):
            return builder.operation(operator, [remap[operand] for operand in operands])
        case _ as unreachable:
            assert_never(unreachable)
