"""Shared HIR rewrite helper."""

from typing import assert_never

from ._const import Const
from ._ir import Hir, HirBuilder, InPort, Node, Operation, StateRead, ValueId


def copy_node(builder: HirBuilder, node: Node, remap: dict[ValueId, ValueId]) -> ValueId:
    """Rebuild ``node`` into ``builder`` with operands remapped."""
    match node:
        case InPort(name=name, type=type):
            return builder.input(name, type)
        case StateRead(slot=slot, type=type):
            return builder.state_read(slot, type)
        case Const():
            return builder.const_node(node)
        case Operation(operator=operator, operands=operands):
            return builder.operation(operator, [remap[operand] for operand in operands])
        case _ as unreachable:
            assert_never(unreachable)


def copy_state_slots(builder: HirBuilder, hir: Hir, remap: dict[ValueId, ValueId]) -> None:
    """Re-emit every persistent state slot with its live-out value remapped (slots survive every rewrite pass)."""
    for slot in hir.state_slots:
        builder.state_slot(slot.name, slot.reset_value, remap[slot.live_out])
