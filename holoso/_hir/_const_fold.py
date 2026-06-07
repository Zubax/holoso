"""HIR constant folding."""

from ._const import Const
from ._copy import copy_node, rebuild
from ._ir import Hir, HirBuilder, Node, Operation, Phi, ValueId


def run(hir: Hir) -> Hir:
    """
    Fold operations whose operands are all constants into a single constant, and collapse a phi all of whose arms are
    the same constant.
    """
    cval: dict[ValueId, Const] = {}

    def uniform_const_arm(arms: tuple[tuple[int, ValueId], ...], remap: dict[ValueId, ValueId]) -> Const | None:
        if not arms:
            return None
        values = [cval.get(remap[arm]) for _, arm in arms]
        first = values[0]
        return first if first is not None and all(value == first for value in values) else None

    def build_value(builder: HirBuilder, vid: ValueId, node: Node, remap: dict[ValueId, ValueId]) -> ValueId:
        folded: Const | None = None
        match node:
            case Const():
                folded = node
            case Operation(operator=operator, operands=operands) if all(remap[op] in cval for op in operands):
                folded = operator.fold_constants([cval[remap[op]] for op in operands])
            case Phi(arms=arms):
                folded = uniform_const_arm(arms, remap)
        if folded is None:
            return copy_node(builder, node, remap)
        new_id = builder.const_node(folded)
        cval[new_id] = folded
        return new_id

    return rebuild(hir, build_value)
