"""HIR constant folding."""

from ._copy import copy_node
from ._ir import Const, Hir, HirBuilder, Operation, ValueId


def run(hir: Hir) -> Hir:
    """Fold operations whose operands are all constants into a single constant."""
    builder = HirBuilder()
    remap: dict[ValueId, ValueId] = {}
    cval: dict[ValueId, float] = {}
    for old_id in sorted(hir.nodes):
        node = hir.nodes[old_id]
        match node:
            case Const(value=value):
                new_id = builder.const(value)
                cval[new_id] = value
            case Operation(operator=operator, operands=operands) if all(remap[operand] in cval for operand in operands):
                values = [cval[remap[operand]] for operand in operands]
                folded = operator.evaluate(*values)
                if folded is None:
                    new_id = copy_node(builder, node, remap)
                else:
                    new_id = builder.const(folded)
                    cval[new_id] = folded
            case _:
                new_id = copy_node(builder, node, remap)
        remap[old_id] = new_id
    for out in hir.outputs:
        builder.output(out.name, remap[out.value])
    return builder.finish()
