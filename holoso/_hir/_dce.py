"""HIR dead-code elimination."""

from ._copy import copy_node
from ._ir import Hir, HirBuilder, Operation, ValueId


def run(hir: Hir) -> Hir:
    """Drop nodes not reachable from any output; all input ports are retained as the module signature."""
    reachable: set[ValueId] = set()
    stack = [out.value for out in hir.outputs]
    while stack:
        vid = stack.pop()
        if vid in reachable:
            continue
        reachable.add(vid)
        match hir.nodes[vid]:
            case Operation(operands=operands):
                stack.extend(operands)
            case _:
                pass
    keep = reachable | set(hir.input_ids)
    builder = HirBuilder()
    remap: dict[ValueId, ValueId] = {}
    for old_id in sorted(keep):
        remap[old_id] = copy_node(builder, hir.nodes[old_id], remap)
    for out in hir.outputs:
        builder.output(out.name, remap[out.value])
    return builder.finish()
