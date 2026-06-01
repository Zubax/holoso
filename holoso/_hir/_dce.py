"""HIR dead-code elimination."""

from ._copy import copy_node, copy_state_slots
from ._ir import Hir, HirBuilder, Operation, ValueId


def run(hir: Hir) -> Hir:
    """Drop nodes unreachable from any output or persistent state; input ports are kept as the module signature."""
    reachable: set[ValueId] = set()
    stack = [out.value for out in hir.outputs] + [slot.live_out for slot in hir.state_slots]
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
    copy_state_slots(builder, hir, remap)
    return builder.finish()
