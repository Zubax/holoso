"""HIR dead-code elimination."""

from dataclasses import replace

from ._copy import rebuild
from .._util import ValueId
from ._ir import Branch, Hir, StateRead, references


def eliminate_dead_code(hir: Hir) -> Hir:
    """
    Drop values unreachable from any output or branch condition, and every state slot whose live-in nothing reads:
    a slot is observable only through that read, a public attribute's port being an ordinary output of its live-out.
    Inputs are kept as the module signature. Block structure is preserved (a structured CFG has no dead blocks at
    this stage).
    """
    live_out_of = {slot.name: slot.live_out for slot in hir.state_slots}
    kept: set[ValueId] = set()
    pending = [out.value for out in hir.outputs]
    pending += [block.terminator.cond for block in hir.blocks if isinstance(block.terminator, Branch)]
    while pending:
        vid = pending.pop()
        if vid not in kept:
            kept.add(vid)
            node = hir.nodes[vid]
            pending.extend(references(node))
            if isinstance(node, StateRead):
                pending.append(live_out_of[node.slot])
    read = {node.slot for vid, node in hir.nodes.items() if vid in kept and isinstance(node, StateRead)}
    return rebuild(replace(hir, state_slots=[slot for slot in hir.state_slots if slot.name in read]), keep=kept)
