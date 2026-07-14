"""
Linear jump-chain fusion (basic-block merging).

The front-end emitter mirrors its own CFG one block per region seam -- the function entry, an inlined call boundary,
each unrolled loop iteration -- so a straight-line region reaches HIR split across a chain of blocks linked by
unconditional jumps. Every such boundary costs a fixed fetch/drain overhead in the scheduled microprogram and fences
cross-boundary overlap (each block schedules independently), so the split region schedules strictly worse than the
same operations fused into one block.

This pass fuses ``A -> B`` whenever ``A`` ends in ``Jump(B)``, ``B`` is not the entry, ``A`` is ``B``'s sole
predecessor, and ``B`` carries no phis: ``B``'s operations append to ``A``'s, ``A`` adopts ``B``'s terminator, and a
phi arm elsewhere taken from ``B`` re-keys to ``A`` (``A`` now owns the edge ``B`` supplied). Repeating to a fixpoint
collapses a whole chain into its head. The complement of ``_thread_merges`` (empty blocks WITH phis composed into a
successor's merge) and ``_prune_empty`` (phi-less empty blocks bypassed without moving anything): only this pass moves
operations between blocks. It runs before if-conversion -- a diamond arm split by an inlined call boundary must fuse
into the single-block arm the diamond matcher recognizes -- and again after the late CFG cleanups expose new chains.
"""

import logging

from ._ir import Block, Hir, Jump, Phi, predecessors, renumber, validate_phi_predecessors

_logger = logging.getLogger(__name__)


def _find_fusable(hir: Hir) -> tuple[Block, Block] | None:
    preds = predecessors(hir.blocks)
    by_id = {block.id: block for block in hir.blocks}
    for block in hir.blocks:
        match block.terminator:
            case Jump(target=target) if target != block.id and target != hir.entry:
                successor = by_id[target]
                if preds[target] == {block.id} and not successor.phis:
                    return block, successor
    return None


def _fuse(hir: Hir, head: Block, tail: Block) -> Hir:
    blocks = [
        Block(b.id, b.phis, b.operations + tail.operations, tail.terminator) if b.id == head.id else b
        for b in hir.blocks
        if b.id != tail.id
    ]
    nodes = dict(hir.nodes)
    for vid, node in hir.nodes.items():
        if isinstance(node, Phi) and any(pred == tail.id for pred, _ in node.arms):
            arms = tuple((head.id if pred == tail.id else pred, value) for pred, value in node.arms)
            nodes[vid] = Phi(type=node.type, arms=arms)
    return Hir(nodes, blocks, hir.input_ids, hir.outputs, hir.state_slots)


def run(hir: Hir) -> Hir:
    fused = 0
    while (candidate := _find_fusable(hir)) is not None:
        head, tail = candidate
        hir = _fuse(hir, head, tail)
        fused += 1
    if fused:
        _logger.info("Jump-chain fusion: %d block(s) fused into their predecessors; %d remain", fused, len(hir.blocks))
        hir = renumber(hir)
        validate_phi_predecessors(hir)
    return hir
