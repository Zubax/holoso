"""
Jump-chain fusion. A block reached only by a jump from one predecessor runs exactly when that predecessor ends, so it
is fused into it: the predecessor takes its operations after its own and its terminator, and a successor's phi arm
from it now comes from the predecessor. A block boundary costs the machine a drain and bounds if-conversion and the
sharing of identical expressions, so an unfused chain -- the path a settled branch leaves behind, or a return site's
jump to the exit -- costs cycles and operators for no work.

Fusion follows only jump edges and moves a terminator without retargeting it, so no branch comes to point at a loop
header, and a fused block never carries a phi: with one predecessor its phis merge one value, which pruning has folded.
"""

from dataclasses import replace
import logging

from ._ir import Block, Hir, Jump, Phi, predecessors, renumber, validate_phi_predecessors

_logger = logging.getLogger(__name__)


def _chain_link(hir: Hir) -> tuple[Block, Block] | None:
    preds = predecessors(hir.blocks)
    blocks_by_id = {block.id: block for block in hir.blocks}
    for block in hir.blocks:
        if block.id == hir.entry or len(preds[block.id]) != 1:
            continue
        (pred,) = preds[block.id]
        assert pred != block.id, "a block reached only from itself is unreachable, which pruning removes"
        if isinstance(blocks_by_id[pred].terminator, Jump):
            return blocks_by_id[pred], block
    return None


def _fuse(hir: Hir, pred: Block, block: Block) -> Hir:
    assert not block.phis, "a single-predecessor block's phis merge one value, which pruning folds"
    fused = Block(pred.id, pred.phis, pred.operations + block.operations, block.terminator)
    blocks = [fused if survivor.id == pred.id else survivor for survivor in hir.blocks if survivor.id != block.id]
    nodes = dict(hir.nodes)
    for survivor in blocks:
        for phi_id in survivor.phis:
            phi = nodes[phi_id]
            assert isinstance(phi, Phi)
            arms = tuple((pred.id if arm_pred == block.id else arm_pred, value) for arm_pred, value in phi.arms)
            nodes[phi_id] = Phi(type=phi.type, arms=arms)
    return replace(hir, nodes=nodes, blocks=blocks)


def run(hir: Hir) -> Hir | None:
    fused = 0
    while (link := _chain_link(hir)) is not None:
        hir = _fuse(hir, *link)
        fused += 1
    if not fused:
        return None
    _logger.info("Jump-chain fusion: %d block(s) fused; %d block(s) remain", fused, len(hir.blocks))
    hir = renumber(hir)
    validate_phi_predecessors(hir)
    return hir
