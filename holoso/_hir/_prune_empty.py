"""
Empty non-merge block elimination.

The complement of ``_thread_merges``, which threads empty blocks that carry phis (composing them into a successor's
merge). This pass removes empty blocks that carry NO phis, the trampolines the structured single-exit emitter leaves on
straight-line and branch-exit paths:

- an empty ``Ret`` block whose sole predecessor jumps to it -- the canonical exit of a function that needs no merge;
  the predecessor's ``Jump`` becomes the ``Ret`` and the empty block is dropped;
- an empty ``Jump`` block with no phis whose successor also has no phis -- a pure pass-through on a branch or exit path;
  each predecessor (``Jump`` or ``Branch`` arm) is retargeted straight to the successor and the block is dropped.

Both are safe because no phi arm is composed or created: the successor has no merge to feed, so retargeting even a
``Branch`` predecessor onto it introduces no forbidden branch-block phi arm. An empty block whose successor DOES carry
phis is left for ``_thread_merges`` (or kept as the edge split a phi arm needs). Each empty block is a fixed per-block
boundary drain in the schedule for zero work, so removing it shortens the static schedule without changing any value.
"""

import logging

from .._util import BlockId
from ._ir import Block, Branch, Hir, Jump, Ret, predecessors, renumber, validate_phi_predecessors

_logger = logging.getLogger(__name__)


def _is_empty(block: Block) -> bool:
    return not block.phis and not block.operations


def _retarget(terminator: Jump | Branch | Ret, victim: BlockId, successor: BlockId) -> Jump | Branch | Ret:
    match terminator:
        case Jump(target=target):
            return Jump(successor) if target == victim else terminator
        case Branch(cond=cond, if_true=if_true, if_false=if_false):
            new_true = successor if if_true == victim else if_true
            new_false = successor if if_false == victim else if_false
            return Jump(new_true) if new_true == new_false else Branch(cond, new_true, new_false)
        case Ret():
            return terminator


def _find_removable(hir: Hir) -> tuple[Block, BlockId | None] | None:
    preds = predecessors(hir.blocks)
    by_id = {block.id: block for block in hir.blocks}
    for block in hir.blocks:
        if block.id == hir.entry or not _is_empty(block):
            continue
        match block.terminator:
            case Ret() if len(preds[block.id]) == 1:
                (sole,) = preds[block.id]
                if isinstance(by_id[sole].terminator, Jump):
                    return block, None  # fold the Ret up into the sole jumping predecessor
            case Jump(target=target) if target != block.id and preds[block.id] and not by_id[target].phis:
                return block, target  # thread the pass-through onto the phi-less successor
    return None


def _remove(hir: Hir, victim: Block, successor: BlockId | None) -> Hir:
    if successor is None:  # empty Ret block: its sole predecessor becomes the Ret
        blocks = [
            Block(b.id, b.phis, b.operations, Ret() if victim.id in _jump_targets(b) else b.terminator)
            for b in hir.blocks
            if b.id != victim.id
        ]
    else:
        blocks = [
            Block(b.id, b.phis, b.operations, _retarget(b.terminator, victim.id, successor))
            for b in hir.blocks
            if b.id != victim.id
        ]
    return Hir(hir.nodes, blocks, hir.input_ids, hir.outputs, hir.state_slots)


def _jump_targets(block: Block) -> set[BlockId]:
    return {block.terminator.target} if isinstance(block.terminator, Jump) else set()


def run(hir: Hir) -> Hir:
    removed = 0
    while (candidate := _find_removable(hir)) is not None:
        block, successor = candidate
        hir = _remove(hir, block, successor)
        removed += 1
    if removed:
        _logger.info("Empty-block elimination: %d empty block(s) removed; %d remain", removed, len(hir.blocks))
        hir = renumber(hir)
        validate_phi_predecessors(hir)
    return hir
