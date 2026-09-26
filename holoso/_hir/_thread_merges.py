"""
Empty merge-block elimination (jump-threading).

If-conversion collapses a small, speculatable diamond into an `fselect`; a diamond whose arm holds a non-speculatable
operation (a variable-divisor division) stays a real branch with a separate phi merge block. When that merge feeds a
following control structure -- a loop header, a sibling/cascade merge, another diamond -- and has no operation of its
own to host, it is a pure pass-through: phis but no operations, a single `Jump` out, and predecessors that all jump
to it. Such a block contributes only a fixed per-block
boundary drain (the fetch refill plus the result-landing tail) for zero work.

This pass threads such a block `M` onto its predecessors: each predecessor's `Jump` is retargeted from `M` to
`M`'s successor `S`, and `M`'s phi arms compose into `S`'s phis -- an arm `(M, v)` of an `S` phi becomes one
arm per predecessor `Q` of `M` (`(Q, a_Q)` when `v` is an `M` phi `phi(Q: a_Q)`, else the pass-through value
`(Q, v)`). No composed arm comes from a branching block, whose install would fire on every edge out of it: every
predecessor `Q` of `M` is `Jump`-terminated.

Scope. The pass fires only when every `M` phi is consumed SOLELY as the arm an `S` phi takes FROM `M`, the one arm
composition rewrites; a merge phi reached any other way keeps `M` a real branch (see DESIGN.md). Chained merges (an
`M` whose successor is itself an `M`) collapse by repeating to a fixpoint, innermost-reachable first.
"""

from collections import Counter
from dataclasses import replace
import logging

from .._util import BlockId, ValueId
from ._ir import Block, Hir, Jump, Phi, predecessors, renumber, validate_phi_predecessors

_logger = logging.getLogger(__name__)


def _jumps_to(block: Block, target: BlockId) -> bool:
    return isinstance(block.terminator, Jump) and block.terminator.target == target


def _phis_consumed_only_as_successor_arms(
    hir: Hir, merge_phis: set[ValueId], merge_id: BlockId, successor: BlockId
) -> bool:
    """
    Every use of every merge phi is the arm a `successor` phi takes FROM the merge block (the `pred == merge_id`
    arm) -- the only arm composition rewrites. Any other use is fatal to threading, because composition leaves it
    untouched yet the merge phi is deleted, so the reference would dangle: an operation operand, a branch condition, an
    output, a state live-out, an arm of a phi outside `successor`, OR -- the subtle one -- an arm of a successor phi
    taken from a DIFFERENT predecessor (a loop-invariant value carried on the loop header's back-edge arm; the merge
    block dominates the back-edge through the header, so its value legitimately reaches that arm). Such a merge stays a
    real branch: it is exactly the deferred self-latch rematerialization case.
    """
    (block,) = [block for block in hir.blocks if block.id == successor]
    permitted: Counter[ValueId] = Counter()
    for phi_id in block.phis:
        phi = hir.nodes[phi_id]
        assert isinstance(phi, Phi)
        permitted.update(value for pred, value in phi.arms if pred == merge_id)
    uses = hir.use_counts()
    return all(uses[phi] == permitted[phi] for phi in merge_phis)


def _find_empty_merge(hir: Hir) -> tuple[Block, BlockId] | None:
    preds = predecessors(hir.blocks)
    blocks_by_id = {block.id: block for block in hir.blocks}
    for block in hir.blocks:
        if block.id == hir.entry or block.operations or not isinstance(block.terminator, Jump):
            continue
        successor = block.terminator.target
        if successor == block.id or not preds[block.id]:
            continue
        if not all(_jumps_to(blocks_by_id[pred], block.id) for pred in preds[block.id]):
            continue
        if _phis_consumed_only_as_successor_arms(hir, set(block.phis), block.id, successor):
            return block, successor
    return None


def _thread(hir: Hir, merge: Block, successor: BlockId) -> Hir:
    arm_preds = sorted(predecessors(hir.blocks)[merge.id])  # deterministic predecessor order
    merge_arms: dict[ValueId, dict[BlockId, ValueId]] = {}  # merge phi -> {predecessor: that arm's value}
    for phi_id in merge.phis:
        phi = hir.nodes[phi_id]
        assert isinstance(phi, Phi)
        merge_arms[phi_id] = dict(phi.arms)
    merge_phis = set(merge.phis)

    def expand(value: ValueId) -> list[tuple[BlockId, ValueId]]:
        # The arm an S phi took from M becomes one arm per predecessor of M: the predecessor's own arm of the merged
        # phi, or the pass-through value itself when defined above M (so it flows unchanged from each predecessor).
        if value in merge_phis:
            return [(pred, merge_arms[value][pred]) for pred in arm_preds]
        return [(pred, value) for pred in arm_preds]

    nodes = dict(hir.nodes)
    for block in hir.blocks:
        if block.id != successor:
            continue
        for phi_id in block.phis:
            phi = nodes[phi_id]
            assert isinstance(phi, Phi)
            arms: list[tuple[BlockId, ValueId]] = []
            for pred, value in phi.arms:
                arms.extend(expand(value) if pred == merge.id else [(pred, value)])
            nodes[phi_id] = Phi(type=phi.type, arms=tuple(arms))
    for phi_id in merge.phis:  # the merge's phis are now unreferenced
        del nodes[phi_id]

    blocks = [
        (Block(block.id, block.phis, block.operations, Jump(successor)) if block.id in arm_preds else block)
        for block in hir.blocks
        if block.id != merge.id
    ]
    return replace(hir, nodes=nodes, blocks=blocks)


def run(hir: Hir) -> Hir | None:
    threaded = 0
    while (candidate := _find_empty_merge(hir)) is not None:
        merge, successor = candidate
        hir = _thread(hir, merge, successor)
        threaded += 1
    if not threaded:
        return None
    _logger.info("Merge threading: %d empty merge block(s) eliminated; %d blocks remain", threaded, len(hir.blocks))
    hir = renumber(hir)
    validate_phi_predecessors(hir)
    return hir
