"""
Diamond if-conversion: a small, pure branch diamond collapses into `fselect` data muxes.

A diamond is `P: Branch(c, T, F)` where both arms are single-predecessor, phi-free, operation-only blocks jumping
to one merge block `M` whose only predecessors they are. When every arm operation is speculatable and each arm is
small enough, the diamond is spliced into `P`: both arms' operations run unconditionally,
each of `M`'s phis becomes a mux under its original value id -- `fselect(c, true_arm, false_arm)`
for a float phi, `iselect` for an integer one and `bselect` for a boolean one (so downstream references
need no rewrite) -- and `M`'s terminator replaces the branch. The pass repeats until no diamond converts, so nested
chains collapse from the inside out; the dead branch condition (when nothing else reads it) falls to DCE, which runs
after this pass. Conversion turns control dependence into data dependence: a diamond whose merged results are entirely
unused frees its condition cone for DCE like any other dead code, so an error-bearing operation feeding only such a
condition stops reporting -- consistent with the error sideband's contract (executed operators only).

Both arms are computed, so conversion only pays where arms are cheap: the per-arm operation budget is
`Options.ifconv_max_ops`. A boolean mux's constant arms (the common True/False of a state-machine merge)
reduce to and/or/not in the strength-reduction pass that re-runs after if-conversion.
"""

import logging

from ._const import BoolConst
from .._util import BlockId
from ._ir import Block, Branch, Hir, Jump, Operation, Phi, predecessors, renumber
from ._operators import BoolSelect, FloatSelect, IntSelect
from ._types import BoolType, FloatType, IntType

_logger = logging.getLogger(__name__)


def _arm_convertible(hir: Hir, preds: dict[BlockId, set[BlockId]], arm: Block, pred: BlockId, max_ops: int) -> bool:
    """A convertible arm: single-pred from `pred`, phi-free, all-speculatable ops within budget, one Jump out."""
    if preds[arm.id] != {pred} or arm.phis or not isinstance(arm.terminator, Jump):
        return False
    if len(arm.operations) > max_ops:
        return False
    for vid in arm.operations:
        node = hir.nodes[vid]
        assert isinstance(node, Operation)
        if not node.operator.speculatable:
            return False
    return True


def _find_diamond(
    hir: Hir, preds: dict[BlockId, set[BlockId]], max_ops: int
) -> tuple[Block, Block, Block, Block] | None:
    """The first convertible diamond `(P, T, F, M)` in block order, or None."""
    blocks_by_id = {block.id: block for block in hir.blocks}
    for block in hir.blocks:
        if not isinstance(block.terminator, Branch):
            continue
        if block.terminator.if_true == block.terminator.if_false:
            continue
        arm_t, arm_f = blocks_by_id[block.terminator.if_true], blocks_by_id[block.terminator.if_false]
        if not (
            _arm_convertible(hir, preds, arm_t, block.id, max_ops)
            and _arm_convertible(hir, preds, arm_f, block.id, max_ops)
        ):
            continue
        assert isinstance(arm_t.terminator, Jump) and isinstance(arm_f.terminator, Jump)
        if arm_t.terminator.target != arm_f.terminator.target:
            continue
        merge = blocks_by_id[arm_t.terminator.target]
        if merge.id == block.id or preds[merge.id] != {arm_t.id, arm_f.id}:
            continue
        if not all(isinstance(hir.nodes[vid].type, (FloatType, IntType, BoolType)) for vid in merge.phis):
            continue
        assert not isinstance(hir.nodes[block.terminator.cond], BoolConst), "pruning owns a decided diamond"
        return block, arm_t, arm_f, merge
    return None


def _splice(hir: Hir, diamond: tuple[Block, Block, Block, Block]) -> Hir:
    """Collapse the diamond into its branching block; `M`'s phis become selects under their original value ids."""
    pred, arm_t, arm_f, merge = diamond
    terminator = pred.terminator
    assert isinstance(terminator, Branch)
    nodes = dict(hir.nodes)
    for vid in merge.phis:
        phi = nodes[vid]
        assert isinstance(phi, Phi)
        arm_value = dict(phi.arms)
        match phi.type:
            case FloatType():
                op: FloatSelect | IntSelect | BoolSelect = FloatSelect()
            case IntType():
                op = IntSelect()
            case BoolType():
                op = BoolSelect()
            case _:
                raise AssertionError(f"if-conversion reached an unsupported phi type: {phi.type}")
        nodes[vid] = Operation(op, (terminator.cond, arm_value[arm_t.id], arm_value[arm_f.id]))
    spliced = Block(
        id=pred.id,
        phis=pred.phis,
        operations=pred.operations + arm_t.operations + arm_f.operations + merge.phis + merge.operations,
        terminator=merge.terminator,
    )

    dissolved = {arm_t.id, arm_f.id, merge.id}
    blocks = [spliced if block.id == pred.id else block for block in hir.blocks if block.id not in dissolved]
    # A successor phi taking an arm from the dissolved merge block now takes it from the spliced block.
    for survivor in blocks:
        for vid in survivor.phis:
            phi = nodes[vid]
            assert isinstance(phi, Phi)
            arms = tuple((pred.id if arm_pred == merge.id else arm_pred, value) for arm_pred, value in phi.arms)
            nodes[vid] = Phi(type=phi.type, arms=arms)
    return Hir(nodes=nodes, blocks=blocks, input_ids=hir.input_ids, outputs=hir.outputs, state_slots=hir.state_slots)


def run(hir: Hir, max_ops: int) -> Hir | None:
    assert max_ops >= 0
    converted = 0
    while (diamond := _find_diamond(hir, predecessors(hir.blocks), max_ops)) is not None:
        hir = _splice(hir, diamond)
        converted += 1
    if not converted:
        return None
    _logger.info("If-conversion: %d diamond(s) collapsed to selects; %d blocks remain", converted, len(hir.blocks))
    return renumber(hir)
