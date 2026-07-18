"""
Diamond if-conversion: a small, pure branch diamond collapses into ``select`` data muxes.

A diamond is ``P: Branch(c, T, F)`` where both arms are single-predecessor, phi-free, operation-only blocks jumping
to one merge block ``M`` whose only predecessors they are. When every arm operation is speculatable (no error
sideband -- division is excluded, since a speculated div-by-zero would raise the module's error flag for a path
never taken) and each arm is small enough, the diamond is spliced into ``P``: both arms' operations run
unconditionally, each of ``M``'s phis becomes a mux under its original value id -- ``select(c, true_arm, false_arm)``
for a float phi, ``bool_select`` for a boolean phi (so downstream references need no rewrite) -- and ``M``'s
terminator replaces the branch. The pass repeats until no diamond converts, so nested chains collapse from the inside
out; the dead branch condition (when nothing else reads it) falls to DCE, which runs after this pass. Conversion turns
control dependence into data dependence: a diamond whose merged results are entirely unused frees its condition cone
for DCE like any other dead code, so an error-bearing operation feeding only such a condition stops reporting --
consistent with the error sideband's contract (executed operators only; an unused division is dead code with or
without a branch around it).

Both arms are computed, so conversion only pays where arms are cheap: the per-arm operation budget is the
``HOLOSO_IFCONV_MAX_OPS`` knob (developer-only, read once; 0 disables the pass entirely). A diamond with any phi that
is neither float nor boolean (none exist today) is left as a real branch. A boolean mux's constant arms (the common
``True``/``False`` of a state-machine merge) reduce to ``and``/``or``/``not`` in the strength-reduction pass that
re-runs after if-conversion.

Guarded-region predication runs FIRST each iteration: a nested guard ``if A: if B: <effect>`` (no else) lowers to a
two-branch region ``P: A?G:F0`` / ``G: B?T:F1`` (``G`` phi-free; its operations -- typically the cone computing ``B``
-- are hoisted into ``P`` by the fusion, so like a diamond arm they must all be speculatable and fit the op budget)
whose inner merge ``I`` and the outer bypass ``F0`` reconverge at ``O``. When every value ``O`` observes on the
inner-false path equals the value on the outer-false path -- so the two bypasses are interchangeable -- the region is
exactly ``(A and B) ? effect : bypass``. It is rewritten to the single diamond ``P: band(A,B) ? T : F0`` reconverging
at ``O``, which the ordinary splicer then collapses. This recovers the tight ``select(A and B, ...)`` the region
means, rather than the nested ``select(A, select(B, ...))`` a bottom-up diamond collapse leaves. It must precede
diamond conversion and any merge/empty-block cleanup, which would erase the two-merge evidence. Hoisting evaluates
``B``'s cone eagerly, matching the eager ``and`` the frontend emits for the hand-written spelling; that is
unobservable for speculatable operations, while a faulting one (division) refuses the fusion. Any ``G``-defined value
observed at ``O`` can never equal its outer-false peer (it does not dominate ``F0``), so the equality test confines
the hoisted values to the branch condition and the true path; a walrus or state write on the inner path likewise
makes some inner-false value differ from its outer-false peer, failing the same test.
"""

import logging
import os

from ._const import BoolConst
from .._util import BlockId, ValueId
from ._ir import Block, Branch, Hir, Jump, Operation, Phi, predecessors, renumber, validate_phi_predecessors
from ._operators import BoolAnd, BoolSelect, Select
from ._types import BoolType, FloatType

_IFCONV_MAX_OPS = int(os.getenv("HOLOSO_IFCONV_MAX_OPS", "8"))

_logger = logging.getLogger(__name__)


def _speculatable_within_budget(hir: Hir, block: Block) -> bool:
    """Whether every operation of ``block`` may run on a not-taken path: all speculatable, within the op budget."""
    if len(block.operations) > _IFCONV_MAX_OPS:
        return False
    for vid in block.operations:
        node = hir.nodes[vid]
        assert isinstance(node, Operation)
        if not node.operator.speculatable:
            return False
    return True


def _arm_convertible(hir: Hir, preds: dict[BlockId, set[BlockId]], arm: Block, pred: BlockId) -> bool:
    """A convertible arm: single-pred from ``pred``, phi-free, all-speculatable ops within budget, one Jump out."""
    if preds[arm.id] != {pred} or arm.phis or not isinstance(arm.terminator, Jump):
        return False
    return _speculatable_within_budget(hir, arm)


def _find_diamond(hir: Hir, preds: dict[BlockId, set[BlockId]]) -> tuple[Block, Block, Block, Block] | None:
    """The first convertible diamond ``(P, T, F, M)`` in block order, or None."""
    blocks_by_id = {block.id: block for block in hir.blocks}
    for block in hir.blocks:
        if not isinstance(block.terminator, Branch):
            continue
        if block.terminator.if_true == block.terminator.if_false:
            continue
        if isinstance(hir.nodes[block.terminator.cond], BoolConst):
            # The frontend's static-condition detection is deliberately incomplete, so a constant condition CAN
            # reach this pass; converting it would pin the untaken arm live through the select forever. Refusing it
            # keeps "a constant-condition select never exists" true by construction.
            continue
        arm_t, arm_f = blocks_by_id[block.terminator.if_true], blocks_by_id[block.terminator.if_false]
        if not (_arm_convertible(hir, preds, arm_t, block.id) and _arm_convertible(hir, preds, arm_f, block.id)):
            continue
        assert isinstance(arm_t.terminator, Jump) and isinstance(arm_f.terminator, Jump)
        if arm_t.terminator.target != arm_f.terminator.target:
            continue
        merge = blocks_by_id[arm_t.terminator.target]
        if merge.id == block.id or preds[merge.id] != {arm_t.id, arm_f.id}:
            continue
        if not all(isinstance(hir.nodes[vid].type, (FloatType, BoolType)) for vid in merge.phis):
            continue
        return block, arm_t, arm_f, merge
    return None


def _splice(hir: Hir, diamond: tuple[Block, Block, Block, Block]) -> Hir:
    """Collapse the diamond into its branching block; ``M``'s phis become selects under their original value ids."""
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
                op: Select | BoolSelect = Select()
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


def _arm_from(hir: Hir, value: ValueId, inner_phis: set[ValueId], pred: BlockId) -> ValueId:
    """A value the inner merge carries as seen on one incoming edge: an inner phi's arm from ``pred``, else itself."""
    node = hir.nodes[value]
    return dict(node.arms)[pred] if value in inner_phis and isinstance(node, Phi) else value


def _inner_phis_consumed_only_as_outer_arm(hir: Hir, inner: set[ValueId], inner_id: BlockId, outer: BlockId) -> bool:
    """Every use of every inner-merge phi is the arm the outer merge takes from it -- else deleting it would dangle."""
    outer_phis = {phi for block in hir.blocks if block.id == outer for phi in block.phis}
    for vid, node in hir.nodes.items():
        if isinstance(node, Operation) and any(operand in inner for operand in node.operands):
            return False
        if isinstance(node, Phi):
            for pred, value in node.arms:
                if value in inner and (vid not in outer_phis or pred != inner_id):
                    return False
    for block in hir.blocks:
        if isinstance(block.terminator, Branch) and block.terminator.cond in inner:
            return False
    return not (any(o.value in inner for o in hir.outputs) or any(s.live_out in inner for s in hir.state_slots))


def _find_guarded_region(hir: Hir, preds: dict[BlockId, set[BlockId]]) -> tuple[Block, Block, Block, Block] | None:
    """
    The first fusible nested guard ``(P, G, T, F0)``, or None. ``P: A?G:F0``; ``G: B?T:F1`` is phi-free with
    speculatable ops within budget (the fusion hoists them into ``P``); ``T``/``F1`` reconverge at an operation-free
    inner merge ``I`` that meets the outer bypass ``F0`` at ``O``; and every value ``O`` observes on the inner-false
    path equals its outer-false peer.
    """
    by = {block.id: block for block in hir.blocks}
    for p in hir.blocks:
        if not isinstance(p.terminator, Branch) or p.terminator.if_true == p.terminator.if_false:
            continue
        g, f0 = by[p.terminator.if_true], by[p.terminator.if_false]
        if preds[g.id] != {p.id} or g.phis or not isinstance(g.terminator, Branch):
            continue
        if not _speculatable_within_budget(hir, g):
            continue
        if g.terminator.if_true == g.terminator.if_false:
            continue
        t, f1 = by[g.terminator.if_true], by[g.terminator.if_false]
        if preds[t.id] != {g.id} or preds[f1.id] != {g.id} or t.phis or f1.phis:
            continue
        if not (isinstance(t.terminator, Jump) and isinstance(f1.terminator, Jump)):
            continue
        if t.terminator.target != f1.terminator.target:
            continue
        inner = by[t.terminator.target]
        if preds[inner.id] != {t.id, f1.id} or inner.operations or not isinstance(inner.terminator, Jump):
            continue
        outer = by[inner.terminator.target]
        if preds[f0.id] != {p.id} or f0.phis or not isinstance(f0.terminator, Jump):
            continue
        if f0.terminator.target != outer.id or outer.id in (p.id, inner.id):
            continue
        if preds[outer.id] != {inner.id, f0.id}:
            continue
        inner_phis = set(inner.phis)
        if not _inner_phis_consumed_only_as_outer_arm(hir, inner_phis, inner.id, outer.id):
            continue
        arms = [dict(hir.nodes[vid].arms) for vid in outer.phis]  # type: ignore[union-attr]
        if all(_arm_from(hir, arm[inner.id], inner_phis, f1.id) == arm[f0.id] for arm in arms):
            return p, g, t, f0
    return None


def _fuse_guard(hir: Hir, region: tuple[Block, Block, Block, Block]) -> Hir:
    """
    Rewrite the guard into a single diamond ``P: band(A,B)?T:F0`` -> ``O``; the splicer then collapses it. ``G``'s
    operations are hoisted into ``P`` ahead of the combined condition, preserving them (and the cone of ``B``) while
    keeping every def above its uses in ``T`` and in the rewritten outer phis.
    """
    p, g, t, f0 = region
    assert isinstance(p.terminator, Branch) and isinstance(g.terminator, Branch)
    inner = {block.id: block for block in hir.blocks}[t.terminator.target]  # type: ignore[union-attr]
    outer_id = inner.terminator.target  # type: ignore[union-attr]
    inner_phis = set(inner.phis)
    f1_id = g.terminator.if_false if g.terminator.if_true == t.id else g.terminator.if_true

    nodes = dict(hir.nodes)
    combined = max(nodes) + 1
    nodes[combined] = Operation(BoolAnd(), (p.terminator.cond, g.terminator.cond))
    for block in hir.blocks:
        if block.id != outer_id:
            continue
        for vid in block.phis:
            phi = nodes[vid]
            assert isinstance(phi, Phi)
            arm = dict(phi.arms)
            true_value = _arm_from(hir, arm[inner.id], inner_phis, t.id)
            nodes[vid] = Phi(type=phi.type, arms=((t.id, true_value), (f0.id, arm[f0.id])))
    for vid in inner.phis:
        del nodes[vid]

    dissolved = {g.id, f1_id, inner.id}
    blocks = []
    for block in hir.blocks:
        if block.id in dissolved:
            continue
        if block.id == p.id:
            blocks.append(Block(p.id, p.phis, p.operations + g.operations + (combined,), Branch(combined, t.id, f0.id)))
        elif block.id == t.id:
            blocks.append(Block(t.id, t.phis, t.operations, Jump(outer_id)))
        else:
            blocks.append(block)
    return Hir(nodes=nodes, blocks=blocks, input_ids=hir.input_ids, outputs=hir.outputs, state_slots=hir.state_slots)


def run(hir: Hir) -> Hir:
    if _IFCONV_MAX_OPS <= 0:
        return hir
    converted = fused = 0
    while True:
        preds = predecessors(hir.blocks)
        if (region := _find_guarded_region(hir, preds)) is not None:
            hir = _fuse_guard(hir, region)
            fused += 1
        elif (diamond := _find_diamond(hir, preds)) is not None:
            hir = _splice(hir, diamond)
            converted += 1
        else:
            break
    if converted or fused:
        _logger.info(
            "If-conversion: %d guard(s) fused, %d diamond(s) collapsed to selects; %d blocks remain",
            fused,
            converted,
            len(hir.blocks),
        )
        hir = renumber(hir)
        validate_phi_predecessors(hir)
    return hir
