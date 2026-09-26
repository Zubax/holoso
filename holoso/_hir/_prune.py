"""
Constant-branch pruning, the only pass that deletes code a branch was to decide. Without it a proven-dead arm reaches
hardware -- selected, materialized, and refused upon -- for code no input can reach.

A merge left with one value to merge is that value: a merge whose other edges died, or a loop header whose latch only
carries the value around. Substituting it can settle another branch, so the folding shares pruning's loop; it must,
since if-conversion leaves decided diamonds to pruning.
"""

from dataclasses import replace
import logging
from typing import assert_never

from .._errors import UnsupportedConstruct
from .._util import BlockId, ValueId, reverse_postorder_of
from ._const import BoolConst
from ._ir import (
    Block,
    Branch,
    Hir,
    Jump,
    Node,
    Operation,
    OutputPort,
    Phi,
    Ret,
    StateSlot,
    Terminator,
    predecessors,
    references,
    renumber,
    successors,
    validate_phi_predecessors,
)

_logger = logging.getLogger(__name__)


def _taking(blocks: list[Block], decided: BlockId, target: BlockId) -> list[Block]:
    return [
        Block(block.id, block.phis, block.operations, Jump(target)) if block.id == decided else block
        for block in blocks
    ]


def _proven_branch(hir: Hir) -> tuple[Block, BlockId] | None:
    for block in hir.blocks:
        terminator = block.terminator
        if isinstance(terminator, Branch) and isinstance(cond := hir.nodes[terminator.cond], BoolConst):
            return block, (terminator.if_true if cond.value else terminator.if_false)
    return None


def _trivial_phis(hir: Hir) -> dict[ValueId, ValueId]:
    """Every phi whose arms, apart from those naming the phi itself, all name one value, mapped to that value."""
    substitution: dict[ValueId, ValueId] = {}
    for block in hir.blocks:
        for phi_id in block.phis:
            phi = hir.nodes[phi_id]
            assert isinstance(phi, Phi)
            values = {value for _, value in phi.arms if value != phi_id}
            if len(values) == 1:
                (substitution[phi_id],) = values
    return substitution


def _resolve(substitution: dict[ValueId, ValueId], value: ValueId) -> ValueId:
    seen: set[ValueId] = set()
    while (target := substitution.get(value)) is not None:
        assert value not in seen, "phis that merge only one another are reached from nowhere else, so never reached"
        seen.add(value)
        value = target
    return value


def _fold(hir: Hir, substitution: dict[ValueId, ValueId]) -> Hir:
    def resolve(value: ValueId) -> ValueId:
        return _resolve(substitution, value)

    nodes: dict[ValueId, Node] = {}
    for vid, node in hir.nodes.items():
        match node:
            case Operation(operator=operator, operands=operands):
                nodes[vid] = Operation(operator=operator, operands=tuple(resolve(o) for o in operands))
            case Phi(type=type, arms=arms):
                if vid not in substitution:
                    nodes[vid] = Phi(type=type, arms=tuple((pred, resolve(value)) for pred, value in arms))
            case _:
                nodes[vid] = node

    def retarget(terminator: Terminator) -> Terminator:
        match terminator:
            case Jump():
                return terminator
            case Branch(cond=cond, if_true=if_true, if_false=if_false):
                return Branch(cond=resolve(cond), if_true=if_true, if_false=if_false)
            case Ret():
                return terminator
            case _:
                assert_never(terminator)

    blocks = [
        Block(
            block.id,
            tuple(p for p in block.phis if p not in substitution),
            block.operations,
            retarget(block.terminator),
        )
        for block in hir.blocks
    ]
    return replace(
        hir,
        nodes=nodes,
        blocks=blocks,
        outputs=[OutputPort(out.name, resolve(out.value)) for out in hir.outputs],
        state_slots=[StateSlot(s.name, s.reset_value, resolve(s.live_out)) for s in hir.state_slots],
    )


def _take(hir: Hir, decided: Block, target: BlockId) -> Hir:
    blocks = _taking(hir.blocks, decided.id, target)
    live = set(reverse_postorder_of(hir.entry, {block.id: successors(block) for block in blocks}))
    if not any(isinstance(block.terminator, Ret) for block in blocks if block.id in live):
        raise UnsupportedConstruct("the kernel provably never returns, so no output of it is ever raised")
    blocks = [block for block in blocks if block.id in live]
    preds = predecessors(blocks)
    nodes = dict(hir.nodes)
    for block in blocks:
        for phi_id in block.phis:
            phi = nodes[phi_id]
            assert isinstance(phi, Phi)
            arms = tuple((pred, value) for pred, value in phi.arms if pred in preds[block.id])
            assert arms, "a reachable non-entry block has a predecessor, and the entry block carries no phi"
            nodes[phi_id] = Phi(type=phi.type, arms=arms)
    # A reachable use cannot be dominated by an unreachable definition, so nothing that survives can name these.
    for block in hir.blocks:
        if block.id not in live:
            for vid in block.phis + block.operations:
                del nodes[vid]
    return replace(hir, nodes=nodes, blocks=blocks)


def _references(hir: Hir) -> list[ValueId]:
    return [*hir.external_value_references(), *(r for node in hir.nodes.values() for r in references(node))]


def run(hir: Hir) -> Hir | None:
    """
    Repeats because taking one edge can settle the next condition, and folding a merge can settle one too; bounded,
    since each pruning replaces a branch with a jump and each folding deletes a phi, and neither mints one. A merge
    that taking an edge leaves single-armed is folded before anything else. `None` reports that nothing changed.
    """
    pruned = folded = 0
    while True:
        if substitution := _trivial_phis(hir):
            hir = _fold(hir, substitution)
            folded += len(substitution)
        elif (decided := _proven_branch(hir)) is not None:
            hir = _take(hir, *decided)
            pruned += 1
        else:
            break
    if not pruned and not folded:
        return None
    _logger.info(
        "Constant-branch pruning: %d branch(es) proven, %d merge(s) folded; %d block(s) remain",
        pruned,
        folded,
        len(hir.blocks),
    )
    hir = renumber(hir)
    validate_phi_predecessors(hir)
    assert all(reference in hir.nodes for reference in _references(hir)), (
        "pruning named a value it deleted: either an unreachable definition reached a surviving use, or a folded "
        "merge was not substituted everywhere"
    )
    return hir
