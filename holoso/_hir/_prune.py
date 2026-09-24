"""
Constant-branch pruning, the only pass that deletes a control edge. Without it a proven-dead arm reaches hardware --
selected, materialized, and refused upon -- for code no input can reach.
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


def _resolve(substitution: dict[ValueId, ValueId], value: ValueId) -> ValueId:
    seen: set[ValueId] = set()
    while (target := substitution.get(value)) is not None:
        assert value not in seen, "a phi standing for itself is a block whose only predecessor is itself"
        seen.add(value)
        value = target
    return value


def _take(hir: Hir, decided: Block, target: BlockId) -> Hir:
    blocks = _taking(hir.blocks, decided.id, target)
    live = set(reverse_postorder_of(hir.entry, {block.id: successors(block) for block in blocks}))
    if not any(isinstance(block.terminator, Ret) for block in blocks if block.id in live):
        raise UnsupportedConstruct("the kernel provably never returns, so no output of it is ever raised")
    blocks = [block for block in blocks if block.id in live]
    preds = predecessors(blocks)
    nodes = dict(hir.nodes)

    substitution: dict[ValueId, ValueId] = {}
    repaired: list[Block] = []
    for block in blocks:
        surviving: list[ValueId] = []
        for phi_id in block.phis:
            phi = nodes[phi_id]
            assert isinstance(phi, Phi)
            arms = tuple((pred, value) for pred, value in phi.arms if pred in preds[block.id])
            assert arms, "a reachable non-entry block has a predecessor, and the entry block carries no phi"
            if len(arms) == 1:
                substitution[phi_id] = arms[0][1]
                del nodes[phi_id]
            else:
                nodes[phi_id] = Phi(type=phi.type, arms=arms)
                surviving.append(phi_id)
        repaired.append(Block(block.id, tuple(surviving), block.operations, block.terminator))

    # A reachable use cannot be dominated by an unreachable definition, so nothing that survives can name these.
    for block in hir.blocks:
        if block.id not in live:
            for vid in block.phis + block.operations:
                del nodes[vid]

    def resolve(value: ValueId) -> ValueId:
        return _resolve(substitution, value)

    for vid, node in list(nodes.items()):
        match node:
            case Operation(operator=operator, operands=operands):
                nodes[vid] = Operation(operator=operator, operands=tuple(resolve(o) for o in operands))
            case Phi(type=type, arms=arms):
                nodes[vid] = Phi(type=type, arms=tuple((pred, resolve(value)) for pred, value in arms))
            case _:
                pass

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

    return replace(
        hir,
        nodes=nodes,
        blocks=[Block(b.id, b.phis, b.operations, retarget(b.terminator)) for b in repaired],
        outputs=[OutputPort(out.name, resolve(out.value)) for out in hir.outputs],
        state_slots=[StateSlot(s.name, s.reset_value, resolve(s.live_out)) for s in hir.state_slots],
    )


def _references(hir: Hir) -> list[ValueId]:
    return [*hir.external_value_references(), *(r for node in hir.nodes.values() for r in references(node))]


def run(hir: Hir) -> Hir | None:
    """
    Repeats because taking one edge can settle the next condition; bounded, since each pruning replaces a branch
    with a jump and never mints one. `None` reports that nothing was pruned.
    """
    pruned = 0
    while (decided := _proven_branch(hir)) is not None:
        block, target = decided
        hir = _take(hir, block, target)
        pruned += 1
    if not pruned:
        return None
    _logger.info("Constant-branch pruning: %d branch(es) proven; %d block(s) remain", pruned, len(hir.blocks))
    hir = renumber(hir)
    validate_phi_predecessors(hir)
    assert all(reference in hir.nodes for reference in _references(hir)), (
        "pruning named a value it deleted: either an unreachable definition reached a surviving use, or a collapsed "
        "merge was not substituted everywhere"
    )
    return hir
