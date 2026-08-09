"""
Constant-branch pruning: a branch whose condition the graph already names takes one edge, so the other edge and
everything only it reaches are not part of the program.

Nothing else deletes a control edge -- if-conversion refuses a decided diamond, DCE preserves the CFG -- so without
this a proven-dead arm reached hardware: selected, materialized, and refused upon, for code no input can reach.
Pruning and strength reduction feed each other, so they run as a pair to a fixpoint; see :mod:`._optimize`.
"""

import logging
from typing import assert_never

from .._util import BlockId, ValueId
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
    """
    The first block whose branch the graph decides, paired with the target it takes -- but only where taking it leaves
    the sole exit reachable. Orphaning the exit would rewrite a kernel that provably never returns into a module that
    can never raise ``out_valid``, so that one branch is left as written, and is then the only decided branch the
    survivor sweep's enterability walk has to step around.
    """
    exits = [block.id for block in hir.blocks if isinstance(block.terminator, Ret)]
    assert len(exits) == 1, "the frontend emits exactly one function exit, and no pass creates or deletes one"
    (exit_block,) = exits
    for block in hir.blocks:
        terminator = block.terminator
        if not (isinstance(terminator, Branch) and isinstance(cond := hir.nodes[terminator.cond], BoolConst)):
            continue
        target = terminator.if_true if cond.value else terminator.if_false
        if exit_block in _reachable(_taking(hir.blocks, block.id, target), hir.entry):
            return block, target
    return None


def _reachable(blocks: list[Block], entry: BlockId) -> set[BlockId]:
    by_id = {block.id: block for block in blocks}
    seen = {entry}
    stack = [entry]
    while stack:
        for target in successors(by_id[stack.pop()]):
            if target not in seen:
                seen.add(target)
                stack.append(target)
    return seen


def _resolve(substitution: dict[ValueId, ValueId], value: ValueId) -> ValueId:
    """Follow a chain of collapsed phis: one may stand for another, and the chain stands for what is at its end."""
    seen: set[ValueId] = set()
    while (target := substitution.get(value)) is not None:
        assert value not in seen, "a phi standing for itself is a block whose only predecessor is itself"
        seen.add(value)
        value = target
    return value


def _take(hir: Hir, decided: Block, target: BlockId) -> Hir:
    """Take the proven edge out of ``decided``, then delete whatever that leaves unreachable and repair the merges."""
    blocks = _taking(hir.blocks, decided.id, target)
    live = _reachable(blocks, hir.entry)
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

    return Hir(
        nodes=nodes,
        blocks=[Block(b.id, b.phis, b.operations, retarget(b.terminator)) for b in repaired],
        input_ids=hir.input_ids,
        outputs=[OutputPort(out.name, resolve(out.value)) for out in hir.outputs],
        state_slots=[StateSlot(s.name, s.reset_value, resolve(s.live_out)) for s in hir.state_slots],
    )


def _references(hir: Hir) -> list[ValueId]:
    """Every value the graph names, from inside the DAG and from outside it -- the complete use set."""
    refs = list(hir.external_value_references())
    for node in hir.nodes.values():
        match node:
            case Operation(operands=operands):
                refs.extend(operands)
            case Phi(arms=arms):
                refs.extend(value for _pred, value in arms)
            case _:
                pass
    return refs


def run(hir: Hir) -> Hir | None:
    """
    Prune every branch the graph decides, repeating because taking one edge can settle the next condition. ``None``
    means none was decided, which is the only thing that ends the reduce/prune fixpoint in :mod:`._optimize`. Each
    pruning replaces a branch with a jump and never mints one, so the repetition is bounded by the branch count.
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
