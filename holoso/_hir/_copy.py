"""
Shared HIR rewrite helpers: a CFG-aware rebuild driver for passes that keep the block structure; if-conversion,
which dissolves blocks, rebuilds by hand.
"""

from collections.abc import Callable, Set
from typing import assert_never

from ._const import Const
from .._util import BlockId, ValueId
from ._ir import (
    Branch,
    Hir,
    HirBuilder,
    InPort,
    Jump,
    Node,
    Operation,
    Phi,
    Ret,
    StateRead,
    Terminator,
)

# A pass supplies one of these to rebuild each value into the target builder; it returns the new value id and may fold
# an operation into a constant. The builder is already positioned at the value's block.
type BuildValue = Callable[[HirBuilder, ValueId, Node, dict[ValueId, ValueId]], ValueId]


def copy_node(builder: HirBuilder, node: Node, remap: dict[ValueId, ValueId]) -> ValueId:
    """Block ids are preserved across a rebuild, so a phi's predecessor ids need no remapping (only arm values do)."""
    match node:
        case InPort(name=name, type=type):
            return builder.input(name, type)
        case StateRead(slot=slot, type=type):
            return builder.state_read(slot, type)
        case Const():
            return builder.const_node(node)
        case Operation(operator=operator, operands=operands):
            return builder.operation(operator, [remap[operand] for operand in operands])
        case Phi(type=type, arms=arms):
            return builder.phi(type, [(pred, remap[value]) for pred, value in arms])
        case _ as unreachable:
            assert_never(unreachable)


def copy_state_slots(builder: HirBuilder, hir: Hir, remap: dict[ValueId, ValueId]) -> None:
    """Re-emit every persistent state slot with its live-out value remapped (slots survive every rewrite pass)."""
    for slot in hir.state_slots:
        builder.state_slot(slot.name, slot.reset_value, remap[slot.live_out])


def _remap_terminator(terminator: Terminator, remap: dict[ValueId, ValueId]) -> Terminator:
    match terminator:
        case Jump():
            return terminator
        case Branch(cond=cond, if_true=if_true, if_false=if_false):
            return Branch(remap[cond], if_true, if_false)
        case Ret():
            return terminator
        case _ as unreachable:
            assert_never(unreachable)


def _globals_in_order(hir: Hir) -> list[ValueId]:
    """Entry-global pure values (constants and state reads) in id (creation) order; inputs are handled separately."""
    return [vid for vid in sorted(hir.nodes) if isinstance(hir.nodes[vid], (Const, StateRead))]


def reverse_postorder(hir: Hir) -> list[BlockId]:
    """
    Block ids in reverse-postorder of the CFG from the entry. Predecessors precede successors (a back-edge target
    precedes its body), so visiting blocks in this order remaps every operand and forward phi arm before its use --
    numeric id order does not (a nested ``if``'s merge gets a higher id than the outer merge it feeds).
    """
    successors: dict[BlockId, list[BlockId]] = {}
    for block in hir.blocks:
        match block.terminator:
            case Jump(target=target):
                successors[block.id] = [target]
            case Branch(if_true=if_true, if_false=if_false):
                successors[block.id] = [if_true, if_false]
            case Ret():
                successors[block.id] = []
    order: list[BlockId] = []
    visited: set[BlockId] = set()
    # Iterative DFS (explicit stack) rather than recursion: a deep CFG -- e.g. nested unrolled loops, which chain
    # thousands of blocks -- would otherwise exceed Python's recursion limit. Each frame is (node, child_index); a node
    # is appended to the postorder once all its successors have been emitted.
    stack: list[tuple[BlockId, int]] = [(hir.entry, 0)]
    visited.add(hir.entry)
    while stack:
        node, index = stack[-1]
        succs = successors[node]
        if index < len(succs):
            stack[-1] = (node, index + 1)
            successor = succs[index]
            if successor not in visited:
                visited.add(successor)
                stack.append((successor, 0))
        else:
            order.append(node)
            stack.pop()
    return order[::-1]


def rebuild(hir: Hir, build_value: BuildValue, keep: Set[ValueId] | None = None) -> Hir:
    """
    Rebuild ``hir`` into a fresh :class:`HirBuilder`, delegating each value's construction to ``build_value`` and
    copying the CFG structure (blocks, phis, terminators) generically. Block ids are preserved. Values are visited in a
    dominance-respecting order -- inputs, then entry-global constants/state reads, then each block's phis and
    operations in reverse-postorder of the blocks -- so operands and phi arms are already remapped when used. The one
    exception is a loop-header phi's latch (back-edge) arm, which references a body value defined later in the order:
    such a phi is copied open with its available arms (``build_value`` is bypassed, as the pass cannot fold an
    incomplete merge) and its arms are closed once every block has been visited. ``keep`` (DCE) restricts which
    non-input values are emitted; dropped values must not be referenced by any kept value.
    """
    builder = HirBuilder()
    remap: dict[ValueId, ValueId] = {}
    for _ in hir.blocks:
        builder.block()  # recreate block ids 0..n-1 in order, preserving them
    entry = hir.entry
    deferred: list[ValueId] = []  # loop-header phis with a forward-referenced latch arm, closed after every block

    def emit(vid: ValueId) -> None:
        remap[vid] = build_value(builder, vid, hir.nodes[vid], remap)

    builder.position_at(entry)
    for vid in hir.input_ids:
        emit(vid)
    for vid in _globals_in_order(hir):
        if keep is None or vid in keep:
            emit(vid)
    blocks_by_id = {block.id: block for block in hir.blocks}
    for bid in reverse_postorder(hir):
        block = blocks_by_id[bid]
        builder.position_at(bid)
        for vid in block.phis:
            if keep is not None and vid not in keep:
                continue
            node = hir.nodes[vid]
            assert isinstance(node, Phi)
            if all(arm in remap for _, arm in node.arms):
                emit(vid)
            else:  # a loop-header phi: open it with the preheader arm now, close the latch arm after all blocks
                known = [(pred, remap[arm]) for pred, arm in node.arms if arm in remap]
                remap[vid] = builder.open_phi(node.type, known[0])
                deferred.append(vid)
        for vid in block.operations:
            if keep is None or vid in keep:
                emit(vid)
        builder.set_terminator(bid, _remap_terminator(block.terminator, remap))
    for vid in deferred:
        node = hir.nodes[vid]
        assert isinstance(node, Phi)
        builder.set_phi_arms(remap[vid], [(pred, remap[arm]) for pred, arm in node.arms])
    for out in hir.outputs:
        builder.output(out.name, remap[out.value])
    copy_state_slots(builder, hir, remap)
    return builder.finish()
