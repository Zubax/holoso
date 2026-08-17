"""
Pure read-only structural and CFG facts over a MIR graph: the MIR node accessor, the phi-arm liveness / const-branch /
install-bearing-block shape predicates, and the block reverse-postorder and successor maps. These emit no LIR and
depend on nothing in the LIR layer, so they sit at the base of the builder DAG -- shared by construction, layout, and
bank allocation without coupling those stages to one another.
"""

from .._mir import (
    Mir,
    MirBoolView,
    MirBranch,
    MirJump,
    MirOperation,
    MirPhi,
    MirRet,
    MirWideView,
)
from .._util import ValueId


def mir_operation(mir: Mir, vid: ValueId) -> MirOperation:
    node = mir.nodes[vid]
    assert isinstance(node, MirOperation)
    return node


def succ_map(mir: Mir) -> dict[int, list[int]]:
    succ: dict[int, list[int]] = {}
    for block in mir.blocks:
        match block.terminator:
            case MirJump(target=target):
                succ[block.id] = [target]
            case MirBranch(if_true=if_true, if_false=if_false):
                succ[block.id] = [if_true, if_false]
            case MirRet():
                succ[block.id] = []
    return succ


def pred_count(mir: Mir) -> dict[int, int]:
    """
    Predecessor EDGE count per block (a both-arms-same-target branch counts twice): the multi-predecessor fact that
    gates cross-block overlap in the layout and underpins the phi-residency premise checked by the builder.
    """
    count: dict[int, int] = {block.id: 0 for block in mir.blocks}
    for targets in succ_map(mir).values():
        for target in targets:
            count[target] += 1
    return count


def mir_rpo(mir: Mir) -> list[int]:
    """Reverse-postorder of the MIR block CFG from the entry (predecessors before successors)."""
    successors = succ_map(mir)
    order: list[int] = []
    visited: set[int] = set()
    # Iterative DFS (explicit stack) rather than recursion: a deep CFG (e.g. nested unrolled loops chaining thousands
    # of blocks) would otherwise exceed Python's recursion limit. A node is emitted once all its successors are done.
    stack: list[tuple[int, int]] = [(mir.entry, 0)]
    visited.add(mir.entry)
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


def phi_arm_out(mir: Mir, phi_nodes: dict[ValueId, MirPhi], values: set[ValueId]) -> dict[int, frozenset[ValueId]]:
    """
    Per block, the phi-arm values live out of it (each read by the phi's install copy at the block's tail) -- a liveness
    input for both banks. The residual installs (which phi registers a block writes) are derived separately, per chosen
    coalescing, by _residual_installs, so they are not precomputed here.
    """
    arm_out: dict[int, set[ValueId]] = {block.id: set() for block in mir.blocks}
    for _vid, phi in phi_nodes.items():
        for pred, arm, _conditioner in phi.arms:
            if arm in values:
                arm_out[pred].add(arm)
    return {b: frozenset(s) for b, s in arm_out.items()}


def const_branch_conditions(mir: Mir, bool_mir: MirBoolView) -> dict[int, ValueId]:
    """
    Per block, the constant branch condition it materializes at its tail. A block whose `MirBranch` tests a globally
    interned boolean constant has no condition register, so the constant is written into a bool register in the
    branching block. The single source of this CFG-shape fact, shared by `block_has_install` (whose key universe the
    install fixpoint asserts every post-allocation classification stays within) and the allocator's materialization
    (which also needs the condition value to write).
    """
    conditions: dict[int, ValueId] = {}
    for block in mir.blocks:
        term = block.terminator
        if isinstance(term, MirBranch) and term.cond in bool_mir.const_nodes:
            conditions[block.id] = term.cond
    return conditions


def block_has_install(mir: Mir, wide_mir: MirWideView, bool_mir: MirBoolView) -> dict[int, bool]:
    """
    Each install-bearing block mapped to whether it carries an install whose source is the block's OWN operation --
    the only source kind that can commit at the work makespan and push the install one step past it
    (`install_issue_cycle`); everything else (a constant, including a const branch condition, an input, a state
    read, a phi, or a value computed in another block) never pushes. This is the CONSERVATIVE seed for that +1: an
    arm is assumed not to coalesce, so a local-source arm marks its block even if it later coalesces away, and the
    fixpoint's `actual_install_blocks` narrows the bit to the blocks whose source really is the last work, once the
    schedule is known. The liveness boundary and the layout share this classification so the per-block makespan and
    drain agree.
    """
    local_ops = {block.id: set(block.operations) for block in mir.blocks}
    install: dict[int, bool] = {}
    for phi_nodes in (wide_mir.phi_nodes, bool_mir.phi_nodes):
        for phi in phi_nodes.values():
            for pred, value, _conditioner in phi.arms:
                install[pred] = install.get(pred, False) or value in local_ops[pred]
    for block_id in const_branch_conditions(mir, bool_mir):
        install.setdefault(block_id, False)  # a const branch materializes a literal: a settled source, never pushing
    return install
