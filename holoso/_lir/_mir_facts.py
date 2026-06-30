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
    MirConst,
    MirFloatView,
    MirInput,
    MirJump,
    MirNode,
    MirOperation,
    MirPhi,
    MirRet,
    MirStateRead,
)
from .._util import ValueId


def mir_operation(mir: Mir, vid: ValueId) -> MirOperation:
    node = mir.nodes[vid]
    assert isinstance(node, MirOperation)
    return node


def value_resident_at_entry(node: MirNode) -> bool:
    """
    Whether a value is resident in its register from a block's first step rather than produced by the block's own work:
    a literal constant, an input load, or a persistent-state read. A value computed in some block (an operator result or
    a phi) is NOT entry-resident -- it lands mid/late-block. One input to a tail install's PLACEMENT: a resident source
    is available from block entry, so the install sits at the work makespan and pays no terminator cycle; a computed
    source sits there too but is pushed one step when it is the block's own last-committing work (decided by its commit
    cycle in ``install_issue_cycle``). The lone source of this residency fact, shared by the install seed, the
    post-coalescing refinement, the builder, and the allocator residence, so they cannot drift. The positive test means
    a future node kind defaults to non-resident -- the safe direction (the conservative computed-source treatment).
    """
    return isinstance(node, (MirConst, MirInput, MirStateRead))


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
    coalescing, by :func:`_residual_installs`, so they are not precomputed here.
    """
    arm_out: dict[int, set[ValueId]] = {block.id: set() for block in mir.blocks}
    for _vid, phi in phi_nodes.items():
        for pred, arm, _conditioner in phi.arms:
            if arm in values:
                arm_out[pred].add(arm)
    return {b: frozenset(s) for b, s in arm_out.items()}


def const_branch_conditions(mir: Mir, bool_mir: MirBoolView) -> dict[int, ValueId]:
    """
    Per block, the constant branch condition it materializes at its tail. A block whose ``MirBranch`` tests a globally
    interned boolean constant has no condition register, so the constant is written into a bool register in the
    branching block. The single source of this CFG-shape fact, shared by ``block_has_install`` and the install fixpoint
    seed (which must agree, or the monotonicity assert trips) and the allocator's materialization (which also needs the
    condition value to write).
    """
    conditions: dict[int, ValueId] = {}
    for block in mir.blocks:
        term = block.terminator
        if isinstance(term, MirBranch) and term.cond in bool_mir.const_nodes:
            conditions[block.id] = term.cond
    return conditions


def block_has_install(mir: Mir, float_mir: MirFloatView, bool_mir: MirBoolView) -> dict[int, bool]:
    """
    Each install-bearing block mapped to whether it carries a COMPUTED-source install -- a float copy or a bool write
    whose source is produced by the block's own work (an operator result or phi) rather than block-entry RESIDENT (a
    literal constant, including a phi-arm const arm or a const branch condition, an input, or a state read). This is the
    CONSERVATIVE seed for the makespan +1: a computed source MIGHT be the block's own last work, which the install must
    fire one step past to read-first (pushing the makespan); the fixpoint's ``actual_install_blocks`` narrows it to the
    blocks that actually push, once the schedule is known. Determinable from the MIR shape before register assignment
    (``value_resident_at_entry``), conservatively: an arm is assumed not to coalesce, so a computed-source arm marks the
    block even if it later coalesces away (the fixpoint then narrows it). The liveness boundary and the layout share
    this classification so the per-block makespan and drain agree.
    """
    install: dict[int, bool] = {}
    for phi in float_mir.phi_nodes.values():
        for pred, value, _conditioner in phi.arms:
            install[pred] = install.get(pred, False) or not value_resident_at_entry(float_mir.nodes[value])
    for phi in bool_mir.phi_nodes.values():
        for pred, value, _conditioner in phi.arms:
            install[pred] = install.get(pred, False) or not value_resident_at_entry(bool_mir.nodes[value])
    for block_id in const_branch_conditions(mir, bool_mir):
        install.setdefault(block_id, False)  # a const branch materializes a literal: a resident source, never pushing
    return install
