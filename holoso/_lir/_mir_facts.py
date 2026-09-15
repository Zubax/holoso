"""
Read-only structural and control-flow facts over MIR. They depend on nothing in the LIR layer, so they sit at the base
of the builder DAG, shared by construction, layout, and bank allocation without coupling those stages to one another.
"""

from .._mir import (
    Mir,
    MirBoolView,
    MirOperation,
    MirPhi,
    MirWideView,
    successors,
)
from .._util import ValueId


def mir_operation(mir: Mir, vid: ValueId) -> MirOperation:
    node = mir.nodes[vid]
    assert isinstance(node, MirOperation)
    return node


def succ_map(mir: Mir) -> dict[int, list[int]]:
    return {block.id: successors(block) for block in mir.blocks}


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


def phi_arm_out(mir: Mir, phi_nodes: dict[ValueId, MirPhi], values: set[ValueId]) -> dict[int, frozenset[ValueId]]:
    """
    Per block, the phi-arm values live out of it because the phi's install copy reads them at the block's tail. The
    residual installs (which phi registers a block writes) depend on the chosen coalescing, so `_residual_installs`
    derives them instead.
    """
    arm_out: dict[int, set[ValueId]] = {block.id: set() for block in mir.blocks}
    for _vid, phi in phi_nodes.items():
        for pred, arm, _conditioner in phi.arms:
            if arm in values:
                arm_out[pred].add(arm)
    return {b: frozenset(s) for b, s in arm_out.items()}


def block_has_install(mir: Mir, wide_mir: MirWideView, bool_mir: MirBoolView) -> dict[int, bool]:
    """
    Each install-bearing block mapped to whether it carries an install whose source is the block's OWN operation -- the
    only source kind that can commit at the work makespan and push the install one step past it (`install_issue_cycle`);
    everything else (a constant, an input, a state read, a phi, or a value computed in another block) never pushes. This
    is the CONSERVATIVE seed for that +1: an arm is assumed not to coalesce, so a local-source arm marks its block even
    if it later coalesces away, and the fixpoint's `CoalescedLayout.install_blocks` narrows the bit to the blocks whose
    source really is the last work, once the schedule is known. The liveness boundary and the layout share this
    classification so the per-block makespan and drain agree.
    """
    local_ops = {block.id: set(block.operations) for block in mir.blocks}
    install: dict[int, bool] = {}
    for phi_nodes in (wide_mir.phi_nodes, bool_mir.phi_nodes):
        for phi in phi_nodes.values():
            for pred, value, _conditioner in phi.arms:
                install[pred] = install.get(pred, False) or value in local_ops[pred]
    return install
