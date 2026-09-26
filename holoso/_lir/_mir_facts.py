"""
Read-only structural and control-flow facts over MIR. They depend on nothing in the LIR layer, so they sit at the base
of the builder DAG, shared by construction, layout, and bank allocation without coupling those stages to one another.
"""

from .._mir import Mir, MirOperation, MirPhi, successors
from .._util import ValueId


def mir_operation(mir: Mir, vid: ValueId) -> MirOperation:
    node = mir.nodes[vid]
    assert isinstance(node, MirOperation)
    return node


def succ_map(mir: Mir) -> dict[int, list[int]]:
    return {block.id: successors(block) for block in mir.blocks}


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
