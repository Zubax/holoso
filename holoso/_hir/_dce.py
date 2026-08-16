"""HIR dead-code elimination."""

from ._copy import copy_node, rebuild
from .._util import ValueId
from ._ir import Hir, Operation, Phi


def eliminate_dead_code(hir: Hir) -> Hir:
    """
    Drop values unreachable from any output, persistent state, or branch condition; inputs are kept as the module
    signature. Block structure is preserved (a structured CFG has no dead blocks at this stage).
    """
    reachable: set[ValueId] = set()
    stack = hir.external_value_references()
    while stack:
        vid = stack.pop()
        if vid in reachable:
            continue
        reachable.add(vid)
        match hir.nodes[vid]:
            case Operation(operands=operands):
                stack.extend(operands)
            case Phi(arms=arms):
                stack.extend(value for _, value in arms)
            case _:
                pass
    return rebuild(hir, copy_node, keep=reachable | set(hir.input_ids))
