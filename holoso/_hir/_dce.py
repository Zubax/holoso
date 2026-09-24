"""HIR dead-code elimination."""

from ._copy import rebuild
from .._util import ValueId
from ._ir import Hir, references


def eliminate_dead_code(hir: Hir) -> Hir:
    """
    Drop values unreachable from any output, persistent state, or branch condition; inputs are kept as the module
    signature. Block structure is preserved (a structured CFG has no dead blocks at this stage).
    """
    kept: set[ValueId] = set()
    pending = hir.external_value_references()
    while pending:
        vid = pending.pop()
        if vid not in kept:
            kept.add(vid)
            pending.extend(references(hir.nodes[vid]))
    return rebuild(hir, keep=kept)
