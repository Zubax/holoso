"""HIR optimization pipeline."""

from . import _dce, _if_convert, _prune, _strength_reduce, _thread_merges
from ._ir import Hir


def _reduce_and_prune(hir: Hir) -> Hir:
    """
    A mutual fixpoint rather than a sequence: reduction materializes the constants pruning decides on, and a pruned
    edge leaves merges the next reduction folds. Bounded by the branch count, since neither mints a branch. On exit
    the only branch the graph can still decide is one whose taken edge would orphan the exit.
    """
    while True:
        hir = _strength_reduce.run(hir)
        pruned = _prune.run(hir)
        if pruned is None:
            return hir
        hir = pruned


def optimize(hir: Hir, ifconv_max_ops: int) -> Hir:
    """
    Run all hardware-agnostic HIR optimizations. If-conversion sits between the two fixpoints, where no diamond it
    could see is still decidable and arm costs are final; the second fixpoint then reduces the muxes it created and
    re-interns the nodes the splice wrote directly into the graph. Merge threading eliminates the empty pass-through
    merge blocks a non-convertible diamond leaves when its merge feeds a following control structure.

    This reduces and never refuses; what survives is judged at the HIR-to-MIR boundary.
    """
    hir = _reduce_and_prune(hir)
    hir = _if_convert.run(hir, ifconv_max_ops)
    hir = _reduce_and_prune(hir)
    hir = _thread_merges.run(hir)
    return _dce.run(hir)
