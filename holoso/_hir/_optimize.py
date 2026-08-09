"""HIR optimization pipeline."""

from . import _dce, _if_convert, _prune, _strength_reduce, _thread_merges
from ._ir import Hir


def optimize(hir: Hir, ifconv_max_ops: int) -> Hir:
    """
    Run all hardware-agnostic HIR optimizations to a fixpoint. Every pass is another's input, so no fixed ordering is
    right and the round simply repeats until it leaves the graph untouched. The budget makes a pair of rewrites that
    oscillate a crash rather than a hang.

    This reduces, and the only build it declines is one that pruning proves never returns.
    """
    rounds = 2 * (len(hir.blocks) + len(hir.nodes)) + 2
    while True:
        rounds -= 1
        assert rounds > 0, "the optimization round is oscillating rather than converging"
        previous = hir
        hir = _strength_reduce.run(hir)
        hir = _prune.run(hir) or hir
        hir = _if_convert.run(hir, ifconv_max_ops) or hir
        hir = _thread_merges.run(hir) or hir
        hir = _dce.run(hir)
        if hir == previous:
            return hir
