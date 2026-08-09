"""HIR optimization pipeline."""

import logging

from . import _dce, _if_convert, _prune, _strength_reduce, _thread_merges
from ._ir import Hir

_logger = logging.getLogger(__name__)


def optimize(hir: Hir, ifconv_max_ops: int) -> Hir:
    """
    Run all hardware-agnostic HIR optimizations to a fixpoint. Every pass is another's input, so no fixed ordering is
    right and the round simply repeats until it leaves the graph untouched. The budget asserts that convergence
    instead of trusting it, since an oscillating pair of rewrites would otherwise hang.

    This reduces, and the only build it declines is one that pruning proves never returns.
    """
    budget = 2 * (len(hir.blocks) + len(hir.nodes)) + 2
    spent = 0
    while True:
        spent += 1
        assert spent <= budget, "the optimization round is oscillating rather than converging"
        previous = hir
        hir = _strength_reduce.run(hir)
        hir = _prune.run(hir) or hir
        hir = _if_convert.run(hir, ifconv_max_ops) or hir
        hir = _thread_merges.run(hir) or hir
        hir = _dce.run(hir)
        if hir == previous:
            _logger.info(
                "HIR optimization: settled after %d of at most %d round(s); %d block(s), %d node(s) remain",
                spent,
                budget,
                len(hir.blocks),
                len(hir.nodes),
            )
            return hir
