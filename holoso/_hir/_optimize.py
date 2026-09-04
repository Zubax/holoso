"""HIR optimization pipeline."""

import logging

from . import _dce, _if_convert, _linear, _prune, _reciprocal, _strength_reduce, _thread_merges
from ._ir import Hir

_logger = logging.getLogger(__name__)


def optimize(hir: Hir, ifconv_max_ops: int) -> Hir:
    """
    Run all hardware-agnostic HIR optimizations to a fixpoint. Every pass is another's input, so no fixed ordering is
    right and the round simply repeats until it leaves the graph untouched. The budget asserts that convergence
    instead of trusting it, since an oscillating pair of rewrites would otherwise hang.

    The sharing passes wait for the round to SETTLE, being the only ones that adopt a value on behalf of another
    and so the only ones a value about to die can mislead. Settled, not merely swept once: an if-converted diamond
    leaves a select over two equal arms that the NEXT round folds, and only then does its condition's cone die.
    Each is swept after, so the one behind it reads a liveness its own rewrites have not left stale.

    This reduces, and the only build it declines is one that pruning proves never returns.
    """
    # A sharing rewrite deletes an addition or a division, so it fires at most once per node, each firing costing
    # a round to canonicalize and another to settle again.
    budget = 4 * (len(hir.blocks) + len(hir.nodes)) + 2
    spent = 0
    while True:
        spent += 1
        assert spent <= budget, "the optimization round is oscillating rather than converging"
        previous = hir
        hir = _strength_reduce.run(hir)
        hir = _prune.run(hir) or hir
        hir = _if_convert.run(hir, ifconv_max_ops) or hir
        hir = _thread_merges.run(hir) or hir
        hir = _dce.eliminate_dead_code(hir)
        if hir != previous:
            continue
        for share in (_linear.run, _reciprocal.run):
            # Swept between the two: the second would otherwise read a liveness the first has just invalidated.
            hir = _dce.eliminate_dead_code(share(hir))
        if hir == previous:
            _logger.info(
                "HIR optimization: settled after %d of at most %d round(s); %d block(s), %d node(s) remain",
                spent,
                budget,
                len(hir.blocks),
                len(hir.nodes),
            )
            return hir
