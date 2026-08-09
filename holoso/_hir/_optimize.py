"""HIR optimization pipeline."""

from . import _dce, _if_convert, _prune, _refuse_nameless, _strength_reduce, _thread_merges
from ._ir import Hir


def _reduce_and_prune(hir: Hir) -> Hir:
    """
    Strength reduction folds and rewrites in one reverse-postorder walk, so a chain of dependent reductions collapses
    in a single pass; pruning then consumes the branch conditions among the constants it materialized, and a pruned
    edge leaves merges the next reduction can fold. The two are therefore a mutual fixpoint rather than a sequence,
    bounded by the branch count since neither mints a branch.

    Exiting only on a pruning that found nothing is the post-condition: the sole branch the graph can still decide is
    one whose taken edge would orphan the exit.
    """
    while True:
        hir = _strength_reduce.run(hir)
        pruned = _prune.run(hir)
        if pruned is None:
            return hir
        hir = pruned


def optimize(hir: Hir, ifconv_max_ops: int) -> Hir:
    """
    Run all hardware-agnostic HIR optimizations. If-conversion runs between the two reduce/prune fixpoints, where no
    diamond it could see is still decidable and arm costs are final. The second fixpoint reduces the muxes it created
    -- a boolean ``bselect`` with constant
    arms collapses to ``band``/``bor``/``bnot``/passthrough, and an ``fselect`` with identical arms drops out -- and
    re-interns the nodes the splice wrote directly into the graph. Merge threading eliminates the empty pass-through
    merge blocks a non-convertible diamond leaves when its merge feeds a following control structure, deleting its own
    composed-away merge phis. DCE follows (it sweeps any operands the mux reductions left dead).

    The survivor refusal sweep closes the pipeline and changes nothing, because SURVIVING every deletion is what it
    tests: an expression naming no number is the kernel's only once no identity erased it, no guard excluded it, and
    nothing left it dead. Running it any earlier would convict the compiler of expressions its own unrolling and
    inlining substituted into the graph.
    """
    hir = _reduce_and_prune(hir)
    hir = _if_convert.run(hir, ifconv_max_ops)
    hir = _reduce_and_prune(hir)
    hir = _thread_merges.run(hir)
    hir = _dce.run(hir)
    _refuse_nameless.run(hir)
    return hir
