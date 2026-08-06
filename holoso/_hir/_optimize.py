"""HIR optimization pipeline."""

from . import _dce, _if_convert, _refuse_nameless, _strength_reduce, _thread_merges
from ._ir import Hir


def optimize(hir: Hir, ifconv_max_ops: int) -> Hir:
    """
    Run all hardware-agnostic HIR optimizations. Strength reduction folds and rewrites in one reverse-postorder walk,
    so a chain of dependent reductions collapses in a single pass. If-conversion runs after it (the constant conditions
    it must refuse are the ones folding materialized, and arm costs are final). Strength reduction then runs a SECOND
    time, to reduce the muxes if-conversion created: a boolean ``bselect`` with constant arms collapses to
    ``band``/``bor``/``bnot``/passthrough, and an ``fselect`` with identical arms drops out. The re-run also re-interns
    the nodes the splice wrote directly into the graph. Merge threading eliminates the empty pass-through merge blocks
    a non-convertible diamond leaves when its merge feeds a following control structure, deleting its own composed-away
    merge phis. DCE follows (it sweeps a converted diamond's condition cone when nothing else reads it and any operands
    the mux reductions left dead).

    The survivor refusal sweep closes the pipeline and changes nothing, because SURVIVING every deletion is what it
    tests: an expression naming no number is the kernel's only once no identity erased it, no guard excluded it, and
    nothing left it dead. Running it any earlier would convict the compiler of expressions its own unrolling and
    inlining substituted into the graph.
    """
    hir = _strength_reduce.run(hir)
    hir = _if_convert.run(hir, ifconv_max_ops)
    hir = _strength_reduce.run(hir)
    hir = _thread_merges.run(hir)
    hir = _dce.run(hir)
    _refuse_nameless.run(hir)
    return hir
