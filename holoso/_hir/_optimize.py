"""HIR optimization pipeline."""

from . import _const_fold, _dce, _if_convert, _strength_reduce, _thread_merges
from ._ir import Hir


def optimize(hir: Hir) -> Hir:
    """
    Run all hardware-agnostic HIR optimizations. If-conversion runs after folding/strength reduction (the constant
    conditions it must refuse are the ones const-fold materialized -- a condition the frontend could prove never
    emitted a branch at all -- and arm costs are final). Folding and strength reduction then run a SECOND time, to
    reduce the muxes if-conversion created: a boolean ``bool_select`` with constant arms collapses to ``and``/``or``/
    ``not``/passthrough, and a ``select`` with identical arms drops out. The re-run also re-interns the nodes the
    splice wrote directly into the graph. Merge threading then eliminates the empty pass-through merge blocks a non-
    convertible diamond leaves when its merge feeds a following control structure, deleting its own composed-away merge
    phis. DCE runs last (it sweeps a converted diamond's condition cone when nothing else reads it and any operands the
    mux reductions left dead).
    """
    hir = _strength_reduce.run(_const_fold.run(hir))
    hir = _if_convert.run(hir)
    hir = _strength_reduce.run(_const_fold.run(hir))
    hir = _thread_merges.run(hir)
    return _dce.run(hir)
