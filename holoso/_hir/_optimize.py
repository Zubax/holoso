"""HIR optimization pipeline."""

from . import _const_fold, _dce, _if_convert, _strength_reduce
from ._ir import Hir


def optimize(hir: Hir) -> Hir:
    """
    Run all hardware-agnostic HIR optimizations. If-conversion runs after folding/strength reduction (the constant
    conditions it must refuse are the ones const-fold materialized -- a condition the frontend could prove never
    emitted a branch at all -- and arm costs are final) and before DCE (which sweeps a converted diamond's condition
    cone when nothing else reads it).
    """
    return _dce.run(_if_convert.run(_strength_reduce.run(_const_fold.run(hir))))
