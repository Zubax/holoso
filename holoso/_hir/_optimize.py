"""HIR optimization pipeline."""

from . import _const_fold, _dce, _strength_reduce
from ._ir import Hir


def optimize(hir: Hir) -> Hir:
    """Run all hardware-agnostic HIR optimizations."""
    return _dce.run(_strength_reduce.run(_const_fold.run(hir)))
