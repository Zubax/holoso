"""Thin API for the selected hardware-aware mid-level IR."""

from ._ir import (
    Mir as Mir,
    MirBlock as MirBlock,
    MirBoolView as MirBoolView,
    MirBranch as MirBranch,
    MirBuilder as MirBuilder,
    MirConst as MirConst,
    MirInput as MirInput,
    MirJump as MirJump,
    MirNode as MirNode,
    MirOperation as MirOperation,
    MirPhi as MirPhi,
    MirRet as MirRet,
    MirStateRead as MirStateRead,
    MirStateSlot as MirStateSlot,
    MirTerminator as MirTerminator,
    MirWideView as MirWideView,
    reverse_postorder as reverse_postorder,
    successors as successors,
)
from ._interpret import MirInterpreter as MirInterpreter
from ._lower import lower as lower
from ._options import MirOptions as MirOptions
