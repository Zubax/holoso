"""Thin API for the selected hardware-aware mid-level IR."""

from ._ir import (
    Mir as Mir,
    MirBlock as MirBlock,
    MirBranch as MirBranch,
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
    reverse_postorder as reverse_postorder,
    successors as successors,
    thread_arm as thread_arm,
    threadable_arms as threadable_arms,
    predecessors as predecessors,
)
from ._lower import lower as lower
from ._options import MirOptions as MirOptions
