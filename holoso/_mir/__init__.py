"""Thin API for the selected hardware-aware mid-level IR."""

from ._ir import (
    Mir as Mir,
    MirBuilder as MirBuilder,
    MirConst as MirConst,
    MirFloatConst as MirFloatConst,
    MirFloatInput as MirFloatInput,
    MirFloatNode as MirFloatNode,
    MirFloatOperation as MirFloatOperation,
    MirFloatOutput as MirFloatOutput,
    MirFloatView as MirFloatView,
    MirInput as MirInput,
    MirOperation as MirOperation,
)
from ._lower import lower as lower
