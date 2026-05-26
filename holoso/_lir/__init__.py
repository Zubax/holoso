"""Thin API for the low-level IR consumer contract."""

from ._build import build as build, interface_of as interface_of
from ._ir import Lir as Lir
from ._ir import (
    FloatConstRef as FloatConstRef,
    FloatOperand as FloatOperand,
    FloatOperatorInstance as FloatOperatorInstance,
    FloatRegRef as FloatRegRef,
    FloatScheduledOp as FloatScheduledOp,
)
