"""
Executable library stubs and the registry the frontend dispatches named callees through.
resolve(callee) maps a callee object to the Match saying how to lower a call to it, or None when unregistered.
The stub modules below register the lowerings, and `_meanings` declares every scalar meaning whole over them.
"""

from . import _conversions as _conversions
from . import _factories as _factories
from . import _intrinsics as _intrinsics
from . import _join as _join
from . import _linalg as _linalg
from . import _meanings as _meanings
from . import _numpy as _numpy
from . import _pow as _pow
from . import _reductions as _reductions
from ._registry import (
    Array as Array,
    Conversion as Conversion,
    Factory as Factory,
    Operand as Operand,
    Reshape as Reshape,
    ScalarLowering as ScalarLowering,
    Spelling as Spelling,
    VariadicFunction as VariadicFunction,
    resolve as resolve,
)
