"""
Executable library stubs and the registry the frontend dispatches named callees through.
resolve(callee) maps a callee object to the Match saying how to lower a call to it, or None when unregistered.
A subset operator is a key like any callee, so `**` and `@` reach their lowerings the same way a spelled call does.
"""

from . import _conversions as _conversions
from . import _factories as _factories
from . import _intrinsics as _intrinsics
from . import _linalg as _linalg
from . import _numpy as _numpy
from . import _pow as _pow
from . import _reductions as _reductions
from ._registry import (
    Array as Array,
    Conversion as Conversion,
    Factory as Factory,
    Lifted as Lifted,
    Operand as Operand,
    Reshape as Reshape,
    ScalarFunction as ScalarFunction,
    resolve as resolve,
)
