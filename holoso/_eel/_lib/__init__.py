"""
Executable library stubs and the registry boundary the frontend dispatches calls through.
resolve(callee) maps a callee object to the Match saying how to lower a call to it, or None when unregistered.
"""

from . import _conversions as _conversions
from . import _factories as _factories
from . import _intrinsics as _intrinsics
from . import _linalg as _linalg
from . import _numpy as _numpy
from ._linalg import matmul_ as matmul_, transpose_ as transpose_
from ._registry import (
    Conversion as Conversion,
    Factory as Factory,
    Intrinsic as Intrinsic,
    Library as Library,
    resolve as resolve,
)
