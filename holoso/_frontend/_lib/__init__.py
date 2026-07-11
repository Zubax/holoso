"""
Executable library stubs and the registry boundary the frontend dispatches calls through.
resolve(callee) maps a callee object to the Match saying how to lower a call to it, or None when unregistered.
"""

from . import _intrinsics as _intrinsics
from . import _linalg as _linalg
from . import _numpy as _numpy
from ._registry import Intrinsic as Intrinsic, Library as Library, resolve as resolve
