"""
Located rejection with inline-frame re-attribution: the reported location is the OUTERMOST call site (the
only place the user can act), the message is prefixed with the chain, and the innermost real location rides
as a short tail — kept for stubs too, since user helpers need it — omitted when it would merely repeat the
reported location.
"""

import os
from typing import NoReturn

from ..._errors import SourceLocation, SynthesisError, UnsupportedConstruct
from .._ir import Origin


def reject(origin: Origin, message: str, error: type[SynthesisError] = UnsupportedConstruct) -> NoReturn:
    raise error(_chained(origin, message), _located(origin))


def reattribute(origin: Origin, error: SynthesisError) -> NoReturn:
    """
    Re-raise a callee-side failure (its own desugar, typically) under the caller's chain, preserving the
    concrete class; the cause is suppressed as pointing into non-actionable internals.
    """
    tail = "" if error.location is None else _tail(error.location)
    raise type(error)(_prefix(origin) + error.message + tail, _located(origin)) from None


def _located(origin: Origin) -> SourceLocation:
    return origin.frames[0].site if origin.frames else origin.location


def _prefix(origin: Origin) -> str:
    return "".join(f"in {frame.callee}(): " for frame in origin.frames)


def _chained(origin: Origin, message: str) -> str:
    if not origin.frames:
        return message
    tail = "" if origin.location == origin.frames[0].site else _tail(origin.location)
    return _prefix(origin) + message + tail


def _tail(location: SourceLocation) -> str:
    return f" (at {os.path.basename(location.filename)}:{location.lineno})"
