"""
Wrappers the compiler reads through to the function inside.

A JIT dispatcher is read through to the function it merely accelerates, as a target and as a callee, and is never
state; one that declares types of its own converts rather than accelerates, so it is refused, as is any decorator
that advertises itself -- a silent unwrap would drop behavior that IS part of the program.
"""

import sys
import types


def is_wrapper(raw: object) -> bool:
    """Whether this is a wrapper around a function: a callee whatever else it does, and never a component."""
    dispatcher = getattr(sys.modules.get("numba.core.registry"), "CPUDispatcher", None)
    return isinstance(dispatcher, type) and isinstance(raw, dispatcher)


def plain_function(raw: object) -> types.FunctionType | None:
    if isinstance(raw, types.FunctionType):
        return raw
    # A type declared up front -- a whole signature, or a local forced to one -- converts what the function
    # computes. Each reading defaults to refusing, so a renamed internal costs the unwrap rather than an answer.
    if not is_wrapper(raw) or getattr(raw, "_can_compile", False) is not True or getattr(raw, "locals", True):
        return None
    inner = getattr(raw, "py_func", None)
    return inner if isinstance(inner, types.FunctionType) else None
