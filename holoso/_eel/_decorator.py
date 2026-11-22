"""
Callables the compiler reads through to the function inside, as the compile root and as a callee alike.

A JIT dispatcher is read through to the function it merely accelerates, and is never state; one that declares types
of its own converts rather than accelerates, so it is refused. A bound method and an instance called through its
class's `__call__` are read through to the function, which then takes their receiver.
"""

import inspect
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


def mro_attr(klass: type, name: str, default: object = None) -> object:
    """The undescriptored class attribute -- what a structural walk must judge."""
    return next((k.__dict__[name] for k in klass.__mro__ if name in k.__dict__), default)


def resolve_callable(raw: object) -> tuple[types.FunctionType, object | None] | None:
    """
    The plain-Python function behind any calling spelling, with the receiver CPython's descriptor protocol would
    prepend to the arguments, if any. A C-level callable (a numpy dispatcher or ufunc, a partial) resolves to None.
    """
    if (fn := plain_function(raw)) is not None:
        return fn, None
    if inspect.ismethod(raw):
        fn = plain_function(raw.__func__)
        return None if fn is None else (fn, raw.__self__)
    if callable(raw):
        call_attr = mro_attr(type(raw), "__call__")
        if isinstance(call_attr, staticmethod) and (fn := plain_function(call_attr.__func__)) is not None:
            return fn, None
        if isinstance(call_attr, classmethod) and (fn := plain_function(call_attr.__func__)) is not None:
            return fn, type(raw)
        if (fn := plain_function(call_attr)) is not None:
            return fn, raw
    return None
