"""
The library registry: a single `resolve(callee)` dispatch boundary that maps a callee object to the Match that
says how to lower a call to it, or None when it is unregistered.
"""

import types
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from ..._hir import Operator


@dataclass(frozen=True, slots=True)
class Intrinsic:
    """A call that lowers to a single HIR float operator, optionally wrapped to an integer result."""

    operator: Operator
    # True for the int-returning spellings (``math.floor``/``ceil``/``trunc``, one-argument ``round``): the operator
    # runs in float, then the result is a typed integer, so subsequent integer arithmetic stays exact rather than
    # rounding in the float datapath. The numpy spellings (``np.floor`` ...) return float and keep this False.
    returns_int: bool = False


@dataclass(frozen=True, slots=True)
class Library:
    """A call that inlines a composite stub function."""

    stub: types.FunctionType


type Match = Intrinsic | Library

_REGISTRY: dict[object, Match] = {}


def _register(match: Match, keys: Iterable[object]) -> None:
    for key in keys:
        assert callable(key), key
        # A key holds exactly one Match; an alias to an equal Match (e.g. np.atan2 is np.arctan2) is tolerated.
        assert _REGISTRY.get(key, match) == match, key
        _REGISTRY[key] = match


def intrinsic[F: Callable[..., object]](
    operator: Callable[[], Operator], *substituted: object, returns_int: bool = False
) -> Callable[[F], F]:
    op = operator()  # instantiated once here, so the registry stores an operator instance rather than a live factory

    def register(fn: F) -> F:
        assert isinstance(fn, types.FunctionType)
        assert fn.__code__.co_argcount == op.signature.arity
        _register(Intrinsic(op, returns_int), (fn, *substituted))
        return fn

    return register


def lib[F: Callable[..., object]](*substituted: object) -> Callable[[F], F]:
    assert substituted

    def register(fn: F) -> F:
        assert isinstance(fn, types.FunctionType)
        _register(Library(fn), substituted)
        return fn

    return register


def resolve(callee: object) -> Match | None:
    """The Match for a callee object, or None if it is unregistered."""
    try:
        return _REGISTRY.get(callee)
    except TypeError:  # something unhashable -- certainly not in the registry.
        return None
