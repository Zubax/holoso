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
    """A call that lowers to a single HIR float operator."""

    operator: Operator


@dataclass(frozen=True, slots=True)
class Library:
    """A call that inlines a composite stub function."""

    stub: types.FunctionType


@dataclass(frozen=True, slots=True)
class Factory:
    """
    A call the partial evaluator folds by invoking the registered builder stub on static arguments and
    snapshotting the resulting array as a fresh allocation.
    """

    build: types.FunctionType


@dataclass(frozen=True, slots=True)
class Conversion:
    """
    A call the partial evaluator lowers as a structural to-array conversion of its single argument.
    ``copies`` distinguishes np.array (an independent copy -- the A5 escape hatch) from np.asarray (which
    returns the argument itself for an array input, so source and result share).
    """

    copies: bool


type Match = Intrinsic | Library | Factory | Conversion

_REGISTRY: dict[object, Match] = {}


def _register(match: Match, keys: Iterable[object]) -> None:
    for key in keys:
        assert callable(key), key
        # A key holds exactly one Match; an alias to an equal Match (e.g. np.atan2 is np.arctan2) is tolerated.
        assert _REGISTRY.get(key, match) == match, key
        _REGISTRY[key] = match


def intrinsic[F: Callable[..., object]](operator: Callable[[], Operator], *substituted: object) -> Callable[[F], F]:
    op = operator()  # instantiated once here, so the registry stores an operator instance rather than a live factory

    def register(fn: F) -> F:
        assert isinstance(fn, types.FunctionType)
        assert fn.__code__.co_argcount == op.signature.arity
        _register(Intrinsic(op), (fn, *substituted))
        return fn

    return register


def lib[F: Callable[..., object]](*substituted: object) -> Callable[[F], F]:
    assert substituted

    def register(fn: F) -> F:
        assert isinstance(fn, types.FunctionType)
        _register(Library(fn), substituted)
        return fn

    return register


def factory[F: Callable[..., object]](*substituted: object) -> Callable[[F], F]:
    assert substituted

    def register(fn: F) -> F:
        assert isinstance(fn, types.FunctionType)
        _register(Factory(fn), substituted)
        return fn

    return register


def conversion(*keys: object, copies: bool) -> None:
    _register(Conversion(copies), keys)


def resolve(callee: object) -> Match | None:
    """The Match for a callee object, or None if it is unregistered."""
    try:
        return _REGISTRY.get(callee)
    except TypeError:  # something unhashable -- certainly not in the registry.
        return None
