"""
The library registry: a single `resolve(callee)` dispatch boundary over object identity. A class-member
descriptor is a key like any other (``np.ndarray.T`` IS an object): the caller resolves a method read by
looking up the descriptor on the owning class, and ``inspect.isdatadescriptor`` on that same object decides
whether the read already is the call. Only pure readers and derivations may bind members: a stub cannot
express receiver mutation, so a mutating method (``.fill``, ``.sort``) must stay unregistered and draw the
no-supported-attribute rejection.
"""

import inspect
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
    """
    A call that inlines a composite stub function. ``derives`` marks a non-copying derivation on the host
    (``.T``, ``flatten``): the result must carry the source's Allocation as its storage-equivalence token.
    """

    stub: types.FunctionType
    derives: bool = False


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
        if inspect.isdatadescriptor(key):
            assert isinstance(match, Library), "only Library entries may bind class members"
        else:
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


def lib[F: Callable[..., object]](*substituted: object, derives: bool = False) -> Callable[[F], F]:
    assert substituted

    def register(fn: F) -> F:
        assert isinstance(fn, types.FunctionType)
        assert not derives or fn.__code__.co_argcount == 1, "a derivation's result tracks its sole argument"
        _register(Library(fn, derives), (fn, *substituted))
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
