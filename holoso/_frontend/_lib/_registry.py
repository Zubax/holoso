"""
The library registry: a single `resolve(callee)` dispatch boundary that maps a callee object to the Match that
says how to lower a call to it, or None when it is unregistered.
"""

import enum
import types
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from ..._hir import Operator


class IntrinsicResultRule(enum.Enum):
    """
    How an intrinsic's result kind depends on its operand kinds -- declarative so the library registry stays free of
    FIR analysis and target-format knowledge (the analyzer maps these to facts).
    """

    SIGNATURE = enum.auto()  # the operator's own result type; an integer operand promotes to float (float-forcing)
    ALWAYS_INT = enum.auto()  # always a typed integer (math.floor/ceil/trunc, one-argument round)
    INT_OVERLOAD = enum.auto()  # all-integer operands -> the integer implementation; otherwise promote and run float


class IntegerImplementation(enum.Enum):
    """The integer-typed HIR an integer-operand intrinsic produces (contained at MIR; no integer hardware here)."""

    IDENTITY = enum.auto()  # the operand itself (np.floor/math.floor of an integer is the integer)
    ABS = enum.auto()  # IntAbs
    MIN = enum.auto()  # IntRelational(LE) + IntSelect
    MAX = enum.auto()  # IntRelational(GE) + IntSelect


@dataclass(frozen=True, slots=True)
class Intrinsic:
    """A call that lowers to a single HIR float operator, its result kind governed by ``result_rule``."""

    operator: Operator
    result_rule: IntrinsicResultRule = IntrinsicResultRule.SIGNATURE
    integer_implementation: IntegerImplementation | None = None


@dataclass(frozen=True, slots=True)
class Library:
    """A call that inlines a composite stub function."""

    stub: types.FunctionType

    @property
    def display_name(self) -> str:
        """The public spelling: a stub carries a trailing underscore only to avoid shadowing what it implements."""
        return self.stub.__name__.removesuffix("_")


type Match = Intrinsic | Library

_REGISTRY: dict[object, Match] = {}


def _register(match: Match, keys: Iterable[object]) -> None:
    for key in keys:
        assert callable(key), key
        # A key holds exactly one Match; an alias to an equal Match (e.g. np.atan2 is np.arctan2) is tolerated.
        assert _REGISTRY.get(key, match) == match, key
        _REGISTRY[key] = match


def intrinsic[F: Callable[..., object]](
    operator: Callable[[], Operator],
    *substituted: object,
    result_rule: IntrinsicResultRule = IntrinsicResultRule.SIGNATURE,
    integer_implementation: IntegerImplementation | None = None,
) -> Callable[[F], F]:
    op = operator()  # instantiated once here, so the registry stores an operator instance rather than a live factory

    def register(fn: F) -> F:
        assert isinstance(fn, types.FunctionType)
        assert fn.__code__.co_argcount == op.signature.arity
        _register(Intrinsic(op, result_rule, integer_implementation), (fn, *substituted))
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
