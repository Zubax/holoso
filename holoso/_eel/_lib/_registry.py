"""
The library registry: a single `resolve(callee)` dispatch boundary over object identity. A class-member
descriptor is a key like any other (`np.ndarray.T` IS an object): the caller resolves a method read by
looking up the descriptor on the owning class, and `inspect.isdatadescriptor` on that same object decides
whether the read already is the call. Only pure readers and derivations may bind members: a stub cannot
express receiver mutation, so a mutating method (`.fill`, `.sort`) must stay unregistered and draw the
no-supported-attribute rejection.

A scalar callee resolves to a group of typed lowerings: `min` is a hardware operator over floats and a
compare-and-select composite over integers. A lowering's domain is its stub's own annotations.
A domain is per operand position and may carry a refinement (`StaticNegative[int]`) --
which a subset operator reaches as a key like any other, so an operator and its spellings cannot part.
"""

import inspect
import types
import typing
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum

from ..._hir import BoolType, FloatType, IntType, Operator
from .._annotations import accepted_stypes, annotation_stype
from .._ir import BinaryOp, ScalarType

_HIR_TYPES: dict[ScalarType, type] = {
    ScalarType.BOOL: BoolType,
    ScalarType.INT: IntType,
    ScalarType.FLOAT: FloatType,
}


class Sign(Enum):
    NEGATIVE = -1
    ZERO = 0
    POSITIVE = 1


_EVERY_SIGN = frozenset(Sign)


class Refinement(Enum):
    """
    What a declaration demands beyond a type, so one callee may carry a lowering that exists only where the
    compiler already knows the number.
    """

    ANY = "any"
    STATIC_NONNEGATIVE = "static nonnegative"
    STATIC_NEGATIVE = "static negative"


@dataclass(frozen=True, slots=True)
class _Demand:
    runtime: bool
    signs: frozenset[Sign]


# A new refinement is one row here and nothing else.
_DEMANDS: dict[Refinement, _Demand] = {
    Refinement.ANY: _Demand(True, _EVERY_SIGN),
    Refinement.STATIC_NONNEGATIVE: _Demand(False, frozenset({Sign.ZERO, Sign.POSITIVE})),
    Refinement.STATIC_NEGATIVE: _Demand(False, frozenset({Sign.NEGATIVE})),
}

# They erase to their argument for the type checker; the alias object itself is the marker, read back through
# `typing.get_origin`.
type StaticNonNegative[T] = T
type StaticNegative[T] = T

_REFINEMENTS: dict[object, Refinement] = {
    StaticNonNegative: Refinement.STATIC_NONNEGATIVE,
    StaticNegative: Refinement.STATIC_NEGATIVE,
}


@dataclass(frozen=True, slots=True)
class Operand:
    """A null `const` is a runtime operand, not an absent one."""

    stype: ScalarType
    const: bool | int | float | None = None

    @property
    def sign(self) -> Sign:
        assert self.const is not None
        return Sign.POSITIVE if self.const > 0 else Sign.NEGATIVE if self.const < 0 else Sign.ZERO


@dataclass(frozen=True, slots=True)
class Domain:
    stype: ScalarType
    refinement: Refinement = Refinement.ANY

    def accepts(self, operand: Operand) -> bool:
        if operand.stype not in accepted_stypes(self.stype):
            return False
        demand = _DEMANDS[self.refinement]
        return demand.runtime if operand.const is None else operand.sign in demand.signs

    def within(self, other: Domain) -> bool:
        """Whether every operand this domain accepts the other accepts too -- the specificity order."""
        mine, theirs = _DEMANDS[self.refinement], _DEMANDS[other.refinement]
        types = accepted_stypes(self.stype) <= accepted_stypes(other.stype)
        return types and (theirs.runtime or not mine.runtime) and mine.signs <= theirs.signs

    def apart(self, other: Domain) -> bool:
        """Whether no operand at all satisfies both, which lets two incomparable lowerings coexist."""
        mine, theirs = _DEMANDS[self.refinement], _DEMANDS[other.refinement]
        if accepted_stypes(self.stype).isdisjoint(accepted_stypes(other.stype)):
            return True
        return not (mine.runtime and theirs.runtime) and mine.signs.isdisjoint(theirs.signs)


def _annotation_domain(annotation: object) -> Domain | None:
    refinement = _REFINEMENTS.get(typing.get_origin(annotation), Refinement.ANY)
    if refinement is not Refinement.ANY:
        (annotation,) = typing.get_args(annotation)
    stype = annotation_stype(annotation)
    return None if stype is None else Domain(stype, refinement)


@dataclass(frozen=True, slots=True)
class ScalarLowering:
    """A single HIR operation, or -- where `operator` is None -- the stub inlined as ordinary user code."""

    stub: types.FunctionType
    operator: Operator | None
    operands: tuple[Domain, ...]

    def within(self, other: ScalarLowering) -> bool:
        return all(a.within(b) for a, b in zip(self.operands, other.operands, strict=True))

    def apart(self, other: ScalarLowering) -> bool:
        return any(a.apart(b) for a, b in zip(self.operands, other.operands, strict=True))


@dataclass(frozen=True, slots=True)
class ScalarFunction:
    """No two lowerings are ever selectable by the same operands."""

    lowerings: tuple[ScalarLowering, ...]

    @property
    def arity(self) -> int:
        return len(self.lowerings[0].operands)

    @property
    def domains(self) -> list[ScalarType]:
        """Fixed order, so a diagnostic reads the same however the decorations landed."""
        served = {domain.stype for lowering in self.lowerings for domain in lowering.operands}
        return [stype for stype in (ScalarType.INT, ScalarType.FLOAT) if stype in served]

    def select(self, operands: list[Operand]) -> ScalarLowering | None:
        assert len(operands) == self.arity
        candidates = [low for low in self.lowerings if all(d.accepts(o) for d, o in zip(low.operands, operands))]
        if not candidates:
            return None
        chosen = next((low for low in candidates if all(low.within(other) for other in candidates)), None)
        assert chosen is not None, "registration keeps the accepting lowerings of any operand tuple ordered"
        return chosen


@dataclass(frozen=True, slots=True)
class Array:
    """
    An inlined composite whose meaning is rank and shape, so it declares no scalar domain. `derives` marks a
    non-copying derivation on the host (`.T`, `flatten`): the result carries the source's Allocation as its
    storage-equivalence token.
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
    `copies` distinguishes np.array (an independent copy -- the A5 escape hatch) from np.asarray (which
    returns the argument itself for an array input, so source and result share).
    """

    copies: bool


type Match = ScalarFunction | Array | Factory | Conversion

_REGISTRY: dict[object, Match] = {}


def _register(match: Match, keys: Iterable[object]) -> None:
    for key in keys:
        if inspect.isdatadescriptor(key):
            assert isinstance(match, Array), "only array entries may bind class members"
        else:
            assert callable(key) or isinstance(key, BinaryOp), key
        # A key holds exactly one Match; an alias to an equal Match (e.g. np.atan2 is np.arctan2) is tolerated.
        assert _REGISTRY.get(key, match) == match, key
        _REGISTRY[key] = match


def _register_scalar(lowering: ScalarLowering, keys: Iterable[object]) -> None:
    for key in keys:
        assert (callable(key) and not inspect.isdatadescriptor(key)) or isinstance(key, BinaryOp), key
        found = _REGISTRY.get(key)
        if found is None:
            _REGISTRY[key] = ScalarFunction((lowering,))
            continue
        assert isinstance(found, ScalarFunction), key
        assert found.arity == len(lowering.operands), key
        served = next((low for low in found.lowerings if low.operands == lowering.operands), None)
        if served is not None:
            assert served == lowering, key  # np.abs IS np.absolute, so one decoration can name a key twice
            continue
        for other in found.lowerings:
            # Selection takes the most refined candidate, so any two an operand tuple could both reach must be
            # ordered.
            assert (
                lowering.within(other) or other.within(lowering) or lowering.apart(other)
            ), f"{key}: the lowering domains are neither ordered nor separated"
        _REGISTRY[key] = ScalarFunction((*found.lowerings, lowering))


def _declared(stub: types.FunctionType, name: str) -> Domain:
    domain = _annotation_domain(stub.__annotations__.get(name))
    assert domain is not None, (stub.__name__, name)
    return domain


def _scalar_lowering(fn: object, operator: Operator | None) -> ScalarLowering:
    assert isinstance(fn, types.FunctionType)
    code = fn.__code__
    operands = tuple(_declared(fn, name) for name in code.co_varnames[: code.co_argcount])
    assert operands and all(d.stype in (ScalarType.INT, ScalarType.FLOAT) for d in operands), "numeric operands"
    if operator is not None:
        # A single HIR operation consumes whatever the datapath carries, so it cannot demand a binding time.
        assert all(d.refinement is Refinement.ANY for d in operands), "an intrinsic takes unrefined operands"
        signature = operator.signature
        assert isinstance(signature.result_type, _HIR_TYPES[_declared(fn, "return").stype])
        assert len(signature.operand_types) == len(operands)
        assert all(isinstance(ty, _HIR_TYPES[d.stype]) for ty, d in zip(signature.operand_types, operands, strict=True))
    return ScalarLowering(fn, operator, operands)


def intrinsic[F: Callable[..., object]](operator: Callable[[], Operator], *substituted: object) -> Callable[[F], F]:
    op = operator()  # instantiated once here, so the registry stores an operator instance rather than a live factory

    def register(fn: F) -> F:
        _register_scalar(_scalar_lowering(fn, op), (fn, *substituted))
        return fn

    return register


def lib[F: Callable[..., object]](*substituted: object) -> Callable[[F], F]:
    assert substituted

    def register(fn: F) -> F:
        _register_scalar(_scalar_lowering(fn, None), (fn, *substituted))
        return fn

    return register


def array[F: Callable[..., object]](*substituted: object, derives: bool = False) -> Callable[[F], F]:
    assert substituted

    def register(fn: F) -> F:
        assert isinstance(fn, types.FunctionType)
        assert not derives or fn.__code__.co_argcount == 1, "a derivation's result tracks its sole argument"
        _register(Array(fn, derives), (fn, *substituted))
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
