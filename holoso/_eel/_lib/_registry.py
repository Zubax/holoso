"""
The library registry: a single `resolve(callee)` dispatch boundary over object identity. A class-member
descriptor is a key like any other (`np.ndarray.T` IS an object): the caller resolves a method read by
looking up the descriptor on the owning class, and `inspect.isdatadescriptor` on that same object decides
whether the read already is the call. Only pure readers and derivations may bind members: a stub cannot
express receiver mutation, so a mutating method (`.fill`, `.sort`) must stay unregistered and draw the
no-supported-attribute rejection.

A scalar callee resolves to a spelling of a meaning; DESIGN.md states that model. A lowering's domain is its stub's
own annotations, per operand position, and may carry a refinement (`StaticWholeNonNegative[int]`).
"""

import inspect
import types
import typing
from typing import TypeAliasType
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from ..._hir import BoolType, FloatType, IntType, Operator
from .._annotations import accepted_stypes, annotation_stype
from .._ir import BinaryOp, CompareOp, ScalarType, UnaryOp

_HIR_TYPES: dict[ScalarType, type] = {
    ScalarType.BOOL: BoolType,
    ScalarType.INT: IntType,
    ScalarType.FLOAT: FloatType,
}


@dataclass(frozen=True, slots=True)
class Operand:
    """A null `const` is a runtime operand, not an absent one."""

    stype: ScalarType
    const: bool | int | float | None = None


# The refinements: each admits a set of compile-time constants, and they are pairwise disjoint, which is what lets
# the specificity order below compare them by identity alone. They erase to their argument for the type checker;
# the alias object itself is the marker, read back through `typing.get_origin`.
type StaticWholeNonNegative[T] = T
type StaticWholeNegative[T] = T
type StaticOneHalf[T] = T


def _whole(const: bool | int | float) -> bool:
    return not isinstance(const, float) or const.is_integer()


_REFINEMENTS: dict[TypeAliasType, Callable[[bool | int | float], bool]] = {
    StaticWholeNonNegative: lambda c: _whole(c) and c >= 0,
    StaticWholeNegative: lambda c: _whole(c) and c < 0,
    StaticOneHalf: lambda c: c == 0.5,
}

# A refinement overlapping another would make the identity comparisons below unsound.
assert not any(
    one is not other and admits(sample) and _REFINEMENTS[other](sample)
    for one, admits in _REFINEMENTS.items()
    for other in _REFINEMENTS
    for sample in (-1e300, -2, -1, -0.5, 0, 0.5, 1, 2, 1e300)
), "the refinements must be pairwise disjoint"


@dataclass(frozen=True, slots=True)
class Domain:
    stype: ScalarType
    refinement: TypeAliasType | None = None

    def accepts(self, operand: Operand) -> bool:
        if operand.stype not in accepted_stypes(self.stype):
            return False
        if self.refinement is None:
            return True
        return operand.const is not None and _REFINEMENTS[self.refinement](operand.const)

    def within(self, other: Domain) -> bool:
        """Whether every operand this domain accepts the other accepts too -- the specificity order."""
        types = accepted_stypes(self.stype) <= accepted_stypes(other.stype)
        return types and (other.refinement is None or other.refinement is self.refinement)

    def apart(self, other: Domain) -> bool:
        """Whether no operand at all satisfies both, which lets two incomparable lowerings coexist."""
        if accepted_stypes(self.stype).isdisjoint(accepted_stypes(other.stype)):
            return True
        return self.refinement is not None and other.refinement is not None and self.refinement is not other.refinement


def _annotation_domain(annotation: object) -> Domain | None:
    marker = typing.get_origin(annotation)
    refinement = marker if isinstance(marker, TypeAliasType) and marker in _REFINEMENTS else None
    if refinement is not None:
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
class ScalarMeaning:
    """The typed lowerings that implement one meaning; no two are ever selectable by the same operands."""

    lowerings: tuple[ScalarLowering, ...]

    def __post_init__(self) -> None:
        assert self.lowerings
        for index, lowering in enumerate(self.lowerings):
            name = lowering.stub.__name__
            assert len(lowering.operands) == self.arity, name
            for other in self.lowerings[:index]:
                assert other.operands != lowering.operands, name
                # Selection takes the most refined candidate, so any two an operand tuple could both reach must be
                # ordered.
                assert (
                    lowering.within(other) or other.within(lowering) or lowering.apart(other)
                ), f"{name}: the lowering domains are neither ordered nor separated"

    @property
    def arity(self) -> int:
        return len(self.lowerings[0].operands)

    @property
    def signatures(self) -> list[list[ScalarType]]:
        """In fixed order, so a diagnostic reads the same however the lowerings are listed."""
        order = list(ScalarType)
        served = {tuple(domain.stype for domain in lowering.operands) for lowering in self.lowerings}
        return [list(one) for one in sorted(served, key=lambda one: [order.index(stype) for stype in one])]

    def selects_by_value(self, position: int) -> bool:
        return any(lowering.operands[position].refinement is not None for lowering in self.lowerings)

    def select(self, operands: list[Operand]) -> ScalarLowering | None:
        assert len(operands) == self.arity
        candidates = [low for low in self.lowerings if all(d.accepts(o) for d, o in zip(low.operands, operands))]
        if not candidates:
            return None
        chosen = next((low for low in candidates if all(low.within(other) for other in candidates)), None)
        assert chosen is not None, "a meaning keeps the accepting lowerings of any operand tuple ordered"
        return chosen


@dataclass(frozen=True, slots=True)
class VariadicFunction:
    """
    A scalar entry of no fixed arity, `math.hypot` being n-ary in Python. Not a `ScalarLowering`: those compare their
    domains positionally, which a variable-length operand list has no positions for. The operator is therefore a
    factory where a fixed-arity entry stores an instance.
    """

    operator: Callable[[int], Operator]
    domain: Domain
    minimum: int


@dataclass(frozen=True, slots=True)
class Array:
    """
    An inlined composite whose meaning is rank and shape, so it declares no scalar domain. `derives` marks a
    non-copying derivation on the host (`.T`, `flatten`): the result carries the source's Allocation as its
    storage-equivalence token. `sequences` names the argument positions whose gate admits a Python sequence
    (np.polyval's coefficients); everywhere else the build-an-array advice stands.
    """

    stub: types.FunctionType
    derives: bool = False
    sequences: frozenset[int] = frozenset()


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
    A structural to-array conversion with an optional dtype (positional or keyword). `copies` distinguishes
    np.array (always an independent copy -- the A5 escape hatch) from np.asarray, which shares a
    family-preserving array input; a family-CHANGING dtype copies on the host, so both spellings mint fresh
    there. Dtype widths are erased, so a same-family host copy (float32 to float) conservatively shares.
    """

    copies: bool


@dataclass(frozen=True, slots=True)
class Reshape:
    """
    A compile-time restructuring of the same storage; like Conversion, the registry only decides WHICH callees
    mean it. A marker rather than a stub: the shape's int-or-tuple polymorphism has no discriminating spelling
    inside the subset.
    """


@dataclass(frozen=True, slots=True)
class Spelling:
    """
    Array capability is the spelling's, not the meaning's: `np.minimum` maps over an array where `min` does not.
    `protocol` names the method a record or captured object answers the call with before the meaning applies.
    """

    meaning: ScalarMeaning
    elementwise: bool = False
    protocol: str | None = None


type Match = Spelling | VariadicFunction | Array | Factory | Conversion | Reshape

_OPERATOR_KEYS = (BinaryOp, CompareOp, UnaryOp)

_REGISTRY: dict[object, Match] = {}


def _keys(keys: Iterable[object]) -> list[object]:
    """numpy spells one object under several names (np.abs IS np.absolute), so one decoration may name a key twice."""
    return list(dict.fromkeys(keys))


def _register(match: Match, keys: Iterable[object]) -> None:
    for key in _keys(keys):
        if inspect.isdatadescriptor(key):
            assert isinstance(match, Array), "only array entries may bind class members"
        else:
            assert callable(key) or isinstance(key, _OPERATOR_KEYS), key
        assert key not in _REGISTRY, key
        _REGISTRY[key] = match


def meaning(
    *stubs: object, scalar: Sequence[object] = (), elementwise: Sequence[object] = (), protocol: str | None = None
) -> None:
    assert scalar or elementwise, "a meaning with no spelling"
    served = ScalarMeaning(tuple(_lowering_of(stub) for stub in stubs))
    if elementwise:
        _admit_elementwise(served)
    for keys, over_arrays in ((scalar, False), (elementwise, True)):
        _register(Spelling(served, over_arrays, protocol), keys)


def _lowering_of(stub: object) -> ScalarLowering:
    found = _REGISTRY[stub]
    assert isinstance(found, Spelling), stub
    (lowering,) = found.meaning.lowerings
    return lowering


def _stub(lowering: ScalarLowering) -> None:
    """A stub spells its own lowering alone: a composite calls the primitive it needs, not a meaning it lowers."""
    _register(Spelling(ScalarMeaning((lowering,))), (lowering.stub,))


def _declared(stub: types.FunctionType, name: str) -> Domain:
    domain = _annotation_domain(stub.__annotations__.get(name))
    assert domain is not None, (stub.__name__, name)
    return domain


def _scalar_lowering(fn: object, operator: Operator | None) -> ScalarLowering:
    assert isinstance(fn, types.FunctionType)
    assert not fn.__kwdefaults__, "a stub binds positionally"
    code = fn.__code__
    operands = tuple(_declared(fn, name) for name in code.co_varnames[: code.co_argcount])
    assert operands
    if operator is not None:
        # A single HIR operation consumes whatever the datapath carries, so it cannot demand a binding time.
        assert all(d.refinement is None for d in operands), "an intrinsic takes unrefined operands"
        signature = operator.signature
        assert isinstance(signature.result_type, _HIR_TYPES[_declared(fn, "return").stype])
        assert len(signature.operand_types) == len(operands)
        assert all(isinstance(ty, _HIR_TYPES[d.stype]) for ty, d in zip(signature.operand_types, operands, strict=True))
    return ScalarLowering(fn, operator, operands)


def intrinsic[F: Callable[..., object]](operator: Operator) -> Callable[[F], F]:
    def register(fn: F) -> F:
        _stub(_scalar_lowering(fn, operator))
        return fn

    return register


def variadic[F: Callable[..., object]](
    operator: Callable[[int], Operator], *substituted: object, minimum: int
) -> Callable[[F], F]:
    def register(fn: F) -> F:
        assert isinstance(fn, types.FunctionType)
        code = fn.__code__
        assert code.co_argcount == 0 and code.co_flags & inspect.CO_VARARGS, "a variadic entry takes only *args"
        domain = _declared(fn, code.co_varnames[0])
        assert domain.stype in (ScalarType.INT, ScalarType.FLOAT) and domain.refinement is None
        result = _declared(fn, "return")
        for arity in (minimum, minimum + 1):  # arity-uniform, so two samples pin the shape
            signature = operator(arity).signature
            assert signature.arity == arity
            assert isinstance(signature.result_type, _HIR_TYPES[result.stype])
            assert all(isinstance(ty, _HIR_TYPES[domain.stype]) for ty in signature.operand_types)
        _register(VariadicFunction(operator, domain, minimum), (fn, *substituted))
        return fn

    return register


def lib[F: Callable[..., object]](fn: F) -> F:
    _stub(_scalar_lowering(fn, None))
    return fn


def array[F: Callable[..., object]](
    *substituted: object, derives: bool = False, sequences: tuple[int, ...] = ()
) -> Callable[[F], F]:
    assert substituted

    def register(fn: F) -> F:
        assert isinstance(fn, types.FunctionType)
        assert not fn.__kwdefaults__, "a stub binds positionally"
        assert not derives or fn.__code__.co_argcount == 1, "a derivation's result tracks its sole argument"
        _register(Array(fn, derives, frozenset(sequences)), (fn, *substituted))
        return fn

    return register


def _admit_elementwise(served: ScalarMeaning) -> None:
    # An all-boolean lowering is unreachable from a leaf, as no array holds booleans.
    reachable = [low for low in served.lowerings if any(d.stype is not ScalarType.BOOL for d in low.operands)]
    assert reachable, f"{served.lowerings[0].stub.__name__}: no array leaf can reach this meaning"
    for lowering in reachable:
        answer = _declared(lowering.stub, "return").stype
        assert answer in (
            ScalarType.INT,
            ScalarType.FLOAT,
        ), f"{served.lowerings[0].stub.__name__}: no array family holds this answer"


def factory[F: Callable[..., object]](*substituted: object) -> Callable[[F], F]:
    assert substituted

    def register(fn: F) -> F:
        assert isinstance(fn, types.FunctionType)
        _register(Factory(fn), substituted)
        return fn

    return register


def conversion(*keys: object, copies: bool) -> None:
    _register(Conversion(copies), keys)


def reshape(*keys: object) -> None:
    _register(Reshape(), keys)


def resolve(callee: object) -> Match | None:
    """The Match for a callee object, or None if it is unregistered."""
    try:
        return _REGISTRY.get(callee)
    except TypeError:  # something unhashable -- certainly not in the registry.
        return None
