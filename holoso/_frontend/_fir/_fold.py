"""
The fold admission harness: the ONE door through which a call crosses into concrete host evaluation.

Every concrete-evaluation hazard family (live-object reads outside the state machinery, folds on type-unfaithful
reconstructions, snapshot layout observation, compile-time side effects and unbounded cost) enters through the
same mechanism -- executing host Python over carriers the domain cannot vouch for -- under an open-ended variety
of spellings. The harness closes the mechanism instead of chasing spellings: the analyzer hands over the resolved
callee and the argument facts, and :func:`admit_call` either returns (a vetted callable with admissible
arguments) or raises :class:`FoldRefusal`, which the analyzer locates at the call origin. Admission is a CLOSED
WHITELIST -- a callable evaluates concretely only if vetted for value-determined evaluation, with per-argument
admission (no records, no live object references, bounded cost) checked before anything runs; the default for
everything else is refusal. The greppable invariant: admission happens here and nowhere else. The intended
refinement turns the vetted-set membership and the argument walk into declarative per-callee rows with typed
admission classes; the door stays the same.
"""

import enum
import types
from dataclasses import MISSING, fields, is_dataclass

from ._fact import (
    AggregateFact,
    Fact,
    Known,
    ListLayout,
    RecordLayout,
    Reference,
    StructuralLayout,
    TupleLayout,
    ValueLayout,
)
from ._value import MetaInt, NpInt, StaticRange, StaticStr


class FoldRefusal(Exception):
    """A refused admission; ``library_diagnostic`` selects the public unimplemented-library error type."""

    def __init__(self, message: str, library_diagnostic: bool = False) -> None:
        super().__init__(message)
        self.library_diagnostic = library_diagnostic


def is_unimplemented_library(target: object) -> bool:
    """A numpy ufunc or a ``math`` module member: a recognized library primitive, distinct from an arbitrary call."""
    import math

    import numpy as np

    return isinstance(target, np.ufunc) or any(target is member for member in vars(math).values())


def range_size(value: StaticRange) -> int:
    try:
        return len(range(value.start, value.stop, value.step))
    except OverflowError:  # astronomically large: oversized by definition
        return 1 << 62


def contains_record(layout: "ValueLayout") -> bool:
    match layout:
        case RecordLayout():
            return True
        case TupleLayout(items=items) | ListLayout(items=items) | StructuralLayout(items=items):
            return any(contains_record(item) for item in items)
        case _:
            return False


def _inert_type_referents() -> tuple[type, ...]:
    import numpy as np

    return (float, int, bool, np.float64, np.int64, np.bool_)


def classinfo_types(fact: "Fact | None") -> list[type] | None:
    """
    The plain types an isinstance classinfo resolves to, or None when any member is opaque (a non-type, a typing
    generic, an unresolvable reference). Tuples and unions unpack recursively, so a precomputed ``(float, Mode)``
    or ``str | Mode`` is inspected member by member instead of slipping past as one reference.
    """
    pending: list[object] = []
    match fact:
        case Reference(obj=obj):
            pending = [obj]
        case AggregateFact(leaves=leaves):
            for leaf in leaves:
                if not isinstance(leaf, Reference):
                    return None
                pending.append(leaf.obj)
        case _:
            return None
    resolved: list[type] = []
    while pending:
        entry = pending.pop()
        if isinstance(entry, tuple):
            pending.extend(entry)
        elif isinstance(entry, types.UnionType):
            pending.extend(entry.__args__)
        elif isinstance(entry, type):
            resolved.append(entry)
        else:
            return None
    return resolved


def _vetted_concrete_target(target: object) -> bool:
    """
    The closed set of callables admitted for concrete evaluation beyond the library registry: value casts and
    constructors whose results depend only on argument VALUES, never on object identity, memory layout, or
    provenance a reconstruction erases. Bound methods are NOT vetted here: a pre-bound builtin captured from the
    user's namespace carries a LIVE receiver (a captured list.pop emptied the user's list at compile time), so
    only the analyzer's own value-method bind site -- which mints methods off the domain's reconstruction under
    its own guards -- may admit one, by identity.
    """
    import operator as operator_module

    import numpy as np

    vetted = (
        float,
        int,
        bool,
        len,
        range,
        slice,
        sum,
        divmod,
        isinstance,
        operator_module.index,
        np.array,
        np.asarray,
        np.asanyarray,
        np.bool_,
        np.int64,
        np.float64,
    )
    if any(target is entry for entry in vetted):
        return True
    if isinstance(target, type) and is_dataclass(target):
        # Record construction is real construction -- but only through the GENERATED machinery (plain field
        # assignment compiled from synthesized source). A user __init__, __post_init__, __new__, __setattr__,
        # metaclass, or field default_factory is arbitrary code that would run at compile time, possibly on
        # erasure-reconstructed arguments (an IntEnum field arrives as its base int) or against live state
        # (a default_factory popped the user's list once per analysis round).
        init = next((c.__dict__["__init__"] for c in target.__mro__ if "__init__" in c.__dict__), None)
        generated_init = isinstance(init, types.FunctionType) and init.__code__.co_filename == "<string>"
        hooked = any(
            name in c.__dict__ for name in ("__post_init__", "__new__") for c in target.__mro__ if c is not object
        ) or any(
            isinstance(member, types.FunctionType) and member.__code__.co_filename != "<string>"
            for c in target.__mro__
            if c is not object
            for name in ("__setattr__", "__delattr__")
            if (member := c.__dict__.get(name)) is not None
        )
        factories = any(field.default_factory is not MISSING for field in fields(target))
        return generated_init and not hooked and not factories and type(target) is type
    return False


def admit_call(
    target: object,
    positional: list["Fact"],
    keywords: list["Fact"],
    minted: bool,
    registry_resolved: bool,
) -> None:
    """
    Admit or refuse a concrete call. ``positional``/``keywords`` are the argument facts in order; ``minted`` marks
    an analyzer-minted value method (admitted by identity at the bind site); ``registry_resolved`` marks a library
    registry member (vetted by its own dispatch). Raises :class:`FoldRefusal` with the established message texts.
    """
    if minted:
        # A minted value method runs on the domain's reconstruction, but its ARGUMENTS size its work and its
        # result: "x".ljust(10**12) or int.to_bytes(10**12) would allocate gigabytes at compile time. The same
        # 2^20 bound that guards static ranges applies to its integer arguments.
        for fact in [*positional, *keywords]:
            if (
                isinstance(fact, Known)
                and isinstance(fact.value, (MetaInt, NpInt))
                and abs(fact.value.value) > (1 << 20)
            ):
                raise FoldRefusal("an oversized integer argument to a value method is not supported")
    if not registry_resolved and not minted and not _vetted_concrete_target(target):
        name = getattr(target, "__name__", repr(target))
        if is_unimplemented_library(target):
            raise FoldRefusal(f"library function {name!r} is not implemented yet", library_diagnostic=True)
        raise FoldRefusal(f"call to {name!r} is not supported in a kernel")
    if target is isinstance:
        # Enum members normalize to their base value at admission (the sanctioned erasure), so an isinstance
        # query answers wrong whenever erasure can matter: the SUBJECT must not carry an erasure-capable
        # provenance (a Known Python int or str may be a normalized IntEnum/StrEnum member -- indistinguishable
        # after admission, and a plain mixin base of an enum makes even an enum-free classinfo lie), and the
        # classinfo must RESOLVE COMPLETELY to enum-free plain types whose instance check is type's own (an
        # ABC's register()/__instancecheck__ distinguishes the live member from its erased value).
        subject = positional[0] if positional else None
        if isinstance(subject, Known) and isinstance(subject.value, (MetaInt, StaticStr)):
            raise FoldRefusal("isinstance of a static int/str is not decidable (it may be a normalized enum member)")
        classinfo = positional[1] if len(positional) == 2 else None
        kinds = classinfo_types(classinfo)
        if (
            kinds is None
            or any(issubclass(kind, enum.Enum) for kind in kinds)
            or any(type(kind).__instancecheck__ is not type.__instancecheck__ for kind in kinds)
        ):
            raise FoldRefusal(
                "isinstance requires a static, enum-free class or tuple of classes with the plain "
                "instance check (enum members normalize to their base value)"
            )
    # Argument admission: a record never crosses into a concrete evaluation (nested inside a tuple/list included)
    # -- the callable, or even the dataclass-generated __repr__, would run on a reconstruction that is
    # value-faithful but not type-faithful (an enum field rebuilds as its base value). An object reference never
    # crosses either, except as isinstance's classinfo: a stateful component's dunder would read the live
    # reset-time object while the kernel's writes exist only as state facts (float(self) stepped [1.0, 1.0]
    # where Python steps [3.0, 5.0]). A referenced dtype-ish builtin TYPE is inert.
    positional_count = len(positional)
    for position, fact in enumerate([*positional, *keywords]):
        if isinstance(fact, AggregateFact) and contains_record(fact.layout):
            raise FoldRefusal("a record cannot cross into a concrete call; access its fields directly")
        classinfo_position = target is isinstance and position == 1 and position < positional_count
        if (
            isinstance(fact, AggregateFact)
            and not classinfo_position
            and any(isinstance(leaf, Reference) for leaf in fact.leaves)
        ):
            # sum((self,)) would hand the callable the live object through the rebuilt container; the inline
            # classinfo tuple of isinstance is the one sanctioned carrier (resolved member by member).
            raise FoldRefusal("an object reference cannot cross into a concrete call")
        oversized = (
            isinstance(fact, Known) and isinstance(fact.value, StaticRange) and range_size(fact.value) > (1 << 20)
        ) or (
            isinstance(fact, AggregateFact)
            and any(
                isinstance(leaf, Known) and isinstance(leaf.value, StaticRange) and range_size(leaf.value) > (1 << 20)
                for leaf in fact.leaves
            )
        )
        if oversized:
            raise FoldRefusal("a static fold over an oversized range is not supported")
        if isinstance(fact, Reference):
            if classinfo_position:
                continue
            referent = fact.obj
            if isinstance(referent, type) and any(referent is kind for kind in _inert_type_referents()):
                continue  # a dtype-ish builtin type carries no live state
            raise FoldRefusal("an object reference cannot cross into a concrete call")
