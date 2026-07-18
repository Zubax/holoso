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

import types
from dataclasses import MISSING, dataclass, fields

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
from ._value import MetaInt, NpInt, StaticRange


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
        list,
        tuple,
        len,
        range,
        slice,
        sum,
        divmod,
        operator_module.index,
        np.array,
        np.asarray,
        np.asanyarray,
        np.bool_,
        np.int64,
        np.float64,
    )
    return any(target is entry for entry in vetted)


@dataclass(frozen=True, slots=True)
class FieldSchema:
    """One constructible record field, snapshot into immutable form (a live Field object is mutable metadata)."""

    name: str
    kw_only: bool
    default: object  # dataclasses.MISSING when the field is required


def construction_schema(target: type) -> tuple[FieldSchema, ...]:
    """
    The validated field schema of a structurally constructible record class, in declaration order, or a
    :class:`FoldRefusal`. Structural construction never executes the class's machinery -- not even the generated
    ``__init__`` -- so eligibility is exactly the schema-match question: does plain per-field assignment of the
    call's arguments reproduce what Python's construction would, and does plain projection reproduce what a later
    field read would? A custom metaclass or ``__init__``, a ``__post_init__``/``__new__``/``__del__`` (implicit
    lifetime hooks included), a ``__getattr__``/``__getattribute__`` (which would warp later field reads), or ANY
    ``__setattr__``/``__delattr__`` beyond the dataclass-generated ones (a callable descriptor or ``None`` entry
    is not a function to source-check) makes construction or observation run user code; a ``default_factory``
    draws from live state per construction; a descriptor-backed field routes assignment through user code (a
    slots field's OWN member descriptor is the field itself, but an alien one aliased from another class writes
    into a different layout); an ``InitVar`` or ``init=False`` field makes the ``__init__`` signature diverge
    from the field schema. The parameter NAMES must be exactly the declared fields (positional ones in
    declaration order, then the kw_only ones, which is what licenses mapping call arguments onto fields
    directly) and the positional/keyword-only BOUNDARY must match the field partition. All checks are by
    presence on the class and its declared fields; a class mutated after decoration constructs per its declared
    schema (scope ruling T7 removed the decoration-vs-live forensics).
    """
    name = target.__name__
    if type(target) is not type:
        raise FoldRefusal(f"record class '{name}' has a custom metaclass, which is not supported in a kernel")
    init = next((c.__dict__["__init__"] for c in target.__mro__ if "__init__" in c.__dict__), None)
    if not (isinstance(init, types.FunctionType) and init.__code__.co_filename == "<string>"):
        raise FoldRefusal(f"record class '{name}' defines its own __init__, which is not supported in a kernel")
    hooked = any(
        name_ in c.__dict__
        for name_ in ("__post_init__", "__new__", "__del__", "__getattr__", "__getattribute__")
        for c in target.__mro__
        if c is not object
    ) or any(
        # Presence-based: a None entry (``__setattr__ = None``) is as construction-breaking as a custom hook —
        # Python raises calling it, so admitting the class would miscompile.
        not (isinstance(member := c.__dict__[name_], types.FunctionType) and member.__code__.co_filename == "<string>")
        for c in target.__mro__
        if c is not object
        for name_ in ("__setattr__", "__delattr__")
        if name_ in c.__dict__
    )
    if hooked:
        raise FoldRefusal(f"record class '{name}' runs user code in construction, which is not supported in a kernel")
    declared = tuple(fields(target))
    if any(entry.default_factory is not MISSING for entry in declared):
        raise FoldRefusal(f"record class '{name}' has a field default_factory, which is not supported in a kernel")
    for entry in declared:
        slot = next((c.__dict__[entry.name] for c in target.__mro__ if entry.name in c.__dict__), None)
        if slot is None:
            continue
        own_member = (
            isinstance(slot, types.MemberDescriptorType)
            and slot.__name__ == entry.name
            and any(slot.__objclass__ is c for c in target.__mro__)
        )
        if not own_member and (hasattr(type(slot), "__set__") or hasattr(type(slot), "__delete__")):
            raise FoldRefusal(
                f"record class '{name}' has a descriptor-backed field, which is not supported in a kernel"
            )
    code = init.__code__
    positional = tuple(entry for entry in declared if not entry.kw_only)
    keyword_only = tuple(entry for entry in declared if entry.kw_only)
    parameters = code.co_varnames[1 : code.co_argcount + code.co_kwonlyargcount]
    expected = tuple(entry.name for entry in positional) + tuple(entry.name for entry in keyword_only)
    if any(not entry.init for entry in declared) or parameters != expected or code.co_argcount != 1 + len(positional):
        raise FoldRefusal(
            f"record class '{name}' constructs through init-only or init=False fields, "
            "which is not supported in a kernel"
        )
    return tuple(FieldSchema(entry.name, entry.kw_only is True, entry.default) for entry in declared)


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
        # result: (1).to_bytes(10**12) would allocate gigabytes at compile time. The same
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
    # Argument admission: a record never crosses into a concrete evaluation (nested inside a tuple/list included)
    # -- the callable, or even the dataclass-generated __repr__, would run on a reconstruction that is
    # value-faithful but not type-faithful (an enum field rebuilds as its base value). An object reference never
    # crosses either: a stateful component's dunder would read the live reset-time object while the kernel's
    # writes exist only as state facts (float(self) stepped [1.0, 1.0] where Python steps [3.0, 5.0]).
    # A referenced dtype-ish builtin TYPE is inert.
    for fact in [*positional, *keywords]:
        if isinstance(fact, AggregateFact) and contains_record(fact.layout):
            raise FoldRefusal("a record cannot cross into a concrete call; access its fields directly")
        if isinstance(fact, AggregateFact) and any(isinstance(leaf, Reference) for leaf in fact.leaves):
            # sum((self,)) would hand the callable the live object through the rebuilt container.
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
            referent = fact.obj
            if referent is None:
                continue  # the None singleton is inert: a slice bound, an explicit sentinel
            if isinstance(referent, type) and any(referent is kind for kind in _inert_type_referents()):
                continue  # a dtype-ish builtin type carries no live state
            raise FoldRefusal("an object reference cannot cross into a concrete call")
