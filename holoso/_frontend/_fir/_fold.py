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
from ._value import MetaInt, NpInt, ScalarOrigin, StaticRange, StaticStr


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


def validate_classinfo(classinfo: "Fact | None") -> list[type]:
    """
    The isinstance classinfo completely resolved to enum-free plain types whose instance check is type's own, or
    a :class:`FoldRefusal`. An ABC's register()/__instancecheck__ distinguishes a live enum member from its
    erased value, and an enum classinfo would compare against the erased side, so both refuse.
    """
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
    return kinds


def _tuple_only_layout(layout: object) -> bool:
    if layout is None:
        return True
    if isinstance(layout, TupleLayout):
        return all(_tuple_only_layout(item) for item in layout.items)
    return False


def classinfo_types(fact: "Fact | None") -> list[type] | None:
    """
    The plain types an isinstance classinfo resolves to, or None when any member is opaque (a non-type, a typing
    generic, an unresolvable reference) or the container is not a tuple (Python raises TypeError on a LIST
    classinfo, so folding it as if it were a tuple would accept what Python rejects). Tuples and unions unpack
    recursively, so a precomputed ``(float, Mode)`` or ``str | Mode`` is inspected member by member instead of
    slipping past as one reference.
    """
    pending: list[object] = []
    match fact:
        case Reference(obj=obj):
            pending = [obj]
        case AggregateFact(layout=layout, leaves=leaves):
            if not _tuple_only_layout(layout):
                return None
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
        list,
        tuple,
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
    from the field schema. The certification is load-bearing in three parts: the parameter NAMES must be exactly
    the declared fields (positional ones in declaration order, then the kw_only ones, which is what licenses
    mapping call arguments onto fields directly), the positional/keyword-only BOUNDARY must match the field
    partition, and the live initializer DEFAULTS must be the field metadata objects themselves (a
    post-decoration ``__defaults__`` mutation makes Python construct with values the schema never saw). A
    ``__post_init__`` deleted after decoration still leaves its call in the generated bytecode (Python raises
    AttributeError at runtime), so the code object is consulted, not only the class dictionaries.
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
        not (isinstance(member, types.FunctionType) and member.__code__.co_filename == "<string>")
        for c in target.__mro__
        if c is not object
        for name_ in ("__setattr__", "__delattr__")
        if (member := c.__dict__.get(name_)) is not None
    )
    if hooked or "__post_init__" in init.__code__.co_names:
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
    expected_defaults = tuple(entry.default for entry in positional if entry.default is not MISSING)
    actual_defaults = init.__defaults__ or ()
    expected_kwdefaults = {entry.name: entry.default for entry in keyword_only if entry.default is not MISSING}
    actual_kwdefaults = init.__kwdefaults__ or {}
    aligned = (
        len(actual_defaults) == len(expected_defaults)
        and all(a is b for a, b in zip(actual_defaults, expected_defaults))
        and set(actual_kwdefaults) == set(expected_kwdefaults)
        and all(actual_kwdefaults[key] is value for key, value in expected_kwdefaults.items())
    )
    if not aligned:
        raise FoldRefusal(
            f"record class '{name}' has initializer defaults diverging from its field schema, "
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
        # Enum members normalize to their base value at admission with the member retained as the scalar's
        # source, so an isinstance subject reconstructs faithfully (a mixin base of the enum answers exactly as
        # Python) -- EXCEPT when a join dropped the source: a LOST-provenance int/str may be a member the fact
        # no longer names, so the query refuses. The classinfo must still RESOLVE COMPLETELY to enum-free plain
        # types whose instance check is type's own (an ABC's register()/__instancecheck__ distinguishes the
        # live member from its erased value, and an enum classinfo would compare against the erased side).
        subject = positional[0] if positional else None
        if (
            isinstance(subject, Known)
            and isinstance(subject.value, (MetaInt, StaticStr))
            and subject.value.source is ScalarOrigin.LOST
        ):
            raise FoldRefusal("isinstance of a static int/str is not decidable (it may be a normalized enum member)")
        validate_classinfo(positional[1] if len(positional) == 2 else None)
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
