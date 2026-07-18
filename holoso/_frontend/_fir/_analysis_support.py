"""
Pure support surface of the analyzer: the located rejection types, fact joins and scalar typing, the
LOST-provenance taint pools, template remapping, graph reachability, concat/sequence pairing, layout geometry
queries, and the class-hook guards. Everything here is a free function over immutable inputs -- no analyzer
state -- so the SCCP orchestration in ``_analyze`` stays the only stateful surface.
"""

import enum
import itertools
import math
import types
from collections.abc import Callable

import numpy as np

from ..._errors import UnsupportedConstruct, UnsupportedLibraryFunction
from ._fact import (
    ATOM,
    AggregateFact,
    ArrayDType,
    ArrayLayout,
    AtomicFact,
    BoundFact,
    ContainerFlavor,
    Fact,
    Known,
    LayoutMismatch,
    ListLayout,
    MaybeUnbound,
    RecordLayout,
    Reference,
    Residual,
    StructuralLayout,
    TupleLayout,
    Unbound,
    ValueLayout,
    aggregate_of,
    join_layouts,
    leaf_count,
    materialize_static,
    outer_arity,
)
from ._ir import (
    BindingId,
    Fail,
    UnitExit,
    Block,
    BlockId,
    Branch,
    BuildList,
    BuildTuple,
    FunctionUnit,
    Jump,
    LoadConst,
    LoadPlace,
    LoadRef,
    Local,
    Op,
    Origin,
    OriginStack,
    Place,
    PyAttr,
    PyBin,
    PyCall,
    PyCompare,
    PyLen,
    PyNot,
    PySelect,
    PyStoreAttr,
    PySubscript,
    PyTruth,
    PyUn,
    StaticFor,
    StorePlace,
    Terminator,
    UnbindPlace,
)
from ._opsem import BinOp
from ._signature import ArrayParameter, RecordParameter, ScalarParameter
from ._value import (
    MetaInt,
    as_python,
    NpBool,
    NpFloat,
    NpInt,
    ScalarOrigin,
    SemType,
    StaticBool,
    StaticFloat,
    StaticStr,
    StaticValue,
    join_scalar_sources,
    same,
)
from dataclasses import replace

_UNBOUND = Unbound()


class AnalysisRejection(UnsupportedConstruct):
    """A located refusal discovered during analysis (dynamic structure, recursion, possibly-unbound reads...)."""

    def __init__(self, message: str, origin: OriginStack) -> None:
        frame = origin[0]
        super().__init__(f"{frame.function}:{frame.line}:{frame.column}: {message}")
        self.message = message
        self.origin = origin


class LibraryAnalysisRejection(UnsupportedLibraryFunction):
    """A recognized math/numpy library function that has no hardware implementation yet -- a sibling refusal."""

    def __init__(self, message: str, origin: OriginStack) -> None:
        frame = origin[0]
        super().__init__(f"{frame.function}:{frame.line}:{frame.column}: {message}")
        self.message = message
        self.origin = origin


def _residual_type(value: StaticValue) -> SemType | None:
    match value:
        case StaticBool() | NpBool():
            return SemType.BOOL
        case StaticFloat() | NpFloat():
            return SemType.FLOAT
        case MetaInt() | NpInt():
            return SemType.INT
        case _:
            return None


def _scalar_sem(fact: "Fact") -> SemType | None:
    """The scalar kind of a fact, BOOL included; None for aggregates, references, and unbound."""
    match fact:
        case Known(value=value):
            return _residual_type(value)
        case Residual(type=t):
            return t
        case _:
            return None


def _numeric_sem(fact: "Fact") -> SemType | None:
    """FLOAT or INT for a numeric fact (a Known number or a Residual FLOAT/INT); None for bool, aggregate, or unbound."""
    match fact:
        case Known(value=value):
            sem = _residual_type(value)
            return sem if sem in (SemType.FLOAT, SemType.INT) else None
        case Residual(type=t) if t in (SemType.FLOAT, SemType.INT):
            return t
        case _:
            return None


def _float_promoted(fact: AtomicFact, origin: OriginStack) -> AtomicFact:
    """
    The C-style promotion applied at an int/float merge: the integer side becomes float, its provenance kept
    (MetaInt -> StaticFloat, NpInt -> NpFloat) and its rounding accepted under the fastmath charter. An integer
    beyond the binary64 carrier cannot promote at all and is a located rejection, never a raw OverflowError.
    """
    match fact:
        case Known(value=(MetaInt() | NpInt()) as value):
            try:
                promoted = float(value.value)
            except OverflowError:
                bits = value.value.bit_length()  # never via str(): the 4300-digit conversion cap
                raise AnalysisRejection(
                    f"a {bits}-bit integer merged with a float is beyond the binary64 carrier range", origin
                ) from None
            return Known(NpFloat(promoted) if isinstance(value, NpInt) else StaticFloat(promoted))
        case Residual(type=SemType.INT):
            return Residual(SemType.FLOAT)
        case _:
            return fact


def join_facts(a: Fact, b: Fact, origin: OriginStack) -> Fact:
    if a is b:
        return a
    match a, b:
        case (Unbound(), Unbound()):
            return _UNBOUND
        case (Unbound(), (Known() | Residual() | Reference() | AggregateFact()) as bound) | (
            (Known() | Residual() | Reference() | AggregateFact()) as bound,
            Unbound(),
        ):
            return MaybeUnbound(bound)
        case (Unbound(), MaybeUnbound() as half) | (MaybeUnbound() as half, Unbound()):
            return half
        case (MaybeUnbound(inner=x), MaybeUnbound(inner=y)):
            joined = join_facts(x, y, origin)
            assert isinstance(joined, (Known, Residual, Reference, AggregateFact))
            return MaybeUnbound(joined)
        case (MaybeUnbound(inner=x), (Known() | Residual() | Reference() | AggregateFact()) as y) | (
            (Known() | Residual() | Reference() | AggregateFact()) as y,
            MaybeUnbound(inner=x),
        ):
            joined = join_facts(x, y, origin)
            assert isinstance(joined, (Known, Residual, Reference, AggregateFact))
            return MaybeUnbound(joined)
        case (AggregateFact() as x, AggregateFact() as y):
            try:
                layout = join_layouts(x.layout, y.layout)
            except LayoutMismatch as error:
                raise AnalysisRejection(str(error), origin) from None
            assert layout is not None
            leaves = tuple(_join_atoms(p, q, origin) for p, q in zip(x.leaves, y.leaves, strict=True))
            return AggregateFact(layout, leaves)
        case ((Known() | Residual() | Reference()) as p, (Known() | Residual() | Reference()) as q):
            return _join_atoms(p, q, origin)
    raise AnalysisRejection("values of irreconcilable shapes merge here", origin)


def _join_atoms(a: AtomicFact, b: AtomicFact, origin: OriginStack) -> AtomicFact:
    """The scalar join: same-kind residualization plus the C-style int/float promotion (see the module docstring)."""
    if a is b:
        return a
    match a, b:
        case (Reference(), Reference()):
            if a.obj is b.obj:
                return a
            if a.obj is None or b.obj is None:
                raise AnalysisRejection("None merges with a value here (a conditional None is not supported)", origin)
            raise AnalysisRejection("values of irreconcilable kinds merge here", origin)
        case (Reference() as lone, _) | (_, Reference() as lone):
            if lone.obj is None:
                raise AnalysisRejection("None merges with a value here (a conditional None is not supported)", origin)
            raise AnalysisRejection("values of irreconcilable kinds merge here", origin)
        case (Known(value=x), Known(value=y)):
            if same(x, y):
                return a
            degraded = join_scalar_sources(x, y)
            if degraded is not None:
                return Known(degraded)
            x_type, y_type = _residual_type(x), _residual_type(y)
            if {x_type, y_type} == {SemType.FLOAT, SemType.INT}:  # an int/float merge promotes the integer, C-style
                return _join_atoms(_float_promoted(a, origin), _float_promoted(b, origin), origin)
            if x_type is not None and x_type == y_type:
                return Residual(x_type)
            raise AnalysisRejection("values of irreconcilable kinds merge here", origin)
        case (Known(value=x), Residual(type=t)) | (Residual(type=t), Known(value=x)):
            x_type = _residual_type(x)
            if x_type == t:
                return Residual(t)
            if x_type is not None and {x_type, t} == {SemType.FLOAT, SemType.INT}:
                return _join_atoms(_float_promoted(a, origin), _float_promoted(b, origin), origin)
            raise AnalysisRejection("values of irreconcilable kinds merge here", origin)
        case (Residual(type=x_t), Residual(type=y_t)):
            if x_t == y_t:
                return a
            if {x_t, y_t} == {SemType.FLOAT, SemType.INT}:
                return Residual(SemType.FLOAT)  # a runtime integer merged with a float promotes
            raise AnalysisRejection("values of irreconcilable kinds merge here", origin)
    raise AssertionError((a, b))


class DeferredRejection:
    """
    Collects rejections raised across an unordered iteration and re-raises the lexicographically least, so the
    surfaced diagnostic does not depend on the iteration order (place and state-leaf hashes involve binding
    names and object addresses, so that order is not reproducible across processes).
    """

    def __init__(self) -> None:
        self._best: AnalysisRejection | None = None

    def offer(self, error: AnalysisRejection) -> None:
        if self._best is None or str(error) < str(self._best):
            self._best = error

    def raise_if_deferred(self) -> None:
        if self._best is not None:
            raise self._best


def _identity_place(place: Place) -> Place:
    return place


def _remap_op(op: Op, fresh: Callable[[BindingId], BindingId], remap_place: Callable[[Place], Place]) -> Op:
    match op:
        case LoadConst() | LoadRef():
            return replace(op, dst=fresh(op.dst))
        case LoadPlace():
            return replace(op, dst=fresh(op.dst), place=remap_place(op.place))
        case StorePlace():
            return replace(op, src=fresh(op.src), place=remap_place(op.place))
        case UnbindPlace():
            return replace(op, place=remap_place(op.place))
        case PyBin() | PyCompare():
            return replace(op, dst=fresh(op.dst), lhs=fresh(op.lhs), rhs=fresh(op.rhs))
        case PyUn() | PyNot() | PyTruth():
            return replace(op, dst=fresh(op.dst), operand=fresh(op.operand))
        case PySelect():
            return replace(op, dst=fresh(op.dst), cond=fresh(op.cond), lhs=fresh(op.lhs), rhs=fresh(op.rhs))
        case PyCall():
            return replace(
                op,
                dst=fresh(op.dst),
                callee=fresh(op.callee),
                args=tuple(fresh(arg) for arg in op.args),
                kwargs=tuple((keyword, fresh(value)) for keyword, value in op.kwargs),
            )
        case PyAttr() | PyLen():
            return replace(op, dst=fresh(op.dst), obj=fresh(op.obj))
        case PyStoreAttr():
            return replace(op, obj=fresh(op.obj), src=fresh(op.src))
        case PySubscript():
            return replace(op, dst=fresh(op.dst), obj=fresh(op.obj), index=fresh(op.index))
        case BuildTuple() | BuildList():
            return replace(op, dst=fresh(op.dst), items=tuple(fresh(item) for item in op.items))


def _remap_terminator(
    terminator: Terminator,
    remap_block: Callable[[BlockId], BlockId],
    fresh: Callable[[BindingId], BindingId],
    remap_place: Callable[[Place], Place],
) -> Terminator:
    match terminator:
        case Jump(target=target, origin=origin):
            return Jump(remap_block(target), origin)
        case Branch(cond=cond, then_target=then_target, else_target=else_target, origin=origin):
            return Branch(fresh(cond), remap_block(then_target), remap_block(else_target), origin)
        case StaticFor(
            target=target,
            iterable=iterable,
            body_entry=body_entry,
            exit_target=exit_target,
            body_blocks=body_blocks,
            origin=origin,
        ):
            return StaticFor(
                remap_place(target),
                fresh(iterable),
                remap_block(body_entry),
                remap_block(exit_target),
                frozenset(remap_block(member) for member in body_blocks),
                origin,
            )
        case Fail(parts=parts, origin=origin):
            remapped = tuple(part if isinstance(part, str) else fresh(part) for part in parts)
            return Fail(remapped, origin)  # a COPY: grafting re-attributes origins and must never touch templates
        case UnitExit(origin=origin):
            return UnitExit(origin)
    raise AssertionError(terminator)


def _coreachable(
    unit: FunctionUnit, exit_block: BlockId, executable_edges: set[tuple[BlockId, BlockId]]
) -> set[BlockId]:
    predecessors: dict[BlockId, list[BlockId]] = {}
    for source, target in executable_edges:
        predecessors.setdefault(target, []).append(source)
    seen = {exit_block}
    pending = [exit_block]
    while pending:
        for predecessor in predecessors.get(pending.pop(), ()):
            if predecessor not in seen:
                seen.add(predecessor)
                pending.append(predecessor)
    return seen


def _same_fact(a: Fact, b: Fact) -> bool:
    """Fixed-point stability: Knowns compare by tagged bitwise sameness, everything else structurally."""
    if isinstance(a, Known) and isinstance(b, Known):
        return same(a.value, b.value)
    return a == b


def _lost_scalar_pools(facts: "list[Fact]") -> tuple[set[int], set[str]]:
    """The base values of every LOST-provenance int/str among the given facts (aggregate leaves included)."""
    ints: set[int] = set()
    strs: set[str] = set()

    def visit(value: StaticValue) -> None:
        match value:
            case MetaInt(value=v, source=source) if source is ScalarOrigin.LOST:
                ints.add(v)
            case StaticStr(value=v, source=source) if source is ScalarOrigin.LOST:
                strs.add(v)

    for fact in facts:
        if isinstance(fact, Known):
            visit(fact.value)
        elif isinstance(fact, AggregateFact):
            for leaf in fact.leaves:
                if isinstance(leaf, Known):
                    visit(leaf.value)
    return ints, strs


def _taint_lost(fact: BoundFact, pools: tuple[set[int], set[str]]) -> BoundFact:
    """
    Downgrade PLAIN int/str scalars of a fold's result to LOST when they value-equal a LOST input: an
    identity-preserving callable (min returns its argument, sum returns its start) may have returned the very
    object whose membership the LOST input no longer names, and reconstruction laundered it to a plain value.
    """
    ints, strs = pools
    if not ints and not strs:
        return fact

    def scalar(value: StaticValue) -> StaticValue:
        match value:
            case MetaInt(value=v, source=source) if source is ScalarOrigin.PLAIN and v in ints:
                return MetaInt(v, source=ScalarOrigin.LOST)
            case StaticStr(value=v, source=source) if source is ScalarOrigin.PLAIN and v in strs:
                return StaticStr(v, source=ScalarOrigin.LOST)
        return value

    match fact:
        case Known(value=value):
            return Known(scalar(value))
        case AggregateFact(layout=layout, leaves=leaves):
            return AggregateFact(
                layout, tuple(Known(scalar(leaf.value)) if isinstance(leaf, Known) else leaf for leaf in leaves)
            )
    return fact


def _datapath_zero(value: object) -> object:
    """Normalize a -0.0 fold input to +0.0: the ZKF datapath has no signed zero, so a static fold must not either."""
    return value + 0.0 if isinstance(value, float) and value == 0.0 else value


def _crossing_object(value: "StaticValue | Reference") -> object:
    """
    The Python object an admitted argument denotes at the evaluation boundary: a value reconstructs through
    as_python; an admitted reference (an inert dtype-ish type, or a classinfo reaching a malformed isinstance
    spelling) crosses as the referent itself -- identity, not reconstruction, is its meaning.
    """
    if isinstance(value, Reference):
        return value.obj
    return _datapath_zero(as_python(value))


def _concrete_fact(fact: Fact) -> StaticValue | None:
    """The concrete static value behind a fact, when one exists: a Known directly, an all-Known aggregate rebuilt."""
    if isinstance(fact, Known):
        return fact.value
    if isinstance(fact, AggregateFact):
        return materialize_static(fact)
    return None


def _concat_seqs(bin_op: BinOp, lhs: Fact, rhs: Fact) -> Fact | None:
    if bin_op is BinOp.MUL:
        seq, count = (lhs, rhs) if _seq_side(lhs) is not None else (rhs, lhs)
        lifted = _seq_side(seq)
        # A plain-bool count repeats 0/1 times exactly as Python; the np.bool_ spelling falls through to the
        # arithmetic rejection (numpy 2 dropped its __index__, a Python TypeError). The count must fit Python's
        # ssize_t index range -- beyond it CPython raises OverflowError rather than clamping.
        if lifted is not None and isinstance(count, Known) and isinstance(count.value, (MetaInt, NpInt, StaticBool)):
            repetitions = int(count.value.value)
            if -(2**63) <= repetitions <= 1024:
                children = tuple(lifted.child(i) for i in range(outer_arity(lifted.layout)))
                repeated = children * max(0, repetitions)  # Python: a negative count is the empty sequence
                return aggregate_of(repeated, is_list=isinstance(lifted.layout, ListLayout))
        return None
    if bin_op is not BinOp.ADD:
        return None
    left, right = _seq_side(lhs), _seq_side(rhs)
    if left is None or right is None or type(left.layout) is not type(right.layout):
        return None  # Python: list + tuple is a TypeError, and a flavor-erased structural side has no ``+``
    children = tuple(left.child(i) for i in range(outer_arity(left.layout))) + tuple(
        right.child(i) for i in range(outer_arity(right.layout))
    )
    return aggregate_of(children, is_list=isinstance(left.layout, ListLayout))


def _seq_side(fact: Fact) -> AggregateFact | None:
    """A pure tuple/list-flavored aggregate; arrays and flavor-erased structural joins have different operators."""
    if isinstance(fact, AggregateFact) and isinstance(fact.layout, (TupleLayout, ListLayout)):
        return fact
    return None


def _is_list_fact(fact: Fact) -> bool:
    return isinstance(fact, AggregateFact) and isinstance(fact.layout, ListLayout)


def _is_array_fact(fact: Fact) -> bool:
    return isinstance(fact, AggregateFact) and isinstance(fact.layout, ArrayLayout)


def _contract_structure(
    contract: "ScalarParameter | ArrayParameter | RecordParameter",
) -> tuple[ValueLayout, list[SemType]]:
    """A parameter contract's fact layout plus its leaf kinds in canonical order."""
    match contract:
        case ScalarParameter(kind=kind):
            return None, [kind]
        case ArrayParameter(shape=shape):
            return ArrayLayout(shape, ArrayDType.FLOAT), [SemType.FLOAT] * leaf_count(
                ArrayLayout(shape, ArrayDType.FLOAT)
            )
        case RecordParameter(klass=klass, fields=fields):
            field_layouts: list[tuple[str, ValueLayout]] = []
            kinds: list[SemType] = []
            for name, sub in fields:
                sub_layout, sub_kinds = _contract_structure(sub)
                field_layouts.append((name, sub_layout))
                kinds.extend(sub_kinds)
            return RecordLayout(klass, tuple(field_layouts)), kinds
    raise AssertionError(contract)


def _transpose_routes(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Source leaf ordinal per result leaf of a full transpose: result (i_n..i_0) reads source (i_0..i_n)."""
    import itertools

    strides: list[int] = []
    span = 1
    for dimension in reversed(shape):
        strides.append(span)
        span *= dimension
    strides.reverse()
    routes: list[int] = []
    for coordinates in itertools.product(*(range(dimension) for dimension in reversed(shape))):
        routes.append(sum(c * strides[axis] for axis, c in enumerate(reversed(coordinates))))
    return tuple(routes)


def _fits_float64(value: int) -> bool:
    try:
        float(value)
    except OverflowError:
        return False
    return True


def _rectangular_shape(layout: "ValueLayout") -> tuple[int, ...] | None:
    """
    The array shape a layout tree yields under numpy's rectangular nesting rules (container flavor is
    irrelevant; a 0-d array child acts as a scalar), or None where numpy itself would refuse the ragged form.
    """
    if layout is None:
        return ()
    match layout:
        case ArrayLayout(shape=shape):
            return shape
        case TupleLayout(items=items) | ListLayout(items=items) | StructuralLayout(items=items):
            inner = {_rectangular_shape(item) for item in items}
            if None in inner or len(inner) > 1:
                return None
            common = next(iter(inner), ())
            assert common is not None
            return (len(items), *common)
    return None  # a record never reaches an array factory (the admission walk refuses it as an argument)


def _layout_dtypes(layout: "ValueLayout") -> set[ArrayDType]:
    """Embedded array dtypes: the evidence an empty array child contributes to numpy's dtype discovery."""
    match layout:
        case ArrayLayout(dtype=dtype):
            return {dtype}
        case TupleLayout(items=items) | ListLayout(items=items) | StructuralLayout(items=items):
            return {dtype for item in items for dtype in _layout_dtypes(item)}
    return set()


def _mro_attribute_of(klass: type, name: str) -> object | None:
    return next((c.__dict__[name] for c in klass.__mro__ if name in c.__dict__), None)


def _has_truth_override(klass: type) -> bool:
    """
    Whether a user-defined __bool__/__len__ ENTRY governs the class's truth (object's own default excluded).
    Membership, not value: ``__bool__ = None`` is a real override (Python raises TypeError on its truth).
    """
    return any(name in c.__dict__ for name in ("__bool__", "__len__") for c in klass.__mro__ if c is not object)


def _reject_attribute_hooks(klass: type, origin: OriginStack) -> None:
    plain_setattr = (_mro_attribute_of(object, "__setattr__"), _mro_attribute_of(type, "__setattr__"))
    plain_getattribute = (
        _mro_attribute_of(object, "__getattribute__"),
        _mro_attribute_of(type, "__getattribute__"),
    )
    if _mro_attribute_of(klass, "__setattr__") not in plain_setattr:
        raise AnalysisRejection("components with a custom __setattr__ are not supported", origin)
    if _mro_attribute_of(klass, "__getattribute__") not in plain_getattribute or (
        _mro_attribute_of(klass, "__getattr__") is not None
    ):
        raise AnalysisRejection("components with custom attribute hooks are not supported", origin)


def _reject_descriptor(klass: type, name: str, origin: OriginStack) -> None:
    # Raw MRO lookup, never getattr (which would run a property getter). A data descriptor (``__set__``/``__delete__``)
    # -- a property with or without a setter, a property subclass, or any other data descriptor -- cannot back a
    # writable component attribute, since its accessor would bypass the abstract state. Property GETTER reads are
    # handled earlier; only stores and unsupported reads reach here.
    descriptor = _mro_attribute_of(klass, name)
    if (
        descriptor is not None
        and (hasattr(type(descriptor), "__set__") or hasattr(type(descriptor), "__delete__"))
        and not isinstance(descriptor, types.MemberDescriptorType)  # slots ARE the fields, not accessors
    ):
        raise AnalysisRejection(
            f"descriptor '{name}' on a component is not supported (it would bypass abstract state)", origin
        )
