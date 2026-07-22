"""
Pure support surface of the analyzer: the located rejection types, fact joins and scalar typing, template
remapping, graph reachability, concat/sequence pairing, layout geometry queries, and the class-hook guards.
Everything here is a free function over immutable inputs -- no analyzer state -- so the SCCP orchestration in
``_analyze`` stays the only stateful surface.
"""

import enum
import itertools
import math
import sys
import types
from collections.abc import Callable, Mapping

import numpy as np

from ..._errors import UnsupportedConstruct, UnsupportedLibraryFunction
from ._fact import (
    ATOM,
    AggregateFact,
    ArrayDType,
    ArrayLayout,
    AtomicFact,
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
    StoreRole,
    Terminator,
    UnbindPlace,
    LocatedRejection,
    executable_preorder,
    OriginOrder,
    origin_order,
    source_position,
)
from ._opsem import BinOp
from ._signature import ArrayParameter, RecordParameter, ScalarParameter
from ._value import (
    MetaInt,
    as_python,
    NpBool,
    NpFloat,
    NpInt,
    SemType,
    StaticBool,
    StaticFloat,
    StaticValue,
    same,
)
from dataclasses import dataclass, fields, is_dataclass, replace

_UNBOUND = Unbound()


class AnalysisRejection(LocatedRejection, UnsupportedConstruct):
    """A located refusal discovered during analysis (dynamic structure, recursion, possibly-unbound reads...)."""


class LibraryAnalysisRejection(LocatedRejection, UnsupportedLibraryFunction):
    """A recognized math/numpy library function that has no hardware implementation yet -- a sibling refusal."""


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
    sem = _scalar_sem(fact)
    return sem if sem in (SemType.FLOAT, SemType.INT) else None


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


def _deferral_key(error: LocatedRejection) -> tuple[str, OriginOrder]:
    """Rendered text first, so the historical selection is preserved, then the origin the text cannot show."""
    return str(error), origin_order(error.origin)


class DeferredRejection:
    """
    Collects rejections raised across an unordered iteration and re-raises the lexicographically least, so the
    surfaced diagnostic does not depend on the iteration order (place and state-leaf hashes involve binding
    names and object addresses, so that order is not reproducible across processes).
    """

    def __init__(self) -> None:
        self._best: LocatedRejection | None = None

    def offer(self, error: LocatedRejection) -> None:
        # The rendered text alone is not a total order: two leaves can differ only in which file their store
        # sits in, which the message never names, and then set iteration would decide the public origin.
        if self._best is None or _deferral_key(error) < _deferral_key(self._best):
            self._best = error

    def pending(self) -> bool:
        return self._best is not None

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


def same_fact(a: Fact, b: Fact) -> bool:
    """Fixed-point stability: Knowns compare by tagged bitwise sameness, everything else structurally."""
    if isinstance(a, Known) and isinstance(b, Known):
        return same(a.value, b.value)
    return a == b


def _datapath_zero(value: object) -> object:
    """Normalize a -0.0 fold input to +0.0: the ZKF datapath has no signed zero, so a static fold must not either."""
    return value + 0.0 if isinstance(value, float) and value == 0.0 else value


def _crossing_object(value: "StaticValue | Reference") -> object:
    """
    The Python object an admitted argument denotes at the evaluation boundary: a value reconstructs through
    as_python; an admitted reference (an inert dtype-ish type) crosses as the referent itself -- identity,
    not reconstruction, is its meaning.
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


_MAX_DECIMAL_DIGITS = 4000  # our own ceiling, conservatively below CPython's default 4300-digit conversion cap
_DECIMAL_DIGIT_MARGIN = 16  # headroom below a runtime-lowered cap so the digit overestimate never lands on it


def _decimal_digit_ceiling() -> int:
    # Read the int-to-decimal cap at CALL time: it is mutable at runtime (sys.set_int_max_str_digits,
    # PYTHONINTMAXSTRDIGITS), so a snapshot at import would miss a later lowering and resurface the raw
    # ValueError for values between the lowered cap and our own ceiling. A returned 0 means the cap is disabled
    # (unbounded), in which case only our own ceiling constrains the decimal spelling.
    runtime = sys.get_int_max_str_digits()
    if runtime <= 0:
        return _MAX_DECIMAL_DIGITS
    return min(_MAX_DECIMAL_DIGITS, runtime - _DECIMAL_DIGIT_MARGIN)


def _is_wide_int(value: int) -> bool:
    # 30103/100000 slightly overestimates log10(2), so the digit bound errs toward hex, never toward the cap.
    return value.bit_length() * 30103 // 100000 + 1 > _decimal_digit_ceiling()


def _int_spelling(value: int) -> str:
    return f"{value:#x}" if _is_wide_int(value) else str(value)


def _contains_wide_int(value: object) -> bool:
    """
    Whether any integer reachable through the value's standard structure would trip the digit cap. A dataclass
    is probed through EVERY field, not only its repr=True ones: a custom __repr__/__format__ may decimalize a
    field the generated repr would hide, so the crash-avoidance must not assume the generated traversal.
    """
    match value:
        case bool():
            return False
        case int():
            return _is_wide_int(value)
        case list() | tuple():
            return any(_contains_wide_int(item) for item in value)
        case range():
            return any(_is_wide_int(bound) for bound in (value.start, value.stop, value.step))
        case slice():
            return any(_contains_wide_int(part) for part in (value.start, value.stop, value.step))
        case _ if is_dataclass(value) and not isinstance(value, type):
            return any(_contains_wide_int(getattr(value, f.name)) for f in fields(value))
        case _:
            return False


def render_interpolation(value: object) -> str:
    """
    The f-string spelling of a folded interpolation: format() at the top level and repr() inside containers,
    exactly as Python nests them, except that an integer wide enough to overflow CPython's int-to-decimal digit
    cap spells in hex -- folding admits such integers, so the rule applies recursively through every container
    as_python can produce lest a nested wide element crash the render with the raw conversion ValueError.
    """
    return _render_value(value, nested=False)


def _render_value(value: object, nested: bool) -> str:
    match value:
        case bool():
            return str(value)
        case int():
            return _int_spelling(value)
        case list():
            return f"[{', '.join(_render_value(item, nested=True) for item in value)}]"
        case tuple() if len(value) == 1:
            return f"({_render_value(value[0], nested=True)},)"
        case tuple():
            return f"({', '.join(_render_value(item, nested=True) for item in value)})"
        case range():
            bounds = [value.start, value.stop] + ([value.step] if value.step != 1 else [])
            return f"range({', '.join(_int_spelling(bound) for bound in bounds)})"
        case slice():
            parts = (value.start, value.stop, value.step)
            return f"slice({', '.join(_render_value(part, nested=True) for part in parts)})"
        case _ if is_dataclass(value) and not isinstance(value, type):
            # A folded StaticRecord reconstructs the user's dataclass. Only INTERCEPT when a wide int is actually
            # reachable: the generated repr would then decimalize it and crash on the conversion cap, so spell the
            # record field-by-field through the digit-safe renderer (repr=True fields, declaration order). With no
            # wide int present, native repr/format is byte-identical to Python's f-string -- nested qualname, a
            # custom __repr__, a custom __format__ all preserved -- which the synthetic field list would discard.
            if _contains_wide_int(value):
                body = ", ".join(
                    f"{f.name}={_render_value(getattr(value, f.name), nested=True)}" for f in fields(value) if f.repr
                )
                return f"{type(value).__name__}({body})"
            return repr(value) if nested else format(value)
        case _:
            return repr(value) if nested else format(value)


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
    irrelevant), or None where numpy itself would refuse the ragged form.
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


# ---------------------------------------- the fixed storage schema (B1) ----------------------------------------


@dataclass(frozen=True, slots=True)
class ScalarSchema:
    kind: SemType


@dataclass(frozen=True, slots=True)
class ContradictorySchema:
    """
    Paths established irreconcilable schemas for one place without any single store being a rebinding (possible
    only through ``del`` corners, since a live fact merge of such paths rejects first). No store satisfies it.
    """


type StorageSchema = ScalarSchema | ContradictorySchema


@dataclass(frozen=True, slots=True)
class StoreVerdict:
    """
    The outcome of one store op's last BOUND execution within the current round; ``message`` None means the
    store conformed. The last bound visit runs under the block's converged environment, so this is the fixpoint
    verdict: an earlier violation drawn on a pre-join transient is superseded. An Unbound execution (a producer
    cut by the deferral cascade) records no verdict at all -- it is evidence of nothing, neither conformance
    nor violation -- but marks its origin as executed-unbound, which blocks the bridge pop.
    """

    message: str | None
    origin: OriginStack


def schema_of_fact(fact: Fact) -> ScalarSchema | None:
    """
    The storage schema a stored fact establishes for a source variable: its scalar SemType kind. The schema sees
    SemType kinds only -- an aggregate-valued store is fact-only, like a reference or a string: local aggregates
    are value dataflow whose shape may be re-represented freely (``v = v.reshape(...)``, the accumulator idiom
    the in-place-mutation rejection recommends), and their leaf kinds ride the fact flow. Only persistent state,
    whose reset fixes a reconstruction contract, enforces flavor, geometry, and per-cell kinds (at its own door).
    """
    match fact:
        case Known() | Residual():
            kind = _scalar_sem(fact)
            return ScalarSchema(kind) if kind is not None else None
        case _:
            return None


def join_schemas(a: StorageSchema, b: StorageSchema) -> StorageSchema:
    """The establishing join of independent first definitions: int promotes to float, like the fact join."""
    if a == b:
        return a
    match a, b:
        case (ScalarSchema(kind=x), ScalarSchema(kind=y)):
            if {x, y} == {SemType.INT, SemType.FLOAT}:
                return ScalarSchema(SemType.FLOAT)
            return ContradictorySchema()
        case _:
            return ContradictorySchema()


def _admit_rebinding(current: StorageSchema, stored: StorageSchema) -> StorageSchema | None:
    """The schema after an acceptable rebinding store (bool<-bool, int<-int, float<-float|int), else None."""
    match current, stored:
        case (ScalarSchema(kind=s), ScalarSchema(kind=k)):
            if s is k or (s is SemType.FLOAT and k is SemType.INT):
                return current
            return None
        case _:
            return None


def _binary64_store_image(fact: AtomicFact, subject: str) -> tuple[AtomicFact, str | None]:
    """
    The float fact an integer becomes on the edge of a store into a float-schema variable or state cell, plus
    the exactness violation it commits, if any. A statically Known integer is exact-or-reject, unlike the
    rounding merge promotion: a merge presents the integer in a float position, where fastmath rounding is
    chartered, but a plain assignment silently changing the stored value would be exactly the spelling-dependent
    divergence the storage schema exists to kill -- the explicit float(...) cast is the spelling that accepts
    the rounding. A genuinely runtime integer instead converts at runtime with the hardware conversion's
    round-to-nearest, the same static-refuses/runtime-defers boundary NaN handling uses. Never raises: the
    verdict belongs to the post-stabilization walk, so the carried image (the rounded float, or a float
    residual past the carrier) keeps the fixed point stable even when the fact is a transient pre-join one.
    """
    match fact:
        case Known(value=(MetaInt() | NpInt()) as value):
            try:
                image = float(value.value)
            except OverflowError:
                bits = value.value.bit_length()  # never via str(): the 4300-digit conversion cap
                return Residual(SemType.FLOAT), (
                    f"{subject} is a float; a {bits}-bit integer stored into it is beyond the binary64 carrier range"
                )
            converted = Known(NpFloat(image) if isinstance(value, NpInt) else StaticFloat(image))
            if int(image) != value.value:
                return converted, (
                    f"{subject} is a float; the stored integer is not exactly representable in the "
                    "binary64 carrier (write float(...) to accept the rounding)"
                )
            return converted, None
        case Residual(type=SemType.INT):
            return Residual(SemType.FLOAT), None
    raise AssertionError(fact)


def conform_local_store(
    current: StorageSchema | None, name: str, stored: Fact
) -> tuple[StorageSchema | None, Fact, str | None]:
    """
    The schema a SOURCE store leaves on its local, the fact it stores, and the violation it commits, if any.
    A scalar datapath store establishes an absent schema and must keep an established one; an integer admitted
    into a float schema converts through the binary64 store edge, so an exact integer fact never survives inside
    a float variable. On a rebinding violation the schema and the fact pass through unchanged; on a conversion
    violation the fact carries the store-edge image. Never raises: the verdict belongs to the post-stabilization
    walk, which reports the first violating store in CFG preorder.
    """
    stored_schema = schema_of_fact(stored)
    if stored_schema is None:
        return current, stored, None
    if current is None:
        return stored_schema, stored, None
    admitted = _admit_rebinding(current, stored_schema)
    if admitted is None:
        return (
            current,
            stored,
            f"variable '{name}' is {describe_schema(current)} and cannot be rebound to "
            f"{describe_schema(stored_schema)}; variables are strongly typed (bind a new name instead)",
        )
    if isinstance(admitted, ScalarSchema) and admitted.kind is SemType.FLOAT and stored_schema.kind is SemType.INT:
        assert isinstance(stored, (Known, Residual))
        image, message = _binary64_store_image(stored, f"variable '{name}'")
        return admitted, image, message
    return admitted, stored, None


def describe_schema(schema: StorageSchema) -> str:
    match schema:
        case ScalarSchema(kind=kind):
            return {SemType.BOOL: "a bool", SemType.INT: "an int", SemType.FLOAT: "a float"}[kind]
        case ContradictorySchema():
            return "of no single established type"
    raise AssertionError(schema)


def conform_state_store(name: str, reset: Fact, stored: Fact) -> tuple[Fact, str | None]:
    """
    The fact a state store leaves in its slot plus the schema violation it commits, if any. The reset fixes the
    slot schema -- container flavor, exact geometry, and per-cell kind -- and a store may only keep it: bool
    cells accept bool, int cells int, float cells float or int (the integer converts through the binary64 store
    edge, exactly like the local rule: a Known is exact-or-reject, a runtime integer converts at runtime). A
    violation reports after stabilization, at this store, so the fact carried onward must keep the fixed point
    stable AND free of misleading secondary rejections: an int slot receiving float (a pure numeric widening)
    carries the stored fact, whose W/D join merely descends; a failed conversion carries its store-edge image;
    every other violation carries the residualized reset, since joining the stored fact would raise a
    worse-located mismatch first. A non-datapath stored value (the Unbound left by a deferred producer chain
    included) neither establishes nor violates, in the scalar and aggregate arms alike; the W/D join owns it.
    """
    if isinstance(reset, Reference):
        return reset, (
            f"state attribute '{name}' cannot persist: its reset is not admissible "
            "(a plain numpy array or a flat list of scalars is required)"
        )
    if isinstance(reset, AggregateFact):
        if not isinstance(stored, AggregateFact):
            if _scalar_sem(stored) is None:
                return stored, None
            return reset, f"state attribute '{name}' persists an aggregate; a scalar cannot be stored into it"
        assert isinstance(reset.layout, (ListLayout, ArrayLayout)), "the reset schema was validated at its read"
        if type(stored.layout) is not type(reset.layout):
            flavor = "numpy array" if isinstance(reset.layout, ArrayLayout) else "list"
            return reset, f"state attribute '{name}' persists a {flavor}; store the same container flavor"
        geometry_matches = (
            stored.layout.shape == reset.layout.shape
            if isinstance(reset.layout, ArrayLayout) and isinstance(stored.layout, ArrayLayout)
            else stored.layout == reset.layout
        )
        if not geometry_matches:
            described = (
                f"a {'x'.join(map(str, reset.layout.shape))} array"
                if isinstance(reset.layout, ArrayLayout)
                else f"a {outer_arity(reset.layout)}-element vector"
            )
            return reset, f"state attribute '{name}' persists {described}; the stored value has an incompatible shape"
        cells: list[AtomicFact] = []
        message: str | None = None
        kind_mismatch = False
        widening_only = True
        for ordinal, cell in enumerate(stored.leaves):
            if isinstance(cell, Reference):
                return reset, f"state attribute '{name}' cannot persist an object reference"
            slot_kind = _scalar_sem(reset.leaves[ordinal])
            assert slot_kind is not None, "the reset schema admits datapath cells only"
            stored_kind = _scalar_sem(cell)
            if stored_kind is slot_kind:
                cells.append(cell)
            elif slot_kind is SemType.FLOAT and stored_kind is SemType.INT:
                converted, cell_message = _binary64_store_image(cell, f"state attribute '{name}' cell {ordinal}")
                assert isinstance(converted, (Known, Residual))
                cells.append(converted)
                if cell_message is not None and message is None:
                    message = cell_message
            else:
                kind_mismatch = True
                if message is None:
                    message = f"state attribute '{name}' stores an incompatible type at cell {ordinal}"
                if not (slot_kind is SemType.INT and stored_kind is SemType.FLOAT):
                    widening_only = False
        if kind_mismatch:
            assert message is not None
            return (stored if widening_only else reset), message
        return AggregateFact(reset.layout, tuple(cells)), message
    slot_kind = _scalar_sem(reset)
    assert slot_kind is not None, "a scalar reset fact is a Known bool or a numeric residual"
    if isinstance(stored, AggregateFact):
        return reset, f"state attribute '{name}' persists a scalar; an aggregate cannot be stored into it"
    stored_kind = _scalar_sem(stored)
    if stored_kind is None or stored_kind is slot_kind:
        return stored, None  # a non-datapath value neither establishes nor violates; the W/D join owns it
    assert isinstance(stored, (Known, Residual))
    if slot_kind is SemType.FLOAT and stored_kind is SemType.INT:
        return _binary64_store_image(stored, f"state attribute '{name}'")
    message = f"state attribute '{name}' stores an incompatible type"
    return (stored if slot_kind is SemType.INT and stored_kind is SemType.FLOAT else reset), message


def enforce_storage_schemas(
    unit: FunctionUnit,
    executable_blocks: set[BlockId],
    executable_edges: set[tuple[BlockId, BlockId]],
    store_facts: Mapping[int, Fact],
    schemas_in: Mapping[BlockId, Mapping["Place", StorageSchema]],
    store_verdicts: Mapping[int, StoreVerdict],
    pending_bridge: Mapping[OriginStack, str],
) -> None:
    """
    The one storage-schema resolution site, over the stabilized executable graph: every block's SOURCE stores
    replay against its stable entry schemas -- the environments the analysis flowed beside the facts
    (establishment, merge joins, the store-edge conversion, and compiler scope resets included) -- with each
    store's stable stored fact re-deriving its verdict, and all violations, local rebindings, store-edge
    conversion failures, and the recorded state-store verdicts alike, report as one located rejection at the
    first violating store in CFG preorder (then-arm first, matching the Fail walk). The pending bridge is never
    a verdict source for a store still in the graph: a store that executed only with an Unbound value -- its
    producer cut by the deferral cascade -- is STALE, its entry expires, and the deferral that cut it surfaces
    as the true stable rejection. An origin with no store left in any executable block is STRANDED -- its own
    violation's cascade removed the block, so nothing downstream can testify -- and its entry reports, ranked
    after every in-graph violation and before any deferred rejection, in source order among stranded siblings
    with the message only a tie-break. Runs strictly after the round stabilizes: SCCP discovers executable
    predecessors late, so a mid-flight verdict would be order-dependent. A compiler scope reset (an unchecked
    UnbindPlace: a comprehension entry, an unroll trip's target prelude) clears the binding for a fresh
    per-execution schema, while a user ``del`` does not: variables stay strongly typed across ``del``. This is the
    VERDICT alone; the conformed facts the store edge produces are re-derived where the route plan needs them.
    """
    violations: list[tuple[tuple[int, int], str, OriginStack]] = []
    surviving_origins: set[OriginStack] = set()
    order = executable_preorder(unit, executable_blocks, executable_edges)

    def standing_obligation(op: Op) -> str | None:
        verdict = store_verdicts.get(id(op))
        return verdict.message if verdict is not None else None

    for position, block_id in enumerate(order):
        assert block_id in schemas_in, "every executable-reachable block carries a flowed environment"
        env: dict[Place, StorageSchema] = dict(schemas_in[block_id])
        for index, op in enumerate(unit.blocks[block_id].ops):
            match op:
                case PyStoreAttr():
                    surviving_origins.add(op.origin)
                    message = standing_obligation(op)
                    if message is not None:
                        violations.append(((position, index), message, op.origin))
                case UnbindPlace(place=place, checked=False):
                    env.pop(place, None)
                case StorePlace(place=place, role=StoreRole.SOURCE):
                    surviving_origins.add(op.origin)
                    fact = store_facts.get(id(op))
                    assert fact is not None, "every executable SOURCE store was transferred this round"
                    assert isinstance(place, Local), "a SOURCE store binds a named local"
                    if isinstance(fact, (Unbound, MaybeUnbound)):
                        message = standing_obligation(op)
                        if message is not None:
                            violations.append(((position, index), message, op.origin))
                        continue
                    schema, _, message = conform_local_store(env.get(place), place.binding.name, fact)
                    if schema is not None:
                        env[place] = schema
                    if message is not None:
                        violations.append(((position, index), message, op.origin))
    if violations:
        _, message, origin = min(violations, key=lambda item: item[0])
        raise AnalysisRejection(message, origin)
    stranded = [(message, origin) for origin, message in pending_bridge.items() if origin not in surviving_origins]
    if stranded:
        message, origin = min(stranded, key=lambda entry: (source_position(entry[1]), entry[0]))
        raise AnalysisRejection(message, origin)


def _mro_attribute_of(klass: type, name: str) -> object | None:
    return next((c.__dict__[name] for c in klass.__mro__ if name in c.__dict__), None)


def _has_truth_override(klass: type) -> bool:
    """
    Whether a user-defined __bool__/__len__ ENTRY governs the class's truth (object's own default excluded).
    Membership, not value: ``__bool__ = None`` is a real override (Python raises TypeError on its truth).
    """
    return any(name in c.__dict__ for name in ("__bool__", "__len__") for c in klass.__mro__ if c is not object)


def _reject_attribute_hooks(klass: type, name: "str | None", origin: OriginStack) -> None:
    """
    The one component-attribute refusal: attribute access on a component must be plain instance state, since a
    hook or accessor would run user code the abstract state model cannot mirror. Refused are a class-wide
    custom ``__setattr__``/``__getattr__``/``__getattribute__`` and, when ``name`` is given, a data descriptor
    (``__set__``/``__delete__``) backing that attribute -- a property with or without a setter, a property
    subclass, or any other data descriptor; a slots member descriptor IS the field, and an exact-property
    GETTER read is desugared by the caller before its name is checked here. Raw MRO lookups throughout, never
    getattr, which would run the very accessor being refused.
    """
    plain_setattr = (_mro_attribute_of(object, "__setattr__"), _mro_attribute_of(type, "__setattr__"))
    plain_getattribute = (
        _mro_attribute_of(object, "__getattribute__"),
        _mro_attribute_of(type, "__getattribute__"),
    )
    hooked = (
        _mro_attribute_of(klass, "__setattr__") not in plain_setattr
        or _mro_attribute_of(klass, "__getattribute__") not in plain_getattribute
        or _mro_attribute_of(klass, "__getattr__") is not None
    )
    descriptor = _mro_attribute_of(klass, name) if name is not None else None
    intercepted = (
        descriptor is not None
        and (hasattr(type(descriptor), "__set__") or hasattr(type(descriptor), "__delete__"))
        # A slot IS its own field, but only under its own name: an ALIAS to another slot's member descriptor
        # (``alias = Base.value``) intercepts a different storage location and would miscompile as a fresh slot.
        and not (
            isinstance(descriptor, types.MemberDescriptorType)
            and descriptor.__name__ == name
            and getattr(descriptor, "__objclass__", None) in klass.__mro__
        )
    )
    if hooked or intercepted:
        raise AnalysisRejection(
            "component attributes must be plain values: custom __setattr__/__getattr__/__getattribute__ hooks "
            "and descriptors are not supported",
            origin,
        )
