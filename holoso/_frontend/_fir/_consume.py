"""
Use-specific consumption operations: what a value may be USED AS, and the doctrine each use enforces.

Every rule here is stated ONCE, at the operation for the use it governs, because the same value is legal in one
consumption and refused in another and a single global predicate would therefore be wrong. A boolean is not an
arithmetic operand and not an array index, yet ``& | ^``, comparison and the explicit casts all admit one; a
record is not a subscript key, not a sliced source and has no attributes, yet field access and integer
projection are exactly how a record IS consumed. Splitting the doctrines by use rather than by value keeps
those asymmetries visible instead of forcing them into exceptions to a rule.

The operations are pure functions over facts. They reach no analyzer state, which is why they live here rather
than on the analyzer: nothing in a consumption rule depends on where the fixpoint has got to.
"""

import enum
from collections.abc import Callable, Iterable
from typing import NoReturn

import numpy as np

from ._analysis_support import (
    AnalysisRejection,
    _fits_float64,
    _has_truth_override,
    _layout_dtypes,
    _rectangular_shape,
    _residual_type,
)
from ._fact import (
    AggregateFact,
    ArrayDType,
    ArrayLayout,
    AtomicFact,
    ContainerFlavor,
    Fact,
    Known,
    RecordLayout,
    Reference,
    Residual,
    StructuralLayout,
    ValueLayout,
    leaf_count,
    materialize_static,
    numpy_kinded,
    outer_arity,
)
from ._fold import contains_record
from ._ir import OriginStack
from ._opsem import BinOp, UnOp, static_binop, static_truth, static_unop
from ._value import (
    MetaInt,
    NpBool,
    NpInt,
    SemType,
    StaticBool,
    StaticRecord,
    StaticSeq,
    StaticValue,
    admit,
)

_ELEMENTWISE_OPS = frozenset({BinOp.ADD, BinOp.SUB, BinOp.MUL, BinOp.DIV})

_DTYPE_KIND = {ArrayDType.BOOL: SemType.BOOL, ArrayDType.INT: SemType.INT, ArrayDType.FLOAT: SemType.FLOAT}


class ArrayResultUse(enum.Enum):
    """
    The array-producing consumptions, which share the runtime-integer doctrine but not its wording. The
    difference between the two refusals is preserved rather than unified, the frozen rejection corpus pinning
    the rendered text; carrying the noun as data keeps two literals from drifting apart again.
    """

    ARITHMETIC = "arithmetic"
    CONSTRUCTION = "construction"


# ------------------------------------ crossing into the fact domain ------------------------------------


def reject_zero_dimensional(origin: OriginStack) -> NoReturn:
    """
    A 0-d array is an accident rather than an idiom (scope ruling T3), so the fact domain does not carry one --
    ``ArrayLayout`` asserts the empty shape away and ``admit`` refuses the object. Every door where one could
    enter therefore refuses HERE, with guidance, instead of letting it survive as an opaque reference whose
    later diagnosis names something else. The doors do not agree on their predicate; see ``crossing_fact``.
    """
    raise AnalysisRejection("a 0-dimensional array is not supported; use the scalar directly", origin)


def crossing_fact(value: object, origin: OriginStack) -> StaticValue | None:
    """
    The door where a raw Python object becomes a fact: the admitted static value, or None for an object that
    stays a reference.

    The predicate is ``isinstance``, so a 0-d ndarray SUBCLASS refuses here as 0-dimensional. The builder's
    global-load door and the attribute-snapshot door instead test the EXACT type, matching ``admit``'s own
    gate, and let a subclass through as a reference to be diagnosed as a subclass where it reaches arithmetic.
    The divergence is measured and recorded rather than smoothed over, since either predicate would change
    behaviour at two of the four doors.
    """
    if isinstance(value, np.ndarray) and value.ndim == 0:
        reject_zero_dimensional(origin)
    return admit(value)


def reshape_dimensions(facts: list[Fact], origin: OriginStack) -> tuple[int, ...]:
    """
    The target shape of a relayout, as static dimensions: bare integers or one static tuple. Inference (-1) is
    refused because the domain never guesses a shape, and the empty shape is the 0-d door in shape form.
    """
    if not facts:
        raise AnalysisRejection("reshape() requires a shape argument", origin)
    if len(facts) == 1 and isinstance(facts[0], AggregateFact):
        materialized = materialize_static(facts[0])
        if materialized is not None:
            facts = [Known(materialized)]
    if len(facts) == 1 and isinstance(facts[0], Known) and isinstance(facts[0].value, StaticSeq):
        items: list[StaticValue] = list(facts[0].value.items)
    else:
        checked: list[StaticValue] = []
        for fact in facts:
            if not isinstance(fact, Known):
                raise AnalysisRejection("reshape() requires a static shape", origin)
            checked.append(fact.value)
        items = checked
    shape: list[int] = []
    for item in items:
        if not isinstance(item, (MetaInt, NpInt)):
            raise AnalysisRejection("reshape() requires integer dimensions", origin)
        dimension = int(item.value)
        if dimension < 0:
            raise AnalysisRejection(
                "a -1 (inferred) reshape dimension is not supported; spell the shape explicitly", origin
            )
        shape.append(dimension)
    if not shape:
        reject_zero_dimensional(origin)
    return tuple(shape)


# ------------------------------------ numeric operand consumptions ------------------------------------


def numeric_operand(fact: Fact, origin: OriginStack) -> SemType:
    """The kind of a value consumed as a runtime numeric operand. A bool is admitted here; arithmetic refuses it."""
    match fact:
        case Known(value=value):
            sem = _residual_type(value)
            if sem is None:
                raise AnalysisRejection("a non-numeric value reaches a runtime operation", origin)
            return sem
        case Residual(type=sem):
            return sem
        case Reference(obj=referent) if isinstance(referent, np.ndarray):
            # An unadmitted ndarray reaches arithmetic as a live reference; name the actual problem
            # instead of "non-numeric" -- a SUBCLASS (np.matrix redefines * as the matrix product) or a
            # dtype outside the embeddable boolean/integer/float categories (a timedelta64, a huge uint64).
            if type(referent) is not np.ndarray:
                raise AnalysisRejection(
                    "an ndarray subclass does not participate in arithmetic; use a plain numpy array", origin
                )
            raise AnalysisRejection(
                f"an array of dtype {referent.dtype} is not admitted " "(only boolean/integer/float dtypes embed)",
                origin,
            )
        case Reference():
            raise AnalysisRejection("a non-numeric value reaches a runtime operation", origin)
        case _:
            raise AnalysisRejection("a runtime operation reads an aggregate or unbound value", origin)


def arithmetic_operands(kinds: Iterable[SemType | None], origin: OriginStack) -> None:
    """
    Consume values as ARITHMETIC operands. Python's bool-as-int promotion is not modelled in the datapath, so a
    boolean crosses into arithmetic only through an explicit cast -- which is what ``float()``/``int()`` are
    for. The kinds arrive together rather than one at a time so a non-numeric operand keeps its own diagnosis:
    deriving every kind before judging any is what lets ``a_bool * an_object`` still report the object.
    """
    if SemType.BOOL in kinds:
        raise AnalysisRejection("arithmetic on a bool requires an explicit conversion", origin)


def elementwise_side_kind(side: Fact, origin: OriginStack) -> SemType:
    """The kind an elementwise operand contributes: an array's dtype, otherwise the scalar's own numeric kind."""
    if isinstance(side, AggregateFact):
        assert isinstance(side.layout, ArrayLayout)
        return _DTYPE_KIND[side.layout.dtype]
    return numeric_operand(side, origin)


def unary_residual(fact: Fact, origin: OriginStack) -> Fact:
    """A value consumed by a unary +/- that did not fold. Python's unary on a bool yields an int."""
    match fact:
        case Known(value=value):
            sem = _residual_type(value)
            if sem is None:
                raise AnalysisRejection("a non-numeric value reaches a runtime operation", origin)
            return Residual(SemType.INT if sem is SemType.BOOL else sem)
        case Residual(type=SemType.BOOL):
            return Residual(SemType.INT)
        case Residual():
            return fact  # a unary negation/plus preserves the operand's numeric kind
        case _:
            raise AnalysisRejection("a runtime operation reads an aggregate or unbound value", origin)


def truth_value(fact: Fact, origin: OriginStack) -> Fact:
    """A value consumed as a condition. An array has no unambiguous truth; a record's override cannot be trusted."""
    match fact:
        case Known(value=value):
            verdict = static_truth(value)
            if verdict is None and _residual_type(value) is None:
                raise AnalysisRejection("the truth value of this object is not defined here", origin)
            return Known(StaticBool(verdict)) if verdict is not None else Residual(SemType.BOOL)
        case Reference():
            raise AnalysisRejection("the truth value of this object is not defined here", origin)
        case AggregateFact() as aggregate:
            layout = aggregate.layout
            if isinstance(layout, ArrayLayout) or (
                isinstance(layout, StructuralLayout) and ContainerFlavor.ARRAY in layout.flavors
            ):
                raise AnalysisRejection(
                    "the truth value of an array is ambiguous; use .any() or .all() in plain numpy", origin
                )
            if isinstance(layout, RecordLayout):
                # A class-dictionary __bool__/__len__ entry (even ``__bool__ = None``) rejects outright: the
                # override would run on a value-faithful but not type-faithful reconstruction (an enum field
                # rebuilds as its base value), so folding it can silently diverge from Python.
                if _has_truth_override(layout.klass):
                    raise AnalysisRejection(
                        "the truth of a record with a custom __bool__/__len__ is not supported", origin
                    )
                return Known(StaticBool(True))  # default object truth, regardless of field count
            return Known(StaticBool(outer_arity(layout) != 0))
        case _:
            return Residual(SemType.BOOL)


# ------------------------------------ subscript and attribute consumptions ------------------------------------


def array_index_element(item: StaticValue, origin: OriginStack) -> None:
    """
    A boolean is not an array index. numpy boolean indexing selects by MASK (and prepends an axis for a scalar
    bool) while Python's bool-as-int indexing applies only to tuples and lists, so guessing would miscompile.
    """
    if isinstance(item, (StaticBool, NpBool)):
        raise AnalysisRejection("a boolean index into an array is not supported; use an integer", origin)


def subscript_index(index: Fact, origin: OriginStack) -> None:
    """
    A record is not a subscript key, for ANY subscriptable (a range or string included): the key would resolve
    through a user ``__index__`` running on the reconstruction -- value-faithful but not type-faithful, an enum
    field rebuilding as its base value -- whose semantics the compiler cannot vouch for. A record nested
    anywhere inside a tuple key is the same hazard on a rebuild, which is why the whole layout is consulted.
    """
    if (isinstance(index, AggregateFact) and contains_record(index.layout)) or (
        isinstance(index, Known) and isinstance(index.value, StaticRecord)
    ):
        raise AnalysisRejection("a record subscript index is not supported", origin)


def spanning_subscript_source(layout: ValueLayout, origin: OriginStack) -> None:
    """
    A record-carrying sequence is neither sliced nor multi-axis indexed. Both consumptions leave the leaf
    algebra: a structural flavor cannot truthfully pick a result container, and the concrete fallback rebuilds
    real instances, so a ``__del__`` would fire at compile time. Integer projection and field access are how a
    record IS consumed, and both stay open.
    """
    if contains_record(layout):
        # ``from None`` because one caller guards the TypeError arm of a failed ``operator.index``: the refusal
        # is about the record, and chaining the probe's TypeError behind it would misdirect the reader.
        raise AnalysisRejection(
            "slicing or multi-axis indexing of a record-carrying sequence is not supported", origin
        ) from None


def attribute_receiver(layout: ValueLayout, name: str, origin: OriginStack) -> None:
    """A record-carrying sequence has no attributes: a bound method would run Python's protocols over rebuilds."""
    if contains_record(layout):
        raise AnalysisRejection(f"attribute '{name}' of a record-carrying sequence is not supported", origin)


# ------------------------------------ array-producing consumptions ------------------------------------


def reject_runtime_integer_array(use: ArrayResultUse, origin: OriginStack) -> NoReturn:
    """
    The scalar integer datapath SATURATES (contained at MIR) where numpy int64 wraps, so an array result
    carrying runtime integer leaves would diverge from numpy leafwise the moment integers lower. Refused until
    the integer sprint, at whichever array consumption produced it.
    """
    raise AnalysisRejection(f"runtime integer array {use.value} is not lowerable yet; cast to float first", origin)


def array_result(
    shape: tuple[int, ...], dtype: ArrayDType, count: int, leaf: Callable[[int], AtomicFact]
) -> AggregateFact:
    """
    The one constructor for a leafwise array result: map every ordinal through the consumption's own rule and
    pin the layout invariant ``ArrayLayout`` cannot check for itself -- a leaf's kind IS the array's dtype.
    Three consumptions build arrays leafwise (binary arithmetic, unary arithmetic, the ``np.array`` factory)
    and each carried its own copy of the walk, only one of them checking the invariant.
    """
    kind = _DTYPE_KIND[dtype]
    leaves: list[AtomicFact] = []
    for ordinal in range(count):
        produced = leaf(ordinal)
        assert (produced == Residual(kind)) or (
            isinstance(produced, Known) and _residual_type(produced.value) is kind
        ), "an elementwise leaf diverged from the result dtype"
        leaves.append(produced)
    return AggregateFact(ArrayLayout(shape, dtype), tuple(leaves))


def fold_binary(
    fold: Callable[[StaticValue, StaticValue], StaticValue | None],
    lhs: Fact,
    rhs: Fact,
    origin: OriginStack,
    default: SemType | None = None,
    promotes_to_float: bool = False,
) -> Fact:
    """A scalar binary consumption: fold when both sides are Known, else the residual kind the operands imply."""
    if isinstance(lhs, Known) and isinstance(rhs, Known):
        folded = fold(lhs.value, rhs.value)
        if folded is not None:
            return Known(folded)
    operand_types = [numeric_operand(fact, origin) for fact in (lhs, rhs)]
    if default is None:  # a declared result kind means comparison, which admits booleans; arithmetic does not
        arithmetic_operands(operand_types, origin)
        if promotes_to_float or SemType.FLOAT in operand_types:
            return Residual(SemType.FLOAT)
        return Residual(SemType.INT)
    return Residual(default)


def elementwise_binary(bin_op: BinOp, lhs: Fact, rhs: Fact, origin: OriginStack) -> Fact:
    """
    Elementwise ``+ - * /`` with at least one array operand: the other side is a same-shape array (leaves pair
    in canonical order) or a numeric scalar (broadcast). Each leaf pair takes the SCALAR fold rule, so a fully
    static pair folds through ``static_binop`` on the domain's own numpy-kinded leaf values -- element-wise
    numpy semantics (promotion, wraparound, errstate deferrals) hold per leaf by construction, and general
    broadcasting stays a located rejection rather than a silent alignment. Divergence guards mirror what numpy
    applies ARRAY-WIDE before touching any element (an empty array included): an out-of-range Python-int
    constant is a located rejection where numpy raises OverflowError.
    """
    if bin_op is BinOp.MATMUL:
        raise AnalysisRejection("the matrix product is not lowerable yet", origin)
    if bin_op not in _ELEMENTWISE_OPS:
        raise AnalysisRejection(f"operator {bin_op.value!r} is not supported on arrays", origin)
    for side in (lhs, rhs):
        if isinstance(side, AggregateFact) and not isinstance(side.layout, ArrayLayout):
            raise AnalysisRejection(
                "elementwise arithmetic mixes an array with a non-array container; wrap it in np.array(...)",
                origin,
            )
    promotes = bin_op is BinOp.DIV
    lhs_sem = elementwise_side_kind(lhs, origin)
    rhs_sem = elementwise_side_kind(rhs, origin)
    arithmetic_operands((lhs_sem, rhs_sem), origin)
    arrays = [side for side in (lhs, rhs) if isinstance(side, AggregateFact)]
    shapes = [side.layout.shape for side in arrays if isinstance(side.layout, ArrayLayout)]
    if len(set(shapes)) > 1:
        raise AnalysisRejection(
            f"elementwise arithmetic on mismatched shapes {shapes[0]} and {shapes[1]} " "(only a scalar broadcasts)",
            origin,
        )
    result_sem = SemType.FLOAT if promotes or SemType.FLOAT in (lhs_sem, rhs_sem) else SemType.INT
    for side in (lhs, rhs):
        if isinstance(side, Known) and isinstance(side.value, MetaInt):
            constant = side.value.value
            if result_sem is SemType.INT and not -(2**63) <= constant < 2**63:
                raise AnalysisRejection(
                    "an integer constant beyond the signed 64-bit range does not combine with an integer "
                    "array (numpy raises OverflowError)",
                    origin,
                )
            if result_sem is SemType.FLOAT and not _fits_float64(constant):
                raise AnalysisRejection(
                    "an integer constant too large for float64 does not combine with an array "
                    "(numpy raises OverflowError)",
                    origin,
                )
    if result_sem is SemType.INT and (
        any(isinstance(side, Residual) for side in (lhs, rhs))
        or any(isinstance(leaf, Residual) for array in arrays for leaf in array.leaves)
    ):
        reject_runtime_integer_array(ArrayResultUse.ARITHMETIC, origin)

    def pair(ordinal: int) -> AtomicFact:
        left = lhs.leaves[ordinal] if isinstance(lhs, AggregateFact) else lhs
        right = rhs.leaves[ordinal] if isinstance(rhs, AggregateFact) else rhs
        combined = fold_binary(lambda a, b: static_binop(bin_op, a, b), left, right, origin, promotes_to_float=promotes)
        assert isinstance(combined, (Known, Residual)), combined
        return combined

    dtype = ArrayDType.FLOAT if result_sem is SemType.FLOAT else ArrayDType.INT
    return array_result(shapes[0], dtype, leaf_count(arrays[0].layout), pair)


def elementwise_unary(un_op: UnOp, operand: AggregateFact, origin: OriginStack) -> Fact:
    """Elementwise unary +/-: numpy itself refuses these on a boolean array (a TypeError pointing at ~)."""
    assert isinstance(operand.layout, ArrayLayout)
    arithmetic_operands((_DTYPE_KIND[operand.layout.dtype],), origin)
    if operand.layout.dtype is ArrayDType.INT and any(isinstance(leaf, Residual) for leaf in operand.leaves):
        reject_runtime_integer_array(ArrayResultUse.ARITHMETIC, origin)

    def mapped(ordinal: int) -> AtomicFact:
        leaf = operand.leaves[ordinal]
        folded = static_unop(un_op, leaf.value) if isinstance(leaf, Known) else None
        if folded is not None:
            return Known(folded)
        residual = unary_residual(leaf, origin)
        assert isinstance(residual, Residual), residual
        return residual

    return array_result(operand.layout.shape, operand.layout.dtype, len(operand.leaves), mapped)


def array_factory(source: AggregateFact, origin: OriginStack, force_float: bool = False) -> AggregateFact:
    """
    np.array/asarray/asanyarray over a residual-carrying aggregate: a relayout of the SAME leaves onto the
    rectangular shape the nesting yields, with dtype discovery restricted to the proven subset -- any float
    evidence promotes to FLOAT64 (integer leaves coerce: Knowns re-kind to np.float64 exactly as numpy
    extraction would yield them, residual integers pick up a runtime conversion at emission), an all-boolean
    argument builds a BOOL array, and empty array children contribute their dtype as evidence. Outside the
    subset numpy behaves in ways the domain cannot carry, so the forms reject where numpy would surprise: a
    Python-int leaf beyond signed 64 bits (numpy builds an object array, or silently promotes the uint64
    range to float64), a bool/numeric mix (numpy widens the bool), and a runtime-integer result.
    """
    shape = _rectangular_shape(source.layout)
    if shape is None:
        raise AnalysisRejection("an array literal must be rectangular (numpy raises on ragged nesting)", origin)
    sems: set[SemType | None] = set()
    for leaf in source.leaves:
        assert not isinstance(leaf, Reference), "a reference leaf survived the admission walk"
        sems.add(leaf.type if isinstance(leaf, Residual) else _residual_type(leaf.value))
    sems |= {_DTYPE_KIND[dtype] for dtype in _layout_dtypes(source.layout)}
    if None in sems:  # a string or range leaf: numpy would build a string/object array, outside the domain
        raise AnalysisRejection("an array literal requires numeric or boolean elements", origin)
    assert sems, "an empty argument cannot carry a residual leaf"
    if force_float:
        sems = {SemType.FLOAT}  # the explicit dtype=float casts every leaf; no discovery, no mix question
    if SemType.BOOL in sems and sems != {SemType.BOOL}:
        raise AnalysisRejection(
            "an array literal mixes booleans with numbers (numpy would widen the bool); convert explicitly",
            origin,
        )
    for leaf in source.leaves:
        if isinstance(leaf, Known) and isinstance(leaf.value, MetaInt):
            if not -(2**63) <= leaf.value.value < 2**63:
                raise AnalysisRejection(
                    "an integer beyond the signed 64-bit range in an array literal is not supported "
                    "(numpy would build an object array or promote through uint64)",
                    origin,
                )
    if sems == {SemType.BOOL}:
        dtype = ArrayDType.BOOL
    elif SemType.FLOAT in sems:
        dtype = ArrayDType.FLOAT
    else:
        dtype = ArrayDType.INT
    if dtype is ArrayDType.INT:  # only residual integers reach here: a fully static argument took the concrete fold
        reject_runtime_integer_array(ArrayResultUse.CONSTRUCTION, origin)

    def relayout(ordinal: int) -> AtomicFact:
        leaf = source.leaves[ordinal]
        if isinstance(leaf, Known):
            return numpy_kinded(leaf, dtype)
        assert isinstance(leaf, Residual)
        return Residual(SemType.FLOAT) if dtype is ArrayDType.FLOAT else leaf

    return array_result(shape, dtype, len(source.leaves), relayout)


def fold_bitwise(bin_op: BinOp, lhs: Fact, rhs: Fact, origin: OriginStack) -> Fact:
    """
    Bit-true operators. ``&``/``|``/``^`` on two booleans is a boolean (logical) result -- the one arithmetic-shaped
    consumption that admits a bool -- and every other admitted form is two integers. A float operand, a boolean
    shift, and mixed bool/int all refuse, Python's bool-as-int promotion not being modelled in the datapath. A
    compile-time-known negative shift count refuses (Python raises); a runtime count is the hardware's documented
    reverse-shift deviation. A fully-static form folds Python-exact via ``static_binop``. Operand kinds are
    validated before any diagnostic.
    """
    is_shift = bin_op in (BinOp.LSHIFT, BinOp.RSHIFT)
    ltype, rtype = numeric_operand(lhs, origin), numeric_operand(rhs, origin)
    if SemType.FLOAT in (ltype, rtype):
        raise AnalysisRejection(f"bitwise/shift operator {bin_op.value} requires integer operands", origin)
    if is_shift and isinstance(rhs, Known) and isinstance(rhs.value, (MetaInt, NpInt)) and int(rhs.value.value) < 0:
        raise AnalysisRejection(f"a negative shift count ({int(rhs.value.value)}) is rejected at compile time", origin)
    if ltype is SemType.BOOL and rtype is SemType.BOOL and not is_shift:
        result_type = SemType.BOOL  # & | ^ on two booleans stays in the boolean bank
    elif ltype is SemType.INT and rtype is SemType.INT:
        result_type = SemType.INT
    else:
        raise AnalysisRejection(
            f"bitwise/shift operator {bin_op.value} requires two integers (or two booleans for & | ^)", origin
        )
    if isinstance(lhs, Known) and isinstance(rhs, Known):
        folded = static_binop(bin_op, lhs.value, rhs.value)
        if folded is not None:
            return Known(folded)
        if isinstance(lhs.value, (StaticBool, NpBool)) and isinstance(rhs.value, (StaticBool, NpBool)):
            # static_binop covers only numerics; combine two Known booleans here so ``True & False`` folds to a
            # Known bool and drives edge selection (a dead branch guarded by it is never analyzed). numpy wins
            # the result provenance exactly as np.bool_ & bool yields np.bool_.
            a, b = lhs.value.value, rhs.value.value
            combined = bool(a and b if bin_op is BinOp.BITAND else a or b if bin_op is BinOp.BITOR else a != b)
            numpy_side = isinstance(lhs.value, NpBool) or isinstance(rhs.value, NpBool)
            return Known(NpBool(combined) if numpy_side else StaticBool(combined))
    return Residual(result_type)
