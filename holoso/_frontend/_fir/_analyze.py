"""
The FIR analyzer: optimistic executable-edge abstract interpretation (SCCP-style) with flow-sensitive per-edge
environments over Places. Facts form the lattice Unbound < Known(StaticValue) < Residual(SemType); joins are
per-Place, strong updates on stores, and only executable in-edges contribute. An int/float join promotes the
integer side to float, C-style, its rounding accepted under the fastmath charter (Python instead keeps each
path's runtime kind -- the documented deviation). Static folding is Python-exact and runs on the closed value
domain (the width rule: runtime-typed numeric values never fold; a Known Bool always drives edge selection).
StaticFor headers unroll by cloning the body per trip once the iterable is Known; PyCall sites
expand on demand by grafting the callee's freshly instantiated template into the working graph (recursion is
a located rejection keyed by function and receiver identity). The result is a stable residual graph plus final
facts, validated to contain no unresolved calls, loop templates, or possibly-unbound reads on executable paths.

Everything here operates on a WORKING COPY of builder templates; templates are never mutated, so the state fixed
point can rebuild from scratch each outer round.
"""

import logging
import math
import types
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import MISSING, dataclass, field, is_dataclass, replace
from functools import partial
from typing import NoReturn

import numpy as np

from ..._errors import UnsupportedConstruct, UnsupportedLibraryFunction
from .._ast_support import UNROLL_THRESHOLD
from ._build import BuildRejection, build_unit
from ._ir import (
    BindingId,
    Block,
    BlockId,
    Branch,
    BuildList,
    BuildTuple,
    Fail,
    FunctionUnit,
    Jump,
    LoadConst,
    LoadRef,
    LoadPlace,
    Local,
    LocatedRejection,
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
    ReturnPlace,
    SelectMode,
    StateLeaf,
    StaticFor,
    StorePlace,
    StoreRole,
    Terminator,
    UnbindPlace,
    UnitExit,
    executable_preorder,
    executable_rpo,
    op_dst,
    origin_order,
    source_position,
)
from ._fact import (
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
    AggregateLayout,
    aggregate_of,
    child_slice,
    join_layouts,
    leaf_count,
    materialize_static,
    normalize_static,
    numpy_kinded,
    outer_arity,
    record_of,
)
from ._signature import (
    ArrayParameter,
    ContractError,
    RecordParameter,
    ScalarParameter,
    array_shape,
    is_array_annotation,
)
from ._fold import (
    FieldSchema,
    FoldRefusal,
    admit_call,
    construction_schema,
    contains_record,
    is_unimplemented_library,
    range_size,
)
from ..._util import RelationalOp
from ._analysis_support import (
    AnalysisRejection,
    DeferredRejection,
    LibraryAnalysisRejection,
    StorageSchema,
    StoreVerdict,
    _concat_seqs,
    _concrete_fact,
    _contract_structure,
    _coreachable,
    _crossing_object,
    _fits_float64,
    _has_truth_override,
    _identity_place,
    _is_array_fact,
    _is_list_fact,
    _join_atoms,
    _layout_dtypes,
    _mro_attribute_of,
    _numeric_sem,
    _scalar_sem,
    _rectangular_shape,
    _reject_attribute_hooks,
    _remap_op,
    _remap_terminator,
    _residual_type,
    _same_fact,
    _seq_side,
    _transpose_routes,
    conform_local_store,
    conform_state_store,
    enforce_storage_schemas,
    join_facts,
    join_schemas,
    render_interpolation,
    schema_of_fact,
)
from ._plan import (
    CallLowering,
    CallPlan,
    FieldBindings,
    PlanSite,
    RouteEvidence,
    RoutePlan,
    SourceSelection,
    produce_route_plans,
)
from ._opsem import BinOp, UnOp, static_binop, static_compare, static_truth, static_unop
from ..._hir import BoolType, FloatIsFinite, FloatIsInf, FloatIsNegInf, FloatIsPosInf
from ._value import (
    MetaInt,
    NpBool,
    StaticRecord,
    NpFloat,
    NpInt,
    SemType,
    StaticBool,
    StaticFloat,
    StaticRange,
    StaticSeq,
    StaticSlice,
    StaticStr,
    StaticValue,
    admit,
    as_python,
    same,
)

_logger = logging.getLogger(__name__)

_MAX_BLOCKS = 200_000
_MAX_VISITS = 1_000_000

_BITWISE_OPS = frozenset({BinOp.LSHIFT, BinOp.RSHIFT, BinOp.BITAND, BinOp.BITOR, BinOp.BITXOR})
_ELEMENTWISE_OPS = frozenset({BinOp.ADD, BinOp.SUB, BinOp.MUL, BinOp.DIV})


@dataclass(frozen=True, slots=True)
class _PropertyRead:
    """A component attribute read that resolved to a ``@property`` getter, to be desugared into a bound call."""

    getter: object  # a ``MethodType(fget, component)`` bound to the exact receiver


@dataclass(frozen=True, slots=True)
class _DefaultArgument:
    """A defaulted parameter's value, admitted while binding validates so the graft mutates only a proven call."""

    value: object
    admitted: StaticValue | None


@dataclass(frozen=True, slots=True, eq=False)
class _ArrayMethod:
    """
    A compiler-minted bound-method token for an ARRAY-valued receiver: the honest fact for ``v.flatten`` is a
    bound method object, which the Reference sort carries by identity (never data, call-only -- returning or
    merging it keeps the established reference rejections). Minted once per (receiver binding, name) so SCCP
    fact equality is stable across rounds; the call site rewrites onto the canonical explicit-receiver form.
    """

    receiver: BindingId
    name: str


_UNBOUND = Unbound()


_ARRAY_ATTRIBUTES = ("real", "imag")  # value navigation, all-Known only; .T/shape metadata are structural


def _list_attribute_rejection(name: str, origin: OriginStack) -> AnalysisRejection:
    """The mutator guidance fits only names Python actually gives a list; array-ish spellings point at np.array."""
    if hasattr([], name):
        return AnalysisRejection(
            f"list method '{name}' is not supported (lists are immutable values here); rebind with + instead", origin
        )
    return AnalysisRejection(f"a list has no attribute '{name}'; convert with np.array to use array attributes", origin)


@dataclass(slots=True)
class _Env:
    """
    One abstract environment: Place -> Fact, absent meaning unbound-never-touched, beside Place -> established
    storage schema. The schema lattice rides the same worklist as the facts so a SOURCE store sees the schema
    exactly where it stores (the store-edge int->float conversion needs it); the schema VERDICT still resolves
    only after stabilization, over these flowed environments.
    """

    facts: dict[Place, Fact] = field(default_factory=dict)
    schemas: dict[Place, StorageSchema] = field(default_factory=dict)

    def copy(self) -> "_Env":
        return _Env(dict(self.facts), dict(self.schemas))

    def get(self, place: Place) -> Fact:
        return self.facts.get(place, _UNBOUND)

    def set(self, place: Place, fact: Fact) -> None:
        self.facts[place] = fact

    def join_with(self, other: "_Env", origin: OriginStack, default: "Callable[[Place], Fact] | None" = None) -> bool:
        changed = False
        for place in set(self.schemas) | set(other.schemas):
            ours, others = self.schemas.get(place), other.schemas.get(place)
            joined_schema = join_schemas(ours, others) if ours is not None and others is not None else (ours or others)
            assert joined_schema is not None
            if joined_schema != self.schemas.get(place):
                self.schemas[place] = joined_schema
                changed = True
        # Per-place fact joins are independent, so a rejection defers until every place has joined.
        deferred = DeferredRejection()
        for place in set(self.facts) | set(other.facts):
            mine, theirs = self.facts.get(place), other.facts.get(place)
            if default is not None and (mine is None or theirs is None):
                fallback = default(place)
                mine = fallback if mine is None else mine
                theirs = fallback if theirs is None else theirs
            try:
                joined = join_facts(
                    mine if mine is not None else _UNBOUND, theirs if theirs is not None else _UNBOUND, origin
                )
            except AnalysisRejection as error:
                deferred.offer(error)
                continue
            if joined != self.facts.get(place, _UNBOUND):
                self.facts[place] = joined
                changed = True
        deferred.raise_if_deferred()
        return changed


class _TargetTest(ABC):
    """
    The dispatch key of a call row. Matching is by object IDENTITY throughout, never equality and never a
    hash lookup: a call target may be an unhashable shadow of a builtin name (a bound array, a dict) that has
    to miss cleanly rather than raise, and equality on a numpy callable is not a predicate to build on.
    """

    @abstractmethod
    def admits(self, target: object) -> bool: ...


@dataclass(frozen=True, slots=True, eq=False)
class _AnyOf(_TargetTest):
    targets: tuple[object, ...]

    def admits(self, target: object) -> bool:
        return any(target is candidate for candidate in self.targets)


@dataclass(frozen=True, slots=True, eq=False)
class _Matching(_TargetTest):
    """A target no table can name when it is built: a user's record class, a registry membership, anything."""

    predicate: Callable[[object], bool]

    def admits(self, target: object) -> bool:
        return self.predicate(target)


@dataclass(frozen=True, slots=True)
class _CallSite:
    """A call as the dispatch rows see it: the resolved target beside its operand facts, positional and keyword."""

    target: object
    call: PyCall
    env: _Env
    args: list[Fact]
    kwargs: list[tuple[str, Fact]]

    @property
    def operands(self) -> list[Fact]:
        return [*self.args, *(fact for _, fact in self.kwargs)]

    @property
    def sole_argument(self) -> Fact | None:
        """The single positional operand of a keyword-free call, which most structural arms require."""
        return self.args[0] if len(self.args) == 1 and not self.kwargs else None


@dataclass(frozen=True, slots=True, eq=False)
class _CallRow:
    """
    One arm of an ordered call-dispatch table. A selected row CONSUMES the call -- it binds the destination or
    refuses -- so nothing falls through past a matched action and the guard has to carry the whole
    applicability condition rather than its plausible part.
    """

    target: _TargetTest
    guard: Callable[[_CallSite], bool]
    apply: Callable[["Analyzer", _CallSite], None]

    def selects(self, site: _CallSite) -> bool:
        return self.target.admits(site.target) and self.guard(site)


@dataclass(frozen=True, slots=True, eq=False)
class _ShapeRow:
    """
    One arm of the library call-shape table. A selected row always REFUSES; a spelling that matches no row, or
    whose row finds no violation, passes through to its stub. First match decides which violation is reported,
    so a spelling's arity rule has to precede its operand rule.
    """

    target: _TargetTest
    violated: Callable[[_CallSite], bool]
    reject: Callable[[_CallSite], NoReturn]

    def selects(self, site: _CallSite) -> bool:
        return self.target.admits(site.target) and self.violated(site)


@dataclass(frozen=True, slots=True)
class _AttrSite:
    """An attribute read as the intercept rows see it."""

    op: PyAttr
    block: Block
    index: int
    env: _Env
    receiver: Fact


@dataclass(frozen=True, slots=True, eq=False)
class _AttrRow:
    """
    One arm of the ordered table of attribute reads resolved structurally, ahead of generic attribute
    resolution. A selected action reports whether it REWROTE the op, in which case the transfer revisits it.
    """

    names: tuple[str, ...]
    receiver: Callable[[Fact], bool]
    apply: Callable[["Analyzer", _AttrSite], bool]

    def selects(self, site: _AttrSite) -> bool:
        return site.op.name in self.names and self.receiver(site.receiver)


def _anything(target: object) -> bool:
    return True


def _is_record_class(target: object) -> bool:
    return isinstance(target, type) and is_dataclass(target)


def _always(site: _CallSite) -> bool:
    return True


def _sole_operand(site: _CallSite) -> bool:
    return site.sole_argument is not None


def _sole_array_operand(site: _CallSite) -> bool:
    sole = site.sole_argument
    return sole is not None and _is_array_fact(sole)


def _sole_aggregate_operand(site: _CallSite) -> bool:
    return isinstance(site.sole_argument, AggregateFact)


def _sole_residual_operand(site: _CallSite) -> bool:
    return isinstance(site.sole_argument, Residual)


def _explicit_float_dtype(site: _CallSite) -> bool:
    """An explicit ``dtype=float`` IS the conversion the implicit-widening rejections demand: every leaf casts."""
    if len(site.kwargs) != 1:
        return False
    keyword, fact = site.kwargs[0]
    return keyword == "dtype" and isinstance(fact, Reference) and any(fact.obj is kind for kind in (float, np.float64))


def _residual_array_source(site: _CallSite) -> bool:
    if len(site.args) != 1 or (site.kwargs and not _explicit_float_dtype(site)):
        return False
    source = site.args[0]
    return isinstance(source, AggregateFact) and any(isinstance(leaf, Residual) for leaf in source.leaves)


def _reduction_axis_given(site: _CallSite) -> bool:
    sole_keyword_operand = not site.call.args and [keyword for keyword, _ in site.call.kwargs] == ["a"]
    return not (len(site.call.args) == 1 and not site.call.kwargs) and not sole_keyword_operand


def _non_array_operand(site: _CallSite) -> bool:
    return any(not _is_array_fact(fact) for fact in site.operands)


def _integer_operand(site: _CallSite) -> bool:
    return any(_is_integer_operand(fact) for fact in site.args)


def _library_spelling(target: object) -> str:
    return getattr(target, "__name__", "?")


def _callee_name(target: object) -> str:
    return getattr(target, "__name__", repr(target))


def _reject_reduction_axis(site: _CallSite) -> NoReturn:
    raise AnalysisRejection(
        f"np.{_library_spelling(site.target)} supports only the default axis: exactly one array "
        "argument (reduce the other axis explicitly instead of passing an axis)",
        site.call.origin,
    )


def _reject_matrix_operand(site: _CallSite) -> NoReturn:
    raise AnalysisRejection(
        "the matrix product requires array operands on both sides; a scalar, list, or "
        "tuple does not acquire matrix semantics (wrap it in np.array(...))",
        site.call.origin,
    )


def _reject_array_operand(site: _CallSite) -> NoReturn:
    raise AnalysisRejection(
        f"np.{_library_spelling(site.target)} requires array operands; a scalar, list, or "
        "tuple does not acquire array semantics (wrap it in np.array(...))",
        site.call.origin,
    )


def _reject_integer_sign(site: _CallSite) -> NoReturn:
    # np.sign is int-polymorphic like abs (np.sign of an integer is an integer); its float composite would round
    # subsequent integer arithmetic, and there is no integer sign yet, so an integer operand refuses.
    raise AnalysisRejection("an integer operand to np.sign is not yet lowerable; cast to float first", site.call.origin)


def _positional_arity_rule(targets: tuple[object, ...], expected: int) -> _ShapeRow:
    """
    These spell positional ufunc-style calls: numpy itself refuses matmul keywords, and an offset or axis
    argument reaches machinery the subset does not model.
    """

    def violated(site: _CallSite) -> bool:
        return bool(site.call.kwargs) or len(site.call.args) != expected

    def reject(site: _CallSite) -> NoReturn:
        raise AnalysisRejection(
            f"np.{_library_spelling(site.target)} takes exactly {expected} positional array " "argument(s) here",
            site.call.origin,
        )

    return _ShapeRow(_AnyOf(targets), violated, reject)


def _check_call_shape(site: _CallSite) -> None:
    for row in _LIBRARY_SHAPE_ROWS:
        if row.selects(site):
            row.reject(site)


def _is_integer_operand(fact: Fact | None) -> bool:
    return fact == Residual(SemType.INT) or (isinstance(fact, Known) and isinstance(fact.value, (MetaInt, NpInt)))


def _stub_frames(origin: OriginStack, display: str | None) -> OriginStack:
    """A registry stub's template frames rebrand to its public spelling; a user callee's pass through untouched."""
    if display is None:
        return origin
    return tuple(replace(frame, function=display) for frame in origin)


class _UnrollRestart(Exception):
    """
    A loop header's iterable fact descended past the already-unrolled shape mid-round: the round must rerun
    with the joined fact seeded at the header, so the next unroll builds the stable shape in one pass. Seeds
    key the loop's ORIGIN STACK, not its block id: grafted and cloned headers get fresh block ids every round,
    while origins are re-attributed identically, so the origin is the round-stable identity. Clones of one
    source loop (an outer unroll's trips) share a seed, which only widens their join — still sound.
    """

    def __init__(self, origin: OriginStack, fact: Fact) -> None:
        super().__init__(origin)
        self.origin = origin
        self.fact = fact


def verify_plan_totality(result: "ResidualUnit") -> None:
    """
    Read back the recorder's postcondition before emission walks: every op emission subscripts a table for has
    an entry, over the blocks emission will visit.

    HONEST SCOPE, because the first two versions of this claimed more than they checked. This CANNOT FAIL for
    any `ResidualUnit` the analyzer can produce today, and it is not a general totality check:

    - the block sets do not diverge in practice. `_check_reachability_settled` runs BEFORE `_finalize` over
      the same `executable_rpo` walk and refuses a marked block the walk never reaches, and an edge leaving an
      unmarked one; a walked-but-unmarked SINK slips past both arms, which is why this checks that direction
      itself -- without it such a block reaches emission and crashes unlocated ("block N was not sealed with a
      terminator"), measured. For every other shape the gate reports first, located;
    - `block_in` coverage is already implied -- `_finalize` bare-subscripts it for every executable block --
      so this arm restates a property that would have raised earlier in the same call;
    - cell routing is NOT checked here at all any more: `route_plans` is total over a derivable site set, so
      `verify_route_plans` re-derives that set and every row rather than reading a table back;
    - `binding_facts` is checked, but only for what reaches `result`: since M1, finalization bare-subscripts
      its own per-visit records to build that table, so a destination the recorder never wrote crashes there
      first and never reaches this arm. What this still catches is a table that loses an entry between
      finalization and emission;
    - the J6 obligation the ruling assigned to this function by name -- every kind promotion consumed from an
      explicit plan row rather than derived by inspecting emitted nodes -- is NOT implemented here, and
      production emission still does it (`docs/decisions/arch-ruling.md`, outstanding M2/M3 work).

    What it is FOR is M1, which rewrites recording to be evidence-atomic: a recorder that stops writing a plan
    for a `PyCall` inside a block emission still visits is the regression the plan arm catches. The other arms
    cover shapes the upstream gate provably does not, and they cover DIFFERENT failures. The serious ones are
    silent: over the bundled corpus, 171 of 314 severed JUMP edges emit different HIR with no error, and
    a dropped parameter fact emits a differently typed input port. A severed BRANCH arm is never silent across
    196 severances, but "crash" overstates it -- 21 of them, including 16 of the 20 folded ones, come out as
    ORDINARY LOCATED refusals, and only the rest are raw. A walked-but-unmarked sink is a raw crash.
    Kept deliberately as a postcondition read-back placed before the change that can break it.
    """
    walked = executable_rpo(result.unit.entry, result.executable_edges)
    unmarked = [block_id for block_id in walked if block_id not in result.executable_blocks]
    assert not unmarked, f"emission walks blocks the analysis did not mark executable: {unmarked}"
    missing_env = [block_id for block_id in walked if block_id not in result.block_in]
    assert not missing_env, f"emission walks blocks with no recorded environment: {missing_env}"
    # Parameters are not op destinations, so the binding-fact arm below cannot see them -- and emission reads
    # their facts from the entry environment to type the module's INPUT PORTS. Dropping one was measured to pass every
    # other arm and silently change a port from bool to float: an ABI divergence with no error at all, the
    # worst class this scaffold exists to catch, and squarely in what M1 rewrites.
    # Scoped exactly to what emission reads: the ENTRY environment only, and not the bound `self`, which
    # emission skips when it builds ports. Requiring `binding_facts` too, or including `self`, made the
    # verifier refuse results emission handles perfectly well -- a guard that fires where nothing is wrong
    # teaches its reader to edit the number.
    entry_facts = result.block_in[result.unit.entry].facts  # the entry is walked, so `missing_env` covered it
    ported = result.unit.params[1:] if result.unit.bound_self is not None else result.unit.params
    unfaced = [param for param in ported if Local(param) not in entry_facts]
    assert not unfaced, f"parameters with no recorded fact for emission to type the input port: {unfaced}"
    # M1's subject is fact recording, not only plans, and emission reads a fact for every destination it
    # materializes. A missing one surfaces as a named assert deep in the walk rather than a raw crash, which is
    # milder than the tables above -- but it is exactly what a rewritten recorder can drop, so it is checked
    # here where the failure can name the block and the op.
    missing_facts = [
        (block_id, op_dst(op))
        for block_id in walked
        for op in result.unit.blocks[block_id].ops
        if op_dst(op) is not None and op_dst(op) not in result.binding_facts
    ]
    assert not missing_facts, "; ".join(
        f"block {block_id} defines {dst} with no recorded fact" for block_id, dst in missing_facts
    )
    # A block that executes hands control to its Jump target, so that edge must be recorded. Dropping one
    # whose target keeps another predecessor leaves every block still walked and every table still total, so
    # the arms above see nothing -- and emission silently produces DIFFERENT HIR, which is the worst outcome
    # this scaffold exists to make impossible. Branch arms are checked too, per the note below.
    severed = []
    for block_id in walked:
        terminator = result.unit.blocks[block_id].terminator
        if isinstance(terminator, Jump) and (block_id, terminator.target) not in result.executable_edges:
            severed.append(block_id)
        # A branch whose condition did not settle takes BOTH arms, so both edges are obligatory. Severing one
        # leaves every block walked and every table total, then dies inside emission with a phi that has no arm
        # for a predecessor: an unlocated CRASH, not the silent divergence the jump case above produces --
        # measured, 31 severances, none silent.
        #
        # A FOLDED branch keeps only the arm its condition selects, and THAT arm is obligatory: severing it
        # reaches emission as a refusal on an innocent line -- measured, a located "the function never returns
        # on any path" rather than the raw KeyError an earlier note here claimed. That note called the rule
        # "measurably false" because it failed 44 tests; that measurement was a bug of mine -- the condition's
        # `value` is a StaticBool WRAPPER and so always truthy, which selected the wrong arm. Unwrapped through
        # `as_python` the rule passes, and the hole is closed.
        if isinstance(terminator, Branch):
            condition = result.binding_facts.get(terminator.cond)
            if isinstance(condition, Known):
                taken = as_python(condition.value)
                assert isinstance(taken, bool)
                obligatory: tuple[BlockId, ...] = (terminator.then_target if taken else terminator.else_target,)
            else:
                obligatory = (terminator.then_target, terminator.else_target)
            for target in obligatory:
                if (block_id, target) not in result.executable_edges:
                    severed.append(block_id)
    assert not severed, f"blocks whose obligatory outgoing edge is missing from the executable set: {severed}"
    missing_plans = [
        (block_id, op)
        for block_id in walked
        for op in result.unit.blocks[block_id].ops
        if isinstance(op, PyCall) and op.dst not in result.call_plans
    ]
    assert not missing_plans, "; ".join(
        f"block {block_id} call at {op.origin[0]} has no call plan" for block_id, op in missing_plans
    )


@dataclass(slots=True)
class ResidualUnit:
    """
    The analyzer's single authoritative output: the stabilized working graph, the per-edge environments, and the
    typed emission plan (final binding facts, call plans, state discovery, component provenance). Emission consumes
    only this -- it never re-derives a fold, re-resolves the library registry, or replays the transfer function.
    """

    unit: FunctionUnit
    block_in: dict[BlockId, _Env]
    executable_blocks: set[BlockId]
    executable_edges: set[tuple[BlockId, BlockId]]
    binding_facts: dict[BindingId, Fact] = field(default_factory=dict)
    call_plans: dict[BindingId, CallPlan] = field(default_factory=dict)
    route_plans: dict[PlanSite, RoutePlan] = field(default_factory=dict)
    # The pinned per-class field schemas, so the plan verifier can re-derive a construction's field binding from
    # the schema the analysis actually used rather than from live dataclass metadata, which can disagree in time.
    construction_schemas: dict[int, tuple[type, tuple[FieldSchema, ...]]] = field(default_factory=dict)
    store_order: list[StateLeaf] = field(default_factory=list)
    runtime_state: set[StateLeaf] = field(default_factory=set)
    state_livein: dict[StateLeaf, Fact] = field(default_factory=dict)
    state_resets: dict[StateLeaf, "StaticValue | str"] = field(default_factory=dict)
    provenance: dict[int, tuple[str, ...]] = field(default_factory=dict)
    store_origins: dict[StateLeaf, OriginStack] = field(default_factory=dict)


def _validate(result: ResidualUnit, classified: Mapping[int, CallLowering]) -> None:
    for block_id in sorted(result.executable_blocks, key=lambda block_id: block_id.index):
        block = result.unit.blocks[block_id]
        for op in block.ops:
            assert not isinstance(op, PyCall) or id(op) in classified, f"{block_id}: unexpanded call survived"
        assert not isinstance(block.terminator, StaticFor), f"{block_id}: loop template survived analysis"


class Analyzer:
    def __init__(self, fn: object) -> None:
        self._root_template = build_unit(fn, root=True)
        self._templates: dict[tuple[int, int | None], tuple[object, FunctionUnit]] = {}
        self._block_ancestry: dict[BlockId, tuple[tuple[int, int | None], ...]] = {}
        self._temp_serial = 1_000_000
        self._binding_serial = 1_000_000
        self._block_serial = 1_000_000
        self._runtime_state: set[StateLeaf] = set()
        # Where each runtime-state leaf came from: the source-earliest store that promoted it, latched on the
        # ROUND that first promoted it. Monotonicity keeps the LEAF in the set, not that store in the graph, so
        # the latched store may well be one a later round proves unreachable -- which is exactly what the
        # stale-leaf refusal is about. Whether any OTHER diagnostic should prefer it is a genuine trade, not a
        # settled question; `_state_origin` weighs it. Its provenance cannot live in the per-round store map,
        # which is empty for it once that store stops executing.
        self._runtime_state_origins: dict[StateLeaf, OriginStack] = {}
        self._state_livein: dict[StateLeaf, Fact] = {}
        # Still block-keyed and still additive -- the shape that made the store records above need op keys.
        # It does not bite here, but only incidentally: W-promotion also requires the block to be co-reachable
        # with the exit, and after a graft a block reaches the exit only through the continuation, so a
        # relocated store is promoted exactly when the op-keyed record covers it anyway.
        self._discovered_stores: set[tuple[BlockId, StateLeaf]] = set()
        self._discovered_store_origins: dict[tuple[BlockId, StateLeaf], OriginStack] = {}
        # Recorded AT THE VISIT that computes them, overwritten each time. The fixpoint revisits a block
        # whenever an incoming fact changes, so a block that is not revisited did not change: the last write
        # is the stabilized one. This is what lets `_finalize` stop replaying the transfer, which used to run
        # every host fold a second time.
        # Which leaf each store op writes, keyed by the OP, because a graft RELOCATES the store rather than
        # destroying it: the record is never removed, and what keeps it out of the plan is that `_finalize`
        # walks ops per executable block, so the record is consulted only where the op now lives. Keyed by
        # block instead, a store the graft moved into an unreachable continuation would stay in the store
        # order and be snapshotted as a nonexistent attribute -- measured, it falsely rejected a kernel the
        # replay accepted.
        self._store_leaf_of_op: dict[int, StateLeaf] = {}
        self._visit_facts: dict[BindingId, Fact] = {}
        self._visit_call_plans: dict[BindingId, CallPlan] = {}
        # How each surviving call lowers, decided at the visit that classifies it. Classification is never
        # inferred from whether a route exists: an identity conversion and a zero-cell one are both conversions.
        self._call_lowering: dict[int, CallLowering] = {}
        # The one routing record: which source cell (or which field binding) feeds each result cell, for the
        # sites whose selection the final facts cannot recover. Finalization turns it into the route plan.
        self._route_evidence: dict[int, RouteEvidence] = {}
        # Schema and default snapshots, one per class per ANALYSIS (never per visit): a mutable field default
        # (an eq=False record holding a list, say) must not move a fact between fixpoint visits. Defaults are
        # admitted LAZILY at the first construction that actually omits the field (Python never observes an
        # overridden default). The pinned class reference keeps ids stable.
        self._construction_schemas: dict[int, tuple[type, tuple[FieldSchema, ...]]] = {}
        self._default_snapshots: dict[tuple[int, str], BoundFact] = {}
        self._unroll_cache: dict[BlockId, tuple[Fact, BlockId]] = {}
        self._unroll_seeds: dict[OriginStack, Fact] = {}  # survives rounds: the joined facts of restarted headers
        self._store_origins: dict[StateLeaf, OriginStack] = {}
        self._store_verdicts: dict[int, StoreVerdict] = {}  # op id -> the last bound execution's verdict this round
        self._unbound_store_origins: set[OriginStack] = set()  # origins that executed with an Unbound value
        # The pending bridge keeps the deferral net closed across round boundaries and nothing more: op ids are
        # round-scoped (grafted and cloned ops are fresh objects every round), so each true round boundary
        # reconciles the still-standing verdicts into this origin-keyed map -- origins are re-attributed
        # identically across rounds, the same identity the unroll seeds key on. It is never popped mid-round
        # (a conforming unroll clone must not open a window for its violating same-origin sibling), carries
        # through an _UnrollRestart untouched (a restart is a mid-round event, so a partial round is evidence
        # of nothing), and is never a verdict source for a store still in the graph: at stabilization an origin
        # with no store left in any executable block is stranded -- its own violation's cascade removed the
        # block -- and reports from here, while an origin whose store survived resolves from the walk alone.
        self._pending_bridge: dict[OriginStack, str] = {}
        self._store_facts: dict[int, Fact] = {}  # SOURCE store op id -> its stored fact at the last visit
        self._transfer_deferrals: dict[int, LocatedRejection] = {}  # op id -> rejection deferred behind a store
        # The live worklist state, instance-held so a mid-round graft can reconcile it with the CFG it mutates: a
        # graft that replaces a block's terminator retracts the recorded out-edges of the replaced terminator and
        # drops any successor thereby left with no in-edge, so its stale env is re-derived by the continuation.
        self._executable_edges: set[tuple[BlockId, BlockId]] = set()
        self._executable_blocks: set[BlockId] = set()
        self._block_in: dict[BlockId, _Env] = {}
        self._bound_methods: dict[tuple[int, str], object] = {}
        self._array_methods: dict[tuple[BindingId, str], _ArrayMethod] = {}
        self._class_annotations: dict[int, tuple[type, "Mapping[str, object]"]] = {}
        self._component_reads: dict[tuple[int, str], tuple[object, object, StaticValue | None]] = {}
        self._value_methods: dict[tuple[StaticValue, str], object] = {}
        self._roots: dict[int, tuple[str, ...]] = {}  # root component id -> the empty member path
        self._component_edges: set[tuple[int, str, int]] = set()  # (parent id, attribute, child id) sub-object graph

    def fixpoint(self, param_facts: dict[str, Fact] | None = None) -> ResidualUnit:
        """
        The outer W/D state fixed point: W (runtime-capable leaves) accumulates store sites that are executable
        AND graph-co-reachable with the canonical exit over executable edges; D (live-in facts) starts at
        Known(reset) and joins executable exit live-outs, descending only. Each round rebuilds the working graph
        from immutable templates; the final round's facts are computed under stable typing. A failing state join
        never aborts a round: the leaf's live-in freezes at its last joinable value (holding W/D descent, so the
        fixed point still stabilizes) and the failure re-derives each round, reporting only at stabilization --
        ranked below every recorded store obligation, since the cascade of a violating store can provoke exactly
        such a merge (an Unbound from a deferred producer joining a carried fact).
        """
        for round_index in range(1_000):
            try:
                result = self.analyze(param_facts)
            except _UnrollRestart as restart:
                # Facts descend monotonically and seeds only join downward, so reseeding terminates within the
                # round fuel. A restart is a mid-round event: the interrupted round's partial verdicts are
                # discarded without reconciling the bridge, which carries into the rerun unchanged.
                self._unroll_seeds[restart.origin] = restart.fact
                self._reset_round()
                _logger.info("state round %d: unroll reseeded at %s", round_index + 1, restart.origin[0])
                continue
            exit_env = result.block_in.get(result.unit.exit, _Env())
            reachable = _coreachable(result.unit, result.unit.exit, result.executable_edges)
            new_w = set(self._runtime_state)
            # Only a store that is executable AND co-reachable with the exit promotes its leaf, so provenance is
            # taken from those blocks alone: the per-leaf minimum over the whole round would happily blame a
            # store standing in a block the exit cannot be reached from.
            promoting: dict[StateLeaf, OriginStack] = {}
            for block_id, leaf in self._discovered_stores:
                if block_id in result.executable_blocks and block_id in reachable:
                    origin = self._discovered_store_origins[(block_id, leaf)]
                    earliest = promoting.get(leaf)
                    if earliest is None or origin_order(origin) < origin_order(earliest):
                        promoting[leaf] = origin
            new_w |= promoting.keys()
            for leaf, origin in promoting.items():
                self._runtime_state_origins.setdefault(leaf, origin)
            new_d = dict(self._state_livein)
            deferred = DeferredRejection()
            for leaf in new_w:
                try:
                    reset = self._state_reset_fact(leaf)
                    exit_fact = exit_env.get(leaf)
                    incoming = (
                        reset
                        if isinstance(exit_fact, Unbound)
                        else join_facts(reset, exit_fact, self._state_origin(leaf))
                    )
                    previous = new_d.get(leaf)
                    new_d[leaf] = (
                        incoming if previous is None else join_facts(previous, incoming, self._state_origin(leaf))
                    )
                except AnalysisRejection as error:
                    deferred.offer(error)
            if new_w == self._runtime_state and new_d == self._state_livein:
                _logger.info("state fixpoint stable after %d round(s): %d runtime leaves", round_index + 1, len(new_w))
                self._check_storage_schemas(result)
                deferred.raise_if_deferred()
                self._raise_transfer_deferrals(result)
                self._reject_executable_fails(result)
                # The chain splice runs after the rejection walks, which rank deferrals by the identity of the
                # ops and terminators they were recorded under; swapping a cached header's StaticFor for the
                # Jump earlier would orphan a join-layer deferral keyed on the replaced terminator.
                for header_id, (_, chain_entry) in self._unroll_cache.items():
                    header = result.unit.blocks[header_id]
                    assert isinstance(header.terminator, StaticFor)
                    header.terminator = Jump(chain_entry, header.terminator.origin)
                # Before _validate, whose asserts describe a graph the speculation gate may already know is
                # inconsistent: an unresolved call left on a speculated arm trips "unexpanded call survived" as a
                # bare AssertionError, where the gate has a located diagnostic for the same cause. Running it
                # second also inverted the -O contract, the debug build crashing where the optimized one explained.
                self._check_reachability_settled(result, self._executable_rank(result))
                _validate(result, self._call_lowering)
                self._finalize(result)
                return result
            self._runtime_state = new_w
            self._state_livein = new_d
            self._reconcile_bridge()
            self._reset_round()
            _logger.info("state round %d: %d runtime leaves, %d live-in facts", round_index + 1, len(new_w), len(new_d))
        raise AnalysisRejection(
            "state fixpoint failed to stabilize", (Origin(self._root_template.name, 0, 0, self._root_template.file),)
        )

    def _check_storage_schemas(self, result: ResidualUnit) -> None:
        enforce_storage_schemas(
            result.unit,
            result.executable_blocks,
            result.executable_edges,
            self._store_facts,
            {block_id: env.schemas for block_id, env in result.block_in.items()},
            self._store_verdicts,
            self._pending_bridge,
        )

    def _violations_pending(self) -> bool:
        return bool(self._pending_bridge) or any(
            verdict.message is not None for verdict in self._store_verdicts.values()
        )

    def _record_store_verdict(self, op: Op, message: str | None) -> None:
        # Last-bound-wins: the worklist can re-execute one CFG op under several environments, and the last
        # visit's env is the converged one, so its verdict is the fixpoint verdict -- an earlier violation
        # drawn on a pre-join transient (a Known integer one arm feeds a merge) is superseded, keeping the
        # merge-chartered store-edge conversion legal in either arm order. An Unbound execution records
        # nothing, so a bound verdict never settles an op the cascade also cut: the origin stays exempt from
        # the bridge pop through _unbound_store_origins.
        self._store_verdicts[id(op)] = StoreVerdict(message, op.origin)

    def _raise_transfer_deferrals(self, result: ResidualUnit) -> None:
        """
        Rejections deferred behind a pending store violation whose ops never went clean again. Reached only when
        the resolution walk and the state-join deferrals all came up clean (the pending violations were
        transient, or their stores fell dead), so each lingering entry is a real rejection re-derived on the
        op's stable facts. The one surfaced is the first in executable preorder. This is deliberately NOT the
        same selection a violation-free run makes: with nothing pending, the first rejection the worklist
        encounters raises immediately (LIFO visit order, with a min-by-str pick among the places of one failing
        edge join), so a kernel with several rejections can report a different one depending on whether an
        unrelated transient violation forced the deferral path. Both selections are deterministic and
        hash-seed-stable -- the worklist order is driven by integer block ids and the deferral walk by the
        preorder -- they just differ from each other; unifying them means deferring every rejection to
        stabilization, which today breaks call-expansion invariants (a deferred graft leaves executable SOURCE
        stores untransferred), so it needs its own redesign.
        """
        if not self._transfer_deferrals:
            return
        for block_id in executable_preorder(result.unit, result.executable_blocks, result.executable_edges):
            block = result.unit.blocks[block_id]
            for op in block.ops:
                error = self._transfer_deferrals.get(id(op))
                if error is not None:
                    raise error
            error = self._transfer_deferrals.get(id(block.terminator))
            if error is not None:
                raise error
        # A leftover is discardable only because its op provably sits on a dead path: binding validation
        # precedes the graft's destructive mutation and the graft re-keys the one op it destroys, so every key
        # still names an op or terminator in the graph -- necessarily in a block the stable round never
        # reached, where the rejection was derived on facts that no longer flow (or an executable clone
        # re-derived it and ranked above). A key absent from the whole graph would be a rejection silently
        # lost, which this assert makes a loud invariant violation instead.
        anchored = {id(op) for graph_block in result.unit.blocks.values() for op in graph_block.ops}
        anchored |= {id(graph_block.terminator) for graph_block in result.unit.blocks.values()}
        assert anchored >= set(self._transfer_deferrals), "a transfer deferral key left the graph"
        _logger.info("discarding %d dead-path transfer deferral(s)", len(self._transfer_deferrals))

    def _reconcile_bridge(self) -> None:
        # A violating verdict enters (superseding any older entry for its origin, the earliest-recorded message
        # winning among same-origin clones), and an origin leaves only on complete evidence: every execution
        # bound and conforming, none unbound. A clone that executed with an Unbound value recorded nothing, so
        # its conforming sibling alone must not pop the shared obligation -- the following round would run with
        # an open deferral net.
        violating: dict[OriginStack, str] = {}
        conforming: set[OriginStack] = set()
        for verdict in self._store_verdicts.values():
            if verdict.message is None:
                conforming.add(verdict.origin)
            else:
                violating.setdefault(verdict.origin, verdict.message)
        for origin in conforming - set(violating) - self._unbound_store_origins:
            self._pending_bridge.pop(origin, None)
        self._pending_bridge.update(violating)

    def _reset_round(self) -> None:
        self._block_ancestry = {}
        self._discovered_stores = set()
        self._discovered_store_origins = {}
        self._store_leaf_of_op = {}
        # Cleared with the rest of the round's evidence: an unroll restart re-runs the worklist, so every
        # surviving destination is recorded again. Retained across restarts these grew quadratically on a
        # deep inlined state chain -- 50,056 facts for 392 final destinations, 10.4 MB against 2.2 MB.
        self._visit_facts = {}
        self._visit_call_plans = {}
        self._store_origins = {}
        self._store_verdicts = {}
        self._unbound_store_origins = set()
        self._store_facts = {}
        self._transfer_deferrals = {}
        self._call_lowering = {}
        self._route_evidence = {}
        self._unroll_cache = {}

    def _read_attribute_snapshot(
        self, owner: object, name: str, origin: OriginStack
    ) -> tuple[object, StaticValue | None]:
        """
        One live read AND one admission per (owner, attribute) per analysis: every later consultation -- W/D
        rounds, reset facts, namespace lookups, plan finalization -- sees the first read's ADMITTED value,
        so neither a drifting live object nor a mutated referent (admission snapshots contents at admit time)
        can move a fact after it is first formed. The owner reference pins the id against reuse for the memo's
        lifetime. AttributeError propagates to the caller's located rejection. A 0-d ndarray refuses right here,
        never entering the memo: this read is its creation door (scope ruling T3) for state resets, component
        reads, and namespace lookups alike, mirroring the builder's global-load door.
        """
        key = (id(owner), name)
        hit = self._component_reads.get(key)
        if hit is not None and hit[0] is owner:
            return hit[1], hit[2]
        value = getattr(owner, name)
        if type(value) is np.ndarray and value.ndim == 0:
            raise AnalysisRejection("a 0-dimensional array is not supported; use the scalar directly", origin)
        admitted = admit(value)
        self._component_reads[key] = (owner, value, admitted)
        return value, admitted

    def _state_origin(self, leaf: StateLeaf) -> OriginStack:
        """
        State rejections locate at a store to the leaf: __init__ is never analyzed, so a store is the line the
        user can act on. THE PER-ROUND MAP LEADS and the promotion origin stands behind it, which is the
        historical order; the fallback exists for the cross-round case, where the map is empty and the
        alternative is no location at all. NEITHER SOURCE DOMINATES, and the order is not a fix for anything:
        the map can name a store in a block the exit cannot be reached from, the latch can name one the
        stabilized facts later prove dead (it is fixed at the round that first promoted the leaf, and the state
        set's monotonicity keeps the LEAF, not that store's reachability), and each order is better than the
        other on a witness the seam already has, and both are pinned. Worse, a verdict raised BEFORE THE
        LEAF HAS A LATCH ENTRY -- the latch is per leaf and survives round resets, so this covers every round
        up to the one whose end-of-round pass would fill it, and all rounds if a verdict aborts the analysis
        first -- finds only whatever stores the worklist has reached, speculated arms included, so a dead arm
        can take the anchor under either order. That residual belongs to the deferral seam's documented class
        and is pinned as a witness rather than patched here.
        """
        stored = self._store_origins.get(leaf)
        recorded = stored if stored is not None else self._runtime_state_origins.get(leaf)
        return recorded if recorded is not None else (Origin(self._root_template.name, 0, 0, self._root_template.file),)

    def _snapshot_leaf(self, leaf: StateLeaf) -> Fact:
        current, admitted = self._walk_snapshot(leaf)
        return normalize_static(admitted) if admitted is not None else Reference(current)

    def _walk_snapshot(self, leaf: StateLeaf) -> tuple[object, StaticValue | None]:
        current: object = leaf.component
        admitted: StaticValue | None = None
        for attribute in leaf.path:
            try:
                current, admitted = self._read_attribute_snapshot(current, attribute, self._state_origin(leaf))
            except AttributeError:
                raise AnalysisRejection(
                    f"state attribute '{'.'.join(leaf.path)}' does not exist on the component at compile time "
                    "(assign it in __init__)",
                    self._state_origin(leaf),
                ) from None
        return current, admitted

    def _state_reset_fact(self, leaf: StateLeaf) -> Fact:
        current, admitted = self._walk_snapshot(leaf)
        if admitted is None:
            return Reference(current)
        origin = self._state_origin(leaf)
        name = ".".join(leaf.path)
        reset = normalize_static(admitted)
        self._validate_state_annotation(leaf, reset, origin)
        if isinstance(reset, AggregateFact):
            self._validate_state_reset_schema(leaf, reset.layout, origin)
            cells: list[AtomicFact] = []
            for cell in reset.leaves:
                assert isinstance(cell, Known)
                cell_sem = _residual_type(cell.value)
                if cell_sem is None:
                    raise AnalysisRejection(f"state attribute '{name}' has an unsupported reset type", origin)
                # The scalar reset rule applies per cell: a Known Bool folds exactly, numerics run as residuals.
                cells.append(cell if cell_sem is SemType.BOOL else Residual(cell_sem))
            return AggregateFact(reset.layout, tuple(cells))
        sem = _residual_type(admitted)
        if sem is None:
            raise AnalysisRejection(f"state attribute '{name}' has an unsupported reset type", origin)
        if sem is SemType.BOOL:
            return Known(admitted)  # a Known Bool folds exactly (no width) and keeps invariant flags folding
        return Residual(sem)

    def _validate_state_reset_schema(self, leaf: StateLeaf, layout: AggregateLayout, origin: OriginStack) -> None:
        """
        Persistent aggregate state is a FLAT list of scalars or a nonempty 1-D/2-D plain ndarray: the reset fixes
        the slot geometry the next transaction reconstructs, cell names come from its leaf paths, and a declared
        jaxtyping field annotation must agree with it. Everything else (a nested list, a tuple, a 3-D array, an
        empty aggregate) has no honest per-cell slot decomposition yet and rejects by name.
        """
        name = ".".join(leaf.path)
        match layout:
            case ListLayout(items=items):
                if not all(item is None for item in items):
                    raise AnalysisRejection(
                        f"state attribute '{name}' must be a flat list of scalars to persist", origin
                    )
            case ArrayLayout(shape=shape):
                if len(shape) > 2:
                    raise AnalysisRejection(
                        f"state attribute '{name}' has a {len(shape)}-D array reset; only 1-D and 2-D arrays "
                        "persist",
                        origin,
                    )
            case _:
                raise AnalysisRejection(f"state attribute '{name}' has an unsupported reset type", origin)
        if leaf_count(layout) == 0:
            raise AnalysisRejection(f"state attribute '{name}' has an empty aggregate reset", origin)

    def _validate_state_annotation(self, leaf: StateLeaf, reset: Fact, origin: OriginStack) -> None:
        """
        A declared jaxtyping FIELD annotation must agree with the reset -- for every reset kind, a scalar
        included (a declared 2-vector seeded with a float is the honest mistake this catches). Annotations are
        read in the deferred FORWARDREF format: PEP 649 evaluates them lazily, and a TYPE_CHECKING-only name
        anywhere on the class (ordinary typing practice) must neither crash the analysis nor block state --
        an unresolved proxy is simply not an array annotation. A raising annotation body is a located
        rejection, never a leaked exception.
        """
        name = ".".join(leaf.path)
        annotation: object = None
        for klass in type(leaf.component).__mro__:
            annotations = self._class_annotations_of(klass, name, origin)
            if leaf.path[-1] in annotations:
                annotation = annotations[leaf.path[-1]]
                break
        if annotation is None or not is_array_annotation(annotation):
            return
        try:
            declared = array_shape(annotation)
        except ContractError as error:
            raise AnalysisRejection(f"state attribute '{name}': {error}", origin) from None
        matches = (
            isinstance(reset, AggregateFact)
            and isinstance(reset.layout, ArrayLayout)
            and reset.layout.shape == declared
        )
        if not matches:
            raise AnalysisRejection(
                f"state attribute '{name}' has a reset diverging from its declared array type "
                f"{'x'.join(map(str, declared))}",
                origin,
            )

    def _class_annotations_of(self, klass: type, state_name: str, origin: OriginStack) -> "Mapping[str, object]":
        import annotationlib

        hit = self._class_annotations.get(id(klass))
        if hit is not None and hit[0] is klass:
            return hit[1]
        try:
            annotations = annotationlib.get_annotations(klass, format=annotationlib.Format.FORWARDREF)
        except Exception as error:
            raise AnalysisRejection(
                f"state attribute '{state_name}': the annotations of class {klass.__name__!r} fail to "
                f"evaluate ({error})",
                origin,
            ) from None
        self._class_annotations[id(klass)] = (klass, annotations)
        return annotations

    def _reject_executable_fails(self, result: ResidualUnit) -> None:
        """
        An executable Fail terminator sits on a path taken unconditionally (or under a residual guard the
        hardware cannot signal): a located rejection carrying the raise's own message, with any f-string
        interpolations rendered from the compile-time facts at the raise site. The walk is a preorder over
        executable edges (then-arm first), so the raise reported is the first one execution can reach —
        unroll-clone block indices do not follow iteration order (a reversed range), so index order would
        misreport which raise fires.
        """
        for block_id in executable_preorder(result.unit, result.executable_blocks, result.executable_edges):
            block = result.unit.blocks[block_id]
            terminator = block.terminator
            if isinstance(terminator, Fail):
                env = result.block_in[block_id].copy()
                for index, op in enumerate(block.ops):
                    self._transfer(result.unit, block, index, op, env)
                raise AnalysisRejection(self._render_fail(terminator.parts, env), terminator.origin)

    def _render_fail(self, parts: tuple[str | BindingId, ...], env: _Env) -> str:
        rendered: list[str] = []
        for part in parts:
            if isinstance(part, str):
                rendered.append(part)
                continue
            concrete = _concrete_fact(env.get(Local(part)))
            if concrete is None:
                return "raise with a runtime-interpolated message"
            rendered.append(render_interpolation(as_python(concrete)))
        return "".join(rendered)

    def _finalize(self, result: ResidualUnit) -> None:
        """
        The emission plan is assembled from evidence recorded at the visits that computed it: the authoritative fact per
        binding (temporaries are write-once, so one pass records each), a typed plan per surviving call (keyed by
        the call's destination binding, never by op identity), and state leaves in first-store SOURCE order (the
        order key is the storing op's origin with the user call site primary, so a store nested in a branch has a
        higher block id than a later top-level store yet comes first in the source text, and two call sites
        inlining one setter order by the call sites, tie-broken by the callee frames). Clones of ONE store op
        (unroll trips of a loop over components) share the whole origin chain, so equal origin keys tie-break by
        the storing block's execution rank -- trip order, the source order of the iterable -- never by raw block
        id, which the unroller hands out in reverse trip order. One independence is given up by not replaying:
        the store order and the runtime-state set now derive from the same recording lines, so the stale-leaf
        refusal can no longer catch a disagreement between what a round RECORDED and what the stabilized graph
        CONTAINS -- only the cross-round staleness it was written for. Nothing here re-runs the transfer, which used to
        execute every concrete library fold a second time; the walk over the stabilized ops only reads what the
        fixpoint already recorded, which is also what keeps a store the graph no longer contains out of the
        plan. Emission consumes only this plan.
        """
        rank = self._executable_rank(result)
        blocks_in_order = sorted(result.executable_blocks, key=lambda block_id: block_id.index)
        first_store: dict[StateLeaf, tuple[tuple[tuple[int, int], ...], int]] = {}
        for block_id in blocks_in_order:
            for op in result.unit.blocks[block_id].ops:
                leaf = self._store_leaf_of_op.get(id(op)) if isinstance(op, PyStoreAttr) else None
                if leaf is not None:
                    position = (source_position(op.origin), rank[block_id])
                    if leaf not in first_store or position < first_store[leaf]:
                        first_store[leaf] = position
                        result.store_origins[leaf] = op.origin
                if isinstance(op, PyCall):
                    # Bare subscripts on purpose, here and for the fact below: every surviving op was visited,
                    # so a miss means the recording premise itself broke, and the assert names which record is
                    # missing where a raw KeyError would only name the line.
                    assert op.dst in self._visit_call_plans, f"call {op.dst} was never visited"
                    result.call_plans[op.dst] = self._visit_call_plans[op.dst]
                dst = op_dst(op)
                if dst is not None:
                    assert dst in self._visit_facts, f"binding {dst} was never visited"
                    result.binding_facts[dst] = self._visit_facts[dst]
        # Before the branch check, not after: that check bare-subscripts the condition's fact, so seeding
        # parameters afterwards would make a parameter-conditioned branch the one shape that crashes there.
        # No such branch exists today -- every condition is a `PyTruth` destination -- but the check's
        # correctness should not rest on which pass happens to run first.
        entry_env = result.block_in[result.unit.entry]
        for param in result.unit.params:
            result.binding_facts.setdefault(param, entry_env.get(Local(param)))
        for block_id in blocks_in_order:
            self._check_branch_settled(result, block_id)
        result.store_order = sorted(first_store, key=lambda leaf: first_store[leaf])
        # A leaf enters W when a store to it is discovered, and W only ever grows: a store that a LATER round
        # proves unreachable leaves its leaf behind as runtime state anyway. That is unsound rather than merely
        # untidy, because the slot's reset then materializes in the target carrier instead of folding at
        # binary64, so a guard reading it can flip -- the same harm the branch rule refuses, arriving across
        # rounds rather than within one, where every per-round check passes on the stable graph.
        stale_leaves = sorted(
            (leaf for leaf in self._runtime_state if leaf not in first_store),
            # Position alone does not separate two components whose stores sit at the same line and column in
            # different files, so the frames' identities come next. Unroll clones of ONE store over two
            # components still tie on everything, including the path -- harmlessly, since the message names only
            # the path and the shared origin, so either choice renders identically.
            key=lambda leaf: (origin_order(self._runtime_state_origins[leaf]), leaf.path),
        )
        if stale_leaves:
            leaf = stale_leaves[0]
            raise AnalysisRejection(
                f"state attribute {'.'.join(leaf.path)!r} was discovered as runtime state on an earlier "
                "analysis round whose store the stabilized facts leave unreachable, so its reset would be "
                "materialized as if it were written; the result cannot be trusted",
                self._runtime_state_origins[leaf],
            )
        result.runtime_state = set(self._runtime_state)
        result.state_livein = dict(self._state_livein)
        for leaf in {*result.runtime_state, *result.store_order, *result.state_livein}:
            raw, admitted = self._walk_snapshot(leaf)
            result.state_resets[leaf] = admitted if admitted is not None else type(raw).__name__
        result.provenance = self._component_provenance()
        result.construction_schemas = dict(self._construction_schemas)
        # Last, because it consumes the state resets and the binding facts assembled above.
        result.route_plans = produce_route_plans(
            result.unit,
            result.executable_edges,
            {block_id: env.schemas for block_id, env in result.block_in.items()},
            result.binding_facts,
            result.call_plans,
            self._route_evidence,
            result.state_resets,
        )

    def _block_origin(self, unit: FunctionUnit, block_id: BlockId) -> OriginStack:
        terminator = unit.blocks[block_id].terminator
        if terminator is not None:
            return terminator.origin
        return (Origin(self._root_template.name, 0, 0, self._root_template.file),)

    def _executable_rank(self, result: ResidualUnit) -> dict[BlockId, int]:
        return {
            block_id: position
            for position, block_id in enumerate(executable_rpo(result.unit.entry, result.executable_edges))
        }

    def _check_reachability_settled(self, result: ResidualUnit, rank: dict[BlockId, int]) -> None:
        """
        Refuse when the recorded reachability disagrees with the stabilized facts, whole-graph and once --
        with one measured exception: a block the walk reaches but the analysis never marked executable is seen
        only through that block's own out-edges, so a SINK in that state passes here. `verify_plan_totality`
        checks that direction before emission, which is where it would otherwise surface as a raw crash.

        Optimistic traversal explores an arm on the strength of a condition that is not yet settled -- which is
        what lets W/D discover runtime state at all -- and marks are add-only, so an arm the stable facts later
        prove dead stays marked and is analyzed, emitted, and counted by the read-only-attribute scan. That last
        effect is a MISCOMPILE and not merely wasted logic: a store on the dead arm promotes an attribute from a
        constant folded at binary64 into a runtime slot whose reset is materialized in the target carrier, so a
        guard reading it can flip. The companion per-branch check is ``_check_branch_settled``.

        Retracting the stale mark is not available here -- environments are joined destructively, so removing an
        edge would mean recomputing downstream environments, schemas, reachability, W/D discoveries, and phis --
        and refusing the condition instead starves the outer fixed point (a measured regression). Until
        residualization becomes a total post-stabilization pass that recomputes reachability from the stable facts
        rather than inheriting these sets, this gate converts the unsound acceptance into an honest refusal.
        """
        # An edge out of a block the stable facts leave unexecutable: the graft that removed the source's mark and
        # env left its out-edges standing, so the target keeps a predecessor that never runs and its phi has no
        # arm for it. Ordered by block index so the first report is stable across hash seeds.
        stale = sorted(
            (edge for edge in result.executable_edges if edge[0] not in result.executable_blocks),
            key=lambda edge: (edge[0].index, edge[1].index),
        )
        if stale:
            source, _ = stale[0]
            raise AnalysisRejection(
                "analysis kept an execution path out of a region the stabilized facts leave unreachable, so a "
                "value here would be read on a path that never runs; the result cannot be trusted",
                self._block_origin(result.unit, source),
            )
        unreached = sorted(
            (block_id for block_id in result.executable_blocks if block_id not in rank), key=lambda b: b.index
        )
        if unreached:  # marked executable, yet no executable edge chain reaches it from the entry
            raise AnalysisRejection(
                "analysis explored this region speculatively and the stabilized facts leave it unreachable; "
                "the result cannot be trusted",
                self._block_origin(result.unit, unreached[0]),
            )

    def _check_branch_settled(self, result: ResidualUnit, block_id: BlockId) -> None:
        """
        Companion of ``_check_reachability_settled``, run on the recorded fact of the branch's condition.

        The condition is always a write-once ``PyTruth`` destination, so its recorded binding fact IS the one
        ``_resolve_terminator`` branched on -- which is why this needs no replayed environment.

        Deliberately unconditional. Two narrowings were tried, each meant to spare kernels whose speculated arm
        looked harmless, and each reintroduced a silent miscompile: testing only arms that store misses an inert
        arm that poisons the merge phi, keeping a DOWNSTREAM guard residual so ITS store does the promoting, and
        scoping that test to the arm's exclusive region silently disables it altogether inside a loop, where the
        back-edge puts the dead arm within the live arm's reach. Judging which speculated arm is harmless needs
        the very reachability this gate exists because the analyzer got wrong, so it does not try.
        """
        terminator = result.unit.blocks[block_id].terminator
        if not isinstance(terminator, Branch) or terminator.then_target == terminator.else_target:
            return
        # Deliberately NOT `.get()`: a miss would leave `settled` None and skip the check silently, and the two
        # narrowings recorded above each reintroduced a silent miscompile by sparing an arm. The premise is that
        # a branch condition is a write-once `PyTruth` in this very block, so a miss is a broken invariant, not
        # a case to tolerate.
        assert terminator.cond in result.binding_facts, f"branch condition {terminator.cond} has no fact"
        truth = result.binding_facts[terminator.cond]
        settled = static_truth(truth.value) if isinstance(truth, Known) else None
        if settled is None:
            return
        dead = terminator.else_target if settled else terminator.then_target
        if (block_id, dead) in result.executable_edges:
            raise AnalysisRejection(
                "analysis explored the branch arm that the stabilized facts prove unreachable, so the emitted "
                "logic would not match this source; simplify the condition or the value feeding it",
                terminator.origin,
            )

    def _call_plan(self, call: PyCall, env: _Env) -> CallPlan:
        # Optimistic SCCP may reclassify a cast across revisits (int(y) is an identity while y is still integer and a
        # conversion once the other edge promotes it), so a cast plan deliberately carries no same-kind/cross-kind
        # split: emission decides from the FINAL facts, which only stabilized rounds produce.
        lowering = self._call_lowering[id(call)]
        if lowering is not CallLowering.INTRINSIC:
            return CallPlan(lowering)
        from .._lib import Intrinsic, resolve

        callee_fact = env.get(Local(call.callee))
        assert isinstance(callee_fact, Reference)
        match = resolve(callee_fact.obj)
        assert isinstance(match, Intrinsic)
        return CallPlan(CallLowering.INTRINSIC, match)

    def _component_provenance(self) -> dict[int, tuple[str, ...]]:
        # Canonical member path per component: the shortest path from a root over the recorded sub-object edges, ties
        # broken lexicographically. Bellman-Ford-style relaxation to a fixpoint, so the result is independent of both
        # edge-discovery order and set-iteration order (paths only ever shrink, so it converges).
        result: dict[int, tuple[str, ...]] = dict(self._roots)
        changed = True
        while changed:
            changed = False
            for parent_id, name, child_id in self._component_edges:
                if parent_id in result:
                    candidate = result[parent_id] + (name,)
                    existing = result.get(child_id)
                    if existing is None or (len(candidate), candidate) < (len(existing), existing):
                        result[child_id] = candidate
                        changed = True
        return result

    def _template(self, fn: object) -> FunctionUnit:
        key: tuple[int, int | None]
        if isinstance(fn, types.MethodType):
            key = (id(fn.__func__), id(fn.__self__))
            anchor: object = fn.__func__
        else:
            key = (id(fn), None)
            anchor = fn
        cached = self._templates.get(key)
        if cached is not None and cached[0] is anchor:
            return cached[1]
        template = build_unit(fn)
        self._templates[key] = (anchor, template)
        return template

    # The working graph is rebuilt for every outer state round; blocks are private copies.

    def analyze(self, param_facts: dict[str, Fact] | None = None) -> ResidualUnit:
        unit = self._instantiate_root()
        origin = (Origin(unit.name, 0, 0, unit.file),)
        self._block_in = {unit.entry: _Env()}
        block_in = self._block_in
        entry_env = block_in[unit.entry]
        for param in unit.params:
            contract = unit.param_contracts.get(param.name)
            default: Fact
            if isinstance(contract, (ArrayParameter, RecordParameter)):
                port_layout, port_kinds = _contract_structure(contract)
                assert port_layout is not None
                default = AggregateFact(port_layout, tuple(Residual(kind) for kind in port_kinds))
            else:
                assert contract is None or isinstance(contract, ScalarParameter)
                default = Residual(contract.kind if contract is not None else SemType.FLOAT)
            fact = (param_facts or {}).get(param.name, default)
            entry_env.set(Local(param), fact)
        if unit.bound_self is not None and unit.params:
            entry_env.set(Local(unit.params[0]), Reference(unit.bound_self))
            self._roots = {id(unit.bound_self): ()}  # the root component anchors the member-path tree
        else:
            self._roots = {}
        for param in unit.params:
            seed = schema_of_fact(entry_env.get(Local(param)))
            if seed is not None:  # a root scalar parameter's annotation contract establishes its schema
                entry_env.schemas[Local(param)] = seed
        self._component_edges = set()
        self._executable_edges = set()
        self._executable_blocks = set()
        executable_blocks = self._executable_blocks

        def edge_default(place: Place) -> Fact:
            if isinstance(place, StateLeaf):
                if place in self._state_livein:
                    return self._state_livein[place]
                try:
                    return self._snapshot_leaf(place)
                except AttributeError:
                    return _UNBOUND
            return _UNBOUND

        worklist: list[BlockId] = [unit.entry]
        visits = 0
        while worklist:
            visits += 1
            if visits > _MAX_VISITS:
                raise AnalysisRejection("analysis fuel exhausted", origin)
            block_id = worklist.pop()
            if block_id not in block_in:
                continue  # orphaned by a graft that retracted its only in-edge; a live edge re-enqueues it
            executable_blocks.add(block_id)
            env = block_in[block_id].copy()
            block = unit.blocks[block_id]
            index = 0
            while index < len(block.ops):
                op = block.ops[index]
                try:
                    expanded = self._transfer(unit, block, index, op, env)
                except LocatedRejection as error:
                    # A rejection downstream of a pending store violation is (potentially) provoked by the fact
                    # the violating store carried -- the library sibling refusal alike, since registry matching
                    # rejects on operand kinds too (callee builds rewrap at the call site and emission runs after
                    # analysis, so no other located kind can arise mid-transfer) -- so it defers and the round
                    # runs on to stabilization, where the resolution walk reports the causal store instead. The
                    # op's destination stays unbound; anything it feeds defers the same way. A clean revisit
                    # clears the entry, so a lingering one was re-derived on the op's stable facts.
                    assert isinstance(error, (AnalysisRejection, LibraryAnalysisRejection))
                    if not self._violations_pending():
                        raise
                    self._transfer_deferrals[id(op)] = error
                    index += 1
                    continue
                if self._transfer_deferrals:
                    self._transfer_deferrals.pop(id(op), None)
                if expanded:
                    continue  # the graph changed under us: re-run this op slot (now a different op)
                if isinstance(op, PyCall):
                    self._visit_call_plans[op.dst] = self._call_plan(op, env)
                dst = op_dst(op)
                if dst is not None:
                    self._visit_facts[dst] = env.get(Local(dst))
                index += 1
            try:
                successors = self._resolve_terminator(unit, block, env)
            except LocatedRejection as error:
                # The terminator counterpart of the op-level deferral (an unrollable iterable fed by a deferred
                # op, say): the successors stay unexplored and the walk ranks over the graph reached so far.
                assert isinstance(error, (AnalysisRejection, LibraryAnalysisRejection))
                if not self._violations_pending():
                    raise
                self._transfer_deferrals[id(block.terminator)] = error
                continue
            if self._transfer_deferrals:
                self._transfer_deferrals.pop(id(block.terminator), None)
            assert block.terminator is not None
            join_origin = block.terminator.origin
            for successor in successors:
                edge = (block.id, successor)
                target_env = block_in.get(successor)
                if target_env is None:
                    block_in[successor] = env.copy()
                    self._executable_edges.add(edge)
                    worklist.append(successor)
                    continue
                first_traversal = edge not in self._executable_edges
                self._executable_edges.add(edge)
                try:
                    changed = target_env.join_with(env, join_origin, edge_default)
                except LocatedRejection as error:
                    # The join counterpart: an irreconcilable merge of a violating store's carried fact defers
                    # like any provoked rejection, keyed on the terminator whose edges join here (mutually
                    # exclusive with a terminator deferral within one visit, and the clean pop above re-arms the
                    # slot; setdefault keeps the first failing edge, then-arm first). The per-place joins that
                    # did succeed were applied with their change tracking lost, so the successor re-enqueues
                    # unconditionally; the failing place holds its old fact, which cannot oscillate.
                    assert isinstance(error, (AnalysisRejection, LibraryAnalysisRejection))
                    if not self._violations_pending():
                        raise
                    self._transfer_deferrals.setdefault(id(block.terminator), error)
                    worklist.append(successor)
                    continue
                if changed or (first_traversal and successor not in executable_blocks):
                    worklist.append(successor)
        return ResidualUnit(unit, block_in, executable_blocks, self._executable_edges)

    # ------------------------------------ instantiation and grafting ------------------------------------

    def _instantiate_root(self) -> FunctionUnit:
        blocks = {
            block_id: Block(block_id, list(block.ops), block.terminator)
            for block_id, block in self._root_template.blocks.items()
        }
        return replace(self._root_template, blocks=blocks)

    def _fresh_block_id(self) -> BlockId:
        self._block_serial += 1
        return BlockId(self._block_serial)

    # ------------------------------------ transfer functions ------------------------------------

    def _transfer(
        self,
        unit: FunctionUnit,
        block: Block,
        index: int,
        op: Op,
        env: _Env,
    ) -> bool:
        result: Fact
        match op:
            case LoadConst(dst=dst, value=value):
                env.set(Local(dst), normalize_static(value))
            case LoadRef(dst=dst, obj=obj_referent):
                env.set(Local(dst), Reference(obj_referent))
            case LoadPlace(dst=dst, place=place):
                fact = env.get(place)
                if isinstance(fact, (Unbound, MaybeUnbound)) and isinstance(place, Local):
                    raise AnalysisRejection(
                        f"local '{place.binding.name}' may be unbound here (Python would raise)", op.origin
                    )
                env.set(Local(dst), fact)
            case StorePlace(place=place, src=src):
                stored = env.get(Local(src))
                if op.role is StoreRole.SOURCE and isinstance(place, Local):
                    # The verdict belongs to the post-stabilization walk, which re-derives it from the stored
                    # fact recorded here (stable at the last visit); the mid-flight verdict record only gates
                    # the downstream-rejection deferral and the round-boundary bridge reconcile. An Unbound
                    # value is no evidence either way, so it leaves the recorded verdict untouched.
                    self._store_facts[id(op)] = stored
                    bound = not isinstance(stored, (Unbound, MaybeUnbound))
                    schema, stored, message = conform_local_store(env.schemas.get(place), place.binding.name, stored)
                    if schema is not None:
                        env.schemas[place] = schema
                    if bound:
                        self._record_store_verdict(op, message)
                    else:
                        self._unbound_store_origins.add(op.origin)
                env.set(place, stored)
            case UnbindPlace(place=place, checked=checked):
                if checked and isinstance(env.get(place), (Unbound, MaybeUnbound)):
                    raise AnalysisRejection(f"'{place}' may be unbound at this del (Python would raise)", op.origin)
                if not checked:  # a compiler scope reset opens a fresh per-execution schema; user del does not
                    env.schemas.pop(place, None)
                env.set(place, _UNBOUND)
            case PyBin(dst=dst, op=bin_op, lhs=lhs, rhs=rhs):
                lhs_fact, rhs_fact = env.get(Local(lhs)), env.get(Local(rhs))
                if (
                    bin_op is BinOp.MATMUL
                    and not isinstance(lhs_fact, AggregateFact)
                    and not isinstance(rhs_fact, AggregateFact)
                ):
                    raise AnalysisRejection("@ is not defined for scalars", op.origin)
                if op.inplace and _is_list_fact(lhs_fact):
                    raise AnalysisRejection(
                        "in-place list mutation is not supported (aliases would observe it); rebind instead",
                        op.origin,
                    )
                if op.inplace and _is_array_fact(lhs_fact):
                    raise AnalysisRejection(
                        "in-place array mutation is not supported (aliases would observe it); rebind instead",
                        op.origin,
                    )
                if bin_op is BinOp.MATMUL:  # both scalar was rejected above; any aggregate side lands here
                    if not (_is_array_fact(lhs_fact) and _is_array_fact(rhs_fact)):
                        raise AnalysisRejection(
                            "the matrix product requires array operands on both sides; a scalar, list, or "
                            "tuple does not acquire matrix semantics (wrap it in np.array(...))",
                            op.origin,
                        )
                    # ``@`` IS the library function: rewrite onto the spelled np.matmul call so the operator
                    # and the call cannot drift apart (they inline the same registry stub).
                    matmul_callee = self._fresh_temp()
                    block.ops[index : index + 1] = [
                        LoadRef(matmul_callee, np.matmul, op.origin),
                        PyCall(dst, matmul_callee, (lhs, rhs), (), op.origin),
                    ]
                    return True
                concat = _concat_seqs(bin_op, lhs_fact, rhs_fact)
                if concat is not None:
                    env.set(Local(dst), concat)
                elif _is_array_fact(lhs_fact) or _is_array_fact(rhs_fact):
                    env.set(Local(dst), self._elementwise_binary(bin_op, lhs_fact, rhs_fact, op.origin))
                elif isinstance(lhs_fact, AggregateFact) or isinstance(rhs_fact, AggregateFact):
                    raise AnalysisRejection("arithmetic on an aggregate value", op.origin)
                elif bin_op in _BITWISE_OPS:
                    env.set(Local(dst), self._fold_bitwise(bin_op, lhs_fact, rhs_fact, op.origin))
                else:
                    # Python's / always yields float. A residual power stays integer only for an integer base with a
                    # compile-time nonnegative integer exponent (a multiply chain); any other exponent may go float
                    # (negative exponents reciprocate), so the static common type is float.
                    integer_power = (
                        bin_op is BinOp.POW
                        and _numeric_sem(lhs_fact) is SemType.INT
                        and isinstance(rhs_fact, Known)
                        and isinstance(rhs_fact.value, (MetaInt, NpInt))
                        and int(rhs_fact.value.value) >= 0
                    )
                    env.set(
                        Local(dst),
                        self._fold_binary(
                            lambda a, b: static_binop(bin_op, a, b),
                            lhs_fact,
                            rhs_fact,
                            op.origin,
                            promotes_to_float=bin_op is BinOp.DIV or (bin_op is BinOp.POW and not integer_power),
                        ),
                    )
            case PyUn(dst=dst, op=un_op, operand=operand):
                operand_fact = env.get(Local(operand))
                if (isinstance(operand_fact, Known) and isinstance(operand_fact.value, (StaticBool, NpBool))) or (
                    isinstance(operand_fact, Residual) and operand_fact.type is SemType.BOOL
                ):
                    raise AnalysisRejection("arithmetic on a bool requires an explicit conversion", op.origin)
                if isinstance(operand_fact, AggregateFact) and isinstance(operand_fact.layout, ArrayLayout):
                    env.set(Local(dst), self._elementwise_unary(un_op, operand_fact, op.origin))
                elif isinstance(operand_fact, Known):
                    folded = static_unop(un_op, operand_fact.value)
                    env.set(
                        Local(dst),
                        Known(folded) if folded is not None else self._residual_of(operand_fact, op.origin),
                    )
                else:
                    env.set(Local(dst), self._residual_of(operand_fact, op.origin))
            case PyCompare(dst=dst, op=rel, lhs=lhs, rhs=rhs):
                lhs_fact, rhs_fact = env.get(Local(lhs)), env.get(Local(rhs))
                if _is_array_fact(lhs_fact) or _is_array_fact(rhs_fact):
                    # Trimmed (scope ruling T2, utility grounds): a mask's only onward use here is scalar extraction —
                    # no boolean indexing, any/all, or array truth — and the machinery lowered integer arrays in
                    # float (A2).
                    raise AnalysisRejection(
                        "elementwise array comparison is not supported; compare the elements explicitly",
                        op.origin,
                    )
                else:
                    compared = self._fold_binary(
                        lambda a, b: static_compare(rel, a, b),
                        lhs_fact,
                        rhs_fact,
                        op.origin,
                        default=SemType.BOOL,
                    )
                    if isinstance(compared, Residual):
                        # A residual comparison reaches the datapath, where bool never converts implicitly;
                        # a fully static one already folded Python-exactly above.
                        sems = {_scalar_sem(lhs_fact), _scalar_sem(rhs_fact)}
                        if SemType.BOOL in sems and len(sems) > 1:
                            raise AnalysisRejection(
                                "a comparison mixes a boolean and a non-boolean without a cast", op.origin
                            )
                        if sems == {SemType.BOOL} and rel not in (RelationalOp.EQ, RelationalOp.NE):
                            raise AnalysisRejection("only == and != are defined between boolean values", op.origin)
                    env.set(Local(dst), compared)
            case PyNot(dst=dst, operand=operand):
                truth = self._truth_fact(env.get(Local(operand)), op.origin)
                if isinstance(truth, Known):
                    result = Known(StaticBool(not as_python(truth.value)))
                else:
                    result = Residual(SemType.BOOL)
                env.set(Local(dst), result)
            case PyTruth(dst=dst, operand=operand):
                result = self._truth_fact(env.get(Local(operand)), op.origin)
                env.set(Local(dst), result)
            case PySelect(dst=dst, mode=mode, cond=cond, lhs=lhs, rhs=rhs):
                condition = env.get(Local(cond))
                if isinstance(condition, Known):
                    taken = as_python(condition.value)
                    assert isinstance(taken, bool)
                    if mode is SelectMode.AND:
                        chosen = rhs if taken else lhs
                    else:
                        chosen = lhs if taken else rhs
                    result = env.get(Local(chosen))
                else:
                    lhs_fact, rhs_fact = env.get(Local(lhs)), env.get(Local(rhs))
                    # The merge is evaluated unconditionally for its kind-check: a non-boolean operand reached before an
                    # absorbing constant (``float(x) or True``) is still irreconcilable and must reject, never
                    # fold away.
                    merged = join_facts(lhs_fact, rhs_fact, op.origin)
                    # A boolean identity holds even with a runtime condition: ``A or True`` is always True and
                    # ``A and False`` always False (the arm chosen when the condition is false is a decisive constant),
                    # so the connective folds and a branch that consumes it has a statically dead arm carrying no state.
                    rhs_const = as_python(rhs_fact.value) if isinstance(rhs_fact, Known) else None
                    if (mode is SelectMode.OR and rhs_const is True) or (mode is SelectMode.AND and rhs_const is False):
                        result = rhs_fact
                    else:
                        result = merged
                env.set(Local(dst), result)
            case BuildTuple(dst=dst, items=items) | BuildList(dst=dst, items=items):
                children = []
                for item in items:
                    fact = env.get(Local(item))
                    if not isinstance(fact, (Known, Residual, Reference, AggregateFact)):
                        raise AnalysisRejection("an unbound value flows into an aggregate literal", op.origin)
                    children.append(fact)
                env.set(Local(dst), aggregate_of(tuple(children), is_list=isinstance(op, BuildList)))
            case PyLen(dst=dst, obj=obj):
                obj_fact = env.get(Local(obj))
                if isinstance(obj_fact, AggregateFact):
                    length = admit(outer_arity(obj_fact.layout))
                    assert length is not None
                    result = Known(length)
                elif isinstance(obj_fact, Reference):
                    # len()/unpacking of a live object would run its __len__ outside the state machinery.
                    raise AnalysisRejection("len() of an object is not supported", op.origin)
                elif isinstance(obj_fact, Known):
                    concrete = as_python(obj_fact.value)
                    try:
                        length = admit(len(concrete))  # type: ignore[arg-type]
                    except TypeError as error:
                        raise AnalysisRejection(str(error), op.origin) from None
                    except OverflowError:
                        raise AnalysisRejection("length of an oversized range is not supported", op.origin) from None
                    assert length is not None
                    result = Known(length)
                else:
                    raise AnalysisRejection("length of a runtime value", op.origin)
                env.set(Local(dst), result)
            case PySubscript(dst=dst, obj=obj, index=idx):
                result = self._subscript(op, env.get(Local(obj)), env.get(Local(idx)))
                env.set(Local(dst), result)
            case PyAttr(dst=dst, obj=obj, name=name):
                receiver_fact = env.get(Local(obj))
                attribute_site = _AttrSite(op, block, index, env, receiver_fact)
                for attribute_row in _ATTRIBUTE_INTERCEPT_ROWS:
                    if attribute_row.selects(attribute_site):
                        return attribute_row.apply(self, attribute_site)
                attr = self._attribute(env, receiver_fact, name, op.origin)
                if isinstance(attr, _PropertyRead):
                    # Desugar the property read into a bound zero-argument call and re-run: the generic call-expansion
                    # machinery then inlines the getter, remaps its return, and threads state through unchanged.
                    callee = BindingId(f"%p{self._binding_serial}", self._binding_serial)
                    self._binding_serial += 1
                    block.ops[index : index + 1] = [
                        LoadRef(callee, attr.getter, op.origin),
                        PyCall(dst, callee, (), (), op.origin),
                    ]
                    return True
                if isinstance(receiver_fact, AggregateFact) and isinstance(receiver_fact.layout, RecordLayout):
                    names = [field for field, _ in receiver_fact.layout.fields]
                    _, start, stop = child_slice(receiver_fact.layout, names.index(name))
                    self._route_evidence[id(op)] = SourceSelection(tuple(range(start, stop)))
                env.set(Local(dst), attr)
            case PyStoreAttr(obj=obj, name=name, src=src):
                obj_fact = env.get(Local(obj))
                if not isinstance(obj_fact, Reference):
                    raise AnalysisRejection("attribute store on a non-component value", op.origin)
                if isinstance(obj_fact.obj, (types.ModuleType, type)):
                    # A module/class is a compile-time namespace, not runtime state: mutating it would make later
                    # reads (which snapshot the live object) disagree with the store. Reject, as production does.
                    raise AnalysisRejection("assignment to a module or class attribute is not supported", op.origin)
                _reject_attribute_hooks(type(obj_fact.obj), name, op.origin)
                src_fact = env.get(Local(src))
                if isinstance(src_fact, Reference):
                    # Storing a component/sub-object into an attribute would change the component topology per
                    # transaction (a slot's owner is fixed at the initial snapshot); reject it at the store, located.
                    raise AnalysisRejection(
                        f"component member '{name}' cannot be rebound; component topology is fixed", op.origin
                    )
                leaf = StateLeaf(obj_fact.obj, (name,))
                recorded = self._store_origins.get(leaf)
                # Deliberately source_position, not the total origin_order used for set-ordered selections:
                # ties here are already broken by the deterministic order stores are transferred in, which is
                # execution order, and that attributes better than the lexically-first filename would.
                if recorded is None or source_position(op.origin) < source_position(recorded):
                    self._store_origins[leaf] = op.origin
                self._discovered_stores.add((block.id, leaf))
                here = self._discovered_store_origins.get((block.id, leaf))
                if here is None or source_position(op.origin) < source_position(here):
                    self._discovered_store_origins[(block.id, leaf)] = op.origin
                self._store_leaf_of_op[id(op)] = leaf
                # The reset fixes the slot schema; a violating store carries a fixpoint-stable fact onward and
                # the recorded verdict reports after stabilization, at this store. An Unbound value is no
                # evidence either way, so it leaves the recorded verdict untouched.
                conformed, violation = conform_state_store(name, self._state_reset_fact(leaf), src_fact)
                if not isinstance(src_fact, (Unbound, MaybeUnbound)):
                    self._record_store_verdict(op, violation)
                else:
                    self._unbound_store_origins.add(op.origin)
                env.set(leaf, conformed)
            case PyCall(dst=dst, callee=callee):
                return self._expand_call(unit, block, index, op, env)
        return False

    def _residual_of(self, fact: Fact, origin: OriginStack) -> Fact:
        match fact:
            case Known(value=value):
                sem = _residual_type(value)
                if sem is None:
                    raise AnalysisRejection("a non-numeric value reaches a runtime operation", origin)
                return Residual(SemType.INT if sem is SemType.BOOL else sem)  # Python: unary on bool yields int
            case Residual(type=SemType.BOOL):
                return Residual(SemType.INT)
            case Residual():
                return fact  # a unary negation/plus preserves the operand's numeric kind
            case _:
                raise AnalysisRejection("a runtime operation reads an aggregate or unbound value", origin)

    def _fold_binary(
        self,
        fold: Callable[[StaticValue, StaticValue], StaticValue | None],
        lhs: Fact,
        rhs: Fact,
        origin: OriginStack,
        default: SemType | None = None,
        promotes_to_float: bool = False,
    ) -> Fact:
        if isinstance(lhs, Known) and isinstance(rhs, Known):
            folded = fold(lhs.value, rhs.value)
            if folded is not None:
                return Known(folded)
        operand_types = [self._operand_type(fact, origin) for fact in (lhs, rhs)]
        if default is None and SemType.BOOL in operand_types:
            raise AnalysisRejection("arithmetic on a bool requires an explicit conversion", origin)
        if default is not None:
            return Residual(default)
        if promotes_to_float or SemType.FLOAT in operand_types:
            return Residual(SemType.FLOAT)
        return Residual(SemType.INT)

    def _elementwise_binary(self, bin_op: BinOp, lhs: Fact, rhs: Fact, origin: OriginStack) -> Fact:
        """
        Elementwise ``+ - * /`` with at least one array operand: the other side is a same-shape array (leaves pair
        in canonical order) or a numeric scalar (broadcast). Each leaf pair takes the SCALAR fold rule, so a fully
        static pair folds through ``static_binop`` on the domain's own numpy-kinded leaf values -- element-wise
        numpy semantics (promotion, wraparound, errstate deferrals) hold per leaf by construction, and general
        broadcasting stays a located rejection rather than a silent alignment. Divergence guards mirror what numpy
        applies ARRAY-WIDE before touching any element (an empty array included): an out-of-range Python-int
        constant is a located rejection where numpy raises OverflowError. A runtime-integer result rejects until
        the integer sprint: the scalar integer datapath saturates where numpy wraps, so lowering it would diverge
        leafwise.
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
        lhs_sem = self._elementwise_side_sem(lhs, origin)
        rhs_sem = self._elementwise_side_sem(rhs, origin)
        if SemType.BOOL in (lhs_sem, rhs_sem):
            raise AnalysisRejection("arithmetic on a bool requires an explicit conversion", origin)
        arrays = [side for side in (lhs, rhs) if isinstance(side, AggregateFact)]
        shapes = [side.layout.shape for side in arrays if isinstance(side.layout, ArrayLayout)]
        if len(set(shapes)) > 1:
            raise AnalysisRejection(
                f"elementwise arithmetic on mismatched shapes {shapes[0]} and {shapes[1]} "
                "(only a scalar broadcasts)",
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
            # The scalar integer datapath saturates (contained at MIR) where numpy int64 wraps; lowering a
            # runtime-integer array op leafwise onto it would silently diverge the moment integers lower.
            raise AnalysisRejection(
                "runtime integer array arithmetic is not lowerable yet; cast to float first", origin
            )

        def pair(ordinal: int) -> Fact:
            left = lhs.leaves[ordinal] if isinstance(lhs, AggregateFact) else lhs
            right = rhs.leaves[ordinal] if isinstance(rhs, AggregateFact) else rhs
            return self._fold_binary(
                lambda a, b: static_binop(bin_op, a, b), left, right, origin, promotes_to_float=promotes
            )

        dtype = ArrayDType.FLOAT if result_sem is SemType.FLOAT else ArrayDType.INT
        leaves: list[AtomicFact] = []
        for ordinal in range(leaf_count(arrays[0].layout)):
            leaf = pair(ordinal)
            assert isinstance(leaf, (Known, Residual)), leaf
            assert (leaf == Residual(result_sem)) or (
                isinstance(leaf, Known) and _residual_type(leaf.value) is result_sem
            ), "an elementwise leaf diverged from the result dtype"
            leaves.append(leaf)
        return AggregateFact(ArrayLayout(shapes[0], dtype), tuple(leaves))

    def _static_reshape_target(
        self, shape_args: tuple[BindingId, ...], env: _Env, origin: OriginStack
    ) -> tuple[int, ...]:
        """The reshape target as static dimensions: bare integers or one static tuple; -1 inference rejects."""
        if not shape_args:
            raise AnalysisRejection("reshape() requires a shape argument", origin)
        facts = [env.get(Local(arg)) for arg in shape_args]
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
        dimensions: list[int] = []
        for item in items:
            if not isinstance(item, (MetaInt, NpInt)):
                raise AnalysisRejection("reshape() requires integer dimensions", origin)
            dimension = int(item.value)
            if dimension < 0:
                raise AnalysisRejection(
                    "a -1 (inferred) reshape dimension is not supported; spell the shape explicitly", origin
                )
            dimensions.append(dimension)
        if not dimensions:
            raise AnalysisRejection("a 0-dimensional array is not supported; use the scalar directly", origin)
        return tuple(dimensions)

    def _elementwise_unary(self, un_op: UnOp, operand: AggregateFact, origin: OriginStack) -> Fact:
        assert isinstance(operand.layout, ArrayLayout)
        if operand.layout.dtype is ArrayDType.BOOL:
            # numpy itself refuses unary +/- on a boolean array (TypeError pointing at ~/logical ops).
            raise AnalysisRejection("arithmetic on a bool requires an explicit conversion", origin)
        if operand.layout.dtype is ArrayDType.INT and any(isinstance(leaf, Residual) for leaf in operand.leaves):
            raise AnalysisRejection(
                "runtime integer array arithmetic is not lowerable yet; cast to float first", origin
            )
        leaves: list[AtomicFact] = []
        for leaf in operand.leaves:
            folded = static_unop(un_op, leaf.value) if isinstance(leaf, Known) else None
            if folded is not None:
                leaves.append(Known(folded))
            else:
                residual = self._residual_of(leaf, origin)
                assert isinstance(residual, Residual), residual
                leaves.append(residual)
        return AggregateFact(operand.layout, tuple(leaves))

    def _array_factory(self, source: AggregateFact, origin: OriginStack, force_float: bool = False) -> AggregateFact:
        """
        np.array/asarray/asanyarray over a residual-carrying aggregate: a relayout of the SAME leaves onto the
        rectangular shape the nesting yields, with dtype discovery restricted to the proven subset -- any float
        evidence promotes to FLOAT64 (integer leaves coerce: Knowns re-kind to np.float64 exactly as numpy
        extraction would yield them, residual integers pick up a runtime conversion at emission), an all-boolean
        argument builds a BOOL array, and empty array children contribute their dtype as evidence. Outside the
        subset numpy behaves in ways the domain cannot carry, so the forms reject where numpy would surprise: a
        Python-int leaf beyond signed 64 bits (numpy builds an object array, or silently promotes the uint64
        range to float64), a bool/numeric mix (numpy widens the bool), and a runtime-integer result (the integer
        datapath saturates where numpy wraps).
        """
        shape = _rectangular_shape(source.layout)
        if shape is None:
            raise AnalysisRejection("an array literal must be rectangular (numpy raises on ragged nesting)", origin)
        sems: set[SemType | None] = set()
        for leaf in source.leaves:
            assert not isinstance(leaf, Reference), "a reference leaf survived the admission walk"
            sems.add(leaf.type if isinstance(leaf, Residual) else _residual_type(leaf.value))
        sems |= {
            {ArrayDType.BOOL: SemType.BOOL, ArrayDType.INT: SemType.INT, ArrayDType.FLOAT: SemType.FLOAT}[dtype]
            for dtype in _layout_dtypes(source.layout)
        }
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
        if dtype is ArrayDType.INT:
            # Only residual integers can reach here (a fully static argument took the concrete fold), and the
            # scalar integer datapath saturates where numpy int64 wraps.
            raise AnalysisRejection(
                "runtime integer array construction is not lowerable yet; cast to float first", origin
            )
        leaves: list[AtomicFact] = []
        for leaf in source.leaves:
            if isinstance(leaf, Known):
                leaves.append(numpy_kinded(leaf, dtype))
            else:
                assert isinstance(leaf, Residual)
                leaves.append(Residual(SemType.FLOAT) if dtype is ArrayDType.FLOAT else leaf)
        return AggregateFact(ArrayLayout(shape, dtype), tuple(leaves))

    def _elementwise_side_sem(self, side: Fact, origin: OriginStack) -> SemType:
        if isinstance(side, AggregateFact):
            assert isinstance(side.layout, ArrayLayout)
            match side.layout.dtype:
                case ArrayDType.BOOL:
                    return SemType.BOOL
                case ArrayDType.INT:
                    return SemType.INT
                case ArrayDType.FLOAT:
                    return SemType.FLOAT
        return self._operand_type(side, origin)

    def _fold_bitwise(self, bin_op: BinOp, lhs: Fact, rhs: Fact, origin: OriginStack) -> Fact:
        # Bit-true operators. ``&``/``|``/``^`` on two booleans is a boolean (logical) result; every other admitted form
        # is two integers. A float operand, a boolean shift, and mixed bool/int all refuse -- Python's bool-as-int
        # promotion is not modelled in the datapath, so an explicit cast is required. A compile-time-known negative
        # shift count refuses (Python raises); a runtime count is the hardware's documented reverse-shift deviation. A
        # fully-static form folds Python-exact via ``static_binop``. Operand kinds are validated before any diagnostic.
        is_shift = bin_op in (BinOp.LSHIFT, BinOp.RSHIFT)
        ltype, rtype = self._operand_type(lhs, origin), self._operand_type(rhs, origin)
        if SemType.FLOAT in (ltype, rtype):
            raise AnalysisRejection(f"bitwise/shift operator {bin_op.value} requires integer operands", origin)
        if is_shift and isinstance(rhs, Known) and isinstance(rhs.value, (MetaInt, NpInt)) and int(rhs.value.value) < 0:
            raise AnalysisRejection(
                f"a negative shift count ({int(rhs.value.value)}) is rejected at compile time", origin
            )
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

    def _operand_type(self, fact: Fact, origin: OriginStack) -> SemType:
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

    def _truth_fact(self, fact: Fact, origin: OriginStack) -> Fact:
        match fact:
            case Known(value=value):
                truth = static_truth(value)
                if truth is None and _residual_type(value) is None:
                    raise AnalysisRejection("the truth value of this object is not defined here", origin)
                return Known(StaticBool(truth)) if truth is not None else Residual(SemType.BOOL)
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

    def _subscript(self, op: PySubscript, obj: Fact, index: Fact) -> Fact:
        import operator

        origin = op.origin
        if isinstance(index, AggregateFact):
            if contains_record(index.layout):  # a record anywhere in the key would run __index__ on a rebuild
                raise AnalysisRejection("a record subscript index is not supported", origin)
            key = materialize_static(index)  # a static tuple key (m[1, 0]); runtime keys reject below
            if key is not None:
                index = Known(key)
        if isinstance(index, Known) and isinstance(index.value, StaticRecord):
            # Rejected for ANY subscriptable (a range or string included): the key would resolve through a user
            # __index__ running on the reconstruction, whose semantics the compiler cannot vouch for.
            raise AnalysisRejection("a record subscript index is not supported", origin)
        if isinstance(index, Reference):
            # A referenced key would resolve through the LIVE object's __index__ at compile time (repeatedly:
            # once per fixpoint visit of its block), reading reset-time state the kernel's
            # writes never touch. The state machinery is the honest path: index with int(self.attr).
            raise AnalysisRejection("an object subscript index is not supported", origin)
        if isinstance(obj, AggregateFact) and isinstance(index, Known):
            if isinstance(obj.layout, RecordLayout):
                raise AnalysisRejection("a record is not subscriptable; access its fields by name", origin)
            if isinstance(index.value, (StaticBool, NpBool)) and (
                isinstance(obj.layout, ArrayLayout)
                or (isinstance(obj.layout, StructuralLayout) and ContainerFlavor.ARRAY in obj.layout.flavors)
            ):
                # numpy boolean indexing selects by mask (and prepends an axis for a scalar bool); Python's
                # bool-as-int semantics apply only to tuples/lists, so guessing here would miscompile.
                raise AnalysisRejection("a boolean index into an array is not supported; use an integer", origin)
            if isinstance(index.value, NpBool):
                # numpy 2 removed np.bool_.__index__, so Python itself refuses it as a sequence index; only the
                # plain Python bool keeps bool-as-int indexing.
                raise AnalysisRejection(
                    "an np.bool_ subscript index is a TypeError in Python; use a plain bool", origin
                )
            if isinstance(index.value, StaticSlice) and isinstance(obj.layout, (TupleLayout, ListLayout)):
                # A slice of a positional container is a WINDOW operation over the same children -- runtime
                # leaves included, exactly like conversion and projection -- so nothing materializes and
                # nothing crosses. Records still refuse (their consumptions are field access and integer
                # projection); a structural flavor cannot truthfully pick a result container, so it keeps the
                # concrete-fallback rejection below.
                if contains_record(obj.layout):
                    raise AnalysisRejection(
                        "slicing or multi-axis indexing of a record-carrying sequence is not supported", origin
                    )
                window = as_python(index.value)
                assert isinstance(window, slice)
                try:
                    selected = range(*window.indices(outer_arity(obj.layout)))
                except ValueError as error:  # a zero step, exactly as Python raises
                    raise AnalysisRejection(f"subscript fails here: {error}", origin) from None
                ordinals: list[int] = []
                for position in selected:
                    _, start, stop = child_slice(obj.layout, position)
                    ordinals.extend(range(start, stop))
                self._route_evidence[id(op)] = SourceSelection(tuple(ordinals))
                children = tuple(obj.child(position) for position in selected)
                return aggregate_of(children, is_list=isinstance(obj.layout, ListLayout))
            if isinstance(obj.layout, ArrayLayout) and (
                isinstance(index.value, StaticSlice) or (isinstance(index.value, StaticSeq) and not index.value.is_list)
            ):
                # Only a TUPLE key is basic multi-axis indexing; a LIST key is numpy ADVANCED (fancy) indexing
                # with entirely different result geometry, so it falls through -- an all-Known object folds it
                # concretely through numpy itself, a runtime one keeps the located rejection.
                return self._array_subscript(op, obj, index.value)
            try:
                position = operator.index(as_python(index.value))  # type: ignore[arg-type]  # np ints qualify
            except TypeError:
                # A non-integer static key (a tuple key ``m[1, 0]``, a slice) applies concretely to an all-Known
                # aggregate; on a runtime-leaf aggregate it awaits the slicing/multi-axis stages. A record
                # anywhere in the OBJECT refuses first: the concrete fallback rebuilds real instances (a __del__
                # would fire at compile time), and records never cross into host evaluation -- integer projection
                # and field access are their consumptions.
                if contains_record(obj.layout):
                    raise AnalysisRejection(
                        "slicing or multi-axis indexing of a record-carrying sequence is not supported", origin
                    ) from None
                concrete = materialize_static(obj)
                if concrete is None:
                    raise AnalysisRejection(
                        "slicing or multi-axis indexing of a runtime aggregate is not supported yet", origin
                    ) from None
                return self._concrete_subscript(concrete, index, origin)
            except Exception as error:  # a raising __index__ (a referenced key's real object): locate, not leak
                raise AnalysisRejection(f"subscript index fails here: {error}", origin) from None
            arity = outer_arity(obj.layout)
            if not -arity <= position < arity:
                raise AnalysisRejection("sequence index out of range", origin)
            selection = position + arity if position < 0 else position
            _, start, stop = child_slice(obj.layout, selection)
            self._route_evidence[id(op)] = SourceSelection(tuple(range(start, stop)))
            return obj.child(selection)
        if isinstance(obj, Reference):
            # A live object's __getitem__ would read reset-time attribute state outside the state machinery.
            raise AnalysisRejection("subscript of an object is not supported", origin)
        if isinstance(obj, Known) and isinstance(index, Known):
            return self._concrete_subscript(obj.value, index, origin)
        if isinstance(obj, (Known, AggregateFact)):
            raise AnalysisRejection("subscript with a runtime index is not supported yet", origin)
        raise AnalysisRejection("subscript of a runtime scalar is not supported", origin)

    def _array_subscript(self, op: PySubscript, obj: AggregateFact, key: "StaticSlice | StaticSeq") -> Fact:
        """
        numpy basic indexing of an array by a STATIC slice or tuple key: a pure leaf SELECTION over leading
        axes (an integer collapses its axis, a slice keeps its window, trailing axes stay whole), recorded as
        the emission plan's source ordinals. Advanced indexing -- a boolean anywhere in the key, an array or
        nested sequence as a key element -- changes numpy's result geometry entirely, so it refuses rather than
        being misread as a positional pick; a non-integer scalar key element is numpy's own TypeError.
        """
        origin = op.origin
        assert isinstance(obj.layout, ArrayLayout)
        shape = obj.layout.shape
        raw = [key] if isinstance(key, StaticSlice) else list(key.items)
        if len(raw) > len(shape):
            raise AnalysisRejection(
                f"too many indices for a {len(shape)}-dimensional array ({len(raw)} were given)", origin
            )
        for item in raw:
            if isinstance(item, (StaticBool, NpBool)):
                raise AnalysisRejection("a boolean index into an array is not supported; use an integer", origin)
        axes: list[list[int]] = []
        kept: list[int] = []
        for axis, item in enumerate(raw):
            dimension = shape[axis]
            if isinstance(item, StaticSlice):
                window = as_python(item)
                assert isinstance(window, slice)
                try:
                    selected = list(range(*window.indices(dimension)))
                except ValueError as error:  # a zero step, exactly as numpy raises
                    raise AnalysisRejection(f"subscript fails here: {error}", origin) from None
                axes.append(selected)
                kept.append(len(selected))
            elif isinstance(item, (MetaInt, NpInt)):
                position = int(item.value)
                if not -dimension <= position < dimension:
                    raise AnalysisRejection(
                        f"array index {position} is out of range for axis {axis} of size {dimension}", origin
                    )
                axes.append([position + dimension if position < 0 else position])
            else:
                raise AnalysisRejection(
                    "an array subscript key must hold integers and slices (advanced indexing is not supported)",
                    origin,
                )
        for dimension in shape[len(raw) :]:  # unindexed trailing axes stay whole
            axes.append(list(range(dimension)))
            kept.append(dimension)
        strides: list[int] = []
        span = 1
        for dimension in reversed(shape):
            strides.append(span)
            span *= dimension
        strides.reverse()
        ordinals: list[int] = []

        def enumerate_coordinates(axis: int, offset: int) -> None:
            if axis == len(shape):
                ordinals.append(offset)
                return
            for position in axes[axis]:
                enumerate_coordinates(axis + 1, offset + position * strides[axis])

        enumerate_coordinates(0, 0)
        self._route_evidence[id(op)] = SourceSelection(tuple(ordinals))
        picked = tuple(obj.leaves[ordinal] for ordinal in ordinals)
        if not kept:  # every axis collapsed by an integer: the element itself, numpy's scalar sort
            assert len(picked) == 1
            return picked[0]
        return AggregateFact(ArrayLayout(tuple(kept), obj.layout.dtype), picked)

    def _concrete_subscript(self, value: StaticValue, index: Known, origin: OriginStack) -> Fact:
        try:
            concrete = as_python(value)[as_python(index.value)]  # type: ignore[index]
        except Exception as error:
            raise AnalysisRejection(f"subscript fails here: {error}", origin) from None
        admitted = admit(concrete)
        if admitted is None:
            return Reference(concrete)
        return normalize_static(admitted)

    def _attribute(self, env: _Env, obj: Fact, name: str, origin: OriginStack) -> "Fact | _PropertyRead":
        if isinstance(obj, AggregateFact):
            if isinstance(obj.layout, RecordLayout):
                names = [field for field, _ in obj.layout.fields]
                if name in names:
                    return obj.child(names.index(name))  # record field projection works on runtime leaves too
                # A non-field attribute (a property, a method) would execute user code on the reconstruction,
                # which is not type-faithful (an enum field rebuilds as its base value).
                raise AnalysisRejection(f"record attribute '{name}' is not supported (only field access)", origin)
            if isinstance(obj.layout, ListLayout):
                raise _list_attribute_rejection(name, origin)
            if isinstance(obj.layout, ArrayLayout) and name in ("ndim", "shape", "size"):
                # Layout-determined metadata: folds identically on runtime leaves, no element consulted, and
                # value-identical to the concrete navigation an all-Known snapshot would take.
                metadata = {
                    "ndim": len(obj.layout.shape),
                    "shape": tuple(obj.layout.shape),
                    "size": leaf_count(obj.layout),
                }[name]
                admitted_metadata = admit(metadata)
                assert admitted_metadata is not None
                return normalize_static(admitted_metadata)
            if isinstance(obj.layout, ArrayLayout) and name not in _ARRAY_ATTRIBUTES:
                # The admitted array is a private C-contiguous SNAPSHOT: identity- and layout-dependent attributes
                # (.base, .strides, .flags, .data) observe the snapshot, not the user's object, so only the
                # value-determined navigation set folds.
                raise AnalysisRejection(f"array attribute '{name}' is not supported", origin)
            if contains_record(obj.layout):
                # A bound method of a record-carrying sequence would run Python's protocols over the rebuilt
                # records; records are consumed by field access only.
                raise AnalysisRejection(f"attribute '{name}' of a record-carrying sequence is not supported", origin)
            concrete = materialize_static(obj)
            if concrete is None:
                raise AnalysisRejection(f"attribute '{name}' of a runtime aggregate is not supported yet", origin)
            # Static navigation (``.T``, ``.shape``, ``.ndim``, ``.flatten`` on an all-Known array; a value method)
            # folds through the concrete object, exactly as a Known value does.
            obj = Known(concrete)
        if isinstance(obj, Reference):
            component = obj.obj
            if isinstance(component, (types.ModuleType, type)):
                # A namespace (math, np, a class), not a stateful component: attribute access is a plain lookup,
                # so math.sqrt/np.floor resolve to the callable the call site then dispatches through the registry.
                try:
                    attribute, admitted = self._read_attribute_snapshot(component, name, origin)
                except AttributeError as error:
                    raise AnalysisRejection(str(error), origin) from None
                return normalize_static(admitted) if admitted is not None else Reference(attribute)
            _reject_attribute_hooks(type(component), None, origin)
            class_attribute = _mro_attribute_of(type(component), name)
            if type(class_attribute) is property:  # an exact property (not a subclass) wins over any __dict__ entry
                if not isinstance(class_attribute.fget, types.FunctionType):
                    raise AnalysisRejection(f"property {name!r} has an unsupported getter", origin)
                # Bind the getter to the exact receiver so its ``self.stored`` reads resolve to the same StateLeaf/Known
                # a direct read would, and so recursion identity and the ``self`` parameter bind correctly.
                return _PropertyRead(types.MethodType(class_attribute.fget, component))
            _reject_attribute_hooks(type(component), name, origin)
            if name not in getattr(component, "__dict__", {}) and isinstance(
                class_attribute, (types.FunctionType, classmethod, staticmethod)
            ):
                key = (id(component), name)
                if key not in self._bound_methods:
                    self._bound_methods[key] = getattr(component, name)
                return Reference(self._bound_methods[key])
            if (
                class_attribute is not None
                and hasattr(type(class_attribute), "__get__")
                and not isinstance(class_attribute, (types.FunctionType, classmethod, staticmethod))
                and not (
                    isinstance(class_attribute, types.MemberDescriptorType)
                    and class_attribute.__name__ == name
                    and getattr(class_attribute, "__objclass__", None) in type(component).__mro__
                )
            ):
                raise AnalysisRejection(f"descriptor attribute '{name}' on a component is not supported", origin)
            leaf = StateLeaf(component, (name,))
            if leaf in env.facts:
                return env.get(leaf)
            if leaf in self._state_livein:
                fact = self._state_livein[leaf]
                env.set(leaf, fact)
                return fact
            try:
                snapshot, admitted = self._read_attribute_snapshot(component, name, origin)
            except AttributeError as error:
                raise AnalysisRejection(str(error), origin) from None
            if admitted is None:
                # ``snapshot`` is a sub-object (a potential child component): record the parent -> child graph edge.
                # Canonical member paths are resolved from these edges by a shortest-path fixpoint in ``provenance()``,
                # so a child's slot name is order-independent even when a lexicographically-smaller alias is discovered
                # later (the state-leaf cache above would otherwise freeze a stale first-seen path).
                self._component_edges.add((id(component), name, id(snapshot)))
            fact = normalize_static(admitted) if admitted is not None else Reference(snapshot)
            env.set(leaf, fact)
            return fact
        if isinstance(obj, Known):
            if isinstance(obj.value, StaticStr):
                # Trimmed (scope ruling T6): a str constant stays an inert value (equality, len, concatenation
                # all fold), but its methods are host machinery a kernel does not need -- every honest use
                # precomputes the constant. Refusing the fetch keeps the whole str method surface closed.
                raise AnalysisRejection(
                    "str methods are not supported in a kernel; strings are inert constants here", origin
                )
            receiver = as_python(obj.value)
            try:
                concrete = getattr(receiver, name)
            except AttributeError as error:
                raise AnalysisRejection(str(error), origin) from None
            admitted = admit(concrete)
            if admitted is None and callable(concrete):
                if isinstance(obj.value, StaticRange) and range_size(obj.value) > (1 << 20):
                    # range.count/.index fall back to linear iteration for non-int arguments.
                    raise AnalysisRejection("a method of an oversized range is not supported", origin)
                if name.startswith("__"):
                    # A dunder bound off a value (t.__repr__) is the reconstruction-observation spelling of the
                    # protocol the concrete-call whitelist refuses.
                    raise AnalysisRejection(f"dunder attribute '{name}' access on a value is not supported", origin)
                if isinstance(obj.value, StaticRecord):
                    raise AnalysisRejection(f"method '{name}' on a record value is not supported yet", origin)
                if isinstance(obj.value, StaticSeq):
                    # tuple.count/.index are identity-and-equality games the reconstruction cannot vouch for
                    # (a NaN element matches by identity in Python, never after a rebuild).
                    raise AnalysisRejection(f"sequence method '{name}' is not supported", origin)
                value_key = (obj.value, name)
                if value_key not in self._value_methods:
                    self._value_methods[value_key] = concrete
                return Reference(self._value_methods[value_key])
            return normalize_static(admitted) if admitted is not None else Reference(concrete)
        raise AnalysisRejection("attribute access on a runtime value", origin)

    # ------------------------------------ terminators ------------------------------------

    def _resolve_terminator(self, unit: FunctionUnit, block: Block, env: _Env) -> list[BlockId]:
        terminator = block.terminator
        assert terminator is not None
        match terminator:
            case Jump(target=target):
                return [target]
            case Branch(cond=cond, then_target=then_target, else_target=else_target):
                fact = env.get(Local(cond))
                if isinstance(fact, Known):  # a Known Bool always drives edge selection (the width rule exception)
                    taken = as_python(fact.value)
                    assert isinstance(taken, bool)
                    return [then_target if taken else else_target]
                # An unavailable condition still opens both arms. Holding the block back instead looks safer and
                # is not: a Branch inside a loop body sits BEFORE the body's trailing back-edge Jump, so deferring
                # it stops the loop re-flowing at all, the transiently-inexact store never sees its operand
                # promote, and valid kernels are refused -- measured, and the same shape round-10's edge
                # withholding regressed. Speculation is therefore left intact and the unsound RESULT of it is
                # caught after stabilization instead, by _check_branch_settled.
                return [then_target, else_target]
            case StaticFor():
                iterable_fact = env.get(Local(terminator.iterable))
                seed = self._unroll_seeds.get(terminator.origin)
                if seed is not None:
                    iterable_fact = join_facts(seed, iterable_fact, terminator.origin)
                cached = self._unroll_cache.get(block.id)
                if cached is not None:
                    cached_fact, chain_entry = cached
                    if _same_fact(iterable_fact, cached_fact):
                        return [chain_entry]
                    raise _UnrollRestart(terminator.origin, join_facts(cached_fact, iterable_fact, terminator.origin))
                chain_entry = self._unroll(unit, block, terminator, iterable_fact)
                self._unroll_cache[block.id] = (iterable_fact, chain_entry)
                return [chain_entry]
            case Fail() | UnitExit():
                return []
        raise AssertionError(terminator)

    # ------------------------------------ StaticFor unrolling ------------------------------------

    def _unroll(self, unit: FunctionUnit, header: Block, loop: StaticFor, iterable: Fact) -> BlockId:
        if isinstance(iterable, Reference):
            # A live object's __iter__/__len__ would run at compile time against reset-time state.
            raise AnalysisRejection("iteration over an object is not supported", loop.origin)
        per_trip: list[Known | Reference | int] = []
        if isinstance(iterable, AggregateFact):
            if isinstance(iterable.layout, RecordLayout):
                # Materializing would drive Python's iteration protocol (a user __len__/__getitem__/__iter__) on
                # the reconstruction -- a demonstrated wrong-value and non-termination hazard.
                raise AnalysisRejection("iteration over a record is not supported", loop.origin)
            trip_count = outer_arity(iterable.layout)
            if trip_count <= UNROLL_THRESHOLD:  # sized before materializing: a 32k table must reject instantly
                for position in range(trip_count):
                    child: Fact = iterable.child(position)
                    if isinstance(child, AggregateFact):
                        materialized = materialize_static(child)
                        child = Known(materialized) if materialized is not None else child
                    if isinstance(child, (Known, Reference)):
                        per_trip.append(child)
                    else:
                        # A runtime element: the trip binds through a synthesized projection prelude, so the
                        # child's cells (a scalar leaf or a whole row) flow exactly as an explicit v[k] would.
                        per_trip.append(position)
        elif isinstance(iterable, Known) and isinstance(iterable.value, StaticSeq):
            trip_count = len(iterable.value.items)
            if trip_count <= UNROLL_THRESHOLD:
                per_trip = [Known(item) for item in iterable.value.items]
        elif isinstance(iterable, Known):
            concrete = as_python(iterable.value)
            try:
                trip_count = len(concrete)  # type: ignore[arg-type]  # sized BEFORE materializing (range(10**9)!)
            except TypeError:
                raise AnalysisRejection("loop iterable has no static length", loop.origin) from None
            except OverflowError:  # len() of an astronomically large range (range(10**38)): far past any threshold
                raise AnalysisRejection(
                    f"loop trip count exceeds the unroll threshold {UNROLL_THRESHOLD}; a counted back-edge loop is "
                    "not supported yet",
                    loop.origin,
                ) from None
            if trip_count <= UNROLL_THRESHOLD:
                for element in list(concrete):  # type: ignore[call-overload]
                    admitted = admit(element)
                    assert admitted is not None, "an element of a closed-domain container must re-admit"
                    per_trip.append(Known(admitted))
        else:
            raise AnalysisRejection("loop trip count is not static here", loop.origin)
        if trip_count > UNROLL_THRESHOLD:
            raise AnalysisRejection(
                f"trip count {trip_count} exceeds the unroll threshold {UNROLL_THRESHOLD}; a counted back-edge loop "
                "is not supported yet",
                loop.origin,
            )
        _logger.info("unrolling %d trip(s) at %s", trip_count, loop.origin[0])
        chain_target = loop.exit_target
        for element_fact in reversed(per_trip):
            body_entry = self._clone_subgraph(unit, loop.body_entry, header.id, chain_target, loop)
            prelude = Block(self._fresh_block_id())
            # A fresh per-trip scope for the target, exactly like the builder's comprehension-entry reset: the
            # schema flow clears at the unchecked unbind, so each trip's bind establishes its own Python-faithful
            # kind (trip 1 of ``(1, 2.5)`` sees an int, trip 2 a float) instead of a cross-trip rebinding.
            prelude.ops.append(UnbindPlace(loop.target, False, loop.origin))
            temp = BindingId(f"%u{self._temp_serial}", self._temp_serial)
            self._temp_serial += 1
            if isinstance(element_fact, Known):
                prelude.ops.append(LoadConst(temp, element_fact.value, loop.origin))
            elif isinstance(element_fact, Reference):
                prelude.ops.append(LoadRef(temp, element_fact.obj, loop.origin))
            else:
                index_temp = BindingId(f"%u{self._temp_serial}", self._temp_serial)
                self._temp_serial += 1
                position_key = admit(element_fact)
                assert position_key is not None
                prelude.ops.append(LoadConst(index_temp, position_key, loop.origin))
                prelude.ops.append(PySubscript(temp, loop.iterable, index_temp, loop.origin))
            prelude.ops.append(StorePlace(loop.target, temp, loop.origin, StoreRole.SOURCE))
            prelude.terminator = Jump(body_entry, loop.origin)
            unit.blocks[prelude.id] = prelude
            chain_target = prelude.id
        return chain_target

    def _clone_subgraph(
        self, unit: FunctionUnit, entry: BlockId, header: BlockId, continue_target: BlockId, loop: StaticFor
    ) -> BlockId:
        if len(unit.blocks) + len(loop.body_blocks) > _MAX_BLOCKS:
            raise AnalysisRejection("unroll fuel exhausted", loop.origin)
        mapping = {member: self._fresh_block_id() for member in loop.body_blocks}
        temp_map: dict[BindingId, BindingId] = {}

        def fresh_temp(binding: BindingId) -> BindingId:
            if not binding.is_temp:
                return binding
            if binding not in temp_map:
                self._temp_serial += 1
                temp_map[binding] = BindingId(f"%c{self._temp_serial}", self._temp_serial)
            return temp_map[binding]

        def remap_block(target: BlockId) -> BlockId:
            if target == header:
                return continue_target  # the back edge advances to the next trip (or the loop exit)
            # A target outside the recorded body (the unit exit for return, an enclosing loop's blocks for
            # break/continue, the loop exit itself) passes through untouched.
            return mapping.get(target, target)

        for member in loop.body_blocks:
            source = unit.blocks[member]
            clone = Block(mapping[member], [_remap_op(op, fresh_temp, _identity_place) for op in source.ops])
            assert source.terminator is not None
            clone.terminator = _remap_terminator(source.terminator, remap_block, fresh_temp, _identity_place)
            unit.blocks[clone.id] = clone
            self._block_ancestry[clone.id] = self._block_ancestry.get(member, ())
        return mapping[entry]

    # ------------------------------------ call expansion ------------------------------------

    def _construction_schema(self, klass: type) -> tuple[FieldSchema, ...]:
        key = id(klass)
        hit = self._construction_schemas.get(key)
        if hit is not None and hit[0] is klass:
            return hit[1]
        declared = construction_schema(klass)
        self._construction_schemas[key] = (klass, declared)
        return declared

    def _default_snapshot(self, klass: type, entry: FieldSchema) -> BoundFact:
        key = (id(klass), entry.name)
        hit = self._default_snapshots.get(key)
        if hit is None:
            admitted = admit(entry.default)
            hit = normalize_static(admitted) if admitted is not None else Reference(entry.default)
            self._default_snapshots[key] = hit
        return hit

    def _expand_construction(self, klass: type, call: PyCall, env: _Env) -> None:
        try:
            declared = self._construction_schema(klass)
        except FoldRefusal as refusal:
            raise AnalysisRejection(str(refusal), call.origin) from None
        name = klass.__name__
        positional_names = [entry.name for entry in declared if not entry.kw_only]
        if len(call.args) > len(positional_names):
            raise AnalysisRejection(
                f"record class '{name}' takes {len(positional_names)} positional argument(s), "
                f"{len(call.args)} given",
                call.origin,
            )
        assignments: dict[str, BindingId] = dict(zip(positional_names, call.args))
        for keyword, binding in call.kwargs:
            if keyword not in {entry.name for entry in declared}:
                raise AnalysisRejection(
                    f"record class '{name}' has no field '{keyword}' (an unexpected keyword argument)", call.origin
                )
            if keyword in assignments:
                raise AnalysisRejection(f"record class '{name}' got multiple values for field '{keyword}'", call.origin)
            assignments[keyword] = binding
        children: list[tuple[str, BoundFact]] = []
        mapping: list[BindingId | None] = []
        for entry in declared:
            source = assignments.get(entry.name)
            if source is not None:
                fact = env.get(Local(source))
                assert isinstance(fact, (Known, Residual, Reference, AggregateFact)), fact
                children.append((entry.name, fact))
                mapping.append(source)
            elif entry.default is not MISSING:
                children.append((entry.name, self._default_snapshot(klass, entry)))
                mapping.append(None)
            else:
                raise AnalysisRejection(
                    f"record class '{name}' is missing the required field '{entry.name}' here", call.origin
                )
        env.set(Local(call.dst), record_of(klass, tuple(children)))
        self._call_lowering[id(call)] = CallLowering.CONSTRUCTION
        self._route_evidence[id(call)] = FieldBindings(tuple(mapping))

    def _fresh_temp(self) -> BindingId:
        self._temp_serial += 1
        return BindingId(f"%c{self._temp_serial}", self._temp_serial)

    # ------------------------------------ dispatch row actions ------------------------------------

    def _dispatch(self, rows: tuple[_CallRow, ...], site: _CallSite) -> bool:
        for row in rows:
            if row.selects(site):
                row.apply(self, site)
                return True
        return False

    def _reject_getattr(self, site: _CallSite) -> NoReturn:
        # Trimmed (scope ruling T1): the static name getattr would require anyway makes it pure spelling
        # redundancy over the dotted access, and letting it near the concrete path was a demonstrated
        # miscompile habitat. The row stays as the refusal site so the guidance is specific.
        raise AnalysisRejection(
            "getattr is not supported in a kernel; spell the attribute access directly (x.name)",
            site.call.origin,
        )

    def _reject_isinstance(self, site: _CallSite) -> NoReturn:
        # Trimmed (scope ruling T4): values are statically typed here, so an honest query answers itself
        # at authoring time, while a faithful compile-time verdict demanded real machinery -- member
        # provenance, complete classinfo resolution, record-layout folds -- with a demonstrated
        # miscompile history. One refusal at the dispatch covers every spelling.
        raise AnalysisRejection(
            "isinstance is not supported in a kernel: values are statically typed", site.call.origin
        )

    def _route_transpose(self, site: _CallSite) -> None:
        # A pure structural relayout: the same leaves under the reversed shape, recorded as a route plan
        # (source ordinal per result cell). Precedes admission: nothing crosses.
        pivoted = site.sole_argument
        assert isinstance(pivoted, AggregateFact)
        pivoted_layout = pivoted.layout
        assert isinstance(pivoted_layout, ArrayLayout)
        routes = _transpose_routes(pivoted_layout.shape)
        site.env.set(
            Local(site.call.dst),
            AggregateFact(
                ArrayLayout(pivoted_layout.shape[::-1], pivoted_layout.dtype),
                tuple(pivoted.leaves[k] for k in routes),
            ),
        )
        self._call_lowering[id(site.call)] = CallLowering.CONVERSION
        self._route_evidence[id(site.call)] = SourceSelection(routes)

    def _fold_rank(self, site: _CallSite) -> None:
        # Deliberately narrow (np.ndim of a LIST would observe structure the fact model erases at atomic
        # leaves): a numeric scalar is rank 0, an array is its layout rank, everything else rejects. The
        # linalg stubs probe ranks through this spelling.
        probed = site.sole_argument
        assert probed is not None
        if isinstance(probed, AggregateFact) and isinstance(probed.layout, ArrayLayout):
            rank = len(probed.layout.shape)
        elif isinstance(probed, Residual) or (isinstance(probed, Known) and _residual_type(probed.value) is not None):
            rank = 0
        else:
            raise AnalysisRejection("np.ndim of this value is not supported here", site.call.origin)
        admitted_rank = admit(rank)
        assert admitted_rank is not None
        site.env.set(Local(site.call.dst), Known(admitted_rank))
        self._call_lowering[id(site.call)] = CallLowering.FOLDED

    def _construct_record(self, site: _CallSite) -> None:
        # Record construction is STRUCTURAL, never an evaluation: the layout is the class's validated field
        # schema and the children are the argument facts THEMSELVES -- runtime leaves and reference leaves ride
        # through untouched, and no host code (not even the generated __init__) ever runs. Like the transpose
        # relayout, this precedes the admission harness: there is nothing to admit because nothing crosses.
        assert isinstance(site.target, type)
        self._expand_construction(site.target, site.call, site.env)

    def _reflavor_container(self, site: _CallSite) -> None:
        # A container conversion over an aggregate is a LAYOUT operation, never an evaluation: the same leaves
        # (runtime ones included) re-aggregate under the requested flavor. Concrete containers (a range, a
        # string, an all-Known tuple) miss the row and fall through to the vetted evaluation.
        source_fact = site.sole_argument
        assert isinstance(source_fact, AggregateFact)  # a record never reaches here: admission refused it
        children = tuple(source_fact.child(i) for i in range(outer_arity(source_fact.layout)))
        site.env.set(Local(site.call.dst), aggregate_of(children, is_list=site.target is list))
        self._call_lowering[id(site.call)] = CallLowering.CONVERSION

    def _fold_length(self, site: _CallSite) -> None:
        # Length is layout-determined: it folds on runtime leaves exactly as the unpacking arity check (the
        # PyLen op) does, records having been refused by the admission walk already.
        sized = site.sole_argument
        assert isinstance(sized, AggregateFact)
        length = admit(outer_arity(sized.layout))
        assert length is not None
        site.env.set(Local(site.call.dst), Known(length))
        self._call_lowering[id(site.call)] = CallLowering.FOLDED

    def _build_array(self, site: _CallSite) -> None:
        # A residual-carrying array construction is the same LAYOUT operation under numpy's discovery rules,
        # restricted to the proven subset; a fully static argument misses the row and falls through to the
        # vetted concrete call, where numpy itself decides every discovery corner (object promotion, the
        # uint64 range, bool widening) and the result normalizes back exactly.
        source = site.args[0]
        assert isinstance(source, AggregateFact)
        site.env.set(
            Local(site.call.dst),
            self._array_factory(source, site.call.origin, force_float=_explicit_float_dtype(site)),
        )
        self._call_lowering[id(site.call)] = CallLowering.CONVERSION

    def _cast_scalar(self, site: _CallSite, kind: SemType) -> None:
        # ``float()``/``int()``/``bool()`` on a runtime scalar: a same-kind cast is the identity (a documented
        # no-op); a cross-kind cast lowers to a conversion op (int<->float truncation/promotion, truthiness,
        # bool widening). Explicit casts are how bool crosses into arithmetic and how float truncates to int.
        site.env.set(Local(site.call.dst), Residual(kind))
        self._call_lowering[id(site.call)] = CallLowering.CAST

    def _reject_unimplemented_library(self, site: _CallSite) -> NoReturn:
        # A recognized math/numpy function with no fast-math hardware equivalent (erf, spacing, a ufunc): a
        # distinct public error so the user knows it is a missing library primitive, not a bad call.
        raise LibraryAnalysisRejection(
            f"library function {_callee_name(site.target)!r} is not implemented yet", site.call.origin
        )

    def _reject_runtime_arguments(self, site: _CallSite) -> NoReturn:
        raise AnalysisRejection(
            f"call to {_callee_name(site.target)} with runtime arguments is not supported yet", site.call.origin
        )

    def _bind_array_method(self, site: _AttrSite) -> bool:
        method_key = (site.op.obj, site.op.name)
        if method_key not in self._array_methods:
            self._array_methods[method_key] = _ArrayMethod(site.op.obj, site.op.name)
        site.env.set(Local(site.op.dst), Reference(self._array_methods[method_key]))
        return False

    def _rewrite_array_transpose(self, site: _AttrSite) -> bool:
        # Transpose is a pure structural relayout (a permutation of the same leaves), so ``.T`` rewrites to the
        # spelled np.transpose call and both spellings share one lowering.
        callee = self._fresh_temp()
        site.block.ops[site.index : site.index + 1] = [
            LoadRef(callee, np.transpose, site.op.origin),
            PyCall(site.op.dst, callee, (site.op.obj,), (), site.op.origin),
        ]
        return True

    def _expand_call(self, unit: FunctionUnit, block: Block, index: int, call: PyCall, env: _Env) -> bool:
        from .._lib import IntrinsicResultRule, Library, resolve
        from .._lib import Intrinsic

        callee_fact = env.get(Local(call.callee))
        if not isinstance(callee_fact, Reference):
            raise AnalysisRejection("call target is not resolvable here", call.origin)
        if any(call.starred):
            # f(*t) flattens BEFORE any dispatch: each starred argument must be a positional container of
            # static arity, and its children become ordinary arguments through synthesized projections --
            # the rewritten call (no stars left) then re-enters every path unchanged.
            replacement: list[Op] = []
            flattened: list[BindingId] = []
            for position, arg in enumerate(call.args):
                if position < len(call.starred) and call.starred[position]:
                    fact = env.get(Local(arg))
                    if not (
                        isinstance(fact, AggregateFact)
                        and isinstance(fact.layout, (TupleLayout, ListLayout, ArrayLayout))
                    ):
                        raise AnalysisRejection(
                            "argument unpacking requires a tuple, list, or array of static arity here", call.origin
                        )
                    for child in range(outer_arity(fact.layout)):
                        index_temp = self._fresh_temp()
                        child_key = admit(child)
                        assert child_key is not None
                        replacement.append(LoadConst(index_temp, child_key, call.origin))
                        element = self._fresh_temp()
                        replacement.append(PySubscript(element, arg, index_temp, call.origin))
                        flattened.append(element)
                else:
                    flattened.append(arg)
            replacement.append(PyCall(call.dst, call.callee, tuple(flattened), call.kwargs, call.origin))
            block.ops[index : index + 1] = replacement
            return True
        if isinstance(callee_fact.obj, _ArrayMethod):
            method = callee_fact.obj
            if call.args[:1] == (method.receiver,) and not call.kwargs:
                # The canonical explicit-receiver form (installed by the rewrite below): flatten/ravel/reshape
                # are pure RELAYOUTS of the same leaves -- the source dtype survives structurally even with
                # zero elements -- and emission is the ordinary conversion copy.
                receiver_fact = env.get(Local(method.receiver))
                assert isinstance(receiver_fact, AggregateFact) and isinstance(receiver_fact.layout, ArrayLayout)
                cells = leaf_count(receiver_fact.layout)
                if method.name == "reshape":
                    reshaped = self._static_reshape_target(call.args[1:], env, call.origin)
                    if math.prod(reshaped) != cells:
                        raise AnalysisRejection(
                            f"cannot reshape an array of {cells} element(s) into shape "
                            f"({', '.join(map(str, reshaped))})",
                            call.origin,
                        )
                    relayout = ArrayLayout(reshaped, receiver_fact.layout.dtype)
                else:
                    if len(call.args) != 1:
                        raise AnalysisRejection(
                            f"{method.name}() accepts no arguments here (only the default C order is " "supported)",
                            call.origin,
                        )
                    relayout = ArrayLayout((cells,), receiver_fact.layout.dtype)
                env.set(Local(call.dst), AggregateFact(relayout, receiver_fact.leaves))
                self._call_lowering[id(call)] = CallLowering.CONVERSION
                return False
            if call.kwargs or (method.name != "reshape" and call.args):
                raise AnalysisRejection(
                    (
                        f"{method.name}() accepts no arguments here (only the default C order is supported)"
                        if method.name != "reshape"
                        else "reshape() accepts a static shape only (no keyword arguments)"
                    ),
                    call.origin,
                )
            block.ops[index : index + 1] = [replace(call, args=(method.receiver, *call.args))]
            return True
        site = _CallSite(
            callee_fact.obj,
            call,
            env,
            [env.get(Local(arg)) for arg in call.args],
            [(keyword, env.get(Local(value))) for keyword, value in call.kwargs],
        )
        match = resolve(site.target)
        stub_display: str | None = None
        if isinstance(match, Library):
            _check_call_shape(site)
            # A composite library stub inlines exactly like a user function, but its grafted frames display the
            # SPELLED callee the user resolved (np.dot reads "in dot():" even though matmul_ implements it); the
            # stub's own stripped name is the fallback for a callee with no __name__.
            stub_display = getattr(site.target, "__name__", None) or match.display_name
            site = replace(site, target=match.stub)
        elif isinstance(match, Intrinsic):
            argument_facts = site.operands
            if all(_concrete_fact(fact) is not None for fact in argument_facts):
                pass  # fully static (an all-Known aggregate included): fold concretely below through the callable
            else:
                # A runtime-operand intrinsic (sqrt(x), sin(x)...) becomes an HIR operation at emission; keep the
                # PyCall in the graph, typed by the operator's result, and let emission resolve the registry match.
                if call.kwargs:
                    raise AnalysisRejection("keyword arguments to a hardware intrinsic are not supported", call.origin)
                arity = match.operator.signature.arity
                if len(call.args) != arity:
                    raise AnalysisRejection(f"intrinsic expects {arity} argument(s), got {len(call.args)}", call.origin)
                for fact in argument_facts:
                    if _numeric_sem(fact) is None:
                        raise AnalysisRejection("a non-numeric operand reaches a numeric intrinsic", call.origin)
                signature_result = match.operator.signature.result_type
                if isinstance(signature_result, BoolType) and all(
                    _numeric_sem(fact) is SemType.INT for fact in argument_facts
                ):
                    # A classification of an integer folds ideally: an integer is always finite and never an infinity
                    # (hardware integers saturate, so this holds in the datapath too, not only in Python). The fold's
                    # closed world is the four registered classifiers; a new bool-result intrinsic (a signbit, say)
                    # must extend this fold rather than silently inherit the isfinite/isinf split.
                    assert isinstance(match.operator, (FloatIsFinite, FloatIsInf, FloatIsPosInf, FloatIsNegInf))
                    verdict = isinstance(match.operator, FloatIsFinite)
                    numpy_spelling = getattr(site.target, "__module__", "").startswith("numpy")
                    env.set(Local(call.dst), Known(NpBool(verdict) if numpy_spelling else StaticBool(verdict)))
                    self._call_lowering[id(call)] = CallLowering.FOLDED
                    return False
                # The result kind follows the spelling's declared rule (see the library registry): an all-integer
                # operand list keeps an integer-overloaded spelling integer (contained at MIR); any float operand,
                # or a float-forcing spelling, promotes the integer operands C-style and runs the float operator.
                rule = match.result_rule
                if rule is IntrinsicResultRule.ALWAYS_INT:
                    result: Fact = Residual(SemType.INT)
                elif rule is IntrinsicResultRule.SIGNATURE:
                    result = Residual(SemType.BOOL if isinstance(signature_result, BoolType) else SemType.FLOAT)
                else:  # INT_OVERLOAD
                    all_int = all(_numeric_sem(fact) is SemType.INT for fact in argument_facts)
                    result = Residual(SemType.INT) if all_int else Residual(SemType.FLOAT)
                env.set(Local(call.dst), result)
                self._call_lowering[id(call)] = CallLowering.INTRINSIC
                return False
        if not isinstance(site.target, (types.FunctionType, types.MethodType)) and not (
            hasattr(type(site.target), "__call__")
            and isinstance(getattr(type(site.target), "__call__", None), types.FunctionType)
        ):
            # A builtin (range, float, abs...) or a fully-static intrinsic evaluates concretely under the snapshot
            # doctrine; its runtime-operand form was already routed to an HIR operation above.
            if self._dispatch(_PRE_ADMISSION_ROWS, site):
                return False
            # Concrete evaluation is a CLOSED WHITELIST behind one door: the fold admission harness. The
            # analyzer contributes only what the harness cannot know -- per-Analyzer minted-method identity and
            # library-registry resolution -- and locates the refusal at the call origin.
            minted = any(site.target is method for method in self._value_methods.values())
            try:
                admit_call(
                    site.target,
                    site.args,
                    [fact for _, fact in site.kwargs],
                    minted=minted,
                    registry_resolved=resolve(site.target) is not None,
                )
            except FoldRefusal as refusal:
                if refusal.library_diagnostic:
                    raise LibraryAnalysisRejection(str(refusal), call.origin) from None
                raise AnalysisRejection(str(refusal), call.origin) from None
            if self._dispatch(_POST_ADMISSION_ROWS, site):
                return False
            concrete_args: list[StaticValue | Reference | None] = [
                fact if isinstance(fact, Reference) else _concrete_fact(fact) for fact in site.args
            ]
            concrete_kwargs = [
                (keyword, fact if isinstance(fact, Reference) else _concrete_fact(fact))
                for keyword, fact in site.kwargs
            ]
            if any(value is None for value in concrete_args) or any(v is None for _, v in concrete_kwargs):
                consumed = self._dispatch(_RUNTIME_OPERAND_ROWS, site)
                assert consumed, "the runtime-operand table is total: its last row refuses unconditionally"
                return False
            try:
                concrete = site.target(  # type: ignore[operator]
                    *[_crossing_object(value) for value in concrete_args if value is not None],
                    **{keyword: _crossing_object(value) for keyword, value in concrete_kwargs if value is not None},
                )
            except Exception as error:
                raise AnalysisRejection(f"call fails here: {error}", call.origin) from None
            admitted = admit(concrete)
            if admitted is None:
                if isinstance(concrete, np.ndarray) and concrete.ndim == 0:
                    raise AnalysisRejection(
                        "a 0-dimensional array is not supported; use the scalar directly", call.origin
                    )
                env.set(Local(call.dst), Reference(concrete))
            else:
                env.set(Local(call.dst), normalize_static(admitted))
            self._call_lowering[id(call)] = CallLowering.FOLDED
            return False
        target = site.target
        receiver: object | None = None
        if isinstance(target, types.MethodType):
            receiver = target.__self__
        elif not isinstance(target, types.FunctionType):
            call_hook = getattr(type(target), "__call__", None)
            if isinstance(call_hook, types.FunctionType):
                target = types.MethodType(call_hook, target)
                receiver = target.__self__
            else:
                raise AnalysisRejection(f"call target {target!r} is not a supported callable", call.origin)
        key = (
            id(target.__func__ if isinstance(target, types.MethodType) else target),
            id(receiver) if receiver is not None else None,
        )
        ancestry = self._block_ancestry.get(block.id, ())
        if key in ancestry:
            raise AnalysisRejection("recursive call", call.origin)
        try:
            template = self._template(target)
        except BuildRejection as rejection:
            raise AnalysisRejection(
                rejection.message, _stub_frames(rejection.origin, stub_display) + call.origin
            ) from None
        if len(unit.blocks) > _MAX_BLOCKS:
            raise AnalysisRejection("expansion fuel exhausted", call.origin)
        # Binding validates COMPLETELY before the graft mutates anything: a rejection raised past the point
        # where the call op leaves the CFG would be deferred under a key no stabilization walk can find, and an
        # open deferral net that later stabilizes legal would then silently compile the invalid call away.
        params = list(template.params)
        bound_params = params[1:] if template.bound_self is not None else params
        positional = list(call.args)
        keyword = dict(call.kwargs)
        fn_object = target.__func__ if isinstance(target, types.MethodType) else target
        raw_defaults = fn_object.__defaults__ or ()
        kw_defaults = fn_object.__kwdefaults__ or {}
        self_offset = 1 if template.bound_self is not None else 0
        positional_count = fn_object.__code__.co_argcount - self_offset
        positional_only = {p.name for p in bound_params[: max(0, fn_object.__code__.co_posonlyargcount - self_offset)]}
        positional_params = bound_params[:positional_count]
        default_by_name: dict[str, object] = dict(
            zip((p.name for p in positional_params[len(positional_params) - len(raw_defaults) :]), raw_defaults)
        )
        default_by_name.update(kw_defaults)
        if len(positional) > len(positional_params):
            raise AnalysisRejection("too many positional arguments", call.origin)
        sources: list[BindingId | _DefaultArgument] = []
        for offset, param in enumerate(bound_params):
            if offset < len(positional):
                if param.name in keyword:
                    raise AnalysisRejection(f"duplicate argument '{param.name}'", call.origin)
                sources.append(positional[offset])
            elif param.name in keyword and param.name not in positional_only:
                sources.append(keyword.pop(param.name))
            elif param.name in default_by_name:
                default_value = default_by_name[param.name]
                if isinstance(default_value, np.ndarray) and default_value.ndim == 0:
                    raise AnalysisRejection(
                        "a 0-dimensional array is not supported; use the scalar directly", call.origin
                    )
                sources.append(_DefaultArgument(default_value, admit(default_value)))
            else:
                raise AnalysisRejection(f"missing argument '{param.name}'", call.origin)
        if keyword:
            raise AnalysisRejection(f"unexpected keyword argument '{next(iter(keyword))}'", call.origin)
        binding_map: dict[BindingId, BindingId] = {}

        def fresh(binding: BindingId) -> BindingId:
            if binding not in binding_map:
                self._binding_serial += 1
                binding_map[binding] = BindingId(
                    binding.name if not binding.is_temp else f"%g{self._binding_serial}", self._binding_serial
                )
            return binding_map[binding]

        return_local = BindingId(f"ret@{template.name}", self._binding_serial + 500_000)
        block_map: dict[BlockId, BlockId] = {b: self._fresh_block_id() for b in template.blocks}
        continuation = Block(self._fresh_block_id())
        continuation.ops = list(block.ops[index + 1 :])
        continuation.terminator = block.terminator
        unit.blocks[continuation.id] = continuation
        self._block_ancestry[continuation.id] = ancestry

        def graft_place(place: Place) -> Place:
            match place:
                case Local(binding=binding):
                    return Local(fresh(binding))
                case ReturnPlace():
                    return Local(return_local)
                case _:
                    return place

        for template_block in template.blocks.values():
            clone = Block(block_map[template_block.id])
            for op in template_block.ops:
                remapped = _remap_op(op, fresh, graft_place)
                # Diagnostics point back at the user call site; a registry stub's own frames graft under the
                # public spelling so the context reads "in matmul():", never the shadow-avoidance "matmul_".
                remapped.origin = _stub_frames(remapped.origin, stub_display) + call.origin
                clone.ops.append(remapped)
            assert template_block.terminator is not None
            if isinstance(template_block.terminator, UnitExit):
                clone.terminator = Jump(continuation.id, call.origin)
            else:
                clone.terminator = _remap_terminator(
                    template_block.terminator, lambda b: block_map[b], fresh, graft_place
                )
                clone.terminator.origin = _stub_frames(clone.terminator.origin, stub_display) + call.origin
            unit.blocks[clone.id] = clone
            self._block_ancestry[clone.id] = ancestry + (key,)
        # The call site becomes: bind arguments -> jump into the graft; the continuation reads the return local.
        block.ops = block.ops[:index]
        if template.bound_self is not None:
            self_temp = BindingId(f"%s{self._binding_serial}", self._binding_serial)
            self._binding_serial += 1
            block.ops.append(LoadRef(self_temp, template.bound_self, call.origin))
            block.ops.append(StorePlace(Local(fresh(params[0])), self_temp, call.origin, StoreRole.SOURCE))
        for param, source in zip(bound_params, sources, strict=True):
            if isinstance(source, _DefaultArgument):
                default_temp = BindingId(f"%d{self._binding_serial}", self._binding_serial)
                self._binding_serial += 1
                if source.admitted is not None:
                    block.ops.append(LoadConst(default_temp, source.admitted, call.origin))
                else:
                    block.ops.append(LoadRef(default_temp, source.value, call.origin))
                argument = default_temp
            else:
                argument = source
            block.ops.append(StorePlace(Local(fresh(param)), argument, call.origin, StoreRole.SOURCE))
        # A revisit graft replaces a terminator an earlier visit may have already resolved (the call deferred
        # behind a pending violation, so the visit ran past it): the edges recorded for the replaced terminator
        # -- every recorded out-edge of this block, since out-edges are recorded only at ITS terminator's
        # resolution -- would otherwise survive as phantom edges from this block to the continuation's
        # successors, leaking a path that skips the graft into the stable graph. Retract them, and drop the env
        # of any successor thereby left with no in-edge: it was derived on the phantom path with the call result
        # unbound, and leaving it standing would poison the continuation's later join into that same successor
        # (bound-joins-unbound = maybe-unbound) instead of a clean replace. The displaced terminator re-records
        # its edges -- and re-establishes those successors' envs -- from the continuation when that executes.
        # This retraction is only one edge deep, so a TRANSITIVE successor of the graft block can retain a
        # phantom-unbound env: the residual false-rejection class documented in TODO.md, which the Stage-4
        # resolved-IR boundary dissolves by making residualization a total pass after the fixpoint.
        retracted = {edge for edge in self._executable_edges if edge[0] == block.id}
        self._executable_edges -= retracted
        for _, orphan in retracted:
            if not any(edge[1] == orphan for edge in self._executable_edges):
                self._block_in.pop(orphan, None)
                self._executable_blocks.discard(orphan)
        block.terminator = Jump(block_map[template.entry], call.origin)
        continuation.ops.insert(0, LoadPlace(call.dst, Local(return_local), call.origin))
        # The graft destroys the call op, so an earlier visit's deferral keyed on it must not dangle off the
        # graph (the graph-anchored assert in _raise_transfer_deferrals). Dropping it is enough: the continuation
        # re-derives or clears any residual exactly as a clean revisit of the call would have.
        self._transfer_deferrals.pop(id(call), None)
        return True


# ---------------------------------------- dispatch tables ----------------------------------------
#
# ORDERED, FIRST MATCH WINS, and the order is SEMANTIC rather than cosmetic. `_PRE_ADMISSION_ROWS` runs ahead of
# the fold-admission harness because its arms either resolve structurally, with nothing crossing the host
# boundary, or refuse with guidance the harness cannot phrase; `_POST_ADMISSION_ROWS` runs behind it, so an arm
# there has already been vetted for crossing; `_RUNTIME_OPERAND_ROWS` is reached only once some operand is known
# not to be concrete and is TOTAL -- its last row refuses unconditionally, so no call leaves it unconsumed.

_PRE_ADMISSION_ROWS: tuple[_CallRow, ...] = (
    _CallRow(_AnyOf((getattr,)), _always, Analyzer._reject_getattr),
    _CallRow(_AnyOf((isinstance,)), _always, Analyzer._reject_isinstance),
    _CallRow(_AnyOf((np.transpose,)), _sole_array_operand, Analyzer._route_transpose),
    _CallRow(_AnyOf((np.ndim,)), _sole_operand, Analyzer._fold_rank),
    _CallRow(_Matching(_is_record_class), _always, Analyzer._construct_record),
)

_POST_ADMISSION_ROWS: tuple[_CallRow, ...] = (
    _CallRow(_AnyOf((list, tuple)), _sole_aggregate_operand, Analyzer._reflavor_container),
    _CallRow(_AnyOf((len,)), _sole_aggregate_operand, Analyzer._fold_length),
    _CallRow(_AnyOf((np.array, np.asarray, np.asanyarray)), _residual_array_source, Analyzer._build_array),
)

_RUNTIME_OPERAND_ROWS: tuple[_CallRow, ...] = (
    _CallRow(_AnyOf((float,)), _sole_residual_operand, partial(Analyzer._cast_scalar, kind=SemType.FLOAT)),
    _CallRow(_AnyOf((int,)), _sole_residual_operand, partial(Analyzer._cast_scalar, kind=SemType.INT)),
    _CallRow(_AnyOf((bool,)), _sole_residual_operand, partial(Analyzer._cast_scalar, kind=SemType.BOOL)),
    _CallRow(_Matching(is_unimplemented_library), _always, Analyzer._reject_unimplemented_library),
    _CallRow(_Matching(_anything), _always, Analyzer._reject_runtime_arguments),
)

# The linalg and reduction stubs are defined over arrays only; a scalar/list/tuple operand must not acquire
# array semantics through the spelled call any more than it does through an operator. A spelling's arity rule
# precedes its operand rule, and the matrix product's operand refusal precedes the shared one, because first
# match decides which of a call's several violations is the one reported.

_REDUCTIONS = (np.max, np.amax, np.mean)
_BINARY_LINALG = (np.matmul, np.dot, np.outer)
_ARRAY_ONLY_SPELLINGS = (*_REDUCTIONS, *_BINARY_LINALG, np.trace)

_LIBRARY_SHAPE_ROWS: tuple[_ShapeRow, ...] = (
    _ShapeRow(_AnyOf(_REDUCTIONS), _reduction_axis_given, _reject_reduction_axis),
    _positional_arity_rule(_BINARY_LINALG, 2),
    _positional_arity_rule((np.trace,), 1),
    _ShapeRow(_AnyOf((np.matmul, np.dot)), _non_array_operand, _reject_matrix_operand),
    _ShapeRow(_AnyOf(_ARRAY_ONLY_SPELLINGS), _non_array_operand, _reject_array_operand),
    _ShapeRow(_AnyOf((np.sign,)), _integer_operand, _reject_integer_sign),
)

_ATTRIBUTE_INTERCEPT_ROWS: tuple[_AttrRow, ...] = (
    _AttrRow(("flatten", "ravel", "reshape"), _is_array_fact, Analyzer._bind_array_method),
    _AttrRow(("T",), _is_array_fact, Analyzer._rewrite_array_transpose),
)
