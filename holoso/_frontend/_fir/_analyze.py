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

import enum
import logging
import math
import types
from typing import TYPE_CHECKING
from collections.abc import Callable, Mapping
from dataclasses import MISSING, dataclass, field, is_dataclass, replace

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
    Terminator,
    UnbindPlace,
    UnitExit,
    op_dst,
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
    classinfo_types,
    construction_schema,
    contains_record,
    is_unimplemented_library,
    range_size,
    validate_classinfo,
)
from ..._util import RelationalOp
from ._analysis_support import (
    AnalysisRejection,
    DeferredRejection,
    LibraryAnalysisRejection,
    _concat_seqs,
    _concrete_fact,
    _contract_structure,
    _coreachable,
    _crossing_object,
    _datapath_zero,
    _fits_float64,
    _float_promoted,
    _has_truth_override,
    _identity_place,
    _is_array_fact,
    _is_list_fact,
    _join_atoms,
    _layout_dtypes,
    _lost_scalar_pools,
    _mro_attribute_of,
    _numeric_sem,
    _scalar_sem,
    _rectangular_shape,
    _reject_attribute_hooks,
    _reject_descriptor,
    _remap_op,
    _remap_terminator,
    _residual_type,
    _same_fact,
    _seq_side,
    _taint_lost,
    _transpose_routes,
    join_facts,
)
from ._opsem import BinOp, UnOp, static_binop, static_compare, static_truth, static_unop
from ..._hir import BoolType, FloatIsFinite, FloatIsInf, FloatIsNegInf, FloatIsPosInf
from ._value import (
    MetaInt,
    NpBool,
    StaticRecord,
    NpFloat,
    NpInt,
    ScalarOrigin,
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
    join_scalar_sources,
    same,
    strip_source,
)

if TYPE_CHECKING:
    from .._lib import Intrinsic

_logger = logging.getLogger(__name__)

_MAX_BLOCKS = 200_000
_MAX_VISITS = 1_000_000

_BITWISE_OPS = frozenset({BinOp.LSHIFT, BinOp.RSHIFT, BinOp.BITAND, BinOp.BITOR, BinOp.BITXOR})
_ELEMENTWISE_OPS = frozenset({BinOp.ADD, BinOp.SUB, BinOp.MUL, BinOp.DIV})


@dataclass(frozen=True, slots=True)
class _PropertyRead:
    """A component attribute read that resolved to a ``@property`` getter, to be desugared into a bound call."""

    getter: object  # a ``MethodType(fget, component)`` bound to the exact receiver


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


@dataclass(slots=True)
class _Env:
    """One abstract environment: Place -> Fact, absent meaning unbound-never-touched."""

    facts: dict[Place, Fact] = field(default_factory=dict)

    def copy(self) -> "_Env":
        return _Env(dict(self.facts))

    def get(self, place: Place) -> Fact:
        return self.facts.get(place, _UNBOUND)

    def set(self, place: Place, fact: Fact) -> None:
        self.facts[place] = fact

    def join_with(self, other: "_Env", origin: OriginStack, default: "Callable[[Place], Fact] | None" = None) -> bool:
        # Per-place joins are independent, so a rejection defers until every place has joined.
        changed = False
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


class _UnrollRestart(Exception):
    """
    A loop header's iterable fact descended past the already-unrolled shape mid-round: the round must rerun
    with the joined fact seeded at the header, so the next unroll builds the stable shape in one pass.
    """

    def __init__(self, header: BlockId, fact: Fact) -> None:
        super().__init__(header)
        self.header = header
        self.fact = fact


class CallLowering(enum.Enum):
    """How a PyCall surviving in the residual graph lowers; expanded calls no longer exist as calls."""

    FOLDED = enum.auto()  # a concrete static fold: the destination fact is Known, nothing to emit
    CAST = enum.auto()  # a scalar float()/int()/bool() cast; same-kind-vs-conversion is decided by the FINAL facts
    INTRINSIC = enum.auto()  # a registered hardware intrinsic: ``intrinsic`` carries the resolved registry match
    CONVERSION = enum.auto()  # list()/tuple() over an aggregate: the argument's leaves re-flavor onto the result
    CONSTRUCTION = enum.auto()  # a record built structurally: argument cells install into per-field windows


@dataclass(frozen=True, slots=True)
class CallPlan:
    lowering: CallLowering
    intrinsic: "Intrinsic | None" = None  # the resolved registry match for INTRINSIC
    construction: "tuple[BindingId | None, ...] | None" = None  # per-field source bindings; None = default-filled


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
    subscript_plans: dict[BindingId, tuple[int, ...]] = field(default_factory=dict)
    route_plans: dict[BindingId, tuple[int, ...]] = field(default_factory=dict)
    store_order: list[StateLeaf] = field(default_factory=list)
    runtime_state: set[StateLeaf] = field(default_factory=set)
    state_livein: dict[StateLeaf, Fact] = field(default_factory=dict)
    state_resets: dict[StateLeaf, "StaticValue | str"] = field(default_factory=dict)
    provenance: dict[int, tuple[str, ...]] = field(default_factory=dict)


def _validate(result: ResidualUnit, concrete_calls: set[int]) -> None:
    for block_id in sorted(result.executable_blocks, key=lambda block_id: block_id.index):
        block = result.unit.blocks[block_id]
        for op in block.ops:
            assert not isinstance(op, PyCall) or id(op) in concrete_calls, f"{block_id}: unexpanded call survived"
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
        self._state_livein: dict[StateLeaf, Fact] = {}
        self._discovered_stores: set[tuple[BlockId, StateLeaf]] = set()
        self._concrete_calls: set[int] = set()
        self._intrinsic_calls: set[int] = set()
        self._cast_calls: set[int] = set()  # runtime float()/int()/bool() casts (identity or conversion at emission)
        self._conversion_calls: set[int] = set()  # list()/tuple() layout conversions over aggregates
        self._subscript_selections: dict[int, tuple[int, ...]] = {}  # op id -> source leaf ordinals
        self._conversion_routes: dict[int, tuple[int, ...]] = {}  # op id -> permuted source ordinals (transpose)
        self._construction_calls: dict[int, tuple[BindingId | None, ...]] = {}  # record builds: per-field sources
        # Schema and default snapshots, one per class per ANALYSIS (never per visit): a mutable field default
        # (an eq=False record holding a list, say) must not move a fact between fixpoint visits or into the
        # emission replay. Defaults are admitted LAZILY at the first construction that actually omits the field
        # (Python never observes an overridden default). The pinned class reference keeps ids stable.
        self._construction_schemas: dict[int, tuple[type, tuple[FieldSchema, ...]]] = {}
        self._default_snapshots: dict[tuple[int, str], BoundFact] = {}
        self._unroll_cache: dict[BlockId, tuple[Fact, BlockId]] = {}
        self._unroll_seeds: dict[BlockId, Fact] = {}  # survives rounds: the joined facts of restarted headers
        self._store_origins: dict[StateLeaf, OriginStack] = {}
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
        from immutable templates; the final round's facts are computed under stable typing.
        """
        for round_index in range(1_000):
            try:
                result = self.analyze(param_facts)
            except _UnrollRestart as restart:
                # Seeds key root-template block ids, which the per-round instantiation preserves; facts descend
                # monotonically, so reseeding terminates within the round fuel.
                self._unroll_seeds[restart.header] = restart.fact
                self._reset_round()
                _logger.info("state round %d: unroll reseeded at %s", round_index + 1, restart.header)
                continue
            exit_env = result.block_in.get(result.unit.exit, _Env())
            reachable = _coreachable(result.unit, result.unit.exit, result.executable_edges)
            new_w = set(self._runtime_state)
            for block_id, leaf in self._discovered_stores:
                if block_id in result.executable_blocks and block_id in reachable:
                    new_w.add(leaf)
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
            deferred.raise_if_deferred()
            if new_w == self._runtime_state and new_d == self._state_livein:
                _logger.info("state fixpoint stable after %d round(s): %d runtime leaves", round_index + 1, len(new_w))
                for header_id, (_, chain_entry) in self._unroll_cache.items():
                    header = result.unit.blocks[header_id]
                    assert isinstance(header.terminator, StaticFor)
                    header.terminator = Jump(chain_entry, header.terminator.origin)
                self._reject_executable_fails(result)
                _validate(
                    result,
                    self._concrete_calls
                    | self._intrinsic_calls
                    | self._cast_calls
                    | self._conversion_calls
                    | set(self._construction_calls),
                )
                self._finalize(result)
                return result
            self._runtime_state = new_w
            self._state_livein = new_d
            self._reset_round()
            _logger.info("state round %d: %d runtime leaves, %d live-in facts", round_index + 1, len(new_w), len(new_d))
        raise AnalysisRejection("state fixpoint failed to stabilize", (Origin(self._root_template.name, 0, 0),))

    def _reset_round(self) -> None:
        self._block_ancestry = {}
        self._discovered_stores = set()
        self._store_origins = {}
        self._concrete_calls = set()
        self._intrinsic_calls = set()
        self._cast_calls = set()
        self._conversion_calls = set()
        self._subscript_selections = {}
        self._conversion_routes = {}
        self._construction_calls = {}
        self._unroll_cache = {}

    def _read_attribute_snapshot(self, owner: object, name: str) -> tuple[object, StaticValue | None]:
        """
        One live read AND one admission per (owner, attribute) per analysis: every later consultation -- W/D
        rounds, reset facts, namespace lookups, the final-plan replay -- sees the first read's ADMITTED value,
        so neither a drifting live object nor a mutated referent (admission snapshots contents at admit time)
        can move a fact after it is first formed. The owner reference pins the id against reuse for the memo's
        lifetime. AttributeError propagates to the caller's located rejection.
        """
        key = (id(owner), name)
        hit = self._component_reads.get(key)
        if hit is not None and hit[0] is owner:
            return hit[1], hit[2]
        value = getattr(owner, name)
        admitted = admit(value)
        self._component_reads[key] = (owner, value, admitted)
        return value, admitted

    def _state_origin(self, leaf: StateLeaf) -> OriginStack:
        """
        State rejections locate at the leaf's first store: __init__ is never analyzed, so the store that
        promoted the leaf is the line the user can act on.
        """
        return self._store_origins.get(leaf, (Origin(self._root_template.name, 0, 0),))

    def _snapshot_leaf(self, leaf: StateLeaf) -> Fact:
        current, admitted = self._walk_snapshot(leaf)
        return normalize_static(admitted) if admitted is not None else Reference(current)

    def _walk_snapshot(self, leaf: StateLeaf) -> tuple[object, StaticValue | None]:
        current: object = leaf.component
        admitted: StaticValue | None = None
        for attribute in leaf.path:
            try:
                current, admitted = self._read_attribute_snapshot(current, attribute)
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
                if not shape:
                    raise AnalysisRejection(
                        f"state attribute '{name}' has a 0-dimensional array reset; persist a scalar instead",
                        origin,
                    )
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

    def _admit_state_store(self, leaf: StateLeaf, reset: Fact, src: Fact, origin: OriginStack) -> Fact:
        """
        An aggregate-flavored store against the reset-derived schema: the reset fixes the container flavor and
        the exact geometry (the next transaction reconstructs that Python object, so a structural degrade would
        lie), and each cell's carrier kind comes from the stabilized live-in with the reset as the first-round
        fallback -- an integer leaf promotes into a float cell exactly as the scalar store does, and a
        bool/numeric mix is a located rejection rather than a silent reinterpretation.
        """
        name = ".".join(leaf.path)
        if not isinstance(reset, AggregateFact):
            if isinstance(reset, Reference):
                raise AnalysisRejection(
                    f"state attribute '{name}' cannot persist: its reset is not admissible "
                    "(a plain numpy array or a flat list of scalars is required)",
                    origin,
                )
            raise AnalysisRejection(
                f"state attribute '{name}' persists a scalar; an aggregate cannot be stored into it", origin
            )
        if not isinstance(src, AggregateFact):
            raise AnalysisRejection(
                f"state attribute '{name}' persists an aggregate; a scalar cannot be stored into it", origin
            )
        assert isinstance(reset.layout, (ListLayout, ArrayLayout))
        if type(src.layout) is not type(reset.layout):
            flavor = "numpy array" if isinstance(reset.layout, ArrayLayout) else "list"
            raise AnalysisRejection(
                f"state attribute '{name}' persists a {flavor}; store the same container flavor", origin
            )
        geometry_matches = (
            src.layout.shape == reset.layout.shape
            if isinstance(reset.layout, ArrayLayout) and isinstance(src.layout, ArrayLayout)
            else src.layout == reset.layout
        )
        if not geometry_matches:
            described = (
                f"a {'x'.join(map(str, reset.layout.shape))} array"
                if isinstance(reset.layout, ArrayLayout)
                else f"a {outer_arity(reset.layout)}-element vector"
            )
            raise AnalysisRejection(
                f"state attribute '{name}' persists {described}; the stored value has an incompatible shape",
                origin,
            )
        livein = self._state_livein.get(leaf)
        cells: list[AtomicFact] = []
        for ordinal, stored in enumerate(src.leaves):
            if isinstance(stored, Reference):
                raise AnalysisRejection(f"state attribute '{name}' cannot persist an object reference", origin)
            slot_cell = livein.leaves[ordinal] if isinstance(livein, AggregateFact) else reset.leaves[ordinal]
            slot_sem = slot_cell.type if isinstance(slot_cell, Residual) else _residual_type(slot_cell.value)  # type: ignore[union-attr]
            stored_sem = stored.type if isinstance(stored, Residual) else _residual_type(stored.value)
            if stored_sem is None or (slot_sem is SemType.BOOL) != (stored_sem is SemType.BOOL):
                raise AnalysisRejection(
                    f"state attribute '{name}' stores an incompatible type at cell {ordinal}", origin
                )
            if slot_sem is SemType.FLOAT and stored_sem is SemType.INT:
                promoted = _float_promoted(stored, origin)
                assert isinstance(promoted, (Known, Residual))
                cells.append(promoted)
            else:
                cells.append(stored)
        layout: AggregateLayout = reset.layout
        if isinstance(reset.layout, ArrayLayout):
            # The stored cells fix the read-back dtype: an int-reset slot holding promoted float cells reads
            # back as a float array (the layout carried by the fact must agree with its own cell kinds).
            sems = set()
            for cell in cells:
                assert isinstance(cell, (Known, Residual)), cell
                sems.add(cell.type if isinstance(cell, Residual) else _residual_type(cell.value))
            dtype = (
                ArrayDType.BOOL
                if sems == {SemType.BOOL}
                else ArrayDType.FLOAT if SemType.FLOAT in sems else ArrayDType.INT
            )
            layout = ArrayLayout(reset.layout.shape, dtype)
        return AggregateFact(layout, tuple(cells))

    def _reject_executable_fails(self, result: ResidualUnit) -> None:
        """
        An executable Fail terminator sits on a path taken unconditionally (or under a residual guard the
        hardware cannot signal): a located rejection carrying the raise's own message, with any f-string
        interpolations rendered from the compile-time facts at the raise site. The walk is a preorder over
        executable edges (then-arm first), so the raise reported is the first one execution can reach —
        unroll-clone block indices do not follow iteration order (a reversed range), so index order would
        misreport which raise fires.
        """
        stack = [result.unit.entry]
        seen: set[BlockId] = set()
        while stack:
            block_id = stack.pop()
            if block_id in seen or block_id not in result.executable_blocks:
                continue
            seen.add(block_id)
            block = result.unit.blocks[block_id]
            terminator = block.terminator
            if isinstance(terminator, Fail):
                env = result.block_in[block_id].copy()
                for index, op in enumerate(block.ops):
                    self._transfer(result.unit, block, index, op, env)
                raise AnalysisRejection(self._render_fail(terminator.parts, env), terminator.origin)
            successors: list[BlockId] = []
            match terminator:
                case Jump(target=target):
                    successors = [target]
                case Branch(then_target=then_target, else_target=else_target):
                    successors = [then_target, else_target]
                case _:
                    pass
            for successor in reversed(successors):
                if (block_id, successor) in result.executable_edges:
                    stack.append(successor)

    def _render_fail(self, parts: tuple[str | BindingId, ...], env: _Env) -> str:
        rendered: list[str] = []
        for part in parts:
            if isinstance(part, str):
                rendered.append(part)
                continue
            concrete = _concrete_fact(env.get(Local(part)))
            if concrete is None:
                return "raise with a runtime-interpolated message"
            rendered.append(format(as_python(concrete)))
        return "".join(rendered)

    def _finalize(self, result: ResidualUnit) -> None:
        """
        One replay of the transfer over the stabilized graph fills the emission plan: the authoritative fact per
        binding (temporaries are write-once, so one pass records each), a typed plan per surviving call (keyed by
        the call's destination binding, never by op identity), and state leaves in first-store SOURCE order (the
        order key is the storing op's origin -- a store nested in a branch has a higher block id than a later
        top-level store yet comes first in the source text). The replay does not mutate the graph: every call is
        already expanded, folded, or classified. Emission consumes only this plan.
        """
        first_store: dict[StateLeaf, tuple[int, int]] = {}
        for block_id in result.executable_blocks:
            block = result.unit.blocks[block_id]
            env = result.block_in[block_id].copy()
            for index, op in enumerate(block.ops):
                if isinstance(op, PyStoreAttr):
                    obj_fact = env.get(Local(op.obj))
                    if isinstance(obj_fact, Reference):
                        leaf = StateLeaf(obj_fact.obj, (op.name,))
                        position = (op.origin[0].line, op.origin[0].column)
                        if leaf not in first_store or position < first_store[leaf]:
                            first_store[leaf] = position
                if isinstance(op, PyCall):
                    result.call_plans[op.dst] = self._call_plan(op, env)
                self._transfer(result.unit, block, index, op, env)
                if isinstance(op, PySubscript) and id(op) in self._subscript_selections:
                    result.subscript_plans[op.dst] = self._subscript_selections[id(op)]
                if isinstance(op, PyCall) and id(op) in self._conversion_routes:
                    result.route_plans[op.dst] = self._conversion_routes[id(op)]
                dst = op_dst(op)
                if dst is not None:
                    result.binding_facts[dst] = env.get(Local(dst))
        entry_env = result.block_in[result.unit.entry]
        for param in result.unit.params:
            result.binding_facts.setdefault(param, entry_env.get(Local(param)))
        result.store_order = sorted(first_store, key=lambda leaf: first_store[leaf])
        result.runtime_state = set(self._runtime_state)
        result.state_livein = dict(self._state_livein)
        for leaf in {*result.runtime_state, *result.store_order, *result.state_livein}:
            raw, admitted = self._walk_snapshot(leaf)
            result.state_resets[leaf] = admitted if admitted is not None else type(raw).__name__
        result.provenance = self._component_provenance()

    def _call_plan(self, call: PyCall, env: _Env) -> CallPlan:
        # Optimistic SCCP may reclassify a cast across revisits (int(y) is an identity while y is still integer and a
        # conversion once the other edge promotes it), so a cast plan deliberately carries no same-kind/cross-kind
        # split: emission decides from the FINAL facts, which only stabilized rounds produce.
        if id(call) in self._conversion_calls:
            return CallPlan(CallLowering.CONVERSION)
        construction = self._construction_calls.get(id(call))
        if construction is not None:
            return CallPlan(CallLowering.CONSTRUCTION, construction=construction)
        if id(call) in self._cast_calls:
            return CallPlan(CallLowering.CAST)
        if id(call) in self._intrinsic_calls:
            from .._lib import Intrinsic, resolve

            callee_fact = env.get(Local(call.callee))
            assert isinstance(callee_fact, Reference)
            match = resolve(callee_fact.obj)
            assert isinstance(match, Intrinsic)
            return CallPlan(CallLowering.INTRINSIC, match)
        assert id(call) in self._concrete_calls, "an unclassified call survived validation"
        return CallPlan(CallLowering.FOLDED)

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
        origin = (Origin(unit.name, 0, 0),)
        block_in: dict[BlockId, _Env] = {unit.entry: _Env()}
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
        self._component_edges = set()
        executable_edges: set[tuple[BlockId, BlockId]] = set()
        executable_blocks: set[BlockId] = set()

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
            executable_blocks.add(block_id)
            env = block_in[block_id].copy()
            block = unit.blocks[block_id]
            index = 0
            while index < len(block.ops):
                op = block.ops[index]
                expanded = self._transfer(unit, block, index, op, env)
                if expanded:
                    continue  # the graph changed under us: re-run this op slot (now a different op)
                index += 1
            successors = self._resolve_terminator(unit, block, env)
            assert block.terminator is not None
            join_origin = block.terminator.origin
            for successor in successors:
                edge = (block.id, successor)
                target_env = block_in.get(successor)
                if target_env is None:
                    block_in[successor] = env.copy()
                    executable_edges.add(edge)
                    worklist.append(successor)
                elif edge not in executable_edges:
                    executable_edges.add(edge)
                    if target_env.join_with(env, join_origin, edge_default) or successor not in executable_blocks:
                        worklist.append(successor)
                else:
                    if target_env.join_with(env, join_origin, edge_default):
                        worklist.append(successor)
        return ResidualUnit(unit, block_in, executable_blocks, executable_edges)

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
                env.set(place, env.get(Local(src)))
            case UnbindPlace(place=place, checked=checked):
                if checked and isinstance(env.get(place), (Unbound, MaybeUnbound)):
                    raise AnalysisRejection(f"'{place}' may be unbound at this del (Python would raise)", op.origin)
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
                    import numpy as np

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
                    env.set(Local(dst), self._elementwise_compare(rel, lhs_fact, rhs_fact, op.origin))
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
                    # absorbing constant (``float(x) or True``) is still irreconcilable and must reject, never fold away.
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
                    if isinstance(obj_fact.layout, ArrayLayout) and not obj_fact.layout.shape:
                        raise AnalysisRejection("len() of a 0-dimensional array is undefined", op.origin)
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
                if (
                    isinstance(receiver_fact, AggregateFact)
                    and isinstance(receiver_fact.layout, ArrayLayout)
                    and name in ("flatten", "ravel", "reshape")
                ):
                    method_key = (obj, name)
                    if method_key not in self._array_methods:
                        self._array_methods[method_key] = _ArrayMethod(obj, name)
                    env.set(Local(dst), Reference(self._array_methods[method_key]))
                    return False
                if (
                    isinstance(receiver_fact, AggregateFact)
                    and isinstance(receiver_fact.layout, ArrayLayout)
                    and name == "T"
                ):
                    # Transpose is a pure structural relayout (a permutation of the same leaves), so ``.T``
                    # rewrites to the spelled np.transpose call and both spellings share one lowering.
                    import numpy as np

                    callee = self._fresh_temp()
                    block.ops[index : index + 1] = [
                        LoadRef(callee, np.transpose, op.origin),
                        PyCall(dst, callee, (obj,), (), op.origin),
                    ]
                    return True
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
                env.set(Local(dst), attr)
            case PyStoreAttr(obj=obj, name=name, src=src):
                obj_fact = env.get(Local(obj))
                if not isinstance(obj_fact, Reference):
                    raise AnalysisRejection("attribute store on a non-component value", op.origin)
                if isinstance(obj_fact.obj, (types.ModuleType, type)):
                    # A module/class is a compile-time namespace, not runtime state: mutating it would make later
                    # reads (which snapshot the live object) disagree with the store. Reject, as production does.
                    raise AnalysisRejection("assignment to a module or class attribute is not supported", op.origin)
                _reject_attribute_hooks(type(obj_fact.obj), op.origin)
                _reject_descriptor(type(obj_fact.obj), name, op.origin)
                src_fact = env.get(Local(src))
                if isinstance(src_fact, Reference):
                    # Storing a component/sub-object into an attribute would change the component topology per
                    # transaction (a slot's owner is fixed at the initial snapshot); reject it at the store, located.
                    raise AnalysisRejection(
                        f"component member '{name}' cannot be rebound; component topology is fixed", op.origin
                    )
                leaf = StateLeaf(obj_fact.obj, (name,))
                recorded = self._store_origins.get(leaf)
                if recorded is None or (op.origin[0].line, op.origin[0].column) < (
                    recorded[0].line,
                    recorded[0].column,
                ):
                    self._store_origins[leaf] = op.origin
                self._discovered_stores.add((block.id, leaf))
                reset_fact = self._state_reset_fact(leaf)
                if isinstance(reset_fact, AggregateFact) or isinstance(src_fact, AggregateFact):
                    src_fact = self._admit_state_store(leaf, reset_fact, src_fact, op.origin)
                elif (
                    (slot_sem := self._leaf_kind(leaf)) is not None
                    and isinstance(src_fact, (Known, Residual))
                    and (
                        stored_sem := (
                            src_fact.type if isinstance(src_fact, Residual) else _residual_type(src_fact.value)
                        )
                    )
                    is not None
                    and (slot_sem is SemType.BOOL) != (stored_sem is SemType.BOOL)
                ):
                    raise AnalysisRejection(f"state attribute '{name}' stores an incompatible type", op.origin)
                elif (
                    self._leaf_kind(leaf) is SemType.FLOAT
                    and isinstance(src_fact, (Known, Residual))
                    and _numeric_sem(src_fact) is SemType.INT
                ):
                    # The datapath store rounds the integer into the float slot, so the fact a read-back sees must be
                    # the promoted (rounded) float -- an exact-integer fact here would fold against a value the slot
                    # does not hold. On the first W/D round the slot kind falls back to the reset's; the fixed point
                    # then converges with the same descending joins as the live-in itself.
                    src_fact = _float_promoted(src_fact, op.origin)
                env.set(leaf, src_fact)
            case PyCall(dst=dst, callee=callee):
                return self._expand_call(unit, block, index, op, env)
        return False

    def _leaf_kind(self, leaf: StateLeaf) -> SemType | None:
        """The numeric kind the slot stores at: the current live-in's, else the reset's; None when unresolvable."""
        fact = self._state_livein.get(leaf)
        if fact is None:
            try:
                fact = self._state_reset_fact(leaf)
            except AnalysisRejection:
                return None
        return _numeric_sem(fact)

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
        constant is a located rejection where numpy raises OverflowError, and 0-d operands yield the SCALAR sort
        (numpy returns np.float64/np.int64, never a 0-d array). A runtime-integer result rejects until the integer
        sprint: the scalar integer datapath saturates where numpy wraps, so lowering it would diverge leafwise.
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
        if _is_array_fact(lhs) and lhs.layout.shape == ():  # type: ignore[union-attr]
            lhs = lhs.leaves[0]  # type: ignore[union-attr]  # numpy broadcasts a 0-d array like a scalar
        if _is_array_fact(rhs) and rhs.layout.shape == ():  # type: ignore[union-attr]
            rhs = rhs.leaves[0]  # type: ignore[union-attr]
        promotes = bin_op is BinOp.DIV
        if not (_is_array_fact(lhs) or _is_array_fact(rhs)):
            # Both operands were 0-d arrays: the result is the scalar sort, exactly as numpy yields it.
            return self._fold_binary(
                lambda a, b: static_binop(bin_op, a, b), lhs, rhs, origin, promotes_to_float=promotes
            )
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
        return tuple(dimensions)

    def _elementwise_compare(self, rel: RelationalOp, lhs: Fact, rhs: Fact, origin: OriginStack) -> Fact:
        """
        An elementwise NUMERIC comparison producing a boolean mask: the same same-shape/scalar pairing as
        arithmetic, each leaf pair taking the scalar comparison rule (numpy provenance rides the fold).
        Boolean operands keep the scalar doctrine rather than gaining a blanket array admission.
        """
        if _is_array_fact(lhs) and lhs.layout.shape == ():  # type: ignore[union-attr]
            lhs = lhs.leaves[0]  # type: ignore[union-attr]
        if _is_array_fact(rhs) and rhs.layout.shape == ():  # type: ignore[union-attr]
            rhs = rhs.leaves[0]  # type: ignore[union-attr]
        if not (_is_array_fact(lhs) or _is_array_fact(rhs)):
            return self._fold_binary(lambda a, b: static_compare(rel, a, b), lhs, rhs, origin, default=SemType.BOOL)
        for side in (lhs, rhs):
            if isinstance(side, AggregateFact) and not isinstance(side.layout, ArrayLayout):
                raise AnalysisRejection(
                    "elementwise comparison mixes an array with a non-array container; wrap it in np.array(...)",
                    origin,
                )
        if SemType.BOOL in (self._elementwise_side_sem(lhs, origin), self._elementwise_side_sem(rhs, origin)):
            raise AnalysisRejection("an elementwise comparison requires numeric operands", origin)
        arrays = [side for side in (lhs, rhs) if isinstance(side, AggregateFact)]
        shapes = [side.layout.shape for side in arrays if isinstance(side.layout, ArrayLayout)]
        if len(set(shapes)) > 1:
            raise AnalysisRejection(
                f"elementwise comparison on mismatched shapes {shapes[0]} and {shapes[1]} "
                "(only a scalar broadcasts)",
                origin,
            )
        leaves: list[AtomicFact] = []
        for ordinal in range(leaf_count(arrays[0].layout)):
            left = lhs.leaves[ordinal] if isinstance(lhs, AggregateFact) else lhs
            right = rhs.leaves[ordinal] if isinstance(rhs, AggregateFact) else rhs
            leaf = self._fold_binary(lambda a, b: static_compare(rel, a, b), left, right, origin, default=SemType.BOOL)
            assert isinstance(leaf, (Known, Residual)), leaf
            leaves.append(leaf)
        return AggregateFact(ArrayLayout(shapes[0], ArrayDType.BOOL), tuple(leaves))

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
        if operand.layout.shape == ():
            return leaves[0]  # numpy: unary +/- on a 0-d array also yields the scalar sort
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
        # promotion is not modelled in the datapath, so an explicit cast is required. A compile-time-known negative shift
        # count refuses (Python raises); a runtime count is the hardware's documented reverse-shift deviation. A
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
        import numpy as np

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
            # per analysis visit, in the replay, and again at emission), reading reset-time state the kernel's
            # writes never touch. The state machinery is the honest path: index with int(self.attr).
            raise AnalysisRejection("an object subscript index is not supported", origin)
        if isinstance(obj, AggregateFact) and isinstance(index, Known):
            if isinstance(obj.layout, RecordLayout):
                raise AnalysisRejection("a record is not subscriptable; access its fields by name", origin)
            if isinstance(obj.layout, ArrayLayout) and not obj.layout.shape:
                raise AnalysisRejection(
                    "a 0-dimensional array cannot be indexed; convert it with float() instead", origin
                )
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
                self._subscript_selections[id(op)] = tuple(ordinals)
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
            return obj.child(position + arity if position < 0 else position)
        if isinstance(obj, Reference):
            # A live object's __getitem__ would read reset-time attribute state outside the state machinery.
            raise AnalysisRejection("subscript of an object is not supported", origin)
        if isinstance(obj, Known) and isinstance(index, Known):
            return self._concrete_subscript(obj.value, index, origin)
        raise AnalysisRejection("subscript of a runtime value is not supported yet", origin)

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
        self._subscript_selections[id(op)] = tuple(ordinals)
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
        return _taint_lost(normalize_static(admitted), _lost_scalar_pools([Known(value)]))

    def _attribute(self, env: _Env, obj: Fact, name: str, origin: OriginStack) -> "Fact | _PropertyRead":
        if isinstance(obj, AggregateFact):
            if isinstance(obj.layout, RecordLayout):
                names = [field for field, _ in obj.layout.fields]
                if name in names:
                    return obj.child(names.index(name))  # record field projection works on runtime leaves too
                # A non-field attribute (a property, a method) would execute user code on the reconstruction,
                # whose provenance the compiler cannot fully vouch for (an enum field rebuilds as its base value).
                raise AnalysisRejection(f"record attribute '{name}' is not supported (only field access)", origin)
            if isinstance(obj.layout, ListLayout):
                raise AnalysisRejection(
                    f"list method '{name}' is not supported (lists are immutable values here); rebind with + instead",
                    origin,
                )
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
                    attribute, admitted = self._read_attribute_snapshot(component, name)
                except AttributeError as error:
                    raise AnalysisRejection(str(error), origin) from None
                return normalize_static(admitted) if admitted is not None else Reference(attribute)
            _reject_attribute_hooks(type(component), origin)
            class_attribute = _mro_attribute_of(type(component), name)
            if type(class_attribute) is property:  # an exact property (not a subclass) wins over any __dict__ entry
                if not isinstance(class_attribute.fget, types.FunctionType):
                    raise AnalysisRejection(f"property {name!r} has an unsupported getter", origin)
                # Bind the getter to the exact receiver so its ``self.stored`` reads resolve to the same StateLeaf/Known
                # a direct read would, and so recursion identity and the ``self`` parameter bind correctly.
                return _PropertyRead(types.MethodType(class_attribute.fget, component))
            _reject_descriptor(type(component), name, origin)
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
                and not isinstance(class_attribute, types.MemberDescriptorType)
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
                snapshot, admitted = self._read_attribute_snapshot(component, name)
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
            if _is_list_fact(obj):
                raise AnalysisRejection(
                    f"list method '{name}' is not supported (lists are immutable values here); rebind with + instead",
                    origin,
                )
            base_receiver = as_python(strip_source(obj.value))  # base-type surface: enum attributes never resolve
            try:
                concrete = getattr(base_receiver, name)
            except AttributeError as error:
                if isinstance(obj.value, (MetaInt, StaticStr)) and not isinstance(obj.value.source, ScalarOrigin):
                    raise AnalysisRejection(
                        f"enum member attribute '{name}' is not supported (the member folds as its base value)",
                        origin,
                    ) from None
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
                if isinstance(obj.value, StaticStr) and name in ("format", "format_map"):
                    # format's conversions (!r) observe the repr of erasure-reconstructed arguments.
                    raise AnalysisRejection(f"str.{name} is not supported in a kernel", origin)
                value_key = (obj.value, name)
                if value_key not in self._value_methods:
                    # The method comes from the BASE TYPE (looked up above, so an enum-defined method never
                    # resolves) but binds onto the FAITHFUL receiver: an identity-preserving method (partition's
                    # no-match head returns self) then yields the retained member itself, and re-admission keeps
                    # it, exactly as Python.
                    slot = _mro_attribute_of(type(base_receiver), name)
                    faithful = as_python(obj.value)
                    getter = getattr(type(slot), "__get__", None)
                    bound = getter(slot, faithful, type(faithful)) if getter is not None else slot
                    self._value_methods[value_key] = bound
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
                return [then_target, else_target]
            case StaticFor():
                iterable_fact = env.get(Local(terminator.iterable))
                seed = self._unroll_seeds.get(block.id)
                if seed is not None:
                    iterable_fact = join_facts(seed, iterable_fact, terminator.origin)
                cached = self._unroll_cache.get(block.id)
                if cached is not None:
                    cached_fact, chain_entry = cached
                    if _same_fact(iterable_fact, cached_fact):
                        return [chain_entry]
                    raise _UnrollRestart(block.id, join_facts(cached_fact, iterable_fact, terminator.origin))
                chain_entry = self._unroll(unit, block, terminator, iterable_fact)
                self._unroll_cache[block.id] = (iterable_fact, chain_entry)
                return [chain_entry]
            case Fail() | UnitExit():
                return []
        raise AssertionError(terminator)

    # ------------------------------------ StaticFor unrolling ------------------------------------

    def _unroll(self, unit: FunctionUnit, header: Block, loop: StaticFor, iterable: Fact) -> BlockId:
        if isinstance(iterable, Reference):
            # A live object's __iter__/__len__ would run against reset-time state, twice (analysis + replay).
            raise AnalysisRejection("iteration over an object is not supported", loop.origin)
        per_trip: list[Known | Reference | int] = []
        if isinstance(iterable, AggregateFact):
            if isinstance(iterable.layout, RecordLayout):
                # Materializing would drive Python's iteration protocol (a user __len__/__getitem__/__iter__) on
                # the reconstruction -- a demonstrated wrong-value and non-termination hazard.
                raise AnalysisRejection("iteration over a record is not supported", loop.origin)
            if isinstance(iterable.layout, ArrayLayout) and not iterable.layout.shape:
                raise AnalysisRejection("iteration over a 0-dimensional array is undefined", loop.origin)
            trip_count = outer_arity(iterable.layout)
            if trip_count > UNROLL_THRESHOLD:  # sized before materializing: a 32k table must reject instantly
                raise AnalysisRejection(
                    f"trip count {trip_count} exceeds the unroll threshold {UNROLL_THRESHOLD}; a counted "
                    "back-edge loop is not supported yet",
                    loop.origin,
                )
            for position in range(trip_count):
                child: Fact = iterable.child(position)
                if isinstance(child, AggregateFact):
                    materialized = materialize_static(child)
                    child = Known(materialized) if materialized is not None else child
                if isinstance(child, (Known, Reference)):
                    per_trip.append(child)
                else:
                    # A runtime element: the trip binds through a synthesized projection prelude, so the child's
                    # cells (a scalar leaf or a whole row) flow exactly as an explicit v[k] would.
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
            prelude.ops.append(StorePlace(loop.target, temp, loop.origin))
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
        self._construction_calls[id(call)] = tuple(mapping)

    def _fresh_temp(self) -> BindingId:
        self._temp_serial += 1
        return BindingId(f"%c{self._temp_serial}", self._temp_serial)

    def _expand_call(self, unit: FunctionUnit, block: Block, index: int, call: PyCall, env: _Env) -> bool:
        import numpy as np

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
                    if isinstance(fact.layout, ArrayLayout) and not fact.layout.shape:
                        raise AnalysisRejection("iteration over a 0-dimensional array is undefined", call.origin)
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
                self._conversion_calls.add(id(call))
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
        target = callee_fact.obj
        match = resolve(target)
        if isinstance(match, Library):
            reduction = any(target is fn for fn in (np.max, np.amax, np.mean))
            if reduction and (len(call.args) != 1 or call.kwargs):
                raise AnalysisRejection(
                    f"np.{getattr(target, '__name__', '?')} supports only the default axis: exactly one array "
                    "argument (reduce the other axis explicitly instead of passing an axis)",
                    call.origin,
                )
            if reduction or any(target is fn for fn in (np.matmul, np.dot, np.trace, np.outer)):
                # The linalg and reduction stubs are defined over arrays only; a scalar/list/tuple operand
                # must not acquire array semantics through the spelled call any more than through an operator.
                for arg in call.args:
                    operand_fact = env.get(Local(arg))
                    if not (isinstance(operand_fact, AggregateFact) and isinstance(operand_fact.layout, ArrayLayout)):
                        if any(target is fn for fn in (np.matmul, np.dot)):
                            raise AnalysisRejection(
                                "the matrix product requires array operands on both sides; a scalar, list, or "
                                "tuple does not acquire matrix semantics (wrap it in np.array(...))",
                                call.origin,
                            )
                        raise AnalysisRejection(
                            f"np.{getattr(target, '__name__', '?')} requires array operands; a scalar, list, or "
                            "tuple does not acquire array semantics (wrap it in np.array(...))",
                            call.origin,
                        )

            # np.sign is int-polymorphic like abs (np.sign of an integer is an integer); its float composite would
            # round subsequent integer arithmetic, and there is no integer sign yet, so an integer operand refuses.
            if target is np.sign and any(
                env.get(Local(arg)) == Residual(SemType.INT)
                or (isinstance(env.get(Local(arg)), Known) and isinstance(env.get(Local(arg)).value, (MetaInt, NpInt)))  # type: ignore[union-attr]
                for arg in call.args
            ):
                raise AnalysisRejection(
                    "an integer operand to np.sign is not yet lowerable; cast to float first", call.origin
                )
            target = match.stub  # a composite library stub inlines exactly like a user function
        elif isinstance(match, Intrinsic):
            argument_facts = [env.get(Local(arg)) for arg in call.args] + [
                env.get(Local(value)) for _, value in call.kwargs
            ]
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
                    numpy_spelling = getattr(target, "__module__", "").startswith("numpy")
                    env.set(Local(call.dst), Known(NpBool(verdict) if numpy_spelling else StaticBool(verdict)))
                    self._concrete_calls.add(id(call))
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
                self._intrinsic_calls.add(id(call))
                return False
        if not isinstance(target, (types.FunctionType, types.MethodType)) and not (
            hasattr(type(target), "__call__")
            and isinstance(getattr(type(target), "__call__", None), types.FunctionType)
        ):
            # A builtin (range, float, abs...) or a fully-static intrinsic evaluates concretely under the snapshot
            # doctrine; its runtime-operand form was already routed to an HIR operation above.
            argument_facts = [env.get(Local(arg)) for arg in call.args]
            keyword_facts = [(keyword, env.get(Local(value))) for keyword, value in call.kwargs]
            if target is getattr:
                # Trimmed (scope ruling T1): the static name getattr would require anyway makes it pure spelling
                # redundancy over the dotted access, and letting it near the concrete path was a demonstrated
                # miscompile habitat. The arm stays as the refusal site so the guidance is specific. Identity
                # comparisons throughout: ``target`` may be an unhashable shadow of a builtin name, which must
                # miss cleanly.
                raise AnalysisRejection(
                    "getattr is not supported in a kernel; spell the attribute access directly (x.name)",
                    call.origin,
                )
            if (
                target is np.transpose
                and not keyword_facts
                and len(argument_facts) == 1
                and isinstance(argument_facts[0], AggregateFact)
                and isinstance(argument_facts[0].layout, ArrayLayout)
            ):
                # A pure structural relayout: the same leaves under the reversed shape, recorded as a route
                # plan (source ordinal per result cell). Precedes admission: nothing crosses.
                pivoted = argument_facts[0]
                pivoted_layout = pivoted.layout
                assert isinstance(pivoted_layout, ArrayLayout)
                routes = _transpose_routes(pivoted_layout.shape)
                env.set(
                    Local(call.dst),
                    AggregateFact(
                        ArrayLayout(pivoted_layout.shape[::-1], pivoted_layout.dtype),
                        tuple(pivoted.leaves[k] for k in routes),
                    ),
                )
                self._conversion_calls.add(id(call))
                self._conversion_routes[id(call)] = routes
                return False
            if target is np.ndim and not keyword_facts and len(argument_facts) == 1:
                # Deliberately narrow (np.ndim of a LIST would observe structure the fact model erases at
                # atomic leaves): a numeric scalar is rank 0, an array is its layout rank, everything else
                # rejects. The linalg stubs probe ranks through this spelling.
                probed = argument_facts[0]
                if isinstance(probed, AggregateFact) and isinstance(probed.layout, ArrayLayout):
                    rank = len(probed.layout.shape)
                elif isinstance(probed, Residual) or (
                    isinstance(probed, Known) and _residual_type(probed.value) is not None
                ):
                    rank = 0
                else:
                    raise AnalysisRejection("np.ndim of this value is not supported here", call.origin)
                admitted_rank = admit(rank)
                assert admitted_rank is not None
                env.set(Local(call.dst), Known(admitted_rank))
                self._concrete_calls.add(id(call))
                return False
            if isinstance(target, type) and is_dataclass(target):
                # Record construction is STRUCTURAL, never an evaluation: the layout is the class's validated
                # field schema and the children are the argument facts THEMSELVES -- runtime leaves, enum
                # provenance, LOST taint, and reference leaves all ride through untouched, and no host code
                # (not even the generated __init__) ever runs. Like the getattr rewrite above, this precedes
                # the admission harness: there is nothing to admit because nothing crosses.
                self._expand_construction(target, call, env)
                return False
            if (
                target is isinstance
                and not keyword_facts
                and len(argument_facts) == 2
                and isinstance(argument_facts[0], AggregateFact)
                and isinstance(argument_facts[0].layout, RecordLayout)
            ):
                # A record subject answers from the layout's class identity alone -- runtime, reference, and
                # oversized-range leaves included, because no field is consulted -- so like construction this
                # precedes admission: nothing reconstructs and nothing crosses. Raw type metadata and identity
                # scans run no user code (an instance-side __class__ override, a metaclass __mro__ hook, or a
                # metaclass __eq__ under tuple membership could otherwise warp the verdict); a class overriding
                # __class__ refuses outright, since CPython's check consults the observed __class__ where the
                # real type misses.
                klass = argument_facts[0].layout.klass
                mro = type.__getattribute__(klass, "__mro__")
                if any("__class__" in c.__dict__ for c in mro if c is not object):
                    raise AnalysisRejection(
                        "isinstance of a record whose class overrides __class__ is not supported", call.origin
                    )
                try:
                    record_kinds = validate_classinfo(argument_facts[1])
                except FoldRefusal as refusal:
                    raise AnalysisRejection(str(refusal), call.origin) from None
                verdict = any(entry is kind for kind in record_kinds for entry in mro)
                env.set(Local(call.dst), Known(StaticBool(verdict)))
                self._concrete_calls.add(id(call))
                return False
            # Concrete evaluation is a CLOSED WHITELIST behind one door: the fold admission harness. The
            # analyzer contributes only what the harness cannot know -- per-Analyzer minted-method identity and
            # library-registry resolution -- and locates the refusal at the call origin.
            minted = any(target is method for method in self._value_methods.values())
            try:
                admit_call(
                    target,
                    argument_facts,
                    [fact for _, fact in keyword_facts],
                    minted=minted,
                    registry_resolved=resolve(target) is not None,
                )
            except FoldRefusal as refusal:
                if refusal.library_diagnostic:
                    raise LibraryAnalysisRejection(str(refusal), call.origin) from None
                raise AnalysisRejection(str(refusal), call.origin) from None
            if (target is list or target is tuple) and not keyword_facts and len(argument_facts) == 1:
                # A container conversion over an aggregate is a LAYOUT operation, never an evaluation: the same
                # leaves (runtime ones included) re-aggregate under the requested flavor. Concrete containers
                # (a range, a string, an all-Known tuple) fall through to the vetted evaluation below.
                source_fact = argument_facts[0]
                if isinstance(source_fact, AggregateFact):
                    # A record never reaches here: the admission walk already refused it as an argument.
                    if isinstance(source_fact.layout, ArrayLayout) and not source_fact.layout.shape:
                        raise AnalysisRejection("iteration over a 0-dimensional array is undefined", call.origin)
                    children = tuple(source_fact.child(i) for i in range(outer_arity(source_fact.layout)))
                    env.set(Local(call.dst), aggregate_of(children, is_list=target is list))
                    self._conversion_calls.add(id(call))
                    return False
            if (
                target is len
                and not keyword_facts
                and len(argument_facts) == 1
                and isinstance(argument_facts[0], AggregateFact)
            ):
                # Length is layout-determined: it folds on runtime leaves exactly as the unpacking arity check
                # (the PyLen op) does, records having been refused by the admission walk already.
                sized = argument_facts[0]
                if isinstance(sized.layout, ArrayLayout) and not sized.layout.shape:
                    raise AnalysisRejection("len() of a 0-dimensional array is undefined", call.origin)
                length = admit(outer_arity(sized.layout))
                assert length is not None
                env.set(Local(call.dst), Known(length))
                self._concrete_calls.add(id(call))
                return False
            explicit_float_dtype = (
                len(keyword_facts) == 1
                and keyword_facts[0][0] == "dtype"
                and isinstance(keyword_facts[0][1], Reference)
                and any(keyword_facts[0][1].obj is kind for kind in (float, np.float64))
            )
            if (
                any(target is factory for factory in (np.array, np.asarray, np.asanyarray))
                and (not keyword_facts or explicit_float_dtype)
                and len(argument_facts) == 1
                and isinstance(argument_facts[0], AggregateFact)
                and any(isinstance(leaf, Residual) for leaf in argument_facts[0].leaves)
            ):
                # A residual-carrying array construction is the same LAYOUT operation under numpy's discovery
                # rules, restricted to the proven subset; a fully static argument falls through to the vetted
                # concrete call below, where numpy itself decides every discovery corner (object promotion,
                # the uint64 range, bool widening) and the result normalizes back exactly. An explicit
                # dtype=float IS the conversion the implicit-widening rejections demand: every leaf casts.
                env.set(
                    Local(call.dst),
                    self._array_factory(argument_facts[0], call.origin, force_float=explicit_float_dtype),
                )
                self._conversion_calls.add(id(call))
                return False
            if target is isinstance and not keyword_facts and len(argument_facts) == 2:
                # isinstance folds through the RESOLVED classinfo types, never through a generic evaluation: the
                # classinfo's sanctioned carriers (a referenced class, an inline tuple of references) are not
                # data and cannot cross the evaluation boundary. Flattening is Python's own equivalence --
                # isinstance over nested tuples and unions is the disjunction over their members.
                subject_value = _concrete_fact(argument_facts[0])
                if subject_value is not None:
                    kinds = classinfo_types(argument_facts[1])
                    assert kinds is not None, "admission proved the classinfo resolves"
                    verdict = isinstance(_datapath_zero(as_python(subject_value)), tuple(kinds))
                    env.set(Local(call.dst), Known(StaticBool(verdict)))
                    self._concrete_calls.add(id(call))
                    return False
            concrete_args: list[StaticValue | Reference | None] = [
                fact if isinstance(fact, Reference) else _concrete_fact(fact) for fact in argument_facts
            ]
            concrete_kwargs = [
                (keyword, fact if isinstance(fact, Reference) else _concrete_fact(fact))
                for keyword, fact in keyword_facts
            ]
            if any(value is None for value in concrete_args) or any(v is None for _, v in concrete_kwargs):
                name = getattr(target, "__name__", repr(target))
                # ``float()``/``int()``/``bool()`` on a runtime scalar: a same-kind cast is the identity (a documented
                # no-op); a cross-kind cast lowers to a conversion op (int<->float truncation/promotion, truthiness,
                # bool widening). Explicit casts are how bool crosses into arithmetic and how float truncates to int.
                # Identity comparisons, never a dict/set membership test: ``target`` may be an unhashable shadow of a
                # builtin name (a bound array, a dict), which must miss cleanly rather than raise on hashing.
                cast_target = (
                    SemType.FLOAT
                    if target is float
                    else SemType.INT if target is int else SemType.BOOL if target is bool else None
                )
                cast_source = argument_facts[0] if argument_facts else None
                if (
                    isinstance(cast_source, AggregateFact)
                    and isinstance(cast_source.layout, ArrayLayout)
                    and cast_source.layout.shape == ()
                ):
                    cast_source = cast_source.leaves[0]  # numpy defines float()/int()/bool() of a 0-d array
                if (
                    cast_target is not None
                    and not keyword_facts
                    and len(argument_facts) == 1
                    and isinstance(cast_source, Residual)
                ):
                    env.set(Local(call.dst), Residual(cast_target))
                    self._cast_calls.add(id(call))
                    return False
                if is_unimplemented_library(target):
                    # A recognized math/numpy function with no fast-math hardware equivalent (erf, spacing, a ufunc):
                    # a distinct public error so the user knows it is a missing library primitive, not a bad call.
                    raise LibraryAnalysisRejection(f"library function {name!r} is not implemented yet", call.origin)
                raise AnalysisRejection(f"call to {name} with runtime arguments is not supported yet", call.origin)
            try:
                concrete = target(  # type: ignore[operator]
                    *[_crossing_object(value) for value in concrete_args if value is not None],
                    **{keyword: _crossing_object(value) for keyword, value in concrete_kwargs if value is not None},
                )
            except Exception as error:
                raise AnalysisRejection(f"call fails here: {error}", call.origin) from None
            admitted = admit(concrete)
            if admitted is None:
                env.set(Local(call.dst), Reference(concrete))
            else:
                taint_inputs: list[Fact] = [*argument_facts, *(fact for _, fact in keyword_facts)]
                if minted:
                    receiver_value = next(k[0] for k, m in self._value_methods.items() if m is target)
                    taint_inputs.append(Known(receiver_value))
                env.set(Local(call.dst), _taint_lost(normalize_static(admitted), _lost_scalar_pools(taint_inputs)))
            self._concrete_calls.add(id(call))
            return False
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
            raise AnalysisRejection(rejection.message, rejection.origin + call.origin) from None
        if len(unit.blocks) > _MAX_BLOCKS:
            raise AnalysisRejection("expansion fuel exhausted", call.origin)
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
                remapped.origin = remapped.origin + call.origin  # diagnostics point back at the user call site
                clone.ops.append(remapped)
            assert template_block.terminator is not None
            if isinstance(template_block.terminator, UnitExit):
                clone.terminator = Jump(continuation.id, call.origin)
            else:
                clone.terminator = _remap_terminator(
                    template_block.terminator, lambda b: block_map[b], fresh, graft_place
                )
                clone.terminator.origin = clone.terminator.origin + call.origin
            unit.blocks[clone.id] = clone
            self._block_ancestry[clone.id] = ancestry + (key,)
        # The call site becomes: bind arguments -> jump into the graft; the continuation reads the return local.
        block.ops = block.ops[:index]
        params = list(template.params)
        positional = list(call.args)
        keyword = dict(call.kwargs)
        if template.bound_self is not None:
            self_temp = BindingId(f"%s{self._binding_serial}", self._binding_serial)
            self._binding_serial += 1
            block.ops.append(LoadRef(self_temp, template.bound_self, call.origin))
            block.ops.append(StorePlace(Local(fresh(params[0])), self_temp, call.origin))
            params = params[1:]
        fn_object = target.__func__ if isinstance(target, types.MethodType) else target
        raw_defaults = fn_object.__defaults__ or ()
        kw_defaults = fn_object.__kwdefaults__ or {}
        self_offset = 1 if template.bound_self is not None else 0
        positional_count = fn_object.__code__.co_argcount - self_offset
        positional_only = {p.name for p in params[: max(0, fn_object.__code__.co_posonlyargcount - self_offset)]}
        positional_params = params[:positional_count]
        default_by_name: dict[str, object] = dict(
            zip((p.name for p in positional_params[len(positional_params) - len(raw_defaults) :]), raw_defaults)
        )
        default_by_name.update(kw_defaults)
        if len(positional) > len(positional_params):
            raise AnalysisRejection("too many positional arguments", call.origin)
        for offset, param in enumerate(params):
            if offset < len(positional):
                source = positional[offset]
                if param.name in keyword:
                    raise AnalysisRejection(f"duplicate argument '{param.name}'", call.origin)
            elif param.name in keyword and param.name not in positional_only:
                source = keyword.pop(param.name)
            elif param.name in default_by_name:
                admitted = admit(default_by_name[param.name])
                default_temp = BindingId(f"%d{self._binding_serial}", self._binding_serial)
                self._binding_serial += 1
                if admitted is not None:
                    block.ops.append(LoadConst(default_temp, admitted, call.origin))
                else:
                    block.ops.append(LoadRef(default_temp, default_by_name[param.name], call.origin))
                source = default_temp
            else:
                raise AnalysisRejection(f"missing argument '{param.name}'", call.origin)
            block.ops.append(StorePlace(Local(fresh(param)), source, call.origin))
        if keyword:
            raise AnalysisRejection(f"unexpected keyword argument '{next(iter(keyword))}'", call.origin)
        block.terminator = Jump(block_map[template.entry], call.origin)
        continuation.ops.insert(0, LoadPlace(call.dst, Local(return_local), call.origin))
        return True
