"""
The typed emission plan: how each surviving call lowers, and one TOTAL route plan per cell-routing site.

A route plan names an action for EVERY logical cell of its target, so absence carries no meaning any more: a site
that defines nothing still owns a plan, of `NoCell` rows. Emission executes rows and derives nothing of its own --
no window offsets, no source selection, no kind promotion chosen by inspecting an emitted node. `NoCell` is
SITE-RELATIVE ("this site defines no cell for this ordinal"), never a property of the fact: the same datapath
`Known` leaf materializes at a projection and stays fact-only inside a fully static construction.

`verify_route_plans` re-derives the site set, every target, every width, every per-ordinal DISPOSITION and every
expected source place from the op kinds and the stabilized facts ALONE -- never from the recorded routing
evidence and never by reading a plan back -- so a producer that stops recording, records a surplus site, or
records a plausible-but-wrong row fails hard instead of passing as a default. Disposition derivation is the
load-bearing half: source availability alone was measured to catch almost nothing.

What no structural check can reach is an in-range WRONG permutation, which is why the behavioural witnesses in
`tests/test_frontend_routing.py` carry that weight and this verifier does not replace them.
"""

import enum
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ._analysis_support import AnalysisRejection, StorageSchema, conform_local_store, join_facts, same_fact
from ._fact import (
    AggregateFact,
    ArrayLayout,
    AtomicFact,
    Fact,
    Known,
    RecordLayout,
    Reference,
    Residual,
    Unbound,
    child_slice,
    leaf_count,
    normalize_static,
)
from ._fold import FieldSchema
from ._ir import (
    BindingId,
    BlockId,
    BuildList,
    BuildTuple,
    FunctionUnit,
    LoadPlace,
    Local,
    Op,
    Place,
    PyAttr,
    PyBin,
    PyCall,
    PySelect,
    PyStoreAttr,
    PySubscript,
    SelectMode,
    StateLeaf,
    StorePlace,
    StoreRole,
    UnbindPlace,
    executable_rpo,
    op_dst,
)
from ._opsem import BinOp
from ._value import SemType, StaticValue, as_python, conform_value, datapath_value, same, value_kind

if TYPE_CHECKING:
    from .._lib import Intrinsic

# ---------------------------------------- call lowering ----------------------------------------


class CallLowering(enum.Enum):
    """How a PyCall surviving in the residual graph lowers; expanded calls no longer exist as calls."""

    FOLDED = enum.auto()  # a concrete static fold: the destination fact is Known, nothing to emit
    CAST = enum.auto()  # a scalar float()/int()/bool() cast; same-kind-vs-conversion is decided by the FINAL facts
    INTRINSIC = enum.auto()  # a registered hardware intrinsic: ``intrinsic`` carries the resolved registry match
    CONVERSION = enum.auto()  # list()/tuple()/np.array over an aggregate: the argument's leaves re-flavor onto it
    CONSTRUCTION = enum.auto()  # a record built structurally: argument cells install into per-field windows


@dataclass(frozen=True, slots=True)
class CallPlan:
    """
    Classification is recorded at the visit and is NEVER inferred from whether a route exists: an identity
    conversion and a zero-cell conversion are both still conversions.
    """

    lowering: CallLowering
    intrinsic: "Intrinsic | None" = None  # the resolved registry match for INTRINSIC


# ---------------------------------------- the route record ----------------------------------------


@dataclass(frozen=True, slots=True)
class CellRef:
    """
    A source cell, addressed by PLACE and ordinal rather than by an index into the op's operand list: operand
    lists have no authoritative meaning here (a sequence is not always operand 0, a state source is reachable
    from no operand at all), so an operand-index bound can pass while naming the wrong value.
    """

    place: Place
    ordinal: int

    def __str__(self) -> str:
        return f"{self.place}#{self.ordinal}"


class CellTransfer(enum.Enum):
    """
    The kind coercion a copy applies between its source cell and its target cell. Closed at three because M2
    covers ROUTING sites only; scalar casts and truth conversions live in their own lowering and would need more
    values the moment they are absorbed.
    """

    IDENTITY = enum.auto()
    INT_TO_FLOAT = enum.auto()
    BOOL_TO_FLOAT = enum.auto()


@dataclass(frozen=True, slots=True)
class CopyCell:
    source: CellRef
    transfer: CellTransfer


@dataclass(frozen=True, slots=True)
class ConstantCell:
    """``value`` is the TARGET-SIDE image: an integer routed onto a float slot carries the conformed float."""

    value: StaticValue
    kind: SemType


@dataclass(frozen=True, slots=True)
class NoCell:
    """This SITE defines no datapath cell for this target ordinal."""


type CellAction = CopyCell | ConstantCell | NoCell


@dataclass(frozen=True, slots=True)
class PlanSite:
    """
    A PHASE-LOCAL op identity, valid from finalization to the end of emission and no further: emission mutates
    neither the block set nor any block's op list, which was measured rather than assumed. A later pass that
    mutates the finalized graph must rebuild its plans rather than expect these keys to survive.
    """

    block: BlockId
    index: int

    def __str__(self) -> str:
        return f"{self.block}@{self.index}"


@dataclass(frozen=True, slots=True)
class RoutePlan:
    target: Place
    actions: tuple[CellAction, ...]  # indexed by the target's LOGICAL leaf ordinal


# ---------------------------------------- recorded routing evidence ----------------------------------------


@dataclass(frozen=True, slots=True)
class SourceSelection:
    """Per result cell, its source leaf ordinal within the op's single source aggregate."""

    ordinals: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class FieldBindings:
    """Per record field in declaration order, the argument binding filling it, or None for a schema default."""

    sources: tuple[BindingId | None, ...]


type RouteEvidence = SourceSelection | FieldBindings


# ---------------------------------------- shared derivations ----------------------------------------


class Disposition(enum.Enum):
    """What a site must do at one target ordinal, independently of which cell it copies from."""

    COPY = enum.auto()
    CONSTANT = enum.auto()
    NONE = enum.auto()


def _logical_width(fact: Fact) -> int:
    return len(fact.leaves) if isinstance(fact, AggregateFact) else 1


def _leaf_disposition(leaf: Fact) -> Disposition:
    if isinstance(leaf, Residual):
        return Disposition.COPY
    if isinstance(leaf, Known) and datapath_value(leaf.value):
        return Disposition.CONSTANT
    return Disposition.NONE


def _skipping_disposition(fact: Fact) -> Disposition:
    """The scalar arms of `LoadPlace` and of projection, which skip a Known destination the aggregate walks emit."""
    return Disposition.COPY if isinstance(fact, Residual) else Disposition.NONE


def _transfer(source: SemType, target: SemType) -> CellTransfer:
    if source is target:
        return CellTransfer.IDENTITY
    if source is SemType.INT and target is SemType.FLOAT:
        return CellTransfer.INT_TO_FLOAT
    assert source is SemType.BOOL and target is SemType.FLOAT, f"no routing transfer from {source} to {target}"
    return CellTransfer.BOOL_TO_FLOAT


def _atom_kind(fact: AtomicFact) -> SemType:
    if isinstance(fact, Residual):
        return fact.type
    assert isinstance(fact, Known) and datapath_value(fact.value)
    return value_kind(fact.value)


def _slot_kind(snapshot: "StaticValue | str", ordinal: int) -> SemType:
    """
    The fixed kind of a state cell, straight from the reset snapshot. A state store whose reset is not an
    admissible datapath scalar is refused during ANALYSIS, so every store site reaches a decidable kind here.
    """
    assert not isinstance(snapshot, str), "a store to an inadmissible reset is refused during analysis"
    normalized = normalize_static(snapshot)
    cell = normalized.leaves[ordinal] if isinstance(normalized, AggregateFact) else normalized
    assert isinstance(cell, Known)
    return value_kind(cell.value)


def _state_width(snapshot: "StaticValue | str") -> int:
    assert not isinstance(snapshot, str), "a store to an inadmissible reset is refused during analysis"
    normalized = normalize_static(snapshot)
    return leaf_count(normalized.layout) if isinstance(normalized, AggregateFact) else 1


def _stored_image(stored: Fact, snapshot: "StaticValue | str") -> Fact:
    """What a state slot holds after a store: the stored fact re-kinded onto the reset-fixed cell kinds."""
    leaves = stored.leaves if isinstance(stored, AggregateFact) else (stored,)
    imaged: list[AtomicFact] = []
    for ordinal, leaf in enumerate(leaves):
        kind = _slot_kind(snapshot, ordinal)
        if isinstance(leaf, Known):
            imaged.append(Known(conform_value(leaf.value, kind)))
        else:
            assert isinstance(leaf, Residual)
            imaged.append(Residual(kind))
    if isinstance(stored, AggregateFact):
        return AggregateFact(stored.layout, tuple(imaged))
    return imaged[0]


def _selected_arm(mode: SelectMode, taken: bool, lhs: BindingId, rhs: BindingId) -> BindingId:
    # POLARITY: `and` yields the RIGHT operand when the condition holds, `or` the left one.
    return (rhs if taken else lhs) if mode is SelectMode.AND else (lhs if taken else rhs)


def _known_condition(fact: Fact) -> bool | None:
    if not isinstance(fact, Known):
        return None
    taken = as_python(fact.value)
    assert isinstance(taken, bool)
    return taken


def _field_assignments(declared: tuple[FieldSchema, ...], call: PyCall) -> list[BindingId | None]:
    """
    Which argument binding fills each declared field, from the call's positional and keyword structure -- Python's
    own call semantics, so re-deriving it is a check rather than a second routing authority.
    """
    positional = [entry.name for entry in declared if not entry.kw_only]
    assignments: dict[str, BindingId] = dict(zip(positional, call.args))
    for keyword, binding in call.kwargs:
        assignments[keyword] = binding
    return [assignments.get(entry.name) for entry in declared]


def _store_conformance(
    unit: FunctionUnit,
    executable_edges: set[tuple[BlockId, BlockId]],
    binding_facts: Mapping[BindingId, Fact],
    schemas_in: Mapping[BlockId, Mapping[Place, StorageSchema]],
) -> dict[int, Fact]:
    """
    The post-store fact each SOURCE store leaves on its local, by replaying the stable storage-schema walk. A
    dst-less `StorePlace` owns no destination binding fact, so this walk is the ONLY authority on the value, kind
    and transfer its plan row carries -- and both the producer and the verifier replay it rather than reading each
    other's derivation. Every block starts from its own recorded entry schemas, so the walk order is immaterial.
    """
    images: dict[int, Fact] = {}
    for block_id in executable_rpo(unit.entry, executable_edges):
        env: dict[Place, StorageSchema] = dict(schemas_in[block_id])
        for op in unit.blocks[block_id].ops:
            match op:
                case UnbindPlace(place=place, checked=False):
                    env.pop(place, None)  # a compiler scope reset clears the binding; a user `del` does not
                case StorePlace(place=place, src=src, role=StoreRole.SOURCE):
                    assert isinstance(place, Local), "a SOURCE store binds a named local"
                    schema, conformed, message = conform_local_store(
                        env.get(place), place.binding.name, binding_facts[src]
                    )
                    assert message is None, f"a storage-schema violation survived analysis: {message}"
                    if schema is not None:
                        env[place] = schema
                    images[id(op)] = conformed
                case _:
                    pass
    return images


# ---------------------------------------- the producer ----------------------------------------


def produce_route_plans(
    unit: FunctionUnit,
    executable_edges: set[tuple[BlockId, BlockId]],
    schemas_in: Mapping[BlockId, Mapping[Place, StorageSchema]],
    binding_facts: Mapping[BindingId, Fact],
    call_plans: Mapping[BindingId, CallPlan],
    evidence: Mapping[int, RouteEvidence],
    state_resets: Mapping[StateLeaf, "StaticValue | str"],
) -> dict[PlanSite, RoutePlan]:
    """
    One total route plan per routing site over the stabilized graph, translating the analyzer's op-keyed routing
    evidence onto position keys. This is the single place a routing decision is turned into a plan.
    """
    producer = _Producer(unit, binding_facts, call_plans, evidence, state_resets)
    producer.images = _store_conformance(unit, executable_edges, binding_facts, schemas_in)
    plans: dict[PlanSite, RoutePlan] = {}
    for block_id in executable_rpo(unit.entry, executable_edges):
        for index, op in enumerate(unit.blocks[block_id].ops):
            plan = producer.plan_for(op)
            if plan is not None:
                plans[PlanSite(block_id, index)] = plan
    return plans


class _Producer:
    def __init__(
        self,
        unit: FunctionUnit,
        binding_facts: Mapping[BindingId, Fact],
        call_plans: Mapping[BindingId, CallPlan],
        evidence: Mapping[int, RouteEvidence],
        state_resets: Mapping[StateLeaf, "StaticValue | str"],
    ) -> None:
        self._unit = unit
        self._facts = binding_facts
        self._call_plans = call_plans
        self._evidence = evidence
        self._state_resets = state_resets
        self.images: dict[int, Fact] = {}

    def _fact(self, binding: BindingId) -> Fact:
        fact = self._facts[binding]
        assert fact is not None
        return fact

    def _selection(self, op: Op) -> SourceSelection:
        # A bare subscript on purpose: every routing selection is recorded at the visit that computes it, so a
        # miss means the recording premise itself broke rather than that this site routes positionally.
        recorded = self._evidence[id(op)]
        assert isinstance(recorded, SourceSelection)
        return recorded

    def plan_for(self, op: Op) -> RoutePlan | None:
        match op:
            case LoadPlace(dst=dst, place=place):
                return self._load(dst, place)
            case StorePlace(place=place, src=src):
                return self._store(op, place, src)
            case PyBin(dst=dst, op=bin_op, lhs=lhs, rhs=rhs):
                return self._sequence_bin(dst, bin_op, lhs, rhs)
            case PySelect(dst=dst, mode=mode, cond=cond, lhs=lhs, rhs=rhs):
                return self._select(dst, mode, cond, lhs, rhs)
            case BuildTuple(dst=dst, items=items) | BuildList(dst=dst, items=items):
                return self._build(dst, items)
            case PySubscript(dst=dst, obj=obj):
                return self._projection(op, dst, obj)
            case PyAttr(dst=dst, obj=obj, name=name):
                return self._attribute(op, dst, obj, name)
            case PyStoreAttr(obj=obj, name=name, src=src):
                return self._store_attr(obj, name, src)
            case PyCall():
                return self._call(op)
            case _:
                return None

    def _copy_window(self, actions: list[CellAction], start: int, source: BindingId) -> None:
        """One item's cells into the target window at [start, ...): a build item or a construction field."""
        fact = self._fact(source)
        leaves = fact.leaves if isinstance(fact, AggregateFact) else (fact,)
        for ordinal, leaf in enumerate(leaves):
            assert isinstance(leaf, (Known, Residual, Reference))
            actions[start + ordinal] = self._aggregate_action(leaf, CellRef(Local(source), ordinal))

    @staticmethod
    def _aggregate_action(
        leaf: AtomicFact, source: CellRef, transfer: CellTransfer = CellTransfer.IDENTITY
    ) -> CellAction:
        match _leaf_disposition(leaf):
            case Disposition.CONSTANT:
                assert isinstance(leaf, Known)
                return ConstantCell(leaf.value, value_kind(leaf.value))
            case Disposition.COPY:
                return CopyCell(source, transfer)
            case Disposition.NONE:
                return NoCell()

    def _load(self, dst: BindingId, place: Place) -> RoutePlan:
        fact = self._fact(dst)
        if isinstance(fact, AggregateFact):
            return RoutePlan(
                Local(dst),
                tuple(
                    self._aggregate_action(leaf, CellRef(place, ordinal)) for ordinal, leaf in enumerate(fact.leaves)
                ),
            )
        # ASYMMETRIC against the aggregate arm above: a scalar Known destination emits nothing at all here.
        source = CellRef(place, 0)
        return RoutePlan(
            Local(dst), (CopyCell(source, CellTransfer.IDENTITY) if isinstance(fact, Residual) else NoCell(),)
        )

    def _store(self, op: Op, place: Place, src: BindingId) -> RoutePlan:
        source_fact = self._fact(src)
        if isinstance(source_fact, AggregateFact):
            return RoutePlan(
                place,
                tuple(
                    self._aggregate_action(leaf, CellRef(Local(src), ordinal))
                    for ordinal, leaf in enumerate(source_fact.leaves)
                ),
            )
        # The width and the kind both come from the POST-STORE fact: a store edge may conform an integer onto a
        # float variable, and only the storage-schema walk knows it.
        stored = self.images.get(id(op), source_fact)
        if isinstance(stored, Known):
            action: CellAction = (
                ConstantCell(stored.value, value_kind(stored.value)) if datapath_value(stored.value) else NoCell()
            )
        elif isinstance(stored, Reference):
            action = NoCell()
        else:
            assert isinstance(stored, Residual) and isinstance(source_fact, Residual)
            action = CopyCell(CellRef(Local(src), 0), _transfer(source_fact.type, stored.type))
        return RoutePlan(place, (action,))

    def _sequence_bin(self, dst: BindingId, bin_op: BinOp, lhs: BindingId, rhs: BindingId) -> RoutePlan | None:
        fact = self._fact(dst)
        # Decided by the LAYOUT, not the op kind: a sequence aggregate concatenates or repeats and emits no HIR at
        # all, while an array aggregate computes elementwise and a scalar computes.
        if not isinstance(fact, AggregateFact) or isinstance(fact.layout, ArrayLayout):
            return None
        actions: list[CellAction] = [NoCell()] * len(fact.leaves)
        if bin_op is BinOp.ADD:
            offset = 0
            for operand in (lhs, rhs):
                operand_fact = self._fact(operand)
                assert isinstance(operand_fact, AggregateFact)
                for ordinal, leaf in enumerate(operand_fact.leaves):
                    actions[offset + ordinal] = self._aggregate_action(leaf, CellRef(Local(operand), ordinal))
                offset += len(operand_fact.leaves)
            assert offset == len(fact.leaves)
        else:
            assert bin_op is BinOp.MUL
            sequence = self._repeated_operand(lhs, rhs)
            unit_fact = self._fact(sequence)
            assert isinstance(unit_fact, AggregateFact)
            width = len(unit_fact.leaves)
            for repetition in range(len(fact.leaves) // width if width else 0):
                for ordinal, leaf in enumerate(unit_fact.leaves):
                    actions[repetition * width + ordinal] = self._aggregate_action(
                        leaf, CellRef(Local(sequence), ordinal)
                    )
        return RoutePlan(Local(dst), tuple(actions))

    def _repeated_operand(self, lhs: BindingId, rhs: BindingId) -> BindingId:
        # `3 * seq` is as valid as `seq * 3`, so the sequence is found by its FACT, never by operand position.
        return next(operand for operand in (lhs, rhs) if isinstance(self._fact(operand), AggregateFact))

    def _select(
        self, dst: BindingId, mode: SelectMode, cond: BindingId, lhs: BindingId, rhs: BindingId
    ) -> RoutePlan | None:
        result = self._fact(dst)
        if isinstance(result, (Known, Reference)):
            return None  # the analyzer already picked a Known/reference arm; every use folds
        taken = _known_condition(self._fact(cond))
        if taken is None:
            return None  # a residual condition COMPUTES a Select operation
        chosen = _selected_arm(mode, taken, lhs, rhs)
        if isinstance(result, AggregateFact):
            return RoutePlan(
                Local(dst),
                tuple(
                    self._aggregate_action(leaf, CellRef(Local(chosen), ordinal))
                    for ordinal, leaf in enumerate(result.leaves)
                ),
            )
        assert isinstance(result, Residual)
        return RoutePlan(Local(dst), (CopyCell(CellRef(Local(chosen), 0), CellTransfer.IDENTITY),))

    def _build(self, dst: BindingId, items: tuple[BindingId, ...]) -> RoutePlan:
        fact = self._fact(dst)
        if not isinstance(fact, AggregateFact):
            return RoutePlan(Local(dst), (NoCell(),))
        actions: list[CellAction] = [NoCell()] * len(fact.leaves)
        for index, item in enumerate(items):
            _, start, _ = child_slice(fact.layout, index)
            self._copy_window(actions, start, item)
        return RoutePlan(Local(dst), tuple(actions))

    def _window_plan(self, target: Place, dst_fact: Fact, source: Place, ordinals: tuple[int, ...]) -> RoutePlan:
        if isinstance(dst_fact, AggregateFact):
            assert len(ordinals) == len(dst_fact.leaves), "a selection misaligns its result"
            return RoutePlan(
                target,
                tuple(
                    self._aggregate_action(leaf, CellRef(source, ordinals[ordinal]))
                    for ordinal, leaf in enumerate(dst_fact.leaves)
                ),
            )
        assert len(ordinals) == 1 and isinstance(dst_fact, Residual)
        return RoutePlan(target, (CopyCell(CellRef(source, ordinals[0]), CellTransfer.IDENTITY),))

    def _projects_cells(self, dst_fact: Fact) -> bool:
        return isinstance(dst_fact, Residual) or (
            isinstance(dst_fact, AggregateFact) and any(isinstance(leaf, Residual) for leaf in dst_fact.leaves)
        )

    def _projection(self, op: Op, dst: BindingId, obj: BindingId) -> RoutePlan:
        dst_fact = self._fact(dst)
        obj_fact = self._fact(obj)
        if not (isinstance(obj_fact, AggregateFact) and self._projects_cells(dst_fact)):
            # An all-Known projection emits nothing, so the plan is all-`NoCell` rather than absent: the old
            # `needs_cells` gate stops being an independent decision and becomes a consequence of dispositions.
            return RoutePlan(Local(dst), (NoCell(),) * _logical_width(dst_fact))
        return self._window_plan(Local(dst), dst_fact, Local(obj), self._selection(op).ordinals)

    def _attribute(self, op: Op, dst: BindingId, obj: BindingId, name: str) -> RoutePlan:
        dst_fact = self._fact(dst)
        if isinstance(dst_fact, (Known, Reference)):
            return RoutePlan(Local(dst), (NoCell(),) * _logical_width(dst_fact))
        obj_fact = self._fact(obj)
        if isinstance(obj_fact, AggregateFact):
            return self._projection(op, dst, obj)
        # A component attribute: the source is a STATE root, reachable from no operand of this op.
        assert isinstance(obj_fact, Reference)
        root = StateLeaf(obj_fact.obj, (name,))
        if isinstance(dst_fact, AggregateFact):
            return RoutePlan(
                Local(dst),
                tuple(
                    self._aggregate_action(leaf, CellRef(root, ordinal)) for ordinal, leaf in enumerate(dst_fact.leaves)
                ),
            )
        assert isinstance(dst_fact, Residual)
        return RoutePlan(Local(dst), (CopyCell(CellRef(root, 0), CellTransfer.IDENTITY),))

    def _store_attr(self, obj: BindingId, name: str, src: BindingId) -> RoutePlan:
        obj_fact = self._fact(obj)
        assert isinstance(obj_fact, Reference)
        leaf = StateLeaf(obj_fact.obj, (name,))
        snapshot = self._state_resets[leaf]
        source_fact = self._fact(src)
        leaves = source_fact.leaves if isinstance(source_fact, AggregateFact) else (source_fact,)
        actions: list[CellAction] = []
        for ordinal, stored in enumerate(leaves):
            # A state store has NO skip: analysis refuses a non-datapath or reference value before emission, so
            # every cell of a state target is defined and none of them may be a `NoCell`.
            kind = _slot_kind(snapshot, ordinal)
            if isinstance(stored, Known):
                assert datapath_value(stored.value), "a non-datapath state store is refused during analysis"
                actions.append(ConstantCell(conform_value(stored.value, kind), kind))
            else:
                assert isinstance(stored, Residual), "a reference state store is refused during analysis"
                actions.append(CopyCell(CellRef(Local(src), ordinal), _transfer(stored.type, kind)))
        return RoutePlan(leaf, tuple(actions))

    def _call(self, op: PyCall) -> RoutePlan | None:
        plan = self._call_plans[op.dst]
        fact = self._fact(op.dst)
        match plan.lowering:
            case CallLowering.CONVERSION:
                if not isinstance(fact, AggregateFact):
                    return None  # a scalar conversion result carries no cells to route
                source = op.args[0]
                source_fact = self._fact(source)
                assert isinstance(source_fact, AggregateFact)
                ordinals = self._conversion_ordinals(op, len(fact.leaves))
                actions: list[CellAction] = []
                for ordinal, leaf in enumerate(fact.leaves):
                    origin = source_fact.leaves[ordinals[ordinal]]
                    transfer = CellTransfer.IDENTITY
                    if isinstance(leaf, Residual):
                        assert isinstance(origin, Residual), "a conversion re-kinded a static leaf into a runtime one"
                        transfer = _transfer(origin.type, leaf.type)
                    actions.append(self._aggregate_action(leaf, CellRef(Local(source), ordinals[ordinal]), transfer))
                return RoutePlan(Local(op.dst), tuple(actions))
            case CallLowering.CONSTRUCTION:
                assert isinstance(fact, AggregateFact)
                bindings = self._evidence[id(op)]
                assert isinstance(bindings, FieldBindings)
                rows: list[CellAction] = [NoCell()] * len(fact.leaves)
                # An INACTIVE (fully static) construction emits nothing at all, not even its datapath constants:
                # every use folds from the facts, and eager constants would shift HIR node ordering.
                if any(isinstance(leaf, Residual) for leaf in fact.leaves):
                    for index, source_binding in enumerate(bindings.sources):
                        _, start, stop = child_slice(fact.layout, index)
                        if source_binding is not None:
                            self._copy_window(rows, start, source_binding)
                        else:
                            for ordinal in range(start, stop):
                                filled = fact.leaves[ordinal]
                                assert isinstance(filled, (Known, Reference)), "a default grew runtime cells"
                                rows[ordinal] = self._aggregate_action(filled, CellRef(Local(op.dst), ordinal))
                return RoutePlan(Local(op.dst), tuple(rows))
            case _:
                return None  # FOLDED, CAST and INTRINSIC compute; they route nothing

    def _conversion_ordinals(self, op: PyCall, width: int) -> tuple[int, ...]:
        recorded = self._evidence.get(id(op))
        if recorded is None:
            return tuple(range(width))  # an aligned re-flavor; only a relayout records a permutation
        assert isinstance(recorded, SourceSelection)
        return recorded.ordinals


# ---------------------------------------- the independent verifier ----------------------------------------


def verify_route_plans(
    unit: FunctionUnit,
    executable_edges: set[tuple[BlockId, BlockId]],
    facts_in: Mapping[BlockId, Mapping[Place, Fact]],
    schemas_in: Mapping[BlockId, Mapping[Place, StorageSchema]],
    binding_facts: Mapping[BindingId, Fact],
    call_plans: Mapping[BindingId, CallPlan],
    construction_schemas: Mapping[int, tuple[type, tuple[FieldSchema, ...]]],
    state_resets: Mapping[StateLeaf, "StaticValue | str"],
    runtime_state: set[StateLeaf],
    plans: Mapping[PlanSite, RoutePlan],
) -> None:
    """
    Re-derive the whole route plan from the ops and the stabilized facts and refuse any disagreement, without
    consulting the analyzer's routing evidence at any point. `runtime_state` is an input so that W itself can be
    CHECKED rather than trusted: a row over an unpromoted leaf is legal exactly when that leaf's snapshot really
    does survive the canonical exit, which this re-derives from the exit facts. It is deliberately not used to
    veto such a row outright -- a state place carries its value across the step whether or not the leaf is
    promoted -- because that veto rejects correct plans while letting an under-promoted W through silently.
    """
    _Verifier(unit, facts_in, binding_facts, call_plans, construction_schemas, state_resets, runtime_state, plans).run(
        executable_edges, schemas_in
    )


class _Verifier:
    def __init__(
        self,
        unit: FunctionUnit,
        facts_in: Mapping[BlockId, Mapping[Place, Fact]],
        binding_facts: Mapping[BindingId, Fact],
        call_plans: Mapping[BindingId, CallPlan],
        construction_schemas: Mapping[int, tuple[type, tuple[FieldSchema, ...]]],
        state_resets: Mapping[StateLeaf, "StaticValue | str"],
        runtime_state: set[StateLeaf],
        plans: Mapping[PlanSite, RoutePlan],
    ) -> None:
        self._unit = unit
        self._facts_in = facts_in
        self._facts = binding_facts
        self._call_plans = call_plans
        self._schemas = construction_schemas
        self._state_resets = state_resets
        self._runtime_state = runtime_state
        self._plans = plans
        self._complaints: list[str] = []

    def _fact(self, binding: BindingId) -> Fact:
        return self._facts[binding]

    def run(
        self,
        executable_edges: set[tuple[BlockId, BlockId]],
        schemas_in: Mapping[BlockId, Mapping[Place, StorageSchema]],
    ) -> None:
        images = _store_conformance(self._unit, executable_edges, self._facts, schemas_in)
        expected: set[PlanSite] = set()
        for block_id in executable_rpo(self._unit.entry, executable_edges):
            env: dict[Place, Fact] = dict(self._facts_in[block_id])
            for index, op in enumerate(self._unit.blocks[block_id].ops):
                site = PlanSite(block_id, index)
                if self._routes(op):
                    expected.add(site)
                    plan = self._plans.get(site)
                    if plan is None:
                        self._complaints.append(f"{site}: {type(op).__name__} routes but has no plan")
                    else:
                        self._check(site, env, op, plan, images)
                self._advance(env, op, images)
        surplus = sorted(set(self._plans) - expected, key=str)
        for site in surplus:
            self._complaints.append(f"{site}: a plan exists where the op routes nothing")
        assert not self._complaints, "route plan verification failed:\n  " + "\n  ".join(self._complaints)

    # ---- the pre-op environment, rebuilt from the block live-ins and the final facts alone ----

    def _advance(self, env: dict[Place, Fact], op: Op, images: Mapping[int, Fact]) -> None:
        match op:
            case StorePlace(place=place, src=src):
                # The POST-STORE fact, not the source's: a store edge may conform an integer onto a float
                # variable, and a later read of that variable copies the conformed cell with no promotion left.
                env[place] = images.get(id(op), self._fact(src))
            case UnbindPlace(place=place):
                env.pop(place, None)
            case PyStoreAttr(obj=obj, name=name, src=src):
                obj_fact = self._fact(obj)
                assert isinstance(obj_fact, Reference)
                # A state slot's kind is fixed by its reset, so what the cell holds afterwards is the slot kind.
                leaf = StateLeaf(obj_fact.obj, (name,))
                stored = self._fact(src)
                env[leaf] = _stored_image(stored, self._state_resets[leaf])
            case _:
                dst = op_dst(op)
                if dst is not None:
                    env[Local(dst)] = self._fact(dst)

    # ---- the site set, derived from the op kind and the final facts ----

    def _routes(self, op: Op) -> bool:
        match op:
            case LoadPlace() | StorePlace() | PySubscript() | PyAttr() | PyStoreAttr():
                return True
            case BuildTuple() | BuildList():
                return True
            case PyBin(dst=dst):
                fact = self._fact(dst)
                return isinstance(fact, AggregateFact) and not isinstance(fact.layout, ArrayLayout)
            case PySelect(dst=dst, cond=cond):
                result = self._fact(dst)
                if isinstance(result, (Known, Reference)):
                    return False
                return _known_condition(self._fact(cond)) is not None
            case PyCall(dst=dst):
                lowering = self._call_plans[dst].lowering
                if lowering is CallLowering.CONSTRUCTION:
                    return True
                return lowering is CallLowering.CONVERSION and isinstance(self._fact(dst), AggregateFact)
            case _:
                return False

    def _target(self, op: Op) -> Place:
        match op:
            case StorePlace(place=place):
                return place
            case PyStoreAttr(obj=obj, name=name):
                obj_fact = self._fact(obj)
                assert isinstance(obj_fact, Reference)
                return StateLeaf(obj_fact.obj, (name,))
            case _:
                dst = op_dst(op)
                assert dst is not None
                return Local(dst)

    def _target_fact(self, op: Op, images: Mapping[int, Fact]) -> Fact:
        match op:
            case StorePlace(src=src):
                return images.get(id(op), self._fact(src))
            case PyStoreAttr(src=src):
                return self._fact(src)
            case _:
                dst = op_dst(op)
                assert dst is not None
                return self._fact(dst)

    def _width(self, op: Op, images: Mapping[int, Fact]) -> int:
        target = self._target(op)
        if isinstance(target, StateLeaf):
            # A state target's width is fixed by the RESET schema, never by what this store happens to carry.
            return _state_width(self._state_resets[target])
        return _logical_width(self._target_fact(op, images))

    # ---- the expected DISPOSITION per ordinal: the load-bearing check ----

    def _dispositions(self, op: Op, images: Mapping[int, Fact]) -> list[Disposition]:
        fact = self._target_fact(op, images)
        leaves: list[Fact] | None = list(fact.leaves) if isinstance(fact, AggregateFact) else None
        match op:
            case LoadPlace():
                return (
                    [_leaf_disposition(leaf) for leaf in leaves]
                    if leaves is not None
                    else [_skipping_disposition(fact)]
                )
            case StorePlace() | PyBin() | BuildTuple() | BuildList() | PySelect():
                return [_leaf_disposition(leaf) for leaf in leaves] if leaves is not None else [_leaf_disposition(fact)]
            case PyStoreAttr():
                # A state store materializes every cell; there is no Known/Reference skip anywhere on this path.
                stored: list[Fact] = leaves if leaves is not None else [fact]
                return [Disposition.CONSTANT if isinstance(leaf, Known) else Disposition.COPY for leaf in stored]
            case PySubscript(obj=obj) | PyAttr(obj=obj):
                return self._projection_dispositions(op, fact, self._fact(obj), leaves)
            case PyCall(dst=dst):
                assert leaves is not None
                if self._call_plans[dst].lowering is CallLowering.CONSTRUCTION and not any(
                    isinstance(leaf, Residual) for leaf in leaves
                ):
                    return [Disposition.NONE] * len(leaves)
                return [_leaf_disposition(leaf) for leaf in leaves]
            case _:
                raise AssertionError(f"no disposition rule for {type(op).__name__}")

    def _projection_dispositions(
        self, op: Op, fact: Fact, obj_fact: Fact, leaves: "list[Fact] | None"
    ) -> list[Disposition]:
        if isinstance(op, PyAttr) and isinstance(obj_fact, Reference) and not isinstance(fact, (Known, Reference)):
            # A component-state read, not a projection: every runtime cell copies from its own slot.
            return [_leaf_disposition(leaf) for leaf in leaves] if leaves is not None else [Disposition.COPY]
        routes = isinstance(obj_fact, AggregateFact) and (
            isinstance(fact, Residual) or (leaves is not None and any(isinstance(leaf, Residual) for leaf in leaves))
        )
        if isinstance(fact, (Known, Reference)) or not routes:
            return [Disposition.NONE] * _logical_width(fact)
        # The scalar projection arm carries the same Known skip as the scalar `LoadPlace` arm.
        return [_leaf_disposition(leaf) for leaf in leaves] if leaves is not None else [_skipping_disposition(fact)]

    # ---- the expected SOURCE PLACE per ordinal: an arbitrary in-range place must not pass ----

    def _sources(self, op: Op, width: int) -> list[Place | None]:
        match op:
            case LoadPlace(place=place):
                return [place] * width
            case StorePlace(src=src) | PyStoreAttr(src=src) | PySubscript(obj=src):
                return [Local(src)] * width
            case PyAttr(obj=obj, name=name):
                obj_fact = self._fact(obj)
                if isinstance(obj_fact, Reference):
                    return [StateLeaf(obj_fact.obj, (name,))] * width
                return [Local(obj)] * width
            case PySelect(dst=dst, mode=mode, cond=cond, lhs=lhs, rhs=rhs):
                taken = _known_condition(self._fact(cond))
                assert taken is not None
                return [Local(_selected_arm(mode, taken, lhs, rhs))] * width
            case PyBin(dst=dst, op=bin_op, lhs=lhs, rhs=rhs):
                return self._sequence_sources(dst, bin_op, lhs, rhs, width)
            case BuildTuple(dst=dst, items=items) | BuildList(dst=dst, items=items):
                return self._window_sources(dst, [(index, item) for index, item in enumerate(items)], width)
            case PyCall(dst=dst, args=args):
                if self._call_plans[dst].lowering is CallLowering.CONVERSION:
                    return [Local(args[0])] * width
                return self._construction_sources(op, dst, width)
            case _:
                raise AssertionError(f"no source rule for {type(op).__name__}")

    def _sequence_sources(
        self, dst: BindingId, bin_op: BinOp, lhs: BindingId, rhs: BindingId, width: int
    ) -> list[Place | None]:
        out: list[Place | None] = [None] * width
        if bin_op is BinOp.ADD:
            offset = 0
            for operand in (lhs, rhs):
                operand_fact = self._fact(operand)
                assert isinstance(operand_fact, AggregateFact)
                for ordinal in range(len(operand_fact.leaves)):
                    out[offset + ordinal] = Local(operand)
                offset += len(operand_fact.leaves)
            return out
        assert bin_op is BinOp.MUL
        sequence = next(operand for operand in (lhs, rhs) if isinstance(self._fact(operand), AggregateFact))
        return [Local(sequence)] * width

    def _window_sources(
        self, dst: BindingId, windows: list[tuple[int, BindingId | None]], width: int
    ) -> list[Place | None]:
        fact = self._fact(dst)
        assert isinstance(fact, AggregateFact)
        out: list[Place | None] = [None] * width
        for index, source in windows:
            _, start, stop = child_slice(fact.layout, index)
            for ordinal in range(start, stop):
                out[ordinal] = Local(source) if source is not None else None
        return out

    def _construction_sources(self, op: PyCall, dst: BindingId, width: int) -> list[Place | None]:
        # RE-DERIVED from the call's positional/keyword structure against the IMMUTABLE schema snapshot -- reading
        # the recorded field mapping back would make this check vacuous, and live dataclass metadata can disagree
        # with the snapshot the analysis actually used.
        fact = self._fact(dst)
        assert isinstance(fact, AggregateFact) and isinstance(fact.layout, RecordLayout)
        klass = fact.layout.klass
        pinned = self._schemas[id(klass)]
        assert pinned[0] is klass, "the pinned construction schema is a different class"
        assignments = _field_assignments(pinned[1], op)
        return self._window_sources(dst, list(enumerate(assignments)), width)

    # ---- source availability: a supporting check, useless alone but never falsely objecting ----

    def _available(self, env: Mapping[Place, Fact], ref: CellRef) -> str | None:
        if isinstance(ref.place, StateLeaf):
            snapshot = self._state_resets.get(ref.place)
            if snapshot is None:
                return "names a state leaf with no reset snapshot"
            width = _state_width(snapshot)
            if not 0 <= ref.ordinal < width:
                return f"is outside the state slot's {width} cell(s)"
            # A promoted leaf reads its slot; an unpromoted one reads the snapshot, which is sound only because W
            # promotes whenever the exit can move a leaf off it. That premise is W's, so re-derive it here instead
            # of trusting it: an under-promoted leaf would otherwise route a stale constant where a carried value
            # belongs, and emit a silently wrong design rather than a located refusal.
            return None if ref.place in self._runtime_state else self._carries_its_snapshot(ref.place, snapshot)
        return self._local_available(env, ref)

    def _carries_its_snapshot(self, leaf: StateLeaf, snapshot: "StaticValue | str") -> str | None:
        """
        Whether an unpromoted leaf really does hand every transaction its reset snapshot back, re-derived from the
        canonical exit rather than read off W. An absent or unbound exit fact leaves the snapshot standing, which
        is what the state fixed point concludes for those cases too; a join that cannot be taken belongs to the
        analyzer's own deferred rejection, so it is not re-reported here.
        """
        exit_fact = self._facts_in.get(self._unit.exit, {}).get(leaf)
        if exit_fact is None or isinstance(exit_fact, Unbound):
            return None
        reset = normalize_static(snapshot) if not isinstance(snapshot, str) else None
        if reset is None:
            return None
        try:
            moved = not same_fact(join_facts(reset, exit_fact, ()), reset)
        except AnalysisRejection:
            return None
        return "names an unpromoted state leaf whose canonical exit moves off its reset snapshot" if moved else None

    def _local_available(self, env: Mapping[Place, Fact], ref: CellRef) -> str | None:
        fact = env.get(ref.place)
        if fact is None:
            return "names a place with no fact at this point"
        if isinstance(fact, AggregateFact):
            if not 0 <= ref.ordinal < len(fact.leaves):
                return f"is outside the source's {len(fact.leaves)} cell(s)"
            leaf: Fact = fact.leaves[ref.ordinal]
        elif ref.ordinal != 0:
            return "indexes a scalar source beyond ordinal 0"
        else:
            leaf = fact
        # NARROWER than "the cell is defined": a materialized constant is defined and holds no datapath value.
        return None if isinstance(leaf, Residual) else "holds no datapath value this plan may copy"

    # ---- one site ----

    def _check(
        self, site: PlanSite, env: Mapping[Place, Fact], op: Op, plan: RoutePlan, images: Mapping[int, Fact]
    ) -> None:
        def complain(message: str) -> None:
            self._complaints.append(f"{site}: {type(op).__name__} {message}")

        target = self._target(op)
        if target != plan.target:
            complain(f"targets {target}, plan says {plan.target}")
        width = self._width(op, images)
        if width != len(plan.actions):
            complain(f"has {width} logical cell(s), plan has {len(plan.actions)} row(s)")
            return
        dispositions = self._dispositions(op, images)
        assert len(dispositions) == width, f"{site}: derived {len(dispositions)} dispositions for {width} cells"
        sources = self._sources(op, width)
        for ordinal, (want, action) in enumerate(zip(dispositions, plan.actions)):
            got = (
                Disposition.COPY
                if isinstance(action, CopyCell)
                else Disposition.CONSTANT if isinstance(action, ConstantCell) else Disposition.NONE
            )
            if want is not got:
                complain(f"cell {ordinal} is {got.name}, expected {want.name}")
                continue
            match action:
                case CopyCell(source=source, transfer=transfer):
                    self._check_copy(complain, env, op, ordinal, source, transfer, sources, images)
                case ConstantCell(value=value, kind=kind):
                    self._check_constant(complain, op, ordinal, value, kind, images)
                case NoCell():
                    pass

    def _check_copy(
        self,
        complain: "object",
        env: Mapping[Place, Fact],
        op: Op,
        ordinal: int,
        source: CellRef,
        transfer: CellTransfer,
        sources: list[Place | None],
        images: Mapping[int, Fact],
    ) -> None:
        assert callable(complain)
        wanted = sources[ordinal]
        if wanted is not None and source.place != wanted:
            complain(f"cell {ordinal} copies from {source.place}, expected {wanted}")
        reason = self._available(env, source)
        if reason is not None:
            complain(f"cell {ordinal} copies {source}, which {reason}")
            return
        origin = self._cell_kind(env, source)
        if transfer is not _transfer(origin, self._target_kind(op, ordinal, images)):
            complain(f"cell {ordinal} declares {transfer.name} from {origin.value}")

    def _check_constant(
        self,
        complain: "object",
        op: Op,
        ordinal: int,
        value: StaticValue,
        kind: SemType,
        images: Mapping[int, Fact],
    ) -> None:
        assert callable(complain)
        target_kind = self._target_kind(op, ordinal, images)
        if kind is not target_kind:
            complain(f"cell {ordinal} is a {kind.value} constant where the target is {target_kind.value}")
            return
        leaf = self._target_leaf(op, ordinal, images)
        assert isinstance(leaf, Known)
        if not same(value, conform_value(leaf.value, target_kind)):
            complain(f"cell {ordinal} carries {value}, not the target-side image of {leaf.value}")

    def _target_leaf(self, op: Op, ordinal: int, images: Mapping[int, Fact]) -> Fact:
        fact = self._target_fact(op, images)
        return fact.leaves[ordinal] if isinstance(fact, AggregateFact) else fact

    def _target_kind(self, op: Op, ordinal: int, images: Mapping[int, Fact]) -> SemType:
        target = self._target(op)
        if isinstance(target, StateLeaf):
            return _slot_kind(self._state_resets[target], ordinal)
        leaf = self._target_leaf(op, ordinal, images)
        assert isinstance(leaf, (Known, Residual))
        return _atom_kind(leaf)

    def _cell_kind(self, env: Mapping[Place, Fact], ref: CellRef) -> SemType:
        if isinstance(ref.place, StateLeaf) and ref.place not in env:
            return _slot_kind(self._state_resets[ref.place], ref.ordinal)
        fact = env[ref.place]
        leaf = fact.leaves[ref.ordinal] if isinstance(fact, AggregateFact) else fact
        assert isinstance(leaf, (Known, Residual))
        return _atom_kind(leaf)
