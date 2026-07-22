"""
The definitive settlement of the emission plan: the block order emission walks and the hardware slot each state
leaf gets, decided once over the stabilized spine.

Every refusal here used to be raised during emission, which is the wrong phase for a decision: by then the
compiler has committed, and emission's job is to execute a plan mechanically. Each of these is a function of the
resolved graph alone -- final reachability for the block order, final provenance, store origins and reset
snapshots for the slots -- so it belongs in the definitive post-stabilization resolution, where the facts are
final, and NOT in the iterative transfer, where speculative paths and the deferral net would see it.

Emission executes all three tables without re-deriving anything, so `verify_settlement` stands where
`verify_route_plans` stands for routing: it RE-DERIVES each table from sources the producer did not consult and
refuses any disagreement, rather than reading the producer's answer back into itself. The three independent
sources are the executable EDGES (a plain reachability closure, sharing no code with the reverse-postorder walk),
the `PyStoreAttr` OPS with their origins (which fix the slot set, the port-ABI order and every attribute name
without consulting the store order the analyzer recorded), and the RESET SNAPSHOTS (the only outside evidence
about a slot's cells, exactly as they are for the routing verifier's per-cell fold check).
"""

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ..._hir import BoolConst, FloatConst, IntConst
from ._analysis_support import AnalysisRejection
from ._fact import (
    AggregateFact,
    ArrayIndex,
    ArrayLayout,
    ContainerFlavor,
    Fact,
    Known,
    LeafPath,
    ListIndex,
    ListLayout,
    RecordField,
    RecordLayout,
    Reference,
    StructuralIndex,
    StructuralLayout,
    TupleIndex,
    TupleLayout,
    ValueLayout,
    child_layouts,
    leaf_paths,
    normalize_static,
)
from ._ir import (
    BindingId,
    BlockId,
    FunctionUnit,
    OriginStack,
    Place,
    PyStoreAttr,
    ReturnPlace,
    StateLeaf,
    StoreOrder,
    StorePlace,
    executable_rpo,
)
from ._signature import (
    ArrayReturn,
    ListReturn,
    RecordReturn,
    ReturnContract,
    ScalarReturn,
    TupleReturn,
    VariadicTupleReturn,
    VoidReturn,
)
from ._value import SemType, StaticValue, as_python, datapath_value

_logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class StateSlot:
    """One CARRIED cell of a state leaf: a hardware register, its port-ABI name and the constant it resets to."""

    name: str
    reset: FloatConst | BoolConst | IntConst


@dataclass(frozen=True, slots=True)
class FoldedCell:
    """
    A cell the fixed point proved the transaction leaves on its reset, so the design does not carry it.

    It gets no register and no state port, for the same reason an attribute the kernel only reads gets neither:
    reads of it materialize the reset at full precision, and a register would hold the reset NARROWED to the
    target format. Publishing that narrowed image beside reads that use the exact one lets a design contradict
    itself -- `state_a_0` reporting 1.0 while `out_0` reports that same cell as greater than 1.0 -- whenever the
    reset is inexact in the carrier. The reset is kept because reads still materialize it.
    """

    reset: FloatConst | BoolConst | IntConst


type StateCell = StateSlot | FoldedCell


def settle_block_order(unit: FunctionUnit, executable_edges: set[tuple[BlockId, BlockId]]) -> list[BlockId]:
    """
    The blocks emission walks, in reverse postorder over executable edges.

    A unit whose canonical exit no path reaches (an unconditional ``while True`` with no break) produces no
    output, so there is nothing to synthesize. Attributed to the deepest reachable terminator, which lives
    inside the non-returning region, so a helper that never returns blames its call site with the callee
    context rather than the root's def line.
    """
    order = executable_rpo(unit.entry, executable_edges)
    assert order, "the entry block is always executable"
    if unit.exit not in order:
        deepest = unit.blocks[order[-1]].terminator
        assert deepest is not None
        raise AnalysisRejection("the function never returns on any path", deepest.origin)
    return order


def settle_state_slots(
    store_order: list[StateLeaf],
    state_resets: dict[StateLeaf, "StaticValue | str"],
    provenance: dict[int, tuple[str, ...]],
    store_origins: dict[StateLeaf, OriginStack],
    runtime_state: set[StateLeaf],
    state_livein: dict[StateLeaf, Fact],
) -> dict[StateLeaf, list[StateCell]]:
    """
    Every state leaf's cells, in the first-store source order that is also the port ABI order, each settled as a
    carried register or a folded constant.

    WHAT THE DESIGN CARRIES IS DECIDED PER CELL, not per leaf, because that is the granularity the fixed point
    decides invariance at: a promoted leaf with one moving cell and one invariant cell carries only the mover.
    A cell is a register exactly when its leaf is promoted AND the settled live-in leaves it runtime-unknown;
    anything the live-in pins to a constant is a `FoldedCell`. That single rule subsumes the leaf-granular one it
    replaces -- an unpromoted leaf reads as its snapshot, so every one of its cells folds and it publishes
    nothing, which is what emission did for it before by filtering on promotion.

    The slot name is the owning component's canonical member path from the root joined to the leaf attribute by
    a double underscore, so a top-level attribute ``m`` stays the bare ``m`` while a nested child's ``m``
    becomes ``child__m``. An aggregate slot appends its cell's canonical coordinates with single underscores
    (``x_0``, ``m_0_1``). This is injective except when an attribute name literally spans a boundary (a
    dunder-ish name, or a scalar attribute spelled like another slot's cell); that alias is a located collision
    rejection, never a silent merge, and the leaf blamed is the one that finds the name already claimed. A folded
    cell claims no name, so it can neither collide nor be collided with.
    """
    assert set(state_resets) == set(store_order), "every discovered leaf is stored, and every store is discovered"
    assert not set(store_order) - set(store_origins), "the resolved graph knows where every leaf was stored"
    slots: dict[StateLeaf, list[StateCell]] = {}
    claimed: dict[str, StateLeaf] = {}
    for leaf in store_order:
        origin = store_origins[leaf]
        snapshot = state_resets[leaf]
        if isinstance(snapshot, str):
            # An inadmissible snapshot is spelled as its type name. No kernel is known to arrive here: the
            # storage-schema check earlier in this same resolution refuses an inadmissible reset by name, and
            # an admitted-but-non-numeric one is refused where the live-in fact is derived. Kept because
            # neither of those guarantees is written down as an invariant this step may rely on.
            raise AnalysisRejection(f"state '{'.'.join(leaf.path)}' has a reset of unsupported type {snapshot}", origin)
        normalized = normalize_static(snapshot)
        layout = normalized.layout if isinstance(normalized, AggregateFact) else None
        path = provenance.get(id(leaf.component))
        if path is None:
            raise AnalysisRejection(
                "a stateful component reached only through an unanchored reference is not supported; "
                "hold it as a direct attribute of the synthesized component",
                origin,
            )
        stem = "__".join(path + leaf.path)
        carried = state_livein.get(leaf) if leaf in runtime_state else normalized
        cells: list[StateCell] = []
        for ordinal in range(1 if layout is None else len(leaf_paths(layout))):
            reset = _reset_const(leaf, normalized, snapshot, ordinal, origin)
            if _cell_is_folded(carried, ordinal):
                cells.append(FoldedCell(reset))
                continue
            name = stem if layout is None else stem + _cell_suffix(leaf_paths(layout)[ordinal])
            owner = claimed.setdefault(name, leaf)
            if owner is not leaf:
                raise AnalysisRejection(
                    f"state slot name collision on '{name}' between distinct component attributes", origin
                )
            cells.append(StateSlot(name, reset))
        slots[leaf] = cells
    _logger.debug(
        "settled %d state leaf/leaves into %d register(s) and %d folded cell(s)",
        len(slots),
        sum(isinstance(cell, StateSlot) for cells in slots.values() for cell in cells),
        sum(isinstance(cell, FoldedCell) for cells in slots.values() for cell in cells),
    )
    return slots


def _cell_is_folded(carried: Fact | None, ordinal: int) -> bool:
    """
    Whether the fixed point pinned this cell to a constant for the whole transaction.

    Only a `Known` per-cell view folds. A live-in that is absent, runtime-unknown or shaped so it has no view at
    this ordinal all leave the cell CARRIED, which is the safe direction: an extra register costs area, a missing
    one loses state. That those cases do not arise is `verify_route_plans`'s per-cell fold check, which re-derives
    the premise from the reset snapshot and the exit facts rather than trusting the live-in this reads.
    """
    if isinstance(carried, AggregateFact):
        cell: Fact | None = carried.leaves[ordinal] if 0 <= ordinal < len(carried.leaves) else None
    else:
        cell = carried if ordinal == 0 else None
    return isinstance(cell, Known)


def _cell_suffix(segments: tuple[object, ...]) -> str:
    suffix = ""
    for segment in segments:
        if isinstance(segment, ArrayIndex):
            suffix += "".join(f"_{coordinate}" for coordinate in segment.coordinates)
        else:
            suffix += f"_{segment.value}"  # type: ignore[attr-defined]  # list cells carry an integer index
    return suffix


def _reset_const(
    leaf: StateLeaf, normalized: Fact, snapshot: StaticValue, ordinal: int, origin: OriginStack
) -> FloatConst | BoolConst | IntConst:
    """
    The slot's reset constant, from the analyzer's one-read attribute snapshot rather than a fresh getattr: a
    live read here could observe state that a permitted compile-time evaluation mutated after stabilization.
    """
    import numpy as np

    if isinstance(normalized, AggregateFact):
        cell = normalized.leaves[ordinal]
        assert isinstance(cell, Known), "an aggregate reset cell must be a concrete scalar"
        current = as_python(cell.value)
    else:
        current = as_python(snapshot)
    if isinstance(current, (bool, np.bool_)):
        return BoolConst(bool(current))
    if isinstance(current, (int, np.integer)):
        return IntConst(int(current))
    if isinstance(current, (float, np.floating)):
        # The same carrier policy a datapath constant follows, decided here because the reset is settled here:
        # a value beyond the binary64 range, or a NaN, has no representable slot image.
        try:
            carried = float(current)
        except OverflowError:
            bits = int(current).bit_length()
            raise AnalysisRejection(
                f"a {bits}-bit integer constant is beyond the binary64 carrier range", origin
            ) from None
        if math.isnan(carried):
            raise AnalysisRejection(
                "Holoso cannot represent a NaN constant. Only [in]finite numbers are supported.", origin
            )
        return FloatConst(carried)
    raise AnalysisRejection(
        f"state '{'.'.join(leaf.path)}' has a reset of unsupported type {type(current).__name__}", origin
    )


@dataclass(frozen=True, slots=True)
class ReturnsNothing:
    """A ``-> None`` contract over an exit that carries no value."""


@dataclass(frozen=True, slots=True)
class ReturnsScalar:
    """One out port of the declared scalar kind."""

    kind: SemType


@dataclass(frozen=True, slots=True)
class ReturnsLeaves:
    """One row per returned cell in canonical leaf order: its typed path and the kind the contract declares."""

    rows: list[tuple[LeafPath, SemType]]


type ReturnPlan = ReturnsNothing | ReturnsScalar | ReturnsLeaves


@dataclass(frozen=True, slots=True)
class SettledReturn:
    """What the unit returns, decided against the declared contract, and where a refusal about it is attributed."""

    plan: ReturnPlan
    origin: OriginStack


def settle_return(unit: FunctionUnit, executable_blocks: set[BlockId], exit_facts: dict[Place, Fact]) -> SettledReturn:
    """
    The declared return contract against what the resolved exit actually carries.

    Every divergence of SHAPE -- a scalar against an aggregate, a wrong arity or array geometry, a container
    flavor erased by a join, an object where a number is declared -- is settled here, because all of it is a
    function of the exit environment and the reachable return stores. What stays in emission is the one check
    that is not: whether the value a leaf ends up EMITTED as carries the declared kind, which can only be read
    off the node once it exists (an integer carried through a state boundary is float-carried while its fact
    still reads integer).
    """
    contract = unit.return_contract
    assert contract is not None, "the root unit always declares a return contract"
    origin = _return_origin(unit, executable_blocks)
    return_fact = exit_facts.get(ReturnPlace())
    returns_value = return_fact is not None and not isinstance(return_fact, Reference)
    returns_object = isinstance(return_fact, Reference) and return_fact.obj is not None
    match contract:
        case VoidReturn():
            if returns_value:
                raise AnalysisRejection("annotated '-> None' but returns a value", origin)
            if returns_object:
                raise AnalysisRejection("annotated '-> None' but returns an object", origin)
            return SettledReturn(ReturnsNothing(), origin)
        case ScalarReturn(kind=kind):
            if not returns_value:
                if returns_object:
                    raise AnalysisRejection(f"return type mismatch: declared {kind.value}, returns an object", origin)
                raise AnalysisRejection(f"return type mismatch: declared {kind.value}, returns nothing", origin)
            if isinstance(return_fact, AggregateFact):
                raise AnalysisRejection(f"return type mismatch: declared {kind.value}, returns an aggregate", origin)
            return SettledReturn(ReturnsScalar(kind), origin)
        case ArrayReturn():
            if not returns_value:
                raise AnalysisRejection("declared an array return but returns nothing", origin)
            if not isinstance(return_fact, AggregateFact):
                raise AnalysisRejection("return shape mismatch: declared an array, returns a scalar", origin)
            _check_layout(contract, return_fact.layout, origin)
            kinds = [SemType.FLOAT] * len(return_fact.leaves)
        case _:  # a tuple/list/record contract: the value must be an aggregate of the declared structure
            if not returns_value:
                raise AnalysisRejection("declared an aggregate return but returns nothing", origin)
            if not isinstance(return_fact, AggregateFact):
                raise AnalysisRejection("declared an aggregate return but returns a scalar", origin)
            _check_layout(contract, return_fact.layout, origin)
            kinds = [_leaf_kind(contract, path) for path in leaf_paths(return_fact.layout)]
    rows: list[tuple[LeafPath, SemType]] = []
    for path, kind, leaf in zip(leaf_paths(return_fact.layout), kinds, return_fact.leaves, strict=True):
        if isinstance(leaf, Known) and not datapath_value(leaf.value):
            spelled = type(as_python(leaf.value)).__name__
            raise AnalysisRejection(
                f"return type mismatch at leaf {_port_keys(path)}: declared {kind.value}, returns a {spelled}", origin
            )
        if isinstance(leaf, Reference):
            raise AnalysisRejection(
                f"return type mismatch at leaf {_port_keys(path)}: declared {kind.value}, returns an object", origin
            )
        rows.append((path, kind))
    return SettledReturn(ReturnsLeaves(rows), origin)


def _return_origin(unit: FunctionUnit, executable_blocks: set[BlockId]) -> OriginStack:
    """
    The earliest return store in source order (the implicit fall-off ``return None`` included), which is what a
    contract refusal is attributed to. The exit terminator only when no reachable path stores a return.
    """
    stores = [
        op.origin
        for block_id in sorted(executable_blocks, key=lambda block_id: block_id.index)
        for op in unit.blocks[block_id].ops
        if isinstance(op, StorePlace) and isinstance(op.place, ReturnPlace)
    ]
    if not stores:
        terminator = unit.blocks[unit.exit].terminator
        assert terminator is not None
        return terminator.origin
    return min(stores, key=lambda origin: origin.position)


def _port_keys(path: LeafPath) -> list[int | str]:
    """A typed leaf path as the established port-name key sequence (indices flatten; array coordinates spread)."""
    keys: list[int | str] = []
    for segment in path:
        match segment:
            case TupleIndex(value=value) | ListIndex(value=value) | StructuralIndex(value=value):
                keys.append(value)
            case ArrayIndex(coordinates=coordinates):
                keys.extend(coordinates)
            case RecordField(name=name):
                keys.append(name)
    return keys


def _check_layout(contract: ReturnContract, layout: "ValueLayout", origin: OriginStack) -> None:
    """The declared return structure against the resolved layout; any shape/arity/flavor divergence rejects."""
    match contract:
        case ScalarReturn():
            if layout is not None:
                raise AnalysisRejection("return type mismatch: declared a scalar, returns an aggregate", origin)
        case TupleReturn(items=items):
            children = _positional_children(layout, ContainerFlavor.TUPLE, "tuple", origin)
            if len(children) != len(items):
                raise AnalysisRejection(
                    f"return arity mismatch: declared a {len(items)}-tuple, returns {len(children)} values", origin
                )
            for item, child in zip(items, children):
                _check_layout(item, child, origin)
        case VariadicTupleReturn(item=item):
            for child in _positional_children(layout, ContainerFlavor.TUPLE, "tuple", origin):
                _check_layout(item, child, origin)
        case ListReturn(item=item):
            for child in _positional_children(layout, ContainerFlavor.LIST, "list", origin):
                _check_layout(item, child, origin)
        case RecordReturn(klass=klass, fields=record_fields):
            if not isinstance(layout, RecordLayout):
                raise AnalysisRejection(
                    f"return type mismatch: declared record {klass.__name__!r}, returns a different value", origin
                )
            if layout.klass is not klass:
                raise AnalysisRejection(
                    f"return type mismatch: declared record {klass.__name__!r}, returns {layout.klass.__name__!r}",
                    origin,
                )
            layout_fields = dict(layout.fields)
            for field_name, field_contract in record_fields:
                _check_layout(field_contract, layout_fields[field_name], origin)
        case ArrayReturn(shape=shape):
            # STRICT flavor: the annotation promises the caller an ndarray of that exact shape, and the model
            # reconstructs one; a list of matching geometry is an observable reflavoring, not RTL plumbing
            # (np.array([...]) is the explicit conversion). The dtype axis is the leaf-kind check's job.
            if not isinstance(layout, ArrayLayout):
                described = "a scalar" if layout is None else "a different container"
                raise AnalysisRejection(
                    f"return shape mismatch: declared a {'x'.join(map(str, shape))} array, returns {described}", origin
                )
            if layout.shape != shape:
                raise AnalysisRejection(
                    f"return shape mismatch: declared {'x'.join(map(str, shape))}, "
                    f"returns {'x'.join(map(str, layout.shape)) or 'a scalar shape'}",
                    origin,
                )
        case _:
            raise AssertionError(f"unhandled return contract {contract}")


def _positional_children(
    layout: "ValueLayout", flavor: ContainerFlavor, spelled: str, origin: OriginStack
) -> tuple["ValueLayout", ...]:
    match layout:
        case TupleLayout() if flavor is ContainerFlavor.TUPLE:
            return child_layouts(layout)
        case ListLayout() if flavor is ContainerFlavor.LIST:
            return child_layouts(layout)
        case StructuralLayout():
            # Strict contracts refuse a flavor-erased join outright: one path returned the declared container and
            # another did not, and picking the declared flavor would silently bless the diverging path.
            raise AnalysisRejection(
                f"return type mismatch: declared a {spelled}, but the container flavor diverges across paths", origin
            )
        case None:
            raise AnalysisRejection(f"return type mismatch: declared a {spelled}, returns a scalar", origin)
        case _:
            raise AnalysisRejection(
                f"return type mismatch: declared a {spelled}, returns a different container", origin
            )


def _leaf_kind(contract: ReturnContract, path: LeafPath) -> SemType:
    """The declared scalar kind governing the leaf at ``path`` (the leaf's contract, walked structurally)."""
    current = contract
    for segment in path:
        match current, segment:
            case (TupleReturn(items=items), TupleIndex(value=value) | StructuralIndex(value=value)):
                current = items[value]
            case (VariadicTupleReturn(item=item), TupleIndex() | StructuralIndex()):
                current = item
            case (ListReturn(item=item), ListIndex() | StructuralIndex()):
                current = item
            case (ArrayReturn(), ArrayIndex()):
                return SemType.FLOAT  # every array-annotation leaf is a float port
            case (RecordReturn(fields=record_fields), RecordField(name=field_name)):
                current = dict(record_fields)[field_name]
            case _:
                raise AssertionError(f"contract walk diverged at {segment} under {current}")
    assert isinstance(current, ScalarReturn), current
    return current.kind


# ---------------------------------------- the independent verifier ----------------------------------------


def verify_settlement(
    unit: FunctionUnit,
    executable_edges: set[tuple[BlockId, BlockId]],
    binding_facts: Mapping[BindingId, Fact],
    exit_facts: Mapping[Place, Fact],
    state_resets: Mapping[StateLeaf, "StaticValue | str"],
    runtime_state: set[StateLeaf],
    state_livein: Mapping[StateLeaf, Fact],
    store_order: list[StateLeaf],
    emission_order: list[BlockId],
    state_slots: Mapping[StateLeaf, list[StateCell]],
    settled_return: SettledReturn,
) -> None:
    """
    Re-derive all three settled tables from the graph and refuse any disagreement with what the producer settled.

    `store_order` is verified alongside them although it is settled a step earlier, because the state-port ABI is
    the PAIR of it and the slot table: emission takes the port order from the store order and only the names and
    resets from the slots, so verifying the slot table alone leaves the half that fixes the order unchecked. That
    is not a supposition -- reversing the slot table's own key order was measured to emit byte-identical HIR for
    every corpus kernel, so a check that read the ABI off the slot table would have been dead on arrival.

    Only one direction of the never-returns decision is reachable from here: a producer that wrongly PROCEEDED
    leaves the canonical exit out of the order it hands over, which is checked. A producer that wrongly refused
    raised instead of returning, so no verifier that runs afterwards can see it -- that direction belongs to the
    behavioural corpus.
    """
    complaints: list[str] = []
    _check_block_order(unit, executable_edges, emission_order, complaints)
    if not complaints:
        # Every check below ranks stores by position in the emission order, so a broken order would report the
        # port ABI as wrong when what is actually wrong is the rank it was measured against.
        _check_state_slots(
            unit,
            binding_facts,
            exit_facts,
            state_resets,
            runtime_state,
            state_livein,
            store_order,
            emission_order,
            state_slots,
            complaints,
        )
    _check_settled_return(unit, exit_facts, settled_return, complaints)
    assert not complaints, "settlement verification failed:\n  " + "\n  ".join(complaints)
    _logger.debug(
        "settlement verified: %d block(s), %d slot(s), %s",
        len(emission_order),
        sum(map(len, state_slots.values())),
        type(settled_return.plan).__name__,
    )


def _check_block_order(
    unit: FunctionUnit,
    executable_edges: set[tuple[BlockId, BlockId]],
    emission_order: list[BlockId],
    complaints: list[str],
) -> None:
    """
    The blocks emission walks, against a plain reachability closure over the executable edges.

    The closure is deliberately not `executable_rpo`: re-running the producer's own walk would agree with it by
    construction and prove nothing. What the closure cannot express is the ORDER, so that is checked by the
    defining property of a reverse postorder instead -- every reachable block other than the entry is preceded by
    one of its own predecessors, which is its depth-first tree parent. Emission seals a block's phis once all its
    predecessors have been emitted, so an order violating this leaves phi arms open at the block that needs them.
    """
    successors: dict[BlockId, list[BlockId]] = {}
    predecessors: dict[BlockId, list[BlockId]] = {}
    for source, target in executable_edges:
        successors.setdefault(source, []).append(target)
        predecessors.setdefault(target, []).append(source)
    reachable = {unit.entry}
    pending = [unit.entry]
    while pending:
        for successor in successors.get(pending.pop(), ()):
            if successor not in reachable:
                reachable.add(successor)
                pending.append(successor)
    seen = set(emission_order)
    duplicated = sorted({block_id for block_id in seen if emission_order.count(block_id) > 1}, key=str)
    if duplicated:
        complaints.append(f"emission order visits {duplicated} more than once")
    missing = sorted(reachable - seen, key=str)
    if missing:
        complaints.append(f"executable blocks the emission order omits: {missing}")
    surplus = sorted(seen - reachable, key=str)
    if surplus:
        complaints.append(f"emission order walks blocks no executable path reaches: {surplus}")
    if not emission_order or emission_order[0] != unit.entry:
        complaints.append("the emission order does not start at the entry block, where the input ports are built")
    if unit.exit not in seen:
        complaints.append("the canonical exit is not in the emission order, yet the unit was not refused")
    position = {block_id: index for index, block_id in enumerate(emission_order)}
    unreached = [
        block_id
        for index, block_id in enumerate(emission_order)
        if index > 0 and not any(position.get(p, index) < index for p in predecessors.get(block_id, ()))
    ]
    if unreached:
        complaints.append(f"emission order places {unreached} before every predecessor, so it is not a postorder")


def _check_state_slots(
    unit: FunctionUnit,
    binding_facts: Mapping[BindingId, Fact],
    exit_facts: Mapping[Place, Fact],
    state_resets: Mapping[StateLeaf, "StaticValue | str"],
    runtime_state: set[StateLeaf],
    state_livein: Mapping[StateLeaf, Fact],
    store_order: list[StateLeaf],
    emission_order: list[BlockId],
    state_slots: Mapping[StateLeaf, list[StateCell]],
    complaints: list[str],
) -> None:
    """
    The port ABI against the stores that create it and the snapshots that fix its cells.

    Which leaves exist and in WHICH ORDER is re-derived from the executable `PyStoreAttr` ops -- their origins
    and their object facts -- rather than from the store order the analyzer recorded, so a store the recorder
    lost or mis-ranked cannot agree with itself. Each leaf's cells then come from its reset snapshot: the count,
    the canonical coordinates that spell each name's suffix, and the constant each cell resets to. The one part
    no outside source fixes is the component's provenance prefix, so the prefix is solved from the leaf's own
    first slot and required to be consistent across its cells, to end at the attribute the STORE names, and to be
    injective across the whole table.
    """
    rank = {block_id: index for index, block_id in enumerate(emission_order)}
    first_store: dict[StateLeaf, StoreOrder] = {}
    for block_id in sorted(emission_order, key=lambda block_id: block_id.index):
        for op in unit.blocks[block_id].ops:
            if not isinstance(op, PyStoreAttr):
                continue
            obj_fact = binding_facts.get(op.obj)
            if not isinstance(obj_fact, Reference):
                complaints.append(f"the store at {op.origin.site} targets no component, so no slot can be settled")
                continue
            leaf = StateLeaf(obj_fact.obj, (op.name,))
            key = StoreOrder(op.origin.position, rank[block_id])
            if leaf not in first_store or key < first_store[leaf]:
                first_store[leaf] = key
    expected_order = sorted(first_store, key=lambda leaf: first_store[leaf])
    if store_order != expected_order:
        complaints.append(
            "the store order disagrees with the stores in the resolved graph, and it IS the state-port order: "
            f"settled {[_spell(leaf) for leaf in store_order]} against {[_spell(leaf) for leaf in expected_order]}"
        )
    if list(state_slots) != store_order:
        complaints.append(
            f"the slot table covers {[_spell(leaf) for leaf in state_slots]} where the store order names "
            f"{[_spell(leaf) for leaf in store_order]}"
        )
    claimed: dict[str, StateLeaf] = {}
    for leaf, slots in state_slots.items():
        snapshot = state_resets.get(leaf)
        if snapshot is None or isinstance(snapshot, str):
            complaints.append(f"state '{_spell(leaf)}' has cells but no admitted reset snapshot to settle them from")
            continue
        _check_leaf_slots(leaf, snapshot, exit_facts.get(leaf), slots, complaints)
        # WHICH cells exist in hardware, against the fixed point's own per-cell invariance answer. This compares
        # two producer tables rather than re-deriving from outside, and that is sound only because D's premise is
        # itself re-derived elsewhere: `verify_route_plans` checks every folded cell against the reset snapshot
        # and the canonical exit. So D is verified against outside evidence, and the slot table against D. Nothing
        # else compares them, and they must agree -- emission reads a register exactly where the table says one
        # exists while the route plan constant-folds exactly where D says the cell is pinned, so a disagreement
        # either reads a register the plan never fills or publishes a port the design contradicts.
        carried = state_livein.get(leaf) if leaf in runtime_state else normalize_static(snapshot)
        for ordinal, cell in enumerate(slots):
            folded = _cell_is_folded(carried, ordinal)
            if folded and isinstance(cell, StateSlot):
                complaints.append(
                    f"state '{_spell(leaf)}' cell {ordinal} is settled as register '{cell.name}', but the fixed "
                    "point folded it to a constant, so its port would publish the reset narrowed to the carrier"
                )
            elif not folded and isinstance(cell, FoldedCell):
                complaints.append(
                    f"state '{_spell(leaf)}' cell {ordinal} is settled as folded, but the fixed point carries it, "
                    "so the design would lose it between transactions"
                )
        for cell in slots:
            if not isinstance(cell, StateSlot):
                continue  # a folded cell claims no name, so it can neither collide nor be collided with
            owner = claimed.setdefault(cell.name, leaf)
            if owner is not leaf:
                complaints.append(f"slot name '{cell.name}' is shared by '{_spell(owner)}' and '{_spell(leaf)}'")


def _check_leaf_slots(
    leaf: StateLeaf,
    snapshot: StaticValue,
    exit_fact: Fact | None,
    slots: list[StateCell],
    complaints: list[str],
) -> None:
    """
    One leaf's cells against its reset snapshot.

    Whether each cell is CARRIED or FOLDED is not re-derived here: that premise is the fixed point's per-cell
    invariance claim, and `verify_route_plans` already re-derives it from the reset snapshot and the exit facts,
    which is the only outside evidence about a fold. What this owes is everything that follows from the split --
    the cell count, the canonical coordinates each carried cell's name spells, and every reset constant, folded
    cells included, since a folded cell's reset is what its reads materialize.
    """
    normalized = normalize_static(snapshot)
    if isinstance(normalized, AggregateFact):
        paths = leaf_paths(normalized.layout)
        cells: tuple[Fact, ...] = normalized.leaves
    else:
        paths = ((),)
        cells = (normalized,)
    if len(slots) != len(cells):
        complaints.append(f"state '{_spell(leaf)}' settled {len(slots)} cell(s) over a {len(cells)}-cell reset")
        return
    # A second, independent witness to the width: the value the canonical exit leaves in the leaf is what every
    # state port publishes, so a slot count the exit cannot fill is wrong however well it agrees with the reset.
    if isinstance(exit_fact, AggregateFact) and len(exit_fact.leaves) != len(slots):
        complaints.append(
            f"state '{_spell(leaf)}' settled {len(slots)} slot(s) but the canonical exit carries "
            f"{len(exit_fact.leaves)} cell(s)"
        )
    attribute = "__".join(leaf.path)
    aggregate = isinstance(normalized, AggregateFact)
    suffixes = [_cell_suffix(path) if aggregate else "" for path in paths]
    # The component prefix is the one part of a name no outside source fixes, so it is solved from the leaf's
    # first CARRIED cell -- not necessarily cell 0, since a folded cell claims no name -- and then required to be
    # the same for every other carried cell of the leaf.
    anchor = next(((ordinal, cell) for ordinal, cell in enumerate(slots) if isinstance(cell, StateSlot)), None)
    prefix: str | None = None
    if anchor is not None:
        index, first = anchor
        tail = attribute + suffixes[index]
        if not first.name.endswith(tail):
            complaints.append(f"slot '{first.name}' does not name the attribute '{attribute}' the store writes")
        else:
            prefix = first.name[: len(first.name) - len(tail)]
            if prefix and not prefix.endswith("__"):
                complaints.append(f"slot '{first.name}' joins its component path to '{attribute}' without a separator")
    for ordinal, (slot, suffix, cell) in enumerate(zip(slots, suffixes, cells, strict=True)):
        if isinstance(slot, StateSlot) and prefix is not None:
            expected = prefix + attribute + suffix
            if slot.name != expected:
                complaints.append(f"state '{_spell(leaf)}' cell {ordinal} is named '{slot.name}', not '{expected}'")
        if not isinstance(cell, Known):
            complaints.append(f"state '{_spell(leaf)}' cell {ordinal} resets to no concrete value")
            continue
        disagreement = _reset_disagreement(slot.reset, as_python(cell.value))
        if disagreement is not None:
            complaints.append(f"state '{_spell(leaf)}' cell {ordinal} {disagreement}")


def _reset_disagreement(reset: FloatConst | BoolConst | IntConst, expected: object) -> str | None:
    """
    The settled reset constant against the value the snapshot itself holds at that cell's canonical path.

    A NaN or an out-of-carrier magnitude is a producer refusal, so reaching one here means the refusal was
    skipped rather than that the value is legal -- hence a complaint instead of the exception rebuilding the
    constant would raise.
    """
    import numpy as np

    wanted: FloatConst | BoolConst | IntConst
    if isinstance(expected, (bool, np.bool_)):
        wanted = BoolConst(bool(expected))
    elif isinstance(expected, (int, np.integer)):
        wanted = IntConst(int(expected))
    elif isinstance(expected, (float, np.floating)):
        carried = float(expected)
        if math.isnan(carried):
            return "resets to a NaN, which the settlement refuses rather than carries"
        wanted = FloatConst(carried)
    else:
        return f"resets to a {type(expected).__name__}, which no slot can hold"
    return None if reset == wanted else f"resets to {reset}, but its snapshot cell holds {wanted}"


def _spell(leaf: StateLeaf) -> str:
    return ".".join(leaf.path)


def _check_settled_return(
    unit: FunctionUnit,
    exit_facts: Mapping[Place, Fact],
    settled: SettledReturn,
    complaints: list[str],
) -> None:
    """
    The settled return plan against the declared contract, enumerated in the opposite direction to the producer.

    `settle_return` walks the resolved exit LAYOUT and looks each leaf's declared kind up in the contract; this
    walks the CONTRACT down and takes arity from the layout only where the contract genuinely leaves it free (a
    variadic tuple, a list, an array). So a row that is dropped, duplicated, mis-ordered or given a kind from the
    wrong leaf disagrees between the two enumerations, where re-running the producer's own direction could not.
    The attribution origin is not re-derived: it selects among true diagnostics rather than deciding the ABI, and
    the corpus pins it where it matters.
    """
    contract = unit.return_contract
    assert contract is not None, "the root unit always declares a return contract"
    return_fact = exit_facts.get(ReturnPlace())
    match contract, settled.plan:
        case (VoidReturn(), ReturnsNothing()):
            return
        case (ScalarReturn(kind=declared), ReturnsScalar(kind=settled_kind)):
            if settled_kind is not declared:
                complaints.append(f"the return plan declares {settled_kind.value} where the contract says {declared}")
            if isinstance(return_fact, AggregateFact):
                complaints.append("a scalar return plan settled over an aggregate exit value")
            return
        case (VoidReturn() | ScalarReturn(), _) | (_, ReturnsNothing() | ReturnsScalar()):
            complaints.append(f"the contract {type(contract).__name__} settled as {type(settled.plan).__name__}")
            return
    assert isinstance(settled.plan, ReturnsLeaves)
    if not isinstance(return_fact, AggregateFact):
        complaints.append("a leaf-wise return plan settled over an exit value that is not an aggregate")
        return
    expected = _contract_rows(contract, return_fact.layout)
    if expected is None:
        complaints.append("the declared return contract and the resolved exit layout do not correspond")
        return
    if expected != settled.plan.rows:
        complaints.append(
            f"the settled return rows disagree with the contract walked leaf by leaf: settled "
            f"{[(_port_keys(path), kind.value) for path, kind in settled.plan.rows]} against "
            f"{[(_port_keys(path), kind.value) for path, kind in expected]}"
        )


def _contract_rows(contract: ReturnContract, layout: "ValueLayout") -> "list[tuple[LeafPath, SemType]] | None":
    """Every returned leaf as the CONTRACT spells it, or None where the contract and the layout do not correspond."""
    match contract, layout:
        case (ScalarReturn(kind=kind), None):
            return [((), kind)]
        case (ArrayReturn(shape=shape), ArrayLayout(shape=layout_shape)) if shape == layout_shape:
            return [(path, SemType.FLOAT) for path in leaf_paths(layout)]
        case (TupleReturn(items=items), TupleLayout()) if len(items) == len(child_layouts(layout)):
            return _join_rows(items, child_layouts(layout), TupleIndex)
        case (VariadicTupleReturn(item=item), TupleLayout()):
            children = child_layouts(layout)
            return _join_rows([item] * len(children), children, TupleIndex)
        case (ListReturn(item=item), ListLayout()):
            children = child_layouts(layout)
            return _join_rows([item] * len(children), children, ListIndex)
        case (RecordReturn(klass=klass, fields=fields), RecordLayout(klass=layout_klass)) if klass is layout_klass:
            by_name = dict(layout.fields)
            if {name for name, _ in fields} != set(by_name):
                return None
            rows: list[tuple[LeafPath, SemType]] = []
            for name, field_contract in fields:
                nested = _contract_rows(field_contract, by_name[name])
                if nested is None:
                    return None
                rows += [((RecordField(name), *path), kind) for path, kind in nested]
            return rows
    return None


def _join_rows(
    contracts: "Sequence[ReturnContract]", children: "tuple[ValueLayout, ...]", segment: "type[TupleIndex | ListIndex]"
) -> "list[tuple[LeafPath, SemType]] | None":
    rows: list[tuple[LeafPath, SemType]] = []
    for index, (item, child) in enumerate(zip(contracts, children, strict=True)):
        nested = _contract_rows(item, child)
        if nested is None:
            return None
        rows += [((segment(index), *path), kind) for path, kind in nested]
    return rows
