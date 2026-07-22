"""
The definitive settlement of the emission plan: the block order emission walks and the hardware slot each state
leaf gets, decided once over the stabilized spine.

Every refusal here used to be raised during emission, which is the wrong phase for a decision: by then the
compiler has committed, and emission's job is to execute a plan mechanically. Each of these is a function of the
resolved graph alone -- final reachability for the block order, final provenance, store origins and reset
snapshots for the slots -- so it belongs in the definitive post-stabilization resolution, where the facts are
final, and NOT in the iterative transfer, where speculative paths and the deferral net would see it.
"""

import logging
import math
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
    BlockId,
    FunctionUnit,
    OriginStack,
    Place,
    ReturnPlace,
    StateLeaf,
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
    """One hardware slot of a state leaf: its port-ABI name and the constant it resets to."""

    name: str
    reset: FloatConst | BoolConst | IntConst


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
) -> dict[StateLeaf, list[StateSlot]]:
    """
    Every state leaf's slots, named and reset, in the first-store source order that is also the port ABI order.

    The slot name is the owning component's canonical member path from the root joined to the leaf attribute by
    a double underscore, so a top-level attribute ``m`` stays the bare ``m`` while a nested child's ``m``
    becomes ``child__m``. An aggregate slot appends its cell's canonical coordinates with single underscores
    (``x_0``, ``m_0_1``). This is injective except when an attribute name literally spans a boundary (a
    dunder-ish name, or a scalar attribute spelled like another slot's cell); that alias is a located collision
    rejection, never a silent merge, and the leaf blamed is the one that finds the name already claimed.
    """
    assert set(state_resets) == set(store_order), "every discovered leaf is stored, and every store is discovered"
    assert not set(store_order) - set(store_origins), "the resolved graph knows where every leaf was stored"
    slots: dict[StateLeaf, list[StateSlot]] = {}
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
        cells: list[StateSlot] = []
        for ordinal in range(1 if layout is None else len(leaf_paths(layout))):
            name = stem if layout is None else stem + _cell_suffix(leaf_paths(layout)[ordinal])
            owner = claimed.setdefault(name, leaf)
            if owner is not leaf:
                raise AnalysisRejection(
                    f"state slot name collision on '{name}' between distinct component attributes", origin
                )
            cells.append(StateSlot(name, _reset_const(leaf, normalized, snapshot, ordinal, origin)))
        slots[leaf] = cells
    _logger.debug("settled %d state leaf/leaves into %d slot(s)", len(slots), sum(map(len, slots.values())))
    return slots


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
