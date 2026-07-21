"""
FIR -> HIR emission over the analyzer's stabilized residual graph and its emission plan: only executable blocks and
edges are walked, every fact and call plan comes from the analysis result (emission never folds, never resolves the
library registry, never replays the transfer), Known values materialize as constants at their use sites, and
residual operations become typed HIR operations through one typed materializer. Value numbering is Braun-style
sealed-block SSA over Places (named locals, state leaves, the return place): straight-line reads chase the unique
predecessor, joins create phis, loop headers read through open phis closed once their latch block is emitted, and
single-value joins collapse without a phi. HIR phi arms are keyed by predecessor block, which the 1:1 block mapping
preserves.

State: every read/written leaf becomes a state_read and, if in the promoted set, a state slot named by its
attribute path with its reset snapshot and canonical-exit live-out; the returned value becomes the out port,
following the established port ABI.
"""

import logging
import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from ..._errors import UnsupportedConstruct
from ..._util import RelationalOp
from .._ast_support import port_name, state_port_name
from ..._hir import (
    BoolAnd,
    BoolConst,
    BoolNot,
    BoolOr,
    BoolSelect,
    BoolToFloat,
    BoolType,
    FloatAdd,
    FloatConst,
    FloatDiv,
    FloatExp2,
    FloatLog2,
    FloatMul,
    FloatNeg,
    FloatRelational,
    BoolXor,
    BoolToInt,
    FloatToBool,
    FloatToInt,
    FloatType,
    Hir,
    HirBuilder,
    IntAbs,
    IntAdd,
    IntAnd,
    IntDivFloor,
    IntMod,
    IntMul,
    IntNeg,
    IntOr,
    IntRelational,
    IntSelect,
    IntShiftLeft,
    IntShiftRight,
    IntSub,
    IntConst,
    IntToBool,
    IntToFloat,
    IntType,
    IntXor,
    Operation,
    Operator,
    Phi,
    Select,
    Type,
)
from ._analyze import Analyzer, CallLowering, CallPlan, ResidualUnit, verify_plan_totality
from ._fact import (
    AggregateFact,
    AggregateLayout,
    ArrayDType,
    ArrayIndex,
    ArrayLayout,
    AtomicFact,
    ContainerFlavor,
    Fact,
    Known,
    Reference,
    LeafPath,
    ListIndex,
    ListLayout,
    RecordField,
    RecordLayout,
    Residual,
    StructuralIndex,
    StructuralLayout,
    TupleIndex,
    TupleLayout,
    ValueLayout,
    child_layouts,
    child_slice,
    leaf_count,
    leaf_paths,
    materialize_static,
    normalize_static,
    outer_arity,
)
from ._signature import (
    ArrayReturn,
    RecordReturn,
    ListReturn,
    ReturnContract,
    ScalarReturn,
    TupleReturn,
    VariadicTupleReturn,
    VoidReturn,
)
from ._ir import (
    BindingId,
    BlockId as FirBlockId,
    Branch as FirBranch,
    BuildList,
    BuildTuple,
    executable_rpo,
    Jump as FirJump,
    LoadConst,
    LoadRef,
    LoadPlace,
    Local,
    Op,
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
    OriginStack,
    ReturnPlace,
    SelectMode,
    StateLeaf,
    StorePlace,
    UnbindPlace,
    UnitExit,
    LocatedRejection,
    source_position,
)
from ._opsem import BinOp, UnOp
from ._value import (
    MetaInt,
    NpBool,
    NpFloat,
    NpInt,
    SemType,
    StaticBool,
    StaticFloat,
    StaticValue,
    admit,
    as_python,
)

_logger = logging.getLogger(__name__)

# A literal power expands to |exponent|-1 chained multiplies; this bounds that expansion so ``x**(10**9)`` refuses
# instead of hanging emission, while leaving any realistic exponent (a degree-N monomial) free to expand.
_MAX_POWER_CHAIN = 1024


class EmissionRejection(LocatedRejection, UnsupportedConstruct):
    """
    A located refusal discovered during HIR emission: an unsupported construct that survived analysis. A rejection
    that can be decided during analysis belongs there; emission attributes each refusal to the op, state store, or
    return store it was lowering.
    """


def lower_fir(kernel: object) -> Hir:
    """The front-end pipeline: build, analyze to the W/D fixed point, emit HIR from the analysis plan."""
    result = Analyzer(kernel).fixpoint()
    verify_plan_totality(result)
    return _Emitter(result).emit()


def _carrier_float(value: object, origin: OriginStack) -> float:
    # A finite inexact integer (2**53 + 1) rounds into the binary64 carrier -- accepted C-style precision loss under
    # the fastmath charter. A value beyond the carrier range entirely (10**400) is a located rejection, and so is a
    # NaN, which the HIR constant domain would otherwise refuse unlocated deeper down.
    try:
        result = float(value)  # type: ignore[arg-type]
    except OverflowError:
        bits = int(value).bit_length()  # type: ignore[call-overload]  # never via str(): 4300-digit conversion cap
        raise EmissionRejection(f"a {bits}-bit integer constant is beyond the binary64 carrier range", origin) from None
    if math.isnan(result):
        raise EmissionRejection(
            "Holoso cannot represent a NaN constant. Only [in]finite numbers are supported.", origin
        )
    return result


@dataclass(frozen=True, slots=True)
class _LeafPlace:
    """
    One scalar cell of a root Place: the SSA unit, keyed by the cell's ORDINAL in the canonical flat leaf order.
    Every join the fact domain admits preserves arity and leaf count (a flavor-degrading tuple-meets-list join
    included), so ordinals align across arms whose typed path vocabularies differ; typed paths serve only the port
    ABI at the exit. A scalar root is ordinal 0.
    """

    root: Place
    ordinal: int


def _port_path(path: LeafPath) -> list[int | str]:
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


def _validate_return_layout(contract: ReturnContract, layout: "ValueLayout", origin: OriginStack) -> None:
    """The declared return structure against the emitted layout; any shape/arity/flavor divergence rejects."""
    match contract:
        case ScalarReturn():
            if layout is not None:
                raise EmissionRejection("return type mismatch: declared a scalar, returns an aggregate", origin)
        case TupleReturn(items=items):
            children = _positional_children(layout, ContainerFlavor.TUPLE, "tuple", origin)
            if len(children) != len(items):
                raise EmissionRejection(
                    f"return arity mismatch: declared a {len(items)}-tuple, returns {len(children)} values", origin
                )
            for item, child in zip(items, children):
                _validate_return_layout(item, child, origin)
        case VariadicTupleReturn(item=item):
            for child in _positional_children(layout, ContainerFlavor.TUPLE, "tuple", origin):
                _validate_return_layout(item, child, origin)
        case ListReturn(item=item):
            for child in _positional_children(layout, ContainerFlavor.LIST, "list", origin):
                _validate_return_layout(item, child, origin)
        case RecordReturn(klass=klass, fields=record_fields):
            if not isinstance(layout, RecordLayout):
                raise EmissionRejection(
                    f"return type mismatch: declared record {klass.__name__!r}, returns a different value", origin
                )
            if layout.klass is not klass:
                raise EmissionRejection(
                    f"return type mismatch: declared record {klass.__name__!r}, returns {layout.klass.__name__!r}",
                    origin,
                )
            layout_fields = dict(layout.fields)
            for field_name, field_contract in record_fields:
                _validate_return_layout(field_contract, layout_fields[field_name], origin)
        case ArrayReturn(shape=shape):
            # STRICT flavor: the annotation promises the caller an ndarray of that exact shape, and the model
            # reconstructs one; a list of matching geometry is an observable reflavoring, not RTL plumbing
            # (np.array([...]) is the explicit conversion). The dtype axis is the leaf-kind check's job.
            if not isinstance(layout, ArrayLayout):
                described = "a scalar" if layout is None else "a different container"
                raise EmissionRejection(
                    f"return shape mismatch: declared a {'x'.join(map(str, shape))} array, returns {described}", origin
                )
            if layout.shape != shape:
                raise EmissionRejection(
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
            raise EmissionRejection(
                f"return type mismatch: declared a {spelled}, but the container flavor diverges across paths", origin
            )
        case None:
            raise EmissionRejection(f"return type mismatch: declared a {spelled}, returns a scalar", origin)
        case _:
            raise EmissionRejection(
                f"return type mismatch: declared a {spelled}, returns a different container", origin
            )


def _contract_leaf_kind(contract: ReturnContract, path: LeafPath) -> SemType:
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


def _is_bool_fact(fact: Fact | None) -> bool:
    match fact:
        case Residual(type=SemType.BOOL):
            return True
        case Known(value=value):
            return isinstance(value, (StaticBool, NpBool))
        case _:
            return False


class _Emitter:
    def __init__(self, result: ResidualUnit) -> None:
        self._result = result
        exit_terminator = result.unit.blocks[result.unit.exit].terminator
        assert isinstance(exit_terminator, UnitExit)
        # The refusal attribution cursor: the origin of whatever emission is currently lowering -- each op as it is
        # visited, the branch condition at a terminator, the earliest return store during exit processing -- so the
        # deep shared helpers (materializer, constant carrier) locate their rejections without threading an origin
        # through every signature. State-leaf refusals bypass it for the leaf's own first-store origin.
        self._origin: OriginStack = exit_terminator.origin
        self._builder = HirBuilder()
        self._fir_to_hir: dict[FirBlockId, int] = {}
        self._definitions: dict[tuple[FirBlockId, _LeafPlace], int] = {}
        self._sealed: set[FirBlockId] = set()
        self._emitted: set[FirBlockId] = set()
        self._open_phis: dict[FirBlockId, list[tuple[_LeafPlace, int]]] = {}
        self._predecessors: dict[FirBlockId, list[FirBlockId]] = {}
        for source, target in sorted(result.executable_edges, key=lambda e: (e[0].index, e[1].index)):
            self._predecessors.setdefault(target, []).append(source)
        self._state_reads: dict[tuple[StateLeaf, int], int] = {}
        self._state_order: list[StateLeaf] = []
        self._exit_identity_memo: dict[int, object] = {}
        self._slot_names: dict[str, tuple[StateLeaf, int]] = {}  # rendered slot name -> owning cell (collisions)

    def emit(self) -> Hir:
        unit = self._result.unit
        order = executable_rpo(unit.entry, self._result.executable_edges)
        if unit.exit not in order:
            # No path reaches the canonical exit (e.g. an unconditional `while True` with no break): the kernel
            # produces no output, so there is nothing to synthesize. A located refusal, not a downstream crash --
            # attributed to the deepest reachable terminator, which lives inside the non-returning region, so a
            # helper that never returns blames its call site with the callee context rather than the root's def line.
            deepest = self._result.unit.blocks[order[-1]].terminator
            assert deepest is not None
            raise EmissionRejection("the function never returns on any path", deepest.origin)
        for fir_id in order:
            self._fir_to_hir[fir_id] = self._builder.block()
        self._builder.position_at(self._fir_to_hir[unit.entry])
        parameters = unit.params[1:] if unit.bound_self is not None else unit.params
        entry_facts = self._result.block_in[unit.entry].facts
        for parameter in parameters:
            entry_fact = entry_facts.get(Local(parameter))
            if isinstance(entry_fact, AggregateFact):
                # An array or record parameter decomposes into one input port per leaf under the shared
                # path-name convention (array coordinates and record field names alike), each typed by its
                # own leaf kind so a boolean record field gets a boolean port.
                for ordinal, path in enumerate(leaf_paths(entry_fact.layout)):
                    cell = parameter.name + "".join(f"_{key}" for key in _port_path(path))
                    vid = self._builder.input(cell, self._fact_port_type(entry_fact.leaves[ordinal]))
                    self._write(unit.entry, _LeafPlace(Local(parameter), ordinal), vid)
            else:
                vid = self._builder.input(parameter.name, self._fact_port_type(entry_fact))
                self._write(unit.entry, Local(parameter), vid)
        for fir_id in order:
            self._emit_block(fir_id)
            self._emitted.add(fir_id)
            for successor in list(self._predecessors):
                if successor not in self._sealed and all(p in self._emitted for p in self._predecessors[successor]):
                    self._seal(successor)
        self._finish_exit()
        return self._builder.finish()

    # ---------------------------------------- SSA over Places ----------------------------------------

    def _type_of(self, vid: int) -> Type:
        return self._builder.type_of(vid)

    def _write(self, block: FirBlockId, place: "Place | _LeafPlace", vid: int) -> None:
        leaf = place if isinstance(place, _LeafPlace) else _LeafPlace(place, 0)
        self._definitions[(block, leaf)] = vid

    def _read(self, block: FirBlockId, place: "Place | _LeafPlace") -> int:
        if not isinstance(place, _LeafPlace):
            place = _LeafPlace(place, 0)
        # Canonical Braun read: a multi-predecessor block ALWAYS caches an empty phi before reading its operands, so a
        # read cycling back through the same block (a loop's latch, whether or not the header is already sealed) finds
        # the phi instead of recursing forever. The phi's arms are filled straight away for a sealed block, or at
        # sealing for a not-yet-sealed one; a trivial phi (one distinct real operand) collapses to that operand.
        if (block, place) in self._definitions:
            return self._definitions[(block, place)]
        # Chase a straight-line single-predecessor chain ITERATIVELY (a long static unroll can be thousands of blocks
        # deep, past Python's recursion limit) down to a source: a defined block, the entry, or a real join. The join
        # is resolved once (recursion there is bounded by CFG nesting, not trip count), then the resolved value is
        # written back into every block on the chain.
        chain: list[FirBlockId] = []
        cursor = block
        while (cursor, place) not in self._definitions:
            predecessors = self._predecessors.get(cursor, [])
            if len(predecessors) == 1 and cursor in self._sealed:
                chain.append(cursor)
                cursor = predecessors[0]
            else:
                break
        vid = self._resolve_source(cursor, place)
        for member in chain:
            self._write(member, place, vid)
        return vid

    def _resolve_source(self, block: FirBlockId, place: _LeafPlace) -> int:
        if (block, place) in self._definitions:
            return self._definitions[(block, place)]
        if not self._predecessors.get(block):
            vid = self._entry_value(place)
            self._write(block, place, vid)
            return vid
        with self._at(block):
            phi = self._builder.empty_phi(self._place_type(block, place))
        self._write(block, place, phi)  # cache BEFORE reading operands -- this is the cycle break
        if block not in self._sealed:
            self._open_phis.setdefault(block, []).append((place, phi))
            return phi
        return self._fill_phi(block, place, phi)

    def _fill_phi(self, block: FirBlockId, place: _LeafPlace, phi: int) -> int:
        phi_type = self._place_type(block, place)
        arms: list[tuple[int, int]] = []
        for p in self._predecessors[block]:
            arm = self._read(p, place)
            if isinstance(phi_type, FloatType) and isinstance(self._type_of(arm), IntType):
                with self._at(p):  # a mixed int/float join promotes the integer arm to float on its own edge
                    arm = self._builder.operation(IntToFloat(), [arm])
            arms.append((self._fir_to_hir[p], arm))
        self._builder.set_phi_arms(phi, arms)
        distinct = {vid for _, vid in arms if vid != phi}  # ignore the phi's own self-reference
        if len(distinct) == 1:
            trivial = next(iter(distinct))
            self._write(block, place, trivial)  # a trivial phi is just its single real operand
            return trivial
        return phi

    def _place_type(self, block: FirBlockId, place: _LeafPlace) -> Type:
        # The phi's type is the Place's own type, known from the analyzer fact independently of its operands. A state
        # leaf whose live-in was never carried in the block environment falls back to its reset snapshot's type.
        fact = self._result.block_in[block].facts.get(place.root)
        if fact is None and isinstance(place.root, StateLeaf):
            fact = self._leaf_fact(place.root)
        if isinstance(fact, AggregateFact):
            atom: Fact | None = fact.leaves[place.ordinal]
        else:
            assert place.ordinal == 0, f"a leaf ordinal into a scalar fact at {place}"
            atom = fact
        return self._fact_port_type(atom)

    @staticmethod
    def _sem_of(ty: Type) -> SemType:
        if isinstance(ty, BoolType):
            return SemType.BOOL
        if isinstance(ty, IntType):
            return SemType.INT
        return SemType.FLOAT

    @staticmethod
    def _fact_port_type(fact: Fact | None) -> Type:
        assert not isinstance(fact, AggregateFact), "an aggregate parameter must decompose before port typing"
        if _is_bool_fact(fact):
            return BoolType()
        if fact == Residual(SemType.INT) or (isinstance(fact, Known) and isinstance(fact.value, (MetaInt, NpInt))):
            return IntType()
        return FloatType()

    @staticmethod
    def _fact_sem(fact: Fact) -> SemType:
        if _is_bool_fact(fact):
            return SemType.BOOL
        if isinstance(fact, Residual) and fact.type is SemType.INT:
            return SemType.INT
        if isinstance(fact, Known) and isinstance(fact.value, (MetaInt, NpInt)):
            return SemType.INT
        return SemType.FLOAT

    @contextmanager
    def _at(self, block: FirBlockId) -> Iterator[None]:
        # A phi inserted during a read must not move the builder off the block currently being emitted.
        saved = self._builder.current_block_or_none()
        self._builder.position_at(self._fir_to_hir[block])
        try:
            yield
        finally:
            if saved is not None:
                self._builder.position_at(saved)

    def _seal(self, block: FirBlockId) -> None:
        self._sealed.add(block)
        for place, phi in self._open_phis.pop(block, []):
            self._fill_phi(block, place, phi)

    def _entry_value(self, place: _LeafPlace) -> int:
        if isinstance(place.root, StateLeaf):
            return self._state_read(place.root, place.ordinal)
        raise AssertionError(f"read of an undefined place '{place}' escaped analysis")

    def _slot_name(self, leaf: StateLeaf, ordinal: int = 0) -> str:
        # The slot name is the owning component's canonical member path from the root joined to the leaf attribute by a
        # double underscore, so a top-level attribute ``m`` stays the bare ``m`` (the established port ABI) while a
        # nested child's ``m`` becomes ``child__m``. An aggregate slot appends its cell's canonical coordinates with
        # single underscores (``x_0``, ``m_0_1``). This is injective except when an attribute name literally spans a
        # boundary (a dunder-ish name, or a scalar attribute spelled like another slot's cell); that alias is a
        # located collision rejection, never a silent merge.
        path = self._result.provenance.get(id(leaf.component))
        if path is None:
            raise EmissionRejection(
                "a stateful component reached only through an unanchored reference is not supported; "
                "hold it as a direct attribute of the synthesized component",
                self._leaf_origin(leaf),
            )
        name = "__".join(path + leaf.path)
        layout = self._reset_layout(leaf)
        if layout is not None:
            segments = leaf_paths(layout)[ordinal]
            for segment in segments:
                if isinstance(segment, ArrayIndex):
                    name += "".join(f"_{coordinate}" for coordinate in segment.coordinates)
                else:
                    name += f"_{segment.value}"  # type: ignore[union-attr]  # list cells carry an integer index
        owner = self._slot_names.setdefault(name, (leaf, ordinal))
        if owner != (leaf, ordinal):
            raise EmissionRejection(
                f"state slot name collision on '{name}' between distinct component attributes", self._leaf_origin(leaf)
            )
        return name

    def _leaf_origin(self, leaf: StateLeaf) -> OriginStack:
        return self._result.store_origins.get(leaf, self._origin)

    def _reset_layout(self, leaf: StateLeaf) -> "AggregateLayout | None":
        snapshot = self._result.state_resets.get(leaf)
        assert snapshot is not None, f"reset for {leaf} missing from the analysis plan"
        if isinstance(snapshot, str):
            raise EmissionRejection(
                f"state '{'.'.join(leaf.path)}' has a reset of unsupported type {snapshot}", self._leaf_origin(leaf)
            )
        normalized = normalize_static(snapshot)
        return normalized.layout if isinstance(normalized, AggregateFact) else None

    def _state_cells(self, leaf: StateLeaf) -> int:
        layout = self._reset_layout(leaf)
        return 1 if layout is None else leaf_count(layout)

    def _state_read(self, leaf: StateLeaf, ordinal: int = 0) -> int:
        if (leaf, ordinal) not in self._state_reads:
            reset = self._leaf_reset(leaf, ordinal)
            slot_type: Type = (
                BoolType()
                if isinstance(reset, BoolConst)
                else IntType() if isinstance(reset, IntConst) else FloatType()
            )
            self._state_reads[(leaf, ordinal)] = self._builder.state_read(self._slot_name(leaf, ordinal), slot_type)
            if leaf not in self._state_order:
                self._state_order.append(leaf)
        return self._state_reads[(leaf, ordinal)]

    def _slot_kind(self, leaf: StateLeaf, ordinal: int = 0) -> SemType:
        """The slot's fixed kind, straight from the reset snapshot (the storage schema pins every store to it)."""
        reset = self._leaf_reset(leaf, ordinal)
        if isinstance(reset, BoolConst):
            return SemType.BOOL
        return SemType.INT if isinstance(reset, IntConst) else SemType.FLOAT

    def _leaf_reset(self, leaf: StateLeaf, ordinal: int = 0) -> FloatConst | BoolConst | IntConst:
        import numpy as np

        # The reset comes from the analyzer's one-read attribute snapshot, never a fresh getattr: a live read
        # here could observe state a permitted compile-time evaluation mutated after analysis stabilized.
        snapshot = self._result.state_resets.get(leaf)
        assert snapshot is not None, f"reset for {leaf} missing from the analysis plan"
        if isinstance(snapshot, str):
            raise EmissionRejection(
                f"state '{'.'.join(leaf.path)}' has a reset of unsupported type {snapshot}", self._leaf_origin(leaf)
            )
        normalized = normalize_static(snapshot)
        if isinstance(normalized, AggregateFact):
            cell = normalized.leaves[ordinal]
            assert isinstance(cell, Known), "an aggregate reset cell must be a concrete scalar"
            current = as_python(cell.value)
        else:
            current = as_python(snapshot)
        if isinstance(current, bool) or isinstance(current, np.bool_):
            return BoolConst(bool(current))
        if isinstance(current, (int, np.integer)):
            return IntConst(int(current))
        if isinstance(current, (int, float, np.integer, np.floating)):
            # The same carrier policy as a datapath constant.
            return FloatConst(_carrier_float(current, self._leaf_origin(leaf)))
        raise EmissionRejection(
            f"state '{'.'.join(leaf.path)}' has a reset of unsupported type {type(current).__name__}",
            self._leaf_origin(leaf),
        )

    # ---------------------------------------- values and ops ----------------------------------------

    def _const(self, value: StaticValue) -> int:
        concrete = as_python(value)
        import numpy as np

        if isinstance(concrete, (bool, np.bool_)):
            return self._builder.bool_const(bool(concrete))
        if isinstance(concrete, (int, float, np.integer, np.floating)):
            return self._builder.float_const(_carrier_float(concrete, self._origin))
        raise EmissionRejection(f"a {type(concrete).__name__} value cannot materialize in the datapath", self._origin)

    def _emit_concat(
        self, block: FirBlockId, dst: BindingId, bin_op: BinOp, lhs: BindingId, rhs: BindingId, result: AggregateFact
    ) -> None:
        """Sequence concat (+) and repeat (*): pure leaf routing, no HIR -- the layout work happened in analysis."""
        if bin_op is BinOp.ADD:
            offset = 0
            for operand in (lhs, rhs):
                fact = self._fact(operand)
                assert isinstance(fact, AggregateFact)
                self._install(block, Local(dst), offset, operand)
                offset += len(fact.leaves)
            assert offset == len(result.leaves)
        else:
            assert bin_op is BinOp.MUL
            seq, seq_fact = next(
                (operand, fact) for operand in (lhs, rhs) if isinstance(fact := self._fact(operand), AggregateFact)
            )
            assert isinstance(seq_fact, AggregateFact)
            count = len(seq_fact.leaves)
            for repetition in range(len(result.leaves) // count if count else 0):
                self._install(block, Local(dst), repetition * count, seq)

    def _atom_vid(self, leaf: Known) -> int:
        """A Known leaf in its own kind (an integer leaf stays an IntConst; interning keeps duplicates free)."""
        if isinstance(leaf.value, (MetaInt, NpInt)):
            return self._builder.int_const(int(leaf.value.value))
        return self._const(leaf.value)

    @staticmethod
    def _datapath_known(leaf: Known) -> bool:
        """
        Whether a Known leaf is a datapath scalar. A non-datapath Known (a string, a function reference, a range)
        stays fact-only: its every use folds at analysis, and it can never become a merge operand (a join keeps it
        Known-same or rejects), so defining its cell would only force a spurious materialization rejection.
        """
        return isinstance(leaf.value, (StaticBool, NpBool, MetaInt, NpInt, StaticFloat, NpFloat))

    def _emit_conversion(
        self,
        block: FirBlockId,
        source: BindingId,
        dst: BindingId,
        result: AggregateFact,
        route: "tuple[int, ...] | None" = None,
    ) -> None:
        """
        A conversion's leaf copy: identical to :meth:`_copy_leaves` for the flavor conversions (whose result
        leaves ARE the source facts), plus the kind coercion an array factory introduces -- a residual integer
        leaf under a float dtype reads its source cell and promotes, exactly as the scalar materializer would.
        A ROUTE plan (a transpose) names the source ordinal feeding each result cell; its absence is the
        aligned identity.
        """
        source_fact = self._fact(source)
        assert isinstance(source_fact, AggregateFact) and len(source_fact.leaves) == len(result.leaves)
        for ordinal, leaf in enumerate(result.leaves):
            source_ordinal = route[ordinal] if route is not None else ordinal
            if isinstance(leaf, Known):
                if self._datapath_known(leaf):
                    self._write(block, _LeafPlace(Local(dst), ordinal), self._atom_vid(leaf))
            elif not isinstance(leaf, Reference):
                assert isinstance(leaf, Residual)
                # An unchanged leaf copies as carried (the flavor-conversion identity); only a leaf the factory
                # re-semmed coerces onto its new kind. A boolean source under a float destination is the ONE
                # sanctioned bool crossing -- the user's explicit dtype=float IS the conversion -- scoped here
                # so the scalar materializer keeps rejecting implicit bool arithmetic everywhere else.
                source_leaf = source_fact.leaves[source_ordinal]
                if leaf.type is SemType.FLOAT and source_leaf == Residual(SemType.BOOL):
                    flag = self._materialize_atom(
                        source_leaf,
                        lambda: self._read(block, _LeafPlace(Local(source), source_ordinal)),
                        SemType.BOOL,
                    )
                    vid = self._builder.operation(BoolToFloat(), [flag])
                else:
                    expected = None if source_leaf == leaf else leaf.type
                    vid = self._materialize_atom(
                        source_leaf,
                        lambda: self._read(block, _LeafPlace(Local(source), source_ordinal)),
                        expected,
                    )
                self._write(block, _LeafPlace(Local(dst), ordinal), vid)

    def _copy_leaves(self, block: FirBlockId, source: Place, fact: AggregateFact, target: Place) -> None:
        """
        Define every datapath leaf of ``target``: a Known leaf as its constant, a residual leaf as the source's
        aligned SSA cell. All-datapath-leaf definition is the invariant a later per-leaf CFG merge relies on -- a
        phi arm must resolve on every predecessor, including one whose leaf happened to be a constant on that path.
        """
        for ordinal, leaf in enumerate(fact.leaves):
            if isinstance(leaf, Known):
                if self._datapath_known(leaf):
                    self._write(block, _LeafPlace(target, ordinal), self._atom_vid(leaf))
            elif not isinstance(leaf, Reference):  # a reference leaf stays fact-only, like a non-datapath Known
                self._write(block, _LeafPlace(target, ordinal), self._read(block, _LeafPlace(source, ordinal)))

    def _project(self, block: FirBlockId, source: Place, start: int, dst: BindingId) -> None:
        """Define dst's cells from the source's cell window at [start, ...): a subscript or field projection."""
        dst_fact = self._fact(dst)
        if isinstance(dst_fact, AggregateFact):
            for ordinal, leaf in enumerate(dst_fact.leaves):
                if isinstance(leaf, Known):
                    if self._datapath_known(leaf):
                        self._write(block, _LeafPlace(Local(dst), ordinal), self._atom_vid(leaf))
                elif not isinstance(leaf, Reference):
                    vid = self._read(block, _LeafPlace(source, start + ordinal))
                    self._write(block, _LeafPlace(Local(dst), ordinal), vid)
        elif not isinstance(dst_fact, (Known, Reference)):
            self._write(block, Local(dst), self._read(block, _LeafPlace(source, start)))

    def _install(self, block: FirBlockId, target: Place, start: int, item: BindingId) -> None:
        """Write item's value cells into target's cells at [start, ...): the insertion mirror of ``_project``."""
        item_fact = self._fact(item)
        if isinstance(item_fact, AggregateFact):
            for ordinal, leaf in enumerate(item_fact.leaves):
                if isinstance(leaf, Known):
                    if self._datapath_known(leaf):
                        self._write(block, _LeafPlace(target, start + ordinal), self._atom_vid(leaf))
                elif not isinstance(leaf, Reference):
                    vid = self._read(block, _LeafPlace(Local(item), ordinal))
                    self._write(block, _LeafPlace(target, start + ordinal), vid)
        elif isinstance(item_fact, Known):
            if self._datapath_known(item_fact):
                self._write(block, _LeafPlace(target, start), self._atom_vid(item_fact))
        elif not isinstance(item_fact, Reference):
            self._write(block, _LeafPlace(target, start), self._read(block, Local(item)))

    def _fact(self, binding: BindingId) -> Fact:
        fact = self._result.binding_facts.get(binding)
        assert fact is not None, f"binding {binding} missing from the analysis plan"
        return fact

    def _materialize(self, block: FirBlockId, binding: BindingId, expected: SemType | None = None) -> int:
        """
        The one typed materializer. A Known value becomes a constant of the expected kind: a Known integer stays an
        IntConst in an integer context and rounds into a float constant in a float context (the carrier policy). A
        residual value is its SSA read, coerced onto the expected kind where the coercion is a genuine promotion
        (int -> float) and a located rejection otherwise. ``expected=None`` materializes the value as carried: a
        Known in its own kind, a residual as already emitted -- the carried kind can differ from the fact's (a
        runtime integer promoted at a state boundary is float-carried while its fact still reads integer).
        """
        fact = self._fact(binding)
        assert not isinstance(fact, AggregateFact), "an aggregate value reaches a scalar operand position"
        return self._materialize_atom(fact, lambda: self._read(block, Local(binding)), expected)

    def _materialize_atom(self, fact: Fact, read: Callable[[], int], expected: SemType | None) -> int:
        """The scalar materializer over one atom: a binding's own cell or a single leaf cell of an aggregate."""
        if isinstance(fact, Known):
            if isinstance(fact.value, (MetaInt, NpInt)) and expected in (SemType.INT, None):
                return self._builder.int_const(int(fact.value.value))
            vid = self._const(fact.value)
        else:
            vid = read()
        if expected is None:
            return vid
        actual = self._sem_of(self._type_of(vid))
        if actual is expected:
            return vid
        if actual is SemType.INT and expected is SemType.FLOAT:
            return self._builder.operation(IntToFloat(), [vid])  # Python promotes int -> float
        if expected is SemType.FLOAT:
            raise EmissionRejection("a boolean value reaches a float operation", self._origin)
        if expected is SemType.INT:
            raise EmissionRejection("a non-integer value reaches an integer operation", self._origin)
        raise EmissionRejection("a non-boolean value reaches a boolean operation", self._origin)

    def _emit_block(self, fir_id: FirBlockId) -> None:
        self._builder.position_at(self._fir_to_hir[fir_id])
        block = self._result.unit.blocks[fir_id]
        for op in block.ops:
            self._emit_op(fir_id, op)
        match block.terminator:
            case FirJump(target=target):
                if target in self._result.executable_blocks:
                    self._builder.jump(self._fir_to_hir[target])
            case FirBranch(cond=cond, then_target=then_target, else_target=else_target, origin=origin):
                live = [t for t in (then_target, else_target) if (fir_id, t) in self._result.executable_edges]
                if len(live) == 1:
                    self._builder.jump(self._fir_to_hir[live[0]])
                else:
                    self._origin = origin
                    condition = self._materialize(fir_id, cond, SemType.BOOL)
                    self._builder.branch(condition, self._fir_to_hir[then_target], self._fir_to_hir[else_target])
            case UnitExit():
                pass  # _finish_exit seals the exit block with outputs, slots, and the single Ret
            case other:
                raise AssertionError(f"terminator {type(other).__name__} survived analysis into emission")

    def _emit_op(self, fir_id: FirBlockId, op: Op) -> None:
        self._origin = op.origin

        def define(dst: BindingId, vid: int) -> None:
            self._write(fir_id, Local(dst), vid)

        match op:
            case LoadConst() | LoadRef() | UnbindPlace():
                pass  # facts and boundness are the analyzer's; nothing materializes here
            case LoadPlace(dst=dst, place=place):
                fact = self._fact(dst)
                if isinstance(fact, AggregateFact):
                    self._copy_leaves(fir_id, place, fact, Local(dst))
                elif not isinstance(fact, (Known, Reference)):
                    define(dst, self._read(fir_id, place))
            case StorePlace(place=place, src=src):
                fact = self._fact(src)
                if isinstance(fact, AggregateFact):
                    self._copy_leaves(fir_id, Local(src), fact, place)
                elif isinstance(fact, Reference) or (isinstance(fact, Known) and not self._datapath_known(fact)):
                    pass  # a reference or non-datapath Known (a string, a range): every use folds
                else:
                    # The analyzer's resolution walk marked the stores whose value converts int->float on the
                    # store edge (a runtime int into a float-schema local); the cell must carry the converted
                    # kind the flowed facts promise, exactly as the explicit float(...) spelling would.
                    expected = SemType.FLOAT if id(op) in self._result.store_conversions else None
                    self._write(fir_id, place, self._materialize(fir_id, src, expected))
            case PyBin(dst=dst, op=bin_op, lhs=lhs, rhs=rhs):
                self._emit_binary(fir_id, dst, bin_op, lhs, rhs)
            case PyUn(dst=dst, op=un_op, operand=operand):
                self._emit_unary(fir_id, dst, un_op, operand)
            case PyCompare(dst=dst, op=rel, lhs=lhs, rhs=rhs):
                compare_fact = self._fact(dst)
                assert not isinstance(compare_fact, AggregateFact), "elementwise comparisons reject at analysis"
                if not isinstance(compare_fact, Known):
                    lsem, rsem = self._fact_sem(self._fact(lhs)), self._fact_sem(self._fact(rhs))
                    if lsem is SemType.BOOL or rsem is SemType.BOOL:
                        if lsem is not rsem:
                            raise EmissionRejection(
                                "a comparison mixes a boolean and a non-boolean without a cast", self._origin
                            )
                        left = self._materialize(fir_id, lhs, SemType.BOOL)
                        right = self._materialize(fir_id, rhs, SemType.BOOL)
                        define(dst, self._bool_compare(rel, left, right))  # bool ==/!= is XNOR/XOR
                    elif lsem is SemType.INT and rsem is SemType.INT:
                        left = self._materialize(fir_id, lhs, SemType.INT)
                        right = self._materialize(fir_id, rhs, SemType.INT)
                        define(dst, self._builder.operation(IntRelational(rel), [left, right]))
                    else:  # a float on at least one side: the integer side promotes C-style and compares in float
                        left = self._materialize(fir_id, lhs, SemType.FLOAT)
                        right = self._materialize(fir_id, rhs, SemType.FLOAT)
                        define(dst, self._builder.operation(FloatRelational(rel), [left, right]))
            case PyNot(dst=dst, operand=operand):
                if not isinstance(self._fact(dst), Known):
                    define(dst, self._not(self._truth_value(fir_id, operand)))
            case PyTruth(dst=dst, operand=operand):
                if not isinstance(self._fact(dst), Known):
                    define(dst, self._truth_value(fir_id, operand))
            case PySelect(dst=dst, mode=mode, cond=cond, lhs=lhs, rhs=rhs):
                result_fact = self._fact(dst)
                if isinstance(result_fact, (Known, Reference)):
                    pass  # the analyzer selected a Known/reference arm or folded a boolean identity (``A or True``)
                elif isinstance(cond_fact := self._fact(cond), Known):
                    taken = as_python(cond_fact.value)
                    assert isinstance(taken, bool)
                    chosen = (rhs if taken else lhs) if mode is SelectMode.AND else (lhs if taken else rhs)
                    if isinstance(result_fact, AggregateFact):
                        self._copy_leaves(fir_id, Local(chosen), result_fact, Local(dst))
                    else:
                        define(dst, self._materialize(fir_id, chosen))
                else:
                    # The merged fact fixes the selection kind; each arm materializes onto it, so a mixed int/float
                    # select promotes the integer arm on its own edge exactly like a phi. An aggregate never reaches
                    # a residual condition: the condition is the lhs's own truth, which folds (or rejects) for every
                    # aggregate, and a scalar lhs cannot join with an aggregate rhs.
                    assert not isinstance(result_fact, AggregateFact)
                    then_binding, else_binding = (rhs, lhs) if mode is SelectMode.AND else (lhs, rhs)
                    condition = self._materialize(fir_id, cond, SemType.BOOL)
                    sem = self._fact_sem(result_fact)
                    operator: Operator = (
                        BoolSelect() if sem is SemType.BOOL else IntSelect() if sem is SemType.INT else Select()
                    )
                    then_vid = self._materialize(fir_id, then_binding, sem)
                    else_vid = self._materialize(fir_id, else_binding, sem)
                    define(dst, self._builder.operation(operator, [condition, then_vid, else_vid]))
            case BuildTuple(dst=dst, items=items) | BuildList(dst=dst, items=items):
                fact = self._fact(dst)
                if isinstance(fact, AggregateFact):
                    for index, item in enumerate(items):
                        _, start, _ = child_slice(fact.layout, index)
                        self._install(fir_id, Local(dst), start, item)
            case PySubscript(dst=dst, obj=obj, index=index):
                obj_fact = self._fact(obj)
                dst_fact = self._fact(dst)
                needs_cells = isinstance(dst_fact, Residual) or (
                    isinstance(dst_fact, AggregateFact) and any(isinstance(leaf, Residual) for leaf in dst_fact.leaves)
                )
                if isinstance(obj_fact, AggregateFact) and needs_cells:
                    selection = self._result.subscript_plans.get(dst)
                    if selection is not None:
                        # A slice window or an array gather: the analyzer's plan names the source leaf ordinal
                        # feeding each result cell; Known result leaves materialize as their own constants.
                        if isinstance(dst_fact, AggregateFact):
                            assert len(selection) == len(dst_fact.leaves), "a selection misaligns its result"
                            for ordinal, window_leaf in enumerate(dst_fact.leaves):
                                if isinstance(window_leaf, Known):
                                    if self._datapath_known(window_leaf):
                                        self._write(
                                            fir_id,
                                            _LeafPlace(Local(dst), ordinal),
                                            self._atom_vid(window_leaf),
                                        )
                                elif not isinstance(window_leaf, Reference):
                                    vid = self._read(fir_id, _LeafPlace(Local(obj), selection[ordinal]))
                                    self._write(fir_id, _LeafPlace(Local(dst), ordinal), vid)
                        else:
                            assert len(selection) == 1
                            define(dst, self._read(fir_id, _LeafPlace(Local(obj), selection[0])))
                    else:
                        # A residual dst leaf without a selection plan means the analyzer projected a positional
                        # child, so the key resolves under operator.index -- either a Known directly or an
                        # all-Known aggregate (an __index__-able record), materialized exactly as the analyzer
                        # materialized it.
                        import operator as operator_module

                        index_fact = self._fact(index)
                        assert isinstance(index_fact, (Known, AggregateFact)), index_fact
                        key = index_fact.value if isinstance(index_fact, Known) else materialize_static(index_fact)
                        assert key is not None, index_fact
                        position = operator_module.index(as_python(key))  # type: ignore[arg-type]
                        width = outer_arity(obj_fact.layout)
                        position = position + width if position < 0 else position
                        _, start, _ = child_slice(obj_fact.layout, position)
                        self._project(fir_id, Local(obj), start, dst)
                # else the analyzer folded the subscript concretely: the Known facts materialize at their use sites
            case PyLen():
                pass  # always folded by the analyzer (static shape)
            case PyAttr(dst=dst, obj=obj, name=name):
                dst_fact = self._fact(dst)
                if not isinstance(dst_fact, (Known, Reference)):
                    obj_fact = self._fact(obj)
                    needs_cells = isinstance(dst_fact, Residual) or (
                        isinstance(dst_fact, AggregateFact)
                        and any(isinstance(leaf, Residual) for leaf in dst_fact.leaves)
                    )
                    if isinstance(obj_fact, AggregateFact) and not needs_cells:
                        pass  # concrete navigation of an all-Known aggregate (.T, .shape, ...): every use folds
                    elif isinstance(obj_fact, AggregateFact):
                        # A record with residual leaves projects the named field's cell window, exactly as a
                        # positional subscript projects a child's (the only residual-producing attribute source).
                        layout = obj_fact.layout
                        assert isinstance(layout, RecordLayout), layout
                        names = [field for field, _ in layout.fields]
                        _, start, _ = child_slice(layout, names.index(name))
                        self._project(fir_id, Local(obj), start, dst)
                    elif isinstance(dst_fact, AggregateFact):
                        # An aggregate component attribute: a Known leaf is its frozen snapshot constant, and a
                        # residual leaf is runtime aggregate state backed by its own per-cell slot.
                        assert isinstance(obj_fact, Reference)
                        state_root = StateLeaf(obj_fact.obj, (name,))
                        for ordinal, leaf_fact in enumerate(dst_fact.leaves):
                            if isinstance(leaf_fact, Known):
                                if self._datapath_known(leaf_fact):
                                    self._write(fir_id, _LeafPlace(Local(dst), ordinal), self._atom_vid(leaf_fact))
                            elif not isinstance(leaf_fact, Reference):
                                vid = self._read(fir_id, _LeafPlace(state_root, ordinal))
                                self._write(fir_id, _LeafPlace(Local(dst), ordinal), vid)
                    else:
                        # Otherwise only a runtime component attribute is residual; it is backed by a state slot.
                        assert isinstance(obj_fact, Reference)
                        define(dst, self._read(fir_id, StateLeaf(obj_fact.obj, (name,))))
            case PyStoreAttr(obj=obj, name=name, src=src):
                obj_fact = self._fact(obj)
                assert isinstance(obj_fact, Reference)
                leaf = StateLeaf(obj_fact.obj, (name,))
                src_fact = self._fact(src)
                if isinstance(src_fact, AggregateFact):
                    for ordinal, stored in enumerate(src_fact.leaves):
                        vid = self._materialize_atom(
                            stored,
                            lambda: self._read(fir_id, _LeafPlace(Local(src), ordinal)),
                            self._slot_kind(leaf, ordinal),
                        )
                        self._write(fir_id, _LeafPlace(leaf, ordinal), vid)
                        self._state_read(leaf, ordinal)  # register every cell slot even if never read
                else:
                    self._write(fir_id, leaf, self._materialize(fir_id, src, self._slot_kind(leaf)))
                    self._state_read(leaf)  # register the slot even if the entry live-in was never read
            case PyCall(dst=dst, args=args):
                plan = self._result.call_plans[dst]
                match plan.lowering:
                    case CallLowering.FOLDED:
                        pass  # a concrete fold; the Known value materializes at its use sites
                    case CallLowering.CAST:
                        define(dst, self._emit_cast(fir_id, args[0], dst))
                    case CallLowering.INTRINSIC:
                        define(dst, self._emit_intrinsic(fir_id, plan, list(args)))
                    case CallLowering.CONVERSION:
                        conversion_fact = self._fact(dst)
                        if isinstance(conversion_fact, AggregateFact):
                            route = self._result.route_plans.get(dst)
                            self._emit_conversion(fir_id, args[0], dst, conversion_fact, route)
                    case CallLowering.CONSTRUCTION:
                        record_fact = self._fact(dst)
                        assert isinstance(record_fact, AggregateFact) and plan.construction is not None
                        # A fully static construction emits nothing, exactly like the folded-call era: every use
                        # folds from the facts, and eager constants would only shift HIR node ordering.
                        if any(isinstance(leaf, Residual) for leaf in record_fact.leaves):
                            for index, source in enumerate(plan.construction):
                                _, start, stop = child_slice(record_fact.layout, index)
                                if source is not None:
                                    source_fact = self._fact(source)
                                    width = len(source_fact.leaves) if isinstance(source_fact, AggregateFact) else 1
                                    assert width == stop - start, "a construction source misaligns its field window"
                                    self._install(fir_id, Local(dst), start, source)
                                else:  # a default-filled field: its leaves are the admitted schema constants
                                    for ordinal in range(start, stop):
                                        filled = record_fact.leaves[ordinal]
                                        assert isinstance(filled, (Known, Reference)), "a default grew runtime cells"
                                        if isinstance(filled, Known) and self._datapath_known(filled):
                                            self._write(fir_id, _LeafPlace(Local(dst), ordinal), self._atom_vid(filled))
            case _:
                raise AssertionError(f"operation {type(op).__name__} survived analysis into emission")

    def _emit_intrinsic(self, block: FirBlockId, plan: CallPlan, args: list[BindingId]) -> int:
        from .._lib import Intrinsic, IntrinsicResultRule

        match_ = plan.intrinsic
        assert isinstance(match_, Intrinsic)
        rule = match_.result_rule
        all_int = all(self._fact_sem(self._fact(arg)) is SemType.INT for arg in args)
        if rule is IntrinsicResultRule.ALWAYS_INT and all_int:
            return self._materialize(block, args[0], SemType.INT)  # rounding an integer is the integer (identity)
        if rule is IntrinsicResultRule.ALWAYS_INT:
            operands = [self._materialize(block, arg, SemType.FLOAT) for arg in args]
            return self._builder.operation(FloatToInt(), [self._builder.operation(match_.operator, operands)])
        if rule is IntrinsicResultRule.INT_OVERLOAD and all_int:
            operands = [self._materialize(block, arg, SemType.INT) for arg in args]
            return self._integer_intrinsic(match_.integer_implementation, operands)
        # SIGNATURE, or a float operand present: promote the integer operands and run the float operator.
        operands = [self._materialize(block, arg, SemType.FLOAT) for arg in args]
        return self._builder.operation(match_.operator, operands)

    def _integer_intrinsic(self, implementation: object, vids: list[int]) -> int:
        """
        The integer-typed HIR for an integer-operand intrinsic (contained at MIR; no integer hardware lowers). min/max
        use an inclusive comparison so the first operand wins a tie, matching Python's ``min``/``max``.
        """
        from .._lib import IntegerImplementation

        match implementation:
            case IntegerImplementation.IDENTITY:
                return vids[0]
            case IntegerImplementation.ABS:
                return self._builder.operation(IntAbs(), [vids[0]])
            case IntegerImplementation.MIN:
                condition = self._builder.operation(IntRelational(RelationalOp.LE), [vids[0], vids[1]])
                return self._builder.operation(IntSelect(), [condition, vids[0], vids[1]])
            case IntegerImplementation.MAX:
                condition = self._builder.operation(IntRelational(RelationalOp.GE), [vids[0], vids[1]])
                return self._builder.operation(IntSelect(), [condition, vids[0], vids[1]])
        raise AssertionError(f"no integer implementation for {implementation!r}")

    def _leaf_fact(self, leaf: StateLeaf) -> Fact:
        livein = self._result.state_livein
        if leaf in livein:
            return livein[leaf]
        reset = self._leaf_reset(leaf)
        if isinstance(reset, BoolConst):
            return Known(StaticBool(reset.value))
        admitted = admit(reset.value)
        assert admitted is not None
        return Known(admitted)

    def _emit_binary(self, block: FirBlockId, dst: BindingId, bin_op: BinOp, lhs: BindingId, rhs: BindingId) -> None:
        result_fact = self._fact(dst)
        if isinstance(result_fact, Known):
            return  # the analyzer folded it (a static fold, a sequence concat/repeat, a bool &/|/^ identity)
        if isinstance(result_fact, AggregateFact):
            if isinstance(result_fact.layout, ArrayLayout):
                self._emit_elementwise(block, dst, bin_op, lhs, rhs, result_fact)
            else:
                self._emit_concat(block, dst, bin_op, lhs, rhs, result_fact)
            return

        def define(vid: int) -> None:
            self._write(block, Local(dst), vid)

        if bin_op is BinOp.POW:  # dispatched on the exponent, in the base's own kind (an integer chain stays integer)
            define(self._emit_power(block, lhs, rhs))
            return
        if result_fact == Residual(SemType.INT):  # a runtime integer result: stay in the integer datapath
            left, right = self._materialize(block, lhs, SemType.INT), self._materialize(block, rhs, SemType.INT)
            define(self._emit_int_binary(bin_op, left, right))
            return
        if result_fact == Residual(SemType.BOOL):  # &/|/^ on two booleans lowers to the boolean-bank operators
            left, right = self._materialize(block, lhs, SemType.BOOL), self._materialize(block, rhs, SemType.BOOL)
            match bin_op:
                case BinOp.BITAND:
                    define(self._builder.operation(BoolAnd(), [left, right]))
                case BinOp.BITOR:
                    define(self._builder.operation(BoolOr(), [left, right]))
                case _:
                    define(self._builder.operation(BoolXor(), [left, right]))
            return
        left, right = self._materialize(block, lhs, SemType.FLOAT), self._materialize(block, rhs, SemType.FLOAT)
        define(self._emit_float_binary(bin_op, left, right))

    def _emit_elementwise(
        self, block: FirBlockId, dst: BindingId, bin_op: BinOp, lhs: BindingId, rhs: BindingId, result: AggregateFact
    ) -> None:
        """
        Elementwise arithmetic over an array: one scalar operation per residual result leaf, each array-side
        operand read from its aligned leaf cell and a scalar side materialized once and broadcast. Known result
        leaves (folded pairs) emit nothing, exactly like a folded scalar; Known operand leaves under a residual
        result materialize as constants of the result kind. A residual integer result never reaches here (the
        analyzer rejects it: the scalar integer datapath saturates where numpy wraps).
        """
        assert isinstance(result.layout, ArrayLayout) and result.layout.dtype is not ArrayDType.BOOL
        lhs_fact, rhs_fact = self._fact(lhs), self._fact(rhs)
        broadcast: dict[BindingId, int] = {}

        def operand(binding: BindingId, fact: Fact, ordinal: int) -> int:
            if isinstance(fact, AggregateFact):
                return self._materialize_atom(
                    fact.leaves[ordinal],
                    lambda: self._read(block, _LeafPlace(Local(binding), ordinal)),
                    SemType.FLOAT,
                )
            # A scalar side: one materialization broadcast across every leaf.
            if binding not in broadcast:
                broadcast[binding] = self._materialize(block, binding, SemType.FLOAT)
            return broadcast[binding]

        for ordinal, leaf in enumerate(result.leaves):
            if not isinstance(leaf, Residual):
                continue
            assert result.layout.dtype is ArrayDType.FLOAT, "a runtime integer array op escaped analysis"
            left = operand(lhs, lhs_fact, ordinal)
            right = operand(rhs, rhs_fact, ordinal)
            self._write(block, _LeafPlace(Local(dst), ordinal), self._emit_float_binary(bin_op, left, right))

    def _emit_float_binary(self, bin_op: BinOp, left: int, right: int) -> int:
        match bin_op:
            case BinOp.ADD:
                return self._builder.operation(FloatAdd(), [left, right])
            case BinOp.SUB:
                return self._builder.operation(FloatAdd(), [left, self._builder.operation(FloatNeg(), [right])])
            case BinOp.MUL:
                return self._builder.operation(FloatMul(), [left, right])
            case BinOp.DIV:
                return self._builder.operation(FloatDiv(), [left, right])
            case _:
                raise EmissionRejection(f"operator {bin_op.value} is not lowerable yet", self._origin)

    def _emit_cast(self, block: FirBlockId, arg: BindingId, dst: BindingId) -> int:
        # A runtime scalar float()/int()/bool() cast, kinded by the FINAL facts (the analyzer's optimistic revisits
        # may have seen either kind mid-flight). A same-kind cast is the identity; int<->float is truncation toward
        # zero / promotion; bool casts are a truthiness test (to bool) or a 0/1 widening (from bool).
        src, target = self._fact_sem(self._fact(arg)), self._fact_sem(self._fact(dst))
        if src is target:
            return self._materialize(block, arg, target)
        match (src, target):
            case (SemType.INT, SemType.FLOAT):
                return self._materialize(block, arg, SemType.FLOAT)
            case (SemType.FLOAT, SemType.INT):
                return self._builder.operation(FloatToInt(), [self._materialize(block, arg, SemType.FLOAT)])
            case (SemType.BOOL, SemType.FLOAT):
                return self._builder.operation(BoolToFloat(), [self._materialize(block, arg, SemType.BOOL)])
            case (SemType.FLOAT, SemType.BOOL):
                return self._builder.operation(FloatToBool(), [self._materialize(block, arg, SemType.FLOAT)])
            case (SemType.BOOL, SemType.INT):
                return self._builder.operation(BoolToInt(), [self._materialize(block, arg, SemType.BOOL)])
            case (SemType.INT, SemType.BOOL):
                return self._builder.operation(IntToBool(), [self._materialize(block, arg, SemType.INT)])
            case _:
                raise EmissionRejection(f"conversion from {src.value} to {target.value} is not supported", self._origin)

    def _emit_int_binary(self, bin_op: BinOp, left: int, right: int) -> int:
        # Signed-integer arithmetic in the integer datapath. Floor-division and modulo are floor-coupled (one hardware
        # divmod at the wiring milestone); ``/`` never reaches here (Python true division promotes to float upstream).
        match bin_op:
            case BinOp.ADD:
                return self._builder.operation(IntAdd(), [left, right])
            case BinOp.SUB:
                return self._builder.operation(IntSub(), [left, right])
            case BinOp.MUL:
                return self._builder.operation(IntMul(), [left, right])
            case BinOp.FLOORDIV:
                return self._builder.operation(IntDivFloor(), [left, right])
            case BinOp.MOD:
                return self._builder.operation(IntMod(), [left, right])
            case BinOp.LSHIFT:
                return self._builder.operation(IntShiftLeft(), [left, right])
            case BinOp.RSHIFT:
                return self._builder.operation(IntShiftRight(), [left, right])
            case BinOp.BITAND:
                return self._builder.operation(IntAnd(), [left, right])
            case BinOp.BITOR:
                return self._builder.operation(IntOr(), [left, right])
            case BinOp.BITXOR:
                return self._builder.operation(IntXor(), [left, right])
            case _:
                raise EmissionRejection(f"integer operator {bin_op.value!r} is not lowerable yet", self._origin)

    def _emit_power(self, block: FirBlockId, base: BindingId, exponent: BindingId) -> int:
        # A base raised to a COMPILE-TIME integer exponent expands to a multiply chain (x**3 -> x*x*x) in the base's
        # own kind -- an integer base with an integer exponent stays exact integer (contained at MIR), a float base
        # multiplies in float -- both bounded like the loop unroller so ``x ** 10**9`` refuses instead of hanging. A
        # negative exponent is the reciprocal of the float chain (Python: a negative power is float even on an
        # integer base), and a Known INTEGRAL float exponent (x ** 3.0) chains identically under fastmath rather
        # than paying the exp2/log2 pair and its domain-error cone.
        exp_fact = self._fact(exponent)
        exponent_is_int = isinstance(exp_fact, Known) and isinstance(exp_fact.value, (MetaInt, NpInt))
        power: int | None = None
        if exponent_is_int:
            assert isinstance(exp_fact, Known) and isinstance(exp_fact.value, (MetaInt, NpInt))
            power = int(exp_fact.value.value)
            if abs(power) > _MAX_POWER_CHAIN:
                raise EmissionRejection(
                    f"a compile-time power exponent of {power} is too large to expand", self._origin
                )
        elif isinstance(exp_fact, Known) and isinstance(exp_fact.value, (StaticFloat, NpFloat)):
            exact = float(exp_fact.value.value)
            if exact.is_integer():  # only a FRACTIONAL float exponent falls to the exp2/log2 path below
                if abs(exact) > _MAX_POWER_CHAIN:
                    raise EmissionRejection(
                        f"a compile-time power exponent of {exact:.0f} is too large to expand", self._origin
                    )
                power = int(exact)
        if power is not None:
            if exponent_is_int and self._fact_sem(self._fact(base)) is SemType.INT and power >= 0:
                if power == 0:
                    return self._builder.int_const(1)  # x**0 is the INTEGER 1 on an integer base, as in Python
                source = self._materialize(block, base, SemType.INT)
                chain = source
                for _ in range(power - 1):
                    chain = self._builder.operation(IntMul(), [chain, source])
                return chain
            if power == 0:
                return self._builder.float_const(1.0)
            source = self._materialize(block, base, SemType.FLOAT)
            chain = source
            for _ in range(abs(power) - 1):
                chain = self._builder.operation(FloatMul(), [chain, source])
            if power < 0:
                return self._builder.operation(FloatDiv(), [self._builder.float_const(1.0), chain])
            return chain
        # A runtime exponent computes in float as the direct fastmath identity exp2(e * log2(b)); base two skips the
        # log2 (the common 2**e spelling costs one exp2). An integer base or exponent promotes C-style. A Known
        # nonpositive base would assert the log2 pole/domain error on every transaction where Python computes a
        # plain value, so it is a located rejection; a RUNTIME zero or negative base keeps the documented C-style
        # log2 domain-error behavior. No identity-guard diamonds: strength reduction owns the provable identities,
        # and the ZKF zero/infinity algebra (0*inf = 0) covers the IEEE corner b == 1, e = inf.
        exponent_vid = self._materialize(block, exponent, SemType.FLOAT)
        base_fact = self._fact(base)
        if isinstance(base_fact, Known) and isinstance(base_fact.value, (MetaInt, NpInt, StaticFloat, NpFloat)):
            base_value = base_fact.value.value
            if base_value == 2:
                return self._builder.operation(FloatExp2(), [exponent_vid])
            if not base_value > 0:
                raise EmissionRejection(
                    "a power with a runtime exponent requires a positive base (log2 domain)", self._origin
                )
        log_base = self._builder.operation(FloatLog2(), [self._materialize(block, base, SemType.FLOAT)])
        return self._builder.operation(FloatExp2(), [self._builder.operation(FloatMul(), [exponent_vid, log_base])])

    def _emit_unary(self, block: FirBlockId, dst: BindingId, un_op: UnOp, operand: BindingId) -> None:
        result_fact = self._fact(dst)
        if isinstance(result_fact, Known):
            return  # folded by the analyzer; materializes at its use sites
        if isinstance(result_fact, AggregateFact):
            assert isinstance(result_fact.layout, ArrayLayout), "only an array admits elementwise unary arithmetic"
            operand_fact = self._fact(operand)
            assert isinstance(operand_fact, AggregateFact)
            for ordinal, leaf in enumerate(result_fact.leaves):
                if not isinstance(leaf, Residual):
                    continue
                assert result_fact.layout.dtype is ArrayDType.FLOAT, "a runtime integer array op escaped analysis"
                source = self._materialize_atom(
                    operand_fact.leaves[ordinal],
                    lambda: self._read(block, _LeafPlace(Local(operand), ordinal)),
                    SemType.FLOAT,
                )
                if un_op is not UnOp.POS:
                    source = self._builder.operation(FloatNeg(), [source])
                self._write(block, _LeafPlace(Local(dst), ordinal), source)
            return
        if result_fact == Residual(SemType.INT):  # runtime integer negation stays in the integer datapath
            source = self._materialize(block, operand, SemType.INT)
            result = source if un_op is UnOp.POS else self._builder.operation(IntNeg(), [source])
        else:
            source = self._materialize(block, operand, SemType.FLOAT)
            result = source if un_op is UnOp.POS else self._builder.operation(FloatNeg(), [source])
        self._write(block, Local(dst), result)

    def _truth_value(self, block: FirBlockId, operand: BindingId) -> int:
        vid = self._materialize(block, operand)
        vtype = self._type_of(vid)
        if isinstance(vtype, BoolType):
            return vid
        if isinstance(vtype, IntType):
            return self._builder.operation(IntToBool(), [vid])  # int truthiness (rejects cleanly at MIR, never floats)
        return self._builder.operation(FloatToBool(), [vid])

    def _not(self, condition: int) -> int:
        return self._builder.operation(BoolNot(), [condition])

    def _bool_compare(self, rel: RelationalOp, left: int, right: int) -> int:
        if rel not in (RelationalOp.EQ, RelationalOp.NE):
            raise EmissionRejection("only == and != are defined between boolean values", self._origin)
        xor = self._builder.operation(BoolXor(), [left, right])
        return xor if rel is RelationalOp.NE else self._builder.operation(BoolNot(), [xor])

    def _exit_identity(self, vid: int, depth: int = 64) -> object:
        """
        The dedup identity of an exit value: a phi is its (type, arms) structure and a pure operation is its
        operator over its operands' identities, not the value id. The return leaf and a state live-out read
        through DIFFERENT places, so a value that is by dataflow the same merge arrives as two distinct-but-
        identical exit phis -- possibly under coercion wrappers minted separately on each side (a store-side and a
        return-side IntToFloat live in different blocks, which the per-block interner keeps distinct), or as
        nested merges whose inner phis differ only by id. Every truncation of the walk degrades to the value id
        itself, so equal identities always imply equal values; a truncation only forgoes a dedup. The memo shares
        the identity of a shared operand (a repeated-squaring DAG stays linear instead of exponential) and breaks
        phi cycles (a loop header's self-arm resolves to the in-progress marker); the depth cap bounds the
        recursion on long chains (x**1024 is a 1024-deep multiply chain, past the interpreter limit).
        """
        memo = self._exit_identity_memo
        if vid in memo:
            return memo[vid]
        if depth == 0:
            return vid  # not memoized: the same node may be reachable shallower on another path
        node = self._builder.node_of(vid)
        identity: object
        if isinstance(node, (Phi, Operation)):
            memo[vid] = vid  # the in-progress marker: a cyclic back-reference resolves to the plain vid
            if isinstance(node, Phi):
                identity = (node.type, tuple((pred, self._exit_identity(arm, depth - 1)) for pred, arm in node.arms))
            else:
                identity = (node.operator, tuple(self._exit_identity(op, depth - 1) for op in node.operands))
        else:
            identity = vid
        memo[vid] = identity
        return identity

    def _exit_leaf(self, exit_block: FirBlockId, path: LeafPath, ordinal: int, leaf: AtomicFact, kind: SemType) -> int:
        """One returned leaf, coerced onto its declared contract kind exactly like a scalar return."""
        if isinstance(leaf, Known):
            if not self._datapath_known(leaf):
                got_name = type(as_python(leaf.value)).__name__
                raise EmissionRejection(
                    f"return type mismatch at leaf {_port_path(path)}: declared {kind.value}, returns a {got_name}",
                    self._origin,
                )
            if isinstance(leaf.value, (MetaInt, NpInt)) and kind is SemType.INT:
                vid = self._builder.int_const(int(leaf.value.value))
            else:
                vid = self._const(leaf.value)
        elif isinstance(leaf, Reference):
            raise EmissionRejection(
                f"return type mismatch at leaf {_port_path(path)}: declared {kind.value}, returns an object",
                self._origin,
            )
        else:
            vid = self._read(exit_block, _LeafPlace(ReturnPlace(), ordinal))
        if kind is SemType.FLOAT and isinstance(self._type_of(vid), IntType):
            vid = self._builder.operation(IntToFloat(), [vid])  # an integer leaf returned as a declared float
        got = self._sem_of(self._type_of(vid))
        if got is not kind:
            raise EmissionRejection(
                f"return type mismatch at leaf {_port_path(path)}: declared {kind.value}, returns {got.value}",
                self._origin,
            )
        return vid

    # ---------------------------------------- exit ----------------------------------------

    def _return_origin(self) -> OriginStack:
        """
        The primary attribution of the exit's contract checks: the earliest return store in source order (the
        implicit fall-off ``return None`` included). The unit-level origin only when no path stores a return.
        """
        stores = [
            op.origin
            for block_id in sorted(self._result.executable_blocks, key=lambda block_id: block_id.index)
            for op in self._result.unit.blocks[block_id].ops
            if isinstance(op, StorePlace) and isinstance(op.place, ReturnPlace)
        ]
        if not stores:
            return self._origin
        return min(stores, key=source_position)

    def _finish_exit(self) -> None:
        unit = self._result.unit
        self._origin = self._return_origin()
        exit_env = self._result.block_in[unit.exit]
        self._builder.position_at(self._fir_to_hir[unit.exit])
        return_fact = exit_env.facts.get(ReturnPlace())
        contract = unit.return_contract
        assert contract is not None, "emission runs only on the root unit"
        returns_value = return_fact is not None and not isinstance(return_fact, Reference)
        return_vid: int | None = None
        return_leaves: list[tuple[LeafPath, int]] = []
        match contract:
            case VoidReturn():
                if returns_value:
                    raise EmissionRejection("annotated '-> None' but returns a value", self._origin)
                if isinstance(return_fact, Reference) and return_fact.obj is not None:
                    raise EmissionRejection("annotated '-> None' but returns an object", self._origin)
            case ScalarReturn(kind=kind):
                if not returns_value:
                    if isinstance(return_fact, Reference) and return_fact.obj is not None:
                        raise EmissionRejection(
                            f"return type mismatch: declared {kind.value}, returns an object", self._origin
                        )
                    raise EmissionRejection(
                        f"return type mismatch: declared {kind.value}, returns nothing", self._origin
                    )
                if isinstance(return_fact, AggregateFact):
                    raise EmissionRejection(
                        f"return type mismatch: declared {kind.value}, returns an aggregate", self._origin
                    )
                if isinstance(return_fact, Known) and isinstance(return_fact.value, (MetaInt, NpInt)):
                    if kind is SemType.INT:
                        return_vid = self._builder.int_const(int(return_fact.value.value))  # an integer port
                    else:
                        return_vid = self._const(return_fact.value)
                elif isinstance(return_fact, Known):
                    return_vid = self._const(return_fact.value)
                else:
                    return_vid = self._read(unit.exit, ReturnPlace())
                if kind is SemType.FLOAT and isinstance(self._type_of(return_vid), IntType):
                    return_vid = self._builder.operation(IntToFloat(), [return_vid])  # int returned as declared float
                got = self._sem_of(self._type_of(return_vid))
                if got is not kind:
                    raise EmissionRejection(
                        f"return type mismatch: declared {kind.value}, returns {got.value}", self._origin
                    )
            case ArrayReturn():
                if not returns_value:
                    raise EmissionRejection("declared an array return but returns nothing", self._origin)
                if not isinstance(return_fact, AggregateFact):
                    raise EmissionRejection("return shape mismatch: declared an array, returns a scalar", self._origin)
                _validate_return_layout(contract, return_fact.layout, self._origin)
                zipped = zip(leaf_paths(return_fact.layout), return_fact.leaves, strict=True)
                for ordinal, (path, leaf_fact) in enumerate(zipped):
                    return_leaves.append((path, self._exit_leaf(unit.exit, path, ordinal, leaf_fact, SemType.FLOAT)))
            case _:  # a tuple/list contract: the value must be an aggregate of the declared structure
                if not returns_value:
                    raise EmissionRejection("declared an aggregate return but returns nothing", self._origin)
                if not isinstance(return_fact, AggregateFact):
                    raise EmissionRejection("declared an aggregate return but returns a scalar", self._origin)
                _validate_return_layout(contract, return_fact.layout, self._origin)
                zipped = zip(leaf_paths(return_fact.layout), return_fact.leaves, strict=True)
                for ordinal, (path, leaf_fact) in enumerate(zipped):
                    kind = _contract_leaf_kind(contract, path)
                    return_leaves.append((path, self._exit_leaf(unit.exit, path, ordinal, leaf_fact, kind)))
        promoted = self._result.runtime_state
        # Slots and public ports emit in first-STORE source order (matching production), not the order the RPO walk
        # happened to touch attributes; a leaf touched only by a read still trails a leaf stored earlier. An
        # aggregate slot expands to its cells in canonical leaf order.
        store_order = [leaf for leaf in self._result.store_order if leaf in promoted]
        store_order += [leaf for leaf in self._state_order if leaf in promoted and leaf not in store_order]
        public_live_outs: set[object] = set()
        for leaf in store_order:
            for ordinal in range(self._state_cells(leaf)):
                live_out = self._read(unit.exit, _LeafPlace(leaf, ordinal))
                self._builder.state_slot(self._slot_name(leaf, ordinal), self._leaf_reset(leaf, ordinal), live_out)
                if not leaf.path[-1].startswith("_"):
                    public_live_outs.add(self._exit_identity(live_out))
        if return_vid is not None and self._exit_identity(return_vid) not in public_live_outs:
            self._builder.output(port_name([0]), return_vid)  # a scalar return is leaf 0 of the bundle
        for path, vid in return_leaves:
            if self._exit_identity(vid) not in public_live_outs:  # equal to a public live-out: ride the state port
                self._builder.output(port_name(_port_path(path)), vid)
        for leaf in store_order:
            if not leaf.path[-1].startswith("_"):
                for ordinal in range(self._state_cells(leaf)):
                    self._builder.output(
                        state_port_name(self._slot_name(leaf, ordinal)),
                        self._read(unit.exit, _LeafPlace(leaf, ordinal)),
                    )
        self._builder.ret()
