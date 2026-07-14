"""
The FIR analyzer: optimistic executable-edge abstract interpretation (SCCP-style) with flow-sensitive per-edge
environments over Places. Facts form the lattice Unbound < Known(StaticValue) < Residual(SemType); joins are
per-Place, strong updates on stores, and only executable in-edges contribute. Static folding is Python-exact and
runs on the closed value domain (the width rule: runtime-typed numeric values never fold; a Known Bool always
drives edge selection). StaticFor headers unroll by cloning the body per trip once the iterable is Known; PyCall
sites expand on demand by grafting the callee's freshly instantiated template into the working graph (recursion is
a located rejection keyed by function and receiver identity). The result is a stable residual graph plus final
facts, validated to contain no unresolved calls, loop templates, or possibly-unbound reads on executable paths.

Everything here operates on a WORKING COPY of builder templates; templates are never mutated, so the state fixed
point can rebuild from scratch each outer round.
"""

import enum
import logging
import types
from collections.abc import Callable
from dataclasses import dataclass, field, replace

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
)
from ._opsem import BinOp, static_binop, static_compare, static_truth, static_unop
from ..._hir import BoolType, FloatAbs, FloatMax, FloatMin
from ._value import (
    MetaInt,
    StaticRecord,
    NpFloat,
    NpInt,
    ObjectRef,
    SemType,
    StaticBool,
    StaticFloat,
    StaticSeq,
    StaticValue,
    admit,
    as_python,
    same,
)

_logger = logging.getLogger(__name__)

_MAX_BLOCKS = 200_000
_MAX_VISITS = 1_000_000

_BITWISE_OPS = frozenset({BinOp.LSHIFT, BinOp.RSHIFT, BinOp.BITAND, BinOp.BITOR, BinOp.BITXOR})


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


def _is_unimplemented_library(target: object) -> bool:
    """A numpy ufunc or a ``math`` module member: a recognized library primitive, distinct from an arbitrary call."""
    import math

    import numpy as np

    return isinstance(target, np.ufunc) or any(target is member for member in vars(math).values())


def _datapath_zero(value: object) -> object:
    """Normalize a -0.0 fold input to +0.0: the ZKF datapath has no signed zero, so a static fold must not either."""
    return value + 0.0 if isinstance(value, float) and value == 0.0 else value


@dataclass(frozen=True, slots=True)
class _PropertyRead:
    """A component attribute read that resolved to a ``@property`` getter, to be desugared into a bound call."""

    getter: object  # a ``MethodType(fget, component)`` bound to the exact receiver


@dataclass(frozen=True, slots=True)
class Unbound:
    pass


@dataclass(frozen=True, slots=True)
class Known:
    value: StaticValue


@dataclass(frozen=True, slots=True)
class Residual:
    type: SemType


@dataclass(frozen=True, slots=True)
class FactSeq:
    """An aggregate whose elements are facts, present when at least one element is not Known (all-Known aggregates
    stay Known(StaticSeq)); subscripts and lengths stay static even when the payload is runtime."""

    items: tuple["Fact", ...]
    is_list: bool


@dataclass(frozen=True, slots=True)
class MaybeUnbound:
    """Joined bound-and-unbound: reading this is a located rejection (Python may raise here)."""

    inner: "Known | Residual | FactSeq"


type Fact = Unbound | Known | Residual | FactSeq | MaybeUnbound

_UNBOUND = Unbound()


def _residual_type(value: StaticValue) -> SemType | None:
    match value:
        case StaticBool():
            return SemType.BOOL
        case StaticFloat() | NpFloat():
            return SemType.FLOAT
        case MetaInt() | NpInt():
            return SemType.INT
        case _:
            return None


def join_facts(a: Fact, b: Fact, origin: OriginStack) -> Fact:
    if a is b:
        return a
    match a, b:
        case (Unbound(), Unbound()):
            return _UNBOUND
        case (Unbound(), (Known() | Residual() | FactSeq()) as bound) | (
            (Known() | Residual() | FactSeq()) as bound,
            Unbound(),
        ):
            return MaybeUnbound(bound)
        case (Unbound(), MaybeUnbound() as half) | (MaybeUnbound() as half, Unbound()):
            return half
        case (MaybeUnbound(inner=x), MaybeUnbound(inner=y)):
            joined = join_facts(x, y, origin)
            assert isinstance(joined, (Known, Residual, FactSeq))
            return MaybeUnbound(joined)
        case (MaybeUnbound(inner=x), (Known() | Residual() | FactSeq()) as y) | (
            (Known() | Residual() | FactSeq()) as y,
            MaybeUnbound(inner=x),
        ):
            joined = join_facts(x, y, origin)
            assert isinstance(joined, (Known, Residual, FactSeq))
            return MaybeUnbound(joined)
        case (Known(value=x), Known(value=y)):
            if same(x, y):
                return a
            if (
                isinstance(x, StaticSeq)
                and isinstance(y, StaticSeq)
                and x.is_list == y.is_list
                and len(x.items) == len(y.items)
            ):
                return join_facts(_lift_seq(x), _lift_seq(y), origin)
            x_type, y_type = _residual_type(x), _residual_type(y)
            if x_type is not None and x_type == y_type:
                return Residual(x_type)
            if {x_type, y_type} <= {SemType.FLOAT, SemType.INT}:  # BOOL never promotes: explicit conversions only
                return Residual(SemType.FLOAT)
            raise AnalysisRejection("values of irreconcilable kinds merge here", origin)
        case (Known(value=x), Residual(type=t)) | (Residual(type=t), Known(value=x)):
            x_type = _residual_type(x)
            if x_type == t:
                return Residual(t)
            if x_type is not None and SemType.BOOL not in (x_type, t):
                return Residual(SemType.FLOAT)
            raise AnalysisRejection("values of irreconcilable kinds merge here", origin)
        case (Residual(type=x_t), Residual(type=y_t)):
            if x_t == y_t:
                return a
            if SemType.BOOL in (x_t, y_t):
                raise AnalysisRejection("values of irreconcilable kinds merge here", origin)
            return Residual(SemType.FLOAT)
        case (FactSeq() as x, FactSeq() as y) if len(x.items) == len(y.items) and x.is_list == y.is_list:
            return _pack_seq(tuple(join_facts(p, q, origin) for p, q in zip(x.items, y.items)), x.is_list)
        case (FactSeq() as x, Known(value=StaticSeq() as y)) | (Known(value=StaticSeq() as y), FactSeq() as x):
            lifted = _lift_seq(y)
            return join_facts(x, lifted, origin)
    raise AnalysisRejection("values of irreconcilable shapes merge here", origin)


def _lift_seq(value: StaticSeq) -> FactSeq:
    return FactSeq(tuple(Known(item) for item in value.items), value.is_list)


def _pack_seq(items: tuple[Fact, ...], is_list: bool) -> Fact:
    if all(isinstance(item, Known) for item in items):
        return Known(StaticSeq(tuple(item.value for item in items if isinstance(item, Known)), is_list=is_list))
    return FactSeq(items, is_list)


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
        changed = False
        for place in set(self.facts) | set(other.facts):
            mine, theirs = self.facts.get(place), other.facts.get(place)
            if default is not None and (mine is None or theirs is None):
                fallback = default(place)
                mine = fallback if mine is None else mine
                theirs = fallback if theirs is None else theirs
            joined = join_facts(
                mine if mine is not None else _UNBOUND, theirs if theirs is not None else _UNBOUND, origin
            )
            if joined != self.facts.get(place, _UNBOUND):
                self.facts[place] = joined
                changed = True
        return changed


@dataclass(slots=True)
class ResidualUnit:
    """The analyzer's output: the stabilized working graph plus final facts for emission."""

    unit: FunctionUnit
    block_in: dict[BlockId, _Env]
    executable_blocks: set[BlockId]
    executable_edges: set[tuple[BlockId, BlockId]]


class Analyzer:
    def __init__(self, fn: object) -> None:
        self._root_template = build_unit(fn)
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
        self._identity_calls: set[int] = set()
        self._conversion_calls: set[int] = set()  # runtime float()/int()/bool() casts, lowered to a conversion op
        self._unroll_cache: dict[BlockId, tuple[Fact, BlockId]] = {}
        self._bound_methods: dict[tuple[int, str], object] = {}
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
            result = self.analyze(param_facts)
            exit_env = result.block_in.get(result.unit.exit, _Env())
            reachable = _coreachable(result.unit, result.unit.exit, result.executable_edges)
            new_w = set(self._runtime_state)
            for block_id, leaf in self._discovered_stores:
                if block_id in result.executable_blocks and block_id in reachable:
                    new_w.add(leaf)
            new_d = dict(self._state_livein)
            for leaf in new_w:
                reset = self._state_reset_fact(leaf)
                exit_fact = exit_env.get(leaf)
                incoming = (
                    reset
                    if isinstance(exit_fact, Unbound)
                    else join_facts(reset, exit_fact, (Origin(result.unit.name, 0, 0),))
                )
                previous = new_d.get(leaf)
                new_d[leaf] = (
                    incoming if previous is None else join_facts(previous, incoming, (Origin(result.unit.name, 0, 0),))
                )
            if new_w == self._runtime_state and new_d == self._state_livein:
                _logger.info("state fixpoint stable after %d round(s): %d runtime leaves", round_index + 1, len(new_w))
                for header_id, (_, chain_entry) in self._unroll_cache.items():
                    header = result.unit.blocks[header_id]
                    assert isinstance(header.terminator, StaticFor)
                    header.terminator = Jump(chain_entry, header.terminator.origin)
                _validate(
                    result,
                    self._concrete_calls | self._intrinsic_calls | self._identity_calls | self._conversion_calls,
                )
                return result
            self._runtime_state = new_w
            self._state_livein = new_d
            self._block_ancestry = {}
            self._discovered_stores = set()
            self._concrete_calls = set()
            self._intrinsic_calls = set()
            self._identity_calls = set()
            self._conversion_calls = set()
            self._unroll_cache = {}
            _logger.info("state round %d: %d runtime leaves, %d live-in facts", round_index + 1, len(new_w), len(new_d))
        raise AnalysisRejection("state fixpoint failed to stabilize", (Origin(self._root_template.name, 0, 0),))

    def _snapshot_leaf(self, leaf: StateLeaf) -> Fact:
        current: object = leaf.component
        for attribute in leaf.path:
            try:
                current = getattr(current, attribute)
            except AttributeError:
                raise AnalysisRejection(
                    f"state attribute '{'.'.join(leaf.path)}' does not exist on the component at compile time "
                    "(assign it in __init__)",
                    (Origin(self._root_template.name, 0, 0),),
                ) from None
        admitted = admit(current)
        return Known(admitted) if admitted is not None else Known(ObjectRef(current))

    def _state_reset_fact(self, leaf: StateLeaf) -> Fact:
        current: object = leaf.component
        for attribute in leaf.path:
            try:
                current = getattr(current, attribute)
            except AttributeError:
                raise AnalysisRejection(
                    f"state attribute '{'.'.join(leaf.path)}' does not exist on the component at compile time "
                    "(assign it in __init__)",
                    (Origin(self._root_template.name, 0, 0),),
                ) from None
        admitted = admit(current)
        if admitted is None:
            return Known(ObjectRef(current))
        sem = _residual_type(admitted)
        if sem is None:
            raise AnalysisRejection(
                f"state attribute '{'.'.join(leaf.path)}' has an unsupported reset type",
                (Origin(self._root_template.name, 0, 0),),
            )
        if sem is SemType.BOOL:
            return Known(admitted)  # a Known Bool folds exactly (no width) and keeps invariant flags folding
        return Residual(sem)

    def state_store_order(self, result: ResidualUnit) -> list[StateLeaf]:
        """
        State leaves in first-store SOURCE order, matching the production front-end's port contract. The order key is
        the storing op's source origin (line, column), not the CFG block index -- a store nested in a branch has a
        higher block id than a later top-level store yet comes first in the source text.
        """
        first_store: dict[StateLeaf, tuple[int, int]] = {}
        for block_id in result.executable_blocks:
            block = result.unit.blocks[block_id]
            env = result.block_in[block_id].copy()
            for index, op in enumerate(block.ops):
                if isinstance(op, PyStoreAttr):
                    obj_fact = env.get(Local(op.obj))
                    if isinstance(obj_fact, Known) and isinstance(obj_fact.value, ObjectRef):
                        leaf = StateLeaf(obj_fact.value.obj, (op.name,))
                        position = (op.origin[0].line, op.origin[0].column)
                        if leaf not in first_store or position < first_store[leaf]:
                            first_store[leaf] = position
                self._transfer(result.unit, block, index, op, env)
        return sorted(first_store, key=lambda leaf: first_store[leaf])

    def binding_facts(self, result: ResidualUnit) -> dict[BindingId, Fact]:
        """
        The final fact of every op-produced binding, by replaying the transfer over the stabilized graph. Temporaries
        are write-once, so one pass records each binding's authoritative fact; emission consults it instead of
        re-deriving folds. The replay does not mutate the graph (all calls are already expanded or concrete-folded).
        """
        from ._ir import op_dst

        facts: dict[BindingId, Fact] = {}
        for block_id in result.executable_blocks:
            block = result.unit.blocks[block_id]
            env = result.block_in[block_id].copy()
            for index, op in enumerate(block.ops):
                self._transfer(result.unit, block, index, op, env)
                dst = op_dst(op)
                if dst is not None:
                    facts[dst] = env.get(Local(dst))
        return facts

    def intrinsic_calls(self) -> set[int]:
        return set(self._intrinsic_calls)

    def identity_calls(self) -> set[int]:
        return set(self._identity_calls)

    def conversion_calls(self) -> set[int]:
        return set(self._conversion_calls)

    def provenance(self) -> dict[int, tuple[str, ...]]:
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

    def runtime_state(self) -> set[StateLeaf]:
        return set(self._runtime_state)

    def state_livein(self) -> dict[StateLeaf, Fact]:
        return dict(self._state_livein)

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
            default = Residual(unit.param_types.get(param.name, SemType.FLOAT))
            fact = (param_facts or {}).get(param.name, default)
            entry_env.set(Local(param), fact)
        if unit.bound_self is not None and unit.params:
            entry_env.set(Local(unit.params[0]), Known(ObjectRef(unit.bound_self)))
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
                env.set(Local(dst), Known(value))
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
                if bin_op is BinOp.MATMUL and _seq_side(lhs_fact) is None and _seq_side(rhs_fact) is None:
                    raise AnalysisRejection("@ is not defined for scalars", op.origin)
                if op.inplace and _is_list_fact(lhs_fact):
                    raise AnalysisRejection(
                        "in-place list mutation is not supported (aliases would observe it); rebind instead",
                        op.origin,
                    )
                concat = _concat_seqs(bin_op, lhs_fact, rhs_fact)
                if concat is not None:
                    env.set(Local(dst), concat)
                elif isinstance(lhs_fact, FactSeq) or isinstance(rhs_fact, FactSeq):
                    raise AnalysisRejection("arithmetic on an aggregate value", op.origin)
                elif bin_op in _BITWISE_OPS:
                    env.set(Local(dst), self._fold_bitwise(bin_op, lhs_fact, rhs_fact, op.origin))
                else:
                    env.set(
                        Local(dst),
                        self._fold_binary(
                            lambda a, b: static_binop(bin_op, a, b),
                            lhs_fact,
                            rhs_fact,
                            op.origin,
                            # Python's / always yields float; residual int**int may too (negative exponents).
                            promotes_to_float=bin_op in (BinOp.DIV, BinOp.POW),
                        ),
                    )
            case PyUn(dst=dst, op=un_op, operand=operand):
                operand_fact = env.get(Local(operand))
                if (isinstance(operand_fact, Known) and isinstance(operand_fact.value, StaticBool)) or (
                    isinstance(operand_fact, Residual) and operand_fact.type is SemType.BOOL
                ):
                    raise AnalysisRejection("arithmetic on a bool requires an explicit conversion", op.origin)
                if isinstance(operand_fact, Known):
                    folded = static_unop(un_op, operand_fact.value)
                    env.set(
                        Local(dst),
                        Known(folded) if folded is not None else self._residual_of(operand_fact, op.origin),
                    )
                else:
                    env.set(Local(dst), self._residual_of(operand_fact, op.origin))
            case PyCompare(dst=dst, op=rel, lhs=lhs, rhs=rhs):
                env.set(
                    Local(dst),
                    self._fold_binary(
                        lambda a, b: static_compare(rel, a, b),
                        env.get(Local(lhs)),
                        env.get(Local(rhs)),
                        op.origin,
                        default=SemType.BOOL,
                    ),
                )
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
                facts = tuple(env.get(Local(item)) for item in items)
                result = _pack_seq(facts, is_list=isinstance(op, BuildList))
                env.set(Local(dst), result)
            case PyLen(dst=dst, obj=obj):
                obj_fact = env.get(Local(obj))
                if isinstance(obj_fact, FactSeq):
                    length = admit(len(obj_fact.items))
                    assert length is not None
                    result = Known(length)
                elif isinstance(obj_fact, Known):
                    concrete = as_python(obj_fact.value)
                    try:
                        length = admit(len(concrete))  # type: ignore[arg-type]
                    except TypeError as error:
                        raise AnalysisRejection(str(error), op.origin) from None
                    assert length is not None
                    result = Known(length)
                else:
                    raise AnalysisRejection("length of a runtime value", op.origin)
                env.set(Local(dst), result)
            case PySubscript(dst=dst, obj=obj, index=idx):
                result = self._subscript(env.get(Local(obj)), env.get(Local(idx)), op.origin)
                env.set(Local(dst), result)
            case PyAttr(dst=dst, obj=obj, name=name):
                attr = self._attribute(env, env.get(Local(obj)), name, op.origin)
                if isinstance(attr, _PropertyRead):
                    # Desugar the property read into a bound zero-argument call and re-run: the generic call-expansion
                    # machinery then inlines the getter, remaps its return, and threads state through unchanged.
                    callee = BindingId(f"%p{self._binding_serial}", self._binding_serial)
                    self._binding_serial += 1
                    block.ops[index : index + 1] = [
                        LoadConst(callee, ObjectRef(attr.getter), op.origin),
                        PyCall(dst, callee, (), (), op.origin),
                    ]
                    return True
                env.set(Local(dst), attr)
            case PyStoreAttr(obj=obj, name=name, src=src):
                obj_fact = env.get(Local(obj))
                if not (isinstance(obj_fact, Known) and isinstance(obj_fact.value, ObjectRef)):
                    raise AnalysisRejection("attribute store on a non-component value", op.origin)
                if isinstance(obj_fact.value.obj, (types.ModuleType, type)):
                    # A module/class is a compile-time namespace, not runtime state: mutating it would make later
                    # reads (which snapshot the live object) disagree with the store. Reject, as production does.
                    raise AnalysisRejection("assignment to a module or class attribute is not supported", op.origin)
                _reject_attribute_hooks(type(obj_fact.value.obj), op.origin)
                _reject_descriptor(type(obj_fact.value.obj), name, op.origin)
                src_fact = env.get(Local(src))
                if isinstance(src_fact, Known) and isinstance(src_fact.value, ObjectRef):
                    # Storing a component/sub-object into an attribute would change the component topology per
                    # transaction (a slot's owner is fixed at the initial snapshot); reject it at the store, located.
                    raise AnalysisRejection(
                        f"component member '{name}' cannot be rebound; component topology is fixed", op.origin
                    )
                leaf = StateLeaf(obj_fact.value.obj, (name,))
                self._discovered_stores.add((block.id, leaf))
                env.set(leaf, src_fact)
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
                return fact
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
            if isinstance(lhs.value, StaticBool) and isinstance(rhs.value, StaticBool):
                # static_binop covers only numerics; combine two Known booleans here so ``True & False`` folds to a
                # Known bool and drives edge selection (a dead branch guarded by it is never analyzed).
                a, b = lhs.value.value, rhs.value.value
                combined = a and b if bin_op is BinOp.BITAND else a or b if bin_op is BinOp.BITOR else a != b
                return Known(StaticBool(bool(combined)))
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
            case _:
                raise AnalysisRejection("a runtime operation reads an aggregate or unbound value", origin)

    def _truth_fact(self, fact: Fact, origin: OriginStack) -> Fact:
        match fact:
            case Known(value=value):
                truth = static_truth(value)
                if truth is None and _residual_type(value) is None:
                    raise AnalysisRejection("the truth value of this object is not defined here", origin)
                return Known(StaticBool(truth)) if truth is not None else Residual(SemType.BOOL)
            case FactSeq(items=items):
                return Known(StaticBool(len(items) != 0))
            case _:
                return Residual(SemType.BOOL)

    def _subscript(self, obj: Fact, index: Fact, origin: OriginStack) -> Fact:
        if isinstance(obj, FactSeq) and isinstance(index, Known):
            import operator

            try:
                position = operator.index(as_python(index.value))  # type: ignore[arg-type]  # np ints qualify
            except TypeError:
                raise AnalysisRejection("sequence index is not an integer", origin) from None
            if not -len(obj.items) <= position < len(obj.items):
                raise AnalysisRejection("sequence index out of range", origin)
            return obj.items[position]
        if isinstance(obj, Known) and isinstance(index, Known):
            try:
                concrete = as_python(obj.value)[as_python(index.value)]  # type: ignore[index]
            except Exception as error:
                raise AnalysisRejection(f"subscript fails here: {error}", origin) from None
            admitted = admit(concrete)
            if admitted is None:
                return Known(ObjectRef(concrete))
            return Known(admitted)
        raise AnalysisRejection("subscript of a runtime value is not supported yet", origin)

    def _attribute(self, env: _Env, obj: Fact, name: str, origin: OriginStack) -> "Fact | _PropertyRead":
        if isinstance(obj, Known) and isinstance(obj.value, ObjectRef):
            component = obj.value.obj
            if isinstance(component, (types.ModuleType, type)):
                # A namespace (math, np, a class), not a stateful component: attribute access is a plain lookup,
                # so math.sqrt/np.floor resolve to the callable the call site then dispatches through the registry.
                try:
                    attribute = getattr(component, name)
                except AttributeError as error:
                    raise AnalysisRejection(str(error), origin) from None
                admitted = admit(attribute)
                return Known(admitted) if admitted is not None else Known(ObjectRef(attribute))
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
                return Known(ObjectRef(self._bound_methods[key]))
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
                concrete = getattr(component, name)
            except AttributeError as error:
                raise AnalysisRejection(str(error), origin) from None
            admitted = admit(concrete)
            if admitted is None:
                # ``concrete`` is a sub-object (a potential child component): record the parent -> child graph edge.
                # Canonical member paths are resolved from these edges by a shortest-path fixpoint in ``provenance()``,
                # so a child's slot name is order-independent even when a lexicographically-smaller alias is discovered
                # later (the state-leaf cache above would otherwise freeze a stale first-seen path).
                self._component_edges.add((id(component), name, id(concrete)))
            fact = Known(admitted) if admitted is not None else Known(ObjectRef(concrete))
            env.set(leaf, fact)
            return fact
        if isinstance(obj, Known):
            if _is_list_fact(obj):
                raise AnalysisRejection(
                    f"list method '{name}' is not supported (lists are immutable values here); rebind with + instead",
                    origin,
                )
            try:
                concrete = getattr(as_python(obj.value), name)
            except AttributeError as error:
                raise AnalysisRejection(str(error), origin) from None
            admitted = admit(concrete)
            if admitted is None and callable(concrete):
                if isinstance(obj.value, StaticRecord):
                    raise AnalysisRejection(f"method '{name}' on a record value is not supported yet", origin)
                value_key = (obj.value, name)
                if value_key not in self._value_methods:
                    self._value_methods[value_key] = concrete
                return Known(ObjectRef(self._value_methods[value_key]))
            return Known(admitted) if admitted is not None else Known(ObjectRef(concrete))
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
                cached = self._unroll_cache.get(block.id)
                if cached is not None:
                    cached_fact, chain_entry = cached
                    if (
                        isinstance(iterable_fact, Known)
                        and isinstance(cached_fact, Known)
                        and same(iterable_fact.value, cached_fact.value)
                    ):
                        return [chain_entry]
                    raise AnalysisRejection("loop iterable is not stable across analysis rounds", terminator.origin)
                chain_entry = self._unroll(unit, block, terminator, env)
                self._unroll_cache[block.id] = (iterable_fact, chain_entry)
                return [chain_entry]
            case Fail() | UnitExit():
                return []
        raise AssertionError(terminator)

    # ------------------------------------ StaticFor unrolling ------------------------------------

    def _unroll(self, unit: FunctionUnit, header: Block, loop: StaticFor, env: _Env) -> BlockId:
        iterable = env.get(Local(loop.iterable))
        if not isinstance(iterable, Known):
            raise AnalysisRejection("loop trip count is not static here", loop.origin)
        concrete = as_python(iterable.value)
        try:
            trip_count = len(concrete)  # type: ignore[arg-type]  # sized BEFORE materializing (range(10**9)!)
        except TypeError:
            raise AnalysisRejection("loop iterable has no static length", loop.origin) from None
        except OverflowError:  # len() of an astronomically large range (range(10**38)): far past any threshold
            raise AnalysisRejection(
                f"loop trip count exceeds the unroll threshold {UNROLL_THRESHOLD}; a counted back-edge loop is not "
                "supported yet",
                loop.origin,
            ) from None
        if trip_count > UNROLL_THRESHOLD:
            raise AnalysisRejection(
                f"trip count {trip_count} exceeds the unroll threshold {UNROLL_THRESHOLD}; a counted back-edge loop "
                "is not supported yet",
                loop.origin,
            )
        elements = list(concrete)  # type: ignore[call-overload]
        _logger.info("unrolling %d trip(s) at %s", trip_count, loop.origin[0])
        chain_target = loop.exit_target
        for element in reversed(elements):
            admitted = admit(element)
            value: StaticValue = admitted if admitted is not None else ObjectRef(element)
            body_entry = self._clone_subgraph(unit, loop.body_entry, header.id, chain_target, loop)
            prelude = Block(self._fresh_block_id())
            temp = BindingId(f"%u{self._temp_serial}", self._temp_serial)
            self._temp_serial += 1
            prelude.ops.append(LoadConst(temp, value, loop.origin))
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

    def _expand_call(self, unit: FunctionUnit, block: Block, index: int, call: PyCall, env: _Env) -> bool:
        from .._lib import Intrinsic, Library, resolve

        callee_fact = env.get(Local(call.callee))
        if not (isinstance(callee_fact, Known) and isinstance(callee_fact.value, ObjectRef)):
            raise AnalysisRejection("call target is not resolvable here", call.origin)
        target = callee_fact.value.obj
        match = resolve(target)
        if isinstance(match, Library):
            target = match.stub  # a composite library stub inlines exactly like a user function
        elif isinstance(match, Intrinsic):
            argument_facts = [env.get(Local(arg)) for arg in call.args] + [
                env.get(Local(value)) for _, value in call.kwargs
            ]
            if all(isinstance(fact, Known) for fact in argument_facts):
                pass  # fully static: fold concretely below through the original callable
            else:
                # A runtime-operand intrinsic (sqrt(x), sin(x)...) becomes an HIR operation at emission; keep the
                # PyCall in the graph, typed by the operator's result, and let emission resolve the registry match.
                if call.kwargs:
                    raise AnalysisRejection("keyword arguments to a hardware intrinsic are not supported", call.origin)
                arity = match.operator.signature.arity
                if len(call.args) != arity:
                    raise AnalysisRejection(f"intrinsic expects {arity} argument(s), got {len(call.args)}", call.origin)
                # abs/min/max are int-polymorphic in Python: on an integer operand (runtime OR a Known integer that can
                # be the min/max winner) they return an EXACT integer, so promoting to float here would let subsequent
                # integer arithmetic round (min(2**24, x) + 1 + 1). There is no integer abs/min/max operator yet, so an
                # integer operand is a located rejection, never a silent float promotion (cast an operand to float first).
                if isinstance(match.operator, (FloatAbs, FloatMin, FloatMax)) and any(
                    fact == Residual(SemType.INT)
                    or (isinstance(fact, Known) and isinstance(fact.value, (MetaInt, NpInt)))
                    for fact in argument_facts
                ):
                    raise AnalysisRejection(
                        f"an integer operand to {match.operator.mnemonic} is not yet lowerable; cast to float first",
                        call.origin,
                    )
                sem = (
                    SemType.INT
                    if match.returns_int
                    else SemType.BOOL if isinstance(match.operator.signature.result_type, BoolType) else SemType.FLOAT
                )
                env.set(Local(call.dst), Residual(sem))
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
            if not all(isinstance(fact, Known) for fact in argument_facts + [fact for _, fact in keyword_facts]):
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
                if (
                    cast_target is not None
                    and not keyword_facts
                    and len(argument_facts) == 1
                    and isinstance(argument_facts[0], Residual)
                ):
                    if argument_facts[0].type is cast_target:
                        env.set(Local(call.dst), argument_facts[0])  # same kind: identity
                        self._identity_calls.add(id(call))
                    else:
                        env.set(Local(call.dst), Residual(cast_target))
                        self._conversion_calls.add(id(call))
                    return False
                if _is_unimplemented_library(target):
                    # A recognized math/numpy function with no fast-math hardware equivalent (erf, spacing, a ufunc):
                    # a distinct public error so the user knows it is a missing library primitive, not a bad call.
                    raise LibraryAnalysisRejection(f"library function {name!r} is not implemented yet", call.origin)
                raise AnalysisRejection(f"call to {name} with runtime arguments is not supported yet", call.origin)
            try:
                concrete = target(  # type: ignore[operator]
                    *[_datapath_zero(as_python(fact.value)) for fact in argument_facts if isinstance(fact, Known)],
                    **{
                        keyword: _datapath_zero(as_python(fact.value))
                        for keyword, fact in keyword_facts
                        if isinstance(fact, Known)
                    },
                )
            except Exception as error:
                raise AnalysisRejection(f"call fails here: {error}", call.origin) from None
            admitted = admit(concrete)
            folded: StaticValue = admitted if admitted is not None else ObjectRef(concrete)
            env.set(Local(call.dst), Known(folded))
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
            block.ops.append(LoadConst(self_temp, ObjectRef(template.bound_self), call.origin))
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
                default_value: StaticValue = (
                    admitted if admitted is not None else ObjectRef(default_by_name[param.name])
                )
                default_temp = BindingId(f"%d{self._binding_serial}", self._binding_serial)
                self._binding_serial += 1
                block.ops.append(LoadConst(default_temp, default_value, call.origin))
                source = default_temp
            else:
                raise AnalysisRejection(f"missing argument '{param.name}'", call.origin)
            block.ops.append(StorePlace(Local(fresh(param)), source, call.origin))
        if keyword:
            raise AnalysisRejection(f"unexpected keyword argument '{next(iter(keyword))}'", call.origin)
        block.terminator = Jump(block_map[template.entry], call.origin)
        continuation.ops.insert(0, LoadPlace(call.dst, Local(return_local), call.origin))
        return True


def _identity_place(place: Place) -> Place:
    return place


def _remap_op(op: Op, fresh: Callable[[BindingId], BindingId], remap_place: Callable[[Place], Place]) -> Op:
    match op:
        case LoadConst():
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
        case Fail(message=message, origin=origin):
            return Fail(message, origin)  # a COPY: grafting re-attributes origins and must never touch templates
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


def _validate(result: ResidualUnit, concrete_calls: set[int]) -> None:
    for block_id in result.executable_blocks:
        block = result.unit.blocks[block_id]
        for op in block.ops:
            assert not isinstance(op, PyCall) or id(op) in concrete_calls, f"{block_id}: unexpanded call survived"
        assert not isinstance(block.terminator, StaticFor), f"{block_id}: loop template survived analysis"
        if isinstance(block.terminator, Fail):
            raise AnalysisRejection(block.terminator.message, block.terminator.origin)


def _concat_seqs(bin_op: BinOp, lhs: Fact, rhs: Fact) -> Fact | None:
    if bin_op is BinOp.MUL:
        seq, count = (lhs, rhs) if isinstance(lhs, (FactSeq, Known)) and _seq_side(lhs) else (rhs, lhs)
        lifted = _seq_side(seq)
        if lifted is not None and isinstance(count, Known) and isinstance(count.value, (MetaInt, NpInt)):
            repetitions = count.value.value
            if 0 <= repetitions <= 1024:
                return _pack_seq(lifted.items * repetitions, lifted.is_list)
        return None
    if bin_op is not BinOp.ADD:
        return None
    sides = []
    for fact in (lhs, rhs):
        if isinstance(fact, FactSeq):
            sides.append(fact)
        elif isinstance(fact, Known) and isinstance(fact.value, StaticSeq):
            sides.append(_lift_seq(fact.value))
        else:
            return None
    if sides[0].is_list != sides[1].is_list:
        return None
    return _pack_seq(sides[0].items + sides[1].items, sides[0].is_list)


def _seq_side(fact: Fact) -> FactSeq | None:
    match fact:
        case FactSeq():
            return fact
        case Known(value=StaticSeq() as seq):
            return _lift_seq(seq)
        case _:
            return None


def _is_list_fact(fact: Fact) -> bool:
    match fact:
        case FactSeq(is_list=True):
            return True
        case Known(value=StaticSeq(is_list=True)):
            return True
        case _:
            return False


def _mro_attribute_of(klass: type, name: str) -> object | None:
    return next((c.__dict__[name] for c in klass.__mro__ if name in c.__dict__), None)


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
