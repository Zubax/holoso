"""
FIR -> HIR emission over the analyzer's stabilized residual graph: only executable blocks and edges are walked,
per-block facts are replayed from the final environments, Known values materialize as constants at their use
sites, and residual operations become typed HIR operations. Value numbering is Braun-style sealed-block SSA over
Places (named locals, state leaves, the return place): straight-line reads chase the unique predecessor, joins
create phis, loop headers read through open phis closed once their latch block is emitted, and single-value joins
collapse without a phi. HIR phi arms are keyed by predecessor block, which the 1:1 block mapping preserves.

State: every read/written leaf becomes a state_read and, if in the promoted set, a state slot named by its
attribute path with its reset snapshot and canonical-exit live-out; the returned value becomes the out port. The
port contract mirrors the production front-end so the differential harness compares model I/O across front-ends.
"""

import logging
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
    Operator,
    Select,
    Type,
)
from ._analyze import Analyzer, Fact, FactSeq, Known, Residual, ResidualUnit
from ._ir import (
    BindingId,
    BlockId as FirBlockId,
    Branch as FirBranch,
    BuildList,
    BuildTuple,
    Jump as FirJump,
    LoadConst,
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
    ReturnPlace,
    SelectMode,
    StateLeaf,
    StorePlace,
    UnbindPlace,
    UnitExit,
)
from ._opsem import BinOp, UnOp
from ._value import MetaInt, NpFloat, NpInt, ObjectRef, SemType, StaticFloat, StaticSeq, StaticValue, as_python

_logger = logging.getLogger(__name__)

# A literal power expands to |exponent|-1 chained multiplies; this bounds that expansion so ``x**(10**9)`` refuses
# instead of hanging emission, while leaving any realistic exponent (a degree-N monomial) free to expand.
_MAX_POWER_CHAIN = 1024

type FactOf = Callable[[BindingId], Fact]
type ValueOf = Callable[[BindingId], int]


class EmissionRejection(UnsupportedConstruct):
    """A located refusal discovered during HIR emission: an unsupported construct that survived analysis."""


def lower_fir(kernel: object) -> Hir:
    """The shadow front-end pipeline: build, analyze to the W/D fixed point, emit HIR."""
    analyzer = Analyzer(kernel)
    result = analyzer.fixpoint()
    return _Emitter(result, analyzer).emit()


def _exact_float(value: object) -> float:
    # A datapath value must be binary64-exact: an integer too wide (2**53 + 1) would silently round on the way in
    # and change a comparison against a runtime value, so it is a located rejection per the materialization policy.
    import numpy as np

    try:
        result = float(value)  # type: ignore[arg-type]
    except OverflowError:  # an integer too large to convert (10**400): far beyond binary64 range
        raise EmissionRejection(
            "integer constant is not exactly representable in the float datapath (too large)"
        ) from None
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)) and result != int(value):
        raise EmissionRejection(f"integer constant {value} is not exactly representable in the float datapath")
    return result


@dataclass(frozen=True, slots=True)
class _KnownHandle:
    """A Known scalar element of a runtime aggregate; materializes as a constant only when used."""

    value: StaticValue


@dataclass(frozen=True, slots=True)
class _ValueHandle:
    """A residual scalar element, already emitted as the given HIR value."""

    vid: int


@dataclass(frozen=True, slots=True)
class _SeqHandle:
    """A nested-sequence element (or a whole aggregate layout), carrying its list/tuple flavor honestly."""

    items: "tuple[_Handle, ...]"
    is_list: bool


type _Handle = _KnownHandle | _ValueHandle | _SeqHandle


def _is_bool_fact(fact: Fact | None) -> bool:
    match fact:
        case Residual(type=SemType.BOOL):
            return True
        case Known(value=value):
            from ._value import StaticBool

            return isinstance(value, StaticBool)
        case _:
            return False


class _Emitter:
    def __init__(self, result: ResidualUnit, analyzer: Analyzer) -> None:
        self._result = result
        self._analyzer = analyzer
        self._builder = HirBuilder()
        self._fir_to_hir: dict[FirBlockId, int] = {}
        self._definitions: dict[tuple[FirBlockId, Place], int] = {}
        self._sealed: set[FirBlockId] = set()
        self._emitted: set[FirBlockId] = set()
        self._open_phis: dict[FirBlockId, list[tuple[Place, int]]] = {}
        self._predecessors: dict[FirBlockId, list[FirBlockId]] = {}
        for source, target in sorted(result.executable_edges, key=lambda e: (e[0].index, e[1].index)):
            self._predecessors.setdefault(target, []).append(source)
        self._state_reads: dict[StateLeaf, int] = {}
        self._state_order: list[StateLeaf] = []
        self._provenance = analyzer.provenance()  # component id -> canonical member path from the root
        self._slot_names: dict[str, StateLeaf] = {}  # encoded slot name -> its leaf, to catch a rare name collision
        self._layouts: dict[BindingId, _SeqHandle] = {}  # FactSeq bindings -> their typed element layout
        self._binding_fact_cache: dict[BindingId, Fact] | None = None

    def emit(self) -> Hir:
        unit = self._result.unit
        order = self._reverse_postorder()
        if unit.exit not in order:
            # No path reaches the canonical exit (e.g. an unconditional `while True` with no break): the kernel
            # produces no output, so there is nothing to synthesize. A located refusal, not a downstream crash.
            raise EmissionRejection("the function never returns on any path")
        for fir_id in order:
            self._fir_to_hir[fir_id] = self._builder.block()
        self._builder.position_at(self._fir_to_hir[unit.entry])
        parameters = unit.params[1:] if unit.bound_self is not None else unit.params
        entry_facts = self._result.block_in[unit.entry].facts
        for parameter in parameters:
            vid = self._builder.input(parameter.name, self._fact_port_type(entry_facts.get(Local(parameter))))
            self._write(unit.entry, Local(parameter), vid)
        for fir_id in order:
            self._emit_block(fir_id)
            self._emitted.add(fir_id)
            for successor in list(self._predecessors):
                if successor not in self._sealed and all(p in self._emitted for p in self._predecessors[successor]):
                    self._seal(successor)
        self._finish_exit()
        return self._builder.finish()

    def _reverse_postorder(self) -> list[FirBlockId]:
        # Iterative post-order: a large static unroll (thousands of blocks, within the analyzer's trip budget) would
        # overflow a recursive DFS, so the traversal keeps its own explicit stack.
        seen: set[FirBlockId] = {self._result.unit.entry}
        order: list[FirBlockId] = []
        stack: list[tuple[FirBlockId, list[FirBlockId]]] = [
            (self._result.unit.entry, sorted(self._successors(self._result.unit.entry), key=lambda b: b.index))
        ]
        while stack:
            block_id, pending = stack[-1]
            advanced = False
            while pending:
                successor = pending.pop(0)
                if successor in self._result.executable_blocks and successor not in seen:
                    seen.add(successor)
                    stack.append((successor, sorted(self._successors(successor), key=lambda b: b.index)))
                    advanced = True
                    break
            if not advanced:
                order.append(block_id)
                stack.pop()
        order.reverse()
        assert order and order[0] == self._result.unit.entry
        return order

    def _successors(self, block_id: FirBlockId) -> list[FirBlockId]:
        return [t for (s, t) in self._result.executable_edges if s == block_id]

    # ---------------------------------------- SSA over Places ----------------------------------------

    def _type_of(self, vid: int) -> Type:
        return self._builder.type_of(vid)

    def _write(self, block: FirBlockId, place: Place, vid: int) -> None:
        self._definitions[(block, place)] = vid

    def _read(self, block: FirBlockId, place: Place) -> int:
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

    def _resolve_source(self, block: FirBlockId, place: Place) -> int:
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

    def _fill_phi(self, block: FirBlockId, place: Place, phi: int) -> int:
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

    def _place_type(self, block: FirBlockId, place: Place) -> Type:
        # The phi's type is the Place's own type, known from the analyzer fact independently of its operands. A state
        # leaf whose live-in was never carried in the block environment falls back to its reset snapshot's type.
        fact = self._result.block_in[block].facts.get(place)
        if fact is None and isinstance(place, StateLeaf):
            fact = self._leaf_fact(place)
        return self._fact_port_type(fact)

    @staticmethod
    def _sem_of(ty: Type) -> SemType:
        if isinstance(ty, BoolType):
            return SemType.BOOL
        if isinstance(ty, IntType):
            return SemType.INT
        return SemType.FLOAT

    @staticmethod
    def _fact_port_type(fact: Fact | None) -> Type:
        if _is_bool_fact(fact):
            return BoolType()
        if fact == Residual(SemType.INT):
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

    def _entry_value(self, place: Place) -> int:
        if isinstance(place, StateLeaf):
            return self._state_read(place)
        raise AssertionError(f"read of an undefined place '{place}' escaped analysis")

    def _slot_name(self, leaf: StateLeaf) -> str:
        # The slot name is the owning component's canonical member path from the root joined to the leaf attribute by a
        # double underscore, so a top-level attribute ``m`` stays the bare ``m`` (the established port ABI) while a
        # nested child's ``m`` becomes ``child__m``. This is injective except when an attribute name literally spans a
        # ``__`` boundary (a rare dunder-ish name); that alias is a located collision rejection, never a silent merge.
        path = self._provenance.get(id(leaf.component))
        if path is None:
            raise EmissionRejection(
                "a stateful component reached only through an unanchored reference is not supported; "
                "hold it as a direct attribute of the synthesized component"
            )
        name = "__".join(path + leaf.path)
        owner = self._slot_names.setdefault(name, leaf)
        if owner != leaf:
            raise EmissionRejection(f"state slot name collision on '{name}' between distinct component attributes")
        return name

    def _state_read(self, leaf: StateLeaf) -> int:
        if leaf not in self._state_reads:
            reset = self._leaf_reset(leaf)
            slot_type: Type = (
                BoolType()
                if isinstance(reset, BoolConst)
                else IntType() if isinstance(reset, IntConst) else FloatType()
            )
            self._state_reads[leaf] = self._builder.state_read(self._slot_name(leaf), slot_type)
            self._state_order.append(leaf)
        return self._state_reads[leaf]

    def _leaf_is_int(self, leaf: StateLeaf) -> bool:
        """An integer-typed persistent leaf: the analyzer carries a runtime integer across the initiation boundary."""
        return self._analyzer.state_livein().get(leaf) == Residual(SemType.INT)

    def _leaf_reset(self, leaf: StateLeaf) -> FloatConst | BoolConst | IntConst:
        import numpy as np

        current: object = leaf.component
        for attribute in leaf.path:
            current = getattr(current, attribute)
        if isinstance(current, bool) or isinstance(current, np.bool_):
            return BoolConst(bool(current))
        # An integer reset stays integer only when the analyzer typed the leaf as a runtime integer; an int literal
        # seeding a float accumulator resets to 0.0 like any float, matching the leaf's promoted datapath type.
        if isinstance(current, (int, np.integer)) and self._leaf_is_int(leaf):
            return IntConst(int(current))
        if isinstance(current, (int, float, np.integer, np.floating)):
            return FloatConst(_exact_float(current))  # the same exactness guard as a datapath constant
        raise EmissionRejection(
            f"state '{'.'.join(leaf.path)}' has a reset of unsupported type {type(current).__name__}"
        )

    # ---------------------------------------- values and ops ----------------------------------------

    def _const(self, value: StaticValue) -> int:
        concrete = as_python(value)
        if isinstance(concrete, bool):
            return self._builder.bool_const(concrete)
        import numpy as np

        if isinstance(concrete, (int, float, np.integer, np.floating)) and not isinstance(concrete, np.bool_):
            return self._builder.float_const(_exact_float(concrete))
        raise EmissionRejection(f"a {type(concrete).__name__} value cannot materialize in the datapath")

    def _emit_block(self, fir_id: FirBlockId) -> None:
        block = self._result.unit.blocks[fir_id]
        self._builder.position_at(self._fir_to_hir[fir_id])
        env: dict[Place, Fact] = dict(self._result.block_in[fir_id].facts)

        def fact_of(binding: BindingId) -> Fact:
            fact = env.get(Local(binding))
            assert fact is not None, f"binding {binding} missing a fact during emission"
            return fact

        def value_of(binding: BindingId) -> int:
            fact = fact_of(binding)
            if isinstance(fact, Known):
                return self._const(fact.value)
            return self._read(fir_id, Local(binding))  # a residual temp is an SSA-numbered place, cross-block safe

        def arm_value(binding: BindingId) -> int:
            # Like value_of but a Known integer materializes as an IntConst, so a select/phi over it keeps the
            # analyzer's integer type; value_of would float it via _const and mismatch an IntSelect or an integer phi.
            fact = fact_of(binding)
            if isinstance(fact, Known) and isinstance(fact.value, (MetaInt, NpInt)):
                return self._builder.int_const(int(fact.value.value))
            return value_of(binding)

        def as_float(binding: BindingId) -> int:
            # Dispatch on the actual HIR type, not the analyzer fact: a value can be float-carried (a MixedNumeric, or a
            # runtime integer already promoted at a state boundary) while its fact still reads integer.
            vid = value_of(binding)
            ty = self._type_of(vid)
            if isinstance(ty, FloatType):
                return vid
            if isinstance(ty, IntType):
                return self._builder.operation(IntToFloat(), [vid])  # Python promotes int -> float
            raise EmissionRejection("a boolean value reaches a float operation")

        def as_int(binding: BindingId) -> int:
            fact = fact_of(binding)
            if isinstance(fact, Known) and isinstance(fact.value, (MetaInt, NpInt)):
                return self._builder.int_const(int(fact.value.value))  # a Known integer materializes as IntConst
            vid = value_of(binding)
            if isinstance(self._type_of(vid), IntType):
                return vid
            raise EmissionRejection("a non-integer value reaches an integer operation")

        for op in block.ops:
            self._emit_op(fir_id, op, env, fact_of, value_of, arm_value, as_float, as_int)
        match block.terminator:
            case FirJump(target=target):
                if target in self._result.executable_blocks:
                    self._builder.jump(self._fir_to_hir[target])
            case FirBranch(cond=cond, then_target=then_target, else_target=else_target):
                live = [t for t in (then_target, else_target) if (fir_id, t) in self._result.executable_edges]
                if len(live) == 1:
                    self._builder.jump(self._fir_to_hir[live[0]])
                else:
                    self._builder.branch(value_of(cond), self._fir_to_hir[then_target], self._fir_to_hir[else_target])
            case UnitExit():
                pass  # _finish_exit seals the exit block with outputs, slots, and the single Ret
            case other:
                raise AssertionError(f"terminator {type(other).__name__} survived analysis into emission")

    def _emit_op(
        self,
        fir_id: FirBlockId,
        op: Op,
        env: dict[Place, Fact],
        fact_of: "FactOf",
        value_of: "ValueOf",
        arm_value: "ValueOf",
        as_float: "ValueOf",
        as_int: "ValueOf",
    ) -> None:
        def define(dst: BindingId, vid: int) -> None:
            self._write(fir_id, Local(dst), vid)

        match op:
            case LoadConst(dst=dst, value=value):
                env[Local(dst)] = Known(value)
            case LoadPlace(dst=dst, place=place):
                fact = env.get(place)
                if fact is None and isinstance(place, StateLeaf):
                    fact = self._leaf_fact(place)
                assert fact is not None
                env[Local(dst)] = fact
                if not isinstance(fact, Known):
                    define(dst, self._read(fir_id, place))
            case StorePlace(place=place, src=src):
                fact = fact_of(src)
                env[place] = fact
                is_aggregate = isinstance(fact, FactSeq) or (
                    isinstance(fact, Known) and isinstance(fact.value, StaticSeq)
                )
                if isinstance(place, ReturnPlace) and is_aggregate:
                    raise EmissionRejection("aggregate (tuple/list) returns are not emitted yet")  # per-leaf: stage 9
                if isinstance(fact, FactSeq):
                    raise EmissionRejection("a runtime aggregate in a local is not supported yet")  # layout: stage 9
                if isinstance(fact, Known) and isinstance(fact.value, (StaticSeq, ObjectRef)):
                    pass  # a fully-static sequence (subscripts fold) or a non-datapath value: no HIR value flows
                elif isinstance(fact, Known) and isinstance(fact.value, (MetaInt, NpInt)):
                    # a Known integer stored to a local materializes as IntConst so an integer phi over it stays typed;
                    # a downstream float use promotes it on that edge (phi arm coercion / as_float).
                    self._write(fir_id, place, self._builder.int_const(int(fact.value.value)))
                elif isinstance(fact, Known):
                    self._write(fir_id, place, self._const(fact.value))
                else:
                    self._write(fir_id, place, value_of(src))
            case UnbindPlace(place=place):
                env[place] = _unbound()
            case PyBin(dst=dst, op=bin_op, lhs=lhs, rhs=rhs):
                self._emit_binary(fir_id, dst, bin_op, lhs, rhs, env, fact_of, value_of, as_float, as_int, op)
            case PyUn(dst=dst, op=un_op, operand=operand):
                self._emit_unary(fir_id, dst, un_op, operand, env, fact_of, as_float, as_int)
            case PyCompare(dst=dst, op=rel, lhs=lhs, rhs=rhs):
                lhs_fact, rhs_fact = fact_of(lhs), fact_of(rhs)
                folded = self._replay_compare(rel, lhs_fact, rhs_fact)
                env[Local(dst)] = folded
                if not isinstance(folded, Known):
                    lsem, rsem = self._fact_sem(lhs_fact), self._fact_sem(rhs_fact)
                    if lsem is SemType.BOOL or rsem is SemType.BOOL:
                        if lsem is not rsem:
                            raise EmissionRejection("a comparison mixes a boolean and a non-boolean without a cast")
                        define(dst, self._bool_compare(rel, value_of(lhs), value_of(rhs)))  # bool ==/!= is XNOR/XOR
                    elif lsem is SemType.INT and rsem is SemType.INT:
                        define(dst, self._builder.operation(IntRelational(rel), [as_int(lhs), as_int(rhs)]))
                    else:  # both float, or mixed int/float -- promote the integer edge and compare in the float domain
                        define(dst, self._builder.operation(FloatRelational(rel), [as_float(lhs), as_float(rhs)]))
            case PyNot(dst=dst, operand=operand):
                fact = fact_of(operand)
                truth = self._replay_truth(fact)
                if isinstance(truth, Known):
                    concrete = as_python(truth.value)
                    assert isinstance(concrete, bool)
                    env[Local(dst)] = Known(_static_bool(not concrete))
                else:
                    env[Local(dst)] = Residual(SemType.BOOL)
                    define(dst, self._not(self._truth_value(operand, fact, value_of)))
            case PyTruth(dst=dst, operand=operand):
                fact = fact_of(operand)
                truth = self._replay_truth(fact)
                env[Local(dst)] = truth
                if not isinstance(truth, Known):
                    define(dst, self._truth_value(operand, fact, value_of))
            case PySelect(dst=dst, mode=mode, cond=cond, lhs=lhs, rhs=rhs):
                cond_fact = fact_of(cond)
                if isinstance(cond_fact, Known):
                    taken = as_python(cond_fact.value)
                    assert isinstance(taken, bool)
                    chosen = (rhs if taken else lhs) if mode is SelectMode.AND else (lhs if taken else rhs)
                    env[Local(dst)] = fact_of(chosen)
                    if not isinstance(env[Local(dst)], Known):
                        define(dst, value_of(chosen))
                else:
                    rhs_fact = fact_of(rhs)
                    rhs_const = as_python(rhs_fact.value) if isinstance(rhs_fact, Known) else None
                    if (mode is SelectMode.OR and rhs_const is True) or (mode is SelectMode.AND and rhs_const is False):
                        # Boolean identity with a runtime condition: ``A or True`` is always True and ``A and False``
                        # always False. The analyzer folded the result to this constant; emission honors the same fold
                        # so a consumer never builds a select over the bool/float arm pair a residual would leave.
                        env[Local(dst)] = rhs_fact
                    else:
                        then_binding, else_binding = (rhs, lhs) if mode is SelectMode.AND else (lhs, rhs)
                        then_vid = arm_value(then_binding)
                        else_vid = arm_value(else_binding)
                        condition = value_of(cond)
                        then_type, else_type = self._type_of(then_vid), self._type_of(else_vid)
                        operator: Operator
                        if isinstance(then_type, BoolType) and isinstance(else_type, BoolType):
                            operator = BoolSelect()
                        elif isinstance(then_type, IntType) and isinstance(else_type, IntType):
                            operator = IntSelect()  # integer and/or: IntSelect, then the MIR refusal
                        else:  # a mixed int/float select promotes the integer arm to float on its own edge, like a phi
                            if isinstance(then_type, IntType):
                                then_vid = self._builder.operation(IntToFloat(), [then_vid])
                            if isinstance(else_type, IntType):
                                else_vid = self._builder.operation(IntToFloat(), [else_vid])
                            operator = Select()
                        env[Local(dst)] = self._binding_facts()[
                            dst
                        ]  # keep the analyzer's fact (a MixedNumeric stays mixed)
                        define(dst, self._builder.operation(operator, [condition, then_vid, else_vid]))
            case BuildTuple(dst=dst, items=items) | BuildList(dst=dst, items=items):
                from ._analyze import _pack_seq

                is_list = isinstance(op, BuildList)
                facts = tuple(fact_of(item) for item in items)
                packed = _pack_seq(facts, is_list=is_list)
                env[Local(dst)] = packed
                if isinstance(packed, FactSeq):
                    self._layouts[dst] = _SeqHandle(
                        tuple(self._handle_of(item, fact_of, value_of) for item in items), is_list=is_list
                    )
            case PySubscript(dst=dst, obj=obj, index=index):
                self._emit_subscript(fir_id, dst, obj, index, env, fact_of, value_of)
            case PyLen(dst=dst, obj=obj):
                from ._value import admit

                obj_fact = fact_of(obj)
                length = len(self._layouts[obj].items) if obj in self._layouts else len(as_python(obj_fact.value))  # type: ignore[union-attr,arg-type]
                admitted = admit(length)
                assert admitted is not None
                env[Local(dst)] = Known(admitted)
            case PyAttr(dst=dst, obj=obj, name=name):
                self._emit_attr_read(fir_id, dst, obj, name, env, fact_of)
            case PyStoreAttr(obj=obj, name=name, src=src):
                obj_fact = fact_of(obj)
                assert isinstance(obj_fact, Known) and isinstance(obj_fact.value, ObjectRef)
                leaf = StateLeaf(obj_fact.value.obj, (name,))
                src_fact = fact_of(src)
                env[leaf] = src_fact
                if (
                    isinstance(src_fact, Known)
                    and isinstance(src_fact.value, (MetaInt, NpInt))
                    and self._leaf_is_int(leaf)
                ):
                    stored = self._builder.int_const(int(src_fact.value.value))  # a Known int keeps an int-typed leaf
                elif isinstance(src_fact, Known):
                    stored = self._const(src_fact.value)  # a Known int stored to a FLOAT leaf floats via the guard
                elif src_fact == Residual(SemType.INT) and not self._leaf_is_int(leaf):
                    stored = as_float(src)  # a runtime integer stored to a float leaf promotes on its own edge
                else:
                    stored = value_of(src)
                self._write(fir_id, leaf, stored)
                self._state_read(leaf)  # register the slot even if the entry live-in was never read
            case PyCall(dst=dst, callee=callee, args=args):
                fact = self._binding_facts()[dst]
                env[Local(dst)] = fact
                if isinstance(fact, Known):
                    pass  # a concrete builtin fold; the value materializes at its use sites
                elif id(op) in self._analyzer.identity_calls():
                    define(dst, value_of(args[0]))  # float(x) on a float: dst aliases the argument's value
                elif id(op) in self._analyzer.conversion_calls():
                    define(dst, self._emit_conversion(fact_of(args[0]), fact, args[0], value_of, as_float, as_int))
                elif id(op) in self._analyzer.intrinsic_calls():
                    from .._lib import Intrinsic, IntrinsicResultRule, resolve

                    callee_fact = fact_of(callee)
                    assert isinstance(callee_fact, Known) and isinstance(callee_fact.value, ObjectRef)
                    match = resolve(callee_fact.value.obj)
                    assert isinstance(match, Intrinsic)
                    arg_vids = [value_of(arg) for arg in args]
                    first_is_int = isinstance(self._type_of(arg_vids[0]), IntType)
                    all_int = all(isinstance(self._type_of(vid), IntType) for vid in arg_vids)
                    rule = match.result_rule
                    if rule is IntrinsicResultRule.ALWAYS_INT and first_is_int:
                        result = arg_vids[0]  # rounding an integer is the integer (identity)
                    elif rule is IntrinsicResultRule.ALWAYS_INT:
                        rounded = self._builder.operation(match.operator, [as_float(arg) for arg in args])
                        result = self._builder.operation(FloatToInt(), [rounded])
                    elif rule is IntrinsicResultRule.PRESERVE and first_is_int:
                        result = self._integer_intrinsic(match.integer_implementation, arg_vids)
                    elif rule in (IntrinsicResultRule.NUMPY_PROMOTE, IntrinsicResultRule.SELECT) and all_int:
                        result = self._integer_intrinsic(match.integer_implementation, arg_vids)
                    else:  # SIGNATURE, or a float/mixed operand: promote every operand and run the float operator
                        result = self._builder.operation(match.operator, [as_float(arg) for arg in args])
                    define(dst, result)
                else:
                    raise AssertionError(f"call {dst} left an unexpected residual value in the graph")
            case _:
                raise AssertionError(f"operation {type(op).__name__} survived analysis into emission")

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

    def _handle_of(self, binding: BindingId, fact_of: "FactOf", value_of: "ValueOf") -> _Handle:
        # A per-element handle for a FactSeq: nested layout, a Known scalar (materialize on use), or an HIR value.
        if binding in self._layouts:
            return self._layouts[binding]
        fact = fact_of(binding)
        if isinstance(fact, Known):
            return _KnownHandle(fact.value)
        return _ValueHandle(value_of(binding))

    def _bind_handle(self, fir_id: FirBlockId, dst: BindingId, handle: _Handle, env: dict[Place, Fact]) -> None:
        env[Local(dst)] = self._handle_fact(handle)
        match handle:
            case _ValueHandle(vid=vid):
                self._write(fir_id, Local(dst), vid)
            case _SeqHandle():
                self._layouts[dst] = handle
            case _KnownHandle():
                pass

    def _handle_fact(self, handle: _Handle) -> Fact:
        match handle:
            case _KnownHandle(value=value):
                return Known(value)
            case _ValueHandle(vid=vid):
                return Residual(SemType.BOOL if isinstance(self._type_of(vid), BoolType) else SemType.FLOAT)
            case _SeqHandle(items=items, is_list=is_list):
                return FactSeq(tuple(self._handle_fact(item) for item in items), is_list=is_list)

    def _emit_subscript(
        self,
        fir_id: FirBlockId,
        dst: BindingId,
        obj: BindingId,
        index: BindingId,
        env: dict[Place, Fact],
        fact_of: "FactOf",
        value_of: "ValueOf",
    ) -> None:
        import operator

        from ._value import admit

        index_fact = fact_of(index)
        assert isinstance(index_fact, Known)
        concrete_index = as_python(index_fact.value)
        if obj in self._layouts:
            position = operator.index(concrete_index)  # type: ignore[arg-type]  # a single int index (np.int64 too)
            self._bind_handle(fir_id, dst, self._layouts[obj].items[position], env)
            return
        obj_fact = fact_of(obj)
        assert isinstance(obj_fact, Known)
        # A Known aggregate accepts any static index Python does -- an int, np.int64, or a multi-dim tuple index
        # (TABLE[1, 0]) -- because the whole indexing folds concretely; only the RESULT must admit into the domain.
        element = as_python(obj_fact.value)[concrete_index]  # type: ignore[index]
        admitted = admit(element)
        if admitted is None:
            raise EmissionRejection("a subscript yields a value outside the datapath domain")
        env[Local(dst)] = Known(admitted)

    def _emit_attr_read(
        self,
        fir_id: FirBlockId,
        dst: BindingId,
        obj: BindingId,
        name: str,
        env: dict[Place, Fact],
        fact_of: "FactOf",
    ) -> None:
        # The analyzer already classified this attribute: a bound method or namespace member is Known(ObjectRef)
        # (no datapath value), a static attribute is a Known scalar, and only a runtime component attribute is
        # residual and backed by a state slot. Deferring to that fact keeps the emitter from re-deriving it and
        # mistaking a helper method for state.
        fact = self._binding_facts()[dst]
        env[Local(dst)] = fact
        if isinstance(fact, Known):
            return
        obj_fact = fact_of(obj)
        assert isinstance(obj_fact, Known) and isinstance(obj_fact.value, ObjectRef)
        self._write(fir_id, Local(dst), self._read(fir_id, StateLeaf(obj_fact.value.obj, (name,))))

    def _binding_facts(self) -> dict[BindingId, Fact]:
        if self._binding_fact_cache is None:
            self._binding_fact_cache = self._analyzer.binding_facts(self._result)
        return self._binding_fact_cache

    def _leaf_fact(self, leaf: StateLeaf) -> Fact:
        livein = self._analyzer.state_livein()
        if leaf in livein:
            return livein[leaf]
        reset = self._leaf_reset(leaf)
        if isinstance(reset, BoolConst):
            return Known(_static_bool(reset.value))
        from ._value import admit

        admitted = admit(reset.value)
        assert admitted is not None
        return Known(admitted)

    def _emit_binary(
        self,
        fir_id: FirBlockId,
        dst: BindingId,
        bin_op: BinOp,
        lhs: BindingId,
        rhs: BindingId,
        env: dict[Place, Fact],
        fact_of: "FactOf",
        value_of: "ValueOf",
        as_float: "ValueOf",
        as_int: "ValueOf",
        op: Op,
    ) -> None:
        from ._opsem import static_binop

        result_fact = self._binding_facts()[dst]
        if isinstance(result_fact, Known) and isinstance(result_fact.value, (StaticSeq, ObjectRef)):
            env[Local(dst)] = result_fact  # a fully-static sequence concat/repeat (the comprehension accumulator):
            return  # it flows as a Known fact whose eventual subscripts fold, exactly like a static list literal
        if isinstance(result_fact, FactSeq):
            raise EmissionRejection("a runtime aggregate operation is not supported yet")  # layout: stage 9
        lhs_fact, rhs_fact = fact_of(lhs), fact_of(rhs)
        if isinstance(lhs_fact, Known) and isinstance(rhs_fact, Known):
            folded = static_binop(bin_op, lhs_fact.value, rhs_fact.value)
            if folded is not None:
                env[Local(dst)] = Known(folded)
                return
            if isinstance(result_fact, Known):
                # A Known result static_binop cannot produce -- ``&``/``|``/``^`` on two booleans, which the analyzer
                # folds. Replay that fold so it materializes at its use site rather than reaching a float operation.
                env[Local(dst)] = result_fact
                return
        if result_fact == Residual(SemType.INT):  # a runtime integer result: stay in the integer datapath
            env[Local(dst)] = result_fact
            self._write(fir_id, Local(dst), self._emit_int_binary(bin_op, as_int(lhs), as_int(rhs)))
            return
        if result_fact == Residual(SemType.BOOL):  # &/|/^ on two booleans lowers to the boolean-bank operators
            env[Local(dst)] = result_fact
            operands = [value_of(lhs), value_of(rhs)]
            match bin_op:
                case BinOp.BITAND:
                    result = self._builder.operation(BoolAnd(), operands)
                case BinOp.BITOR:
                    result = self._builder.operation(BoolOr(), operands)
                case _:
                    result = self._builder.operation(BoolXor(), operands)
            self._write(fir_id, Local(dst), result)
            return
        env[Local(dst)] = Residual(SemType.FLOAT)
        if bin_op is BinOp.POW:
            self._write(fir_id, Local(dst), self._emit_power(lhs, rhs, fact_of, as_float, op))
            return
        left, right = as_float(lhs), as_float(rhs)
        match bin_op:
            case BinOp.ADD:
                result = self._builder.operation(FloatAdd(), [left, right])
            case BinOp.SUB:
                result = self._builder.operation(FloatAdd(), [left, self._builder.operation(FloatNeg(), [right])])
            case BinOp.MUL:
                result = self._builder.operation(FloatMul(), [left, right])
            case BinOp.DIV:
                result = self._builder.operation(FloatDiv(), [left, right])
            case _:
                raise EmissionRejection(f"operator {bin_op.value} is not lowerable yet")
        self._write(fir_id, Local(dst), result)

    def _emit_conversion(
        self,
        src_fact: Fact,
        dst_fact: Fact,
        arg: BindingId,
        value_of: "ValueOf",
        as_float: "ValueOf",
        as_int: "ValueOf",
    ) -> int:
        # A runtime scalar cast between kinds. int<->float is truncation toward zero / exact promotion; bool casts are a
        # truthiness test (to bool) or a 0/1 widening (from bool). A same-kind cast is handled as an identity upstream.
        src, dst = self._fact_sem(src_fact), self._fact_sem(dst_fact)
        match (src, dst):
            case (SemType.INT, SemType.FLOAT):
                return self._builder.operation(IntToFloat(), [as_int(arg)])
            case (SemType.FLOAT, SemType.INT):
                return self._builder.operation(FloatToInt(), [as_float(arg)])
            case (SemType.BOOL, SemType.FLOAT):
                return self._builder.operation(BoolToFloat(), [value_of(arg)])
            case (SemType.FLOAT, SemType.BOOL):
                return self._builder.operation(FloatToBool(), [as_float(arg)])
            case (SemType.BOOL, SemType.INT):
                return self._builder.operation(BoolToInt(), [value_of(arg)])
            case (SemType.INT, SemType.BOOL):
                return self._builder.operation(IntToBool(), [as_int(arg)])
            case _:
                raise EmissionRejection(f"conversion from {src.value} to {dst.value} is not supported")

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
                raise EmissionRejection(f"integer operator {bin_op.value!r} is not lowerable yet")

    def _emit_power(self, base: BindingId, exponent: BindingId, fact_of: "FactOf", as_float: "ValueOf", op: Op) -> int:
        # A runtime base raised to a COMPILE-TIME integer exponent expands to a multiply chain (x**3 -> x*x*x), bounded
        # like the loop unroller so ``x ** 10**9`` refuses instead of hanging; a negative exponent is the reciprocal of
        # the chain, and x**0 is 1.0.
        from ._value import MetaInt, NpInt

        exp_fact = fact_of(exponent)
        if isinstance(exp_fact, Known) and isinstance(exp_fact.value, (MetaInt, NpInt)):
            power = int(exp_fact.value.value)
            if abs(power) > _MAX_POWER_CHAIN:
                raise EmissionRejection(f"a compile-time power exponent of {power} is too large to expand")
            if self._fact_sem(fact_of(base)) is SemType.INT:
                # An integer base raised to a power is exact integer exponentiation in Python (int(x)**0 is the integer
                # 1, not 1.0); a float chain would round it and int(x)**0 would silently float. No integer-power operator
                # yet, so it is a located rejection -- checked before the power==0 shortcut so int(x)**0 refuses too.
                raise EmissionRejection("an integer raised to a compile-time power is not yet lowerable")
            if power == 0:
                return self._builder.float_const(1.0)
            source = as_float(base)
            chain = source
            for _ in range(abs(power) - 1):
                chain = self._builder.operation(FloatMul(), [chain, source])
            if power < 0:
                return self._builder.operation(FloatDiv(), [self._builder.float_const(1.0), chain])
            return chain
        # A runtime exponent: only base two lowers, and only when the result is float. ``2.0**i``, ``2**xf`` and
        # ``2.0**xf`` are float (-> exp2, the integer exponent promoting through as_float); ``2**i`` is int-or-float by
        # the sign of i (no integer-power operator) and every non-two base grows unbounded, so both refuse.
        base_fact = fact_of(base)
        if isinstance(base_fact, Known) and isinstance(base_fact.value, (MetaInt, NpInt, StaticFloat, NpFloat)):
            base_is_float = isinstance(base_fact.value, (StaticFloat, NpFloat))
            exp_sem = self._fact_sem(exp_fact)
            if as_python(base_fact.value) == 2 and (
                exp_sem is SemType.FLOAT or (base_is_float and exp_sem is SemType.INT)
            ):
                return self._builder.operation(FloatExp2(), [as_float(exponent)])
        raise EmissionRejection("a power with a runtime exponent is not supported")

    def _emit_unary(
        self,
        fir_id: FirBlockId,
        dst: BindingId,
        un_op: UnOp,
        operand: BindingId,
        env: dict[Place, Fact],
        fact_of: "FactOf",
        as_float: "ValueOf",
        as_int: "ValueOf",
    ) -> None:
        from ._opsem import static_unop

        fact = fact_of(operand)
        if isinstance(fact, Known):
            folded = static_unop(un_op, fact.value)
            if folded is not None:
                env[Local(dst)] = Known(folded)
                return
        result_fact = self._binding_facts()[dst]
        if result_fact == Residual(SemType.INT):  # runtime integer negation stays in the integer datapath
            env[Local(dst)] = result_fact
            source = as_int(operand)
            self._write(
                fir_id, Local(dst), source if un_op is UnOp.POS else self._builder.operation(IntNeg(), [source])
            )
            return
        env[Local(dst)] = Residual(SemType.FLOAT)
        source = as_float(operand)
        self._write(fir_id, Local(dst), source if un_op is UnOp.POS else self._builder.operation(FloatNeg(), [source]))

    def _replay_compare(self, rel: RelationalOp, lhs: Fact, rhs: Fact) -> Fact:
        from ._opsem import static_compare

        if isinstance(lhs, Known) and isinstance(rhs, Known):
            folded = static_compare(rel, lhs.value, rhs.value)
            if folded is not None:
                return Known(folded)
        return Residual(SemType.BOOL)

    def _replay_truth(self, fact: Fact) -> Fact:
        from ._opsem import static_truth

        if isinstance(fact, Known):
            truth = static_truth(fact.value)
            if truth is not None:
                return Known(_static_bool(truth))
        return Residual(SemType.BOOL)

    def _truth_value(self, operand: BindingId, fact: Fact, value_of: "ValueOf") -> int:
        vid = value_of(operand)
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
            raise EmissionRejection("only == and != are defined between boolean values")
        xor = self._builder.operation(BoolXor(), [left, right])
        return xor if rel is RelationalOp.NE else self._builder.operation(BoolNot(), [xor])

    # ---------------------------------------- exit ----------------------------------------

    def _finish_exit(self) -> None:
        unit = self._result.unit
        exit_env = self._result.block_in[unit.exit]
        self._builder.position_at(self._fir_to_hir[unit.exit])
        return_fact = exit_env.facts.get(ReturnPlace())
        if isinstance(return_fact, FactSeq) or (
            isinstance(return_fact, Known) and isinstance(return_fact.value, StaticSeq)
        ):
            raise EmissionRejection("aggregate (tuple/list) returns are not emitted yet")  # per-leaf return: stage 9
        return_vid: int | None = None
        if return_fact is not None and not (
            isinstance(return_fact, Known) and isinstance(return_fact.value, ObjectRef)
        ):
            if (
                isinstance(return_fact, Known)
                and isinstance(return_fact.value, (MetaInt, NpInt))
                and unit.declared_return is SemType.INT
            ):
                return_vid = self._builder.int_const(int(return_fact.value.value))  # a Known integer -> integer port
            elif isinstance(return_fact, Known):
                return_vid = self._const(return_fact.value)
            else:
                return_vid = self._read(unit.exit, ReturnPlace())
        if (
            unit.declared_return is SemType.FLOAT
            and return_vid is not None
            and isinstance(self._type_of(return_vid), IntType)
        ):
            return_vid = self._builder.operation(
                IntToFloat(), [return_vid]
            )  # Python returns an int where float declared
        if unit.declared_return is not None:  # a scalar return type was declared: the value must match it
            if return_vid is None:
                raise EmissionRejection(f"return type mismatch: declared {unit.declared_return.value}, returns nothing")
            got = self._sem_of(self._type_of(return_vid))
            if got is not unit.declared_return:
                raise EmissionRejection(
                    f"return type mismatch: declared {unit.declared_return.value}, returns {got.value}"
                )
        promoted = self._analyzer.runtime_state()
        # Slots and public ports emit in first-STORE source order (matching production), not the order the RPO walk
        # happened to touch attributes; a leaf touched only by a read still trails a leaf stored earlier.
        store_order = [leaf for leaf in self._analyzer.state_store_order(self._result) if leaf in promoted]
        store_order += [leaf for leaf in self._state_order if leaf in promoted and leaf not in store_order]
        public_live_outs: set[int] = set()
        for leaf in store_order:
            live_out = self._read(unit.exit, leaf)
            self._builder.state_slot(self._slot_name(leaf), self._leaf_reset(leaf), live_out)
            if not leaf.path[-1].startswith("_"):
                public_live_outs.add(live_out)
        if return_vid is not None and return_vid not in public_live_outs:
            self._builder.output(port_name([0]), return_vid)  # a scalar return is leaf 0 of the bundle
        for leaf in store_order:
            if not leaf.path[-1].startswith("_"):
                self._builder.output(state_port_name(self._slot_name(leaf)), self._read(unit.exit, leaf))
        self._builder.ret()


def _unbound() -> Fact:
    from ._analyze import _UNBOUND

    return _UNBOUND


def _static_bool(value: bool) -> StaticValue:
    from ._value import StaticBool

    return StaticBool(value)
