"""Lower optimized HIR to selected MIR."""

import logging
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from .._errors import UnsupportedConstruct
from .._hir import (
    copy_node,
    eliminate_dead_code,
    rebuild,
    references,
    BoolAnd,
    BoolConst,
    BoolNot,
    BoolOr,
    BoolSelect,
    BoolToFloat,
    BoolToInt,
    BoolType as HirBoolType,
    BoolXor,
    Branch,
    Const,
    FloatAbs,
    FloatAdd,
    FloatAtan2Turns,
    FloatCeil,
    FloatComparison,
    FloatConst,
    FloatCosTurns,
    FloatDiv,
    FloatEqual,
    FloatExp2,
    FloatFloor,
    FloatFma,
    HirBuilder,
    FloatGreater,
    FloatGreaterOrEqual,
    FloatILog2,
    FloatIsFinite,
    FloatIsInf,
    FloatIsNegInf,
    FloatIsPosInf,
    FloatLess,
    FloatLessOrEqual,
    FloatLog2,
    FloatMax,
    FloatMin,
    FloatMul,
    FloatMulPow2,
    FloatMulPow2Dynamic,
    FloatNeg,
    FloatNotEqual,
    FloatRound,
    FloatSelect,
    FloatSinTurns,
    FloatSqrt,
    FloatToBool,
    FloatToInt,
    FloatTrunc,
    FloatType as HirFloatType,
    Hir,
    InPort,
    IntAbs,
    IntAdd,
    IntBwAnd,
    IntBwNot,
    IntBwOr,
    IntBwXor,
    IntComparison,
    IntConst,
    IntDivFloor,
    IntEqual,
    IntGreater,
    IntGreaterOrEqual,
    IntLess,
    IntLessOrEqual,
    IntMod,
    IntMul,
    IntMulPow2,
    IntNeg,
    IntNotEqual,
    IntPopcount,
    IntSelect,
    IntShiftLeft,
    IntShiftRight,
    IntSub,
    IntToBool,
    IntToFloat,
    IntType as HirIntType,
    Jump,
    Node,
    Operation,
    Operator,
    Phi,
    Ret,
    Scaling,
    StateRead,
    StateSlot,
    Terminator,
    optimize,
    reverse_postorder,
)
from ._refuse import refuse
from ._specialize import constant_shift_count, left_shifts, specialize
from ._rescale import rescale
from ._trig_abi import trig_abi
from .._util import ValueId
from .._operators import (
    BoolAndOperator,
    BoolInversion,
    BoolOrOperator,
    BoolToFloatOperator,
    BoolToIntOperator,
    BoolXorOperator,
    FloatClassificationOperator,
    FloatHardwareOperator,
    FloatIsFiniteOperator,
    FloatIsNegInfOperator,
    FloatIsPosInfOperator,
    IAbsOperator,
    IAddOperator,
    ICmpOperator,
    IDivOperator,
    IPopcntOperator,
    IShlOperator,
    IShrOperator,
    ISubOperator,
    IntBwAndOperator,
    IntBwNotOperator,
    IntBwOrOperator,
    IntBwXorOperator,
    IntIdentity,
    IntShiftConstOperator,
    IntToBoolOperator,
    FloatSignControl,
    FloatToBoolOperator,
    HardwareOperator,
    OpConfig,
    PortConditioner,
    PooledHardwareOperator,
    Relation,
    RoundMode,
    SelectOperator,
    require,
)
from .._type import (
    BoolType as ScalarBoolType,
    FloatFormat,
    FloatType as ScalarFloatType,
    IntFormat,
    IntType as ScalarIntType,
    ScalarType,
)
from ._ir import Mir, MirBuilder, MirOperation
from ._hypot import count as hypot_count, expand_unfused, plan_fusions
from ._options import MirOptions
from ._signs import collapse_signs, sign_chain, sign_of

_logger = logging.getLogger(__name__)

# The seam between the semantic relation and the comparator flag it taps; the two vocabularies meet only here.
# Both comparator families share it: an integer comparison taps `icmp` exactly as a float one taps `fcmp`.
_RELATION_OF: dict[type[Operator], Relation] = {
    FloatLess: Relation.LT,
    FloatLessOrEqual: Relation.LE,
    FloatEqual: Relation.EQ,
    FloatNotEqual: Relation.NE,
    FloatGreaterOrEqual: Relation.GE,
    FloatGreater: Relation.GT,
    IntLess: Relation.LT,
    IntLessOrEqual: Relation.LE,
    IntEqual: Relation.EQ,
    IntNotEqual: Relation.NE,
    IntGreaterOrEqual: Relation.GE,
    IntGreater: Relation.GT,
}


# Shared by the standalone `fround` and the `ftoint` that absorbs one: the same field on both.
_ROUND_MODE_OF: dict[type[Operator], RoundMode] = {
    FloatRound: RoundMode.NEAREST_EVEN,
    FloatFloor: RoundMode.FLOOR,
    FloatCeil: RoundMode.CEIL,
    FloatTrunc: RoundMode.TRUNC,
}


@dataclass(frozen=True, slots=True)
class _Read:
    """`site` is None outside any operation."""

    site: ValueId | None
    absorbing: bool


def _absorbable(reads: Iterable[_Read]) -> bool:
    """Every read of the product is an absorbing port of a distinct addition, so no add rounds it twice."""
    reads = list(reads)
    return bool(reads) and all(read.absorbing for read in reads) and len({read.site for read in reads}) == len(reads)


def _exact_scale(fmt: FloatFormat, k: int) -> float | None:
    """
    `2**k` only where this format holds it exactly: the host rails past k = 1023 while composed exponents are
    unbounded, and a format without subnormals rounds a small power onto a neighbour it is not.
    """
    value = Scaling(1.0, k, False).magnitude()
    return value if value is not None and fmt.round(value) == value else None


def _collapse_bool_inversions(nodes: dict[ValueId, Node], vid: ValueId) -> tuple[ValueId, BoolInversion]:
    """
    Peel a chain of semantic NOT operations, returning the base value and the combined inversion -- the boolean dual
    of collapse_signs. Folding happens on the CONSUMER side only: a NOT over a comparison must never flip
    the producer's tap conditioner (two taps of one comparator port with different inversions cannot fuse and would
    serialize two firings), and consumer-side folding keeps one shared producer for both polarities of a value.
    """
    invert = False
    node = nodes[vid]
    while isinstance(node, Operation) and isinstance(node.operator, BoolNot):
        invert = not invert
        (vid,) = node.operands
        node = nodes[vid]
    return vid, BoolInversion(invert=invert)


def _collapse_conditioner(nodes: dict[ValueId, Node], vid: ValueId) -> tuple[ValueId, PortConditioner]:
    """
    Collapse the type's own sideband chain: sign operations over a float value, NOTs over a boolean one. An integer
    has no free sideband, so `ineg`/`iabs` are hardware and nothing collapses.
    """
    ty = nodes[vid].type
    match ty:
        case HirBoolType():
            return _collapse_bool_inversions(nodes, vid)
        case HirFloatType():
            return collapse_signs(nodes, vid)
        case HirIntType():
            return vid, IntIdentity()
        case _:
            raise UnsupportedConstruct(f"no conditioner-collapse rule for HIR type {ty!r}")


@dataclass(frozen=True, slots=True)
class _ValueFmaPlan:
    """
    A contraction of `a*b + c` into one `ffma`: `product` is the FloatMul whose standalone MIR op it suppresses,
    `product_sign` the sign peeled off the product operand.
    """

    product: ValueId
    a: ValueId
    b: ValueId
    c: ValueId
    product_sign: FloatSignControl


@dataclass(frozen=True, slots=True)
class _ScaleFmaPlan:
    """
    The same over an exponent scaling, `a*2**k + c`. The scaler is exact, so the fma may only stand in for it where
    the format holds `2**k` exactly -- checked at planning, since the plan materializes what the scaler never did.
    """

    product: ValueId
    a: ValueId
    scale: float
    c: ValueId
    product_sign: FloatSignControl


type _FmaPlan = _ValueFmaPlan | _ScaleFmaPlan


@dataclass(frozen=True, slots=True)
class _DirectionalInfPlan:
    """A planned contraction of `isinf(x) and x` sign tests into one directional infinity predicate."""

    operand: ValueId
    sign: FloatSignControl
    semantic: FloatIsPosInf | FloatIsNegInf
    members: frozenset[ValueId]


def _is_zero_float(hir: Hir, vid: ValueId) -> bool:
    base, _ = collapse_signs(hir.nodes, vid)
    node = hir.nodes[base]
    return isinstance(node, FloatConst) and node.value == 0.0


def _match_isinf_operand(hir: Hir, vid: ValueId) -> ValueId | None:
    match hir.nodes[vid]:
        case Operation(operator=FloatIsInf(), operands=(operand,)):
            base, _ = collapse_signs(hir.nodes, operand)
            return base
        case _:
            return None


def _match_zero_sign_relation(
    hir: Hir, vid: ValueId
) -> tuple[ValueId, FloatSignControl, FloatIsPosInf | FloatIsNegInf] | None:
    """Recognize zero-sided sign tests that distinguish positive from negative infinity."""
    match hir.nodes[vid]:
        case Operation(operator=FloatGreater() | FloatGreaterOrEqual(), operands=(left, right)):
            greater = True
        case Operation(operator=FloatLess() | FloatLessOrEqual(), operands=(left, right)):
            greater = False
        case _:  # equality tests say nothing about the side
            return None
    left_zero = _is_zero_float(hir, left)
    right_zero = _is_zero_float(hir, right)
    if left_zero == right_zero:
        return None
    operand, sign = collapse_signs(hir.nodes, left if right_zero else right)
    if sign.absolute:
        return None
    # `x > 0` and `0 < x` both test the positive side; either mirroring alone flips it.
    return operand, sign, FloatIsPosInf() if greater == right_zero else FloatIsNegInf()


def _directional_inf_plan(hir: Hir, isinf_id: ValueId, relation_id: ValueId) -> _DirectionalInfPlan | None:
    isinf_operand = _match_isinf_operand(hir, isinf_id)
    relation = _match_zero_sign_relation(hir, relation_id)
    if isinf_operand is None or relation is None:
        return None
    operand, sign, semantic = relation
    if operand != isinf_operand:
        return None
    return _DirectionalInfPlan(operand, sign, semantic, frozenset((isinf_id, relation_id)))


def _plan_directional_inf_fusions(hir: Hir, use_counts: dict[ValueId, int]) -> dict[ValueId, _DirectionalInfPlan]:
    candidates: dict[ValueId, _DirectionalInfPlan] = {}
    consumers_by_member: dict[ValueId, set[ValueId]] = {}
    for vid, node in hir.nodes.items():
        if not (isinstance(node, Operation) and isinstance(node.operator, BoolAnd)):
            continue
        a, b = node.operands
        plan = _directional_inf_plan(hir, a, b)
        if plan is None:
            plan = _directional_inf_plan(hir, b, a)
        if plan is not None:
            candidates[vid] = plan
            for member in plan.members:
                consumers_by_member.setdefault(member, set()).add(vid)
    plans: dict[ValueId, _DirectionalInfPlan] = {}
    for vid, plan in candidates.items():
        fused_members = frozenset(
            member for member in plan.members if use_counts[member] == len(consumers_by_member[member])
        )
        if fused_members:
            plans[vid] = _DirectionalInfPlan(plan.operand, plan.sign, plan.semantic, fused_members)
    return plans


def _fma_plan(hir: Hir, fmt: FloatFormat, product: ValueId, sign: FloatSignControl, addend: ValueId) -> _FmaPlan | None:
    node = hir.nodes[product]
    if not isinstance(node, Operation):
        return None
    match node.operator:
        case FloatMul():
            a, b = node.operands
            return _ValueFmaPlan(product=product, a=a, b=b, c=addend, product_sign=sign)
        case FloatMulPow2(k=k) if (scale := _exact_scale(fmt, k)) is not None:
            (a,) = node.operands
            return _ScaleFmaPlan(product=product, a=a, scale=scale, c=addend, product_sign=sign)
        case _:
            return None


def _contract_fmas(hir: Hir, ops: OpConfig) -> Hir:
    """
    Rewrite every contractible `a*b + c` into the semantic `FloatFma` the kernel could have written itself, so one
    lowering serves both spellings. The product sign rides HIR sign operations, which selection folds onto the
    operand conditioners as it does for any other operand.

    Runs after judgement: the constant an exponent scaling materializes is the machine's own, not the program's.
    """
    plans = _plan_fma_fusions(hir, ops)
    if not plans:
        return hir
    # Operations intern per block, so a block and a node name one value -- which is what `BuildValue`, handed the
    # node and not its id, needs to find the plan.
    planned = {
        (block.id, hir.nodes[vid]): plans[vid] for block in hir.blocks for vid in block.operations if vid in plans
    }

    def build_value(builder: HirBuilder, node: Node, remap: dict[ValueId, ValueId]) -> ValueId:
        plan = planned.get((builder.current_block, node))
        if plan is None:
            return copy_node(builder, node, remap)
        signed = _signed(builder, remap[plan.a], plan.product_sign)
        match plan:
            case _ValueFmaPlan(b=b):
                # `|a*b|` is `|a||b|`, so an absolute product signs both multipliers; a negation signs only one.
                other = _signed(builder, remap[b], FloatSignControl(absolute=plan.product_sign.absolute))
            case _ScaleFmaPlan(scale=scale):
                other = builder.const_node(FloatConst(scale))
        return builder.operation(FloatFma(), [signed, other, remap[plan.c]])

    _logger.info("FMA contraction: %d addition(s) contracted with their product", len(plans))
    return eliminate_dead_code(rebuild(hir, build_value))


def _signed(builder: HirBuilder, value: ValueId, sign: FloatSignControl) -> ValueId:
    if sign.absolute:
        value = builder.operation(FloatAbs(), [value])
    return builder.operation(FloatNeg(), [value]) if sign.negate else value


def _readers(hir: Hir) -> dict[ValueId, list[ValueId | None]]:
    readers: dict[ValueId, list[ValueId | None]] = defaultdict(list)
    for vid, node in hir.nodes.items():
        for referenced in references(node):
            readers[referenced].append(vid)
    for referenced in hir.external_value_references():
        readers[referenced].append(None)
    return readers


def _plan_fma_fusions(hir: Hir, ops: OpConfig) -> dict[ValueId, _FmaPlan]:
    """
    A product is carried by every add that names it or by none, since one add would single-round a product observed
    elsewhere.
    """
    if ops.ffma is None:
        return {}
    fmt = ops.float_format
    readers = _readers(hir)
    position = {vid: index for index, vid in enumerate(vid for block in hir.blocks for vid in block.operations)}
    sites: dict[ValueId, list[tuple[ValueId, _FmaPlan]]] = defaultdict(list)
    signs: dict[ValueId, set[ValueId]] = defaultdict(set)
    for vid, node in hir.nodes.items():
        if not (isinstance(node, Operation) and isinstance(node.operator, FloatAdd)):
            continue
        op0, op1 = node.operands
        for product_operand, addend in ((op0, op1), (op1, op0)):
            base, control, chain = sign_chain(hir.nodes, product_operand)
            plan = _fma_plan(hir, fmt, base, control, addend)
            if plan is not None:
                sites[plan.product].append((vid, plan))
                signs[plan.product].update(chain)

    plans: dict[ValueId, _FmaPlan] = {}
    # Fewest adds first, so an exclusive product -- which nothing else can want -- never loses its add to a shared
    # one that then fails to be claimed whole.
    for product in sorted(sites, key=lambda product: (len(sites[product]), position[product])):
        adds = {add for add, _ in sites[product]}
        cone = {product, *signs[product]}
        reads = [_Read(site, site in adds) for member in cone for site in readers[member] if site not in cone]
        if _absorbable(reads) and not any(add in plans for add in adds):
            plans.update(sites[product])
    return plans


def _wholly_taken(sites: Counter[ValueId], use_counts: dict[ValueId, int]) -> set[ValueId]:
    """Values the counted sites consume entirely, leaving no reader; a lowering may then emit nothing for them."""
    return {vid for vid, taken in sites.items() if taken == use_counts[vid]}


def _plan_folded_shift_counts(hir: Hir, use_counts: dict[ValueId, int]) -> set[ValueId]:
    """
    Decided before any block, because constants are lowered entry-globally. A count the shift below answers from is
    consumed as an immediate, not held as a value, exactly as an absorbed rounding is -- and a count something ELSE
    reads is an ordinary value, which is where one too wide for the machine is still refused.
    """
    folds: Counter[ValueId] = Counter()
    for node in hir.nodes.values():
        if isinstance(node, Operation) and isinstance(node.operator, (IntShiftLeft, IntShiftRight)):
            _, count = node.operands
            if constant_shift_count(hir, count) is not None:
                folds[count] += 1
    return _wholly_taken(folds, use_counts)


def _absorbed_rounding(hir: Hir, operand: ValueId) -> tuple[RoundMode, ValueId] | None:
    """
    The mode a `FloatToInt` carries instead of reading a rounding's result, with the value it rounds. Only a
    rounding read DIRECTLY: anything between conditions the ROUNDED value, and `-floor(x)` is not `floor(-x)`.
    """
    node = hir.nodes[operand]
    if not isinstance(node, Operation) or type(node.operator) not in _ROUND_MODE_OF:
        return None
    (rounded,) = node.operands
    return _ROUND_MODE_OF[type(node.operator)], rounded


def _plan_absorbed_roundings(hir: Hir, use_counts: dict[ValueId, int]) -> set[ValueId]:
    """
    Roundings with no reader left once the conversions absorb them. The absorption itself is unconditional: it
    shortens the dependency either way. Where a reader survives, both operators are emitted and each rounds the same
    value -- the fastmath charter licenses that, where the ffma contraction declines it only because contracting a
    shared product buys nothing to pay the divergence with.
    """
    absorptions: Counter[ValueId] = Counter()
    for vid, node in hir.nodes.items():
        if isinstance(node, Operation) and isinstance(node.operator, FloatToInt):
            (operand,) = node.operands
            if _absorbed_rounding(hir, operand) is not None:
                absorptions[operand] += 1
    return _wholly_taken(absorptions, use_counts)


class _LoweringContext:
    def __init__(self, hir: Hir, ops: OpConfig) -> None:
        self.hir = hir
        self.ops = ops
        self.float_format = ops.float_format
        self.int_format = ops.int_format
        self.builder = MirBuilder(self.float_format, self.int_format)
        self.remap: dict[ValueId, ValueId] = {}
        # Every plan below reads the same whole-DAG reference count.
        use_counts = hir.use_counts()
        self.directional_inf_plans = _plan_directional_inf_fusions(hir, use_counts)
        self.fused_directional_inf_members = {
            member for plan in self.directional_inf_plans.values() for member in plan.members
        }
        self.fused_hypots = plan_fusions(hir, ops)
        # The derivation expanded every unfused hypotenuse, so the survivors all ride an atan2's magnitude port.
        assert hypot_count(hir) == len(self.fused_hypots)
        self.folded_shift_counts = _plan_folded_shift_counts(hir, use_counts)
        self.absorbed_roundings = _plan_absorbed_roundings(hir, use_counts)
        self.float_lowerer = _FloatLowerer(self)
        self.int_lowerer = _IntLowerer(self)

    def run(self) -> Mir:
        for _ in self.hir.blocks:
            self.builder.block()  # preserve block ids 0..n-1
        # Entry-global pure values first: inputs in signature order, then constants and state reads.
        self.builder.position_at(self.hir.entry)
        for vid in self.hir.input_ids:
            self._lower_node(vid, self.hir.nodes[vid])
        for vid in sorted(self.hir.nodes):
            if isinstance(self.hir.nodes[vid], (Const, StateRead)):
                self._lower_node(vid, self.hir.nodes[vid])
        # Then each block's phis and operations in reverse-postorder (predecessors first), then its terminator, so
        # every operand and phi arm is remapped before use even when branches nest.
        blocks_by_id = {block.id: block for block in self.hir.blocks}
        deferred: list[ValueId] = []  # loop-header phis whose latch arm is a body value lowered later; closed below
        for bid in reverse_postorder(self.hir):
            block = blocks_by_id[bid]
            self.builder.position_at(bid)
            for vid in block.phis:
                self._lower_phi(vid, self.hir.nodes[vid], deferred)
            for vid in block.operations:
                self._lower_node(vid, self.hir.nodes[vid])
            self._seal(block.terminator)
        for vid in deferred:
            self._close_phi(vid, self.hir.nodes[vid])
        for out in self.hir.outputs:
            self._lower_output(out.name, out.value)
        for slot in self.hir.state_slots:
            self._lower_state_slot(slot)
        mir = self.builder.finish()
        for vid, node in self.hir.nodes.items():
            if isinstance(node, Operation):
                assert not node.operator.sideband or vid not in self.remap
                selected = mir.nodes[self.remap[vid]] if vid in self.remap else None
                if isinstance(selected, MirOperation) and isinstance(selected.operator, PooledHardwareOperator):
                    assert not (node.operator.speculatable and selected.operator.error_ports)
        return mir

    def _seal(self, terminator: Terminator) -> None:
        match terminator:
            case Jump(target=target):
                self.builder.jump(target)
            case Branch(cond=cond, if_true=if_true, if_false=if_false):
                # A NOT on the condition is free: take the complementary target instead of inverting the register.
                base, inversion = _collapse_bool_inversions(self.hir.nodes, cond)
                if inversion.invert:
                    if_true, if_false = if_false, if_true
                self.builder.branch(self.remap[base], if_true, if_false)
            case Ret():
                self.builder.ret()

    def _lower_phi(self, old_id: ValueId, node: Node, deferred: list[ValueId]) -> None:
        # Each arm folds its type's OWN sideband chain into the arm conditioner, applied by the merge install: a sign
        # chain on a float arm (a branch assigning `-x`/`abs(x)`), a NOT chain on a boolean arm (`f = not g`).
        # A loop-header phi's latch arm is a body value lowered after the header: open the phi with its available arms
        # now (so the body can reference it) and close it once every block is lowered.
        assert isinstance(node, Phi)
        bases = [(pred, _collapse_conditioner(self.hir.nodes, value)) for pred, value in node.arms]
        scalar_type = self._phi_scalar_type(node)
        if all(base in self.remap for _, (base, _) in bases):
            arms = [(pred, self.remap[base], conditioner) for pred, (base, conditioner) in bases]
            self.remap[old_id] = self.builder.phi(scalar_type, arms)
        else:
            known = [(pred, self.remap[base], conditioner) for pred, (base, conditioner) in bases if base in self.remap]
            self.remap[old_id] = self.builder.open_phi(scalar_type, known[0])
            deferred.append(old_id)

    def _close_phi(self, old_id: ValueId, node: Node) -> None:
        assert isinstance(node, Phi)
        arms = [
            (pred, self.remap[base], conditioner)
            for pred, value in node.arms
            for base, conditioner in [_collapse_conditioner(self.hir.nodes, value)]
        ]
        self.builder.set_phi_arms(self.remap[old_id], arms)

    def _phi_scalar_type(self, node: Phi) -> ScalarType:
        match node.type:
            case HirFloatType():
                return ScalarFloatType(self.float_format)
            case HirIntType():
                return ScalarIntType(self.int_format)
            case HirBoolType():
                return ScalarBoolType()
            case _:
                raise UnsupportedConstruct(f"no MIR lowering rule for phi of type {node.type!r}")

    def _lower_node(self, old_id: ValueId, node: Node) -> None:
        if self.float_lowerer.lower_node(old_id, node):
            return
        if self.int_lowerer.lower_node(old_id, node):
            return
        if self._lower_bool_node(old_id, node):
            return
        match node:
            case Const(type=type):
                raise UnsupportedConstruct(f"no MIR lowering rule for HIR constant type {type!r}")
            case InPort(type=type):
                raise UnsupportedConstruct(f"no MIR lowering rule for HIR input type {type!r}")
            case Operation(operator=operator):
                raise UnsupportedConstruct(f"no hardware lowering rule for HIR operator {operator.mnemonic!r}")

    def _lower_bool_node(self, old_id: ValueId, node: Node) -> bool:
        if old_id in self.fused_directional_inf_members:
            return True
        plan = self.directional_inf_plans.get(old_id)
        if plan is not None:
            self.remap[old_id] = self.float_lowerer.lower_directional_inf(plan)
            return True
        match node:
            case InPort(name=name, type=HirBoolType()):
                self.remap[old_id] = self.builder.bool_input(name, ScalarBoolType())
                return True
            case StateRead(slot=slot, type=HirBoolType()):
                self.remap[old_id] = self.builder.bool_state_read(slot, ScalarBoolType())
                return True
            case BoolConst(value=value):
                self.remap[old_id] = self.builder.bool_const(value, ScalarBoolType())
                return True
            case Operation(operator=FloatComparison() as semantic, operands=(a, b)):
                # A relation is one comparator output port with an optional inversion (the ZKF ordering is total and
                # the flags one-hot), so every relation -- and every comparison over the same operand pair -- selects
                # into the same pooled fcmp operator and can fuse into one firing.
                base_a, sign_a = collapse_signs(self.hir.nodes, a)
                base_b, sign_b = collapse_signs(self.hir.nodes, b)
                fcmp = require(self.ops.fcmp, "fcmp")
                port, inversion = fcmp.tap_of(_RELATION_OF[type(semantic)])
                self.remap[old_id] = self.builder.operation(
                    fcmp,
                    [self.remap[base_a], self.remap[base_b]],
                    [sign_a, sign_b],
                    output_port=port,
                    output_conditioner=inversion,
                )
                return True
            case Operation(operator=IntComparison() as semantic, operands=(a, b)):
                # Two's complement is totally ordered, so the integer comparator taps exactly as the float one does.
                icmp = ICmpOperator(self.int_format)
                port, inversion = icmp.tap_of(_RELATION_OF[type(semantic)])
                self.remap[old_id] = self.builder.operation(
                    icmp,
                    [self.remap[a], self.remap[b]],
                    [IntIdentity(), IntIdentity()],
                    output_port=port,
                    output_conditioner=inversion,
                )
                return True
            case Operation(operator=IntToBool(), operands=(a,)):
                self.remap[old_id] = self.builder.operation(
                    IntToBoolOperator(self.int_format), [self.remap[a]], [IntIdentity()]
                )
                return True
            case Operation(
                operator=(FloatIsFinite() | FloatIsInf() | FloatIsPosInf() | FloatIsNegInf()) as semantic,
                operands=(a,),
            ):
                self._lower_float_classification(old_id, semantic, a)
                return True
            case Operation(operator=BoolAnd(), operands=(a, b)):
                self._lower_bool_logic(old_id, BoolAndOperator(), [a, b])
                return True
            case Operation(operator=BoolOr(), operands=(a, b)):
                self._lower_bool_logic(old_id, BoolOrOperator(), [a, b])
                return True
            case Operation(operator=BoolXor(), operands=(a, b)):
                self._lower_bool_logic(old_id, BoolXorOperator(), [a, b])
                return True
            case Operation(operator=BoolSelect(), operands=(cond, a, b)):
                # The boolean if-conversion mux: a NOT chain on the condition or either arm folds into that operand's
                # inversion conditioner, exactly like FloatSelect's sign folding -- so `a if not c else b` is free.
                self._lower_bool_logic(old_id, SelectOperator(ScalarBoolType()), [cond, a, b])
                return True
            case Operation(operator=BoolNot() as semantic, operands=(_,)):
                assert semantic.sideband
                return True
            case Operation(operator=FloatToBool(), operands=(a,)):
                # `bool(x)` reads a float operand (its sign is irrelevant: the exponent test is sign-invariant) and
                # writes the boolean bank, like the comparison but with an inline exponent reduction in place of fcmp.
                base, sign = collapse_signs(self.hir.nodes, a)
                self.remap[old_id] = self.builder.operation(
                    FloatToBoolOperator(self.float_format), [self.remap[base]], [sign]
                )
                return True
            case _:
                return False

    def _lower_bool_logic(self, old_id: ValueId, operator: HardwareOperator, operands: list[ValueId]) -> None:
        # NOT chains over the operands fold into the operand conditioners: `band(~a, b)` is one gate.
        bases = [_collapse_bool_inversions(self.hir.nodes, operand) for operand in operands]
        self.remap[old_id] = self.builder.operation(
            operator, [self.remap[base] for base, _ in bases], [inversion for _, inversion in bases]
        )

    def _lower_float_classification(
        self, old_id: ValueId, semantic: FloatIsFinite | FloatIsInf | FloatIsPosInf | FloatIsNegInf, a: ValueId
    ) -> None:
        operator, output = self.float_lowerer.classification_lowering(semantic)
        base, sign = collapse_signs(self.hir.nodes, a)
        self.remap[old_id] = self.builder.operation(operator, [self.remap[base]], [sign], output_conditioner=output)

    def _lower_output(self, name: str, value: ValueId) -> None:
        if self.float_lowerer.lower_output(name, value):
            return
        if self.int_lowerer.lower_output(name, value):
            return
        if self._lower_bool_output(name, value):
            return
        raise UnsupportedConstruct(f"no MIR lowering rule for HIR output type {self.hir.nodes[value].type!r}")

    def _lower_bool_output(self, name: str, value: ValueId) -> bool:
        if not isinstance(self.hir.nodes[value].type, HirBoolType):
            return False
        base, inversion = _collapse_bool_inversions(self.hir.nodes, value)
        self.builder.bool_output(name, self.remap[base], inversion)
        return True

    def _lower_state_slot(self, slot: StateSlot) -> None:
        if self.float_lowerer.lower_state_slot(slot):
            return
        if self.int_lowerer.lower_state_slot(slot):
            return
        if self._lower_bool_state_slot(slot):
            return
        raise UnsupportedConstruct(f"no MIR lowering rule for HIR state slot {slot.name!r}")

    def _lower_bool_state_slot(self, slot: StateSlot) -> bool:
        if not isinstance(self.hir.nodes[slot.live_out].type, HirBoolType):
            return False
        base, inversion = _collapse_bool_inversions(self.hir.nodes, slot.live_out)
        assert isinstance(slot.reset_value, BoolConst), "the gate owns a slot whose reset is of the wrong family"
        self.builder.bool_state_slot(slot.name, slot.reset_value.value, self.remap[base], inversion)
        return True


class _FloatLowerer:
    def __init__(self, context: _LoweringContext) -> None:
        self.context = context
        self.float_type = ScalarFloatType(context.float_format)

    def lower_node(self, old_id: ValueId, node: Node) -> bool:
        match node:
            case InPort(name=name, type=HirFloatType()):
                self.context.remap[old_id] = self.context.builder.float_input(name, self.float_type)
                return True
            case StateRead(slot=slot, type=HirFloatType()):
                self.context.remap[old_id] = self.context.builder.float_state_read(slot, self.float_type)
                return True
            case FloatConst(value=value):
                self.context.remap[old_id] = self._lower_float_const(value)
                return True
            case Operation(operator=semantic) if sign_of(node) is not None:
                assert semantic.sideband
                return True
            case Operation() as operation:
                if old_id in self.context.absorbed_roundings:
                    return True  # absorbed into an adjacent conversion's mode immediate
                atan2_id = self.context.fused_hypots.get(old_id)
                if atan2_id is not None:
                    self.context.remap[old_id] = self._lower_fused_hypot(atan2_id)
                    return True
                lowered = self._lower_operation(operation)
                if lowered is None:
                    return False
                self.context.remap[old_id] = lowered
                return True
            case _:
                return False

    def _lower_float_const(self, value: float) -> ValueId:
        return self.context.builder.float_const(value, self.float_type)

    def lower_directional_inf(self, plan: _DirectionalInfPlan) -> ValueId:
        operator: FloatClassificationOperator = (
            FloatIsPosInfOperator(self.context.float_format)
            if isinstance(plan.semantic, FloatIsPosInf)
            else FloatIsNegInfOperator(self.context.float_format)
        )
        return self.context.builder.operation(operator, [self.context.remap[plan.operand]], [plan.sign])

    def _lower_operation(self, node: Operation) -> ValueId | None:
        match node:
            case Operation(operator=FloatAdd(), operands=(a, b)):
                return self._emit_float(require(self.context.ops.fadd, "fadd"), [a, b])
            case Operation(operator=FloatMul(), operands=(a, b)):
                return self._emit_float(require(self.context.ops.fmul, "fmul"), [a, b])
            case Operation(operator=FloatDiv(), operands=(a, b)):
                return self._emit_float(require(self.context.ops.fdiv, "fdiv"), [a, b])
            case Operation(operator=FloatMulPow2(k=k), operands=(a,)):
                return self._lower_float_mul_pow2(a, k)
            case Operation(operator=FloatMulPow2Dynamic(), operands=(a, k)):
                base, sign = collapse_signs(self.context.hir.nodes, a)
                return self._emit_scale(self.context.remap[base], sign, self.context.remap[k])
            case Operation(
                operator=(FloatRound() | FloatFloor() | FloatCeil() | FloatTrunc()) as semantic, operands=(a,)
            ):
                return self._lower_round(semantic, a)
            case Operation(operator=FloatExp2(), operands=(a,)):
                return self._emit_float(require(self.context.ops.fexp2, "fexp2"), [a])
            case Operation(operator=FloatLog2(), operands=(a,)):
                return self._emit_float(require(self.context.ops.flog2, "flog2"), [a])
            case Operation(operator=(FloatSinTurns() | FloatCosTurns()) as semantic, operands=(a,)):
                return self._lower_sincos(semantic, a)
            case Operation(operator=FloatSqrt(), operands=(a,)):
                return self._emit_float(require(self.context.ops.fsqrt, "fsqrt"), [a])
            case Operation(operator=FloatAtan2Turns(), operands=(y, x)):
                return self._lower_atan2(y, x)
            case Operation(operator=(FloatMin() | FloatMax()) as semantic, operands=(a, b)):
                return self._lower_minmax(semantic, a, b)
            case Operation(operator=FloatFma(), operands=(a, b, c)):
                return self._emit_float(require(self.context.ops.ffma, "ffma"), [a, b, c])
            case Operation(operator=BoolToFloat(), operands=(a,)):
                # `float(cond)` crosses from the boolean bank into the wide bank; a NOT chain folds into the
                # operand conditioner.
                base, inversion = _collapse_bool_inversions(self.context.hir.nodes, a)
                return self.context.builder.operation(
                    BoolToFloatOperator(self.context.float_format),
                    [self.context.remap[base]],
                    [inversion],
                )
            case Operation(operator=IntToFloat(), operands=(a,)):
                return self.context.builder.operation(
                    require(self.context.ops.ffromint, "ffromint"),
                    [self.context.remap[a]],
                    [IntIdentity()],
                )
            case Operation(operator=FloatSelect(), operands=(cond, a, b)):
                # The if-conversion mux: arm signs and a condition NOT chain fold into the operand conditioners
                # (`x if c else -x` and `a if not c else b` cost no hardware beyond the mux itself).
                base_c, inv_c = _collapse_bool_inversions(self.context.hir.nodes, cond)
                base_a, sign_a = collapse_signs(self.context.hir.nodes, a)
                base_b, sign_b = collapse_signs(self.context.hir.nodes, b)
                return self.context.builder.operation(
                    SelectOperator(ScalarFloatType(self.context.float_format)),
                    [self.context.remap[base_c], self.context.remap[base_a], self.context.remap[base_b]],
                    [inv_c, sign_a, sign_b],
                )
            case _:
                return None

    def _emit_float(
        self,
        hardware: FloatHardwareOperator,
        operands: list[ValueId],
        output_port: int = 0,
        immediates: tuple[int, ...] = (),
    ) -> ValueId:
        # Each operand's sign chain folds onto its conditioner, applied before the op: min(-a, b) feeds the sorter -a,
        # floor(-x) feeds -x.
        bases, signs = zip(*(collapse_signs(self.context.hir.nodes, operand) for operand in operands))
        return self.context.builder.operation(
            hardware, [self.context.remap[base] for base in bases], list(signs), output_port, immediates=immediates
        )

    def _lower_round(self, semantic: FloatRound | FloatFloor | FloatCeil | FloatTrunc, a: ValueId) -> ValueId:
        mode = _ROUND_MODE_OF[type(semantic)]
        return self._emit_float(require(self.context.ops.fround, "fround"), [a], immediates=(int(mode),))

    def _lower_minmax(self, semantic: FloatMin | FloatMax, a: ValueId, b: ValueId) -> ValueId:
        # min taps the low output port, max the high one; a min and a max over one pair fuse into one sorter firing.
        operator = require(self.context.ops.fsort, "fsort")
        return self._emit_float(operator, [a, b], 0 if isinstance(semantic, FloatMin) else 1)

    def _lower_sincos(self, semantic: FloatSinTurns | FloatCosTurns, a: ValueId) -> ValueId:
        # zkf_sincos already counts in turns, so the operand needs no scaling here at all -- whatever conversion the
        # kernel's angle wanted was stated in HIR and folded there. The sign rides the core's own operand, so one
        # value serves sin(-x)/cos(-x) and a sin+cos over one argument fuse into one firing.
        operator = require(self.context.ops.fsincos, "fsincos")
        base, sign = collapse_signs(self.context.hir.nodes, a)
        return self.context.builder.operation(
            operator,
            [self.context.remap[base]],
            [sign],
            output_port=0 if isinstance(semantic, FloatSinTurns) else 1,
        )

    def _lower_atan2(self, y: ValueId, x: ValueId) -> ValueId:
        # zkf_atan2 returns theta in turns, which is what the operator now means. Its magnitude port is tapped only by
        # a fusible adjacent hypot (intercepted in lower_node).
        return self._emit_float(require(self.context.ops.fatan2, "fatan2"), [y, x])

    def _lower_fused_hypot(self, atan2_id: ValueId) -> ValueId:
        # Emitting the fatan2 firing from the ATAN2's own operands/signs makes the two collapse into one CORDIC; the
        # magnitude is symmetric and sign-invariant, so the hypot's own operand order/signs are immaterial.
        node = self.context.hir.nodes[atan2_id]
        assert isinstance(node, Operation)
        y, x = node.operands
        operator = self.context.ops.fatan2
        assert operator is not None  # only reached for a planned fusion, which exists only when fatan2 is configured
        return self._emit_float(operator, [y, x], 1)

    def classification_lowering(
        self, semantic: FloatIsFinite | FloatIsInf | FloatIsPosInf | FloatIsNegInf
    ) -> tuple[FloatClassificationOperator, BoolInversion]:
        fmt = self.context.float_format
        match semantic:
            case FloatIsFinite():
                return FloatIsFiniteOperator(fmt), BoolInversion()
            case FloatIsInf():
                return FloatIsFiniteOperator(fmt), BoolInversion(invert=True)
            case FloatIsPosInf():
                return FloatIsPosInfOperator(fmt), BoolInversion()
            case FloatIsNegInf():
                return FloatIsNegInfOperator(fmt), BoolInversion()

    def _lower_float_mul_pow2(self, a: ValueId, k: int) -> ValueId:
        base, sign = collapse_signs(self.context.hir.nodes, a)
        return self._emit_scale_pow2(self.context.remap[base], sign, k)

    def _emit_scale_pow2(self, operand: ValueId, sign: FloatSignControl, k: int) -> ValueId:
        """
        The exponent rides as a materialized integer constant, clamped rather than refused: any count past the int
        format already lies far beyond the float's dynamic range, where the scaler rails or flushes identically.
        """
        ifmt = self.context.int_format
        return self._emit_scale(operand, sign, self.context.builder.int_const(ifmt.saturate(k), ScalarIntType(ifmt)))

    def _emit_scale(self, operand: ValueId, sign: FloatSignControl, exponent: ValueId) -> ValueId:
        return self.context.builder.operation(
            require(self.context.ops.fmul_ilog2, "fmul_ilog2"),
            [operand, exponent],
            [sign, IntIdentity()],
        )

    def lower_output(self, name: str, value: ValueId) -> bool:
        base, sign = collapse_signs(self.context.hir.nodes, value)
        if not isinstance(self.context.hir.nodes[base].type, HirFloatType):
            return False
        self.context.builder.float_output(name, self.context.remap[base], sign)
        return True

    def lower_state_slot(self, slot: StateSlot) -> bool:
        base, sign = collapse_signs(self.context.hir.nodes, slot.live_out)
        if not isinstance(self.context.hir.nodes[base].type, HirFloatType):
            return False
        assert isinstance(slot.reset_value, FloatConst), "the gate owns a slot whose reset is of the wrong family"
        self.context.builder.float_state_slot(slot.name, slot.reset_value.value, self.context.remap[base], sign)
        return True


class _IntLowerer:
    """
    The integer dual of _FloatLowerer, owning every operation whose RESULT is an integer. Its operands never
    carry a folded sideband: an integer port conditions with the identity alone, so `ineg` and `iabs` are hardware
    where their float counterparts are free. Integer operators are never optional, so only the two conversions that
    cross into the float half go through require.
    """

    def __init__(self, context: _LoweringContext) -> None:
        self.context = context
        self.int_type = ScalarIntType(context.int_format)

    def lower_node(self, old_id: ValueId, node: Node) -> bool:
        match node:
            case InPort(name=name, type=HirIntType()):
                self.context.remap[old_id] = self.context.builder.int_input(name, self.int_type)
                return True
            case StateRead(slot=slot, type=HirIntType()):
                self.context.remap[old_id] = self.context.builder.int_state_read(slot, self.int_type)
                return True
            case IntConst(value=value):
                if old_id not in self.context.folded_shift_counts:
                    self.context.remap[old_id] = self._const(value)
                return True
            case Operation() as operation:
                lowered = self._lower_operation(operation)
                if lowered is None:
                    return False
                self.context.remap[old_id] = lowered
                return True
            case _:
                return False

    def _lower_operation(self, node: Operation) -> ValueId | None:
        fmt = self.context.int_format
        match node:
            case Operation(operator=IntAdd(), operands=(a, b)):
                return self._emit(IAddOperator(fmt), a, b)
            case Operation(operator=IntSub(), operands=(a, b)):
                return self._emit(ISubOperator(fmt), a, b)
            case Operation(operator=IntMul(), operands=(a, b)):
                return self._emit(self.context.ops.imul, a, b)
            case Operation(operator=IntMulPow2() as semantic, operands=(a,)):
                return self._scale_by_pow2(semantic, a)
            case Operation(operator=IntNeg(), operands=(a,)):
                return self._negate(self.context.remap[a])
            case Operation(operator=IntAbs(), operands=(a,)):
                return self._emit(IAbsOperator(fmt), a)
            case Operation(operator=IntPopcount(), operands=(a,)):
                return self._emit(IPopcntOperator(fmt), a)
            # The quotient and the remainder are two taps of one divider: written from the same operands with the same
            # conditioners, they share a MIR intern key up to the port and fuse into a single firing at LIR build.
            case Operation(operator=IntDivFloor(), operands=(a, b)):
                return self._emit(IDivOperator(fmt), a, b, output_port=0)
            case Operation(operator=IntMod(), operands=(a, b)):
                return self._emit(IDivOperator(fmt), a, b, output_port=1)
            case Operation(operator=(IntShiftLeft() | IntShiftRight()) as semantic, operands=(a, b)):
                return self._lower_shift(semantic, a, b)
            case Operation(operator=IntBwAnd(), operands=(a, b)):
                return self._emit(IntBwAndOperator(fmt), a, b)
            case Operation(operator=IntBwOr(), operands=(a, b)):
                return self._emit(IntBwOrOperator(fmt), a, b)
            case Operation(operator=IntBwXor(), operands=(a, b)):
                return self._emit(IntBwXorOperator(fmt), a, b)
            case Operation(operator=IntBwNot(), operands=(a,)):
                return self._emit(IntBwNotOperator(fmt), a)
            case Operation(operator=IntSelect(), operands=(cond, a, b)):
                # The integer if-conversion mux: only the condition folds (`a if not c else b` costs no extra gate).
                base_c, inv_c = _collapse_bool_inversions(self.context.hir.nodes, cond)
                return self.context.builder.operation(
                    SelectOperator(self.int_type),
                    [self.context.remap[base_c], self.context.remap[a], self.context.remap[b]],
                    [inv_c, IntIdentity(), IntIdentity()],
                )
            case Operation(operator=BoolToInt(), operands=(a,)):
                base, inversion = _collapse_bool_inversions(self.context.hir.nodes, a)
                return self.context.builder.operation(BoolToIntOperator(fmt), [self.context.remap[base]], [inversion])
            case Operation(operator=FloatToInt(), operands=(a,)):
                return self._lower_to_int(a)
            case Operation(operator=FloatILog2(), operands=(a,)):
                base, _ = collapse_signs(self.context.hir.nodes, a)  # the exponent cannot see the sign
                return self.context.builder.operation(
                    require(self.context.ops.filog2, "filog2"), [self.context.remap[base]], [FloatSignControl()]
                )
            case _:
                return None

    def _lower_to_int(self, a: ValueId) -> ValueId:
        """
        `int(x)` truncates toward zero, and a rounding it reads becomes its mode instead of a module of its own.
        The surviving operand's sign chain folds onto the float port, applied before the rounding as the source has it.
        """
        mode, operand = _absorbed_rounding(self.context.hir, a) or (RoundMode.TRUNC, a)
        base, sign = collapse_signs(self.context.hir.nodes, operand)
        return self.context.builder.operation(
            require(self.context.ops.ftoint, "ftoint"),
            [self.context.remap[base]],
            [sign],
            immediates=(int(mode),),
        )

    def _lower_shift(self, semantic: IntShiftLeft | IntShiftRight, a: ValueId, count: ValueId) -> ValueId:
        constant = constant_shift_count(self.context.hir, count)
        if constant is None:
            return self._runtime_shift(semantic, a, count)
        assert constant >= 0, "the gate owns a negative count, which no machine word settles"
        return self._constant_shift(semantic, a, constant)

    def _runtime_shift(self, semantic: IntShiftLeft | IntShiftRight, a: ValueId, count: ValueId) -> ValueId:
        """
        Each direction has the module that names it, so neither negates its count to reach the other's. The left
        shifter is tapped on its raw reading, because `<<` drops what leaves the word rather than saturating. Both
        modules clamp the amount at the word, which is where the two readings of an unbounded count meet -- a left
        shift past the word answers zero and a right shift past it answers the sign fill, as Python's own unbounded
        shift does once the word truncates it.
        """
        fmt = self.context.int_format
        hardware = IShrOperator(fmt) if isinstance(semantic, IntShiftRight) else IShlOperator(fmt)
        return self._emit(hardware, a, count)

    def _constant_shift(self, semantic: IntShiftLeft | IntShiftRight, a: ValueId, count: int) -> ValueId:
        """The same raw reading as the runtime shifter, without either module."""
        width = self.context.int_format.width
        assert count > 0, "strength reduction elides a zero-count shift"
        if isinstance(semantic, IntShiftLeft):
            assert count < width, "specialization owns a left shift past the word, whose answer is zero regardless"
            shamt = count
        else:
            # Clamped here and not in HIR: past the top bit a right shift repeats the sign fill only for a value the
            # word already holds, and this operand is one where an HIR constant need not have been.
            shamt = -min(count, width - 1)
        return self.context.builder.operation(
            IntShiftConstOperator(self.context.int_format, shamt),
            [self.context.remap[a]],
            [IntIdentity()],
        )

    def _scale_by_pow2(self, semantic: IntMulPow2, a: ValueId) -> ValueId:
        """
        The left shifter's OTHER reading: `prod` saturates where `shft` lets the high bits fall off the word, and
        saturating is what a multiplication does. The count is unbounded where the word is not, so it clamps at the
        width -- past that every count rails the same operand the same way, and only zero survives either. It is
        itself a machine integer, and a two-bit word cannot hold its own width, so it clamps into the word as well:
        every count from `width - 1` up rails identically, and `max` is never below that.
        """
        fmt = self.context.int_format
        return self.context.builder.operation(
            IShlOperator(fmt),
            [self.context.remap[a], self._const(fmt.saturate(min(semantic.k, fmt.width)))],
            [IntIdentity(), IntIdentity()],
            output_port=1,
        )

    def _negate(self, value: ValueId) -> ValueId:
        """`0 - x`: there is no negation module, and the subtractor saturates `-MIN` correctly."""
        return self.context.builder.operation(
            ISubOperator(self.context.int_format),
            [self._const(0), value],
            [IntIdentity(), IntIdentity()],
        )

    def _const(self, value: int) -> ValueId:
        return self.context.builder.int_const(value, self.int_type)

    def _emit(self, hardware: HardwareOperator, *operands: ValueId, output_port: int = 0) -> ValueId:
        return self.context.builder.operation(
            hardware,
            [self.context.remap[operand] for operand in operands],
            [IntIdentity()] * len(operands),
            output_port=output_port,
        )

    def lower_output(self, name: str, value: ValueId) -> bool:
        if not isinstance(self.context.hir.nodes[value].type, HirIntType):
            return False
        self.context.builder.int_output(name, self.context.remap[value])
        return True

    def lower_state_slot(self, slot: StateSlot) -> bool:
        if not isinstance(self.context.hir.nodes[slot.live_out].type, HirIntType):
            return False
        assert isinstance(slot.reset_value, IntConst), "the gate owns a slot whose reset is of the wrong family"
        self.context.builder.int_state_slot(slot.name, slot.reset_value.value, self.context.remap[slot.live_out])
        return True


def _derive(hir: Hir, ops: OpConfig, ifconv_max_ops: int) -> Hir:
    """
    Everything this machine knows, told to the graph. The trigonometric unit goes first, since it is the optimizer's
    input rather than its consumer, and the word width from within the fixpoint, since only a fold can reveal a count.
    """
    hir = optimize(trig_abi(hir), ifconv_max_ops)
    fuel = left_shifts(hir) + hypot_count(hir) + 1  # each rewrite deletes one of these, and no pass mints either
    while True:
        rewritten = specialize(hir, ops.int_format) or expand_unfused(hir, ops)
        if rewritten is None:
            break
        fuel -= 1
        assert fuel > 0, "the derivation is not settling"
        hir = optimize(rewritten, ifconv_max_ops)
    return rescale(hir, ops)  # last: strength reduction would compose the pair it splits straight back


def _widest_word(options: MirOptions) -> int:
    return max(options.wint_min, options.float_format.width)


def _word(hir: Hir, options: MirOptions) -> int:
    return _widest_word(options) if HirFloatType() in hir.value_types() else options.wint_min


def lower(hir: Hir, options: MirOptions) -> Mir:
    """
    Optimize the front end's HIR against this machine, judge what survives, then select hardware operators for it and
    fold semantic signs onto MIR sign controls.

    The machine word is written into the graph, not read off it: a constant shift past the word folds to zero and
    cascades, so the graph is derived at the widest word the configuration admits and again at the narrower one the
    surviving families ask for, adopting the narrower machine only where its own graph asks for it in turn. No fixpoint
    need exist, since an identity can erase work as the word narrows; a graph the narrower word cannot build is a
    narrowing that failed. Optimization is not the caller's to run, and neither is judgement: a substituted constant
    cascades, so the passes run again after every substitution and `refuse` waits for the graph that is actually built.
    """
    original, widest = hir, _widest_word(options)
    ops = OpConfig(options.operator, options.float_format, IntFormat(widest), options.wmultiplier)
    hir = _derive(original, ops, options.ifconv_max_ops)
    if (word := _word(hir, options)) != widest:
        narrow = OpConfig(options.operator, options.float_format, IntFormat(word), options.wmultiplier)
        try:
            candidate = _derive(original, narrow, options.ifconv_max_ops)
        except UnsupportedConstruct as ex:
            _logger.warning("Machine word: int%d, since int%d leaves nothing buildable (%s)", widest, word, ex)
        else:
            if _word(candidate, options) == word:
                ops, hir = narrow, candidate
                _logger.info("Machine word: int%d, narrowed from int%d by the families the kernel keeps", word, widest)
            else:
                _logger.warning("Machine word: int%d, since int%d revives work the wider word erased", widest, word)
    else:
        _logger.info("Machine word: int%d, the widest the configuration admits", widest)
    _logger.info(
        "Optimized HIR:\n\tinputs=%s\n\toutputs=%s\n\tnodes=%d\n\tblocks=%d",
        hir.input_ids,
        hir.outputs,
        len(hir.nodes),
        len(hir.blocks),
    )
    refuse(hir)
    return _LoweringContext(_contract_fmas(hir, ops), ops).run()
