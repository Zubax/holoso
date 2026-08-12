"""Lower optimized HIR to selected MIR."""

import logging
from collections import Counter
from dataclasses import dataclass

from .._errors import UnsupportedConstruct
from .._hir import (
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
    FloatGreater,
    FloatGreaterOrEqual,
    FloatHypot2,
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
    StateRead,
    StateSlot,
    Terminator,
    optimize,
    reverse_postorder,
)
from ._refuse import refuse
from ._specialize import constant_shift_count, left_shifts, specialize
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
    FloatType as ScalarFloatType,
    IntType as ScalarIntType,
    ScalarType,
)
from ._ir import Mir, MirBuilder

_logger = logging.getLogger(__name__)

# The seam between the semantic relation and the comparator flag it taps; the two vocabularies meet only here.
# Both comparator families share it: an integer comparison taps ``icmp`` exactly as a float one taps ``fcmp``.
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


# Shared by the standalone ``fround`` and the ``ftoint`` that absorbs one: the same field on both.
_ROUND_MODE_OF: dict[type[Operator], RoundMode] = {
    FloatRound: RoundMode.NEAREST_EVEN,
    FloatFloor: RoundMode.FLOOR,
    FloatCeil: RoundMode.CEIL,
    FloatTrunc: RoundMode.TRUNC,
}


def _select_hardware(semantic: Operator, hardware: HardwareOperator) -> HardwareOperator:
    """
    The single choke point where a semantic operator meets the hardware operator selected for it. ``speculatable``
    (semantic side) and ``error_ports`` (hardware side) declare one fact -- whether evaluation on a never-taken path
    is observable -- so the two declarations are enforced in lockstep here: a speculatable semantic operator must
    never lower to error-bearing hardware, or if-conversion would assert the module error flag for untaken paths.
    """
    error_ports = hardware.error_ports if isinstance(hardware, PooledHardwareOperator) else []
    assert not (
        semantic.speculatable and error_ports
    ), f"{semantic.mnemonic} is speculatable but lowers to error-bearing {hardware.mnemonic}"
    return hardware


def _sign_of(node: Operation) -> FloatSignControl | None:
    match node:
        case Operation(operator=FloatNeg()):
            return FloatSignControl(negate=True)
        case Operation(operator=FloatAbs()):
            return FloatSignControl(absolute=True)
        case _:
            return None


def _collapse_bool_inversions(nodes: dict[ValueId, Node], vid: ValueId) -> tuple[ValueId, BoolInversion]:
    """
    Peel a chain of semantic NOT operations, returning the base value and the combined inversion -- the boolean dual
    of :func:`_collapse_signs`. Folding happens on the CONSUMER side only: a NOT over a comparison must never flip
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
    has no free sideband, so ``ineg``/``iabs`` are hardware and nothing collapses.
    """
    ty = nodes[vid].type
    match ty:
        case HirBoolType():
            return _collapse_bool_inversions(nodes, vid)
        case HirFloatType():
            return _collapse_signs(nodes, vid)
        case HirIntType():
            return vid, IntIdentity()
        case _:
            raise UnsupportedConstruct(f"no conditioner-collapse rule for HIR type {ty!r}")


def _collapse_signs(nodes: dict[ValueId, Node], vid: ValueId) -> tuple[ValueId, FloatSignControl]:
    """Peel a chain of semantic sign operations, returning the non-sign base value and combined sign control."""
    chain: list[FloatSignControl] = []
    node = nodes[vid]
    while isinstance(node, Operation) and (sign := _sign_of(node)) is not None:
        chain.append(sign)
        (vid,) = node.operands
        node = nodes[vid]
    control = FloatSignControl()
    for sign in reversed(chain):  # innermost first
        control = control.then(sign)
    return vid, control


@dataclass(frozen=True, slots=True)
class _FmaPlan:
    """
    A planned contraction of ``a*b + c`` into one ``ffma``. ``mul`` is the FloatMul whose standalone MIR op is
    suppressed, ``ma``/``mb`` its operands, ``c`` the addend, ``product_sign`` the sign peeled off the product operand.
    """

    mul: ValueId
    ma: ValueId
    mb: ValueId
    c: ValueId
    product_sign: FloatSignControl


@dataclass(frozen=True, slots=True)
class _DirectionalInfPlan:
    """A planned contraction of ``isinf(x) and x`` sign tests into one directional infinity predicate."""

    operand: ValueId
    sign: FloatSignControl
    semantic: FloatIsPosInf | FloatIsNegInf
    members: frozenset[ValueId]


def _compute_use_counts(hir: Hir) -> dict[ValueId, int]:
    """Total reference count per value across every use site: operation operands, phi arms, and external references."""
    counts: dict[ValueId, int] = {vid: 0 for vid in hir.nodes}
    for node in hir.nodes.values():
        if isinstance(node, Operation):
            for operand in node.operands:
                counts[operand] += 1
        elif isinstance(node, Phi):
            for _, value in node.arms:
                counts[value] += 1
    for vid in hir.external_value_references():
        counts[vid] += 1
    return counts


def _is_zero_float(hir: Hir, vid: ValueId) -> bool:
    base, _ = _collapse_signs(hir.nodes, vid)
    node = hir.nodes[base]
    return isinstance(node, FloatConst) and node.value == 0.0


def _match_isinf_operand(hir: Hir, vid: ValueId) -> ValueId | None:
    match hir.nodes[vid]:
        case Operation(operator=FloatIsInf(), operands=(operand,)):
            base, _ = _collapse_signs(hir.nodes, operand)
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
    operand, sign = _collapse_signs(hir.nodes, left if right_zero else right)
    if sign.absolute:
        return None
    # ``x > 0`` and ``0 < x`` both test the positive side; either mirroring alone flips it.
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


def _exclusive_mul(hir: Hir, use_counts: dict[ValueId, int], vid: ValueId) -> tuple[ValueId, FloatSignControl] | None:
    """
    If ``vid`` is a single-use ``a*b`` reached through single-use sign ops, return the FloatMul and its combined sign;
    else None. Every node on the path must have use-count 1, so the rounded product is observed nowhere else -- only
    then is contracting to a single rounding faithful. A FloatMulPow2 (``a*2**k``) is not a FloatMul, so never matches.
    """
    signs: list[FloatSignControl] = []
    node = hir.nodes[vid]
    while isinstance(node, Operation) and (sign := _sign_of(node)) is not None:
        if use_counts[vid] != 1:
            return None
        signs.append(sign)
        (vid,) = node.operands
        node = hir.nodes[vid]
    if not (isinstance(node, Operation) and isinstance(node.operator, FloatMul)) or use_counts[vid] != 1:
        return None
    product_sign = FloatSignControl()
    for sign in reversed(signs):
        product_sign = product_sign.then(sign)
    return vid, product_sign


def _plan_fma_fusions(hir: Hir, ops: OpConfig, use_counts: dict[ValueId, int]) -> dict[ValueId, _FmaPlan]:
    """
    Map each FloatAdd that will contract into an ``ffma`` to its plan (only when ``ffma`` is configured). When both
    addends are exclusive products only the first contracts -- one fma carries one product.
    """
    if ops.ffma is None:
        return {}
    plans: dict[ValueId, _FmaPlan] = {}
    for vid, node in hir.nodes.items():
        if not (isinstance(node, Operation) and isinstance(node.operator, FloatAdd)):
            continue
        op0, op1 = node.operands
        for product_operand, addend in ((op0, op1), (op1, op0)):
            found = _exclusive_mul(hir, use_counts, product_operand)
            if found is not None:
                mul_vid, product_sign = found
                mul_node = hir.nodes[mul_vid]
                assert isinstance(mul_node, Operation)
                ma, mb = mul_node.operands
                plans[vid] = _FmaPlan(mul=mul_vid, ma=ma, mb=mb, c=addend, product_sign=product_sign)
                break
    return plans


def _operand_base_set(hir: Hir, node: Operation) -> tuple[ValueId, ...]:
    """
    Order-independent: hypot is commutative and sign-invariant, as is the fatan2 magnitude it taps, so a hypot fuses
    with any same-block atan2 over the same value pair.
    """
    return tuple(sorted(_collapse_signs(hir.nodes, operand)[0] for operand in node.operands))


def _plan_hypot_fusions(hir: Hir, ops: OpConfig) -> dict[ValueId, ValueId]:
    """
    Map each FloatHypot2 to a same-block FloatAtan2Turns over the same value pair so MIR can tap the atan2's magnitude
    port (the two fuse into one CORDIC) rather than decompose into primitives. Block-local, like the LIR firing fusion
    it feeds.
    """
    if ops.fatan2 is None:
        return {}
    plans: dict[ValueId, ValueId] = {}
    for block in hir.blocks:
        atan2_by_pair: dict[tuple[ValueId, ...], ValueId] = {}
        for vid in block.operations:
            if isinstance(node := hir.nodes[vid], Operation) and isinstance(node.operator, FloatAtan2Turns):
                atan2_by_pair.setdefault(_operand_base_set(hir, node), vid)
        for vid in block.operations:
            if isinstance(node := hir.nodes[vid], Operation) and isinstance(node.operator, FloatHypot2):
                match = atan2_by_pair.get(_operand_base_set(hir, node))
                if match is not None:
                    plans[vid] = match
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
    The mode a ``FloatToInt`` carries instead of reading a rounding's result, with the value it rounds. Only a
    rounding read DIRECTLY: anything between conditions the ROUNDED value, and ``-floor(x)`` is not ``floor(-x)``.
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
        use_counts = _compute_use_counts(hir)
        self.fma_plans = _plan_fma_fusions(hir, ops, use_counts)
        self.fused_muls = {plan.mul for plan in self.fma_plans.values()}
        self.directional_inf_plans = _plan_directional_inf_fusions(hir, use_counts)
        self.fused_directional_inf_members = {
            member for plan in self.directional_inf_plans.values() for member in plan.members
        }
        self.fused_hypots = _plan_hypot_fusions(hir, ops)
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
        return self.builder.finish()

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
        # chain on a float arm (a branch assigning ``-x``/``abs(x)``), a NOT chain on a boolean arm (``f = not g``).
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
                base_a, sign_a = _collapse_signs(self.hir.nodes, a)
                base_b, sign_b = _collapse_signs(self.hir.nodes, b)
                fcmp = require(self.ops.fcmp, "fcmp")
                port, inversion = fcmp.tap_of(_RELATION_OF[type(semantic)])
                self.remap[old_id] = self.builder.operation(
                    _select_hardware(semantic, fcmp),
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
                    _select_hardware(semantic, icmp),
                    [self.remap[a], self.remap[b]],
                    [IntIdentity(), IntIdentity()],
                    output_port=port,
                    output_conditioner=inversion,
                )
                return True
            case Operation(operator=IntToBool() as semantic, operands=(a,)):
                self.remap[old_id] = self.builder.operation(
                    _select_hardware(semantic, IntToBoolOperator(self.int_format)), [self.remap[a]], [IntIdentity()]
                )
                return True
            case Operation(
                operator=(FloatIsFinite() | FloatIsInf() | FloatIsPosInf() | FloatIsNegInf()) as semantic,
                operands=(a,),
            ):
                self._lower_float_classification(old_id, semantic, a)
                return True
            case Operation(operator=BoolAnd() as semantic, operands=(a, b)):
                self._lower_bool_logic(old_id, _select_hardware(semantic, BoolAndOperator()), [a, b])
                return True
            case Operation(operator=BoolOr() as semantic, operands=(a, b)):
                self._lower_bool_logic(old_id, _select_hardware(semantic, BoolOrOperator()), [a, b])
                return True
            case Operation(operator=BoolXor() as semantic, operands=(a, b)):
                self._lower_bool_logic(old_id, _select_hardware(semantic, BoolXorOperator()), [a, b])
                return True
            case Operation(operator=BoolSelect() as semantic, operands=(cond, a, b)):
                # The boolean if-conversion mux: a NOT chain on the condition or either arm folds into that operand's
                # inversion conditioner, exactly like FloatSelect's sign folding -- so ``a if not c else b`` is free.
                self._lower_bool_logic(
                    old_id, _select_hardware(semantic, SelectOperator(ScalarBoolType())), [cond, a, b]
                )
                return True
            case Operation(operator=BoolNot(), operands=(_,)):
                # A NOT never materializes hardware: every consumer position collapses the chain into its own
                # conditioner directly from the HIR nodes, so the NOT's own vid is deliberately left unmapped -- a
                # consumer that bypassed the collapse would fail loudly on the missing remap entry.
                return True
            case Operation(operator=FloatToBool() as semantic, operands=(a,)):
                # ``bool(x)`` reads a float operand (its sign is irrelevant: the exponent test is sign-invariant) and
                # writes the boolean bank, like the comparison but with an inline exponent reduction in place of fcmp.
                base, sign = _collapse_signs(self.hir.nodes, a)
                self.remap[old_id] = self.builder.operation(
                    _select_hardware(semantic, FloatToBoolOperator(self.float_format)), [self.remap[base]], [sign]
                )
                return True
            case _:
                return False

    def _lower_bool_logic(self, old_id: ValueId, operator: HardwareOperator, operands: list[ValueId]) -> None:
        # NOT chains over the operands fold into the operand conditioners: ``band(~a, b)`` is one gate.
        bases = [_collapse_bool_inversions(self.hir.nodes, operand) for operand in operands]
        self.remap[old_id] = self.builder.operation(
            operator, [self.remap[base] for base, _ in bases], [inversion for _, inversion in bases]
        )

    def _lower_float_classification(
        self, old_id: ValueId, semantic: FloatIsFinite | FloatIsInf | FloatIsPosInf | FloatIsNegInf, a: ValueId
    ) -> None:
        operator, output = self.float_lowerer.classification_lowering(semantic)
        base, sign = _collapse_signs(self.hir.nodes, a)
        self.remap[old_id] = self.builder.operation(
            _select_hardware(semantic, operator), [self.remap[base]], [sign], output_conditioner=output
        )

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
            case Operation() if _sign_of(node) is not None:
                return True
            case Operation() as operation:
                if old_id in self.context.fused_muls:
                    return True  # this product is contracted into an adjacent fma; it has no standalone MIR op
                if old_id in self.context.absorbed_roundings:
                    return True  # absorbed into an adjacent conversion's mode immediate
                plan = self.context.fma_plans.get(old_id)
                if plan is not None:
                    assert isinstance(operation.operator, FloatAdd)
                    self.context.remap[old_id] = self._emit_ffma(
                        operation.operator, plan.ma, plan.mb, plan.c, plan.product_sign
                    )
                    return True
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
        return self.context.builder.operation(
            _select_hardware(plan.semantic, operator), [self.context.remap[plan.operand]], [plan.sign]
        )

    def _lower_operation(self, node: Operation) -> ValueId | None:
        match node:
            case Operation(operator=FloatAdd() as semantic, operands=(a, b)):
                return self._lower_binary_float(semantic, require(self.context.ops.fadd, "fadd"), a, b)
            case Operation(operator=FloatMul() as semantic, operands=(a, b)):
                return self._lower_binary_float(semantic, require(self.context.ops.fmul, "fmul"), a, b)
            case Operation(operator=FloatDiv() as semantic, operands=(a, b)):
                return self._lower_binary_float(semantic, require(self.context.ops.fdiv, "fdiv"), a, b)
            case Operation(operator=FloatMulPow2(k=k) as semantic, operands=(a,)):
                return self._lower_float_mul_pow2(semantic, a, k)
            case Operation(
                operator=(FloatRound() | FloatFloor() | FloatCeil() | FloatTrunc()) as semantic, operands=(a,)
            ):
                return self._lower_round(semantic, a)
            case Operation(operator=FloatExp2() as semantic, operands=(a,)):
                return self._lower_unary_pooled(semantic, require(self.context.ops.fexp2, "fexp2"), a)
            case Operation(operator=FloatLog2() as semantic, operands=(a,)):
                return self._lower_unary_pooled(semantic, require(self.context.ops.flog2, "flog2"), a)
            case Operation(operator=(FloatSinTurns() | FloatCosTurns()) as semantic, operands=(a,)):
                return self._lower_sincos(semantic, a)
            case Operation(operator=FloatSqrt() as semantic, operands=(a,)):
                base, sign = _collapse_signs(self.context.hir.nodes, a)
                return self._sqrt_via_exp2_log2(semantic, self.context.remap[base], sign)
            case Operation(operator=FloatAtan2Turns() as semantic, operands=(y, x)):
                return self._lower_atan2(semantic, y, x)
            case Operation(operator=FloatHypot2() as semantic, operands=(y, x)):
                return self._lower_hypot2_naive(semantic, y, x)  # a fusible hypot is intercepted in lower_node
            case Operation(operator=(FloatMin() | FloatMax()) as semantic, operands=(a, b)):
                return self._lower_minmax(semantic, a, b)
            case Operation(operator=FloatFma() as semantic, operands=(a, b, c)):
                return self._emit_ffma(semantic, a, b, c, FloatSignControl())
            case Operation(operator=BoolToFloat() as semantic, operands=(a,)):
                # ``float(cond)`` crosses from the boolean bank into the wide bank; a NOT chain folds into the
                # operand conditioner.
                base, inversion = _collapse_bool_inversions(self.context.hir.nodes, a)
                return self.context.builder.operation(
                    _select_hardware(semantic, BoolToFloatOperator(self.context.float_format)),
                    [self.context.remap[base]],
                    [inversion],
                )
            case Operation(operator=IntToFloat() as semantic, operands=(a,)):
                return self.context.builder.operation(
                    _select_hardware(semantic, require(self.context.ops.ffromint, "ffromint")),
                    [self.context.remap[a]],
                    [IntIdentity()],
                )
            case Operation(operator=FloatSelect() as semantic, operands=(cond, a, b)):
                # The if-conversion mux: arm signs and a condition NOT chain fold into the operand conditioners
                # (``x if c else -x`` and ``a if not c else b`` cost no hardware beyond the mux itself).
                base_c, inv_c = _collapse_bool_inversions(self.context.hir.nodes, cond)
                base_a, sign_a = _collapse_signs(self.context.hir.nodes, a)
                base_b, sign_b = _collapse_signs(self.context.hir.nodes, b)
                return self.context.builder.operation(
                    _select_hardware(semantic, SelectOperator(ScalarFloatType(self.context.float_format))),
                    [self.context.remap[base_c], self.context.remap[base_a], self.context.remap[base_b]],
                    [inv_c, sign_a, sign_b],
                )
            case _:
                return None

    def _lower_binary_float(
        self, semantic: Operator, hardware: FloatHardwareOperator, a: ValueId, b: ValueId, output_port: int = 0
    ) -> ValueId:
        # Each operand's sign chain folds onto its conditioner, applied before the op (min(-a, b) feeds the sorter -a).
        base_a, sign_a = _collapse_signs(self.context.hir.nodes, a)
        base_b, sign_b = _collapse_signs(self.context.hir.nodes, b)
        return self.context.builder.operation(
            _select_hardware(semantic, hardware),
            [self.context.remap[base_a], self.context.remap[base_b]],
            [sign_a, sign_b],
            output_port=output_port,
        )

    def _emit_ffma(
        self, semantic: Operator, a: ValueId, b: ValueId, c: ValueId, product_sign: FloatSignControl
    ) -> ValueId:
        """
        y = product_sign(a*b) + c as one ffma (explicit math.fma passes an identity product_sign).
        The product sign distributes onto the multiplier operands -- negation onto a only (-(a*b) = (-a)*b),
        absolute onto both (|a*b| = |a||b|) -- composed with each operand's own folded chain; c keeps its own.
        The raise fires only for explicit math.fma, since a contraction plan exists only when ffma is configured.
        """
        operator = require(self.context.ops.ffma, "ffma")
        base_a, sign_a = _collapse_signs(self.context.hir.nodes, a)
        base_b, sign_b = _collapse_signs(self.context.hir.nodes, b)
        base_c, sign_c = _collapse_signs(self.context.hir.nodes, c)
        cond_a = sign_a.then(product_sign)
        cond_b = sign_b.then(FloatSignControl(absolute=product_sign.absolute))
        return self.context.builder.operation(
            _select_hardware(semantic, operator),
            [self.context.remap[base_a], self.context.remap[base_b], self.context.remap[base_c]],
            [cond_a, cond_b, sign_c],
        )

    def _lower_round(self, semantic: FloatRound | FloatFloor | FloatCeil | FloatTrunc, a: ValueId) -> ValueId:
        mode = _ROUND_MODE_OF[type(semantic)]
        return self._lower_unary_pooled(
            semantic, require(self.context.ops.fround, "fround"), a, immediates=(int(mode),)
        )

    def _lower_unary_pooled(
        self,
        semantic: Operator,
        operator: FloatHardwareOperator,
        a: ValueId,
        immediates: tuple[int, ...] = (),
    ) -> ValueId:
        # Sign chain folds onto the operand (applied before the op): floor(-x)/exp2(-x) feed -x, not -floor(x)/-exp2(x).
        base, sign = _collapse_signs(self.context.hir.nodes, a)
        return self.context.builder.operation(
            _select_hardware(semantic, operator),
            [self.context.remap[base]],
            [sign],
            immediates=immediates,
        )

    def _lower_minmax(self, semantic: FloatMin | FloatMax, a: ValueId, b: ValueId) -> ValueId:
        # min taps the low output port, max the high one; a min and a max over one pair fuse into one sorter firing.
        operator = require(self.context.ops.fsort, "fsort")
        return self._lower_binary_float(
            semantic, operator, a, b, output_port=0 if isinstance(semantic, FloatMin) else 1
        )

    def _lower_sincos(self, semantic: FloatSinTurns | FloatCosTurns, a: ValueId) -> ValueId:
        # zkf_sincos already counts in turns, so the operand needs no scaling here at all -- whatever conversion the
        # kernel's angle wanted was stated in HIR and folded there. The sign rides the core's own operand, so one
        # value serves sin(-x)/cos(-x) and a sin+cos over one argument fuse into one firing.
        operator = require(self.context.ops.fsincos, "fsincos")
        base, sign = _collapse_signs(self.context.hir.nodes, a)
        return self.context.builder.operation(
            _select_hardware(semantic, operator),
            [self.context.remap[base]],
            [sign],
            output_port=0 if isinstance(semantic, FloatSinTurns) else 1,
        )

    def _lower_atan2(self, semantic: FloatAtan2Turns, y: ValueId, x: ValueId) -> ValueId:
        # zkf_atan2 returns theta in turns, which is what the operator now means. Its magnitude port is tapped only by
        # a fusible adjacent hypot (intercepted in lower_node).
        return self._lower_binary_float(semantic, require(self.context.ops.fatan2, "fatan2"), y, x)

    def _lower_fused_hypot(self, atan2_id: ValueId) -> ValueId:
        # Emitting the fatan2 firing from the ATAN2's own operands/signs makes the two collapse into one CORDIC; the
        # magnitude is symmetric and sign-invariant, so the hypot's own operand order/signs are immaterial.
        node = self.context.hir.nodes[atan2_id]
        assert isinstance(node, Operation)
        y, x = node.operands
        operator = self.context.ops.fatan2
        assert operator is not None  # only reached for a planned fusion, which exists only when fatan2 is configured
        return self._lower_binary_float(node.operator, operator, y, x, output_port=1)

    def _lower_hypot2_naive(self, semantic: FloatHypot2, y: ValueId, x: ValueId) -> ValueId:
        """
        Standalone hypot has no dedicated primitive, so it uses h*sqrt((x/h)^2 + (y/h)^2), h=max(|x|,|y|).
        The h==0 and h==inf cases are semantically valid but would feed 0/0 or inf/inf into fdiv causing err latch.
        The mux keeps those internal divisions finite while the surrounding formula still returns 0 or inf.
        """
        ops = self.context.ops
        fsort = require(ops.fsort, "fsort")
        fmul = require(ops.fmul, "fmul")
        base_y, sign_y = _collapse_signs(self.context.hir.nodes, y)
        base_x, sign_x = _collapse_signs(self.context.hir.nodes, x)
        my, mx = self.context.remap[base_y], self.context.remap[base_x]
        absolute = FloatSignControl(absolute=True)
        h = self.context.builder.operation(
            _select_hardware(semantic, fsort), [mx, my], [absolute, absolute], output_port=1
        )
        bypass = self._bool_or(self._float_eq_zero(h, FloatSignControl()), self._float_isinf(h))
        safe_h = self._float_select(bypass, self._lower_float_const(1.0), h)
        xn = self.context.builder.operation(
            _select_hardware(semantic, require(ops.fdiv, "fdiv")), [mx, safe_h], [sign_x, FloatSignControl()]
        )
        yn = self.context.builder.operation(
            _select_hardware(semantic, require(ops.fdiv, "fdiv")), [my, safe_h], [sign_y, FloatSignControl()]
        )
        squares = self._mir_op(
            semantic,
            require(ops.fadd, "fadd"),
            [self._mir_op(semantic, fmul, [xn, xn]), self._mir_op(semantic, fmul, [yn, yn])],
        )
        return self._mir_op(semantic, fmul, [h, self._sqrt_via_exp2_log2(semantic, squares, FloatSignControl())])

    def _sqrt_via_exp2_log2(self, semantic: Operator, operand: ValueId, conditioner: FloatSignControl) -> ValueId:
        """
        sqrt is expanded as exp2(log2(x)*0.5) until a native sqrt primitive exists.
        Zero is a valid sqrt input but a pole for log2, so the log sees 1.0 and the result is muxed back to 0.0.
        Negative nonzero inputs are not sanitized, so they still reach log2 and report the domain error.
        """
        ops = self.context.ops
        is_zero = self._float_eq_zero(operand, conditioner)
        zero = self._lower_float_const(0.0)
        safe_operand = self._float_select(is_zero, self._lower_float_const(1.0), operand, sign_b=conditioner)
        log = self.context.builder.operation(
            _select_hardware(semantic, require(ops.flog2, "flog2")),
            [safe_operand],
            [FloatSignControl()],
        )
        half = self._emit_scale_pow2(semantic, log, FloatSignControl(), -1)
        sqrt = self._mir_op(semantic, require(ops.fexp2, "fexp2"), [half])
        return self._float_select(is_zero, zero, sqrt)

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

    def _float_eq_zero(self, operand: ValueId, conditioner: FloatSignControl) -> ValueId:
        fcmp = require(self.context.ops.fcmp, "fcmp")
        port, inversion = fcmp.tap_of(_RELATION_OF[FloatEqual])
        return self.context.builder.operation(
            _select_hardware(FloatEqual(), fcmp),
            [operand, self._lower_float_const(0.0)],
            [conditioner, FloatSignControl()],
            output_port=port,
            output_conditioner=inversion,
        )

    def _float_isinf(self, operand: ValueId, conditioner: FloatSignControl = FloatSignControl()) -> ValueId:
        semantic = FloatIsInf()
        operator, output = self.classification_lowering(semantic)
        return self.context.builder.operation(
            _select_hardware(semantic, operator), [operand], [conditioner], output_conditioner=output
        )

    def _bool_or(self, a: ValueId, b: ValueId) -> ValueId:
        return self.context.builder.operation(
            _select_hardware(BoolOr(), BoolOrOperator()), [a, b], [BoolInversion(), BoolInversion()]
        )

    def _float_select(
        self,
        cond: ValueId,
        a: ValueId,
        b: ValueId,
        *,
        sign_a: FloatSignControl = FloatSignControl(),
        sign_b: FloatSignControl = FloatSignControl(),
    ) -> ValueId:
        return self.context.builder.operation(
            _select_hardware(FloatSelect(), SelectOperator(ScalarFloatType(self.context.float_format))),
            [cond, a, b],
            [BoolInversion(), sign_a, sign_b],
        )

    def _mir_op(
        self, semantic: Operator, hardware: HardwareOperator, operands: list[ValueId], output_port: int = 0
    ) -> ValueId:
        # Operands are already-lowered MIR ids taking identity conditioners -- a decomposition's interior.
        return self.context.builder.operation(
            _select_hardware(semantic, hardware),
            operands,
            [FloatSignControl()] * len(operands),
            output_port=output_port,
        )

    def _lower_float_mul_pow2(self, semantic: Operator, a: ValueId, k: int) -> ValueId:
        base, sign = _collapse_signs(self.context.hir.nodes, a)
        return self._emit_scale_pow2(semantic, self.context.remap[base], sign, k)

    def _emit_scale_pow2(self, semantic: Operator, operand: ValueId, sign: FloatSignControl, k: int) -> ValueId:
        """
        The exponent rides as a materialized integer constant, clamped rather than refused: any count past the int
        format already lies far beyond the float's dynamic range, where the scaler rails or flushes identically.
        """
        ifmt = self.context.int_format
        exponent = self.context.builder.int_const(ifmt.saturate(k), ScalarIntType(ifmt))
        return self.context.builder.operation(
            _select_hardware(semantic, require(self.context.ops.fmul_ilog2, "fmul_ilog2")),
            [operand, exponent],
            [sign, IntIdentity()],
        )

    def lower_output(self, name: str, value: ValueId) -> bool:
        base, sign = _collapse_signs(self.context.hir.nodes, value)
        if not isinstance(self.context.hir.nodes[base].type, HirFloatType):
            return False
        self.context.builder.float_output(name, self.context.remap[base], sign)
        return True

    def lower_state_slot(self, slot: StateSlot) -> bool:
        base, sign = _collapse_signs(self.context.hir.nodes, slot.live_out)
        if not isinstance(self.context.hir.nodes[base].type, HirFloatType):
            return False
        assert isinstance(slot.reset_value, FloatConst), "the gate owns a slot whose reset is of the wrong family"
        self.context.builder.float_state_slot(slot.name, slot.reset_value.value, self.context.remap[base], sign)
        return True


class _IntLowerer:
    """
    The integer dual of :class:`_FloatLowerer`, owning every operation whose RESULT is an integer. Its operands never
    carry a folded sideband: an integer port conditions with the identity alone, so ``ineg`` and ``iabs`` are hardware
    where their float counterparts are free. Integer operators are never optional, so only the two conversions that
    cross into the float half go through :func:`require`.
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
            case Operation(operator=IntAdd() as semantic, operands=(a, b)):
                return self._emit(semantic, IAddOperator(fmt), a, b)
            case Operation(operator=IntSub() as semantic, operands=(a, b)):
                return self._emit(semantic, ISubOperator(fmt), a, b)
            case Operation(operator=IntMul() as semantic, operands=(a, b)):
                return self._emit(semantic, self.context.ops.imul, a, b)
            case Operation(operator=IntMulPow2() as semantic, operands=(a,)):
                return self._scale_by_pow2(semantic, a)
            case Operation(operator=IntNeg() as semantic, operands=(a,)):
                return self._negate(semantic, self.context.remap[a])
            case Operation(operator=IntAbs() as semantic, operands=(a,)):
                return self._emit(semantic, IAbsOperator(fmt), a)
            # The quotient and the remainder are two taps of one divider: written from the same operands with the same
            # conditioners, they share a MIR intern key up to the port and fuse into a single firing at LIR build.
            case Operation(operator=IntDivFloor() as semantic, operands=(a, b)):
                return self._emit(semantic, IDivOperator(fmt), a, b, output_port=0)
            case Operation(operator=IntMod() as semantic, operands=(a, b)):
                return self._emit(semantic, IDivOperator(fmt), a, b, output_port=1)
            case Operation(operator=(IntShiftLeft() | IntShiftRight()) as semantic, operands=(a, b)):
                return self._lower_shift(semantic, a, b)
            case Operation(operator=IntBwAnd() as semantic, operands=(a, b)):
                return self._emit(semantic, IntBwAndOperator(fmt), a, b)
            case Operation(operator=IntBwOr() as semantic, operands=(a, b)):
                return self._emit(semantic, IntBwOrOperator(fmt), a, b)
            case Operation(operator=IntBwXor() as semantic, operands=(a, b)):
                return self._emit(semantic, IntBwXorOperator(fmt), a, b)
            case Operation(operator=IntBwNot() as semantic, operands=(a,)):
                return self._emit(semantic, IntBwNotOperator(fmt), a)
            case Operation(operator=IntSelect() as semantic, operands=(cond, a, b)):
                # The integer if-conversion mux: only the condition folds (``a if not c else b`` costs no extra gate).
                base_c, inv_c = _collapse_bool_inversions(self.context.hir.nodes, cond)
                return self.context.builder.operation(
                    _select_hardware(semantic, SelectOperator(self.int_type)),
                    [self.context.remap[base_c], self.context.remap[a], self.context.remap[b]],
                    [inv_c, IntIdentity(), IntIdentity()],
                )
            case Operation(operator=BoolToInt() as semantic, operands=(a,)):
                base, inversion = _collapse_bool_inversions(self.context.hir.nodes, a)
                return self.context.builder.operation(
                    _select_hardware(semantic, BoolToIntOperator(fmt)), [self.context.remap[base]], [inversion]
                )
            case Operation(operator=FloatToInt() as semantic, operands=(a,)):
                return self._lower_to_int(semantic, a)
            case _:
                return None

    def _lower_to_int(self, semantic: FloatToInt, a: ValueId) -> ValueId:
        """
        ``int(x)`` truncates toward zero, and a rounding it reads becomes its mode instead of a module of its own.
        The surviving operand's sign chain folds onto the float port, applied before the rounding as the source has it.
        """
        mode, operand = _absorbed_rounding(self.context.hir, a) or (RoundMode.TRUNC, a)
        base, sign = _collapse_signs(self.context.hir.nodes, operand)
        return self.context.builder.operation(
            _select_hardware(semantic, require(self.context.ops.ftoint, "ftoint")),
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
        shifter is tapped on its raw reading, because ``<<`` drops what leaves the word rather than saturating. Both
        modules clamp the amount at the word, which is where the two readings of an unbounded count meet -- a left
        shift past the word answers zero and a right shift past it answers the sign fill, as Python's own unbounded
        shift does once the word truncates it.
        """
        fmt = self.context.int_format
        hardware = IShrOperator(fmt) if isinstance(semantic, IntShiftRight) else IShlOperator(fmt)
        return self._emit(semantic, hardware, a, count)

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
            _select_hardware(semantic, IntShiftConstOperator(self.context.int_format, shamt)),
            [self.context.remap[a]],
            [IntIdentity()],
        )

    def _scale_by_pow2(self, semantic: IntMulPow2, a: ValueId) -> ValueId:
        """
        The left shifter's OTHER reading: ``prod`` saturates where ``shft`` lets the high bits fall off the word, and
        saturating is what a multiplication does. The count is unbounded where the word is not, so it clamps at the
        width -- past that every count rails the same operand the same way, and only zero survives either.
        """
        fmt = self.context.int_format
        return self.context.builder.operation(
            _select_hardware(semantic, IShlOperator(fmt)),
            [self.context.remap[a], self._const(min(semantic.k, fmt.width))],
            [IntIdentity(), IntIdentity()],
            output_port=1,
        )

    def _negate(self, semantic: Operator, value: ValueId) -> ValueId:
        """``0 - x``: there is no negation module, and the subtractor saturates ``-MIN`` correctly."""
        return self.context.builder.operation(
            _select_hardware(semantic, ISubOperator(self.context.int_format)),
            [self._const(0), value],
            [IntIdentity(), IntIdentity()],
        )

    def _const(self, value: int) -> ValueId:
        return self.context.builder.int_const(value, self.int_type)

    def _emit(
        self, semantic: Operator, hardware: HardwareOperator, *operands: ValueId, output_port: int = 0
    ) -> ValueId:
        return self.context.builder.operation(
            _select_hardware(semantic, hardware),
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


def lower(hir: Hir, ops: OpConfig, ifconv_max_ops: int) -> Mir:
    """
    Optimize the front end's HIR against this machine, judge what survives, then select hardware operators for it and
    fold semantic signs onto MIR sign controls.

    Optimization is not the caller's to run. A substituted constant cascades, and can leave another count constant for
    the next round, so the passes run again after every substitution and the judgement waits for the last graph: a
    caller who optimized first would also have judged first, convicting whatever a later round erases. What this
    machine knows is told to the graph before any of it: the trigonometric unit up front, since it is the optimizer's
    input rather than its consumer, and the word width from within the fixpoint, since only a fold can reveal a count.

    Semantic sign operations are never emitted as standalone scheduled operators. Exact power-of-two scaling selects
    ``fmul_ilog2`` with the exponent materialized as an integer constant; no exponent is refused.
    """
    hir = trig_abi(hir)
    hir = optimize(hir, ifconv_max_ops)
    rounds = left_shifts(hir) + 1  # every substitution deletes one, and no pass mints one
    while (substituted := specialize(hir, ops.int_format)) is not None:
        rounds -= 1
        assert rounds > 0, "specialization unsettled what an earlier round settled"
        hir = optimize(substituted, ifconv_max_ops)
    _logger.info(
        "Optimized HIR:\n\tinputs=%s\n\toutputs=%s\n\tnodes=%d\n\tblocks=%d",
        hir.input_ids,
        hir.outputs,
        len(hir.nodes),
        len(hir.blocks),
    )
    refuse(hir)
    return _LoweringContext(hir, ops).run()
