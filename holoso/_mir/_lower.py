"""Lower optimized HIR to selected MIR."""

import logging
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Sequence

from .._errors import SynthesisError
from .._hir import (
    BoolAnd,
    BoolNot,
    BoolOr,
    BoolSelect,
    BoolToFloat,
    BoolToInt,
    BoolType as HirBoolType,
    BoolXor,
    Branch,
    Const,
    FloatAdd,
    FloatAtan2Turns,
    FloatComparison,
    FloatCosTurns,
    FloatDiv,
    FloatExp2,
    FloatFma,
    FloatILog2,
    FloatIsFinite,
    FloatIsInf,
    FloatIsNegInf,
    FloatIsPosInf,
    FloatLog2,
    FloatMax,
    FloatMin,
    FloatMul,
    FloatMulPow2,
    FloatMulPow2Dynamic,
    FloatRounding,
    FloatSelect,
    FloatSinTurns,
    FloatSqrt,
    FloatToBool,
    FloatToInt,
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
    IntDivFloor,
    IntMod,
    IntMul,
    IntMulPow2,
    IntNeg,
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
    Phi,
    Rounding,
    Ret,
    StateRead,
    Terminator,
    Type as HirType,
    const_value,
    optimize,
    reverse_postorder,
)
from ._refuse import refuse
from ._specialize import constant_shift_count, left_shifts, specialize
from ._rescale import rescale
from ._trig_abi import trig_abi
from .._util import ValueId
from .._operators import (
    BoolAndPrimitive,
    BoolInversion,
    BoolOrPrimitive,
    BoolToFloatPrimitive,
    BoolToIntPrimitive,
    BoolXorPrimitive,
    FAddPrimitive,
    FAtan2Primitive,
    FCmpPrimitive,
    FCosPrimitive,
    FHypotPrimitive,
    FMaxPrimitive,
    FMinPrimitive,
    FDivPrimitive,
    FExp2Primitive,
    FFmaPrimitive,
    FFromIntPrimitive,
    FILog2Primitive,
    FloatIsFinitePrimitive,
    FloatIsNegInfPrimitive,
    FloatIsPosInfPrimitive,
    FloatToBoolPrimitive,
    FLog2Primitive,
    FMulILog2Primitive,
    FMulPrimitive,
    FRintPrimitive,
    FRintToIntPrimitive,
    FSinPrimitive,
    FSqrtPrimitive,
    IAbsPrimitive,
    IAddPrimitive,
    IFloorDivPrimitive,
    IModPrimitive,
    IMulPrimitive,
    IntBwAndPrimitive,
    IntBwNotPrimitive,
    IntBwOrPrimitive,
    IntBwXorPrimitive,
    IntIdentity,
    IntShiftConstPrimitive,
    IntToBoolPrimitive,
    IPopcntPrimitive,
    IShlPrimitive,
    IShrPrimitive,
    ICmpPrimitive,
    ISubPrimitive,
    OpConfig,
    PooledPrimitive,
    Primitive,
    RoundMode,
    SelectPrimitive,
)
from .._type import (
    BoolType as ScalarBoolType,
    FloatType as ScalarFloatType,
    IntFormat,
    IntType as ScalarIntType,
    ScalarType,
)
from ._ir import Mir, MirBuilder, MirOperation
from ._mirrored_comparisons import plan_mirrored_comparisons
from ._hypot import count as hypot_count, expand_unfused, plan_magnitude_fusions
from ._options import MirOptions
from ._fma import expand_fmas
from ._signs import collapse_bool_inversions, collapse_conditioner, collapse_signs, sign_of

_logger = logging.getLogger(__name__)

_ROUND_MODE_OF: dict[Rounding, RoundMode] = {
    Rounding.NEAREST_EVEN: RoundMode.NEAREST_EVEN,
    Rounding.FLOOR: RoundMode.FLOOR,
    Rounding.CEIL: RoundMode.CEIL,
    Rounding.TRUNC: RoundMode.TRUNC,
}


def _folded_shift_counts(hir: Hir) -> set[ValueId]:
    """
    Decided before any block, because constants are lowered entry-globally. A count the shift below answers from is
    consumed as the shift primitive's parameter, not held as a value -- and a count something ELSE reads is an ordinary
    value, which is where one too wide for the machine is still refused.
    """
    folds = Counter(
        node.operands[1]
        for node in hir.nodes.values()
        if isinstance(node, Operation)
        and isinstance(node.operator, (IntShiftLeft, IntShiftRight))
        and constant_shift_count(hir, node.operands[1]) is not None
    )
    use_counts = hir.use_counts()
    return {vid for vid, taken in folds.items() if taken == use_counts[vid]}


class _LoweringContext:
    def __init__(self, hir: Hir, ops: OpConfig) -> None:
        self._hir = hir
        self._ops = ops
        self._builder = MirBuilder(ops.float_format, ops.int_format)
        self._remap: dict[ValueId, ValueId] = {}
        fused_hypots = plan_magnitude_fusions(hir, ops)
        # The derivation expanded every unfused magnitude, so the survivors all ride an atan2's magnitude port.
        assert hypot_count(hir) == len(fused_hypots)
        # Values their readers consume whole, so no MIR value stands for them: a sign or NOT chain folds into every
        # reader's conditioner, a constant shift count into its shift primitive's parameter.
        self._absorbed = {
            *(vid for vid, node in hir.nodes.items() if isinstance(node, Operation) and _is_sideband(node)),
            *_folded_shift_counts(hir),
        }
        self._lowerers: dict[type[HirType], _FamilyLowerer] = {
            HirFloatType: _FloatLowerer(hir, ops, self._builder, self._remap, fused_hypots),
            HirIntType: _IntLowerer(hir, ops, self._builder, self._remap),
            HirBoolType: _BoolLowerer(hir, ops, self._builder, self._remap, plan_mirrored_comparisons(hir)),
        }

    def _scalar_type(self, hir_type: HirType) -> ScalarType:
        match hir_type:
            case HirFloatType():
                return ScalarFloatType(self._ops.float_format)
            case HirIntType():
                return ScalarIntType(self._ops.int_format)
            case HirBoolType():
                return ScalarBoolType()
        assert False, f"no scalar type for {hir_type!r}"

    def run(self) -> Mir:
        for _ in self._hir.blocks:
            self._builder.block()  # preserve block ids 0..n-1
        self._builder.position_at(self._hir.entry)
        for vid in self._hir.input_ids:
            self._lower_node(vid, self._hir.nodes[vid])
        for vid in sorted(self._hir.nodes):
            if isinstance(self._hir.nodes[vid], (Const, StateRead)):
                self._lower_node(vid, self._hir.nodes[vid])
        # Reverse postorder, so every operand and phi arm is remapped before use even when branches nest.
        blocks_by_id = {block.id: block for block in self._hir.blocks}
        deferred: list[ValueId] = []  # loop-header phis whose latch arm is a body value lowered later; closed below
        for bid in reverse_postorder(self._hir):
            block = blocks_by_id[bid]
            self._builder.position_at(bid)
            for vid in block.phis:
                self._lower_phi(vid, self._hir.nodes[vid], deferred)
            for vid in block.operations:
                self._lower_node(vid, self._hir.nodes[vid])
            self._seal(block.terminator)
        for vid in deferred:
            self._close_phi(vid, self._hir.nodes[vid])
        for out in self._hir.outputs:
            base, conditioner = collapse_conditioner(self._hir.nodes, out.value)
            self._builder.output(out.name, self._remap[base], conditioner)
        for slot in self._hir.state_slots:
            base, conditioner = collapse_conditioner(self._hir.nodes, slot.live_out)
            self._builder.state_slot(slot.name, const_value(slot.reset_value), self._remap[base], conditioner)
        mir = self._builder.finish()
        for vid, node in self._hir.nodes.items():
            if isinstance(node, Operation):
                selected = mir.nodes[self._remap[vid]] if vid in self._remap else None
                if isinstance(selected, MirOperation) and isinstance(selected.primitive, PooledPrimitive):
                    assert not (node.operator.speculatable and selected.primitive.operator.error_ports)
        return mir

    def _seal(self, terminator: Terminator) -> None:
        match terminator:
            case Jump(target=target):
                self._builder.jump(target)
            case Branch(cond=cond, if_true=if_true, if_false=if_false):
                # A NOT on the condition is free: take the complementary target instead of inverting the register.
                base, inversion = collapse_bool_inversions(self._hir.nodes, cond)
                if inversion.invert:
                    if_true, if_false = if_false, if_true
                self._builder.branch(self._remap[base], if_true, if_false)
            case Ret():
                self._builder.ret()

    def _lower_phi(self, old_id: ValueId, node: Node, deferred: list[ValueId]) -> None:
        # Each arm folds its type's OWN sideband chain into the arm conditioner, applied by the merge install: a sign
        # chain on a float arm (a branch assigning `-x`/`abs(x)`), a NOT chain on a boolean arm (`f = not g`).
        # A loop-header phi's latch arm is a body value lowered after the header: open the phi with its available arms
        # now (so the body can reference it) and close it once every block is lowered.
        assert isinstance(node, Phi)
        bases = [(pred, collapse_conditioner(self._hir.nodes, value)) for pred, value in node.arms]
        scalar_type = self._scalar_type(node.type)
        known = [(pred, self._remap[base], conditioner) for pred, (base, conditioner) in bases if base in self._remap]
        self._remap[old_id] = self._builder.phi(scalar_type, known)
        if len(known) < len(bases):
            deferred.append(old_id)

    def _close_phi(self, old_id: ValueId, node: Node) -> None:
        assert isinstance(node, Phi)
        arms = [
            (pred, self._remap[base], conditioner)
            for pred, value in node.arms
            for base, conditioner in [collapse_conditioner(self._hir.nodes, value)]
        ]
        self._builder.set_phi_arms(self._remap[old_id], arms)

    def _lower_node(self, old_id: ValueId, node: Node) -> None:
        if old_id in self._absorbed:
            return
        match node:
            case InPort(name=name, type=hir_type):
                self._remap[old_id] = self._builder.input(name, self._scalar_type(hir_type))
            case StateRead(slot=slot, type=hir_type):
                self._remap[old_id] = self._builder.state_read(slot, self._scalar_type(hir_type))
            case Const():
                self._remap[old_id] = self._builder.const(const_value(node), self._scalar_type(node.type))
            case Operation():
                self._remap[old_id] = self._lowerers[type(node.type)].lower_operation(old_id, node)


def _is_sideband(node: Operation) -> bool:
    return sign_of(node) is not None or isinstance(node.operator, BoolNot)


class _FamilyLowerer(ABC):
    """
    One per scalar family, owning every operation whose RESULT is of that family. Every operand folds its own
    family's sideband chain into its conditioner, whichever family reads it: `min(-a, b)` feeds the comparator `-a`,
    `band(~a, b)` is one gate, and an integer operand folds nothing.
    """

    def __init__(self, hir: Hir, ops: OpConfig, builder: MirBuilder, remap: dict[ValueId, ValueId]) -> None:
        self._hir = hir
        self._ops = ops
        self._builder = builder
        self._remap = remap

    @abstractmethod
    def lower_operation(self, old_id: ValueId, node: Operation) -> ValueId: ...

    def _emit(
        self,
        primitive: Primitive,
        operands: Sequence[ValueId],
        output_inversion: BoolInversion = BoolInversion(),
    ) -> ValueId:
        bases = [collapse_conditioner(self._hir.nodes, operand) for operand in operands]
        return self._builder.operation(
            primitive,
            [self._remap[base] for base, _ in bases],
            [conditioner for _, conditioner in bases],
            output_inversion,
        )


class _FloatLowerer(_FamilyLowerer):
    def __init__(
        self,
        hir: Hir,
        ops: OpConfig,
        builder: MirBuilder,
        remap: dict[ValueId, ValueId],
        fused_hypots: dict[ValueId, ValueId],
    ) -> None:
        super().__init__(hir, ops, builder, remap)
        self._fused_hypots = fused_hypots

    def lower_operation(self, old_id: ValueId, node: Operation) -> ValueId:
        atan2_id = self._fused_hypots.get(old_id)
        if atan2_id is not None:
            # The atan2's own operands make the two collapse into one CORDIC firing; the magnitude is symmetric and
            # sign-invariant, so the hypot's own operand order and signs are immaterial.
            atan2 = self._hir.nodes[atan2_id]
            assert isinstance(atan2, Operation)
            return self._emit(FHypotPrimitive(self._ops.fcordic), atan2.operands)
        ops, fmt = self._ops, self._ops.float_format
        match node:
            case Operation(operator=FloatAdd(), operands=operands):
                return self._emit(FAddPrimitive(ops.fadd), operands)
            case Operation(operator=FloatMul(), operands=operands):
                return self._emit(FMulPrimitive(ops.fmul), operands)
            case Operation(operator=FloatDiv(), operands=operands):
                return self._emit(FDivPrimitive(ops.fdivsqrt), operands)
            case Operation(operator=FloatMulPow2(k=k), operands=(a,)):
                # The exponent rides as a materialized integer constant, clamped rather than refused: any count past
                # the int format already lies far beyond the float's dynamic range, where the scaler rails or flushes
                # identically.
                base, sign = collapse_signs(self._hir.nodes, a)
                exponent = self._builder.const(ops.int_format.saturate(k), ScalarIntType(ops.int_format))
                return self._builder.operation(
                    FMulILog2Primitive(ops.fmul_ilog2), [self._remap[base], exponent], [sign, IntIdentity()]
                )
            case Operation(operator=FloatMulPow2Dynamic(), operands=operands):
                return self._emit(FMulILog2Primitive(ops.fmul_ilog2), operands)
            case Operation(operator=FloatRounding(rounding=rounding), operands=operands):
                # The float result of the rounder; the conversion of the same rounding of the same operand is its
                # integer result, and the two fuse into one firing.
                return self._emit(FRintPrimitive(ops.frint, _ROUND_MODE_OF[rounding]), operands)
            case Operation(operator=FloatExp2(), operands=operands):
                return self._emit(FExp2Primitive(ops.fexp2), operands)
            case Operation(operator=FloatLog2(), operands=operands):
                return self._emit(FLog2Primitive(ops.flog2), operands)
            case Operation(operator=(FloatSinTurns() | FloatCosTurns()) as semantic, operands=operands):
                # The CORDIC already counts in turns, so whatever conversion the kernel's angle wanted was stated in HIR
                # and folded there. The sign rides the core's own operand, so one value serves sin(-x)/cos(-x) and a
                # sin+cos over one argument fuse into one firing.
                rotation = FSinPrimitive if isinstance(semantic, FloatSinTurns) else FCosPrimitive
                return self._emit(rotation(ops.fcordic), operands)
            case Operation(operator=FloatSqrt(), operands=operands):
                return self._emit(FSqrtPrimitive(ops.fdivsqrt), operands)
            case Operation(operator=FloatAtan2Turns(), operands=operands):
                # Vectoring returns theta in turns, which is what the operator means.
                return self._emit(FAtan2Primitive(ops.fcordic), operands)
            case Operation(operator=(FloatMin() | FloatMax()) as semantic, operands=operands):
                # A min and a max over one pair, and the relations over it, fuse into one comparator firing.
                extreme = FMaxPrimitive if isinstance(semantic, FloatMax) else FMinPrimitive
                return self._emit(extreme(ops.fcmp), operands)
            case Operation(operator=FloatFma(), operands=operands):
                return self._emit(FFmaPrimitive(ops.ffma), operands)
            case Operation(operator=BoolToFloat(), operands=operands):
                return self._emit(BoolToFloatPrimitive(fmt), operands)
            case Operation(operator=IntToFloat(), operands=operands):
                return self._emit(FFromIntPrimitive(ops.ffromint), operands)
            case Operation(operator=FloatSelect(), operands=operands):
                return self._emit(SelectPrimitive(ScalarFloatType(fmt)), operands)
            case _:
                assert False, f"no float lowering for {node.operator.mnemonic!r}"


class _IntLowerer(_FamilyLowerer):
    """
    `ineg` and `iabs` are hardware where their float counterparts are free. Integer operators are never optional, so
    only the conversions that cross into the float half can find their operator unconfigured.
    """

    def lower_operation(self, old_id: ValueId, node: Operation) -> ValueId:
        ops, fmt = self._ops, self._ops.int_format
        match node:
            case Operation(operator=IntAdd(), operands=operands):
                return self._emit(IAddPrimitive(ops.iadds), operands)
            case Operation(operator=IntSub(), operands=operands):
                return self._emit(ISubPrimitive(ops.iadds), operands)
            case Operation(operator=IntMul(), operands=operands):
                return self._emit(IMulPrimitive(ops.imuls), operands)
            case Operation(operator=IntMulPow2(k=k), operands=(a,)):
                # HIR extracts a scaling by a power of two as an operation of its own, which a dedicated operator may
                # serve one day; today it is an ordinary multiplication by the power of two: the product rails where a
                # left shift would drop what leaves the word, and the shifter has no such reading. The factor is
                # therefore a number the word must hold, and a scaling whose factor it cannot is refused.
                return self._builder.operation(
                    IMulPrimitive(ops.imuls), [self._remap[a], self._const(1 << k)], [IntIdentity()] * 2
                )
            case Operation(operator=IntNeg(), operands=(a,)):
                # `0 - x`: there is no negation module, and the subtraction saturates `-MIN` correctly.
                return self._builder.operation(
                    ISubPrimitive(ops.iadds), [self._const(0), self._remap[a]], [IntIdentity()] * 2
                )
            case Operation(operator=IntAbs(), operands=operands):
                return self._emit(IAbsPrimitive(ops.iabss), operands)
            case Operation(operator=IntPopcount(), operands=operands):
                return self._emit(IPopcntPrimitive(ops.ipopcnt), operands)
            # The quotient and the remainder are two results of one drive of the divider: written from the same
            # operands, they fuse into a single firing at LIR build.
            case Operation(operator=IntDivFloor(), operands=operands):
                return self._emit(IFloorDivPrimitive(ops.idivs), operands)
            case Operation(operator=IntMod(), operands=operands):
                return self._emit(IModPrimitive(ops.idivs), operands)
            case Operation(operator=(IntShiftLeft() | IntShiftRight()) as semantic, operands=(a, count)):
                return self._lower_shift(semantic, a, count)
            case Operation(operator=IntBwAnd(), operands=operands):
                return self._emit(IntBwAndPrimitive(fmt), operands)
            case Operation(operator=IntBwOr(), operands=operands):
                return self._emit(IntBwOrPrimitive(fmt), operands)
            case Operation(operator=IntBwXor(), operands=operands):
                return self._emit(IntBwXorPrimitive(fmt), operands)
            case Operation(operator=IntBwNot(), operands=operands):
                return self._emit(IntBwNotPrimitive(fmt), operands)
            case Operation(operator=IntSelect(), operands=operands):
                return self._emit(SelectPrimitive(ScalarIntType(fmt)), operands)
            case Operation(operator=BoolToInt(), operands=operands):
                return self._emit(BoolToIntPrimitive(fmt), operands)
            case Operation(operator=FloatToInt(rounding=rounding), operands=operands):
                return self._emit(FRintToIntPrimitive(ops.frint, _ROUND_MODE_OF[rounding]), operands)
            case Operation(operator=FloatILog2(), operands=operands):
                return self._emit(FILog2Primitive(ops.filog2), operands)
            case _:
                assert False, f"no integer lowering for {node.operator.mnemonic!r}"

    def _lower_shift(self, semantic: IntShiftLeft | IntShiftRight, a: ValueId, count: ValueId) -> ValueId:
        """
        Each direction is a primitive of the one shifter, so neither negates its count to reach the other. The shifter
        clamps the amount at the word, which is where the two readings of an unbounded count meet -- a left shift
        past the word answers zero and a right shift past it answers the sign fill, as Python's own unbounded shift
        does once the word truncates it.

        A constant count needs no module and is never materialized.
        """
        constant = constant_shift_count(self._hir, count)
        if constant is None:
            shifter = (
                IShrPrimitive(self._ops.ishft)
                if isinstance(semantic, IntShiftRight)
                else IShlPrimitive(self._ops.ishft)
            )
            return self._emit(shifter, [a, count])
        width = self._ops.int_format.width
        assert constant > 0, "strength reduction elides a zero-count shift and the gate owns a negative one"
        if isinstance(semantic, IntShiftLeft):
            assert constant < width, "specialization owns a left shift past the word, whose answer is zero regardless"
            shamt = constant
        else:
            # Clamped here and not in HIR: past the top bit a right shift repeats the sign fill only for a value the
            # word already holds, and this operand is one where an HIR constant need not have been.
            shamt = -min(constant, width - 1)
        return self._emit(IntShiftConstPrimitive(self._ops.int_format, shamt), [a])

    def _const(self, value: int) -> ValueId:
        return self._builder.const(value, ScalarIntType(self._ops.int_format))


class _BoolLowerer(_FamilyLowerer):
    def __init__(
        self,
        hir: Hir,
        ops: OpConfig,
        builder: MirBuilder,
        remap: dict[ValueId, ValueId],
        mirrored_comparisons: dict[ValueId, ValueId],
    ) -> None:
        super().__init__(hir, ops, builder, remap)
        self._mirrored_comparisons = mirrored_comparisons

    def lower_operation(self, old_id: ValueId, node: Operation) -> ValueId:
        match node:
            case Operation(operator=FloatComparison() | IntComparison() as semantic, operands=operands):
                # A relation is one order flag with an optional inversion (both orderings are total and the flags
                # one-hot), so the relations over one operand pair tap the firing that orders it, beside a min, a max
                # or a difference of the pair. A comparison reading a sort's or a difference's operands the other
                # way round takes their order and asks for the mirrored relation.
                relation, host = semantic.relation, self._mirrored_comparisons.get(old_id)
                if host is not None:
                    host_node = self._hir.nodes[host]
                    assert isinstance(host_node, Operation)
                    relation, operands = relation.mirror, host_node.operands
                flag, inversion = (
                    FCmpPrimitive.of(self._ops.fcmp, relation)
                    if isinstance(semantic, FloatComparison)
                    else ICmpPrimitive.of(self._ops.iadds, relation)
                )
                return self._emit(flag, operands, output_inversion=inversion)
            case Operation(operator=IntToBool(), operands=operands):
                return self._emit(IntToBoolPrimitive(self._ops.int_format), operands)
            case Operation(operator=FloatIsFinite(), operands=operands):
                return self._emit(FloatIsFinitePrimitive(self._ops.float_format), operands)
            case Operation(operator=FloatIsInf(), operands=operands):
                return self._emit(
                    FloatIsFinitePrimitive(self._ops.float_format),
                    operands,
                    output_inversion=BoolInversion(invert=True),
                )
            case Operation(operator=FloatIsPosInf(), operands=operands):
                return self._emit(FloatIsPosInfPrimitive(self._ops.float_format), operands)
            case Operation(operator=FloatIsNegInf(), operands=operands):
                return self._emit(FloatIsNegInfPrimitive(self._ops.float_format), operands)
            case Operation(operator=BoolAnd(), operands=operands):
                return self._emit(BoolAndPrimitive(), operands)
            case Operation(operator=BoolOr(), operands=operands):
                return self._emit(BoolOrPrimitive(), operands)
            case Operation(operator=BoolXor(), operands=operands):
                return self._emit(BoolXorPrimitive(), operands)
            case Operation(operator=BoolSelect(), operands=operands):
                return self._emit(SelectPrimitive(ScalarBoolType()), operands)
            case Operation(operator=FloatToBool(), operands=operands):
                return self._emit(FloatToBoolPrimitive(self._ops.float_format), operands)
            case _:
                assert False, f"no boolean lowering for {node.operator.mnemonic!r}"


def _derive(hir: Hir, ops: OpConfig, ifconv_max_ops: int) -> Hir:
    """
    Everything this machine knows, told to the graph. The trigonometric unit is restated first, being the optimizer's
    input rather than its consumer. So is every fused multiply-add of a machine that cannot fuse, again after each
    magnitude is expanded, since its expansion spells more of them. The word width is told from within the fixpoint,
    since only a fold can reveal a count.
    """
    hir = optimize(expand_fmas(trig_abi(hir), ops), ifconv_max_ops)
    # Each rewrite deletes one of these and no pass grows either count: a magnitude is re-minted only in place of
    # the one it replaced.
    fuel = left_shifts(hir) + hypot_count(hir) + 1
    while True:
        rewritten = specialize(hir, ops.int_format) or expand_unfused(hir, ops)
        if rewritten is None:
            break
        fuel -= 1
        assert fuel > 0, "the derivation is not settling"
        hir = optimize(expand_fmas(rewritten, ops), ifconv_max_ops)
    return rescale(hir, ops)  # last: strength reduction would compose the pair it splits straight back


def _widest_word(options: MirOptions) -> int:
    return max(options.wint_min, options.float_format.wfull)


def _word(hir: Hir, options: MirOptions) -> int:
    # A family absent from the graph needs no room in the wide register, however the machine is configured. Node types
    # alone are complete: every operand is itself a node, and a slot's reset shares its live-out's type.
    floats = any(isinstance(node.type, HirFloatType) for node in hir.nodes.values())
    return _widest_word(options) if floats else options.wint_min


def lower(hir: Hir, options: MirOptions) -> Mir:
    """
    Optimize the front end's HIR against this machine, judge what survives, then select primitives for it and
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
            refuse(candidate)
        except SynthesisError as ex:
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
    return _LoweringContext(hir, ops).run()
