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
    BoolAndOperator,
    BoolInversion,
    BoolOrOperator,
    BoolToFloatOperator,
    BoolToIntOperator,
    BoolXorOperator,
    FloatClassificationOperator,
    FloatIsFiniteOperator,
    FloatIsNegInfOperator,
    FloatIsPosInfOperator,
    IntBwAndOperator,
    IntBwNotOperator,
    IntBwOrOperator,
    IntBwXorOperator,
    IntIdentity,
    IntShiftConstOperator,
    IntToBoolOperator,
    FloatToBoolOperator,
    HardwareOperator,
    OpConfig,
    PooledHardwareOperator,
    PortConditioner,
    RoundMode,
    SelectOperator,
    require,
)
from .._type import (
    BoolType as ScalarBoolType,
    FloatType as ScalarFloatType,
    IntFormat,
    IntType as ScalarIntType,
    ScalarType,
)
from ._ir import Mir, MirBuilder, MirOperation
from ._hypot import count as hypot_count, expand_unfused, plan_fusions
from ._options import MirOptions
from ._fma import contract_fmas
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
    consumed as an immediate, not held as a value -- and a count something ELSE reads is an ordinary value, which is
    where one too wide for the machine is still refused.
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
        self.hir = hir
        self.ops = ops
        self.builder = MirBuilder(ops.float_format, ops.int_format)
        self.remap: dict[ValueId, ValueId] = {}
        self.fused_hypots = plan_fusions(hir, ops)
        # The derivation expanded every unfused magnitude, so the survivors all ride an atan2's magnitude port.
        assert hypot_count(hir) == len(self.fused_hypots)
        # Values their readers consume whole, so no MIR value stands for them: a sign or NOT chain folds into every
        # reader's conditioner, a constant shift count into an immediate.
        self.absorbed = {
            *(vid for vid, node in hir.nodes.items() if isinstance(node, Operation) and _is_sideband(node)),
            *_folded_shift_counts(hir),
        }
        self.lowerers: dict[type[HirType], _FamilyLowerer] = {
            HirFloatType: _FloatLowerer(self),
            HirIntType: _IntLowerer(self),
            HirBoolType: _BoolLowerer(self),
        }

    def scalar_type(self, hir_type: HirType) -> ScalarType:
        match hir_type:
            case HirFloatType():
                return ScalarFloatType(self.ops.float_format)
            case HirIntType():
                return ScalarIntType(self.ops.int_format)
            case HirBoolType():
                return ScalarBoolType()
        raise AssertionError(f"no scalar type for {hir_type!r}")

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
            base, conditioner = collapse_conditioner(self.hir.nodes, out.value)
            self.builder.output(out.name, self.remap[base], conditioner)
        for slot in self.hir.state_slots:
            base, conditioner = collapse_conditioner(self.hir.nodes, slot.live_out)
            self.builder.state_slot(slot.name, const_value(slot.reset_value), self.remap[base], conditioner)
        mir = self.builder.finish()
        for vid, node in self.hir.nodes.items():
            if isinstance(node, Operation):
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
                base, inversion = collapse_bool_inversions(self.hir.nodes, cond)
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
        bases = [(pred, collapse_conditioner(self.hir.nodes, value)) for pred, value in node.arms]
        scalar_type = self.scalar_type(node.type)
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
            for base, conditioner in [collapse_conditioner(self.hir.nodes, value)]
        ]
        self.builder.set_phi_arms(self.remap[old_id], arms)

    def _lower_node(self, old_id: ValueId, node: Node) -> None:
        if old_id in self.absorbed:
            return
        match node:
            case InPort(name=name, type=hir_type):
                self.remap[old_id] = self.builder.input(name, self.scalar_type(hir_type))
            case StateRead(slot=slot, type=hir_type):
                self.remap[old_id] = self.builder.state_read(slot, self.scalar_type(hir_type))
            case Const():
                self.remap[old_id] = self.builder.const(const_value(node), self.scalar_type(node.type))
            case Operation():
                self.remap[old_id] = self.lowerers[type(node.type)].lower_operation(old_id, node)


def _is_sideband(node: Operation) -> bool:
    return sign_of(node) is not None or isinstance(node.operator, BoolNot)


class _FamilyLowerer(ABC):
    """
    One per scalar family, owning every operation whose RESULT is of that family. Every operand folds its own
    family's sideband chain into its conditioner, whichever family reads it: `min(-a, b)` feeds the sorter `-a`,
    `band(~a, b)` is one gate, and an integer operand folds nothing.
    """

    def __init__(self, context: _LoweringContext) -> None:
        self.context = context
        self.hir = context.hir
        self.ops = context.ops
        self.builder = context.builder
        self.remap = context.remap

    @abstractmethod
    def lower_operation(self, old_id: ValueId, node: Operation) -> ValueId: ...

    def emit(
        self,
        hardware: HardwareOperator,
        operands: Sequence[ValueId],
        output_port: int = 0,
        output_conditioner: PortConditioner | None = None,
        immediates: tuple[int, ...] = (),
    ) -> ValueId:
        bases = [collapse_conditioner(self.hir.nodes, operand) for operand in operands]
        return self.builder.operation(
            hardware,
            [self.remap[base] for base, _ in bases],
            [conditioner for _, conditioner in bases],
            output_port,
            output_conditioner,
            immediates,
        )


class _FloatLowerer(_FamilyLowerer):
    def lower_operation(self, old_id: ValueId, node: Operation) -> ValueId:
        atan2_id = self.context.fused_hypots.get(old_id)
        if atan2_id is not None:
            # The atan2's own operands make the two collapse into one CORDIC firing; the magnitude is symmetric and
            # sign-invariant, so the hypot's own operand order and signs are immaterial.
            atan2 = self.hir.nodes[atan2_id]
            assert isinstance(atan2, Operation)
            return self.emit(require(self.ops.fatan2, "fatan2"), atan2.operands, output_port=1)
        ops, fmt = self.ops, self.ops.float_format
        match node:
            case Operation(operator=FloatAdd(), operands=operands):
                return self.emit(require(ops.fadd, "fadd"), operands)
            case Operation(operator=FloatMul(), operands=operands):
                return self.emit(require(ops.fmul, "fmul"), operands)
            case Operation(operator=FloatDiv(), operands=operands):
                return self.emit(require(ops.fdiv, "fdiv"), operands)
            case Operation(operator=FloatMulPow2(k=k), operands=(a,)):
                # The exponent rides as a materialized integer constant, clamped rather than refused: any count past
                # the int format already lies far beyond the float's dynamic range, where the scaler rails or flushes
                # identically.
                base, sign = collapse_signs(self.hir.nodes, a)
                exponent = self.builder.const(ops.int_format.saturate(k), ScalarIntType(ops.int_format))
                return self.builder.operation(
                    require(ops.fmul_ilog2, "fmul_ilog2"), [self.remap[base], exponent], [sign, IntIdentity()]
                )
            case Operation(operator=FloatMulPow2Dynamic(), operands=operands):
                return self.emit(require(ops.fmul_ilog2, "fmul_ilog2"), operands)
            case Operation(operator=FloatRounding(rounding=rounding), operands=operands):
                return self.emit(require(ops.fround, "fround"), operands, immediates=(int(_ROUND_MODE_OF[rounding]),))
            case Operation(operator=FloatExp2(), operands=operands):
                return self.emit(require(ops.fexp2, "fexp2"), operands)
            case Operation(operator=FloatLog2(), operands=operands):
                return self.emit(require(ops.flog2, "flog2"), operands)
            case Operation(operator=(FloatSinTurns() | FloatCosTurns()) as semantic, operands=operands):
                # zkf_sincos already counts in turns, so whatever conversion the kernel's angle wanted was stated in HIR
                # and folded there. The sign rides the core's own operand, so one value serves sin(-x)/cos(-x) and a
                # sin+cos over one argument fuse into one firing.
                port = 0 if isinstance(semantic, FloatSinTurns) else 1
                return self.emit(require(ops.fsincos, "fsincos"), operands, output_port=port)
            case Operation(operator=FloatSqrt(), operands=operands):
                return self.emit(require(ops.fsqrt, "fsqrt"), operands)
            case Operation(operator=FloatAtan2Turns(), operands=operands):
                # zkf_atan2 returns theta in turns, which is what the operator means; its magnitude port is tapped
                # only by a fused hypot (above).
                return self.emit(require(ops.fatan2, "fatan2"), operands)
            case Operation(operator=(FloatMin() | FloatMax()) as semantic, operands=operands):
                # min taps the low output port, max the high one; a min and a max over one pair fuse into one firing.
                return self.emit(require(ops.fsort, "fsort"), operands, output_port=int(isinstance(semantic, FloatMax)))
            case Operation(operator=FloatFma(), operands=operands):
                return self.emit(require(ops.ffma, "ffma"), operands)
            case Operation(operator=BoolToFloat(), operands=operands):
                return self.emit(BoolToFloatOperator(fmt), operands)
            case Operation(operator=IntToFloat(), operands=operands):
                return self.emit(require(ops.ffromint, "ffromint"), operands)
            case Operation(operator=FloatSelect(), operands=operands):
                return self.emit(SelectOperator(ScalarFloatType(fmt)), operands)
            case _:
                raise AssertionError(f"no float lowering for {node.operator.mnemonic!r}")


class _IntLowerer(_FamilyLowerer):
    """
    `ineg` and `iabs` are hardware where their float counterparts are free. Integer operators are never optional, so
    only the conversions that cross into the float half go through require.
    """

    def lower_operation(self, old_id: ValueId, node: Operation) -> ValueId:
        ops, fmt = self.ops, self.ops.int_format
        match node:
            case Operation(operator=IntAdd(), operands=operands):
                return self.emit(ops.iadd, operands)
            case Operation(operator=IntSub(), operands=operands):
                return self.emit(ops.isub, operands)
            case Operation(operator=IntMul(), operands=operands):
                return self.emit(ops.imul, operands)
            case Operation(operator=IntMulPow2(k=k), operands=(a,)):
                # The left shifter's OTHER reading: `prod` saturates where `shft` lets the high bits fall off the word,
                # and saturating is what a multiplication does. The count clamps at the width -- past that every count
                # rails the same operand the same way -- and into the word, since a two-bit word cannot hold its own
                # width: every count from `width - 1` up rails identically.
                count = self._const(fmt.saturate(min(k, fmt.width)))
                return self.builder.operation(ops.ishl, [self.remap[a], count], [IntIdentity()] * 2, output_port=1)
            case Operation(operator=IntNeg(), operands=(a,)):
                # `0 - x`: there is no negation module, and the subtractor saturates `-MIN` correctly.
                return self.builder.operation(ops.isub, [self._const(0), self.remap[a]], [IntIdentity()] * 2)
            case Operation(operator=IntAbs(), operands=operands):
                return self.emit(ops.iabs, operands)
            case Operation(operator=IntPopcount(), operands=operands):
                return self.emit(ops.ipopcnt, operands)
            # The quotient and the remainder are two taps of one divider: written from the same operands, they share
            # a MIR intern key up to the port and fuse into a single firing at LIR build.
            case Operation(operator=IntDivFloor(), operands=operands):
                return self.emit(ops.idiv, operands, output_port=0)
            case Operation(operator=IntMod(), operands=operands):
                return self.emit(ops.idiv, operands, output_port=1)
            case Operation(operator=(IntShiftLeft() | IntShiftRight()) as semantic, operands=(a, count)):
                return self._lower_shift(semantic, a, count)
            case Operation(operator=IntBwAnd(), operands=operands):
                return self.emit(IntBwAndOperator(fmt), operands)
            case Operation(operator=IntBwOr(), operands=operands):
                return self.emit(IntBwOrOperator(fmt), operands)
            case Operation(operator=IntBwXor(), operands=operands):
                return self.emit(IntBwXorOperator(fmt), operands)
            case Operation(operator=IntBwNot(), operands=operands):
                return self.emit(IntBwNotOperator(fmt), operands)
            case Operation(operator=IntSelect(), operands=operands):
                return self.emit(SelectOperator(ScalarIntType(fmt)), operands)
            case Operation(operator=BoolToInt(), operands=operands):
                return self.emit(BoolToIntOperator(fmt), operands)
            case Operation(operator=FloatToInt(rounding=rounding), operands=operands):
                return self.emit(require(ops.ftoint, "ftoint"), operands, immediates=(int(_ROUND_MODE_OF[rounding]),))
            case Operation(operator=FloatILog2(), operands=operands):
                return self.emit(require(ops.filog2, "filog2"), operands)
            case _:
                raise AssertionError(f"no integer lowering for {node.operator.mnemonic!r}")

    def _lower_shift(self, semantic: IntShiftLeft | IntShiftRight, a: ValueId, count: ValueId) -> ValueId:
        """
        Each direction has the module that names it, so neither negates its count to reach the other's. The left
        shifter is tapped on its raw reading, because `<<` drops what leaves the word rather than saturating. Both
        modules clamp the amount at the word, which is where the two readings of an unbounded count meet -- a left
        shift past the word answers zero and a right shift past it answers the sign fill, as Python's own unbounded
        shift does once the word truncates it.

        A constant count needs neither module and is never materialized.
        """
        constant = constant_shift_count(self.hir, count)
        if constant is None:
            return self.emit(self.ops.ishr if isinstance(semantic, IntShiftRight) else self.ops.ishl, [a, count])
        width = self.ops.int_format.width
        assert constant > 0, "strength reduction elides a zero-count shift and the gate owns a negative one"
        if isinstance(semantic, IntShiftLeft):
            assert constant < width, "specialization owns a left shift past the word, whose answer is zero regardless"
            shamt = constant
        else:
            # Clamped here and not in HIR: past the top bit a right shift repeats the sign fill only for a value the
            # word already holds, and this operand is one where an HIR constant need not have been.
            shamt = -min(constant, width - 1)
        return self.emit(IntShiftConstOperator(self.ops.int_format, shamt), [a])

    def _const(self, value: int) -> ValueId:
        return self.builder.const(value, ScalarIntType(self.ops.int_format))


class _BoolLowerer(_FamilyLowerer):
    def lower_operation(self, old_id: ValueId, node: Operation) -> ValueId:
        match node:
            case Operation(operator=FloatComparison() | IntComparison() as semantic, operands=operands):
                # A relation is one comparator output port with an optional inversion (both orderings are total and
                # the flags one-hot), so every relation over one operand pair selects the same comparator and fuses
                # into one firing.
                comparator = require(self.ops.fcmp, "fcmp") if isinstance(semantic, FloatComparison) else self.ops.icmp
                port, inversion = comparator.tap_of(semantic.relation)
                return self.emit(comparator, operands, output_port=port, output_conditioner=inversion)
            case Operation(operator=IntToBool(), operands=operands):
                return self.emit(IntToBoolOperator(self.ops.int_format), operands)
            case Operation(
                operator=(FloatIsFinite() | FloatIsInf() | FloatIsPosInf() | FloatIsNegInf()) as semantic,
                operands=operands,
            ):
                return self._classify(semantic, operands)
            case Operation(operator=BoolAnd(), operands=operands):
                return self.emit(BoolAndOperator(), operands)
            case Operation(operator=BoolOr(), operands=operands):
                return self.emit(BoolOrOperator(), operands)
            case Operation(operator=BoolXor(), operands=operands):
                return self.emit(BoolXorOperator(), operands)
            case Operation(operator=BoolSelect(), operands=operands):
                return self.emit(SelectOperator(ScalarBoolType()), operands)
            case Operation(operator=FloatToBool(), operands=operands):
                return self.emit(FloatToBoolOperator(self.ops.float_format), operands)
            case _:
                raise AssertionError(f"no boolean lowering for {node.operator.mnemonic!r}")

    def _classify(
        self, semantic: FloatIsFinite | FloatIsInf | FloatIsPosInf | FloatIsNegInf, operands: Sequence[ValueId]
    ) -> ValueId:
        fmt = self.ops.float_format
        match semantic:
            case FloatIsFinite():
                operator: FloatClassificationOperator = FloatIsFiniteOperator(fmt)
                output = BoolInversion()
            case FloatIsInf():
                operator, output = FloatIsFiniteOperator(fmt), BoolInversion(invert=True)
            case FloatIsPosInf():
                operator, output = FloatIsPosInfOperator(fmt), BoolInversion()
            case FloatIsNegInf():
                operator, output = FloatIsNegInfOperator(fmt), BoolInversion()
        return self.emit(operator, operands, output_conditioner=output)


def _derive(hir: Hir, ops: OpConfig, ifconv_max_ops: int) -> Hir:
    """
    Everything this machine knows, told to the graph. The trigonometric unit goes first, since it is the optimizer's
    input rather than its consumer, and the word width from within the fixpoint, since only a fold can reveal a count.
    """
    hir = optimize(trig_abi(hir), ifconv_max_ops)
    # Each rewrite deletes one of these and no pass grows either count: a magnitude is re-minted only in place of
    # the one it replaced.
    fuel = left_shifts(hir) + hypot_count(hir) + 1
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
    # A family absent from the graph needs no room in the wide register, however the machine is configured. Node types
    # alone are complete: every operand is itself a node, and a slot's reset shares its live-out's type.
    floats = any(isinstance(node.type, HirFloatType) for node in hir.nodes.values())
    return _widest_word(options) if floats else options.wint_min


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
    return _LoweringContext(contract_fmas(hir, ops), ops).run()
