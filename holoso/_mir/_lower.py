"""Lower optimized HIR to selected MIR."""

from dataclasses import dataclass

from .._errors import UnsupportedConstruct
from .._hir import (
    BoolAnd,
    BoolConst,
    BoolNot,
    BoolOr,
    BoolSelect,
    BoolToFloat,
    BoolType as HirBoolType,
    BoolXor,
    Branch,
    Const,
    FloatAbs,
    FloatAdd,
    FloatCeil,
    FloatConst,
    FloatDiv,
    FloatExp2,
    FloatFloor,
    FloatFma,
    FloatLog2,
    FloatMax,
    FloatMin,
    FloatMul,
    FloatMulPow2,
    FloatNeg,
    FloatRelational,
    FloatRound,
    FloatToBool,
    FloatTrunc,
    FloatType as HirFloatType,
    Hir,
    InPort,
    Jump,
    Node,
    Operation,
    Operator,
    Phi,
    Ret,
    Select,
    StateRead,
    StateSlot,
    Terminator,
    reverse_postorder,
)
from .._util import ValueId
from .._operators import (
    BoolAndOperator,
    BoolInversion,
    BoolOrOperator,
    BoolSelectOperator,
    BoolToFloatOperator,
    BoolXorOperator,
    FloatHardwareOperator,
    FloatSignControl,
    FloatToBoolOperator,
    FRoundOperator,
    HardwareOperator,
    OpConfig,
    PortConditioner,
    PooledHardwareOperator,
    SelectOperator,
)
from .._type import BoolType as ScalarBoolType, FloatType as ScalarFloatType, ScalarType
from ._ir import Mir, MirBuilder


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
    """Collapse the type's own sideband chain: sign operations over a float value, NOTs over a boolean one."""
    ty = nodes[vid].type
    match ty:
        case HirBoolType():
            return _collapse_bool_inversions(nodes, vid)
        case HirFloatType():
            return _collapse_signs(nodes, vid)
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


def _plan_fma_fusions(hir: Hir, ops: OpConfig) -> dict[ValueId, _FmaPlan]:
    """
    Map each FloatAdd that will contract into an ``ffma`` to its plan (only when ``ffma`` is configured; else no
    contraction, and the whole-DAG use-count is skipped). When both addends are exclusive products only the first
    contracts -- one fma carries one product.
    """
    if ops.ffma is None:
        return {}
    use_counts = _compute_use_counts(hir)
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


class _LoweringContext:
    def __init__(self, hir: Hir, ops: OpConfig) -> None:
        self.hir = hir
        self.ops = ops
        self.builder = MirBuilder(ops.float_format)
        self.remap: dict[ValueId, ValueId] = {}
        self.fma_plans = _plan_fma_fusions(hir, ops)
        self.fused_muls = {plan.mul for plan in self.fma_plans.values()}
        self.float_lowerer = _FloatLowerer(self)

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
                return ScalarFloatType(self.ops.float_format)
            case HirBoolType():
                return ScalarBoolType()
            case _:
                raise UnsupportedConstruct(f"no MIR lowering rule for phi of type {node.type!r}")

    def _lower_node(self, old_id: ValueId, node: Node) -> None:
        if self.float_lowerer.lower_node(old_id, node):
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
            case Operation(operator=FloatRelational(op=relation) as semantic, operands=(a, b)):
                # A relation is one comparator output port with an optional inversion (the ZKF ordering is total and
                # the flags one-hot), so every relation -- and every comparison over the same operand pair -- selects
                # into the same pooled fcmp operator and can fuse into one firing.
                base_a, sign_a = _collapse_signs(self.hir.nodes, a)
                base_b, sign_b = _collapse_signs(self.hir.nodes, b)
                port, inversion = self.ops.fcmp.tap_of(relation)
                self.remap[old_id] = self.builder.operation(
                    _select_hardware(semantic, self.ops.fcmp),
                    [self.remap[base_a], self.remap[base_b]],
                    [sign_a, sign_b],
                    output_port=port,
                    output_conditioner=inversion,
                )
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
                # inversion conditioner, exactly like float Select's sign folding -- so ``a if not c else b`` is free.
                self._lower_bool_logic(old_id, _select_hardware(semantic, BoolSelectOperator()), [cond, a, b])
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
                    _select_hardware(semantic, FloatToBoolOperator(self.ops.float_format)), [self.remap[base]], [sign]
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

    def _lower_output(self, name: str, value: ValueId) -> None:
        if self.float_lowerer.lower_output(name, value):
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
        if self._lower_bool_state_slot(slot):
            return
        raise UnsupportedConstruct(f"no MIR lowering rule for HIR state slot {slot.name!r}")

    def _lower_bool_state_slot(self, slot: StateSlot) -> bool:
        if not isinstance(self.hir.nodes[slot.live_out].type, HirBoolType):
            return False
        base, inversion = _collapse_bool_inversions(self.hir.nodes, slot.live_out)
        if not isinstance(slot.reset_value, BoolConst):
            raise UnsupportedConstruct(f"boolean state slot {slot.name!r} must have a boolean reset value")
        self.builder.bool_state_slot(slot.name, slot.reset_value.value, self.remap[base], inversion)
        return True


class _FloatLowerer:
    def __init__(self, context: _LoweringContext) -> None:
        self.context = context
        self.float_type = ScalarFloatType(context.ops.float_format)

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
                plan = self.context.fma_plans.get(old_id)
                if plan is not None:
                    assert isinstance(operation.operator, FloatAdd)
                    self.context.remap[old_id] = self._emit_ffma(
                        operation.operator, plan.ma, plan.mb, plan.c, plan.product_sign
                    )
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

    def _lower_operation(self, node: Operation) -> ValueId | None:
        match node:
            case Operation(operator=FloatAdd() as semantic, operands=(a, b)):
                return self._lower_binary_float(semantic, self.context.ops.fadd, a, b)
            case Operation(operator=FloatMul() as semantic, operands=(a, b)):
                return self._lower_binary_float(semantic, self.context.ops.fmul, a, b)
            case Operation(operator=FloatDiv() as semantic, operands=(a, b)):
                return self._lower_binary_float(semantic, self.context.ops.fdiv, a, b)
            case Operation(operator=FloatMulPow2(k=k) as semantic, operands=(a,)):
                return self._lower_float_mul_pow2(semantic, a, k)
            case Operation(
                operator=(FloatRound() | FloatFloor() | FloatCeil() | FloatTrunc()) as semantic, operands=(a,)
            ):
                return self._lower_round(semantic, a)
            case Operation(operator=FloatExp2() as semantic, operands=(a,)):
                return self._lower_unary_pooled(semantic, self.context.ops.fexp2, "fexp2", a)
            case Operation(operator=FloatLog2() as semantic, operands=(a,)):
                return self._lower_unary_pooled(semantic, self.context.ops.flog2, "flog2", a)
            case Operation(operator=(FloatMin() | FloatMax()) as semantic, operands=(a, b)):
                return self._lower_minmax(semantic, a, b)
            case Operation(operator=FloatFma() as semantic, operands=(a, b, c)):
                return self._emit_ffma(semantic, a, b, c, FloatSignControl())
            case Operation(operator=BoolToFloat() as semantic, operands=(a,)):
                # ``float(cond)`` crosses from the boolean bank into the wide bank; a NOT chain folds into the
                # operand conditioner.
                base, inversion = _collapse_bool_inversions(self.context.hir.nodes, a)
                return self.context.builder.operation(
                    _select_hardware(semantic, BoolToFloatOperator(self.context.ops.float_format)),
                    [self.context.remap[base]],
                    [inversion],
                )
            case Operation(operator=Select() as semantic, operands=(cond, a, b)):
                # The if-conversion mux: arm signs and a condition NOT chain fold into the operand conditioners
                # (``x if c else -x`` and ``a if not c else b`` cost no hardware beyond the mux itself).
                base_c, inv_c = _collapse_bool_inversions(self.context.hir.nodes, cond)
                base_a, sign_a = _collapse_signs(self.context.hir.nodes, a)
                base_b, sign_b = _collapse_signs(self.context.hir.nodes, b)
                return self.context.builder.operation(
                    _select_hardware(semantic, SelectOperator(self.context.ops.float_format)),
                    [self.context.remap[base_c], self.context.remap[base_a], self.context.remap[base_b]],
                    [inv_c, sign_a, sign_b],
                )
            case _:
                return None

    def _lower_binary_float(
        self, semantic: Operator, hardware: FloatHardwareOperator, a: ValueId, b: ValueId
    ) -> ValueId:
        base_a, sign_a = _collapse_signs(self.context.hir.nodes, a)
        base_b, sign_b = _collapse_signs(self.context.hir.nodes, b)
        return self.context.builder.operation(
            _select_hardware(semantic, hardware),
            [self.context.remap[base_a], self.context.remap[base_b]],
            [sign_a, sign_b],
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
        operator = self.context.ops.ffma
        if operator is None:
            raise UnsupportedConstruct(
                "the kernel uses math.fma but no 'ffma' operator is configured; add it to OpConfig"
            )
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
        mode = {
            FloatRound: FRoundOperator.Mode.ROUND,
            FloatFloor: FRoundOperator.Mode.FLOOR,
            FloatCeil: FRoundOperator.Mode.CEIL,
            FloatTrunc: FRoundOperator.Mode.TRUNC,
        }[type(semantic)]
        return self._lower_unary_pooled(semantic, self.context.ops.fround, "fround", a, immediates=(int(mode),))

    def _lower_unary_pooled(
        self,
        semantic: Operator,
        operator: FloatHardwareOperator | None,
        config_field: str,
        a: ValueId,
        immediates: tuple[int, ...] = (),
    ) -> ValueId:
        if operator is None:
            raise UnsupportedConstruct(
                f"the kernel uses {semantic.mnemonic!r} but no {config_field!r} operator is configured; "
                "add it to OpConfig"
            )
        # Sign chain folds onto the operand (applied before the op): floor(-x)/exp2(-x) feed -x, not -floor(x)/-exp2(x).
        base, sign = _collapse_signs(self.context.hir.nodes, a)
        return self.context.builder.operation(
            _select_hardware(semantic, operator),
            [self.context.remap[base]],
            [sign],
            immediates=immediates,
        )

    def _lower_minmax(self, semantic: FloatMin | FloatMax, a: ValueId, b: ValueId) -> ValueId:
        # Each input sign chain folds onto its operand conditioner, applied before the sort: min(-a, b) is the sorter
        # fed (-a, b). min taps the low output port, max the high one; a min and a max over one pair fuse at LIR build.
        operator = self.context.ops.fsort
        if operator is None:
            raise UnsupportedConstruct(
                f"the kernel uses {semantic.mnemonic!r} but no 'fsort' operator is configured; add it to OpConfig"
            )
        base_a, sign_a = _collapse_signs(self.context.hir.nodes, a)
        base_b, sign_b = _collapse_signs(self.context.hir.nodes, b)
        return self.context.builder.operation(
            _select_hardware(semantic, operator),
            [self.context.remap[base_a], self.context.remap[base_b]],
            [sign_a, sign_b],
            output_port=0 if isinstance(semantic, FloatMin) else 1,
        )

    def _lower_float_mul_pow2(self, semantic: Operator, a: ValueId, k: int) -> ValueId:
        base, sign = _collapse_signs(self.context.hir.nodes, a)
        try:
            operator = _select_hardware(semantic, self.context.ops.fmul_ilog2.instantiate(k))
        except ValueError as exc:
            # An out-of-range exponent is rejected rather than lowered to a constant multiply by 2**k: such a k always
            # lies outside the format's representable range, so the constant would overflow to a (rejected) infinity or
            # underflow to zero -- the fallback multiply would be degenerate, so there is nothing useful to fall back
            # to.
            raise UnsupportedConstruct(f"unsupported power-of-two float scale 2**{k}: {exc}") from exc
        return self.context.builder.operation(operator, [self.context.remap[base]], [sign])

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
        if not isinstance(slot.reset_value, FloatConst):
            raise UnsupportedConstruct(f"floating-point state slot {slot.name!r} must have a float reset value")
        self.context.builder.float_state_slot(slot.name, slot.reset_value.value, self.context.remap[base], sign)
        return True


def lower(hir: Hir, ops: OpConfig) -> Mir:
    """
    Select hardware operators from the configuration and fold semantic signs onto MIR sign controls.

    Semantic sign operations are never emitted as standalone scheduled operators. Exact power-of-two scaling selects
    ``fmul_ilog2_const`` when supported by the configured float format; unsupported exponents are rejected.
    """
    return _LoweringContext(hir, ops).run()
