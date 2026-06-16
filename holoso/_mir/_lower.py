"""Lower optimized HIR to selected MIR."""

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
    FloatConst,
    FloatDiv,
    FloatMul,
    FloatMulPow2,
    FloatNeg,
    FloatRelational,
    FloatToBool,
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
    ValueId,
    reverse_postorder,
)
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
    if isinstance(nodes[vid].type, HirBoolType):
        return _collapse_bool_inversions(nodes, vid)
    return _collapse_signs(nodes, vid)


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


class _LoweringContext:
    def __init__(self, hir: Hir, ops: OpConfig) -> None:
        self.hir = hir
        self.ops = ops
        self.builder = MirBuilder(ops.float_format)
        self.remap: dict[ValueId, ValueId] = {}
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

    def _lower_float_mul_pow2(self, semantic: Operator, a: ValueId, k: int) -> ValueId:
        base, sign = _collapse_signs(self.context.hir.nodes, a)
        try:
            operator = _select_hardware(semantic, self.context.ops.fmul_ilog2.instantiate(k))
        except ValueError as exc:
            # An out-of-range exponent is rejected rather than lowered to a constant multiply by 2**k: such a k always
            # lies outside the format's representable range, so the constant would overflow to a (rejected) infinity or
            # underflow to zero -- the fallback multiply would be degenerate, so there is nothing useful to fall back to.
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
