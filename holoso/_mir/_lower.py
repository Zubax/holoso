"""Lower optimized HIR to selected MIR."""

from .._errors import UnsupportedConstruct
from .._hir import (
    BoolAnd,
    BoolConst,
    BoolNot,
    BoolOr,
    BoolToFloat,
    BoolType as HirBoolType,
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
    Phi,
    Ret,
    StateRead,
    StateSlot,
    Terminator,
    ValueId,
    reverse_postorder,
)
from .._operators import (
    BoolAndOperator,
    BoolNotOperator,
    BoolOrOperator,
    BoolToFloatOperator,
    FComparisonOperator,
    FloatHardwareOperator,
    FloatSignControl,
    FloatToBoolOperator,
    HardwareOperator,
    OpConfig,
)
from .._type import BoolType as ScalarBoolType, FloatType as ScalarFloatType, ScalarType
from ._ir import Mir, MirBuilder


def _sign_of(node: Operation) -> FloatSignControl | None:
    match node:
        case Operation(operator=FloatNeg()):
            return FloatSignControl(negate=True)
        case Operation(operator=FloatAbs()):
            return FloatSignControl(absolute=True)
        case _:
            return None


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
                self.builder.branch(self.remap[cond], if_true, if_false)
            case Ret():
                self.builder.ret()

    def _lower_phi(self, old_id: ValueId, node: Node, deferred: list[ValueId]) -> None:
        # A float arm may carry a folded sign (a branch assigning ``-x`` or ``abs(x)``); the merge install applies it.
        # A boolean value is never under a negation/abs, so ``_collapse_signs`` gives the identity sign for a bool arm.
        # A loop-header phi's latch arm is a body value lowered after the header: open the phi with its available arms
        # now (so the body can reference it) and close it once every block is lowered.
        assert isinstance(node, Phi)
        bases = [(pred, _collapse_signs(self.hir.nodes, value)) for pred, value in node.arms]
        scalar_type = self._phi_scalar_type(node)
        if all(base in self.remap for _, (base, _) in bases):
            arms = [(pred, self.remap[base], sign) for pred, (base, sign) in bases]
            self.remap[old_id] = self.builder.phi(scalar_type, arms)
        else:
            known = [(pred, self.remap[base], sign) for pred, (base, sign) in bases if base in self.remap]
            self.remap[old_id] = self.builder.open_phi(scalar_type, known[0])
            deferred.append(old_id)

    def _close_phi(self, old_id: ValueId, node: Node) -> None:
        assert isinstance(node, Phi)
        arms = [
            (pred, self.remap[base], sign)
            for pred, value in node.arms
            for base, sign in [_collapse_signs(self.hir.nodes, value)]
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
            case StateRead(slot=slot, type=HirBoolType()):
                self.remap[old_id] = self.builder.bool_state_read(slot, ScalarBoolType())
                return True
            case BoolConst(value=value):
                self.remap[old_id] = self.builder.bool_const(value, ScalarBoolType())
                return True
            case Operation(operator=FloatRelational(op=relation), operands=(a, b)):
                base_a, sign_a = _collapse_signs(self.hir.nodes, a)
                base_b, sign_b = _collapse_signs(self.hir.nodes, b)
                self.remap[old_id] = self.builder.operation(
                    FComparisonOperator(self.ops.fcmp, relation),
                    [self.remap[base_a], self.remap[base_b]],
                    [sign_a, sign_b],
                )
                return True
            case Operation(operator=BoolAnd(), operands=(a, b)):
                self._lower_bool_logic(old_id, BoolAndOperator(), [a, b])
                return True
            case Operation(operator=BoolOr(), operands=(a, b)):
                self._lower_bool_logic(old_id, BoolOrOperator(), [a, b])
                return True
            case Operation(operator=BoolNot(), operands=(a,)):
                self._lower_bool_logic(old_id, BoolNotOperator(), [a])
                return True
            case Operation(operator=FloatToBool(), operands=(a,)):
                # ``bool(x)`` reads a float operand (its sign is irrelevant: the exponent test is sign-invariant) and
                # writes the boolean bank, like the comparison but with an inline exponent reduction in place of fcmp.
                base, sign = _collapse_signs(self.hir.nodes, a)
                self.remap[old_id] = self.builder.operation(
                    FloatToBoolOperator(self.ops.float_format), [self.remap[base]], [sign]
                )
                return True
            case _:
                return False

    def _lower_bool_logic(self, old_id: ValueId, operator: HardwareOperator, operands: list[ValueId]) -> None:
        # Boolean operands carry no sign control (booleans have no sign); they are remapped directly.
        self.remap[old_id] = self.builder.operation(
            operator, [self.remap[operand] for operand in operands], [FloatSignControl() for _ in operands]
        )

    def _lower_output(self, name: str, value: ValueId) -> None:
        if self.float_lowerer.lower_output(name, value):
            return
        if self._lower_bool_output(name, value):
            return
        raise UnsupportedConstruct(f"no MIR lowering rule for HIR output type {self.hir.nodes[value].type!r}")

    def _lower_bool_output(self, name: str, value: ValueId) -> bool:
        base, sign = _collapse_signs(self.hir.nodes, value)
        if not isinstance(self.hir.nodes[base].type, HirBoolType):
            return False
        if sign != FloatSignControl():
            raise UnsupportedConstruct("a boolean output cannot carry a sign control")
        self.builder.bool_output(name, self.remap[base])
        return True

    def _lower_state_slot(self, slot: StateSlot) -> None:
        if self.float_lowerer.lower_state_slot(slot):
            return
        if self._lower_bool_state_slot(slot):
            return
        raise UnsupportedConstruct(f"no MIR lowering rule for HIR state slot {slot.name!r}")

    def _lower_bool_state_slot(self, slot: StateSlot) -> bool:
        base, sign = _collapse_signs(self.hir.nodes, slot.live_out)
        if not isinstance(self.hir.nodes[base].type, HirBoolType):
            return False
        if sign != FloatSignControl():
            raise UnsupportedConstruct("a boolean state slot cannot carry a sign control")
        if not isinstance(slot.reset_value, BoolConst):
            raise UnsupportedConstruct(f"boolean state slot {slot.name!r} must have a boolean reset value")
        self.builder.bool_state_slot(slot.name, slot.reset_value.value, self.remap[base])
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
            case Operation(operator=FloatAdd(), operands=(a, b)):
                return self._lower_binary_float(self.context.ops.fadd, a, b)
            case Operation(operator=FloatMul(), operands=(a, b)):
                return self._lower_binary_float(self.context.ops.fmul, a, b)
            case Operation(operator=FloatDiv(), operands=(a, b)):
                return self._lower_binary_float(self.context.ops.fdiv, a, b)
            case Operation(operator=FloatMulPow2(k=k), operands=(a,)):
                return self._lower_float_mul_pow2(a, k)
            case Operation(operator=BoolToFloat(), operands=(a,)):
                # ``float(cond)`` is a float-result combinational op reading a boolean operand (no sign control on a
                # boolean): the one operator that crosses from the boolean bank into the float bank.
                return self.context.builder.operation(
                    BoolToFloatOperator(self.context.ops.float_format), [self.context.remap[a]], [FloatSignControl()]
                )
            case _:
                return None

    def _lower_binary_float(self, operator: FloatHardwareOperator, a: ValueId, b: ValueId) -> ValueId:
        base_a, sign_a = _collapse_signs(self.context.hir.nodes, a)
        base_b, sign_b = _collapse_signs(self.context.hir.nodes, b)
        return self.context.builder.operation(
            operator,
            [self.context.remap[base_a], self.context.remap[base_b]],
            [sign_a, sign_b],
        )

    def _lower_float_mul_pow2(self, a: ValueId, k: int) -> ValueId:
        base, sign = _collapse_signs(self.context.hir.nodes, a)
        try:
            operator = self.context.ops.fmul_ilog2.instantiate(k)
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
