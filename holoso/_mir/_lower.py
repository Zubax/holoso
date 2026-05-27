"""Lower optimized HIR to selected MIR."""

import math

from .._errors import UnsupportedConstruct
from .._hir import (
    Const,
    FloatAbs,
    FloatAdd,
    FloatConst,
    FloatDiv,
    FloatType as HirFloatType,
    FloatMul,
    FloatMulPow2,
    FloatNeg,
    Hir,
    InPort,
    Node,
    Operation,
    ValueId,
)
from .._operators import FloatHardwareOperator, OpConfig, FloatSignControl
from .._type import FloatType as ScalarFloatType
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


def _ilog2_feasible(ops: OpConfig, k: int) -> bool:
    return abs(k) < (1 << (ops.float_format.wexp - 1))


def _pow2(k: int) -> float:
    try:
        return math.ldexp(1.0, k)
    except OverflowError:
        return math.inf


class _LoweringContext:
    def __init__(self, hir: Hir, ops: OpConfig) -> None:
        self.hir = hir
        self.ops = ops
        self.builder = MirBuilder()
        self.remap: dict[ValueId, ValueId] = {}
        self.float_lowerer = _FloatLowerer(self)

    def run(self) -> Mir:
        for old_id in sorted(self.hir.nodes):
            self._lower_node(old_id, self.hir.nodes[old_id])
        for out in self.hir.outputs:
            self._lower_output(out.name, out.value)
        return self.builder.finish()

    def _lower_node(self, old_id: ValueId, node: Node) -> None:
        if self.float_lowerer.lower_node(old_id, node):
            return
        match node:
            case Const(type=type):
                raise UnsupportedConstruct(f"no MIR lowering rule for HIR constant type {type!r}")
            case InPort(type=type):
                raise UnsupportedConstruct(f"no MIR lowering rule for HIR input type {type!r}")
            case Operation(operator=operator):
                raise UnsupportedConstruct(f"no hardware lowering rule for HIR operator {operator.mnemonic!r}")

    def _lower_output(self, name: str, value: ValueId) -> None:
        if self.float_lowerer.lower_output(name, value):
            return
        raise UnsupportedConstruct(f"no MIR lowering rule for HIR output type {self.hir.nodes[value].type!r}")


class _FloatLowerer:
    def __init__(self, context: _LoweringContext) -> None:
        self.context = context
        self.float_type = ScalarFloatType(context.ops.float_format)

    def lower_node(self, old_id: ValueId, node: Node) -> bool:
        match node:
            case InPort(name=name, type=HirFloatType()):
                self.context.remap[old_id] = self.context.builder.float_input(name, self.float_type)
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
            case _:
                return None

    def _lower_binary_float(self, operator: FloatHardwareOperator, a: ValueId, b: ValueId) -> ValueId:
        base_a, sign_a = _collapse_signs(self.context.hir.nodes, a)
        base_b, sign_b = _collapse_signs(self.context.hir.nodes, b)
        return self.context.builder.float_operation(
            operator,
            [self.context.remap[base_a], self.context.remap[base_b]],
            [sign_a, sign_b],
        )

    def _lower_float_mul_pow2(self, a: ValueId, k: int) -> ValueId:
        base, sign = _collapse_signs(self.context.hir.nodes, a)
        if _ilog2_feasible(self.context.ops, k):
            return self.context.builder.float_operation(
                self.context.ops.fmul_ilog2.instantiate(k), [self.context.remap[base]], [sign]
            )
        return self.context.builder.float_operation(
            self.context.ops.fmul,
            [self.context.remap[base], self.context.builder.float_const(_pow2(k), self.float_type)],
            [sign, FloatSignControl()],
        )

    def lower_output(self, name: str, value: ValueId) -> bool:
        base, sign = _collapse_signs(self.context.hir.nodes, value)
        if not isinstance(self.context.hir.nodes[base].type, HirFloatType):
            return False
        self.context.builder.float_output(name, self.context.remap[base], sign)
        return True


def lower(hir: Hir, ops: OpConfig) -> Mir:
    """
    Select hardware operators from the configuration and fold semantic signs onto MIR sign controls.

    Semantic sign operations are never emitted as standalone scheduled operators. Exact power-of-two scaling selects
    ``fmul_ilog2_const`` when feasible for the configured float format, otherwise it falls back to ordinary multiply
    by a constant factor.
    """
    return _LoweringContext(hir, ops).run()
