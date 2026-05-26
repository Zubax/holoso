"""Lower optimized HIR to selected MIR."""

import math
from typing import assert_never

from ._hir import Abs, Add, Const, Div, Hir, InPort, Mul, MulPow2, Neg, Node, Operation, ValueId
from ._mir import Mir, MirBuilder
from ._operators import OpConfig, SignControl


def _sign_of(node: Operation) -> SignControl | None:
    match node:
        case Operation(operator=Neg()):
            return SignControl(negate=True)
        case Operation(operator=Abs()):
            return SignControl(absolute=True)
        case _:
            return None


def _collapse_signs(nodes: dict[ValueId, Node], vid: ValueId) -> tuple[ValueId, SignControl]:
    """Peel a chain of semantic sign operations, returning the non-sign base value and combined sign control."""
    chain: list[SignControl] = []
    node = nodes[vid]
    while isinstance(node, Operation) and (sign := _sign_of(node)) is not None:
        chain.append(sign)
        (vid,) = node.operands
        node = nodes[vid]
    control = SignControl()
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


def lower(hir: Hir, ops: OpConfig) -> Mir:
    """
    Select hardware operators from the configuration and fold semantic signs onto MIR sign controls.

    Semantic sign operations are never emitted as standalone scheduled operators. Exact power-of-two scaling selects
    ``fmul_ilog2_const`` when feasible for the configured float format, otherwise it falls back to ordinary multiply
    by a constant factor.
    """
    builder = MirBuilder()
    remap: dict[ValueId, ValueId] = {}
    for old_id in sorted(hir.nodes):
        node = hir.nodes[old_id]
        match node:
            case InPort(name=name):
                remap[old_id] = builder.input(name)
            case Const(value=value):
                remap[old_id] = builder.const(value)
            case Operation() if _sign_of(node) is not None:
                continue
            case Operation(operator=Add(), operands=(a, b)):
                base_a, sign_a = _collapse_signs(hir.nodes, a)
                base_b, sign_b = _collapse_signs(hir.nodes, b)
                remap[old_id] = builder.operation(ops.fadd, [remap[base_a], remap[base_b]], [sign_a, sign_b])
            case Operation(operator=Mul(), operands=(a, b)):
                base_a, sign_a = _collapse_signs(hir.nodes, a)
                base_b, sign_b = _collapse_signs(hir.nodes, b)
                remap[old_id] = builder.operation(ops.fmul, [remap[base_a], remap[base_b]], [sign_a, sign_b])
            case Operation(operator=Div(), operands=(a, b)):
                base_a, sign_a = _collapse_signs(hir.nodes, a)
                base_b, sign_b = _collapse_signs(hir.nodes, b)
                remap[old_id] = builder.operation(ops.fdiv, [remap[base_a], remap[base_b]], [sign_a, sign_b])
            case Operation(operator=MulPow2(k=k), operands=(a,)):
                base, sign = _collapse_signs(hir.nodes, a)
                if _ilog2_feasible(ops, k):
                    remap[old_id] = builder.operation(ops.fmul_ilog2.instantiate(k), [remap[base]], [sign])
                else:
                    remap[old_id] = builder.operation(
                        ops.fmul,
                        [remap[base], builder.const(_pow2(k))],
                        [sign, SignControl()],
                    )
            case Operation(operator=operator):
                raise RuntimeError(f"no hardware lowering rule for HIR operator {operator.mnemonic!r}")
            case _ as unreachable:
                assert_never(unreachable)
    for out in hir.outputs:
        base, sign = _collapse_signs(hir.nodes, out.value)
        builder.output(out.name, remap[base], sign)
    return builder.finish()
