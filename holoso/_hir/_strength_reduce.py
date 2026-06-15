"""HIR algebraic strength reduction."""

import math

from ._const import BoolConst, FloatConst
from ._copy import copy_node, rebuild
from ._ir import Hir, HirBuilder, Node, Operation, ValueId
from ._operators import BoolAnd, BoolNot, BoolOr, BoolSelect, FloatDiv, FloatMul, FloatMulPow2, Select


def _ilog2_exact(c: float) -> int | None:
    """Return ``k`` if ``c == 2**k`` exactly for a positive ``c``, else ``None``."""
    if c <= 0.0 or not math.isfinite(c):
        return None
    mantissa, exponent = math.frexp(c)  # c == mantissa * 2**exponent, mantissa in [0.5, 1)
    return exponent - 1 if mantissa == 0.5 else None


def run(hir: Hir) -> Hir:
    """
    Rewrite exact power-of-two scaling and finite constant division, and reduce the if-conversion muxes: a ``select``
    with identical arms drops out, and a ``bool_select`` with one or two constant arms collapses to ``and``/``or``/
    ``not``/passthrough (the common state-machine merge with ``True``/``False`` arms). All before hardware selection.
    """
    cval: dict[ValueId, float] = {}

    def bool_const(vid: ValueId) -> bool | None:
        node = hir.nodes[vid]
        return node.value if isinstance(node, BoolConst) else None

    def build_value(builder: HirBuilder, vid: ValueId, node: Node, remap: dict[ValueId, ValueId]) -> ValueId:
        match node:
            case FloatConst(value=value):
                new_id = builder.float_const(value)
                cval[new_id] = value
                return new_id
            case Operation(operator=FloatMul(), operands=(a, b)):
                return _reduce_mul(builder, remap[a], remap[b], cval)
            case Operation(operator=FloatDiv(), operands=(a, b)):
                return _reduce_div(builder, remap[a], remap[b], cval)
            case Operation(operator=Select(), operands=(_cond, a, b)) if remap[a] == remap[b]:
                return remap[a]  # select(c, X, X) == X
            case Operation(operator=BoolSelect(), operands=(cond, a, b)):
                return _reduce_bool_select(builder, remap[cond], remap[a], remap[b], bool_const(a), bool_const(b))
            case _:
                return copy_node(builder, node, remap)

    return rebuild(hir, build_value)


def _reduce_bool_select(
    builder: HirBuilder, cond: ValueId, a: ValueId, b: ValueId, a_const: bool | None, b_const: bool | None
) -> ValueId:
    """Reduce ``bool_select(cond, a, b)`` using its constant arms; the NOTs fold consumer-side at MIR lowering."""
    if a == b:
        return a  # bool_select(c, X, X) == X (covers both arms the same interned constant)
    if a_const is not None and b_const is not None:  # both constant and distinct -> True/False or False/True
        return cond if a_const else builder.operation(BoolNot(), [cond])
    if a_const is True:
        return builder.operation(BoolOr(), [cond, b])  # (c, True, b) == c or b
    if a_const is False:
        return builder.operation(BoolAnd(), [builder.operation(BoolNot(), [cond]), b])  # (c, False, b) == ~c and b
    if b_const is True:
        return builder.operation(BoolOr(), [builder.operation(BoolNot(), [cond]), a])  # (c, a, True) == ~c or a
    if b_const is False:
        return builder.operation(BoolAnd(), [cond, a])  # (c, a, False) == c and a
    return builder.operation(BoolSelect(), [cond, a, b])  # both arms dynamic: keep the mux


def _reduce_mul(builder: HirBuilder, a: ValueId, b: ValueId, cval: dict[ValueId, float]) -> ValueId:
    for const_side, other in ((b, a), (a, b)):
        if const_side in cval:
            k = _ilog2_exact(cval[const_side])
            if k is not None:
                return builder.operation(FloatMulPow2(k), [other])
    return builder.operation(FloatMul(), [a, b])


def _reduce_div(builder: HirBuilder, a: ValueId, b: ValueId, cval: dict[ValueId, float]) -> ValueId:
    if b in cval:
        c = cval[b]
        k = _ilog2_exact(c)
        if k is not None:
            return builder.operation(FloatMulPow2(-k), [a])
        if c != 0.0 and math.isfinite(c):
            return builder.operation(FloatMul(), [a, builder.float_const(1.0 / c)])
    return builder.operation(FloatDiv(), [a, b])
