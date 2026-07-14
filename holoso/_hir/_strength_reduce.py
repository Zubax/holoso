"""HIR algebraic strength reduction."""

import math

from ._const import BoolConst, FloatConst
from ._copy import copy_node, rebuild
from .._util import ValueId
from ._ir import Hir, HirBuilder, Node, Operation
from ._operators import (
    BoolAnd,
    BoolNot,
    BoolOr,
    BoolSelect,
    BoolToFloat,
    FloatAdd,
    FloatCeil,
    FloatDiv,
    FloatFloor,
    FloatMul,
    FloatMulPow2,
    FloatNeg,
    FloatRound,
    FloatToInt,
    FloatTrunc,
    IntToFloat,
    Operator,
    Select,
)


def _ilog2_exact(c: float) -> int | None:
    """Return ``k`` if ``c == 2**k`` exactly for a positive ``c``, else ``None``."""
    if c <= 0.0 or not math.isfinite(c):
        return None
    mantissa, exponent = math.frexp(c)  # c == mantissa * 2**exponent, mantissa in [0.5, 1)
    return exponent - 1 if mantissa == 0.5 else None


def run(hir: Hir) -> Hir:
    """
    Rewrite trivial fast-math float identities, exact power-of-two scaling, and finite constant division, and reduce
    the if-conversion muxes. All before hardware selection.
    """
    cval: dict[ValueId, float] = {}
    neg_of: dict[ValueId, ValueId] = {}

    def is_integral(vid: ValueId) -> bool:
        """A value provably equal to an integer-valued float: a rounding result, or a bool widened to 0.0/1.0."""
        node = hir.nodes[vid]
        return isinstance(node, Operation) and isinstance(
            node.operator, (FloatFloor, FloatCeil, FloatRound, FloatTrunc, BoolToFloat)
        )

    def emit_float_const(builder: HirBuilder, value: float) -> ValueId:
        new_id = builder.float_const(value)
        cval[new_id] = value
        return new_id

    def emit_float_operation(builder: HirBuilder, operation: Operator, operands: list[ValueId]) -> ValueId:
        return builder.operation(operation, operands)

    def is_zero(vid: ValueId) -> bool:
        return cval.get(vid) == 0.0

    def is_one(vid: ValueId) -> bool:
        return cval.get(vid) == 1.0

    def is_neg_one(vid: ValueId) -> bool:
        return cval.get(vid) == -1.0

    def make_neg(builder: HirBuilder, value: ValueId) -> ValueId:
        base = neg_of.get(value)
        if base is not None:
            return base
        new_id = emit_float_operation(builder, FloatNeg(), [value])
        neg_of[new_id] = value
        return new_id

    def reduce_add(builder: HirBuilder, a: ValueId, b: ValueId) -> ValueId:
        if is_zero(a):
            return b
        if is_zero(b):
            return a
        if neg_of.get(a) == b or neg_of.get(b) == a:
            return emit_float_const(builder, 0.0)
        return emit_float_operation(builder, FloatAdd(), [a, b])

    def reduce_mul(builder: HirBuilder, a: ValueId, b: ValueId) -> ValueId:
        if is_zero(a) or is_zero(b):
            return emit_float_const(builder, 0.0)
        if is_one(a):
            return b
        if is_one(b):
            return a
        if is_neg_one(a):
            return make_neg(builder, b)
        if is_neg_one(b):
            return make_neg(builder, a)
        for const_side, other in ((b, a), (a, b)):
            if const_side in cval:
                k = _ilog2_exact(cval[const_side])
                if k is not None:
                    return emit_float_operation(builder, FloatMulPow2(k), [other])
        return emit_float_operation(builder, FloatMul(), [a, b])

    def reduce_div(builder: HirBuilder, a: ValueId, b: ValueId) -> ValueId:
        if a == b:
            return emit_float_const(builder, 1.0)  # Fast-math fold: intentionally ignores div0 for 0.0 / 0.0.
        if is_zero(a):
            return emit_float_const(builder, 0.0)  # Fast-math fold: intentionally drops the fdiv error sideband.
        if is_one(b):
            return a
        if is_neg_one(b):
            return make_neg(builder, a)
        if b in cval:
            c = cval[b]
            k = _ilog2_exact(c)
            if k is not None:
                return emit_float_operation(builder, FloatMulPow2(-k), [a])
            if c != 0.0 and math.isfinite(c):
                return emit_float_operation(builder, FloatMul(), [a, emit_float_const(builder, 1.0 / c)])
        return emit_float_operation(builder, FloatDiv(), [a, b])

    def bool_const(vid: ValueId) -> bool | None:
        node = hir.nodes[vid]
        return node.value if isinstance(node, BoolConst) else None

    def build_value(builder: HirBuilder, vid: ValueId, node: Node, remap: dict[ValueId, ValueId]) -> ValueId:
        match node:
            case FloatConst(value=value):
                return emit_float_const(builder, value)
            case Operation(operator=FloatNeg(), operands=(a,)):
                return make_neg(builder, remap[a])
            case Operation(operator=FloatAdd(), operands=(a, b)):
                return reduce_add(builder, remap[a], remap[b])
            case Operation(operator=FloatMul(), operands=(a, b)):
                return reduce_mul(builder, remap[a], remap[b])
            case Operation(operator=FloatDiv(), operands=(a, b)):
                return reduce_div(builder, remap[a], remap[b])
            case Operation(operator=IntToFloat(), operands=(a,)):
                # i2f(f2i(x)) is a round-toward-zero of the FLOAT x, which is FloatTrunc(x) exactly (f2i(x) is x's
                # integer value, exactly representable back as a float because x already was one). It collapses to x
                # itself only when x is already integer-valued, so the truncation is provably a no-op. The reverse,
                # f2i(i2f(n)), is NOT rewritten: i2f rounds an integer wider than the mantissa, so it is not identity.
                inner = hir.nodes[a]
                if isinstance(inner, Operation) and isinstance(inner.operator, FloatToInt):
                    x = inner.operands[0]
                    return remap[x] if is_integral(x) else emit_float_operation(builder, FloatTrunc(), [remap[x]])
                return copy_node(builder, node, remap)
            case Operation(
                operator=FloatFloor() | FloatCeil() | FloatRound() | FloatTrunc(), operands=(a,)
            ) if is_integral(a):
                return remap[a]  # rounding an already-integer-valued float is idempotent
            case Operation(operator=FloatMulPow2(k=0), operands=(a,)):
                return remap[a]
            case Operation(operator=FloatMulPow2(k=k), operands=(a,)):
                return emit_float_operation(builder, FloatMulPow2(k), [remap[a]])
            case Operation(operator=Select(), operands=(_cond, a, b)) if remap[a] == remap[b]:
                return remap[a]  # select(c, X, X) == X
            case Operation(operator=BoolSelect(), operands=(cond, a, b)):
                if remap[a] == remap[b]:
                    return remap[a]
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
