"""HIR algebraic strength reduction."""

import math

from ._const import BoolConst, FloatConst, IntConst
from ._copy import copy_node, rebuild
from .._util import ValueId
from ._ir import Hir, HirBuilder, Node, Operation, Phi
from ._types import FloatType, IntType
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


def _fits_binary64(value: int) -> bool:
    try:
        float(value)
    except OverflowError:
        return False
    return True


def _const_int_phis_used_only_as_float(hir: Hir) -> frozenset[ValueId]:
    """
    Integer phis whose every arm is a binary64-convertible IntConst and whose every use is an IntToFloat promotion.
    An arm beyond the carrier range disqualifies the phi: it stays integer and meets the MIR containment instead of
    leaking a raw OverflowError out of the rewrite.
    """
    candidates = {
        vid
        for vid, node in hir.nodes.items()
        if isinstance(node, Phi)
        and isinstance(node.type, IntType)
        and all(isinstance(arm := hir.nodes[value], IntConst) and _fits_binary64(arm.value) for _, value in node.arms)
    }
    if not candidates:
        return frozenset()
    for vid, node in hir.nodes.items():
        if isinstance(node, Operation):
            promotes = isinstance(node.operator, IntToFloat)
            candidates -= {operand for operand in node.operands if not promotes and operand in candidates}
        elif isinstance(node, Phi):
            candidates -= {value for _, value in node.arms if value in candidates and value != vid}
    for slot in hir.state_slots:
        candidates.discard(slot.live_out)
    for output in hir.outputs:
        candidates.discard(output.value)
    return frozenset(candidates)


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
    promotable_phis = _const_int_phis_used_only_as_float(hir)

    def is_integral(vid: ValueId) -> bool:
        """A value provably equal to an integer-valued float: a rounding result, a promoted integer, or a widened bool."""
        node = hir.nodes[vid]
        return isinstance(node, Operation) and isinstance(
            node.operator, (FloatFloor, FloatCeil, FloatRound, FloatTrunc, BoolToFloat, IntToFloat)
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

    def defers_to_runtime(vid: ValueId, keep_zero: bool = False) -> bool:
        """
        Whether an identity rewrite over this operand must yield to the runtime operator: the compiler can see
        the operand is non-finite (or zero, where zero breaks the identity), so the fold's answer could differ
        from the ZKF one. An unknown runtime operand keeps the chartered fast-math fold.
        """
        constant = cval.get(vid)
        if constant is None:
            return False
        return not math.isfinite(constant) or (not keep_zero and constant == 0.0)

    def reduce_add(builder: HirBuilder, a: ValueId, b: ValueId) -> ValueId:
        if is_zero(a):
            return b
        if is_zero(b):
            return a
        cancelled = b if neg_of.get(a) == b else a if neg_of.get(b) == a else None
        if cancelled is not None and not defers_to_runtime(cancelled, keep_zero=True):
            return emit_float_const(builder, 0.0)
        return emit_float_operation(builder, FloatAdd(), [a, b])

    def reduce_mul(builder: HirBuilder, a: ValueId, b: ValueId) -> ValueId:
        if is_zero(a) or is_zero(b):
            # Needs no runtime-divergence guard: ZKF zero kills unconditionally (0 x inf is 0.0, there is no
            # NaN), so this fold matches the hardware for every other operand.
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
        # x/x folds to 1.0 for a runtime operand under the charter, but a shared operand the compiler can see is
        # zero or non-finite (constant interning makes both sides of a deferred 0/0 or inf/inf the same node)
        # defers to the runtime divider, whose ZKF answer is 0.0, not 1.0.
        if a == b and not defers_to_runtime(a):
            return emit_float_const(builder, 1.0)
        if is_zero(a):
            # 0/y is the ZKF quotient for every y (zero kills; 0/0 and 0/inf are 0.0 in hardware too), so the
            # fold never diverges from the runtime; it does drop the fdiv error sideband for a zero divisor.
            return emit_float_const(builder, 0.0)
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
            case Operation(operator=IntToFloat(), operands=(a,)) if a in promotable_phis:
                return remap[a]  # the phi itself was rebuilt as a float phi; the conversion dissolves
            case Phi(type=IntType(), arms=arms) if vid in promotable_phis:
                # An all-constant integer phi consumed only as float promotes wholesale (the ``1 if c else 0``
                # idiom): the merge happens in the float bank instead of surviving to the integer containment.
                float_arms = []
                for pred, value in arms:
                    arm_node = hir.nodes[value]
                    assert isinstance(arm_node, IntConst)
                    float_arms.append((pred, emit_float_const(builder, float(arm_node.value))))
                return builder.phi(FloatType(), float_arms)
            case Operation(operator=IntToFloat(), operands=(a,)):
                # i2f(f2i(x)) is a round-toward-zero of the FLOAT x, which is FloatTrunc(x) exactly (f2i(x) is x's
                # integer value, exactly representable back as a float because x already was one). It collapses to x
                # itself only when x is already integer-valued, so the truncation is provably a no-op.
                inner = hir.nodes[a]
                if isinstance(inner, Operation) and isinstance(inner.operator, FloatToInt):
                    x = inner.operands[0]
                    return remap[x] if is_integral(x) else emit_float_operation(builder, FloatTrunc(), [remap[x]])
                return copy_node(builder, node, remap)
            case Operation(operator=FloatToInt(), operands=(a,)):
                # f2i(i2f(n)) collapses to n per the fastmath charter: int -> float -> int is the identity with the
                # promotion's precision loss deliberately ignored (an integer wider than the mantissa notwithstanding).
                # A promoted phi is excluded: its remap is the FLOAT phi, so the collapse would silently retype an
                # integer expression as float; the conversion stays and meets the integer containment downstream.
                inner = hir.nodes[a]
                if (
                    isinstance(inner, Operation)
                    and isinstance(inner.operator, IntToFloat)
                    and inner.operands[0] not in promotable_phis
                ):
                    return remap[inner.operands[0]]
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
    if b == cond:
        return builder.operation(BoolAnd(), [cond, a])  # (c, a, c) == c and a: Python's eager ``and`` shape
    if a == cond:
        return builder.operation(BoolOr(), [cond, b])  # (c, c, b) == c or b: Python's eager ``or`` shape
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
