"""
HIR algebraic strength reduction and constant folding. Every rewrite here is licensed by the fastmath charter in
DESIGN.md (Direction): the identities hold unconditionally for every value, and no rewrite may consult a numeric format
or decline because the datapath would answer differently.
"""

import math

from ._const import BoolConst, Const, FloatConst
from ._copy import copy_node, rebuild
from .._util import BlockId, ValueId
from ._ir import Hir, HirBuilder, Node, Operation, Phi
from ._operators import (
    BoolAnd,
    BoolNot,
    BoolOr,
    BoolSelect,
    FloatAdd,
    FloatCeil,
    FloatDiv,
    FloatFloor,
    FloatMul,
    FloatMulPow2,
    FloatNeg,
    FloatRound,
    FloatSelect,
    FloatToInt,
    FloatTrunc,
    IntSelect,
    IntToFloat,
    NoNumber,
    Operator,
)

_MUX = (FloatSelect, BoolSelect, IntSelect)  # the three scalar families share both universal mux identities


def _sole_operand(node: Node) -> ValueId:
    assert isinstance(node, Operation) and len(node.operands) == 1
    return node.operands[0]


def _ilog2_exact(c: float) -> int | None:
    """Return ``k`` if ``c == 2**k`` exactly for a positive ``c``, else ``None``."""
    if c <= 0.0 or not math.isfinite(c):
        return None
    mantissa, exponent = math.frexp(c)  # c == mantissa * 2**exponent, mantissa in [0.5, 1)
    return exponent - 1 if mantissa == 0.5 else None


def run(hir: Hir) -> Hir:
    """
    Fold every constant expression, rewrite the fast-math float identities, exact power-of-two scaling, constant
    division and the conversion round trips, and reduce the if-conversion muxes. All before hardware selection.
    """
    known: dict[ValueId, Const] = {}  # constants this pass established, keyed by the id it built them under
    neg_of: dict[ValueId, ValueId] = {}
    integral: set[ValueId] = set()  # integer-valued floats a rounding is the identity over (a constant one folds)

    def emit_const(builder: HirBuilder, const: Const) -> ValueId:
        new_id = builder.const_node(const)
        known[new_id] = const
        return new_id

    def emit_float_const(builder: HirBuilder, value: float) -> ValueId:
        return emit_const(builder, FloatConst(value))

    def float_of(vid: ValueId) -> float | None:
        const = known.get(vid)
        return const.value if isinstance(const, FloatConst) else None

    def is_one(vid: ValueId) -> bool:
        return float_of(vid) == 1.0

    def is_neg_one(vid: ValueId) -> bool:
        return float_of(vid) == -1.0

    def make_neg(builder: HirBuilder, value: ValueId) -> ValueId:
        base = neg_of.get(value)
        if base is not None:
            return base
        new_id = builder.operation(FloatNeg(), [value])
        neg_of[new_id] = value
        return new_id

    def opposites(a: ValueId, b: ValueId) -> bool:
        return neg_of.get(a) == b or neg_of.get(b) == a

    def uniform_const_arm(arms: tuple[tuple[BlockId, ValueId], ...], remap: dict[ValueId, ValueId]) -> Const | None:
        values = [known.get(remap[arm]) for _, arm in arms]
        first = values[0] if values else None
        return first if first is not None and all(value == first for value in values) else None

    def reduce_algebra(builder: HirBuilder, operator: Operator, operands: list[ValueId]) -> ValueId:
        """
        The algebra an operator declares, applied over an operand the compiler cannot see: an absorbing operand fixes
        the result regardless of the others (``x or True``), and an identity operand drops out (``x and True`` -> x).
        Sound for any associative operator that declares them, which every declaration here is. This is the shared
        fallback of the reductions rather than a pass of its own, so no rewrite escapes it.
        """
        consts = [known.get(operand) for operand in operands]
        absorbing = operator.absorbing()
        if absorbing is not None and absorbing in consts:
            return emit_const(builder, absorbing)
        identity = operator.identity()
        if identity is not None:
            survivors = [operand for operand, const in zip(operands, consts, strict=True) if const != identity]
            if len(survivors) == 1:
                return survivors[0]
        return builder.operation(operator, operands)

    def reduce_add(builder: HirBuilder, a: ValueId, b: ValueId) -> ValueId:
        if opposites(a, b):
            return emit_float_const(builder, 0.0)
        return reduce_algebra(builder, FloatAdd(), [a, b])

    def reduce_mul(builder: HirBuilder, a: ValueId, b: ValueId) -> ValueId:
        if is_neg_one(a):
            return make_neg(builder, b)
        if is_neg_one(b):
            return make_neg(builder, a)
        for const_side, other in ((b, a), (a, b)):
            scale = float_of(const_side)
            if scale is not None:
                k = _ilog2_exact(scale)
                if k:  # a zero exponent is ``x*1``, left to the declared identity rather than minted as a shift
                    return builder.operation(FloatMulPow2(k), [other])
        return reduce_algebra(builder, FloatMul(), [a, b])

    def reduce_div(builder: HirBuilder, a: ValueId, b: ValueId) -> ValueId:
        if a == b:
            return emit_float_const(builder, 1.0)
        if float_of(a) == 0.0:
            return emit_float_const(builder, 0.0)  # ``0/x == 0``: a numerator rule, so no operator algebra states it
        if is_one(b):
            return a
        if is_neg_one(b):
            return make_neg(builder, a)
        divisor = float_of(b)
        # A zero divisor is excluded because there is no reciprocal to multiply by at all. An infinite one is excluded
        # only because nothing has needed the fold; ``1/inf`` is ``0.0``, a perfectly good second factor.
        if divisor is not None and divisor != 0.0 and math.isfinite(divisor):
            k = _ilog2_exact(divisor)
            if k is not None:
                return builder.operation(FloatMulPow2(-k), [a])
            return builder.operation(FloatMul(), [a, emit_float_const(builder, 1.0 / divisor)])
        return reduce_algebra(builder, FloatDiv(), [a, b])

    def emit_integral(builder: HirBuilder, operator: Operator, value: ValueId) -> ValueId:
        """Emit an operation whose result is an integer-valued float, so a rounding of it is later recognized free."""
        new_id = builder.operation(operator, [value])
        integral.add(new_id)
        return new_id

    def reduce_rounding(builder: HirBuilder, operator: Operator, value: ValueId) -> ValueId:
        return value if value in integral else emit_integral(builder, operator, value)

    def inner_operator(vid: ValueId) -> Operator | None:
        node = hir.nodes[vid]
        return node.operator if isinstance(node, Operation) else None

    def bool_of(vid: ValueId) -> bool | None:
        const = known.get(vid)
        return const.value if isinstance(const, BoolConst) else None

    def reduce_bselect(builder: HirBuilder, cond: ValueId, a: ValueId, b: ValueId) -> ValueId:
        """
        Reduce ``bselect(cond, a, b)`` using its constant arms, which the universal mux identity in ``build_value``
        has already made distinct; the NOTs fold consumer-side at MIR lowering. Every connective minted here goes
        through the declared algebra, because a constant arm often makes the gate it becomes a constant in turn -- a
        one-shot latch reduces to ``first and False``, which is the latch's live-out written the long way.
        """
        assert a != b, "equal arms would read as the True/False entry below; the mux identity must have reduced them"
        a_const, b_const = bool_of(a), bool_of(b)
        if b == cond:
            return reduce_algebra(builder, BoolAnd(), [cond, a])  # (c, a, c) == c and a: Python's eager ``and`` shape
        if a == cond:
            return reduce_algebra(builder, BoolOr(), [cond, b])  # (c, c, b) == c or b: Python's eager ``or`` shape
        if a_const is not None and b_const is not None:  # both constant and distinct -> True/False or False/True
            return cond if a_const else builder.operation(BoolNot(), [cond])
        if a_const is True:
            return reduce_algebra(builder, BoolOr(), [cond, b])  # (c, True, b) == c or b
        if a_const is False:
            not_cond = builder.operation(BoolNot(), [cond])
            return reduce_algebra(builder, BoolAnd(), [not_cond, b])  # (c, False, b) == ~c and b
        if b_const is True:
            not_cond = builder.operation(BoolNot(), [cond])
            return reduce_algebra(builder, BoolOr(), [not_cond, a])  # (c, a, True) == ~c or a
        if b_const is False:
            return reduce_algebra(builder, BoolAnd(), [cond, a])  # (c, a, False) == c and a
        return builder.operation(BoolSelect(), [cond, a, b])  # both arms dynamic: keep the mux

    def build_value(builder: HirBuilder, node: Node, remap: dict[ValueId, ValueId]) -> ValueId:
        if isinstance(node, Operation):
            # Ask what the operation names, but only where every operand is known -- and then no identity applies,
            # because an identity speaks for exactly the operand the compiler cannot see.
            consts = [const for operand in node.operands if (const := known.get(remap[operand])) is not None]
            if len(consts) == len(node.operands):
                try:
                    folded = node.operator.evaluate(consts)
                except NoNumber:
                    # The operation names no number, so it is copied verbatim and no rewrite below is offered it: with
                    # every operand in view there is nothing left for an identity to speak for, and ``inf*0`` is not
                    # the absorbing zero. Whether this costs the build is settled by the survivor sweep, once every
                    # deletion has had its turn.
                    return copy_node(builder, node, remap)
                return emit_const(builder, folded)
        match node:
            case Const():
                return emit_const(builder, node)
            case Phi(arms=arms) if (uniform := uniform_const_arm(arms, remap)) is not None:
                return emit_const(builder, uniform)  # every arm merges the same constant, so the merge names it too
            case Operation(operator=mux, operands=(cond, a, b)) if (
                isinstance(mux, _MUX) and bool_of(remap[cond]) is not None
            ):
                # A known selector picks an arm and the other becomes irrelevant, so neither is evaluated here. What
                # the unselected one names is settled where it is folded, not by the mux that never selects it.
                return remap[a] if bool_of(remap[cond]) else remap[b]
            case Operation(operator=mux, operands=(_cond, a, b)) if isinstance(mux, _MUX) and remap[a] == remap[b]:
                # The selector cannot matter once both arms name one value -- the shape an if-converted diamond leaves
                # once its spliced arms are interned into one block. Reducing it here is also what keeps the
                # constant-arm rules below reachable only for arms that DIFFER.
                return remap[a]
            case Operation(operator=IntToFloat(), operands=(a,)) if isinstance(inner_operator(a), FloatToInt):
                return reduce_rounding(builder, FloatTrunc(), remap[_sole_operand(hir.nodes[a])])  # float(int(x))
            case Operation(operator=FloatNeg(), operands=(a,)):
                return make_neg(builder, remap[a])
            case Operation(operator=FloatAdd(), operands=(a, b)):
                return reduce_add(builder, remap[a], remap[b])
            case Operation(operator=FloatMul(), operands=(a, b)):
                return reduce_mul(builder, remap[a], remap[b])
            case Operation(operator=FloatDiv(), operands=(a, b)):
                return reduce_div(builder, remap[a], remap[b])
            case Operation(operator=(FloatRound() | FloatFloor() | FloatCeil() | FloatTrunc()) as op, operands=(a,)):
                return reduce_rounding(builder, op, remap[a])
            case Operation(operator=IntToFloat(), operands=(a,)):
                return emit_integral(builder, IntToFloat(), remap[a])
            case Operation(operator=BoolSelect(), operands=(cond, a, b)):
                return reduce_bselect(builder, remap[cond], remap[a], remap[b])
            case Operation(operator=operator, operands=operands):
                return reduce_algebra(builder, operator, [remap[o] for o in operands])
            case _:
                return copy_node(builder, node, remap)

    return rebuild(hir, build_value)
