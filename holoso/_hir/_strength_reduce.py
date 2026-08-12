"""
HIR algebraic strength reduction and constant folding. Every rewrite here is licensed by the fastmath charter in
DESIGN.md (Direction): the identities hold unconditionally for every value, and no rewrite may consult a numeric format
or decline because the datapath would answer differently.
"""

import math
import sys

from ._const import BoolConst, Const, FloatConst, IntConst
from ._copy import copy_node, rebuild
from ._scaling import Scaling, scaling_of
from .._util import BlockId, ValueId
from ._ir import Hir, HirBuilder, Node, Operation, Phi
from ._operators import (
    BoolAnd,
    BoolNot,
    BoolOr,
    BoolSelect,
    BoolXor,
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
    IntSelect,
    IntShiftLeft,
    IntShiftRight,
    IntSub,
    IntToFloat,
    NoNumber,
    Operator,
)

_MUX = (FloatSelect, BoolSelect, IntSelect)  # the three scalar families share both universal mux identities


def _sole_operand(node: Node) -> ValueId:
    assert isinstance(node, Operation) and len(node.operands) == 1
    return node.operands[0]


def _int_pow2_exponent(c: int) -> int | None:
    """Return ``k`` if ``c == 2**k`` for a positive ``c``, else ``None``."""
    return c.bit_length() - 1 if c > 0 and c & (c - 1) == 0 else None


def run(hir: Hir) -> Hir:
    """
    Fold every constant expression, rewrite the fast-math float and integer identities, exact power-of-two scaling,
    constant division and the conversion round trips, and reduce the if-conversion muxes. All before hardware
    selection, and none of it minting a LEFT shift, whose count the machine-word substitution below MIR settles.
    """
    known: dict[ValueId, Const] = {}  # constants this pass established, keyed by the id it built them under
    neg_of: dict[ValueId, ValueId] = {}  # both numeric families share it: an id names one node of one family
    bwnot_of: dict[ValueId, ValueId] = {}
    integral: set[ValueId] = set()  # integer-valued floats a rounding is the identity over (a constant one folds)

    def emit_const(builder: HirBuilder, const: Const) -> ValueId:
        new_id = builder.const_node(const)
        known[new_id] = const
        return new_id

    def emit_float_const(builder: HirBuilder, value: float) -> ValueId:
        return emit_const(builder, FloatConst(value))

    def emit_int_const(builder: HirBuilder, value: int) -> ValueId:
        return emit_const(builder, IntConst(value))

    def float_of(vid: ValueId) -> float | None:
        const = known.get(vid)
        return const.value if isinstance(const, FloatConst) else None

    def int_of(vid: ValueId) -> int | None:
        const = known.get(vid)
        return const.value if isinstance(const, IntConst) else None

    def is_one(vid: ValueId) -> bool:
        return float_of(vid) == 1.0

    def is_neg_one(vid: ValueId) -> bool:
        return float_of(vid) == -1.0

    def involution(builder: HirBuilder, memo: dict[ValueId, ValueId], operator: Operator, value: ValueId) -> ValueId:
        """Apply a self-inverse operator: over one this pass already minted, the base answers instead of a new node."""
        base = memo.get(value)
        if base is not None:
            return base
        new_id = builder.operation(operator, [value])
        memo[new_id] = value
        return new_id

    def make_neg(builder: HirBuilder, value: ValueId) -> ValueId:
        return involution(builder, neg_of, FloatNeg(), value)

    def make_ineg(builder: HirBuilder, value: ValueId) -> ValueId:
        return involution(builder, neg_of, IntNeg(), value)

    def make_ibwnot(builder: HirBuilder, value: ValueId) -> ValueId:
        return involution(builder, bwnot_of, IntBwNot(), value)

    def opposites(a: ValueId, b: ValueId) -> bool:
        return neg_of.get(a) == b or neg_of.get(b) == a

    def complements(a: ValueId, b: ValueId) -> bool:
        return bwnot_of.get(a) == b or bwnot_of.get(b) == a

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

    def emit_exponent(builder: HirBuilder, base: ValueId, scaling: Scaling) -> ValueId:
        """An exponent scaling over a value: the scaler, the sign sideband it rides with, or at zero exponent neither."""
        assert scaling.is_power_of_two
        scaled = builder.operation(FloatMulPow2(scaling.k), [base]) if scaling.k else base
        return make_neg(builder, scaled) if scaling.negative else scaled

    def emit_scaling(builder: HirBuilder, base: ValueId, scaling: Scaling) -> ValueId | None:
        """
        The one node a scaling becomes over a value, or None where no host float names its coefficient and the
        operands must stand as written -- each of which the machine layer can still carry apart, where this cannot.
        """
        if scaling.is_power_of_two:
            return emit_exponent(builder, base, scaling)
        coefficient = scaling.coefficient()
        return (
            None
            if coefficient is None
            else reduce_algebra(builder, FloatMul(), [base, emit_float_const(builder, coefficient)])
        )

    def scale_or_multiply(builder: HirBuilder, a: ValueId, b: ValueId) -> ValueId:
        if is_neg_one(a):
            return make_neg(builder, b)
        if is_neg_one(b):
            return make_neg(builder, a)
        for const_side, other in ((b, a), (a, b)):
            scale = float_of(const_side)
            # A unit scaling is ``x*1``, left to the declared identity rather than minted as a shift; ``x*-1`` never
            # reaches here, the negation above having claimed it.
            if (
                scale is not None
                and (scaling := scaling_of(scale)) is not None
                and scaling.is_power_of_two
                and scaling.k
            ):
                return emit_exponent(builder, other, scaling)
        return reduce_algebra(builder, FloatMul(), [a, b])

    def scaling(remap: dict[ValueId, ValueId], old_id: ValueId) -> tuple[ValueId, Scaling] | None:
        """The value an old-graph node scales and the constant it scales it by; ``None`` if it scales nothing."""
        match hir.nodes[old_id]:
            case Operation(operator=FloatMulPow2(k=k), operands=(x,)):
                return x, Scaling(1.0, k, False)
            case Operation(operator=FloatNeg(), operands=(x,)):
                return x, Scaling(1.0, 0, True)  # a negation is a scaling by -1, and the cheapest one there is
            case Operation(operator=FloatMul(), operands=(x, y)):
                for base, other in ((x, y), (y, x)):
                    factor = float_of(remap[other])
                    if factor is not None and (found := scaling_of(factor)) is not None:
                        return base, found
        return None

    def reassociate(
        builder: HirBuilder, remap: dict[ValueId, ValueId], outer: Scaling, old_operand: ValueId
    ) -> ValueId | None:
        """
        Scaling an already-scaled value scales it once. Cost is never the question: the composed node replaces the
        outer scaling one for one, and the inner survives only while another consumer still wants it.
        """
        found = scaling(remap, old_operand)
        if found is None:
            return None
        base, inner = found
        return emit_scaling(builder, remap[base], outer.compose(inner))

    def reduce_mul(builder: HirBuilder, remap: dict[ValueId, ValueId], old_a: ValueId, old_b: ValueId) -> ValueId:
        # The only reduction that reads the shape of the OLD graph, hence the old ids: composition needs the operand's
        # own operator, which the rebuilt operand no longer names once it has been reduced.
        for const_side, other in ((old_b, old_a), (old_a, old_b)):
            factor = float_of(remap[const_side])
            if factor is not None and (outer := scaling_of(factor)) is not None:
                fused = reassociate(builder, remap, outer, other)
                if fused is not None:
                    return fused
        return scale_or_multiply(builder, remap[old_a], remap[old_b])

    def reduce_mul_pow2(builder: HirBuilder, remap: dict[ValueId, ValueId], old_a: ValueId, k: int) -> ValueId:
        fused = reassociate(builder, remap, Scaling(1.0, k, False), old_a)
        return fused if fused is not None else builder.operation(FloatMulPow2(k), [remap[old_a]])

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
            scaling = scaling_of(divisor)
            if scaling is not None and scaling.is_power_of_two:  # by a signed power of two: its negated exponent
                return emit_exponent(builder, a, Scaling(1.0, -scaling.k, scaling.negative))
            # The same normality test as the composition above: a divisor whose reciprocal rails or falls into the
            # host's subnormals has none this arithmetic can carry, so the division stands as written.
            reciprocal = 1.0 / divisor
            if sys.float_info.min <= abs(reciprocal) < math.inf:
                return builder.operation(FloatMul(), [a, emit_float_const(builder, reciprocal)])
        return reduce_algebra(builder, FloatDiv(), [a, b])

    def reduce_iadd(builder: HirBuilder, a: ValueId, b: ValueId) -> ValueId:
        if opposites(a, b):
            return emit_int_const(builder, 0)
        return reduce_algebra(builder, IntAdd(), [a, b])

    def reduce_isub(builder: HirBuilder, a: ValueId, b: ValueId) -> ValueId:
        if a == b:
            return emit_int_const(builder, 0)
        if int_of(b) == 0:
            return a  # stated here, not as a declared identity, which the shared algebra would drop from either side
        if int_of(a) == 0:
            return make_ineg(builder, b)
        return reduce_algebra(builder, IntSub(), [a, b])

    def reduce_imul(builder: HirBuilder, a: ValueId, b: ValueId) -> ValueId:
        if int_of(a) == -1:
            return make_ineg(builder, b)
        if int_of(b) == -1:
            return make_ineg(builder, a)
        for const_side, other in ((b, a), (a, b)):
            scale = int_of(const_side)
            if scale is not None:
                k = _int_pow2_exponent(scale)
                if k:  # a zero exponent is ``x*1``, left to the declared identity rather than minted as a scaling
                    return builder.operation(IntMulPow2(k), [other])
        return reduce_algebra(builder, IntMul(), [a, b])

    def reduce_idiv(builder: HirBuilder, a: ValueId, b: ValueId) -> ValueId:
        if a == b:
            return emit_int_const(builder, 1)
        if int_of(a) == 0:
            return emit_int_const(builder, 0)
        divisor = int_of(b)
        if divisor == 1:
            return a
        if divisor == -1:
            return make_ineg(builder, a)
        if divisor is not None:
            k = _int_pow2_exponent(divisor)
            if k is not None:
                # The arithmetic right shift IS the floor division, exactly, negative dividends included.
                return builder.operation(IntShiftRight(), [a, emit_int_const(builder, k)])
        return reduce_algebra(builder, IntDivFloor(), [a, b])

    def reduce_imod(builder: HirBuilder, a: ValueId, b: ValueId) -> ValueId:
        if a == b or int_of(a) == 0:
            return emit_int_const(builder, 0)
        divisor = int_of(b)
        if divisor is not None:
            if abs(divisor) == 1:
                return emit_int_const(builder, 0)
            k = _int_pow2_exponent(divisor)
            if k is not None:
                # The floor remainder over a positive power of two IS the infinite two's-complement mask, negative
                # operands included, which is why the sign of the dividend never has to be asked.
                return builder.operation(IntBwAnd(), [a, emit_int_const(builder, divisor - 1)])
        return reduce_algebra(builder, IntMod(), [a, b])

    def reduce_ixor(builder: HirBuilder, a: ValueId, b: ValueId) -> ValueId:
        if a == b:
            return emit_int_const(builder, 0)
        if complements(a, b):
            return emit_int_const(builder, -1)
        if int_of(a) == -1:
            return make_ibwnot(builder, b)
        if int_of(b) == -1:
            return make_ibwnot(builder, a)  # ``x ^ -1`` is the complement at every width
        return reduce_algebra(builder, IntBwXor(), [a, b])

    def reduce_iand(builder: HirBuilder, a: ValueId, b: ValueId) -> ValueId:
        if a == b:
            return a
        if complements(a, b):
            return emit_int_const(builder, 0)
        return reduce_algebra(builder, IntBwAnd(), [a, b])

    def reduce_ior(builder: HirBuilder, a: ValueId, b: ValueId) -> ValueId:
        if a == b:
            return a
        if complements(a, b):
            return emit_int_const(builder, -1)
        return reduce_algebra(builder, IntBwOr(), [a, b])

    def reduce_bxor(builder: HirBuilder, a: ValueId, b: ValueId) -> ValueId:
        if a == b:
            return emit_const(builder, BoolConst(False))
        if bool_of(a) is True:
            return builder.operation(BoolNot(), [b])
        if bool_of(b) is True:
            return builder.operation(BoolNot(), [a])  # the 1-bit complement; MIR folds it into its consumer
        return reduce_algebra(builder, BoolXor(), [a, b])

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
                    # the absorbing zero. Whether this costs the build is settled by the refusal gate, once every
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
            case Operation(operator=IntShiftLeft() | IntShiftRight(), operands=(a, count)) if int_of(remap[count]) == 0:
                # Stated here, not at selection, because this is where the if-conversion budget counts the op --
                # and the shared algebra cannot state it, dropping an identity operand wherever it sits.
                return remap[a]
            case Operation(operator=IntToFloat(), operands=(a,)) if isinstance(inner_operator(a), FloatToInt):
                return reduce_rounding(builder, FloatTrunc(), remap[_sole_operand(hir.nodes[a])])  # float(int(x))
            case Operation(operator=FloatNeg(), operands=(a,)):
                return make_neg(builder, remap[a])
            case Operation(operator=FloatAdd(), operands=(a, b)):
                return reduce_add(builder, remap[a], remap[b])
            case Operation(operator=FloatMul(), operands=(a, b)):
                return reduce_mul(builder, remap, a, b)
            case Operation(operator=FloatMulPow2(k=k), operands=(a,)):
                return reduce_mul_pow2(builder, remap, a, k)
            case Operation(operator=FloatDiv(), operands=(a, b)):
                return reduce_div(builder, remap[a], remap[b])
            case Operation(operator=(FloatRound() | FloatFloor() | FloatCeil() | FloatTrunc()) as op, operands=(a,)):
                return reduce_rounding(builder, op, remap[a])
            case Operation(operator=IntToFloat(), operands=(a,)):
                return emit_integral(builder, IntToFloat(), remap[a])
            case Operation(operator=IntNeg(), operands=(a,)):
                return make_ineg(builder, remap[a])
            case Operation(operator=IntBwNot(), operands=(a,)):
                return make_ibwnot(builder, remap[a])
            case Operation(operator=IntAdd(), operands=(a, b)):
                return reduce_iadd(builder, remap[a], remap[b])
            case Operation(operator=IntSub(), operands=(a, b)):
                return reduce_isub(builder, remap[a], remap[b])
            case Operation(operator=IntMul(), operands=(a, b)):
                return reduce_imul(builder, remap[a], remap[b])
            case Operation(operator=IntDivFloor(), operands=(a, b)):
                return reduce_idiv(builder, remap[a], remap[b])
            case Operation(operator=IntMod(), operands=(a, b)):
                return reduce_imod(builder, remap[a], remap[b])
            case Operation(operator=IntBwXor(), operands=(a, b)):
                return reduce_ixor(builder, remap[a], remap[b])
            case Operation(operator=IntBwAnd(), operands=(a, b)):
                return reduce_iand(builder, remap[a], remap[b])
            case Operation(operator=IntBwOr(), operands=(a, b)):
                return reduce_ior(builder, remap[a], remap[b])
            case Operation(operator=BoolXor(), operands=(a, b)):
                return reduce_bxor(builder, remap[a], remap[b])
            case Operation(operator=BoolAnd() | BoolOr(), operands=(a, b)) if remap[a] == remap[b]:
                return remap[a]  # the idempotent connectives: both operands one value, so the gate names it too
            case Operation(operator=IntComparison() as relation, operands=(a, b)) if remap[a] == remap[b]:
                # Reflexive over every integer -- no NaN to except -- so the relation's answer over one value is
                # its answer over any.
                return emit_const(builder, relation.evaluate([IntConst(0), IntConst(0)]))
            case Operation(operator=BoolSelect(), operands=(cond, a, b)):
                return reduce_bselect(builder, remap[cond], remap[a], remap[b])
            case Operation(operator=operator, operands=operands):
                return reduce_algebra(builder, operator, [remap[o] for o in operands])
            case _:
                return copy_node(builder, node, remap)

    return rebuild(hir, build_value)
