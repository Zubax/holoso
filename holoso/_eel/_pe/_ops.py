"""
The single HIR-facing module of the partial evaluator: operator selection tables and constant folding.

Folding calls the selected operator's own ``evaluate``, so a value the partial evaluator computes statically and
a value HIR folding computes for the same residual expression cannot differ -- one expression, one answer, per
the fastmath charter. Every other ``_pe`` module stays free of direct HIR imports; the confinement is enforced
by ``tests/test_eel_layering.py``.
"""

from ..._hir import (
    BoolAnd,
    BoolConst,
    BoolNot,
    BoolOr,
    BoolToFloat,
    BoolToInt,
    BoolXor,
    BoolType as _BoolType,
    Const as Const,
    FloatAdd,
    FloatConst,
    FloatDiv,
    FloatExp2,
    FloatMul,
    FloatNeg,
    FloatRelational,
    FloatToBool,
    FloatToInt,
    FloatType,
    IntAdd,
    IntAnd,
    IntConst,
    IntDivFloor,
    IntMod,
    IntMul,
    IntNeg,
    IntNot,
    IntOr,
    IntRelational,
    IntShiftLeft,
    IntShiftRight,
    IntSub,
    IntToBool,
    IntToFloat,
    IntType,
    IntXor,
    NoNumber as NoNumber,
    Operator as Operator,
    RelationalOp,
)
from .._ir import BinaryOp, CompareOp, ScalarType


def make_const(value: bool | int | float) -> Const:
    if type(value) is bool:
        return BoolConst(value)
    if type(value) is int:
        return IntConst(value)
    assert type(value) is float
    return FloatConst(value)


def scalar_type(const: Const) -> ScalarType:
    match const:
        case BoolConst():
            return ScalarType.BOOL
        case IntConst():
            return ScalarType.INT
        case FloatConst():
            return ScalarType.FLOAT
    raise AssertionError(const)


def const_value(const: Const) -> bool | int | float:
    assert isinstance(const, (BoolConst, IntConst, FloatConst))
    return const.value


def result_stype(operator: Operator) -> ScalarType:
    result = operator.signature.result_type
    if isinstance(result, _BoolType):
        return ScalarType.BOOL
    if isinstance(result, IntType):
        return ScalarType.INT
    assert isinstance(result, FloatType)
    return ScalarType.FLOAT


INT_BINARY: dict[BinaryOp, Operator] = {
    BinaryOp.ADD: IntAdd(),
    BinaryOp.SUB: IntSub(),
    BinaryOp.MUL: IntMul(),
    BinaryOp.FLOORDIV: IntDivFloor(),
    BinaryOp.MOD: IntMod(),
    BinaryOp.LSHIFT: IntShiftLeft(),
    BinaryOp.RSHIFT: IntShiftRight(),
    BinaryOp.BITAND: IntAnd(),
    BinaryOp.BITOR: IntOr(),
    BinaryOp.BITXOR: IntXor(),
}

FLOAT_ADD: Operator = FloatAdd()
FLOAT_MUL: Operator = FloatMul()
FLOAT_DIV: Operator = FloatDiv()
FLOAT_NEG: Operator = FloatNeg()
FLOAT_EXP2: Operator = FloatExp2()
INT_NEG: Operator = IntNeg()
INT_NOT: Operator = IntNot()
BOOL_NOT: Operator = BoolNot()
BOOL_XOR: Operator = BoolXor()

BOOL_GATES: dict[BinaryOp, Operator] = {
    BinaryOp.AND: BoolAnd(),
    BinaryOp.OR: BoolOr(),
    BinaryOp.BITAND: BoolAnd(),
    BinaryOp.BITOR: BoolOr(),
    BinaryOp.BITXOR: BoolXor(),
}

CONVERT: dict[tuple[ScalarType, ScalarType], Operator] = {
    (ScalarType.INT, ScalarType.FLOAT): IntToFloat(),
    (ScalarType.FLOAT, ScalarType.INT): FloatToInt(),
    (ScalarType.BOOL, ScalarType.FLOAT): BoolToFloat(),
    (ScalarType.FLOAT, ScalarType.BOOL): FloatToBool(),
    (ScalarType.BOOL, ScalarType.INT): BoolToInt(),
    (ScalarType.INT, ScalarType.BOOL): IntToBool(),
}

_RELATION: dict[CompareOp, RelationalOp] = {
    CompareOp.LT: RelationalOp.LT,
    CompareOp.LE: RelationalOp.LE,
    CompareOp.GT: RelationalOp.GT,
    CompareOp.GE: RelationalOp.GE,
    CompareOp.EQ: RelationalOp.EQ,
    CompareOp.NE: RelationalOp.NE,
}


def int_relational(op: CompareOp) -> Operator:
    return IntRelational(_RELATION[op])


def float_relational(op: CompareOp) -> Operator:
    return FloatRelational(_RELATION[op])
