"""
The single HIR-importing module of the partial evaluator: the operator selection tables, and the const type
the operators fold over. The fold itself is applied one level up, in ``_express``, and it calls the selected
operator's own ``evaluate``, so a value the partial evaluator computes statically and a value HIR folding computes
for the same residual expression cannot differ -- one expression, one answer, per the fastmath charter. Every other
``_pe`` module stays free of direct HIR imports; the confinement is enforced by ``tests/test_eel_layering.py``.
"""

from ..._hir import (
    BoolAnd,
    BoolConst,
    BoolNot,
    BoolOr,
    BoolToFloat,
    BoolToInt,
    BoolType as _BoolType,
    BoolXor,
    Const as Const,
    FloatAdd,
    FloatConst,
    FloatDiv,
    FloatEqual,
    FloatGreater,
    FloatGreaterOrEqual,
    FloatLess,
    FloatLessOrEqual,
    FloatMul,
    FloatNeg,
    FloatNotEqual,
    FloatToBool,
    FloatToInt,
    FloatType,
    IntAdd,
    IntBwAnd,
    IntBwNot,
    IntBwOr,
    IntBwXor,
    IntConst,
    IntDivFloor,
    IntEqual,
    IntGreater,
    IntGreaterOrEqual,
    IntLess,
    IntLessOrEqual,
    IntMod,
    IntMul,
    IntNeg,
    IntNotEqual,
    IntShiftLeft,
    IntShiftRight,
    IntSub,
    IntToBool,
    IntToFloat,
    IntType,
    NoNumber as NoNumber,
    Operator as Operator,
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


def _stype_of(ty: object) -> ScalarType:
    if isinstance(ty, _BoolType):
        return ScalarType.BOOL
    if isinstance(ty, IntType):
        return ScalarType.INT
    assert isinstance(ty, FloatType)
    return ScalarType.FLOAT


def result_stype(operator: Operator) -> ScalarType:
    return _stype_of(operator.signature.result_type)


def operand_stypes(operator: Operator) -> list[ScalarType]:
    return [_stype_of(ty) for ty in operator.signature.operand_types]


INT_BINARY: dict[BinaryOp, Operator] = {
    BinaryOp.ADD: IntAdd(),
    BinaryOp.SUB: IntSub(),
    BinaryOp.MUL: IntMul(),
    BinaryOp.FLOORDIV: IntDivFloor(),
    BinaryOp.MOD: IntMod(),
    BinaryOp.LSHIFT: IntShiftLeft(),
    BinaryOp.RSHIFT: IntShiftRight(),
    BinaryOp.BITAND: IntBwAnd(),
    BinaryOp.BITOR: IntBwOr(),
    BinaryOp.BITXOR: IntBwXor(),
}

FLOAT_ADD: Operator = FloatAdd()
FLOAT_MUL: Operator = FloatMul()
FLOAT_DIV: Operator = FloatDiv()
FLOAT_NEG: Operator = FloatNeg()
INT_NEG: Operator = IntNeg()
INT_BW_NOT: Operator = IntBwNot()
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

INT_COMPARE: dict[CompareOp, Operator] = {
    CompareOp.LT: IntLess(),
    CompareOp.LE: IntLessOrEqual(),
    CompareOp.GT: IntGreater(),
    CompareOp.GE: IntGreaterOrEqual(),
    CompareOp.EQ: IntEqual(),
    CompareOp.NE: IntNotEqual(),
}

FLOAT_COMPARE: dict[CompareOp, Operator] = {
    CompareOp.LT: FloatLess(),
    CompareOp.LE: FloatLessOrEqual(),
    CompareOp.GT: FloatGreater(),
    CompareOp.GE: FloatGreaterOrEqual(),
    CompareOp.EQ: FloatEqual(),
    CompareOp.NE: FloatNotEqual(),
}
