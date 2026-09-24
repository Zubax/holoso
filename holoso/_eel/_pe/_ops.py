"""
The single HIR-importing module of the partial evaluator: the conversion table, and the const type the
operators fold over. The fold itself is applied one level up, in `_express`, and it calls the selected
operator's own `evaluate`, so a value the partial evaluator computes statically and a value HIR folding computes
for the same residual expression cannot differ -- one expression, one answer, per the fastmath charter. Every other
`_pe` module stays free of direct HIR imports; the confinement is enforced by `tests/test_eel_layering.py`.
"""

from ..._hir import (
    BoolConst,
    BoolToFloat,
    BoolToInt,
    BoolType as _BoolType,
    Const as Const,
    FloatConst,
    FloatToBool,
    FloatToInt,
    FloatType,
    IntConst,
    IntToBool,
    IntToFloat,
    IntType,
    NoNumber as NoNumber,
    Operator as Operator,
)
from .._ir import ScalarType


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


CONVERT: dict[tuple[ScalarType, ScalarType], Operator] = {
    (ScalarType.INT, ScalarType.FLOAT): IntToFloat(),
    (ScalarType.FLOAT, ScalarType.INT): FloatToInt(),
    (ScalarType.BOOL, ScalarType.FLOAT): BoolToFloat(),
    (ScalarType.FLOAT, ScalarType.BOOL): FloatToBool(),
    (ScalarType.BOOL, ScalarType.INT): BoolToInt(),
    (ScalarType.INT, ScalarType.BOOL): IntToBool(),
}
