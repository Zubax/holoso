"""
The single HIR-importing module of the partial evaluator: the const type the operators fold over. The fold itself is
applied one level up, in `_express`, and it calls the selected operator's own `evaluate`, so a value the partial
evaluator computes statically and a value HIR folding computes for the same residual expression cannot differ -- one
expression, one answer, per the fastmath charter. Every other `_pe` module stays free of direct HIR imports; the
confinement is enforced by `tests/test_eel_layering.py`.
"""

from ..._hir import (
    BoolType as _BoolType,
    Const as Const,
    FloatType,
    IntType,
    NoNumber as NoNumber,
    Operator as Operator,
    const_value as const_value,
    make_const as make_const,
)
from .._ir import ScalarType


def stype_of(ty: object) -> ScalarType:
    if isinstance(ty, _BoolType):
        return ScalarType.BOOL
    if isinstance(ty, IntType):
        return ScalarType.INT
    assert isinstance(ty, FloatType)
    return ScalarType.FLOAT
