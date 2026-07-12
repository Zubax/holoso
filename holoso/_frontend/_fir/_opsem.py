"""
Static execution semantics for the normalized scalar operations: one evaluation table over the closed value domain,
replacing the per-query-site ``_static_*`` pattern matching. Evaluation happens through real Python/numpy on the
domain's own objects, so each variant's provenance applies its own semantics by construction (a MetaInt computes
exact; an np scalar exactly as numpy would, 64-bit wraparound included -- faithful, so it folds). A None result means
"not statically evaluable here" -- the operation stays runtime. Zero division and invalid operations defer to
runtime, where the hardware's own defined error semantics and error flags apply; a float overflow folds to the
infinity DESIGN.md sanctions where the host produces one, and defers where the host raises instead (float ``**``);
a NaN never participates in a fold, neither as an operand nor as a result (ZKF has none).
"""

import enum
import math
import operator
from collections.abc import Callable

import numpy as np

from ..._util import RelationalOp
from ._value import MetaInt, NpFloat, NpInt, StaticBool, StaticFloat, StaticSeq, StaticValue, admit, as_python

_MAX_FOLD_BITS = 1 << 16  # refuse to fold an integer whose result would be astronomically wide


class BinOp(enum.Enum):
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    FLOORDIV = "//"
    MOD = "%"
    POW = "**"


class UnOp(enum.Enum):
    NEG = "-"
    POS = "+"


_NUMERIC = (MetaInt, NpInt, StaticFloat, NpFloat)

_BINOP_FN: dict[BinOp, Callable[[object, object], object]] = {
    BinOp.ADD: operator.add,
    BinOp.SUB: operator.sub,
    BinOp.MUL: operator.mul,
    BinOp.DIV: operator.truediv,
    BinOp.FLOORDIV: operator.floordiv,
    BinOp.MOD: operator.mod,
    BinOp.POW: operator.pow,
}


def _renumber(result: object) -> StaticValue | None:
    value = admit(result)
    if isinstance(value, (StaticFloat, NpFloat)) and math.isnan(value.value):
        return None  # ZKF has no NaN; a NaN fold defers to runtime, whose own semantics never produce one
    if isinstance(value, MetaInt) and value.value.bit_length() > _MAX_FOLD_BITS:
        return None  # one chokepoint bounds every integer op: a compact squaring chain must not exhaust the compiler
    return value


def _has_nan(*operands: StaticValue) -> bool:
    return any(isinstance(v, (StaticFloat, NpFloat)) and math.isnan(v.value) for v in operands)


def _too_wide(*operands: StaticValue) -> bool:
    # A pre-check, before any evaluation: an already-astronomical operand must not cost CPU or memory to refold.
    return any(isinstance(v, MetaInt) and v.value.bit_length() > _MAX_FOLD_BITS for v in operands)


def _evaluate(fn: Callable[[], object]) -> object | None:
    """
    The one evaluation harness: every host evaluation runs under the same errstate flags and defers on the same
    host faults, so the three operation kinds cannot drift apart. Evaluations never legitimately produce None.
    """
    try:
        with np.errstate(divide="raise", invalid="raise", over="ignore", under="ignore"):
            result = fn()
    except (ZeroDivisionError, OverflowError, ValueError, FloatingPointError):
        return None
    assert result is not None
    return result


def _pow_is_bounded(left: StaticValue, right: StaticValue) -> bool:
    """
    Pre-check before the power is even computed (the post-hoc width bound in :func:`_renumber` would arrive after
    the evaluation of ``2 ** 10**9`` has already exhausted the compiler).
    """
    if not isinstance(left, (MetaInt, NpInt)) or not isinstance(right, (MetaInt, NpInt)):
        return True  # float powers do not grow without bound
    exponent = right.value
    return exponent <= 0 or max(left.value.bit_length(), 1) * exponent <= _MAX_FOLD_BITS


def static_binop(op: BinOp, left: StaticValue, right: StaticValue) -> StaticValue | None:
    """
    Numeric arithmetic on static scalars, under each operand's own provenance semantics (exact bigint, numpy scalar,
    float64 fast-math). Sequences fall outside: list/tuple ``+``/``*`` are structural operations owned by the
    aggregate layer, not scalar arithmetic.
    """
    if not isinstance(left, _NUMERIC) or not isinstance(right, _NUMERIC) or _has_nan(left, right):
        return None
    if _too_wide(left, right):
        return None
    if op is BinOp.POW and not _pow_is_bounded(left, right):
        return None
    result = _evaluate(lambda: _BINOP_FN[op](as_python(left), as_python(right)))
    return None if result is None else _renumber(result)


_UNOP_FN: dict[UnOp, Callable[[object], object]] = {
    UnOp.NEG: operator.neg,  # type: ignore[dict-item]
    UnOp.POS: operator.pos,  # type: ignore[dict-item]
}


def static_unop(op: UnOp, operand: StaticValue) -> StaticValue | None:
    if not isinstance(operand, _NUMERIC) or _has_nan(operand) or _too_wide(operand):
        return None
    result = _evaluate(lambda: _UNOP_FN[op](as_python(operand)))
    return None if result is None else _renumber(result)


def static_compare(relation: RelationalOp, left: StaticValue, right: StaticValue) -> StaticBool | None:
    """
    A relational link over static scalars: Python compares a Python int with an int or float exactly, a numpy scalar
    applies numpy's own conversion rules to the whole pair, and booleans admit equality only (ordering booleans is
    rejected at lowering, not here). The comparison runs on the domain's own objects, so those semantics apply by
    construction.
    """
    if isinstance(left, StaticBool) != isinstance(right, StaticBool):
        return None
    if isinstance(left, StaticBool) and isinstance(right, StaticBool):
        if relation not in (RelationalOp.EQ, RelationalOp.NE):
            return None
        equal = left.value == right.value
        return StaticBool(equal if relation is RelationalOp.EQ else not equal)
    if not isinstance(left, _NUMERIC) or not isinstance(right, _NUMERIC) or _has_nan(left, right):
        return None
    result = _evaluate(lambda: bool(relation.apply(as_python(left), as_python(right))))  # type: ignore[arg-type]
    if result is None:
        return None
    assert isinstance(result, bool)
    return StaticBool(result)


def static_truth(value: StaticValue) -> bool | None:
    """Python truthiness for the value kinds whose truth is defined in the subset; None where it is not static."""
    match value:
        case StaticBool(value=v):
            return v
        case MetaInt(value=v) | NpInt(value=v):
            return v != 0
        case StaticFloat(value=v) | NpFloat(value=v):
            return None if math.isnan(v) else v != 0.0
        case StaticSeq(items=items):
            return len(items) != 0
        case _:
            return None
