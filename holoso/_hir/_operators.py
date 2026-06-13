"""Semantic HIR operators."""

import enum
import math
import operator
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

from ._const import BoolConst, Const, FloatConst
from ._types import BoolType, FloatType, Signature


def _float_signature(arity: int) -> Signature:
    ty = FloatType()
    return Signature((ty,) * arity, ty)


def _bool_signature(arity: int) -> Signature:
    ty = BoolType()
    return Signature((ty,) * arity, ty)


def _float_const(const: Const) -> FloatConst:
    if not isinstance(const, FloatConst):
        raise TypeError(f"expected FloatConst, got {const!r}")
    return const


def _bool_const(const: Const) -> BoolConst:
    if not isinstance(const, BoolConst):
        raise TypeError(f"expected BoolConst, got {const!r}")
    return const


@dataclass(frozen=True, slots=True)
class Operator(ABC):
    """A reusable semantic operation definition referenced by HIR operation nodes."""

    mnemonic: ClassVar[str]
    # Whether evaluating this operation on a not-taken path is unobservable: a speculatable operation has no error
    # sideband and no effect beyond its result value, so if-conversion may execute it unconditionally. Division is
    # not speculatable (a speculated div-by-zero would assert the module's error flag for a branch never taken).
    # The default is False so a future error-bearing operator that omits the declaration is a missed optimization
    # rather than a silent spurious-error bug; pure operators opt in explicitly.
    speculatable: ClassVar[bool] = False

    @property
    @abstractmethod
    def signature(self) -> Signature:
        """Semantic operand/result types."""

    @property
    def arity(self) -> int:
        return self.signature.arity

    @abstractmethod
    def fold_constants(self, operands: list[Const]) -> Const | None:
        """Return the folded constant node, or ``None`` if this operation should not be constant-folded."""

    def absorbing(self) -> Const | None:
        """
        The constant operand that forces the result to that constant regardless of the others (the absorbing element):
        ``True`` for ``or``, ``False`` for ``and``. None if the operator has no absorbing element. The constant folder
        uses it to fold a partially-constant expression like ``x or True`` to a constant.
        """
        return None

    def identity(self) -> Const | None:
        """
        The constant operand that leaves the result equal to the other operand (the identity element): ``False`` for
        ``or``, ``True`` for ``and``. None if the operator has none. The constant folder drops it (``x and True`` -> x).
        """
        return None


@dataclass(frozen=True, slots=True)
class FloatAdd(Operator):
    mnemonic: ClassVar[str] = "add"
    speculatable: ClassVar[bool] = True

    @property
    def signature(self) -> Signature:
        return _float_signature(2)

    def fold_constants(self, operands: list[Const]) -> Const:
        a, b = [_float_const(operand) for operand in operands]
        return FloatConst(a.value + b.value)


@dataclass(frozen=True, slots=True)
class FloatMul(Operator):
    mnemonic: ClassVar[str] = "mul"
    speculatable: ClassVar[bool] = True

    @property
    def signature(self) -> Signature:
        return _float_signature(2)

    def fold_constants(self, operands: list[Const]) -> Const:
        a, b = [_float_const(operand) for operand in operands]
        return FloatConst(a.value * b.value)


@dataclass(frozen=True, slots=True)
class FloatDiv(Operator):
    mnemonic: ClassVar[str] = "div"

    @property
    def signature(self) -> Signature:
        return _float_signature(2)

    def fold_constants(self, operands: list[Const]) -> Const | None:
        a, b = [_float_const(operand) for operand in operands]
        return FloatConst(a.value / b.value) if b.value != 0 else None


@dataclass(frozen=True, slots=True)
class FloatNeg(Operator):
    mnemonic: ClassVar[str] = "neg"
    speculatable: ClassVar[bool] = True

    @property
    def signature(self) -> Signature:
        return _float_signature(1)

    def fold_constants(self, operands: list[Const]) -> Const:
        (a,) = [_float_const(operand) for operand in operands]
        return FloatConst(-a.value)


@dataclass(frozen=True, slots=True)
class FloatAbs(Operator):
    mnemonic: ClassVar[str] = "abs"
    speculatable: ClassVar[bool] = True

    @property
    def signature(self) -> Signature:
        return _float_signature(1)

    def fold_constants(self, operands: list[Const]) -> Const:
        (a,) = [_float_const(operand) for operand in operands]
        return FloatConst(abs(a.value))


@dataclass(frozen=True, slots=True)
class FloatMulPow2(Operator):
    """Exact semantic scaling by a power of two, introduced by strength reduction."""

    mnemonic: ClassVar[str] = "mul_pow2"
    speculatable: ClassVar[bool] = True
    k: int

    @property
    def signature(self) -> Signature:
        return _float_signature(1)

    def fold_constants(self, operands: list[Const]) -> Const:
        (a,) = [_float_const(operand) for operand in operands]
        return FloatConst(math.ldexp(a.value, self.k))


class RelationalOp(enum.Enum):
    """A two-operand ordering/equality test on floats, producing a boolean."""

    LT = "lt"
    LE = "le"
    GT = "gt"
    GE = "ge"
    EQ = "eq"
    NE = "ne"

    def holds(self, ordering: int) -> bool:
        """Apply this relation to a three-way comparison result (-1/0/+1), the bit-exact model's comparison path."""
        return _RELATIONAL_FN[self](ordering, 0)


_RELATIONAL_FN: dict[RelationalOp, Callable[[float, float], bool]] = {
    RelationalOp.LT: operator.lt,
    RelationalOp.LE: operator.le,
    RelationalOp.GT: operator.gt,
    RelationalOp.GE: operator.ge,
    RelationalOp.EQ: operator.eq,
    RelationalOp.NE: operator.ne,
}


@dataclass(frozen=True, slots=True)
class FloatRelational(Operator):
    """A float comparison ``a <op> b`` returning a boolean."""

    mnemonic: ClassVar[str] = "frelational"
    speculatable: ClassVar[bool] = True
    op: RelationalOp

    @property
    def signature(self) -> Signature:
        return Signature((FloatType(), FloatType()), BoolType())

    def fold_constants(self, operands: list[Const]) -> Const:
        a, b = [_float_const(operand) for operand in operands]
        return BoolConst(_RELATIONAL_FN[self.op](a.value, b.value))


@dataclass(frozen=True, slots=True)
class BoolAnd(Operator):
    """A boolean conjunction ``a and b`` (both operands genuine booleans)."""

    mnemonic: ClassVar[str] = "band"
    speculatable: ClassVar[bool] = True

    @property
    def signature(self) -> Signature:
        return _bool_signature(2)

    def fold_constants(self, operands: list[Const]) -> Const:
        a, b = [_bool_const(operand) for operand in operands]
        return BoolConst(a.value and b.value)

    def absorbing(self) -> Const:
        return BoolConst(False)  # x and False == False

    def identity(self) -> Const:
        return BoolConst(True)  # x and True == x


@dataclass(frozen=True, slots=True)
class BoolOr(Operator):
    """A boolean disjunction ``a or b`` (both operands genuine booleans)."""

    mnemonic: ClassVar[str] = "bor"
    speculatable: ClassVar[bool] = True

    @property
    def signature(self) -> Signature:
        return _bool_signature(2)

    def fold_constants(self, operands: list[Const]) -> Const:
        a, b = [_bool_const(operand) for operand in operands]
        return BoolConst(a.value or b.value)

    def absorbing(self) -> Const:
        return BoolConst(True)  # x or True == True

    def identity(self) -> Const:
        return BoolConst(False)  # x or False == x


@dataclass(frozen=True, slots=True)
class BoolNot(Operator):
    """A boolean negation ``not a``."""

    mnemonic: ClassVar[str] = "bnot"
    speculatable: ClassVar[bool] = True

    @property
    def signature(self) -> Signature:
        return _bool_signature(1)

    def fold_constants(self, operands: list[Const]) -> Const:
        (a,) = [_bool_const(operand) for operand in operands]
        return BoolConst(not a.value)


@dataclass(frozen=True, slots=True)
class Select(Operator):
    """
    A data mux ``a if cond else b`` over float values, produced exclusively by the if-conversion pass, which
    refuses constant conditions -- so a constant-condition select never exists, and since selects are created after
    the constant folder runs, ``fold_constants`` never sees one.
    """

    mnemonic: ClassVar[str] = "select"
    speculatable: ClassVar[bool] = True

    @property
    def signature(self) -> Signature:
        return Signature((BoolType(), FloatType(), FloatType()), FloatType())

    def fold_constants(self, operands: list[Const]) -> None:
        return None


@dataclass(frozen=True, slots=True)
class FloatToBool(Operator):
    """A scalar cast ``bool(x)``: a float is truthy iff it is nonzero (the ZKF exponent-nonzero test)."""

    mnemonic: ClassVar[str] = "float_to_bool"
    speculatable: ClassVar[bool] = True

    @property
    def signature(self) -> Signature:
        return Signature((FloatType(),), BoolType())

    def fold_constants(self, operands: list[Const]) -> None:
        # Deliberately NOT folded at the (format-agnostic) HIR level: ``bool(c)`` is the ZKF exponent-nonzero test on
        # the constant *encoded into the configured float format*, so a magnitude too small to represent (which encodes
        # to zero) is False. A raw float64 ``c != 0.0`` here would disagree; the cast is evaluated by the hardware
        # operator instead, where the format is known.
        return None


@dataclass(frozen=True, slots=True)
class BoolToFloat(Operator):
    """A scalar cast ``float(cond)``: ``1.0`` when the boolean is true, ``0.0`` when false."""

    mnemonic: ClassVar[str] = "bool_to_float"
    speculatable: ClassVar[bool] = True

    @property
    def signature(self) -> Signature:
        return Signature((BoolType(),), FloatType())

    def fold_constants(self, operands: list[Const]) -> Const:
        (a,) = [_bool_const(operand) for operand in operands]
        return FloatConst(1.0 if a.value else 0.0)
