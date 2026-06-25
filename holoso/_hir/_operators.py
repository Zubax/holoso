"""Semantic HIR operators."""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from .._util import RelationalOp
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
    mnemonic: ClassVar[str]
    # Whether evaluating this operation on a not-taken path is unobservable: a speculatable operation has no error
    # sideband and no effect beyond its result value, so if-conversion may execute it unconditionally. Division is
    # not speculatable (a speculated div-by-zero would assert the module's error flag for a branch never taken).
    # The default is False so a future error-bearing operator that omits the declaration is a missed optimization
    # rather than a silent spurious-error bug; pure operators opt in explicitly.
    speculatable: ClassVar[bool] = False

    @property
    @abstractmethod
    def signature(self) -> Signature: ...

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


@dataclass(frozen=True, slots=True)
class FloatRelational(Operator):
    mnemonic: ClassVar[str] = "frelational"
    speculatable: ClassVar[bool] = True
    op: RelationalOp

    @property
    def signature(self) -> Signature:
        return Signature((FloatType(), FloatType()), BoolType())

    def fold_constants(self, operands: list[Const]) -> Const:
        a, b = [_float_const(operand) for operand in operands]
        return BoolConst(self.op.apply(a.value, b.value))


@dataclass(frozen=True, slots=True)
class BoolAnd(Operator):
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
class BoolXor(Operator):
    mnemonic: ClassVar[str] = "bxor"
    speculatable: ClassVar[bool] = True

    @property
    def signature(self) -> Signature:
        return _bool_signature(2)

    def fold_constants(self, operands: list[Const]) -> Const:
        a, b = [_bool_const(operand) for operand in operands]
        return BoolConst(a.value != b.value)

    def identity(self) -> Const:
        return BoolConst(False)  # x ^ False == x (there is no absorbing element: x ^ True == ~x, not a constant)


@dataclass(frozen=True, slots=True)
class BoolNot(Operator):
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
    A data mux ``a if cond else b`` over float values, produced exclusively by the if-conversion pass, which refuses
    constant conditions -- so a constant-condition select never exists and ``fold_constants`` never fires (the folder
    re-runs after if-conversion, so it does see selects, but never one whose condition is constant).
    """

    mnemonic: ClassVar[str] = "select"
    speculatable: ClassVar[bool] = True

    @property
    def signature(self) -> Signature:
        return Signature((BoolType(), FloatType(), FloatType()), FloatType())

    def fold_constants(self, operands: list[Const]) -> None:
        return None


@dataclass(frozen=True, slots=True)
class BoolSelect(Operator):
    """
    A boolean mux ``a if cond else b`` over boolean values, the 1-bit dual of :class:`Select`. Produced exclusively by
    if-conversion of a boolean-phi diamond; like ``Select`` it refuses constant conditions, so a constant-condition
    bool select never exists and ``fold_constants`` never fires. Its constant arms (the common ``True``/``False`` arms
    of a state-machine merge) are reduced to ``and``/``or``/``not``/passthrough by strength reduction.
    """

    mnemonic: ClassVar[str] = "bool_select"
    speculatable: ClassVar[bool] = True

    @property
    def signature(self) -> Signature:
        return Signature((BoolType(), BoolType(), BoolType()), BoolType())

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
    mnemonic: ClassVar[str] = "bool_to_float"
    speculatable: ClassVar[bool] = True

    @property
    def signature(self) -> Signature:
        return Signature((BoolType(),), FloatType())

    def fold_constants(self, operands: list[Const]) -> Const:
        (a,) = [_bool_const(operand) for operand in operands]
        return FloatConst(1.0 if a.value else 0.0)
