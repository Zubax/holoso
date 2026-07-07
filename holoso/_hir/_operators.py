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

    def fold_constants(self, operands: list[Const]) -> Const | None:
        """
        The folded constant node, or None to leave the operation unfolded (the default; like ``absorbing``/``identity``,
        a foldable operator opts in by overriding). The HIR folder is format-agnostic (float64), so folding is faithful
        only where that float64 result is an accepted fast-math approximation of the hardware; a format-critical
        operator stays unfolded and is evaluated by the hardware operator, where the format is known.
        """
        return None

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
class FloatRound(Operator):
    """Round a float to the nearest integral-valued float, ties to even."""

    mnemonic: ClassVar[str] = "round"
    speculatable: ClassVar[bool] = True

    @property
    def signature(self) -> Signature:
        return _float_signature(1)


@dataclass(frozen=True, slots=True)
class FloatFloor(Operator):
    """Round a float toward negative infinity to an integral-valued float."""

    mnemonic: ClassVar[str] = "floor"
    speculatable: ClassVar[bool] = True

    @property
    def signature(self) -> Signature:
        return _float_signature(1)


@dataclass(frozen=True, slots=True)
class FloatCeil(Operator):
    """Round a float toward positive infinity to an integral-valued float."""

    mnemonic: ClassVar[str] = "ceil"
    speculatable: ClassVar[bool] = True

    @property
    def signature(self) -> Signature:
        return _float_signature(1)


@dataclass(frozen=True, slots=True)
class FloatTrunc(Operator):
    """Round a float toward zero to an integral-valued float."""

    mnemonic: ClassVar[str] = "trunc"
    speculatable: ClassVar[bool] = True

    @property
    def signature(self) -> Signature:
        return _float_signature(1)


@dataclass(frozen=True, slots=True)
class FloatExp2(Operator):
    mnemonic: ClassVar[str] = "exp2"
    speculatable: ClassVar[bool] = True

    @property
    def signature(self) -> Signature:
        return _float_signature(1)

    def fold_constants(self, operands: list[Const]) -> Const | None:
        (a,) = [_float_const(operand) for operand in operands]
        if not math.isfinite(a.value):
            return None
        try:
            result = math.exp2(a.value)
        except OverflowError:
            return None
        return FloatConst(result) if math.isfinite(result) else None


@dataclass(frozen=True, slots=True)
class FloatLog2(Operator):
    mnemonic: ClassVar[str] = "log2"

    @property
    def signature(self) -> Signature:
        return _float_signature(1)

    def fold_constants(self, operands: list[Const]) -> Const | None:
        (a,) = [_float_const(operand) for operand in operands]
        if not (math.isfinite(a.value) and a.value > 0.0):
            return None
        result = math.log2(a.value)
        return FloatConst(result) if math.isfinite(result) else None


@dataclass(frozen=True, slots=True)
class FloatSin(Operator):
    mnemonic: ClassVar[str] = "sin"
    speculatable: ClassVar[bool] = True

    @property
    def signature(self) -> Signature:
        return _float_signature(1)

    def fold_constants(self, operands: list[Const]) -> Const | None:
        (a,) = [_float_const(operand) for operand in operands]
        return FloatConst(math.sin(a.value)) if math.isfinite(a.value) else None


@dataclass(frozen=True, slots=True)
class FloatCos(Operator):
    mnemonic: ClassVar[str] = "cos"
    speculatable: ClassVar[bool] = True

    @property
    def signature(self) -> Signature:
        return _float_signature(1)

    def fold_constants(self, operands: list[Const]) -> Const | None:
        (a,) = [_float_const(operand) for operand in operands]
        return FloatConst(math.cos(a.value)) if math.isfinite(a.value) else None


@dataclass(frozen=True, slots=True)
class FloatSqrt(Operator):
    mnemonic: ClassVar[str] = "sqrt"

    @property
    def signature(self) -> Signature:
        return _float_signature(1)

    def fold_constants(self, operands: list[Const]) -> Const | None:
        (a,) = [_float_const(operand) for operand in operands]
        return FloatConst(math.sqrt(a.value)) if math.isfinite(a.value) and a.value >= 0.0 else None


@dataclass(frozen=True, slots=True)
class FloatAtan2(Operator):
    mnemonic: ClassVar[str] = "atan2"
    speculatable: ClassVar[bool] = True

    @property
    def signature(self) -> Signature:
        return _float_signature(2)

    def fold_constants(self, operands: list[Const]) -> Const | None:
        y, x = [_float_const(operand) for operand in operands]
        if not (math.isfinite(y.value) and math.isfinite(x.value)):
            return None
        return FloatConst(math.atan2(y.value, x.value))


@dataclass(frozen=True, slots=True)
class FloatHypot2(Operator):
    """
    A semantic op only because it may be computed as a byproduct of atan2 -- the MIR op selector will manage this.
    If not, it is expected to be lowered as the ordinary sqrt(x**2+y**2).
    """

    mnemonic: ClassVar[str] = "hypot2"

    @property
    def signature(self) -> Signature:
        return _float_signature(2)

    def fold_constants(self, operands: list[Const]) -> Const | None:
        a, b = [_float_const(operand) for operand in operands]
        if not (math.isfinite(a.value) and math.isfinite(b.value)):
            return None
        result = math.hypot(a.value, b.value)
        return FloatConst(result) if math.isfinite(result) else None


@dataclass(frozen=True, slots=True)
class FloatIsFinite(Operator):
    mnemonic: ClassVar[str] = "isfinite"
    speculatable: ClassVar[bool] = True

    @property
    def signature(self) -> Signature:
        return Signature((FloatType(),), BoolType())


@dataclass(frozen=True, slots=True)
class FloatIsInf(Operator):
    mnemonic: ClassVar[str] = "isinf"
    speculatable: ClassVar[bool] = True

    @property
    def signature(self) -> Signature:
        return Signature((FloatType(),), BoolType())


@dataclass(frozen=True, slots=True)
class FloatIsPosInf(Operator):
    mnemonic: ClassVar[str] = "isposinf"
    speculatable: ClassVar[bool] = True

    @property
    def signature(self) -> Signature:
        return Signature((FloatType(),), BoolType())


@dataclass(frozen=True, slots=True)
class FloatIsNegInf(Operator):
    mnemonic: ClassVar[str] = "isneginf"
    speculatable: ClassVar[bool] = True

    @property
    def signature(self) -> Signature:
        return Signature((FloatType(),), BoolType())


@dataclass(frozen=True, slots=True)
class FloatFma(Operator):
    """
    Fused multiply-add ``a*b + c`` from an explicit ``math.fma`` call: always single-rounds.
    Never constant-folded -- a fold in the format-agnostic HIR could only use float64, which double-rounds relative
    to the hardware's single round -- exactly the double-rounding FMA exists to avoid.
    """

    mnemonic: ClassVar[str] = "fma"
    speculatable: ClassVar[bool] = True

    @property
    def signature(self) -> Signature:
        return _float_signature(3)


@dataclass(frozen=True, slots=True)
class FloatMin(Operator):
    """Binary minimum of two floats."""

    mnemonic: ClassVar[str] = "min"
    speculatable: ClassVar[bool] = True

    @property
    def signature(self) -> Signature:
        return _float_signature(2)

    def fold_constants(self, operands: list[Const]) -> Const | None:
        a, b = [_float_const(operand) for operand in operands]
        if not (math.isfinite(a.value) and math.isfinite(b.value)):
            return None  # do not let selection silently discard a non-finite operand
        return a if a.value < b.value else b


@dataclass(frozen=True, slots=True)
class FloatMax(Operator):
    """Binary maximum of two floats."""

    mnemonic: ClassVar[str] = "max"
    speculatable: ClassVar[bool] = True

    @property
    def signature(self) -> Signature:
        return _float_signature(2)

    def fold_constants(self, operands: list[Const]) -> Const | None:
        a, b = [_float_const(operand) for operand in operands]
        if not (math.isfinite(a.value) and math.isfinite(b.value)):
            return None  # do not let selection silently discard a non-finite operand
        return b if a.value < b.value else a


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
    A data mux ``a if cond else b`` over float values. In HIR it is produced by the if-conversion pass, which refuses
    constant conditions; MIR composite lowerings may also use the selected inline hardware mux directly.
    """

    mnemonic: ClassVar[str] = "select"
    speculatable: ClassVar[bool] = True

    @property
    def signature(self) -> Signature:
        return Signature((BoolType(), FloatType(), FloatType()), FloatType())


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


@dataclass(frozen=True, slots=True)
class FloatToBool(Operator):
    """
    A scalar cast ``bool(x)``: a float is truthy iff it is nonzero (the ZKF exponent-nonzero test). Never
    constant-folded: the test is on the constant *encoded into the configured format*, so a magnitude too small to
    represent (encoding to zero) is False -- which a format-agnostic float64 ``c != 0.0`` would get wrong.
    """

    mnemonic: ClassVar[str] = "float_to_bool"
    speculatable: ClassVar[bool] = True

    @property
    def signature(self) -> Signature:
        return Signature((FloatType(),), BoolType())


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
