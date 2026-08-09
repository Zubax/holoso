"""Semantic HIR operators."""

import math

import numpy as np
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

from .._errors import HolosoError
from ._const import BoolConst, Const, FloatConst, IntConst
from ._types import BoolType, FloatType, IntType, Signature


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


def _int_const(const: Const) -> IntConst:
    if not isinstance(const, IntConst):
        raise TypeError(f"expected IntConst, got {const!r}")
    return const


class NoNumber(HolosoError):
    """
    The signal :meth:`Operator.evaluate` raises when the operands prove the expression names no number -- an
    indeterminate form like ``inf - inf``, or an argument outside a function's domain like ``sqrt(-1)``. It is not an
    answer, so it is not a return value.

    It is a signal and not a refusal. Every pass that folds speculatively catches it and leaves the operation exactly
    as it stands, because unrolling and inlining SUBSTITUTE values and so manufacture expressions the kernel never
    wrote -- ``for w in [1.0, 0.0]: if w > 0.0: x / w`` becomes ``x / 0.0`` -- and convicting one of those is the
    compiler answering for its own transformation. What refuses is the gate at the HIR-to-MIR boundary, over what is
    left once every deletion and substitution has run. See the fastmath charter in DESIGN.md.

    ``what`` names the expression for that diagnostic; the signal carries no message of its own because where it is
    raised is not where it is reported.
    """

    def __init__(self, what: str) -> None:
        super().__init__(what)
        self.what = what


def _fold_float(operands: list[Const], name: str, evaluate: Callable[..., float]) -> Const:
    """
    Host-precision float folding -- never the target format. Whatever the operator's own reference declines to answer,
    we decline too: a domain fault, a result past the carrier, a NaN. Guessing which infinity an overflow meant is
    inventing a value the expression never named, and the exception could not tell us anyway. A reference that
    SATURATES instead of raising therefore hands back the infinity as an ordinary value, which is the answer rather
    than a guess at one.
    """
    try:
        # The fold must not depend on the user's process-global numpy error policy; faults are judged by the
        # NaN value below, never by the signal.
        with np.errstate(all="ignore"):
            value = evaluate(*[_float_const(operand).value for operand in operands])
    except (ValueError, OverflowError, ZeroDivisionError):
        value = math.nan
    if math.isnan(value):
        raise NoNumber(f"{name} of {operands}")
    return FloatConst(value)


def _fold_int(operands: list[Const], name: str, evaluate: Callable[..., int]) -> Const:
    """
    Arbitrary-precision integer folding -- no width, no saturation, and no size limit. An expression the user wrote is
    one the user asked for: ``1 << 10**9`` takes as long as it takes, exactly as it would in the Python the kernel is
    written in. A host fault means the operands are outside the operation's domain, so no value exists.
    """
    try:
        return IntConst(evaluate(*[_int_const(operand).value for operand in operands]))
    except (ZeroDivisionError, ValueError, OverflowError):
        raise NoNumber(f"{name} of {operands}") from None


@dataclass(frozen=True, slots=True)
class Operator(ABC):
    mnemonic: ClassVar[str]
    # Whether evaluating this operation on a not-taken path is unobservable: a speculatable operation has no error
    # sideband and no effect beyond its result value, so if-conversion may execute it unconditionally. Division is
    # not speculatable (a speculated div-by-zero would assert the module's error flag for a branch never taken).
    # A partial fold is not an error sideband and does not bar speculation: mul, add, the casts and the shifts all
    # decline to name a number somewhere in their domain, and the mux discards that arm in hardware regardless.
    # The default is False so a future error-bearing operator that omits the declaration is a missed optimization
    # rather than a silent spurious-error bug; pure operators opt in explicitly.
    speculatable: ClassVar[bool] = False

    @property
    @abstractmethod
    def signature(self) -> Signature: ...

    @abstractmethod
    def evaluate(self, operands: list[Const]) -> Const:
        """
        The value of this operation over operands the compiler knows in full, which is a number or no build at all.
        Evaluation runs in the compiler's own arithmetic -- unbounded integers, host-precision floats -- and never in
        the target format, so a folded constant may differ from what the datapath would compute; see the fastmath
        charter in DESIGN.md. Nothing declines a fold: not size, not representability, and never a result the hardware
        would disagree with. Where no value exists at all it raises :class:`NoNumber`.

        A caller asks only where it sees EVERY operand. One it cannot see leaves the expression unnamed, and then the
        algebraic identities below speak for it instead -- which is why ``x*0`` is zero for an unknown ``x`` while
        ``inf*0`` names no number.
        """

    def absorbing(self) -> Const | None:
        """
        The constant operand that forces the result to that constant regardless of the others (the absorbing element):
        ``True`` for ``or``, ``False`` for ``and``. None if the operator has none. Strength reduction uses it to reduce
        a partially-constant expression like ``x or True`` to a constant.
        """
        return None

    def identity(self) -> Const | None:
        """
        The constant operand that leaves the result unchanged (the identity element): ``False`` for ``or``,
        ``True`` for ``and``. None if the operator has none. Strength reduction drops it (``x and True`` -> x).
        """
        return None


@dataclass(frozen=True, slots=True)
class FloatAdd(Operator):
    mnemonic: ClassVar[str] = "fadd"
    speculatable: ClassVar[bool] = True

    @property
    def signature(self) -> Signature:
        return _float_signature(2)

    def evaluate(self, operands: list[Const]) -> Const:
        return _fold_float(operands, "the sum", lambda a, b: a + b)

    def identity(self) -> Const | None:
        # ``x + 0 == x``, declared rather than hand-coded so every rewrite agrees about it
        return FloatConst(0.0)


@dataclass(frozen=True, slots=True)
class FloatMul(Operator):
    mnemonic: ClassVar[str] = "fmul"
    speculatable: ClassVar[bool] = True

    @property
    def signature(self) -> Signature:
        return _float_signature(2)

    def evaluate(self, operands: list[Const]) -> Const:
        return _fold_float(operands, "the product", lambda a, b: a * b)

    # ``x*0 == 0`` is the charter's identity, declared here rather than hand-coded in one pass so that every rewrite
    # reasoning about known values sees it. It cannot fire on two constants: an all-known product is evaluated first,
    # and ``inf*0`` names no number.
    def absorbing(self) -> Const | None:
        return FloatConst(0.0)

    def identity(self) -> Const | None:
        return FloatConst(1.0)


@dataclass(frozen=True, slots=True)
class FloatDiv(Operator):
    mnemonic: ClassVar[str] = "fdiv"

    @property
    def signature(self) -> Signature:
        return _float_signature(2)

    def evaluate(self, operands: list[Const]) -> Const:
        return _fold_float(operands, "the quotient", lambda a, b: a / b)


@dataclass(frozen=True, slots=True)
class FloatNeg(Operator):
    mnemonic: ClassVar[str] = "fneg"
    speculatable: ClassVar[bool] = True

    @property
    def signature(self) -> Signature:
        return _float_signature(1)

    def evaluate(self, operands: list[Const]) -> Const:
        (a,) = [_float_const(operand) for operand in operands]
        return FloatConst(-a.value)


@dataclass(frozen=True, slots=True)
class FloatAbs(Operator):
    mnemonic: ClassVar[str] = "fabs"
    speculatable: ClassVar[bool] = True

    @property
    def signature(self) -> Signature:
        return _float_signature(1)

    def evaluate(self, operands: list[Const]) -> Const:
        (a,) = [_float_const(operand) for operand in operands]
        return FloatConst(abs(a.value))


@dataclass(frozen=True, slots=True)
class FloatMulPow2(Operator):
    """Exact semantic scaling by a power of two, introduced by strength reduction."""

    mnemonic: ClassVar[str] = "fmul_pow2"
    speculatable: ClassVar[bool] = True
    k: int

    @property
    def signature(self) -> Signature:
        return _float_signature(1)

    def evaluate(self, operands: list[Const]) -> Const:
        # ``np.ldexp`` rather than ``math.ldexp`` because this operator stands for a multiplication, which saturates:
        # the raise is the math module's, not the operation's, and answering it would be inventing a value.
        return _fold_float(operands, "the scaling", lambda a: float(np.ldexp(a, self.k)))


@dataclass(frozen=True, slots=True)
class FloatRound(Operator):
    """Round a float to the nearest integral-valued float, ties to even."""

    mnemonic: ClassVar[str] = "fround"
    speculatable: ClassVar[bool] = True

    @property
    def signature(self) -> Signature:
        return _float_signature(1)

    def evaluate(self, operands: list[Const]) -> Const:
        return _fold_float(operands, "the rounding", lambda a: float(np.rint(a)))


@dataclass(frozen=True, slots=True)
class FloatFloor(Operator):
    """Round a float toward negative infinity to an integral-valued float."""

    mnemonic: ClassVar[str] = "ffloor"
    speculatable: ClassVar[bool] = True

    @property
    def signature(self) -> Signature:
        return _float_signature(1)

    def evaluate(self, operands: list[Const]) -> Const:
        return _fold_float(operands, "the floor", lambda a: float(np.floor(a)))


@dataclass(frozen=True, slots=True)
class FloatCeil(Operator):
    """Round a float toward positive infinity to an integral-valued float."""

    mnemonic: ClassVar[str] = "fceil"
    speculatable: ClassVar[bool] = True

    @property
    def signature(self) -> Signature:
        return _float_signature(1)

    def evaluate(self, operands: list[Const]) -> Const:
        return _fold_float(operands, "the ceiling", lambda a: float(np.ceil(a)))


@dataclass(frozen=True, slots=True)
class FloatTrunc(Operator):
    """Round a float toward zero to an integral-valued float."""

    mnemonic: ClassVar[str] = "ftrunc"
    speculatable: ClassVar[bool] = True

    @property
    def signature(self) -> Signature:
        return _float_signature(1)

    def evaluate(self, operands: list[Const]) -> Const:
        return _fold_float(operands, "the truncation", lambda a: float(np.trunc(a)))


@dataclass(frozen=True, slots=True)
class FloatExp2(Operator):
    mnemonic: ClassVar[str] = "fexp2"
    speculatable: ClassVar[bool] = True

    @property
    def signature(self) -> Signature:
        return _float_signature(1)

    def evaluate(self, operands: list[Const]) -> Const:
        # ``np.exp2`` is this operator's reference -- the intrinsic stub is registered against it -- and it saturates.
        # ``math.exp2`` raises instead, which is the math module's convention and not this operation's answer.
        return _fold_float(operands, "the exponential", lambda a: float(np.exp2(a)))


@dataclass(frozen=True, slots=True)
class FloatLog2(Operator):
    mnemonic: ClassVar[str] = "flog2"

    @property
    def signature(self) -> Signature:
        return _float_signature(1)

    def evaluate(self, operands: list[Const]) -> Const:
        # np.log2 answers -inf at the 0.0 pole like the hardware; math.log2 raises there.
        return _fold_float(operands, "the base-2 logarithm", lambda a: float(np.log2(a)))


@dataclass(frozen=True, slots=True)
class FloatSin(Operator):
    mnemonic: ClassVar[str] = "fsin"
    speculatable: ClassVar[bool] = True

    @property
    def signature(self) -> Signature:
        return _float_signature(1)

    def evaluate(self, operands: list[Const]) -> Const:
        return _fold_float(operands, "the sine", math.sin)


@dataclass(frozen=True, slots=True)
class FloatCos(Operator):
    mnemonic: ClassVar[str] = "fcos"
    speculatable: ClassVar[bool] = True

    @property
    def signature(self) -> Signature:
        return _float_signature(1)

    def evaluate(self, operands: list[Const]) -> Const:
        return _fold_float(operands, "the cosine", math.cos)


@dataclass(frozen=True, slots=True)
class FloatSqrt(Operator):
    mnemonic: ClassVar[str] = "fsqrt"

    @property
    def signature(self) -> Signature:
        return _float_signature(1)

    def evaluate(self, operands: list[Const]) -> Const:
        return _fold_float(operands, "the square root", math.sqrt)


@dataclass(frozen=True, slots=True)
class FloatAtan2(Operator):
    mnemonic: ClassVar[str] = "fatan2"
    speculatable: ClassVar[bool] = True

    @property
    def signature(self) -> Signature:
        return _float_signature(2)

    def evaluate(self, operands: list[Const]) -> Const:
        return _fold_float(operands, "the arctangent", math.atan2)


@dataclass(frozen=True, slots=True)
class FloatHypot2(Operator):
    """
    A semantic op only because it may be computed as a byproduct of atan2 -- the MIR op selector will manage this.
    If not, it is expected to be lowered as the ordinary sqrt(x**2+y**2).
    """

    mnemonic: ClassVar[str] = "fhypot2"

    @property
    def signature(self) -> Signature:
        return _float_signature(2)

    def evaluate(self, operands: list[Const]) -> Const:
        return _fold_float(operands, "the magnitude", math.hypot)


@dataclass(frozen=True, slots=True)
class FloatIsFinite(Operator):
    mnemonic: ClassVar[str] = "fisfinite"
    speculatable: ClassVar[bool] = True

    @property
    def signature(self) -> Signature:
        return Signature((FloatType(),), BoolType())

    def evaluate(self, operands: list[Const]) -> Const:
        (a,) = [_float_const(operand) for operand in operands]
        return BoolConst(math.isfinite(a.value))


@dataclass(frozen=True, slots=True)
class FloatIsInf(Operator):
    mnemonic: ClassVar[str] = "fisinf"
    speculatable: ClassVar[bool] = True

    @property
    def signature(self) -> Signature:
        return Signature((FloatType(),), BoolType())

    def evaluate(self, operands: list[Const]) -> Const:
        (a,) = [_float_const(operand) for operand in operands]
        return BoolConst(math.isinf(a.value))


@dataclass(frozen=True, slots=True)
class FloatIsPosInf(Operator):
    mnemonic: ClassVar[str] = "fisposinf"
    speculatable: ClassVar[bool] = True

    @property
    def signature(self) -> Signature:
        return Signature((FloatType(),), BoolType())

    def evaluate(self, operands: list[Const]) -> Const:
        (a,) = [_float_const(operand) for operand in operands]
        return BoolConst(a.value == math.inf)


@dataclass(frozen=True, slots=True)
class FloatIsNegInf(Operator):
    mnemonic: ClassVar[str] = "fisneginf"
    speculatable: ClassVar[bool] = True

    @property
    def signature(self) -> Signature:
        return Signature((FloatType(),), BoolType())

    def evaluate(self, operands: list[Const]) -> Const:
        (a,) = [_float_const(operand) for operand in operands]
        return BoolConst(a.value == -math.inf)


@dataclass(frozen=True, slots=True)
class FloatFma(Operator):
    """Fused multiply-add ``a*b + c`` from an explicit ``math.fma`` call: always single-rounds."""

    mnemonic: ClassVar[str] = "ffma"
    speculatable: ClassVar[bool] = True

    @property
    def signature(self) -> Signature:
        return _float_signature(3)

    def evaluate(self, operands: list[Const]) -> Const:
        return _fold_float(operands, "the fused multiply-add", math.fma)


@dataclass(frozen=True, slots=True)
class FloatMin(Operator):
    mnemonic: ClassVar[str] = "fmin"
    speculatable: ClassVar[bool] = True

    @property
    def signature(self) -> Signature:
        return _float_signature(2)

    def evaluate(self, operands: list[Const]) -> Const:
        a, b = [_float_const(operand) for operand in operands]
        return a if a.value < b.value else b


@dataclass(frozen=True, slots=True)
class FloatMax(Operator):
    mnemonic: ClassVar[str] = "fmax"
    speculatable: ClassVar[bool] = True

    @property
    def signature(self) -> Signature:
        return _float_signature(2)

    def evaluate(self, operands: list[Const]) -> Const:
        a, b = [_float_const(operand) for operand in operands]
        return b if a.value < b.value else a


@dataclass(frozen=True, slots=True)
class FloatComparison(Operator, ABC):
    """One relation over two floats; the six spellings are distinct operators sharing everything but their truth."""

    speculatable: ClassVar[bool] = True

    @property
    def signature(self) -> Signature:
        return Signature((FloatType(), FloatType()), BoolType())


@dataclass(frozen=True, slots=True)
class FloatLess(FloatComparison):
    mnemonic: ClassVar[str] = "flt"

    def evaluate(self, operands: list[Const]) -> Const:
        a, b = [_float_const(operand) for operand in operands]
        return BoolConst(a.value < b.value)


@dataclass(frozen=True, slots=True)
class FloatLessOrEqual(FloatComparison):
    mnemonic: ClassVar[str] = "fle"

    def evaluate(self, operands: list[Const]) -> Const:
        a, b = [_float_const(operand) for operand in operands]
        return BoolConst(a.value <= b.value)


@dataclass(frozen=True, slots=True)
class FloatEqual(FloatComparison):
    mnemonic: ClassVar[str] = "feq"

    def evaluate(self, operands: list[Const]) -> Const:
        a, b = [_float_const(operand) for operand in operands]
        return BoolConst(a.value == b.value)


@dataclass(frozen=True, slots=True)
class FloatNotEqual(FloatComparison):
    mnemonic: ClassVar[str] = "fne"

    def evaluate(self, operands: list[Const]) -> Const:
        a, b = [_float_const(operand) for operand in operands]
        return BoolConst(a.value != b.value)


@dataclass(frozen=True, slots=True)
class FloatGreaterOrEqual(FloatComparison):
    mnemonic: ClassVar[str] = "fge"

    def evaluate(self, operands: list[Const]) -> Const:
        a, b = [_float_const(operand) for operand in operands]
        return BoolConst(a.value >= b.value)


@dataclass(frozen=True, slots=True)
class FloatGreater(FloatComparison):
    mnemonic: ClassVar[str] = "fgt"

    def evaluate(self, operands: list[Const]) -> Const:
        a, b = [_float_const(operand) for operand in operands]
        return BoolConst(a.value > b.value)


@dataclass(frozen=True, slots=True)
class BoolAnd(Operator):
    mnemonic: ClassVar[str] = "band"
    speculatable: ClassVar[bool] = True

    @property
    def signature(self) -> Signature:
        return _bool_signature(2)

    def evaluate(self, operands: list[Const]) -> Const:
        a, b = [_bool_const(operand) for operand in operands]
        return BoolConst(a.value and b.value)

    def absorbing(self) -> Const | None:
        # x and False == False
        return BoolConst(False)

    def identity(self) -> Const | None:
        # x and True == x
        return BoolConst(True)


@dataclass(frozen=True, slots=True)
class BoolOr(Operator):
    mnemonic: ClassVar[str] = "bor"
    speculatable: ClassVar[bool] = True

    @property
    def signature(self) -> Signature:
        return _bool_signature(2)

    def evaluate(self, operands: list[Const]) -> Const:
        a, b = [_bool_const(operand) for operand in operands]
        return BoolConst(a.value or b.value)

    def absorbing(self) -> Const | None:
        # x or True == True
        return BoolConst(True)

    def identity(self) -> Const | None:
        # x or False == x
        return BoolConst(False)


@dataclass(frozen=True, slots=True)
class BoolXor(Operator):
    mnemonic: ClassVar[str] = "bxor"
    speculatable: ClassVar[bool] = True

    @property
    def signature(self) -> Signature:
        return _bool_signature(2)

    def evaluate(self, operands: list[Const]) -> Const:
        a, b = [_bool_const(operand) for operand in operands]
        return BoolConst(a.value != b.value)

    def identity(self) -> Const | None:
        # x ^ False == x (there is no absorbing element: x ^ True == ~x, not a constant)
        return BoolConst(False)


@dataclass(frozen=True, slots=True)
class BoolNot(Operator):
    mnemonic: ClassVar[str] = "bnot"
    speculatable: ClassVar[bool] = True

    @property
    def signature(self) -> Signature:
        return _bool_signature(1)

    def evaluate(self, operands: list[Const]) -> Const:
        (a,) = [_bool_const(operand) for operand in operands]
        return BoolConst(not a.value)


@dataclass(frozen=True, slots=True)
class FloatSelect(Operator):
    """
    A data mux ``a if cond else b`` over float values. In HIR it is produced by the if-conversion pass, which refuses
    constant conditions; MIR composite lowerings may also use the selected inline hardware mux directly.
    """

    mnemonic: ClassVar[str] = "fselect"
    speculatable: ClassVar[bool] = True

    @property
    def signature(self) -> Signature:
        return Signature((BoolType(), FloatType(), FloatType()), FloatType())

    def evaluate(self, operands: list[Const]) -> Const:
        cond, a, b = operands
        return _float_const(a) if _bool_const(cond).value else _float_const(b)


@dataclass(frozen=True, slots=True)
class BoolSelect(Operator):
    """
    A boolean mux ``a if cond else b`` over boolean values, the 1-bit dual of :class:`FloatSelect`. Produced only by
    if-conversion of a boolean-phi diamond. Its constant arms (the common ``True``/``False`` arms of a state-machine
    merge) are reduced to ``and``/``or``/``not``/passthrough by strength reduction.
    """

    mnemonic: ClassVar[str] = "bselect"
    speculatable: ClassVar[bool] = True

    @property
    def signature(self) -> Signature:
        return Signature((BoolType(), BoolType(), BoolType()), BoolType())

    def evaluate(self, operands: list[Const]) -> Const:
        cond, a, b = operands
        return _bool_const(a) if _bool_const(cond).value else _bool_const(b)


@dataclass(frozen=True, slots=True)
class FloatToBool(Operator):
    """A scalar cast ``bool(x)``: a float is truthy iff it is nonzero."""

    mnemonic: ClassVar[str] = "float_to_bool"
    speculatable: ClassVar[bool] = True

    @property
    def signature(self) -> Signature:
        return Signature((FloatType(),), BoolType())

    def evaluate(self, operands: list[Const]) -> Const:
        (a,) = [_float_const(operand) for operand in operands]
        return BoolConst(a.value != 0.0)


@dataclass(frozen=True, slots=True)
class BoolToFloat(Operator):
    mnemonic: ClassVar[str] = "bool_to_float"
    speculatable: ClassVar[bool] = True

    @property
    def signature(self) -> Signature:
        return Signature((BoolType(),), FloatType())

    def evaluate(self, operands: list[Const]) -> Const:
        (a,) = [_bool_const(operand) for operand in operands]
        return FloatConst(1.0 if a.value else 0.0)


# Signed integers before hardware width selection. Folding is exact at arbitrary precision -- no width, no saturation
# -- so a fully static integer expression disappears before MIR, whose integer refusal therefore only ever sees a
# runtime one. Add/sub/mul/neg/abs and the bitwise and shift operators are speculatable; floor-division and modulo
# assert the div-by-zero error flag, so they are not.


def _int_signature(arity: int) -> Signature:
    ty = IntType()
    return Signature((ty,) * arity, ty)


@dataclass(frozen=True, slots=True)
class IntAdd(Operator):
    mnemonic: ClassVar[str] = "iadd"
    speculatable: ClassVar[bool] = True

    @property
    def signature(self) -> Signature:
        return _int_signature(2)

    def evaluate(self, operands: list[Const]) -> Const:
        return _fold_int(operands, "the sum", lambda a, b: a + b)

    def identity(self) -> Const | None:
        return IntConst(0)


@dataclass(frozen=True, slots=True)
class IntSub(Operator):
    mnemonic: ClassVar[str] = "isub"
    speculatable: ClassVar[bool] = True

    @property
    def signature(self) -> Signature:
        return _int_signature(2)

    def evaluate(self, operands: list[Const]) -> Const:
        return _fold_int(operands, "the difference", lambda a, b: a - b)


@dataclass(frozen=True, slots=True)
class IntMul(Operator):
    mnemonic: ClassVar[str] = "imul"
    speculatable: ClassVar[bool] = True

    @property
    def signature(self) -> Signature:
        return _int_signature(2)

    def evaluate(self, operands: list[Const]) -> Const:
        return _fold_int(operands, "the product", lambda a, b: a * b)

    def absorbing(self) -> Const | None:
        return IntConst(0)

    def identity(self) -> Const | None:
        return IntConst(1)


@dataclass(frozen=True, slots=True)
class IntMulPow2(Operator):
    """
    The integer dual of :class:`FloatMulPow2`: exact scaling by a power of two. It is a MULTIPLICATION and not the
    ``<<`` that shares its arithmetic -- what leaves the word rails here where the shift drops it -- which is why the
    two cannot be one operator however alike their folding looks.
    """

    mnemonic: ClassVar[str] = "imul_pow2"
    speculatable: ClassVar[bool] = True
    k: int

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError(f"scaling by 2**{self.k} is no multiplication for a shift to serve")

    @property
    def signature(self) -> Signature:
        return _int_signature(1)

    def evaluate(self, operands: list[Const]) -> Const:
        return _fold_int(operands, "the scaling", lambda a: a << self.k)


@dataclass(frozen=True, slots=True)
class IntNeg(Operator):
    mnemonic: ClassVar[str] = "ineg"
    speculatable: ClassVar[bool] = True

    @property
    def signature(self) -> Signature:
        return _int_signature(1)

    def evaluate(self, operands: list[Const]) -> Const:
        return _fold_int(operands, "the negation", lambda a: -a)


@dataclass(frozen=True, slots=True)
class IntAbs(Operator):
    mnemonic: ClassVar[str] = "iabs"
    speculatable: ClassVar[bool] = True

    @property
    def signature(self) -> Signature:
        return _int_signature(1)

    def evaluate(self, operands: list[Const]) -> Const:
        return _fold_int(operands, "the magnitude", abs)


@dataclass(frozen=True, slots=True)
class IntDivFloor(Operator):
    mnemonic: ClassVar[str] = "idivfloor"

    @property
    def signature(self) -> Signature:
        return _int_signature(2)

    def evaluate(self, operands: list[Const]) -> Const:
        return _fold_int(operands, "the quotient", lambda a, b: a // b)


@dataclass(frozen=True, slots=True)
class IntMod(Operator):
    mnemonic: ClassVar[str] = "imod"

    @property
    def signature(self) -> Signature:
        return _int_signature(2)

    def evaluate(self, operands: list[Const]) -> Const:
        return _fold_int(operands, "the remainder", lambda a, b: a % b)


@dataclass(frozen=True, slots=True)
class IntShiftLeft(Operator):
    mnemonic: ClassVar[str] = "ishl"
    speculatable: ClassVar[bool] = True

    @property
    def signature(self) -> Signature:
        return _int_signature(2)

    def evaluate(self, operands: list[Const]) -> Const:
        return _fold_int(operands, "the left shift", lambda a, b: a << b)


@dataclass(frozen=True, slots=True)
class IntShiftRight(Operator):
    mnemonic: ClassVar[str] = "ishr"
    speculatable: ClassVar[bool] = True

    @property
    def signature(self) -> Signature:
        return _int_signature(2)

    def evaluate(self, operands: list[Const]) -> Const:
        return _fold_int(operands, "the right shift", lambda a, b: a >> b)


@dataclass(frozen=True, slots=True)
class IntBwAnd(Operator):
    mnemonic: ClassVar[str] = "ibwand"
    speculatable: ClassVar[bool] = True

    @property
    def signature(self) -> Signature:
        return _int_signature(2)

    def evaluate(self, operands: list[Const]) -> Const:
        return _fold_int(operands, "the conjunction", lambda a, b: a & b)

    def absorbing(self) -> Const | None:
        return IntConst(0)

    def identity(self) -> Const | None:
        # all ones at every width
        return IntConst(-1)


@dataclass(frozen=True, slots=True)
class IntBwOr(Operator):
    mnemonic: ClassVar[str] = "ibwor"
    speculatable: ClassVar[bool] = True

    @property
    def signature(self) -> Signature:
        return _int_signature(2)

    def evaluate(self, operands: list[Const]) -> Const:
        return _fold_int(operands, "the disjunction", lambda a, b: a | b)

    def absorbing(self) -> Const | None:
        # all ones at every width
        return IntConst(-1)

    def identity(self) -> Const | None:
        return IntConst(0)


@dataclass(frozen=True, slots=True)
class IntBwXor(Operator):
    mnemonic: ClassVar[str] = "ibwxor"
    speculatable: ClassVar[bool] = True

    @property
    def signature(self) -> Signature:
        return _int_signature(2)

    def evaluate(self, operands: list[Const]) -> Const:
        return _fold_int(operands, "the exclusive disjunction", lambda a, b: a ^ b)

    def identity(self) -> Const | None:
        return IntConst(0)


@dataclass(frozen=True, slots=True)
class IntBwNot(Operator):
    mnemonic: ClassVar[str] = "ibwnot"
    speculatable: ClassVar[bool] = True

    @property
    def signature(self) -> Signature:
        return _int_signature(1)

    def evaluate(self, operands: list[Const]) -> Const:
        return _fold_int(operands, "the complement", lambda a: ~a)


@dataclass(frozen=True, slots=True)
class IntComparison(Operator, ABC):
    """The integer dual of :class:`FloatComparison`: one relation per operator, exact at any magnitude."""

    speculatable: ClassVar[bool] = True

    @property
    def signature(self) -> Signature:
        return Signature((IntType(), IntType()), BoolType())


@dataclass(frozen=True, slots=True)
class IntLess(IntComparison):
    mnemonic: ClassVar[str] = "ilt"

    def evaluate(self, operands: list[Const]) -> Const:
        a, b = [_int_const(operand) for operand in operands]
        return BoolConst(a.value < b.value)


@dataclass(frozen=True, slots=True)
class IntLessOrEqual(IntComparison):
    mnemonic: ClassVar[str] = "ile"

    def evaluate(self, operands: list[Const]) -> Const:
        a, b = [_int_const(operand) for operand in operands]
        return BoolConst(a.value <= b.value)


@dataclass(frozen=True, slots=True)
class IntEqual(IntComparison):
    mnemonic: ClassVar[str] = "ieq"

    def evaluate(self, operands: list[Const]) -> Const:
        a, b = [_int_const(operand) for operand in operands]
        return BoolConst(a.value == b.value)


@dataclass(frozen=True, slots=True)
class IntNotEqual(IntComparison):
    mnemonic: ClassVar[str] = "ine"

    def evaluate(self, operands: list[Const]) -> Const:
        a, b = [_int_const(operand) for operand in operands]
        return BoolConst(a.value != b.value)


@dataclass(frozen=True, slots=True)
class IntGreaterOrEqual(IntComparison):
    mnemonic: ClassVar[str] = "ige"

    def evaluate(self, operands: list[Const]) -> Const:
        a, b = [_int_const(operand) for operand in operands]
        return BoolConst(a.value >= b.value)


@dataclass(frozen=True, slots=True)
class IntGreater(IntComparison):
    mnemonic: ClassVar[str] = "igt"

    def evaluate(self, operands: list[Const]) -> Const:
        a, b = [_int_const(operand) for operand in operands]
        return BoolConst(a.value > b.value)


@dataclass(frozen=True, slots=True)
class IntSelect(Operator):
    """A data mux ``a if cond else b`` over integer values, the integer dual of :class:`FloatSelect`."""

    mnemonic: ClassVar[str] = "iselect"
    speculatable: ClassVar[bool] = True

    @property
    def signature(self) -> Signature:
        return Signature((BoolType(), IntType(), IntType()), IntType())

    def evaluate(self, operands: list[Const]) -> Const:
        cond, a, b = operands
        return _int_const(a) if _bool_const(cond).value else _int_const(b)


@dataclass(frozen=True, slots=True)
class IntToFloat(Operator):
    mnemonic: ClassVar[str] = "int_to_float"
    speculatable: ClassVar[bool] = True

    @property
    def signature(self) -> Signature:
        return Signature((IntType(),), FloatType())

    def evaluate(self, operands: list[Const]) -> Const:
        (a,) = [_int_const(operand) for operand in operands]
        try:
            return FloatConst(float(a.value))
        except OverflowError:
            raise NoNumber(f"the float value of {a.value}") from None


@dataclass(frozen=True, slots=True)
class FloatToInt(Operator):
    """A truncation-toward-zero cast ``int(x)``; no error sideband, so speculatable."""

    mnemonic: ClassVar[str] = "float_to_int"
    speculatable: ClassVar[bool] = True

    @property
    def signature(self) -> Signature:
        return Signature((FloatType(),), IntType())

    def evaluate(self, operands: list[Const]) -> Const:
        (a,) = [_float_const(operand) for operand in operands]
        try:
            return IntConst(int(a.value))
        except OverflowError:
            raise NoNumber(f"the integer part of {operands}") from None


@dataclass(frozen=True, slots=True)
class IntToBool(Operator):
    mnemonic: ClassVar[str] = "int_to_bool"
    speculatable: ClassVar[bool] = True

    @property
    def signature(self) -> Signature:
        return Signature((IntType(),), BoolType())

    def evaluate(self, operands: list[Const]) -> Const:
        (a,) = [_int_const(operand) for operand in operands]
        return BoolConst(a.value != 0)


@dataclass(frozen=True, slots=True)
class BoolToInt(Operator):
    mnemonic: ClassVar[str] = "bool_to_int"
    speculatable: ClassVar[bool] = True

    @property
    def signature(self) -> Signature:
        return Signature((BoolType(),), IntType())

    def evaluate(self, operands: list[Const]) -> Const:
        (a,) = [_bool_const(operand) for operand in operands]
        return IntConst(1 if a.value else 0)
