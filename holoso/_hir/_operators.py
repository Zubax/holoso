"""Semantic HIR operators."""

import math

import numpy as np
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

from .._errors import HolosoError
from .._util import Relation
from ._const import BoolConst, Const, FloatConst, IntConst, const_value
from ._types import BoolType, FloatType, IntType, Signature


def _float_const(const: Const) -> FloatConst:
    assert isinstance(const, FloatConst), const
    return const


def _bool_const(const: Const) -> BoolConst:
    assert isinstance(const, BoolConst), const
    return const


def _int_const(const: Const) -> IntConst:
    assert isinstance(const, IntConst), const
    return const


class NoNumber(HolosoError):
    """
    The signal Operator.evaluate raises when the operands prove the expression names no number -- an
    indeterminate form like `inf - inf`, or an argument outside a function's domain like `sqrt(-1)`. It is not an
    answer, so it is not a return value.

    It is a signal and not a refusal. Every pass that folds speculatively catches it and leaves the operation exactly
    as it stands, because unrolling and inlining SUBSTITUTE values and so manufacture expressions the kernel never
    wrote -- `for w in [1.0, 0.0]: if w > 0.0: x / w` becomes `x / 0.0` -- and convicting one of those is the
    compiler answering for its own transformation. What refuses is the gate at the HIR-to-MIR boundary, over what is
    left once every deletion and substitution has run. See the fastmath charter in DESIGN.md.

    `what` names the expression for that diagnostic; the signal carries no message of its own because where it is
    raised is not where it is reported.
    """

    def __init__(self, what: str) -> None:
        super().__init__(what)
        self.what = what


def _spelled(operands: list[Const]) -> str:
    return ", ".join(repr(const_value(operand)) for operand in operands)


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
        raise NoNumber(f"{name} of {_spelled(operands)}")
    return FloatConst(value)


def _fold_int(operands: list[Const], name: str, evaluate: Callable[..., int]) -> Const:
    """
    Arbitrary-precision integer folding -- no width, no saturation, and no size limit. An expression the user wrote is
    one the user asked for: `1 << 10**9` takes as long as it takes, exactly as it would in the Python the kernel is
    written in. A host fault means the operands are outside the operation's domain, so no value exists.
    """
    try:
        return IntConst(evaluate(*[_int_const(operand).value for operand in operands]))
    except (ZeroDivisionError, ValueError, OverflowError):
        raise NoNumber(f"{name} of {_spelled(operands)}") from None


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
    # The algebra strength reduction states once for every operator: the constant operand that forces the result to
    # itself whatever the other is (`False` for `and`), the constant RIGHT operand that leaves the left one unchanged
    # (`True` for `and`, `0` for `-` and `<<`), and whether `x op x == x`.
    absorbing: ClassVar[Const | None] = None
    identity: ClassVar[Const | None] = None
    idempotent: ClassVar[bool] = False

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
        would disagree with. Where no value exists at all it raises NoNumber.

        A caller asks only where it sees EVERY operand. One it cannot see leaves the expression unnamed, and then the
        algebraic identities below speak for it instead -- which is why `x*0` is zero for an unknown `x` while
        `inf*0` names no number.
        """

    @property
    def mirror(self) -> "Operator | None":
        """
        Declared rather than inferred from the algebra, because the answer is about bits: `fmin`/`fmax` break ties
        toward the second operand, so exchanging them flips the sign of a zero.
        """
        return None


@dataclass(frozen=True, slots=True)
class CommutativeOperator(Operator, ABC):
    @property
    def mirror(self) -> Operator:
        return self


@dataclass(frozen=True, slots=True)
class FloatAdd(CommutativeOperator):
    mnemonic: ClassVar[str] = "fadd"
    signature: ClassVar[Signature] = Signature((FloatType(), FloatType()), FloatType())
    speculatable: ClassVar[bool] = True
    identity: ClassVar[Const | None] = FloatConst(0.0)

    def evaluate(self, operands: list[Const]) -> Const:
        return _fold_float(operands, "the sum", lambda a, b: a + b)


@dataclass(frozen=True, slots=True)
class FloatMul(CommutativeOperator):
    mnemonic: ClassVar[str] = "fmul"
    signature: ClassVar[Signature] = Signature((FloatType(), FloatType()), FloatType())
    speculatable: ClassVar[bool] = True
    # `x*0 == 0` is the charter's identity, declared here rather than hand-coded in one pass so that every rewrite
    # reasoning about known values sees it. It cannot fire on two constants: an all-known product is evaluated first,
    # and `inf*0` names no number.
    absorbing: ClassVar[Const | None] = FloatConst(0.0)
    identity: ClassVar[Const | None] = FloatConst(1.0)

    def evaluate(self, operands: list[Const]) -> Const:
        return _fold_float(operands, "the product", lambda a, b: a * b)


@dataclass(frozen=True, slots=True)
class FloatDiv(Operator):
    mnemonic: ClassVar[str] = "fdiv"
    signature: ClassVar[Signature] = Signature((FloatType(), FloatType()), FloatType())

    def evaluate(self, operands: list[Const]) -> Const:
        return _fold_float(operands, "the quotient", lambda a, b: a / b)


@dataclass(frozen=True, slots=True)
class FloatNeg(Operator):
    mnemonic: ClassVar[str] = "fneg"
    signature: ClassVar[Signature] = Signature((FloatType(),), FloatType())
    speculatable: ClassVar[bool] = True

    def evaluate(self, operands: list[Const]) -> Const:
        (a,) = [_float_const(operand) for operand in operands]
        return FloatConst(-a.value)


@dataclass(frozen=True, slots=True)
class FloatAbs(Operator):
    mnemonic: ClassVar[str] = "fabs"
    signature: ClassVar[Signature] = Signature((FloatType(),), FloatType())
    speculatable: ClassVar[bool] = True

    def evaluate(self, operands: list[Const]) -> Const:
        (a,) = [_float_const(operand) for operand in operands]
        return FloatConst(abs(a.value))


@dataclass(frozen=True, slots=True)
class FloatMulPow2(Operator):
    """Exact semantic scaling by a power of two, introduced by strength reduction."""

    mnemonic: ClassVar[str] = "fmul_pow2"
    signature: ClassVar[Signature] = Signature((FloatType(),), FloatType())
    speculatable: ClassVar[bool] = True
    k: int

    def evaluate(self, operands: list[Const]) -> Const:
        # `np.ldexp` rather than `math.ldexp` because this operator stands for a multiplication, which saturates:
        # the raise is the math module's, not the operation's, and answering it would be inventing a value.
        return _fold_float(operands, "the scaling", lambda a: float(np.ldexp(a, self.k)))


@dataclass(frozen=True, slots=True)
class FloatILog2(Operator):
    """
    `floor(log2|x|)`, the extraction half of the pair `FloatMulPow2Dynamic` scales by. Zero answers `-bias` and an
    infinity `bias + 1`, outside the finite span in the direction their magnitude implies, so an extremum over the
    answer needs no special case. The bias is told by the machine; HIR asks no format.
    """

    mnemonic: ClassVar[str] = "filog2"
    signature: ClassVar[Signature] = Signature((FloatType(),), IntType())
    speculatable: ClassVar[bool] = True
    bias: int

    def evaluate(self, operands: list[Const]) -> Const:
        (a,) = [_float_const(operand).value for operand in operands]
        if a == 0.0:
            return IntConst(-self.bias)
        if math.isinf(a):
            return IntConst(self.bias + 1)
        return IntConst(math.frexp(abs(a))[1] - 1)  # not floor(log2 x), which answers 3 just below 8


@dataclass(frozen=True, slots=True)
class FloatMulPow2Dynamic(Operator):
    """
    Scaling by a power of two whose exponent is data, `FloatMulPow2` being the constant spelling the scaling
    algebra composes.
    """

    mnemonic: ClassVar[str] = "fmul_pow2_dyn"
    signature: ClassVar[Signature] = Signature((FloatType(), IntType()), FloatType())
    speculatable: ClassVar[bool] = True

    def evaluate(self, operands: list[Const]) -> Const:
        a, k = operands  # unpacked here, `_fold_float` taking one family only
        return FloatMulPow2(_int_const(k).value).evaluate([a])


class Rounding(Enum):
    """How a float becomes an integral value: to the nearest (ties to even), or toward -inf, +inf or zero."""

    NEAREST_EVEN = "rounding"
    FLOOR = "floor"
    CEIL = "ceiling"
    TRUNC = "truncation"

    def __repr__(self) -> str:
        return self.name

    def apply(self, a: float) -> float:
        return float(_ROUNDINGS[self](a))


_ROUNDINGS: dict[Rounding, Callable[[float], float]] = {
    Rounding.NEAREST_EVEN: np.rint,
    Rounding.FLOOR: np.floor,
    Rounding.CEIL: np.ceil,
    Rounding.TRUNC: np.trunc,
}


@dataclass(frozen=True, slots=True)
class FloatRounding(Operator):
    """The integral-valued float `rounding` makes of its operand."""

    mnemonic: ClassVar[str] = "fround"
    signature: ClassVar[Signature] = Signature((FloatType(),), FloatType())
    speculatable: ClassVar[bool] = True
    rounding: Rounding

    def evaluate(self, operands: list[Const]) -> Const:
        return _fold_float(operands, f"the {self.rounding.value}", self.rounding.apply)


@dataclass(frozen=True, slots=True)
class FloatExp2(Operator):
    mnemonic: ClassVar[str] = "fexp2"
    signature: ClassVar[Signature] = Signature((FloatType(),), FloatType())
    speculatable: ClassVar[bool] = True

    def evaluate(self, operands: list[Const]) -> Const:
        # `np.exp2` is this operator's reference -- the intrinsic stub is registered against it -- and it saturates.
        # `math.exp2` raises instead, which is the math module's convention and not this operation's answer.
        return _fold_float(operands, "the exponential", lambda a: float(np.exp2(a)))


@dataclass(frozen=True, slots=True)
class FloatLog2(Operator):
    mnemonic: ClassVar[str] = "flog2"
    signature: ClassVar[Signature] = Signature((FloatType(),), FloatType())

    def evaluate(self, operands: list[Const]) -> Const:
        # np.log2 answers -inf at the 0.0 pole like the hardware; math.log2 raises there.
        return _fold_float(operands, "the base-2 logarithm", lambda a: float(np.log2(a)))


@dataclass(frozen=True, slots=True)
class FloatSin(Operator):
    mnemonic: ClassVar[str] = "fsin"
    signature: ClassVar[Signature] = Signature((FloatType(),), FloatType())
    speculatable: ClassVar[bool] = True

    def evaluate(self, operands: list[Const]) -> Const:
        return _fold_float(operands, "the sine", math.sin)


@dataclass(frozen=True, slots=True)
class FloatCos(Operator):
    mnemonic: ClassVar[str] = "fcos"
    signature: ClassVar[Signature] = Signature((FloatType(),), FloatType())
    speculatable: ClassVar[bool] = True

    def evaluate(self, operands: list[Const]) -> Const:
        return _fold_float(operands, "the cosine", math.cos)


_QUARTER_TURN = math.tau / 4.0


def _turn_sincos(a: float) -> tuple[float, float]:
    """
    Sine and cosine of an angle in turns, reduced in turns. Every step of the reduction is exact -- a remainder by
    one, a scaling by four, a split at the integer -- so a whole or quarter turn arrives as an angle of exactly zero
    and answers exactly, and no residual argument ever exceeds an eighth of a revolution. Reducing in radians can do
    neither: the reduction is inexact from the first multiplication.

    The remainder is signed, which a nonnegative one could not be without cancelling a tiny negative phase away
    entirely, so the quadrant is taken over its magnitude and the parity of each function restores the sign.
    """
    r = math.remainder(a, 1.0)  # exact, in [-0.5, 0.5]; raises over an infinity, which the caller answers for
    quadrant, fraction = divmod(abs(r) * 4.0, 1.0)
    s, c = math.sin(fraction * _QUARTER_TURN), math.cos(fraction * _QUARTER_TURN)
    sine, cosine = ((s, c), (c, -s), (-s, -c))[int(quadrant)]
    return (-sine if r < 0.0 else sine) + 0.0, cosine + 0.0


@dataclass(frozen=True, slots=True)
class FloatSinTurns(Operator):
    """
    The turn-native trigonometric vocabulary, in the spirit of C's `sinpi`: one turn is a full revolution, so a
    phase already counted in turns needs no unit conversion at all.
    """

    mnemonic: ClassVar[str] = "fsin_turns"
    signature: ClassVar[Signature] = Signature((FloatType(),), FloatType())
    speculatable: ClassVar[bool] = True

    def evaluate(self, operands: list[Const]) -> Const:
        return _fold_float(operands, "the sine", lambda a: _turn_sincos(a)[0])


@dataclass(frozen=True, slots=True)
class FloatCosTurns(Operator):
    mnemonic: ClassVar[str] = "fcos_turns"
    signature: ClassVar[Signature] = Signature((FloatType(),), FloatType())
    speculatable: ClassVar[bool] = True

    def evaluate(self, operands: list[Const]) -> Const:
        return _fold_float(operands, "the cosine", lambda a: _turn_sincos(a)[1])


@dataclass(frozen=True, slots=True)
class FloatSqrt(Operator):
    mnemonic: ClassVar[str] = "fsqrt"
    signature: ClassVar[Signature] = Signature((FloatType(),), FloatType())

    def evaluate(self, operands: list[Const]) -> Const:
        return _fold_float(operands, "the square root", math.sqrt)


@dataclass(frozen=True, slots=True)
class FloatAtan2(Operator):
    mnemonic: ClassVar[str] = "fatan2"
    signature: ClassVar[Signature] = Signature((FloatType(), FloatType()), FloatType())
    speculatable: ClassVar[bool] = True

    def evaluate(self, operands: list[Const]) -> Const:
        return _fold_float(operands, "the arctangent", math.atan2)


@dataclass(frozen=True, slots=True)
class FloatAtan2Turns(Operator):
    mnemonic: ClassVar[str] = "fatan2_turns"
    signature: ClassVar[Signature] = Signature((FloatType(), FloatType()), FloatType())
    speculatable: ClassVar[bool] = True

    def evaluate(self, operands: list[Const]) -> Const:
        return _fold_float(operands, "the arctangent", lambda y, x: math.atan2(y, x) / math.tau)


@dataclass(frozen=True, slots=True)
class FloatHypot(Operator):
    """
    Semantic because a pair may be computed as a byproduct of atan2, and because the expansion serving every other
    case needs a scaling window only the float format can supply. Speculatable: every input has an answer, and
    neither lowering can fault -- the atan2 raises nothing, and the expansion's root sees a sum of squares.
    """

    mnemonic: ClassVar[str] = "fhypot"
    speculatable: ClassVar[bool] = True
    arity: int

    def __post_init__(self) -> None:
        assert self.arity >= 1

    @property
    def signature(self) -> Signature:
        return Signature((FloatType(),) * self.arity, FloatType())

    def evaluate(self, operands: list[Const]) -> Const:
        return _fold_float(operands, "the magnitude", math.hypot)


@dataclass(frozen=True, slots=True)
class FloatIsFinite(Operator):
    mnemonic: ClassVar[str] = "fisfinite"
    signature: ClassVar[Signature] = Signature((FloatType(),), BoolType())
    speculatable: ClassVar[bool] = True

    def evaluate(self, operands: list[Const]) -> Const:
        (a,) = [_float_const(operand) for operand in operands]
        return BoolConst(math.isfinite(a.value))


@dataclass(frozen=True, slots=True)
class FloatIsInf(Operator):
    mnemonic: ClassVar[str] = "fisinf"
    signature: ClassVar[Signature] = Signature((FloatType(),), BoolType())
    speculatable: ClassVar[bool] = True

    def evaluate(self, operands: list[Const]) -> Const:
        (a,) = [_float_const(operand) for operand in operands]
        return BoolConst(math.isinf(a.value))


@dataclass(frozen=True, slots=True)
class FloatIsPosInf(Operator):
    mnemonic: ClassVar[str] = "fisposinf"
    signature: ClassVar[Signature] = Signature((FloatType(),), BoolType())
    speculatable: ClassVar[bool] = True

    def evaluate(self, operands: list[Const]) -> Const:
        (a,) = [_float_const(operand) for operand in operands]
        return BoolConst(a.value == math.inf)


@dataclass(frozen=True, slots=True)
class FloatIsNegInf(Operator):
    mnemonic: ClassVar[str] = "fisneginf"
    signature: ClassVar[Signature] = Signature((FloatType(),), BoolType())
    speculatable: ClassVar[bool] = True

    def evaluate(self, operands: list[Const]) -> Const:
        (a,) = [_float_const(operand) for operand in operands]
        return BoolConst(a.value == -math.inf)


@dataclass(frozen=True, slots=True)
class FloatFma(Operator):
    """Always single-rounds, so the contraction may not absorb another addition into it."""

    mnemonic: ClassVar[str] = "ffma"
    signature: ClassVar[Signature] = Signature((FloatType(), FloatType(), FloatType()), FloatType())
    speculatable: ClassVar[bool] = True

    def evaluate(self, operands: list[Const]) -> Const:
        return _fold_float(operands, "the fused multiply-add", math.fma)


@dataclass(frozen=True, slots=True)
class FloatMin(Operator):
    mnemonic: ClassVar[str] = "fmin"
    signature: ClassVar[Signature] = Signature((FloatType(), FloatType()), FloatType())
    speculatable: ClassVar[bool] = True
    idempotent: ClassVar[bool] = True

    def evaluate(self, operands: list[Const]) -> Const:
        a, b = [_float_const(operand) for operand in operands]
        return a if a.value < b.value else b


@dataclass(frozen=True, slots=True)
class FloatMax(Operator):
    mnemonic: ClassVar[str] = "fmax"
    signature: ClassVar[Signature] = Signature((FloatType(), FloatType()), FloatType())
    speculatable: ClassVar[bool] = True
    idempotent: ClassVar[bool] = True

    def evaluate(self, operands: list[Const]) -> Const:
        a, b = [_float_const(operand) for operand in operands]
        return b if a.value < b.value else a


@dataclass(frozen=True, slots=True)
class FloatComparison(Operator):
    mnemonic: ClassVar[str] = "fcmp"
    signature: ClassVar[Signature] = Signature((FloatType(), FloatType()), BoolType())
    speculatable: ClassVar[bool] = True
    relation: Relation

    def evaluate(self, operands: list[Const]) -> Const:
        a, b = [_float_const(operand) for operand in operands]
        return BoolConst(self.relation.holds(a.value, b.value))

    @property
    def mirror(self) -> Operator:
        return FloatComparison(self.relation.mirror)


@dataclass(frozen=True, slots=True)
class BoolAnd(CommutativeOperator):
    mnemonic: ClassVar[str] = "band"
    signature: ClassVar[Signature] = Signature((BoolType(), BoolType()), BoolType())
    speculatable: ClassVar[bool] = True
    idempotent: ClassVar[bool] = True
    absorbing: ClassVar[Const | None] = BoolConst(False)
    identity: ClassVar[Const | None] = BoolConst(True)

    def evaluate(self, operands: list[Const]) -> Const:
        a, b = [_bool_const(operand) for operand in operands]
        return BoolConst(a.value and b.value)


@dataclass(frozen=True, slots=True)
class BoolOr(CommutativeOperator):
    mnemonic: ClassVar[str] = "bor"
    signature: ClassVar[Signature] = Signature((BoolType(), BoolType()), BoolType())
    speculatable: ClassVar[bool] = True
    idempotent: ClassVar[bool] = True
    absorbing: ClassVar[Const | None] = BoolConst(True)
    identity: ClassVar[Const | None] = BoolConst(False)

    def evaluate(self, operands: list[Const]) -> Const:
        a, b = [_bool_const(operand) for operand in operands]
        return BoolConst(a.value or b.value)


@dataclass(frozen=True, slots=True)
class BoolXor(CommutativeOperator):
    mnemonic: ClassVar[str] = "bxor"
    signature: ClassVar[Signature] = Signature((BoolType(), BoolType()), BoolType())
    speculatable: ClassVar[bool] = True
    # x ^ False == x (there is no absorbing element: x ^ True == ~x, not a constant)
    identity: ClassVar[Const | None] = BoolConst(False)

    def evaluate(self, operands: list[Const]) -> Const:
        a, b = [_bool_const(operand) for operand in operands]
        return BoolConst(a.value != b.value)


@dataclass(frozen=True, slots=True)
class BoolNot(Operator):
    mnemonic: ClassVar[str] = "bnot"
    signature: ClassVar[Signature] = Signature((BoolType(),), BoolType())
    speculatable: ClassVar[bool] = True

    def evaluate(self, operands: list[Const]) -> Const:
        (a,) = [_bool_const(operand) for operand in operands]
        return BoolConst(not a.value)


@dataclass(frozen=True, slots=True)
class FloatSelect(Operator):
    """A data mux `a if cond else b` over float values, produced by if-conversion."""

    mnemonic: ClassVar[str] = "fselect"
    signature: ClassVar[Signature] = Signature((BoolType(), FloatType(), FloatType()), FloatType())
    speculatable: ClassVar[bool] = True

    def evaluate(self, operands: list[Const]) -> Const:
        cond, a, b = operands
        return _float_const(a) if _bool_const(cond).value else _float_const(b)


@dataclass(frozen=True, slots=True)
class BoolSelect(Operator):
    """
    A boolean mux `a if cond else b` over boolean values, the 1-bit dual of FloatSelect. Produced only by
    if-conversion of a boolean-phi diamond. Its constant arms (the common `True`/`False` arms of a state-machine
    merge) are reduced to `and`/`or`/`not`/passthrough by strength reduction.
    """

    mnemonic: ClassVar[str] = "bselect"
    signature: ClassVar[Signature] = Signature((BoolType(), BoolType(), BoolType()), BoolType())
    speculatable: ClassVar[bool] = True

    def evaluate(self, operands: list[Const]) -> Const:
        cond, a, b = operands
        return _bool_const(a) if _bool_const(cond).value else _bool_const(b)


@dataclass(frozen=True, slots=True)
class FloatToBool(Operator):
    """A scalar cast `bool(x)`: a float is truthy iff it is nonzero."""

    mnemonic: ClassVar[str] = "float_to_bool"
    signature: ClassVar[Signature] = Signature((FloatType(),), BoolType())
    speculatable: ClassVar[bool] = True

    def evaluate(self, operands: list[Const]) -> Const:
        (a,) = [_float_const(operand) for operand in operands]
        return BoolConst(a.value != 0.0)


@dataclass(frozen=True, slots=True)
class BoolToFloat(Operator):
    mnemonic: ClassVar[str] = "bool_to_float"
    signature: ClassVar[Signature] = Signature((BoolType(),), FloatType())
    speculatable: ClassVar[bool] = True

    def evaluate(self, operands: list[Const]) -> Const:
        (a,) = [_bool_const(operand) for operand in operands]
        return FloatConst(1.0 if a.value else 0.0)


# Signed integers before hardware width selection. Folding is exact at arbitrary precision -- no width, no saturation
# -- so a fully static integer expression disappears before MIR ever has to hold it in a machine word.
# Floor-division and modulo assert the div-by-zero error flag, so they are not speculatable.


@dataclass(frozen=True, slots=True)
class IntAdd(CommutativeOperator):
    mnemonic: ClassVar[str] = "iadd"
    signature: ClassVar[Signature] = Signature((IntType(), IntType()), IntType())
    speculatable: ClassVar[bool] = True
    identity: ClassVar[Const | None] = IntConst(0)

    def evaluate(self, operands: list[Const]) -> Const:
        return _fold_int(operands, "the sum", lambda a, b: a + b)


@dataclass(frozen=True, slots=True)
class IntSub(Operator):
    mnemonic: ClassVar[str] = "isub"
    signature: ClassVar[Signature] = Signature((IntType(), IntType()), IntType())
    speculatable: ClassVar[bool] = True
    identity: ClassVar[Const | None] = IntConst(0)

    def evaluate(self, operands: list[Const]) -> Const:
        return _fold_int(operands, "the difference", lambda a, b: a - b)


@dataclass(frozen=True, slots=True)
class IntMul(CommutativeOperator):
    mnemonic: ClassVar[str] = "imul"
    signature: ClassVar[Signature] = Signature((IntType(), IntType()), IntType())
    speculatable: ClassVar[bool] = True
    absorbing: ClassVar[Const | None] = IntConst(0)
    identity: ClassVar[Const | None] = IntConst(1)

    def evaluate(self, operands: list[Const]) -> Const:
        return _fold_int(operands, "the product", lambda a, b: a * b)


@dataclass(frozen=True, slots=True)
class IntMulPow2(Operator):
    """
    The integer dual of FloatMulPow2: exact scaling by a power of two. It is a MULTIPLICATION and not the
    `<<` that shares its arithmetic -- what leaves the word rails here where the shift drops it -- which is why the
    two cannot be one operator however alike their folding looks.
    """

    mnemonic: ClassVar[str] = "imul_pow2"
    signature: ClassVar[Signature] = Signature((IntType(),), IntType())
    speculatable: ClassVar[bool] = True
    k: int

    def __post_init__(self) -> None:
        assert self.k >= 1

    def evaluate(self, operands: list[Const]) -> Const:
        return _fold_int(operands, "the scaling", lambda a: a << self.k)


@dataclass(frozen=True, slots=True)
class IntNeg(Operator):
    mnemonic: ClassVar[str] = "ineg"
    signature: ClassVar[Signature] = Signature((IntType(),), IntType())
    speculatable: ClassVar[bool] = True

    def evaluate(self, operands: list[Const]) -> Const:
        return _fold_int(operands, "the negation", lambda a: -a)


@dataclass(frozen=True, slots=True)
class IntAbs(Operator):
    mnemonic: ClassVar[str] = "iabs"
    signature: ClassVar[Signature] = Signature((IntType(),), IntType())
    speculatable: ClassVar[bool] = True

    def evaluate(self, operands: list[Const]) -> Const:
        return _fold_int(operands, "the magnitude", abs)


@dataclass(frozen=True, slots=True)
class IntPopcount(Operator):
    mnemonic: ClassVar[str] = "ipopcnt"
    signature: ClassVar[Signature] = Signature((IntType(),), IntType())
    speculatable: ClassVar[bool] = True

    def evaluate(self, operands: list[Const]) -> Const:
        return _fold_int(operands, "the population count", int.bit_count)


@dataclass(frozen=True, slots=True)
class IntDivFloor(Operator):
    mnemonic: ClassVar[str] = "idivfloor"
    signature: ClassVar[Signature] = Signature((IntType(), IntType()), IntType())
    identity: ClassVar[Const | None] = IntConst(1)

    def evaluate(self, operands: list[Const]) -> Const:
        return _fold_int(operands, "the quotient", lambda a, b: a // b)


@dataclass(frozen=True, slots=True)
class IntMod(Operator):
    mnemonic: ClassVar[str] = "imod"
    signature: ClassVar[Signature] = Signature((IntType(), IntType()), IntType())

    def evaluate(self, operands: list[Const]) -> Const:
        return _fold_int(operands, "the remainder", lambda a, b: a % b)


@dataclass(frozen=True, slots=True)
class IntShiftLeft(Operator):
    mnemonic: ClassVar[str] = "ishl"
    signature: ClassVar[Signature] = Signature((IntType(), IntType()), IntType())
    speculatable: ClassVar[bool] = True
    identity: ClassVar[Const | None] = IntConst(0)

    def evaluate(self, operands: list[Const]) -> Const:
        return _fold_int(operands, "the left shift", lambda a, b: a << b)


@dataclass(frozen=True, slots=True)
class IntShiftRight(Operator):
    mnemonic: ClassVar[str] = "ishr"
    signature: ClassVar[Signature] = Signature((IntType(), IntType()), IntType())
    speculatable: ClassVar[bool] = True
    identity: ClassVar[Const | None] = IntConst(0)

    def evaluate(self, operands: list[Const]) -> Const:
        return _fold_int(operands, "the right shift", lambda a, b: a >> b)


@dataclass(frozen=True, slots=True)
class IntBwAnd(CommutativeOperator):
    mnemonic: ClassVar[str] = "ibwand"
    signature: ClassVar[Signature] = Signature((IntType(), IntType()), IntType())
    speculatable: ClassVar[bool] = True
    idempotent: ClassVar[bool] = True
    absorbing: ClassVar[Const | None] = IntConst(0)
    # all ones at every width
    identity: ClassVar[Const | None] = IntConst(-1)

    def evaluate(self, operands: list[Const]) -> Const:
        return _fold_int(operands, "the conjunction", lambda a, b: a & b)


@dataclass(frozen=True, slots=True)
class IntBwOr(CommutativeOperator):
    mnemonic: ClassVar[str] = "ibwor"
    signature: ClassVar[Signature] = Signature((IntType(), IntType()), IntType())
    speculatable: ClassVar[bool] = True
    idempotent: ClassVar[bool] = True
    # all ones at every width
    absorbing: ClassVar[Const | None] = IntConst(-1)
    identity: ClassVar[Const | None] = IntConst(0)

    def evaluate(self, operands: list[Const]) -> Const:
        return _fold_int(operands, "the disjunction", lambda a, b: a | b)


@dataclass(frozen=True, slots=True)
class IntBwXor(CommutativeOperator):
    mnemonic: ClassVar[str] = "ibwxor"
    signature: ClassVar[Signature] = Signature((IntType(), IntType()), IntType())
    speculatable: ClassVar[bool] = True
    identity: ClassVar[Const | None] = IntConst(0)

    def evaluate(self, operands: list[Const]) -> Const:
        return _fold_int(operands, "the exclusive disjunction", lambda a, b: a ^ b)


@dataclass(frozen=True, slots=True)
class IntBwNot(Operator):
    mnemonic: ClassVar[str] = "ibwnot"
    signature: ClassVar[Signature] = Signature((IntType(),), IntType())
    speculatable: ClassVar[bool] = True

    def evaluate(self, operands: list[Const]) -> Const:
        return _fold_int(operands, "the complement", lambda a: ~a)


@dataclass(frozen=True, slots=True)
class IntComparison(Operator):
    mnemonic: ClassVar[str] = "icmp"
    signature: ClassVar[Signature] = Signature((IntType(), IntType()), BoolType())
    speculatable: ClassVar[bool] = True
    relation: Relation

    def evaluate(self, operands: list[Const]) -> Const:
        a, b = [_int_const(operand) for operand in operands]
        return BoolConst(self.relation.holds(a.value, b.value))

    @property
    def mirror(self) -> Operator:
        return IntComparison(self.relation.mirror)


@dataclass(frozen=True, slots=True)
class IntSelect(Operator):
    """A data mux `a if cond else b` over integer values, the integer dual of FloatSelect."""

    mnemonic: ClassVar[str] = "iselect"
    signature: ClassVar[Signature] = Signature((BoolType(), IntType(), IntType()), IntType())
    speculatable: ClassVar[bool] = True

    def evaluate(self, operands: list[Const]) -> Const:
        cond, a, b = operands
        return _int_const(a) if _bool_const(cond).value else _int_const(b)


@dataclass(frozen=True, slots=True)
class IntToFloat(Operator):
    mnemonic: ClassVar[str] = "int_to_float"
    signature: ClassVar[Signature] = Signature((IntType(),), FloatType())
    speculatable: ClassVar[bool] = True

    def evaluate(self, operands: list[Const]) -> Const:
        (a,) = [_int_const(operand) for operand in operands]
        try:
            return FloatConst(float(a.value))
        except OverflowError:
            raise NoNumber(f"the float value of {a.value}") from None


@dataclass(frozen=True, slots=True)
class FloatToInt(Operator):
    """`int(rounding(x))`: `int(x)` truncates, and a conversion reading a rounding converts in that rounding's mode."""

    mnemonic: ClassVar[str] = "float_to_int"
    signature: ClassVar[Signature] = Signature((FloatType(),), IntType())
    speculatable: ClassVar[bool] = True
    rounding: Rounding = Rounding.TRUNC

    def evaluate(self, operands: list[Const]) -> Const:
        (a,) = [_float_const(operand) for operand in operands]
        try:
            return IntConst(int(self.rounding.apply(a.value)))
        except OverflowError:
            raise NoNumber(f"the integer part of {_spelled(operands)}") from None


@dataclass(frozen=True, slots=True)
class IntToBool(Operator):
    mnemonic: ClassVar[str] = "int_to_bool"
    signature: ClassVar[Signature] = Signature((IntType(),), BoolType())
    speculatable: ClassVar[bool] = True

    def evaluate(self, operands: list[Const]) -> Const:
        (a,) = [_int_const(operand) for operand in operands]
        return BoolConst(a.value != 0)


@dataclass(frozen=True, slots=True)
class BoolToInt(Operator):
    mnemonic: ClassVar[str] = "bool_to_int"
    signature: ClassVar[Signature] = Signature((BoolType(),), IntType())
    speculatable: ClassVar[bool] = True

    def evaluate(self, operands: list[Const]) -> Const:
        (a,) = [_bool_const(operand) for operand in operands]
        return IntConst(1 if a.value else 0)
