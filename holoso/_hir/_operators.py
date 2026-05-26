"""Semantic HIR operators."""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar
from typing import TYPE_CHECKING

from ._types import FloatType, Signature

if TYPE_CHECKING:
    from ._ir import Const


def _float_signature(arity: int) -> Signature:
    ty = FloatType()
    return Signature((ty,) * arity, ty)


@dataclass(frozen=True, slots=True)
class Operator(ABC):
    """A reusable semantic operation definition referenced by HIR operation nodes."""

    mnemonic: ClassVar[str]

    @property
    @abstractmethod
    def signature(self) -> Signature:
        """Semantic operand/result types."""

    @property
    def arity(self) -> int:
        return self.signature.arity

    @abstractmethod
    def fold_constants(self, operands: list["Const"]) -> "Const | None":
        """Return the folded constant node, or ``None`` if this operation should not be constant-folded."""


@dataclass(frozen=True, slots=True)
class FloatAdd(Operator):
    mnemonic: ClassVar[str] = "add"

    @property
    def signature(self) -> Signature:
        return _float_signature(2)

    def fold_constants(self, operands: list["Const"]) -> "Const":
        from ._ir import FloatConst

        a, b = operands
        assert isinstance(a, FloatConst)
        assert isinstance(b, FloatConst)
        return FloatConst(a.value + b.value)


@dataclass(frozen=True, slots=True)
class FloatMul(Operator):
    mnemonic: ClassVar[str] = "mul"

    @property
    def signature(self) -> Signature:
        return _float_signature(2)

    def fold_constants(self, operands: list["Const"]) -> "Const":
        from ._ir import FloatConst

        a, b = operands
        assert isinstance(a, FloatConst)
        assert isinstance(b, FloatConst)
        return FloatConst(a.value * b.value)


@dataclass(frozen=True, slots=True)
class FloatDiv(Operator):
    mnemonic: ClassVar[str] = "div"

    @property
    def signature(self) -> Signature:
        return _float_signature(2)

    def fold_constants(self, operands: list["Const"]) -> "Const | None":
        from ._ir import FloatConst

        a, b = operands
        assert isinstance(a, FloatConst)
        assert isinstance(b, FloatConst)
        return FloatConst(a.value / b.value) if b.value != 0 else None


@dataclass(frozen=True, slots=True)
class FloatNeg(Operator):
    mnemonic: ClassVar[str] = "neg"

    @property
    def signature(self) -> Signature:
        return _float_signature(1)

    def fold_constants(self, operands: list["Const"]) -> "Const":
        from ._ir import FloatConst

        (a,) = operands
        assert isinstance(a, FloatConst)
        return FloatConst(-a.value)


@dataclass(frozen=True, slots=True)
class FloatAbs(Operator):
    mnemonic: ClassVar[str] = "abs"

    @property
    def signature(self) -> Signature:
        return _float_signature(1)

    def fold_constants(self, operands: list["Const"]) -> "Const":
        from ._ir import FloatConst

        (a,) = operands
        assert isinstance(a, FloatConst)
        return FloatConst(abs(a.value))


@dataclass(frozen=True, slots=True)
class FloatMulPow2(Operator):
    """Exact semantic scaling by a power of two, introduced by strength reduction."""

    mnemonic: ClassVar[str] = "mul_pow2"
    k: int

    @property
    def signature(self) -> Signature:
        return _float_signature(1)

    def fold_constants(self, operands: list["Const"]) -> "Const":
        from ._ir import FloatConst

        (a,) = operands
        assert isinstance(a, FloatConst)
        return FloatConst(math.ldexp(a.value, self.k))
