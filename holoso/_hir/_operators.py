"""Semantic HIR operators."""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True, slots=True)
class Operator(ABC):
    """A reusable semantic operation definition referenced by HIR operation nodes."""

    mnemonic: ClassVar[str]
    arity: ClassVar[int]

    @abstractmethod
    def evaluate(self, *operands: float) -> float | None:
        """Return the folded value, or ``None`` if this operation should not be constant-folded."""

    @abstractmethod
    def render(self, *operands: str) -> str:
        """Human-friendly expression for diagnostics."""


@dataclass(frozen=True, slots=True)
class Add(Operator):
    mnemonic: ClassVar[str] = "add"
    arity: ClassVar[int] = 2

    def evaluate(self, *operands: float) -> float:
        a, b = operands
        return a + b

    def render(self, *operands: str) -> str:
        a, b = operands
        return f"{a}+{b}"


@dataclass(frozen=True, slots=True)
class Mul(Operator):
    mnemonic: ClassVar[str] = "mul"
    arity: ClassVar[int] = 2

    def evaluate(self, *operands: float) -> float:
        a, b = operands
        return a * b

    def render(self, *operands: str) -> str:
        a, b = operands
        return f"{a}×{b}"


@dataclass(frozen=True, slots=True)
class Div(Operator):
    mnemonic: ClassVar[str] = "div"
    arity: ClassVar[int] = 2

    def evaluate(self, *operands: float) -> float | None:
        a, b = operands
        return a / b if b != 0 else None

    def render(self, *operands: str) -> str:
        a, b = operands
        return f"{a}/{b}"


@dataclass(frozen=True, slots=True)
class Neg(Operator):
    mnemonic: ClassVar[str] = "neg"
    arity: ClassVar[int] = 1

    def evaluate(self, *operands: float) -> float:
        (a,) = operands
        return -a

    def render(self, *operands: str) -> str:
        (a,) = operands
        return f"-{a}"


@dataclass(frozen=True, slots=True)
class Abs(Operator):
    mnemonic: ClassVar[str] = "abs"
    arity: ClassVar[int] = 1

    def evaluate(self, *operands: float) -> float:
        (a,) = operands
        return abs(a)

    def render(self, *operands: str) -> str:
        (a,) = operands
        return f"|{a}|"


@dataclass(frozen=True, slots=True)
class MulPow2(Operator):
    """Exact semantic scaling by a power of two, introduced by strength reduction."""

    mnemonic: ClassVar[str] = "mul_pow2"
    arity: ClassVar[int] = 1
    k: int

    def evaluate(self, *operands: float) -> float:
        (a,) = operands
        return math.ldexp(a, self.k)

    def render(self, *operands: str) -> str:
        (a,) = operands
        return f"{a}×2^{self.k}"
