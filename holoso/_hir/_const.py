"""Typed HIR constant values."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ._types import FloatType, Type


@dataclass(frozen=True, slots=True)
class Const(ABC):
    """A typed HIR constant value."""

    @property
    @abstractmethod
    def type(self) -> Type:
        """The HIR type of this constant."""


@dataclass(frozen=True, slots=True)
class FloatConst(Const):
    """A floating-point constant."""

    value: float

    @property
    def type(self) -> FloatType:
        return FloatType()
