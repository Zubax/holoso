"""Typed HIR constant values."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ._types import BoolType, FloatType, Type


@dataclass(frozen=True, slots=True)
class Const(ABC):
    @property
    @abstractmethod
    def type(self) -> Type: ...


@dataclass(frozen=True, slots=True)
class FloatConst(Const):
    value: float

    @property
    def type(self) -> FloatType:
        return FloatType()


@dataclass(frozen=True, slots=True)
class BoolConst(Const):
    value: bool

    @property
    def type(self) -> BoolType:
        return BoolType()
