"""Typed HIR constant values."""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .._errors import UnsupportedConstruct
from ._types import BoolType, FloatType, IntType, Type


@dataclass(frozen=True, slots=True)
class Const(ABC):
    @property
    @abstractmethod
    def type(self) -> Type: ...


@dataclass(frozen=True, slots=True)
class FloatConst(Const):
    value: float

    def __post_init__(self) -> None:
        if math.isnan(self.value):
            raise UnsupportedConstruct("Holoso cannot represent a NaN constant. Only [in]finite numbers are supported.")
        if self.value == 0.0:  # Normalize -0.0 to +0.0.
            object.__setattr__(self, "value", 0.0)

    @property
    def type(self) -> FloatType:
        return FloatType()


@dataclass(frozen=True, slots=True)
class BoolConst(Const):
    value: bool

    @property
    def type(self) -> BoolType:
        return BoolType()


@dataclass(frozen=True, slots=True)
class IntConst(Const):
    value: int

    @property
    def type(self) -> IntType:
        return IntType()
