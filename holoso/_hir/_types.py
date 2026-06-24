"""Format-free HIR value types."""

from abc import ABC
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Type(ABC): ...


@dataclass(frozen=True, slots=True)
class FloatType(Type):
    """A semantic floating-point scalar before hardware format selection."""


@dataclass(frozen=True, slots=True)
class BoolType(Type):
    """A semantic single-bit boolean: branch conditions, comparison results, and boolean state."""


@dataclass(frozen=True, slots=True)
class Signature:
    operand_types: tuple[Type, ...]
    result_type: Type

    @property
    def arity(self) -> int:
        return len(self.operand_types)
