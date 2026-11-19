"""The boolean primitives: plain gates folded into their register's write, needing no configuration."""

from abc import ABC
from dataclasses import dataclass

from .._value import ScalarValue
from .._type import BoolType
from ._common import InlinePrimitive, ScalarSignature


@dataclass(frozen=True, slots=True)
class _BoolLogicPrimitive(InlinePrimitive, ABC):
    """
    A boolean-logic primitive (AND/OR/XOR): a plain `& | ^` gate folded into its boolean register's write.
    Never added to OpConfig -- it has no module and no configuration.
    """

    @property
    def signature(self) -> ScalarSignature:
        return ScalarSignature((BoolType(), BoolType()), BoolType())

    def _validated_operands(self, operands: tuple[ScalarValue, ...]) -> tuple[bool, ...]:
        validated: list[bool] = []
        for operand in super()._validated_operands(operands):
            assert isinstance(operand, bool)
            validated.append(operand)
        return tuple(validated)


@dataclass(frozen=True, slots=True)
class BoolAndPrimitive(_BoolLogicPrimitive):
    def verilog_expr(self, *operand_nets: str) -> str:
        a, b = operand_nets
        return f"({a}) & ({b})"

    def evaluate(self, *operands: ScalarValue) -> bool:
        a, b = self._validated_operands(operands)
        return a and b


@dataclass(frozen=True, slots=True)
class BoolOrPrimitive(_BoolLogicPrimitive):
    def verilog_expr(self, *operand_nets: str) -> str:
        a, b = operand_nets
        return f"({a}) | ({b})"

    def evaluate(self, *operands: ScalarValue) -> bool:
        a, b = self._validated_operands(operands)
        return a or b


@dataclass(frozen=True, slots=True)
class BoolXorPrimitive(_BoolLogicPrimitive):
    def verilog_expr(self, *operand_nets: str) -> str:
        a, b = operand_nets
        return f"({a}) ^ ({b})"

    def evaluate(self, *operands: ScalarValue) -> bool:
        a, b = self._validated_operands(operands)
        return a != b
