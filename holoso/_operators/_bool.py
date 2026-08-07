"""The boolean operators: plain gates folded into their register's write, needing no configuration."""

from abc import ABC
from dataclasses import dataclass
from typing import ClassVar

from .._value import ScalarValue
from .._type import BoolType
from ._common import InlineHardwareOperator, ScalarSignature


@dataclass(frozen=True, slots=True)
class BoolLogicOperator(InlineHardwareOperator, ABC):
    """
    A boolean-logic operator (AND/OR/XOR): a plain ``& | ^`` gate folded into its boolean register's write.
    Never added to :class:`OpConfig` -- it has no module and no configuration.
    """

    @property
    def signature(self) -> ScalarSignature:
        return ScalarSignature((BoolType(), BoolType()), (BoolType(),))

    def _validated_operands(self, operands: tuple[ScalarValue, ...]) -> tuple[bool, ...]:
        validated: list[bool] = []
        for operand in super()._validated_operands(operands):
            assert isinstance(operand, bool)
            validated.append(operand)
        return tuple(validated)


@dataclass(frozen=True, slots=True)
class BoolAndOperator(BoolLogicOperator):
    mnemonic: ClassVar[str] = "band"

    def verilog_expr(self, *operand_nets: str) -> str:
        a, b = operand_nets
        return f"{a} & {b}"

    def evaluate(self, *operands: ScalarValue, immediates: tuple[int, ...] = ()) -> tuple[bool, ...]:
        a, b = self._validated_operands(operands)
        return (a and b,)


@dataclass(frozen=True, slots=True)
class BoolOrOperator(BoolLogicOperator):
    mnemonic: ClassVar[str] = "bor"

    def verilog_expr(self, *operand_nets: str) -> str:
        a, b = operand_nets
        return f"{a} | {b}"

    def evaluate(self, *operands: ScalarValue, immediates: tuple[int, ...] = ()) -> tuple[bool, ...]:
        a, b = self._validated_operands(operands)
        return (a or b,)


@dataclass(frozen=True, slots=True)
class BoolXorOperator(BoolLogicOperator):
    mnemonic: ClassVar[str] = "bxor"

    def verilog_expr(self, *operand_nets: str) -> str:
        a, b = operand_nets
        return f"{a} ^ {b}"

    def evaluate(self, *operands: ScalarValue, immediates: tuple[int, ...] = ()) -> tuple[bool, ...]:
        a, b = self._validated_operands(operands)
        return (a != b,)
