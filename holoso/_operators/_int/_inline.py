from abc import ABC
from dataclasses import dataclass

from ..._type import BoolType, IntFormat, IntType
from ..._value import IntValue, ScalarValue
from .._common import InlinePrimitive, ScalarSignature


@dataclass(frozen=True, slots=True)
class _IntInlinePrimitive(InlinePrimitive, ABC):
    fmt: IntFormat

    @property
    def _scalar_type(self) -> IntType:
        return IntType(self.fmt)

    def _validated_operands(self, operands: tuple[ScalarValue, ...]) -> tuple[IntValue, ...]:
        validated: list[IntValue] = []
        for operand in super()._validated_operands(operands):
            assert isinstance(operand, IntValue)
            validated.append(operand)
        return tuple(validated)


@dataclass(frozen=True, slots=True)
class _IntBitwisePrimitive(_IntInlinePrimitive, ABC):
    @property
    def signature(self) -> ScalarSignature:
        return ScalarSignature((self._scalar_type,) * 2, self._scalar_type)


@dataclass(frozen=True, slots=True)
class IntBwAndPrimitive(_IntBitwisePrimitive):
    def verilog_expr(self, *operand_nets: str) -> str:
        a, b = operand_nets
        return f"({a}) & ({b})"

    def evaluate(self, *operands: ScalarValue) -> IntValue:
        a, b = self._validated_operands(operands)
        return a & b


@dataclass(frozen=True, slots=True)
class IntBwOrPrimitive(_IntBitwisePrimitive):
    def verilog_expr(self, *operand_nets: str) -> str:
        a, b = operand_nets
        return f"({a}) | ({b})"

    def evaluate(self, *operands: ScalarValue) -> IntValue:
        a, b = self._validated_operands(operands)
        return a | b


@dataclass(frozen=True, slots=True)
class IntBwXorPrimitive(_IntBitwisePrimitive):
    def verilog_expr(self, *operand_nets: str) -> str:
        a, b = operand_nets
        return f"({a}) ^ ({b})"

    def evaluate(self, *operands: ScalarValue) -> IntValue:
        a, b = self._validated_operands(operands)
        return a ^ b


@dataclass(frozen=True, slots=True)
class IntBwNotPrimitive(_IntInlinePrimitive):
    @property
    def signature(self) -> ScalarSignature:
        return ScalarSignature((self._scalar_type,), self._scalar_type)

    def verilog_expr(self, *operand_nets: str) -> str:
        (a,) = operand_nets
        return f"~({a})"

    def evaluate(self, *operands: ScalarValue) -> IntValue:
        (a,) = self._validated_operands(operands)
        return ~a


@dataclass(frozen=True, slots=True)
class IntShiftConstPrimitive(_IntInlinePrimitive):
    """
    An arithmetic shift by a count fixed at compile time, left when positive; the raw bit shift, as the pooled shifter
    computes for a runtime count, so a left shift drops what leaves the word.
    """

    shamt: int

    def __post_init__(self) -> None:
        super().__post_init__()
        # Zero is the identity, which HIR or MIR folds; a count reaching the word answers a constant or a sign fill, and
        # clamping a width-less count to the word is the lowering's job.
        assert 0 < abs(self.shamt) < self.fmt.width, self.shamt

    @property
    def signature(self) -> ScalarSignature:
        return ScalarSignature((self._scalar_type,), self._scalar_type)

    def verilog_expr(self, *operand_nets: str) -> str:
        (a,) = operand_nets
        if self.shamt < 0:
            return f"{{$signed({a}) >>> {-self.shamt}}}"
        return f"({a}) << {self.shamt}"

    def render(self, *operands: str) -> str:
        (a,) = operands
        return f"{a}<<{self.shamt}" if self.shamt > 0 else f"{a}>>{-self.shamt}"

    def evaluate(self, *operands: ScalarValue) -> IntValue:
        (a,) = self._validated_operands(operands)
        return a.shift_left(IntValue.from_int(self.fmt, self.shamt))


@dataclass(frozen=True, slots=True)
class IntToBoolPrimitive(InlinePrimitive):
    fmt: IntFormat

    @property
    def signature(self) -> ScalarSignature:
        return ScalarSignature((IntType(self.fmt),), BoolType())

    def render(self, *operands: str) -> str:
        (a,) = operands
        return f"bool({a})"

    def verilog_expr(self, *operand_nets: str) -> str:
        (a,) = operand_nets
        return f"|({a})"

    def evaluate(self, *operands: ScalarValue) -> bool:
        (a,) = self._validated_operands(operands)
        assert isinstance(a, IntValue)
        return a.value != 0


@dataclass(frozen=True, slots=True)
class BoolToIntPrimitive(InlinePrimitive):
    fmt: IntFormat

    @property
    def signature(self) -> ScalarSignature:
        return ScalarSignature((BoolType(),), IntType(self.fmt))

    def render(self, *operands: str) -> str:
        (a,) = operands
        return f"int({a})"

    def verilog_expr(self, *operand_nets: str) -> str:
        (a,) = operand_nets
        # Concatenation makes the widening self-determined: an operand net carrying a folded inversion arrives as
        # `~net`, and a bare one spliced into a wide register write would complement the carrier, not the single bit.
        return f"{{1'b0, {a}}}"

    def evaluate(self, *operands: ScalarValue) -> IntValue:
        (a,) = self._validated_operands(operands)
        assert isinstance(a, bool)
        return IntValue.from_int(self.fmt, int(a))
