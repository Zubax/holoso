"""The integer operators. Each carries its own closed-form latency and its own saturating arithmetic."""

from abc import ABC
from dataclasses import dataclass
from typing import ClassVar

from .._value import IntValue, ScalarValue
from .._type import IntFormat, IntType
from ._common import ComparatorOperator, IntIdentity, PooledHardwareOperator, PortConditioner, ScalarSignature


@dataclass(frozen=True, slots=True)
class IntHardwareOperator(PooledHardwareOperator, ABC):
    """
    The dual of :class:`FloatHardwareOperator`, each operator carries its own closed-form latency and RTL parameters.
    Saturation is what the integer type does at its extremes rather than a failure, and HIR marks the saturating
    operations speculatable, so the ``saturated`` sideband every module raises stays unconnected and unmodeled -- an
    if-converted arm that saturates must not raise the machine's error flag. Only a division by zero is an error.
    """

    fmt: IntFormat

    @property
    def latency(self) -> int:
        """Every integer core latches its inputs and its outputs; one with internal stages adds them to this."""
        return 2

    @property
    def scalar_type(self) -> IntType:
        return IntType(self.fmt)

    @property
    def params(self) -> dict[str, int]:
        return {"W": self.fmt.width, "LATENCY": self.latency}

    def _validated_operands(self, operands: tuple[ScalarValue, ...], arity: int) -> tuple[IntValue, ...]:
        if len(operands) != arity:
            raise ValueError(f"{self.mnemonic} expected {arity} operands, got {len(operands)}")
        validated: list[IntValue] = []
        for index, operand in enumerate(operands):
            if not isinstance(operand, IntValue):
                raise TypeError(f"{self.mnemonic} operand {index} must be IntValue, got {type(operand).__name__}")
            if operand.fmt != self.fmt:
                raise ValueError(f"{self.mnemonic} operand {index} has {operand.fmt}, expected {self.fmt}")
            validated.append(operand)
        return tuple(validated)


@dataclass(frozen=True, slots=True)
class IAddOperator(IntHardwareOperator):
    mnemonic: ClassVar[str] = "iadds"
    operand_hdl_ports: ClassVar[list[str]] = ["a", "b"]
    output_hdl_ports: ClassVar[list[str]] = ["y"]
    swap_output_permutation: ClassVar[tuple[int, ...]] = (0,)

    @property
    def signature(self) -> ScalarSignature:
        return ScalarSignature((self.scalar_type,) * 2, (self.scalar_type,))

    def evaluate(self, *operands: ScalarValue, immediates: tuple[int, ...] = ()) -> tuple[IntValue, ...]:
        a, b = self._validated_operands(operands, 2)
        return (a + b,)

    def render(self, *operands: str, immediates: tuple[int, ...] = ()) -> str:
        a, b = operands
        return f"{a}+{b}"


@dataclass(frozen=True, slots=True)
class ISubOperator(IntHardwareOperator):
    """Also serves negation as ``0-x``: there is no negation module, and this one saturates ``-MIN`` correctly."""

    mnemonic: ClassVar[str] = "isubs"
    operand_hdl_ports: ClassVar[list[str]] = ["a", "b"]
    output_hdl_ports: ClassVar[list[str]] = ["y"]

    @property
    def signature(self) -> ScalarSignature:
        return ScalarSignature((self.scalar_type,) * 2, (self.scalar_type,))

    def evaluate(self, *operands: ScalarValue, immediates: tuple[int, ...] = ()) -> tuple[IntValue, ...]:
        a, b = self._validated_operands(operands, 2)
        return (a - b,)

    def render(self, *operands: str, immediates: tuple[int, ...] = ()) -> str:
        a, b = operands
        return f"{a}-{b}"


@dataclass(frozen=True, slots=True)
class IMulOperator(IntHardwareOperator):
    @dataclass(frozen=True, slots=True)
    class Options:
        stage_product: int = 0  # splitting the product is rarely useful unless the width exceeds the DSP slice input

    mnemonic: ClassVar[str] = "imuls"
    operand_hdl_ports: ClassVar[list[str]] = ["a", "b"]
    output_hdl_ports: ClassVar[list[str]] = ["y"]
    swap_output_permutation: ClassVar[tuple[int, ...]] = (0,)
    opt: Options

    def __post_init__(self) -> None:
        if not 0 <= self.opt.stage_product <= 4:
            raise ValueError(f"imuls stage_product must be in 0..4, got {self.opt.stage_product}")

    @property
    def latency(self) -> int:
        return 2 + self.opt.stage_product

    @property
    def params(self) -> dict[str, int]:
        return {"W": self.fmt.width, "STAGE_PRODUCT": self.opt.stage_product, "LATENCY": self.latency}

    @property
    def signature(self) -> ScalarSignature:
        return ScalarSignature((self.scalar_type,) * 2, (self.scalar_type,))

    def evaluate(self, *operands: ScalarValue, immediates: tuple[int, ...] = ()) -> tuple[IntValue, ...]:
        a, b = self._validated_operands(operands, 2)
        return (a * b,)

    def render(self, *operands: str, immediates: tuple[int, ...] = ()) -> str:
        a, b = operands
        return f"{a}×{b}"


@dataclass(frozen=True, slots=True)
class IDivOperator(IntHardwareOperator):
    """Floor division and its remainder together: one firing answers both, as the sorter answers min and max."""

    mnemonic: ClassVar[str] = "idivs"
    operand_hdl_ports: ClassVar[list[str]] = ["num", "den"]
    output_hdl_ports: ClassVar[list[str]] = ["quo", "rem"]
    error_ports: ClassVar[list[str]] = ["div0"]

    @property
    def latency(self) -> int:
        return 3 + (self.fmt.width + 1) // 2  # one radix-4 step per two quotient bits

    @property
    def params(self) -> dict[str, int]:
        # Floor, because that is what Python's ``//`` and ``%`` mean and HIR has no other division; the core's
        # truncating mode is unreachable from a kernel.
        return {"W": self.fmt.width, "QUOTIENT_FLOOR": 1, "LATENCY": self.latency}

    @property
    def signature(self) -> ScalarSignature:
        return ScalarSignature((self.scalar_type,) * 2, (self.scalar_type,) * 2)

    def evaluate(self, *operands: ScalarValue, immediates: tuple[int, ...] = ()) -> tuple[IntValue, ...]:
        a, b = self._validated_operands(operands, 2)
        return a.divmod_floor(b)

    def render(self, *operands: str, immediates: tuple[int, ...] = ()) -> str:
        a, b = operands
        return f"{a}//{b}"

    def render_output(
        self, port: int, conditioner: PortConditioner, *operands: str, immediates: tuple[int, ...] = ()
    ) -> str:
        assert isinstance(conditioner, IntIdentity)
        a, b = operands
        return f"{a}//{b}" if port == 0 else f"{a}%{b}"


@dataclass(frozen=True, slots=True)
class IAbsOperator(IntHardwareOperator):
    mnemonic: ClassVar[str] = "iabss"
    operand_hdl_ports: ClassVar[list[str]] = ["x"]
    output_hdl_ports: ClassVar[list[str]] = ["y"]

    @property
    def signature(self) -> ScalarSignature:
        return ScalarSignature((self.scalar_type,) * 1, (self.scalar_type,))

    def evaluate(self, *operands: ScalarValue, immediates: tuple[int, ...] = ()) -> tuple[IntValue, ...]:
        (a,) = self._validated_operands(operands, 1)
        return (abs(a),)

    def render(self, *operands: str, immediates: tuple[int, ...] = ()) -> str:
        (a,) = operands
        return f"|{a}|"


@dataclass(frozen=True, slots=True)
class IShiftOperator(IntHardwareOperator):
    """
    An arithmetic shift by a signed amount, left when positive. It emits both readings of a left shift at once:
    ``shft`` lets the high bits fall off the word, while ``prod`` is the multiplication by a power of two, saturating
    instead. Which one a shift wants is a lowering decision, so the operator commits to neither.
    """

    mnemonic: ClassVar[str] = "ishift"
    operand_hdl_ports: ClassVar[list[str]] = ["x", "shamt"]
    output_hdl_ports: ClassVar[list[str]] = ["shft", "prod"]

    @property
    def signature(self) -> ScalarSignature:
        return ScalarSignature((self.scalar_type,) * 2, (self.scalar_type,) * 2)

    def evaluate(self, *operands: ScalarValue, immediates: tuple[int, ...] = ()) -> tuple[IntValue, ...]:
        a, b = self._validated_operands(operands, 2)
        return a.shift(b)

    def render(self, *operands: str, immediates: tuple[int, ...] = ()) -> str:
        a, b = operands
        return f"{a}<<{b}"

    def render_output(
        self, port: int, conditioner: PortConditioner, *operands: str, immediates: tuple[int, ...] = ()
    ) -> str:
        assert isinstance(conditioner, IntIdentity)
        return f"{self.output_hdl_ports[port]}({', '.join(operands)})"


@dataclass(frozen=True, slots=True)
class ICmpOperator(IntHardwareOperator, ComparatorOperator):
    """Two's complement is totally ordered."""

    mnemonic: ClassVar[str] = "icmp"

    def evaluate(self, *operands: ScalarValue, immediates: tuple[int, ...] = ()) -> tuple[bool, ...]:
        a, b = self._validated_operands(operands, 2)
        ordering = a.compare(b)
        return ordering > 0, ordering == 0, ordering < 0
