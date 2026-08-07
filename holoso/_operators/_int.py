"""
The integer operators. The pooled ones each carry their own closed-form latency and their own saturating arithmetic;
the inline ones are native Verilog over the whole wide bank, sound only under the carrier contract in DESIGN.md.
"""

from abc import ABC
from dataclasses import dataclass
from typing import ClassVar

from .._value import IntValue, ScalarValue
from .._type import BoolType, IntFormat, IntType
from ._common import (
    ComparatorOperator,
    InlineHardwareOperator,
    IntIdentity,
    PooledHardwareOperator,
    PortConditioner,
    ScalarSignature,
)


@dataclass(frozen=True, slots=True)
class IntHardwareOperator(PooledHardwareOperator, ABC):
    """
    The dual of :class:`FloatHardwareOperator`, each operator carries its own closed-form latency and RTL parameters.
    Saturation is what the integer type does at its extremes rather than a failure, and HIR marks the saturating
    operations speculatable, so the ``saturated`` sideband every module raises stays unconnected and unmodeled -- an
    if-converted arm that saturates must not raise the machine's error flag.
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

    def _validated_operands(self, operands: tuple[ScalarValue, ...]) -> tuple[IntValue, ...]:
        validated: list[IntValue] = []
        for operand in super()._validated_operands(operands):
            assert isinstance(operand, IntValue)
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
        a, b = self._validated_operands(operands)
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
        a, b = self._validated_operands(operands)
        return (a - b,)

    def render(self, *operands: str, immediates: tuple[int, ...] = ()) -> str:
        a, b = operands
        return f"{a}-{b}"


@dataclass(frozen=True, slots=True)
class IMulOperator(IntHardwareOperator):
    @dataclass(frozen=True, slots=True)
    class Options:
        stage_product: int = 0
        """Splitting the product is useful when the width exceeds the DSP slice input. See Verilog holoso_imuls."""

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
        a, b = self._validated_operands(operands)
        return (a * b,)

    def render(self, *operands: str, immediates: tuple[int, ...] = ()) -> str:
        a, b = operands
        return f"{a}×{b}"


@dataclass(frozen=True, slots=True)
class IDivOperator(IntHardwareOperator):
    """Floor division and its remainder together: one firing answers both."""

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
        a, b = self._validated_operands(operands)
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
        (a,) = self._validated_operands(operands)
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
        a, b = self._validated_operands(operands)
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
        a, b = self._validated_operands(operands)
        ordering = a.compare(b)
        return ordering > 0, ordering == 0, ordering < 0


@dataclass(frozen=True, slots=True)
class IntInlineOperator(InlineHardwareOperator, ABC):
    """The inline dual of :class:`IntHardwareOperator`."""

    fmt: IntFormat

    @property
    def scalar_type(self) -> IntType:
        return IntType(self.fmt)

    def _validated_operands(self, operands: tuple[ScalarValue, ...]) -> tuple[IntValue, ...]:
        validated: list[IntValue] = []
        for operand in super()._validated_operands(operands):
            assert isinstance(operand, IntValue)
            validated.append(operand)
        return tuple(validated)


@dataclass(frozen=True, slots=True)
class IntBitwiseOperator(IntInlineOperator, ABC):
    @property
    def signature(self) -> ScalarSignature:
        return ScalarSignature((self.scalar_type,) * 2, (self.scalar_type,))


@dataclass(frozen=True, slots=True)
class IntBwAndOperator(IntBitwiseOperator):
    mnemonic: ClassVar[str] = "ibwand"

    def verilog_expr(self, *operand_nets: str) -> str:
        a, b = operand_nets
        return f"({a}) & ({b})"

    def evaluate(self, *operands: ScalarValue, immediates: tuple[int, ...] = ()) -> tuple[IntValue, ...]:
        a, b = self._validated_operands(operands)
        return (a & b,)


@dataclass(frozen=True, slots=True)
class IntBwOrOperator(IntBitwiseOperator):
    mnemonic: ClassVar[str] = "ibwor"

    def verilog_expr(self, *operand_nets: str) -> str:
        a, b = operand_nets
        return f"({a}) | ({b})"

    def evaluate(self, *operands: ScalarValue, immediates: tuple[int, ...] = ()) -> tuple[IntValue, ...]:
        a, b = self._validated_operands(operands)
        return (a | b,)


@dataclass(frozen=True, slots=True)
class IntBwXorOperator(IntBitwiseOperator):
    mnemonic: ClassVar[str] = "ibwxor"

    def verilog_expr(self, *operand_nets: str) -> str:
        a, b = operand_nets
        return f"({a}) ^ ({b})"

    def evaluate(self, *operands: ScalarValue, immediates: tuple[int, ...] = ()) -> tuple[IntValue, ...]:
        a, b = self._validated_operands(operands)
        return (a ^ b,)


@dataclass(frozen=True, slots=True)
class IntBwNotOperator(IntInlineOperator):
    mnemonic: ClassVar[str] = "ibwnot"

    @property
    def signature(self) -> ScalarSignature:
        return ScalarSignature((self.scalar_type,), (self.scalar_type,))

    def verilog_expr(self, *operand_nets: str) -> str:
        (a,) = operand_nets
        return f"~({a})"

    def evaluate(self, *operands: ScalarValue, immediates: tuple[int, ...] = ()) -> tuple[IntValue, ...]:
        (a,) = self._validated_operands(operands)
        return (~a,)


@dataclass(frozen=True, slots=True)
class IntShiftConstOperator(IntInlineOperator):
    """
    An arithmetic shift by a count fixed at compile time, left when positive; the raw bit shift, so a left shift
    drops what leaves the word rather than saturating as the pooled ``holoso_ishift`` also offers.
    Shift by 0 or by an amount exceeding the operand width is a compile-time error (must fold/reject before LIR).
    """

    mnemonic: ClassVar[str] = "ishiftc"
    shamt: int

    def __post_init__(self) -> None:
        if self.shamt == 0:
            raise ValueError("ishiftc by 0 is the identity, which HIR or MIR must fold rather than select hardware")
        if abs(self.shamt) >= self.fmt.width:
            raise ValueError(
                f"ishiftc by {self.shamt} is not a shift within {self.fmt}: a count reaching the word answers a "
                f"constant or a sign fill, and clamping a width-less count to the word is the lowering's job"
            )

    @property
    def signature(self) -> ScalarSignature:
        return ScalarSignature((self.scalar_type,), (self.scalar_type,))

    def verilog_expr(self, *operand_nets: str) -> str:
        (a,) = operand_nets
        if self.shamt < 0:
            return f"{{$signed({a}) >>> {-self.shamt}}}"
        return f"({a}) << {self.shamt}"

    def render(self, *operands: str, immediates: tuple[int, ...] = ()) -> str:
        (a,) = operands
        return f"{a}<<{self.shamt}" if self.shamt > 0 else f"{a}>>{-self.shamt}"

    def evaluate(self, *operands: ScalarValue, immediates: tuple[int, ...] = ()) -> tuple[IntValue, ...]:
        (a,) = self._validated_operands(operands)
        return (a.shift(IntValue.from_int(self.fmt, self.shamt)).shft,)


@dataclass(frozen=True, slots=True)
class IntToBoolOperator(InlineHardwareOperator):
    mnemonic: ClassVar[str] = "itobool"
    fmt: IntFormat

    @property
    def signature(self) -> ScalarSignature:
        return ScalarSignature((IntType(self.fmt),), (BoolType(),))

    def render(self, *operands: str, immediates: tuple[int, ...] = ()) -> str:
        (a,) = operands
        return f"bool({a})"

    def verilog_expr(self, *operand_nets: str) -> str:
        (a,) = operand_nets
        return f"|({a})"

    def evaluate(self, *operands: ScalarValue, immediates: tuple[int, ...] = ()) -> tuple[bool, ...]:
        (a,) = self._validated_operands(operands)
        assert isinstance(a, IntValue)
        return (a.value != 0,)


@dataclass(frozen=True, slots=True)
class BoolToIntOperator(InlineHardwareOperator):
    mnemonic: ClassVar[str] = "ifrombool"
    fmt: IntFormat

    @property
    def signature(self) -> ScalarSignature:
        return ScalarSignature((BoolType(),), (IntType(self.fmt),))

    def render(self, *operands: str, immediates: tuple[int, ...] = ()) -> str:
        (a,) = operands
        return f"int({a})"

    def verilog_expr(self, *operand_nets: str) -> str:
        (a,) = operand_nets
        # Concatenation makes the widening self-determined: an operand net carrying a folded inversion arrives as
        # `~net`, and a bare one spliced into a wide register write would complement the carrier, not the single bit.
        return f"{{1'b0, {a}}}"

    def evaluate(self, *operands: ScalarValue, immediates: tuple[int, ...] = ()) -> tuple[IntValue, ...]:
        (a,) = self._validated_operands(operands)
        assert isinstance(a, bool)
        return (IntValue.from_int(self.fmt, int(a)),)
