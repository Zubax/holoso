from dataclasses import dataclass, replace
from collections.abc import Mapping
from typing import ClassVar, Self

import zkf
from amaranth.hdl import Cat, Const, Module, Mux, Signal, Value, signed
from amaranth.lib import wiring
from amaranth.lib.wiring import In, Out

from ..._value import IntValue, ScalarValue
from ..._type import IntFormat
from .._common import BaseOperatorOptions, HardwareOperator, PooledPrimitive, agreed, int_operand
from ._common import Datapath, IntBlock, int_ports


@dataclass(frozen=True, kw_only=True, eq=False)
class IShft(IntBlock):
    """
    Arithmetic shift of `x` by the signed `shamt`: left when `right` is low and right when it is high, a negative
    `shamt` reversing whichever of the two that is. It is the raw bit shift: a left shift lets the high bits fall off
    the word instead of saturating, and a right shift fills with the sign.
    """

    operator = "ishft"
    _right: bool | None = None

    def specialized(self, *, right: bool) -> IShft:
        """This configuration built for the left shift alone or the right shift alone."""
        return replace(self, _right=right)

    @property
    def timing(self) -> zkf.Timing:
        return zkf.Timing(self.stage_input + self.stage_output, 1)

    @property
    def signature(self) -> wiring.Signature:
        word = signed(self.wint)
        return wiring.Signature(
            {
                **({"right": In(1)} if self._right is None else {}),
                "x": In(word),
                "shamt": In(word),
                "shft": Out(word),
            }
        )

    def component(self) -> _IShft:
        return _IShft(self, self._right)

    def evaluate(self, right: int, x: int, shamt: int) -> int:
        assert self._right is None or right == self._right, "a specialized block serves its own immediate alone"
        value, count = self._word(x), self._word(shamt)
        return (value.shift_right(count) if right else value.shift_left(count)).value


class _IShft(wiring.Component):
    right: Value
    """The port, or the constant the model is specialized to."""
    x: Signal
    shamt: Signal
    shft: Signal

    def __init__(self, model: IShft, right: bool | None) -> None:
        """`right` is the value the model holds the input at, or None where it is a port."""
        self._model = model
        super().__init__(model.signature)
        if right is not None:
            self.right = Const(right, 1)

    def elaborate(self, platform: object) -> Module:
        m = Module()
        model, w = self._model, self._model.wint
        field_width = (w - 1).bit_length()
        assert 0 < field_width < w
        span = 1 << field_width
        path = Datapath(m)

        right, x, shamt = path.register("in", model.stage_input, right=self.right, x=self.x, shamt=self.shamt)
        negative = shamt[w - 1]
        go_right = Signal(name="go_right")
        m.d.comb += go_right.eq(right ^ negative)

        # The magnitude of a negative amount is the two's-complement negation of its low bits, which inverts every bit
        # above the lowest set one; a prefix OR finds those bits without a carry chain.
        narrow = shamt[:field_width]
        prefix = narrow
        for level in range((field_width - 1).bit_length()):
            widened = Signal(field_width, name=f"prefix{level + 1}")
            m.d.comb += widened.eq(prefix | (prefix << (1 << level))[:field_width])
            prefix = widened
        amount = Signal(field_width, name="amount")
        m.d.comb += amount.eq(narrow ^ ((prefix << 1)[:field_width] & negative.replicate(field_width)))

        # The shifter answers every amount its count field can express, because the fill below reaches as far as the
        # field does; only a magnitude past the field itself needs the fill word chosen here.
        upper = shamt[field_width:]
        past_field = Signal(name="past_field")
        m.d.comb += past_field.eq(Mux(negative, ~upper.all() | ~narrow.any(), upper.any()))
        fill = Signal(name="fill")
        m.d.comb += fill.eq(go_right & x[w - 1])

        # A right shift is the left shift of the reversed word, reversed back, so that one shifter serves both.
        entering = Signal(w, name="entering")
        m.d.comb += entering.eq(Mux(go_right, x[::-1], x))
        spanned = Signal(w + span, name="spanned")
        m.d.comb += spanned.eq(Cat(fill.replicate(span), entering) << amount)
        leaving = spanned[span : span + w]
        shifted = Signal(signed(w), name="shifted")
        with m.Switch(Cat(go_right, past_field)):
            with m.Case("1-"):
                m.d.comb += shifted.eq(fill.replicate(w))
            with m.Case("00"):
                m.d.comb += shifted.eq(leaving)
            with m.Case("01"):
                m.d.comb += shifted.eq(leaving[::-1])

        (shft,) = path.register("out", model.stage_output, shft=shifted)
        m.d.comb += self.shft.eq(shft)
        assert path.depth == model.timing.latency
        return m


@dataclass(frozen=True, slots=True, kw_only=True)
class IShftOptions(BaseOperatorOptions):
    stage_input: int = 0
    stage_output: int = 0


class IShftOperator(HardwareOperator[IShft]):
    """
    One barrel shifter serving the left shift and the right shift, chosen per firing on `right`, at one latency. The
    shift is the raw bit shift either way, so nothing saturates.
    """

    __slots__ = ()
    name = "ishft"

    @classmethod
    def build(cls, fmt: IntFormat, options: IShftOptions) -> Self:
        block = IShft(wint=fmt.width, stage_input=options.stage_input, stage_output=options.stage_output)
        return cls(block, options.instances, int_ports(fmt, "x", "shamt"), int_ports(fmt, "shft"))

    def instance_block(self, primitives: frozenset[PooledPrimitive]) -> IShft:
        right = agreed(primitives, "right", bool)
        return self.block if right is None else self.block.specialized(right=right)


@dataclass(frozen=True, slots=True)
class IShlPrimitive(PooledPrimitive[IShft]):
    """
    An arithmetic shift by a signed amount, left when positive and right when negative. A left shift lets the high
    bits fall off the word: saturating instead is what a multiplication by a power of two does, on the multiplier.
    """

    operator: IShftOperator

    immediates: ClassVar[Mapping[str, int]] = {"right": False}

    def evaluate(self, *operands: ScalarValue, block: zkf.Block | None = None) -> IntValue:
        block = self.operator.block if block is None else block
        assert isinstance(block, IShft)
        x, shamt = map(int_operand, self._validated_operands(operands))
        return self._int_result(block.evaluate(self.immediates["right"], x, shamt))

    def render(self, *operands: str) -> str:
        a, b = operands
        return f"{a}<<{b}"


@dataclass(frozen=True, slots=True)
class IShrPrimitive(PooledPrimitive[IShft]):
    """
    The mirror of IShlPrimitive on the same operator: right when positive and left when negative, that left shift
    dropping what leaves the word as well.
    """

    operator: IShftOperator

    immediates: ClassVar[Mapping[str, int]] = {"right": True}

    def evaluate(self, *operands: ScalarValue, block: zkf.Block | None = None) -> IntValue:
        block = self.operator.block if block is None else block
        assert isinstance(block, IShft)
        x, shamt = map(int_operand, self._validated_operands(operands))
        return self._int_result(block.evaluate(self.immediates["right"], x, shamt))

    def render(self, *operands: str) -> str:
        a, b = operands
        return f"{a}>>{b}"
