from dataclasses import dataclass, field
from typing import ClassVar, NamedTuple, Self

import zkf
from amaranth.hdl import Cat, Const, Module, Mux, Signal, Value, signed
from amaranth.lib import wiring
from amaranth.lib.wiring import In, Out

from ..._value import IntValue, ScalarValue
from ..._type import IntFormat
from .._common import BaseOperatorOptions, HardwareOperator, PooledPrimitive, int_operand
from ._common import Datapath, IntBlock, int_ports


class IDivsResult(NamedTuple):
    quo: int
    rem: int
    div0: bool


@dataclass(frozen=True, kw_only=True, eq=False)
class IDivs(IntBlock):
    """
    Floor division and its remainder, Python's `//` and `%`, by restoring radix-4 steps of two quotient bits each.
    `MIN // -1` saturates to `(MAX, 0)`, and a division by zero raises `div0` and answers `MIN` for a negative numerator
    and `MAX` otherwise, keeping the numerator as the remainder.
    """

    operator = "idivs"
    keep_attrs: zkf.Attrs | None = field(default=None, compare=False)
    """On the four mutually exclusive results of the floor correction, which a synthesizer must not fold together."""

    @property
    def timing(self) -> zkf.Timing:
        return zkf.Timing(self.stage_input + 1 + _steps(self.wint) + self.stage_output, 1)

    @property
    def signature(self) -> wiring.Signature:
        word = signed(self.wint)
        return wiring.Signature({"num": In(word), "den": In(word), "quo": Out(word), "rem": Out(word), "div0": Out(1)})

    def component(self) -> _IDivs:
        return _IDivs(self)

    def evaluate(self, num: int, den: int) -> IDivsResult:
        numerator, denominator = self._word(num), self._word(den)
        quotient, remainder = numerator.divmod_floor(denominator)
        return IDivsResult(quotient.value, remainder.value, denominator.value == 0)


def _steps(w: int) -> int:
    return (w + 1) // 2


class _IDivs(wiring.Component):
    num: Signal
    den: Signal
    quo: Signal
    rem: Signal
    div0: Signal

    def __init__(self, model: IDivs) -> None:
        self._model = model
        super().__init__(model.signature)

    def elaborate(self, platform: object) -> Module:
        m = Module()
        model, w = self._model, self._model.wint
        fmt = IntFormat(w)
        steps = _steps(w)
        wpad = 2 * steps
        assert wpad in (w, w + 1)
        path = Datapath(m)

        num, den = path.register("in", model.stage_input, num=self.num, den=self.den)
        num_magnitude = Signal(w, name="num_magnitude")
        m.d.comb += num_magnitude.eq(Mux(num[w - 1], -num, num))
        den_magnitude = Signal(w, name="den_magnitude")
        m.d.comb += den_magnitude.eq(Mux(den[w - 1], -den, den))
        den_magnitude3 = Signal(w + 2, name="den_magnitude3")
        m.d.comb += den_magnitude3.eq(Cat(Const(0, 1), den_magnitude) + den_magnitude)

        # The work word is the partial remainder above a field that shifts left by two bits per step: the numerator's
        # magnitude leaves it at the top into the remainder, and the quotient digits enter it at the bottom.
        work, den_q, den3_q, num_negative, den_negative, div0, overflow = path.register(
            "step0",
            1,
            work=Cat(num_magnitude, Const(0, wpad)),
            den=den_magnitude,
            den3=den_magnitude3,
            num_negative=num[w - 1],
            den_negative=den[w - 1],
            div0=den == 0,
            overflow=(num == fmt.min) & (den == -1),
        )
        for step in range(1, steps + 1):
            remainder, digit = _radix4_step(m, f"step{step}", den_q, den3_q, work[wpad:], work[wpad - 2 : wpad])
            work, den_q, den3_q, num_negative, den_negative, div0, overflow = path.register(
                f"step{step}",
                1,
                work=Cat(digit, work[: wpad - 2], remainder),
                den=den_q,
                den3=den3_q,
                num_negative=num_negative,
                den_negative=den_negative,
                div0=div0,
                overflow=overflow,
            )
        assert len(work) == w + wpad

        quotient_magnitude = work[:w]
        remainder_magnitude = work[wpad:]
        signs_differ = Signal(name="signs_differ")
        m.d.comb += signs_differ.eq(num_negative ^ den_negative)
        remainder_nonzero = Signal(name="remainder_nonzero")
        m.d.comb += remainder_nonzero.eq(remainder_magnitude.any())

        # Unless kept apart, these four mutually exclusive results are folded by a synthesizer sharing arithmetic onto
        # one adder behind operand multiplexers: smaller, with a much longer path into the output registers.
        def kept(name: str, value: Value) -> Signal:
            result = Signal(w, name=name, attrs=dict(model.keep_attrs or {}))
            m.d.comb += result.eq(value)
            return result

        negative_quo = kept("negative_quo", -quotient_magnitude)
        rem_minus_den = kept("rem_minus_den", remainder_magnitude - den_q)
        den_minus_rem = kept("den_minus_rem", den_q - remainder_magnitude)
        negative_rem = kept("negative_rem", -remainder_magnitude)
        unequal_rem = Mux(den_negative, rem_minus_den, den_minus_rem)
        unequal_rem_or_zero = Mux(remainder_nonzero, unequal_rem, 0)
        equal_rem = Mux(den_negative, negative_rem, remainder_magnitude)
        floor_quo = Signal(w, name="floor_quo")
        with m.Switch(Cat(remainder_nonzero, signs_differ)):
            with m.Case("10"):
                m.d.comb += floor_quo.eq(negative_quo)
            with m.Case("11"):
                m.d.comb += floor_quo.eq(~quotient_magnitude)
            with m.Default():
                m.d.comb += floor_quo.eq(quotient_magnitude)

        quo = Signal(signed(w), name="quo_result")
        rem = Signal(signed(w), name="rem_result")
        with m.If(div0):
            m.d.comb += quo.eq(Mux(num_negative, Const(fmt.min, signed(w)), Const(fmt.max, signed(w))))
            m.d.comb += rem.eq(Mux(num_negative, -remainder_magnitude, remainder_magnitude))
        with m.Elif(overflow):
            m.d.comb += quo.eq(fmt.max)
            m.d.comb += rem.eq(0)
        with m.Else():
            m.d.comb += quo.eq(floor_quo)
            m.d.comb += rem.eq(Mux(signs_differ, unequal_rem_or_zero, equal_rem))

        quo, rem, div0 = path.register("out", model.stage_output, quo=quo, rem=rem, div0=div0)
        m.d.comb += [self.quo.eq(quo), self.rem.eq(rem), self.div0.eq(div0)]
        assert path.depth == model.timing.latency
        return m


def _radix4_step(
    m: Module, name: str, den: Value, den3: Value, partial_rem: Value, next_bits: Value
) -> tuple[Signal, Signal]:
    """
    The next two quotient bits by three trial subtractions of `den` times 1, 2 and 3 from the partial remainder with
    the next two numerator bits appended, the largest that does not borrow winning.
    """
    w = len(den)
    assert len(den3) == w + 2 and len(partial_rem) == w and len(next_bits) == 2
    candidate = Signal(w + 2, name=f"{name}_candidate")
    m.d.comb += candidate.eq(Cat(next_bits, partial_rem))
    differences = []
    for multiple, subtrahend in ((1, den), (2, Cat(Const(0, 1), den)), (3, den3)):
        difference = Signal(w + 3, name=f"{name}_diff{multiple}")
        m.d.comb += difference.eq(candidate - subtrahend)
        differences.append(difference)
    diff1, diff2, diff3 = differences
    rem_next = Signal(w, name=f"{name}_rem")
    digit = Signal(2, name=f"{name}_digit")
    with m.Switch(Cat(~diff1[w + 2], ~diff2[w + 2], ~diff3[w + 2])):
        with m.Case("1--"):
            m.d.comb += [rem_next.eq(diff3[:w]), digit.eq(3)]
        with m.Case("01-"):
            m.d.comb += [rem_next.eq(diff2[:w]), digit.eq(2)]
        with m.Case("001"):
            m.d.comb += [rem_next.eq(diff1[:w]), digit.eq(1)]
        with m.Default():
            m.d.comb += [rem_next.eq(candidate[:w]), digit.eq(0)]
    return rem_next, digit


@dataclass(frozen=True, slots=True, kw_only=True)
class IDivsOptions(BaseOperatorOptions):
    stage_input: int = 0
    stage_output: int = 0
    keep_attrs: zkf.Attrs = field(default_factory=dict)
    """
    On the floor correction's four candidate results, which a synthesizer sharing arithmetic would otherwise fold onto
    one adder behind operand multiplexers: smaller, with a much longer path into the output registers.
    """


class IDivOperator(HardwareOperator[IDivs]):
    """Floor division, because that is what Python's `//` and `%` mean and HIR has no other division."""

    __slots__ = ()
    name = "idivs"
    error_ports = ("div0",)

    @classmethod
    def build(cls, fmt: IntFormat, options: IDivsOptions) -> Self:
        block = IDivs(
            wint=fmt.width,
            stage_input=options.stage_input,
            stage_output=options.stage_output,
            keep_attrs=options.keep_attrs,
        )
        return cls(block, options.instances, int_ports(fmt, "num", "den"), int_ports(fmt, "quo", "rem"))


@dataclass(frozen=True, slots=True)
class IFloorDivPrimitive(PooledPrimitive[IDivs]):
    """The floor quotient, which the divider answers beside the remainder, so the two are one firing."""

    operator: IDivOperator
    output_name: ClassVar[str] = "quo"

    def evaluate(self, *operands: ScalarValue, block: zkf.Block | None = None) -> IntValue:
        block = self.operator.block if block is None else block
        assert isinstance(block, IDivs)
        a, b = map(int_operand, self._validated_operands(operands))
        return self._int_result(block.evaluate(a, b).quo)

    def render(self, *operands: str) -> str:
        a, b = operands
        return f"{a}//{b}"


@dataclass(frozen=True, slots=True)
class IModPrimitive(PooledPrimitive[IDivs]):
    """The remainder of the floor quotient."""

    operator: IDivOperator
    output_name: ClassVar[str] = "rem"

    def evaluate(self, *operands: ScalarValue, block: zkf.Block | None = None) -> IntValue:
        block = self.operator.block if block is None else block
        assert isinstance(block, IDivs)
        a, b = map(int_operand, self._validated_operands(operands))
        return self._int_result(block.evaluate(a, b).rem)

    def render(self, *operands: str) -> str:
        a, b = operands
        return f"{a}%{b}"
