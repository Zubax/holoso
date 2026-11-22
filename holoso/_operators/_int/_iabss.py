from dataclasses import dataclass
from typing import Self

import zkf
from amaranth.hdl import Const, Module, Mux, Signal, signed
from amaranth.lib import wiring
from amaranth.lib.wiring import In, Out

from ..._value import IntValue, ScalarValue
from ..._type import IntFormat
from .._common import BaseOperatorOptions, HardwareOperator, PooledPrimitive, int_operand
from ._common import Datapath, IntBlock, int_ports


@dataclass(frozen=True, kw_only=True, eq=False)
class IAbss(IntBlock):
    """`y = |x|`, saturating: `|MIN|` is `MAX`."""

    operator = "iabss"

    @property
    def timing(self) -> zkf.Timing:
        return zkf.Timing(self.stage_input + self.stage_output, 1)

    @property
    def signature(self) -> wiring.Signature:
        word = signed(self.wint)
        return wiring.Signature({"x": In(word), "y": Out(word)})

    def component(self) -> _IAbss:
        return _IAbss(self)

    def evaluate(self, x: int) -> int:
        return abs(self._word(x)).value


class _IAbss(wiring.Component):
    x: Signal
    y: Signal

    def __init__(self, model: IAbss) -> None:
        self._model = model
        super().__init__(model.signature)

    def elaborate(self, platform: object) -> Module:
        m = Module()
        model, w = self._model, self._model.wint
        fmt = IntFormat(w)
        path = Datapath(m)

        (x,) = path.register("in", model.stage_input, x=self.x)
        negated = Signal(signed(w), name="negated")
        m.d.comb += negated.eq(-x)
        clamp = Signal(name="clamp")
        m.d.comb += clamp.eq(x == fmt.min)
        result = Signal(signed(w), name="result")
        m.d.comb += result.eq(Mux(x[w - 1], Mux(clamp, Const(fmt.max, signed(w)), negated), x))

        (y,) = path.register("out", model.stage_output, y=result)
        m.d.comb += self.y.eq(y)
        assert path.depth == model.timing.latency
        return m


@dataclass(frozen=True, slots=True, kw_only=True)
class IAbssOptions(BaseOperatorOptions):
    stage_input: int = 0
    stage_output: int = 0


class IAbsOperator(HardwareOperator[IAbss]):
    __slots__ = ()
    name = "iabss"

    @classmethod
    def build(cls, fmt: IntFormat, options: IAbssOptions) -> Self:
        block = IAbss(wint=fmt.width, stage_input=options.stage_input, stage_output=options.stage_output)
        return cls(block, options.instances, int_ports(fmt, "x"), int_ports(fmt, "y"))


@dataclass(frozen=True, slots=True)
class IAbsPrimitive(PooledPrimitive[IAbss]):
    operator: IAbsOperator

    def evaluate(self, *operands: ScalarValue, block: zkf.Block | None = None) -> IntValue:
        block = self.operator.block if block is None else block
        assert isinstance(block, IAbss)
        (a,) = map(int_operand, self._validated_operands(operands))
        return self._int_result(block.evaluate(a))

    def render(self, *operands: str) -> str:
        (a,) = operands
        return f"|{a}|"
