from dataclasses import dataclass
from typing import ClassVar, Self

import zkf
from amaranth.hdl import Const, Module, Mux, Signal, signed
from amaranth.lib import wiring
from amaranth.lib.wiring import In, Out

from ..._value import IntValue, ScalarValue
from ..._type import IntFormat
from .._common import BaseOperatorOptions, HardwareOperator, PooledPrimitive, int_operand
from ._common import Datapath, IntBlock, int_ports


@dataclass(frozen=True, kw_only=True, eq=False)
class IMuls(IntBlock):
    """`y = a * b`, saturating."""

    operator = "imuls"
    wmultiplier: int
    """The operand width of the target's multipliers, as `zkf.Pmul` takes it."""
    stage_core_mul: int = 0
    """The register stages of the embedded `zkf.Pmul`, taken in its order (see `zkf.Mul.stage_core_mul`)."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.stage_core_mul < 0:
            raise ValueError(f"stage_core_mul={self.stage_core_mul} is negative")

    @property
    def timing(self) -> zkf.Timing:
        return zkf.Timing(self.stage_input + self.stage_core_mul + self.stage_output, 1)

    @property
    def signature(self) -> wiring.Signature:
        word = signed(self.wint)
        return wiring.Signature({"a": In(word), "b": In(word), "y": Out(word)})

    def component(self) -> _IMuls:
        return _IMuls(self)

    def evaluate(self, a: int, b: int) -> int:
        return (self._word(a) * self._word(b)).value


class _IMuls(wiring.Component):
    a: Signal
    b: Signal
    y: Signal

    def __init__(self, model: IMuls) -> None:
        self._model = model
        super().__init__(model.signature)

    def elaborate(self, platform: object) -> Module:
        m = Module()
        model, w = self._model, self._model.wint
        fmt = IntFormat(w)
        path = Datapath(m)

        a, b = path.register("in", model.stage_input, a=self.a, b=self.b)
        # The valid flags of the product drive nothing: every register here runs in every cycle.
        m.submodules.pmul = pmul = zkf.Pmul(
            w, w, signed=True, wmultiplier=model.wmultiplier, stages=model.stage_core_mul
        )
        m.d.comb += [pmul.a.eq(a), pmul.b.eq(b)]
        path.depth += pmul.latency
        product = pmul.p
        assert len(product) == 2 * w
        overflow = Signal(name="overflow")
        m.d.comb += overflow.eq((product[w:] ^ product[w - 1].replicate(w)).any())
        saturated = Signal(signed(w), name="saturated")
        rail = Mux(product[2 * w - 1], Const(fmt.min, signed(w)), Const(fmt.max, signed(w)))
        m.d.comb += saturated.eq(Mux(overflow, rail, product[:w].as_signed()))

        (y,) = path.register("out", model.stage_output, y=saturated)
        m.d.comb += self.y.eq(y)
        assert path.depth == model.timing.latency
        return m


@dataclass(frozen=True, slots=True, kw_only=True)
class IMulsOptions(BaseOperatorOptions):
    stage_input: int = 0
    stage_core_mul: int = 0
    """The register stages of the product, which `wmultiplier` cuts into a grid of slices as ZKF's products are."""
    stage_output: int = 0


class IMulOperator(HardwareOperator[IMuls]):
    __slots__ = ()
    name = "imuls"

    @classmethod
    def build(cls, fmt: IntFormat, options: IMulsOptions, wmultiplier: int) -> Self:
        block = IMuls(
            wint=fmt.width,
            wmultiplier=wmultiplier,
            stage_input=options.stage_input,
            stage_core_mul=options.stage_core_mul,
            stage_output=options.stage_output,
        )
        return cls(block, options.instances, int_ports(fmt, "a", "b"), int_ports(fmt, "y"))


@dataclass(frozen=True, slots=True)
class IMulPrimitive(PooledPrimitive[IMuls]):
    operator: IMulOperator
    _commutative: ClassVar[bool] = True

    def evaluate(self, *operands: ScalarValue, block: zkf.Block | None = None) -> IntValue:
        block = self.operator.block if block is None else block
        assert isinstance(block, IMuls)
        a, b = map(int_operand, self._validated_operands(operands))
        return self._int_result(block.evaluate(a, b))

    def render(self, *operands: str) -> str:
        a, b = operands
        return f"{a}×{b}"
