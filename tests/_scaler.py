"""
A test-only operator whose immediates have two shapes -- a one-bit `mode` and a signed multi-bit `gain` -- and whose
block answers on two outputs, so specialization is held to its contract beyond the shapes the shipped operators happen
to have. Module-level, so a numerical model embedding it pickles into a cosimulation bench.
"""

from dataclasses import dataclass, replace
from typing import ClassVar, Self

import zkf
from amaranth.hdl import Const, Module, Mux, Signal, Value, signed
from amaranth.lib import wiring
from amaranth.lib.wiring import In, Out

from holoso._operators import HardwareOperator, PooledPrimitive
from holoso._operators._common import agreed, int_operand
from holoso._operators._int._common import Datapath, IntBlock, int_ports
from holoso._type import IntFormat
from holoso._value import IntValue, ScalarValue

GAIN_WIDTH = 4


@dataclass(frozen=True, kw_only=True, eq=False)
class Scaler(IntBlock):
    """
    `y = a + gain` with `mode` low, else `y = a - gain`, and `z = a * gain` whatever the mode, both wrapping. The mode
    high at the most negative gain is reserved: no block is built for it, so nothing may enumerate the combinations.
    """

    operator = "scaler"
    _mode: bool | None = None
    _gain: int | None = None

    def specialized(self, *, mode: bool | None, gain: int | None) -> Scaler:
        if mode and gain == -(1 << (GAIN_WIDTH - 1)):
            raise ValueError("the mode high at the most negative gain is reserved")
        return replace(self, _mode=mode, _gain=gain)

    @property
    def timing(self) -> zkf.Timing:
        return zkf.Timing(self.stage_input + self.stage_output, 1)

    @property
    def signature(self) -> wiring.Signature:
        word = signed(self.wint)
        return wiring.Signature(
            {
                **({"mode": In(1)} if self._mode is None else {}),
                **({"gain": In(signed(GAIN_WIDTH))} if self._gain is None else {}),
                "a": In(word),
                "y": Out(word),
                "z": Out(word),
            }
        )

    def component(self) -> _Scaler:
        return _Scaler(self, self._mode, self._gain)

    def evaluate(self, mode: int, gain: int, a: int) -> tuple[int, int]:
        assert self._mode is None or mode == self._mode
        assert self._gain is None or gain == self._gain
        return self._wrapped(a - gain if mode else a + gain), self._wrapped(a * gain)

    def _wrapped(self, value: int) -> int:
        half = 1 << (self.wint - 1)
        return (value + half) % (2 * half) - half


class _Scaler(wiring.Component):  # type: ignore[misc]
    mode: Value
    gain: Value
    a: Signal
    y: Signal
    z: Signal

    def __init__(self, model: Scaler, mode: bool | None, gain: int | None) -> None:
        self._model = model
        super().__init__(model.signature)
        if mode is not None:
            self.mode = Const(mode, 1)
        if gain is not None:
            self.gain = Const(gain, signed(GAIN_WIDTH))

    def elaborate(self, platform: object) -> Module:
        m = Module()
        model, w = self._model, self._model.wint
        path = Datapath(m)
        mode, gain, a = path.register("in", model.stage_input, mode=self.mode, gain=self.gain, a=self.a)
        y, z = Signal(signed(w), name="y_d"), Signal(signed(w), name="z_d")
        m.d.comb += [y.eq(Mux(mode, a - gain, a + gain)[:w]), z.eq((a * gain)[:w])]
        y, z = path.register("out", model.stage_output, y=y, z=z)
        m.d.comb += [self.y.eq(y), self.z.eq(z)]
        assert path.depth == model.timing.latency
        return m


class ScalerOperator(HardwareOperator[Scaler]):
    __slots__ = ()
    name = "scaler"

    @classmethod
    def build(cls, fmt: IntFormat, stage_input: int = 0, stage_output: int = 0) -> Self:
        block = Scaler(wint=fmt.width, stage_input=stage_input, stage_output=stage_output)
        return cls(block, 1, int_ports(fmt, "a"), int_ports(fmt, "y", "z"))

    def instance_block(self, primitives: frozenset[PooledPrimitive]) -> Scaler:
        return self.block.specialized(mode=agreed(primitives, "mode", bool), gain=agreed(primitives, "gain", int))


@dataclass(frozen=True, slots=True)
class OffsetPrimitive(PooledPrimitive[Scaler]):
    """`a + gain`, or `a - gain` with the mode high: the block's `y`."""

    operator: ScalerOperator
    mode: bool
    gain: int
    output_name: ClassVar[str] = "y"

    @property
    def immediates(self) -> dict[str, int]:
        return {"mode": self.mode, "gain": self.gain}

    def evaluate(self, *operands: ScalarValue, block: zkf.Block | None = None) -> IntValue:
        block = self.operator.block if block is None else block
        assert isinstance(block, Scaler)
        (a,) = map(int_operand, self._validated_operands(operands))
        return self._int_result(block.evaluate(self.mode, self.gain, a)[0])

    def render(self, *operands: str) -> str:
        (a,) = operands
        return f"{a}{'-' if self.mode else '+'}{self.gain}"


@dataclass(frozen=True, slots=True)
class ProductPrimitive(PooledPrimitive[Scaler]):
    """`a * gain`, the block's `z`, which answers at either mode, so the primitive names the one it drives."""

    operator: ScalerOperator
    mode: bool
    gain: int
    output_name: ClassVar[str] = "z"

    @property
    def immediates(self) -> dict[str, int]:
        return {"mode": self.mode, "gain": self.gain}

    def evaluate(self, *operands: ScalarValue, block: zkf.Block | None = None) -> IntValue:
        block = self.operator.block if block is None else block
        assert isinstance(block, Scaler)
        (a,) = map(int_operand, self._validated_operands(operands))
        return self._int_result(block.evaluate(self.mode, self.gain, a)[1])

    def render(self, *operands: str) -> str:
        (a,) = operands
        return f"{a}×{self.gain}"
