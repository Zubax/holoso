from dataclasses import dataclass, replace
from collections.abc import Mapping
from typing import ClassVar, NamedTuple, Self

import zkf
from amaranth.hdl import Cat, Const, Module, Mux, Signal, Value, signed
from amaranth.lib import wiring
from amaranth.lib.wiring import In, Out

from ..._value import IntValue, ScalarValue
from ..._type import BoolType, IntFormat, IntType
from .._common import (
    BaseOperatorOptions,
    HardwareOperator,
    ComparatorPrimitive,
    PooledPrimitive,
    agreed,
    int_operand,
)
from ._common import Datapath, IntBlock, int_ports


class IAddsResult(NamedTuple):
    y: int
    a_gt_b: bool
    a_eq_b: bool
    a_lt_b: bool


@dataclass(frozen=True, kw_only=True, eq=False)
class IAdds(IntBlock):
    """
    `y = a + b` when `sub` is low, else `y = a - b`, saturating. A subtraction also orders `a` against `b` on the flags
    as a side effect; an addition leaves them meaningless.
    """

    operator = "iadds"
    fast: bool = False
    """Trades area for shorter paths: the equality comes from a comparator beside the adder, not the zero difference."""
    _sub: bool | None = None

    def specialized(self, *, sub: bool) -> IAdds:
        """This configuration built for the addition alone or the subtraction alone."""
        return replace(self, _sub=sub)

    @property
    def timing(self) -> zkf.Timing:
        return zkf.Timing(self.stage_input + self.stage_output, 1)

    @property
    def signature(self) -> wiring.Signature:
        word = signed(self.wint)
        return wiring.Signature(
            {
                **({"sub": In(1)} if self._sub is None else {}),
                "a": In(word),
                "b": In(word),
                "y": Out(word),
                "a_gt_b": Out(1),
                "a_eq_b": Out(1),
                "a_lt_b": Out(1),
            }
        )

    def component(self) -> _IAdds:
        return _IAdds(self, self._sub)

    def evaluate(self, sub: int, a: int, b: int) -> IAddsResult:
        assert self._sub is None or sub == self._sub, "a specialized block serves its own immediate alone"
        left, right = self._word(a), self._word(b)
        if not sub:
            return IAddsResult((left + right).value, False, False, False)
        ordering = left.compare(right)
        return IAddsResult((left - right).value, ordering > 0, ordering == 0, ordering < 0)


class _IAdds(wiring.Component):
    sub: Value
    """The port, or the constant the model is specialized to."""
    a: Signal
    b: Signal
    y: Signal
    a_gt_b: Signal
    a_eq_b: Signal
    a_lt_b: Signal

    def __init__(self, model: IAdds, sub: bool | None) -> None:
        """`sub` is the value the model holds the input at, or None where it is a port."""
        self._model = model
        super().__init__(model.signature)
        if sub is not None:
            self.sub = Const(sub, 1)

    def elaborate(self, platform: object) -> Module:
        m = Module()
        model, w = self._model, self._model.wint
        fmt = IntFormat(w)
        path = Datapath(m)

        sub, a, b = path.register("in", model.stage_input, sub=self.sub, a=self.a, b=self.b)
        rhs = Signal(w, name="rhs")
        m.d.comb += rhs.eq(b ^ sub.replicate(w))
        # `{0, a} + {0, rhs} + sub` as one addition, the carry-in riding in the low bit of both operands, so that every
        # tool builds one carry chain of it rather than a second adder for the increment.
        total = Cat(sub, a) + Cat(sub, rhs)
        assert len(total) == w + 2
        sum_ext = Signal(w + 1, name="sum_ext")
        m.d.comb += sum_ext.eq(total[1:])
        carry_into_sign = sum_ext[w - 1] ^ a[w - 1] ^ rhs[w - 1]
        overflow = Signal(name="overflow")
        m.d.comb += overflow.eq(carry_into_sign ^ sum_ext[w])
        less = Signal(name="less")  # the sign of the unbounded difference
        m.d.comb += less.eq(sum_ext[w - 1] ^ overflow)
        equal = Signal(name="equal")
        m.d.comb += equal.eq((a == b) if model.fast else ~sum_ext[:w].any())
        saturated = Signal(signed(w), name="saturated")
        rail = Mux(a[w - 1], Const(fmt.min, signed(w)), Const(fmt.max, signed(w)))
        m.d.comb += saturated.eq(Mux(overflow, rail, sum_ext[:w].as_signed()))

        y, gt, eq, lt = path.register(
            "out", model.stage_output, y=saturated, a_gt_b=~less & ~equal, a_eq_b=equal, a_lt_b=less
        )
        m.d.comb += [self.y.eq(y), self.a_gt_b.eq(gt), self.a_eq_b.eq(eq), self.a_lt_b.eq(lt)]
        assert path.depth == model.timing.latency
        return m


@dataclass(frozen=True, slots=True, kw_only=True)
class IAddsOptions(BaseOperatorOptions):
    """
    The adder also runs every integer subtraction, negation and comparison, one firing per cycle, so a kernel issuing
    several of them at once shortens by raising `instances`.
    """

    stage_input: int = 0
    stage_output: int = 0
    fast: bool = False
    """Trades area for shorter combinational paths."""


class IAddOperator(HardwareOperator[IAdds]):
    """
    The saturating adder, which also subtracts, chosen per firing on `sub`, at one latency. A subtraction orders `a`
    against `b` on the flags as a side effect, so the drive answers them beside the difference. Additions and
    subtractions contend for the one adder.
    """

    __slots__ = ()
    name = "iadds"

    @classmethod
    def build(cls, fmt: IntFormat, options: IAddsOptions) -> Self:
        block = IAdds(
            wint=fmt.width,
            stage_input=options.stage_input,
            stage_output=options.stage_output,
            fast=options.fast,
        )
        outputs = {"y": IntType(fmt), "a_gt_b": BoolType(), "a_eq_b": BoolType(), "a_lt_b": BoolType()}
        return cls(block, options.instances, int_ports(fmt, "a", "b"), outputs)

    def instance_block(self, primitives: frozenset[PooledPrimitive]) -> IAdds:
        sub = agreed(primitives, "sub", bool)
        return self.block if sub is None else self.block.specialized(sub=sub)


@dataclass(frozen=True, slots=True)
class IAddPrimitive(PooledPrimitive[IAdds]):
    """The saturating sum alone: an addition leaves the order flags meaningless."""

    operator: IAddOperator
    _commutative: ClassVar[bool] = True
    immediates: ClassVar[Mapping[str, int]] = {"sub": False}
    output_name: ClassVar[str] = "y"

    def evaluate(self, *operands: ScalarValue, block: zkf.Block | None = None) -> IntValue:
        block = self.operator.block if block is None else block
        assert isinstance(block, IAdds)
        a, b = map(int_operand, self._validated_operands(operands))
        return self._int_result(block.evaluate(self.immediates["sub"], a, b).y)

    def render(self, *operands: str) -> str:
        a, b = operands
        return f"{a}+{b}"


@dataclass(frozen=True, slots=True)
class ISubPrimitive(PooledPrimitive[IAdds]):
    """
    The saturating difference. Also serves negation as `0-x`: there is no negation module, and the subtraction
    saturates `-MIN` correctly. It does not survive exchanging the operands.
    """

    operator: IAddOperator
    immediates: ClassVar[Mapping[str, int]] = {"sub": True}
    output_name: ClassVar[str] = "y"

    def evaluate(self, *operands: ScalarValue, block: zkf.Block | None = None) -> IntValue:
        block = self.operator.block if block is None else block
        assert isinstance(block, IAdds)
        a, b = map(int_operand, self._validated_operands(operands))
        return self._int_result(block.evaluate(self.immediates["sub"], a, b).y)

    def render(self, *operands: str) -> str:
        a, b = operands
        return f"{a}-{b}"


@dataclass(frozen=True, slots=True)
class ICmpPrimitive(ComparatorPrimitive[IAdds]):
    """
    An ordering of the operands of a subtraction, which the adder answers beside the difference, so the relations
    over one operand pair ride the subtraction of that pair, and a comparison alone is the subtraction read through
    its flags. Two's complement is totally ordered.
    """

    operator: IAddOperator
    immediates: ClassVar[Mapping[str, int]] = {"sub": True}

    def evaluate(self, *operands: ScalarValue, block: zkf.Block | None = None) -> bool:
        block = self.operator.block if block is None else block
        assert isinstance(block, IAdds)
        a, b = map(int_operand, self._validated_operands(operands))
        result = block.evaluate(self.immediates["sub"], a, b)
        return self._flag(result.a_gt_b, result.a_eq_b, result.a_lt_b)
