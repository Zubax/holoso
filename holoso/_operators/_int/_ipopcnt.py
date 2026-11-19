from dataclasses import dataclass
from typing import Self

import zkf
from amaranth.hdl import Cat, Const, Module, Mux, Signal, Value, signed
from amaranth.lib import wiring
from amaranth.lib.wiring import In, Out

from ..._value import IntValue, ScalarValue
from ..._type import IntFormat
from .._common import BaseOperatorOptions, HardwareOperator, PooledPrimitive, int_operand
from ._common import Datapath, IntBlock, int_ports


@dataclass(frozen=True, kw_only=True, eq=False)
class IPopcnt(IntBlock):
    """
    The population count of the magnitude, as Python's `abs(x).bit_count()`, so a negative operand counts the ones of
    `-x`. The negation that overflows a signed word is exactly the magnitude `2**(wint-1)` read unsigned, so nothing
    saturates and the count never reaches `wint`: the count is only as wide as that needs, `(wint - 1).bit_length()`
    bits, through its registers, and `y` widens it to the word with a zero fill.
    """

    operator = "ipopcnt"

    @property
    def timing(self) -> zkf.Timing:
        return zkf.Timing(self.stage_input + self.stage_output, 1)

    @property
    def signature(self) -> wiring.Signature:
        word = signed(self.wint)
        return wiring.Signature({"x": In(word), "y": Out(word)})

    def component(self) -> _IPopcnt:
        return _IPopcnt(self)

    def evaluate(self, x: int) -> int:
        count = abs(self._word(x).value).bit_count()
        assert 0 <= count < self.wint
        return count


def _count_width(w: int) -> int:
    return (w - 1).bit_length()


# Negating first would put a carry chain as long as the word in series with the reduction and misses timing by a wide
# margin, so the sign is folded into the operand instead: -x = ~x + 1 agrees with ~x everywhere above the lowest set bit
# of x, and below it the increment clears a run of tz(x) ones and sets one, tz being the trailing-zero count; hence
# popcount(-x) = popcount(~x) + 1 - tz(x). Counting x ^ sign costs nothing (the sign is just another item), the +1 rides
# in as a constant item, and the trailing-zero count comes from its own shallow priority tree beside the reduction.
#
# The reduction is built of explicit counters and carry-save levels rather than an addition tree because a synthesizer
# flattens a chain of additions into one carry-save tree that consumes every operand in its first compression level.
# That schedule ignores arrival times and would stack the whole reduction underneath the trailing-zero cone. Here the
# cone lands beside a carry-save pair instead, and only the closing addition sees it. A level counts up to seven values
# per column into three weighted results, and three values collapse to a carry-save pair, so the value count runs
# 7 -> 3 -> 2 at common widths: one logic level each. None of that is paid for in fabric, which is the counterintuitive
# part: measured against the addition tree it replaces, this network holds the flip-flop count exactly and spends fewer
# 4-input lookup tables at common widths, because explicit counters pack into them where a generic multiply-accumulate
# expansion does not. A synthesizer that retimes will still grow its own registers here, but that is the speed being
# taken, not the structure costing it.
class _IPopcnt(wiring.Component):
    x: Signal
    y: Signal

    def __init__(self, model: IPopcnt) -> None:
        self._model = model
        super().__init__(model.signature)

    def elaborate(self, platform: object) -> Module:
        m = Module()
        model, w = self._model, self._model.wint
        wy = _count_width(w)
        assert (1 << (wy - 1)) <= w - 1 < (1 << wy)
        path = Datapath(m)

        (x,) = path.register("in", model.stage_input, x=self.x)
        sign = x[w - 1]

        # The items to count: the magnitude bits below the sign (the topmost one is zero after folding), the sign, and
        # the constant that completes the two's complement of the inverted trailing-zero count.
        items = [x[k] ^ sign for k in range(w - 1)] + [sign, Const(1, 1)]
        items += [Const(0, 1)] * (-len(items) % 7)
        values = []
        for k in range(len(items) // 7):
            leaf = Signal(wy, name=f"value0_{k}")
            m.d.comb += leaf.eq(_count7(items[7 * k : 7 * k + 7]))
            values.append(leaf)
        level = 0
        while len(values) > 2:
            level += 1
            values = _compress(m, level, values, wy)
        pair_hi = values[1] if len(values) > 1 else Const(0, wy)

        tz_value = _trailing_zeros(m, x, wy)
        count = Signal(wy, name="count")
        m.d.comb += count.eq(values[0] + pair_hi + ~(tz_value & sign.replicate(wy)))

        (y,) = path.register("out", model.stage_output, y=count)
        m.d.comb += self.y.eq(y)
        assert path.depth == model.timing.latency
        return m


def _compress(m: Module, level: int, values: list[Signal], wy: int) -> list[Signal]:
    """One level: seven-input column counters into three weighted results each, or a carry-save adder of three."""
    assert len(values) > 2
    out = []
    if len(values) == 3:
        a, b, c = values
        for index, value in enumerate((a ^ b ^ c, ((a & b) | (a & c) | (b & c)) << 1)):
            node = Signal(wy, name=f"value{level}_{index}")
            m.d.comb += node.eq(value)
            out.append(node)
        return out
    for chunk in range(-(-len(values) // 7)):
        group = values[7 * chunk : 7 * chunk + 7]
        columns = [_count7([v[bit] for v in group] + [Const(0, 1)] * (7 - len(group))) for bit in range(wy)]
        for weight in range(3):
            node = Signal(wy, name=f"value{level}_{3 * chunk + weight}")
            m.d.comb += node.eq(Cat(Const(0, weight), *(columns[bit][weight] for bit in range(wy - weight))))
            out.append(node)
    return out


def _count7(v: list[Value]) -> Value:
    """A seven-input counter, written as gates so that it stays one logic level rather than a chain."""
    assert len(v) == 7

    def majority(a: Value, b: Value, c: Value) -> Value:
        return (a & b) | (a & c) | (b & c)

    s0, k0 = v[0] ^ v[1] ^ v[2], majority(v[0], v[1], v[2])
    s1, k1 = v[3] ^ v[4] ^ v[5], majority(v[3], v[4], v[5])
    s2, k2 = s0 ^ s1 ^ v[6], majority(s0, s1, v[6])
    return Cat(s2, k0 ^ k1 ^ k2, (k0 & k1) | ((k0 ^ k1) & k2))


def _trailing_zeros(m: Module, x: Value, wy: int) -> Value:
    """
    The trailing-zero count of a nonzero `x` by a four-way priority tree over its nibbles, padded with zero nibbles to a
    power of four; a zero nibble is resolved one level up.
    """
    nibbles = Cat(x, Const(0, 4))
    count = (len(x) + 3) // 4
    levels = ((count - 1).bit_length() + 1) // 2
    nodes: list[tuple[Value, Value]] = []
    for k in range(1 << (2 * levels)):
        if k < count:
            nibble = nibbles[4 * k : 4 * k + 4]
            zero = Signal(name=f"tz0_{k}_zero")
            m.d.comb += zero.eq(~nibble.any())
            value = Signal(wy, name=f"tz0_{k}")
            m.d.comb += value.eq(Mux(nibble[0], 0, Mux(nibble[1], 1, Mux(nibble[2], 2, 3))))
            nodes.append((zero, value))
        else:
            nodes.append((Const(1, 1), Const(0, wy)))
    for level in range(1, levels + 1):
        span = 1 << (2 * level)  # bits spanned by one child, so a digit is a set bit
        parents = []
        for p in range(len(nodes) // 4):
            (z0, v0), (z1, v1), (z2, v2), (z3, v3) = nodes[4 * p : 4 * p + 4]
            zero = Signal(name=f"tz{level}_{p}_zero")
            m.d.comb += zero.eq(z0 & z1 & z2 & z3)
            value = Signal(wy, name=f"tz{level}_{p}")
            m.d.comb += value.eq(Mux(~z0, v0, Mux(~z1, v1 | span, Mux(~z2, v2 | (2 * span), v3 | (3 * span)))))
            parents.append((zero, value))
        nodes = parents
    assert len(nodes) == 1
    return nodes[0][1]


@dataclass(frozen=True, slots=True, kw_only=True)
class IPopcntOptions(BaseOperatorOptions):
    stage_input: int = 0
    stage_output: int = 0


class IPopcntOperator(HardwareOperator[IPopcnt]):
    __slots__ = ()
    name = "ipopcnt"

    @classmethod
    def build(cls, fmt: IntFormat, options: IPopcntOptions) -> Self:
        block = IPopcnt(wint=fmt.width, stage_input=options.stage_input, stage_output=options.stage_output)
        return cls(block, options.instances, int_ports(fmt, "x"), int_ports(fmt, "y"))


@dataclass(frozen=True, slots=True)
class IPopcntPrimitive(PooledPrimitive[IPopcnt]):
    """
    The population count of the magnitude, as Python's `int.bit_count()`, so a negative operand counts the ones of `-x`.
    The negation that overflows a signed word is exactly the magnitude `2**(W-1)` read unsigned, so unlike
    IAbsPrimitive nothing saturates and the count never reaches the width.
    """

    operator: IPopcntOperator

    def evaluate(self, *operands: ScalarValue, block: zkf.Block | None = None) -> IntValue:
        block = self.operator.block if block is None else block
        assert isinstance(block, IPopcnt)
        (a,) = map(int_operand, self._validated_operands(operands))
        return self._int_result(block.evaluate(a))

    def render(self, *operands: str) -> str:
        (a,) = operands
        return f"popcnt({a})"
