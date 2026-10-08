"""
Several primitives on one operator, in modes of different latency, initiation interval and operand prefix: the timing is
per firing, a firing holds its instance for its own mode's initiation interval, and a port a mode does not read costs
no mux arm. The test-only operator has no RTL, so the numerical model against the MIR interpreter is the oracle here.
"""

import re
from dataclasses import dataclass
from typing import Self

import pytest

from holoso import FloatFormat, IAddOptions
from holoso._backend.verilog import generate as generate_verilog
from holoso._lir import OpWriteSource, RegallocTuning, landing_cycle, read_sources_per_port, write_events
from holoso._backend.verilog._microcode import build_microcode, f_mode, read_codebook, write_codebook
from holoso._mir import Mir
from holoso._mir._interpret import MirInterpreter
from holoso._mir._ir import MirBuilder
from holoso._operators import (
    AddMode,
    HardwareOperator,
    IAddOperator,
    ICmpPrimitive,
    IntIdentity,
    ModePort,
    OperatorMode,
    OperatorPort,
    PooledPrimitive,
)
from holoso._type import IntType
from holoso._value import IntValue, ScalarValue

from ._modelref import build_lir, build_model, default_ifmt

FMT = FloatFormat(6, 18)
_IFMT = default_ifmt(FMT)
_SLOW, _FAST = 0, 1  # codes on the modal operator's `fast` mode port


class _ModalOperator(HardwareOperator):
    """
    An operator serving two primitives of different timing and operand prefix: SLOW reads `a` alone, FAST reads both
    ports and is fully pipelined. Sharing FAST's output port, SLOW must hold its instance for its latency or one step
    past it, as a CORDIC core does; on a port of its own it may be pipelined too, its results overtaken by FAST's.
    """

    __slots__ = ()
    name = "modal"
    mode_port = ModePort("fast", 1)

    @classmethod
    def build(cls, instances: int, slow_interval: int, split: bool) -> Self:
        ints = IntType(_IFMT)
        outputs = (OperatorPort("y", ints), OperatorPort("z", ints)) if split else (OperatorPort("y", ints),)
        slow = OperatorMode(_SLOW, "LATENCY_SLOW", slow_interval, 1, (1,) if split else (0,))
        fast = OperatorMode(_FAST, "LATENCY_FAST", 1, 2, (0,))
        # MODE 2 serves both modes; an instance running only one is elaborated for it alone.
        return cls(
            {"W": _IFMT.width, "MODE": 2, "LATENCY_SLOW": 4, "LATENCY_FAST": 1},
            instances,
            (OperatorPort("a", ints), OperatorPort("b", ints)),
            outputs,
            (slow, fast),
            {
                slow: {"W": _IFMT.width, "MODE": _SLOW, "LATENCY_SLOW": 4},
                fast: {"W": _IFMT.width, "MODE": _FAST, "LATENCY_FAST": 1},
            },
        )

    @property
    def slow(self) -> OperatorMode:
        return self.mode_of(_SLOW)

    @property
    def fast(self) -> OperatorMode:
        return self.mode_of(_FAST)


@dataclass(frozen=True, slots=True)
class _Slow(PooledPrimitive):
    operator: _ModalOperator

    @property
    def mode(self) -> OperatorMode:
        return self.operator.slow

    def evaluate(self, *operands: ScalarValue) -> tuple[ScalarValue, ...]:
        (a,) = self._validated_operands(operands)
        return (a,)

    def render(self, *operands: str) -> str:
        (a,) = operands
        return f"slow({a})"


@dataclass(frozen=True, slots=True)
class _Fast(PooledPrimitive):
    operator: _ModalOperator

    @property
    def mode(self) -> OperatorMode:
        return self.operator.fast

    def evaluate(self, *operands: ScalarValue) -> tuple[ScalarValue, ...]:
        a, b = self._validated_operands(operands)
        assert isinstance(a, IntValue) and isinstance(b, IntValue)
        return (a ^ b,)

    def render(self, *operands: str) -> str:
        a, b = operands
        return f"fast({a},{b})"


def _modal_kernel(instances: int, slow_interval: int, split: bool) -> Mir:
    """
    Both adjacent orders of the two modes in a dependent chain, an independent FAST beside them, and a branch whose
    overlapping block ends on a SLOW result that spills into both arms.
    """
    operator = _ModalOperator.build(instances, slow_interval, split)
    slow, fast = _Slow(operator), _Fast(operator)
    ints = IntType(_IFMT)
    builder = MirBuilder(FMT, _IFMT)
    entry, left, right, merge = builder.block(), builder.block(), builder.block(), builder.block()
    builder.position_at(entry)
    x, y = builder.input("x", ints), builder.input("y", ints)
    chained = builder.operation(
        slow,
        [builder.operation(fast, [builder.operation(slow, [x], [IntIdentity()]), y], [IntIdentity()] * 2)],
        [IntIdentity()],
    )
    spilled = builder.operation(slow, [builder.operation(fast, [x, y], [IntIdentity()] * 2)], [IntIdentity()])
    comparator = ICmpPrimitive(IAddOperator.build(_IFMT, IAddOptions()))
    builder.branch(builder.operation(comparator, [x, y], [IntIdentity()] * 2, result=2), left, right)
    builder.position_at(left)
    from_left = builder.operation(fast, [spilled, x], [IntIdentity()] * 2)
    builder.jump(merge)
    builder.position_at(right)
    from_right = builder.operation(fast, [y, spilled], [IntIdentity()] * 2)
    builder.jump(merge)
    builder.position_at(merge)
    builder.output("chained", chained)
    builder.output("merged", builder.phi(ints, [(left, from_left, IntIdentity()), (right, from_right, IntIdentity())]))
    builder.ret()
    return builder.finish()


# Pinned because the annealer reaches a rebinding that a mis-sized busy window would wrongly admit only at some efforts.
_TUNING = RegallocTuning(effort=1000, register_price=2.0)

_MODAL_VECTORS = [(0, 0), (1, 2), (-5, 3), (7, -7), (123, 45), (-1000, -999), (32767, -32768)]


@pytest.mark.parametrize("slow_interval, split", [(4, False), (5, False), (1, True)])
@pytest.mark.parametrize("instances", [1, 2])
def test_primitives_of_different_timing_share_one_operator(instances: int, slow_interval: int, split: bool) -> None:
    # The model replays the LIR's own timing, so agreeing with the MIR interpreter shows the mixed chain is ordered
    # soundly but not that each mode runs at its own latency; the commit check below pins that. A busy window or an
    # allocator slot keyed wrongly trips the LIR's own busy-window check once the allocator takes the move it admits.
    mir = _modal_kernel(instances, slow_interval, split)
    lir = build_lir(mir, "modal", _TUNING)
    firings = [op for block in lir.blocks for op in block.ops if isinstance(op.primitive, (_Slow, _Fast))]
    assert {type(op.primitive) for op in firings} == {_Slow, _Fast}
    assert len({op.inst for op in firings}) == instances
    for op in firings:
        assert op.commit_cycle - op.issue_cycle == (4 if isinstance(op.primitive, _Slow) else 1)
    orders: set[tuple[type, type]] = set()
    for block in lir.blocks:
        for inst in {op.inst for op in block.ops}:
            ordered = sorted((op for op in block.ops if op.inst == inst), key=lambda op: op.issue_cycle)
            for earlier, later in zip(ordered, ordered[1:]):
                spacing = slow_interval if isinstance(earlier.primitive, _Slow) else 1
                assert later.issue_cycle - earlier.issue_cycle >= spacing
                orders.add((type(earlier.primitive), type(later.primitive)))
    if instances == 1:
        assert {(_Slow, _Fast), (_Fast, _Slow)} <= orders
    overtaken = any(
        slow.inst == fast.inst and slow.issue_cycle < fast.issue_cycle and fast.commit_cycle < slow.commit_cycle
        for block in lir.blocks
        for slow in block.ops
        for fast in block.ops
        if isinstance(slow.primitive, _Slow) and isinstance(fast.primitive, _Fast)
    )
    if split and instances == 1:
        assert overtaken, "a pipelined SLOW on a port of its own is overtaken by a FAST issued behind it"
    if not split:
        assert not overtaken, "on a shared port results leave in issue order"
    entry = lir.blocks[0]
    assert any(
        landing_cycle(op.commit_cycle, lir.fetch_lag) > entry.term_offset
        for op in entry.ops
        if isinstance(op.primitive, _Slow)
    ), "a SLOW result spills past the overlapping entry block's terminator"
    # The mode port alone tells the RTL which mode a firing runs in, and the model never reads it.
    events = write_events(lir)
    fields = build_microcode(lir, read_codebook(lir), write_codebook(events), events)
    # A result leaves on the port its mode drives, not on the port its result index names.
    lanes = {(event.source.inst, event.source.port) for event in events if isinstance(event.source, OpWriteSource)}
    expected = {(op.inst, 1 if split and isinstance(op.primitive, _Slow) else 0) for op in firings}
    assert {lane for lane in lanes if lane[0].operator.name == "modal"} == expected
    for block in lir.blocks:
        for op in (op for op in block.ops if isinstance(op.primitive, (_Slow, _Fast))):
            step = lir.block_base[block.index] + op.issue_cycle
            assert fields[f_mode(op.inst.name)].values[step] == (_SLOW if isinstance(op.primitive, _Slow) else _FAST)
    model, interpreter = build_model(lir), MirInterpreter(mir)
    for vector in _MODAL_VECTORS:
        assert model.run(*vector) == interpreter.run(*vector), vector


def test_an_operand_port_no_firing_reads_costs_no_mux_and_is_tied_off() -> None:
    # A mode reading a shorter operand prefix than its operator declares leaves the rest of the ports unread; an
    # instance running only such firings has a port with nothing to read, which the emitter ties off.
    builder = MirBuilder(FMT, _IFMT)
    builder.block()
    x = builder.input("x", IntType(_IFMT))
    builder.output("out", builder.operation(_Slow(_ModalOperator.build(1, 4, False)), [x], [IntIdentity()]))
    builder.ret()
    mir = builder.finish()
    lir = build_lir(mir, "modal_slow")
    (inst,) = lir.instances
    assert read_sources_per_port(lir)[(inst, 1)] == []
    assert "s_modal_0_b = 0;" in generate_verilog(lir).verilog
    model, interpreter = build_model(lir), MirInterpreter(mir)
    for x_value, _ in _MODAL_VECTORS:
        assert model.run(x_value) == interpreter.run(x_value), x_value


def _elaborations(verilog: str) -> dict[str, str]:
    """Each modal instance's parameter list as the emitted Verilog instantiates it, by instance name."""
    return {name: params for params, name in re.findall(r"holoso_modal #\(\n\s*(.*?)\n\) u_(\w+) \(", verilog)}


def _single_instance_kernel(chain: str) -> Mir:
    operator = _ModalOperator.build(1, 4, False)
    slow, fast = _Slow(operator), _Fast(operator)
    builder = MirBuilder(FMT, _IFMT)
    builder.block()
    x, y = builder.input("x", IntType(_IFMT)), builder.input("y", IntType(_IFMT))
    value = x
    for letter in chain:
        if letter == "s":
            value = builder.operation(slow, [value], [IntIdentity()])
        else:
            value = builder.operation(fast, [value, y], [IntIdentity()] * 2)
    builder.output("out", value)
    builder.ret()
    return builder.finish()


@pytest.mark.parametrize(
    "chain, expected",
    [
        ("s", f".W({_IFMT.width}), .MODE(0), .LATENCY_SLOW(4)"),
        ("ff", f".W({_IFMT.width}), .MODE(1), .LATENCY_FAST(1)"),
        ("fs", f".W({_IFMT.width}), .MODE(2), .LATENCY_SLOW(4), .LATENCY_FAST(1)"),
    ],
)
def test_an_instance_running_one_mode_is_elaborated_for_it_alone(chain: str, expected: str) -> None:
    mir = _single_instance_kernel(chain)
    lir = build_lir(mir, f"modal_{chain}")
    assert _elaborations(generate_verilog(lir).verilog) == {"modal_0": expected}
    model, interpreter = build_model(lir), MirInterpreter(mir)
    for vector in _MODAL_VECTORS:
        assert model.run(*vector) == interpreter.run(*vector), vector


def test_each_instance_is_elaborated_for_the_modes_its_own_firings_run() -> None:
    # A SLOW and a FAST issue together and so take both instances; the FAST's instance also runs a SLOW in each arm
    # of the branch. That instance serves both modes, while the other is elaborated for SLOW alone.
    operator = _ModalOperator.build(2, 4, False)
    slow, fast = _Slow(operator), _Fast(operator)
    ints = IntType(_IFMT)
    builder = MirBuilder(FMT, _IFMT)
    entry, left, right, merge = builder.block(), builder.block(), builder.block(), builder.block()
    builder.position_at(entry)
    x, y = builder.input("x", ints), builder.input("y", ints)
    alone = builder.operation(slow, [x], [IntIdentity()])
    mixed = builder.operation(fast, [x, y], [IntIdentity()] * 2)
    comparator = ICmpPrimitive(IAddOperator.build(_IFMT, IAddOptions()))
    builder.branch(builder.operation(comparator, [x, y], [IntIdentity()] * 2, result=2), left, right)
    builder.position_at(left)
    from_left = builder.operation(slow, [mixed], [IntIdentity()])
    builder.jump(merge)
    builder.position_at(right)
    from_right = builder.operation(slow, [y], [IntIdentity()])
    builder.jump(merge)
    builder.position_at(merge)
    builder.output("alone", alone)
    builder.output("merged", builder.phi(ints, [(left, from_left, IntIdentity()), (right, from_right, IntIdentity())]))
    builder.ret()
    mir = builder.finish()
    lir = build_lir(mir, "modal_pair", _TUNING)
    codes = {inst.name: {mode.code for mode in modes} for inst, modes in lir.instance_modes.items()}
    assert codes == {
        "modal_0": {_SLOW},
        "modal_1": {_SLOW, _FAST},
        "iadds_0": {AddMode.SUB},
    }, "the premise of this test"
    assert _elaborations(generate_verilog(lir).verilog) == {
        "modal_0": f".W({_IFMT.width}), .MODE(0), .LATENCY_SLOW(4)",
        "modal_1": f".W({_IFMT.width}), .MODE(2), .LATENCY_SLOW(4), .LATENCY_FAST(1)",
    }
    model, interpreter = build_model(lir), MirInterpreter(mir)
    for vector in _MODAL_VECTORS:
        assert model.run(*vector) == interpreter.run(*vector), vector
