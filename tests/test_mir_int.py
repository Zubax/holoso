"""
The integer half below HIR, in two halves of its own. The first drives :class:`MirInterpreter` over graphs built
directly with :class:`MirBuilder`, pinning what MIR carries; the second drives real kernels through selection and
LIR construction, pinning what the lowering chooses and what the transport keeps.

Both stay white-box to pin selection internals directly (mnemonics, immediates, conditioners, the carriage into
LIR); the black-box end-to-end coverage lives in ``test_int_synthesis`` and ``test_eel_int_corpus``, which drive
``synthesize`` and the numerical model.

No expectation here calls back into ``IntValue``: the rails, the division degeneracies and the two shift readings
are literals or CPython's own operators, so a defect in the value layer cannot vouch for itself.
"""

import math
from dataclasses import replace
from collections.abc import Callable

import pytest

import holoso
from holoso import (
    FAddOptions,
    FCmpOptions,
    FFromIntOptions,
    FMulOptions,
    FRoundOptions,
    FToIntOptions,
    OperatorOptions,
    Options,
    UnsupportedConstruct,
)
from holoso._eel import lower as lower_frontend
from holoso._hir import (
    FloatFloor,
    FloatNeg,
    FloatToInt,
    FloatType as HirFloatType,
    HirBuilder,
    IntMulPow2,
    IntShiftLeft,
    IntShiftRight,
    IntType as HirIntType,
)
from holoso._lir import Lir, PooledScheduledOp, RegRef, WideOutputWire
from holoso._mir import (
    Mir,
    MirBoolView,
    MirBuilder,
    MirFloatConst,
    MirFloatInput,
    MirFloatOutput,
    MirFloatStateSlot,
    MirInterpreter,
    MirIntConst,
    MirIntInput,
    MirIntOutput,
    MirIntStateRead,
    MirIntStateSlot,
    MirOperation,
    MirWideView,
    lower as lower_to_mir,
)
from holoso._operators import (
    BoolInversion,
    BoolToIntOperator,
    FAddOperator,
    FFromIntOperator,
    FMulILog2VarOperator,
    FToIntOperator,
    FloatSignControl,
    HardwareOperator,
    IAbsOperator,
    IAddOperator,
    ICmpOperator,
    IDivOperator,
    IMulOperator,
    IShlOperator,
    ISubOperator,
    IntBwAndOperator,
    IntBwNotOperator,
    IntBwOrOperator,
    IntBwXorOperator,
    IntIdentity,
    IntShiftConstOperator,
    IntToBoolOperator,
    Relation,
    RoundMode,
    SelectOperator,
)
from holoso._type import BoolType, FloatFormat, FloatType, IntFormat, IntType
from holoso._value import FloatValue, IntValue, ScalarValue

from ._modelref import build_lir, build_ops
from ._writetimeline import OperationProducer, build_write_timeline, latest_producer_before

pytestmark = pytest.mark.whitebox

FMT = FloatFormat(5, 11)
IFMT = IntFormat(16)
ITYPE = IntType(IFMT)
FTYPE = FloatType(FMT)
MIN, MAX = -32768, 32767  # spelled out: the reference must not share the code under test


def _clamp(value: int) -> int:
    return min(max(value, MIN), MAX)


def _int(value: int) -> IntValue:
    return IntValue.from_int(IFMT, value)


def _binary_graph(operator: HardwareOperator, *output_ports: int) -> Mir:
    """``a`` and ``b`` feeding one operation per tapped output port, each port exported as its own integer output."""
    builder = MirBuilder(FMT, IFMT)
    builder.block()
    a = builder.int_input("a", ITYPE)
    b = builder.int_input("b", ITYPE)
    for port in output_ports:
        builder.int_output(f"y{port}", builder.operation(operator, [a, b], [IntIdentity()] * 2, output_port=port))
    builder.ret()
    return builder.finish()


def _unary_graph(operator: HardwareOperator) -> Mir:
    builder = MirBuilder(FMT, IFMT)
    builder.block()
    x = builder.int_input("x", ITYPE)
    builder.int_output("y", builder.operation(operator, [x], [IntIdentity()]))
    builder.ret()
    return builder.finish()


def _running_maximum() -> Mir:
    """``y = max(a, y_prev)`` as a diamond: an integer crosses a control-flow edge and merges at a phi."""
    builder = MirBuilder(FMT, IFMT)
    entry = builder.block()
    keep_a = builder.block()
    keep_prev = builder.block()
    join = builder.block()
    builder.position_at(entry)
    a = builder.int_input("a", ITYPE)
    prev = builder.int_state_read("prev", ITYPE)
    port, inversion = ICmpOperator.tap_of(Relation.GT)
    builder.branch(
        builder.operation(
            ICmpOperator(IFMT), [a, prev], [IntIdentity()] * 2, output_port=port, output_conditioner=inversion
        ),
        keep_a,
        keep_prev,
    )
    for block in (keep_a, keep_prev):
        builder.position_at(block)
        builder.jump(join)
    builder.position_at(join)
    merged = builder.phi(ITYPE, [(keep_a, a, IntIdentity()), (keep_prev, prev, IntIdentity())])
    builder.int_output("y", merged)
    builder.int_state_slot("prev", MIN, merged)
    builder.ret()
    return builder.finish()


def _swap_loop() -> Mir:
    """``while n > 0: x, y, n = y, x, n - 1``: resolved in sequence rather than as a snapshot, the pair collapses."""
    builder = MirBuilder(FMT, IFMT)
    entry = builder.block()
    header = builder.block()
    body = builder.block()
    done = builder.block()
    builder.position_at(entry)
    x0 = builder.int_input("x", ITYPE)
    y0 = builder.int_input("y", ITYPE)
    n0 = builder.int_input("n", ITYPE)
    zero = builder.int_const(0, ITYPE)
    one = builder.int_const(1, ITYPE)
    builder.jump(header)
    builder.position_at(header)
    x = builder.open_phi(ITYPE, (entry, x0, IntIdentity()))
    y = builder.open_phi(ITYPE, (entry, y0, IntIdentity()))
    n = builder.open_phi(ITYPE, (entry, n0, IntIdentity()))
    port, inversion = ICmpOperator.tap_of(Relation.GT)
    builder.branch(
        builder.operation(
            ICmpOperator(IFMT), [n, zero], [IntIdentity()] * 2, output_port=port, output_conditioner=inversion
        ),
        body,
        done,
    )
    builder.position_at(body)
    remaining = builder.operation(ISubOperator(IFMT), [n, one], [IntIdentity()] * 2)
    builder.jump(header)
    builder.set_phi_arms(x, [(entry, x0, IntIdentity()), (body, y, IntIdentity())])
    builder.set_phi_arms(y, [(entry, y0, IntIdentity()), (body, x, IntIdentity())])
    builder.set_phi_arms(n, [(entry, n0, IntIdentity()), (body, remaining, IntIdentity())])
    builder.position_at(done)
    builder.int_output("x", x)
    builder.int_output("y", y)
    builder.ret()
    return builder.finish()


@pytest.mark.parametrize(
    "operator,reference",
    [
        (IAddOperator(IFMT), lambda a, b: _clamp(a + b)),
        (ISubOperator(IFMT), lambda a, b: _clamp(a - b)),
        (IMulOperator(IFMT, IMulOperator.Options()), lambda a, b: _clamp(a * b)),
    ],
)
def test_integer_arithmetic_saturates_at_the_rails(
    operator: HardwareOperator, reference: Callable[[int, int], int]
) -> None:
    interpreter = MirInterpreter(_binary_graph(operator, 0))
    for a, b in [(0, 0), (7, 3), (-7, 3), (MAX, 1), (MIN, -1), (MAX, MAX), (MIN, MIN), (MIN, MAX), (-1, -1)]:
        assert interpreter.run(a, b) == [_int(reference(a, b))], f"{operator.mnemonic}({a}, {b})"


@pytest.mark.parametrize(
    "operator,reference",
    [
        (IntBwAndOperator(IFMT), lambda a, b: a & b),
        (IntBwOrOperator(IFMT), lambda a, b: a | b),
        (IntBwXorOperator(IFMT), lambda a, b: a ^ b),
    ],
)
def test_integer_bitwise_combination_never_leaves_the_word(
    operator: HardwareOperator, reference: Callable[[int, int], int]
) -> None:
    """Bitwise combination cannot escape the range a two's-complement word already spans, so nothing saturates."""
    interpreter = MirInterpreter(_binary_graph(operator, 0))
    for a, b in [(0, 0), (0x0F0F, 0x00FF), (MIN, -1), (MAX, MIN), (-1, -1), (-12345, 6789)]:
        assert interpreter.run(a, b) == [_int(reference(a, b))], f"{operator.mnemonic}({a}, {b})"


@pytest.mark.parametrize("value", [0, -1, 1, MIN, MAX, 12345])
def test_integer_bitwise_complement_maps_the_rails_onto_each_other(value: int) -> None:
    assert MirInterpreter(_unary_graph(IntBwNotOperator(IFMT))).run(value) == [_int(~value)]


@pytest.mark.parametrize("value,expected", [(0, 0), (5, 5), (-5, 5), (MAX, MAX), (MIN, MAX)])
def test_integer_absolute_value_of_the_minimum_rails(value: int, expected: int) -> None:
    interpreter = MirInterpreter(_unary_graph(IAbsOperator(IFMT)))
    assert interpreter.run(_int(value)) == [_int(expected)]  # an already-encoded input


def test_integer_division_taps_quotient_and_remainder_from_one_operation() -> None:
    """Both readings come off ONE ``idivs`` over one operand pair, which is what lets LIR fuse them later."""
    mir = _binary_graph(IDivOperator(IFMT), 0, 1)
    operations = [node for node in mir.nodes.values() if isinstance(node, MirOperation)]
    assert {op.output_port for op in operations} == {0, 1}
    assert len({(op.operator, tuple(op.operands)) for op in operations}) == 1
    interpreter = MirInterpreter(mir)
    for a, b in [(7, 3), (-7, 3), (7, -3), (-7, -3), (MIN, -1), (5, 0), (-5, 0), (0, 0)]:
        if b == 0:  # a rail by the numerator's sign, keeping the numerator as the remainder
            expected = (MIN if a < 0 else MAX, a)
        elif (a, b) == (MIN, -1):
            expected = (MAX, 0)
        else:
            expected = divmod(a, b)
        assert interpreter.run(a, b) == [_int(expected[0]), _int(expected[1])], f"{a} // {b}"


@pytest.mark.parametrize(
    "value,count,shft,prod",
    [
        (3, 2, 12, 12),
        (-3, 2, -12, -12),
        (1, 20, 0, MAX),  # past the word a left shift loses every bit while the saturating reading rails
        (-1, 20, 0, MIN),
        (-12345, -3, -1544, -1544),
        (12345, -20, 0, 0),  # past the word a right shift is a sign fill
        (-12345, -20, -1, -1),
    ],
)
def test_integer_shift_emits_both_readings_past_the_word(value: int, count: int, shft: int, prod: int) -> None:
    interpreter = MirInterpreter(_binary_graph(IShlOperator(IFMT), 0, 1))
    assert interpreter.run(value, count) == [_int(shft), _int(prod)]


@pytest.mark.parametrize("value,shamt,expected", [(3, 2, 12), (-3, 2, -12), (-12345, -3, -1544), (1, 15, MIN)])
def test_constant_integer_shift_is_the_raw_bit_shift(value: int, shamt: int, expected: int) -> None:
    """The inline constant shift keeps the ``shft`` reading, so a bit shifted into the sign is a value, not a rail."""
    assert MirInterpreter(_unary_graph(IntShiftConstOperator(IFMT, shamt))).run(value) == [_int(expected)]


def test_integer_comparison_serves_every_relation_from_one_comparator() -> None:
    builder = MirBuilder(FMT, IFMT)
    builder.block()
    a = builder.int_input("a", ITYPE)
    b = builder.int_input("b", ITYPE)
    operator = ICmpOperator(IFMT)
    for relation in (Relation.LT, Relation.GE, Relation.EQ):
        port, inversion = operator.tap_of(relation)
        vid = builder.operation(operator, [a, b], [IntIdentity()] * 2, output_port=port, output_conditioner=inversion)
        builder.bool_output(relation.name.lower(), vid)
    builder.ret()
    mir = builder.finish()
    operations = [node for node in mir.nodes.values() if isinstance(node, MirOperation)]
    assert len({(op.operator, tuple(op.operands)) for op in operations}) == 1
    interpreter = MirInterpreter(mir)
    for a_value, b_value in [(1, 2), (2, 2), (3, 2), (MIN, MAX), (MAX, MIN)]:
        assert interpreter.run(a_value, b_value) == [a_value < b_value, a_value >= b_value, a_value == b_value]


def test_integer_state_slot_carries_across_transactions() -> None:
    """A running total: the slot is read-first, so each transaction observes the previous one's live-out."""
    builder = MirBuilder(FMT, IFMT)
    builder.block()
    x = builder.int_input("x", ITYPE)
    total = builder.int_state_read("total", ITYPE)
    builder.int_output("was", total)
    builder.int_state_slot("total", 7, builder.operation(IAddOperator(IFMT), [total, x], [IntIdentity()] * 2))
    builder.ret()
    interpreter = MirInterpreter(builder.finish())
    assert [interpreter.run(step)[0] for step in (1, 2, 3)] == [_int(7), _int(8), _int(10)]
    interpreter.reset()
    assert interpreter.run(0) == [_int(7)]


def test_integer_phi_merges_the_arms_of_a_branch() -> None:
    interpreter = MirInterpreter(_running_maximum())
    running = MIN
    for value in (3, -7, 3, 100, 99, MAX, MIN):
        running = max(running, value)
        assert interpreter.run(value) == [_int(running)], f"after {value}"
    interpreter.reset()
    assert interpreter.run(MIN) == [_int(MIN)]


@pytest.mark.parametrize("trips", [0, 1, 2, 3, 7, 20, -4])
def test_integer_phis_swap_in_parallel_across_a_back_edge(trips: int) -> None:
    swapped = max(trips, 0) % 2 == 1  # a non-positive count never enters the body
    assert MirInterpreter(_swap_loop()).run(11, -22, trips) == [
        _int(-22 if swapped else 11),
        _int(11 if swapped else -22),
    ]


def test_cross_family_operators_carry_mixed_conditioner_lists() -> None:
    """Operands and result in different banks: the boolean casts, a select on a folded inversion, the scaler."""
    builder = MirBuilder(FMT, IFMT)
    builder.block()
    flag = builder.bool_input("flag", BoolType())
    a = builder.int_input("a", ITYPE)
    b = builder.int_input("b", ITYPE)
    k = builder.int_input("k", ITYPE)
    builder.int_output("from_bool", builder.operation(BoolToIntOperator(IFMT), [flag], [BoolInversion()]))
    builder.bool_output("to_bool", builder.operation(IntToBoolOperator(IFMT), [a], [IntIdentity()]))
    builder.int_output(  # the condition arrives inverted, so the arms answer the other way round
        "picked",
        builder.operation(
            SelectOperator(ITYPE), [flag, a, b], [BoolInversion(invert=True), IntIdentity(), IntIdentity()]
        ),
    )
    builder.float_output(
        "scaled",
        builder.operation(
            FMulILog2VarOperator(FMT, IFMT, FMulILog2VarOperator.Options()),
            [builder.float_const(1.5, FTYPE), k],
            [FloatSignControl(), IntIdentity()],
        ),
    )
    builder.ret()
    interpreter = MirInterpreter(builder.finish())
    for flag_value, a_value, b_value, k_value in [(True, 3, -4, 2), (False, 0, 7, -3), (True, MIN, MAX, 0)]:
        assert interpreter.run(flag_value, a_value, b_value, k_value) == [
            _int(int(flag_value)),
            a_value != 0,
            _int(b_value if flag_value else a_value),
            FloatValue.from_float(FMT, 1.5 * 2.0**k_value),
        ]


def test_integer_constants_and_conversions_cross_the_family_boundary() -> None:
    """``float(int(f) & 0xFF)`` -- an integer constant, both mixed-format operators, and an inline bitwise between."""
    builder = MirBuilder(FMT, IFMT)
    builder.block()
    f = builder.float_input("f", FTYPE)
    truncated = builder.operation(
        FToIntOperator(FMT, IFMT, FToIntOperator.Options()),
        [f],
        [FloatSignControl()],
        immediates=(int(RoundMode.TRUNC),),
    )
    masked = builder.operation(IntBwAndOperator(IFMT), [truncated, builder.int_const(0xFF, ITYPE)], [IntIdentity()] * 2)
    builder.float_output(
        "y", builder.operation(FFromIntOperator(FMT, IFMT, FFromIntOperator.Options()), [masked], [IntIdentity()])
    )
    builder.int_output("i", masked)
    builder.ret()
    interpreter = MirInterpreter(builder.finish())
    for value in (0.0, 3.75, 300.0, -3.75, -300.0):
        expected = int(value) & 0xFF  # CPython truncates toward zero too
        assert interpreter.run(FloatValue.from_float(FMT, value)) == [
            FloatValue.from_float(FMT, float(expected)),
            _int(expected),
        ]


def test_wide_view_admits_both_wide_families_while_the_bool_view_admits_neither() -> None:
    builder = MirBuilder(FMT, IFMT)
    builder.block()
    f = builder.float_input("f", FTYPE)
    i = builder.int_input("i", ITYPE)
    flag = builder.bool_input("flag", BoolType())
    scaled = builder.operation(
        IMulOperator(IFMT, IMulOperator.Options()), [i, builder.int_const(3, ITYPE)], [IntIdentity()] * 2
    )
    port, inversion = ICmpOperator.tap_of(Relation.LT)
    below = builder.operation(
        ICmpOperator(IFMT), [scaled, i], [IntIdentity()] * 2, output_port=port, output_conditioner=inversion
    )
    widened = builder.operation(FFromIntOperator(FMT, IFMT, FFromIntOperator.Options()), [scaled], [IntIdentity()])
    biased = builder.operation(
        FAddOperator(FMT, FAddOperator.Options()),
        [widened, builder.float_const(0.5, FTYPE)],
        [FloatSignControl()] * 2,
    )
    builder.float_output("y", biased)
    builder.int_output("n", scaled)
    builder.bool_output("below", below)
    builder.float_state_slot("acc", 0.5, f)
    builder.int_state_slot("count", -1, scaled)
    builder.bool_state_slot("seen", False, flag)
    builder.ret()
    mir = builder.finish()

    wide = MirWideView.from_mir(mir)
    assert wide.float_format == FMT and wide.int_format == IFMT
    assert {type(node) for node in wide.input_nodes.values()} == {MirFloatInput, MirIntInput}
    assert {type(node) for node in wide.const_nodes.values()} == {MirFloatConst, MirIntConst}
    assert {type(out) for out in wide.outputs} == {MirFloatOutput, MirIntOutput}
    assert {type(slot) for slot in wide.state_slots} == {MirFloatStateSlot, MirIntStateSlot}
    assert [out.conditioner for out in wide.outputs if isinstance(out, MirIntOutput)] == [IntIdentity()]
    assert below not in wide.nodes, "a comparator tap is a boolean value however wide its operands are"

    boolean = MirBoolView.from_mir(mir)
    assert not set(boolean.nodes) & set(wide.nodes), "the two banks must partition the graph"
    assert below in boolean.operation_nodes
    assert [slot.name for slot in boolean.state_slots] == ["seen"]


def test_wide_view_admits_an_integer_phi_and_state_read() -> None:
    wide = MirWideView.from_mir(_running_maximum())
    assert {node.scalar_type for node in wide.phi_nodes.values()} == {ITYPE}
    assert {type(node) for node in wide.state_read_nodes.values()} == {MirIntStateRead}
    assert {type(slot) for slot in wide.state_slots} == {MirIntStateSlot}


def _foreign_format_leaf() -> Mir:
    builder = MirBuilder(FMT, IntFormat(24))
    builder.block()
    builder.int_output("y", builder.int_input("x", ITYPE))
    builder.ret()
    return builder.finish()


def _foreign_format_operation() -> Mir:
    """The stray format reaches the view only through an operation result; every leaf here is a float."""
    builder = MirBuilder(FMT, IntFormat(24))
    builder.block()
    f = builder.float_input("f", FTYPE)
    converted = builder.operation(
        FToIntOperator(FMT, IFMT, FToIntOperator.Options()), [f], [FloatSignControl()], immediates=(0,)
    )
    builder.int_output("y", converted)
    builder.ret()
    return builder.finish()


def _foreign_format_beside_a_conforming_one() -> Mir:
    """A conforming leaf beside the stray one, so the diagnostic has something it could wrongly include."""
    builder = MirBuilder(FMT, IntFormat(24))
    builder.block()
    builder.int_output("wide", builder.int_input("wide", IntType(IntFormat(24))))
    builder.int_output("stray", builder.int_input("stray", ITYPE))
    builder.ret()
    return builder.finish()


@pytest.mark.parametrize(
    "build", [_foreign_format_leaf, _foreign_format_operation, _foreign_format_beside_a_conforming_one]
)
def test_wide_view_refuses_an_integer_of_a_foreign_format(build: Callable[[], Mir]) -> None:
    """Anchored because ``int24`` is the configured format: unanchored would pass on a message listing both."""
    with pytest.raises(ValueError, match=r"got int16$"):
        MirWideView.from_mir(build())


# ----------------------------------------------------------------------------------------------------------------
# Selection and LIR carriage, driven by real kernels rather than hand-built graphs.
#
# ``FMT`` is 16 bits wide and ``wint_min`` defaults to 16, so these kernels run on exactly the ``IFMT`` machine the
# hand-built graphs above use.


KERNEL_OPTIONS = Options(
    OperatorOptions(
        fadd=FAddOptions(),
        fmul=FMulOptions(),
        fcmp=FCmpOptions(),
        fround=FRoundOptions(),
        ffromint=FFromIntOptions(),
        ftoint=FToIntOptions(),
    ),
    ffmt=FMT,
)


def _select_with_budget(target: Callable[..., object], ifconv_max_ops: int) -> Mir:
    """Front end through selection, exactly the chain ``synthesize`` runs before it builds the LIR."""
    return lower_to_mir(
        lower_frontend(target).hir,
        build_ops(KERNEL_OPTIONS),
        KERNEL_OPTIONS.ffmt,
        KERNEL_OPTIONS.ifmt,
        ifconv_max_ops,
    )


def _select(target: Callable[..., object]) -> Mir:
    return _select_with_budget(target, KERNEL_OPTIONS.ifconv_max_ops)


def _plain(value: ScalarValue) -> int | float | bool:
    match value:
        case bool():
            return value
        case IntValue():
            return int(value)
        case _:
            return float(value)


def _run(interpreter: MirInterpreter, *args: int | float | bool) -> list[int | float | bool]:
    return [_plain(value) for value in interpreter.run(*args)]


def _expected(target: Callable[..., object], *args: int | float | bool) -> list[object]:
    result = target(*args)
    return list(result) if isinstance(result, tuple) else [result]


def _operations(mir: Mir, mnemonic: str) -> list[MirOperation]:
    return [
        node for node in mir.nodes.values() if isinstance(node, MirOperation) and node.operator.mnemonic == mnemonic
    ]


def _mnemonics(mir: Mir) -> list[str]:
    return sorted(node.operator.mnemonic for node in mir.nodes.values() if isinstance(node, MirOperation))


def _wide_firings(lir: Lir) -> list[PooledScheduledOp]:
    return [op for block in lir.blocks for op in block.ops]


def divmod_pair(a: int, b: int) -> tuple[int, int]:
    return a // b, a % b


def three_relations(a: int, b: int) -> tuple[bool, bool, bool]:
    return a < b, a == b, a > b


def four_relations(a: int, b: int) -> tuple[bool, bool, bool, bool]:
    return a <= b, a != b, a >= b, a > b


def sign_ops(a: int, b: int) -> tuple[int, int]:
    return -a, abs(b)


def bitwise_ops(a: int, b: int) -> tuple[int, int, int, int]:
    return a & b, a | b, a ^ b, ~a


def mux_and_casts(a: int, b: int, c: bool) -> tuple[int, bool, int]:
    return (a if c else b), bool(a), int(c) + a


def family_crossings(x: float, n: int) -> tuple[int, float]:
    return int(x), float(n)


def negated_crossing(x: float) -> int:
    return int(-x)


def shift_pair(x: int, n: int) -> tuple[int, int]:
    return x << n, x >> n


def boundary_outputs(a: int, b: int) -> tuple[int, int, int]:
    """Three integer outputs, one of them computed first and then left idle while a long chain runs."""
    return a + b, a - b, ((a * b) * (a - b)) * (a + 2) * (b + 3)


def mixed_constants(x: float, n: int) -> tuple[float, int, int]:
    """``1`` beside ``1.0`` is the pool collision; ``7`` lands at an index that is not its own value."""
    return x + 1.0, n + 1, n + 7


def rounded_to_float(x: float) -> float:
    return float(math.floor(x)) + 1.0


def countdown(n: int) -> int:
    steps = 0
    while n > 0:
        n = n - 3
        steps = steps + 1
    return steps


class MixedState:
    def __init__(self) -> None:
        self.count = 0
        self.level = 0.0

    def step(self, n: int, x: float) -> tuple[int, float]:
        self.count = self.count + n
        self.level = self.level + x
        return self.count, self.level


class Accumulator:
    def __init__(self) -> None:
        self.total = 0

    def step(self, x: int) -> int:
        self.total = self.total + x
        return self.total


class InputLatch:
    def __init__(self) -> None:
        self.prev = 0

    def step(self, x: int, y: int) -> int:
        """The slot live-out is the integer INPUT itself, which is what lets the install run ahead of the boundary."""
        out = self.prev * y + x * 3 - y * y
        self.prev = x
        return out


@pytest.mark.parametrize(
    "target,args,selected",
    [
        (divmod_pair, (17, 5), ["idivs", "idivs"]),
        (three_relations, (3, 9), ["icmp", "icmp", "icmp"]),
        (sign_ops, (7, -9), ["iabss", "isubs"]),
        (bitwise_ops, (0x0F0F, 0x00FF), ["ibwand", "ibwnot", "ibwor", "ibwxor"]),
        (mux_and_casts, (5, 6, True), ["iadds", "ifrombool", "itobool", "select"]),
        (family_crossings, (3.75, -4), ["ffromint", "ftoint"]),
        (shift_pair, (5, 2), ["ishl", "ishr"]),
        (countdown, (10,), ["iadds", "icmp", "isubs"]),
    ],
)
def test_a_kernel_selects_its_integer_modules_and_answers_as_cpython_does(
    target: Callable[..., object], args: tuple[int | float | bool, ...], selected: list[str]
) -> None:
    """
    Every integer operator the lowering can choose, named in one place: ``ineg`` selects ``isubs`` because there is
    no negation module, while each shift direction selects the module that names it.
    """
    mir = _select(target)
    assert _mnemonics(mir) == selected
    assert _run(MirInterpreter(mir), *args) == _expected(target, *args)


def test_a_negated_operand_folds_onto_the_conversion() -> None:
    """``int(-x)`` conditions the ``ftoint`` float port rather than emitting a sign operator of its own."""
    mir = _select(negated_crossing)
    assert _mnemonics(mir) == ["ftoint"]
    for x in (3.75, -3.75, 0.0):
        assert _run(MirInterpreter(mir), x) == _expected(negated_crossing, x)


def test_the_quotient_and_the_remainder_share_one_divider_firing() -> None:
    """``a // b`` beside ``a % b`` is one activation with two taps -- counted, because fusion is not automatic."""
    lir = build_lir(_select(divmod_pair), "divmod_pair")
    (firing,) = _wide_firings(lir)
    assert [instance.operator.mnemonic for instance in lir.instances] == ["idivs"]
    assert sorted(write.port for write in firing.writes) == [0, 1]


def test_relations_over_one_operand_pair_fuse_into_one_comparator_firing() -> None:
    """Three relations, three flags, one activation -- counted, because fusion is not automatic."""
    lir = build_lir(_select(three_relations), "three_relations")
    (firing,) = _wide_firings(lir)
    assert [instance.operator.mnemonic for instance in lir.instances] == ["icmp"]
    assert len(firing.writes) == 3


def test_two_relations_reading_one_flag_still_share_the_comparator() -> None:
    """
    A firing taps each port at most once, so ``a <= b`` and ``a > b`` -- the same flag under opposite inversions --
    need an activation each. They still bind to the one pooled comparator: the cost is a cycle, never a module.
    """
    lir = build_lir(_select(four_relations), "four_relations")
    assert [instance.operator.mnemonic for instance in lir.instances] == ["icmp"]
    assert len(_wide_firings(lir)) == 2


def test_each_integer_output_still_holds_its_own_value_at_the_boundary() -> None:
    """
    The register an output taps must not be recycled before the boundary reads it. This is the one wide-bank
    predicate an integer output reaches on its own, and getting it wrong is silent: the allocator simply frees the
    register early and a later firing lands in it, so the port reads a stranger's value with no exception anywhere.
    """
    lir = build_lir(_select(boundary_outputs), "boundary_outputs")
    timeline = build_write_timeline(lir)
    resolved: dict[str, str] = {}
    for wire in lir.outputs:
        assert isinstance(wire, WideOutputWire) and isinstance(wire.tap.source, RegRef)
        producer = latest_producer_before(timeline, wire.tap.source, lir.initiation_interval)
        assert isinstance(producer, OperationProducer)
        resolved[wire.name] = lir.ops[producer.index].inst.operator.mnemonic
    assert resolved == {"out_0": "iadds", "out_1": "isubs", "out_2": "imuls"}


def test_a_slot_fed_by_an_integer_input_installs_ahead_of_the_boundary() -> None:
    """
    A live-out that is an input is available immediately, so the slot installs at once and frees nothing else to wait
    on; the boundary install a narrower predicate would force is correct but costs the register for the whole run.
    """
    lir = build_lir(_select(InputLatch().step), "input_latch")
    (slot,) = lir.wide_state_slots
    assert slot.install_cycle == 1 < lir.initiation_interval


def test_the_lir_carries_the_scalar_family_of_every_wide_port() -> None:
    """A wide carrier no longer names its family, so the port metadata the RTL and the model share must."""
    lir = build_lir(_select(family_crossings), "family_crossings")
    assert [(port.name, port.scalar_type) for port in lir.input_ports] == [("in_x", FTYPE), ("in_n", ITYPE)]
    assert [(port.name, port.scalar_type) for port in lir.output_ports] == [("out_0", ITYPE), ("out_1", FTYPE)]


def test_a_state_slot_carries_its_scalar_family_too() -> None:
    """The third wide carrier: a slot's family comes from its live-out, not from what its reset literal happens
    to be, so a float slot reset to a whole number is still a float slot."""
    lir = build_lir(_select(MixedState().step), "mixed_state")
    assert {slot.name: slot.reset_value for slot in lir.wide_state_slots} == {
        "count": IntValue.from_int(IFMT, 0),
        "level": FloatValue.from_float(FMT, 0.0),
    }


def test_a_kernel_mixing_integer_and_float_state_keeps_them_apart() -> None:
    interpreter = MirInterpreter(_select(MixedState().step))
    reference = MixedState()
    for n, x in ((3, 0.5), (-10, 0.25), (4, -1.0)):
        assert _run(interpreter, n, x) == _expected(reference.step, n, x)


def test_the_two_families_keep_separate_entries_in_one_constant_pool() -> None:
    """``1`` and ``1.0`` hash and compare equal in Python while naming different words, so the keying must differ."""
    lir = build_lir(_select(mixed_constants), "mixed_constants")
    assert lir.wide_consts == [FloatValue.from_float(FMT, 1.0), _int(1), _int(7)]


def test_an_integer_slot_carries_its_value_across_transactions() -> None:
    interpreter = MirInterpreter(_select(Accumulator().step))
    reference = Accumulator()
    for x in (3, 4, -10, 9000, -12):  # short of the rails, where CPython's unbounded sum is still the machine's
        assert _run(interpreter, x) == _expected(reference.step, x)


def test_a_data_dependent_loop_merges_integers_at_its_header() -> None:
    """A residual loop the front end cannot unroll: the trip count and the counter both merge at an integer phi."""
    interpreter = MirInterpreter(_select(countdown))
    for n in (0, 1, 10, -5, 100):
        assert _run(interpreter, n) == _expected(countdown, n)


def _wrap(value: int) -> int:
    """Into the word, spelled out rather than borrowed from ``IntFormat``."""
    return ((value + 32768) % 65536) - 32768


@pytest.mark.parametrize("x", [0, 1, -1, 5, -5, 12345, MIN, MAX])
@pytest.mark.parametrize("n", [0, 1, 3, 14, 15, 16, 17, 40])
def test_a_shift_answers_as_python_does_once_the_word_truncates_it(x: int, n: int) -> None:
    """
    The shifter clamps its count at the word, which is exactly where an unbounded shift stops saying anything new:
    a right shift past the word is the sign fill CPython also gives, and a left shift past it drops every bit either
    way. So CPython is the reference for every non-negative count, with only the left shift's wrap applied.
    """
    interpreter = MirInterpreter(_select(shift_pair))
    assert _run(interpreter, x, n) == [_wrap(x << n), x >> n]


@pytest.mark.parametrize("x,n,expected", [(1, -2, [0, 4]), (-8, -2, [-2, -32]), (12345, -20, [0, 0])])
def test_a_negative_runtime_shift_count_reverses_the_direction(x: int, n: int, expected: list[int]) -> None:
    """
    CPython refuses a negative count; each shifter is total over every representable one and reads it as its other
    direction. A kernel reaches this only through a value it did not constrain, so the hardware's answer is the
    definition -- there is no other.
    """
    assert _run(MirInterpreter(_select(shift_pair)), x, n) == expected


def test_a_right_shift_costs_exactly_what_a_left_shift_costs() -> None:
    """
    The point of a second shifter: a right shift no longer negates its count, so it pays neither the subtractor nor
    the cycles that dependency cost. Matching schedules alone would pass if both regressed, so the module each
    kernel builds is pinned too.
    """
    left, right = (build_lir(_select(target), "shift") for target in (shift_left_only, shift_right_only))
    assert [inst.operator.mnemonic for inst in left.instances] == ["ishl"]
    assert [inst.operator.mnemonic for inst in right.instances] == ["ishr"]
    assert left.last_pc == right.last_pc
    assert left.min_initiation_interval == right.min_initiation_interval


def _scaled_by_a_power_of_two(k: int) -> Mir:
    """``x * 2**k`` built directly, no pass minting it yet."""
    builder = HirBuilder()
    builder.block()
    builder.output("y", builder.operation(IntMulPow2(k), [builder.input("x", HirIntType())]))
    builder.ret()
    return lower_to_mir(
        builder.finish(),
        build_ops(KERNEL_OPTIONS),
        KERNEL_OPTIONS.ffmt,
        KERNEL_OPTIONS.ifmt,
        KERNEL_OPTIONS.ifconv_max_ops,
    )


@pytest.mark.parametrize("x", [0, 1, -1, 3, -3, 1000, -1000, MIN, MAX])
@pytest.mark.parametrize("k", [1, 2, 14, 15, 16, 40])
def test_a_power_of_two_scaling_reads_the_shifter_where_it_saturates(k: int, x: int) -> None:
    """
    The one thing separating this operator from ``x << k``: a multiplication rails where the raw shift drops what
    leaves the word, and the shifter emits both readings, so the tap is the whole decision. The count is unbounded
    where the word is not, so every one past the width rails the same operand the same way.
    """
    mir = _scaled_by_a_power_of_two(k)
    assert _mnemonics(mir) == ["ishl"]
    assert _run(MirInterpreter(mir), x) == [_clamp(x * 2**k)]


def test_a_power_of_two_scaling_and_the_raw_shift_share_one_module() -> None:
    """Both readings come off one firing, so a kernel wanting each pays for a single shifter."""
    scaled, shifted = _scaled_by_a_power_of_two(1), _select(shift_left_only)
    assert _mnemonics(scaled) == _mnemonics(shifted) == ["ishl"]
    assert _run(MirInterpreter(scaled), MAX) == [MAX], "the product rails"
    assert _run(MirInterpreter(shifted), MAX, 1) == [_wrap(MAX << 1)], "the raw shift does not"


def times_eight(x: int) -> int:
    return x * 8


def eighth(x: int) -> int:
    return x // 8


def eighth_remainder(x: int) -> int:
    return x % 8


def past_the_word_quotient(x: int) -> int:
    return x // 2**40


def negated_by_product(x: int) -> int:
    return x * -1


def third(x: int) -> int:
    return x // 3


@pytest.mark.parametrize("x", [0, 1, -1, 5, -5, -8, 12345, -12345, MIN, MAX])
def test_a_minted_power_of_two_product_saturates_like_the_multiplication(x: int) -> None:
    """Strength reduction hands ``x * 8`` to the shifter's saturating tap, so the rails answer as the product."""
    mir = _select(times_eight)
    assert _mnemonics(mir) == ["ishl"]
    assert _run(MirInterpreter(mir), x) == [_clamp(x * 8)]


@pytest.mark.parametrize("x", [0, 1, -1, 5, -5, -8, 12345, -12345, MIN, MAX])
def test_a_minted_power_of_two_quotient_is_one_inline_shift(x: int) -> None:
    """``x // 8`` pays neither the divider nor any module: the arithmetic shift IS the floor division."""
    mir = _select(eighth)
    assert _mnemonics(mir) == ["ishiftc"]
    assert _run(MirInterpreter(mir), x) == [x // 8]


@pytest.mark.parametrize("x", [MAX, 1, 0, -1, MIN])
def test_a_minted_quotient_past_the_word_is_the_sign_fill(x: int) -> None:
    """The clamp at the top bit answers exactly the floor of an in-word operand over so large a divisor."""
    assert _run(MirInterpreter(_select(past_the_word_quotient)), x) == [x // 2**40]


def past_the_word_product(x: int) -> int:
    return x * 2**40


@pytest.mark.parametrize("x", [MAX, 1, 0, -1, MIN])
def test_a_minted_product_past_the_word_rails_by_sign(x: int) -> None:
    """A count past the width rails every nonzero operand, exactly as the multiplication it stands for would."""
    mir = _select(past_the_word_product)
    assert _mnemonics(mir) == ["ishl"]
    assert _run(MirInterpreter(mir), x) == [_clamp(x * 2**40)]


def boundary_remainder(x: int) -> int:
    return x % 2**15


def boundary_quotient(x: int) -> int:
    return x // 2**15


@pytest.mark.parametrize("x", [0, 1, -1, 5, -5, 12345, -12345, MIN, MAX])
def test_the_boundary_exponent_builds_where_its_divisor_constant_could_not(x: int) -> None:
    """
    ``2**15`` fits no int16 word, so the spelled-out divisor is refused at selection; the mask and the shift the
    rewrites leave are in-word and exact, so the boundary exponent builds and answers as CPython does.
    """
    assert _run(MirInterpreter(_select(boundary_remainder)), x) == [x % 2**15]
    assert _run(MirInterpreter(_select(boundary_quotient)), x) == [x // 2**15]


@pytest.mark.parametrize("x", [0, 1, -1, 5, -5, -8, 12345, -12345, MIN, MAX])
def test_a_minted_power_of_two_remainder_is_the_mask(x: int) -> None:
    """``x % 8`` is the two's-complement mask, negative dividends included, with no divider error sideband."""
    mir = _select(eighth_remainder)
    assert _mnemonics(mir) == ["ibwand"]
    assert _run(MirInterpreter(mir), x) == [x % 8]


def test_a_product_with_minus_one_negates_on_the_subtractor() -> None:
    mir = _select(negated_by_product)
    assert _mnemonics(mir) == ["isubs"]
    assert _run(MirInterpreter(mir), MIN) == [MAX], "the negation saturates at the rail"
    assert _run(MirInterpreter(mir), MAX) == [MIN + 1]


def test_a_quotient_by_a_non_power_of_two_still_pays_the_divider() -> None:
    assert _mnemonics(_select(third)) == ["idivs"]


def shift_left_only(x: int, n: int) -> int:
    return x << n


def shift_right_only(x: int, n: int) -> int:
    return x >> n


def shift_left_by_a_negative_constant(x: int) -> int:
    return x << -1


def shift_right_by_a_negative_constant(x: int) -> int:
    return x >> -1


def shift_by_a_count_a_loop_phi_carries(x: int, n: int) -> int:
    # An optimizer that stops before the phi folds leaves a runtime count, and the shifter reverses it into ``x >> 1``.
    count = (x * 0) - 1
    t = n
    while t > 0:
        count = (x * 0) - 1
        t = t - 1
    return x << count


@pytest.mark.parametrize(
    "target",
    [shift_left_by_a_negative_constant, shift_right_by_a_negative_constant, shift_by_a_count_a_loop_phi_carries],
)
def test_a_constant_negative_shift_count_is_refused_rather_than_reversed(target: Callable[..., object]) -> None:
    """
    CPython raises on a negative count, and the shifter would read one as its OPPOSITE direction --
    a wrong answer, not a rail. HIR cannot fold it away here because the shifted value is a runtime input.
    """
    with pytest.raises(UnsupportedConstruct, match=r"shift count -1 is negative; Python has no such shift"):
        _select(target)


def shift_by_nothing(x: int) -> tuple[int, int]:
    return x << 0, x >> 0


def shift_by_one(x: int) -> tuple[int, int]:
    return x << 1, x >> 1


def shift_by_the_top_bit(x: int) -> tuple[int, int]:
    return x << 15, x >> 15


def shift_by_the_whole_word(x: int) -> tuple[int, int]:
    return x << 16, x >> 16


def shift_past_every_word(x: int) -> tuple[int, int]:
    """A count no machine word can hold: legal Python, and only the fold can carry it to an answer."""
    return x << 100000, x >> 100000


def shift_by_a_count_used_as_a_value(x: int) -> tuple[int, int]:
    return x << 3, x + 3


def a_count_two_rounds_make_constant(x: int, y: int) -> int:
    """The count is a select until the guard above it is settled, which only the first substitution round does."""
    guard = x << 100000
    n = 100000 if guard == 0 else 1
    return y << n


def a_count_a_runtime_select_keeps(y: int, c: bool) -> int:
    return y << (100000 if c else 3)


def an_oversize_shift_multiplying_a_cone(x: int, y: int) -> int:
    return (x << 100000) * (y * y * y + 5)


def an_oversize_shift_proving_a_guard(x: int, y: int) -> int:
    return y * y if (x << 100000) != 0 else y + 1


def test_a_count_no_single_round_settles_needs_the_fixpoint() -> None:
    # One substitution round leaves the outer count standing: the select is undecidable until the inner shift is
    # settled, so only a second round reaches it.
    mir = _select(a_count_two_rounds_make_constant)
    assert _mnemonics(mir) == [], "no operator may survive: both shifts are settled and the rest is dead"
    assert _run(MirInterpreter(mir), 3, 5) == [0]


def test_a_count_a_runtime_select_keeps_is_still_a_value_the_machine_must_hold() -> None:
    # The negative pin: a genuinely runtime count is settled by nothing, so the literal really is a value the
    # machine must hold.
    with pytest.raises(UnsupportedConstruct, match="100000 does not fit"):
        _select(a_count_a_runtime_select_keeps)


def test_an_oversize_shift_takes_the_cone_it_multiplies_with_it() -> None:
    # MIR has no DCE of its own, so before the substitution moved into HIR the absorbed cone was still emitted.
    mir = _select(an_oversize_shift_multiplying_a_cone)
    assert _mnemonics(mir) == []
    assert build_lir(mir, "cone").instances == []
    assert _run(MirInterpreter(mir), 3, 5) == [0]


def test_an_oversize_shift_proves_the_guard_that_reads_it() -> None:
    mir = _select(an_oversize_shift_proving_a_guard)
    assert _mnemonics(mir) == ["iadds"], "only the taken arm may survive"
    for y in (0, 1, -7, 12345):
        assert _run(MirInterpreter(mir), 3, y) == [y + 1]


def a_nameless_quotient_the_word_erases(x: int) -> int:
    return (x << 100000) * (5 // 0)


def a_nameless_quotient_a_settled_guard_excludes(x: int, y: int) -> int:
    r = y
    if (x << 100000) != 0:
        r = 5 // 0
    return r


def a_negative_count_the_word_erases(x: int, y: int) -> int:
    return (x << 100000) * (y << -1)


def a_quotient_only_the_word_names(x: int) -> int:
    return 7 // (x << 100000)


@pytest.mark.parametrize(
    "target,args,expected",
    [
        (a_nameless_quotient_the_word_erases, (3,), 0),
        (a_nameless_quotient_a_settled_guard_excludes, (3, 5), 5),
        (a_negative_count_the_word_erases, (3, 5), 0),
    ],
    ids=lambda value: getattr(value, "__name__", str(value)),
)
def test_what_the_word_erases_is_no_longer_judged(
    target: Callable[..., object], args: tuple[int, ...], expected: int
) -> None:
    # Each was refused before the word could speak. Judging before the substitution rounds convicts the compiler of
    # expressions its own machine erases.
    assert _run(MirInterpreter(_select(target)), *args) == [expected]


def test_what_only_the_word_names_is_judged_after_all() -> None:
    # Why the judgement cannot stay in HIR: nothing names this quotient until the word settles its divisor.
    with pytest.raises(holoso.SynthesisError, match="names no number"):
        _select(a_quotient_only_the_word_names)


def a_count_past_every_carrier(x: int) -> int:
    return (1 << 2**63) + x


def test_a_count_no_host_can_shift_by_is_settled_all_the_same() -> None:
    # CPython raises on this shift, so the fold names no number and only the word can answer it. The rule answers
    # zero for every operand, the count included, and zero is what the machine computes.
    assert _run(MirInterpreter(_select(a_count_past_every_carrier)), 7) == [7]


def a_right_shift_over_a_value_no_word_holds(x: int) -> int:
    """The shifted value is a select until the word settles the guard, and what it settles to is out of range."""
    payload = MAX + 1
    return (payload if (x << 100000) == 0 else x) >> 100000


def test_a_right_shift_is_clamped_where_its_operand_is_a_machine_value() -> None:
    # Clamped in HIR the round after this select folds, the shift answered ``(MAX + 1) >> 15 == 1`` against the 0
    # both the machine and CPython give: the clamp holds only for a value the word already holds.
    assert _run(MirInterpreter(_select(a_right_shift_over_a_value_no_word_holds)), 0) == [0]


def a_shift_past_the_word_on_a_latch_arm(x: int, n: int) -> int:
    acc = x
    t = n
    while t > 0:
        acc = (acc << 100000) + 1
        t = t - 1
    return acc


def test_the_word_reaches_a_value_carried_across_a_back_edge() -> None:
    # A loop-header phi is opened early and closed after every block, so it is the one rebuild path the ordinary
    # walk never takes.
    interpreter = MirInterpreter(_select(a_shift_past_the_word_on_a_latch_arm))
    for x, n in ((7, 0), (7, 1), (7, 3), (-9, 2)):
        assert _run(interpreter, x, n) == [1 if n > 0 else x]


@pytest.mark.parametrize(
    "target,count",
    [(shift_by_nothing, 0), (shift_by_one, 1), (shift_by_the_top_bit, 15), (shift_by_the_whole_word, 16)],
)
@pytest.mark.parametrize("x", [0, 1, -1, 5, -5, 12345, MIN, MAX])
def test_a_folded_shift_answers_exactly_as_the_runtime_shifter_does(
    target: Callable[..., object], count: int, x: int
) -> None:
    """Folding removes hardware without changing the answer: it agrees with the module driven by the same count."""
    assert _run(MirInterpreter(_select(target)), x) == _run(MirInterpreter(_select(shift_pair)), x, count)


@pytest.mark.parametrize("x", [0, 1, -1, 5, -5, 12345, MIN, MAX])
def test_a_shift_past_every_word_answers_where_it_used_to_be_refused(x: int) -> None:
    """The shifter cannot be handed this count -- it is not a representable operand -- so only the fold answers it."""
    assert _run(MirInterpreter(_select(shift_past_every_word)), x) == [_wrap(x << 100000), x >> 100000]


@pytest.mark.parametrize(
    "target,selected,constants",
    [
        (shift_by_nothing, [], []),
        (shift_by_one, ["ishiftc", "ishiftc"], []),
        (shift_by_the_top_bit, ["ishiftc", "ishiftc"], []),
        (shift_by_the_whole_word, ["ishiftc"], [0]),
        (shift_past_every_word, ["ishiftc"], [0]),
        (shift_by_a_count_used_as_a_value, ["iadds", "ishiftc"], [3]),
    ],
)
def test_a_constant_shift_count_costs_neither_a_module_nor_a_pooled_constant(
    target: Callable[..., object], selected: list[str], constants: list[int]
) -> None:
    """
    What each fold costs. The count itself must not reach MIR either -- past the format it is not representable --
    unless something else reads it, which the last row keeps honest.
    """
    mir = _select(target)
    assert _mnemonics(mir) == selected
    assert sorted(node.value for node in mir.nodes.values() if isinstance(node, MirIntConst)) == constants


def a_shift_and_a_sum_over_one_huge_literal(x: int) -> tuple[int, int]:
    return x << 100000, x + 100000


def test_a_literal_too_wide_for_the_machine_is_still_refused_where_it_is_read_as_a_value() -> None:
    """The fold carries an over-wide count because nothing reads it; one the kernel also adds is an ordinary value."""
    with pytest.raises(UnsupportedConstruct, match=r"100000 does not fit"):
        _select(a_shift_and_a_sum_over_one_huge_literal)


def diamond_over_zero_shifts(x: int, y: int, c: bool) -> int:
    if c:
        r = ((x << 0) >> 0) << 0
    else:
        r = ((y << 0) >> 0) << 0
    return r


def diamond_over_bare_operands(x: int, y: int, c: bool) -> int:
    if c:
        r = x
    else:
        r = y
    return r


@pytest.mark.parametrize("budget", [0, 2, 8])
def test_a_shift_by_nothing_is_not_charged_against_the_if_conversion_budget(budget: int) -> None:
    """An arm of zero-count shifts must if-convert wherever the same arm written without them does."""
    assert _mnemonics(_select_with_budget(diamond_over_zero_shifts, budget)) == _mnemonics(
        _select_with_budget(diamond_over_bare_operands, budget)
    )


def _shift_both_ways(width: int, count: int) -> Mir:
    """Both directions by one constant count on a machine of the given integer width, built without the front end."""
    options = Options(OperatorOptions(), ffmt=FMT, wint_min=width)
    builder = HirBuilder()
    builder.block()
    x = builder.input("x", HirIntType())
    shamt = builder.int_const(count)
    builder.output("l", builder.operation(IntShiftLeft(), [x, shamt]))
    builder.output("r", builder.operation(IntShiftRight(), [x, shamt]))
    builder.ret()
    return lower_to_mir(builder.finish(), build_ops(options), options.ffmt, options.ifmt, options.ifconv_max_ops)


@pytest.mark.parametrize("width", [16, 24, 33])
def test_the_word_the_fold_clamps_to_is_the_machine_word_and_not_a_fixed_one(width: int) -> None:
    """A machine other than this module's 16-bit one is what keeps the two width-stated bounds off a constant."""
    limit = 1 << (width - 1)
    for count in (1, width - 1, width, width + 1, 100000):
        interpreter = MirInterpreter(_shift_both_ways(width, count))
        for x in (0, 1, -1, 12345, -limit, limit - 1):
            left = ((x << count) + limit) % (2 * limit) - limit
            assert _run(interpreter, x) == [left, x >> count], (width, count, x)


def test_a_float_only_kernel_reaching_an_integer_operator_still_synthesizes() -> None:
    """``float(math.floor(x))`` folds the conversion pair away, so the built machine is float-only throughout."""
    holoso.synthesize(rounded_to_float, KERNEL_OPTIONS, name="RoundedToFloat")


def rounded_to_int(x: float) -> int:
    return int(round(x))


def floored_to_int(x: float) -> int:
    return int(math.floor(x))


def ceiled_to_int(x: float) -> int:
    return int(math.ceil(x))


def truncated_to_int(x: float) -> int:
    """``math.trunc`` already answers an integer, so the float-valued truncation comes from a conversion round trip."""
    return int(float(int(x)))


def negated_then_floored_to_int(x: float) -> int:
    return int(math.floor(-x))


def floored_for_two_readers(x: float) -> tuple[int, float]:
    return int(math.floor(x)), math.floor(x) + 1.0


_ROUNDINGS = [0.0, 0.5, -0.5, 1.5, -1.5, 2.5, -2.5, 3.75, -3.75, 7.0, -7.0, 100.25, -100.25]


@pytest.mark.parametrize(
    "target,mode",
    [
        (rounded_to_int, RoundMode.NEAREST_EVEN),
        (floored_to_int, RoundMode.FLOOR),
        (ceiled_to_int, RoundMode.CEIL),
        (truncated_to_int, RoundMode.TRUNC),
        (negated_then_floored_to_int, RoundMode.FLOOR),
    ],
)
def test_a_conversion_carries_its_rounding_as_a_mode_rather_than_a_second_module(
    target: Callable[..., object], mode: RoundMode
) -> None:
    """A rounding feeding a conversion is a field on it, not a module before it; the last row folds a sign inside."""
    mir = _select(target)
    assert _mnemonics(mir) == ["ftoint"]
    (operation,) = [node for node in mir.nodes.values() if isinstance(node, MirOperation)]
    assert operation.immediates == (int(mode),)
    for x in _ROUNDINGS:
        assert _run(MirInterpreter(mir), x) == _expected(target, x)


def test_a_rounding_another_reader_observes_is_still_emitted_beside_the_conversion() -> None:
    """
    A second reader adds the standalone rounding rather than cancelling the absorption. Asserting on the conversion's
    OPERAND is what tells that apart from a conversion gated on exclusivity, whose values agree in this format.
    """
    mir = _select(floored_for_two_readers)
    assert _mnemonics(mir) == ["fadd", "fround", "ftoint"]
    (conversion,) = _operations(mir, "ftoint")
    (rounding,) = _operations(mir, "fround")
    assert conversion.operands == rounding.operands, "the conversion reads the value, not the rounding's result"
    assert conversion.immediates == rounding.immediates
    for x in _ROUNDINGS:
        assert _run(MirInterpreter(mir), x) == _expected(floored_for_two_readers, x)


def test_a_sign_applied_after_the_rounding_blocks_the_absorption() -> None:
    """
    The conversion's operand conditioner applies before it rounds, so a negation applied after the rounding cannot
    move there -- ``-floor(x)`` is not ``floor(-x)``. The front end sinks such a negation to the integer side, which
    is why the shape is built directly rather than written in Python.
    """
    builder = HirBuilder()
    builder.block()
    x = builder.input("x", HirFloatType())
    floored = builder.operation(FloatFloor(), [x])
    builder.output("y", builder.operation(FloatToInt(), [builder.operation(FloatNeg(), [floored])]))
    builder.ret()
    mir = lower_to_mir(
        builder.finish(),
        build_ops(KERNEL_OPTIONS),
        KERNEL_OPTIONS.ffmt,
        KERNEL_OPTIONS.ifmt,
        KERNEL_OPTIONS.ifconv_max_ops,
    )
    assert _mnemonics(mir) == ["fround", "ftoint"]
    for value in _ROUNDINGS:
        assert _run(MirInterpreter(mir), value) == [int(-math.floor(value))]


def test_an_absorbed_rounding_is_still_emitted_for_the_reader_that_survives_it() -> None:
    """
    A cancelling sign chain is gone before selection sees it, so the conversion absorbs the rounding after all -- and
    the rounding is nonetheless emitted, because a second output observes it. Each then rounds the value on its own.
    """
    builder = HirBuilder()
    builder.block()
    x = builder.input("x", HirFloatType())
    floored = builder.operation(FloatFloor(), [x])
    negated = builder.operation(FloatNeg(), [floored])
    builder.output("y", builder.operation(FloatToInt(), [builder.operation(FloatNeg(), [negated])]))
    builder.output("n", negated)
    builder.ret()
    mir = lower_to_mir(
        builder.finish(),
        build_ops(KERNEL_OPTIONS),
        KERNEL_OPTIONS.ffmt,
        KERNEL_OPTIONS.ifmt,
        KERNEL_OPTIONS.ifconv_max_ops,
    )
    assert _mnemonics(mir) == ["fround", "ftoint"]
    (conversion,) = [n for n in mir.nodes.values() if isinstance(n, MirOperation) and n.operator.mnemonic == "ftoint"]
    (rounding,) = [n for n in mir.nodes.values() if isinstance(n, MirOperation) and n.operator.mnemonic == "fround"]
    assert conversion.immediates == (int(RoundMode.FLOOR),), "the conversion must carry the mode, not read the result"
    assert conversion.operands == rounding.operands, "both must round the same value independently"
    for value in _ROUNDINGS:
        assert _run(MirInterpreter(mir), value) == [int(math.floor(value)), -math.floor(value)]


def truncated_and_floored(x: float) -> tuple[int, int]:
    return int(x), int(math.floor(x))


def test_two_conversions_over_one_value_stay_apart_on_their_modes_alone() -> None:
    """One operator, operand, conditioner and port, differing only in an immediate: they must not fuse."""
    mir = _select(truncated_and_floored)
    conversions = _operations(mir, "ftoint")
    assert [c.immediates for c in conversions] == [(int(RoundMode.TRUNC),), (int(RoundMode.FLOOR),)]
    assert len({c.operands[0] for c in conversions}) == 1
    lir = build_lir(mir, "truncated_and_floored")
    assert [inst.operator.mnemonic for inst in lir.instances] == ["ftoint"]
    assert len(_wide_firings(lir)) == 2
    for x in _ROUNDINGS:
        assert _run(MirInterpreter(mir), x) == _expected(truncated_and_floored, x)


def test_a_rounding_that_only_a_conversion_reads_needs_no_rounding_operator_configured() -> None:
    """Absorbed, the rounding is never selected, so a kernel that only converts one no longer demands ``fround``."""
    without_fround = replace(KERNEL_OPTIONS, operator=replace(KERNEL_OPTIONS.operator, fround=None))
    mir = lower_to_mir(
        lower_frontend(floored_to_int).hir,
        build_ops(without_fround),
        without_fround.ffmt,
        without_fround.ifmt,
        without_fround.ifconv_max_ops,
    )
    assert _mnemonics(mir) == ["ftoint"]
    with pytest.raises(UnsupportedConstruct, match=r"'fround'"):
        lower_to_mir(
            lower_frontend(floored_for_two_readers).hir,
            build_ops(without_fround),
            without_fround.ffmt,
            without_fround.ifmt,
            without_fround.ifconv_max_ops,
        )


def shifted_to_zero_then_added(x: int, y: int) -> int:
    return y + (x << 100000)


def shifted_to_zero_then_multiplied(x: int, y: int) -> int:
    return y * (x << 100000)


@pytest.mark.parametrize(
    "target,expected,pool",
    [(shifted_to_zero_then_added, 7, []), (shifted_to_zero_then_multiplied, 0, [_int(0)])],
)
def test_the_zero_a_shift_folds_to_is_absorbed_by_what_reads_it(
    target: Callable[..., object], expected: int, pool: list[IntValue]
) -> None:
    """The word writes the zero into HIR, where the declared algebra that absorbs it already lives."""
    mir = _select(target)
    assert _mnemonics(mir) == []
    assert build_lir(mir, target.__name__).wide_consts == pool  # the product IS the zero; the sum drops it entirely
    assert _run(MirInterpreter(mir), 3, 7) == [expected]
