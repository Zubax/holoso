"""
MIR's integer half, driven through :class:`MirInterpreter` on graphs built directly with :class:`MirBuilder`,
because ``_reject_integers`` still refuses every integer before ``synthesize`` can reach one -- hence the
``whitebox`` marker on the whole module.

No expectation here calls back into ``IntValue``: the rails, the division degeneracies and the two shift readings
are literals or CPython's own operators, so a defect in the value layer cannot vouch for itself.
"""

from collections.abc import Callable

import pytest

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
    IShiftOperator,
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
from holoso._value import FloatValue, IntValue

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
    interpreter = MirInterpreter(_binary_graph(IShiftOperator(IFMT), 0, 1))
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
