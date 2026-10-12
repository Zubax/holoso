"""
The white-box integer selection sentinels: fabric, latency and resource contracts that no public artifact names.
Values, typed ports and the public initiation interval cannot tell one shared firing from two, an inline primitive
from a module, or a conditioner fold from a survived sign chain -- these few tests pin them directly, while all
value coverage lives in `test_int_synthesis`.
"""

import math
from collections.abc import Callable

import pytest

import holoso
from holoso import (
    FAddOptions,
    FCmpOptions,
    FFromIntOptions,
    FILog2Options,
    FloatFormat,
    FMulILog2Options,
    FMulOptions,
    FRintOptions,
    OperatorOptions,
    Options,
)
import holoso._operators as operators
from holoso._backend.verilog._emit import generate
from holoso._eel import lower as lower_frontend
from holoso._hir import FloatNeg, FloatRounding, Rounding, FloatToInt, FloatType as HirFloatType, HirBuilder
from holoso._lir import Lir, PooledScheduledOp, WideOperand
from holoso._mir import Mir, MirConst, MirOperation, lower as lower_to_mir
from holoso._mir._ir import MirBuilder
from holoso._mir._interpret import MirInterpreter
from holoso._operators import (
    FILog2Operator,
    FILog2Primitive,
    FloatIsFinitePrimitive,
    FloatIsNegInfPrimitive,
    FloatIsPosInfPrimitive,
    FloatSignControl,
    FloatToBoolPrimitive,
    FMulILog2Operator,
    FMulILog2Primitive,
    FRintPrimitive,
    IAddPrimitive,
    ICmpPrimitive,
    IntIdentity,
    ISubPrimitive,
    PooledPrimitive,
    Primitive,
    RoundMode,
)
from holoso._type import FloatType, IntType
from holoso._value import FloatValue, IntValue

from ._modelref import default_ifmt, build_lir, early_install, mir_options, hardware_name, DEFAULT_UNROLL_MAX_TRIPS
from .test_eel_calls import _min_max_of_ints
from .test_int_synthesis import (
    cross_boundary,
    divmod_pair,
    eighth,
    eighth_remainder,
    family_crossings,
    floored_for_two_readers,
    mux_and_casts,
    negated_by_product,
    negated_crossing,
    popcount_of,
    shift_pair,
    times_eight,
    truncated_and_floored,
)

OPTIONS = Options(
    OperatorOptions(
        fadd=FAddOptions(),
        fmul=FMulOptions(),
        fcmp=FCmpOptions(),
        frint=FRintOptions(),
        ffromint=FFromIntOptions(),
    ),
    ffmt=FloatFormat(5, 11),
)


def _select(target: Callable[..., object]) -> Mir:
    return lower_to_mir(lower_frontend(target, DEFAULT_UNROLL_MAX_TRIPS).hir, mir_options(OPTIONS))


def _mnemonics(mir: Mir) -> list[str]:
    return sorted(hardware_name(node.primitive) for node in mir.nodes.values() if isinstance(node, MirOperation))


def _operations(mir: Mir, name: str) -> list[MirOperation]:
    return [
        node for node in mir.nodes.values() if isinstance(node, MirOperation) and hardware_name(node.primitive) == name
    ]


def _rounding(operation: MirOperation) -> RoundMode:
    assert isinstance(operation.primitive, FRintPrimitive)
    return operation.primitive.rounding


def three_relations(a: int, b: int) -> tuple[bool, bool, bool]:
    return a < b, a == b, a > b


def four_relations(a: int, b: int) -> tuple[bool, bool, bool, bool]:
    return a <= b, a != b, a >= b, a > b


def sign_ops(a: int, b: int) -> tuple[int, int]:
    return -a, abs(b)


def bitwise_ops(a: int, b: int) -> tuple[int, int, int, int]:
    return a & b, a | b, a ^ b, ~a


def countdown(n: int) -> int:
    steps = 0
    while n > 0:
        n = n - 3
        steps = steps + 1
    return steps


@pytest.mark.parametrize(
    "target,selected",
    [
        (divmod_pair, ["idivs", "idivs"]),
        (three_relations, ["iadds", "iadds", "iadds"]),
        (sign_ops, ["iabss", "iadds"]),
        (bitwise_ops, ["ibwand", "ibwnot", "ibwor", "ibwxor"]),
        (mux_and_casts, ["iadds", "ifrombool", "itobool", "select"]),
        (_min_max_of_ints, ["iadds", "iadds", "iadds", "imuls", "select", "select"]),
        (family_crossings, ["ffromint", "frint"]),
        (shift_pair, ["ishft", "ishft"]),
        (countdown, ["iadds", "iadds", "iadds"]),
        (times_eight, ["imuls"]),
        (eighth, ["ishiftc"]),
        (eighth_remainder, ["ibwand"]),
        (negated_by_product, ["iadds"]),
        (popcount_of, ["ipopcnt"]),
        (cross_boundary, ["ffromint", "frint", "ibwand"]),
    ],
    ids=lambda value: getattr(value, "__name__", str(value)),
)
def test_the_lowering_names_each_integer_operator_in_one_table(
    target: Callable[..., object], selected: list[str]
) -> None:
    """
    Every primitive the lowering can choose, named in one place: `-x` selects the adder because there is no
    negation module, both shift directions select the one shifter, the strength rewrites pick the
    inline `ishiftc`/`ibwand` no public artifact can name, and `min`/`max` become one compare-and-select
    pair each rather than branches.
    """
    assert _mnemonics(_select(target)) == selected


@pytest.mark.parametrize(
    "target,primitives",
    [
        (three_relations, [ICmpPrimitive, ICmpPrimitive, ICmpPrimitive]),
        (sign_ops, [ISubPrimitive]),
        (mux_and_casts, [IAddPrimitive]),
        (_min_max_of_ints, [IAddPrimitive, ICmpPrimitive, ICmpPrimitive]),
        (countdown, [IAddPrimitive, ICmpPrimitive, ISubPrimitive]),
        (negated_by_product, [ISubPrimitive]),
    ],
    ids=lambda value: getattr(value, "__name__", str(value)),
)
def test_the_lowering_selects_the_adder_primitive_each_use_means(
    target: Callable[..., object], primitives: list[type[PooledPrimitive]]
) -> None:
    """
    The adder's name cannot tell an addition from a subtraction or a comparison, so the primitive each use selects is
    the sentinel: a negation is the subtraction from zero, a relation the comparison, and a sum the addition.
    """
    selected = sorted(type(operation.primitive).__name__ for operation in _operations(_select(target), "iadds"))
    assert selected == [primitive.__name__ for primitive in primitives]


def _wide_firings(lir: Lir) -> list[PooledScheduledOp]:
    return [op for block in lir.blocks for op in block.ops]


def test_the_quotient_and_the_remainder_share_one_divider_firing() -> None:
    """`a // b` beside `a % b` is one activation with two taps -- counted, because fusion is not automatic."""
    lir = build_lir(_select(divmod_pair), "divmod_pair")
    (firing,) = _wide_firings(lir)
    assert [instance.operator.name for instance in lir.instances] == ["idivs"]
    assert sorted(write.result for write in firing.writes) == [0, 1]


def test_relations_fuse_into_one_firing_and_opposite_inversions_split() -> None:
    """
    Three relations, three flags, one activation. A firing taps each port at most once, so `a <= b` and `a > b`
    -- the same flag under opposite inversions -- need an activation each, still bound to the one pooled adder: the
    cost is a cycle, never a module.
    """
    fused = build_lir(_select(three_relations), "three_relations")
    (firing,) = _wide_firings(fused)
    assert [instance.operator.name for instance in fused.instances] == ["iadds"]
    assert isinstance(firing.primitive, ICmpPrimitive) and len(firing.writes) == 3
    split = build_lir(_select(four_relations), "four_relations")
    assert [instance.operator.name for instance in split.instances] == ["iadds"]
    assert [type(firing.primitive) for firing in _wide_firings(split)] == [ICmpPrimitive, ICmpPrimitive]


def strength_mix(x: int, n: int) -> tuple[int, int, int, int]:
    return x * 2, x << n, x // 5, x << 3


def test_strength_selection_keeps_the_multiplier_the_shifter_and_the_divider() -> None:
    """
    One graph holding every strength decision: a power-of-two product is a multiplication like any other (it rails
    where a shift would wrap), the runtime shift takes the shifter, the non-power-of-two quotient still pays the
    divider, and the constant count `3` is an `ishiftc` immediate -- neither a module nor a pooled constant.
    """
    mir = _select(strength_mix)
    assert _mnemonics(mir) == ["idivs", "imuls", "ishft", "ishiftc"]
    lir = build_lir(mir, "strength_mix")
    assert sorted(instance.operator.name for instance in lir.instances) == ["idivs", "imuls", "ishft"]
    assert 3 not in {
        node.value
        for node in mir.nodes.values()
        if isinstance(node, MirConst) and isinstance(node.scalar_type, IntType)
    }


def test_a_runtime_exponent_scaling_carries_mixed_conditioner_lists() -> None:
    """
    The frontend can only spell a STATIC `FloatMulPow2(k)`, which lowering materializes as a constant, so the
    runtime-exponent `fmul_ilog2` contract -- a float port beside an integer port, each with its own conditioner
    algebra -- is reachable only as a hand-built graph.
    """
    fmt = OPTIONS.ffmt
    ifmt = default_ifmt(fmt)  # hand-built, so the word is named rather than settled
    builder = MirBuilder(fmt, ifmt)
    builder.block()
    k = builder.input("k", IntType(ifmt))
    builder.output(
        "scaled",
        builder.operation(
            FMulILog2Primitive(FMulILog2Operator.build(fmt, ifmt, FMulILog2Options())),
            [builder.const(1.5, FloatType(fmt)), k],
            [FloatSignControl(), IntIdentity()],
        ),
    )
    builder.ret()
    interpreter = MirInterpreter(builder.finish())
    for exponent in (2, -3, 0):
        assert interpreter.run(exponent) == [FloatValue.from_float(fmt, 1.5 * 2.0**exponent)], exponent


def test_exponent_extraction_places_the_limit_cases_outside_the_finite_span() -> None:
    """
    A hand-built graph reaches the primitive directly, without the hypotenuse expansion that is its only source in a
    real kernel. The limit answers carry the weight: zero and an infinity must fall below and above every finite
    exponent, or an
    extremum taken over the answer -- which is how a scaled magnitude picks its normalizing exponent -- would need a
    special case for each of them.
    """
    fmt = OPTIONS.ffmt
    ifmt = default_ifmt(fmt)
    bias = (1 << (fmt.wexp - 1)) - 1
    builder = MirBuilder(fmt, ifmt)
    builder.block()
    builder.output(
        "exponent",
        builder.operation(
            FILog2Primitive(FILog2Operator.build(fmt, ifmt, FILog2Options())),
            [builder.input("x", FloatType(fmt))],
            [FloatSignControl()],
        ),
    )
    builder.ret()
    interpreter = MirInterpreter(builder.finish())

    def extracted(value: float) -> int:
        (answer,) = interpreter.run(FloatValue.from_float(fmt, value))
        assert isinstance(answer, IntValue)  # the extraction crosses into the integer family
        return answer.value

    assert extracted(0.0) == -bias
    assert extracted(math.inf) == extracted(-math.inf) == bias + 1
    # The finite span's own ends included, since that is where the limit answers must not collide with a real one.
    for magnitude in (1.0, 1.5, 3.0, 0.5, 2.0 ** (1 - bias), 2.0**bias):
        assert extracted(magnitude) == extracted(-magnitude) == math.floor(math.log2(magnitude))
        assert -bias < extracted(magnitude) < bias + 1


def test_a_rounding_and_its_conversion_are_two_taps_of_one_firing() -> None:
    """
    Values cannot distinguish a conversion that reads the already-rounded node in this format, nor one firing from
    two: the operand identity, the rounding modes and the firing count are the contract.
    """
    mir = _select(floored_for_two_readers)
    assert _mnemonics(mir) == ["fadd", "frint", "frint"]
    rounding, conversion = sorted(_operations(mir, "frint"), key=lambda operation: operation.result)
    assert (rounding.result, conversion.result) == (0, 1)
    assert conversion.operands == rounding.operands, "the conversion reads the value, not the rounding's result"
    assert _rounding(conversion) == _rounding(rounding) == RoundMode.FLOOR
    lir = build_lir(mir, "floored_for_two_readers")
    (firing,) = [op for op in _wide_firings(lir) if op.inst.operator.name == "frint"]
    assert sorted(write.result for write in firing.writes) == [0, 1], "the float and the integer leave one firing"


def test_two_roundings_of_one_value_share_the_instance_but_not_a_firing() -> None:
    """One firing runs one rounding mode, so a truncation and a floor of the same operand are two firings."""
    mir = _select(truncated_and_floored)
    conversions = _operations(mir, "frint")
    assert [_rounding(c) for c in conversions] == [RoundMode.TRUNC, RoundMode.FLOOR]
    assert len({c.operands[0] for c in conversions}) == 1
    lir = build_lir(mir, "truncated_and_floored")
    assert [instance.operator.name for instance in lir.instances] == ["frint"]
    assert len(_wide_firings(lir)) == 2


def test_a_negated_operand_folds_onto_the_conversion() -> None:
    """
    `int(-x)` conditions the `frint` float port rather than emitting a sign primitive of its own; the public
    module regex cannot see an inline sign, so the exact mnemonic list is the sentinel.
    """
    assert _mnemonics(_select(negated_crossing)) == ["frint"]


def test_a_sign_applied_after_the_rounding_blocks_the_absorption() -> None:
    """
    The conversion's operand conditioner applies before it rounds, so a negation applied after the rounding cannot
    move there -- `-floor(x)` is not `floor(-x)`. The front end sinks such a negation to the integer side, which
    is why the shape is built directly rather than written in Python.
    """
    builder = HirBuilder()
    builder.block()
    x = builder.input("x", HirFloatType())
    floored = builder.operation(FloatRounding(Rounding.FLOOR), [x])
    builder.output("y", builder.operation(FloatToInt(), [builder.operation(FloatNeg(), [floored])]))
    builder.ret()
    mir = lower_to_mir(builder.finish(), mir_options(OPTIONS))
    assert _mnemonics(mir) == ["frint", "frint"]
    assert len({operation.operands for operation in _operations(mir, "frint")}) == 2, "two firings, not two taps"
    interpreter = MirInterpreter(mir)
    for value in (0.0, 0.5, -0.5, 1.5, -1.5, 2.5, -2.5, 3.75, -3.75, 7.0, -7.0, 100.25, -100.25):
        (converted,) = interpreter.run(value)
        assert isinstance(converted, IntValue) and int(converted) == int(-math.floor(value)), value


class InputLatch:
    def __init__(self) -> None:
        self.prev = 0

    def step(self, x: int, y: int) -> int:
        out = self.prev * y + x * 3 - y * y
        self.prev = x
        return out


def test_a_slot_fed_by_an_integer_input_installs_ahead_of_the_boundary() -> None:
    """
    A register-pressure claim with no public spelling (values and II are unchanged); it belongs beside the
    schedule allocation contracts.
    """
    lir = build_lir(_select(InputLatch().step), "input_latch")
    (slot,) = lir.wide_state_slots
    assert early_install(lir, slot)[1].issue_cycle == 0


def test_an_unconditioned_operand_binds_no_port_and_keeps_none_through_lowering() -> None:
    """
    `filog2` reads the exponent field alone, so its operand is declared unconditioned: no conditioner may reach it
    and the wrapper binds no sign port. The NEGATIVE constant is the case that matters -- the pool stores a float as
    its magnitude and folds the sign onto whoever reads it, below the normalization in the builder, so it is the one
    way a sign could reappear on a port that has none.
    """
    fmt = OPTIONS.ffmt
    ifmt = default_ifmt(fmt)
    primitive = FILog2Primitive(FILog2Operator.build(fmt, ifmt, FILog2Options()))

    for value, expected in ((3.5, 1), (-3.5, 1)):
        builder = MirBuilder(fmt, ifmt)
        builder.block()
        builder.output(
            "exponent",
            builder.operation(primitive, [builder.const(value, FloatType(fmt))], [FloatSignControl()]),
        )
        builder.ret()
        mir = builder.finish()
        assert MirInterpreter(mir).run() == [IntValue.from_int(ifmt, expected)], value

        lir = build_lir(mir, f"filog2_sign_{'neg' if value < 0 else 'pos'}")
        (firing,) = [op for block in lir.blocks for op in block.ops]
        for operand in firing.operands:
            assert isinstance(operand, WideOperand)  # a float port, so the wide bank
            assert operand.conditioner.is_identity, value
        assert ".a_sgnop(" not in generate(lir).verilog, value


def test_a_sign_that_cannot_be_observed_costs_no_conditioner() -> None:
    """
    `bool(x)` reads the exponent alone, so the negation feeding the second call is unobservable. The two must name
    ONE operation and the emitted datapath must carry no sign conditioner at all.
    """

    def kernel(x: float) -> tuple[bool, bool]:
        return bool(x), bool(-x)

    # The conditioner count is the sentinel; `bool(x)` and `bool(-x)` agree for every x, so the values below only
    # confirm that the erasure left the answer intact.
    result = holoso.synthesize(kernel, OPTIONS, name="unobservable_sign")
    assert result.verilog_output.verilog.count("holoso_fsgnop(") == 0
    model = result.numerical_model.elaborate()
    for x in (2.5, -2.5, 0.0):
        assert tuple(bool(value) for value in model.run(x)) == kernel(x), x


def _declaring_classes() -> set[str]:
    """
    Read off the package's exported catalogue, NOT `__subclasses__`, which `@dataclass(slots=True)` makes unusable:
    it builds a new class object and leaves the pre-slots original registered, so every primitive appears twice.
    """
    return {
        name
        for name, value in vars(operators).items()
        if isinstance(value, type) and issubclass(value, Primitive) and value.unconditioned_operands
    }


def test_every_unconditioned_declaration_holds_over_every_result() -> None:
    """
    The compiler ERASES a sign transform on the strength of these declarations and nothing derives them, so each is
    swept directly: all four transforms must leave every result port alone. The expected set is named as well, or
    the sweep would pass vacuously the day a declaration goes missing -- and the directional classifiers, which DO
    read the sign bit, are pinned as declaring nothing.
    """
    fmt = OPTIONS.ffmt
    ifmt = default_ifmt(fmt)
    bias = (1 << (fmt.wexp - 1)) - 1
    declaring = [
        FILog2Primitive(FILog2Operator.build(fmt, ifmt, FILog2Options())),
        FloatToBoolPrimitive(fmt),
        FloatIsFinitePrimitive(fmt),
    ]
    assert all(primitive.unconditioned_operands == frozenset({0}) for primitive in declaring)
    assert FloatIsPosInfPrimitive(fmt).unconditioned_operands == frozenset()
    assert FloatIsNegInfPrimitive(fmt).unconditioned_operands == frozenset()
    # Walked rather than listed, so a fourth declaration cannot slip past this sweep: nothing else would catch it,
    # the numerical model and the RTL both reading the conditioner the compiler erased and so agreeing.
    assert _declaring_classes() == {type(primitive).__name__ for primitive in declaring}

    # Zero under a negation is the reachable NEGATIVE ZERO encoding, which is the case the weaker argument
    # ("no negative zero exists") would have missed. ZKF has no NaN, so none appears.
    magnitudes = [0.0, 1.0, 3.5, 2.0 ** (1 - bias), 2.0**bias, math.inf]
    for primitive in declaring:
        for magnitude in magnitudes:
            base = FloatValue.from_float(fmt, magnitude)
            answers = {
                primitive.evaluate(base.apply_sign(negate=negate, absolute=absolute))
                for negate in (False, True)
                for absolute in (False, True)
            }
            assert len(answers) == 1, (type(primitive).__name__, magnitude, answers)
