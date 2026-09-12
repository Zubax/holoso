"""
The white-box integer selection sentinels: fabric, latency and resource contracts that no public artifact names.
Values, typed ports and the public initiation interval cannot tell one shared firing from two, an inline operator
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
    FMulOptions,
    FRoundOptions,
    FToIntOptions,
    FloatFormat,
    OperatorOptions,
    Options,
)
import holoso._operators as operators
from holoso._backend.verilog._emit import generate
from holoso._eel import lower as lower_frontend
from holoso._hir import FloatFloor, FloatNeg, FloatToInt, FloatType as HirFloatType, HirBuilder
from holoso._lir import Lir, PooledScheduledOp, WideEarlyInstall, WideOperand
from holoso._mir import Mir, MirBuilder, MirIntConst, MirInterpreter, MirOperation, lower as lower_to_mir
from holoso._operators import (
    FILog2Operator,
    HardwareOperator,
    FMulILog2Operator,
    FloatIsFiniteOperator,
    FloatIsNegInfOperator,
    FloatIsPosInfOperator,
    FloatSignControl,
    FloatToBoolOperator,
    IntIdentity,
    RoundMode,
)
from holoso._type import FloatType, IntType
from holoso._value import FloatValue, IntValue

from ._modelref import default_ifmt, build_lir, mir_options, DEFAULT_UNROLL_MAX_TRIPS
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
        fround=FRoundOptions(),
        ffromint=FFromIntOptions(),
        ftoint=FToIntOptions(),
    ),
    ffmt=FloatFormat(5, 11),
)


def _select(target: Callable[..., object]) -> Mir:
    return lower_to_mir(lower_frontend(target, DEFAULT_UNROLL_MAX_TRIPS).hir, mir_options(OPTIONS))


def _mnemonics(mir: Mir) -> list[str]:
    return sorted(node.operator.mnemonic for node in mir.nodes.values() if isinstance(node, MirOperation))


def _operations(mir: Mir, mnemonic: str) -> list[MirOperation]:
    return [
        node for node in mir.nodes.values() if isinstance(node, MirOperation) and node.operator.mnemonic == mnemonic
    ]


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
        (three_relations, ["icmp", "icmp", "icmp"]),
        (sign_ops, ["iabss", "isubs"]),
        (bitwise_ops, ["ibwand", "ibwnot", "ibwor", "ibwxor"]),
        (mux_and_casts, ["iadds", "ifrombool", "itobool", "select"]),
        (_min_max_of_ints, ["iadds", "icmp", "icmp", "imuls", "select", "select"]),
        (family_crossings, ["ffromint", "ftoint"]),
        (shift_pair, ["ishl", "ishr"]),
        (countdown, ["iadds", "icmp", "isubs"]),
        (times_eight, ["ishl"]),
        (eighth, ["ishiftc"]),
        (eighth_remainder, ["ibwand"]),
        (negated_by_product, ["isubs"]),
        (popcount_of, ["ipopcnt"]),
    ],
    ids=lambda value: getattr(value, "__name__", str(value)),
)
def test_the_lowering_names_each_integer_operator_in_one_table(
    target: Callable[..., object], selected: list[str]
) -> None:
    """
    Every operator the lowering can choose, named in one place: `-x` selects `isubs` because there is no
    negation module, each shift direction selects the module that names it, the strength rewrites pick the
    inline `ishiftc`/`ibwand` no public artifact can name, and `min`/`max` become one compare-and-select
    pair each rather than branches.
    """
    assert _mnemonics(_select(target)) == selected


def _wide_firings(lir: Lir) -> list[PooledScheduledOp]:
    return [op for block in lir.blocks for op in block.ops]


def test_the_quotient_and_the_remainder_share_one_divider_firing() -> None:
    """`a // b` beside `a % b` is one activation with two taps -- counted, because fusion is not automatic."""
    lir = build_lir(_select(divmod_pair), "divmod_pair")
    (firing,) = _wide_firings(lir)
    assert [instance.operator.mnemonic for instance in lir.instances] == ["idivs"]
    assert sorted(write.port for write in firing.writes) == [0, 1]


def test_relations_fuse_into_one_firing_and_opposite_inversions_split() -> None:
    """
    Three relations, three flags, one activation. A firing taps each port at most once, so `a <= b` and `a > b`
    -- the same flag under opposite inversions -- need an activation each, still bound to the one pooled comparator:
    the cost is a cycle, never a module.
    """
    fused = build_lir(_select(three_relations), "three_relations")
    (firing,) = _wide_firings(fused)
    assert [instance.operator.mnemonic for instance in fused.instances] == ["icmp"]
    assert len(firing.writes) == 3
    split = build_lir(_select(four_relations), "four_relations")
    assert [instance.operator.mnemonic for instance in split.instances] == ["icmp"]
    assert len(_wide_firings(split)) == 2


def strength_mix(x: int, n: int) -> tuple[int, int, int, int]:
    return x * 2, x << n, x // 5, x << 3


def test_strength_selection_shares_the_shifter_and_keeps_the_divider() -> None:
    """
    One graph holding every strength decision: the saturating scale and the raw runtime shift bind to a SINGLE
    `ishl` instance through their opposite taps, the non-power-of-two quotient still pays the divider, and the
    constant count `3` is an `ishiftc` immediate -- neither a module nor a pooled constant.
    """
    mir = _select(strength_mix)
    assert _mnemonics(mir) == ["idivs", "ishiftc", "ishl", "ishl"]
    lir = build_lir(mir, "strength_mix")
    assert sorted(instance.operator.mnemonic for instance in lir.instances) == ["idivs", "ishl"]
    shifter_taps = {
        write.port for op in _wide_firings(lir) if op.inst.operator.mnemonic == "ishl" for write in op.writes
    }
    assert shifter_taps == {0, 1}, "the raw reading and the saturating reading must come off the one module"
    assert 3 not in {node.value for node in mir.nodes.values() if isinstance(node, MirIntConst)}


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
    k = builder.int_input("k", IntType(ifmt))
    builder.float_output(
        "scaled",
        builder.operation(
            FMulILog2Operator(fmt, ifmt, FMulILog2Operator.Options()),
            [builder.float_const(1.5, FloatType(fmt)), k],
            [FloatSignControl(), IntIdentity()],
        ),
    )
    builder.ret()
    interpreter = MirInterpreter(builder.finish())
    for exponent in (2, -3, 0):
        assert interpreter.run(exponent) == [FloatValue.from_float(fmt, 1.5 * 2.0**exponent)], exponent


def test_exponent_extraction_places_the_limit_cases_outside_the_finite_span() -> None:
    """
    A hand-built graph reaches the operator directly, without the hypotenuse expansion that is its only source in a
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
    builder.int_output(
        "exponent",
        builder.operation(
            FILog2Operator(fmt, ifmt, FILog2Operator.Options()),
            [builder.float_input("x", FloatType(fmt))],
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


def test_conversions_share_one_instance_and_read_the_unrounded_value() -> None:
    """
    Values cannot distinguish a conversion that reads the already-rounded node in this format, nor one shared
    configured operator from two instances: the operand identity and the mode immediates are the contract.
    """
    mir = _select(floored_for_two_readers)
    assert _mnemonics(mir) == ["fadd", "fround", "ftoint"]
    (conversion,) = _operations(mir, "ftoint")
    (rounding,) = _operations(mir, "fround")
    assert conversion.operands == rounding.operands, "the conversion reads the value, not the rounding's result"
    assert conversion.immediates == rounding.immediates == (int(RoundMode.FLOOR),)

    mir = _select(truncated_and_floored)
    conversions = _operations(mir, "ftoint")
    assert [c.immediates for c in conversions] == [(int(RoundMode.TRUNC),), (int(RoundMode.FLOOR),)]
    assert len({c.operands[0] for c in conversions}) == 1
    lir = build_lir(mir, "truncated_and_floored")
    assert [instance.operator.mnemonic for instance in lir.instances] == ["ftoint"]
    assert len(_wide_firings(lir)) == 2

    mir = _select(cross_boundary)  # the folded negation's dual: an integer constant crossing beside both conversions
    assert _mnemonics(mir) == ["ffromint", "ftoint", "ibwand"]


def test_a_negated_operand_folds_onto_the_conversion() -> None:
    """
    `int(-x)` conditions the `ftoint` float port rather than emitting a sign operator of its own; the public
    module regex cannot see an inline sign, so the exact mnemonic list is the sentinel.
    """
    assert _mnemonics(_select(negated_crossing)) == ["ftoint"]


def test_a_sign_applied_after_the_rounding_blocks_the_absorption() -> None:
    """
    The conversion's operand conditioner applies before it rounds, so a negation applied after the rounding cannot
    move there -- `-floor(x)` is not `floor(-x)`. The front end sinks such a negation to the integer side, which
    is why the shape is built directly rather than written in Python.
    """
    builder = HirBuilder()
    builder.block()
    x = builder.input("x", HirFloatType())
    floored = builder.operation(FloatFloor(), [x])
    builder.output("y", builder.operation(FloatToInt(), [builder.operation(FloatNeg(), [floored])]))
    builder.ret()
    mir = lower_to_mir(builder.finish(), mir_options(OPTIONS))
    assert _mnemonics(mir) == ["fround", "ftoint"]
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
    assert isinstance(slot.install, WideEarlyInstall) and slot.install.cycle == 1 < lir.initiation_interval


def test_an_unconditioned_operand_binds_no_port_and_keeps_none_through_lowering() -> None:
    """
    `filog2` reads the exponent field alone, so its operand is declared unconditioned: no conditioner may reach it
    and the wrapper binds no sign port. The NEGATIVE constant is the case that matters -- the pool stores a float as
    its magnitude and folds the sign onto whoever reads it, below the normalization in the builder, so it is the one
    way a sign could reappear on a port that has none.
    """
    fmt = OPTIONS.ffmt
    ifmt = default_ifmt(fmt)
    operator = FILog2Operator(fmt, ifmt, FILog2Operator.Options())

    for value, expected in ((3.5, 1), (-3.5, 1)):
        builder = MirBuilder(fmt, ifmt)
        builder.block()
        builder.int_output(
            "exponent",
            builder.operation(operator, [builder.float_const(value, FloatType(fmt))], [FloatSignControl()]),
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
    ONE operation and the emitted datapath must carry no sign conditioner at all; before the operand was declared
    the operand was declared unconditioned this kernel emitted one `holoso_fsgnop` call for a sign that could not
    change the answer.
    """

    def kernel(x: float) -> tuple[bool, bool]:
        return bool(x), bool(-x)

    # The conditioner count is the sentinel; the values below cannot distinguish the fix, `bool(x)` and `bool(-x)`
    # agreeing for every x, and stand only as insurance that the erasure did not disturb the answer.
    result = holoso.synthesize(kernel, OPTIONS, name="unobservable_sign")
    assert result.verilog_output.verilog.count("holoso_fsgnop(") == 0
    model = result.numerical_model.elaborate()
    for x in (2.5, -2.5, 0.0):
        assert tuple(bool(value) for value in model.run(x)) == kernel(x), x


def _declaring_classes() -> set[str]:
    """
    Read off the package's exported catalogue, NOT `__subclasses__`, which `@dataclass(slots=True)` makes unusable:
    it builds a new class object and leaves the pre-slots original registered, so every operator appears twice.
    """
    return {
        name
        for name, value in vars(operators).items()
        if isinstance(value, type) and issubclass(value, HardwareOperator) and value.unconditioned_operands
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
        FILog2Operator(fmt, ifmt, FILog2Operator.Options()),
        FloatToBoolOperator(fmt),
        FloatIsFiniteOperator(fmt),
    ]
    assert all(operator.unconditioned_operands == frozenset({0}) for operator in declaring)
    assert FloatIsPosInfOperator(fmt).unconditioned_operands == frozenset()
    assert FloatIsNegInfOperator(fmt).unconditioned_operands == frozenset()
    # Walked rather than listed, so a fourth declaration cannot slip past this sweep: nothing else would catch it,
    # the numerical model and the RTL both reading the conditioner the compiler erased and so agreeing.
    assert _declaring_classes() == {type(operator).__name__ for operator in declaring}

    # Zero under a negation is the reachable NEGATIVE ZERO encoding, which is the case the weaker argument
    # ("no negative zero exists") would have missed. ZKF has no NaN, so none appears.
    magnitudes = [0.0, 1.0, 3.5, 2.0 ** (1 - bias), 2.0**bias, math.inf]
    for operator in declaring:
        for magnitude in magnitudes:
            base = FloatValue.from_float(fmt, magnitude)
            answers = {
                operator.evaluate(base.apply_sign(negate=negate, absolute=absolute))
                for negate in (False, True)
                for absolute in (False, True)
            }
            assert len(answers) == 1, (operator.mnemonic, magnitude, answers)
