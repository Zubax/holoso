"""
Black-box integer synthesis through the public API: every kernel drives `synthesize` and the numerical model
against CPython or independent literals. Selection facts are asserted through the one public spelling they have --
`holoso_<mnemonic> #(` instantiations present or absent in `verilog_output.verilog` -- and refusal diagnostics
verbatim. `wint_min` alone pins the machine word of a float-free kernel, so the vectors that depend on it say so; the
inline operators (`ishiftc`, `ibwand`, ...) have no public name, so their selection lives in
`test_int_selection`.
"""

import dataclasses
import math
import pickle
import re
from collections.abc import Callable

import pytest

import holoso

_OPTIONS = holoso.Options(holoso.OperatorOptions())

_INT16 = holoso.Options(
    holoso.OperatorOptions(
        fadd=holoso.FAddOptions(),
        fmul=holoso.FMulOptions(),
        fcmp=holoso.FCmpOptions(),
        fround=holoso.FRoundOptions(),
        ffromint=holoso.FFromIntOptions(),
        ftoint=holoso.FToIntOptions(),
    ),
    ffmt=holoso.FloatFormat(5, 11),
)

MIN, MAX = -32768, 32767  # spelled out: the reference must not share the code under test


def _clamp(value: int) -> int:
    return min(max(value, MIN), MAX)


def _wrap(value: int) -> int:
    return ((value + 32768) % 65536) - 32768


def _as_int(value: object) -> int:
    assert isinstance(value, holoso.IntValue)
    return int(value)


def _plain(value: object) -> int | float | bool:
    match value:
        case bool():
            return value
        case holoso.IntValue():
            return int(value)
        case holoso.FloatValue():
            return float(value)
    raise AssertionError(value)


def _run(sim: holoso.NumericalSimulator, *args: int | float | bool) -> list[int | float | bool]:
    return [_plain(value) for value in sim.run(*args)]


def _expected(target: Callable[..., object], *args: int | float | bool) -> list[object]:
    result = target(*args)
    return list(result) if isinstance(result, tuple) else [result]


def _modules(result: holoso.SynthesisResult) -> list[str]:
    return sorted(set(re.findall(r"holoso_(\w+) #", result.verilog_output.verilog)))


def _elaborate(target: Callable[..., object], name: str) -> holoso.NumericalSimulator:
    return holoso.synthesize(target, _INT16, name=name).numerical_model.elaborate()


class Accumulator:
    def __init__(self) -> None:
        self.total = 0

    def step(self, x: int) -> int:
        self.total = self.total + x
        return abs(self.total)


@pytest.fixture(scope="module")
def accumulator_result() -> holoso.SynthesisResult:
    return holoso.synthesize(Accumulator().step, _OPTIONS, name="IntAccumulator")


def test_an_integer_state_kernel_synthesizes_and_the_model_matches_python(
    accumulator_result: holoso.SynthesisResult,
) -> None:
    fmt = accumulator_result.int_format
    model = pickle.loads(pickle.dumps(accumulator_result.numerical_model))  # the blob a generated bench embeds
    assert [(port.name, port.scalar_type) for port in model.inputs] == [("x", holoso.IntType(fmt))]
    assert [(port.name, port.scalar_type) for port in model.outputs] == [
        ("out_0", holoso.IntType(fmt)),
        ("state_total", holoso.IntType(fmt)),
    ]
    sim = model.elaborate()
    total = 0
    for x in [5, -12, 7, fmt.max, fmt.max, -3, fmt.min, fmt.min, 100]:
        total = fmt.saturate(total + x)
        assert [_as_int(value) for value in sim.run(x)] == [fmt.saturate(abs(total)), total]


def mixed_constants(x: float, n: int) -> tuple[float, int, int]:
    """`1` beside `1.0` pools a FloatValue and an IntValue side by side; `7` reads at a non-value index."""
    return x + 1.0, n + 1, n + 7


def test_a_mixed_family_kernel_synthesizes_end_to_end() -> None:
    options = holoso.Options(holoso.OperatorOptions(fadd=holoso.FAddOptions()))
    result = holoso.synthesize(mixed_constants, options, name="MixedConstants")
    sim = result.numerical_model.elaborate()
    word = result.int_format
    for x, n in [(2.5, 40), (-1.0, -3), (0.0, word.max)]:
        out_x, out_n1, out_n7 = sim.run(x, n)
        assert isinstance(out_x, holoso.FloatValue)
        assert (float(out_x), _as_int(out_n1), _as_int(out_n7)) == (
            x + 1.0,
            word.saturate(n + 1),
            word.saturate(n + 7),
        )


class PhaseDecimator:
    def __init__(self) -> None:
        self.acc = 0

    def step(self, x: int) -> tuple[int, int]:
        self.acc = (self.acc + x * 4) % 4096
        return self.acc // 16, self.acc * -1


def test_power_of_two_strength_reduction_survives_synthesis_and_matches_python() -> None:
    """
    Every constant power-of-two rewrite in one stateful kernel -- the saturating scaling, the mask, the inline
    shift and the negation -- driven through `synthesize` against the same class running in CPython. The vectors
    stay short of the rails, where the machine's answer and CPython's coincide exactly.
    """
    # x*4 reaches 400_000 on these vectors, and no float lends this kernel a word, so it asks for one that holds it.
    options = dataclasses.replace(_OPTIONS, wint_min=24)
    result = holoso.synthesize(PhaseDecimator().step, options, name="PhaseDecimator")
    sim = result.numerical_model.elaborate()
    reference = PhaseDecimator()
    for x in [5, -12, 4095, -4096, 100_000, -100_000, 0, 77, -1]:
        expected = list(reference.step(x))
        assert [_as_int(value) for value in sim.run(x)] == expected + [reference.acc], x


def divmod_pair(a: int, b: int) -> tuple[int, int]:
    return a // b, a % b


def test_a_divider_kernel_model_matches_python() -> None:
    """The RTL cosim shares the model's LIR, so a swapped quotient/remainder lane is visible only upstream."""
    result = holoso.synthesize(divmod_pair, _OPTIONS, name="DivmodPair")
    sim = result.numerical_model.elaborate()
    fmt = result.int_format
    for a, b in [(7, 3), (-7, 3), (7, -3), (-7, -3), (17, 5), (fmt.max, 2), (fmt.min, 7), (0, 5)]:
        assert [_as_int(value) for value in sim.run(a, b)] == [a // b, a % b], (a, b)
    assert [_as_int(value) for value in sim.run(fmt.min, -1)] == [fmt.max, 0]
    for a in (5, -5, 0):  # a rail by the numerator's sign, keeping the numerator as the remainder
        assert [_as_int(value) for value in sim.run(a, 0)] == [fmt.min if a < 0 else fmt.max, a]


def test_an_integer_input_rejects_a_bool(accumulator_result: holoso.SynthesisResult) -> None:
    """`bool` is an `int` subclass, so a silently float-free miscoercion -- not a crash -- is the failure mode."""
    sim = accumulator_result.numerical_model.elaborate()
    with pytest.raises(TypeError, match=r"input 0 must be IntValue or int"):
        sim.run(True)


# ----------------------------------------------------------------------------------------------------------------
# The integer vocabulary, comparisons, casts and state, promoted from the retired MIR-level module.


def integer_vocabulary(a: int, b: int) -> tuple[int, int, int, int, int, int, int, int, int]:
    return a + b, a - b, a * b, abs(a), -b, a & b, a | b, a ^ b, ~a


def test_the_integer_vocabulary_answers_as_cpython_does_at_the_rails() -> None:
    """Arithmetic saturates at the rails; bitwise combination cannot leave the word, so nothing saturates there."""
    sim = _elaborate(integer_vocabulary, "IntVocabulary")
    vectors = [(0, 0), (7, 3), (-7, 3), (MAX, 1), (MIN, -1), (MAX, MAX), (MIN, MIN), (MIN, MAX), (-1, -1)]
    vectors += [(0x0F0F, 0x00FF), (-12345, 6789), (MAX, MIN), (12345, 0), (MIN, 0), (1, MIN)]
    for a, b in vectors:
        expected = [_clamp(a + b), _clamp(a - b), _clamp(a * b), _clamp(abs(a)), _clamp(-b), a & b, a | b, a ^ b, ~a]
        assert _run(sim, a, b) == expected, (a, b)


def six_relations(a: int, b: int) -> tuple[bool, bool, bool, bool, bool, bool]:
    return a < b, a <= b, a == b, a != b, a >= b, a > b


def test_every_relation_answers_and_stays_boolean() -> None:
    sim = _elaborate(six_relations, "SixRelations")
    for a, b in [(1, 2), (2, 2), (3, 2), (MIN, MAX), (MAX, MIN)]:
        assert _run(sim, a, b) == [a < b, a <= b, a == b, a != b, a >= b, a > b], (a, b)


def mux_and_casts(a: int, b: int, c: bool) -> tuple[int, bool, int]:
    return (a if c else b), bool(a), int(c) + a


def inverted_pick(c: bool, a: int, b: int) -> int:
    return a if not c else b


def test_a_select_and_the_boolean_casts_cross_the_families() -> None:
    sim = _elaborate(mux_and_casts, "MuxAndCasts")
    for a, b, c in [(5, 6, True), (5, 6, False), (0, -3, True), (MIN, MAX, False), (MIN, MAX, True)]:
        assert _run(sim, a, b, c) == _expected(mux_and_casts, a, b, c), (a, b, c)
    inverted = _elaborate(inverted_pick, "InvertedPick")
    for c, a, b in [(True, 3, -4), (False, 3, -4), (True, MIN, MAX), (False, MIN, MAX)]:
        assert _run(inverted, c, a, b) == [_expected(inverted_pick, c, a, b)[0]], (c, a, b)


def family_crossings(x: float, n: int) -> tuple[int, float]:
    return int(x), float(n)


def test_a_conversion_pair_crosses_the_family_boundary_and_types_its_ports() -> None:
    """The typed-port metadata is public, so a wide carrier that lost its scalar family fails right here."""
    result = holoso.synthesize(family_crossings, _INT16, name="FamilyCrossings")
    ifmt, ffmt = result.int_format, _INT16.ffmt
    assert [(p.name, p.scalar_type) for p in result.input_ports] == [
        ("in_x", holoso.FloatType(ffmt)),
        ("in_n", holoso.IntType(ifmt)),
    ]
    assert [(p.name, p.scalar_type) for p in result.output_ports] == [
        ("out_0", holoso.IntType(ifmt)),
        ("out_1", holoso.FloatType(ffmt)),
    ]
    sim = result.numerical_model.elaborate()
    for x, n in [(3.75, -4), (-3.75, 100), (0.0, 0)]:
        assert _run(sim, x, n) == [int(x), float(n)], (x, n)


def cross_boundary(f: float) -> tuple[float, int]:
    """`float(int(f) & 0xFF)` -- an integer constant, both mixed-format operators, and an inline bitwise between."""
    masked = int(f) & 0xFF
    return float(masked), masked


def test_an_integer_constant_and_both_conversions_cross_in_one_kernel() -> None:
    sim = _elaborate(cross_boundary, "CrossBoundary")
    for value in (0.0, 3.75, 300.0, -3.75, -300.0):
        expected = int(value) & 0xFF  # CPython truncates toward zero too
        assert _run(sim, value) == [float(expected), expected], value


def scaled_by_a_constant_power_of_two(x: float) -> float:
    return x * 4.0


def test_a_constant_power_of_two_float_scaling_reads_the_scaler_not_the_multiplier() -> None:
    options = holoso.Options(
        holoso.OperatorOptions(
            fadd=holoso.FAddOptions(), fmul=holoso.FMulOptions(), fmul_ilog2=holoso.FMulILog2Options()
        ),
        ffmt=holoso.FloatFormat(5, 11),
    )
    result = holoso.synthesize(scaled_by_a_constant_power_of_two, options, name="ScaledByConst")
    assert _modules(result) == ["fmul_ilog2"]
    sim = result.numerical_model.elaborate()
    for x in (1.5, -0.25, 0.0, 100.0):
        assert _run(sim, x) == [x * 4.0], x


class RunningTotal:
    """The declared reset is NONZERO, so loading the reset literal is distinguishable from clearing the register."""

    def __init__(self) -> None:
        self.total = 7

    def step(self, x: int) -> int:
        was = self.total
        self.total = was + x
        return was


def test_a_state_slot_loads_its_declared_reset_and_carries_across_transactions() -> None:
    result = holoso.synthesize(RunningTotal().step, _INT16, name="RunningTotal")
    assert [(p.name, p.scalar_type) for p in result.output_ports] == [
        ("out_0", holoso.IntType(result.int_format)),
        ("state_total", holoso.IntType(result.int_format)),
    ]
    sim = result.numerical_model.elaborate()
    reference = RunningTotal()
    for x in (1, 2, 3, 4, -10, 9000, -12):
        assert _run(sim, x) == [reference.step(x), reference.total], x
    sim.reset()
    assert _run(sim, 0) == [7, 7]


class RunningMaximum:
    def __init__(self) -> None:
        self.peak = -32768

    def step(self, value: int) -> int:
        if value > self.peak:
            self.peak = value
        return self.peak


def test_a_branch_merge_carries_integer_state() -> None:
    sim = _elaborate(RunningMaximum().step, "RunningMaximum")
    reference = RunningMaximum()
    for value in (3, -7, 3, 100, 99, MAX, MIN):
        assert _run(sim, value) == [reference.step(value)], value
    sim.reset()
    assert _run(sim, MIN) == [MIN]


def swap_loop(x: int, y: int, n: int) -> tuple[int, int]:
    while n > 0:
        x, y, n = y, x, n - 1
    return x, y


def test_integer_phis_swap_in_parallel_across_a_back_edge() -> None:
    result = holoso.synthesize(swap_loop, _INT16, name="SwapLoop")
    assert result.initiation_interval[1] is None, "the swap must survive as a real data-dependent loop"
    sim = result.numerical_model.elaborate()
    for trips in (0, 1, 2, 3, 7, 20, -4):
        swapped = max(trips, 0) % 2 == 1  # a non-positive count never enters the body
        assert _run(sim, 11, -22, trips) == [-22 if swapped else 11, 11 if swapped else -22], trips


class MixedState:
    def __init__(self) -> None:
        self.count = 0
        self.level = 0.0

    def step(self, n: int, x: float) -> tuple[int, float]:
        self.count = self.count + n
        self.level = self.level + x
        return self.count, self.level


def test_a_kernel_mixing_integer_and_float_state_keeps_them_apart() -> None:
    result = holoso.synthesize(MixedState().step, _INT16, name="MixedState")
    # The typed ports pin the scalar families: Python `==` over the values alone conflates 3 and 3.0.
    assert [(port.name, port.scalar_type) for port in result.output_ports] == [
        ("state_count", holoso.IntType(result.int_format)),
        ("state_level", holoso.FloatType(_INT16.ffmt)),
    ]
    sim = result.numerical_model.elaborate()
    reference = MixedState()
    for n, x in ((3, 0.5), (-10, 0.25), (4, -1.0)):
        assert _run(sim, n, x) == _expected(reference.step, n, x), (n, x)
    sim.reset()
    reference = MixedState()
    assert _run(sim, 1, 0.5) == _expected(reference.step, 1, 0.5)


def boundary_outputs(a: int, b: int) -> tuple[int, int, int]:
    """Three integer outputs, one of them computed first and then left idle while a long chain runs."""
    return a + b, a - b, ((a * b) * (a - b)) * (a + 2) * (b + 3)


def test_each_integer_output_still_holds_its_own_value_at_the_boundary() -> None:
    """
    The register an output taps must not be recycled before the boundary reads it; getting it wrong is silent, the
    port simply reads a stranger's value. The vectors make every producer answer differently, rails included.
    """
    sim = _elaborate(boundary_outputs, "BoundaryOutputs")
    for a, b in [(3, 5), (7, -9), (MAX, 1), (MIN, -1), (2, 2), (0, 1)]:
        chain = _clamp(_clamp(_clamp(_clamp(a * b) * _clamp(a - b)) * _clamp(a + 2)) * _clamp(b + 3))
        assert _run(sim, a, b) == [_clamp(a + b), _clamp(a - b), chain], (a, b)


# ----------------------------------------------------------------------------------------------------------------
# Shifts: the runtime shifters, the constant folds, and the machine-word substitution fixpoint.


def shift_pair(x: int, n: int) -> tuple[int, int]:
    return x << n, x >> n


@pytest.fixture(scope="module")
def shift_pair_sim() -> holoso.NumericalSimulator:
    return _elaborate(shift_pair, "ShiftPair")


def test_a_shift_answers_as_python_does_once_the_word_truncates_it(shift_pair_sim: holoso.NumericalSimulator) -> None:
    """
    The shifter clamps its count at the word, which is exactly where an unbounded shift stops saying anything new:
    a right shift past the word is the sign fill CPython also gives, and a left shift past it drops every bit either
    way. So CPython is the reference for every non-negative count, with only the left shift's wrap applied.
    """
    for x in (0, 1, -1, 5, -5, 12345, -12345, MIN, MAX):
        for n in (0, 1, 3, 14, 15, 16, 17, 40):
            assert _run(shift_pair_sim, x, n) == [_wrap(x << n), x >> n], (x, n)


@pytest.mark.parametrize(
    "x,n,expected",
    [(1, -2, [0, 4]), (-8, -2, [-2, -32]), (12345, -20, [0, 0]), (-12345, -3, [-1544, 32312]), (-12345, -20, [-1, 0])],
)
def test_a_negative_runtime_shift_count_reverses_the_direction(
    shift_pair_sim: holoso.NumericalSimulator, x: int, n: int, expected: list[int]
) -> None:
    """
    CPython refuses a negative count; each shifter is total over every representable one and reads it as its other
    direction. A kernel reaches this only through a value it did not constrain, so the hardware's answer is the
    definition -- there is no other.
    """
    assert _run(shift_pair_sim, x, n) == expected


def _fixed_shift(count: int) -> Callable[[int], tuple[int, int]]:
    def fixed_shift(x: int) -> tuple[int, int]:
        return x << count, x >> count

    return fixed_shift


@pytest.mark.parametrize("count", [0, 1, 15, 16])
def test_a_folded_shift_answers_exactly_as_the_runtime_shifter_does(
    shift_pair_sim: holoso.NumericalSimulator, count: int
) -> None:
    """Folding removes hardware without changing the answer: both agree with CPython at the shared count."""
    sim = _elaborate(_fixed_shift(count), f"FixedShift{count}")
    for x in (0, 1, -1, 5, -5, 12345, MIN, MAX):
        want = [_wrap(x << count), x >> count]
        assert _run(sim, x) == want, (count, x)
        assert _run(shift_pair_sim, x, count) == want, (count, x)


def shift_past_every_word(x: int) -> tuple[int, int]:
    """A count no machine word can hold: legal Python, and only the fold can carry it to an answer."""
    return x << 100000, x >> 100000


def test_a_shift_past_every_word_answers_where_it_used_to_be_refused() -> None:
    sim = _elaborate(shift_past_every_word, "PastEveryWord")
    for x in (0, 1, -1, 5, -5, 12345, MIN, MAX):
        assert _run(sim, x) == [_wrap(x << 100000), x >> 100000], x


@pytest.mark.parametrize("width", [16, 24, 33])
def test_the_word_the_fold_clamps_to_is_the_machine_word_and_not_a_fixed_one(width: int) -> None:
    """A machine other than this module's 16-bit one is what keeps the two width-stated bounds off a constant."""
    options = dataclasses.replace(_INT16, wint_min=width)
    limit = 1 << (width - 1)
    for count in (1, width - 1, width, width + 1, 100000):
        sim = holoso.synthesize(_fixed_shift(count), options, name=f"Word{width}By{count}").numerical_model.elaborate()
        for x in (0, 1, -1, 12345, -limit, limit - 1):
            left = ((x << count) + limit) % (2 * limit) - limit
            assert _run(sim, x) == [left, x >> count], (width, count, x)


def a_count_past_every_carrier(x: int) -> int:
    return (1 << 2**63) + x


def test_a_count_no_host_can_shift_by_is_settled_all_the_same() -> None:
    """CPython raises on this shift, so only the word can answer it: zero for every operand, the count included."""
    assert _run(_elaborate(a_count_past_every_carrier, "PastEveryCarrier"), 7) == [7]


def a_right_shift_over_a_value_no_word_holds(x: int) -> int:
    """The shifted value is a select until the word settles the guard, and what it settles to is out of range."""
    payload = MAX + 1
    return (payload if (x << 100000) == 0 else x) >> 100000


def test_a_right_shift_is_clamped_where_its_operand_is_a_machine_value() -> None:
    # Clamped in HIR the round after this select folds, the shift answered `(MAX + 1) >> 15 == 1` against the 0
    # both the machine and CPython give: the clamp holds only for a value the word already holds.
    assert _run(_elaborate(a_right_shift_over_a_value_no_word_holds, "ClampedOperand"), 0) == [0]


def a_shift_past_the_word_on_a_latch_arm(x: int, n: int) -> int:
    acc = x
    t = n
    while t > 0:
        acc = (acc << 100000) + 1
        t = t - 1
    return acc


def test_the_word_reaches_a_value_carried_across_a_back_edge() -> None:
    sim = _elaborate(a_shift_past_the_word_on_a_latch_arm, "LatchArmShift")
    for x, n in ((7, 0), (7, 1), (7, 3), (-9, 2)):
        assert _run(sim, x, n) == [1 if n > 0 else x], (x, n)


def a_count_two_rounds_make_constant(x: int, y: int) -> int:
    """The count is a select until the guard above it is settled, which only the first substitution round does."""
    guard = x << 100000
    n = 100000 if guard == 0 else 1
    return y << n


def test_a_count_no_single_round_settles_needs_the_fixpoint() -> None:
    result = holoso.synthesize(a_count_two_rounds_make_constant, _INT16, name="Fixpoint")
    assert _modules(result) == [], "no operator may survive: both shifts are settled and the rest is dead"
    assert _run(result.numerical_model.elaborate(), 3, 5) == [0]


def an_oversize_shift_multiplying_a_cone(x: int, y: int) -> int:
    return (x << 100000) * (y * y * y + 5)


def test_an_oversize_shift_takes_the_cone_it_multiplies_with_it() -> None:
    result = holoso.synthesize(an_oversize_shift_multiplying_a_cone, _INT16, name="AbsorbedCone")
    assert _modules(result) == [], "the multiplier and adder cone must be dead, not merely unread"
    assert _run(result.numerical_model.elaborate(), 3, 5) == [0]


def an_oversize_shift_proving_a_guard(x: int, y: int) -> int:
    return y * y if (x << 100000) != 0 else y + 1


def test_an_oversize_shift_proves_the_guard_that_reads_it() -> None:
    result = holoso.synthesize(an_oversize_shift_proving_a_guard, _INT16, name="ProvenGuard")
    assert _modules(result) == ["iadds"], "only the taken arm may survive"
    sim = result.numerical_model.elaborate()
    for y in (0, 1, -7, 12345):
        assert _run(sim, 3, y) == [y + 1], y


def a_nameless_quotient_the_word_erases(x: int) -> int:
    return (x << 100000) * (5 // 0)


def a_nameless_quotient_a_settled_guard_excludes(x: int, y: int) -> int:
    r = y
    if (x << 100000) != 0:
        r = 5 // 0
    return r


def a_negative_count_the_word_erases(x: int, y: int) -> int:
    return (x << 100000) * (y << -1)


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
    assert _run(_elaborate(target, "Erased"), *args) == [expected]


def a_quotient_only_the_word_names(x: int) -> int:
    return 7 // (x << 100000)


def test_what_only_the_word_names_is_judged_after_all() -> None:
    # Why the judgement cannot stay in HIR: nothing names this quotient until the word settles its divisor.
    with pytest.raises(holoso.SynthesisError, match="names no number"):
        holoso.synthesize(a_quotient_only_the_word_names, _INT16, name="NamesNoNumber")


def shifted_to_zero_then_added(x: int, y: int) -> int:
    return y + (x << 100000)


def shifted_to_zero_then_multiplied(x: int, y: int) -> int:
    return y * (x << 100000)


@pytest.mark.parametrize(
    "target,expected",
    [(shifted_to_zero_then_added, 7), (shifted_to_zero_then_multiplied, 0)],
    ids=lambda value: getattr(value, "__name__", str(value)),
)
def test_the_zero_a_shift_folds_to_is_absorbed_by_what_reads_it(target: Callable[..., object], expected: int) -> None:
    """The word writes the zero into HIR, where the declared algebra that absorbs it already lives."""
    result = holoso.synthesize(target, _INT16, name="AbsorbedZero")
    assert _modules(result) == []
    assert _run(result.numerical_model.elaborate(), 3, 7) == [expected]


def shift_left_by_a_negative_constant(x: int) -> int:
    return x << -1


def shift_right_by_a_negative_constant(x: int) -> int:
    return x >> -1


def shift_by_a_count_a_loop_phi_carries(x: int, n: int) -> int:
    # An optimizer that stops before the phi folds leaves a runtime count, and the shifter reverses it into `x >> 1`.
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
    with pytest.raises(holoso.UnsupportedConstruct, match=r"shift count -1 is negative; Python has no such shift"):
        holoso.synthesize(target, _INT16, name="NegativeCount")


def a_count_a_runtime_select_keeps(y: int, c: bool) -> int:
    return y << (100000 if c else 3)


def test_a_count_a_runtime_select_keeps_is_still_a_value_the_machine_must_hold() -> None:
    # The negative pin: a genuinely runtime count is settled by nothing, so the literal really is a value the
    # machine must hold.
    with pytest.raises(holoso.UnsupportedConstruct, match="100000 does not fit"):
        holoso.synthesize(a_count_a_runtime_select_keeps, _INT16, name="RuntimeCount")


def a_shift_and_a_sum_over_one_huge_literal(x: int) -> tuple[int, int]:
    return x << 100000, x + 100000


def test_a_literal_too_wide_for_the_machine_is_still_refused_where_it_is_read_as_a_value() -> None:
    """The fold carries an over-wide count because nothing reads it; one the kernel also adds is an ordinary value."""
    with pytest.raises(holoso.UnsupportedConstruct, match=r"100000 does not fit"):
        holoso.synthesize(a_shift_and_a_sum_over_one_huge_literal, _INT16, name="HugeLiteral")


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
    options = dataclasses.replace(_INT16, ifconv_max_ops=budget)
    zero_shifts = holoso.synthesize(diamond_over_zero_shifts, options, name="ZeroShiftDiamond")
    bare = holoso.synthesize(diamond_over_bare_operands, options, name="BareDiamond")
    assert zero_shifts.initiation_interval == bare.initiation_interval
    zero_sim, bare_sim = zero_shifts.numerical_model.elaborate(), bare.numerical_model.elaborate()
    for x, y, c in [(3, -4, True), (3, -4, False), (MIN, MAX, True), (MIN, MAX, False)]:
        assert _run(zero_sim, x, y, c) == _run(bare_sim, x, y, c) == [x if c else y], (x, y, c)


def shift_left_only(x: int, n: int) -> int:
    return x << n


def shift_right_only(x: int, n: int) -> int:
    return x >> n


def test_a_right_shift_costs_exactly_what_a_left_shift_costs() -> None:
    """
    The point of a second shifter: a right shift no longer negates its count, so it pays neither the subtractor nor
    the cycles that dependency cost. Matching latency alone would pass if both regressed, so the module each
    kernel builds is pinned too.
    """
    left = holoso.synthesize(shift_left_only, _INT16, name="LeftOnly")
    right = holoso.synthesize(shift_right_only, _INT16, name="RightOnly")
    assert _modules(left) == ["ishl"] and _modules(right) == ["ishr"]
    assert left.initiation_interval == right.initiation_interval
    left_sim, right_sim = left.numerical_model.elaborate(), right.numerical_model.elaborate()
    for x, n in [(5, 2), (-5, 2), (12345, 15), (MIN, 1)]:
        assert _run(left_sim, x, n) == [_wrap(x << n)] and _run(right_sim, x, n) == [x >> n], (x, n)


# ----------------------------------------------------------------------------------------------------------------
# Constant power-of-two strength reduction: values through the model, area through module presence/absence.


def _scale_by(k: int) -> Callable[[int], int]:
    factor: int = 2**k

    def scaled(x: int) -> int:
        return x * factor

    return scaled


@pytest.mark.parametrize("k", [1, 2, 14, 15, 16, 40])
def test_a_power_of_two_scaling_reads_the_shifter_where_it_saturates(k: int) -> None:
    """
    The one thing separating this operator from `x << k`: a multiplication rails where the raw shift drops what
    leaves the word. The count is unbounded where the word is not, so every one past the width rails the same way.
    """
    result = holoso.synthesize(_scale_by(k), _INT16, name=f"ScaleBy{k}")
    assert _modules(result) == ["ishl"], "the scaling must ride the shifter, not a multiplier"
    sim = result.numerical_model.elaborate()
    for x in (0, 1, -1, 3, -3, 1000, -1000, MIN, MAX):
        assert _run(sim, x) == [_clamp(x * 2**k)], (k, x)


def times_eight(x: int) -> int:
    return x * 8


def eighth(x: int) -> int:
    return x // 8


def eighth_remainder(x: int) -> int:
    return x % 8


def past_the_word_quotient(x: int) -> int:
    return x // 2**40


def past_the_word_product(x: int) -> int:
    return x * 2**40


def negated_by_product(x: int) -> int:
    return x * -1


def test_a_minted_power_of_two_product_saturates_like_the_multiplication() -> None:
    """Strength reduction hands `x * 8` to the shifter's saturating tap, so the rails answer as the product."""
    result = holoso.synthesize(times_eight, _INT16, name="TimesEight")
    assert _modules(result) == ["ishl"]
    sim = result.numerical_model.elaborate()
    for x in (0, 1, -1, 5, -5, -8, 12345, -12345, MIN, MAX):
        assert _run(sim, x) == [_clamp(x * 8)], x


def test_a_minted_power_of_two_quotient_is_one_inline_shift() -> None:
    """`x // 8` pays neither the divider nor any module: the arithmetic shift IS the floor division."""
    result = holoso.synthesize(eighth, _INT16, name="Eighth")
    assert _modules(result) == []
    sim = result.numerical_model.elaborate()
    for x in (0, 1, -1, 5, -5, -8, 12345, -12345, MIN, MAX):
        assert _run(sim, x) == [x // 8], x


def test_a_minted_power_of_two_remainder_is_the_mask() -> None:
    """`x % 8` is the two's-complement mask, negative dividends included, with no divider error sideband."""
    result = holoso.synthesize(eighth_remainder, _INT16, name="EighthRemainder")
    assert _modules(result) == []
    sim = result.numerical_model.elaborate()
    for x in (0, 1, -1, 5, -5, -8, 12345, -12345, MIN, MAX):
        assert _run(sim, x) == [x % 8], x


def test_a_minted_quotient_past_the_word_is_the_sign_fill() -> None:
    """The clamp at the top bit answers exactly the floor of an in-word operand over so large a divisor."""
    sim = _elaborate(past_the_word_quotient, "PastWordQuotient")
    for x in (MAX, 1, 0, -1, MIN):
        assert _run(sim, x) == [x // 2**40], x


def test_a_minted_product_past_the_word_rails_by_sign() -> None:
    """A count past the width rails every nonzero operand, exactly as the multiplication it stands for would."""
    result = holoso.synthesize(past_the_word_product, _INT16, name="PastWordProduct")
    assert _modules(result) == ["ishl"]
    sim = result.numerical_model.elaborate()
    for x in (MAX, 1, 0, -1, MIN):
        assert _run(sim, x) == [_clamp(x * 2**40)], x


def boundary_remainder(x: int) -> int:
    return x % 2**15


def boundary_quotient(x: int) -> int:
    return x // 2**15


def test_the_boundary_exponent_builds_where_its_divisor_constant_could_not() -> None:
    """
    `2**15` fits no int16 word, so the spelled-out divisor is refused at selection; the mask and the shift the
    rewrites leave are in-word and exact, so the boundary exponent builds and answers as CPython does.
    """
    remainder_sim = _elaborate(boundary_remainder, "BoundaryRemainder")
    quotient_sim = _elaborate(boundary_quotient, "BoundaryQuotient")
    for x in (0, 1, -1, 5, -5, 12345, -12345, MIN, MAX):
        assert _run(remainder_sim, x) == [x % 2**15], x
        assert _run(quotient_sim, x) == [x // 2**15], x


def test_a_product_with_minus_one_negates_on_the_subtractor() -> None:
    result = holoso.synthesize(negated_by_product, _INT16, name="NegatedByProduct")
    assert _modules(result) == ["isubs"]
    sim = result.numerical_model.elaborate()
    assert _run(sim, MIN) == [MAX], "the negation saturates at the rail"
    assert _run(sim, MAX) == [MIN + 1]


# ----------------------------------------------------------------------------------------------------------------
# Float-to-integer conversions: the rounding rides the conversion as a mode, never a second module.


def rounded_to_int(x: float) -> int:
    return int(round(x))


def floored_to_int(x: float) -> int:
    return int(math.floor(x))


def ceiled_to_int(x: float) -> int:
    return int(math.ceil(x))


def truncated_to_int(x: float) -> int:
    """`math.trunc` already answers an integer, so the float-valued truncation comes from a conversion round trip."""
    return int(float(int(x)))


def negated_then_floored_to_int(x: float) -> int:
    return int(math.floor(-x))


def negated_crossing(x: float) -> int:
    return int(-x)


def floored_for_two_readers(x: float) -> tuple[int, float]:
    return int(math.floor(x)), math.floor(x) + 1.0


def truncated_and_floored(x: float) -> tuple[int, int]:
    return int(x), int(math.floor(x))


_ROUNDINGS = [0.0, 0.5, -0.5, 1.5, -1.5, 2.5, -2.5, 3.75, -3.75, 7.0, -7.0, 100.25, -100.25]


@pytest.mark.parametrize(
    "target",
    [rounded_to_int, floored_to_int, ceiled_to_int, truncated_to_int, negated_then_floored_to_int, negated_crossing],
    ids=lambda value: getattr(value, "__name__", str(value)),
)
def test_a_conversion_carries_its_rounding_as_a_mode_rather_than_a_second_module(
    target: Callable[..., object],
) -> None:
    """
    Only the conversion is instantiated: no `holoso_fround` (the rounding is a mode field) and no
    `holoso_isubs` (a sign folds onto the conversion's operand rather than costing a module).
    """
    result = holoso.synthesize(target, _INT16, name="Conv")
    assert _modules(result) == ["ftoint"]
    sim = result.numerical_model.elaborate()
    for x in _ROUNDINGS:
        assert _run(sim, x) == _expected(target, x), x


def test_a_rounding_another_reader_observes_is_still_emitted_beside_the_conversion() -> None:
    """A second reader adds the standalone rounding rather than cancelling the absorption."""
    result = holoso.synthesize(floored_for_two_readers, _INT16, name="TwoReaders")
    assert _modules(result) == ["fadd", "fround", "ftoint"]
    sim = result.numerical_model.elaborate()
    for x in _ROUNDINGS:
        assert _run(sim, x) == _expected(floored_for_two_readers, x), x


def test_two_conversions_over_one_value_stay_apart_on_their_modes_alone() -> None:
    """One shared instance, two firings: the outputs differ wherever truncation and floor do."""
    result = holoso.synthesize(truncated_and_floored, _INT16, name="TruncAndFloor")
    assert result.verilog_output.verilog.count("holoso_ftoint #") == 1
    assert _modules(result) == ["ftoint"]
    sim = result.numerical_model.elaborate()
    for x in _ROUNDINGS:
        assert _run(sim, x) == _expected(truncated_and_floored, x), x


def test_a_rounding_that_only_a_conversion_reads_needs_no_rounding_operator_configured() -> None:
    """Absorbed, the rounding is never selected, so a kernel that only converts one no longer demands `fround`."""
    without_fround = holoso.Options(
        holoso.OperatorOptions(
            fadd=holoso.FAddOptions(), ffromint=holoso.FFromIntOptions(), ftoint=holoso.FToIntOptions()
        ),
        ffmt=holoso.FloatFormat(5, 11),
    )
    holoso.synthesize(floored_to_int, without_fround, name="NoFround")
    with pytest.raises(holoso.UnsupportedConstruct, match=r"'fround'"):
        holoso.synthesize(floored_for_two_readers, without_fround, name="NoFroundTwoReaders")


def rounded_to_float(x: float) -> float:
    return float(math.floor(x)) + 1.0


def test_a_float_only_kernel_reaching_an_integer_operator_still_synthesizes() -> None:
    """`float(math.floor(x))` folds the conversion pair away, so the built machine is float-only throughout."""
    holoso.synthesize(rounded_to_float, _INT16, name="RoundedToFloat")


def popcount_of(x: int) -> int:
    return x.bit_count()


def popcount_parity(x: int) -> bool:
    return bool((x & 0xFF).bit_count() & 1)


def test_the_population_count_counts_the_magnitude_on_its_own_module() -> None:
    """
    `int.bit_count()` is the one spelling, and it counts the magnitude exactly as CPython does -- so the rails
    answer 1 rather than the width, and no absolute value is spent reaching them.
    """
    result = holoso.synthesize(popcount_of, _INT16, name="Popcount")
    assert _modules(result) == ["ipopcnt"], "the count is one pooled module and nothing else"
    sim = result.numerical_model.elaborate()
    for x in [0, 1, -1, 7, -7, 255, MIN, MIN + 1, MAX, -12345]:
        assert _run(sim, x) == [abs(x).bit_count()], x


def test_the_parity_of_a_masked_byte_rides_the_count() -> None:
    parity = holoso.synthesize(popcount_parity, _INT16, name="PopcountParity")
    assert _modules(parity) == ["ipopcnt"], "the mask and the parity bit are inline; only the count is a module"
    sim = parity.numerical_model.elaborate()
    for x in [0, 1, -1, 0xFF, 0x7F, 256, MIN, MAX]:
        assert _run(sim, x) == [bool((x & 0xFF).bit_count() & 1)], x


def popcount_of_a_literal(x: int) -> int:
    return (0x1234).bit_count() + x


def test_a_static_population_count_folds_away() -> None:
    result = holoso.synthesize(popcount_of_a_literal, _INT16, name="PopcountStatic")
    assert _modules(result) == ["iadds"], "a count of a literal is answered before hardware is selected"
    assert _run(result.numerical_model.elaborate(), 3) == [(0x1234).bit_count() + 3]
