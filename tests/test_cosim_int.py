"""
RTL-vs-model cosimulation of integer kernels: corpus FSM shapes, the fused divider at the rails via explicit
vectors, and the bench's own bounded integer sweep.
"""

import dataclasses
import re
from collections.abc import Callable

import pytest

import holoso
from holoso import FloatFormat, IntFormat, Options
from ._cosim import run_cosim
from ._eel_corpus import INT_CASES, rows
from ._eeloracle import InputRow
from ._modelref import adder_modes, default_options
from .hdl.hdl_float_oracle import SIMULATORS
from .test_int_selection import countdown
from .test_int_synthesis import (
    difference_and_order,
    difference_only,
    divmod_pair,
    every_rounding_both_ways,
    order_beside_reversed_difference,
    order_only,
    popcount_of,
    shift_pair,
    shift_right_only,
    sum_difference_and_order,
    sum_only,
)

# NcoPhase sums a 2**30 increment over a 32-bit mask, so exactness needs at least a 34-bit word.
_OPTIONS = dataclasses.replace(default_options(FloatFormat(wexp=6, wman=18)), wint_min=34)
_IFMT = IntFormat(_OPTIONS.wint_min)  # these kernels carry no float, so the floor alone is the machine word

# A deliberately small state-machine subset of the corpus; the full matrix is owned model-vs-CPython by
# the integer acceptance suite, and this checks the same witnesses model-vs-RTL.
_CORPUS_SUBSET = [pytest.param(case, id=case[0]) for case in INT_CASES if case[0] in ("crc8", "pwm", "nco_phase")]


@pytest.mark.cosim
@pytest.mark.parametrize("case", _CORPUS_SUBSET)
@pytest.mark.parametrize("sim", SIMULATORS)
def test_int_corpus_cosim(sim: str, case: tuple[str, Callable[[], Callable[..., object]], list[InputRow]]) -> None:
    name, factory, vectors = case
    run_cosim(sim, holoso.synthesize(factory(), _OPTIONS, name=f"{name}_int"), vectors=vectors)


def divmod_rails(a: int, b: int) -> tuple[int, int, int]:
    return a // b, a % b, a + b


@pytest.mark.cosim
@pytest.mark.parametrize("sim", SIMULATORS)
def test_int_popcount_cosim(sim: str) -> None:
    """
    The count port is narrower than the word it lands in, so the emitted top zero-extends it. The bench's own sweep
    draws small operands whose counts never set the port's top bit, which cannot tell a zero fill from a sign fill:
    only a large-magnitude vector discriminates, so the rails are driven explicitly.
    """
    values = [0, 1, -1, 7, -7, _IFMT.min, _IFMT.min + 1, _IFMT.max, _IFMT.max - 1, -12345]
    count_width = (_IFMT.width - 1).bit_length()
    assert _IFMT.max.bit_count() >= 1 << (count_width - 1), "a rail must set the count port's top bit"
    result = holoso.synthesize(popcount_of, _OPTIONS, name="popcount_int")
    run_cosim(sim, result, vectors=[{"x": x} for x in values])


@pytest.mark.cosim
@pytest.mark.parametrize("sim", SIMULATORS)
def test_int_divmod_and_rails_cosim(sim: str) -> None:
    """The pooled divider's fused quotient/remainder firing plus a saturating add, driven to the format rails."""
    pairs = [(7, 3), (-7, 3), (7, -3), (-7, -3), (_IFMT.min, -1), (_IFMT.max, _IFMT.max), (_IFMT.min, 1), (0, 5)]
    result = holoso.synthesize(divmod_rails, _OPTIONS, name="divmod_rails_int")
    run_cosim(sim, result, vectors=[{"a": a, "b": b} for a, b in pairs])


def int_float_crossing(x: float, n: int) -> tuple[int, float]:
    return int(round(x)) + n, float(n) + x


def _crossing_options() -> Options:
    operator = dataclasses.replace(
        _OPTIONS.operator,
        frint=holoso.FRintOptions(),
        ffromint=holoso.FFromIntOptions(),
    )
    return dataclasses.replace(_OPTIONS, operator=operator)


@pytest.mark.cosim
@pytest.mark.parametrize("sim", SIMULATORS)
def test_int_float_crossing_cosim(sim: str) -> None:
    """Pooled frint (rounding carried on its mode port) and ffromint inside one scheduled kernel, random sweep."""
    run_cosim(sim, holoso.synthesize(int_float_crossing, _crossing_options(), name="int_float_crossing"))


@pytest.mark.cosim
@pytest.mark.parametrize("wint_min", (16, 34), ids=("word_equals_float", "word_wider_than_float"))
@pytest.mark.parametrize("sim", SIMULATORS)
def test_a_rounding_answered_as_float_and_integer_cosim(sim: str, wint_min: int) -> None:
    """
    Each firing of the rounder writes a float register and an integer one. Where the integer word outgrows the float,
    the float write fills the register's high bits with don't-cares, which this drives through real RTL.
    """
    options = dataclasses.replace(_crossing_options(), wint_min=wint_min)
    result = holoso.synthesize(every_rounding_both_ways, options, name=f"rounded_both_ways_w{wint_min}")
    assert (result.int_format.width > options.ffmt.width) == (wint_min == 34)
    run_cosim(sim, result)


@pytest.mark.cosim
@pytest.mark.parametrize("sim", SIMULATORS)
def test_int_division_random_sweep_cosim(sim: str) -> None:
    """The default draw once included zero, so a defined x//0 transaction tripped the bench's err_pc assert."""
    run_cosim(sim, holoso.synthesize(divmod_pair, _OPTIONS, name="ratio_int"))


def sat_mix(a: int, b: int) -> tuple[int, int]:
    return a + b, (a * b) ^ b


@pytest.mark.cosim
@pytest.mark.parametrize("sim", SIMULATORS)
def test_int_random_sweep_cosim(sim: str) -> None:
    """`vectors=None` draws the bench's own bounded integer sweep through pooled add/multiply and an inline xor."""
    run_cosim(sim, holoso.synthesize(sat_mix, _OPTIONS, name="sat_mix_int"))


def pow2_strength(x: int) -> tuple[int, int, int, int, int]:
    return x * 4, x // 8, x % 32, x * -1, x // 2**40


@pytest.mark.cosim
@pytest.mark.parametrize("sim", SIMULATORS)
def test_int_pow2_strength_reduction_cosim(sim: str) -> None:
    """
    The minted power-of-two forms in RTL: the inline right shift both in-word and clamped past the word, the mask,
    and negation as a subtraction, beside a power-of-two product, which stays on the multiplier; driven through the
    rails and the negative dividends whose floor/mask behavior the rewrites must preserve.
    """
    values = [0, 1, -1, 7, -7, 8, -8, 31, -33, 4095, -4096, _IFMT.max, _IFMT.min, _IFMT.max // 4 + 1]
    result = holoso.synthesize(pow2_strength, _OPTIONS, name="pow2_strength_int")
    run_cosim(sim, result, vectors=rows("x", values))


# Each operand under the counts that separate the shifter's cases: inside the word, at it and past it, and the
# negative ones, which reverse the direction.
_SHIFT_VECTORS = [
    {"x": x, "n": n}
    for x in (0, 1, -1, 12345, -12345, _IFMT.min, _IFMT.max)
    for n in (
        *(0, 1, 5, _IFMT.width - 1, _IFMT.width, _IFMT.width + 1, 100, _IFMT.max),
        *(-1, -5, 1 - _IFMT.width, -_IFMT.width, -_IFMT.width - 1, -100, _IFMT.min),
    )
]


@pytest.mark.cosim
@pytest.mark.parametrize("sim", SIMULATORS)
def test_int_shift_in_both_directions_cosim(sim: str) -> None:
    """`x << n` and `x >> n` as two firings of one shifter, the microcode switching its direction bit between them."""
    result = holoso.synthesize(shift_pair, _OPTIONS, name="shift_pair_int")
    assert result.verilog_output.verilog.count("holoso_ishft #") == 1
    run_cosim(sim, result, vectors=_SHIFT_VECTORS)


@pytest.mark.cosim
@pytest.mark.parametrize("sim", SIMULATORS)
def test_int_shift_right_alone_cosim(sim: str) -> None:
    """
    A kernel that only shifts right holds the shifter's direction bit constant across the program, so the emitted top
    ties it high instead of driving it from the microcode. Tied low, the module would shift left, which the model --
    never reading the bit -- cannot see.
    """
    result = holoso.synthesize(shift_right_only, _OPTIONS, name="shift_right_int")
    assert re.search(r"\buc_ishft_0_mode\s*=\s*1'd1;", result.verilog_output.verilog), "the premise of this test"
    run_cosim(sim, result, vectors=_SHIFT_VECTORS)


# Sums and differences inside the word and railed in either direction, and orders whose difference overflows.
_ADDER_VECTORS = [
    {"a": a, "b": b}
    for a, b in (
        *((0, 0), (7, 3), (3, 7), (-7, -7), (12345, -6789), (_IFMT.max, 0), (_IFMT.min, 0)),
        *((_IFMT.max, 1), (_IFMT.max, _IFMT.max), (_IFMT.min, -1), (_IFMT.min, _IFMT.min)),
        *((_IFMT.max, _IFMT.min), (_IFMT.min, _IFMT.max), (_IFMT.max, -1), (_IFMT.min, 1), (0, _IFMT.min)),
    )
]


@pytest.mark.cosim
@pytest.mark.parametrize("sim", SIMULATORS)
def test_int_adder_in_every_mode_cosim(sim: str) -> None:
    """A sum, a difference and an order as three firings of one adder, the microcode switching `sub` between them."""
    result = holoso.synthesize(sum_difference_and_order, _OPTIONS, name="adder_modes_int")
    assert adder_modes(result) == [2], "the premise of this test"
    run_cosim(sim, result, vectors=_ADDER_VECTORS)


@pytest.mark.cosim
@pytest.mark.parametrize(
    "target,mode",
    [(sum_only, 0), (difference_only, 1), (order_only, 1), (difference_and_order, 1)],
    ids=["add", "subtract", "compare", "subtract_and_compare"],
)
@pytest.mark.parametrize("sim", SIMULATORS)
def test_int_adder_fixed_to_one_mode_cosim(sim: str, target: Callable[..., object], mode: int) -> None:
    """
    A kernel that only adds, or never adds, has the adder elaborated for that alone. Elaborated for the other, the
    module would compute the wrong operation, which the model -- never reading the parameter -- cannot see.
    """
    result = holoso.synthesize(target, _OPTIONS, name=f"adder_{target.__name__}_int")
    assert adder_modes(result) == [mode], "the premise of this test"
    run_cosim(sim, result, vectors=_ADDER_VECTORS)


@pytest.mark.cosim
@pytest.mark.parametrize("sim", SIMULATORS)
def test_int_comparison_on_a_reversed_subtraction_cosim(sim: str) -> None:
    """The relations tap the flags of `b - a` mirrored, which only the RTL's port wiring can get wrong."""
    result = holoso.synthesize(order_beside_reversed_difference, _OPTIONS, name="reversed_difference_int")
    assert adder_modes(result) == [1] and result.initiation_interval == (5, 5), "the premise of this test"
    run_cosim(sim, result, vectors=_ADDER_VECTORS)


@pytest.mark.cosim
@pytest.mark.parametrize("sim", SIMULATORS)
def test_int_fast_adder_cosim(sim: str) -> None:
    """The equality detector beside the adder is a second arm of the RTL, which the model cannot tell from the first."""
    operators = dataclasses.replace(_OPTIONS.operator, iadds=holoso.IAddsOptions(fast=True))
    result = holoso.synthesize(
        sum_difference_and_order, dataclasses.replace(_OPTIONS, operator=operators), name="fast_adder_int"
    )
    assert ".FAST(1)" in result.verilog_output.verilog, "the premise of this test"
    run_cosim(sim, result, vectors=_ADDER_VECTORS)


@pytest.mark.cosim
@pytest.mark.parametrize("sim", SIMULATORS)
def test_int_counting_loop_random_sweep_cosim(sim: str) -> None:
    """An unbounded draw once handed this counting loop a multi-million-cycle transaction (runaway ceiling)."""
    run_cosim(sim, holoso.synthesize(countdown, _OPTIONS, name="countdown_int"))


def counted_scan(n: int, seed: int) -> tuple[int, int]:
    acc = seed
    last = 0
    for i in range(n):
        if i == 2:
            continue
        acc = acc + i
        last = i
    return acc, last


class CountedState:
    def __init__(self) -> None:
        self.total = 0

    def step(self, n: int) -> int:
        for i in range(n):
            self.total = self.total + i + 1
        return self.total


@pytest.mark.cosim
@pytest.mark.parametrize("sim", SIMULATORS)
def test_int_counted_for_cosim(sim: str) -> None:
    """The counted back-edge for: a runtime trip count with a continue lane, and a state carry across trips."""
    scan = holoso.synthesize(counted_scan, _OPTIONS, name="counted_scan_int")
    run_cosim(sim, scan, vectors=[{"n": 0, "seed": 5}, {"n": 1, "seed": -3}, {"n": 4, "seed": 0}, {"n": 7, "seed": 9}])
    state = holoso.synthesize(CountedState().step, _OPTIONS, name="counted_state_int")
    run_cosim(sim, state, vectors=[{"n": 0}, {"n": 3}, {"n": 1}, {"n": 6}])
