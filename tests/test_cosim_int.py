"""
RTL-vs-model cosimulation of integer kernels: corpus FSM shapes, the fused divider at the rails via explicit
vectors, and the bench's own bounded integer sweep.
"""

import dataclasses

import pytest

import holoso
from holoso import FloatFormat, Options
from ._cosim import run_cosim
from ._eel_corpus import Crc8, NcoPhase, Pwm
from ._modelref import default_options
from .hdl.hdl_float_oracle import SIMULATORS

# NcoPhase sums a 2**30 increment over a 32-bit mask, so exactness needs at least a 34-bit word.
_OPTIONS = dataclasses.replace(default_options(FloatFormat(wexp=6, wman=18)), wint_min=34)
_IFMT = _OPTIONS.ifmt


def _rows(name: str, values: list[int]) -> list[dict[str, int]]:
    return [{name: value} for value in values]


@pytest.mark.cosim
@pytest.mark.parametrize("sim", SIMULATORS)
def test_crc8_cosim(sim: str) -> None:
    vectors = _rows("byte", [0x31, 0x32, 0x33, 0xFF, 0x00, 0x80, 0x01])
    run_cosim(sim, Crc8().step, _OPTIONS, "crc8_int", vectors=vectors)


@pytest.mark.cosim
@pytest.mark.parametrize("sim", SIMULATORS)
def test_pwm_cosim(sim: str) -> None:
    vectors = _rows("duty", [3] * 12 + [0] * 3 + [5] * 6)
    run_cosim(sim, Pwm(top=5).step, _OPTIONS, "pwm_int", vectors=vectors)


@pytest.mark.cosim
@pytest.mark.parametrize("sim", SIMULATORS)
def test_nco_phase_cosim(sim: str) -> None:
    vectors = _rows("increment", [0x40000000] * 5 + [0x3FFFFFFF] * 3 + [1, 0])
    run_cosim(sim, NcoPhase().step, _OPTIONS, "nco_phase_int", vectors=vectors)


def divmod_rails(a: int, b: int) -> tuple[int, int, int]:
    return a // b, a % b, a + b


@pytest.mark.cosim
@pytest.mark.parametrize("sim", SIMULATORS)
def test_int_divmod_and_rails_cosim(sim: str) -> None:
    """The pooled divider's fused quotient/remainder firing plus a saturating add, driven to the format rails."""
    pairs = [(7, 3), (-7, 3), (7, -3), (-7, -3), (_IFMT.min, -1), (_IFMT.max, _IFMT.max), (_IFMT.min, 1), (0, 5)]
    run_cosim(sim, divmod_rails, _OPTIONS, "divmod_rails_int", vectors=[{"a": a, "b": b} for a, b in pairs])


def int_float_crossing(x: float, n: int) -> tuple[int, float]:
    return int(round(x)) + n, float(n) + x


def _crossing_options() -> Options:
    operator = dataclasses.replace(
        _OPTIONS.operator,
        fround=holoso.FRoundOptions(),
        ffromint=holoso.FFromIntOptions(),
        ftoint=holoso.FToIntOptions(),
    )
    return dataclasses.replace(_OPTIONS, operator=operator)


@pytest.mark.cosim
@pytest.mark.parametrize("sim", SIMULATORS)
def test_int_float_crossing_cosim(sim: str) -> None:
    """Pooled ftoint (rounding carried as its immediate) and ffromint inside one scheduled kernel, random sweep."""
    run_cosim(sim, int_float_crossing, _crossing_options(), "int_float_crossing")


def ratio(a: int, b: int) -> tuple[int, int]:
    return a // b, a % b


@pytest.mark.cosim
@pytest.mark.parametrize("sim", SIMULATORS)
def test_int_division_random_sweep_cosim(sim: str) -> None:
    """The default draw once included zero, so a defined x//0 transaction tripped the bench's err_pc assert."""
    run_cosim(sim, ratio, _OPTIONS, "ratio_int")


def sat_mix(a: int, b: int) -> tuple[int, int]:
    return a + b, (a * b) ^ b


@pytest.mark.cosim
@pytest.mark.parametrize("sim", SIMULATORS)
def test_int_random_sweep_cosim(sim: str) -> None:
    """``vectors=None`` draws the bench's own bounded integer sweep through pooled add/multiply and an inline xor."""
    run_cosim(sim, sat_mix, _OPTIONS, "sat_mix_int")


def pow2_strength(x: int) -> tuple[int, int, int, int, int]:
    return x * 4, x // 8, x % 32, x * -1, x // 2**40


@pytest.mark.cosim
@pytest.mark.parametrize("sim", SIMULATORS)
def test_int_pow2_strength_reduction_cosim(sim: str) -> None:
    """
    The minted power-of-two forms in RTL: the shifter's saturating product tap (its first end-to-end reader),
    the inline right shift both in-word and clamped past the word, the mask, and negation on the subtractor,
    driven through the rails and the negative dividends whose floor/mask behavior the rewrites must preserve.
    """
    values = [0, 1, -1, 7, -7, 8, -8, 31, -33, 4095, -4096, _IFMT.max, _IFMT.min, _IFMT.max // 4 + 1]
    run_cosim(sim, pow2_strength, _OPTIONS, "pow2_strength_int", vectors=_rows("x", values))


def countdown(n: int) -> int:
    steps = 0
    while n > 0:
        n = n - 3
        steps = steps + 1
    return steps


@pytest.mark.cosim
@pytest.mark.parametrize("sim", SIMULATORS)
def test_int_counting_loop_random_sweep_cosim(sim: str) -> None:
    """An unbounded draw once handed this counting loop a multi-million-cycle transaction (runaway ceiling)."""
    run_cosim(sim, countdown, _OPTIONS, "countdown_int")
