"""
RTL-vs-model cosimulation of integer kernels: corpus FSM shapes, the fused divider at the rails via explicit
vectors, and the bench's own bounded integer sweep.
"""

import dataclasses
from collections.abc import Callable

import pytest

import holoso
from holoso import FloatFormat, Options
from ._cosim import run_cosim
from ._eel_corpus import INT_CASES, rows
from ._eeloracle import InputRow
from ._modelref import default_options
from .hdl.hdl_float_oracle import SIMULATORS
from .test_int_selection import countdown
from .test_int_synthesis import divmod_pair

# NcoPhase sums a 2**30 increment over a 32-bit mask, so exactness needs at least a 34-bit word.
_OPTIONS = dataclasses.replace(default_options(FloatFormat(wexp=6, wman=18)), wint_min=34)
_IFMT = _OPTIONS.ifmt

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
        fround=holoso.FRoundOptions(),
        ffromint=holoso.FFromIntOptions(),
        ftoint=holoso.FToIntOptions(),
    )
    return dataclasses.replace(_OPTIONS, operator=operator)


@pytest.mark.cosim
@pytest.mark.parametrize("sim", SIMULATORS)
def test_int_float_crossing_cosim(sim: str) -> None:
    """Pooled ftoint (rounding carried as its immediate) and ffromint inside one scheduled kernel, random sweep."""
    run_cosim(sim, holoso.synthesize(int_float_crossing, _crossing_options(), name="int_float_crossing"))


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
    """``vectors=None`` draws the bench's own bounded integer sweep through pooled add/multiply and an inline xor."""
    run_cosim(sim, holoso.synthesize(sat_mix, _OPTIONS, name="sat_mix_int"))


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
    result = holoso.synthesize(pow2_strength, _OPTIONS, name="pow2_strength_int")
    run_cosim(sim, result, vectors=rows("x", values))


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
