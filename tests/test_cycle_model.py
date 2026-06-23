"""
Tests for the cycle-accurate :class:`NumericalSimulator`: the per-clock ``tick`` interface, the data-dependent latency
a caller recovers by counting ticks, and bounded-memory execution of an arbitrarily deep loop.

The bit-exactness of the outputs themselves is covered by the broad model-equivalence suites (which call the model
transaction-level via ``run``) and, against the RTL, by the cycle-accurate cosim lockstep. Here we exercise the
cycle-level behaviour: that ``tick`` reaches ``out_valid`` on the cycle the RTL would, that the count is data-dependent
on a branchy/looping kernel, and that a hard loop does not blow up state.
"""

import random
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from holoso import FloatFormat
from holoso._backend.numerical import NumericalSimulator
from holoso._frontend import lower
from holoso._hir import optimize
from holoso._lir import Lir, build
from holoso._mir import lower as lower_to_mir

from ._modelref import default_ops

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import madd  # noqa: E402
import poly3  # noqa: E402
from cordic_sincos import CordicSinCos  # noqa: E402
from ekf1_stateless import update_x_P  # noqa: E402
from iir1_lpf import IIR1LPF  # noqa: E402
from majority_voter import MajorityVoter  # noqa: E402
from octave_index import octave_index  # noqa: E402
from phase_frequency_detector import PhaseFrequencyDetector  # noqa: E402
from quadrature_encoder import QuadratureEncoder  # noqa: E402
from recip_newton import NewtonReciprocal  # noqa: E402
from remainder import remainder  # noqa: E402
from schmitt_trigger import SchmittTrigger  # noqa: E402

_FMT = FloatFormat(8, 36)


def _random_inputs(lir: Lir, rng: random.Random) -> list[object]:
    return [
        bool(rng.randint(0, 1)) if type(load).__name__ == "BoolInputLoad" else rng.uniform(-3.0, 3.0)
        for load in lir.inputs
    ]


def _drive(model: NumericalSimulator, inputs: list[object]) -> tuple[tuple[object, ...], int]:
    """
    Drive one transaction tick by tick (as a cosimulator would), returning the outputs and the in_valid->out_valid
    latency in cycles -- the count from just after the accept edge to out_valid, exactly the cosim bench's ``waited``.
    """
    model.set_inputs(*inputs)
    while not model.in_ready:
        model.tick(in_valid=False, out_ready=True)
    model.tick(in_valid=True, out_ready=False)  # accept: pc 0 -> 1 (the accept edge itself is not part of waited)
    waited = 0
    while not model.out_valid:
        model.tick(in_valid=False, out_ready=False)
        waited += 1
    outputs = model.output_values
    model.tick(in_valid=False, out_ready=True)  # accept the output, advancing the persistent state
    return outputs, waited


@pytest.mark.parametrize("name,factory", [("madd", lambda: madd.madd), ("poly3", lambda: poly3.poly3)])
def test_tick_and_call_agree(name: str, factory: Callable[[], Callable[..., object]]) -> None:
    # The hand-driven tick loop and the run() convenience must produce the same outputs: run() is just a tick
    # driver, and the cycle count it would observe is the loop count below.
    lir = build(lower_to_mir(optimize(lower(factory())), default_ops(_FMT)), name)
    by_tick = NumericalSimulator(lir)
    by_call = NumericalSimulator(lir)
    rng = random.Random(1)
    for _ in range(8):
        inputs = _random_inputs(lir, rng)
        ticked_out, _cycles = _drive(by_tick, inputs)
        called_out = by_call.run(*inputs)
        assert [v.bits for v in ticked_out] == [v.bits for v in called_out]  # type: ignore[union-attr]


@pytest.mark.parametrize("name", ["madd", "poly3", "cordic_sincos", "ekf1_stateless"])
def test_single_path_latency_is_the_static_initiation_interval(name: str) -> None:
    # A kernel with one forward path takes exactly ``initiation_interval - 1`` cycles from the accept edge (pc==1) to
    # out_valid (pc==LASTPC) -- the cycle count a caller recovers by counting ticks, here the static lower bound.
    factory = {
        "madd": lambda: madd.madd,
        "poly3": lambda: poly3.poly3,
        "cordic_sincos": lambda: CordicSinCos().__call__,
        "ekf1_stateless": lambda: update_x_P,
    }[name]
    lir = build(lower_to_mir(optimize(lower(factory())), default_ops(_FMT)), name)
    model = NumericalSimulator(lir)
    rng = random.Random(7)
    for _ in range(8):
        _outputs, cycles = _drive(model, _random_inputs(lir, rng))
        assert cycles == lir.initiation_interval - 1


def test_loop_latency_grows_with_the_trip_count() -> None:
    # Newton's reciprocal iterates until convergence, so a harder input runs more loop trips and a strictly longer
    # transaction -- the data-dependent latency a fixed ``initiation_interval`` cannot express.
    lir = build(lower_to_mir(optimize(lower(NewtonReciprocal().__call__)), default_ops(_FMT)), "recip")
    model = NumericalSimulator(lir)
    latencies = {_drive(model, [x])[1] for x in (0.5, 0.9, 1.3, 1.7, 2.5, 3.5, 6.0, 12.0)}
    assert len(latencies) >= 3, f"loop latency should vary with the trip count, saw {sorted(latencies)}"
    assert min(latencies) > lir.min_initiation_interval, "every realized latency exceeds the not-taken lower bound"


def test_deep_loop_runs_in_bounded_memory() -> None:
    # The per-clock model holds only the register files and a small in-flight buffer, so a loop with a very high trip
    # count executes without unbounded growth. ``count_down`` runs ``n`` iterations; at n=20000 that is hundreds of
    # thousands of ticks completing in bounded memory and time -- a global-timeline design would be O(trips^2).
    lir = build(lower_to_mir(optimize(lower(_count_down)), default_ops(_FMT)), "count_down")
    shallow_cycles = _drive(NumericalSimulator(lir), [10.0])[1]
    model = NumericalSimulator(lir)
    out, deep_cycles = _drive(model, [20000.0])
    assert abs(float(out[0])) <= 1e-3  # type: ignore[arg-type]
    assert deep_cycles > shallow_cycles * 100, "the deep run must genuinely iterate thousands of times"
    # The in-flight landing buffers drain every cycle, so after the transaction they hold nothing -- the state the
    # model retains does not grow with the trip count, which is what keeps an arbitrarily deep loop bounded.
    assert not model._pending  # noqa: SLF001  (the in-flight buffer drains every cycle)


def _count_down(n):  # type: ignore[no-untyped-def]
    # A runtime-trip-count loop: subtract 1.0 until the value is no longer positive. The trip count is the input, so
    # it can be made arbitrarily large to stress the model's bounded state.
    while n > 0.0:
        n = n - 1.0
    return n


# The realized worst-case in_valid->out_valid latency over a FIXED adversarial input sequence per kernel, frozen on the
# B1+B2 build. This is the regression guard for the project's true goal -- realized per-transaction latency in multi-
# block kernels -- which the static last_pc gate alone cannot express: a per-iteration drain regression is amplified by
# the loop trip count (recip_newton, remainder), and a branchy kernel's worst arm may not be its longest static path.
# Each tuple is (factory, input vectors, frozen worst-case waited). The vectors are chosen to hit the draining/long
# paths (schmitt's deadband, quadrature's simultaneous-change fault, pfd's both-pending, the loops' high-trip inputs).
# The bound is ``<=`` so a future optimization may lower it; a regression that re-inflates a per-block drain trips it.
# If-conversion collapsed schmitt, pfd, and iir1_lpf to a single block (a fixed-latency transaction, down from tens of
# cycles of branch-block drain); quadrature/recip/remainder stay multi-block and data-dependent. iir1_lpf is the
# canonical "if-conversion RAISES min_ii but LOWERS realized latency" case (min_ii 15->21 as its rare cheap first-sample
# path is unified away, yet every realized transaction is the steady-state 20 cycles, which only this guard pins -- the
# static metrics gate sees only the raised min_ii).
_T, _F = True, False
_WORST_CASE_LATENCY: dict[str, tuple[Callable[[], Callable[..., object]], list[list[object]], int]] = {
    "schmitt_trigger": (lambda: SchmittTrigger().__call__, [[2.0], [-2.0], [0.0], [0.5], [-0.5], [3.0]], 6),
    "iir1_lpf": (lambda: IIR1LPF().__call__, [[1.0], [-2.0], [0.5], [3.0], [-1.5], [0.0]], 20),
    "quadrature_encoder": (
        lambda: QuadratureEncoder().__call__,
        [[_T, _F], [_T, _T], [_F, _F], [_F, _T], [_T, _T], [_F, _F]],
        5,
    ),
    "phase_frequency_detector": (
        lambda: PhaseFrequencyDetector().__call__,
        [[_T, _F, _F], [_F, _T, _F], [_T, _T, _F], [_F, _F, _T], [_T, _F, _F], [_F, _F, _T]],
        5,
    ),
    "majority_voter": (
        lambda: MajorityVoter().__call__,
        [[_T, _T, _T, _F, _F, _F], [_T, _T, _T, _T, _T, _T], [_T, _F, _T, _F, _T, _F], [_F, _T, _T, _T, _T, _T]],
        17,
    ),
    "recip_newton": (lambda: NewtonReciprocal().__call__, [[0.5], [1.0], [2.0], [1.7], [2.9], [0.35]], 244),
    "remainder": (
        lambda: remainder,
        [[1.0, 1.0], [7.0, 3.0], [1000.0, 1.0], [123.0, 4.0], [50.0, 7.0], [2.5, 2.5]],
        381,
    ),
    "octave_index": (lambda: octave_index, [[8.0], [0.1], [1.0], [32.0], [0.03], [-3.0]], 135),
}


@pytest.mark.parametrize("name", list(_WORST_CASE_LATENCY))
def test_realized_worst_case_latency_does_not_regress(name: str) -> None:
    # The realized per-transaction latency (the cosim bench's ``waited``) over a fixed adversarial input set must not
    # exceed its frozen worst case. This is the teeth behind the latency-reduction work: a per-block drain regression
    # re-inflates the count, amplified across loop trips. Freezing the post-optimization figure makes the gate fail on
    # any future change that lengthens a transaction, while still allowing a genuine improvement (the bound is ``<=``).
    factory, vectors, worst = _WORST_CASE_LATENCY[name]
    lir = build(lower_to_mir(optimize(lower(factory())), default_ops(_FMT)), name)
    model = NumericalSimulator(lir)
    waited = [_drive(model, inputs)[1] for inputs in vectors]
    assert max(waited) <= worst, f"{name}: realized worst-case latency regressed {worst} -> {max(waited)} ({waited})"
