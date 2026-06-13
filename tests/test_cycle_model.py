"""
Tests for the cycle-accurate :class:`NumericalModel`: the per-clock ``tick`` interface, the data-dependent latency a
caller recovers by counting ticks, and bounded-memory execution of an arbitrarily deep loop.

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
from recip_newton import NewtonReciprocal  # noqa: E402

_FMT = FloatFormat(8, 36)


def _random_inputs(lir: Lir, rng: random.Random) -> list[object]:
    return [
        bool(rng.randint(0, 1)) if type(load).__name__ == "BoolInputLoad" else rng.uniform(-3.0, 3.0)
        for load in lir.inputs
    ]


def _drive(model: NumericalModel, inputs: list[object]) -> tuple[tuple[object, ...], int]:
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
