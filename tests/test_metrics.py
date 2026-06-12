"""
Steering/area non-regression gate for the LIR build.

Every currently-synthesizing example (all except ``iir1_hpf`` and ``finite_set_current_controller``, which the
frontend cannot yet lower) is built to LIR and measured on the metrics that bound the synthesized fabric: the wide and
boolean register counts, the per-port read-mux fan-in and per-register write-select fan-in (the steering cost that
dominates the LUTs), and the statically-known latency lower bound. The baseline below was re-frozen on the converged
build with the per-bank timing model, at the default register-allocation effort and ``PYTHONHASHSEED=0`` (the nox
``tests`` session pins the seed: MIR value numbering is hash-dependent upstream of the allocator, whose tie-breaks
key on value ids, so unpinned runs land anywhere in a small band of equal-quality colorings and the frozen steering
figures would flap across processes; seed-independent value numbering is tracked as follow-up compiler work).

The contract: NO example may regress past its frozen baseline on any metric. A single-block (straight-line) kernel is
expected to hit equality -- the unified allocator subsumes the former straight-line path. The control-flow rows
themselves encode the convergence win (cross-block reuse and coalescing collapsed the former fresh-per-value register
explosion), so any backslide toward it fails the same gate.

These read/write fan-in figures are exact steering proxies only for single-block kernels (the read-set/write-set
properties union across the flat op stream, which conflates mutually-exclusive blocks on a CFG); for control-flow
kernels the register counts are the directly-comparable figures, with the fan-in kept as a same-kernel before/after
monotonicity guard.
"""

import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest

from holoso import FloatFormat
from holoso._frontend import lower
from holoso._hir import optimize
from holoso._lir import Lir, build
from holoso._mir import lower as lower_to_mir
from ._modelref import default_ops

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import madd  # noqa: E402
import poly3  # noqa: E402
from cordic_sincos import CordicSinCos  # noqa: E402
from ekf1_stateful import Ekf1  # noqa: E402
from ekf1_stateless import update_x_P  # noqa: E402
from iir1_lpf import IIR1LPF  # noqa: E402
from phase_frequency_detector import PhaseFrequencyDetector  # noqa: E402
from pid import PID  # noqa: E402
from quadrature_encoder import QuadratureEncoder  # noqa: E402
from recip_newton import NewtonReciprocal  # noqa: E402
from remainder import remainder  # noqa: E402
from schmitt_trigger import SchmittTrigger  # noqa: E402
from signal_window import signal_window  # noqa: E402
from trapezoidal_leaky_streaming_integrator import TrapezoidalLeakyStreamingIntegrator  # noqa: E402

_FMT = FloatFormat(8, 36)

_EXAMPLES: dict[str, Callable[[], Callable[..., object]]] = {
    "madd": lambda: madd.madd,
    "poly3": lambda: poly3.poly3,
    "signal_window": lambda: signal_window,
    "iir1_lpf": lambda: IIR1LPF().__call__,
    "pid": lambda: PID().__call__,
    "schmitt_trigger": lambda: SchmittTrigger().__call__,
    "quadrature_encoder": lambda: QuadratureEncoder().__call__,
    "phase_frequency_detector": lambda: PhaseFrequencyDetector().__call__,
    "recip_newton": lambda: NewtonReciprocal().__call__,
    "remainder": lambda: remainder,
    "cordic_sincos": lambda: CordicSinCos().__call__,
    "integrator": lambda: TrapezoidalLeakyStreamingIntegrator(k=2**-22).__call__,
    "ekf1_stateless": lambda: update_x_P,
    "ekf1_stateful": lambda: Ekf1().update,
}


@dataclass(frozen=True, slots=True)
class Metrics:
    """
    The non-regression figures sampled off a built :class:`Lir`.

    ``steering`` is the total sparse-regfile mux fan-in -- read-mux fan-in plus write-select fan-in -- which is exactly
    the allocator's primary objective. The read/write split between the two is an artifact of set-iteration order
    (``PYTHONHASHSEED``-sensitive) that can shift between two equal-cost colorings; their sum is stable, so the gate
    asserts on the sum.
    """

    straight_line: bool
    nreg: int
    bnreg: int
    steering: int
    min_ii: int


@pytest.fixture(autouse=True)
def _pinned_regalloc_knobs(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Pin every register-allocation tuning knob to its shipped default so the frozen baselines are reproducible
    regardless of the developer's environment (``HOLOSO_REGALLOC_EFFORT`` speed-ups, write-cap/price experiments).
    The knobs are env-read-once at import, so the module attributes are patched to the named defaults; changing a
    default deliberately re-freezes the baselines.
    """
    import holoso._lir._regalloc as regalloc

    # Pinned to the knobs' default values (the getenv fallbacks in _regalloc), restated here as literals: the
    # baselines are frozen against the defaults, so an env override must not leak into this gate.
    monkeypatch.setattr(regalloc, "_REFINE_MAXITER", 5000)
    monkeypatch.setattr(regalloc, "_REG_REUSE_WRITE_CAP", 2)
    monkeypatch.setattr(regalloc, "_REG_PRICE", 2.0)


def _measure(name: str) -> Metrics:
    lir: Lir = build(lower_to_mir(optimize(lower(_EXAMPLES[name]())), default_ops(_FMT)), name)
    straight = (
        len(lir.blocks) == 1
        and not lir.bool_state_slots
        and not any(b.inline_ops or b.copies or b.bool_writes for b in lir.blocks)
        and lir.bool_regfile.nreg == 0
    )
    read_fanin = sum(max(0, len(regs) - 1) for regs in lir.read_set_per_port.values())
    write_fanin = sum(
        max(0, len(lanes) - 1)
        for sets in (lir.write_set_per_register, lir.bool_write_set_per_register)
        for lanes in sets.values()
    )
    return Metrics(
        straight_line=straight,
        nreg=lir.regfile.nreg,
        bnreg=lir.bool_regfile.nreg,
        steering=read_fanin + write_fanin,
        min_ii=lir.min_initiation_interval,
    )


# Re-frozen on the converged build with the per-bank timing model, at the default register-allocation effort. Each is
# an upper bound: a build must be <= every field. Relative to the pre-convergence dev-branching freeze this folds in
# three named mechanisms: the unified cross-block allocator (register/steering drops, e.g. cordic 143/12 -> 10/1);
# the bank-true dependency edges of the latch-free boolean bank (min_ii drops on the boolean-logic/cast-dense kernels:
# signal_window 54 -> 43, quadrature_encoder 29 -> 20; float-dominated paths are unchanged); and the modeling of
# phi-arm installs as real predecessor-tail register writes, which forbids a phi from sharing a register still
# occupied at the install step (small honest costs on the phi-dense kernels: pid steering 5 -> 6, schmitt bnreg
# 2 -> 3, phase_frequency_detector bnreg 8 -> 12, recip_newton nreg 4 -> 5; the prior smaller figures relied on
# sharings that were safe only because the frontend never emits a phi arm from a branching block). The pooled-fcmp
# normalization then raised ONLY the steering of the comparison-heavy kernels (signal_window 0 -> 3, remainder
# 3 -> 7, cordic 22 -> 25; registers and min_ii byte-identical): the comparator's operand muxes existed before as a
# PC-keyed mux the metric could not see, and as ordinary read ports they are now counted -- and visible to the
# coloring objective, which is why cordic then shed a register and three mux arms (10/25 -> 9/22) while remainder's
# reshaped annealing landscape settled one arm higher (7 -> 8). Making the comparator commutative (operand swap
# with the gt/lt tap exchange) then clawed comparator mux arms back: signal_window 3 -> 1, remainder 8 -> 7. The
# straight-line float-only kernels are byte-identical to the pre-convergence freeze, proving the unified path
# subsumes the old one. Write fan-in counts BOTH banks' lanes: the boolean lanes' write-address selects are real
# steering this taxonomy created, so they enter the gate alongside the wide selects (signal_window +2, pid +1,
# schmitt +1, remainder +2 select arms on the comparison kernels at the inclusion re-freeze).
#
# (PYTHONHASHSEED note: see the module docstring.)
BASELINE: dict[str, Metrics] = {
    "madd": Metrics(True, nreg=4, bnreg=0, steering=1, min_ii=20),
    "poly3": Metrics(True, nreg=5, bnreg=0, steering=3, min_ii=35),
    "signal_window": Metrics(False, nreg=6, bnreg=6, steering=3, min_ii=43),
    "iir1_lpf": Metrics(False, nreg=3, bnreg=2, steering=1, min_ii=15),
    "pid": Metrics(False, nreg=9, bnreg=2, steering=7, min_ii=62),
    "schmitt_trigger": Metrics(False, nreg=1, bnreg=3, steering=1, min_ii=17),
    "quadrature_encoder": Metrics(False, nreg=1, bnreg=15, steering=0, min_ii=20),
    "phase_frequency_detector": Metrics(False, nreg=1, bnreg=12, steering=0, min_ii=15),
    "recip_newton": Metrics(False, nreg=5, bnreg=1, steering=1, min_ii=26),
    "remainder": Metrics(False, nreg=7, bnreg=3, steering=9, min_ii=82),
    "cordic_sincos": Metrics(False, nreg=9, bnreg=1, steering=22, min_ii=274),
    "integrator": Metrics(True, nreg=5, bnreg=0, steering=2, min_ii=24),
    "ekf1_stateless": Metrics(True, nreg=39, bnreg=0, steering=81, min_ii=129),
    "ekf1_stateful": Metrics(True, nreg=38, bnreg=0, steering=86, min_ii=132),
}


@pytest.mark.parametrize("name", list(_EXAMPLES))
def test_metrics_do_not_regress(name: str) -> None:
    base = BASELINE[name]
    got = _measure(name)
    assert got.straight_line == base.straight_line, f"{name}: control-flow classification changed"
    for field in ("nreg", "bnreg", "steering", "min_ii"):
        assert getattr(got, field) <= getattr(
            base, field
        ), f"{name}: {field} regressed {getattr(base, field)} -> {getattr(got, field)}"


def test_build_is_deterministic() -> None:
    """The allocator's annealing is ``seed=0``; two builds of the same kernel must agree, so the baseline is stable."""
    first = _measure("ekf1_stateless")
    second = _measure("ekf1_stateless")
    assert first == second
