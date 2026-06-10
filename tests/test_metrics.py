"""
Steering/area non-regression gate for the LIR build.

Every currently-synthesizing example (all except ``iir1_hpf`` and ``finite_set_current_controller``, which the
frontend cannot yet lower) is built to LIR and measured on the metrics that bound the synthesized fabric: the wide and
boolean register counts, the per-port read-mux fan-in and per-register write-select fan-in (the steering cost that
dominates the LUTs), and the statically-known latency lower bound. The baseline below was frozen on the pre-convergence
``dev-branching`` build at the default register-allocation effort.

The contract: NO example may regress past its frozen baseline on any metric. A single-block (straight-line) kernel must
stay exactly not-worse -- the unified allocator subsumes the former straight-line path, so equality is the expected
outcome. A control-flow kernel must additionally improve substantially on register count (see ``CFG_TARGET``): the
former CFG path gave every value a fresh register, and convergence reclaims that via cross-block reuse and coalescing.

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


def _measure(name: str) -> Metrics:
    lir: Lir = build(lower_to_mir(optimize(lower(_EXAMPLES[name]())), default_ops(_FMT)), name)
    straight = (
        len(lir.blocks) == 1
        and not lir.bool_state_slots
        and not any(b.comb_ops or b.copies or b.bool_writes for b in lir.blocks)
    )
    read_fanin = sum(max(0, len(regs) - 1) for regs in lir.read_set_per_port.values())
    write_fanin = sum(max(0, len(insts) - 1) for insts in lir.write_set_per_register.values())
    return Metrics(
        straight_line=straight,
        nreg=lir.regfile.nreg,
        bnreg=lir.bool_regfile.nreg,
        steering=read_fanin + write_fanin,
        min_ii=lir.min_initiation_interval,
    )


# Frozen on the pre-convergence build. Each is an upper bound: a converged build must be <= every field. A single-block
# kernel is expected to hit equality; a control-flow kernel must drop its register counts (see CFG_TARGET).
BASELINE: dict[str, Metrics] = {
    "madd": Metrics(True, nreg=4, bnreg=0, steering=1, min_ii=20),
    "poly3": Metrics(True, nreg=5, bnreg=0, steering=3, min_ii=35),
    "signal_window": Metrics(False, nreg=10, bnreg=13, steering=0, min_ii=54),
    "iir1_lpf": Metrics(False, nreg=6, bnreg=2, steering=2, min_ii=15),
    "pid": Metrics(False, nreg=17, bnreg=3, steering=8, min_ii=62),
    "schmitt_trigger": Metrics(False, nreg=1, bnreg=5, steering=0, min_ii=17),
    "quadrature_encoder": Metrics(False, nreg=1, bnreg=30, steering=0, min_ii=29),
    "phase_frequency_detector": Metrics(False, nreg=1, bnreg=18, steering=0, min_ii=15),
    "recip_newton": Metrics(False, nreg=9, bnreg=1, steering=4, min_ii=26),
    "remainder": Metrics(False, nreg=18, bnreg=9, steering=5, min_ii=82),
    "cordic_sincos": Metrics(False, nreg=143, bnreg=12, steering=82, min_ii=274),
    "integrator": Metrics(True, nreg=5, bnreg=0, steering=2, min_ii=24),
    "ekf1_stateless": Metrics(True, nreg=39, bnreg=0, steering=81, min_ii=129),
    "ekf1_stateful": Metrics(True, nreg=38, bnreg=0, steering=86, min_ii=132),
}

# Wide-register ceilings for control-flow kernels: the hardware-frame liveness allocation must keep each at or below
# the figure the convergence achieved, so the gate proves the reuse win (a fresh-per-value regression would blow past
# these) rather than merely guarding the pre-convergence baseline. CORDIC is the headline -- 143 fresh registers down
# to 11. The three at 1 are boolean-dominated kernels whose wide bank was already minimal.
CFG_TARGET: dict[str, int] = {
    "signal_window": 8,
    "iir1_lpf": 4,
    "pid": 10,
    "schmitt_trigger": 1,
    "quadrature_encoder": 1,
    "phase_frequency_detector": 1,
    "recip_newton": 5,
    "remainder": 8,
    "cordic_sincos": 11,
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
    if not base.straight_line:
        assert got.nreg <= CFG_TARGET[name], f"{name}: nreg {got.nreg} exceeds CFG target {CFG_TARGET[name]}"


def test_build_is_deterministic() -> None:
    """The allocator's annealing is ``seed=0``; two builds of the same kernel must agree, so the baseline is stable."""
    first = _measure("ekf1_stateless")
    second = _measure("ekf1_stateless")
    assert first == second
