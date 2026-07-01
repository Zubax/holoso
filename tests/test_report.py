"""
The HTML report is tested against every compilable example: each is synthesized and rendered, and the schedule section
is checked for the features the report must reveal -- the operator legend, register-liveness tint, and (where the
kernel uses them) the boolean operators (logic and the float<->bool casts) and persistent state. This guards the
report renderer against regressions on the full ZISC feature set, which the cosimulation tests do not exercise.
"""

import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from holoso import FloatFormat
from holoso._backend.html import generate as generate_report
from holoso._backend.verilog import generate as generate_verilog
from holoso._frontend import lower
from holoso._hir import optimize
from holoso._lir import build, RegRef
from holoso._mir import lower as lower_to_mir
from ._modelref import default_ops, overlap_spill_kernel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import madd  # noqa: E402
import poly3  # noqa: E402
from cordic_sincos import CordicSinCos  # noqa: E402
from iir1_lpf import IIR1LPF  # noqa: E402
from pid import PID  # noqa: E402
from phase_frequency_detector import PhaseFrequencyDetector  # noqa: E402
from quadrature_encoder import QuadratureEncoder  # noqa: E402
from recip_newton import NewtonReciprocal  # noqa: E402
from remainder import remainder  # noqa: E402
from schmitt_trigger import SchmittTrigger  # noqa: E402
from signal_window import signal_window  # noqa: E402
from trapezoidal_leaky_streaming_integrator import TrapezoidalLeakyStreamingIntegrator  # noqa: E402

_FMT = FloatFormat(8, 36)

# Every compilable scalar example (the EKF kernels render through the same machinery as these but are large and add no
# report-feature coverage). Each maps to a factory so a stateful instance is fresh per render.
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
}


def _report(name: str) -> str:
    lir = build(lower_to_mir(optimize(lower(_EXAMPLES[name]())), default_ops(_FMT)), name, fetch_stages=3)
    return generate_report(lir, generate_verilog(lir)).html


@pytest.mark.parametrize("name", list(_EXAMPLES))
def test_report_renders_for_each_example(name: str) -> None:
    lir = build(lower_to_mir(optimize(lower(_EXAMPLES[name]())), default_ops(_FMT)), name, fetch_stages=3)
    html = generate_report(lir, generate_verilog(lir)).html
    assert html.lstrip().startswith("<!")
    assert "<h2>Schedule</h2>" in html
    assert "class='gridkey'" in html
    assert "register holds a live value" in html
    # The wide-register column appears only when the kernel uses the wide bank; a purely-boolean kernel (no wide
    # registers) shows the boolean bank instead.
    if lir.regfile.nreg:
        assert "title='registers'>registers" in html
        assert "r0" in html
    else:
        assert "title='bool registers'>bool registers" in html
        assert "b0" in html
    assert "class='live'" in html or "live'>" in html


def test_report_reveals_boolean_operators_and_casts() -> None:
    # signal_window uses boolean connectives (and/or), a chained comparison, and both float<->bool casts -- all of
    # which the schedule must now render (operator legend colors and the per-op chips), not just comparisons.
    html = _report("signal_window")
    for mnemonic in ("fcmp", "band", "bor", "ftobool", "ffrombool"):
        assert f">{mnemonic}<" in html, f"the legend should list the {mnemonic} operator"
    assert "bool(" in html  # a float->bool cast chip
    assert "&amp;" in html  # a boolean-AND chip


def test_report_shows_persistent_boolean_state() -> None:
    # pid carries a boolean ``_started`` state; the report must show persistent state and a boolean register bank.
    html = _report("pid")
    assert "persistent state" in html
    assert "b0" in html


def test_report_draws_per_arm_edges_for_a_multi_arm_spill() -> None:
    # Regression (P3b): under cross-block overlap a result spills into BOTH single-pred arms, landing at a distinct PC
    # in each (Lir.write_landing_pcs). The schedule must draw that result's dataflow edges (and ops chip) on EVERY arm's
    # landing row, not only the first -- else the non-fall-through arm shows a bare commit cell with no provenance and
    # the report is not path-exact. Crash-before: edges/chips anchored only to landing_pcs[0], so the second arm's
    # landing cell never appeared as an edge source.
    import json

    from holoso._backend.html._schedule import render_schedule

    lir = build(lower_to_mir(optimize(lower(overlap_spill_kernel)), default_ops(_FMT)), "overlap_spill", fetch_stages=3)
    html = render_schedule(lir)
    marker = "var data = "
    payload, _ = json.JSONDecoder().raw_decode(html, html.index(marker) + len(marker))
    edge_sources = {edge[0] for edge in payload["edges"]}

    checked = 0
    for block in lir.blocks:
        for op in block.ops:
            for write in op.writes:
                if not isinstance(write.dst, RegRef):
                    continue
                landing_pcs = lir.write_landing_pcs(block, op)
                if len(landing_pcs) <= 1:
                    continue
                for pc in landing_pcs:  # a wide register's column ordinal is its index (the wide bank renders first)
                    cell = f"g{write.dst.index}_{pc}"
                    assert cell in edge_sources, f"no dataflow edge anchored to the arm landing cell {cell}"
                checked += 1
    assert checked > 0, "overlap_spill_kernel produced no multi-arm spill -- the regression is vacuous"
