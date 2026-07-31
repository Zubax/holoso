"""
The shadow parity table, v1: every example entry target's expected Eel stage, one declarative row each.
A row is TEMPORARY until its owner milestone advances it to the terminal expectation; the two permanent
refusals record their terminal refusal class in ``terminal``. The table fails on any unexpected status, so a
regression or an unplanned advance both surface here. The mechanism is deleted at cutover.

Also pins the shadow containment contract: a shadow failure of any kind must not affect the primary output.
"""

import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest

import holoso
from holoso._eel._shadow import ShadowStage, probe

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import ekf1_stateful  # noqa: E402
import ekf1_stateless  # noqa: E402
import equal_temperament  # noqa: E402
import imu_frame_transform  # noqa: E402
import kepler  # noqa: E402
import madd  # noqa: E402
import octave_index  # noqa: E402
import poly3  # noqa: E402
import polar  # noqa: E402
import remainder  # noqa: E402
import signal_window  # noqa: E402
from cordic_sincos import CordicSinCos  # noqa: E402
from finite_set_current_controller import FiniteSetCurrentController  # noqa: E402
from iir1_hpf import IIR1HPF  # noqa: E402
from iir1_lpf import IIR1LPF  # noqa: E402
from latching_fault_register import LatchingFaultRegister  # noqa: E402
from majority_voter import MajorityVoter  # noqa: E402
from phase_frequency_detector import PhaseFrequencyDetector  # noqa: E402
from pid import PID  # noqa: E402
from quadrature_encoder import QuadratureEncoder  # noqa: E402
from recip_newton import NewtonReciprocal  # noqa: E402
from schmitt_trigger import SchmittTrigger  # noqa: E402
from trapezoidal_leaky_streaming_integrator import TrapezoidalLeakyStreamingIntegrator  # noqa: E402
from uart import UartRx, UartTx  # noqa: E402


@dataclass(frozen=True)
class ParityRow:
    """
    ``expect`` is the stage required TODAY; ``owner`` is the milestone expected to advance the row next;
    ``terminal`` is the end state the row is driving toward (ORACLE_OK, or the refusal class for the two
    permanent refusals).
    """

    name: str
    make: Callable[[], object]
    expect: ShadowStage
    owner: str
    terminal: str


_D = ShadowStage.DESUGARED

ROWS: list[ParityRow] = [
    ParityRow("madd", lambda: madd.madd, _D, "M3", "ORACLE_OK"),
    ParityRow("poly3", lambda: poly3.poly3, _D, "M3", "ORACLE_OK"),
    ParityRow("signal_window", lambda: signal_window.signal_window, _D, "M3", "ORACLE_OK"),
    ParityRow("equal_temperament", lambda: equal_temperament.equal_temperament, _D, "M4", "ORACLE_OK"),
    ParityRow("octave_index", lambda: octave_index.octave_index, _D, "M7", "ORACLE_OK"),
    ParityRow("remainder", lambda: remainder.remainder, _D, "M7", "ORACLE_OK"),
    ParityRow("kepler", lambda: kepler.eccentric_anomaly, _D, "M7", "ORACLE_OK"),
    ParityRow("recip_newton", lambda: NewtonReciprocal().__call__, _D, "M7", "ORACLE_OK"),
    ParityRow("cordic_sincos", lambda: CordicSinCos(iterations=12).__call__, _D, "M7", "ORACLE_OK"),
    ParityRow("ekf1_stateless", lambda: ekf1_stateless.update_x_P, _D, "M5", "ORACLE_OK"),
    ParityRow("to_polar", lambda: polar.to_polar, _D, "M5", "ORACLE_OK"),
    ParityRow("from_polar", lambda: polar.from_polar, _D, "M5", "ORACLE_OK"),
    ParityRow("imu_frame_transform", lambda: imu_frame_transform.transform, _D, "M7", "ORACLE_OK"),
    ParityRow("iir1_lpf", lambda: IIR1LPF().__call__, _D, "M8", "ORACLE_OK"),
    ParityRow("pid", lambda: PID().__call__, _D, "M8", "ORACLE_OK"),
    ParityRow("schmitt_trigger", lambda: SchmittTrigger().__call__, _D, "M8", "ORACLE_OK"),
    ParityRow("quadrature_encoder", lambda: QuadratureEncoder().__call__, _D, "M8", "ORACLE_OK"),
    ParityRow("phase_frequency_detector", lambda: PhaseFrequencyDetector().__call__, _D, "M8", "ORACLE_OK"),
    ParityRow("latching_fault_register", lambda: LatchingFaultRegister().__call__, _D, "M8", "ORACLE_OK"),
    ParityRow("majority_voter", lambda: MajorityVoter().__call__, _D, "M8", "ORACLE_OK"),
    ParityRow("integrator", lambda: TrapezoidalLeakyStreamingIntegrator().__call__, _D, "M8", "ORACLE_OK"),
    ParityRow("uart_tx", lambda: UartTx(parity=False).__call__, _D, "M8", "ORACLE_OK"),
    ParityRow("uart_rx", lambda: UartRx(parity=False).__call__, _D, "M8", "ORACLE_OK"),
    ParityRow("ekf1_stateful", lambda: ekf1_stateful.Ekf1().update, _D, "M8", "ORACLE_OK"),
    ParityRow("iir1_hpf", lambda: IIR1HPF().step, _D, "M5", "REJECTED: stateful sub-object at snapshot conversion"),
    ParityRow(
        "finite_set_current_controller",
        lambda: FiniteSetCurrentController().__call__,
        _D,
        "M5",
        "REJECTED: unsupported (dataclass/shapeless) parameters at parameter binding",
    ),
]


@pytest.mark.parametrize("row", ROWS, ids=[row.name for row in ROWS])
def test_parity(row: ParityRow) -> None:
    report = probe(row.make())
    assert (
        report.stage is row.expect
    ), f"{row.name}: expected {row.expect.value}, got {report.stage.value}: {report.error}"
    if report.stage is ShadowStage.DESUGARED:
        assert report.text
        assert report.error is None


def test_probe_rejects_non_function_targets_gracefully() -> None:
    report = probe(print)  # a builtin: no code object, no source; must reject, not crash
    assert report.stage is ShadowStage.REJECTED
    assert report.error is not None


def test_shadow_failure_cannot_affect_primary_output(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def explode(fn: object) -> object:
        raise RuntimeError("deliberate shadow failure")

    monkeypatch.setattr("holoso._eel._shadow.desugar", explode)
    fmt = holoso.FloatFormat(wexp=6, wman=18)
    ops = holoso.OpConfig(
        holoso.FAddOperator(fmt),
        holoso.FMulOperator(fmt),
        holoso.FDivOperator(fmt),
        holoso.FMulILog2OperatorFamily(fmt),
        holoso.FCmpOperator(fmt),
    )
    with caplog.at_level("DEBUG", logger="holoso._eel._shadow"):
        result = holoso.synthesize(madd.madd, ops=ops)
    assert result.verilog_output.verilog
    # The exception record proves synthesize() actually invoked the shadow hook — the containment test must
    # not pass vacuously if the _api.py hook is ever dropped.
    assert any("Eel shadow failed" in record.message for record in caplog.records)
