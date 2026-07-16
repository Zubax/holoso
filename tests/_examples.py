"""
Shared example-kernel catalogue: each compilable example plus the domain knowledge needed to drive it -- a factory, a
baseline, curated and random vector generators, and the datapath format(s). Consumed by both the cosimulation suite
(``test_cosim_examples.py``, RTL vs the embedded model) and the Python-reference suite (``test_example_reference.py``,
the model vs the original Python), so the two views stay in lockstep over one source of truth.
"""

import math
import os
import sys

import pytest
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np

from holoso import FloatFormat
from ._modelref import (
    bounded,
    encode_inputs,
    format_edge_bits,
    log_uniform_positive,
    spd_matrix,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import ekf1_stateful as ekf1_stateful  # noqa: E402
import ekf1_stateless as ekf1_stateless  # noqa: E402
import imu_frame_transform as imu_frame_transform  # noqa: E402  # synth matrix only; matrix/vector I/O has no scalar SPEC
import kepler  # noqa: E402
import madd  # noqa: E402
import polar as polar  # noqa: E402  # scalar-driven below; vector I/O pinned in test_verify
import poly3  # noqa: E402
from polar import from_polar, to_polar  # noqa: E402  # bare names so the frontend inlines them into the wrappers
from cordic_sincos import CordicSinCos as CordicSinCos  # noqa: E402
from equal_temperament import equal_temperament as equal_temperament  # noqa: E402
from iir1_lpf import IIR1LPF as IIR1LPF  # noqa: E402
from iir1_hpf import IIR1HPF as IIR1HPF  # noqa: E402
from latching_fault_register import LatchingFaultRegister  # noqa: E402
from majority_voter import MajorityVoter  # noqa: E402
from octave_index import octave_index  # noqa: E402
from pid import PID as PID  # noqa: E402
from phase_frequency_detector import PhaseFrequencyDetector as PhaseFrequencyDetector  # noqa: E402
from quadrature_encoder import QuadratureEncoder  # noqa: E402
from recip_newton import NewtonReciprocal  # noqa: E402
from remainder import remainder as remainder  # noqa: E402
from schmitt_trigger import SchmittTrigger as SchmittTrigger  # noqa: E402
from signal_window import signal_window  # noqa: E402
from trapezoidal_leaky_streaming_integrator import TrapezoidalLeakyStreamingIntegrator  # noqa: E402
from uart import OVERSAMPLE, UartRx, UartTx  # noqa: E402

# The wide scalar datapath; the plan permits synthesizing only this configuration for the example matrix.
_FMT = FloatFormat(8, 36)
# Frozen random vectors per example (over and above the manual and edge vectors); scale via the env knob to trade
# coverage for cosimulation wall-clock.
_RANDOM_COUNT = int(os.environ.get("HOLOSO_TEST_RANDOM_COUNT", "48"))
_SEED = 0x05EED

# Canonical format edges (zero, ±0.5, ±1, ±smallest-normal, ±largest-finite); the EKF variants stay finite and keep the
# divisor anchored, so they swap the ±largest-finite extreme for a large but non-overflowing magnitude.
_WIDE_EDGES = tuple(_FMT.decode(bits) for bits in format_edge_bits(_FMT))
_MIN_NORMAL = _WIDE_EDGES[5]
_EKF_EDGES = (*_WIDE_EDGES[:7], 1e6, -1e6)
_POSITIVE_DIVISOR_EDGES = (0.5, 1.0, _MIN_NORMAL, 1e6)
_PID_INPUTS = ("setpoint", "measurement", "dt")
_PID_MANUAL = [  # first update (D suppressed), then a varying measurement (D active) driving both saturation rails
    {"setpoint": 0.5, "measurement": 0.0, "dt": 2.0},
    {"setpoint": 0.75, "measurement": 0.0, "dt": 0.5},
    {"setpoint": 10.0, "measurement": 0.0, "dt": 1.0},
    {"setpoint": 10.0, "measurement": 0.5, "dt": 0.5},
    {"setpoint": 0.0, "measurement": 1.0, "dt": 1.0},
    {"setpoint": 0.5, "measurement": 0.5, "dt": 1.0},
    {"setpoint": -10.0, "measurement": 0.0, "dt": 0.25},
    {"setpoint": -10.0, "measurement": -0.5, "dt": 1.5},
    {"setpoint": 0.0, "measurement": 0.0, "dt": 1.0},
]


type InputVector = dict[str, float | bool]
"""One input vector: input-name -> scalar value (a float, or a bool for a boolean-typed port)."""


class ReferenceComparison(Enum):
    """
    How the Python-reference suite (``test_example_reference``) treats a kernel's scalar outputs.
    EXACT: every float output is exact in the format (boolean logic, integer-valued counters/bytes, or exact
    Sterbenz reductions), so it must match the float64 reference bit-for-bit. APPROXIMATE: float outputs
    accumulate rounding (continuous arithmetic), so the comparison allows a format-derived tolerance. EXCLUDED:
    the generic scalar-lane harness cannot drive the kernel -- it has public VECTOR state it would read by a
    non-existent per-element attribute -- so its aggregate-state read-back is validated against Python separately in
    ``test_verify``.
    """

    EXACT = "exact"
    APPROXIMATE = "approximate"
    EXCLUDED = "excluded"


@dataclass(frozen=True)
class ExampleSpec:
    """One example kernel plus the domain knowledge to drive it: a factory, a baseline, and vector generators."""

    name: str
    inputs: tuple[str, ...]
    make_kernel: Callable[[], Callable[..., object]]
    nominal: InputVector  # baseline for the per-input edge sweep (each input perturbed in turn)
    manual: list[InputVector]  # sensible vectors; an ordered sequence for stateful kernels
    draw_random: Callable[[np.random.Generator], InputVector]
    edge_values: tuple[float | bool, ...]
    # Per-input edge-sweep overrides: a listed input is swept over its own values instead of ``edge_values`` (e.g. a
    # divisor pinned to positive magnitudes so it never reaches zero). Inputs absent here use ``edge_values``.
    edge_overrides: Mapping[str, tuple[float | bool, ...]] = field(default_factory=dict)
    # The float format(s) to drive at. The matrix is e8m36 by plan; a kernel that wants a second datapath (e.g. a
    # shallow e6m18 alongside the deep e8m36, to exercise both pipeline depths) lists both here.
    formats: tuple[FloatFormat, ...] = (_FMT,)
    reference: ReferenceComparison = ReferenceComparison.EXACT

    def __post_init__(self) -> None:
        inputs = set(self.inputs)
        assert set(self.nominal) == inputs, f"{self.name}: nominal keys {set(self.nominal)} != inputs {inputs}"
        for row in self.manual:
            assert set(row) == inputs, f"{self.name}: manual row keys {set(row)} != inputs {inputs}"
        assert set(self.edge_overrides) <= inputs, f"{self.name}: edge_overrides keys outside inputs {inputs}"

    def vectors(self, fmt: FloatFormat) -> list[dict[str, int]]:
        """The full reproducible input sequence as input-name -> ZKF-bits rows: manual, then random, then edges."""
        return [encode_inputs(fmt, row) for row in self.raw_vectors()]

    def reference_vectors(self) -> list[InputVector]:
        """
        The manual sequence then the random draw -- the inputs on which the ZKF model and the float64 Python reference
        agree to within the per-operation rounding tolerance, so the Python-reference suite drives this subset. The
        per-input format-edge sweep is intentionally excluded: at the format extremes the model legitimately diverges
        from float64 (an operation overflowing to the format's infinity stays finite in float64), a property of the
        datapath rather than a compiler defect, and the cosim suite (RTL == model) covers those edges instead.
        """
        rng = np.random.default_rng(_SEED)
        return [*self.manual, *(self.draw_random(rng) for _ in range(_RANDOM_COUNT))]

    def raw_vectors(self) -> list[InputVector]:
        """The full reproducible input sequence as raw float/bool rows: manual, then random, then per-input edges."""
        rows = self.reference_vectors()
        for name in self.inputs:
            values = self.edge_overrides.get(name, self.edge_values)
            rows += [{**self.nominal, name: value} for value in values]
        return rows


def _draw_ekf_stateless(rng: np.random.Generator) -> dict[str, float]:
    cov = spd_matrix(rng, 3, 0.5, 2.0)
    return {
        "P00": float(cov[0, 0]),
        "P01": float(cov[0, 1]),
        "P02": float(cov[0, 2]),
        "P11": float(cov[1, 1]),
        "P12": float(cov[1, 2]),
        "P22": float(cov[2, 2]),
        "Q_R": log_uniform_positive(rng, 1e-3, 1e-1),
        "Q_g": log_uniform_positive(rng, 1e-3, 1e-1),
        "Q_i": log_uniform_positive(rng, 1e-3, 1e-1),
        "R_ct": log_uniform_positive(rng, 1e1, 1e3),  # large measurement noise keeps the 1/x21 divisor away from zero
        "R_shunt": log_uniform_positive(rng, 1e1, 1e3),
        "dt": bounded(rng, 1e-3, 1e-2),
        "x_R": bounded(rng, -1.0, 1.0),
        "x_g": bounded(rng, -1.0, 1.0),
        "x_i": bounded(rng, -1.0, 1.0),
        "z_ct": bounded(rng, -1.0, 1.0),
        "z_shunt": bounded(rng, -1.0, 1.0),
    }


def _draw_scalars(names: tuple[str, ...], lo: float, hi: float) -> Callable[[np.random.Generator], dict[str, float]]:
    return lambda rng: {name: bounded(rng, lo, hi) for name in names}


_UART_FMT = FloatFormat(4, 8)  # the narrowest format that holds a 0..255 byte exactly (wman=8), per examples/uart.py


def _uart_tx_drive(payload: tuple[int, ...]) -> list[dict[str, float | bool]]:
    """A transmit sequence: assert start for one tick with each byte, then idle through its whole (<= 11-bit) frame."""
    rows: list[dict[str, float | bool]] = []
    for value in payload:
        rows.append({"start": True, "char": float(value)})
        rows += [{"start": False, "char": 0.0}] * (OVERSAMPLE * 11)
    return rows


def _uart_rx_frame(
    value: int, parity_odd: bool, *, flip_parity: bool = False, drop_stop: bool = False
) -> list[dict[str, float | bool]]:
    """
    A receive sequence: one oversampled 8E1/8O1 serial frame -- idle, start, 8 data bits LSB first, parity, stop. With
    ``flip_parity`` the parity bit is corrupted (the receiver must flag ``parity_error``); with ``drop_stop`` the stop
    bit is held low (it must flag ``frame_error``) -- so the error lanes are driven to their non-default value.
    """
    data = [(value >> i) & 1 for i in range(8)]
    parity = (sum(data) % 2 == 1) != parity_odd  # even-parity bit, inverted for odd parity
    levels = [True] * 4 + [False] + [bool(d) for d in data] + [parity != flip_parity] + [not drop_stop] + [True] * 4
    return [{"rx": level} for level in levels for _ in range(OVERSAMPLE)]


def _fresh_stateful_ekf() -> Callable[..., object]:
    # An explicit divisor-safe reset (large, equal measurement noise keeps the 1/x21 divisor anchored), independent of
    # the filter's real-application default reset; a fresh instance per compile so the model's reset snapshot starts
    # each run from the same state.
    filt = ekf1_stateful.Ekf1(
        x=[0.0, 0.0, 0.0],
        P_urt=[1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
        R_diag=[1.0e3, 1.0e3],
        Q_diag=np.array([1.0e-6, 1.0e-6, 1.0e-6]),
    )
    return filt.update


_EKF_STATELESS_INPUTS = (
    "P00", "P01", "P02", "P11", "P12", "P22",
    "Q_R", "Q_g", "Q_i", "R_ct", "R_shunt", "dt",
    "x_R", "x_g", "x_i", "z_ct", "z_shunt",
)  # fmt: skip


# Scalar wrappers so the generic scalar harness can drive the vector-only polar kernels; the frontend inlines the
# bare-name calls, so these reuse the example's own arithmetic.
def polar_to(x: float, y: float) -> tuple[float, float]:
    r = to_polar(np.array([x, y]))
    return r[0], r[1]


def polar_from(magnitude: float, angle: float) -> tuple[float, float]:
    v = from_polar(np.array([magnitude, angle]))
    return v[0], v[1]


SPECS = [
    ExampleSpec(
        name="madd",
        inputs=("a", "b", "c"),
        make_kernel=lambda: madd.madd,
        reference=ReferenceComparison.APPROXIMATE,
        nominal={"a": 1.0, "b": 1.0, "c": 1.0},
        manual=[
            {"a": 1.0, "b": 1.0, "c": 0.0},
            {"a": 2.0, "b": -3.0, "c": 5.0},  # c is a dead input -- value must not matter
            {"a": 0.5, "b": 0.25, "c": -1.0},
            {"a": -1.5, "b": 2.5, "c": 0.0},
        ],
        draw_random=_draw_scalars(("a", "b", "c"), -4.0, 4.0),
        edge_values=_WIDE_EDGES,
    ),
    ExampleSpec(
        name="signal_window",
        inputs=("x", "lo", "hi"),
        make_kernel=lambda: signal_window,
        # Exact: clamped selects an operand verbatim (no arithmetic) and gated multiplies by exactly 0.0 or 1.0 -- both
        # float lanes are bit-exact in the format, so a clamp/select/gate miscompile cannot hide under a tolerance.
        nominal={"x": 0.0, "lo": -1.0, "hi": 1.0},
        manual=[
            {"x": 0.5, "lo": -1.0, "hi": 1.0},  # inside and nonzero -> live
            {"x": 0.0, "lo": -1.0, "hi": 1.0},  # inside but zero -> not live
            {"x": 2.0, "lo": -1.0, "hi": 1.0},  # above -> clamped to hi, outside
            {"x": -2.0, "lo": -1.0, "hi": 1.0},  # below -> clamped to lo, outside
            {"x": 1.0, "lo": -1.0, "hi": 1.0},  # on the hi boundary -> outside (x >= hi), not strictly inside
            {"x": -1.0, "lo": -1.0, "hi": 1.0},  # on the lo boundary
            {"x": 0.25, "lo": -0.5, "hi": 0.5},
        ],
        draw_random=lambda rng: {
            "x": bounded(rng, -3.0, 3.0),
            "lo": bounded(rng, -2.0, 0.0),
            "hi": bounded(rng, 0.0, 2.0),
        },
        edge_values=_WIDE_EDGES,
    ),
    ExampleSpec(
        name="poly3",
        inputs=("x", "c0", "c1", "c2", "c3"),
        make_kernel=lambda: poly3.poly3,
        reference=ReferenceComparison.APPROXIMATE,
        nominal={"x": 1.0, "c0": 1.0, "c1": 1.0, "c2": 1.0, "c3": 1.0},
        manual=[
            {"x": 0.0, "c0": 1.0, "c1": 2.0, "c2": 3.0, "c3": 4.0},  # evaluates to c0
            {"x": 1.0, "c0": 1.0, "c1": 1.0, "c2": 1.0, "c3": 1.0},  # sum of coefficients
            {"x": 2.0, "c0": 1.0, "c1": 0.0, "c2": 0.0, "c3": 1.0},  # x**3 + 1
            {"x": -1.5, "c0": 0.5, "c1": -2.0, "c2": 1.0, "c3": 3.0},
        ],
        draw_random=lambda rng: {
            "x": bounded(rng, -2.0, 2.0),
            **_draw_scalars(("c0", "c1", "c2", "c3"), -4.0, 4.0)(rng),
        },
        edge_values=_WIDE_EDGES,
    ),
    ExampleSpec(
        name="iir1_lpf",
        inputs=("x",),
        make_kernel=lambda: IIR1LPF().__call__,
        reference=ReferenceComparison.APPROXIMATE,
        nominal={"x": 1.0},
        manual=[  # one continuous stream: the first sample latches y=x, then the IIR settles toward the input
            *({"x": v} for v in (1.0, 1.0, 1.0, 1.0)),
            *({"x": v} for v in (5.0, 5.0, 0.0, 0.0)),
            *({"x": v} for v in (-2.0, 3.0, 0.5, -1.0)),
        ],
        draw_random=_draw_scalars(("x",), -4.0, 4.0),
        edge_values=_WIDE_EDGES,
    ),
    ExampleSpec(
        name="iir1_hpf",
        inputs=("x",),
        make_kernel=lambda: IIR1HPF().step,  # a hierarchical component: the HPF holds a stateful LPF child
        reference=ReferenceComparison.APPROXIMATE,
        nominal={"x": 1.0},
        manual=[  # a continuous stream; the high-pass output decays to ~0 for a constant input as the LPF bias tracks it
            *({"x": v} for v in (1.0, 1.0, 1.0, 1.0)),
            *({"x": v} for v in (5.0, 5.0, 0.0, 0.0)),
            *({"x": v} for v in (-2.0, 3.0, 0.5, -1.0)),
        ],
        draw_random=_draw_scalars(("x",), -4.0, 4.0),
        edge_values=_WIDE_EDGES,
    ),
    ExampleSpec(
        name="pid",
        inputs=_PID_INPUTS,
        make_kernel=lambda: PID().__call__,
        reference=ReferenceComparison.APPROXIMATE,
        nominal={"setpoint": 1.0, "measurement": 0.0, "dt": 1.0},
        manual=_PID_MANUAL,
        draw_random=lambda rng: {
            **_draw_scalars(("setpoint", "measurement"), -6.0, 6.0)(rng),
            "dt": log_uniform_positive(rng, 0.125, 4.0),
        },
        edge_values=_WIDE_EDGES,
        edge_overrides={"dt": _POSITIVE_DIVISOR_EDGES},
    ),
    ExampleSpec(
        name="schmitt_trigger",
        inputs=("x",),
        make_kernel=lambda: SchmittTrigger().__call__,
        nominal={"x": 0.0},
        manual=[  # up through HIGH, hold across the deadband, down through LOW, hold, back up (hysteresis)
            {"x": v} for v in (0.0, 0.5, 1.5, 0.5, -0.5, -1.5, -0.5, 0.5, 2.0)
        ],
        draw_random=_draw_scalars(("x",), -3.0, 3.0),
        edge_values=_WIDE_EDGES,
    ),
    ExampleSpec(
        name="quadrature_encoder",
        inputs=("a", "b"),
        make_kernel=lambda: QuadratureEncoder().__call__,
        nominal={"a": False, "b": False},
        manual=[
            {"a": False, "b": False},  # no transition
            {"a": False, "b": True},  # forward sequence: 00 -> 01 -> 11 -> 10 -> 00
            {"a": True, "b": True},
            {"a": True, "b": False},
            {"a": False, "b": False},
            {"a": True, "b": False},  # reverse sequence: 00 -> 10 -> 11 -> 01 -> 00
            {"a": True, "b": True},
            {"a": False, "b": True},
            {"a": False, "b": False},
            {"a": True, "b": True},  # invalid simultaneous change
            {"a": False, "b": False},
            {"a": False, "b": True},
        ],
        draw_random=lambda rng: {
            "a": bool(rng.integers(0, 2)),
            "b": bool(rng.integers(0, 2)),
        },
        edge_values=(False, True),
    ),
    ExampleSpec(
        name="phase_frequency_detector",
        inputs=("ref_edge", "fb_edge", "clear"),
        make_kernel=lambda: PhaseFrequencyDetector().__call__,
        nominal={"ref_edge": False, "fb_edge": False, "clear": False},
        manual=[
            {"ref_edge": False, "fb_edge": False, "clear": True},
            {"ref_edge": True, "fb_edge": False, "clear": False},  # reference leads -> up
            {"ref_edge": False, "fb_edge": False, "clear": False},  # hold up while waiting
            {"ref_edge": False, "fb_edge": True, "clear": False},  # feedback arrives -> reset
            {"ref_edge": False, "fb_edge": True, "clear": False},  # feedback leads -> down
            {"ref_edge": False, "fb_edge": False, "clear": False},  # hold down while waiting
            {"ref_edge": True, "fb_edge": False, "clear": False},  # reference arrives -> reset
            {"ref_edge": True, "fb_edge": True, "clear": False},  # simultaneous edges cancel
            {"ref_edge": True, "fb_edge": False, "clear": False},
            {"ref_edge": False, "fb_edge": False, "clear": True},  # asynchronous software-visible clear
        ],
        draw_random=lambda rng: {
            "ref_edge": bool(rng.integers(0, 2)),
            "fb_edge": bool(rng.integers(0, 2)),
            "clear": bool(rng.integers(0, 2)),
        },
        edge_values=(False, True),
    ),
    ExampleSpec(
        name="latching_fault_register",
        inputs=("overcurrent", "overvoltage", "overtemp"),
        make_kernel=lambda: LatchingFaultRegister().__call__,
        nominal={"overcurrent": False, "overvoltage": False, "overtemp": False},
        manual=[
            {"overcurrent": False, "overvoltage": False, "overtemp": False},  # idle -> nothing latched
            {"overcurrent": True, "overvoltage": False, "overtemp": False},  # overcurrent trips -> latches
            {"overcurrent": False, "overvoltage": False, "overtemp": False},  # transient gone, the latch holds
            {"overcurrent": False, "overvoltage": True, "overtemp": False},  # overvoltage trips -> both latched
            {"overcurrent": False, "overvoltage": False, "overtemp": True},  # overtemp trips -> all three latched
            {"overcurrent": False, "overvoltage": False, "overtemp": False},  # all stay latched (cleared only by reset)
        ],
        draw_random=lambda rng: {
            "overcurrent": bool(rng.integers(0, 2)),
            "overvoltage": bool(rng.integers(0, 2)),
            "overtemp": bool(rng.integers(0, 2)),
        },
        edge_values=(False, True),
    ),
    ExampleSpec(
        name="majority_voter",
        inputs=("enabled", "a", "b", "c", "d", "e"),
        make_kernel=lambda: MajorityVoter().__call__,
        # nominal ``enabled`` is True so the per-input edge sweep actually enters the ``if enabled:`` diagnostic block
        # (perturbing one channel against an all-low background flips the voted value and trips that channel's fault).
        nominal={"enabled": True, "a": False, "b": False, "c": False, "d": False, "e": False},
        manual=[
            # The opening row observes every fault lane LOW (all channels agree with voted=False) before any can latch,
            # so a stuck-high lane is caught. The fault XOR is then exercised against BOTH voted polarities: a high
            # channel disagreeing with a low majority (voted False) AND a low channel disagreeing with a high majority
            # (voted True) -- so a miscompile of the voted value feeding the latches cannot hide behind a constant.
            {"enabled": True, "a": False, "b": False, "c": False, "d": False, "e": False},  # voted False, no fault
            {
                "enabled": True,
                "a": True,
                "b": False,
                "c": False,
                "d": False,
                "e": False,
            },  # voted False, a disagrees ->a
            {"enabled": True, "a": True, "b": True, "c": True, "d": False, "e": False},  # voted True, d, e disagree
            {"enabled": False, "a": False, "b": False, "c": False, "d": False, "e": False},  # disabled: faults hold
            {"enabled": True, "a": True, "b": True, "c": False, "d": True, "e": True},  # voted True, c disagrees -> c
            {
                "enabled": True,
                "a": False,
                "b": True,
                "c": False,
                "d": False,
                "e": False,
            },  # voted False, b disagrees ->b
            {"enabled": True, "a": True, "b": True, "c": True, "d": True, "e": True},  # unanimous, all faults hold
        ],
        draw_random=lambda rng: {name: bool(rng.integers(0, 2)) for name in ("enabled", "a", "b", "c", "d", "e")},
        edge_values=(False, True),
    ),
    ExampleSpec(
        name="uart_tx",
        inputs=("start", "char"),
        make_kernel=lambda: UartTx(parity=False).__call__,
        nominal={"start": False, "char": 0.0},
        # 0x01 and 0x7F have an ODD number of set bits, so the even-parity bit is HIGH for them: the parity XOR
        # reduction must emit a 1, not just the 0 that every even-popcount byte (0x55/0xC3/0x00/0xFF) yields.
        manual=_uart_tx_drive((0x55, 0xC3, 0x00, 0x01, 0x7F, 0xFF)),
        draw_random=lambda rng: {"start": bool(rng.integers(0, 8) == 0), "char": float(rng.integers(0, 256))},
        # The per-input edge sweep is uniform over inputs, so it cannot mix a boolean lane (start) with a float lane
        # (char); the random draw already sweeps char across the whole 0..255 byte range, so no separate edge set.
        edge_values=(),
        formats=(_UART_FMT,),
    ),
    ExampleSpec(
        name="uart_rx",
        inputs=("rx",),
        make_kernel=lambda: UartRx(parity=False).__call__,
        nominal={"rx": True},
        manual=(
            _uart_rx_frame(0x55, False)
            + _uart_rx_frame(0xC3, False)
            + _uart_rx_frame(0x00, False)
            + _uart_rx_frame(
                0x01, False
            )  # odd popcount -> true even-parity bit HIGH, so the recomputed parity must be 1
            + _uart_rx_frame(0x7F, False)  # 7 bits set (odd) -> exercises most of the parity reduction, still no error
            + _uart_rx_frame(0x96, False, flip_parity=True)  # corrupted parity bit -> parity_error asserts
            + _uart_rx_frame(0x3C, False, drop_stop=True)  # stop bit held low -> frame_error asserts
        ),
        draw_random=lambda rng: {"rx": bool(rng.integers(0, 2))},
        edge_values=(False, True),
        formats=(_UART_FMT,),
    ),
    ExampleSpec(
        name="recip_newton",
        inputs=("x",),
        make_kernel=lambda: NewtonReciprocal().__call__,
        reference=ReferenceComparison.APPROXIMATE,
        nominal={"x": 1.0},
        manual=[{"x": v} for v in (0.5, 0.75, 1.0, 1.3, 1.7, 2.0)],  # across the [0.5, 2.0] reciprocal domain
        draw_random=_draw_scalars(("x",), 0.5, 2.0),
        # The Newton iteration only converges on its domain; off-domain x diverges and the data-dependent back-edge
        # loop never terminates, so the edge sweep is pinned to the domain rather than the full format edge set.
        edge_overrides={"x": (0.5, 0.75, 1.0, 1.5, 2.0)},
        edge_values=_WIDE_EDGES,
    ),
    ExampleSpec(
        name="remainder",
        inputs=("x", "y"),
        make_kernel=lambda: remainder,
        nominal={"x": 5.0, "y": 2.0},
        manual=[  # reduction across magnitude ratios, both signs, and round-to-even ties (6/4 -> -2, 2/4 -> 2)
            {"x": x, "y": y}
            for x, y in [(5.0, 3.0), (10.0, 3.0), (7.5, 2.0), (-7.5, 2.0), (13.0, 4.0), (6.0, 4.0), (2.0, 4.0),
                         (1.0, 4.0), (100.0, 7.0), (0.5, 0.25), (3.0, 3.0), (0.0, 2.0)]
        ],  # fmt: skip
        draw_random=lambda rng: {"x": bounded(rng, -8.0, 8.0), "y": log_uniform_positive(rng, 0.25, 4.0)},
        # The divisor must stay nonzero (y == 0 makes the scaled-subtraction loop run forever), and the magnitude
        # ratio is bounded to keep the data-dependent trip count -- hence the simulation length -- small.
        edge_overrides={"y": (0.25, 0.5, 1.0, 2.0, 4.0)},
        edge_values=(0.0, 0.5, -0.5, 1.0, -1.0, 3.0, -3.0, 8.0),
    ),
    ExampleSpec(
        name="octave_index",
        inputs=("x",),
        make_kernel=lambda: octave_index,
        nominal={"x": 1.0},
        manual=[{"x": v} for v in (1.0, 2.0, 8.0, 0.5, 0.1, 32.0, 0.03, -4.0, -0.25)],  # both ranges, both signs
        # x must stay nonzero (x == 0 makes the magnitude loop run forever) and bounded in magnitude (the trip count is
        # the octave distance, hence the simulation length); abs() folds the sign in, so the random sweep is positive.
        draw_random=lambda rng: {"x": log_uniform_positive(rng, 2**-5, 2**5)},
        edge_overrides={"x": (0.25, 0.5, 1.0, 2.0, 8.0)},
        edge_values=(0.25, 0.5, 1.0, 2.0, 8.0),
        formats=(FloatFormat(6, 18), _FMT),  # the shallow and deep datapaths, both bit-exact against the model
    ),
    ExampleSpec(
        name="equal_temperament",
        inputs=("note",),
        make_kernel=lambda: equal_temperament,
        reference=ReferenceComparison.APPROXIMATE,  # continuous transcendental arithmetic rounds each step
        nominal={"note": 69.0},
        manual=[{"note": v} for v in (69.0, 60.0, 81.0, 57.0, 69.5, 0.0, 127.0)],  # landmark notes + MIDI-range ends
        draw_random=lambda rng: {"note": bounded(rng, 0.0, 127.0)},
        edge_values=(0.0, 21.0, 60.0, 69.0, 108.0, 127.0),  # note edges over the MIDI range
    ),
    ExampleSpec(
        name="cordic_sincos",
        inputs=("theta",),
        make_kernel=lambda: CordicSinCos().__call__,
        reference=ReferenceComparison.APPROXIMATE,
        nominal={"theta": 0.5},
        manual=[{"theta": v} for v in (0.0, 0.3, 0.7, -0.5, 1.0, -1.0)],  # angles within the convergence range
        draw_random=_draw_scalars(("theta",), -1.4, 1.4),
        edge_values=_WIDE_EDGES,
    ),
    ExampleSpec(
        name="polar_to",  # fused hypot+atan2 -> one fatan2
        inputs=("x", "y"),
        make_kernel=lambda: polar_to,
        reference=ReferenceComparison.APPROXIMATE,  # the turn scale rounds; the CORDIC is faithful, not exact
        nominal={"x": 1.0, "y": 1.0},
        manual=[
            {"x": 3.0, "y": 4.0},
            {"x": -1.0, "y": 2.0},
            {"x": -2.0, "y": -1.5},
            {"x": 0.5, "y": -0.5},
            {"x": 1.0, "y": 0.0},
            {"x": 0.0, "y": 1.0},
            {"x": 0.0, "y": 0.0},  # origin: the fused path yields (0, 0)
        ],
        draw_random=_draw_scalars(("x", "y"), -4.0, 4.0),
        edge_values=_WIDE_EDGES,
    ),
    ExampleSpec(
        name="polar_from",  # coalesced cos+sin -> one fsincos
        inputs=("magnitude", "angle"),
        make_kernel=lambda: polar_from,
        reference=ReferenceComparison.APPROXIMATE,
        nominal={"magnitude": 1.0, "angle": 0.5},
        manual=[
            {"magnitude": 1.0, "angle": 0.0},
            {"magnitude": 2.0, "angle": math.pi / 2},
            {"magnitude": 1.5, "angle": -math.pi / 2},
            {"magnitude": 0.5, "angle": math.pi},
            {"magnitude": 3.0, "angle": -math.pi},
            {"magnitude": 2.0, "angle": 0.7},
        ],
        draw_random=lambda rng: {"magnitude": bounded(rng, -4.0, 4.0), "angle": bounded(rng, -math.pi, math.pi)},
        edge_values=_WIDE_EDGES,
    ),
    ExampleSpec(
        name="kepler",  # Newton loop; sin(E)+cos(E) coalesce into one fsincos per iteration
        inputs=("mean_anomaly", "eccentricity"),
        make_kernel=lambda: kepler.eccentric_anomaly,
        reference=ReferenceComparison.APPROXIMATE,
        nominal={"mean_anomaly": 0.5, "eccentricity": 0.3},
        manual=[
            {"mean_anomaly": 0.8, "eccentricity": 0.3},
            {"mean_anomaly": 2.5, "eccentricity": 0.6},
            {"mean_anomaly": -1.0, "eccentricity": 0.9},
            {"mean_anomaly": 0.1, "eccentricity": 0.0},  # e=0: E=M immediately
            {"mean_anomaly": math.pi, "eccentricity": 0.5},
            {"mean_anomaly": -math.pi, "eccentricity": 0.8},
        ],
        draw_random=lambda rng: {
            "mean_anomaly": bounded(rng, -math.pi, math.pi),
            # Capping e at 0.7 keeps 1-e*cos >= 0.3 so the Newton update crosses the exit threshold at the same trip
            # count in model and float64; the high-e corner (0.9) is pinned by the manual/edge vectors.
            "eccentricity": bounded(rng, 0.0, 0.7),
        },
        # Off-domain (huge M, or e -> 1) the Newton loop diverges and never terminates, so both inputs are pinned to the
        # convergent domain instead of the format-edge sweep.
        edge_overrides={
            "mean_anomaly": (-math.pi, -1.0, 0.0, 1.0, math.pi),
            "eccentricity": (0.0, 0.3, 0.6, 0.9),
        },
        edge_values=(),
    ),
    ExampleSpec(
        name="integrator",
        inputs=("x", "dt"),
        make_kernel=lambda: TrapezoidalLeakyStreamingIntegrator(k=2**-22).__call__,
        reference=ReferenceComparison.APPROXIMATE,
        nominal={"x": 1.0, "dt": 1.0e-3},
        manual=[  # one continuous stream: settle at zero, a step, an impulse, then a ramp
            *({"x": v, "dt": 1.0e-3} for v in (0.0, 0.0, 1.0, 1.0, 1.0, 1.0)),
            *({"x": v, "dt": 2.0e-3} for v in (0.0, 5.0, 0.0, 0.0)),
            *({"x": v, "dt": 5.0e-4} for v in (1.0, 2.0, 3.0, 4.0)),
        ],
        draw_random=lambda rng: {"x": bounded(rng, -4.0, 4.0), "dt": log_uniform_positive(rng, 1.0e-4, 1.0e-2)},
        edge_overrides={"dt": (0.0, 1.0e-4, 1.0e-3, 1.0e-2)},
        edge_values=_WIDE_EDGES,
    ),
    ExampleSpec(
        name="ekf1_stateless",
        inputs=_EKF_STATELESS_INPUTS,
        make_kernel=lambda: ekf1_stateless.update_x_P,
        reference=ReferenceComparison.APPROXIMATE,
        nominal={
            "P00": 1.0, "P01": 0.0, "P02": 0.0, "P11": 1.0, "P12": 0.0, "P22": 1.0,
            "Q_R": 1e-3, "Q_g": 1e-3, "Q_i": 1e-3, "R_ct": 1e2, "R_shunt": 1e2, "dt": 1e-2,
            "x_R": 0.5, "x_g": 0.5, "x_i": 0.5, "z_ct": 0.5, "z_shunt": 0.5,
        },  # fmt: skip
        manual=[
            {
                **dict.fromkeys(_EKF_STATELESS_INPUTS, 0.0),
                "P00": 1.0, "P11": 1.0, "P22": 1.0, "R_ct": 1e3, "R_shunt": 1e3,
            },  # fmt: skip
            {
                "P00": 2.0, "P01": 0.1, "P02": 0.0, "P11": 1.5, "P12": -0.2, "P22": 0.8,
                "Q_R": 1e-3, "Q_g": 1e-3, "Q_i": 1e-3, "R_ct": 5e2, "R_shunt": 5e2, "dt": 1e-2,
                "x_R": 0.3, "x_g": -0.4, "x_i": 0.2, "z_ct": 0.1, "z_shunt": -0.1,
            },  # fmt: skip
        ],
        draw_random=_draw_ekf_stateless,
        edge_values=_EKF_EDGES,
        edge_overrides={"R_ct": _POSITIVE_DIVISOR_EDGES, "R_shunt": _POSITIVE_DIVISOR_EDGES},
    ),
    ExampleSpec(
        name="ekf1_stateful",
        inputs=("dt", "u_shunt", "di_dt"),
        make_kernel=_fresh_stateful_ekf,
        reference=ReferenceComparison.EXCLUDED,  # carried x/P_urt vectors only; no scalar lane for the reference harness
        nominal={"dt": 1e-2, "u_shunt": 0.5, "di_dt": 0.5},
        manual=[  # a short measurement sequence threaded through the carried state
            {"dt": 1e-2, "u_shunt": 0.0, "di_dt": 0.0},
            {"dt": 1e-2, "u_shunt": 1.0, "di_dt": 0.5},
            {"dt": 1e-2, "u_shunt": 1.0, "di_dt": 0.5},
            {"dt": 1e-2, "u_shunt": -1.0, "di_dt": -0.5},
        ],
        draw_random=lambda rng: {
            "dt": bounded(rng, 1e-3, 1e-2),
            "u_shunt": bounded(rng, -1.0, 1.0),
            "di_dt": bounded(rng, -1.0, 1.0),
        },
        edge_values=_EKF_EDGES,  # only dt reaches the divisor, and the folded R_diag keeps it anchored
    ),
]


# FIR_PARITY_PENDING: examples the new (FIR) front-end cannot lower yet, name -> the feature/stage that unblocks it.
# This is the central registry of the front-end parity debt; every SPECS-driven suite skips the examples in it. Each
# of stages 6-9 removes its own entries as the feature lands, and stage 10 asserts this map is empty. Greppable via
# the FIR_PARITY_PENDING token.
FIR_PARITY_PENDING: dict[str, str] = {
    "ekf1_stateful": "stage 9: aggregate list state (the whole update expression chain lowers)",
    "polar_to": "stage 9: np.array construction",
    "polar_from": "stage 9: np.array construction",
    # Off-catalogue examples (no SPECS entry) whose synth targets and eventual suites key off these names too.
    "imu_frame_transform": "stage 9: jaxtyping ndarray ports + matmul over array parameters",
    "finite_set_current_controller": "stage 9: record ports/returns, reductions, array comparison, Index[N] gather",
}


def parity_marks(name: str) -> tuple[pytest.MarkDecorator, ...]:
    """Pytest marks that skip an example still awaiting front-end parity, or an empty tuple when it is supported."""
    reason = FIR_PARITY_PENDING.get(name)
    return (pytest.mark.skip(reason=f"FIR_PARITY_PENDING: {reason}"),) if reason is not None else ()
