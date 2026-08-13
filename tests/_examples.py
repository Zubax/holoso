"""
Shared example-kernel catalogue: each compilable example plus the domain knowledge needed to drive it -- a factory, a
baseline, curated and random vector generators, and the datapath format(s). Consumed by both the cosimulation suite
(``test_cosim_examples.py``, RTL vs the embedded model) and the Python-reference suite (``test_example_reference.py``,
the model vs the original Python), so the two views stay in lockstep over one source of truth.
"""

import dataclasses
import math
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from holoso import FFromIntOptions, FloatFormat, FToIntOptions, OperatorOptions, Options
from ._modelref import bounded, default_options, format_edge_bits, log_uniform_positive, spd_matrix, unit_roundoff

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import ekf1_stateful as ekf1_stateful  # noqa: E402
import ekf1_stateless as ekf1_stateless  # noqa: E402
import imu_frame_transform as imu_frame_transform  # noqa: E402  # synth matrix only; matrix/vector I/O has no scalar SPEC
import kepler  # noqa: E402
import madd  # noqa: E402
from nco import Nco  # noqa: E402
import polar as polar  # noqa: E402  # scalar-driven below; vector I/O pinned in test_verify
import poly3  # noqa: E402
from polar import from_polar, to_polar  # noqa: E402  # bare names so the frontend inlines them into the wrappers
from biquad import Biquad  # noqa: E402
from cordic_sincos import CordicSinCos as CordicSinCos  # noqa: E402
from crc32 import POLY_IEEE8023, Crc32  # noqa: E402
from debouncer import Debouncer  # noqa: E402
from equal_temperament import equal_temperament as equal_temperament  # noqa: E402
from fir import Fir4  # noqa: E402
from iir1_lpf import IIR1LPF as IIR1LPF  # noqa: E402
from iq_oscillator import IqOscillator  # noqa: E402
from latching_fault_register import LatchingFaultRegister  # noqa: E402
from lfsr16 import Lfsr16  # noqa: E402
from majority_voter import MajorityVoter  # noqa: E402
from octave_index import octave_index  # noqa: E402
from pid import PID as PID  # noqa: E402
from priority_encoder import PriorityEncoder  # noqa: E402
from pwm import Pwm  # noqa: E402
from phase_frequency_detector import PhaseFrequencyDetector as PhaseFrequencyDetector  # noqa: E402
from quadrature_encoder import QuadratureEncoder  # noqa: E402
from recip_newton import NewtonReciprocal  # noqa: E402
from remainder import remainder as remainder  # noqa: E402
from schmitt_trigger import SchmittTrigger as SchmittTrigger  # noqa: E402
from signal_window import signal_window  # noqa: E402
from trapezoidal_leaky_streaming_integrator import TrapezoidalLeakyStreamingIntegrator  # noqa: E402
from uart import OVERSAMPLE, UartRx, UartTx  # noqa: E402

# The wide scalar datapath: the one configuration the example matrix is synthesized in.
_FMT = FloatFormat(8, 36)
# What a kernel that builds no float operator gets by default; it sizes nothing for them, so it only tracks main().
_NARROW = FloatFormat(6, 18)
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


type InputVector = dict[str, float | bool | int]
"""One input vector: input-name -> scalar value, matching the family of its port (float, int, or bool)."""


def _drive(name: str, values: Sequence[float | bool | int]) -> list[InputVector]:
    """The single-input drive sequence: one vector per value, in order."""
    return [{name: value} for value in values]


# The published check message runs first, straight out of the all-ones reset, so the ninth row reproduces the
# catalogue value 0xCBF43926 exactly -- the same number ``zlib.crc32`` reports for those bytes. The byte rails and
# checkerboards then fold into the register that message left behind.
_CRC32_MANUAL = _drive("byte", list(b"123456789") + [0x00, 0xFF, 0x80, 0x01, 0x7F, 0xAA, 0x55, 0x00])

_LFSR_MANUAL = _drive(
    "advance",
    [True] * 20
    + [False] * 3  # gated: both the register and the emitted bit must hold
    + [True] * 8  # and resume in phase
    + [False]
    + [True] * 5,
)

# A clean edge, two rejected bounces, the N-1/N boundary (which discriminates >= from >), and a re-bounce one sample
# after a flip, which must not flip back because the tally restarts on the flip.
_DEBOUNCE_MANUAL = _drive(
    "raw",
    [False] * 2
    + [True] * 5
    + [False, False, False, True] * 2
    + [False] * 7
    + [True] * 3
    + [False]
    + [True] * 4
    + [False, True]
    + [False] * 4,
)

# The idle bus and its sentinel, one bit at each position, multi-bit words where the lowest wins, and words whose
# set bits lie above the scanned range, which the scan must ignore because it reads only shifts 0..7. The port is the
# bus carried signed, so a line above the bus is the sign bit and those words are the negative ones.
_PRIORITY_MANUAL = _drive(
    "request",
    [0, 1, 2, 4, 8, 16, 32, 64, 128] + [0b1010, 0b1100_0000, 0xFF, 0x81, 0b10_1000, 0b110] + [-256, -128, -1, -2],
)

_NCO_PHASE_MASK = (1 << 32) - 1
_NCO_MANUAL: list[InputVector] = [
    {"increment": increment, "phase_offset": offset}
    for increment, offset in (
        [(1 << 31, 0)] * 6  # half the tick rate: the MSB toggles every tick
        + [(1 << 30, 0)] * 8  # a quarter of it: a period-4 square wave, two full wraps
        + [(0x3FFFFFFF, 0)] * 8  # one count short of a quarter turn: the one-tick edge jitter
        + [(0xC0000000, 0)] * 8  # past Nyquist: wraps on three ticks out of four
        + [(_NCO_PHASE_MASK, 0)] * 4  # full scale: the aliased one-LSB-per-tick backward ramp
        + [(0, 0)] * 3  # halted: the phase holds and the output must not move
        # The accumulator is frozen across these, so the offset alone must move the output through a whole turn.
        + [(0, 1 << 30), (0, 1 << 31), (0, 0xC0000000), (0, _NCO_PHASE_MASK)]
        + [(1, 0)] * 3  # the smallest tuning step, from a nonzero phase
        + [(1 << 30, 1 << 31)] * 4  # retuning and offsetting at once
    )
]

_PWM_TOP = 100  # the period the example ships; the vectors below are written in terms of it
_PWM_MANUAL = _drive(
    "duty",
    [_PWM_TOP // 2] * (2 * _PWM_TOP)  # a full triangle period at half duty
    + [0] * _PWM_TOP  # always off
    + [_PWM_TOP] * (2 * _PWM_TOP)  # the fullest duty that still has an off tick
    + [_PWM_TOP + 5] * _PWM_TOP  # above top: always on
    + [1] * (2 * _PWM_TOP)  # the shortest pulse
    + [3] * 5  # ends part-way up the ramp, so the next change lands mid-period
    + [6] * 7  # retuned while counting up, and this segment ends on the way down
    + [2] * 6,  # so this one is retuned while counting down
)


# The I/Q oscillator's exact grid: ``frequency * dt * 2**32`` must be an integer in e8m36 and in float64 alike, so the
# float-to-int conversion rounds identically on both sides and the integer phase stays bit-identical to CPython for
# any run length. Off the grid the two roundings disagree near a half-integer and a diverged accumulator never
# re-converges, which no output tolerance could absorb.
_IQ_DT = 2.0**-10  # 1024 transactions per second
_IQ_FREQ_LSB = 2.0**-22  # exactly one accumulator unit per tick


def _draw_iq(rng: np.random.Generator) -> InputVector:
    return {
        "frequency": float(rng.integers(-(1 << 31), 1 << 31)) * _IQ_FREQ_LSB,
        "dt": _IQ_DT,
        "phase_offset": float(rng.integers(0, 1 << 32)) / 2.0**32,
    }


def _iq(frequency: float, phase_offset: float = 0.0) -> InputVector:
    return {"frequency": frequency, "dt": _IQ_DT, "phase_offset": phase_offset}


@dataclass(frozen=True)
class OutputTolerance:
    """
    The independent accuracy budget of one float output lane whose arithmetic accumulates rounding, owned by the spec
    so a compiler defect cannot loosen its own oracle. The allowed absolute error for one transaction is
    ``(ulps + growth_ulps * age) * u * max(|reference|, floor)`` with ``u`` the format's unit roundoff and ``age`` the
    number of transactions already driven: ``ulps`` bounds the rounding of one pass over the source expression,
    ``growth_ulps`` the error a recurrence carries forward through state, and ``floor`` the scale of the lane's
    intermediate operands over the spec's driven input domain, below which a cancellation-prone reference magnitude
    would make a purely relative budget vacuous. Every budget is derived from the source algorithm and the driven
    domain, never calibrated against the compiler under test.
    """

    ulps: int
    growth_ulps: int = 0
    floor: float = 1.0

    def allowance(self, fmt: FloatFormat, reference: float, age: int) -> float:
        assert self.ulps > 0 and self.growth_ulps >= 0 and self.floor > 0.0 and age >= 0
        return (self.ulps + self.growth_ulps * age) * unit_roundoff(fmt) * max(abs(reference), self.floor)


@dataclass(frozen=True)
class ExampleSpec:
    """One example kernel plus the domain knowledge to drive it: a factory, a baseline, and vector generators."""

    name: str
    inputs: tuple[str, ...]
    make_kernel: Callable[[], Callable[..., object]]
    nominal: InputVector  # baseline for the per-input edge sweep (each input perturbed in turn)
    manual: list[InputVector]  # sensible vectors; an ordered sequence for stateful kernels
    draw_random: Callable[[np.random.Generator], InputVector]
    edge_values: tuple[float | bool | int, ...]
    # Per-input edge-sweep overrides: a listed input is swept over its own values instead of ``edge_values`` (e.g. a
    # divisor pinned to positive magnitudes so it never reaches zero). Inputs absent here use ``edge_values``.
    edge_overrides: Mapping[str, tuple[float | bool | int, ...]] = field(default_factory=dict)
    # The float format(s) to drive at. The matrix is e8m36 by plan; a kernel that wants a second datapath (e.g. a
    # shallow e6m18 alongside the deep e8m36, to exercise both pipeline depths) lists both here.
    formats: tuple[FloatFormat, ...] = (_FMT,)
    # The native integer width the kernel needs, which for a float-free kernel is exactly the word it gets: an
    # accumulator that must wrap at a given modulus needs the word to hold it without saturating on the way.
    wint_min: int = Options(OperatorOptions()).wint_min
    # How this kernel's operator set differs from the one the catalogue shares: a datapath crossing between the
    # families needs the conversions, which no float-only kernel builds.
    operators: Callable[[OperatorOptions], OperatorOptions] = lambda ops: ops
    # The Python-reference accuracy contract, keyed by output port name (``out_*``/``state_*``): a listed float lane
    # is compared within its OutputTolerance allowance; an absent lane (and every bool lane) must match the float64
    # reference bit-for-bit. ``None`` excludes the kernel from the generic scalar reference harness: public VECTOR
    # state cannot be read back through per-element scalar attributes.
    reference: Mapping[str, OutputTolerance] | None = field(default_factory=dict)

    def __post_init__(self) -> None:
        inputs = set(self.inputs)
        assert set(self.nominal) == inputs, f"{self.name}: nominal keys {set(self.nominal)} != inputs {inputs}"
        for row in self.manual:
            assert set(row) == inputs, f"{self.name}: manual row keys {set(row)} != inputs {inputs}"
        assert set(self.edge_overrides) <= inputs, f"{self.name}: edge_overrides keys outside inputs {inputs}"

    def options(self, fmt: FloatFormat) -> Options:
        """The spec's synthesis configuration: the shared default at one format, adjusted to what the kernel needs."""
        return self.configured(default_options(fmt))

    def configured(self, options: Options) -> Options:
        """The same adjustment over an arbitrary base, so the pipeline-depth variants stay in one place."""
        return dataclasses.replace(options, operator=self.operators(options.operator), wint_min=self.wint_min)

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


def _uart_tx_drive(payload: tuple[int, ...]) -> list[InputVector]:
    """A transmit sequence: assert start for one tick with each byte, then idle through its whole (<= 11-bit) frame."""
    rows: list[InputVector] = []
    for value in payload:
        rows.append({"start": True, "char": value})
        rows += [{"start": False, "char": 0}] * (OVERSAMPLE * 11)
    return rows


def _uart_rx_frame(
    value: int, parity: bool | None, *, flip_parity: bool = False, drop_stop: bool = False
) -> list[InputVector]:
    """
    A receive sequence: one oversampled serial frame -- idle, start, 8 data bits LSB first, the parity bit only when
    the framing carries one, stop. With ``flip_parity`` the parity bit is corrupted (the receiver must flag
    ``parity_error``); with ``drop_stop`` the stop bit is held low (it must flag ``frame_error``) -- so the error lanes
    are driven to their non-default value. An 8N1 receiver stops one bit earlier, so feeding it a parity bit would
    make it read that bit as the stop bit and flag a spurious framing error.
    """
    data = [(value >> i) & 1 for i in range(8)]
    levels = [True] * 4 + [False] + [bool(d) for d in data]
    if parity is not None:
        levels.append(((sum(data) % 2 == 1) != parity) != flip_parity)  # even-parity bit, inverted for odd parity
    levels += [not drop_stop] + [True] * 4
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
        reference={"out_0": OutputTolerance(ulps=8, floor=16.0)},  # two roundings; |a*b| <= 16 over the domain
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
        reference={"out_0": OutputTolerance(ulps=16, floor=64.0)},  # six Horner roundings; intermediates <= ~60
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
        reference={"state_y": OutputTolerance(ulps=8, growth_ulps=1, floor=8.0)},  # 3 roundings/step at |x| <= 5
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
        name="pid",
        inputs=_PID_INPUTS,
        make_kernel=lambda: PID().__call__,
        reference={
            "out_0": OutputTolerance(ulps=8, growth_ulps=1, floor=64.0),  # pre-clamp |u| <= kd*|de|/dt_min ~ 64
            "state_integral": OutputTolerance(ulps=8, growth_ulps=1, floor=16.0),  # per-step addend ki*e*dt <= 4
            "state_prev_error": OutputTolerance(ulps=8, floor=16.0),  # one subtraction at the |error| scale
        },
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
        formats=(_NARROW,),  # float-free, so the format sizes nothing; this is the one main() builds
        wint_min=6,  # the five channels pack into as many bits, which a signed word carries one above
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
        name="iq_oscillator",
        inputs=("frequency", "dt", "phase_offset"),
        make_kernel=lambda: IqOscillator().tick,
        # One fsincos firing serves both lanes; the budget is the source's own phase scaling plus the core, and it
        # does not grow with age because the phase recurrence is integer and exact -- no float state drifts. It is a
        # bound rather than a fit: the scaling now composes with the cores' turn ABI into a single exact exponent, so
        # the realized error sits well inside it.
        reference={"out_0": OutputTolerance(ulps=64), "out_1": OutputTolerance(ulps=64)},
        nominal={"frequency": 64.0, "dt": _IQ_DT, "phase_offset": 0.0},
        manual=[
            _iq(0.0),  # DC: the phase must not move
            *[_iq(256.0) for _ in range(4)],  # quarter rate: I/Q hit exact 0 and +-1, returning to the start phase
            # The accumulator is frozen across these three, so the bit-exact phase lane must be identical on all of
            # them while I/Q rotate a quarter turn each -- the offset's independence from the state, checkably.
            _iq(0.0, 0.25),
            _iq(0.0, 0.5),
            _iq(0.0, 0.75),
            _iq(-256.0),  # negative: the phase runs backwards
            _iq(1023.99993896484375),  # just under one turn per tick
            _iq(768.0),  # 0.75 turn/tick, which folds to -0.25
            _iq(68.0, 0.125),  # an ordinary tone, wrapping slowly
            _iq(68.0, 0.125),
        ],
        draw_random=_draw_iq,
        # The edge sweep drives dt and frequency to the format rails, overflowing the product to infinity and taking
        # the conversion to its own rail; those rows reach cosim only, where both sides saturate identically.
        edge_values=_WIDE_EDGES,
        formats=(_FMT,),  # wman >= 32: a shallower format would quantize the grid and measure itself, not the compiler
        wint_min=34,  # a carry bit above the phase, then a sign bit
        operators=lambda ops: dataclasses.replace(ops, ffromint=FFromIntOptions(), ftoint=FToIntOptions()),
    ),
    ExampleSpec(
        name="nco",
        inputs=("increment", "phase_offset"),
        make_kernel=lambda: Nco().tick,
        nominal={"increment": 1 << 30, "phase_offset": 0},
        manual=_NCO_MANUAL,
        # Both control words are architecturally 32 bits and the ports are wider only because the machine has one
        # native integer width, but masking each input keeps every sum inside the word, so the kernel is total over
        # the port and the edge sweep can leave the 32-bit range.
        draw_random=lambda rng: {
            "increment": int(rng.integers(0, 1 << 32)),
            "phase_offset": int(rng.integers(0, 1 << 32)),
        },
        edge_values=(0, 1, 1 << 30, 1 << 31, 0xC0000000, _NCO_PHASE_MASK, 1 << 32, -1),
        formats=(FloatFormat(6, 18),),  # float-free, so the kernel's own wint_min alone sizes the accumulator
        wint_min=34,  # the pre-mask sum reaches 2**33 - 2: one carry bit above the accumulator, plus the sign bit
    ),
    ExampleSpec(
        name="pwm",
        inputs=("duty",),
        make_kernel=lambda: Pwm(top=_PWM_TOP).tick,
        nominal={"duty": _PWM_TOP // 2},
        manual=_PWM_MANUAL,
        draw_random=lambda rng: {"duty": int(rng.integers(0, _PWM_TOP + 2))},
        edge_values=(0, 1, _PWM_TOP // 2, _PWM_TOP - 1, _PWM_TOP, _PWM_TOP + 5),
        formats=(_NARROW,),  # float-free, so the format sizes nothing; this is the one main() builds
        wint_min=8,  # a word holding _PWM_TOP, plus the sign bit
    ),
    ExampleSpec(
        name="debouncer",
        inputs=("raw",),
        make_kernel=lambda: Debouncer(samples=4).__call__,
        nominal={"raw": False},
        manual=_DEBOUNCE_MANUAL,
        draw_random=lambda rng: {"raw": bool(rng.integers(0, 2))},
        edge_values=(False, True),
        formats=(_NARROW,),  # float-free, so the format sizes nothing; this is the one main() builds
        wint_min=4,  # the dwell count reaches samples, plus the sign bit
    ),
    ExampleSpec(
        name="priority_encoder",
        inputs=("request",),
        make_kernel=lambda: PriorityEncoder(width=8).__call__,
        nominal={"request": 0},
        manual=_PRIORITY_MANUAL,
        draw_random=lambda rng: {"request": int(rng.integers(0, 256))},
        edge_values=(0, 1, 0x80, 0xFF, -256, -1),
        formats=(_NARROW,),  # float-free, so the format sizes nothing; this is the one main() builds
        wint_min=9,  # the request bus is width lines, carried signed
    ),
    ExampleSpec(
        name="crc32",
        inputs=("byte",),
        make_kernel=lambda: Crc32(POLY_IEEE8023).__call__,
        nominal={"byte": 0x00},
        manual=_CRC32_MANUAL,
        draw_random=lambda rng: {"byte": int(rng.integers(0, 256))},
        edge_values=(0x00, 0x01, 0x7F, 0x80, 0xFF),
        formats=(_NARROW,),  # float-free, so the format sizes nothing; this is the one main() builds
        wint_min=33,  # the reversed polynomial is 32 unsigned bits, which a signed word carries one above
    ),
    ExampleSpec(
        name="lfsr16",
        inputs=("advance",),
        make_kernel=lambda: Lfsr16().__call__,
        nominal={"advance": True},
        manual=_LFSR_MANUAL,
        draw_random=lambda rng: {"advance": bool(rng.integers(0, 2))},
        edge_values=(False, True),
        formats=(_NARROW,),  # float-free, so the format sizes nothing; this is the one main() builds
        wint_min=17,  # the tap mask 0xB400 does not fit a signed 16-bit word
    ),
    ExampleSpec(
        name="uart_tx",
        inputs=("start", "char"),
        make_kernel=lambda: UartTx(parity=False).tick,
        nominal={"start": False, "char": 0},
        # 0x01 and 0x7F have an ODD number of set bits, so the even-parity bit is HIGH for them. The
        # trailing pair asserts start while the machine is busy: the second byte must be ignored, not latched mid-frame.
        manual=(
            _uart_tx_drive((0x55, 0xC3, 0x00, 0x01, 0x7F, 0xFF))
            + [{"start": True, "char": 0x0F}, {"start": True, "char": 0xA5}]
            + [{"start": False, "char": 0}] * (OVERSAMPLE * 11)
        ),
        draw_random=lambda rng: {"start": bool(rng.integers(0, 8) == 0), "char": int(rng.integers(0, 256))},
        # The per-input sweep is uniform over the inputs, so one value set cannot serve both a boolean lane and a byte
        # lane; and a byte only enters the machine on a latching tick, which a one-row perturbation cannot arrange.
        # The byte edges therefore ride the manual sequence, which latches each of them.
        edge_values=(),
        formats=(_NARROW,),  # float-free, so the format sizes nothing; this is the one main() builds
        wint_min=9,  # the 8-bit character, which a signed word carries one bit above
    ),
    ExampleSpec(
        name="uart_rx",
        inputs=("rx",),
        make_kernel=lambda: UartRx(parity=False).tick,
        nominal={"rx": True},
        manual=(
            _uart_rx_frame(0x55, False)
            + _uart_rx_frame(0xC3, False)
            + _uart_rx_frame(0x00, False)
            + _uart_rx_frame(
                0x01, False
            )  # odd popcount -> true even-parity bit HIGH, so the recomputed parity must be 1
            + _uart_rx_frame(0x7F, False)  # 7 bits set (odd), still no error
            + _uart_rx_frame(0x96, False, flip_parity=True)  # corrupted parity bit -> parity_error asserts
            + _uart_rx_frame(0x3C, False, drop_stop=True)  # stop bit held low -> frame_error asserts
            + [{"rx": level} for level in [False] * 4 + [True] * 24]  # false start: recovers before the mid-bit sample
        ),
        draw_random=lambda rng: {"rx": bool(rng.integers(0, 2))},
        edge_values=(False, True),
        formats=(_NARROW,),  # float-free, so the format sizes nothing; this is the one main() builds
        wint_min=9,  # the 8-bit character, which a signed word carries one bit above
    ),
    ExampleSpec(
        name="recip_newton",
        inputs=("x",),
        make_kernel=lambda: NewtonReciprocal().__call__,
        reference={"out_0": OutputTolerance(ulps=64)},  # a few same-trip Newton iterations; output in [0.5, 2]
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
        reference={
            "out_0": OutputTolerance(ulps=32),  # hertz: the exp2 polynomial is a few-ulp-relative approximation
            "out_1": OutputTolerance(ulps=32, floor=64.0),  # recovered note: 12*log2 cancels against the 69 offset
        },
        nominal={"note": 69.0},
        manual=[{"note": v} for v in (69.0, 60.0, 81.0, 57.0, 69.5, 0.0, 127.0)],  # landmark notes + MIDI-range ends
        draw_random=lambda rng: {"note": bounded(rng, 0.0, 127.0)},
        edge_values=(0.0, 21.0, 60.0, 69.0, 108.0, 127.0),  # note edges over the MIDI range
    ),
    ExampleSpec(
        name="cordic_sincos",
        inputs=("theta",),
        make_kernel=lambda: CordicSinCos().__call__,
        # 12 micro-rotations of ~3 roundings each on unit-norm vectors; the same budget for both lanes.
        reference={"out_0": OutputTolerance(ulps=64), "out_1": OutputTolerance(ulps=64)},
        nominal={"theta": 0.5},
        manual=[{"theta": v} for v in (0.0, 0.3, 0.7, -0.5, 1.0, -1.0)],  # angles within the convergence range
        draw_random=_draw_scalars(("theta",), -1.4, 1.4),
        edge_values=_WIDE_EDGES,
    ),
    ExampleSpec(
        name="polar_to",  # fused hypot+atan2 -> one fatan2
        inputs=("x", "y"),
        make_kernel=lambda: polar_to,
        # The CORDIC operator is faithful (few-ulp), not exact; both lanes carry the same operator-level budget.
        reference={"out_0": OutputTolerance(ulps=64), "out_1": OutputTolerance(ulps=64)},
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
        # A near-axis angle makes one lane |magnitude*eps|, so the floor is the |magnitude| <= 4 operand scale.
        reference={"out_0": OutputTolerance(ulps=64, floor=4.0), "out_1": OutputTolerance(ulps=64, floor=4.0)},
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
        reference={"out_0": OutputTolerance(ulps=64, floor=4.0)},  # same-trip Newton at the |M| + e scale
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
        # Four roundings/step at the per-step addend scale |x|*dt <= 0.0625; the sum carries the error forward.
        reference={"state_y": OutputTolerance(ulps=8, growth_ulps=1, floor=0.0625)},
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
        # The 1e3-scale measurement noise enters only through the relative error of S=P+R and 1/S, so the floors are
        # the per-lane intermediate scales: the state block |x| + K*innovation ~ 4, the covariance block's P-products
        # ~ 16 -- NOT the 1e3 input scale, so a small lane cannot hide a large absolute error.
        reference={
            **{f"out_{i}_0": OutputTolerance(ulps=64, floor=4.0) for i in range(3)},
            **{f"out_{i}_0": OutputTolerance(ulps=64, floor=16.0) for i in range(3, 9)},
        },
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
        name="fir",
        inputs=("x",),
        make_kernel=lambda: Fir4().__call__,
        # No growth: the private delay line holds verbatim samples, so only the output convolution rounds.
        reference={"out_0": OutputTolerance(ulps=16, floor=4.0)},
        nominal={"x": 1.0},
        manual=[  # an impulse walking the whole line, then a step, then a sign flip
            *({"x": v} for v in (1.0, 0.0, 0.0, 0.0, 0.0)),
            *({"x": v} for v in (2.0, 2.0, 2.0, 2.0)),
            *({"x": v} for v in (-1.0, 0.5, -3.0, 0.0)),
        ],
        draw_random=_draw_scalars(("x",), -4.0, 4.0),
        edge_values=_WIDE_EDGES,
    ),
    ExampleSpec(
        name="biquad",
        inputs=("x",),
        make_kernel=lambda: Biquad().__call__,
        # Stable feedback (|a1|, a2 < 1) contracts carried error, so linear growth is a conservative envelope.
        reference={
            "out_0": OutputTolerance(ulps=16, growth_ulps=1, floor=4.0),
            "state_s1": OutputTolerance(ulps=16, growth_ulps=1, floor=4.0),
            "state_s2": OutputTolerance(ulps=16, growth_ulps=1, floor=4.0),
        },
        nominal={"x": 1.0},
        manual=[  # an impulse response, then a step the two accumulators settle through
            *({"x": v} for v in (1.0, 0.0, 0.0, 0.0, 0.0)),
            *({"x": v} for v in (1.0, 1.0, 1.0, 1.0)),
            *({"x": v} for v in (-2.0, 0.25, 3.0, 0.0)),
        ],
        draw_random=_draw_scalars(("x",), -4.0, 4.0),
        edge_values=_WIDE_EDGES,
    ),
    ExampleSpec(
        name="ekf1_stateful",
        inputs=("dt", "u_shunt", "di_dt"),
        make_kernel=_fresh_stateful_ekf,
        reference=None,  # carried x/P_urt VECTOR state has no per-element scalar attribute the harness could read
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
