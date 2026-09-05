"""
Shared example-kernel catalogue: each compilable example plus the domain knowledge needed to drive it -- a factory, a
baseline, curated and random vector generators, and the datapath format(s). Consumed by both the cosimulation suite
(`test_cosim_examples.py`, RTL vs the embedded model) and the Python-reference suite (`test_example_reference.py`,
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

from holoso import FFromIntOptions, FloatFormat, FSortOptions, FToIntOptions, OperatorOptions, Options
from ._modelref import bounded, default_options, format_edge_bits, log_uniform_positive, spd_matrix, unit_roundoff

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import ekf1_stateful as ekf1_stateful  # noqa: E402
import ekf1_stateless as ekf1_stateless  # noqa: E402
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
from finite_set_current_controller import FiniteSetCurrentController  # noqa: E402
from fir import Fir4  # noqa: E402
from flux_observer import FluxObserver  # noqa: E402
from foc import FocController  # noqa: E402
from iir1_hpf import IIR1HPF as IIR1HPF  # noqa: E402
from iir1_lpf import IIR1LPF as IIR1LPF  # noqa: E402
from image_agc_streamed import EXPOSURE_MIN_s, PIXEL_MAX, ImageAgc  # noqa: E402
import imu_fusion as imu_fusion  # noqa: E402  # synth matrix; it synthesizes the shipped realistic-config kernel
from imu_fusion import ImuFusion as ImuFusion  # noqa: E402
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
import rigid_body_rates as rigid_body_rates  # noqa: E402  # synth matrix; scalar-driven via the wrapper below
from rigid_body_rates import update as rigid_body_update  # noqa: E402  # bare name so the frontend inlines it
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
DEFAULT_RANDOM_COUNT = 48
"""The draw the reference suite takes and the accuracy freeze is frozen over; the env knob only tunes the former."""

_RANDOM_COUNT = int(os.environ.get("HOLOSO_TEST_RANDOM_COUNT", str(DEFAULT_RANDOM_COUNT)))
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
# catalogue value 0xCBF43926 exactly -- the same number `zlib.crc32` reports for those bytes. The byte rails and
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


# The exposure control's pixel bus carries a whole beat per transaction, so each vector is one beat plus its target;
# the beat mirrors the example's own bus width. The catalogue drives a 64x4 frame -- the smallest whose central
# quarter is whole beats and whole rows -- so a vector sequence crosses many frame boundaries.
_AGC_BEAT = 16
_AGC_COLS = 4
_AGC_ROWS = 4
_AGC_TARGET = 120


def _agc(pixels: Sequence[int], target: int) -> InputVector:
    return {**{f"pixels_{i}": pixel for i, pixel in enumerate(pixels)}, "target": target}


def _agc_frame(beat_at: Callable[[int, int], Sequence[int]], target: int = _AGC_TARGET) -> list[InputVector]:
    return [_agc(beat_at(x, y), target) for y in range(_AGC_ROWS) for x in range(_AGC_COLS)]


def _agc_flat(value: int) -> Callable[[int, int], Sequence[int]]:
    return lambda x, y: [value] * _AGC_BEAT


def _agc_centre(inside: int, outside: int) -> Callable[[int, int], Sequence[int]]:
    return lambda x, y: [inside if 1 <= x < 3 and 1 <= y < 3 else outside] * _AGC_BEAT


def _agc_hot(saturated: int) -> Callable[[int, int], Sequence[int]]:
    """A dark frame whose first beat carries the given number of saturated pixels; the clip limit is two."""
    return lambda x, y: (
        [PIXEL_MAX] * saturated + [10] * (_AGC_BEAT - saturated) if (x, y) == (0, 0) else [10] * _AGC_BEAT
    )


_AGC_MANUAL = (
    _agc_frame(_agc_flat(40))  # passes through at the reset demand; a dark frame, so the demand starts to rise
    + _agc_frame(_agc_flat(0)) * 8  # black: the largest error, integrating up through all three actuators to the clamp
    + _agc_frame(_agc_flat(40)) * 10  # over-corrected now, so the digital gain converges down into the deadband
    + _agc_frame(_agc_hot(4))  # saturates more than the limit while dark: the increase is refused
    + _agc_frame(_agc_hot(2))  # saturates exactly the limit: the increase goes through
    + _agc_frame(_agc_centre(200, 0)) * 2  # a bright centre in a dark field meters bright, so the demand falls
    + _agc_frame(_agc_flat(250), 1) * 8  # bright against the lowest target: down to the floor and clamped there
    + _agc_frame(_agc_flat(120))  # on target at the floor: held by the deadband
)


# The I/Q oscillator's exact grid: `frequency * dt * 2**32` must be an integer in e8m36 and in float64 alike, so the
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
    `(ulps + growth_ulps * age) * u * max(|reference|, floor)` with `u` the format's unit roundoff and `age` the
    number of transactions already driven: `ulps` bounds the rounding of one pass over the source expression,
    `growth_ulps` the error a recurrence carries forward through state, and `floor` the scale of the lane's
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
    # Per-input edge-sweep overrides: a listed input is swept over its own values instead of `edge_values` (e.g. a
    # divisor pinned to positive magnitudes so it never reaches zero). Inputs absent here use `edge_values`.
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
    # The Python-reference accuracy contract, keyed by output port name (`out_*`/`state_*`): a listed float lane
    # is compared within its OutputTolerance allowance; an absent lane (and every bool lane) must match the float64
    # reference bit-for-bit. `None` states that no budget has been derived, which excludes the kernel from the
    # tolerance check but not from the accuracy freeze, which measures against the reference rather than a budget.
    reference: Mapping[str, OutputTolerance] | None = field(default_factory=dict)
    # The front-end oracle's slack (`test_eel_oracle`), where both sides are float64 in the same operation order and
    # only the host's own library shape differs -- numpy reaching BLAS for a dot or a norm may contract a product the
    # evaluator rounds. A kernel that carries such a difference through a feedback recurrence needs more than the
    # shared default; one that does not must not be given any.
    oracle_ulps: int = 16

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

    def reference_vectors(self, count: int = _RANDOM_COUNT) -> list[InputVector]:
        """
        The manual sequence then the random draw, `count` of them where a caller needs a sequence the environment
        cannot move -- the inputs on which the ZKF model and the float64 Python reference
        agree to within the per-operation rounding tolerance, so the Python-reference suite drives this subset. The
        per-input format-edge sweep is intentionally excluded: at the format extremes the model legitimately diverges
        from float64 (an operation overflowing to the format's infinity stays finite in float64), a property of the
        datapath rather than a compiler defect, and the cosim suite (RTL == model) covers those edges instead.
        """
        rng = np.random.default_rng(_SEED)
        return [*self.manual, *(self.draw_random(rng) for _ in range(count))]

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
    the framing carries one, stop. With `flip_parity` the parity bit is corrupted (the receiver must flag
    `parity_error`); with `drop_stop` the stop bit is held low (it must flag `frame_error`) -- so the error lanes
    are driven to their non-default value. An 8N1 receiver stops one bit earlier, so feeding it a parity bit would
    make it read that bit as the stop bit and flag a spurious framing error.
    """
    data = [(value >> i) & 1 for i in range(8)]
    levels = [True] * 4 + [False] + [bool(d) for d in data]
    if parity is not None:
        levels.append(((sum(data) % 2 == 1) != parity) != flip_parity)  # even-parity bit, inverted for odd parity
    levels += [not drop_stop] + [True] * 4
    return [{"rx": level} for level in levels for _ in range(OVERSAMPLE)]


def _fresh_flux_observer() -> Callable[..., object]:
    return FluxObserver().tick


def _fresh_foc() -> Callable[..., object]:
    return FocController().tick


def _fresh_finite_set_controller() -> Callable[..., object]:
    return FiniteSetCurrentController().__call__


_FSCC_INPUTS = (
    "kin_pos", "kin_vel", "kin_accel",
    "i_ac_0", "i_ac_1", "i_ac_2",
    "di_ac_dt_0", "di_ac_dt_1", "di_ac_dt_2",
    "u_dc",
    "i_dq_ref_0", "i_dq_ref_1",
)  # fmt: skip


def _fscc_row(
    pos: float, i_ac: tuple[float, float, float], di: tuple[float, float, float], u_dc: float, ref: tuple[float, float]
) -> InputVector:
    return {
        "kin_pos": pos,
        "kin_vel": 0.0,
        "kin_accel": 0.0,
        "i_ac_0": i_ac[0],
        "i_ac_1": i_ac[1],
        "i_ac_2": i_ac[2],
        "di_ac_dt_0": di[0],
        "di_ac_dt_1": di[1],
        "di_ac_dt_2": di[2],
        "u_dc": u_dc,
        "i_dq_ref_0": ref[0],
        "i_dq_ref_1": ref[1],
    }


def _draw_fscc(rng: np.random.Generator) -> InputVector:
    return _fscc_row(
        bounded(rng, -3.2, 3.2),
        (bounded(rng, -10.0, 10.0), bounded(rng, -10.0, 10.0), bounded(rng, -10.0, 10.0)),
        (bounded(rng, -1e5, 1e5), bounded(rng, -1e5, 1e5), bounded(rng, -1e5, 1e5)),
        bounded(rng, 0.0, 400.0),
        (bounded(rng, -10.0, 10.0), bounded(rng, -10.0, 10.0)),
    )


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


def rigid_body_scalar(
    i00: float, i01: float, i02: float,
    i10: float, i11: float, i12: float,
    i20: float, i21: float, i22: float,
    w0: float, w1: float, w2: float,
    t0: float, t1: float, t2: float,
    dt: float,
) -> tuple[float, float, float, float, float, float]:  # fmt: skip
    omega_next, momentum = rigid_body_update(
        np.array([[i00, i01, i02], [i10, i11, i12], [i20, i21, i22]]),
        np.array([w0, w1, w2]),
        np.array([t0, t1, t2]),
        dt,
    )
    return omega_next[0], omega_next[1], omega_next[2], momentum[0], momentum[1], momentum[2]


# The machine the manual sequences describe, shared by the observer and the controller that embeds it: the flux
# linkage each adopts on its aligning first transaction, so the clamp rails where the rows expect.
_FLUX_LINKAGE = 0.005
_PHASE_RESISTANCE = 0.05
_PHASE_INDUCTANCE = 2e-5

_FLUX_OBSERVER_INPUTS = (
    "params_R", "params_L_d", "params_flux_linkage",
    "dt", "u_alpha_beta_0", "u_alpha_beta_1", "i_alpha_beta_0", "i_alpha_beta_1",
)  # fmt: skip


def _flux_row(dt: float, u: tuple[float, float], i: tuple[float, float]) -> InputVector:
    return {
        "params_R": _PHASE_RESISTANCE,
        "params_L_d": _PHASE_INDUCTANCE,
        "params_flux_linkage": _FLUX_LINKAGE,
        "dt": dt,
        "u_alpha_beta_0": u[0],
        "u_alpha_beta_1": u[1],
        "i_alpha_beta_0": i[0],
        "i_alpha_beta_1": i[1],
    }


_FLUX_OBSERVER_MANUAL = [  # a hand-steered spin: the carried flux vector visits all four atan2 quadrants in order
    _flux_row(1e-3, (0.0, 0.0), (0.0, 0.0)),  # the aligning transaction, which adopts the prior and integrates nothing
    _flux_row(1e-3, (50.0, 20.0), (5.0, -3.0)),
    _flux_row(1e-3, (-120.0, 10.0), (-10.0, 0.0)),
    _flux_row(1e-3, (0.0, -80.0), (0.0, 0.0)),
    _flux_row(1e-3, (130.0, 0.0), (8.0, 8.0)),
    _flux_row(0.0, (7.0, 7.0), (5.0, 5.0)),  # dt=0: the voltage term vanishes, isolating the L_d*di path
    _flux_row(1e-3, (0.0, 0.0), (5.0, 5.0)),  # repeated current: di=0, pure resistive-drop drift
    _flux_row(1e-3, (0.0, 90.0), (-20.0, 15.0)),
    _flux_row(1e-4, (1.0, 1.0), (0.0, 0.0)),
    # Steps off the rails: the quadrant rows above drive lanes onto both clamp rails (one lane railed while the
    # other is interior around rows 4-5); these final rows move both lanes back inside the box, so the clamp's
    # pass-through side is also observed against carried state.
    _flux_row(1e-3, (-2.5, -7.0), (0.0, 0.0)),
    _flux_row(1e-3, (0.0, 6.0), (1.0, -1.0)),
]


def _draw_flux_observer(rng: np.random.Generator) -> InputVector:
    # The voltage scale straddles the clamp box: a large draw at a long dt rails a flux lane, a small draw at a
    # short dt moves it inside the linear region, so the sweep keeps exercising both sides of the clamp. The machine
    # parameters are drawn too, over the range of small PMSMs, since they are live ports rather than folded constants.
    return {
        "params_R": log_uniform_positive(rng, 0.01, 0.5),
        "params_L_d": log_uniform_positive(rng, 5e-6, 1e-4),
        "params_flux_linkage": log_uniform_positive(rng, 1e-3, 2e-2),
        "dt": log_uniform_positive(rng, 2e-5, 2e-3),
        "u_alpha_beta_0": bounded(rng, -12.0, 12.0),
        "u_alpha_beta_1": bounded(rng, -12.0, 12.0),
        "i_alpha_beta_0": bounded(rng, -20.0, 20.0),
        "i_alpha_beta_1": bounded(rng, -20.0, 20.0),
    }


_FOC_INPUTS = (
    "params_R", "params_L_dq_0", "params_L_dq_1", "params_flux_linkage", "params_speed_filter_gain",
    "dt", "i_ab_0", "i_ab_1", "i_dq_ref_0", "i_dq_ref_1", "v_dc",
)  # fmt: skip


def _foc_row(
    dt: float,
    i_ab: tuple[float, float],
    i_dq_ref: tuple[float, float],
    v_dc: float,
    inductance: float = _PHASE_INDUCTANCE,
) -> InputVector:
    return {
        "params_R": _PHASE_RESISTANCE,
        "params_L_dq_0": inductance,
        "params_L_dq_1": inductance,
        "params_flux_linkage": _FLUX_LINKAGE,
        "params_speed_filter_gain": 0.05,
        "dt": dt,
        "i_ab_0": i_ab[0],
        "i_ab_1": i_ab[1],
        "i_dq_ref_0": i_dq_ref[0],
        "i_dq_ref_1": i_dq_ref[1],
        "v_dc": v_dc,
    }


# One continuous PWM history covering every arm of the controller. It opens on the aligning transaction (the
# observer adopts its prior from the parameters and integrates nothing), walks a rising current through the linear
# regime where the PI integrators advance,
# then commands a setpoint the bus cannot serve so the vector limiter engages and freezes them. The four steered
# rows that follow carry a tenfold inductance, whose L*di term slams the flux estimate clear across the clamp box:
# their currents were solved offline against the state the preceding rows leave, so the estimated angle lands on a
# demanded value and BOTH branch-cut arms fire (a +5.6 rad step and a -5.6 rad step, each a comfortable 2.4 rad
# clear of the +-pi decision), which no smooth trajectory reaches within a short sequence. The last rows settle
# back onto the shipped machine with the limiter released.
_FOC_MANUAL = [
    _foc_row(2e-5, (0.0, 0.0), (0.0, 0.0), 24.0),
    _foc_row(2e-5, (0.5, -0.25), (0.0, 1.5), 24.0),
    _foc_row(2e-5, (1.0, -0.5), (0.0, 1.5), 24.4),
    _foc_row(2e-5, (2.0, -1.0), (-1.5, 2.0), 23.6),
    _foc_row(2e-5, (-1.0, 2.0), (-1.5, 2.0), 24.0),
    _foc_row(2e-5, (0.0, 0.0), (0.0, 40.0), 24.0),
    _foc_row(2e-5, (-5.0, 2.0), (0.0, 40.0), 12.0),  # the limiter engages on a sagging bus
    _foc_row(5e-5, (38.47, -23.16), (0.0, 0.0), 24.0, inductance=2e-4),
    _foc_row(5e-5, (34.66, -9.10), (0.0, 0.0), 24.0, inductance=2e-4),  # wraps through -pi
    _foc_row(5e-5, (30.84, -19.17), (0.0, 0.0), 24.0, inductance=2e-4),  # and back through +pi
    _foc_row(5e-5, (-10.38, 2.70), (0.0, 0.0), 24.0, inductance=2e-4),
    _foc_row(2e-5, (0.0, 0.0), (0.0, 1.0), 24.0),
    _foc_row(2e-5, (0.0, 0.0), (0.0, 1.0), 24.0),
]


def _draw_foc(rng: np.random.Generator) -> InputVector:
    # Small-PMSM machine parameters and a PWM period between 10 and 100 kHz, with the currents and the bus swept over
    # an ESC's whole operating range. The drawn currents bear no relation to the voltage the controller last
    # commanded, so the flux estimate tumbles and the demanded voltage stays far above the ceiling: the random sweep
    # lives on the limiter, and the linear regime is the manual sequence's to cover. The speed filter is the one lane
    # drawn away from the manual sequence's setting, over the slow per-sample gains a PWM-rate drive actually uses
    # (a millisecond-scale time constant): the estimator is a closed loop around a pure integrator, and its gain is
    # what decides whether a rounding difference decays or compounds. On uncorrelated rows the flux estimate
    # occasionally passes near the origin, where the angle it carries is ill-conditioned; a fast filter turns that
    # into an amplifier and the model and the float64 reference separate within a few hundred rows, so this spec's
    # comparison is validated at the default HOLOSO_TEST_RANDOM_COUNT rather than at an arbitrarily raised one.
    return {
        "params_R": log_uniform_positive(rng, 0.02, 0.3),
        "params_L_dq_0": log_uniform_positive(rng, 1e-5, 1e-4),
        "params_L_dq_1": log_uniform_positive(rng, 1e-5, 1e-4),
        "params_flux_linkage": log_uniform_positive(rng, 2e-3, 2e-2),
        "params_speed_filter_gain": log_uniform_positive(rng, 5e-4, 5e-3),
        "dt": log_uniform_positive(rng, 1e-5, 1e-4),
        "i_ab_0": bounded(rng, -20.0, 20.0),
        "i_ab_1": bounded(rng, -20.0, 20.0),
        "i_dq_ref_0": bounded(rng, -10.0, 10.0),
        "i_dq_ref_1": bounded(rng, -10.0, 10.0),
        "v_dc": bounded(rng, 12.0, 48.0),
    }


_RIGID_BODY_INPUTS = (
    "i00", "i01", "i02", "i10", "i11", "i12", "i20", "i21", "i22",
    "w0", "w1", "w2", "t0", "t1", "t2", "dt",
)  # fmt: skip


def _rigid_body_row(inertia: np.ndarray, omega: tuple[float, ...], tau: tuple[float, ...], dt: float) -> InputVector:
    row: InputVector = {f"i{i}{j}": float(inertia[i, j]) for i in range(3) for j in range(3)}
    row |= {f"w{k}": omega[k] for k in range(3)}
    row |= {f"t{k}": tau[k] for k in range(3)}
    row["dt"] = dt
    return row


def _draw_rigid_body(rng: np.random.Generator) -> InputVector:
    # Eigenvalue-controlled SPD inertia (eigenvalues in [0.5, 2], so cond <= 4 and the determinant stays far from
    # zero); spd_matrix is unsuitable here because its bounds constrain the Cholesky diagonal, not the spectrum.
    # Redrawn until every output lane sits well above cancellation scale: the eel oracle compares the lanes
    # floorless at 16 ULPs against LAPACK/BLAS operation order, so a momentum dot that cancels toward zero would
    # convict the benign algorithm mismatch rather than a defect, and would do so only at a raised
    # HOLOSO_TEST_RANDOM_COUNT.
    while True:
        basis, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        inertia = basis @ np.diag(rng.uniform(0.5, 2.0, 3)) @ basis.T
        omega = np.array([bounded(rng, -1.0, 1.0) for _ in range(3)])
        if min(abs(float(w)) for w in omega) >= 0.05 and min(abs(float(l)) for l in inertia @ omega) >= 0.05:
            return _rigid_body_row(
                inertia,
                (float(omega[0]), float(omega[1]), float(omega[2])),
                tuple(bounded(rng, -1.0, 1.0) for _ in range(3)),
                bounded(rng, 1e-3, 1e-2),
            )


_IMU_FUSION_MEAS = ("gyro_0", "gyro_1", "gyro_2", "accel_0", "accel_1", "accel_2")
_IMU_FUSION_CAL = tuple(f"{m}_{i}_{j}" for m in ("gyro_cal", "accel_cal") for i in range(3) for j in range(3))
_IMU_FUSION_INPUTS = (*_IMU_FUSION_MEAS, *_IMU_FUSION_CAL, "temperature", "dt")
# The catalogue's own demonstration calibration (a 90-degree-yaw mounting rotation composed with scale and
# misalignment corrections), matching main()'s local demo values. Every driven row carries it verbatim; the
# matrices are runtime inputs of the kernel, but sweeping them off nominal is only done by the pinned edge sets
# below (a format-rail element would overflow the state exactly like a rail rate).
_FUSION_GYRO_CAL = np.array([[0.002, -1.002, 0.001], [0.998, 0.001, -0.002], [0.001, 0.002, 1.001]])
_FUSION_ACCEL_CAL = np.array([[-0.001, -0.999, 0.002], [1.003, -0.002, 0.001], [-0.002, 0.001, 0.997]])
_IMU_FUSION_CAL_LANES = dict(
    zip(_IMU_FUSION_CAL, [float(v) for v in (*_FUSION_GYRO_CAL.flatten(), *_FUSION_ACCEL_CAL.flatten())])
)
_IMU_FUSION_G = 9.80665


_IMU_BIAS_LIMIT = 0.0005
"""The catalogue's bias clamp: also the floor of the bias lanes' error budget, whose rails resynchronize them."""


def _fresh_imu_fusion() -> Callable[..., object]:
    """
    The oracle-safe catalogue configuration, distinct from the shipped realistic defaults (the synth rows pass the
    shipped kernel explicitly): a tiny bias clamp so every valid row's bias step overshoots rail to rail and the
    clamp resynchronizes `bias` bit-exactly, and a small clip threshold so the clip row latches on tame dynamics
    (a full-scale rate would make the backward-difference angular acceleration amplify the benign host-vs-model
    summation-order noise past the oracle budgets). The attitude needs no override: the manual sequence opens
    with an accepted coarse-alignment row whose strong rates twist the estimate straight off the near-zero lanes.
    """
    return ImuFusion(bias_limit=_IMU_BIAS_LIMIT, gyro_limit=3.0).update


# One continuous history crafted offline by simulating the reference and frozen as literals (the generator lives
# in the design history): an accepted coarse-alignment row first (in-band accel off unit magnitude with strong
# sub-clip rates, so the aligned attitude immediately twists and every lane stays away from zero), three freefall
# rows walking the attitude further, eight in-band rows alternating between two world-frame gravity targets (the
# filter keeps chasing, so every output lane stays far from zero while the bias rails on every row and both clamp
# arms fire), then the high-magnitude reject, the clip latch, a hot row, and a cold row. Every in-band row keeps
# all |out| lanes above ~0.6 m/s^2 -- below that, the benign BLAS-vs-left-fold summation difference outgrows the
# eel oracle's relative budget on the g-scale operands.
_IMU_FUSION_MANUAL = [
    {**dict(zip((*_IMU_FUSION_MEAS, "temperature", "dt"), values)), **_IMU_FUSION_CAL_LANES}
    for values in [
        (
            -2.61717584486634,
            -2.0522804524157405,
            -2.4716780098458457,
            -5.598102556376175,
            -4.81866182149115,
            6.022439209791113,
            25.0,
            0.125,
        ),
        (-0.3162390153313255, -0.297270259847738, 0.35823635333586945, 0.0, 0.0, 0.0, 25.0, 0.3),
        (-0.3162390153313255, -0.297270259847738, 0.35823635333586945, 0.0, 0.0, 0.0, 25.0, 0.3),
        (-0.3162390153313255, -0.297270259847738, 0.35823635333586945, 0.0, 0.0, 0.0, 25.0, 0.3),
        (
            -0.006111273972922761,
            -0.007300797064837048,
            0.009323888979123474,
            -8.00151356541689,
            -5.044460882024048,
            -2.559792077992834,
            25.0,
            0.125,
        ),
        (
            -0.006111273972922761,
            -0.007300797064837048,
            0.009323888979123474,
            2.957177211541472,
            -6.035906761547455,
            7.158172840906673,
            25.0,
            0.125,
        ),
        (
            -0.006111273972922761,
            -0.007300797064837048,
            0.009323888979123474,
            -8.00843360157175,
            -5.131970618841172,
            -2.3543281285198994,
            25.0,
            0.125,
        ),
        (
            -0.006111273972922761,
            -0.007300797064837048,
            0.009323888979123474,
            3.081717836542828,
            -5.892084096288197,
            7.2252006509938385,
            25.0,
            0.125,
        ),
        (
            -0.006111273972922761,
            -0.007300797064837048,
            0.009323888979123474,
            -8.011592919771621,
            -5.212948280506902,
            -2.1557009490155705,
            25.0,
            0.125,
        ),
        (
            -0.006111273972922761,
            -0.007300797064837048,
            0.009323888979123474,
            3.2067913012550733,
            -5.746619593474721,
            7.287686484093886,
            25.0,
            0.125,
        ),
        (
            -0.006111273972922761,
            -0.007300797064837048,
            0.009323888979123474,
            -8.011588072182313,
            -5.290627337884657,
            -1.9555565219149302,
            25.0,
            0.125,
        ),
        (
            -0.006111273972922761,
            -0.007300797064837048,
            0.009323888979123474,
            3.332351581738178,
            -5.599558970526353,
            7.345606512118715,
            25.0,
            0.125,
        ),
        (
            0.00488170800417128,
            -0.006294822056824952,
            -0.004675116946943577,
            -16.01679690978565,
            -10.729956984022571,
            -3.5080056067154195,
            25.0,
            0.01,
        ),
        (
            0.001489072441967786,
            -3.594112805498327,
            0.008490745792761925,
            1.9330238899639955,
            -1.6755254550131902,
            2.9789982682069183,
            25.0,
            0.25,
        ),
        (
            -0.011339616787281058,
            -0.020480427369373753,
            0.03379100946206396,
            4.368418274251716,
            -2.1018157900454635,
            0.6502566518220327,
            85.0,
            0.25,
        ),
        (
            -0.011223811824432859,
            -0.019673619295098217,
            0.029481089960454177,
            -3.3150183899305126,
            -2.6585436891322347,
            2.4408521380638146,
            -40.0,
            0.01,
        ),
    ]
]

_FUSION_CONFIG = ImuFusion()  # only the shipped temperature-model default is read, never mutated
_FUSION_INV_GYRO_CAL = np.linalg.inv(_FUSION_GYRO_CAL)
_FUSION_INV_ACCEL_CAL = np.linalg.inv(_FUSION_ACCEL_CAL)
# The frozen post-manual attitude's body-frame image of the world direction the random rows aim near.
_FUSION_ACCEL_DIR = np.array([0.5431691419956042, -0.6765799876735006, 0.497198957625099])


def _draw_imu_fusion(rng: np.random.Generator) -> InputVector:
    """
    Every random row is DECISIVELY below the acceptance band (0.50..0.85 g near the frozen direction): the gate
    rejects it, freezing `bias` bit-exactly, and every output lane stays proportionate to its operand scale, which is
    what keeps the eel oracle's relative budget satisfied even at a raised HOLOSO_TEST_RANDOM_COUNT (validated to
    x10; far beyond that, the slow attitude creep can still push a lane over -- the same documented residual
    fragility as rigid_body's). The valid arm's coverage lives in the manual prefix. The raw samples are generated
    through the inverse sensor model so the rates stay tiny and the temperature term stays exactly compensated.
    """
    w = rng.uniform(0.003, 0.01, 3) * rng.choice([-1.0, 1.0], 3)
    angle = rng.uniform(-0.05, 0.05)
    c, sn = math.cos(angle), math.sin(angle)
    d = np.array(
        [
            c * _FUSION_ACCEL_DIR[0] - sn * _FUSION_ACCEL_DIR[1],
            sn * _FUSION_ACCEL_DIR[0] + c * _FUSION_ACCEL_DIR[1],
            _FUSION_ACCEL_DIR[2],
        ]
    )
    f = d * (rng.uniform(0.50, 0.85) * _IMU_FUSION_G)
    temperature = rng.uniform(15.0, 45.0)
    dt = float(np.exp(rng.uniform(np.log(0.02), np.log(0.08))))
    t_powers = np.array([1.0, temperature, temperature * temperature])
    g_s = _FUSION_INV_GYRO_CAL @ (w + _FUSION_CONFIG.temp_model @ t_powers)
    f_s = _FUSION_INV_ACCEL_CAL @ f
    row = dict(zip((*_IMU_FUSION_MEAS, "temperature", "dt"), [float(v) for v in (*g_s, *f_s, temperature, dt)]))
    return {**row, **_IMU_FUSION_CAL_LANES}


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
        name="iir1_hpf",
        inputs=("x",),
        make_kernel=lambda: IIR1HPF().step,
        # The LPF recurrence budget rides on the nested slot; the output subtracts the bias from the input, so the
        # x-m cancellation near convergence needs the same floor as the source scale (|x| <= 5 over the domain).
        reference={
            "out_0": OutputTolerance(ulps=8, growth_ulps=1, floor=8.0),
            "state_lpf_y": OutputTolerance(ulps=8, growth_ulps=1, floor=8.0),
        },
        nominal={"x": 1.0},
        manual=[  # one continuous stream: the first sample latches the bias, then the HPF tracks steps off it
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
        # nominal `enabled` is True so the per-input edge sweep actually enters the `if enabled:` diagnostic block
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
        name="image_agc_streamed",
        inputs=tuple(f"pixels_{i}" for i in range(_AGC_BEAT)) + ("target",),
        make_kernel=lambda: ImageAgc(width=_AGC_COLS * _AGC_BEAT, height=_AGC_ROWS).__call__,
        # The demand in stops carries its rounding forward through state, and the gains are its exp2, where an
        # absolute error of d stops is a relative error of ln2*d -- amplified by the demand's magnitude of up to
        # fourteen stops. The pixel sums are small integers, hence exact; the output pixels are integer lanes and
        # match exactly. The exposure's own scale is its floor, or the budget would be absolute at unity.
        reference={
            "state_exposure_s": OutputTolerance(ulps=8, growth_ulps=2, floor=EXPOSURE_MIN_s),
            "state_analog_gain": OutputTolerance(ulps=8, growth_ulps=2),
            "state_digital_gain": OutputTolerance(ulps=8, growth_ulps=2),
        },
        nominal=_agc([128] * _AGC_BEAT, _AGC_TARGET),
        manual=_AGC_MANUAL,
        draw_random=lambda rng: _agc(
            [int(v) for v in rng.integers(0, PIXEL_MAX + 1, _AGC_BEAT)], int(rng.integers(64, 193))
        ),
        # Every port carries an 8-bit quantity, so the sweep leaves that range on purpose: the far end saturates the
        # accumulator, and the rounding of the gained pixel back to an integer saturates too, on both sides alike.
        edge_values=(0, 1, 128, PIXEL_MAX, -1, 1 << 42),
        operators=lambda ops: dataclasses.replace(
            ops, fsort=FSortOptions(), ffromint=FFromIntOptions(), ftoint=FToIntOptions()
        ),
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
        name="rigid_body_scalar",  # pivoted Gauss-Jordan inversion: data-dependent swap branches feeding one fdiv
        inputs=_RIGID_BODY_INPUTS,
        make_kernel=lambda: rigid_body_scalar,
        # omega' lanes: the inversion's forward error over the driven domain (cond <= ~4) enters scaled by dt <= 1e-2
        # on top of |omega| <= 1, well inside the budget at the |omega_dot| <= ~9 operand scale. L lanes: one 3-term
        # dot per lane (the model's left fold vs the host's BLAS order) with operands <= ~6.
        reference={
            "out_0": OutputTolerance(ulps=64, floor=2.0),
            "out_1": OutputTolerance(ulps=64, floor=2.0),
            "out_2": OutputTolerance(ulps=64, floor=2.0),
            "out_3": OutputTolerance(ulps=16, floor=8.0),
            "out_4": OutputTolerance(ulps=16, floor=8.0),
            "out_5": OutputTolerance(ulps=16, floor=8.0),
        },
        nominal=_rigid_body_row(np.diag([2.0, 3.0, 4.0]), (0.5, -0.3, 0.8), (0.1, 0.0, -0.2), 0.005),
        manual=[
            # A permutation inertia takes the swap branch at every pivot column; all arithmetic stays exact.
            _rigid_body_row(
                np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]), (0.5, -0.3, 0.8), (0.1, 0.0, -0.2), 0.005
            ),
            # An SPD matrix whose leading column is off-diagonal-dominant, so exactly the first column swaps.
            _rigid_body_row(
                np.array([[1.0, 2.0, 0.0], [2.0, 5.0, 1.0], [0.0, 1.0, 3.0]]), (-0.6, 0.4, 0.9), (0.3, -0.1, 0.2), 0.008
            ),
            # Diagonally dominant and asymmetric: no swaps, exercising the fall-through pivot path.
            _rigid_body_row(
                np.array([[3.0, 0.5, -0.2], [0.1, 2.5, 0.4], [-0.3, 0.2, 3.5]]),
                (0.9, -0.7, 0.2),
                (-0.2, 0.4, 0.0),
                0.001,
            ),
        ],
        draw_random=_draw_rigid_body,
        edge_values=_WIDE_EDGES,
        # With the diagonal nominal, a single perturbed off-diagonal leaf cannot change the determinant (its cofactor
        # is zero), so only the diagonal leaves must stay positive to keep every elimination pivot away from zero.
        edge_overrides={
            "i00": _POSITIVE_DIVISOR_EDGES,
            "i11": _POSITIVE_DIVISOR_EDGES,
            "i22": _POSITIVE_DIVISOR_EDGES,
        },
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
        name="imu_fusion",
        inputs=_IMU_FUSION_INPUTS,
        make_kernel=_fresh_imu_fusion,
        # The q lanes are an isometric recurrence (linear growth at the per-step rounding count, unit-norm floor);
        # the b lanes are bounded absolutely by the catalogue clamp, whose rails resynchronize them bit-exactly
        # (floor = the clamp); the out lanes inherit the carried q error at the ~1 g operand scale.
        reference={
            **{f"state_attitude_{k}": OutputTolerance(ulps=64, growth_ulps=32, floor=1.0) for k in range(4)},
            **{f"state_bias_{k}": OutputTolerance(ulps=16, growth_ulps=4, floor=_IMU_BIAS_LIMIT) for k in range(3)},
            **{f"out_0_{k}": OutputTolerance(ulps=64, growth_ulps=16, floor=10.0) for k in range(3)},
        },
        nominal=_IMU_FUSION_MANUAL[4],  # the first bias-railing in-band row
        manual=_IMU_FUSION_MANUAL,
        draw_random=_draw_imu_fusion,
        # The accelerometer lanes take the full format rails: an overflowed magnitude compares as infinity, the
        # gate rejects the row, and the ZKF infinity identities keep every flag clear. The gyro, temperature, and
        # dt lanes are pinned instead: a rail rate or temperature would overflow the state through the rate matrix
        # or the T^2 bias term and silently zero the quaternion (inf * 0 == +0), after which the renormalization
        # divides by zero on every later row; dt also divides the backward difference, so it must stay positive.
        edge_overrides={
            "gyro_0": (0.0, 0.25, -0.25, 3.5, -3.5),
            "gyro_1": (0.0, 0.25, -0.25, 3.5, -3.5),
            "gyro_2": (0.0, 0.25, -0.25, 3.5, -3.5),
            **{lane: (0.0, 1.0, -1.0, 0.5, -0.5) for lane in _IMU_FUSION_CAL},
            "temperature": (-40.0, 0.0, 25.0, 85.0, 125.0),
            "dt": (1e-3, 1e-2, 0.25),
        },
        edge_values=_WIDE_EDGES,
        formats=(_FMT, FloatFormat(6, 18)),  # the deep datapath, and the narrow one the synth matrix ships
        operators=lambda ops: dataclasses.replace(ops, fsort=FSortOptions()),
    ),
    ExampleSpec(
        name="ekf1_stateful",
        inputs=("dt", "u_shunt", "di_dt"),
        make_kernel=_fresh_stateful_ekf,
        # No spec-owned error budget has been derived for the carried covariance recurrence (cancellation-prone
        # P products through 1/S), so the tolerance check is not driven; the accuracy freeze measures the lanes
        # anyway and test_verify pins one step.
        reference=None,
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
    ExampleSpec(
        name="finite_set_current_controller",  # stateful; record/array parameters drive decomposed scalar lanes
        inputs=_FSCC_INPUTS,
        make_kernel=_fresh_finite_set_controller,
        # The bool switch lanes must match bit-for-bit -- the driven domain keeps the drive comparisons away from
        # near-ties, and an exact symmetric tie computes bit-equal drives on both sides, so the first-wins scan picks
        # the same candidate. The balance lanes carry the zero-mean recurrence's budget with a unit floor, since a
        # true-zero lane would make a relative budget vacuous.
        reference={f"out_switch_balance_{k}": OutputTolerance(ulps=64, growth_ulps=8, floor=1.0) for k in range(3)},
        nominal=_fscc_row(0.5, (1.0, -0.5, -0.5), (1e4, -1e4, 0.0), 100.0, (2.0, -1.0)),
        manual=[
            _fscc_row(0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0, (0.0, 0.0)),  # quiet: the all-false arm
            _fscc_row(0.0, (1.0, -1.0, 0.0), (0.0, 0.0, 0.0), 350.0, (5.0, 0.0)),
            _fscc_row(2.1, (-2.0, 3.0, -1.0), (5e4, -5e4, 0.0), 200.0, (-3.0, 4.0)),
            _fscc_row(0.0, (1.0, -1.0, 0.0), (0.0, 0.0, 0.0), 350.0, (5.0, 0.0)),  # revisit on evolved balance
            _fscc_row(-1.5, (0.5, 0.5, -1.0), (0.0, 1e5, -1e5), 1.0, (1.0, 1.0)),  # sub-threshold drive
        ],
        draw_random=_draw_fscc,
        # Finite magnitudes only: an infinity edge would send inf - inf through the zero-mean pipeline, whose
        # defined +0 answer raises the error sideband the generic bench asserts silent.
        edge_values=(0.0, 1.0, -1.0, 100.0, -100.0, 1e4, -1e4),
        operators=lambda ops: dataclasses.replace(ops, fsort=FSortOptions()),
    ),
    ExampleSpec(
        name="flux_observer",  # stateful record/2-vector I/O driven through its decomposed scalar port lanes
        inputs=_FLUX_OBSERVER_INPUTS,
        make_kernel=_fresh_flux_observer,
        # The i_last lanes are an identity install of the quantized input, so they must match bit-for-bit; the
        # rounded lanes carry budgets (the atan2 CORDIC is faithful rather than exact, and the flux recurrence
        # accumulates rounding through carried state at the flux-linkage scale).
        reference={
            "out_0": OutputTolerance(ulps=64),
            "state_flux_0": OutputTolerance(ulps=64, growth_ulps=1, floor=_FLUX_LINKAGE),
            "state_flux_1": OutputTolerance(ulps=64, growth_ulps=1, floor=_FLUX_LINKAGE),
        },
        nominal=_flux_row(1e-4, (1.0, 0.5), (2.0, -1.0)),
        manual=_FLUX_OBSERVER_MANUAL,
        draw_random=_draw_flux_observer,
        # The full format-edge sweep is safe on the voltage and current lanes because the clamp bounds the carried
        # flux every row: an infinite update lands as a railed ±flux_linkage, never as an infinity a later
        # opposite-signed row could cancel into inf - inf. The clamp itself is what holds that guarantee, so the
        # machine parameters are swept over physical magnitudes instead -- a railed flux_linkage would open the box
        # and let an infinite flux reach the atan2 with both operands infinite.
        edge_overrides={
            "params_R": (0.0, _PHASE_RESISTANCE, 5.0),
            "params_L_d": (0.0, _PHASE_INDUCTANCE, 1e-2),
            "params_flux_linkage": (0.0, 1e-3, _FLUX_LINKAGE, 1.0),
        },
        edge_values=_WIDE_EDGES,
        operators=lambda ops: dataclasses.replace(ops, fsort=FSortOptions()),
    ),
    ExampleSpec(
        name="foc",  # the capstone: the observer above embedded in a full sensorless current controller
        inputs=_FOC_INPUTS,
        make_kernel=_fresh_foc,
        # Four tiers, each measured against the driven sequence rather than assumed. The current samples the observer
        # carries are one Clarke product of the row, so they hold at the operand scale; the flux integrator and the
        # duty cycles it feeds accumulate one rounding per row; the dq frame and the commanded voltage close a loop
        # through the estimated angle, which is ill-conditioned wherever the flux vector passes near the origin; and
        # the speed estimate divides an angle difference by a PWM period, so it carries whatever the angle carries
        # amplified by the reciprocal of a period as short as ten microseconds -- the widest budget in the catalogue,
        # and still under a thousandth of a radian per second at the speeds these rows reach.
        reference={
            "out_0_0": OutputTolerance(ulps=64, growth_ulps=4),
            "out_0_1": OutputTolerance(ulps=64, growth_ulps=4),
            "out_0_2": OutputTolerance(ulps=64, growth_ulps=4),
            "out_1_0": OutputTolerance(ulps=256, growth_ulps=8, floor=1.0),
            "out_1_1": OutputTolerance(ulps=256, growth_ulps=8, floor=1.0),
            "state_integral_dq_0": OutputTolerance(ulps=128, growth_ulps=4, floor=1.0),
            "state_integral_dq_1": OutputTolerance(ulps=128, growth_ulps=4, floor=1.0),
            "state_observer_flux_0": OutputTolerance(ulps=64, growth_ulps=1, floor=_FLUX_LINKAGE),
            "state_observer_flux_1": OutputTolerance(ulps=64, growth_ulps=1, floor=_FLUX_LINKAGE),
            "state_observer_i_last_0": OutputTolerance(ulps=64, floor=1.0),
            "state_observer_i_last_1": OutputTolerance(ulps=64, floor=1.0),
            "state_omega": OutputTolerance(ulps=1024, growth_ulps=8),
            "state_theta_prev": OutputTolerance(ulps=64, growth_ulps=4),
            "state_u_alpha_beta_0": OutputTolerance(ulps=256, growth_ulps=8),
            "state_u_alpha_beta_1": OutputTolerance(ulps=256, growth_ulps=8),
        },
        # The Clarke and Park products and the voltage norm are numpy calls the host may serve from BLAS, whose
        # contracted products differ from the evaluator's separately rounded ones by an ulp the kernel then divides
        # by a PWM period into the speed estimate. The measured requirement is about two thousand float64 ulps, so
        # this is set a factor of four above it -- loose against a fold-exact kernel, and still four parts in a
        # trillion against a front end that got an operand or a term wrong.
        oracle_ulps=8192,
        nominal=_foc_row(2e-5, (2.0, -1.0), (0.0, 2.0), 24.0),
        manual=_FOC_MANUAL,
        draw_random=_draw_foc,
        # Every lane is pinned to physical magnitudes: the format rails have no meaning for a machine parameter, and
        # this kernel has no clamp standing between them and an undefined form -- a railed current overflows the
        # Park rotation into inf - inf, a subnormal period divides the angle difference into an infinite speed whose
        # sine is a NaN, and a zero bus divides the modulator.
        edge_overrides={
            "params_R": (0.0, _PHASE_RESISTANCE, 5.0),
            "params_L_dq_0": (0.0, _PHASE_INDUCTANCE, 1e-3),
            "params_L_dq_1": (0.0, _PHASE_INDUCTANCE, 1e-3),
            "params_flux_linkage": (0.0, 1e-3, _FLUX_LINKAGE, 0.05),
            "params_speed_filter_gain": (0.0, 0.05, 1.0),
            "dt": (1e-5, 2e-5, 1e-4),
            "v_dc": (6.0, 12.0, 24.0, 48.0),
        },
        edge_values=(0.0, 5.0, -5.0, 40.0, -40.0),  # the current lanes: quiet, nominal, and both saturating rails
        operators=lambda ops: dataclasses.replace(ops, fsort=FSortOptions()),
    ),
]
