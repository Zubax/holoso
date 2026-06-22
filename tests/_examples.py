"""
Shared example-kernel catalogue: each compilable example plus the domain knowledge needed to drive it -- a factory, a
baseline, curated and random vector generators, and the datapath format(s). Consumed by both the cosimulation suite
(``test_cosim_examples.py``, RTL vs the embedded model) and the Python-reference suite (``test_example_reference.py``,
the model vs the original Python), so the two views stay in lockstep over one source of truth.
"""

import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
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
import ekf1_stateful  # noqa: E402
import ekf1_stateless  # noqa: E402
import madd  # noqa: E402
import poly3  # noqa: E402
from cordic_sincos import CordicSinCos  # noqa: E402
from iir1_lpf import IIR1LPF  # noqa: E402
from latching_fault_register import LatchingFaultRegister  # noqa: E402
from majority_voter import MajorityVoter  # noqa: E402
from octave_index import octave_index  # noqa: E402
from pid import PID  # noqa: E402
from phase_frequency_detector import PhaseFrequencyDetector  # noqa: E402
from quadrature_encoder import QuadratureEncoder  # noqa: E402
from recip_newton import NewtonReciprocal  # noqa: E402
from remainder import remainder  # noqa: E402
from schmitt_trigger import SchmittTrigger  # noqa: E402
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
_EKF_POSITIVE_EDGES = (0.5, 1.0, _MIN_NORMAL, 1e6)


@dataclass(frozen=True)
class ExampleSpec:
    """One example kernel plus the domain knowledge to drive it: a factory, a baseline, and vector generators."""

    name: str
    inputs: tuple[str, ...]
    make_kernel: Callable[[], Callable[..., object]]
    nominal: dict[str, float | bool]  # baseline for the per-input edge sweep (each input perturbed in turn)
    manual: list[dict[str, float | bool]]  # sensible vectors; an ordered sequence for stateful kernels
    draw_random: Callable[[np.random.Generator], dict[str, float | bool]]
    edge_values: tuple[float | bool, ...]
    protected: frozenset[str] = frozenset()  # inputs swept only over positive edges to keep a divisor away from zero
    protected_values: tuple[float, ...] = ()
    # The float format(s) to drive at. The matrix is e8m36 by plan; a kernel that wants a second datapath (e.g. a
    # shallow e6m18 alongside the deep e8m36, to exercise both pipeline depths) lists both here.
    formats: tuple[FloatFormat, ...] = (_FMT,)
    # Whether the kernel's float outputs accumulate rounding (continuous arithmetic), so the Python-reference comparison
    # must allow a format-derived tolerance. False (the default) means every float output is exact in the format --
    # boolean logic, integer-valued counters/bytes, or exact (Sterbenz) reductions -- and must match Python bit-for-bit.
    approximate: bool = False
    # Whether the kernel reports its result purely as persistent PUBLIC VECTOR state (no scalar return or scalar state).
    # The generic Python-reference harness (``test_example_reference``) compares only scalar lanes, so such a kernel is
    # excluded there; its aggregate-state read-back is validated against Python separately in ``test_verify``.
    vector_public_state: bool = False

    def vectors(self, fmt: FloatFormat) -> list[dict[str, int]]:
        """The full reproducible input sequence as input-name -> ZKF-bits rows: manual, then random, then edges."""
        return [encode_inputs(fmt, row) for row in self.raw_vectors()]

    def reference_vectors(self) -> list[dict[str, float | bool]]:
        """
        The manual sequence then the random draw -- the inputs on which the ZKF model and the float64 Python reference
        agree to within the per-operation rounding tolerance, so the Python-reference suite drives this subset. The
        per-input format-edge sweep is intentionally excluded: at the format extremes the model legitimately diverges
        from float64 (an operation overflowing to the format's infinity stays finite in float64), a property of the
        datapath rather than a compiler defect, and the cosim suite (RTL == model) covers those edges instead.
        """
        rng = np.random.default_rng(_SEED)
        return [*self.manual, *(self.draw_random(rng) for _ in range(_RANDOM_COUNT))]

    def raw_vectors(self) -> list[dict[str, float | bool]]:
        """The full reproducible input sequence as raw float/bool rows: manual, then random, then per-input edges."""
        rows = self.reference_vectors()
        for name in self.inputs:
            values = self.protected_values if name in self.protected else self.edge_values
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

SPECS = [
    ExampleSpec(
        name="madd",
        inputs=("a", "b", "c"),
        make_kernel=lambda: madd.madd,
        approximate=True,  # continuous float arithmetic -> compare within a format tolerance
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
            {"x": 0.25, "lo": -0.5, "hi": 0.5},  # a narrower window
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
        approximate=True,  # continuous float arithmetic -> compare within a format tolerance
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
        approximate=True,  # continuous float arithmetic -> compare within a format tolerance
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
        inputs=("setpoint", "measurement"),
        make_kernel=lambda: PID().__call__,
        approximate=True,  # continuous float arithmetic -> compare within a format tolerance
        nominal={"setpoint": 1.0, "measurement": 0.0},
        manual=[  # first update (D suppressed), then a varying measurement (D active) driving both saturation rails
            {"setpoint": sp, "measurement": m}
            for sp, m in [(10.0, 0.0), (10.0, 0.5), (0.0, 1.0), (0.5, 0.5), (-10.0, 0.0), (-10.0, -0.5), (0.0, 0.0)]
        ],
        draw_random=_draw_scalars(("setpoint", "measurement"), -6.0, 6.0),
        edge_values=_WIDE_EDGES,
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
        approximate=True,  # continuous float arithmetic -> compare within a format tolerance
        nominal={"x": 1.0},
        manual=[{"x": v} for v in (0.5, 0.75, 1.0, 1.3, 1.7, 2.0)],  # across the [0.5, 2.0] reciprocal domain
        draw_random=_draw_scalars(("x",), 0.5, 2.0),
        # The Newton iteration only converges on its domain; off-domain x diverges and the data-dependent back-edge
        # loop never terminates, so the edge sweep is pinned to the domain rather than the full format edge set.
        protected=frozenset({"x"}),
        protected_values=(0.5, 0.75, 1.0, 1.5, 2.0),
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
        protected=frozenset({"y"}),
        protected_values=(0.25, 0.5, 1.0, 2.0, 4.0),
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
        protected=frozenset({"x"}),
        protected_values=(0.25, 0.5, 1.0, 2.0, 8.0),
        edge_values=(0.25, 0.5, 1.0, 2.0, 8.0),
        formats=(FloatFormat(6, 18), _FMT),  # the shallow and deep datapaths, both bit-exact against the model
    ),
    ExampleSpec(
        name="cordic_sincos",
        inputs=("theta",),
        make_kernel=lambda: CordicSinCos().__call__,
        approximate=True,  # continuous float arithmetic -> compare within a format tolerance
        nominal={"theta": 0.5},
        manual=[{"theta": v} for v in (0.0, 0.3, 0.7, -0.5, 1.0, -1.0)],  # angles within the convergence range
        draw_random=_draw_scalars(("theta",), -1.4, 1.4),
        edge_values=_WIDE_EDGES,
    ),
    ExampleSpec(
        name="integrator",
        inputs=("x", "dt"),
        make_kernel=lambda: TrapezoidalLeakyStreamingIntegrator(k=2**-22).__call__,
        approximate=True,  # continuous float arithmetic -> compare within a format tolerance
        nominal={"x": 1.0, "dt": 1.0e-3},
        manual=[  # one continuous stream: settle at zero, a step, an impulse, then a ramp
            *({"x": v, "dt": 1.0e-3} for v in (0.0, 0.0, 1.0, 1.0, 1.0, 1.0)),
            *({"x": v, "dt": 2.0e-3} for v in (0.0, 5.0, 0.0, 0.0)),
            *({"x": v, "dt": 5.0e-4} for v in (1.0, 2.0, 3.0, 4.0)),
        ],
        draw_random=lambda rng: {"x": bounded(rng, -4.0, 4.0), "dt": log_uniform_positive(rng, 1.0e-4, 1.0e-2)},
        protected=frozenset({"dt"}),
        protected_values=(0.0, 1.0e-4, 1.0e-3, 1.0e-2),
        edge_values=_WIDE_EDGES,
    ),
    ExampleSpec(
        name="ekf1_stateless",
        inputs=_EKF_STATELESS_INPUTS,
        make_kernel=lambda: ekf1_stateless.update_x_P,
        approximate=True,  # continuous float arithmetic -> compare within a format tolerance
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
        protected=frozenset({"R_ct", "R_shunt"}),
        protected_values=_EKF_POSITIVE_EDGES,
    ),
    ExampleSpec(
        name="ekf1_stateful",
        inputs=("dt", "u_shunt", "di_dt"),
        make_kernel=_fresh_stateful_ekf,
        vector_public_state=True,  # result is the carried x/P_urt vectors; excluded from the scalar reference harness
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
