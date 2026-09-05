"""
Per-lane accuracy guard over the example matrix: what `test_latency_freeze.py` does for the schedule, bounded the
way `test_metrics.py` bounds hardware cost rather than pinned to equality.

`test_example_reference.py` already bounds every float lane by a spec-owned `OutputTolerance`, but that budget is
a CONTRACT, derived from the source algorithm and deliberately generous: a lane may drift from 2.7 to 3.1 ulps
inside a budget of 16 and nothing notices, and that drift is exactly what reassociating a sum or contracting a
product moves. The bound here is the opposite kind -- taken from what the build actually achieves, so it sits
against the lane rather than above it, and the first drift trips it.

Bounded per lane, in ulps of the lane's own scale -- the scaling `OutputTolerance.allowance` applies where a lane
has a budget, so a figure reads against the one permitting it -- are the MAXIMUM, which the budget bounds, and the
RMS, which is what actually moves when a rewrite shifts many transactions a little rather than one a lot. A lane
whose figures both round away is omitted, so its bound is zero and any error there is a regression. Every spec is
measured at every format it declares and with `ffma` both absent and configured, since contraction changes which
operand is rounded and the reference suite drives only the first format at the default operator set.

Each figure is an upper bound taken on the build that froze it, not an equality: a rewrite that improves a lane
passes, and re-taking the row keeps the bound tight. Figures are independent of `HOLOSO_REGALLOC_EFFORT`, and of
`HOLOSO_TEST_RANDOM_COUNT` because the draw is pinned here rather than read from the environment.
"""

import dataclasses
import math

import pytest

import holoso
from holoso import FFmaOptions, FloatFormat, Options
from ._examples import DEFAULT_RANDOM_COUNT, SPECS, ExampleSpec
from ._modelref import unit_roundoff
from ._refdrive import FloatLane, drive

# Kernel label -> lane -> the (maximum, RMS) it may not exceed, each an upper bound taken on the build that froze
# it, as in `test_metrics.py`. A lane absent from a row was exact over the whole sequence, so its bound is zero.
# Rounded to two and three decimals: finer is last-bit noise in the float64 oracle, coarser hides a real drift.
_BASELINE: dict[str, dict[str, tuple[float, float]]] = {
    "madd-e8m36": {"out_0": (0.71, 0.216)},
    "madd-e8m36-ffma": {"out_0": (0.71, 0.216)},
    "signal_window-e8m36": {},
    "signal_window-e8m36-ffma": {},
    "poly3-e8m36": {"out_0": (0.18, 0.055)},
    "poly3-e8m36-ffma": {"out_0": (0.24, 0.046)},
    "iir1_lpf-e8m36": {"state_y": (0.17, 0.077)},
    "iir1_lpf-e8m36-ffma": {"state_y": (0.17, 0.077)},
    "iir1_hpf-e8m36": {"out_0": (0.26, 0.098), "state_lpf_y": (0.17, 0.077)},
    "iir1_hpf-e8m36-ffma": {"out_0": (0.26, 0.098), "state_lpf_y": (0.17, 0.077)},
    "pid-e8m36": {"out_0": (0.05, 0.011), "state_integral": (0.03, 0.023), "state_prev_error": (0.25, 0.078)},
    "pid-e8m36-ffma": {"out_0": (0.05, 0.011), "state_integral": (0.03, 0.022), "state_prev_error": (0.25, 0.078)},
    "schmitt_trigger-e8m36": {},
    "schmitt_trigger-e8m36-ffma": {},
    "quadrature_encoder-e8m36": {},
    "quadrature_encoder-e8m36-ffma": {},
    "phase_frequency_detector-e8m36": {},
    "phase_frequency_detector-e8m36-ffma": {},
    "latching_fault_register-e8m36": {},
    "latching_fault_register-e8m36-ffma": {},
    "majority_voter-e6m18": {},
    "majority_voter-e6m18-ffma": {},
    "iq_oscillator-e8m36": {"out_0": (0.41, 0.158), "out_1": (0.43, 0.152)},
    "iq_oscillator-e8m36-ffma": {"out_0": (0.41, 0.158), "out_1": (0.43, 0.152)},
    "nco-e6m18": {},
    "nco-e6m18-ffma": {},
    "image_agc_streamed-e8m36": {
        "state_analog_gain": (7.21, 1.694),
        "state_digital_gain": (8.59, 2.631),
        "state_exposure_s": (3.76, 1.31),
    },
    "image_agc_streamed-e8m36-ffma": {
        "state_analog_gain": (7.21, 1.694),
        "state_digital_gain": (8.59, 2.631),
        "state_exposure_s": (3.76, 1.31),
    },
    "pwm-e6m18": {},
    "pwm-e6m18-ffma": {},
    "debouncer-e6m18": {},
    "debouncer-e6m18-ffma": {},
    "priority_encoder-e6m18": {},
    "priority_encoder-e6m18-ffma": {},
    "crc32-e6m18": {},
    "crc32-e6m18-ffma": {},
    "lfsr16-e6m18": {},
    "lfsr16-e6m18-ffma": {},
    "uart_tx-e6m18": {},
    "uart_tx-e6m18-ffma": {},
    "uart_rx-e6m18": {},
    "uart_rx-e6m18-ffma": {},
    "recip_newton-e8m36": {"out_0": (0.7, 0.293)},
    "recip_newton-e8m36-ffma": {"out_0": (0.56, 0.232)},
    "remainder-e8m36": {},
    "remainder-e8m36-ffma": {},
    "octave_index-e6m18": {},
    "octave_index-e6m18-ffma": {},
    "octave_index-e8m36": {},
    "octave_index-e8m36-ffma": {},
    "equal_temperament-e8m36": {"out_0": (2.43, 0.736), "out_1": (0.75, 0.144)},
    "equal_temperament-e8m36-ffma": {"out_0": (2.43, 0.736), "out_1": (0.75, 0.133)},
    "cordic_sincos-e8m36": {"out_0": (1.12, 0.455), "out_1": (1.15, 0.432)},
    "cordic_sincos-e8m36-ffma": {"out_0": (1.12, 0.455), "out_1": (1.15, 0.432)},
    "polar_to-e8m36": {"out_0": (0.58, 0.206), "out_1": (0.8, 0.294)},
    "polar_to-e8m36-ffma": {"out_0": (0.58, 0.206), "out_1": (0.8, 0.294)},
    "polar_from-e8m36": {"out_0": (0.7, 0.212), "out_1": (1.01, 0.215)},
    "polar_from-e8m36-ffma": {"out_0": (0.7, 0.212), "out_1": (1.01, 0.215)},
    "rigid_body_scalar-e8m36": {
        "out_0": (0.14, 0.065),
        "out_1": (0.12, 0.055),
        "out_2": (0.12, 0.057),
        "out_3": (0.1, 0.035),
        "out_4": (0.1, 0.032),
        "out_5": (0.13, 0.038),
    },
    "rigid_body_scalar-e8m36-ffma": {
        "out_0": (0.12, 0.064),
        "out_1": (0.12, 0.055),
        "out_2": (0.12, 0.057),
        "out_3": (0.08, 0.027),
        "out_4": (0.1, 0.032),
        "out_5": (0.08, 0.023),
    },
    "kepler-e8m36": {"out_0": (0.36, 0.153)},
    "kepler-e8m36-ffma": {"out_0": (0.36, 0.148)},
    "integrator-e8m36": {"state_y": (0.76, 0.32)},
    "integrator-e8m36-ffma": {"state_y": (0.45, 0.176)},
    "ekf1_stateless-e8m36": {
        "out_0_0": (0.22, 0.066),
        "out_1_0": (0.2, 0.065),
        "out_2_0": (0.23, 0.068),
        "out_3_0": (0.33, 0.11),
        "out_4_0": (0.09, 0.026),
        "out_5_0": (0.12, 0.04),
        "out_6_0": (0.33, 0.099),
        "out_7_0": (0.14, 0.035),
        "out_8_0": (0.65, 0.15),
    },
    "ekf1_stateless-e8m36-ffma": {
        "out_0_0": (0.16, 0.059),
        "out_1_0": (0.2, 0.065),
        "out_2_0": (0.19, 0.048),
        "out_3_0": (0.33, 0.106),
        "out_4_0": (0.09, 0.026),
        "out_5_0": (0.12, 0.035),
        "out_6_0": (0.33, 0.099),
        "out_7_0": (0.09, 0.029),
        "out_8_0": (0.65, 0.15),
    },
    "fir-e8m36": {"out_0": (0.3, 0.121)},
    "fir-e8m36-ffma": {"out_0": (0.31, 0.112)},
    "biquad-e8m36": {"out_0": (0.25, 0.102), "state_s1": (0.28, 0.112), "state_s2": (0.12, 0.046)},
    "biquad-e8m36-ffma": {"out_0": (0.56, 0.127), "state_s1": (0.41, 0.11), "state_s2": (0.12, 0.048)},
    "imu_fusion-e8m36": {
        "out_0_0": (3.47, 0.64),
        "out_0_1": (1.35, 0.539),
        "out_0_2": (2.15, 0.657),
        "state_attitude_0": (1.12, 0.496),
        "state_attitude_1": (0.51, 0.247),
        "state_attitude_2": (0.74, 0.39),
        "state_attitude_3": (1.34, 0.619),
        "state_bias_0": (0.16, 0.159),
        "state_bias_1": (0.16, 0.159),
        "state_bias_2": (0.16, 0.159),
    },
    "imu_fusion-e8m36-ffma": {
        "out_0_0": (2.67, 0.537),
        "out_0_1": (1.75, 0.801),
        "out_0_2": (1.85, 0.799),
        "state_attitude_0": (1.62, 0.635),
        "state_attitude_1": (0.84, 0.271),
        "state_attitude_2": (0.74, 0.368),
        "state_attitude_3": (2.2, 1.017),
        "state_bias_0": (0.16, 0.159),
        "state_bias_1": (0.16, 0.159),
        "state_bias_2": (0.16, 0.159),
    },
    "imu_fusion-e6m18": {
        "out_0_0": (2.93, 0.822),
        "out_0_1": (2.3, 0.987),
        "out_0_2": (3.15, 1.392),
        "state_attitude_0": (1.13, 0.502),
        "state_attitude_1": (1.71, 1.028),
        "state_attitude_2": (0.82, 0.352),
        "state_attitude_3": (0.93, 0.652),
        "state_bias_0": (0.27, 0.257),
        "state_bias_1": (0.27, 0.257),
        "state_bias_2": (0.27, 0.257),
    },
    "imu_fusion-e6m18-ffma": {
        "out_0_0": (2.56, 0.734),
        "out_0_1": (1.64, 0.9),
        "out_0_2": (2.35, 0.661),
        "state_attitude_0": (1.21, 0.607),
        "state_attitude_1": (0.96, 0.482),
        "state_attitude_2": (1.26, 0.52),
        "state_attitude_3": (0.91, 0.525),
        "state_bias_0": (0.27, 0.257),
        "state_bias_1": (0.27, 0.257),
        "state_bias_2": (0.27, 0.257),
    },
    "ekf1_stateful-e8m36": {
        "state_P_urt_0": (8.35, 4.183),
        "state_P_urt_1": (2.98, 1.121),
        "state_P_urt_2": (11.18, 5.187),
        "state_P_urt_3": (3.99, 1.97),
        "state_P_urt_4": (7.67, 3.129),
        "state_P_urt_5": (8.06, 5.263),
        "state_x_0": (3.74, 1.354),
        "state_x_1": (4.35, 1.898),
        "state_x_2": (3.76, 1.347),
    },
    "ekf1_stateful-e8m36-ffma": {
        "state_P_urt_0": (8.35, 4.183),
        "state_P_urt_1": (2.23, 0.853),
        "state_P_urt_2": (10.32, 4.44),
        "state_P_urt_3": (3.99, 1.97),
        "state_P_urt_4": (6.79, 2.318),
        "state_P_urt_5": (8.06, 5.263),
        "state_x_0": (3.45, 1.753),
        "state_x_1": (2.86, 1.252),
        "state_x_2": (3.88, 1.783),
    },
    "finite_set_current_controller-e8m36": {
        "out_switch_balance_0": (2.83, 1.235),
        "out_switch_balance_1": (3.0, 1.071),
        "out_switch_balance_2": (8.17, 3.356),
    },
    "finite_set_current_controller-e8m36-ffma": {
        "out_switch_balance_0": (2.83, 1.235),
        "out_switch_balance_1": (3.0, 1.071),
        "out_switch_balance_2": (8.17, 3.356),
    },
    "flux_observer-e8m36": {"out_0": (0.92, 0.369), "state_flux_0": (1.01, 0.311), "state_flux_1": (1.09, 0.241)},
    "flux_observer-e8m36-ffma": {"out_0": (2.33, 0.446), "state_flux_0": (1.01, 0.319), "state_flux_1": (0.67, 0.182)},
    "foc-e8m36": {
        "out_0_0": (9.97, 1.915),
        "out_0_1": (9.97, 2.446),
        "out_0_2": (11.46, 2.276),
        "out_1_0": (83.44, 15.312),
        "out_1_1": (177.39, 34.731),
        "state_integral_dq_0": (34.32, 4.809),
        "state_integral_dq_1": (33.89, 12.423),
        "state_observer_flux_0": (8.36, 3.597),
        "state_observer_flux_1": (4.4, 1.452),
        "state_observer_i_last_1": (3.32, 0.641),
        "state_omega": (280.24, 53.406),
        "state_theta_prev": (21.42, 5.046),
        "state_u_alpha_beta_0": (59.41, 15.855),
        "state_u_alpha_beta_1": (62.44, 17.17),
    },
    "foc-e8m36-ffma": {
        "out_0_0": (3.9, 0.696),
        "out_0_1": (4.17, 1.059),
        "out_0_2": (3.92, 0.897),
        "out_1_0": (35.39, 5.806),
        "out_1_1": (55.99, 11.057),
        "state_integral_dq_0": (9.59, 1.495),
        "state_integral_dq_1": (12.75, 4.751),
        "state_observer_flux_0": (3.08, 1.095),
        "state_observer_flux_1": (1.85, 0.621),
        "state_observer_i_last_1": (1.7, 0.433),
        "state_omega": (238.24, 46.597),
        "state_theta_prev": (7.64, 1.721),
        "state_u_alpha_beta_0": (22.17, 6.261),
        "state_u_alpha_beta_1": (28.65, 7.415),
    },
}


def _label(spec: ExampleSpec, fmt: FloatFormat, fma: bool) -> str:
    return f"{spec.name}-e{fmt.wexp}m{fmt.wman}{'-ffma' if fma else ''}"


def _options(spec: ExampleSpec, fmt: FloatFormat, fma: bool) -> Options:
    """
    Both settings are imposed, not merely offered: a spec that configures `ffma` itself would otherwise make
    the two rows one configuration measured twice.
    """
    options = spec.options(fmt)
    ffma = FFmaOptions() if fma else None
    return dataclasses.replace(options, operator=dataclasses.replace(options.operator, ffma=ffma))


_CASES = [(spec, fmt, fma) for spec in SPECS for fmt in spec.formats for fma in (False, True)]
"""
Every spec, including the one the tolerance check skips: a freeze needs a reference to measure against, not a
budget to stay under, and the lane whose budget was never derived is exactly the one nothing else watches.
"""


def _ulps(got: float, want: float, unit: float, floor: float) -> float:
    """
    The error in ulps of the reference's own magnitude. Agreement is answered before any arithmetic, since a lane
    that legitimately reaches an infinity agrees with the reference exactly and would otherwise divide `inf` by
    `inf`; any disagreement involving a non-finite value is total, and a NaN read as exact through `max` would
    report a broken lane as clean.
    """
    if got == want:
        return 0.0
    if not (math.isfinite(got) and math.isfinite(want)):
        return math.inf
    return abs(got - want) / (unit * max(abs(want), floor))


def _rms(values: list[float]) -> float:
    return math.sqrt(sum(v * v for v in values) / len(values))


def _floor(spec: ExampleSpec, port: str, wants: list[float]) -> float:
    """
    The magnitude a lane's error is read against wherever its own reference value falls below it. A budgeted lane
    takes the spec's floor, so the frozen figure reads against the budget permitting it. A lane with no budget
    takes the scale of its own reference sequence: measured against 1.0, a lane living at 1e-9 reads as exact
    however far it drifts, which is how the covariance cross-terms sat at 0.00 while carrying tens of ulps.
    """
    budget = None if spec.reference is None else spec.reference.get(port)
    if budget is not None:
        return budget.floor
    finite = [abs(want) for want in wants if math.isfinite(want)]
    scale = _rms(finite) if finite else 0.0
    return scale if scale > 0.0 else 1.0


def _lane_errors(spec: ExampleSpec, fmt: FloatFormat, fma: bool) -> dict[str, tuple[float, float]]:
    """Each float lane's (max, RMS) error over the reference sequence, in ulps of the lane's own scale."""
    label = _label(spec, fmt, fma)
    model = holoso.synthesize(spec.make_kernel(), _options(spec, fmt, fma), name=spec.name).numerical_model.elaborate()
    unit = unit_roundoff(fmt)
    lanes: dict[str, list[tuple[float, float]]] = {}
    for row in drive(label, model, spec.make_kernel(), spec.inputs, spec.reference_vectors(DEFAULT_RANDOM_COUNT), fmt):
        if isinstance(row, FloatLane):
            lanes.setdefault(row.port, []).append((float(row.got), row.want))
    errors = {}
    for port, pairs in sorted(lanes.items()):
        floor = _floor(spec, port, [want for _, want in pairs])
        ulps = [_ulps(got, want, unit, floor) for got, want in pairs]
        worst, rms = round(max(ulps), 2), round(_rms(ulps), 3)
        if worst > 0.0 or rms > 0.0:
            errors[port] = (worst, rms)
    return errors


def test_every_case_is_bounded() -> None:
    """
    A row outliving the case it was taken for: the parametrized test below fails on its own where a row is MISSING,
    but a stale one it never reads would sit here unnoticed.
    """
    assert sorted(_BASELINE) == sorted(_label(spec, fmt, fma) for spec, fmt, fma in _CASES)


@pytest.mark.parametrize("spec,fmt,fma", [pytest.param(s, f, m, id=_label(s, f, m)) for s, f, m in _CASES])
def test_lane_accuracy_does_not_regress(spec: ExampleSpec, fmt: FloatFormat, fma: bool) -> None:
    label = _label(spec, fmt, fma)
    baseline = _BASELINE[label]
    regressed = {}
    for port, (worst, rms) in _lane_errors(spec, fmt, fma).items():
        # A lane the row omits was exact when the row was taken, so its bound is zero and any error is a regression.
        bounds = baseline.get(port, (0.0, 0.0))
        if worst > bounds[0] or rms > bounds[1]:
            regressed[port] = {"bound": bounds, "measured": (worst, rms)}
    assert not regressed, (
        f"{label}: per-lane accuracy regressed.\n"
        f"  {regressed}\n"
        "A lane that got worse needs a reason. One that got better passes as it stands; re-take the row in the "
        "same commit to keep the bound tight."
    )
