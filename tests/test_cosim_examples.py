"""
End-to-end cosimulation of every compilable example kernel: each is driven with hand-built sensible vectors, a frozen
random sweep, and format edge cases, then checked bit-for-bit against its embedded model under a lean (no optional
stages) and a deeply pipelined operator configuration at the wide e8m36 datapath.

Excluded examples are frontend feature gaps (not verification scope), confirmed by an in-memory compile probe:
  - iir1_lpf: ``UnsupportedConstruct: If`` -- the data-dependent first-sample branch is not lowerable.
  - iir1_hpf: ``UnsupportedConstruct: call to 'float'``.
  - finite_set_current_controller: ``UnsupportedConstruct`` -- nested/foreign attribute access.
"""

import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from holoso import FloatFormat
from ._cosim import run_cosim
from ._modelref import (
    bounded,
    default_ops,
    encode_inputs,
    format_edge_bits,
    log_uniform_positive,
    spd_matrix,
    staged_ops,
)
from .hdl.hdl_float_oracle import SIMULATORS

pytestmark = pytest.mark.cosim

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import ekf1_stateful  # noqa: E402
import ekf1_stateless  # noqa: E402
import madd  # noqa: E402
import poly3  # noqa: E402
from trapezoidal_leaky_streaming_integrator import TrapezoidalLeakyStreamingIntegrator  # noqa: E402

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
    nominal: dict[str, float]  # baseline for the per-input edge sweep (each input perturbed in turn)
    manual: list[dict[str, float]]  # sensible vectors; an ordered sequence for stateful kernels
    draw_random: Callable[[np.random.Generator], dict[str, float]]
    edge_values: tuple[float, ...]
    protected: frozenset[str] = frozenset()  # inputs swept only over positive edges to keep a divisor away from zero
    protected_values: tuple[float, ...] = ()

    def vectors(self) -> list[dict[str, int]]:
        """The full reproducible input sequence as input-name -> ZKF-bits rows: manual, then random, then edges."""
        rng = np.random.default_rng(_SEED)
        rows: list[dict[str, float]] = [*self.manual]
        rows += [self.draw_random(rng) for _ in range(_RANDOM_COUNT)]
        for name in self.inputs:
            values = self.protected_values if name in self.protected else self.edge_values
            rows += [{**self.nominal, name: value} for value in values]
        return [encode_inputs(_FMT, row) for row in rows]


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

_SPECS = [
    ExampleSpec(
        name="madd",
        inputs=("a", "b", "c"),
        make_kernel=lambda: madd.madd,
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
        name="poly3",
        inputs=("x", "c0", "c1", "c2", "c3"),
        make_kernel=lambda: poly3.poly3,
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
        name="integrator",
        inputs=("x",),
        make_kernel=lambda: TrapezoidalLeakyStreamingIntegrator(k=2**-22).__call__,
        nominal={"x": 1.0},
        manual=[  # one continuous stream: settle at zero, a step, an impulse, then a ramp
            *({"x": v} for v in (0.0, 0.0, 1.0, 1.0, 1.0, 1.0)),
            *({"x": v} for v in (0.0, 5.0, 0.0, 0.0)),
            *({"x": v} for v in (1.0, 2.0, 3.0, 4.0)),
        ],
        draw_random=_draw_scalars(("x",), -4.0, 4.0),
        edge_values=_WIDE_EDGES,
    ),
    ExampleSpec(
        name="ekf1_stateless",
        inputs=_EKF_STATELESS_INPUTS,
        make_kernel=lambda: ekf1_stateless.update_x_P,
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


# Each example is exercised at the lean default schedule and a deeply pipelined one, to explore the schedule and
# handshake at two latency points; both are bit-exact against the same model.
_OP_CONFIGS = [("default", default_ops), ("staged", staged_ops)]


@pytest.mark.parametrize("sim", SIMULATORS)
@pytest.mark.parametrize("config", _OP_CONFIGS, ids=lambda c: c[0])
@pytest.mark.parametrize("spec", _SPECS, ids=lambda s: s.name)
def test_example_cosim(spec: ExampleSpec, config: tuple[str, object], sim: str) -> None:
    label, make_ops = config
    run_cosim(sim, spec.make_kernel(), _FMT, f"{spec.name}_{label}", ops=make_ops(_FMT), vectors=spec.vectors())
