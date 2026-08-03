"""
The Eel frontend differential oracle over example kernels: CPython executing the original kernel against the
evaluator running the unoptimized HIR lowered by ``holoso._eel``. ``ORACLE_COVERED`` is the set of example
names verified here. Array-parameter kernels are driven directly through their
decomposed leaf ports (``v_0``, ``v_1``) with local vector tables; scalar kernels ride the shared SPECS rows.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

from holoso._eel import lower

from ._eeloracle import InputRow, assert_hir_matches_reference
from ._examples import SPECS, ExampleSpec

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import imu_frame_transform  # noqa: E402
import polar  # noqa: E402

ORACLE_COVERED = frozenset(
    {
        "madd",
        "poly3",
        "signal_window",
        "equal_temperament",
        "ekf1_stateless",
        "polar_to",
        "polar_from",
        "to_polar",
        "from_polar",
        "octave_index",
        "remainder",
        "kepler",
        "recip_newton",
        "cordic_sincos",
        "imu_frame_transform",
        "iir1_lpf",
        "pid",
        "schmitt_trigger",
        "quadrature_encoder",
        "phase_frequency_detector",
        "latching_fault_register",
        "majority_voter",
        "integrator",
        "uart_tx",
        "uart_rx",
        "ekf1_stateful",
    }
)

_SPEC_COVERED = {spec.name for spec in SPECS} & ORACLE_COVERED
_SPECS = [spec for spec in SPECS if spec.name in _SPEC_COVERED]


def _polar_vectors() -> list[InputRow]:
    rows: list[InputRow] = [
        {"v_0": 3.0, "v_1": 4.0},
        {"v_0": -1.0, "v_1": 2.0},
        {"v_0": -2.0, "v_1": -1.5},
        {"v_0": 0.5, "v_1": -0.5},
        {"v_0": 1.0, "v_1": 0.0},
        {"v_0": 0.0, "v_1": 1.0},
        {"v_0": 0.0, "v_1": 0.0},
        {"v_0": 2.0, "v_1": math.pi / 2},
        {"v_0": 1.5, "v_1": -math.pi},
    ]
    rng = np.random.default_rng(2024)
    rows += [{"v_0": float(rng.uniform(-4.0, 4.0)), "v_1": float(rng.uniform(-4.0, 4.0))} for _ in range(20)]
    return rows


def _imu_vectors() -> list[InputRow]:
    """A right-angle rotation as the readable landmark, then random poses across all four array inputs."""
    rng = np.random.default_rng(1959)
    yaw90 = {"R_0_0": 0.0, "R_0_1": -1.0, "R_0_2": 0.0, "R_1_0": 1.0, "R_1_1": 0.0, "R_1_2": 0.0}
    yaw90 |= {"R_2_0": 0.0, "R_2_1": 0.0, "R_2_2": 1.0}
    landmark: dict[str, float] = {
        **yaw90,
        "t_0": 1.0,
        "t_1": 2.0,
        "t_2": 3.0,
        "a_meas_0": 0.1,
        "a_meas_1": -0.2,
        "a_meas_2": 9.9,
        "p_sensor_0": 2.0,
        "p_sensor_1": 0.0,
        "p_sensor_2": -1.0,
    }
    names = sorted(landmark)
    rows: list[InputRow] = [landmark]
    rows += [{name: float(rng.uniform(-3.0, 3.0)) for name in names} for _ in range(16)]
    return rows


_VECTOR_KERNELS: list[tuple[str, object, list[InputRow]]] = [
    ("to_polar", polar.to_polar, _polar_vectors()),
    ("from_polar", polar.from_polar, _polar_vectors()),
    ("imu_frame_transform", imu_frame_transform.transform, _imu_vectors()),
]


def test_every_covered_name_is_verified_somewhere() -> None:
    verified = {spec.name for spec in SPECS} | {name for name, _, _ in _VECTOR_KERNELS}
    assert ORACLE_COVERED <= verified


@pytest.mark.parametrize("spec", _SPECS, ids=[spec.name for spec in _SPECS])
def test_eel_oracle_on_examples(spec: ExampleSpec) -> None:
    hir = lower(spec.make_kernel())
    vectors = spec.reference_vectors()
    compared = assert_hir_matches_reference(hir, spec.make_kernel(), vectors, label=spec.name)
    assert compared == len(vectors)


@pytest.mark.parametrize("name,kernel,vectors", _VECTOR_KERNELS, ids=[name for name, _, _ in _VECTOR_KERNELS])
def test_eel_oracle_on_array_kernels(name: str, kernel: object, vectors: list[InputRow]) -> None:
    assert callable(kernel)
    compared = assert_hir_matches_reference(lower(kernel), kernel, vectors, label=name)
    assert compared == len(vectors)
