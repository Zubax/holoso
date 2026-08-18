"""
The Eel frontend differential oracle over example kernels: CPython executing the original kernel against the
evaluator running the unoptimized HIR lowered by `holoso._eel`. Every SPECS row rides its full shared
reference sequence; array-parameter kernels are driven directly through their decomposed leaf ports
(`v_0`, `v_1`) with local vector tables.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

from holoso._eel import lower

from ._eeloracle import InputRow, assert_hir_matches_reference
from ._modelref import DEFAULT_UNROLL_MAX_TRIPS
from ._examples import SPECS, ExampleSpec

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import polar  # noqa: E402
import rigid_body_rates  # noqa: E402


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


def _rigid_body_vectors() -> list[InputRow]:
    """
    A diagonal-inertia landmark, then eigenvalue-controlled SPD draws (the HIR side runs the library's pivoted
    Gauss-Jordan while CPython runs LAPACK, so the vectors keep the inversion mismatch far below the oracle's
    ulp budget: cond(inertia) ≤ 4, every omega component away from zero, and dt small).
    """
    landmark: dict[str, float] = {
        "inertia_0_0": 2.0, "inertia_0_1": 0.0, "inertia_0_2": 0.0,
        "inertia_1_0": 0.0, "inertia_1_1": 3.0, "inertia_1_2": 0.0,
        "inertia_2_0": 0.0, "inertia_2_1": 0.0, "inertia_2_2": 4.0,
        "omega_0": 0.5, "omega_1": -0.3, "omega_2": 0.8,
        "tau_0": 0.1, "tau_1": 0.0, "tau_2": -0.2,
        "dt": 0.005,
    }  # fmt: skip
    rows: list[InputRow] = [landmark]
    rng = np.random.default_rng(1687)
    for _ in range(12):
        while True:
            basis, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            inertia = basis @ np.diag(rng.uniform(0.5, 2.0, 3)) @ basis.T
            omega = np.array([float(rng.uniform(0.2, 1.0)) * float(rng.choice([-1.0, 1.0])) for _ in range(3)])
            if min(abs(float(l)) for l in inertia @ omega) >= 0.05:  # keep the momentum lanes off cancellation
                break
        row: dict[str, float] = {f"inertia_{i}_{j}": float(inertia[i, j]) for i in range(3) for j in range(3)}
        for k in range(3):
            row[f"omega_{k}"] = float(omega[k])
            row[f"tau_{k}"] = float(rng.uniform(-1.0, 1.0))
        row["dt"] = float(rng.uniform(1e-3, 1e-2))
        rows.append(row)
    return rows


_VECTOR_KERNELS: list[tuple[str, object, list[InputRow]]] = [
    ("to_polar", polar.to_polar, _polar_vectors()),
    ("from_polar", polar.from_polar, _polar_vectors()),
    ("rigid_body_rates", rigid_body_rates.update, _rigid_body_vectors()),
]


@pytest.mark.parametrize("spec", SPECS, ids=[spec.name for spec in SPECS])
def test_eel_oracle_on_examples(spec: ExampleSpec) -> None:
    hir = lower(spec.make_kernel(), DEFAULT_UNROLL_MAX_TRIPS).hir
    vectors = spec.reference_vectors()
    compared = assert_hir_matches_reference(hir, spec.make_kernel(), vectors, label=spec.name, ulps=spec.oracle_ulps)
    assert compared == len(vectors)


@pytest.mark.parametrize("name,kernel,vectors", _VECTOR_KERNELS, ids=[name for name, _, _ in _VECTOR_KERNELS])
def test_eel_oracle_on_array_kernels(name: str, kernel: object, vectors: list[InputRow]) -> None:
    assert callable(kernel)
    compared = assert_hir_matches_reference(lower(kernel, DEFAULT_UNROLL_MAX_TRIPS).hir, kernel, vectors, label=name)
    assert compared == len(vectors)
