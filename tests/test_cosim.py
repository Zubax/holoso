"""Functional cosimulation: drive generated modules and check outputs against the float64 reference within tolerance."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from cocotb_tools.runner import get_runner

from holoso.backend_verilog import generate
from holoso.format import FloatFormat
from holoso.frontend import lower
from holoso.passes import run
from holoso.schedule import build, metrics_of
from holoso.verify.cosim import Sampler, build_vectors, generic_sampler

from hdl_float_oracle import HDL_DIR, REPO_ROOT, SIMULATORS, BENCH_DIR, build_args, sources


def _run_cosim(
    sim: str, fn: Callable[..., object], fmt: FloatFormat, name: str, count: int, sampler: Sampler = generic_sampler
) -> None:
    lir = build(run(lower(fn, fmt)), name)
    metrics = metrics_of(lir)
    # Generated sources live outside the cocotb build dir, which the runner wipes on clean=True.
    gen_dir = REPO_ROOT / "build" / "holoso_gen" / f"{name}_w{fmt.wexp}_{fmt.wman}"
    gen_dir.mkdir(parents=True, exist_ok=True)
    build_dir = REPO_ROOT / "build" / "cocotb" / sim / f"synth_{name}_w{fmt.wexp}_{fmt.wman}"
    verilog_path = gen_dir / f"{name}.v"
    verilog_path.write_text(generate(lir))

    spec = build_vectors(
        fn,
        fmt,
        [load.name for load in lir.inputs],
        [wire.name for wire in lir.outputs],
        metrics.op_count,
        count=count,
        rng=np.random.default_rng(0xC0FFEE),
        timeout_cycles=max(64, metrics.ii_cycles * 4),
        cycles=metrics.ii_cycles,  # == schedule.cycle_count(lir); the driver asserts the DUT matches it exactly
        sampler=sampler,
    )
    vectors_path = gen_dir / "vectors.json"
    vectors_path.write_text(json.dumps(spec))

    runner = get_runner(sim)
    runner.build(
        sources=[verilog_path, *sources()],
        includes=[HDL_DIR],
        hdl_toplevel=name,
        parameters={"WEXP": fmt.wexp, "WMAN": fmt.wman},
        build_args=build_args(sim),
        build_dir=str(build_dir),
        clean=True,
        timescale=("1ns", "1ps"),
    )
    runner.test(
        hdl_toplevel=name,
        test_module="cosim_driver",
        test_dir=str(BENCH_DIR),
        build_dir=str(build_dir),
        extra_env={"HOLOSO_VECTORS": str(vectors_path)},
        results_xml=str(build_dir / "results.xml"),
    )


def _positive_sampler(fmt: FloatFormat, names: object, rng: np.random.Generator) -> dict[str, float]:
    return {name: float(rng.uniform(0.5, 4.0)) for name in names}  # type: ignore[attr-defined]


def _ekf1_sampler(fmt: FloatFormat, names: object, rng: np.random.Generator) -> dict[str, float]:
    from holoso.verify.sampling import bounded, log_uniform_positive, spd_matrix

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
        "R_ct": log_uniform_positive(rng, 1e-1, 1.0),
        "R_shunt": log_uniform_positive(rng, 1e-1, 1.0),
        "dt": bounded(rng, 1e-3, 1e-2),
        "x_R": bounded(rng, -1.0, 1.0),
        "x_g": bounded(rng, -1.0, 1.0),
        "x_i": bounded(rng, -1.0, 1.0),
        "z_ct": bounded(rng, -1.0, 1.0),
        "z_shunt": bounded(rng, -1.0, 1.0),
    }


@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_small_kernel(sim: str) -> None:
    def kernel(a, b):  # type: ignore[no-untyped-def]
        return (a - b) * 0.25 + a * b

    _run_cosim(sim, kernel, FloatFormat(8, 24), "kernel", count=64)


@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_division(sim: str) -> None:
    def blend(a, b, c):  # type: ignore[no-untyped-def]
        return a / b + c * 2.0

    _run_cosim(sim, blend, FloatFormat(6, 18), "blend", count=64, sampler=_positive_sampler)


@pytest.mark.slow
@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_ekf1(sim: str) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    import ekf1

    _run_cosim(sim, ekf1.update_x_P, FloatFormat(6, 18), "update_x_P", count=24, sampler=_ekf1_sampler)
