"""Tests for holoso_fisfinite (combinational; y=1 iff exponent != all-ones)."""

from __future__ import annotations

import cocotb
import numpy as np
import pytest
from cocotb.triggers import Timer
from cocotb_tools.runner import get_runner

from hdl_float_oracle import (
    DIRECTED_F32,
    F32_EXP_MASK,
    REPO_ROOT,
    SIMULATORS,
    BENCH_DIR,
    build_args,
    get_random_count,
    get_seed,
    random_zkf_f32,
    sources,
)


@cocotb.test()
async def holoso_fisfinite_cocotb(dut) -> None:
    async def check(x_bits: int) -> None:
        dut.x.value = x_bits
        await Timer(1, unit="ns")
        expected = 0 if (x_bits & F32_EXP_MASK) == F32_EXP_MASK else 1
        actual = int(dut.y.value)
        assert actual == expected, f"x=0x{x_bits:08x}: got {actual}, want {expected}"

    for x in DIRECTED_F32:
        await check(x)

    for x in (0x7FC00001, 0xFFC00001):
        await check(x)

    rng = np.random.default_rng(get_seed())
    for _ in range(get_random_count()):
        await check(random_zkf_f32(rng))


@pytest.mark.parametrize("sim", SIMULATORS)
def test_holoso_fisfinite(sim: str) -> None:
    runner = get_runner(sim)
    build_dir = REPO_ROOT / "build" / "cocotb" / sim / "fisfinite"
    runner.build(
        sources=sources(),
        includes=[REPO_ROOT / "hdl"],
        hdl_toplevel="holoso_fisfinite",
        parameters={"WEXP": 8, "WMAN": 24},
        build_args=build_args(sim),
        build_dir=build_dir,
        clean=True,
        timescale=("1ns", "1ps"),
    )
    runner.test(
        hdl_toplevel="holoso_fisfinite",
        test_module="test_fisfinite",
        test_dir=BENCH_DIR,
        build_dir=build_dir,
        results_xml=str(build_dir / "results.xml"),
    )
