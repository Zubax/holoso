"""
Tests for holoso_fatan2 (operand a is y, b is x; outputs theta in turns and mag = hypot). Like holoso_fsincos the
core holds one transaction in flight, so the bench drives one at a time and waits for out_valid; SIMULATION=1 arms
the over-issue $fatal.
"""

import os
from typing import Any

import cocotb
import numpy as np
import pytest
from cocotb.triggers import RisingEdge, Timer
from cocotb_tools.runner import get_runner

from holoso import FAtan2Operator, FloatFormat

from .hdl_float_oracle import (
    DIRECTED_F32,
    HDL_DIR,
    REPO_ROOT,
    SGNOP_OPS,
    SIMULATORS,
    apply_sgnop,
    atan2_oracle,
    build_args,
    drive_reset,
    get_random_count,
    get_seed,
    random_zkf_f32,
    sources,
    stage_tag,
    start_clock,
)

STAGE_COMBOS: tuple[dict[str, int], ...] = (
    {},
    {"stage_input": 1, "stage_output": 1},
    {"stage_product": 1, "stage_normalize": 1, "stage_pack": 1},
    {"unroll100": 50},
    {"unroll100": 200, "stage_product": 2},
    {"unroll100": 50, "stage_pack": 1, "stage_normalize": 2, "stage_product": 3},  # the to_polar timing-closed config
)


async def _atan2(dut: Any, latency: int, y: int, x: int, ops: tuple[int, int, int, int]) -> tuple[int, int]:
    dut.a.value = y
    dut.b.value = x
    dut.a_sgnop.value, dut.b_sgnop.value, dut.theta_sgnop.value, dut.mag_sgnop.value = ops
    dut.in_valid.value = 1
    await RisingEdge(dut.clk)
    dut.in_valid.value = 0
    for cycle in range(latency + 16):
        await Timer(1, unit="ns")
        if int(dut.out_valid.value) == 1:
            assert (
                cycle == latency - 1
            ), f"out_valid at cycle {cycle}, expected {latency - 1} (fixed accept-to-out_valid)"
            return int(dut.theta.value), int(dut.mag.value)
        await RisingEdge(dut.clk)
    raise AssertionError("out_valid never asserted")


@cocotb.test()
async def holoso_fatan2_cocotb(dut: Any) -> None:
    latency = int(os.environ["HOLOSO_EXPECTED_LATENCY"])
    await start_clock(dut)
    await drive_reset(dut)
    rng = np.random.default_rng(get_seed())

    async def check(y: int, x: int, ops: tuple[int, int, int, int] = (0, 0, 0, 0)) -> None:
        y_op, x_op, theta_op, mag_op = ops
        theta_pre, mag_pre = atan2_oracle(apply_sgnop(y, y_op), apply_sgnop(x, x_op))
        exp_theta, exp_mag = apply_sgnop(theta_pre, theta_op), apply_sgnop(mag_pre, mag_op)
        got_theta, got_mag = await _atan2(dut, latency, y, x, ops)
        assert (
            got_theta == exp_theta
        ), f"theta y=0x{y:08x} x=0x{x:08x} ops={ops}: 0x{got_theta:08x} != 0x{exp_theta:08x}"
        assert got_mag == exp_mag, f"mag y=0x{y:08x} x=0x{x:08x} ops={ops}: 0x{got_mag:08x} != 0x{exp_mag:08x}"
        await RisingEdge(dut.clk)

    # Directed grid over all four quadrants, the axes, and zero.
    grid = list(DIRECTED_F32[:9])
    for y in grid:
        for x in grid:
            await check(y, x)
    sample = [(grid[int(rng.integers(0, len(grid)))], grid[int(rng.integers(0, len(grid)))]) for _ in range(4)]
    for y_op in SGNOP_OPS:
        for x_op in SGNOP_OPS:
            for y, x in sample:
                await check(y, x, (y_op, x_op, 0, 0))
    for _ in range(get_random_count()):
        ops = (int(rng.integers(0, 4)), int(rng.integers(0, 4)), int(rng.integers(0, 4)), int(rng.integers(0, 4)))
        await check(random_zkf_f32(rng), random_zkf_f32(rng), ops)

    await drive_reset(dut)
    for _ in range(8):
        dut.in_valid.value = 0
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        assert int(dut.out_valid.value) == 0


@pytest.mark.parametrize("stages", STAGE_COMBOS, ids=stage_tag)
@pytest.mark.parametrize("sim", SIMULATORS)
def test_holoso_fatan2(sim: str, stages: dict[str, int]) -> None:
    operator = FAtan2Operator(FloatFormat(8, 24), **stages)
    runner = get_runner(sim)
    build_dir = REPO_ROOT / "build" / "cocotb" / sim / f"fatan2_{stage_tag(stages)}"
    runner.build(
        sources=sources(),
        includes=[HDL_DIR],
        hdl_toplevel="holoso_fatan2",
        parameters={"WEXP": 8, "WMAN": 24, **operator.hdl_params(), "LATENCY": operator.latency},
        build_args=build_args(sim),
        defines={"SIMULATION": 1},
        build_dir=build_dir,
        clean=True,
        timescale=("1ns", "1ps"),
    )
    runner.test(
        hdl_toplevel="holoso_fatan2",
        test_module="tests.hdl.test_fatan2",
        test_dir=REPO_ROOT,
        build_dir=build_dir,
        extra_env={"HOLOSO_EXPECTED_LATENCY": str(operator.latency)},
        results_xml=str(build_dir / "results.xml"),
    )
