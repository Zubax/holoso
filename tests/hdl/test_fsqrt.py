"""
Tests for holoso_fsqrt (pipelined; y = sgnop(sqrt(sgnop(a))); domain_error alongside out_valid).

The root is correctly rounded, so the value oracle is numpy -- an independent reference, not the ZKF model the DUT
mirrors. A negative conditioned operand yields -inf and raises domain_error; that poison value still passes through
the output sign conditioner, so the sgnop sweep is seeded with a zero and a negative operand.
"""

import os
from typing import Any

import cocotb
import numpy as np
import pytest
from cocotb.triggers import RisingEdge, Timer
from cocotb_tools.runner import get_runner

from holoso import FSqrtOptions, FloatFormat
from holoso._operators import FSqrtOperator

from .hdl_float_oracle import (
    DIRECTED_F32,
    HDL_DIR,
    PipelineScoreboard,
    REPO_ROOT,
    SGNOP_OPS,
    SIMULATORS,
    apply_sgnop,
    build_args,
    drive_reset,
    f32_to_bits,
    get_random_count,
    get_seed,
    random_zkf_f32,
    sources,
    sqrt_oracle,
    stage_tag,
    start_clock,
)

STAGE_COMBOS: tuple[dict[str, int], ...] = (
    {},
    {"stage_input": 1},
    {"stage_pack": 1, "stage_output": 1},
    {"stage_input": 1, "stage_pack": 1, "stage_output": 1},  # the staged fixture's combo (tests/_modelref.py)
    {"stage_input": 2, "stage_pack": 1, "stage_output": 1},
)


@cocotb.test()
async def holoso_fsqrt_cocotb(dut: Any) -> None:
    await start_clock(dut)
    await drive_reset(dut)

    sb = PipelineScoreboard(
        dut,
        [("y", "y"), ("domain_error", "domain_error")],
        latency=int(os.environ["HOLOSO_EXPECTED_LATENCY"]),
    )
    rng = np.random.default_rng(get_seed())

    async def step_idle() -> None:
        dut.in_valid.value = 0
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        sb.sample()

    async def step(a: int, a_op: int, y_op: int) -> None:
        y_pre, domain_error = sqrt_oracle(apply_sgnop(a, a_op))
        expected = {
            "_desc": f"a=0x{a:08x} ops={a_op}{y_op}",
            "y": apply_sgnop(y_pre, y_op),
            "domain_error": domain_error,
        }
        dut.a.value = a
        dut.a_sgnop.value = a_op
        dut.y_sgnop.value = y_op
        dut.in_valid.value = 1
        sb.push(expected)
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        sb.sample()

    for a in DIRECTED_F32:
        await step(a, 0, 0)
    await sb.drain()

    sample = [DIRECTED_F32[int(rng.integers(0, len(DIRECTED_F32)))] for _ in range(6)]
    sample += [0, f32_to_bits(-3.0)]
    for a_op in SGNOP_OPS:
        for y_op in SGNOP_OPS:
            for a in sample:
                await step(a, a_op, y_op)
    await sb.drain()

    for _ in range(get_random_count()):
        if rng.random() < 0.2:
            await step_idle()
            continue
        await step(random_zkf_f32(rng), int(rng.integers(0, 4)), int(rng.integers(0, 4)))
    await sb.drain()

    await drive_reset(dut)
    for _ in range(8):
        dut.in_valid.value = 0
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        assert int(dut.out_valid.value) == 0


@pytest.mark.parametrize("stages", STAGE_COMBOS, ids=stage_tag)
@pytest.mark.parametrize("sim", SIMULATORS)
def test_holoso_fsqrt(sim: str, stages: dict[str, int]) -> None:
    operator = FSqrtOperator(FloatFormat(8, 24), FSqrtOptions(**stages))
    runner = get_runner(sim)
    build_dir = REPO_ROOT / "build" / "cocotb" / sim / f"fsqrt_{stage_tag(stages)}"
    runner.build(
        sources=sources(),
        includes=[HDL_DIR],
        hdl_toplevel="holoso_fsqrt",
        parameters=operator.params,
        build_args=build_args(sim),
        build_dir=build_dir,
        clean=True,
        timescale=("1ns", "1ps"),
    )
    runner.test(
        hdl_toplevel="holoso_fsqrt",
        test_module="tests.hdl.test_fsqrt",
        test_dir=REPO_ROOT,
        build_dir=build_dir,
        extra_env={"HOLOSO_EXPECTED_LATENCY": str(operator.latency)},
        results_xml=str(build_dir / "results.xml"),
    )
