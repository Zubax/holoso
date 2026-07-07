"""
Tests for holoso_fexp2 (pipelined; y = sgnop(2 ** sgnop(a))).

The oracle is the exact ZKF model (FloatValue.exp2); the vendored zkf_exp2 RTL is the independent hardware anchor, so
this bench proves the two agree bit-for-bit across the directed corner cases, the sign-conditioning sweep, and a random
sweep -- at several pipeline-stage configurations that also exercise the latency contract.
"""

import os
from typing import Any

import cocotb
import numpy as np
import pytest
from cocotb.triggers import RisingEdge, Timer
from cocotb_tools.runner import get_runner

from holoso import FExp2Operator, FloatFormat

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
    exp2_oracle_bits,
    get_random_count,
    get_seed,
    random_zkf_f32,
    sources,
    stage_tag,
    start_clock,
)

# A few stage configurations spanning every knob and a range of latencies (not the full cross-product); keyed by the
# operator's own field names so the DUT parameters come from hdl_params().
STAGE_COMBOS: tuple[dict[str, int], ...] = (
    {},
    {"stage_input": 1, "stage_output": 1},
    {"stage_reduce": 1, "stage_product": 2, "stage_pack": 1},  # stage_product=2 is the DSP-split path used in closure
)


@cocotb.test()
async def holoso_fexp2_cocotb(dut: Any) -> None:
    await start_clock(dut)
    await drive_reset(dut)

    sb = PipelineScoreboard(dut, [("y", "y")], latency=int(os.environ["HOLOSO_EXPECTED_LATENCY"]))
    rng = np.random.default_rng(get_seed())

    async def step_idle() -> None:
        dut.in_valid.value = 0
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        sb.sample()

    async def step(a: int, a_op: int, y_op: int) -> None:
        a_eff = apply_sgnop(a, a_op)
        expected = {
            "_desc": f"a=0x{a:08x} ops={a_op}{y_op}",
            "y": apply_sgnop(exp2_oracle_bits(a_eff), y_op),
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
def test_holoso_fexp2(sim: str, stages: dict[str, int]) -> None:
    operator = FExp2Operator(FloatFormat(8, 24), **stages)
    runner = get_runner(sim)
    build_dir = REPO_ROOT / "build" / "cocotb" / sim / f"fexp2_{stage_tag(stages)}"
    runner.build(
        sources=sources(),
        includes=[HDL_DIR],
        hdl_toplevel="holoso_fexp2",
        parameters={"WEXP": 8, "WMAN": 24, **operator.hdl_params(), "LATENCY": operator.latency},
        build_args=build_args(sim),
        build_dir=build_dir,
        clean=True,
        timescale=("1ns", "1ps"),
    )
    runner.test(
        hdl_toplevel="holoso_fexp2",
        test_module="tests.hdl.test_fexp2",
        test_dir=REPO_ROOT,
        build_dir=build_dir,
        extra_env={"HOLOSO_EXPECTED_LATENCY": str(operator.latency)},
        results_xml=str(build_dir / "results.xml"),
    )
