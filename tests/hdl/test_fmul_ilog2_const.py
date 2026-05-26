"""
Tests for holoso_fmul_ilog2_const (pipelined; y = sgnop(sgnop(a) * 2^K)).

K is a compile-time signed integer exponent shift. The test rebuilds the DUT for several K values to cover negative,
zero, and positive scales, and separately sweeps STAGE_DECODE at K=0. The wrapper delays y_sgnop through the same
number of stages as zkf_mul_ilog2_const, and the scoreboard verifies the documented latency against out_valid timing.
"""

import os

import cocotb
import numpy as np
import pytest
from cocotb.triggers import RisingEdge, Timer
from cocotb_tools.runner import get_runner

from holoso import FloatFormat
from holoso._operators import FMulILog2Op

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
    get_random_count,
    get_seed,
    mul_ilog2_oracle_bits,
    random_zkf_f32,
    sources,
    start_clock,
)

K_VALUES = (-5, -1, 0, 1, 5)


@cocotb.test()
async def holoso_fmul_ilog2_const_cocotb(dut) -> None:
    k_param = int(os.environ["HOLOSO_TEST_K"])

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
        y_pre = mul_ilog2_oracle_bits(a_eff, k_param)
        if y_pre is None:
            await step_idle()
            return
        expected = apply_sgnop(y_pre, y_op)
        dut.a.value = a
        dut.a_sgnop.value = a_op
        dut.y_sgnop.value = y_op
        dut.in_valid.value = 1
        sb.push({"y": expected, "_desc": f"a=0x{a:08x} ops=a{a_op}y{y_op} K={k_param}"})
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        sb.sample()

    # 1) Directed, neutral sgnops.
    for a in DIRECTED_F32:
        await step(a, 0, 0)
    await sb.drain()

    # 2) Sgnop sweep on directed values. Output sign control changes every cycle to verify sideband pipelining.
    for a_op in SGNOP_OPS:
        for a in DIRECTED_F32:
            for y_op in SGNOP_OPS:
                await step(a, a_op, y_op)
    await sb.drain()

    # 3) Random bulk.
    for _ in range(get_random_count()):
        if rng.random() < 0.2:
            await step_idle()
            continue
        a = random_zkf_f32(rng)
        a_op = int(rng.integers(0, 4))
        y_op = int(rng.integers(0, 4))
        await step(a, a_op, y_op)
    await sb.drain()

    # 4) Reset.
    await drive_reset(dut)
    for _ in range(8):
        dut.in_valid.value = 0
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        assert int(dut.out_valid.value) == 0


# Sweep K with STAGE_DECODE=0; add one STAGE_DECODE=1 case at K=0 to cover the extra stage.
CONFIG_MATRIX = [(k, 0) for k in K_VALUES] + [(0, 1)]


@pytest.mark.parametrize("config", CONFIG_MATRIX, ids=lambda c: f"k{c[0]}_d{c[1]}")
@pytest.mark.parametrize("sim", SIMULATORS)
def test_holoso_fmul_ilog2_const(sim: str, config: tuple[int, int]) -> None:
    k, stage_decode = config
    runner = get_runner(sim)
    build_dir = REPO_ROOT / "build" / "cocotb" / sim / f"fmlog_k{k}_d{stage_decode}"
    runner.build(
        sources=sources(),
        includes=[HDL_DIR],
        hdl_toplevel="holoso_fmul_ilog2_const",
        parameters={"WEXP": 8, "WMAN": 24, "K": k, "STAGE_DECODE": stage_decode},
        build_args=build_args(sim),
        build_dir=build_dir,
        clean=True,
        timescale=("1ns", "1ps"),
    )
    runner.test(
        hdl_toplevel="holoso_fmul_ilog2_const",
        test_module="tests.hdl.test_fmul_ilog2_const",
        test_dir=REPO_ROOT,
        build_dir=build_dir,
        extra_env={
            "HOLOSO_TEST_K": str(k),
            "HOLOSO_EXPECTED_LATENCY": str(FMulILog2Op(fmt=FloatFormat(8, 24), k=0, stage_decode=stage_decode).latency),
        },
        results_xml=str(build_dir / "results.xml"),
    )
