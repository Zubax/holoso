"""
Tests for holoso_fmul (pipelined; sgnop on a, b, y; y = sgnop(sgnop(a)*sgnop(b))).

The wrapper delays y_sgnop through the same number of stages as zkf_mul, so all sgnop controls are allowed to vary
every input cycle. The scoreboard verifies the documented wrapper latency against actual out_valid timing.
"""

import os

import cocotb
import numpy as np
import pytest
from cocotb.triggers import RisingEdge, Timer
from cocotb_tools.runner import get_runner

from holoso.format import FloatFormat
from holoso.operators import FMulOp

from hdl_float_oracle import (
    DIRECTED_F32,
    PipelineScoreboard,
    REPO_ROOT,
    SGNOP_OPS,
    SIMULATORS,
    BENCH_DIR,
    apply_sgnop,
    build_args,
    drive_reset,
    get_random_count,
    get_seed,
    mul_oracle_bits,
    random_zkf_f32,
    sources,
    start_clock,
)

STAGE_PRODUCT_VALUES = (0, 1)


@cocotb.test()
async def holoso_fmul_cocotb(dut) -> None:
    await start_clock(dut)
    await drive_reset(dut)

    sb = PipelineScoreboard(dut, [("y", "y")], latency=int(os.environ["HOLOSO_EXPECTED_LATENCY"]))
    rng = np.random.default_rng(get_seed())

    async def step_idle() -> None:
        dut.in_valid.value = 0
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        sb.sample()

    async def step(a: int, b: int, a_op: int, b_op: int, y_op: int) -> None:
        a_eff = apply_sgnop(a, a_op)
        b_eff = apply_sgnop(b, b_op)
        y_pre = mul_oracle_bits(a_eff, b_eff)
        if y_pre is None:
            await step_idle()
            return
        expected = apply_sgnop(y_pre, y_op)
        dut.a.value = a
        dut.b.value = b
        dut.a_sgnop.value = a_op
        dut.b_sgnop.value = b_op
        dut.y_sgnop.value = y_op
        dut.in_valid.value = 1
        sb.push({"y": expected, "_desc": f"a=0x{a:08x} b=0x{b:08x} ops={a_op}{b_op}{y_op}"})
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        sb.sample()

    for a in DIRECTED_F32:
        for b in DIRECTED_F32:
            await step(a, b, 0, 0, 0)
    await sb.drain()

    sample_pairs = [
        (DIRECTED_F32[int(rng.integers(0, len(DIRECTED_F32)))], DIRECTED_F32[int(rng.integers(0, len(DIRECTED_F32)))])
        for _ in range(8)
    ]
    for a_op in SGNOP_OPS:
        for b_op in SGNOP_OPS:
            for a, b in sample_pairs:
                for y_op in SGNOP_OPS:
                    await step(a, b, a_op, b_op, y_op)
    await sb.drain()

    for _ in range(get_random_count()):
        if rng.random() < 0.2:
            await step_idle()
            continue
        a = random_zkf_f32(rng)
        b = random_zkf_f32(rng)
        a_op = int(rng.integers(0, 4))
        b_op = int(rng.integers(0, 4))
        y_op = int(rng.integers(0, 4))
        await step(a, b, a_op, b_op, y_op)
    await sb.drain()

    await drive_reset(dut)
    for _ in range(8):
        dut.in_valid.value = 0
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        assert int(dut.out_valid.value) == 0


@pytest.mark.parametrize("stage_product", STAGE_PRODUCT_VALUES)
@pytest.mark.parametrize("sim", SIMULATORS)
def test_holoso_fmul(sim: str, stage_product: int) -> None:
    runner = get_runner(sim)
    build_dir = REPO_ROOT / "build" / "cocotb" / sim / f"fmul_sp{stage_product}"
    runner.build(
        sources=sources(),
        includes=[REPO_ROOT / "hdl"],
        hdl_toplevel="holoso_fmul",
        parameters={"WEXP": 8, "WMAN": 24, "STAGE_PRODUCT": stage_product},
        build_args=build_args(sim),
        build_dir=build_dir,
        clean=True,
        timescale=("1ns", "1ps"),
    )
    runner.test(
        hdl_toplevel="holoso_fmul",
        test_module="test_fmul",
        test_dir=BENCH_DIR,
        build_dir=build_dir,
        extra_env={"HOLOSO_EXPECTED_LATENCY": str(FMulOp(product=stage_product).latency(FloatFormat(8, 24)))},
        results_xml=str(build_dir / "results.xml"),
    )
