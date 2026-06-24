"""
Tests for holoso_fsort (pipelined; min/max with input + output sgnops).

The wrapper delays min_sgnop and max_sgnop through the same number of stages as zkf_sort, so output sign controls are
allowed to vary every input cycle. The test also skips any case where applying sgnop to an input would produce a
non-canonical -0, since sort preserves the input bit pattern through to its outputs and -0 is outside the ZKF input
contract.
"""

import os

import cocotb
import numpy as np
import pytest
from cocotb.triggers import RisingEdge, Timer
from cocotb_tools.runner import get_runner

from .hdl_float_oracle import (
    DIRECTED_F32,
    PipelineScoreboard,
    HDL_DIR,
    REPO_ROOT,
    SGNOP_OPS,
    SIMULATORS,
    apply_sgnop,
    build_args,
    drive_reset,
    get_random_count,
    get_seed,
    is_neg_zero_f32,
    random_zkf_f32,
    sort_oracle_bits,
    sources,
    start_clock,
)


@cocotb.test()
async def holoso_fsort_cocotb(dut) -> None:
    await start_clock(dut)
    await drive_reset(dut)

    sb = PipelineScoreboard(dut, [("min", "min"), ("max", "max")], latency=int(os.environ["HOLOSO_EXPECTED_LATENCY"]))
    rng = np.random.default_rng(get_seed())

    async def step_idle() -> None:
        dut.in_valid.value = 0
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        sb.sample()

    async def step(a: int, b: int, a_op: int, b_op: int, mn_op: int, mx_op: int) -> None:
        a_eff = apply_sgnop(a, a_op)
        b_eff = apply_sgnop(b, b_op)
        # Skip sgnop-induced non-canonical -0; ZKF doesn't define it as an input, and sort preserves the bit pattern
        # of its inputs through to its outputs, so the DUT would return -0 bits where the oracle says +0 (after
        # flushing).
        if is_neg_zero_f32(a_eff) or is_neg_zero_f32(b_eff):
            await step_idle()
            return
        mn_pre, mx_pre = sort_oracle_bits(a_eff, b_eff)
        mn_exp = apply_sgnop(mn_pre, mn_op)
        mx_exp = apply_sgnop(mx_pre, mx_op)
        dut.a.value = a
        dut.b.value = b
        dut.a_sgnop.value = a_op
        dut.b_sgnop.value = b_op
        dut.min_sgnop.value = mn_op
        dut.max_sgnop.value = mx_op
        dut.in_valid.value = 1
        sb.push(
            {
                "min": mn_exp,
                "max": mx_exp,
                "_desc": f"a=0x{a:08x} b=0x{b:08x} ops={a_op}{b_op}{mn_op}{mx_op}",
            }
        )
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        sb.sample()

    for a in DIRECTED_F32:
        for b in DIRECTED_F32:
            await step(a, b, 0, 0, 0, 0)
    await sb.drain()

    # Full sgnop sweep: output sign controls change every cycle to verify sideband pipelining.
    sample_pairs = [
        (DIRECTED_F32[int(rng.integers(0, len(DIRECTED_F32)))], DIRECTED_F32[int(rng.integers(0, len(DIRECTED_F32)))])
        for _ in range(4)
    ]
    for a_op in SGNOP_OPS:
        for b_op in SGNOP_OPS:
            for a, b in sample_pairs:
                for mn_op in SGNOP_OPS:
                    for mx_op in SGNOP_OPS:
                        await step(a, b, a_op, b_op, mn_op, mx_op)
    await sb.drain()

    for _ in range(get_random_count()):
        if rng.random() < 0.2:
            await step_idle()
            continue
        a = random_zkf_f32(rng)
        b = random_zkf_f32(rng)
        a_op = int(rng.integers(0, 4))
        b_op = int(rng.integers(0, 4))
        mn_op = int(rng.integers(0, 4))
        mx_op = int(rng.integers(0, 4))
        await step(a, b, a_op, b_op, mn_op, mx_op)
    await sb.drain()

    await drive_reset(dut)
    for _ in range(8):
        dut.in_valid.value = 0
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        assert int(dut.out_valid.value) == 0


@pytest.mark.parametrize("sim", SIMULATORS)
def test_holoso_fsort(sim: str) -> None:
    runner = get_runner(sim)
    build_dir = REPO_ROOT / "build" / "cocotb" / sim / "fsort"
    runner.build(
        sources=sources(),
        includes=[HDL_DIR],
        hdl_toplevel="holoso_fsort",
        parameters={"WEXP": 8, "WMAN": 24, "LATENCY": 1},
        build_args=build_args(sim),
        build_dir=build_dir,
        clean=True,
        timescale=("1ns", "1ps"),
    )
    runner.test(
        hdl_toplevel="holoso_fsort",
        test_module="tests.hdl.test_fsort",
        test_dir=REPO_ROOT,
        build_dir=build_dir,
        extra_env={"HOLOSO_EXPECTED_LATENCY": "1"},
        results_xml=str(build_dir / "results.xml"),
    )
