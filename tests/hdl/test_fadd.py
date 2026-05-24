"""Tests for holoso_fadd (pipelined; sgnop on a, b, y; y = sgnop(sgnop(a)+sgnop(b))).

The wrapper delays y_sgnop through the same number of stages as zkf_add, so all sgnop controls are allowed to vary
every input cycle. The scoreboard verifies the documented wrapper latency against actual out_valid timing.
"""

from __future__ import annotations

import os

import cocotb
import numpy as np
import pytest
from cocotb.triggers import RisingEdge, Timer
from cocotb_tools.runner import get_runner

from holoso.format import FloatFormat
from holoso.operators import OpKind, StageConfig, latency_of

from hdl_float_oracle import (
    DIRECTED_F32,
    PipelineScoreboard,
    HDL_DIR,
    REPO_ROOT,
    SGNOP_OPS,
    SIMULATORS,
    BENCH_DIR,
    add_oracle_bits,
    apply_sgnop,
    build_args,
    drive_reset,
    get_random_count,
    get_seed,
    random_zkf_f32,
    sources,
    start_clock,
)


@cocotb.test()
async def holoso_fadd_cocotb(dut) -> None:
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
        y_pre = add_oracle_bits(a_eff, b_eff)
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

    # Phase A: directed x directed, all-zero sgnops -- streams without drain.
    for a in DIRECTED_F32:
        for b in DIRECTED_F32:
            await step(a, b, 0, 0, 0)
    await sb.drain()

    # Phase B: sgnop sweep. Output sign control changes every cycle to verify sideband pipelining.
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

    # Phase C: random bulk. Inject occasional gaps to exercise the valid pipeline.
    for _ in range(get_random_count()):
        a = random_zkf_f32(rng)
        b = random_zkf_f32(rng)
        a_op = int(rng.integers(0, 4))
        b_op = int(rng.integers(0, 4))
        y_op = int(rng.integers(0, 4))
        if rng.random() < 0.2:
            await step_idle()
            continue
        await step(a, b, a_op, b_op, y_op)
    await sb.drain()

    # Phase D: reset behavior. After reset, out_valid must stay deasserted while idle.
    await drive_reset(dut)
    for _ in range(8):
        dut.in_valid.value = 0
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        assert int(dut.out_valid.value) == 0, "out_valid asserted while idle after reset"


STAGE_COMBOS = ((0, 0), (1, 0), (0, 1), (1, 1))


@pytest.mark.parametrize("stages", STAGE_COMBOS, ids=lambda s: f"d{s[0]}a{s[1]}")
@pytest.mark.parametrize("sim", SIMULATORS)
def test_holoso_fadd(sim: str, stages: tuple[int, int]) -> None:
    stage_decode, stage_align = stages
    runner = get_runner(sim)
    build_dir = REPO_ROOT / "build" / "cocotb" / sim / f"fadd_d{stage_decode}a{stage_align}"
    runner.build(
        sources=sources(),
        includes=[HDL_DIR],
        hdl_toplevel="holoso_fadd",
        parameters={"WEXP": 8, "WMAN": 24, "STAGE_DECODE": stage_decode, "STAGE_ALIGN": stage_align},
        build_args=build_args(sim),
        build_dir=build_dir,
        clean=True,
        timescale=("1ns", "1ps"),
    )
    runner.test(
        hdl_toplevel="holoso_fadd",
        test_module="test_fadd",
        test_dir=BENCH_DIR,
        build_dir=build_dir,
        extra_env={
            "HOLOSO_EXPECTED_LATENCY": str(
                latency_of(
                    OpKind.FADD, FloatFormat(8, 24), StageConfig(fadd_decode=stage_decode, fadd_align=stage_align)
                )
            )
        },
        results_xml=str(build_dir / "results.xml"),
    )
