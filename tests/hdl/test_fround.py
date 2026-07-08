"""
Tests for holoso_fround (pipelined; sgnop on a and y, per-firing round_mode immediate;
y = sgnop(round(sgnop(a), round_mode))).

round_mode (0 nearest-even, 1 floor, 2 ceil, 3 trunc) and both sgnop controls are allowed to vary every input cycle;
the wrapper delays y_sgnop through the same number of stages as zkf_round.
"""

import os
from typing import Any

import cocotb
import numpy as np
import pytest
from cocotb.triggers import RisingEdge, Timer
from cocotb_tools.runner import get_runner

from holoso import FloatFormat, FRoundOperator

from .hdl_float_oracle import (
    DIRECTED_F32,
    HDL_DIR,
    PipelineScoreboard,
    REPO_ROOT,
    ROUND_MODES,
    SGNOP_OPS,
    SIMULATORS,
    apply_sgnop,
    build_args,
    drive_reset,
    f32_to_bits,
    get_random_count,
    get_seed,
    random_zkf_f32,
    round_oracle_bits,
    sources,
    start_clock,
)

# (stage_input, stage_decode, stage_pack, stage_output): base (output reg only, latency 1) + each knob alone + all-on.
# At least one stage must be enabled -- the zkf_round core is combinational, so a pooled rounder needs a register.
STAGE_COMBOS = ((0, 0, 0, 1), (1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1), (1, 1, 1, 1))

# Directed corner cases beyond DIRECTED_F32: ties (x.5) exercise nearest-even both parities, and |x| < 1 exercises the
# rounder's sub-one branch (floor(-0.3) = -1, ceil(-0.3) = +0) which the integer-boundary path never reaches.
DIRECTED_ROUND: tuple[int, ...] = DIRECTED_F32 + tuple(
    f32_to_bits(np.float32(v))
    for v in (0.3, -0.3, 0.7, -0.7, 1.5, -1.5, 2.5, -2.5, 3.5, -3.5, 0.49999997, -0.49999997, 8388607.5, -8388607.5)
)


@cocotb.test()
async def holoso_fround_cocotb(dut: Any) -> None:
    await start_clock(dut)
    await drive_reset(dut)

    sb = PipelineScoreboard(dut, [("y", "y")], latency=int(os.environ["HOLOSO_EXPECTED_LATENCY"]))
    rng = np.random.default_rng(get_seed())

    async def step_idle() -> None:
        dut.in_valid.value = 0
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        sb.sample()

    async def step(a: int, a_op: int, mode: int, y_op: int) -> None:
        a_eff = apply_sgnop(a, a_op)
        y_pre = round_oracle_bits(a_eff, mode)
        if y_pre is None:
            await step_idle()
            return
        expected = apply_sgnop(y_pre, y_op)
        dut.a.value = a
        dut.a_sgnop.value = a_op
        dut.round_mode.value = mode
        dut.y_sgnop.value = y_op
        dut.in_valid.value = 1
        sb.push({"y": expected, "_desc": f"a=0x{a:08x} mode={mode} ops={a_op}{y_op}"})
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        sb.sample()

    # Every directed value through every mode, identity sgnops.
    for a in DIRECTED_ROUND:
        for mode in ROUND_MODES:
            await step(a, 0, mode, 0)
    await sb.drain()

    # A sample of directed values through every (a_sgnop, mode, y_sgnop) combination.
    sample = [DIRECTED_ROUND[int(rng.integers(0, len(DIRECTED_ROUND)))] for _ in range(8)]
    for a_op in SGNOP_OPS:
        for mode in ROUND_MODES:
            for a in sample:
                for y_op in SGNOP_OPS:
                    await step(a, a_op, mode, y_op)
    await sb.drain()

    # Random ZKF-legal inputs with random mode and sgnops, interleaved with idle bubbles.
    for _ in range(get_random_count()):
        if rng.random() < 0.2:
            await step_idle()
            continue
        a = random_zkf_f32(rng)
        a_op = int(rng.integers(0, 4))
        mode = int(rng.integers(0, 4))
        y_op = int(rng.integers(0, 4))
        await step(a, a_op, mode, y_op)
    await sb.drain()

    await drive_reset(dut)
    for _ in range(8):
        dut.in_valid.value = 0
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        assert int(dut.out_valid.value) == 0


@pytest.mark.parametrize("stages", STAGE_COMBOS, ids=lambda s: f"i{s[0]}d{s[1]}p{s[2]}o{s[3]}")
@pytest.mark.parametrize("sim", SIMULATORS)
def test_holoso_fround(sim: str, stages: tuple[int, int, int, int]) -> None:
    stage_input, stage_decode, stage_pack, stage_output = stages
    operator = FRoundOperator(
        FloatFormat(8, 24),
        stage_input=stage_input,
        stage_decode=stage_decode,
        stage_pack=stage_pack,
        stage_output=stage_output,
    )
    runner = get_runner(sim)
    build_dir = (
        REPO_ROOT / "build" / "cocotb" / sim / f"fround_i{stage_input}d{stage_decode}p{stage_pack}o{stage_output}"
    )
    runner.build(
        sources=sources(),
        includes=[HDL_DIR],
        hdl_toplevel="holoso_fround",
        parameters=operator.params,
        build_args=build_args(sim),
        build_dir=build_dir,
        clean=True,
        timescale=("1ns", "1ps"),
    )
    runner.test(
        hdl_toplevel="holoso_fround",
        test_module="tests.hdl.test_fround",
        test_dir=REPO_ROOT,
        build_dir=build_dir,
        extra_env={"HOLOSO_EXPECTED_LATENCY": str(operator.latency)},
        results_xml=str(build_dir / "results.xml"),
    )
