"""Tests for holoso_fdiv (pipelined; y = sgnop(sgnop(a)/sgnop(b)); div0 alongside out_valid).

div0 is asserted when the post-sgnop divisor has exp=0 (i.e., the divisor is zero in either sign form). When div0=1
the y output is unspecified by the wrapper contract, so the test skips the y comparison for those cases but still
checks that div0 itself is correct. The wrapper delays y_sgnop through the same number of stages as zkf_div, and the
scoreboard verifies the documented latency against actual out_valid timing.
"""

from __future__ import annotations

import os

import cocotb
import numpy as np
import pytest
from cocotb.triggers import RisingEdge, Timer
from cocotb_tools.runner import get_runner

from hdl_float_oracle import (
    DIRECTED_F32,
    F32_EXP_MASK,
    PipelineScoreboard,
    REPO_ROOT,
    SGNOP_OPS,
    SIMULATORS,
    BENCH_DIR,
    apply_sgnop,
    build_args,
    div_oracle_bits,
    drive_reset,
    f32_to_bits,
    get_random_count,
    get_seed,
    random_zkf_f32,
    sources,
    start_clock,
)

STAGE_INPUT_VALUES = (0, 1)


def _exp_is_zero(bits: int) -> bool:
    return (bits & F32_EXP_MASK) == 0


@cocotb.test()
async def holoso_fdiv_cocotb(dut) -> None:
    await start_clock(dut)
    await drive_reset(dut)

    sb = PipelineScoreboard(dut, [("y", "y"), ("div0", "div0")], latency=int(os.environ["HOLOSO_EXPECTED_LATENCY"]))
    rng = np.random.default_rng(get_seed())

    async def step_idle() -> None:
        dut.in_valid.value = 0
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        sb.sample()

    async def step(a: int, b: int, a_op: int, b_op: int, y_op: int) -> None:
        a_eff = apply_sgnop(a, a_op)
        b_eff = apply_sgnop(b, b_op)
        b_eff_is_zero = _exp_is_zero(b_eff)

        expected: dict = {"_desc": f"a=0x{a:08x} b=0x{b:08x} ops={a_op}{b_op}{y_op}"}
        if b_eff_is_zero:
            # div0 asserts; y is unspecified -- only check div0.
            expected["div0"] = 1
        else:
            # Filter NaN-producing pairs (oracle returns None for inf/inf, etc.).
            y_pre = div_oracle_bits(a_eff, b_eff)
            if y_pre is None:
                await step_idle()
                return
            expected["y"] = apply_sgnop(y_pre, y_op)
            expected["div0"] = 0

        dut.a.value = a
        dut.b.value = b
        dut.a_sgnop.value = a_op
        dut.b_sgnop.value = b_op
        dut.y_sgnop.value = y_op
        dut.in_valid.value = 1
        sb.push(expected)
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        sb.sample()

    # Directed x directed, neutral sgnops.
    for a in DIRECTED_F32:
        for b in DIRECTED_F32:
            await step(a, b, 0, 0, 0)
    await sb.drain()

    # Sgnop sweep. Include zero-divisor cases in each sample to exercise div0 while output sign control changes.
    sample_pairs = [
        (DIRECTED_F32[int(rng.integers(0, len(DIRECTED_F32)))], DIRECTED_F32[int(rng.integers(0, len(DIRECTED_F32)))])
        for _ in range(6)
    ]
    sample_pairs.append((f32_to_bits(1.0), 0))  # div by +0
    sample_pairs.append((f32_to_bits(-1.0), 0))  # also div by +0

    for a_op in SGNOP_OPS:
        for b_op in SGNOP_OPS:
            for a, b in sample_pairs:
                for y_op in SGNOP_OPS:
                    await step(a, b, a_op, b_op, y_op)
    await sb.drain()

    # Random bulk.
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


@pytest.mark.parametrize("stage_input", STAGE_INPUT_VALUES)
@pytest.mark.parametrize("sim", SIMULATORS)
def test_holoso_fdiv(sim: str, stage_input: int) -> None:
    runner = get_runner(sim)
    build_dir = REPO_ROOT / "build" / "cocotb" / sim / f"fdiv_si{stage_input}"
    latency = 4 + ((24 + 2 + ((24 + 2) % 2)) // 2) + int(bool(stage_input))
    runner.build(
        sources=sources(),
        includes=[REPO_ROOT / "hdl"],
        hdl_toplevel="holoso_fdiv",
        parameters={"WEXP": 8, "WMAN": 24, "STAGE_INPUT": stage_input},
        build_args=build_args(sim),
        build_dir=build_dir,
        clean=True,
        timescale=("1ns", "1ps"),
    )
    runner.test(
        hdl_toplevel="holoso_fdiv",
        test_module="test_fdiv",
        test_dir=BENCH_DIR,
        build_dir=build_dir,
        extra_env={"HOLOSO_EXPECTED_LATENCY": str(latency)},
        results_xml=str(build_dir / "results.xml"),
    )
