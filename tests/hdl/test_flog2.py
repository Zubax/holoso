"""
Tests for holoso_flog2 (pipelined; y = sgnop(log2(sgnop(a))); domain_error and pole alongside out_valid).

Unlike fdiv (whose y is unspecified on div0), zkf_log2 defines y = -inf for both error cases, so this bench checks y in
every case and additionally checks the two flags: pole when the conditioned operand is +0, domain_error when it is
negative. The value oracle is the exact ZKF model (FloatValue.log2); the flags are an independent bit classification.
"""

import os
from typing import Any

import cocotb
import numpy as np
import pytest
from cocotb.triggers import RisingEdge, Timer
from cocotb_tools.runner import get_runner

from holoso import FLog2Operator, FloatFormat

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
    log2_oracle,
    random_zkf_f32,
    sources,
    stage_tag,
    start_clock,
)

STAGE_COMBOS: tuple[dict[str, int], ...] = (
    {},
    {"stage_input": 1, "stage_output": 1},
    {"stage_decode": 1, "stage_normalize": 1, "stage_pack": 1},
    {"stage_product": 2, "stage_product_final": 1, "stage_normalize_output": 1},
)


@cocotb.test()
async def holoso_flog2_cocotb(dut: Any) -> None:
    await start_clock(dut)
    await drive_reset(dut)

    sb = PipelineScoreboard(
        dut,
        [("y", "y"), ("domain_error", "domain_error"), ("pole", "pole")],
        latency=int(os.environ["HOLOSO_EXPECTED_LATENCY"]),
    )
    rng = np.random.default_rng(get_seed())

    async def step_idle() -> None:
        dut.in_valid.value = 0
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        sb.sample()

    async def step(a: int, a_op: int, y_op: int) -> None:
        a_eff = apply_sgnop(a, a_op)
        y_pre, domain_error, pole = log2_oracle(a_eff)
        expected = {
            "_desc": f"a=0x{a:08x} ops={a_op}{y_op}",
            "y": apply_sgnop(y_pre, y_op),
            "domain_error": domain_error,
            "pole": pole,
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

    # Sgnop sweep, seeded with a zero and a negative operand so the pole/domain flags fire under output sign changes.
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
def test_holoso_flog2(sim: str, stages: dict[str, int]) -> None:
    operator = FLog2Operator(FloatFormat(8, 24), **stages)
    runner = get_runner(sim)
    build_dir = REPO_ROOT / "build" / "cocotb" / sim / f"flog2_{stage_tag(stages)}"
    runner.build(
        sources=sources(),
        includes=[HDL_DIR],
        hdl_toplevel="holoso_flog2",
        parameters={"WEXP": 8, "WMAN": 24, **operator.hdl_params(), "LATENCY": operator.latency},
        build_args=build_args(sim),
        build_dir=build_dir,
        clean=True,
        timescale=("1ns", "1ps"),
    )
    runner.test(
        hdl_toplevel="holoso_flog2",
        test_module="tests.hdl.test_flog2",
        test_dir=REPO_ROOT,
        build_dir=build_dir,
        extra_env={"HOLOSO_EXPECTED_LATENCY": str(operator.latency)},
        results_xml=str(build_dir / "results.xml"),
    )
