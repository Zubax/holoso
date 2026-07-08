"""
Tests for holoso_fsincos. The core holds one transaction in flight (II = LATENCY+1), so the bench drives one
transaction at a time and waits for out_valid; SIMULATION=1 arms the wrapper's over-issue $fatal.
"""

import os
from typing import Any

import cocotb
import numpy as np
import pytest
from cocotb.triggers import RisingEdge, Timer
from cocotb_tools.runner import get_runner

from holoso import FloatFormat, FSincosOperator

from .hdl_float_oracle import (
    DIRECTED_F32,
    HDL_DIR,
    REPO_ROOT,
    SGNOP_OPS,
    SIMULATORS,
    apply_sgnop,
    build_args,
    drive_reset,
    get_random_count,
    get_seed,
    random_zkf_f32,
    sincos_oracle,
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
    {"stage_pack": 1, "stage_product": 2, "stage_normalize": 1},  # the from_polar timing-closed config
)


async def _sincos(dut: Any, latency: int, a: int, a_op: int, sin_op: int, cos_op: int) -> tuple[int, int]:
    dut.a.value = a
    dut.a_sgnop.value = a_op
    dut.sin_sgnop.value = sin_op
    dut.cos_sgnop.value = cos_op
    dut.in_valid.value = 1
    await RisingEdge(dut.clk)
    dut.in_valid.value = 0
    for cycle in range(latency + 16):
        await Timer(1, unit="ns")
        if int(dut.out_valid.value) == 1:
            assert (
                cycle == latency - 1
            ), f"out_valid at cycle {cycle}, expected {latency - 1} (fixed accept-to-out_valid)"
            return int(dut.sin.value), int(dut.cos.value)
        await RisingEdge(dut.clk)
    raise AssertionError("out_valid never asserted")


@cocotb.test()
async def holoso_fsincos_cocotb(dut: Any) -> None:
    latency = int(os.environ["HOLOSO_EXPECTED_LATENCY"])
    await start_clock(dut)
    await drive_reset(dut)
    rng = np.random.default_rng(get_seed())

    async def check(a: int, a_op: int, sin_op: int, cos_op: int) -> None:
        sin_pre, cos_pre = sincos_oracle(apply_sgnop(a, a_op))
        exp_sin, exp_cos = apply_sgnop(sin_pre, sin_op), apply_sgnop(cos_pre, cos_op)
        got_sin, got_cos = await _sincos(dut, latency, a, a_op, sin_op, cos_op)
        assert got_sin == exp_sin, f"sin a=0x{a:08x} ops={a_op}{sin_op}: got 0x{got_sin:08x} exp 0x{exp_sin:08x}"
        assert got_cos == exp_cos, f"cos a=0x{a:08x} ops={a_op}{cos_op}: got 0x{got_cos:08x} exp 0x{exp_cos:08x}"
        await RisingEdge(dut.clk)  # let in_ready reassert before the next transaction

    for a in DIRECTED_F32:
        await check(a, 0, 0, 0)
    sample = [DIRECTED_F32[int(rng.integers(0, len(DIRECTED_F32)))] for _ in range(4)]
    for a_op in SGNOP_OPS:
        for sin_op in SGNOP_OPS:
            for a in sample:
                await check(a, a_op, sin_op, 0)
    for _ in range(get_random_count()):
        await check(random_zkf_f32(rng), *(int(rng.integers(0, 4)) for _ in range(3)))

    await drive_reset(dut)
    for _ in range(8):
        dut.in_valid.value = 0
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        assert int(dut.out_valid.value) == 0


@pytest.mark.parametrize("stages", STAGE_COMBOS, ids=stage_tag)
@pytest.mark.parametrize("sim", SIMULATORS)
def test_holoso_fsincos(sim: str, stages: dict[str, int]) -> None:
    operator = FSincosOperator(FloatFormat(8, 24), **stages)
    runner = get_runner(sim)
    build_dir = REPO_ROOT / "build" / "cocotb" / sim / f"fsincos_{stage_tag(stages)}"
    runner.build(
        sources=sources(),
        includes=[HDL_DIR],
        hdl_toplevel="holoso_fsincos",
        parameters=operator.params,
        build_args=build_args(sim),
        defines={"SIMULATION": 1},
        build_dir=build_dir,
        clean=True,
        timescale=("1ns", "1ps"),
    )
    runner.test(
        hdl_toplevel="holoso_fsincos",
        test_module="tests.hdl.test_fsincos",
        test_dir=REPO_ROOT,
        build_dir=build_dir,
        extra_env={"HOLOSO_EXPECTED_LATENCY": str(operator.latency)},
        results_xml=str(build_dir / "results.xml"),
    )
