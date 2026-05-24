"""Tests for holoso_fsgnop (combinational, format-independent)."""

from __future__ import annotations

import os
from pathlib import Path

import cocotb
import numpy as np
import pytest
from cocotb.triggers import Timer
from cocotb_tools.runner import get_runner

from hdl_float_oracle import (
    DIRECTED_F32,
    REPO_ROOT,
    SGNOP_OPS,
    SIMULATORS,
    BENCH_DIR,
    apply_sgnop,
    build_args,
    get_random_count,
    get_seed,
    sources,
)

WFULL_VALUES = (6, 12, 24, 32)


@cocotb.test()
async def holoso_fsgnop_cocotb(dut) -> None:
    wfull = int(os.environ["HOLOSO_TEST_WFULL"])

    async def check(x: int, op: int) -> None:
        dut.x.value = x
        dut.op.value = op
        await Timer(1, unit="ns")
        actual = int(dut.y.value)
        expected = apply_sgnop(x, op, wfull)
        assert actual == expected, f"WFULL={wfull} x=0x{x:x} op={op}: got 0x{actual:x}, want 0x{expected:x}"

    if wfull <= 8:
        # Exhaustive over all input bit patterns + all opcodes.
        for x in range(1 << wfull):
            for op in SGNOP_OPS:
                await check(x, op)
        return

    # Directed battery (using the float32 corner-case set, masked to wfull).
    body_mask = (1 << wfull) - 1
    for raw in DIRECTED_F32:
        x = raw & body_mask
        # If wfull != 32, also exercise patterns with the wfull-1 sign bit explicitly set/clear.
        candidates = {x, x ^ (1 << (wfull - 1))}
        for x_eff in candidates:
            for op in SGNOP_OPS:
                await check(x_eff, op)

    # Random sweep: HOLOSO_TEST_RANDOM_COUNT cases, all opcodes per draw.
    rng = np.random.default_rng(get_seed())
    for _ in range(get_random_count()):
        x = int(rng.integers(0, 1 << wfull, dtype=np.uint64))
        for op in SGNOP_OPS:
            await check(x, op)


@pytest.mark.parametrize("wfull", WFULL_VALUES)
@pytest.mark.parametrize("sim", SIMULATORS)
def test_holoso_fsgnop(sim: str, wfull: int) -> None:
    runner = get_runner(sim)
    build_dir = REPO_ROOT / "build" / "cocotb" / sim / f"fsgnop_w{wfull}"
    runner.build(
        sources=sources(),
        includes=[REPO_ROOT / "hdl"],
        hdl_toplevel="holoso_fsgnop",
        parameters={"WFULL": wfull},
        build_args=build_args(sim),
        build_dir=build_dir,
        clean=True,
        timescale=("1ns", "1ps"),
    )
    runner.test(
        hdl_toplevel="holoso_fsgnop",
        test_module="test_fsgnop",
        test_dir=BENCH_DIR,
        build_dir=build_dir,
        extra_env={"HOLOSO_TEST_WFULL": str(wfull)},
        results_xml=str(build_dir / "results.xml"),
    )
