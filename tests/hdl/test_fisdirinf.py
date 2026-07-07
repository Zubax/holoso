"""Tests for holoso_fisposinf and holoso_fisneginf."""

import os
from pathlib import Path
from typing import Any

import cocotb
import numpy as np
import pytest
from cocotb.triggers import Timer
from cocotb_tools.runner import get_runner

from .hdl_float_oracle import (
    DIRECTED_F32,
    F32_EXP_MASK,
    F32_SIGN_MASK,
    HDL_DIR,
    REPO_ROOT,
    SIMULATORS,
    build_args,
    get_random_count,
    get_seed,
    random_zkf_f32,
    sources,
)


@cocotb.test()
async def holoso_fisdirinf_cocotb(dut: Any) -> None:
    positive = bool(int(os.environ["HOLOSO_INF_POSITIVE"]))

    async def check(x_bits: int) -> None:
        dut.x.value = x_bits
        await Timer(1, unit="ns")
        nonfinite = (x_bits & F32_EXP_MASK) == F32_EXP_MASK
        sign = bool(x_bits & F32_SIGN_MASK)
        expected = int(nonfinite and sign != positive)
        actual = int(dut.y.value)
        assert actual == expected, f"x=0x{x_bits:08x}: got {actual}, want {expected}"

    for x in DIRECTED_F32:
        await check(x)

    for x in (0x7FC00001, 0xFFC00001):
        await check(x)

    rng = np.random.default_rng(get_seed())
    for _ in range(get_random_count()):
        await check(random_zkf_f32(rng))


@pytest.mark.parametrize("positive", [True, False], ids=["pos", "neg"])
@pytest.mark.parametrize("sim", SIMULATORS)
def test_holoso_fisdirinf(sim: str, positive: bool) -> None:
    toplevel = "holoso_fisposinf_tb" if positive else "holoso_fisneginf_tb"
    runner = get_runner(sim)
    build_dir = REPO_ROOT / "build" / "cocotb" / sim / toplevel
    runner.build(
        sources=[*sources(), Path(__file__).resolve().parent / "holoso_support_fn_tb.v"],
        includes=[HDL_DIR],
        hdl_toplevel=toplevel,
        parameters={"WEXP": 8, "WMAN": 24},
        build_args=build_args(sim),
        build_dir=build_dir,
        clean=True,
        timescale=("1ns", "1ps"),
    )
    runner.test(
        hdl_toplevel=toplevel,
        test_module="tests.hdl.test_fisdirinf",
        test_dir=REPO_ROOT,
        build_dir=build_dir,
        extra_env={"HOLOSO_INF_POSITIVE": str(int(positive))},
        results_xml=str(build_dir / "results.xml"),
    )
