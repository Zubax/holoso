"""Tests for holoso_ftobool (combinational; exponent-zero values map to false)."""

from __future__ import annotations

import cocotb
import numpy as np
import pytest
from cocotb.triggers import Timer
from cocotb_tools.runner import get_runner

from hdl_float_oracle import (
    DIRECTED_F32,
    F32_EXP_MASK,
    REPO_ROOT,
    SIMULATORS,
    BENCH_DIR,
    build_args,
    get_random_count,
    get_seed,
    sources,
)

DIRECTED_TO_BOOL: tuple[int, ...] = tuple(
    dict.fromkeys(
        (
            *DIRECTED_F32,
            0x80000000,  # -0 is false by exponent test.
            0x00000001,  # Smallest positive subnormal.
            0x007FFFFF,  # Largest positive subnormal.
            0x80000001,  # Smallest negative subnormal.
            0x807FFFFF,  # Largest negative subnormal.
            0x7FC00001,  # NaN-like payload still has a nonzero exponent field.
            0xFFC00001,
        )
    )
)


@cocotb.test()
async def holoso_ftobool_cocotb(dut) -> None:
    async def check(x_bits: int) -> None:
        dut.x.value = x_bits
        await Timer(1, unit="ns")
        expected = 1 if (x_bits & F32_EXP_MASK) else 0
        actual = int(dut.y.value)
        assert actual == expected, f"x=0x{x_bits:08x}: got {actual}, want {expected}"

    for x in DIRECTED_TO_BOOL:
        await check(x)

    rng = np.random.default_rng(get_seed())
    for _ in range(get_random_count()):
        x = int(rng.integers(0, 1 << 32, dtype=np.uint64))
        await check(x)


@pytest.mark.parametrize("sim", SIMULATORS)
def test_holoso_ftobool(sim: str) -> None:
    runner = get_runner(sim)
    build_dir = REPO_ROOT / "build" / "cocotb" / sim / "ftobool"
    runner.build(
        sources=sources(),
        includes=[REPO_ROOT / "hdl"],
        hdl_toplevel="holoso_ftobool",
        parameters={"WEXP": 8, "WMAN": 24},
        build_args=build_args(sim),
        build_dir=build_dir,
        clean=True,
        timescale=("1ns", "1ps"),
    )
    runner.test(
        hdl_toplevel="holoso_ftobool",
        test_module="test_ftobool",
        test_dir=BENCH_DIR,
        build_dir=build_dir,
        results_xml=str(build_dir / "results.xml"),
    )
