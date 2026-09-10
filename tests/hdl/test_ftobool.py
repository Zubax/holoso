"""
Tests for holoso_ftobool (combinational; y=1 iff the exponent field is nonzero).

The sign sweep is the point: `FloatToBoolOperator` declares its operand sign-invariant, and the compiler ERASES
sign conditioning on the strength of that. This is the RTL-side evidence for the claim.
"""

from pathlib import Path
from typing import Any

import cocotb
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
    sources,
)


@cocotb.test()
async def holoso_ftobool_cocotb(dut: Any) -> None:
    async def answer(x_bits: int) -> int:
        dut.x.value = x_bits
        await Timer(1, unit="ns")
        return int(dut.y.value)

    for x in DIRECTED_F32:
        expected = 0 if (x & F32_EXP_MASK) == 0 else 1
        actual = await answer(x)
        assert actual == expected, f"x=0x{x:08x}: got {actual}, want {expected}"
        # Flipping the sign alone must not move the answer, which is what the invariance declaration asserts.
        flipped = await answer(x ^ F32_SIGN_MASK)
        assert flipped == expected, f"x=0x{x:08x}: sign flip moved the answer to {flipped}"


@pytest.mark.parametrize("sim", SIMULATORS)
def test_holoso_ftobool(sim: str) -> None:
    runner = get_runner(sim)
    build_dir = REPO_ROOT / "build" / "cocotb" / sim / "ftobool"
    runner.build(
        sources=[*sources(), Path(__file__).resolve().parent / "holoso_support_fn_tb.v"],
        includes=[HDL_DIR],
        hdl_toplevel="holoso_ftobool_tb",
        parameters={"WEXP": 8, "WMAN": 24},
        build_args=build_args(sim),
        build_dir=build_dir,
        clean=True,
        timescale=("1ns", "1ps"),
    )
    runner.test(
        hdl_toplevel="holoso_ftobool_tb",
        test_module="tests.hdl.test_ftobool",
        test_dir=REPO_ROOT,
        build_dir=build_dir,
        results_xml=str(build_dir / "results.xml"),
    )
