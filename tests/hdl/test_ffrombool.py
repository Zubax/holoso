"""Tests for holoso_ffrombool (combinational; false -> +0, true -> +1.0)."""

import cocotb
import pytest
from cocotb.triggers import Timer
from cocotb_tools.runner import get_runner

from hdl_float_oracle import (
    HDL_DIR,
    REPO_ROOT,
    SIMULATORS,
    BENCH_DIR,
    build_args,
    f32_to_bits,
    sources,
)


@cocotb.test()
async def holoso_ffrombool_cocotb(dut) -> None:
    for x, expected in ((0, 0), (1, f32_to_bits(1.0))):
        dut.x.value = x
        await Timer(1, unit="ns")
        actual = int(dut.y.value)
        assert actual == expected, f"x={x}: got 0x{actual:08x}, want 0x{expected:08x}"


@pytest.mark.parametrize("sim", SIMULATORS)
def test_holoso_ffrombool(sim: str) -> None:
    runner = get_runner(sim)
    build_dir = REPO_ROOT / "build" / "cocotb" / sim / "ffrombool"
    runner.build(
        sources=sources(),
        includes=[HDL_DIR],
        hdl_toplevel="holoso_ffrombool",
        parameters={"WEXP": 8, "WMAN": 24},
        build_args=build_args(sim),
        build_dir=build_dir,
        clean=True,
        timescale=("1ns", "1ps"),
    )
    runner.test(
        hdl_toplevel="holoso_ffrombool",
        test_module="test_ffrombool",
        test_dir=BENCH_DIR,
        build_dir=build_dir,
        results_xml=str(build_dir / "results.xml"),
    )
