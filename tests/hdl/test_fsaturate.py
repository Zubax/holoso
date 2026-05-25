"""Tests for holoso_fsaturate (combinational; +inf -> +max, -inf -> -max, finite passes through)."""

import cocotb
import numpy as np
import pytest
from cocotb.triggers import Timer
from cocotb_tools.runner import get_runner

from hdl_float_oracle import (
    DIRECTED_F32,
    F32_MAX_FIN,
    F32_SIGN_MASK,
    HDL_DIR,
    REPO_ROOT,
    SIMULATORS,
    BENCH_DIR,
    build_args,
    get_random_count,
    get_seed,
    is_inf_f32,
    random_zkf_f32,
    sources,
)


@cocotb.test()
async def holoso_fsaturate_cocotb(dut) -> None:
    async def check(x_bits: int) -> None:
        dut.x.value = x_bits
        await Timer(1, unit="ns")
        if is_inf_f32(x_bits):
            expected = F32_MAX_FIN | (x_bits & F32_SIGN_MASK)
        else:
            expected = x_bits
        actual = int(dut.y.value)
        assert actual == expected, f"x=0x{x_bits:08x}: got 0x{actual:08x}, want 0x{expected:08x}"

    for x in DIRECTED_F32:
        await check(x)

    rng = np.random.default_rng(get_seed())
    for _ in range(get_random_count()):
        await check(random_zkf_f32(rng))


@pytest.mark.parametrize("sim", SIMULATORS)
def test_holoso_fsaturate(sim: str) -> None:
    runner = get_runner(sim)
    build_dir = REPO_ROOT / "build" / "cocotb" / sim / "fsaturate"
    runner.build(
        sources=sources(),
        includes=[HDL_DIR],
        hdl_toplevel="holoso_fsaturate",
        parameters={"WEXP": 8, "WMAN": 24},
        build_args=build_args(sim),
        build_dir=build_dir,
        clean=True,
        timescale=("1ns", "1ps"),
    )
    runner.test(
        hdl_toplevel="holoso_fsaturate",
        test_module="test_fsaturate",
        test_dir=BENCH_DIR,
        build_dir=build_dir,
        results_xml=str(build_dir / "results.xml"),
    )
