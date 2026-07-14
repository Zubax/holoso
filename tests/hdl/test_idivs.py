import os
from typing import Any

import cocotb
import numpy as np
import pytest
from cocotb.triggers import RisingEdge, Timer
from cocotb_tools.runner import get_runner

from .hdl_float_oracle import (
    HDL_DIR,
    REPO_ROOT,
    SIMULATORS,
    PipelineScoreboard,
    build_args,
    drive_reset,
    sources,
    start_clock,
)
from .hdl_integer_oracle import signed


def _expected(num_bits: int, den_bits: int, width: int, quotient_floor: bool) -> dict[str, int | str]:
    mask = (1 << width) - 1
    minimum = -(1 << (width - 1))
    maximum = (1 << (width - 1)) - 1
    num = signed(num_bits, width)
    den = signed(den_bits, width)
    if den == 0:
        quotient = minimum if num < 0 else maximum
        remainder = num
        saturated = 1
        div0 = 1
    elif num == minimum and den == -1:
        quotient = maximum
        remainder = 0
        saturated = 1
        div0 = 0
    else:
        if quotient_floor:
            quotient = num // den
        else:
            quotient_magnitude = abs(num) // abs(den)
            quotient = -quotient_magnitude if (num < 0) != (den < 0) else quotient_magnitude
        remainder = num - den * quotient
        saturated = 0
        div0 = 0
        assert num == den * quotient + remainder
        assert abs(remainder) < abs(den)
        if remainder:
            assert (remainder < 0) == ((den if quotient_floor else num) < 0)
    return {
        "quo": quotient & mask,
        "rem": remainder & mask,
        "saturated": saturated,
        "div0": div0,
    }


def _directed(width: int) -> list[int]:
    minimum = -(1 << (width - 1))
    maximum = (1 << (width - 1)) - 1
    return sorted(
        {
            value & ((1 << width) - 1)
            for value in (
                minimum,
                minimum + 1,
                -3,
                -2,
                -1,
                0,
                1,
                2,
                3,
                maximum - 1,
                maximum,
            )
        }
    )


@cocotb.test()
async def idivs_cocotb(dut: Any) -> None:
    width = int(os.environ["HOLOSO_IDIVS_WIDTH"])
    quotient_floor = bool(int(os.environ["HOLOSO_IDIVS_QUOTIENT_FLOOR"]))
    latency = 2 + (width + 1) // 2
    mask = (1 << width) - 1
    scoreboard = PipelineScoreboard(
        dut,
        (("quo", "quo"), ("rem", "rem"), ("saturated", "saturated"), ("div0", "div0")),
        latency=latency,
    )
    await start_clock(dut)
    await drive_reset(dut)

    async def step(num: int, den: int, valid: bool = True) -> None:
        num &= mask
        den &= mask
        dut.num.value = num
        dut.den.value = den
        dut.in_valid.value = valid
        if valid:
            expected = _expected(num, den, width, quotient_floor)
            expected["_desc"] = (
                f"W={width} floor={int(quotient_floor)} num={signed(num, width)} den={signed(den, width)}"
            )
            scoreboard.push(expected)
        await RisingEdge(dut.clk)
        dut.num.value = (~num) & mask
        dut.den.value = (~den) & mask
        await Timer(1, unit="ns")
        scoreboard.sample()

    if width <= 6:
        for num in range(1 << width):
            for den in range(1 << width):
                await step(num, den)
    else:
        directed = _directed(width)
        for num in directed:
            for den in directed:
                await step(num, den)
        rng = np.random.default_rng(int(os.environ.get("HOLOSO_TEST_SEED", "12345")))
        for _ in range(int(os.environ.get("HOLOSO_IDIVS_RANDOM", "1000"))):
            num = int(rng.integers(0, 1 << width, dtype=np.uint64))
            den = int(rng.integers(0, 1 << width, dtype=np.uint64))
            if rng.random() < 0.1:
                den = int(rng.choice(np.array(_directed(width), dtype=np.uint64)))
            await step(num, den, bool(rng.random() >= 0.2))

    await scoreboard.drain()
    for value in range(latency):
        dut.num.value = value & mask
        dut.den.value = (value + 1) & mask
        dut.in_valid.value = 1
        await RisingEdge(dut.clk)
    dut.rst.value = 1
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    dut.in_valid.value = 0
    for _ in range(latency + 2):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        assert not int(dut.out_valid.value)


@pytest.mark.parametrize("width", (*range(2, 7), 24, 44), ids=lambda width: f"w{width}")
@pytest.mark.parametrize("quotient_floor", (0, 1), ids=lambda mode: "trunc" if mode == 0 else "floor")
@pytest.mark.parametrize("sim", SIMULATORS)
def test_idivs(sim: str, width: int, quotient_floor: int) -> None:
    latency = 2 + (width + 1) // 2
    build_dir = REPO_ROOT / "build" / "cocotb" / sim / f"holoso_idivs_w{width}_f{quotient_floor}"
    runner = get_runner(sim)
    runner.build(
        sources=sources(),
        includes=[HDL_DIR],
        hdl_toplevel="holoso_idivs",
        parameters={"W": width, "QUOTIENT_FLOOR": quotient_floor, "LATENCY": latency},
        build_args=build_args(sim),
        build_dir=build_dir,
        clean=True,
        timescale=("1ns", "1ps"),
    )
    runner.test(
        hdl_toplevel="holoso_idivs",
        test_module="tests.hdl.test_idivs",
        test_dir=REPO_ROOT,
        build_dir=build_dir,
        extra_env={
            "HOLOSO_IDIVS_WIDTH": str(width),
            "HOLOSO_IDIVS_QUOTIENT_FLOOR": str(quotient_floor),
        },
        results_xml=str(build_dir / "results.xml"),
    )
