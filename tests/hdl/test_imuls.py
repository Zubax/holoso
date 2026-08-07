import math
import os
from typing import Any

import cocotb
import numpy as np
import pytest
from cocotb.triggers import RisingEdge, Timer
from cocotb_tools.runner import get_runner

from holoso import IMulOptions, IntFormat
from holoso._operators import IMulOperator

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
from .hdl_integer_oracle import expected_imuls, signed


def _directed(width: int) -> list[int]:
    minimum = -(1 << (width - 1))
    maximum = (1 << (width - 1)) - 1
    root = math.isqrt(maximum)
    values = {
        minimum,
        minimum + 1,
        -(root + 1),
        -root,
        -2,
        -1,
        0,
        1,
        2,
        root,
        root + 1,
        maximum - 1,
        maximum,
    }
    return sorted(value & ((1 << width) - 1) for value in values)


@cocotb.test()
async def imuls_cocotb(dut: Any) -> None:
    width = int(os.environ["HOLOSO_IMULS_WIDTH"])
    stage_product = int(os.environ["HOLOSO_IMULS_STAGE_PRODUCT"])
    latency = int(os.environ["HOLOSO_EXPECTED_LATENCY"])
    mask = (1 << width) - 1
    scoreboard = PipelineScoreboard(dut, (("y", "y"), ("saturated", "saturated")), latency=latency)
    await start_clock(dut)
    await drive_reset(dut)

    async def step(a: int, b: int, valid: bool = True) -> None:
        dut.a.value = a & mask
        dut.b.value = b & mask
        dut.in_valid.value = valid
        if valid:
            desc = f"W={width} stage={stage_product} a={signed(a & mask, width)} b={signed(b & mask, width)}"
            scoreboard.push({**expected_imuls(a & mask, b & mask, width), "_desc": desc})
        await RisingEdge(dut.clk)
        dut.a.value = (~a) & mask
        dut.b.value = (~b) & mask
        await Timer(1, unit="ns")
        scoreboard.sample()

    if width <= 6:
        for a in range(1 << width):
            for b in range(1 << width):
                await step(a, b)
    else:
        directed = _directed(width)
        for a in directed:
            for b in directed:
                await step(a, b)
        rng = np.random.default_rng(int(os.environ.get("HOLOSO_TEST_SEED", "12345")))
        for _ in range(int(os.environ.get("HOLOSO_IMULS_RANDOM", "512"))):
            a = int(rng.integers(0, 1 << width, dtype=np.uint64))
            b = int(rng.integers(0, 1 << width, dtype=np.uint64))
            await step(a, b, bool(rng.random() >= 0.2))

    await scoreboard.drain()

    for value in range(latency):
        dut.a.value = value & mask
        dut.b.value = (value + 1) & mask
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


_CONFIGURATIONS = tuple((width, stage) for width in (*range(2, 7), 24, 44) for stage in range(5))


@pytest.mark.parametrize(
    ("width", "stage_product"),
    _CONFIGURATIONS,
    ids=lambda value: str(value),
)
@pytest.mark.parametrize("sim", SIMULATORS)
def test_imuls(sim: str, width: int, stage_product: int) -> None:
    # The Python operator model supplies the RTL parameters and the expected latency, so a drifted closed form fails.
    hardware = IMulOperator(IntFormat(width), IMulOptions(stage_product=stage_product))
    runner = get_runner(sim)
    tag = f"holoso_imuls_w{width}_s{stage_product}"
    build_dir = REPO_ROOT / "build" / "cocotb" / sim / tag
    runner.build(
        sources=sources(),
        includes=[HDL_DIR],
        hdl_toplevel=hardware.module_name,
        parameters=hardware.params,
        build_args=build_args(sim),
        build_dir=build_dir,
        clean=True,
        timescale=("1ns", "1ps"),
    )
    runner.test(
        hdl_toplevel=hardware.module_name,
        test_module="tests.hdl.test_imuls",
        test_dir=REPO_ROOT,
        build_dir=build_dir,
        extra_env={
            "HOLOSO_IMULS_WIDTH": str(width),
            "HOLOSO_IMULS_STAGE_PRODUCT": str(stage_product),
            "HOLOSO_EXPECTED_LATENCY": str(hardware.latency),
        },
        results_xml=str(build_dir / "results.xml"),
    )
