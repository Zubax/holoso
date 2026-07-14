import os
from pathlib import Path
from typing import Any

import cocotb
import numpy as np
import pytest
from cocotb.triggers import Timer
from cocotb_tools.runner import get_runner

from .hdl_float_oracle import HDL_DIR, REPO_ROOT, SIMULATORS, build_args, sources
from .hdl_integer_oracle import ashift


@cocotb.test()
async def holoso_ashiftc_cocotb(dut: Any) -> None:
    width = int(os.environ["HOLOSO_INTEGER_WIDTH"])
    mask = (1 << width) - 1

    async def check(a: int, b: int) -> None:
        dut.x.value = a
        dut.shamt.value = b
        await Timer(1, unit="ns")
        actual = int(dut.y.value) & mask
        expected = ashift(a, b, width)
        assert actual == expected, f"W={width} a=0x{a:x} b=0x{b:x}: got 0x{actual:x}, want 0x{expected:x}"

    if width <= 6:
        for a in range(1 << width):
            for b in range(1 << width):
                await check(a, b)
    else:
        minimum = 1 << (width - 1)
        values = (0, 1, 2, minimum - 1, minimum, minimum + 1, mask - 1, mask)
        shifts = tuple(value & mask for value in (0, 1, -1, width - 1, 1 - width, width, -width, width + 1, -width - 1))
        for a in values:
            for b in shifts:
                await check(a, b)
        rng = np.random.default_rng(12345)
        for _ in range(1000):
            await check(
                int(rng.integers(0, 1 << width, dtype=np.uint64)),
                int(rng.integers(0, 1 << width, dtype=np.uint64)),
            )


@pytest.mark.parametrize("width", (2, 3, 4, 5, 6, 24, 44), ids=lambda width: f"w{width}")
@pytest.mark.parametrize("sim", SIMULATORS)
def test_holoso_ashiftc(sim: str, width: int) -> None:
    runner = get_runner(sim)
    build_dir = REPO_ROOT / "build" / "cocotb" / sim / f"ashiftc_w{width}"
    runner.build(
        sources=[*sources(), Path(__file__).resolve().parent / "holoso_support_fn_tb.v"],
        includes=[HDL_DIR],
        hdl_toplevel="holoso_ashiftc_tb",
        parameters={"W": width},
        build_args=build_args(sim),
        build_dir=build_dir,
        clean=True,
        timescale=("1ns", "1ps"),
    )
    runner.test(
        hdl_toplevel="holoso_ashiftc_tb",
        test_module="tests.hdl.test_ashiftc",
        test_dir=REPO_ROOT,
        build_dir=build_dir,
        extra_env={"HOLOSO_INTEGER_WIDTH": str(width)},
        results_xml=str(build_dir / "results.xml"),
    )
