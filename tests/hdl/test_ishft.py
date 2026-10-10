"""
Tests for holoso_ishft (pipelined; `right` selects shft = x << shamt or shft = x >> shamt, and a negative shamt
reverses either), with the direction chosen per transaction. The expectations come from the fixed-width oracle and never
from CPython's own shifts, which refuse a negative count and keep the bits a word drops.
"""

import os
from typing import Any

import cocotb
import numpy as np
import pytest
from cocotb.triggers import RisingEdge, Timer
from cocotb_tools.runner import get_runner

from holoso import IntFormat, IShftOptions
from holoso._operators import IShftOperator, ShiftMode

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
from .hdl_integer_oracle import EXHAUSTIVE_MAX_WIDTH, TEST_WIDTHS, ishl, ishr, signed


@cocotb.test()
async def ishft_cocotb(dut: Any) -> None:
    width = int(os.environ["HOLOSO_ISHFT_WIDTH"])
    x_port, shamt_port = os.environ["HOLOSO_ISHFT_OPERANDS"].split(",")
    (result_port,) = os.environ["HOLOSO_ISHFT_RESULTS"].split(",")
    mode_port = IShftOperator.mode_port
    assert mode_port is not None
    mask = (1 << width) - 1
    scoreboard = PipelineScoreboard(
        dut, [(result_port, result_port)], latency=int(os.environ["HOLOSO_EXPECTED_LATENCY"])
    )
    await start_clock(dut)
    await drive_reset(dut)

    async def step(mode: ShiftMode, x: int, shamt: int, valid: bool = True) -> None:
        # Driven through the operator's own port names and its own mode codes, so a misdeclared one miscomputes.
        getattr(dut, x_port).value = x
        getattr(dut, shamt_port).value = shamt
        getattr(dut, mode_port.name).value = int(mode)
        dut.in_valid.value = valid
        if valid:
            shft = ishl(x, shamt, width) if mode is ShiftMode.LEFT else ishr(x, shamt, width)
            desc = f"W={width} {mode.name} x={signed(x, width)} shamt={signed(shamt, width)}"
            scoreboard.push({result_port: shft, "_desc": desc})
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        scoreboard.sample()

    if width <= EXHAUSTIVE_MAX_WIDTH:
        # The direction is the innermost loop, so it also switches between every two consecutive transactions.
        for x in range(1 << width):
            for shamt in range(1 << width):
                for mode in ShiftMode:
                    await step(mode, x, shamt)
    else:
        minimum = 1 << (width - 1)
        # The last four counts straddle the end of the shifter's low count field, which lies past the word: a band
        # every other corner here misses.
        field = 1 << (width - 1).bit_length()
        counts = [
            value & mask
            for value in (
                *(0, 1, -1, width - 1, 1 - width, width, -width, width + 1, -width - 1, -minimum),
                *(field - 1, 1 - field, field, -field),
            )
        ]
        for x in (0, 1, 2, minimum - 1, minimum, minimum + 1, mask - 1, mask):
            for shamt in counts:
                for mode in ShiftMode:
                    await step(mode, x, shamt)
        rng = np.random.default_rng(int(os.environ.get("HOLOSO_TEST_SEED", "12345")))
        for _ in range(int(os.environ.get("HOLOSO_ISHFT_RANDOM", "2000"))):
            x = int(rng.integers(0, 1 << width, dtype=np.uint64))
            shamt = int(rng.integers(0, 1 << width, dtype=np.uint64))
            if rng.random() < 0.5:
                shamt = int(rng.integers(-width - 2, width + 3)) & mask
            await step(ShiftMode(int(rng.integers(0, 2))), x, shamt, bool(rng.random() >= 0.2))

    await scoreboard.drain()
    for value in range(4):
        getattr(dut, x_port).value = value & mask
        getattr(dut, shamt_port).value = (value + 1) & mask
        getattr(dut, mode_port.name).value = value & 1
        dut.in_valid.value = 1
        await RisingEdge(dut.clk)
    dut.rst.value = 1
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    dut.in_valid.value = 0
    for _ in range(4):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        assert not int(dut.out_valid.value)


@pytest.mark.parametrize("width", TEST_WIDTHS, ids=lambda width: f"w{width}")
@pytest.mark.parametrize("sim", SIMULATORS)
def test_ishft(sim: str, width: int) -> None:
    # The operator supplies the module name, the RTL parameters, the port names and the latency, so a declaration
    # that drifted from the hardware fails right here, across every width the sweep covers.
    operator = IShftOperator.build(IntFormat(width), IShftOptions())
    runner = get_runner(sim)
    build_dir = REPO_ROOT / "build" / "cocotb" / sim / f"{operator.module_name}_w{width}"
    runner.build(
        sources=sources(),
        includes=[HDL_DIR],
        hdl_toplevel=operator.module_name,
        parameters=operator.params,
        build_args=build_args(sim),
        build_dir=build_dir,
        clean=True,
        timescale=("1ns", "1ps"),
    )
    runner.test(
        hdl_toplevel=operator.module_name,
        test_module="tests.hdl.test_ishft",
        test_dir=REPO_ROOT,
        build_dir=build_dir,
        extra_env={
            "HOLOSO_ISHFT_WIDTH": str(width),
            "HOLOSO_ISHFT_OPERANDS": ",".join(port.name for port in operator.operand_ports),
            "HOLOSO_ISHFT_RESULTS": ",".join(port.name for port in operator.output_ports),
            "HOLOSO_EXPECTED_LATENCY": str(operator.latencies[0]),
        },
        results_xml=str(build_dir / "results.xml"),
    )
