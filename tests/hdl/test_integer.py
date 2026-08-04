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
from .hdl_integer_oracle import EXHAUSTIVE_MAX_WIDTH, TEST_WIDTHS, ashift, signed

_OPERATORS = ("holoso_iadds", "holoso_isubs", "holoso_iabss", "holoso_icmp", "holoso_ashift")
_UNARY = frozenset(("holoso_iabss",))


def _expected(operator: str, a_bits: int, b_bits: int, width: int, saturating: bool) -> dict[str, int | str]:
    modulus = 1 << width
    minimum = -(1 << (width - 1))
    maximum = (1 << (width - 1)) - 1
    a = signed(a_bits, width)
    b = signed(b_bits, width)
    if operator == "holoso_icmp":
        return {"a_gt_b": int(a > b), "a_eq_b": int(a == b), "a_lt_b": int(a < b)}
    if operator == "holoso_ashift":
        y, sat = ashift(a_bits, b_bits, width, saturating)
        return {"y": y, "saturated": sat}
    exact = {
        "holoso_iadds": a + b,
        "holoso_isubs": a - b,
        "holoso_iabss": abs(a),
    }[operator]
    clamped = min(max(exact, minimum), maximum)
    return {"y": clamped & (modulus - 1), "saturated": int(clamped != exact)}


@cocotb.test()
async def integer_operator_cocotb(dut: Any) -> None:
    operator = os.environ["HOLOSO_INTEGER_OPERATOR"]
    width = int(os.environ["HOLOSO_INTEGER_WIDTH"])
    if operator == "holoso_icmp":
        outputs = [("a_gt_b", "a_gt_b"), ("a_eq_b", "a_eq_b"), ("a_lt_b", "a_lt_b")]
    else:
        outputs = [("y", "y"), ("saturated", "saturated")]
    modes = (False, True) if operator == "holoso_ashift" else (False,)
    scoreboard = PipelineScoreboard(dut, outputs, latency=2)
    await start_clock(dut)
    await drive_reset(dut)

    async def step(a: int, b: int = 0, valid: bool = True, saturating: bool = False) -> None:
        if operator == "holoso_ashift":
            dut.x.value = a
            dut.shamt.value = b
            dut.saturating.value = int(saturating)
        elif operator == "holoso_iabss":
            dut.x.value = a
        else:
            dut.a.value = a
        if operator not in _UNARY and operator != "holoso_ashift":
            dut.b.value = b
        dut.in_valid.value = valid
        if valid:
            expected = _expected(operator, a, b, width, saturating)
            expected["_desc"] = f"{operator} W={width} a=0x{a:x} b=0x{b:x} sat={int(saturating)}"
            scoreboard.push(expected)
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        scoreboard.sample()

    if width <= EXHAUSTIVE_MAX_WIDTH:
        for a in range(1 << width):
            if operator in _UNARY:
                await step(a)
            else:
                for b in range(1 << width):
                    for saturating in modes:
                        await step(a, b, saturating=saturating)
    else:
        mask = (1 << width) - 1
        minimum = 1 << (width - 1)
        directed = [0, 1, 2, minimum - 1, minimum, minimum + 1, mask - 1, mask]
        for a in directed:
            if operator in _UNARY:
                await step(a)
            else:
                operands_b = directed
                if operator == "holoso_ashift":
                    operands_b = [
                        value & mask for value in (0, 1, -1, width - 1, 1 - width, width, -width, width + 1, -width - 1)
                    ]
                for b in operands_b:
                    for saturating in modes:
                        await step(a, b, saturating=saturating)
        if operator == "holoso_ashift":
            # Straddle the exact/overflow boundary of every left shift amount, which is what the overflow mask
            # actually decides. The boundary is asymmetric by one -- -2**k shifts exactly where +2**k already
            # overflows -- so both signs are driven on both sides of it; the trailing corners pin the two operands
            # whose magnitude alone cannot decide them.
            for shift in range(width + 1):
                headroom = 1 << (width - 1 - shift) if shift < width else 0
                for value in (headroom, headroom - 1, -headroom, -headroom - 1):
                    await step(value & mask, shift, saturating=True)
            for value, shift in ((-minimum, 0), (-minimum, 1), (-1, width - 1), (-1, width)):
                await step(value & mask, shift, saturating=True)
        rng = np.random.default_rng(int(os.environ.get("HOLOSO_TEST_SEED", "12345")))
        for _ in range(int(os.environ.get("HOLOSO_INTEGER_RANDOM", "1000"))):
            b = int(rng.integers(0, 1 << width, dtype=np.uint64))
            if operator == "holoso_ashift" and rng.random() < 0.5:
                b = int(rng.integers(-width - 2, width + 3)) & mask
            saturating = operator == "holoso_ashift" and rng.random() < 0.5
            await step(int(rng.integers(0, 1 << width, dtype=np.uint64)), b, bool(rng.random() >= 0.2), saturating)

    await scoreboard.drain()
    mask = (1 << width) - 1
    for value in range(4):
        if operator == "holoso_ashift":
            dut.x.value = value & mask
            dut.shamt.value = (value + 1) & mask
            dut.saturating.value = value & 1
        elif operator == "holoso_iabss":
            dut.x.value = value & mask
        else:
            dut.a.value = value & mask
        if operator not in _UNARY and operator != "holoso_ashift":
            dut.b.value = (value + 1) & mask
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
@pytest.mark.parametrize("operator", _OPERATORS)
@pytest.mark.parametrize("sim", SIMULATORS)
def test_integer_operator(sim: str, operator: str, width: int) -> None:
    runner = get_runner(sim)
    build_dir = REPO_ROOT / "build" / "cocotb" / sim / f"{operator}_w{width}"
    runner.build(
        sources=sources(),
        includes=[HDL_DIR],
        hdl_toplevel=operator,
        parameters={"W": width, "LATENCY": 2},
        build_args=build_args(sim),
        build_dir=build_dir,
        clean=True,
        timescale=("1ns", "1ps"),
    )
    runner.test(
        hdl_toplevel=operator,
        test_module="tests.hdl.test_integer",
        test_dir=REPO_ROOT,
        build_dir=build_dir,
        extra_env={"HOLOSO_INTEGER_OPERATOR": operator, "HOLOSO_INTEGER_WIDTH": str(width)},
        results_xml=str(build_dir / "results.xml"),
    )
