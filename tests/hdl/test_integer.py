import os
from typing import Any

import cocotb
import numpy as np
import pytest
from cocotb.triggers import RisingEdge, Timer
from cocotb_tools.runner import get_runner

from holoso import IntFormat
from holoso._operators import (
    IAbsOperator,
    IAddOperator,
    ICmpOperator,
    IShlOperator,
    ISubOperator,
    IntHardwareOperator,
)

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
from .hdl_integer_oracle import EXHAUSTIVE_MAX_WIDTH, TEST_WIDTHS, expected_simple

# The operator model is the source of the module name, its RTL parameters, its port names and its latency, so a
# declaration that drifted from the hardware fails right here, across every width the sweep covers.
_OPERATORS = (IAddOperator, ISubOperator, IAbsOperator, ICmpOperator, IShlOperator)


@cocotb.test()
async def integer_operator_cocotb(dut: Any) -> None:
    operator = os.environ["HOLOSO_INTEGER_OPERATOR"]
    width = int(os.environ["HOLOSO_INTEGER_WIDTH"])
    operands = os.environ["HOLOSO_INTEGER_OPERANDS"].split(",")
    results = os.environ["HOLOSO_INTEGER_RESULTS"].split(",")
    unary = len(operands) == 1
    outputs = [(port, port) for port in results + (["saturated"] if operator != "holoso_icmp" else [])]
    scoreboard = PipelineScoreboard(dut, outputs, latency=int(os.environ["HOLOSO_EXPECTED_LATENCY"]))
    await start_clock(dut)
    await drive_reset(dut)

    async def step(a: int, b: int = 0, valid: bool = True) -> None:
        # Driven through the operator's own operand port names, in its own order, so a misdeclared pair miscomputes.
        for port, value in zip(operands, (a, b), strict=False):
            getattr(dut, port).value = value
        dut.in_valid.value = valid
        if valid:
            scoreboard.push(
                {**expected_simple(operator, a, b, width), "_desc": f"{operator} W={width} a=0x{a:x} b=0x{b:x}"}
            )
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        scoreboard.sample()

    if width <= EXHAUSTIVE_MAX_WIDTH:
        for a in range(1 << width):
            if unary:
                await step(a)
            else:
                for b in range(1 << width):
                    await step(a, b)
    else:
        mask = (1 << width) - 1
        minimum = 1 << (width - 1)
        directed = [0, 1, 2, minimum - 1, minimum, minimum + 1, mask - 1, mask]
        for a in directed:
            if unary:
                await step(a)
            else:
                operands_b = directed
                if operator == "holoso_ishl":
                    operands_b = [
                        value & mask
                        for value in (0, 1, -1, width - 1, 1 - width, width, -width, width + 1, -width - 1, -minimum)
                    ]
                for b in operands_b:
                    await step(a, b)
        if operator == "holoso_ishl":
            # Straddle the exact/overflow boundary of every left shift amount, which is what the overflow mask
            # actually decides. The boundary is asymmetric by one -- -2**k shifts exactly where +2**k already
            # overflows -- so both signs are driven on both sides of it; the trailing corners pin the two operands
            # whose magnitude alone cannot decide them.
            for shift in range(width + 1):
                headroom = 1 << (width - 1 - shift) if shift < width else 0
                for value in (headroom, headroom - 1, -headroom, -headroom - 1):
                    await step(value & mask, shift)
            for value, shift in ((-minimum, 0), (-minimum, 1), (-1, width - 1), (-1, width)):
                await step(value & mask, shift)
        rng = np.random.default_rng(int(os.environ.get("HOLOSO_TEST_SEED", "12345")))
        for _ in range(int(os.environ.get("HOLOSO_INTEGER_RANDOM", "1000"))):
            b = int(rng.integers(0, 1 << width, dtype=np.uint64))
            if operator == "holoso_ishl" and rng.random() < 0.5:
                b = int(rng.integers(-width - 2, width + 3)) & mask
            await step(int(rng.integers(0, 1 << width, dtype=np.uint64)), b, bool(rng.random() >= 0.2))

    await scoreboard.drain()
    mask = (1 << width) - 1
    for value in range(4):
        for port, driven in zip(operands, (value & mask, (value + 1) & mask), strict=False):
            getattr(dut, port).value = driven
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
@pytest.mark.parametrize("operator_class", _OPERATORS, ids=lambda cls: cls.mnemonic)
@pytest.mark.parametrize("sim", SIMULATORS)
def test_integer_operator(sim: str, operator_class: type[IntHardwareOperator], width: int) -> None:
    hardware = operator_class(IntFormat(width))
    operator = hardware.module_name
    runner = get_runner(sim)
    build_dir = REPO_ROOT / "build" / "cocotb" / sim / f"{operator}_w{width}"
    runner.build(
        sources=sources(),
        includes=[HDL_DIR],
        hdl_toplevel=operator,
        parameters=hardware.params,
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
        extra_env={
            "HOLOSO_INTEGER_OPERATOR": operator,
            "HOLOSO_INTEGER_WIDTH": str(width),
            "HOLOSO_INTEGER_OPERANDS": ",".join(hardware.operand_hdl_ports),
            "HOLOSO_INTEGER_RESULTS": ",".join(hardware.output_hdl_ports),
            "HOLOSO_EXPECTED_LATENCY": str(hardware.latency),
        },
        results_xml=str(build_dir / "results.xml"),
    )
