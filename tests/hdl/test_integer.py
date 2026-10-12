import os
from collections.abc import Callable
from typing import Any

import cocotb
import numpy as np
import pytest
from cocotb.triggers import RisingEdge, Timer
from cocotb_tools.runner import get_runner

from holoso import IAbssOptions, IntFormat, IPopcntOptions
from holoso._operators import HardwareOperator, IAbsOperator, IPopcntOperator

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

# The operator is the source of the module name, its RTL parameters, its port names and its latency, so a declaration
# that drifted from the hardware fails right here, across every width the sweep covers.
_OPERATORS: list[Callable[[IntFormat], HardwareOperator]] = [
    lambda fmt: IAbsOperator.build(fmt, IAbssOptions()),
    lambda fmt: IPopcntOperator.build(fmt, IPopcntOptions()),
]


@cocotb.test()
async def integer_operator_cocotb(dut: Any) -> None:
    operator = os.environ["HOLOSO_INTEGER_OPERATOR"]
    width = int(os.environ["HOLOSO_INTEGER_WIDTH"])
    (operand,) = os.environ["HOLOSO_INTEGER_OPERANDS"].split(",")
    results = os.environ["HOLOSO_INTEGER_RESULTS"].split(",")
    # Value ports from the operator, so a name it declares and the RTL lacks fails here; sidebands from the oracle,
    # which is what knows whether this module raises one.
    sidebands = sorted(set(expected_simple(operator, 0, width)) - set(results))
    outputs = [(port, port) for port in results + sidebands]
    scoreboard = PipelineScoreboard(dut, outputs, latency=int(os.environ["HOLOSO_EXPECTED_LATENCY"]))
    await start_clock(dut)
    await drive_reset(dut)

    async def step(a: int, valid: bool = True) -> None:
        # Driven through the operator's own operand port name, so a misdeclared one fails here.
        getattr(dut, operand).value = a
        dut.in_valid.value = valid
        if valid:
            scoreboard.push({**expected_simple(operator, a, width), "_desc": f"{operator} W={width} a=0x{a:x}"})
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        scoreboard.sample()

    if width <= EXHAUSTIVE_MAX_WIDTH:
        for a in range(1 << width):
            await step(a)
    else:
        mask = (1 << width) - 1
        minimum = 1 << (width - 1)
        for a in (0, 1, 2, minimum - 1, minimum, minimum + 1, mask - 1, mask):
            await step(a)
        rng = np.random.default_rng(int(os.environ.get("HOLOSO_TEST_SEED", "12345")))
        for _ in range(int(os.environ.get("HOLOSO_INTEGER_RANDOM", "1000"))):
            await step(int(rng.integers(0, 1 << width, dtype=np.uint64)), bool(rng.random() >= 0.2))

    await scoreboard.drain()
    mask = (1 << width) - 1
    for value in range(4):
        getattr(dut, operand).value = value & mask
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
@pytest.mark.parametrize("operator_of", _OPERATORS, ids=lambda operator_of: operator_of(IntFormat(2)).name)
@pytest.mark.parametrize("sim", SIMULATORS)
def test_integer_operator(sim: str, operator_of: Callable[[IntFormat], HardwareOperator], width: int) -> None:
    operator = operator_of(IntFormat(width))
    module = operator.module_name
    runner = get_runner(sim)
    build_dir = REPO_ROOT / "build" / "cocotb" / sim / f"{module}_w{width}"
    runner.build(
        sources=sources(),
        includes=[HDL_DIR],
        hdl_toplevel=module,
        parameters=operator.params,
        build_args=build_args(sim),
        build_dir=build_dir,
        clean=True,
        timescale=("1ns", "1ps"),
    )
    runner.test(
        hdl_toplevel=module,
        test_module="tests.hdl.test_integer",
        test_dir=REPO_ROOT,
        build_dir=build_dir,
        extra_env={
            "HOLOSO_INTEGER_OPERATOR": module,
            "HOLOSO_INTEGER_WIDTH": str(width),
            "HOLOSO_INTEGER_OPERANDS": ",".join(port.name for port in operator.operand_ports),
            "HOLOSO_INTEGER_RESULTS": ",".join(port.name for port in operator.output_ports),
            "HOLOSO_EXPECTED_LATENCY": str(operator.latencies[0]),
        },
        results_xml=str(build_dir / "results.xml"),
    )
