"""
Tests for holoso_iadds (pipelined; `sub` selects y = a + b or y = a - b, both saturated, a subtraction also ordering a
against b on the flags), with the operation chosen per transaction and with the adder elaborated for each alone. An
adder fixed to one operation is driven with noise on `sub`, which it must ignore. A sum is scored on `y` and the
saturation sideband, a difference on the flags as well, against the fixed-width oracle. Every case runs with either
equality detector.
"""

import os
from typing import Any

import cocotb
import numpy as np
import pytest
from cocotb.triggers import RisingEdge, Timer
from cocotb_tools.runner import get_runner

from holoso import IAddsOptions, IntFormat
from holoso._operators import AddMode, IAddOperator

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
from .hdl_integer_oracle import (
    EXHAUSTIVE_MAX_WIDTH,
    TEST_WIDTHS,
    expected_iadds_add,
    expected_iadds_cmp,
    expected_iadds_sub,
    signed,
)


@cocotb.test()
async def iadds_cocotb(dut: Any) -> None:
    width = int(os.environ["HOLOSO_IADDS_WIDTH"])
    a_port, b_port = os.environ["HOLOSO_IADDS_OPERANDS"].split(",")
    results = os.environ["HOLOSO_IADDS_RESULTS"].split(",")
    fixed = os.environ["HOLOSO_FIXED_MODE"]
    modes = [AddMode[fixed]] if fixed else list(AddMode)
    mode_port = IAddOperator.mode_port
    assert mode_port is not None
    mask = (1 << width) - 1
    # Value ports from the operator, so a name it declares and the RTL lacks fails here; the saturation sideband, which
    # no operator declares, by the name the oracle gives it.
    scoreboard = PipelineScoreboard(
        dut, [(port, port) for port in [*results, "saturated"]], latency=int(os.environ["HOLOSO_EXPECTED_LATENCY"])
    )
    rng = np.random.default_rng(int(os.environ.get("HOLOSO_TEST_SEED", "12345")))
    await start_clock(dut)
    await drive_reset(dut)

    async def step(mode: AddMode, a: int, b: int, valid: bool = True) -> None:
        # Driven through the operator's own port names and its own mode codes, so a misdeclared one miscomputes.
        getattr(dut, a_port).value = a
        getattr(dut, b_port).value = b
        getattr(dut, mode_port.name).value = int(rng.integers(0, 1 << mode_port.width)) if fixed else int(mode)
        dut.in_valid.value = valid
        if valid:
            match mode:
                case AddMode.ADD:
                    expected = expected_iadds_add(a, b, width)
                case AddMode.SUB:
                    expected = {**expected_iadds_sub(a, b, width), **expected_iadds_cmp(a, b, width)}
            desc = f"W={width} {mode.name} a={signed(a, width)} b={signed(b, width)}"
            scoreboard.push({**expected, "_desc": desc})
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        scoreboard.sample()

    if width <= EXHAUSTIVE_MAX_WIDTH:
        # The mode is the innermost loop, so it also switches between every two consecutive transactions.
        for a in range(1 << width):
            for b in range(1 << width):
                for mode in modes:
                    await step(mode, a, b)
    else:
        minimum = 1 << (width - 1)
        directed = [0, 1, 2, minimum - 1, minimum, minimum + 1, mask - 1, mask]
        for a in directed:
            for b in directed:
                for mode in modes:
                    await step(mode, a, b)
        for _ in range(int(os.environ.get("HOLOSO_IADDS_RANDOM", "3000"))):
            a = int(rng.integers(0, 1 << width, dtype=np.uint64))
            b = int(rng.integers(0, 1 << width, dtype=np.uint64))
            if rng.random() < 0.1:
                b = a  # two uniform words of this width never meet, and equality has a cone of its own
            await step(modes[int(rng.integers(0, len(modes)))], a, b, bool(rng.random() >= 0.2))

    await scoreboard.drain()
    for value in range(4):
        getattr(dut, a_port).value = value & mask
        getattr(dut, b_port).value = (value + 1) & mask
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
@pytest.mark.parametrize("fast", (False, True), ids=("zero_difference", "fast"))
@pytest.mark.parametrize("sim", SIMULATORS)
def test_iadds(sim: str, fast: bool, width: int) -> None:
    _run(sim, width, None, fast)


@pytest.mark.parametrize("width", TEST_WIDTHS, ids=lambda width: f"w{width}")
@pytest.mark.parametrize("fast", (False, True), ids=("zero_difference", "fast"))
@pytest.mark.parametrize("mode", AddMode, ids=lambda mode: mode.name.lower())
@pytest.mark.parametrize("sim", SIMULATORS)
def test_iadds_fixed_to_one_mode(sim: str, mode: AddMode, fast: bool, width: int) -> None:
    _run(sim, width, mode, fast)


def _run(sim: str, width: int, fixed: AddMode | None, fast: bool) -> None:
    """
    `fixed` elaborates the adder for that operation alone; `None` keeps it choosing per transaction. The operator
    supplies the module name, the RTL parameters, the port names and the latency, so a declaration that drifted from the
    hardware fails right here, across every width the sweep covers.
    """
    operator = IAddOperator.build(IntFormat(width), IAddsOptions(fast=fast))
    match fixed:
        case None:
            modes = frozenset(operator.modes)
        case AddMode.ADD:
            modes = frozenset({operator.addition})
        case AddMode.SUB:
            modes = frozenset({operator.subtraction, operator.comparison})
    label = ("all" if fixed is None else fixed.name.lower()) + ("_fast" if fast else "")
    runner = get_runner(sim)
    build_dir = REPO_ROOT / "build" / "cocotb" / sim / f"{operator.module_name}_{label}_w{width}"
    runner.build(
        sources=sources(),
        includes=[HDL_DIR],
        hdl_toplevel=operator.module_name,
        parameters=operator.params_for(modes),
        build_args=build_args(sim),
        build_dir=build_dir,
        clean=True,
        timescale=("1ns", "1ps"),
    )
    runner.test(
        hdl_toplevel=operator.module_name,
        test_module="tests.hdl.test_iadds",
        test_dir=REPO_ROOT,
        build_dir=build_dir,
        extra_env={
            "HOLOSO_IADDS_WIDTH": str(width),
            "HOLOSO_IADDS_OPERANDS": ",".join(port.name for port in operator.operand_ports),
            "HOLOSO_IADDS_RESULTS": ",".join(port.name for port in operator.output_ports),
            "HOLOSO_EXPECTED_LATENCY": str(operator.latencies[0]),
            "HOLOSO_FIXED_MODE": "" if fixed is None else fixed.name,
        },
        results_xml=str(build_dir / "results.xml"),
    )
