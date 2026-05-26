"""
Directed cosim: a divide-by-zero latches ``err_cyc`` to the fdiv's commit cycle, and it resets each run.

Builds the tiny module ``a / b``, then drives three back-to-back invocations: a normal one (err_cyc stays 0), a
zero-divisor one (err_cyc latches the fdiv commit cycle, == cycle_count-1, which is nonzero), and a normal one again
(the per-initiation reset must have cleared the prior error).
"""

import json
import os

import cocotb
import pytest
from cocotb.triggers import RisingEdge, Timer
from cocotb_tools.runner import get_runner

from holoso import FAddOp, FDivOp, FloatFormat, FMulILog2GenericOp, FMulOp, OpConfig
from holoso._backend.verilog import generate
from holoso._frontend import lower
from holoso._passes import run
from holoso._schedule import build

from .hdl_float_oracle import HDL_DIR, REPO_ROOT, SIMULATORS, build_args, drive_reset, sources, start_clock

FMT = FloatFormat(6, 18)
OPS = OpConfig(FAddOp(), FMulOp(), FDivOp(), FMulILog2GenericOp())


def _divide(a, b):  # type: ignore[no-untyped-def]
    return a / b


async def _settle(dut) -> None:  # type: ignore[no-untyped-def]
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")


@cocotb.test()
async def err_cyc_latches_div0(dut) -> None:
    cycles = int(json.loads(os.environ["HOLOSO_ERRCYC"])["cycles"])
    a_bits = FMT.encode(1.0)

    await start_clock(dut)
    await drive_reset(dut)
    dut.out_ready.value = 1

    async def invoke(b: float) -> int:
        while int(dut.in_ready.value) != 1:
            await _settle(dut)
        dut.in_a.value = a_bits
        dut.in_b.value = FMT.encode(b)
        dut.in_valid.value = 1
        await _settle(dut)
        dut.in_valid.value = 0
        while int(dut.out_valid.value) != 1:
            await _settle(dut)
        latched = int(dut.err_cyc.value)
        await _settle(dut)  # accept the result and return to idle
        return latched

    assert await invoke(2.0) == 0, "no-error run must leave err_cyc clear"
    # Divide by zero: the fdiv asserts div0 at its commit cycle, the last compute cycle.
    assert await invoke(0.0) == cycles - 1, "div0 must latch err_cyc to the fdiv commit cycle"
    assert await invoke(2.0) == 0, "the per-initiation reset must clear the previous run's error"


@pytest.mark.parametrize("sim", SIMULATORS)
def test_err_cyc(sim: str) -> None:
    lir = build(run(lower(_divide, FMT), OPS), "divide")
    gen_dir = REPO_ROOT / "build" / "holoso_gen" / f"divide_w{FMT.wexp}_{FMT.wman}"
    gen_dir.mkdir(parents=True, exist_ok=True)
    verilog_path = gen_dir / "divide.v"
    verilog_path.write_text(generate(lir).verilog)
    build_dir = REPO_ROOT / "build" / "cocotb" / sim / f"errcyc_divide_w{FMT.wexp}_{FMT.wman}"

    runner = get_runner(sim)
    runner.build(
        sources=[verilog_path, *sources()],
        includes=[HDL_DIR],
        hdl_toplevel="divide",
        build_args=build_args(sim),
        build_dir=str(build_dir),
        clean=True,
        timescale=("1ns", "1ps"),
    )
    runner.test(
        hdl_toplevel="divide",
        test_module="tests.hdl.test_err_cyc",
        test_dir=str(REPO_ROOT),
        build_dir=str(build_dir),
        extra_env={"HOLOSO_ERRCYC": json.dumps({"cycles": lir.initiation_interval})},
        results_xml=str(build_dir / "results.xml"),
    )
