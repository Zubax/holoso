"""
Directed cosim: a divide-by-zero latches ``err_pc`` to the fdiv's write step, and it resets each run.

Builds the tiny module ``a / b``, then drives three back-to-back invocations: a normal one (err_pc stays 0), a
zero-divisor one (err_pc latches the executing step on which the fdiv result is written back -- the commit step
itself -- which is nonzero), and a normal one again (the per-initiation reset
must have cleared the prior error). Parametrized over the fdiv output stage so the err flag and the result are shown
to latch/land together whether or not an output register stage delays the commit.
"""

import json
import os

import cocotb
import pytest
from cocotb.triggers import RisingEdge, Timer
from cocotb_tools.runner import get_runner

from holoso import (
    FAddOperator,
    FCmpOperator,
    FDivOperator,
    FloatFormat,
    FMulILog2OperatorFamily,
    FMulOperator,
    OpConfig,
)
from holoso._backend.verilog import generate
from holoso._frontend import lower
from holoso._hir import optimize
from holoso._lir import build, pooled_write_word
from holoso._mir import lower as lower_to_mir

from .hdl_float_oracle import HDL_DIR, REPO_ROOT, SIMULATORS, build_args, drive_reset, sources, start_clock

FMT = FloatFormat(6, 18)


def _ops(stage_output: int) -> OpConfig:
    return OpConfig(
        FAddOperator(FMT),
        FMulOperator(FMT),
        FDivOperator(FMT, stage_output=stage_output),
        FMulILog2OperatorFamily(FMT),
        FCmpOperator(FMT),
    )


def _divide(a, b):  # type: ignore[no-untyped-def]
    return a / b


async def _settle(dut) -> None:  # type: ignore[no-untyped-def]
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")


@cocotb.test()
async def err_pc_latches_div0(dut) -> None:
    err_step = int(json.loads(os.environ["HOLOSO_ERRCYC"])["err_step"])
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
        latched = int(dut.err_pc.value)
        await _settle(dut)  # accept the result and return to idle
        return latched

    assert await invoke(2.0) == 0, "no-error run must leave err_pc clear"
    # Divide by zero: the fdiv asserts div0 at its commit; the write step is the commit step itself.
    assert await invoke(0.0) == err_step, "div0 must latch err_pc to the fdiv write step"
    assert await invoke(2.0) == 0, "the per-initiation reset must clear the previous run's error"


@pytest.mark.parametrize("stage_output", [0, 1])
@pytest.mark.parametrize("sim", SIMULATORS)
def test_err_pc(sim: str, stage_output: int) -> None:
    lir = build(lower_to_mir(optimize(lower(_divide)), _ops(stage_output)), "divide")
    # The fdiv asserts div0 at its commit; err_pc latches the write word -- the
    # commit step itself (pooled_write_word). An fdiv output stage pushes the commit later, and the err flag and the
    # result still latch/land together: err_step is recomputed from this build's actual fdiv commit.
    commit_cycle = next(op.commit_cycle for op in lir.ops if isinstance(op.inst.operator, FDivOperator))
    err_step = pooled_write_word(commit_cycle)
    gen_dir = REPO_ROOT / "build" / "holoso_gen" / f"divide_w{FMT.wexp}_{FMT.wman}_s{stage_output}"
    gen_dir.mkdir(parents=True, exist_ok=True)
    verilog_path = gen_dir / "divide.v"
    verilog_path.write_text(generate(lir).verilog)
    build_dir = REPO_ROOT / "build" / "cocotb" / sim / f"errcyc_divide_w{FMT.wexp}_{FMT.wman}_s{stage_output}"

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
        test_module="tests.hdl.test_err_pc",
        test_dir=str(REPO_ROOT),
        build_dir=str(build_dir),
        extra_env={"HOLOSO_ERRCYC": json.dumps({"err_step": err_step})},
        results_xml=str(build_dir / "results.xml"),
    )
