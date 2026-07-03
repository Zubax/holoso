"""
Tests the holoso_fsgnop inline function. The same-named module is co-compiled from the megafile, so this also serves
as a function/module name-coexistence canary across simulators.
"""

from pathlib import Path
import cocotb
import numpy as np
import pytest
from cocotb.triggers import Timer
from cocotb_tools.runner import get_runner

from .hdl_float_oracle import (
    DIRECTED_F32,
    HDL_DIR,
    REPO_ROOT,
    SGNOP_OPS,
    SIMULATORS,
    apply_sgnop,
    build_args,
    get_random_count,
    get_seed,
    random_zkf_f32,
    sources,
)

_WEXP = 8
_WMAN = 24
_WFULL = _WEXP + _WMAN


@cocotb.test()
async def holoso_fsgnop_fn_cocotb(dut) -> None:
    async def check(x_bits: int, op: int) -> None:
        dut.x.value = x_bits
        dut.op.value = op
        await Timer(1, unit="ns")
        actual = int(dut.y.value)
        expected = apply_sgnop(x_bits, op, _WFULL)
        assert actual == expected, f"x=0x{x_bits:08x} op={op}: got 0x{actual:08x}, want 0x{expected:08x}"

    for x in DIRECTED_F32:
        for op in SGNOP_OPS:
            await check(x, op)

    rng = np.random.default_rng(get_seed())
    for _ in range(get_random_count()):
        x = random_zkf_f32(rng)
        for op in SGNOP_OPS:
            await check(x, op)


@pytest.mark.parametrize("sim", SIMULATORS)
def test_holoso_fsgnop_fn(sim: str) -> None:
    runner = get_runner(sim)
    build_dir = REPO_ROOT / "build" / "cocotb" / sim / "fsgnop_fn"
    runner.build(
        sources=[*sources(), Path(__file__).resolve().parent / "holoso_support_fn_tb.v"],
        includes=[HDL_DIR],
        hdl_toplevel="holoso_fsgnop_tb",
        parameters={"WEXP": _WEXP, "WMAN": _WMAN},
        build_args=build_args(sim),
        build_dir=build_dir,
        clean=True,
        timescale=("1ns", "1ps"),
    )
    runner.test(
        hdl_toplevel="holoso_fsgnop_tb",
        test_module="tests.hdl.test_fsgnop_fn",
        test_dir=REPO_ROOT,
        build_dir=build_dir,
        results_xml=str(build_dir / "results.xml"),
    )
