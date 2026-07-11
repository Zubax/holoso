from pathlib import Path
from typing import Any

import cocotb
from cocotb.triggers import RisingEdge, Timer
from cocotb_tools.runner import get_runner

from synth.test_integer_ooc import _render_divider_wrapper

from .hdl_float_oracle import HDL_DIR, REPO_ROOT, build_args, drive_reset, sources, start_clock

_WIDTH = 24
_LATENCY = 14


@cocotb.test()
async def idivs_ooc_wrapper_cocotb(dut: Any) -> None:
    await start_clock(dut)
    await drive_reset(dut)
    dut.out_sel.value = 0
    dut.in_sel.value = 0
    dut.io_in.value = 4
    dut.in_valid.value = 0
    await RisingEdge(dut.clk)
    dut.in_sel.value = 1
    dut.io_in.value = 3
    await RisingEdge(dut.clk)

    expected = [2, 3, 4, 5]
    for numerator in (7, 10, 13, 16):
        dut.in_sel.value = 0
        dut.io_in.value = numerator
        dut.in_valid.value = 1
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        if int(dut.out_valid.value):
            assert int(dut.io_out.value) == expected.pop(0)

    dut.in_valid.value = 0
    for _ in range(_LATENCY + 6):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        if int(dut.out_valid.value):
            assert int(dut.io_out.value) == expected.pop(0)
        if not expected:
            return
    assert not expected


def test_idivs_ooc_wrapper_payload_alignment() -> None:
    top = "holoso_idivs_ooc_wrapper_tb"
    build_dir = REPO_ROOT / "build" / "cocotb" / "icarus" / top
    wrapper = REPO_ROOT / "build" / "ooc_wrapper_sources" / f"{top}.v"
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    wrapper.write_text(_render_divider_wrapper(top, _WIDTH, _LATENCY, 1), encoding="utf-8")
    runner = get_runner("icarus")
    runner.build(
        sources=[*sources(), Path(wrapper)],
        includes=[HDL_DIR],
        hdl_toplevel=top,
        build_args=build_args("icarus"),
        build_dir=build_dir,
        clean=True,
        timescale=("1ns", "1ps"),
    )
    runner.test(
        hdl_toplevel=top,
        test_module="tests.hdl.test_idivs_ooc_wrapper",
        test_dir=REPO_ROOT,
        build_dir=build_dir,
        results_xml=str(build_dir / "results.xml"),
    )
