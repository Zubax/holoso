from __future__ import annotations

import os
from pathlib import Path

import cocotb
import pytest
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge, Timer
from cocotb_tools.runner import get_runner

REPO_ROOT = Path(__file__).resolve().parents[2]
HDL = REPO_ROOT / "hdl" / "holoso_support.v"
BENCH_DIR = REPO_ROOT / "tests" / "hdl"


CASES = {
    "no_pass_1r1w": {"W": 7, "WADDR": 3, "NRD": 1, "NWR": 1, "NREG": 8, "RWPASS": 0},
    "pass_multi": {"W": 5, "WADDR": 3, "NRD": 3, "NWR": 2, "NREG": 8, "RWPASS": 1},
    "pass_limited": {"W": 6, "WADDR": 3, "NRD": 1, "NWR": 1, "NREG": 5, "RWPASS": 1},
}
SIMULATORS = (os.environ["SIM"],) if "SIM" in os.environ else ("icarus", "verilator")


def set_lane(bus: int, width: int, port: int, value: int) -> int:
    mask = (1 << width) - 1
    shift = port * width
    return (bus & ~(mask << shift)) | ((value & mask) << shift)


def get_lane(bus: int, width: int, port: int) -> int:
    return (bus >> (port * width)) & ((1 << width) - 1)


async def settle() -> None:
    await Timer(1, units="ns")


def rd_lane(dut, width: int, port: int) -> int:
    return get_lane(int(dut.rd_data.value), width, port)


def drive_defaults(dut) -> None:
    dut.wr_en.value = 0
    dut.wr_addr.value = 0
    dut.wr_data.value = 0
    dut.rd_addr.value = 0


async def start_clock(dut) -> None:
    drive_defaults(dut)
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await FallingEdge(dut.clk)


async def check_no_pass_1r1w(dut) -> None:
    await start_clock(dut)

    dut.rd_addr.value = 3
    dut.wr_en.value = 1
    dut.wr_addr.value = 3
    dut.wr_data.value = 0x55
    await RisingEdge(dut.clk)
    await settle()

    dut.wr_en.value = 0
    await settle()
    assert int(dut.rd_data.value) == 0x55

    dut.wr_en.value = 1
    dut.wr_addr.value = 3
    dut.wr_data.value = 0x2A
    await settle()
    assert int(dut.rd_data.value) == 0x55, "RWPASS=0 passed same-cycle write data to read"

    await RisingEdge(dut.clk)
    await settle()
    dut.wr_en.value = 0
    await settle()
    assert int(dut.rd_data.value) == 0x2A

    dut.wr_data.value = 0x11
    await RisingEdge(dut.clk)
    await settle()
    assert int(dut.rd_data.value) == 0x2A, "disabled write changed stored value"


async def check_pass_multi(dut) -> None:
    await start_clock(dut)

    rd_addr = 0
    rd_addr = set_lane(rd_addr, 3, 0, 2)
    rd_addr = set_lane(rd_addr, 3, 1, 5)
    rd_addr = set_lane(rd_addr, 3, 2, 2)
    wr_addr = 0
    wr_addr = set_lane(wr_addr, 3, 0, 2)
    wr_addr = set_lane(wr_addr, 3, 1, 5)
    wr_data = 0
    wr_data = set_lane(wr_data, 5, 0, 0x09)
    wr_data = set_lane(wr_data, 5, 1, 0x12)

    dut.rd_addr.value = rd_addr
    dut.wr_en.value = 0b11
    dut.wr_addr.value = wr_addr
    dut.wr_data.value = wr_data
    await settle()
    assert rd_lane(dut, 5, 0) == 0x09
    assert rd_lane(dut, 5, 1) == 0x12
    assert rd_lane(dut, 5, 2) == 0x09

    await RisingEdge(dut.clk)
    await settle()
    dut.wr_en.value = 0
    await settle()
    assert rd_lane(dut, 5, 0) == 0x09
    assert rd_lane(dut, 5, 1) == 0x12
    assert rd_lane(dut, 5, 2) == 0x09

    rd_addr = 0
    rd_addr = set_lane(rd_addr, 3, 0, 6)
    rd_addr = set_lane(rd_addr, 3, 1, 5)
    rd_addr = set_lane(rd_addr, 3, 2, 2)
    wr_addr = 0
    wr_addr = set_lane(wr_addr, 3, 0, 6)
    wr_addr = set_lane(wr_addr, 3, 1, 5)
    wr_data = 0
    wr_data = set_lane(wr_data, 5, 0, 0x1C)
    wr_data = set_lane(wr_data, 5, 1, 0x0F)

    dut.rd_addr.value = rd_addr
    dut.wr_en.value = 0b01
    dut.wr_addr.value = wr_addr
    dut.wr_data.value = wr_data
    await settle()
    assert rd_lane(dut, 5, 0) == 0x1C
    assert rd_lane(dut, 5, 1) == 0x12
    assert rd_lane(dut, 5, 2) == 0x09

    await RisingEdge(dut.clk)
    await settle()
    dut.wr_en.value = 0
    await settle()
    assert rd_lane(dut, 5, 0) == 0x1C

    rd_addr = 0
    rd_addr = set_lane(rd_addr, 3, 0, 1)
    rd_addr = set_lane(rd_addr, 3, 1, 1)
    rd_addr = set_lane(rd_addr, 3, 2, 1)
    wr_addr = 0
    wr_addr = set_lane(wr_addr, 3, 0, 1)
    wr_addr = set_lane(wr_addr, 3, 1, 1)
    wr_data = 0
    wr_data = set_lane(wr_data, 5, 0, 0x05)
    wr_data = set_lane(wr_data, 5, 1, 0x12)

    dut.rd_addr.value = rd_addr
    dut.wr_en.value = 0b11
    dut.wr_addr.value = wr_addr
    dut.wr_data.value = wr_data
    await settle()
    assert rd_lane(dut, 5, 0) == 0x17
    assert rd_lane(dut, 5, 1) == 0x17
    assert rd_lane(dut, 5, 2) == 0x17

    await RisingEdge(dut.clk)
    await settle()
    dut.wr_en.value = 0
    await settle()
    assert rd_lane(dut, 5, 0) == 0x17


async def check_pass_limited(dut) -> None:
    await start_clock(dut)

    dut.rd_addr.value = 4
    dut.wr_en.value = 1
    dut.wr_addr.value = 4
    dut.wr_data.value = 0x2D
    await settle()
    assert int(dut.rd_data.value) == 0x2D

    await RisingEdge(dut.clk)
    await settle()
    dut.wr_en.value = 0
    await settle()
    assert int(dut.rd_data.value) == 0x2D


@cocotb.test()
async def holoso_regfile_cocotb(dut) -> None:
    case = os.environ["HOLOSO_REGFILE_CASE"]
    if case == "no_pass_1r1w":
        await check_no_pass_1r1w(dut)
    elif case == "pass_multi":
        await check_pass_multi(dut)
    elif case == "pass_limited":
        await check_pass_limited(dut)
    else:
        raise AssertionError(f"unknown case {case!r}")


@pytest.mark.parametrize("case_name", CASES)
@pytest.mark.parametrize("sim", SIMULATORS)
def test_holoso_regfile(case_name: str, sim: str) -> None:
    runner = get_runner(sim)
    params = CASES[case_name]
    build_dir = REPO_ROOT / "build" / "cocotb" / sim / case_name
    build_args = []
    if sim == "verilator":
        build_args = [
            "--timing",
            "-Wno-fatal",
            "-Wno-DECLFILENAME",
            "-Wno-UNUSEDSIGNAL",
            "-Wno-WIDTH",
            "-Wno-CMPCONST",
        ]

    runner.build(
        sources=[HDL],
        includes=[REPO_ROOT / "hdl"],
        hdl_toplevel="holoso_regfile",
        parameters=params,
        build_args=build_args,
        build_dir=build_dir,
        clean=True,
        timescale=("1ns", "1ps"),
    )
    results_xml = build_dir / f"results_{case_name}.xml"
    runner.test(
        hdl_toplevel="holoso_regfile",
        test_module="test_regfile",
        test_dir=BENCH_DIR,
        build_dir=build_dir,
        extra_env={"HOLOSO_REGFILE_CASE": case_name},
        results_xml=str(results_xml),
    )
