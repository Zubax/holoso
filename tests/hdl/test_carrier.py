"""
The one property no model can observe: a float occupies only the low bits of the wide register, so nothing ever SETS
a bit above ``WFLT`` (an unwritten datapath register reads X, being outside the reset cone). Both oracles store typed
values rather than register words, so this is checkable only against the RTL, and only where the integer is wider.
"""

import os
from typing import Any

import cocotb
import pytest
from cocotb.triggers import RisingEdge, Timer
from cocotb_tools.runner import get_runner

from holoso import FAddOptions, FCmpOptions, FMulILog2Options, FMulOptions, FloatFormat, OperatorOptions, Options
from holoso._backend.verilog import generate
from holoso._eel import lower
from holoso._mir import lower as lower_to_mir

from .hdl_float_oracle import HDL_DIR, REPO_ROOT, build_args, drive_reset, sources, start_clock
from .._modelref import build_lir, mir_options, DEFAULT_UNROLL_MAX_TRIPS

FMT = FloatFormat(6, 18)
WINT_MIN = 33


class _Accumulate:
    """Reaches every wide-register producer a float-only kernel has: input, constant, operator, sign fold, state."""

    def __init__(self) -> None:
        self.acc = 0.0

    def __call__(self, x: float, y: float) -> float:
        self.acc = abs(self.acc) * 0.5 - -x
        return self.acc + y * 3.7


def _options() -> Options:
    operators = OperatorOptions(
        fadd=FAddOptions(), fmul=FMulOptions(), fmul_ilog2=FMulILog2Options(), fcmp=FCmpOptions()
    )
    return Options(operators, ffmt=FMT, wint_min=WINT_MIN)


async def _settle(dut: Any) -> None:
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")


@cocotb.test()
async def float_leaves_the_upper_carrier_bits_clear(dut: Any) -> None:
    nreg = int(os.environ["HOLOSO_NREG"])
    assert len(str(dut.regs[0].value)) == WINT_MIN > FMT.width, "the premise: the register outsizes the float in it"

    def check(where: str) -> None:
        # A pure datapath register is deliberately outside the reset cone, so an unwritten one reads X; what must
        # never happen is a float write SETTING a bit above WFLT.
        for index in range(nreg):
            bits = str(dut.regs[index].value)
            assert "1" not in bits[: len(bits) - FMT.width], f"{where}: regs[{index}]={bits} sets a bit above WFLT"

    await start_clock(dut)
    await drive_reset(dut)
    check("after reset")
    dut.out_ready.value = 1
    for x, y in ((1.0, 2.0), (-3.5, 0.25), (0.0, -7.0), (-1e5, 1e5)):
        while int(dut.in_ready.value) != 1:
            await _settle(dut)
            check("idle")
        dut.in_x.value = FMT.encode(x)
        dut.in_y.value = FMT.encode(y)
        dut.in_valid.value = 1
        await _settle(dut)
        dut.in_valid.value = 0
        while int(dut.out_valid.value) != 1:
            await _settle(dut)
            check("mid-transaction")
        check("at out_valid")
        await _settle(dut)


@pytest.mark.parametrize("sim", ("icarus",))  # reads the register array through VPI, which verilator does not expose
def test_float_leaves_the_upper_carrier_bits_clear(sim: str) -> None:
    options = _options()
    lir = build_lir(
        lower_to_mir(lower(_Accumulate().__call__, DEFAULT_UNROLL_MAX_TRIPS).hir, mir_options(options)),
        "carrier",
    )
    assert lir.int_format.width == WINT_MIN > options.ffmt.width
    gen_dir = REPO_ROOT / "build" / "holoso_gen" / "carrier"
    gen_dir.mkdir(parents=True, exist_ok=True)
    verilog_path = gen_dir / "carrier.v"
    verilog_path.write_text(generate(lir).verilog)
    build_dir = REPO_ROOT / "build" / "cocotb" / sim / "carrier"

    runner = get_runner(sim)
    runner.build(
        sources=[verilog_path, *sources()],
        includes=[HDL_DIR],
        hdl_toplevel="carrier",
        build_args=build_args(sim),
        build_dir=str(build_dir),
        clean=True,
        timescale=("1ns", "1ps"),
    )
    runner.test(
        hdl_toplevel="carrier",
        test_module="tests.hdl.test_carrier",
        test_dir=str(REPO_ROOT),
        build_dir=str(build_dir),
        extra_env={"HOLOSO_NREG": str(lir.regfile.nreg)},
        results_xml=str(build_dir / "results.xml"),
    )
