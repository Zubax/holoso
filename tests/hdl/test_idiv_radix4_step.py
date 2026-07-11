import os
from typing import Any

import cocotb
import pytest
from cocotb.triggers import Timer
from cocotb_tools.runner import get_runner

from .hdl_float_oracle import HDL_DIR, REPO_ROOT, SIMULATORS, build_args, sources


@cocotb.test()
async def idiv_radix4_step_cocotb(dut: Any) -> None:
    width = int(os.environ["HOLOSO_IDIV_STEP_WIDTH"])
    for den in range(1, 1 << width):
        dut.den.value = den
        dut.den3.value = 3 * den
        for partial_rem in range(den):
            dut.partial_rem.value = partial_rem
            for next_bits in range(4):
                dut.next_bits.value = next_bits
                await Timer(1, unit="ns")
                candidate = 4 * partial_rem + next_bits
                expected_digit = candidate // den
                expected_remainder = candidate % den
                assert expected_digit <= 3
                assert int(dut.digit.value) == expected_digit
                assert int(dut.rem_next.value) == expected_remainder


@pytest.mark.parametrize("width", range(2, 7), ids=lambda width: f"w{width}")
@pytest.mark.parametrize("sim", SIMULATORS)
def test_idiv_radix4_step(sim: str, width: int) -> None:
    build_dir = REPO_ROOT / "build" / "cocotb" / sim / f"holoso_idiv_radix4_step_w{width}"
    runner = get_runner(sim)
    runner.build(
        sources=sources(),
        includes=[HDL_DIR],
        hdl_toplevel="_holoso_idiv_radix4_step",
        parameters={"W": width},
        build_args=build_args(sim),
        build_dir=build_dir,
        clean=True,
        timescale=("1ns", "1ps"),
    )
    runner.test(
        hdl_toplevel="_holoso_idiv_radix4_step",
        test_module="tests.hdl.test_idiv_radix4_step",
        test_dir=REPO_ROOT,
        build_dir=build_dir,
        extra_env={"HOLOSO_IDIV_STEP_WIDTH": str(width)},
        results_xml=str(build_dir / "results.xml"),
    )
