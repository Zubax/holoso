"""Tests for holoso_fconst (elaboration-time constant generator).

Each parametrize case rebuilds the DUT with a different (VALUE, INF) pair and checks the static `y` output against
the float32 bit pattern of the encoded value. There is no clock and no inputs; a single Timer step lets `y` settle
before the comparison.
"""

from __future__ import annotations

import math
import os

import cocotb
import numpy as np
import pytest
from cocotb.triggers import Timer
from cocotb_tools.runner import get_runner

from hdl_float_oracle import (
    F32_NINF,
    F32_PINF,
    REPO_ROOT,
    SIMULATORS,
    TESTS_DIR,
    build_args,
    f32_to_bits,
    sources,
)

# (case_name, VALUE, INF, expected_bits)
CONST_CASES: list[tuple[str, float, int, int]] = [
    ("zero", 0.0, 0, f32_to_bits(0.0)),
    ("one", 1.0, 0, f32_to_bits(1.0)),
    ("neg_one", -1.0, 0, f32_to_bits(-1.0)),
    ("half", 0.5, 0, f32_to_bits(0.5)),
    ("neg_quart", -0.25, 0, f32_to_bits(-0.25)),
    ("three_half", 1.5, 0, f32_to_bits(1.5)),
    ("pi", math.pi, 0, f32_to_bits(np.float32(np.pi))),
    ("neg_pi", -math.pi, 0, f32_to_bits(np.float32(-np.pi))),
    ("pinf", 0.0, 1, F32_PINF),
    ("pinf_big", 0.0, 1000, F32_PINF),
    ("ninf", 0.0, -1, F32_NINF),
    ("ninf_big", 0.0, -1000, F32_NINF),
]


@cocotb.test()
async def holoso_fconst_cocotb(dut) -> None:
    expected = int(os.environ["HOLOSO_TEST_EXPECTED"], 0)
    await Timer(1, unit="ns")
    actual = int(dut.y.value)
    assert actual == expected, f"holoso_fconst output mismatch: got 0x{actual:08x}, want 0x{expected:08x}"


@pytest.mark.parametrize("case", CONST_CASES, ids=[c[0] for c in CONST_CASES])
@pytest.mark.parametrize("sim", SIMULATORS)
def test_holoso_fconst(sim: str, case: tuple[str, float, int, int]) -> None:
    name, value, inf, expected = case
    runner = get_runner(sim)
    build_dir = REPO_ROOT / "build" / "cocotb" / sim / f"fconst_{name}"
    runner.build(
        sources=sources(),
        includes=[REPO_ROOT / "hdl"],
        hdl_toplevel="holoso_fconst",
        parameters={"WEXP": 8, "WMAN": 24, "VALUE": value, "INF": inf},
        build_args=build_args(sim),
        build_dir=build_dir,
        clean=True,
        timescale=("1ns", "1ps"),
    )
    runner.test(
        hdl_toplevel="holoso_fconst",
        test_module="test_hdl_fconst",
        test_dir=TESTS_DIR,
        build_dir=build_dir,
        extra_env={"HOLOSO_TEST_EXPECTED": hex(expected)},
        results_xml=str(build_dir / "results.xml"),
    )
