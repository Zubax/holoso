"""
Tests for holoso_fcordic. The core holds one transaction in flight, its mode chosen per transaction, so the bench
issues each transaction on the cycle after the previous one's out_valid -- the tightest spacing a schedule may use --
with the modes interleaved; SIMULATION=1 arms the wrapper's over-issue $fatal.
"""

import os
from typing import Any

import cocotb
import numpy as np
import pytest
import zkf
from cocotb.triggers import RisingEdge, Timer
from cocotb_tools.runner import get_runner

from .hdl_float_oracle import (
    DIRECTED_F32,
    HDL_DIR,
    REPO_ROOT,
    SGNOP_OPS,
    SIMULATORS,
    apply_sgnop,
    atan2_oracle,
    build_args,
    drive_reset,
    get_random_count,
    get_seed,
    random_zkf_f32,
    sincos_oracle,
    sources,
    stage_tag,
    start_clock,
)

STAGE_COMBOS: tuple[dict[str, int], ...] = (
    {},
    {"stage_input": 1, "stage_output": 1},
    {"stage_product": 1, "stage_normalize": 1, "stage_pack": 1},
    {"unroll100": 50},
    {"unroll100": 200, "stage_product": 2},
    {"stage_pack": 1, "stage_product": 2, "stage_normalize": 2},
)


async def _cordic(dut: Any, vectoring: bool, a: int, b: int, ops: tuple[int, int]) -> tuple[int, int]:
    latency = int(os.environ["HOLOSO_LATENCY_VECTORING" if vectoring else "HOLOSO_LATENCY_ROTATION"])
    dut.vectoring.value = int(vectoring)
    dut.a.value = a
    dut.b.value = b
    dut.a_sgnop.value, dut.b_sgnop.value = ops
    dut.in_valid.value = 1
    await RisingEdge(dut.clk)
    dut.in_valid.value = 0
    for cycle in range(latency + 16):
        await Timer(1, unit="ns")
        if int(dut.out_valid.value) == 1:
            assert cycle == latency - 1, f"out_valid at cycle {cycle}, expected {latency - 1} (vectoring={vectoring})"
            result = int(dut.r0.value), int(dut.r1.value)
            await RisingEdge(dut.clk)  # the core accepts again one cycle after retiring
            return result
        await RisingEdge(dut.clk)
    raise AssertionError("out_valid never asserted")


@cocotb.test()
async def holoso_fcordic_cocotb(dut: Any) -> None:
    await start_clock(dut)
    await drive_reset(dut)
    rng = np.random.default_rng(get_seed())

    async def rotate(a: int, a_op: int) -> None:
        # Rotation must not depend on b, so b and its sign control are driven with garbage.
        b, b_op = random_zkf_f32(rng), int(rng.integers(0, 4))
        expected = sincos_oracle(apply_sgnop(a, a_op))
        got = await _cordic(dut, False, a, b, (a_op, b_op))
        assert got == expected, f"rotation a=0x{a:08x} op={a_op}: got {got} expected {expected}"

    async def vector(y: int, x: int, ops: tuple[int, int]) -> None:
        expected = atan2_oracle(apply_sgnop(y, ops[0]), apply_sgnop(x, ops[1]))
        got = await _cordic(dut, True, y, x, ops)
        assert got == expected, f"vectoring y=0x{y:08x} x=0x{x:08x} ops={ops}: got {got} expected {expected}"

    grid = list(DIRECTED_F32[:9])
    for a in DIRECTED_F32:
        await rotate(a, 0)
    for y in grid:
        for x in grid:
            await vector(y, x, (0, 0))
    for op in SGNOP_OPS:
        await rotate(grid[int(rng.integers(0, len(grid)))], op)
        await vector(grid[int(rng.integers(0, len(grid)))], grid[int(rng.integers(0, len(grid)))], (op, 3 - op))
    for _ in range(get_random_count()):
        if rng.integers(0, 2):
            await vector(random_zkf_f32(rng), random_zkf_f32(rng), (int(rng.integers(0, 4)), int(rng.integers(0, 4))))
        else:
            await rotate(random_zkf_f32(rng), int(rng.integers(0, 4)))

    await drive_reset(dut)
    for _ in range(8):
        dut.in_valid.value = 0
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        assert int(dut.out_valid.value) == 0


@pytest.mark.parametrize("stages", STAGE_COMBOS, ids=stage_tag)
@pytest.mark.parametrize("sim", SIMULATORS)
def test_holoso_fcordic(sim: str, stages: dict[str, int]) -> None:
    model = zkf.CordicModel(zkf.ZkfFormat(8, 24), wmultiplier=0, **stages)
    runner = get_runner(sim)
    build_dir = REPO_ROOT / "build" / "cocotb" / sim / f"fcordic_{stage_tag(stages)}"
    runner.build(
        sources=sources(),
        includes=[HDL_DIR],
        hdl_toplevel="holoso_fcordic",
        parameters=model.params,
        build_args=build_args(sim),
        defines={"SIMULATION": 1},
        build_dir=build_dir,
        clean=True,
        timescale=("1ns", "1ps"),
    )
    runner.test(
        hdl_toplevel="holoso_fcordic",
        test_module="tests.hdl.test_fcordic",
        test_dir=REPO_ROOT,
        build_dir=build_dir,
        extra_env={
            "HOLOSO_LATENCY_ROTATION": str(model.latency_rotation),
            "HOLOSO_LATENCY_VECTORING": str(model.latency_vectoring),
        },
        results_xml=str(build_dir / "results.xml"),
    )
