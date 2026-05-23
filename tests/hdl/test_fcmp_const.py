"""Tests for holoso_fcmp_const (pipelined; compares sign-conditioned a against constant b)."""

from __future__ import annotations

import os

import cocotb
import numpy as np
import pytest
from cocotb.triggers import RisingEdge, Timer
from cocotb_tools.runner import get_runner

from hdl_float_oracle import (
    DIRECTED_F32,
    PipelineScoreboard,
    REPO_ROOT,
    SGNOP_OPS,
    SIMULATORS,
    BENCH_DIR,
    apply_sgnop,
    build_args,
    cmp_oracle,
    drive_reset,
    f32_to_bits,
    get_random_count,
    get_seed,
    random_zkf_f32,
    sources,
    start_clock,
)

FCMP_CONST_CASES: list[tuple[str, float, int]] = [
    ("zero", 0.0, f32_to_bits(0.0)),
    ("one", 1.0, f32_to_bits(1.0)),
    ("neg_one", -1.0, f32_to_bits(-1.0)),
    ("pi", float(np.float32(np.pi)), f32_to_bits(np.float32(np.pi))),
    ("neg_tiny", float(np.float32(-np.finfo(np.float32).tiny)), f32_to_bits(np.float32(-np.finfo(np.float32).tiny))),
]


@cocotb.test()
async def holoso_fcmp_const_cocotb(dut) -> None:
    b_bits = int(os.environ["HOLOSO_TEST_B_BITS"], 0)

    await start_clock(dut)
    await drive_reset(dut)

    sb = PipelineScoreboard(
        dut,
        [
            ("a_gt_b", "gt"),
            ("a_eq_b", "eq"),
            ("a_lt_b", "lt"),
        ],
    )
    rng = np.random.default_rng(get_seed())

    async def step_idle() -> None:
        dut.in_valid.value = 0
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        sb.sample()

    async def step(a: int, a_op: int) -> None:
        a_eff = apply_sgnop(a, a_op)
        gt, eq, lt = cmp_oracle(a_eff, b_bits)
        assert gt + eq + lt == 1, "oracle produced non-one-hot result"
        dut.a.value = a
        dut.a_sgnop.value = a_op
        dut.in_valid.value = 1
        sb.push(
            {
                "gt": gt,
                "eq": eq,
                "lt": lt,
                "_desc": f"a=0x{a:08x} a_op={a_op} b=0x{b_bits:08x}",
            }
        )
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        if int(dut.out_valid.value):
            dgt = int(dut.a_gt_b.value)
            deq = int(dut.a_eq_b.value)
            dlt = int(dut.a_lt_b.value)
            assert dgt + deq + dlt == 1, f"DUT flags not one-hot: gt={dgt} eq={deq} lt={dlt}"
        sb.sample()

    for a in DIRECTED_F32:
        for a_op in SGNOP_OPS:
            await step(a, a_op)
    await sb.drain()

    for _ in range(get_random_count()):
        if rng.random() < 0.2:
            await step_idle()
            continue
        await step(random_zkf_f32(rng), int(rng.integers(0, 4)))
    await sb.drain()

    await drive_reset(dut)
    for _ in range(8):
        dut.in_valid.value = 0
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        assert int(dut.out_valid.value) == 0


@pytest.mark.parametrize("case", FCMP_CONST_CASES, ids=[c[0] for c in FCMP_CONST_CASES])
@pytest.mark.parametrize("sim", SIMULATORS)
def test_holoso_fcmp_const(sim: str, case: tuple[str, float, int]) -> None:
    name, b_value, b_bits = case
    runner = get_runner(sim)
    build_dir = REPO_ROOT / "build" / "cocotb" / sim / f"fcmp_const_{name}"
    runner.build(
        sources=sources(),
        includes=[REPO_ROOT / "hdl"],
        hdl_toplevel="holoso_fcmp_const",
        parameters={"WEXP": 8, "WMAN": 24, "B": b_value},
        build_args=build_args(sim),
        build_dir=build_dir,
        clean=True,
        timescale=("1ns", "1ps"),
    )
    runner.test(
        hdl_toplevel="holoso_fcmp_const",
        test_module="test_fcmp_const",
        test_dir=BENCH_DIR,
        build_dir=build_dir,
        extra_env={"HOLOSO_TEST_B_BITS": hex(b_bits)},
        results_xml=str(build_dir / "results.xml"),
    )
