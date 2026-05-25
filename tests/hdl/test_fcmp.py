"""
Tests for holoso_fcmp (pipelined; comparison with input sgnops only).

Outputs a_gt_b, a_eq_b, a_lt_b are mutually-exclusive one-hot flags. There is no output sgnop, so no drain on
sgnop change is needed; a_sgnop and b_sgnop can vary every cycle. The DUT's one-hot invariant is also checked
on every out_valid pulse in addition to the per-flag comparison.
"""

import cocotb
import numpy as np
import pytest
from cocotb.triggers import RisingEdge, Timer
from cocotb_tools.runner import get_runner

from hdl_float_oracle import (
    DIRECTED_F32,
    PipelineScoreboard,
    HDL_DIR,
    REPO_ROOT,
    SGNOP_OPS,
    SIMULATORS,
    BENCH_DIR,
    apply_sgnop,
    build_args,
    cmp_oracle,
    drive_reset,
    get_random_count,
    get_seed,
    random_zkf_f32,
    sources,
    start_clock,
)


@cocotb.test()
async def holoso_fcmp_cocotb(dut) -> None:
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
        # One-hot invariant only meaningful when out_valid; nothing extra here.

    async def step(a: int, b: int, a_op: int, b_op: int) -> None:
        a_eff = apply_sgnop(a, a_op)
        b_eff = apply_sgnop(b, b_op)
        gt, eq, lt = cmp_oracle(a_eff, b_eff)
        assert gt + eq + lt == 1, "oracle produced non-one-hot result"
        dut.a.value = a
        dut.b.value = b
        dut.a_sgnop.value = a_op
        dut.b_sgnop.value = b_op
        dut.in_valid.value = 1
        sb.push(
            {
                "gt": gt,
                "eq": eq,
                "lt": lt,
                "_desc": f"a=0x{a:08x} b=0x{b:08x} ops={a_op}{b_op}",
            }
        )
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        # Also assert one-hot invariant on the DUT itself when it pulses.
        if int(dut.out_valid.value):
            dgt = int(dut.a_gt_b.value)
            deq = int(dut.a_eq_b.value)
            dlt = int(dut.a_lt_b.value)
            assert dgt + deq + dlt == 1, f"DUT flags not one-hot: gt={dgt} eq={deq} lt={dlt}"
        sb.sample()

    # 1) Directed x directed, neutral sgnops.
    for a in DIRECTED_F32:
        for b in DIRECTED_F32:
            await step(a, b, 0, 0)
    await sb.drain()

    # 2) Sgnop sweep on a sample of operand pairs (varying sgnops per cycle is fine).
    sample_pairs = [
        (DIRECTED_F32[int(rng.integers(0, len(DIRECTED_F32)))], DIRECTED_F32[int(rng.integers(0, len(DIRECTED_F32)))])
        for _ in range(8)
    ]
    for a, b in sample_pairs:
        for a_op in SGNOP_OPS:
            for b_op in SGNOP_OPS:
                await step(a, b, a_op, b_op)
    await sb.drain()

    # 3) Cases that test equality after sgnop normalization (e.g., +3 vs -3 with neg).
    pi_p = DIRECTED_F32[7]  # +pi
    pi_n = DIRECTED_F32[8]  # -pi
    await step(pi_p, pi_n, 0, 1)  # +pi vs neg(-pi)=+pi  -> equal
    await step(pi_p, pi_p, 0, 2)  # +pi vs abs(+pi)=+pi  -> equal
    await step(pi_p, pi_n, 0, 2)  # +pi vs abs(-pi)=+pi  -> equal
    await sb.drain()

    # 4) Random bulk with random sgnops.
    for _ in range(get_random_count()):
        if rng.random() < 0.2:
            await step_idle()
            continue
        a = random_zkf_f32(rng)
        b = random_zkf_f32(rng)
        a_op = int(rng.integers(0, 4))
        b_op = int(rng.integers(0, 4))
        await step(a, b, a_op, b_op)
    await sb.drain()

    await drive_reset(dut)
    for _ in range(8):
        dut.in_valid.value = 0
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        assert int(dut.out_valid.value) == 0


@pytest.mark.parametrize("sim", SIMULATORS)
def test_holoso_fcmp(sim: str) -> None:
    runner = get_runner(sim)
    build_dir = REPO_ROOT / "build" / "cocotb" / sim / "fcmp"
    runner.build(
        sources=sources(),
        includes=[HDL_DIR],
        hdl_toplevel="holoso_fcmp",
        parameters={"WEXP": 8, "WMAN": 24},
        build_args=build_args(sim),
        build_dir=build_dir,
        clean=True,
        timescale=("1ns", "1ps"),
    )
    runner.test(
        hdl_toplevel="holoso_fcmp",
        test_module="test_fcmp",
        test_dir=BENCH_DIR,
        build_dir=build_dir,
        results_xml=str(build_dir / "results.xml"),
    )
