"""
Tests for holoso_ffma (pipelined; sgnop on a, b, c, y; y = sgnop(sgnop(a)*sgnop(b) + sgnop(c)), single rounding).

The wrapper delays y_sgnop through the same number of stages as zkf_fma, so all sgnop controls are allowed to vary
every input cycle. The oracle is the exact FloatValue.fma; the vendored zkf_fma RTL is the independent anchor, so a
bit-exact match proves the single-rounding fused result against hardware (including cases where a separate
multiply-then-add would double-round differently).
"""

import os

import cocotb
import numpy as np
import pytest
from cocotb.triggers import RisingEdge, Timer
from cocotb_tools.runner import get_runner

from holoso import FloatFormat, FFmaOperator

from .hdl_float_oracle import (
    HDL_DIR,
    PipelineScoreboard,
    REPO_ROOT,
    SGNOP_OPS,
    SIMULATORS,
    apply_sgnop,
    build_args,
    drive_reset,
    f32_to_bits,
    fma_oracle_bits,
    get_random_count,
    get_seed,
    random_zkf_f32,
    sources,
    start_clock,
)

# (input, product, decode, align, normalize, pack, output): base (latency 5) + each latency-bearing knob + all-on.
STAGE_COMBOS = (
    (0, 0, 0, 0, 0, 0, 0),
    (1, 0, 0, 0, 0, 0, 0),
    (0, 1, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 2, 0, 0),
    (1, 1, 1, 1, 2, 1, 1),
)

# A directed corner battery: a small value set cubed (covers 0 / +-1 / +-2 / 0.5 / pi / +-inf in every operand slot)
# plus exact-cancellation triples (a*b + -(a*b) -> +0, the signature fused result a separate add cannot guarantee).
_SMALL = tuple(f32_to_bits(np.float32(v)) for v in (0.0, 1.0, -1.0, 2.0, 0.5, np.pi, np.inf, -np.inf))
_CANCELLATION = (
    (f32_to_bits(2.0), f32_to_bits(3.0), f32_to_bits(-6.0)),
    (f32_to_bits(np.float32(1.0 + 2.0**-23)), f32_to_bits(np.float32(1.0 + 2.0**-23)), f32_to_bits(-1.0)),
    (f32_to_bits(np.float32(np.pi)), f32_to_bits(np.float32(np.e)), f32_to_bits(np.float32(-np.pi * np.e))),
    (f32_to_bits(np.float32(1e15)), f32_to_bits(np.float32(1e15)), f32_to_bits(np.float32(-1e30))),
)


@cocotb.test()
async def holoso_ffma_cocotb(dut) -> None:
    await start_clock(dut)
    await drive_reset(dut)

    sb = PipelineScoreboard(dut, [("y", "y")], latency=int(os.environ["HOLOSO_EXPECTED_LATENCY"]))
    rng = np.random.default_rng(get_seed())

    async def step_idle() -> None:
        dut.in_valid.value = 0
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        sb.sample()

    async def step(a: int, b: int, c: int, a_op: int, b_op: int, c_op: int, y_op: int) -> None:
        a_eff, b_eff, c_eff = apply_sgnop(a, a_op), apply_sgnop(b, b_op), apply_sgnop(c, c_op)
        expected = apply_sgnop(fma_oracle_bits(a_eff, b_eff, c_eff), y_op)
        dut.a.value, dut.b.value, dut.c.value = a, b, c
        dut.a_sgnop.value, dut.b_sgnop.value, dut.c_sgnop.value, dut.y_sgnop.value = a_op, b_op, c_op, y_op
        dut.in_valid.value = 1
        sb.push({"y": expected, "_desc": f"a=0x{a:08x} b=0x{b:08x} c=0x{c:08x} ops={a_op}{b_op}{c_op}{y_op}"})
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        sb.sample()

    for a in _SMALL:
        for b in _SMALL:
            for c in _SMALL:
                await step(a, b, c, 0, 0, 0, 0)
    for a, b, c in _CANCELLATION:
        await step(a, b, c, 0, 0, 0, 0)
    await sb.drain()

    triples = [tuple(int(rng.integers(0, len(_SMALL))) for _ in range(3)) for _ in range(6)]
    for a_op in SGNOP_OPS:
        for y_op in SGNOP_OPS:
            for ia, ib, ic in triples:
                await step(_SMALL[ia], _SMALL[ib], _SMALL[ic], a_op, 0, 0, y_op)
    await sb.drain()

    for _ in range(get_random_count()):
        if rng.random() < 0.2:
            await step_idle()
            continue
        a, b, c = random_zkf_f32(rng), random_zkf_f32(rng), random_zkf_f32(rng)
        ops = [int(rng.integers(0, 4)) for _ in range(4)]
        await step(a, b, c, ops[0], ops[1], ops[2], ops[3])
    await sb.drain()

    await drive_reset(dut)
    for _ in range(8):
        dut.in_valid.value = 0
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        assert int(dut.out_valid.value) == 0


@pytest.mark.parametrize("stages", STAGE_COMBOS, ids=lambda s: "".join(str(v) for v in s))
@pytest.mark.parametrize("sim", SIMULATORS)
def test_holoso_ffma(sim: str, stages: tuple[int, int, int, int, int, int, int]) -> None:
    si, sp, sd, sa, sn, spk, so = stages
    latency = FFmaOperator(
        FloatFormat(8, 24),
        stage_input=si,
        stage_product=sp,
        stage_decode=sd,
        stage_align=sa,
        stage_normalize=sn,
        stage_pack=spk,
        stage_output=so,
    ).latency
    runner = get_runner(sim)
    build_dir = REPO_ROOT / "build" / "cocotb" / sim / f"ffma_{''.join(str(v) for v in stages)}"
    runner.build(
        sources=sources(),
        includes=[HDL_DIR],
        hdl_toplevel="holoso_ffma",
        parameters={
            "WEXP": 8,
            "WMAN": 24,
            "STAGE_INPUT": si,
            "STAGE_PRODUCT": sp,
            "STAGE_DECODE": sd,
            "STAGE_ALIGN": sa,
            "STAGE_NORMALIZE": sn,
            "STAGE_PACK": spk,
            "STAGE_OUTPUT": so,
            "LATENCY": latency,
        },
        build_args=build_args(sim),
        build_dir=build_dir,
        clean=True,
        timescale=("1ns", "1ps"),
    )
    runner.test(
        hdl_toplevel="holoso_ffma",
        test_module="tests.hdl.test_ffma",
        test_dir=REPO_ROOT,
        build_dir=build_dir,
        extra_env={"HOLOSO_EXPECTED_LATENCY": str(latency)},
        results_xml=str(build_dir / "results.xml"),
    )
