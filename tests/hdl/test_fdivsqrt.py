"""
Tests for holoso_fdivsqrt (pipelined; `op_sqrt` selects y = sgnop(a)/sgnop(b) or y = sqrt(sgnop(a)); error alongside
out_valid), with the operation chosen per transaction and with the core elaborated for either one alone. A core fixed
to one operation is driven with noise on `op_sqrt`, and a root with noise on `b`, since both must be ignored.
"""

import os
from typing import Any

import cocotb
import numpy as np
import pytest
from cocotb.triggers import RisingEdge, Timer
from cocotb_tools.runner import get_runner

from holoso import FDivsqrtOptions, FloatFormat, FloatValue
from holoso._operators import DivsqrtMode, FDivsqrtOperator

from .hdl_float_oracle import (
    DIRECTED_F32,
    F32_EXP_MASK,
    HDL_DIR,
    PipelineScoreboard,
    REPO_ROOT,
    SGNOP_OPS,
    SIMULATORS,
    apply_sgnop,
    build_args,
    div_oracle_bits,
    drive_reset,
    f32_to_bits,
    get_random_count,
    get_seed,
    random_zkf_f32,
    sources,
    sqrt_oracle,
    stage_tag,
    start_clock,
)

_FMT = FloatFormat(8, 24)

# Base + each knob alone + the staged fixture's combination (tests/_modelref.py) + the synthesis matrix's (the input
# stage with the decode stage) + a multi-stage input.
STAGE_COMBOS: tuple[dict[str, int], ...] = (
    {},
    {"stage_input": 1},
    {"stage_decode": 1},
    {"stage_pack": 1},
    {"stage_output": 1},
    {"stage_input": 1, "stage_pack": 1, "stage_output": 1},
    {"stage_input": 1, "stage_decode": 1},
    {"stage_input": 2, "stage_decode": 1, "stage_pack": 1, "stage_output": 1},
)
# A core fixed to one operation sheds the other's datapath, so each is checked lean and fully staged.
FIXED_STAGE_COMBOS: tuple[dict[str, int], ...] = (
    {},
    {"stage_input": 2, "stage_decode": 1, "stage_pack": 1, "stage_output": 1},
)


def _division(a: int, b: int) -> tuple[int, int]:
    """
    Reference `(y_bits, error)`. Where float32 names no ZKF value -- a zero divisor, inf/inf -- the core still defines
    its answer, which the value model states.
    """
    y = div_oracle_bits(a, b)
    if y is None:
        y = (FloatValue.from_bits(_FMT, a) / FloatValue.from_bits(_FMT, b)).bits
    return y, int((b & F32_EXP_MASK) == 0)


@cocotb.test()
async def holoso_fdivsqrt_cocotb(dut: Any) -> None:
    fixed = os.environ["HOLOSO_FIXED_MODE"]
    operations = [bool(DivsqrtMode[fixed])] if fixed else [False, True]
    await start_clock(dut)
    await drive_reset(dut)

    sb = PipelineScoreboard(dut, [("y", "y"), ("error", "error")], latency=int(os.environ["HOLOSO_EXPECTED_LATENCY"]))
    rng = np.random.default_rng(get_seed())

    async def step_idle() -> None:
        dut.in_valid.value = 0
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        sb.sample()

    async def step(sqrt: bool, a: int, b: int, a_op: int, b_op: int) -> None:
        a_eff = apply_sgnop(a, a_op)
        y, error = sqrt_oracle(a_eff) if sqrt else _division(a_eff, apply_sgnop(b, b_op))
        dut.a.value = a
        dut.b.value = b
        dut.a_sgnop.value = a_op
        dut.b_sgnop.value = b_op
        dut.op_sqrt.value = int(rng.integers(0, 2)) if fixed else int(sqrt)
        dut.in_valid.value = 1
        sb.push({"y": y, "error": error, "_desc": f"sqrt={sqrt} a=0x{a:08x} b=0x{b:08x} ops={a_op}{b_op}"})
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        sb.sample()

    def noise() -> int:
        return random_zkf_f32(rng) if rng.random() < 0.8 else 0

    for sqrt in operations:
        for a in DIRECTED_F32:
            for b in [noise()] if sqrt else DIRECTED_F32:
                await step(sqrt, a, b, 0, 0)
    await sb.drain()

    # Sign controls against the operands that raise the error: a zero divisor in either sign form, a negative radicand.
    sample = [
        (DIRECTED_F32[int(rng.integers(0, len(DIRECTED_F32)))], DIRECTED_F32[int(rng.integers(0, len(DIRECTED_F32)))])
        for _ in range(6)
    ]
    sample += [
        (f32_to_bits(1.0), 0),
        (f32_to_bits(-1.0), 0),
        (0, f32_to_bits(2.0)),
        (f32_to_bits(-3.0), f32_to_bits(2.0)),
    ]
    for sqrt in operations:
        for a_op in SGNOP_OPS:
            for b_op in SGNOP_OPS:
                for a, b in sample:
                    await step(sqrt, a, b, a_op, b_op)
    await sb.drain()

    for _ in range(get_random_count()):
        if rng.random() < 0.2:
            await step_idle()
            continue
        sqrt = operations[int(rng.integers(0, len(operations)))]
        b = noise() if sqrt else random_zkf_f32(rng)
        await step(sqrt, random_zkf_f32(rng), b, int(rng.integers(0, 4)), int(rng.integers(0, 4)))
    await sb.drain()

    await drive_reset(dut)
    for _ in range(8):
        dut.in_valid.value = 0
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        assert int(dut.out_valid.value) == 0


@pytest.mark.parametrize("stages", STAGE_COMBOS, ids=stage_tag)
@pytest.mark.parametrize("sim", SIMULATORS)
def test_holoso_fdivsqrt(sim: str, stages: dict[str, int]) -> None:
    _run(sim, stages, None)


@pytest.mark.parametrize("stages", FIXED_STAGE_COMBOS, ids=stage_tag)
@pytest.mark.parametrize("mode", DivsqrtMode, ids=lambda mode: mode.name.lower())
@pytest.mark.parametrize("sim", SIMULATORS)
def test_holoso_fdivsqrt_fixed_to_one_operation(sim: str, mode: DivsqrtMode, stages: dict[str, int]) -> None:
    _run(sim, stages, mode)


def _run(sim: str, stages: dict[str, int], fixed: DivsqrtMode | None) -> None:
    """`fixed` elaborates the core for that operation alone; `None` keeps it choosing per transaction."""
    operator = FDivsqrtOperator.build(_FMT, FDivsqrtOptions(**stages))
    modes = frozenset(operator.modes if fixed is None else [operator.mode_of_divsqrt(fixed)])
    label = "both" if fixed is None else fixed.name.lower()
    runner = get_runner(sim)
    build_dir = REPO_ROOT / "build" / "cocotb" / sim / f"fdivsqrt_{label}_{stage_tag(stages)}"
    runner.build(
        sources=sources(),
        includes=[HDL_DIR],
        hdl_toplevel=operator.module_name,
        parameters=operator.params_for(modes),
        build_args=build_args(sim),
        build_dir=build_dir,
        clean=True,
        timescale=("1ns", "1ps"),
    )
    runner.test(
        hdl_toplevel=operator.module_name,
        test_module="tests.hdl.test_fdivsqrt",
        test_dir=REPO_ROOT,
        build_dir=build_dir,
        extra_env={
            "HOLOSO_EXPECTED_LATENCY": str(operator.latencies[0]),
            "HOLOSO_FIXED_MODE": "" if fixed is None else fixed.name,
        },
        results_xml=str(build_dir / "results.xml"),
    )
