"""Direct HDL tests for the independent-width saturating ZKF-float-to-integer wrapper."""

import os
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import cocotb
import numpy as np
import pytest
from cocotb.triggers import RisingEdge, Timer
from cocotb_tools.runner import get_runner
from zkf import ToIntModel, ZkfFormat

from .hdl_float_oracle import (
    HDL_DIR,
    PipelineScoreboard,
    REPO_ROOT,
    ROUND_MODES,
    SGNOP_OPS,
    SIMULATORS,
    apply_sgnop,
    build_args,
    drive_reset,
    get_random_count,
    get_seed,
    sources,
    start_clock,
)


@dataclass(frozen=True, slots=True)
class _Config:
    wexp: int
    wman: int
    wint: int
    stage_input: int = 0

    @property
    def model(self) -> ToIntModel:
        return ToIntModel(ZkfFormat(self.wexp, self.wman), wint=self.wint, stage_input=self.stage_input)

    @property
    def label(self) -> str:
        return f"e{self.wexp}m{self.wman}_i{self.wint}_s{self.stage_input}"


_CONFIGS = (_Config(6, 18, 44), _Config(6, 18, 44, 1), _Config(8, 36, 24), _Config(8, 36, 24, 1))


def _directed_values(fmt: ZkfFormat, wint: int) -> list[int]:
    int_min = -(1 << (wint - 1))
    int_max = (1 << (wint - 1)) - 1
    fractions = (
        Fraction(-5, 2),
        Fraction(-3, 2),
        Fraction(-1, 2),
        Fraction(-1, 4),
        Fraction(0),
        Fraction(1, 4),
        Fraction(1, 2),
        Fraction(3, 2),
        Fraction(5, 2),
        Fraction(int_min - 1),
        Fraction(int_min) - Fraction(1, 2),
        Fraction(int_min),
        Fraction(int_min + 1),
        Fraction(int_max - 1),
        Fraction(int_max),
        Fraction(int_max) + Fraction(1, 2),
        Fraction(int_max + 1),
        Fraction(int_max + 2),
        -fmt.max,
        -fmt.lowest,
        fmt.lowest,
        fmt.max,
    )
    values = {fmt.encode(value).bits for value in fractions}
    values.update((fmt.inf(0).bits, fmt.inf(1).bits))
    return sorted(values)


def _integer_oracle(fmt: ZkfFormat, a: int, a_sgnop: int, round_mode: int, wint: int) -> int:
    value = fmt.wrap(apply_sgnop(a, a_sgnop, fmt.wfull))
    if round_mode == 0:
        result = value.round_int(wint)
    elif round_mode == 1:
        result = value.floor_int(wint)
    elif round_mode == 2:
        result = value.ceil_int(wint)
    else:
        assert round_mode == 3
        result = value.trunc_int(wint)
    return result & ((1 << wint) - 1)


@cocotb.test()
async def holoso_ftoint_cocotb(dut: Any) -> None:
    wexp = int(os.environ["HOLOSO_WEXP"])
    wman = int(os.environ["HOLOSO_WMAN"])
    wint = int(os.environ["HOLOSO_WINT"])
    latency = int(os.environ["HOLOSO_EXPECTED_LATENCY"])
    wfull = wexp + wman
    fmt = ZkfFormat(wexp, wman)
    assert len(dut.y) == wint
    assert len(dut.round_mode) == 2
    await start_clock(dut)
    await drive_reset(dut)

    sb = PipelineScoreboard(dut, [("y", "y")], latency=latency)
    rng = np.random.default_rng(get_seed())

    async def step_idle() -> None:
        dut.in_valid.value = 0
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        sb.sample()

    async def step(a: int, a_sgnop: int, round_mode: int) -> None:
        expected = _integer_oracle(fmt, a, a_sgnop, round_mode, wint)
        dut.a.value = a
        dut.a_sgnop.value = a_sgnop
        dut.round_mode.value = round_mode
        dut.in_valid.value = 1
        sb.push({"y": expected, "_desc": f"a=0x{a:x} a_sgnop={a_sgnop} round_mode={round_mode}"})
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        sb.sample()

    values = _directed_values(fmt, wint)
    for a in values:
        for round_mode in ROUND_MODES:
            await step(a, 0, round_mode)
    await sb.drain()

    for a_sgnop in SGNOP_OPS:
        for round_mode in ROUND_MODES:
            for a in values:
                await step(a, a_sgnop, round_mode)
    await sb.drain()

    for _ in range(get_random_count()):
        if rng.random() < 0.2:
            await step_idle()
        else:
            a = fmt.wrap(int(rng.integers(0, 1 << wfull, dtype=np.uint64))).canonicalize().bits
            await step(a, int(rng.integers(0, 4)), int(rng.integers(0, 4)))
    await sb.drain()

    dut.a.value = fmt.encode(1).bits
    dut.a_sgnop.value = 0
    dut.round_mode.value = 3
    dut.in_valid.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")
    assert int(dut.out_valid.value) == 0
    sb.queue.clear()

    await drive_reset(dut)
    for _ in range(latency + 2):
        dut.in_valid.value = 0
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        assert int(dut.out_valid.value) == 0


@pytest.mark.parametrize("config", _CONFIGS, ids=lambda config: config.label)
@pytest.mark.parametrize("sim", SIMULATORS)
def test_holoso_ftoint(sim: str, config: _Config) -> None:
    model = config.model
    runner = get_runner(sim)
    build_dir = REPO_ROOT / "build" / "cocotb" / sim / f"ftoint_{config.label}"
    runner.build(
        sources=sources(),
        includes=[HDL_DIR],
        hdl_toplevel="holoso_ftoint",
        parameters=model.params,
        build_args=build_args(sim),
        build_dir=build_dir,
        clean=True,
        timescale=("1ns", "1ps"),
    )
    runner.test(
        hdl_toplevel="holoso_ftoint",
        test_module="tests.hdl.test_ftoint",
        test_dir=REPO_ROOT,
        build_dir=build_dir,
        extra_env={
            "HOLOSO_WEXP": str(config.wexp),
            "HOLOSO_WMAN": str(config.wman),
            "HOLOSO_WINT": str(config.wint),
            "HOLOSO_EXPECTED_LATENCY": str(model.latency),
        },
        results_xml=str(build_dir / "results.xml"),
    )
