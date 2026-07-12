"""Direct HDL tests for the independent-width signed-integer-to-ZKF-float wrapper."""

import os
from dataclasses import dataclass
from typing import Any

import cocotb
import numpy as np
import pytest
from cocotb.triggers import RisingEdge, Timer
from cocotb_tools.runner import get_runner
from zkf import FromIntModel, ZkfFormat

from .hdl_float_oracle import (
    HDL_DIR,
    PipelineScoreboard,
    REPO_ROOT,
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
    stage_normalize: int = 0
    stage_pack: int = 0
    stage_output: int = 0
    default_latency: bool = False

    @property
    def model(self) -> FromIntModel:
        return FromIntModel(
            ZkfFormat(self.wexp, self.wman),
            wint=self.wint,
            stage_input=self.stage_input,
            stage_normalize=self.stage_normalize,
            stage_pack=self.stage_pack,
            stage_output=self.stage_output,
        )

    @property
    def label(self) -> str:
        return (
            f"e{self.wexp}m{self.wman}_i{self.wint}_"
            f"i{self.stage_input}n{self.stage_normalize}p{self.stage_pack}o{self.stage_output}"
            f"_default{int(self.default_latency)}"
        )


_CONFIGS = (
    _Config(6, 18, 44),
    _Config(6, 18, 44, stage_input=1, default_latency=True),
    _Config(6, 18, 44, stage_normalize=1),
    _Config(6, 18, 44, stage_pack=1),
    _Config(6, 18, 44, stage_output=1),
    _Config(6, 18, 44, stage_input=1, stage_normalize=1, stage_pack=1, stage_output=1),
    _Config(8, 36, 24),
    _Config(3, 4, 12),
)


def _signed_values(wman: int, wint: int) -> list[int]:
    int_min = -(1 << (wint - 1))
    int_max = (1 << (wint - 1)) - 1
    values = {int_min, int_min + 1, -1, 0, 1, int_max - 1, int_max}
    for shift in range(wint - 1):
        values.update((-(1 << shift), 1 << shift))
    for boundary in (1 << (wman - 1), 1 << wman, 1 << (wman + 1)):
        for delta in (-3, -1, 0, 1, 3):
            values.update((boundary + delta, -boundary - delta))
    return sorted(value for value in values if int_min <= value <= int_max)


@cocotb.test()
async def holoso_ffromint_cocotb(dut: Any) -> None:
    wexp = int(os.environ["HOLOSO_WEXP"])
    wman = int(os.environ["HOLOSO_WMAN"])
    wint = int(os.environ["HOLOSO_WINT"])
    latency = int(os.environ["HOLOSO_EXPECTED_LATENCY"])
    wfull = wexp + wman
    fmt = ZkfFormat(wexp, wman)
    int_mask = (1 << wint) - 1

    assert len(dut.y) == wfull
    await start_clock(dut)
    await drive_reset(dut)

    sb = PipelineScoreboard(dut, [("y", "y")], latency=latency)
    rng = np.random.default_rng(get_seed())

    async def step_idle() -> None:
        dut.in_valid.value = 0
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        sb.sample()

    async def step(a: int, y_sgnop: int) -> None:
        expected = apply_sgnop(fmt.from_int(wint, a).bits, y_sgnop, wfull)
        dut.a.value = a & int_mask
        dut.y_sgnop.value = y_sgnop
        dut.in_valid.value = 1
        sb.push({"y": expected, "_desc": f"a={a} y_sgnop={y_sgnop}"})
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        sb.sample()

    values = _signed_values(wman, wint)
    for a in values:
        await step(a, 0)
    await sb.drain()

    for y_sgnop in SGNOP_OPS:
        for a in values:
            await step(a, y_sgnop)
    await sb.drain()

    for _ in range(get_random_count()):
        if rng.random() < 0.2:
            await step_idle()
        else:
            await step(int(rng.integers(-(1 << (wint - 1)), 1 << (wint - 1))), int(rng.integers(0, 4)))
    await sb.drain()

    if latency > 1:
        dut.a.value = 1
        dut.y_sgnop.value = 0
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
def test_holoso_ffromint(sim: str, config: _Config) -> None:
    model = config.model
    parameters = dict(model.params)
    if config.default_latency:
        del parameters["LATENCY"]
    runner = get_runner(sim)
    build_dir = REPO_ROOT / "build" / "cocotb" / sim / f"ffromint_{config.label}"
    runner.build(
        sources=sources(),
        includes=[HDL_DIR],
        hdl_toplevel="holoso_ffromint",
        parameters=parameters,
        build_args=build_args(sim),
        build_dir=build_dir,
        clean=True,
        timescale=("1ns", "1ps"),
    )
    runner.test(
        hdl_toplevel="holoso_ffromint",
        test_module="tests.hdl.test_ffromint",
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
