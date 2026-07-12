"""Direct HDL tests for the independent-width runtime ZKF power-of-two scaler."""

import os
from dataclasses import dataclass
from typing import Any

import cocotb
import numpy as np
import pytest
from cocotb.triggers import RisingEdge, Timer
from cocotb_tools.runner import get_runner
from zkf import MulIlog2Model, ZkfFormat

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
    stage_decode: int = 0

    @property
    def model(self) -> MulIlog2Model:
        return MulIlog2Model(
            ZkfFormat(self.wexp, self.wman),
            wk=self.wint,
            stage_input=self.stage_input,
            stage_decode=self.stage_decode,
        )

    @property
    def label(self) -> str:
        return f"e{self.wexp}m{self.wman}_k{self.wint}_i{self.stage_input}d{self.stage_decode}"


_CONFIGS = (
    _Config(6, 18, 44),
    _Config(6, 18, 44, stage_input=1),
    _Config(6, 18, 44, stage_decode=1),
    _Config(6, 18, 44, stage_input=1, stage_decode=1),
    _Config(8, 36, 24),
    _Config(5, 8, 3),
)

_DEFAULT_CONFIG = _Config(6, 18, 24, stage_input=1)


def _directed_values(fmt: ZkfFormat) -> list[int]:
    return [
        fmt.zero().bits,
        fmt.inf(0).bits,
        fmt.inf(1).bits,
        fmt.encode(-fmt.max).bits,
        fmt.encode(-fmt.lowest).bits,
        fmt.encode(-1).bits,
        fmt.encode(1).bits,
        fmt.encode(fmt.lowest).bits,
        fmt.encode(fmt.max).bits,
    ]


def _shift_values(fmt: ZkfFormat, wint: int) -> list[int]:
    int_min = -(1 << (wint - 1))
    int_max = (1 << (wint - 1)) - 1
    if wint <= 3:
        return list(range(int_min, int_max + 1))
    values = {
        int_min,
        -fmt.exp_inf,
        -fmt.bias,
        -2,
        -1,
        0,
        1,
        2,
        fmt.bias,
        fmt.exp_inf,
        int_max,
    }
    return sorted(value for value in values if int_min <= value <= int_max)


@cocotb.test()
async def holoso_fmul_ilog2_cocotb(dut: Any) -> None:
    wexp = int(os.environ["HOLOSO_WEXP"])
    wman = int(os.environ["HOLOSO_WMAN"])
    wint = int(os.environ["HOLOSO_WINT"])
    latency = int(os.environ["HOLOSO_EXPECTED_LATENCY"])
    wfull = wexp + wman
    fmt = ZkfFormat(wexp, wman)
    k_mask = (1 << wint) - 1

    assert len(dut.y) == wfull
    assert len(dut.k) == wint
    await start_clock(dut)
    await drive_reset(dut)

    sb = PipelineScoreboard(dut, [("y", "y")], latency=latency)
    rng = np.random.default_rng(get_seed())

    async def step_idle() -> None:
        dut.in_valid.value = 0
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        sb.sample()

    async def step(a: int, k: int, a_sgnop: int, y_sgnop: int) -> None:
        a_eff = apply_sgnop(a, a_sgnop, wfull)
        expected = apply_sgnop(fmt.wrap(a_eff).mul_ilog2(k).bits, y_sgnop, wfull)
        dut.a.value = a
        dut.k.value = k & k_mask
        dut.a_sgnop.value = a_sgnop
        dut.y_sgnop.value = y_sgnop
        dut.in_valid.value = 1
        sb.push({"y": expected, "_desc": f"a=0x{a:x} k={k} ops=a{a_sgnop}y{y_sgnop}"})
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        sb.sample()

    values = _directed_values(fmt)
    shifts = _shift_values(fmt, wint)
    for a in values:
        for k in shifts:
            await step(a, k, 0, 0)
    await sb.drain()

    for index, a_sgnop in enumerate(SGNOP_OPS):
        for y_sgnop in SGNOP_OPS:
            for offset, a in enumerate(values):
                await step(a, shifts[(index + offset + y_sgnop) % len(shifts)], a_sgnop, y_sgnop)
    await sb.drain()

    int_min = -(1 << (wint - 1))
    int_max = (1 << (wint - 1)) - 1
    for _ in range(get_random_count()):
        if rng.random() < 0.2:
            await step_idle()
        else:
            a = fmt.wrap(int(rng.integers(0, 1 << wfull, dtype=np.uint64))).canonicalize().bits
            await step(
                a,
                int(rng.integers(int_min, int_max + 1)),
                int(rng.integers(0, 4)),
                int(rng.integers(0, 4)),
            )
    await sb.drain()

    if latency > 1:
        dut.a.value = fmt.encode(1).bits
        dut.k.value = 0
        dut.a_sgnop.value = 0
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
def test_holoso_fmul_ilog2(sim: str, config: _Config) -> None:
    model = config.model
    parameters = {name: value for name, value in model.params.items() if name != "WK"}
    parameters["WINT"] = config.wint
    _run(sim, config, model, parameters, config.label)


@pytest.mark.parametrize("sim", SIMULATORS)
def test_holoso_fmul_ilog2_defaults(sim: str) -> None:
    config = _DEFAULT_CONFIG
    model = config.model
    parameters = {name: value for name, value in model.params.items() if name not in {"WK", "LATENCY"}}
    _run(sim, config, model, parameters, "e6m18_defaults_i1d0")


def _run(sim: str, config: _Config, model: MulIlog2Model, parameters: dict[str, int], label: str) -> None:
    runner = get_runner(sim)
    build_dir = REPO_ROOT / "build" / "cocotb" / sim / f"fmul_ilog2_{label}"
    runner.build(
        sources=sources(),
        includes=[HDL_DIR],
        hdl_toplevel="holoso_fmul_ilog2",
        parameters=parameters,
        build_args=build_args(sim),
        build_dir=build_dir,
        clean=True,
        timescale=("1ns", "1ps"),
    )
    runner.test(
        hdl_toplevel="holoso_fmul_ilog2",
        test_module="tests.hdl.test_fmul_ilog2",
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
