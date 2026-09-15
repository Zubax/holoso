"""Direct HDL tests for the ZKF exponent-extraction wrapper."""

import os
from dataclasses import dataclass
from typing import Any

import cocotb
import numpy as np
import pytest
from cocotb.triggers import RisingEdge, Timer
from cocotb_tools.runner import get_runner
from zkf import ZkfFormat

from holoso import FloatFormat, FloatValue, IntFormat
from holoso._operators import FILog2Operator

from .hdl_float_oracle import (
    HDL_DIR,
    PipelineScoreboard,
    REPO_ROOT,
    SIMULATORS,
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
    def operator(self) -> FILog2Operator:
        """The module name, its RTL parameters and its latency all come from the operator, so a drift fails here."""
        return FILog2Operator(
            FloatFormat(self.wexp, self.wman),
            IntFormat(self.wint),
            FILog2Operator.Options(stage_input=self.stage_input),
        )

    @property
    def label(self) -> str:
        return f"e{self.wexp}m{self.wman}_i{self.wint}_s{self.stage_input}"


_CONFIGS = (_Config(6, 18, 44), _Config(6, 18, 44, 1), _Config(8, 36, 24), _Config(8, 36, 24, 1))


def _every_exponent(fmt: ZkfFormat) -> list[int]:
    """The answer is a field slice, so every exponent is a case and no fraction or sign may disturb it."""
    return [
        (sign << fmt.sign_shift) | (exponent << fmt.wfrac) | fraction
        for exponent in range(1 << fmt.wexp)
        for fraction in (0, 1, fmt.frac_mask)
        for sign in (0, 1)
    ]


@cocotb.test()
async def holoso_filog2_cocotb(dut: Any) -> None:
    wexp = int(os.environ["HOLOSO_WEXP"])
    wman = int(os.environ["HOLOSO_WMAN"])
    wint = int(os.environ["HOLOSO_WINT"])
    latency = int(os.environ["HOLOSO_EXPECTED_LATENCY"])
    wfull = wexp + wman
    fmt = ZkfFormat(wexp, wman)
    ffmt = FloatFormat(wexp, wman)
    operator = FILog2Operator(
        ffmt, IntFormat(wint), FILog2Operator.Options(stage_input=int(os.environ["HOLOSO_STAGE_INPUT"]))
    )
    assert operator.latency == latency, "the oracle must be the configuration the DUT was built from"
    assert len(dut.y) == wint
    await start_clock(dut)
    await drive_reset(dut)

    sb = PipelineScoreboard(dut, [("y", "y")], latency=latency)
    rng = np.random.default_rng(get_seed())

    def oracle(a: int) -> int:
        """Taken through the operator's own reference, so the RTL and the model cannot drift apart."""
        (value,) = operator.evaluate(FloatValue.from_bits(ffmt, a))
        return int(value) & ((1 << wint) - 1)

    async def step_idle() -> None:
        dut.in_valid.value = 0
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        sb.sample()

    async def step(a: int) -> None:
        dut.a.value = a
        dut.in_valid.value = 1
        sb.push({"y": oracle(a), "_desc": f"a=0x{a:x}"})
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        sb.sample()

    # The wrapper binds no sign port, the operand being declared unconditioned. Sign coverage needs no conditioner
    # sweep: `_every_exponent` enumerates BOTH signs of every exponent and fraction while the reference ignores the
    # sign, so this single pass already asserts that a magnitude and its negation answer alike.
    for a in _every_exponent(fmt):
        await step(a)
    await sb.drain()

    for _ in range(get_random_count()):
        if rng.random() < 0.2:
            await step_idle()
        else:
            a = fmt.wrap(int(rng.integers(0, 1 << wfull, dtype=np.uint64))).canonicalize().bits
            await step(a)
    await sb.drain()

    await drive_reset(dut)
    for _ in range(latency + 2):
        dut.in_valid.value = 0
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        assert int(dut.out_valid.value) == 0


@pytest.mark.parametrize("config", _CONFIGS, ids=lambda config: config.label)
@pytest.mark.parametrize("sim", SIMULATORS)
def test_holoso_filog2(sim: str, config: _Config) -> None:
    operator = config.operator
    runner = get_runner(sim)
    build_dir = REPO_ROOT / "build" / "cocotb" / sim / f"filog2_{config.label}"
    runner.build(
        sources=sources(),
        includes=[HDL_DIR],
        hdl_toplevel=operator.module_name,
        parameters=operator.params,
        build_args=build_args(sim),
        build_dir=build_dir,
        clean=True,
        timescale=("1ns", "1ps"),
    )
    runner.test(
        hdl_toplevel=operator.module_name,
        test_module="tests.hdl.test_filog2",
        test_dir=REPO_ROOT,
        build_dir=build_dir,
        extra_env={
            "HOLOSO_WEXP": str(config.wexp),
            "HOLOSO_WMAN": str(config.wman),
            "HOLOSO_WINT": str(config.wint),
            "HOLOSO_STAGE_INPUT": str(config.stage_input),
            "HOLOSO_EXPECTED_LATENCY": str(operator.latency),
        },
        results_xml=str(build_dir / "results.xml"),
    )
