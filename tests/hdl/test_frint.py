"""
Tests for holoso_frint (pipelined; sgnop on a, per-firing round_mode port), which answers one rounding twice at once:
`y_float` as a float of the same format and `y_int` as a saturating signed integer of an independent width. Both are
checked on every transaction.
"""

import os
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import cocotb
import numpy as np
import pytest
from cocotb.triggers import RisingEdge, Timer
from cocotb_tools.runner import get_runner
from zkf import Zkf, ZkfFormat

from holoso import FRintOptions, FloatFormat, IntFormat
from holoso._operators import FRintOperator

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
    round_oracle_bits,
    sources,
    start_clock,
)


@dataclass(frozen=True, slots=True)
class _Config:
    wexp: int
    wman: int
    wint: int
    options: FRintOptions

    @property
    def operator(self) -> FRintOperator:
        """The module name, its RTL parameters and its latency all come from the operator, so a drift fails here."""
        return FRintOperator.build(FloatFormat(self.wexp, self.wman), IntFormat(self.wint), self.options)

    @property
    def label(self) -> str:
        o = self.options
        stages = f"i{o.stage_input}s{o.stage_shift}r{o.stage_round}o{o.stage_output}"
        return f"e{self.wexp}m{self.wman}_i{self.wint}_{stages}"


# The public default (the shift stage alone), each other knob alone, all on, and the formats whose integer is wider
# and narrower than the float. At least one stage must be enabled: the zkf_rint core is combinational without any.
_CONFIGS = (
    _Config(8, 24, 32, FRintOptions()),
    _Config(8, 24, 32, FRintOptions(stage_input=1, stage_shift=0)),
    _Config(8, 24, 32, FRintOptions(stage_shift=0, stage_round=1)),
    _Config(8, 24, 32, FRintOptions(stage_shift=0, stage_output=1)),
    _Config(8, 24, 32, FRintOptions(stage_input=1, stage_round=1, stage_output=1)),
    _Config(6, 18, 44, FRintOptions()),
    _Config(6, 18, 44, FRintOptions(stage_input=2, stage_output=1)),
    _Config(8, 36, 24, FRintOptions()),
    _Config(8, 36, 24, FRintOptions(stage_round=1, stage_output=1)),
)

# Ties (x.5) exercise nearest-even in both parities, and |x| < 1 the sub-one branch (floor(-0.3) = -1,
# ceil(-0.3) = +0) which the integer-boundary values never reach.
_FRACTIONS = tuple(
    Fraction(v)
    for v in (0.3, -0.3, 0.7, -0.7, 0.25, -0.25, 0.5, -0.5, 1.5, -1.5, 2.5, -2.5, 3.5, -3.5, 8388607.5, -8388607.5)
)


def _directed_values(fmt: ZkfFormat, wint: int) -> list[int]:
    int_min = -(1 << (wint - 1))
    int_max = (1 << (wint - 1)) - 1
    fractions = (
        *_FRACTIONS,
        Fraction(0),
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


_FLOAT_ORACLE = (Zkf.round, Zkf.floor, Zkf.ceil, Zkf.trunc)
_INT_ORACLE = (Zkf.round_int, Zkf.floor_int, Zkf.ceil_int, Zkf.trunc_int)


def _oracle(fmt: ZkfFormat, a: int, a_sgnop: int, round_mode: int, wint: int) -> tuple[int, int]:
    """
    Reference `(y_float, y_int)` bits from the ZKF value model, the float one cross-checked against numpy -- an
    INDEPENDENT reference -- where the format is float32's.
    """
    conditioned = apply_sgnop(a, a_sgnop, fmt.wfull)
    value = fmt.wrap(conditioned)
    y_float = _FLOAT_ORACLE[round_mode](value).bits
    if (fmt.wexp, fmt.wman) == (8, 24):
        assert y_float == round_oracle_bits(conditioned, round_mode), f"the oracles disagree on 0x{conditioned:08x}"
    return y_float, _INT_ORACLE[round_mode](value, wint) & ((1 << wint) - 1)


@cocotb.test()
async def holoso_frint_cocotb(dut: Any) -> None:
    wexp = int(os.environ["HOLOSO_WEXP"])
    wman = int(os.environ["HOLOSO_WMAN"])
    wint = int(os.environ["HOLOSO_WINT"])
    latency = int(os.environ["HOLOSO_EXPECTED_LATENCY"])
    wfull = wexp + wman
    fmt = ZkfFormat(wexp, wman)
    assert len(dut.y_float) == wfull and len(dut.y_int) == wint
    mode_port = FRintOperator.mode_port
    assert mode_port is not None
    assert len(getattr(dut, mode_port.name)) == mode_port.width
    await start_clock(dut)
    await drive_reset(dut)

    sb = PipelineScoreboard(dut, [("y_float", "y_float"), ("y_int", "y_int")], latency=latency)
    rng = np.random.default_rng(get_seed())

    async def step_idle() -> None:
        dut.in_valid.value = 0
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        sb.sample()

    async def step(a: int, a_sgnop: int, round_mode: int) -> None:
        y_float, y_int = _oracle(fmt, a, a_sgnop, round_mode, wint)
        dut.a.value = a
        dut.a_sgnop.value = a_sgnop
        dut.round_mode.value = round_mode
        dut.in_valid.value = 1
        desc = f"a=0x{a:x} a_sgnop={a_sgnop} round_mode={round_mode}"
        sb.push({"y_float": y_float, "y_int": y_int, "_desc": desc})
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        sb.sample()

    values = _directed_values(fmt, wint)
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

    # A transaction in flight is dropped by reset.
    dut.a.value = fmt.encode(1).bits
    dut.a_sgnop.value = 0
    dut.round_mode.value = 3
    dut.in_valid.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")

    await drive_reset(dut)
    for _ in range(latency + 2):
        dut.in_valid.value = 0
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        assert int(dut.out_valid.value) == 0


@pytest.mark.parametrize("config", _CONFIGS, ids=lambda config: config.label)
@pytest.mark.parametrize("sim", SIMULATORS)
def test_holoso_frint(sim: str, config: _Config) -> None:
    operator = config.operator
    runner = get_runner(sim)
    build_dir = REPO_ROOT / "build" / "cocotb" / sim / f"frint_{config.label}"
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
        test_module="tests.hdl.test_frint",
        test_dir=REPO_ROOT,
        build_dir=build_dir,
        extra_env={
            "HOLOSO_WEXP": str(config.wexp),
            "HOLOSO_WMAN": str(config.wman),
            "HOLOSO_WINT": str(config.wint),
            "HOLOSO_EXPECTED_LATENCY": str(operator.latencies[0]),
        },
        results_xml=str(build_dir / "results.xml"),
    )
