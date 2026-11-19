"""
Shared scaffolding for the simulator-driven tests: the build helpers (simulator selection and flags), the binary32
stimulus the inline-function benches drive (a directed corner battery and a ZKF-legal random sampler) with the
reference of the sign conditioner, and the cocotb clock and reset.

The stimulus mirrors numpy.float32 (IEEE 754 binary32), so a bench driving it is configured at WEXP=8, WMAN=24. NaN,
subnormals and the negative zero are excluded because ZKF does not define them.
"""

import os
from pathlib import Path
from typing import Any

import holoso
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge

REPO_ROOT = Path(__file__).resolve().parents[2]
HDL_DIR = Path(holoso.__file__).resolve().parent / "_backend" / "verilog"
"""Where `holoso_support_inline.vh` lives, which the inline-function harness includes."""

SIMULATORS = (os.environ["SIM"],) if "SIM" in os.environ else ("icarus", "verilator")

VERILATOR_BUILD_ARGS = [
    "--timing",
    "-Wno-fatal",
    "-Wno-DECLFILENAME",
    "-Wno-UNUSEDSIGNAL",
    "-Wno-WIDTH",
    "-Wno-CMPCONST",
]


def build_args(sim: str) -> list[str]:
    """Icarus reads the generated Verilog as Verilog-2005, as README asks (its SystemVerilog mode misses time zero)."""
    return VERILATOR_BUILD_ARGS if sim == "verilator" else ["-g2005"]


SGNOP_OPS: tuple[int, ...] = (0, 1, 2, 3)
"""The sign-conditioning codes of `holoso_fsgnop`: bit 0 negates, bit 1 takes the magnitude first."""


def apply_sgnop(bits: int, op: int, wfull: int = 32) -> int:
    """Mirror of holoso_fsgnop: s_out = (s_in & ~op[1]) ^ op[0]; body preserved."""
    sign_mask = 1 << (wfull - 1)
    body_mask = sign_mask - 1
    op_abs = (op >> 1) & 1
    op_neg = op & 1
    s_in = (bits >> (wfull - 1)) & 1
    s_out = (s_in & (1 - op_abs)) ^ op_neg
    return (s_out << (wfull - 1)) | (bits & body_mask)


F32_SIGN_MASK = 0x80000000
F32_EXP_MASK = 0x7F800000
F32_FRAC_MASK = 0x007FFFFF
F32_PINF = 0x7F800000
F32_NINF = 0xFF800000
F32_MAX_FIN = 0x7F7FFFFF  # largest finite +
F32_MIN_FIN = 0xFF7FFFFF  # largest finite - (most negative)


def f32_to_bits(x: float | np.floating[Any]) -> int:
    return int(np.float32(x).view(np.uint32))


def is_zkf_legal_f32(bits: int) -> bool:
    """ZKF excludes NaN, subnormals, and negative zero."""
    bits &= 0xFFFFFFFF
    exponent, fraction = bits & F32_EXP_MASK, bits & F32_FRAC_MASK
    nan = exponent == F32_EXP_MASK and fraction != 0
    subnormal = exponent == 0 and fraction != 0
    return not (nan or subnormal or bits == F32_SIGN_MASK)


def random_zkf_f32(rng: np.random.Generator) -> int:
    """Uniformly-random 32-bit pattern, rejecting ZKF-illegal classes."""
    while True:
        bits = int(rng.integers(0, 1 << 32, dtype=np.uint64))
        if is_zkf_legal_f32(bits):
            return bits


DIRECTED_F32: tuple[int, ...] = (
    0,  # +0
    f32_to_bits(1.0),
    f32_to_bits(-1.0),
    f32_to_bits(0.5),
    f32_to_bits(-0.5),
    f32_to_bits(2.0),
    f32_to_bits(-2.0),
    f32_to_bits(np.float32(np.pi)),
    f32_to_bits(np.float32(-np.pi)),
    f32_to_bits(np.float32(1e-30)),
    f32_to_bits(np.float32(-1e-30)),
    f32_to_bits(np.finfo(np.float32).tiny),  # smallest normal +
    f32_to_bits(-np.finfo(np.float32).tiny),  # smallest normal -
    F32_MAX_FIN,
    F32_MIN_FIN,
    F32_PINF,
    F32_NINF,
)


def get_seed(default: int = 0x9E3779B97F4A7C15) -> int:
    return int(os.environ.get("HOLOSO_TEST_SEED", hex(default)), 0)


def get_random_count(default: int = 256) -> int:
    return int(os.environ.get("HOLOSO_TEST_RANDOM_COUNT", str(default)))


async def start_clock(dut: Any, period_ns: int = 10) -> None:
    Clock(dut.clk, period_ns, unit="ns").start()
    await FallingEdge(dut.clk)


async def drive_reset(dut: Any, cycles: int = 4) -> None:
    dut.rst.value = 1
    if hasattr(dut, "in_valid"):
        dut.in_valid.value = 0
    for _ in range(cycles):
        await RisingEdge(dut.clk)
    dut.rst.value = 0
    await FallingEdge(dut.clk)
