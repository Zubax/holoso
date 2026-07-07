"""
Shared scaffolding for the holoso_f* HDL wrapper test suite.

Provides build helpers (source list, verilator flags, simulator selection); a bit-level oracle (binary32 <-> bits,
sgnop emulation, classification); a directed corner-case battery and a ZKF-legal random sampler; and cocotb
scaffolding (start_clock, drive_reset, PipelineScoreboard).

The oracle deliberately mirrors numpy.float32 (IEEE 754 binary32) so the DUT must be configured at WEXP=8, WMAN=24.
Subnormals, NaN, and -0 are excluded from stimulus because ZKF does not define them; arithmetic results that fall into
those classes are mapped through ZKF's zero/MIN_NORMAL boundary rule and canonical-zero rule.
"""

import math
import os
import tempfile
from collections import deque
from pathlib import Path
from typing import Any, Iterable

import cocotb
import holoso
import numpy as np
from holoso._backend.verilog._support import support_files
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge, Timer


def within(actual: float, expected: float, rtol: float, atol: float) -> bool:
    """Whether ``actual`` is within ``atol + rtol*|expected|`` of ``expected`` (infinities must match exactly)."""
    if math.isinf(expected) or math.isinf(actual) or math.isnan(expected) or math.isnan(actual):
        return actual == expected
    return abs(actual - expected) <= atol + rtol * abs(expected)


BENCH_DIR = Path(__file__).resolve().parent  # tests/hdl -- the cocotb test_dir for the benches and cosim driver
REPO_ROOT = BENCH_DIR.parents[1]
HDL_DIR = Path(holoso.__file__).resolve().parent / "_backend" / "verilog"
TESTS_DIR = REPO_ROOT / "tests"
SUPPORT_BUILD_DIR = REPO_ROOT / "build" / "holoso_support"  # where the assembled support library is materialized

SIMULATORS = (os.environ["SIM"],) if "SIM" in os.environ else ("icarus", "verilator")

VERILATOR_BUILD_ARGS = [
    "--timing",
    "-Wno-fatal",
    "-Wno-DECLFILENAME",
    "-Wno-UNUSEDSIGNAL",
    "-Wno-WIDTH",
    "-Wno-CMPCONST",
]


def sources() -> list[Path]:
    """
    The shared support library written out for the simulator: the single self-contained module file
    holoso_support.v, assembled in memory exactly as it ships, which suffices to elaborate any holoso_f* wrapper.
    The holoso_support_inline.vh helper functions are resolved separately from HDL_DIR via the include path (only the
    holoso_support_fn_tb harness includes them; generated DUTs splice them inline).
    """
    SUPPORT_BUILD_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name, content in support_files().items():
        path = SUPPORT_BUILD_DIR / name
        _atomic_write(path, content)
        paths.append(path)
    return paths


def _atomic_write(path: Path, content: str) -> None:
    """
    Write ``content`` to the shared support path atomically (write-temp + replace). Every holoso_f* bench rewrites
    this same file, and a simulator build may read it while the next bench is rewriting it; a plain truncating write
    is observed torn (iverilog fails to find the top module in a half-written file), so the swap must be atomic.
    """
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def build_args(sim: str) -> list[str]:
    return VERILATOR_BUILD_ARGS if sim == "verilator" else []


# Sign-conditioning opcodes -- must match holoso_fsgnop's op encoding in holoso_support.v / FloatSignControl.encoded.
SGNOP_NONE = 0
SGNOP_NEG = 1
SGNOP_ABS = 2
SGNOP_ABS_NEG = 3
SGNOP_OPS: tuple[int, ...] = (SGNOP_NONE, SGNOP_NEG, SGNOP_ABS, SGNOP_ABS_NEG)


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
F32_MIN_NORMAL = 0x00800000
F32_HALF_MIN_NORMAL_FRAC = 0x00400000


def bits_to_f32(bits: int) -> np.float32:
    return np.uint32(bits & 0xFFFFFFFF).view(np.float32)


def f32_to_bits(x: float | np.floating[Any]) -> int:
    return int(np.float32(x).view(np.uint32))


def is_nan_f32(bits: int) -> bool:
    return (bits & F32_EXP_MASK) == F32_EXP_MASK and (bits & F32_FRAC_MASK) != 0


def is_subnormal_f32(bits: int) -> bool:
    return (bits & F32_EXP_MASK) == 0 and (bits & F32_FRAC_MASK) != 0


def is_neg_zero_f32(bits: int) -> bool:
    return (bits & 0xFFFFFFFF) == F32_SIGN_MASK


def is_inf_f32(bits: int) -> bool:
    return (bits & 0x7FFFFFFF) == F32_PINF


def is_zero_f32(bits: int) -> bool:
    return (bits & 0x7FFFFFFF) == 0


def is_zkf_legal_f32(bits: int) -> bool:
    """ZKF excludes NaN, subnormals, and negative zero."""
    bits &= 0xFFFFFFFF
    return not (is_nan_f32(bits) or is_subnormal_f32(bits) or is_neg_zero_f32(bits))


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


def add_oracle_bits(a_bits: int, b_bits: int) -> int | None:
    """ZKF-compatible float32 add. Returns None if the result would be NaN."""
    y = bits_to_f32(a_bits) + bits_to_f32(b_bits)
    yb = f32_to_bits(y)
    if is_nan_f32(yb):
        return None
    return _flush_to_zkf(yb)


def sub_oracle_bits(a_bits: int, b_bits: int) -> int | None:
    y = bits_to_f32(a_bits) - bits_to_f32(b_bits)
    yb = f32_to_bits(y)
    if is_nan_f32(yb):
        return None
    return _flush_to_zkf(yb)


def mul_oracle_bits(a_bits: int, b_bits: int) -> int | None:
    y = bits_to_f32(a_bits) * bits_to_f32(b_bits)
    yb = f32_to_bits(y)
    if is_nan_f32(yb):
        return None
    return _flush_to_zkf(yb)


def div_oracle_bits(a_bits: int, b_bits: int) -> int | None:
    """
    ZKF-compatible float32 divide; returns None whenever the wrapper's y is unspecified.

    Division by +0 is a separately-signalled condition (the div0 flag); the wrapper contract leaves y unspecified
    there, so callers should skip the value check when b == +0 and verify div0 instead.
    """
    if is_zero_f32(b_bits) and is_zero_f32(a_bits):
        return None  # 0/0 -> NaN in float32, undefined in ZKF
    if is_inf_f32(a_bits) and is_inf_f32(b_bits):
        return None  # inf/inf -> NaN
    if is_zero_f32(b_bits):
        return None  # finite/0 -> inf in float32; div0 flag asserts and the value of y is unspecified
    with np.errstate(divide="ignore", invalid="ignore"):
        y = bits_to_f32(a_bits) / bits_to_f32(b_bits)
    yb = f32_to_bits(y)
    if is_nan_f32(yb):
        return None
    return _flush_to_zkf(yb)


_ZKF_F32 = holoso.FloatFormat(8, 24)


def fma_oracle_bits(a_bits: int, b_bits: int, c_bits: int) -> int:
    """
    Reference fused multiply-add a*b + c (single rounding) for ZKF-legal float32 inputs, via the exact
    ``FloatValue.fma``. The vendored zkf_fma RTL is the independent hardware anchor; this bench proves the two agree
    bit-for-bit. No None case: ZKF has no NaN, so fma of legal inputs is always a legal value.
    """
    a = holoso.FloatValue.from_bits(_ZKF_F32, a_bits)
    b = holoso.FloatValue.from_bits(_ZKF_F32, b_bits)
    c = holoso.FloatValue.from_bits(_ZKF_F32, c_bits)
    return holoso.FloatValue.fma(a, b, c).bits


def exp2_oracle_bits(a_bits: int) -> int:
    """
    Reference ``2**a`` via the exact ``FloatValue.exp2``; numpy is not usable as it is correctly rounded while zkf_exp2
    is faithfully rounded (within 1 ULP).
    """
    return holoso.FloatValue.from_bits(_ZKF_F32, a_bits).exp2().bits


def log2_oracle(a_bits: int) -> tuple[int, int, int]:
    """
    Reference ``(y_bits, domain_error, pole)``. The value is the exact ``FloatValue.log2`` (numpy is not usable: it
    rounds the true value, which the zkf table+polynomial core need not match to the last bit). The flags are an
    independent classification: ``pole`` when the operand is zero, ``domain_error`` when it is negative and nonzero.
    """
    y = holoso.FloatValue.from_bits(_ZKF_F32, a_bits).log2().bits
    pole = 1 if is_zero_f32(a_bits) else 0
    domain_error = 1 if ((a_bits & F32_SIGN_MASK) and not is_zero_f32(a_bits)) else 0
    return y, domain_error, pole


def sincos_oracle(a_bits: int) -> tuple[int, int]:
    """
    Reference turn-native ``(sin(2*pi*a), cos(2*pi*a))`` via the exact ``FloatValue.sincos``; numpy is unusable because
    the CORDIC is faithfully rounded, not correctly rounded.
    """
    s, c = holoso.FloatValue.from_bits(_ZKF_F32, a_bits).sincos()
    return s.bits, c.bits


def atan2_oracle(y_bits: int, x_bits: int) -> tuple[int, int]:
    """Reference turn-native ``(theta, magnitude)`` of ``atan2(y, x)`` via the exact ``FloatValue.atan2``."""
    th, mag = holoso.FloatValue.atan2(
        holoso.FloatValue.from_bits(_ZKF_F32, y_bits), holoso.FloatValue.from_bits(_ZKF_F32, x_bits)
    )
    return th.bits, mag.bits


def sort_oracle_bits(a_bits: int, b_bits: int) -> tuple[int, int]:
    """Return (min_bits, max_bits) for the float32 min/max of two ZKF-legal inputs."""
    a = bits_to_f32(a_bits)
    b = bits_to_f32(b_bits)
    mn = np.minimum(a, b)
    mx = np.maximum(a, b)
    return _flush_to_zkf(f32_to_bits(mn)), _flush_to_zkf(f32_to_bits(mx))


def cmp_oracle(a_bits: int, b_bits: int) -> tuple[int, int, int]:
    """Return (a_gt_b, a_eq_b, a_lt_b) one-hot flags for float32 comparison."""
    a = bits_to_f32(a_bits)
    b = bits_to_f32(b_bits)
    return int(a > b), int(a == b), int(a < b)


def mul_ilog2_oracle_bits(a_bits: int, k: int) -> int | None:
    """y = a * 2^k, exactly (when in-range). Returns None on NaN."""
    a = bits_to_f32(a_bits)
    y = np.float32(np.ldexp(float(a), k))
    yb = f32_to_bits(y)
    if is_nan_f32(yb):
        return None
    return _flush_to_zkf(yb)


# Round-mode opcodes -- must match zkf_round's round_mode encoding and FRoundOperator's immediate values.
ROUND_NEAREST_EVEN = 0
ROUND_FLOOR = 1
ROUND_CEIL = 2
ROUND_TRUNC = 3
ROUND_MODES: tuple[int, ...] = (ROUND_NEAREST_EVEN, ROUND_FLOOR, ROUND_CEIL, ROUND_TRUNC)


def round_oracle_bits(a_bits: int, mode: int) -> int | None:
    """
    Round a ZKF-legal float32 to an integral float per the zkf_round mode, using numpy as an INDEPENDENT reference
    (rint is round-half-to-even). Integral float32 results are exact; ``_flush_to_zkf`` canonicalizes a -0 result
    (e.g. ceil(-0.3)) to +0. Returns None on NaN (an inf input rounds to itself, never NaN, so None never occurs here).
    """
    a = bits_to_f32(a_bits)
    if mode == ROUND_NEAREST_EVEN:
        y = np.rint(a)
    elif mode == ROUND_FLOOR:
        y = np.floor(a)
    elif mode == ROUND_CEIL:
        y = np.ceil(a)
    else:
        y = np.trunc(a)
    yb = f32_to_bits(np.float32(y))
    if is_nan_f32(yb):
        return None
    return _flush_to_zkf(yb)


def _flush_to_zkf(bits: int) -> int:
    """
    Map a float32 result to a ZKF-legal bit pattern.

    ZKF has no subnormals and no negative zero. Tiny finite magnitudes below 0.5*MIN_NORMAL round to canonical +0;
    magnitudes at or above that boundary round to signed MIN_NORMAL.
    """
    bits &= 0xFFFFFFFF
    if is_neg_zero_f32(bits):
        return 0
    if is_subnormal_f32(bits):
        sign = bits & F32_SIGN_MASK
        frac = bits & F32_FRAC_MASK
        return sign | F32_MIN_NORMAL if frac >= F32_HALF_MIN_NORMAL_FRAC else 0
    return bits


def get_seed(default: int = 0x9E3779B97F4A7C15) -> int:
    return int(os.environ.get("HOLOSO_TEST_SEED", hex(default)), 0)


def get_random_count(default: int = 256) -> int:
    return int(os.environ.get("HOLOSO_TEST_RANDOM_COUNT", str(default)))


def stage_tag(stages: dict[str, int]) -> str:
    return "_".join(f"{k}{v}" for k, v in stages.items()) or "base"


async def start_clock(dut: Any, period_ns: int = 10) -> None:
    cocotb.start_soon(Clock(dut.clk, period_ns, unit="ns").start())
    await FallingEdge(dut.clk)


async def drive_reset(dut: Any, cycles: int = 4) -> None:
    dut.rst.value = 1
    if hasattr(dut, "in_valid"):
        dut.in_valid.value = 0
    for _ in range(cycles):
        await RisingEdge(dut.clk)
    dut.rst.value = 0
    await FallingEdge(dut.clk)


class PipelineScoreboard:
    """
    Checker for in_valid / out_valid pipelines.

    Workflow per case: drive inputs and set in_valid=1, then `push({...})` the expected payload, then
    `await RisingEdge(dut.clk)` and `await Timer(1, "ns")`, then `sample()`. The invariant is that out_valid
    going high must coincide with a previously-queued expectation reaching the head of the queue; a high out_valid
    with an empty queue is flagged as a spurious assertion. If `latency` is set, out_valid is also required to occur
    exactly that many sampled rising edges after the corresponding push.
    """

    def __init__(self, dut: Any, payload_fields: Iterable[tuple[str, str]], latency: int | None = None):
        self.dut = dut
        self.payload_fields = tuple(payload_fields)
        self.queue: deque[dict[str, Any]] = deque()
        self.latency = latency
        self.cycle = 0

    def push(self, expected: dict[str, Any]) -> None:
        if self.latency is not None:
            expected["_due_cycle"] = self.cycle + self.latency
        self.queue.append(expected)

    def sample(self) -> None:
        self.cycle += 1
        ov = int(self.dut.out_valid.value)
        if self.latency is not None and self.queue:
            due_cycle = self.queue[0]["_due_cycle"]
            desc = self.queue[0].get("_desc", "?")
            if ov:
                assert (
                    self.cycle == due_cycle
                ), f"out_valid latency mismatch in case {desc!r}: got cycle {self.cycle}, expected {due_cycle}"
            else:
                assert self.cycle != due_cycle, f"missing out_valid in case {desc!r} at expected cycle {due_cycle}"
        if not ov:
            return
        assert self.queue, "spurious out_valid: no expected output queued"
        expected = self.queue.popleft()
        desc = expected.get("_desc", "?")
        for sig_name, key in self.payload_fields:
            if key not in expected:
                continue
            actual = int(getattr(self.dut, sig_name).value)
            exp = expected[key]
            assert actual == exp, f"{sig_name} mismatch in case {desc!r}: " f"got 0x{actual:x}, expected 0x{exp:x}"

    async def drain(self, max_cycles: int = 512) -> None:
        """Hold in_valid=0 until the queue empties or max_cycles elapses."""
        self.dut.in_valid.value = 0
        for _ in range(max_cycles):
            if not self.queue:
                return
            await RisingEdge(self.dut.clk)
            await Timer(1, unit="ns")
            self.sample()
        assert not self.queue, f"queue did not drain in {max_cycles} cycles ({len(self.queue)} left)"
