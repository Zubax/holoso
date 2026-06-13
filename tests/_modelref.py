"""Test-only verification helpers."""

import dataclasses
import math
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from holoso import FAddOperator, FCmpOperator, FDivOperator, FMulILog2OperatorFamily, FMulOperator, OpConfig
from holoso._type import FloatFormat
from holoso._frontend._lower import _Path, _port_name


def flatten_value(root: object) -> list[tuple[_Path, Any]]:
    """Flatten a runtime return value into ``(path, leaf)`` pairs."""
    leaves: list[tuple[_Path, Any]] = []

    def walk(node: object, path: _Path) -> None:
        if isinstance(node, (list, tuple)) and not isinstance(node, str):
            for index, item in enumerate(node):
                walk(item, [*path, index])
        elif dataclasses.is_dataclass(node) and not isinstance(node, type):
            for field in dataclasses.fields(node):
                walk(getattr(node, field.name), [*path, field.name])
        else:
            leaves.append((path, node))

    if (isinstance(root, (list, tuple)) and not isinstance(root, str)) or (
        dataclasses.is_dataclass(root) and not isinstance(root, type)
    ):
        walk(root, [])
    else:
        leaves.append(([0], root))
    return leaves


def evaluate_reference(fn: Callable[..., object], inputs: Mapping[str, float]) -> list[float]:
    """Call ``fn`` in float64 with the named inputs and flatten the result into ordered output values."""
    result = fn(**inputs)
    return [float(value) for _, value in flatten_value(result)]


def output_names(root: object) -> list[str]:
    """The ordered output-port names for a runtime return value."""
    return [_port_name(path) for path, _ in flatten_value(root)]


def unit_roundoff(fmt: FloatFormat) -> float:
    """The format's unit roundoff, ``2**-(wman-1)`` (relative spacing of representable values)."""
    return 2.0 ** -(fmt.wman - 1)


def default_tolerance(
    fmt: FloatFormat, op_count: int, magnitude: float = 1.0, rel_factor: float = 16.0
) -> tuple[float, float]:
    """A defensible (rtol, atol) for a kernel of ``op_count`` operations evaluated over operands up to ``magnitude``."""
    u = unit_roundoff(fmt)
    rtol = rel_factor * max(op_count, 1) * u
    atol = rtol * abs(magnitude)
    return rtol, atol


def within(actual: float, expected: float, rtol: float, atol: float) -> bool:
    """Whether ``actual`` is within ``atol + rtol*|expected|`` of ``expected`` (infinities must match exactly)."""
    if math.isinf(expected) or math.isinf(actual) or math.isnan(expected) or math.isnan(actual):
        return actual == expected
    return abs(actual - expected) <= atol + rtol * abs(expected)


def bounded(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(rng.uniform(lo, hi))


def log_uniform_positive(rng: np.random.Generator, lo: float, hi: float) -> float:
    """A strictly-positive value drawn log-uniformly in ``[lo, hi]`` (good for noise/scale parameters)."""
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def random_legal_bits(fmt: FloatFormat, rng: np.random.Generator) -> int:
    """A uniformly random finite, legal ZKF bit pattern (normals and +0; no inf/subnormal/negative zero)."""
    span = 1 << fmt.width
    while True:
        bits = int(rng.integers(0, span, dtype=np.uint64))
        if fmt.is_legal(bits) and fmt.is_finite(bits):
            return bits


def spd_matrix(rng: np.random.Generator, n: int, diag_lo: float = 0.5, diag_hi: float = 2.0) -> np.ndarray:
    """A random symmetric positive-definite ``n x n`` matrix (``L @ L.T`` with a positive-diagonal lower triangle)."""
    lower = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            lower[i, j] = rng.uniform(diag_lo, diag_hi) if i == j else rng.uniform(-1.0, 1.0)
    return lower @ lower.T


def encode_inputs(fmt: FloatFormat, values: dict[str, float | bool]) -> dict[str, int]:
    """Encode a name->float mapping to name->ZKF-bits (the bit pattern the DUT receives)."""
    return {name: int(value) if type(value) is bool else fmt.encode(value) for name, value in values.items()}


def format_edge_bits(fmt: FloatFormat) -> list[int]:
    """
    Canonical legal-ZKF edge bit patterns for one format: zero, ±0.5, ±1, ±smallest-normal, ±largest-finite.
    Built directly from the bit layout so the extremes stay exact even where they would overflow a Python float.
    """
    frac_bits = fmt.wman - 1
    sign_bit = 1 << (fmt.width - 1)
    max_exp = (1 << fmt.wexp) - 2  # the all-ones exponent is infinity, so the largest finite exponent is one below it
    magnitudes = [
        0,  # canonical zero (ZKF has no negative zero, so it has no signed counterpart)
        fmt.encode(0.5),
        fmt.encode(1.0),
        1 << frac_bits,  # smallest normal: exponent 1, zero fraction
        (max_exp << frac_bits) | ((1 << frac_bits) - 1),  # largest finite: max exponent, all-ones fraction
    ]
    edges: list[int] = []
    for magnitude in magnitudes:
        edges.append(magnitude)
        if magnitude != 0:
            edges.append(sign_bit | magnitude)
    return edges


def default_ops(fmt: FloatFormat) -> OpConfig:
    """The operator configuration with no optional pipeline stages: the minimum-latency baseline."""
    return fcmp_staged_ops(fmt, 0)


def fcmp_staged_ops(fmt: FloatFormat, stage_input: int) -> OpConfig:
    """Default operators with only the comparator's stage knob varied (latency ``1 + stage_input``)."""
    return OpConfig(
        FAddOperator(fmt),
        FMulOperator(fmt),
        FDivOperator(fmt),
        FMulILog2OperatorFamily(fmt),
        FCmpOperator(fmt, stage_input=stage_input),
    )


def branch_boundary_kernel(a, b, c):  # type: ignore[no-untyped-def]
    """
    The boundary-slack corner kernel shared by the cosim test and its white-box schedule twin: the comparison is the
    LAST commit in its block and feeds the branch, so its result lands in the condition register exactly one step
    before the terminator reads it. The two tests must exercise the same kernel, so it lives here. The division in
    the else arm is unspeculatable, which keeps the diamond a REAL branch under default if-conversion -- the corner
    under test exists only on a branchy schedule.
    """
    t = a * b + c
    if t > c:
        y = t + 1.0
    else:
        y = (t - 1.0) / (b * b + 1.0)  # structurally nonzero divisor: the bench asserts err_pc == 0 per vector
    return y


def staged_ops(fmt: FloatFormat) -> OpConfig:
    """
    A deeply pipelined configuration, distinct enough from the default to exercise the schedule, register allocation,
    and handshake at a longer latency. Deliberately hardcoded -- it is a test fixture chosen for coverage, not a
    derived enumeration of operator knobs, so it stays valid as new (not necessarily stage-shaped) knobs are added.
    """
    return OpConfig(
        FAddOperator(
            fmt, stage_input=1, stage_decode=1, stage_align=1, stage_normalize=1, stage_pack=1, stage_output=1
        ),
        FMulOperator(fmt, stage_input=1, stage_product=1, stage_pack=1, stage_output=1),
        FDivOperator(fmt, stage_input=1, stage_pack=1, stage_output=1),
        FMulILog2OperatorFamily(fmt, stage_input=1, stage_decode=1),
        FCmpOperator(fmt, stage_input=1),
    )


class ChainedSlots:
    """
    Chained persistent slots: ``_a`` captures ``_b``'s OLD value while ``_b`` advances, behind a long float tail.
    Shared by the schedule-level regression test and its RTL cosim twin -- the two must exercise the same kernel.
    """

    def __init__(self) -> None:
        self._a = 0.0
        self._b = 0.0

    def __call__(self, x):  # type: ignore[no-untyped-def]
        self._a = self._b
        self._b = x + 1.0
        return self._a * 2.0 + (x * 1.5) / (x - 0.5)


class SelectHold:
    """
    A Ret-block select is the slot live-in's LAST reader while the new live-out commits early: pins the read-step
    frame of the state early-install bound. Shared by the white-box schedule test and its RTL cosim twin.
    """

    def __init__(self) -> None:
        self._h = 1.0

    def step(self, x, c):  # type: ignore[no-untyped-def]
        old = self._h
        self._h = x + 1.0
        y = old if c > 0.0 else x
        return y * 2.0 + (x * 1.5) / (x * x + 0.5)  # structurally nonzero divisor (the bench asserts err_pc == 0)
