"""
Test-only verification helpers, relocated out of the holoso library (which now carries a numerical model backend).

Kept confined to this one module for easy removal: the float64 reference, the tolerance model, input samplers, and the
output-name convention. Tests are exempt from the "import only the public surface" rule, so this reaches into holoso's
underscored modules directly.
"""

import math
from collections.abc import Callable, Mapping

import numpy as np

from holoso._format import FloatFormat
from holoso._frontend import flatten_value, port_name


def evaluate_reference(fn: Callable[..., object], inputs: Mapping[str, float]) -> list[float]:
    """Call ``fn`` in float64 with the named inputs and flatten the result into ordered output values."""
    result = fn(**inputs)
    return [float(value) for _, value in flatten_value(result)]


def output_names(root: object) -> list[str]:
    """The ordered output-port names for a runtime return value."""
    return [port_name(path) for path, _ in flatten_value(root)]


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


def encode_inputs(fmt: FloatFormat, values: dict[str, float]) -> dict[str, int]:
    """Encode a name->float mapping to name->ZKF-bits (the bit pattern the DUT receives)."""
    return {name: fmt.encode(value) for name, value in values.items()}
