"""Input-vector sampling for verification.

Provides ZKF-legal random bit patterns and well-conditioned building blocks (bounded values, log-uniform positives,
symmetric-positive-definite matrices) so that test vectors avoid catastrophic cancellation / near-singular
configurations and keep the tolerance band meaningful.
"""

from __future__ import annotations

import numpy as np

from ..format import FloatFormat
from .zkf_codec import encode, is_finite, is_legal


def bounded(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(rng.uniform(lo, hi))


def log_uniform_positive(rng: np.random.Generator, lo: float, hi: float) -> float:
    """A strictly-positive value drawn log-uniformly in ``[lo, hi]`` (good for noise/scale parameters)."""
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def random_legal_bits(fmt: FloatFormat, rng: np.random.Generator) -> int:
    """A uniformly random *finite, legal* ZKF bit pattern (normals and +0; no inf/subnormal/negative zero)."""
    span = 1 << fmt.width
    while True:
        bits = int(rng.integers(0, span, dtype=np.uint64))
        if is_legal(fmt, bits) and is_finite(fmt, bits):
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
    return {name: encode(fmt, value) for name, value in values.items()}
