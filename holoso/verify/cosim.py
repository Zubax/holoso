"""
Build cosimulation vector sets: sample inputs, encode to ZKF bits, and compute the float64 reference + tolerance.
"""

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from ..format import FloatFormat
from . import reference
from .tolerance import default_tolerance

Sampler = Callable[[FloatFormat, Sequence[str], np.random.Generator], dict[str, float]]


def generic_sampler(fmt: FloatFormat, names: Sequence[str], rng: np.random.Generator) -> dict[str, float]:
    """A simple well-bounded sampler: each input uniform in [-4, 4]."""
    return {name: float(rng.uniform(-4.0, 4.0)) for name in names}


def build_vectors(
    fn: Callable[..., object],
    fmt: FloatFormat,
    input_names: Sequence[str],
    output_names: Sequence[str],
    op_count: int,
    *,
    count: int,
    rng: np.random.Generator,
    timeout_cycles: int,
    cycles: int,
    sampler: Sampler = generic_sampler,
) -> dict[str, Any]:
    """
    Produce ``count`` ``(input-bits, expected-floats, tolerance)`` records for the cosim driver.

    ``cycles`` is the model's exact in_valid->out_valid latency; the driver asserts the DUT matches it on every vector.
    """
    vectors: list[dict[str, Any]] = []
    for _ in range(count):
        values = sampler(fmt, input_names, rng)
        bits = {name: fmt.encode(value) for name, value in values.items()}
        decoded = {name: fmt.decode(pattern) for name, pattern in bits.items()}  # the values the DUT actually sees
        expected = reference.evaluate(fn, decoded)
        magnitude = max((abs(v) for v in decoded.values()), default=1.0)
        rtol, atol = default_tolerance(fmt, op_count, magnitude=magnitude)
        vectors.append({"in": bits, "exp": dict(zip(output_names, expected)), "rtol": rtol, "atol": atol})
    return {
        "wexp": fmt.wexp,
        "wman": fmt.wman,
        "inputs": list(input_names),
        "outputs": list(output_names),
        "timeout_cycles": timeout_cycles,
        "cycles": cycles,
        "vectors": vectors,
    }
