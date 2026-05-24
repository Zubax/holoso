"""The verification reference: run the original Python function in float64 and flatten its return to ordered outputs."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from .._shape import flatten_value


def evaluate(fn: Callable[..., object], inputs: Mapping[str, float]) -> list[float]:
    """Call ``fn`` with the named inputs and flatten the result into ordered output values (matching port order)."""
    result = fn(**inputs)
    return [float(value) for _, value in flatten_value(result)]
