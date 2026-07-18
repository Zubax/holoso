"""Trim T4: isinstance is rejected -- values are statically typed."""


def kernel(x: float) -> float:
    return x * 2.0 if isinstance(1.0, float) else x
