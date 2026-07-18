"""E2: a data-dependent raise with an f-string message surfaces the interpolated message, located."""

_LIMIT = 2.0


def kernel(x: float) -> float:
    if x > _LIMIT:
        raise ValueError(f"x exceeds the limit {_LIMIT}")
    return x * 0.5
