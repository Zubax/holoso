"""B1 storage schema: a local established as float cannot be rebound to a bool."""


def kernel(v: float) -> float:
    y = v * 2.0
    y = v > 0.0
    return 1.0 if y else 0.0
