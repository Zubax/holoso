"""Pre-existing shape: a recursive call is rejected, blaming the user call site through the origin stack."""


def spiral(v: float) -> float:
    return spiral(v - 1.0) if v > 0.0 else v


def kernel(x: float) -> float:
    return spiral(x)
