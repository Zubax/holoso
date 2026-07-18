"""Pre-existing shape: a power with a runtime exponent requires a positive base (log2 domain)."""


def kernel(e: float) -> float:
    return (-2.0) ** e
