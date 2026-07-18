"""Trim T6: str value methods are rejected; str constants stay inert."""


def kernel(x: float) -> float:
    return x * float(len("ab".upper()))
