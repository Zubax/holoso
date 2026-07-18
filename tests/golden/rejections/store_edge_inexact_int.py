"""B1 store-edge conversion: an int store into a float variable rejects when inexact in the binary64 carrier."""


def kernel(value: float) -> float:
    current = value
    current = 2**53 + 1
    return current
