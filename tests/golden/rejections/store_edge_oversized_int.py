"""B1 store-edge conversion: an int store into a float variable rejects beyond the binary64 carrier range."""


def kernel(v: float) -> float:
    x = 1.5
    x = 2**7000
    return v
