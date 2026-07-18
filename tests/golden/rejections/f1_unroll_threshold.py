"""F1: the unroll fuel guard -- a counted loop past the threshold rejects before materializing any trip."""


def kernel(a: float) -> float:
    x = a
    for _ in range(1000):
        x = x + a
    return x
