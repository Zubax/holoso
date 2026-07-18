"""Trim T9: a starred element in an assignment target is rejected; plain unpacking stays."""


def kernel(x: float) -> float:
    t: tuple[float, ...] = (x, 2.0, 3.0)
    a, *rest = t
    return a + rest[0]
