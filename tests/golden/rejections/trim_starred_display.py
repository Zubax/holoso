"""Trim T10 (H2): a starred element in a list display is rejected."""


def kernel(a: float, b: float, c: float) -> list[float]:
    v = [a, b, c]
    head = v[0:2]
    return [v[2], *head]
