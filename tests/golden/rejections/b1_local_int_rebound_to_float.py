"""B1 storage schema: a local established as int cannot be rebound to a float."""


def kernel(v: float) -> float:
    x = 0
    x = v
    return x
