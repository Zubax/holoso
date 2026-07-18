"""C2: a comparison mixing a boolean and a non-boolean is refused in analysis, located."""


def kernel(flag: bool, x: float) -> float:
    hit = flag == x
    return 1.0 if hit else 0.0
