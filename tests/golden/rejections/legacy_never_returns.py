"""Pre-existing shape: a kernel that never returns on any path is rejected."""


def kernel(x: float) -> float:
    while True:
        x = x + 1.0
