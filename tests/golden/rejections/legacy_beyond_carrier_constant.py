"""Pre-existing shape: an integer constant beyond the binary64 carrier cannot enter the float domain."""

_HUGE = 2**1024


def kernel(x: float) -> float:
    return 1.0 if _HUGE == x else 0.0
