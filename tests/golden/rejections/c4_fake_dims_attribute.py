"""C4 surface: a class carrying a non-tuple dims attribute is not an array annotation and rejects cleanly."""


class FakeArray:
    dims = None


def kernel(p: FakeArray) -> float:
    return 1.0
