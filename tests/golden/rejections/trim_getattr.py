"""Trim T1: getattr() calls are rejected -- spell the attribute access directly."""


class Accumulator:
    def __init__(self) -> None:
        self.g = 1.0

    def step(self, x: float) -> float:
        self.g = self.g + x
        return getattr(self, "g")
