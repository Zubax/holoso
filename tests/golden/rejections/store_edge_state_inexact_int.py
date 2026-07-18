"""B1 store-edge conversion: a float state slot rejects an int store that is inexact in the binary64 carrier."""


class Total:
    def __init__(self) -> None:
        self.total = 0.0

    def step(self, v: float) -> float:
        self.total = 2**53 + 1
        return self.total + v
