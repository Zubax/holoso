"""B1 storage schema: a persistent int slot rejects a float store that reaches the exit."""


class Counter:
    def __init__(self) -> None:
        self.n = 0

    def step(self, v: float) -> float:
        self.n = v
        return v
