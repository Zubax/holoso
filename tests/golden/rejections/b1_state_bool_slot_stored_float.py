"""B1 storage schema: a persistent bool slot rejects a float store."""


class Latch:
    def __init__(self) -> None:
        self.armed = False

    def step(self, v: float) -> float:
        self.armed = v
        return v
