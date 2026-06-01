class TrapezoidalLeakyStreamingIntegrator:
    def __init__(self, *, k: float = 2**-22) -> None:
        self.k: float = k
        self.y: float = 0.0
        self._x_prev: float = 0.0

    def __call__(self, x: float, /) -> float:
        self.y = (1.0 - self.k) * self.y + 0.5 * (x + self._x_prev)
        self._x_prev = x
        return self.y
