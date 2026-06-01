class IIR1LPF:
    """
    A single-pole low-pass IIR filter. Difference equation: y[n] = y[n-1] + alpha * (x[n] - y[n-1])
    """

    def __init__(self, *, ALPHA: float = 2**-16):
        self.alpha: float = ALPHA
        self.y: float = 0.0
        self._first: bool = True

    def __call__(self, x: float) -> float:
        if self._first:
            self._first = False
            self.y = x
        else:
            self.y += self.alpha * (x - self.y)
        return self.y
