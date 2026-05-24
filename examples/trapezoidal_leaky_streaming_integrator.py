class TrapezoidalLeakyStreamingIntegrator:
    """
    Classes can be used to define stateful modules.
    The __init__ method defines the initial states acquired at module reset; all of these must resolve to constant
    expressions upon constant folding OR be derived from __init__ keyword-only scalar parameters with defaults,
    which map to elaboration-time parameters.
    """

    def __init__(self, *, K: float = 2**-22) -> None:
        """
        K translates to `parameter real K = 2**-22`.
        """
        self.k: float = K
        self.y: float = 0.0
        self._x_prev: float = 0.0

    def __call__(self, x: float, /) -> float:
        self.y = (1.0 - self.k) * self.y + 0.5 * (x + self._x_prev)
        self._x_prev = x
        return self.y
