"""Trim T8 (the retained hook guard): a component with a custom __setattr__ is refused."""


class Clamped:
    def __init__(self) -> None:
        object.__setattr__(self, "gain", 0.5)

    def __setattr__(self, name: str, value: float) -> None:
        object.__setattr__(self, name, min(1.0, max(0.0, value)))

    def step(self, x: float) -> float:
        self.gain = 3.0
        return self.gain * x
