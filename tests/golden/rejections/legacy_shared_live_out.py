"""Pre-existing shape: the backend refuses two state slots that end the transaction holding one value."""


class Shared:
    def __init__(self) -> None:
        self.a = 0.0
        self.b = 1.0

    def step(self, x: float) -> float:
        self.a = x + self.a
        self.a = x + self.a
        self.b = self.a
        return self.b
