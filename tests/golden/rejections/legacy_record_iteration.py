"""Pre-existing shape: iteration over a record is rejected even when it defines __len__/__getitem__."""

import dataclasses


@dataclasses.dataclass(frozen=True)
class Pair:
    a: float
    b: float

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> float:
        return (self.a, self.b)[index]


PAIR = Pair(1.0, 2.0)


def kernel(x: float) -> float:
    acc = x
    for v in PAIR:
        acc = acc + v
    return acc
