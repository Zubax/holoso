"""Trim T7: a record class whose construction would run user code (__post_init__) is rejected."""

import dataclasses


@dataclasses.dataclass(frozen=True)
class Hooked:
    v: float

    def __post_init__(self) -> None:
        raise RuntimeError("user code must never run at compile time")


def kernel(x: float) -> float:
    return Hooked(x).v
