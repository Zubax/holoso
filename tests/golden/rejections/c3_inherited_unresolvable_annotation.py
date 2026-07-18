"""C3 surface: an inherited record field whose annotation fails to evaluate is rejected."""

import dataclasses


@dataclasses.dataclass(frozen=True)
class Base:
    gain: "missing_type"


@dataclasses.dataclass(frozen=True)
class Derived(Base):
    offset: float


def kernel(p: Derived) -> float:
    return p.offset
