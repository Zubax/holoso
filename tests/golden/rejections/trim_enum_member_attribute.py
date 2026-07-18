"""Trim T5: an enum member admits only as its base value, so member attributes reject."""

import enum


class Mode(enum.IntEnum):
    FAST = 3


MODE = Mode.FAST


def kernel(x: float) -> float:
    return x * float(MODE.value)
