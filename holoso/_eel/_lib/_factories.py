"""Each builder stub doubles as a plain-Python reference."""

import numpy as np

from ._registry import factory


@factory(np.zeros)
def zeros_(shape: int | tuple[int, ...]) -> object:
    return np.zeros(shape)


@factory(np.ones)
def ones_(shape: int | tuple[int, ...]) -> object:
    return np.ones(shape)


@factory(np.full)
def full_(shape: int | tuple[int, ...], fill: float) -> object:
    return np.full(shape, fill)


@factory(np.eye)
def eye_(rows: int, columns: int | None = None) -> object:
    return np.eye(rows, columns)
