"""Trim T3 (deleted defect C1): a 0-dimensional array is rejected at creation."""

import numpy as np

Z = np.array(3.0)


def kernel(x: float) -> float:
    return x * float(Z)
