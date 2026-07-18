"""Trim T2 (deleted defect A2): elementwise array comparison produces no mask."""

import numpy as np

COEFFS = np.array([0.25, 0.5, 1.0])


def kernel(s: float) -> tuple[float, float, float]:
    mask = (COEFFS * s) >= 0.5
    return float(mask[0]), float(mask[1]), float(mask[2])
