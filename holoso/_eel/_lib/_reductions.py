"""
Whole-array reductions as balanced pairwise trees, log-deep in the operator's latency where a left fold over a
wide array would serialize on it (numpy's own float sum is pairwise as well, while its product is a left fold, so
`prod` reassociates where the reference does not); no axis/keepdims/dtype forms.
`sum`/`prod`/`min`/`max` preserve the scalar family; `mean` promotes to float before accumulating, as numpy does.
0-D operands follow numpy up to the widthless model: `sum`/`prod`/`mean` promote via `+ 0`/`* 1`/`+ 0.0` (so a
compiled bool refuses), the extrema pass the operand through.
Bool arrays cannot exist in the value model, so the host-side folds are outside the reference contract.
"""

import operator
from typing import Any

import numpy as np

from ._linalg import flatten, pairwise
from ._registry import array


@array(np.sum, np.ndarray.sum)
def sum_(a: np.ndarray) -> Any:
    if a.ndim == 0:
        return a + 0
    return pairwise(flatten(a), operator.add)


@array(np.prod, np.ndarray.prod)
def prod(a: np.ndarray) -> Any:
    if a.ndim == 0:
        return a * 1
    return pairwise(flatten(a), operator.mul)


@array(np.min, np.amin, np.ndarray.min)
def amin(a: np.ndarray) -> Any:
    if a.ndim == 0:
        return a
    return pairwise(flatten(a), min)


@array(np.max, np.amax, np.ndarray.max)
def amax(a: np.ndarray) -> Any:
    if a.ndim == 0:
        return a
    return pairwise(flatten(a), max)


@array(np.mean, np.ndarray.mean)
def mean(a: np.ndarray) -> Any:
    if a.ndim == 0:
        return a + 0.0
    v = flatten(a) + 0.0
    return pairwise(v, operator.add) / len(v)
