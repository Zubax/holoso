"""
Whole-array reductions as balanced pairwise trees, log-deep in the operator's latency where a left fold over a
wide array would serialize on it (numpy's own float sum is pairwise as well); no axis/keepdims/dtype forms.
`sum`/`min`/`max` preserve the scalar family; `mean` promotes to float before accumulating, as numpy does.
Direct tensor iteration sidesteps the unroll threshold, and each level's length is static, so the odd-element
passthrough folds away at compile time. 0-D operands follow numpy up to the widthless model: `sum`/`mean`
promote via `+ 0`/`+ 0.0` (so a compiled bool refuses), the extrema pass the operand through.
Bool arrays cannot exist in the value model, so the host-side folds are outside the reference contract.
"""

from collections.abc import Callable
from typing import Any

import numpy as np

from ._linalg import flatten
from ._registry import array


def _add(a: Any, b: Any) -> Any:
    return a + b


def _pairwise(v: Any, f: Callable[[Any, Any], Any]) -> Any:
    while len(v) > 1:
        v = [f(v[i], v[i + 1]) if i + 1 < len(v) else v[i] for i in range(0, len(v), 2)]
    return v[0]


@array(np.sum, np.ndarray.sum)
def sum_(a: np.ndarray) -> Any:
    if a.ndim == 0:
        return a + 0
    return _pairwise(flatten(a), _add)


@array(np.min, np.amin, np.ndarray.min)
def amin(a: np.ndarray) -> Any:
    if a.ndim == 0:
        return a
    return _pairwise(flatten(a), min)


@array(np.max, np.amax, np.ndarray.max)
def amax(a: np.ndarray) -> Any:
    if a.ndim == 0:
        return a
    return _pairwise(flatten(a), max)


@array(np.mean, np.ndarray.mean)
def mean(a: np.ndarray) -> Any:
    if a.ndim == 0:
        return a + 0.0
    v = flatten(a) + 0.0
    return _pairwise(v, _add) / len(v)
