"""
Whole-array reductions as `_dot`-style left folds (FMA stays reachable); no axis/keepdims/dtype forms.
`sum`/`min`/`max` preserve the scalar family; `mean` promotes to float before accumulating, as numpy does.
Direct tensor iteration sidesteps the unroll threshold; the statically-folded `len(v) > 1` guard keeps
length-one inputs away from the empty-slice refusal. 0-D operands follow numpy up to the widthless model:
`sum`/`mean` promote via `+ 0`/`+ 0.0` (so a compiled bool refuses), the extrema pass the operand through.
Bool arrays cannot exist in the value model, so the host-side folds are outside the reference contract.
"""

from typing import Any

import numpy as np

from ._linalg import flatten
from ._registry import array


@array(np.sum, np.ndarray.sum)
def sum_(a: np.ndarray) -> Any:
    if a.ndim == 0:
        return a + 0
    v = flatten(a)
    acc = v[0]
    if len(v) > 1:
        for x in v[1:]:
            acc = acc + x
    return acc


@array(np.min, np.amin, np.ndarray.min)
def amin(a: np.ndarray) -> Any:
    if a.ndim == 0:
        return a
    v = flatten(a)
    acc = v[0]
    if len(v) > 1:
        for x in v[1:]:
            acc = min(acc, x)
    return acc


@array(np.max, np.amax, np.ndarray.max)
def amax(a: np.ndarray) -> Any:
    if a.ndim == 0:
        return a
    v = flatten(a)
    acc = v[0]
    if len(v) > 1:
        for x in v[1:]:
            acc = max(acc, x)
    return acc


@array(np.mean, np.ndarray.mean)
def mean(a: np.ndarray) -> Any:
    if a.ndim == 0:
        return a + 0.0
    v = flatten(a) + 0.0
    acc = v[0]
    if len(v) > 1:
        for x in v[1:]:
            acc = acc + x
    return acc / len(v)
