"""
Joining arrays along an axis. Every result is a fresh copy, as numpy's is, and mixed families promote the whole
result as numpy's conversion does. The axis is positional only, and axis `None` (flatten first) is not supported;
ranks above 2 are unsupported everywhere. A part must be an array or a scalar, not a nested Python sequence.
"""

from typing import Any

import numpy as np

from ._linalg import transpose
from ._registry import array


def _axis(axis: int, ndim: int) -> int:
    if not -ndim <= axis < ndim:
        raise ValueError(f"axis {axis} is out of bounds for a {ndim}-D result")
    return axis + ndim if axis < 0 else axis


def _chain(parts: Any) -> Any:
    """Pairwise merging, so the expansion budget pays for each item once per tree level rather than once per part."""
    while len(parts) > 1:
        parts = [(*parts[i], *parts[i + 1]) if i + 1 < len(parts) else parts[i] for i in range(0, len(parts), 2)]
    return np.array(parts[0])


@array(np.concatenate, np.concat, sequences=(0,))
def concatenate(arrays: Any, axis: int = 0) -> Any:
    if len(arrays) == 0:
        raise ValueError("need at least one array to concatenate")
    ndim = arrays[0].ndim
    if ndim == 0:
        raise ValueError("zero-dimensional arrays cannot be concatenated")
    if ndim > 2:
        raise ValueError(f"concatenation of {ndim}-D arrays is not supported")
    for a in arrays:
        if a.ndim != ndim:
            raise ValueError(f"all input arrays must have the same number of dimensions, got {ndim}-D and {a.ndim}-D")
    along = _axis(axis, ndim)
    if ndim == 2:
        for a in arrays:
            if a.shape[1 - along] != arrays[0].shape[1 - along]:
                raise ValueError(
                    f"dimension {1 - along} lengths {arrays[0].shape[1 - along]} and {a.shape[1 - along]} disagree"
                    f" off the concatenation axis {along}"
                )
    if along == 0:
        return _chain(arrays)
    return np.array([_chain([a[i] for a in arrays]) for i in range(len(arrays[0]))])


def _atleast_1d(a: Any) -> Any:
    return np.array([a]) if a.ndim == 0 else a


def _atleast_2d(a: Any) -> Any:
    return np.array([_atleast_1d(a)]) if a.ndim < 2 else a


@array(np.vstack, sequences=(0,))
def vstack(tup: Any) -> Any:
    return concatenate([_atleast_2d(a) for a in tup], 0)


@array(np.hstack, sequences=(0,))
def hstack(tup: Any) -> Any:
    parts = [_atleast_1d(a) for a in tup]
    if len(parts) > 0:
        if parts[0].ndim == 2:  # nested because `and` evaluates both operands in the subset
            return concatenate(parts, 1)
    return concatenate(parts, 0)


@array(np.column_stack, sequences=(0,))
def column_stack(tup: Any) -> Any:
    return concatenate([transpose(_atleast_2d(a)) if a.ndim < 2 else a for a in tup], 1)


@array(np.stack, sequences=(0,))
def stack(arrays: Any, axis: int = 0) -> Any:
    if len(arrays) == 0:
        raise ValueError("need at least one array to stack")
    ndim = arrays[0].ndim
    if ndim > 1:
        raise ValueError(f"stacking {ndim}-D arrays is not supported")
    for a in arrays:
        if a.ndim != ndim:
            raise ValueError(f"all input arrays must have the same shape, got {ndim}-D and {a.ndim}-D")
        if ndim == 1:
            if len(a) != len(arrays[0]):
                raise ValueError(
                    f"all input arrays must have the same shape, got lengths {len(arrays[0])} and {len(a)}"
                )
    if _axis(axis, ndim + 1) == 0:
        return np.array(arrays)
    return np.array([[a[i] for a in arrays] for i in range(len(arrays[0]))])
