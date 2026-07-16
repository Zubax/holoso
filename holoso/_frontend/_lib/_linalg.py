"""
Linear algebra library functions.
The stubs are ordinary executable Python over numpy arrays, so each doubles as its own numerical reference.

Batched (`ndim > 2`) operands are unsupported everywhere; there is no runtime sizing.

Booleans reach the arithmetic of every stub and are rejected there; no stub passes one through.

Raise messages stay LITERAL so the builder's message extraction reproduces them verbatim (an f-string
degrades to a bare "raise"). Rank probes use `np.ndim`, whose real Python behavior on a float is zero -- a scalar `.ndim` ATTRIBUTE would
be an AttributeError in plain Python, so the compiler folds only the function spelling. Transpose is not a
stub: `.T` and `np.transpose` lower structurally as a leaf permutation, so `_transpose` here is only the plain
helper the matrix product uses to walk columns.
"""

from typing import Any

import numpy as np

from ._registry import lib


def _dot(u: np.ndarray, v: np.ndarray) -> Any:
    """A left fold to enable FMA contraction."""
    acc = u[0] * v[0]
    for k in range(1, len(u)):
        acc = acc + u[k] * v[k]
    return acc


def _transpose(a: np.ndarray) -> Any:
    return np.array([[a[i][j] for i in range(len(a))] for j in range(len(a[0]))])


@lib(np.matmul, np.dot)
def matmul_(a: np.ndarray, b: np.ndarray) -> Any:
    """
    numpy's shape rules for 1-D and 2-D operands: inner dimensions must agree, a 1-D left operand is promoted to a row
    and a 1-D right operand to a column, and each promoted axis is dropped from the result -- so vector @ vector is a
    scalar dot product. Rows iterate outermost and the contraction innermost, one dot product per output element.
    """
    if np.ndim(a) == 0 or np.ndim(b) == 0:
        raise ValueError("matmul does not accept scalar operands")
    if np.ndim(a) > 2 or np.ndim(b) > 2:
        raise ValueError("matmul operands must be 1-D or 2-D")
    if a.shape[-1] != len(b):
        raise ValueError("matmul dimension mismatch: the inner dimensions disagree")
    if np.ndim(b) == 1:
        if np.ndim(a) == 1:
            return _dot(a, b)
        return np.array([_dot(a[i], b) for i in range(len(a))])
    bt = _transpose(b)  # the columns of b, so every output element is the dot product of two rows
    if np.ndim(a) == 1:
        return np.array([_dot(a, bt[j]) for j in range(len(bt))])
    return np.array([[_dot(a[i], bt[j]) for j in range(len(bt))] for i in range(len(a))])


@lib(np.trace)
def trace_(a: np.ndarray) -> Any:
    if np.ndim(a) != 2:
        raise ValueError("trace requires a matrix")
    if len(a) != len(a[0]):
        raise ValueError("trace requires a square matrix")
    acc = 0.0
    for i in range(len(a)):
        acc = acc + a[i][i]
    return acc


@lib(np.outer)
def outer_(u: np.ndarray, v: np.ndarray) -> Any:
    if np.ndim(u) != 1 or np.ndim(v) != 1:
        raise ValueError("outer requires 1-D operands")
    return np.array([[u[i] * v[j] for j in range(len(v))] for i in range(len(u))])
