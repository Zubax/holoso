"""
Linear algebra library functions.
The stubs are ordinary executable Python over numpy arrays, so each doubles as its own numerical reference.

Batched (`ndim > 2`) operands are unsupported everywhere; there is no runtime sizing.

Booleans reach the arithmetic of every stub and are rejected there; no stub passes one through.

A scalar has `ndim == 0` here, as a numpy scalar does and a Python float does not; that is what lets `matmul`
reject a scalar operand by asking its rank rather than by a type test the subset cannot express.
"""

from typing import Any

import numpy as np

from .._ir import BinaryOp
from ._registry import array


def _dot(u: np.ndarray, v: np.ndarray) -> Any:
    """A left fold to enable FMA contraction."""
    acc = u[0] * v[0]
    for k in range(1, len(u)):
        acc = acc + u[k] * v[k]
    return acc


@array(np.transpose, np.ndarray.T, np.ndarray.transpose, derives=True)
def transpose(a: np.ndarray) -> Any:
    if a.ndim == 0:
        raise ValueError("cannot transpose a scalar value")
    if a.ndim > 2:
        raise ValueError(f"transpose of a {a.ndim}-D value is not supported")
    if a.ndim == 1:
        return a  # numpy leaves a vector alone: it has no second axis to swap
    return np.array([[a[i][j] for i in range(len(a))] for j in range(len(a[0]))])


@array(np.ravel, np.ndarray.flatten, np.ndarray.ravel, derives=True)
def flatten(a: np.ndarray) -> Any:
    if a.ndim == 0:
        raise ValueError("cannot flatten a scalar value")
    if a.ndim > 2:
        raise ValueError(f"flatten of a {a.ndim}-D value is not supported")
    if a.ndim == 1:
        return a
    width = len(a[0])
    return np.asarray([a[k // width, k % width] for k in range(len(a) * width)])


@array(np.dot, np.ndarray.dot, np.matmul, BinaryOp.MATMUL)
def matmul(a: np.ndarray, b: np.ndarray) -> Any:
    """
    numpy's shape rules for 1-D and 2-D operands: inner dimensions must agree, a 1-D left operand is promoted to a row
    and a 1-D right operand to a column, and each promoted axis is dropped from the result -- so vector @ vector is a
    scalar dot product. Rows iterate outermost and the contraction innermost, one dot product per output element.
    FIXME Scalars are currently not supported.
    """
    if a.ndim == 0 or b.ndim == 0:
        raise ValueError("FIXME matmul does not accept scalar operands")
    if a.ndim > 2 or b.ndim > 2:
        raise ValueError(f"matmul operands must be 1-D or 2-D, got {a.ndim}-D @ {b.ndim}-D")
    if a.shape[-1] != len(b):
        raise ValueError(f"matmul dimension mismatch: inner dimensions {a.shape[-1]} and {len(b)} disagree")
    if b.ndim == 1:
        if a.ndim == 1:
            return _dot(a, b)
        return np.array([_dot(a[i], b) for i in range(len(a))])
    bt = transpose(b)  # the columns of b, so every output element is the dot product of two rows
    if a.ndim == 1:
        return np.array([_dot(a, bt[j]) for j in range(len(bt))])
    return np.array([[_dot(a[i], bt[j]) for j in range(len(bt))] for i in range(len(a))])


@array(np.trace, np.ndarray.trace)
def trace(a: np.ndarray) -> Any:
    """FIXME Support non-square matrices by running the shorter diagonal."""
    if a.ndim != 2:
        raise ValueError(f"trace requires a matrix, got a {a.ndim}-D value")
    if len(a) != len(a[0]):
        raise ValueError(f"trace requires a square matrix, got {len(a)}×{len(a[0])}")
    acc = a[0][0]
    for i in range(1, len(a)):
        acc = acc + a[i][i]
    return acc


@array(np.outer)
def outer(u: np.ndarray, v: np.ndarray) -> Any:
    """FIXME Currently does not flatten its operands."""
    if u.ndim != 1 or v.ndim != 1:
        raise ValueError(f"outer requires 1-D operands, got {u.ndim}-D and {v.ndim}-D")
    return np.array([[u[i] * v[j] for j in range(len(v))] for i in range(len(u))])


@array(np.linalg.inv)
def inv(m: np.ndarray) -> Any:
    """
    Gauss-Jordan inversion with partial pivoting; a singular pivot follows the division-by-zero contract
    instead of raising `numpy.linalg.LinAlgError`.
    """
    if m.ndim != 2:
        raise ValueError(f"inv requires a matrix, got a {m.ndim}-D value")
    if m.shape[0] != m.shape[1]:
        raise ValueError(f"inv requires a square matrix, got {m.shape[0]}×{m.shape[1]}")
    n = len(m)
    a = m + 0.0  # Fresh owned float working copy: promotes int elements and never mutates the caller's array.
    r = np.eye(n)
    for k in range(n):
        for p in range(k + 1, n):  # Bubble the largest |pivot| onto the diagonal via conditional swaps at
            if abs(a[p, k]) > abs(a[k, k]):  # static indices, since a runtime-selected index cannot be spelled.
                for j in range(n):
                    t = a[k, j]
                    a[k, j] = a[p, j]
                    a[p, j] = t
                    s = r[k, j]
                    r[k, j] = r[p, j]
                    r[p, j] = s
        piv = a[k, k]
        # Natural divisions, not a minted reciprocal: 1/piv can rail near the format's magnitude limits while
        # every quotient is ordinary (the constant-divisor strength reduction guards the same hazard).
        for j in range(n):
            a[k, j] = a[k, j] / piv
            r[k, j] = r[k, j] / piv
        for i in range(n):
            if i != k:
                f = a[i, k]
                for j in range(n):
                    a[i, j] = a[i, j] - f * a[k, j]
                    r[i, j] = r[i, j] - f * r[k, j]
    return r
