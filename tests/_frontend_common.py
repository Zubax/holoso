"""Shared helpers and constants for the frontend test modules."""

import types
from collections.abc import Callable

import numpy as np
import pytest

import holoso
from holoso import FloatFormat
from holoso._hir import Hir, Operation

from ._modelref import default_ops


def _rebind_globals(fn: Callable[..., object], **overrides: object) -> Callable[..., object]:
    """A copy of ``fn`` whose module globals carry ``overrides`` (its source stays retrievable via the shared code)."""
    assert isinstance(fn, types.FunctionType)
    copy = types.FunctionType(
        fn.__code__, {**fn.__globals__, **overrides}, fn.__name__, fn.__defaults__, fn.__closure__
    )
    copy.__annotations__ = dict(fn.__annotations__)  # FunctionType does not copy these; the entry point reads them
    return copy


def _op_count(hir: Hir, op_type: type) -> int:
    return sum(1 for n in hir.nodes.values() if isinstance(n, Operation) and type(n.operator) is op_type)


_INEXACT_INTEGER = 2**53 + 1
_BIG_A = 2**53
_BIG_B = 1
_INT_TABLE = np.array([[2**53 + 1, 3]], dtype=np.int64)
_BIG_F = float(2**53)


def _assert_shape_kernel_matches_python(fn: Callable[..., float], v: np.ndarray) -> None:
    sim = holoso.synthesize(fn, default_ops(FloatFormat(11, 52)), name=fn.__qualname__.split(".")[0]).numerical_model
    assert [float(x) for x in sim.elaborate().run(*v.tolist())] == pytest.approx([fn(v)])
