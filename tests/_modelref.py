"""
Test-only verification helpers, relocated out of the holoso library (which will gain a model-backend later).

Kept confined to this one module for easy removal: the float64 op-graph evaluator (a simulator-free reference for
the lowered HIR), input samplers, the output-name convention, and the tolerance predicate. Tests are exempt from the
"import only the public surface" rule, so this reaches into holoso's underscored modules directly.
"""

import math
from collections.abc import Mapping

import numpy as np

from holoso._format import FloatFormat
from holoso._frontend import flatten_value, port_name
from holoso._hir import Const, Hir, InPort, OpNode, ValueId
from holoso._operators import Sgnop


def _apply_sgnop(x: float, sgnop: Sgnop) -> float:
    if Sgnop.ABS in sgnop:
        x = abs(x)
    if Sgnop.NEG in sgnop:
        x = -x
    return x


def _eval_op(op: OpNode, val: dict[ValueId, float]) -> float:
    operands = [_apply_sgnop(val[vid], sgnop) for vid, sgnop in zip(op.operands, op.operand_sgnops)]
    return _apply_sgnop(op.op.evaluate(*operands), op.y_sgnop)


def evaluate_opgraph(hir: Hir, inputs: Mapping[str, float]) -> list[float]:
    """Evaluate a fully lowered HIR (InPort/Const/OpNode only) in float64 -- the simulator-free op-graph reference."""
    val: dict[ValueId, float] = {}
    for vid in sorted(hir.nodes):
        node = hir.nodes[vid]
        match node:
            case InPort(name=name):
                val[vid] = float(inputs[name])
            case Const(value=value):
                val[vid] = value
            case OpNode():
                val[vid] = _eval_op(node, val)
            case _:
                raise ValueError("evaluate_opgraph requires a fully lowered HIR (InPort/Const/OpNode only)")
    return [_apply_sgnop(val[out.value], out.sgnop) for out in hir.outputs]


def output_names(root: object) -> list[str]:
    """The ordered output-port names for a runtime return value."""
    return [port_name(path) for path, _ in flatten_value(root)]


def within(actual: float, expected: float, rtol: float, atol: float) -> bool:
    """Whether ``actual`` is within ``atol + rtol*|expected|`` of ``expected`` (infinities must match exactly)."""
    if math.isinf(expected) or math.isinf(actual) or math.isnan(expected) or math.isnan(actual):
        return actual == expected
    return abs(actual - expected) <= atol + rtol * abs(expected)


def bounded(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(rng.uniform(lo, hi))


def log_uniform_positive(rng: np.random.Generator, lo: float, hi: float) -> float:
    """A strictly-positive value drawn log-uniformly in ``[lo, hi]`` (good for noise/scale parameters)."""
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def random_legal_bits(fmt: FloatFormat, rng: np.random.Generator) -> int:
    """A uniformly random finite, legal ZKF bit pattern (normals and +0; no inf/subnormal/negative zero)."""
    span = 1 << fmt.width
    while True:
        bits = int(rng.integers(0, span, dtype=np.uint64))
        if fmt.is_legal(bits) and fmt.is_finite(bits):
            return bits


def spd_matrix(rng: np.random.Generator, n: int, diag_lo: float = 0.5, diag_hi: float = 2.0) -> np.ndarray:
    """A random symmetric positive-definite ``n x n`` matrix (``L @ L.T`` with a positive-diagonal lower triangle)."""
    lower = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            lower[i, j] = rng.uniform(diag_lo, diag_hi) if i == j else rng.uniform(-1.0, 1.0)
    return lower @ lower.T


def encode_inputs(fmt: FloatFormat, values: dict[str, float]) -> dict[str, int]:
    """Encode a name->float mapping to name->ZKF-bits (the bit pattern the DUT receives)."""
    return {name: fmt.encode(value) for name, value in values.items()}
