"""Evaluate a fully lowered HIR op-graph in float64.

This mirrors exactly the operators and folded sign-ops the generated RTL instantiates, so comparing it against the
original Python function (both in float64) checks that the front-end and passes preserve the computation -- a
simulator-free correctness net for everything upstream of the backend.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import assert_never

from ..hir import Const, Hir, InPort, OpNode, ValueId
from ..operators import OpKind


def _apply_sgnop(x: float, sgnop: int) -> float:
    if sgnop & 2:
        x = abs(x)
    if sgnop & 1:
        x = -x
    return x


def _eval_op(op: OpNode, val: dict[ValueId, float]) -> float:
    a = _apply_sgnop(val[op.a], op.a_sgnop)
    match op.kind:
        case OpKind.FADD:
            assert op.b is not None
            r = a + _apply_sgnop(val[op.b], op.b_sgnop)
        case OpKind.FMUL:
            assert op.b is not None
            r = a * _apply_sgnop(val[op.b], op.b_sgnop)
        case OpKind.FDIV:
            assert op.b is not None
            b = _apply_sgnop(val[op.b], op.b_sgnop)
            r = a / b if b != 0.0 else math.copysign(math.inf, a)
        case OpKind.FMUL_ILOG2:
            assert op.k is not None
            r = math.ldexp(a, op.k)
        case _ as unreachable:
            assert_never(unreachable)
    return _apply_sgnop(r, op.y_sgnop)


def evaluate(hir: Hir, inputs: Mapping[str, float]) -> list[float]:
    """Evaluate a lowered HIR (InPort/Const/OpNode only) at the given named inputs, returning the output values."""
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
                raise ValueError("opgraph_eval requires a fully lowered HIR (InPort/Const/OpNode only)")
    return [_apply_sgnop(val[out.value], out.sgnop) for out in hir.outputs]
