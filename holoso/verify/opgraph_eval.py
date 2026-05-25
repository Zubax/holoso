"""
Evaluate a fully lowered HIR op-graph in float64.

This mirrors exactly the operators and folded sign-ops the generated RTL instantiates, so comparing it against the
original Python function (both in float64) checks that the front-end and passes preserve the computation -- a
simulator-free correctness net for everything upstream of the backend.
"""

from collections.abc import Mapping

from ..hir import Const, Hir, InPort, OpNode, ValueId
from ..operators import Sgnop


def _apply_sgnop(x: float, sgnop: Sgnop) -> float:
    if Sgnop.ABS in sgnop:
        x = abs(x)
    if Sgnop.NEG in sgnop:
        x = -x
    return x


def _eval_op(op: OpNode, val: dict[ValueId, float]) -> float:
    operands = [_apply_sgnop(val[vid], sgnop) for vid, sgnop in zip(op.operands, op.operand_sgnops)]
    return _apply_sgnop(op.op.evaluate(*operands), op.y_sgnop)


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
