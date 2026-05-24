"""HIR -> HIR optimization and lowering passes.

Pipeline (:func:`run`): constant folding -> strength reduction -> operator selection with sign folding -> dead-code
elimination. The result is a fully lowered HIR (only ``InPort``/``Const``/``OpNode``), ready for scheduling.

FP math is non-associative; these passes may change results in the last bits, which is accepted (see DESIGN.md).
"""

from __future__ import annotations

import math
from typing import assert_never

from .hir import (
    Arith,
    ArithOp,
    Const,
    Fmul2K,
    Hir,
    HirBuilder,
    InPort,
    Node,
    OpNode,
    SignFix,
    SignOp,
    ValueId,
)
from .operators import OpKind, Sgnop, latency_of

_ARITH_TO_OPKIND: dict[ArithOp, OpKind] = {
    ArithOp.ADD: OpKind.FADD,
    ArithOp.MUL: OpKind.FMUL,
    ArithOp.DIV: OpKind.FDIV,
}


# ----------------------------------------------------------------------------------------------------------------------
# Small numeric helpers


def _eval_arith(op: ArithOp, a: float, b: float) -> float | None:
    match op:
        case ArithOp.ADD:
            return a + b
        case ArithOp.MUL:
            return a * b
        case ArithOp.DIV:
            return a / b if b != 0.0 else None
        case _ as unreachable:
            assert_never(unreachable)


def _eval_sign(op: SignOp, x: float) -> float:
    match op:
        case SignOp.NEG:
            return -x
        case SignOp.ABS:
            return abs(x)
        case _ as unreachable:
            assert_never(unreachable)


def _ilog2_exact(c: float) -> int | None:
    """Return ``k`` if ``c == 2**k`` exactly for a positive ``c``, else ``None``."""
    if c <= 0.0 or not math.isfinite(c):
        return None
    mantissa, exponent = math.frexp(c)  # c == mantissa * 2**exponent, mantissa in [0.5, 1)
    return exponent - 1 if mantissa == 0.5 else None


def _compose_sgnop(inner: Sgnop, outer: Sgnop) -> Sgnop:
    """Combine two sign-ops where ``inner`` is applied first (closest to the value) and ``outer`` after."""
    if Sgnop.ABS in outer:  # abs erases everything underneath
        return Sgnop.ABS | (outer & Sgnop.NEG)
    negated = (Sgnop.NEG in inner) ^ (Sgnop.NEG in outer)
    return (inner & Sgnop.ABS) | (Sgnop.NEG if negated else Sgnop.NONE)


def _collapse_signs(nodes: dict[ValueId, Node], vid: ValueId) -> tuple[ValueId, Sgnop]:
    """Peel a chain of ``SignFix`` nodes, returning the non-``SignFix`` base value and the combined sign-op."""
    chain: list[SignOp] = []
    node = nodes[vid]
    while isinstance(node, SignFix):
        chain.append(node.op)
        vid = node.a
        node = nodes[vid]
    sgnop = Sgnop.NONE
    for op in reversed(chain):  # innermost first
        sgnop = _compose_sgnop(sgnop, Sgnop.NEG if op is SignOp.NEG else Sgnop.ABS)
    return vid, sgnop


def _copy(builder: HirBuilder, node: Node, remap: dict[ValueId, ValueId]) -> ValueId:
    """Rebuild ``node`` into ``builder`` with operands remapped (the default, structure-preserving rewrite)."""
    match node:
        case InPort(name=name):
            return builder.input(name)
        case Const(value=value):
            return builder.const(value)
        case Arith(op=op, a=a, b=b):
            return builder.arith(op, remap[a], remap[b])
        case SignFix(op=op, a=a):
            return builder.signfix(op, remap[a])
        case Fmul2K(a=a, k=k):
            return builder.fmul2k(remap[a], k)
        case OpNode(kind=kind, a=a, b=b, a_sgnop=asg, b_sgnop=bsg, y_sgnop=ysg, k=k, latency=latency):
            nb = remap[b] if b is not None else None
            return builder.opnode(kind, remap[a], nb, asg, bsg, ysg, k, latency)
        case _ as unreachable:
            assert_never(unreachable)


# ----------------------------------------------------------------------------------------------------------------------
# Passes


def const_fold(hir: Hir) -> Hir:
    """Fold operations whose operands are all constants into a single constant."""
    builder = HirBuilder(hir.fmt)
    remap: dict[ValueId, ValueId] = {}
    cval: dict[ValueId, float] = {}
    for old_id in sorted(hir.nodes):
        node = hir.nodes[old_id]
        match node:
            case Const(value=value):
                new_id = builder.const(value)
                cval[new_id] = value
            case Arith(op=op, a=a, b=b) if (
                remap[a] in cval
                and remap[b] in cval
                and (folded := _eval_arith(op, cval[remap[a]], cval[remap[b]])) is not None
            ):
                new_id = builder.const(folded)
                cval[new_id] = folded
            case SignFix(op=op, a=a) if remap[a] in cval:
                folded_sign = _eval_sign(op, cval[remap[a]])
                new_id = builder.const(folded_sign)
                cval[new_id] = folded_sign
            case _:
                new_id = _copy(builder, node, remap)
        remap[old_id] = new_id
    for out in hir.outputs:
        builder.output(out.name, remap[out.value], out.sgnop)
    return builder.finish()


def strength_reduce(hir: Hir) -> Hir:
    """Rewrite multiply/divide by power-of-two constants to ``Fmul2K`` and divide-by-constant to reciprocal-multiply."""
    builder = HirBuilder(hir.fmt)
    remap: dict[ValueId, ValueId] = {}
    cval: dict[ValueId, float] = {}
    k_bound = 1 << (hir.fmt.wexp - 1)  # holoso_fmul_ilog2_const requires |k| < 2**(WEXP-1)
    for old_id in sorted(hir.nodes):
        node = hir.nodes[old_id]
        match node:
            case Const(value=value):
                new_id = builder.const(value)
                cval[new_id] = value
            case Arith(op=ArithOp.MUL, a=a, b=b):
                new_id = _reduce_mul(builder, remap[a], remap[b], cval, k_bound)
            case Arith(op=ArithOp.DIV, a=a, b=b):
                new_id = _reduce_div(builder, remap[a], remap[b], cval, k_bound)
            case _:
                new_id = _copy(builder, node, remap)
        remap[old_id] = new_id
    for out in hir.outputs:
        builder.output(out.name, remap[out.value], out.sgnop)
    return builder.finish()


def _reduce_mul(builder: HirBuilder, na: ValueId, nb: ValueId, cval: dict[ValueId, float], k_bound: int) -> ValueId:
    for const_side, other in ((nb, na), (na, nb)):
        if const_side in cval:
            k = _ilog2_exact(cval[const_side])
            if k is not None and abs(k) < k_bound:
                return builder.fmul2k(other, k)
    return builder.arith(ArithOp.MUL, na, nb)


def _reduce_div(builder: HirBuilder, na: ValueId, nb: ValueId, cval: dict[ValueId, float], k_bound: int) -> ValueId:
    if nb in cval:
        c = cval[nb]
        k = _ilog2_exact(c)
        if k is not None and abs(k) < k_bound:
            return builder.fmul2k(na, -k)
        if c != 0.0 and math.isfinite(c):
            return builder.arith(ArithOp.MUL, na, builder.const(1.0 / c))
    return builder.arith(ArithOp.DIV, na, nb)


def lower_to_operators(hir: Hir) -> Hir:
    """Select hardware operators and fold sign manipulations onto operator/output sign-op ports."""
    builder = HirBuilder(hir.fmt)
    remap: dict[ValueId, ValueId] = {}
    for old_id in sorted(hir.nodes):
        node = hir.nodes[old_id]
        match node:
            case InPort(name=name):
                remap[old_id] = builder.input(name)
            case Const(value=value):
                remap[old_id] = builder.const(value)
            case SignFix():
                continue  # folded at each consumer; never emitted as a standalone node
            case Arith(op=op, a=a, b=b):
                base_a, sgn_a = _collapse_signs(hir.nodes, a)
                base_b, sgn_b = _collapse_signs(hir.nodes, b)
                kind = _ARITH_TO_OPKIND[op]
                remap[old_id] = builder.opnode(
                    kind, remap[base_a], remap[base_b], sgn_a, sgn_b, Sgnop.NONE, None, latency_of(kind, hir.fmt)
                )
            case Fmul2K(a=a, k=k):
                base_a, sgn_a = _collapse_signs(hir.nodes, a)
                remap[old_id] = builder.opnode(
                    OpKind.FMUL_ILOG2,
                    remap[base_a],
                    None,
                    sgn_a,
                    Sgnop.NONE,
                    Sgnop.NONE,
                    k,
                    latency_of(OpKind.FMUL_ILOG2, hir.fmt),
                )
            case OpNode():
                remap[old_id] = _copy(builder, node, remap)
            case _ as unreachable:
                assert_never(unreachable)
    for out in hir.outputs:
        base, sgn = _collapse_signs(hir.nodes, out.value)
        builder.output(out.name, remap[base], _compose_sgnop(sgn, out.sgnop))
    return builder.finish()


def dce(hir: Hir) -> Hir:
    """Drop nodes not reachable from any output (all input ports are retained as the module signature)."""
    reachable: set[ValueId] = set()
    stack = [out.value for out in hir.outputs]
    while stack:
        vid = stack.pop()
        if vid in reachable:
            continue
        reachable.add(vid)
        match hir.nodes[vid]:
            case OpNode(a=a, b=b):
                stack.append(a)
                if b is not None:
                    stack.append(b)
            case Arith(a=a, b=b):
                stack.append(a)
                stack.append(b)
            case SignFix(a=a):
                stack.append(a)
            case Fmul2K(a=a):
                stack.append(a)
            case _:
                pass
    keep = reachable | set(hir.input_ids)
    builder = HirBuilder(hir.fmt)
    remap: dict[ValueId, ValueId] = {}
    for old_id in sorted(keep):
        remap[old_id] = _copy(builder, hir.nodes[old_id], remap)
    for out in hir.outputs:
        builder.output(out.name, remap[out.value], out.sgnop)
    return builder.finish()


def run(hir: Hir) -> Hir:
    """Run the full pass pipeline, returning a fully lowered HIR (only InPort/Const/OpNode)."""
    return dce(lower_to_operators(strength_reduce(const_fold(hir))))
