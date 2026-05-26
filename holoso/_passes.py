"""
HIR -> HIR optimization and lowering passes.

Pipeline (:func:`run`): constant folding -> strength reduction -> operator selection with sign folding -> dead-code
elimination. The result is a fully lowered HIR (only ``InPort``/``Const``/``OpNode``), ready for scheduling.

FP math is non-associative; these passes may change results in the last bits, which is accepted (see DESIGN.md).
"""

import math
from typing import assert_never

from ._hir import (
    ADD,
    DIV,
    MUL,
    Arith,
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
from ._operators import OpConfig, Sgnop
from ._type import FloatFormat

# ----------------------------------------------------------------------------------------------------------------------
# Small numeric helpers


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
        sgnop = _compose_sgnop(sgnop, op.sgnop)
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
        case OpNode(op=op, a=a, b=b, a_sgnop=asg, b_sgnop=bsg, y_sgnop=ysg):
            nb = remap[b] if b is not None else None
            return builder.opnode(op, remap[a], nb, asg, bsg, ysg)
        case _ as unreachable:
            assert_never(unreachable)


# ----------------------------------------------------------------------------------------------------------------------
# Passes


def _const_fold(hir: Hir) -> Hir:
    """Fold operations whose operands are all constants into a single constant."""
    builder = HirBuilder()
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
                and (folded := op.evaluate(cval[remap[a]], cval[remap[b]])) is not None
            ):
                new_id = builder.const(folded)
                cval[new_id] = folded
            case SignFix(op=op, a=a) if remap[a] in cval:
                folded_sign = op.evaluate(cval[remap[a]])
                new_id = builder.const(folded_sign)
                cval[new_id] = folded_sign
            case _:
                new_id = _copy(builder, node, remap)
        remap[old_id] = new_id
    for out in hir.outputs:
        builder.output(out.name, remap[out.value], out.sgnop)
    return builder.finish()


def _strength_reduce(hir: Hir, fmt: FloatFormat) -> Hir:
    """Rewrite multiply/divide by power-of-two constants to ``Fmul2K`` and divide-by-constant to reciprocal-multiply."""
    builder = HirBuilder()
    remap: dict[ValueId, ValueId] = {}
    cval: dict[ValueId, float] = {}
    k_bound = 1 << (fmt.wexp - 1)  # holoso_fmul_ilog2_const requires |k| < 2**(WEXP-1)
    for old_id in sorted(hir.nodes):
        node = hir.nodes[old_id]
        match node:
            case Const(value=value):
                new_id = builder.const(value)
                cval[new_id] = value
            case Arith(op=op, a=a, b=b) if op is MUL:
                new_id = _reduce_mul(builder, remap[a], remap[b], cval, k_bound)
            case Arith(op=op, a=a, b=b) if op is DIV:
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
    return builder.arith(MUL, na, nb)


def _reduce_div(builder: HirBuilder, na: ValueId, nb: ValueId, cval: dict[ValueId, float], k_bound: int) -> ValueId:
    if nb in cval:
        c = cval[nb]
        k = _ilog2_exact(c)
        if k is not None and abs(k) < k_bound:
            return builder.fmul2k(na, -k)
        if c != 0.0 and math.isfinite(c):
            return builder.arith(MUL, na, builder.const(1.0 / c))
    return builder.arith(DIV, na, nb)


def _lower_to_operators(hir: Hir, ops: OpConfig) -> Hir:
    """
    Select hardware operators from the configuration and fold sign manipulations onto operator/output sign-op ports.

    Each program op picks its operator by identity (ADD -> ``ops.fadd``, MUL -> ``ops.fmul``, DIV -> ``ops.fdiv``, a
    power-of-two scaling -> ``ops.fmul_ilog2.instantiate(k)``); the chosen operator instance is baked into the OpNode,
    so its latency and instantiation params travel with the node and cannot drift.
    """
    builder = HirBuilder()
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
                hw = ops.fadd if op is ADD else ops.fmul if op is MUL else ops.fdiv
                remap[old_id] = builder.opnode(hw, remap[base_a], remap[base_b], sgn_a, sgn_b, Sgnop.NONE)
            case Fmul2K(a=a, k=k):
                base_a, sgn_a = _collapse_signs(hir.nodes, a)
                remap[old_id] = builder.opnode(
                    ops.fmul_ilog2.instantiate(k), remap[base_a], None, sgn_a, Sgnop.NONE, Sgnop.NONE
                )
            case OpNode():
                remap[old_id] = _copy(builder, node, remap)
            case _ as unreachable:
                assert_never(unreachable)
    for out in hir.outputs:
        base, sgn = _collapse_signs(hir.nodes, out.value)
        builder.output(out.name, remap[base], _compose_sgnop(sgn, out.sgnop))
    return builder.finish()


def _dce(hir: Hir) -> Hir:
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
    builder = HirBuilder()
    remap: dict[ValueId, ValueId] = {}
    for old_id in sorted(keep):
        remap[old_id] = _copy(builder, hir.nodes[old_id], remap)
    for out in hir.outputs:
        builder.output(out.name, remap[out.value], out.sgnop)
    return builder.finish()


def run(hir: Hir, ops: OpConfig) -> Hir:
    """Run the full pass pipeline, returning a fully lowered HIR (only InPort/Const/OpNode)."""
    return _dce(_lower_to_operators(_strength_reduce(_const_fold(hir), ops.float_format), ops))
