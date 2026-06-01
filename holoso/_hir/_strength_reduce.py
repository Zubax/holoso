"""HIR algebraic strength reduction."""

import math

from ._copy import copy_node, copy_state_slots
from ._const import FloatConst
from ._ir import Hir, HirBuilder, Operation, ValueId
from ._operators import FloatDiv, FloatMul, FloatMulPow2


def _ilog2_exact(c: float) -> int | None:
    """Return ``k`` if ``c == 2**k`` exactly for a positive ``c``, else ``None``."""
    if c <= 0.0 or not math.isfinite(c):
        return None
    mantissa, exponent = math.frexp(c)  # c == mantissa * 2**exponent, mantissa in [0.5, 1)
    return exponent - 1 if mantissa == 0.5 else None


def run(hir: Hir) -> Hir:
    """Rewrite exact power-of-two scaling and finite constant division before hardware selection."""
    builder = HirBuilder()
    remap: dict[ValueId, ValueId] = {}
    cval: dict[ValueId, float] = {}
    for old_id in sorted(hir.nodes):
        node = hir.nodes[old_id]
        match node:
            case FloatConst(value=value):
                new_id = builder.float_const(value)
                cval[new_id] = value
            case Operation(operator=FloatMul(), operands=(a, b)):
                new_id = _reduce_mul(builder, remap[a], remap[b], cval)
            case Operation(operator=FloatDiv(), operands=(a, b)):
                new_id = _reduce_div(builder, remap[a], remap[b], cval)
            case _:
                new_id = copy_node(builder, node, remap)
        remap[old_id] = new_id
    for out in hir.outputs:
        builder.output(out.name, remap[out.value])
    copy_state_slots(builder, hir, remap)
    return builder.finish()


def _reduce_mul(builder: HirBuilder, a: ValueId, b: ValueId, cval: dict[ValueId, float]) -> ValueId:
    for const_side, other in ((b, a), (a, b)):
        if const_side in cval:
            k = _ilog2_exact(cval[const_side])
            if k is not None:
                return builder.operation(FloatMulPow2(k), [other])
    return builder.operation(FloatMul(), [a, b])


def _reduce_div(builder: HirBuilder, a: ValueId, b: ValueId, cval: dict[ValueId, float]) -> ValueId:
    if b in cval:
        c = cval[b]
        k = _ilog2_exact(c)
        if k is not None:
            return builder.operation(FloatMulPow2(-k), [a])
        if c != 0.0 and math.isfinite(c):
            return builder.operation(FloatMul(), [a, builder.float_const(1.0 / c)])
    return builder.operation(FloatDiv(), [a, b])
