"""
The float format, written back into HIR. A constant the optimizer minted may exceed what the format holds even
where every constant the kernel wrote fits; carrying it as a significand and an exponent costs one more operation
and no accuracy the single constant would have had, since it has none.
"""

import logging

from .._hir import (
    FloatConst,
    FloatMul,
    FloatMulPow2,
    Hir,
    HirBuilder,
    Node,
    Operation,
    Scaling,
    copy_node,
    eliminate_dead_code,
    rebuild,
    scaling_of,
)
from .._operators import OpConfig
from .._type import FloatFormat
from .._util import ValueId

_logger = logging.getLogger(__name__)


def _degrades(fmt: FloatFormat, value: float) -> bool:
    bits = fmt.encode(value)
    return bits == 0 or not fmt.is_finite(bits)


def _reaches(fmt: FloatFormat, k: int) -> bool:
    """
    Whether scaling by `2**k` maps any representable magnitude back onto a representable one -- the format's own
    exponent span is the reach. Past it no operand the machine can hold produces a result it can hold, so splitting
    would only compute the zero or the infinity more slowly, and the constant is worth refusing over after all.
    """
    return abs(k) <= (1 << fmt.wexp) - 3


def rescale(hir: Hir, ops: OpConfig) -> Hir:
    """
    Split every multiplication by a constant this format cannot hold into one by its significand, which every format
    holds, and one by its exponent, which no format bounds. Runs once after optimization has settled: strength
    reduction would compose the pair straight back into the constant that does not fit.

    Declined without the exponent scaler to split into, and declined for a scale beyond the format's reach, so the
    kernel is refused over the constant it cannot hold -- which names the trouble and the way out -- rather than over
    an operator it never asked for, or built into a slower way to answer zero.
    """
    if ops.fmul_ilog2 is None:
        return hir
    fmt = ops.float_format
    rewrites = 0

    def split(builder: HirBuilder, base: ValueId, scaling: Scaling) -> ValueId:
        # The significand only ever amplifies, so it goes on the smaller-magnitude side of the exponent step, where
        # the least is at stake against the format's rails.
        assert scaling.k != 0, "a scaling by the significand alone is representable, so it cannot have degraded"
        assert not scaling.is_power_of_two, "an exact power of two is absorbed into the scaler and never materializes"
        significand = builder.float_const(scaling.signed_significand)
        if scaling.k > 0:
            return builder.operation(FloatMulPow2(scaling.k), [builder.operation(FloatMul(), [base, significand])])
        return builder.operation(FloatMul(), [builder.operation(FloatMulPow2(scaling.k), [base]), significand])

    def build_value(builder: HirBuilder, node: Node, remap: dict[ValueId, ValueId]) -> ValueId:
        nonlocal rewrites
        if isinstance(node, Operation) and isinstance(node.operator, FloatMul):
            assert len(node.operands) == 2
            for base, other in (node.operands, node.operands[::-1]):
                constant = hir.nodes[other]
                if isinstance(constant, FloatConst) and _degrades(fmt, constant.value):
                    scaling = scaling_of(constant.value)
                    if scaling is not None and _reaches(fmt, scaling.k):
                        rewrites += 1
                        return split(builder, remap[base], scaling)
        return copy_node(builder, node, remap)

    result = rebuild(hir, build_value)
    if not rewrites:
        return hir
    _logger.info("Rescaling: %d constant multiplier(s) carried as a significand and an exponent in %s", rewrites, fmt)
    # The constant each split replaced is now unreferenced, and an unreferenced constant still materializes.
    return eliminate_dead_code(result)
