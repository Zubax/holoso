"""
The trigonometric unit, written back into HIR: the cores count angles in turns, and only radians are spelled.
"""

import logging
import math

from .._hir import (
    FloatAtan2,
    FloatAtan2Turns,
    FloatCos,
    FloatCosTurns,
    FloatMul,
    FloatSin,
    FloatSinTurns,
    Hir,
    HirBuilder,
    Node,
    Operation,
    copy_node,
    rebuild,
)
from .._util import ValueId

_logger = logging.getLogger(__name__)


def trig_abi(hir: Hir) -> Hir:
    """
    Restate every radian trigonometric operation over the turn-native vocabulary, conserving its meaning by an
    explicit scaling that the optimizer is then free to fold against whatever the kernel wrote next to it. Answering
    this at selection instead would strand that scaling past every fold, where a phase already counted in turns pays
    two multiplies and two roundings to compute an identity.

    Running before optimization rather than after also treats every trigonometric operand alike: one that only a later
    round reveals as constant folds through the same conversion as one constant from the start.
    """
    rewrites = 0

    def build_value(builder: HirBuilder, node: Node, remap: dict[ValueId, ValueId]) -> ValueId:
        nonlocal rewrites
        match node:
            case Operation(operator=FloatSin() | FloatCos() as semantic, operands=(a,)):
                rewrites += 1
                turns = builder.operation(FloatMul(), [remap[a], builder.float_const(1.0 / math.tau)])
                turned = FloatSinTurns() if isinstance(semantic, FloatSin) else FloatCosTurns()
                return builder.operation(turned, [turns])
            case Operation(operator=FloatAtan2(), operands=(y, x)):
                rewrites += 1
                turns = builder.operation(FloatAtan2Turns(), [remap[y], remap[x]])
                return builder.operation(FloatMul(), [turns, builder.float_const(math.tau)])
            case _:
                return copy_node(builder, node, remap)

    result = rebuild(hir, build_value)
    _logger.info("Trigonometric ABI: %d radian operation(s) restated over turns", rewrites)
    return result
