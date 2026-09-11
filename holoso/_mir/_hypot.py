"""The standalone hypotenuse: which ones an adjacent atan2 carries, and the expansion for the rest."""

import logging

from .._errors import UnsupportedConstruct
from .._hir import (
    FloatAdd,
    FloatAtan2Turns,
    FloatHypot2,
    FloatILog2,
    FloatMul,
    FloatMulPow2Dynamic,
    FloatSqrt,
    Hir,
    HirBuilder,
    IntGreater,
    IntNeg,
    IntSelect,
    IntSub,
    Node,
    Operation,
    copy_node,
    rebuild,
)
from .._operators import OpConfig
from .._type import FloatFormat
from .._util import ValueId
from ._signs import collapse_signs

_logger = logging.getLogger(__name__)


def _operand_base_set(hir: Hir, node: Operation) -> tuple[ValueId, ...]:
    """
    Order-independent: hypot is commutative and sign-invariant, as is the fatan2 magnitude it taps, so a hypot fuses
    with any same-block atan2 over the same value pair.
    """
    return tuple(sorted(collapse_signs(hir.nodes, operand)[0] for operand in node.operands))


def plan_fusions(hir: Hir, ops: OpConfig) -> dict[ValueId, ValueId]:
    """
    Map each FloatHypot2 to a same-block FloatAtan2Turns over the same value pair so MIR can tap the atan2's magnitude
    port (the two fuse into one CORDIC) rather than decompose into primitives. Block-local, like the LIR firing fusion
    it feeds.
    """
    if ops.fatan2 is None:
        return {}
    plans: dict[ValueId, ValueId] = {}
    for block in hir.blocks:
        atan2_by_pair: dict[tuple[ValueId, ...], ValueId] = {}
        for vid in block.operations:
            if isinstance(node := hir.nodes[vid], Operation) and isinstance(node.operator, FloatAtan2Turns):
                atan2_by_pair.setdefault(_operand_base_set(hir, node), vid)
        for vid in block.operations:
            if isinstance(node := hir.nodes[vid], Operation) and isinstance(node.operator, FloatHypot2):
                match = atan2_by_pair.get(_operand_base_set(hir, node))
                if match is not None:
                    plans[vid] = match
    if plans:
        _logger.info("Hypotenuse: %d fused onto an adjacent atan2's magnitude port", len(plans))
    return plans


def count(hir: Hir) -> int:
    return sum(isinstance(n, Operation) and isinstance(n.operator, FloatHypot2) for n in hir.nodes.values())


def _scaling_exponent(fmt: FloatFormat) -> tuple[int, int]:
    """
    The bias, and the exponent the expansion normalizes the larger operand to. Bounded both ways -- the dominant
    square must stay normal, `2t >= 1 - bias`, and the sum must stay finite, `2t + 3 <= bias` -- and the window
    they leave is empty exactly at `wexp < 3`, which the caller refuses.
    """
    bias = (1 << (fmt.wexp - 1)) - 1
    return bias, (bias - 3) // 2


def expand_unfused(hir: Hir, ops: OpConfig) -> Hir | None:
    """
    Rewrite each hypotenuse no adjacent atan2 will carry into `2^-k * sqrt((x*2^k)^2 + (y*2^k)^2)`. Being exact,
    the scaling cannot overflow the square; the smaller leg's square still underflows once the significand outruns
    the exponent range, a loss the written form shares and exceeds. Sign chains are dropped rather than folded, so
    two spellings of one magnitude expand alike and can cancel -- which the orphan case turns on.

    Fusion is planned here rather than by the caller because it must be re-planned on every round: an expansion can
    let an expression cancel, and the cancellation can delete the atan2 another hypotenuse was to fuse with.
    """
    fused, fmt = plan_fusions(hir, ops), ops.float_format
    # Operations intern per block, so a block and a node name one value -- which is how `BuildValue`, handed the
    # node and not its id, recognizes a target.
    targets = {
        (block.id, node)
        for block in hir.blocks
        for vid in block.operations
        if isinstance(node := hir.nodes[vid], Operation) and isinstance(node.operator, FloatHypot2) and vid not in fused
    }
    if not targets:
        return None
    if fmt.wexp < 3:
        raise UnsupportedConstruct(
            f"a standalone hypotenuse needs an exponent range that holds a normalized operand's square, which "
            f"{fmt} has not; widen wexp"
        )
    bias, scale = _scaling_exponent(fmt)  # the window's top leaves the smaller operand the most room to flush

    def build_value(builder: HirBuilder, node: Node, remap: dict[ValueId, ValueId]) -> ValueId:
        if (builder.current_block, node) not in targets:
            return copy_node(builder, node, remap)
        assert isinstance(node, Operation) and len(node.operands) == 2  # the integer max below is binary
        legs = [remap[collapse_signs(hir.nodes, operand)[0]] for operand in node.operands]  # signs dropped
        exponents = [builder.operation(FloatILog2(bias), [leg]) for leg in legs]
        larger = builder.operation(
            IntSelect(), [builder.operation(IntGreater(), exponents), exponents[0], exponents[1]]
        )
        k = builder.operation(IntSub(), [builder.int_const(scale), larger])
        squares = [
            builder.operation(FloatMul(), [scaled, scaled])
            for scaled in (builder.operation(FloatMulPow2Dynamic(), [leg, k]) for leg in legs)
        ]
        root = builder.operation(FloatSqrt(), [builder.operation(FloatAdd(), squares)])
        return builder.operation(FloatMulPow2Dynamic(), [root, builder.operation(IntNeg(), [k])])

    _logger.info("Hypotenuse: %d expanded by exact exponent scaling at 2^%d", len(targets), scale)
    return rebuild(hir, build_value)
