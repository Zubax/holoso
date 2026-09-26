"""The standalone magnitude: which ones an adjacent atan2 carries, and the expansion for the rest."""

import logging
from collections.abc import Callable

from .._errors import UnsupportedConstruct
from .._hir import (
    FloatAdd,
    FloatAtan2Turns,
    FloatHypot,
    FloatILog2,
    FloatMul,
    FloatMulPow2Dynamic,
    FloatSqrt,
    Hir,
    HirBuilder,
    IntComparison,
    Relation,
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
    Order-independent: a magnitude is commutative and sign-invariant, as is the fatan2 magnitude it taps, so a
    two-legged one fuses with any same-block atan2 over the same value pair.
    """
    return tuple(sorted(collapse_signs(hir.nodes, operand)[0] for operand in node.operands))


def _pair(hir: Hir, vid: ValueId) -> Operation | None:
    node = hir.nodes[vid]
    if isinstance(node, Operation) and isinstance(node.operator, FloatHypot) and node.operator.arity == 2:
        return node
    return None


def plan_fusions(hir: Hir, ops: OpConfig) -> dict[ValueId, ValueId]:
    """
    Map each two-legged FloatHypot to a same-block FloatAtan2Turns over the same value pair so MIR can tap the atan2's
    magnitude port (the two fuse into one CORDIC) rather than decompose into primitives. Block-local, like the LIR
    firing fusion it feeds; a pair is the only arity that port carries.
    """
    if ops.options.fatan2 is None:
        return {}
    plans: dict[ValueId, ValueId] = {}
    for block in hir.blocks:
        atan2_by_pair: dict[tuple[ValueId, ...], ValueId] = {}
        for vid in block.operations:
            if isinstance(node := hir.nodes[vid], Operation) and isinstance(node.operator, FloatAtan2Turns):
                atan2_by_pair.setdefault(_operand_base_set(hir, node), vid)
        for vid in block.operations:
            if (pair := _pair(hir, vid)) is not None:
                match = atan2_by_pair.get(_operand_base_set(hir, pair))
                if match is not None:
                    plans[vid] = match
    if plans:
        _logger.info("Magnitude: %d fused onto an adjacent atan2's magnitude port", len(plans))
    return plans


def count(hir: Hir) -> int:
    return sum(isinstance(n, Operation) and isinstance(n.operator, FloatHypot) for n in hir.nodes.values())


def _bias(fmt: FloatFormat) -> int:
    return (1 << (fmt.wexp - 1)) - 1


def _scaling_exponent(fmt: FloatFormat, arity: int) -> int | None:
    """
    The exponent the expansion normalizes the LARGEST leg to, or None where the window is empty. The dominant square
    must stay normal, `2t >= 1 - bias`; and each scaled square being below `2^(2t+2)`, `n` of them stay under
    `2^(2t + 2 + ceil(log2 n))`, which held to `bias` leaves the sum a binade below overflow.
    """
    assert arity >= 2
    bias = _bias(fmt)
    top = (bias - 2 - (arity - 1).bit_length()) // 2
    return top if 2 * top >= 1 - bias else None


def _tree(values: list[ValueId], combine: Callable[[ValueId, ValueId], ValueId]) -> ValueId:
    """Balanced, so the reduction is `ceil(log2 n)` operators deep where a fold would be `n-1`."""
    assert values
    while len(values) > 1:
        pairs = range(0, len(values), 2)
        values = [combine(values[i], values[i + 1]) if i + 1 < len(values) else values[i] for i in pairs]
    return values[0]


def expand_unfused(hir: Hir, ops: OpConfig) -> Hir | None:
    """
    Rewrite each magnitude no adjacent atan2 will carry into `2^-k * sqrt(sum((x_i*2^k)^2))`. Being exact, the
    scaling cannot overflow the dominant square; a smaller leg's square still underflows once the significand
    outruns the exponent range, a loss the written form shares and exceeds. Sign chains are dropped rather than
    folded, so two spellings of one magnitude expand alike and can cancel.

    Fusion is planned here rather than by the caller because it must be re-planned on every round: an expansion can
    let an expression cancel, and the cancellation can delete the atan2 another magnitude was to fuse with.
    """
    fused, fmt = plan_fusions(hir, ops), ops.float_format
    targets = {
        vid: node
        for vid, node in hir.nodes.items()
        if isinstance(node, Operation) and isinstance(node.operator, FloatHypot) and vid not in fused
    }
    if not targets:
        return None
    bias = _bias(fmt)
    # The window's top leaves the smaller legs the most room to flush, and it narrows with the arity. Sorted, so a
    # graph holding several refusable arities names the shortest vector it cannot hold.
    scales: dict[int, int] = {}
    for arity in sorted({len(node.operands) for node in targets.values()}):
        scale = _scaling_exponent(fmt, arity)
        if scale is None:
            raise UnsupportedConstruct(
                f"a standalone magnitude over {arity} operands needs an exponent range that holds their scaled "
                f"squares, which {fmt} has not; widen wexp or shorten the vector"
            )
        scales[arity] = scale

    def build_value(builder: HirBuilder, vid: ValueId, node: Node, remap: dict[ValueId, ValueId]) -> ValueId:
        if vid not in targets:
            return copy_node(builder, node, remap)
        assert isinstance(node, Operation)
        assert len(node.operands) >= 2  # strength reduction answers a lone leg with the absolute value
        scale = scales[len(node.operands)]
        legs = [remap[collapse_signs(hir.nodes, operand)[0]] for operand in node.operands]
        exponents = [builder.operation(FloatILog2(bias), [leg]) for leg in legs]
        largest = _tree(
            exponents,
            lambda a, b: builder.operation(IntSelect(), [builder.operation(IntComparison(Relation.GT), [a, b]), a, b]),
        )
        k = builder.operation(IntSub(), [builder.int_const(scale), largest])
        squares = [
            builder.operation(FloatMul(), [scaled, scaled])
            for scaled in (builder.operation(FloatMulPow2Dynamic(), [leg, k]) for leg in legs)
        ]
        summed = _tree(squares, lambda a, b: builder.operation(FloatAdd(), [a, b]))
        root = builder.operation(FloatSqrt(), [summed])
        return builder.operation(FloatMulPow2Dynamic(), [root, builder.operation(IntNeg(), [k])])

    _logger.info("Magnitude: %d expanded by exact exponent scaling; scale per arity: %s", len(targets), scales)
    return rebuild(hir, build_value)
