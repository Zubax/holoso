"""
Reciprocal sharing: a divide whose divisor already has a reciprocal is a multiply by it.

The dual of the constant-divisor rule in strength reduction, for a divisor the compiler cannot see.
Runs on a settled graph rather than inside the reduction round: a reciprocal about to die must not be adopted,
or one divide becomes a divide and a multiply.
"""

import logging

from ._const import FloatConst
from ._copy import copy_node, rebuild
from .._util import BlockId, ValueId
from ._ir import Hir, HirBuilder, Node, Operation
from ._operators import FloatDiv, FloatMul

_logger = logging.getLogger(__name__)


def _is_unit(hir: Hir, vid: ValueId) -> bool:
    node = hir.nodes[vid]
    return isinstance(node, FloatConst) and node.value == 1.0


def _unit_reciprocals(hir: Hir) -> set[tuple[BlockId, ValueId]]:
    """
    Block-scoped to match the builder's interning, so emitting one here shares the existing node rather than minting
    a second divider. Conservative: a reciprocal in a dominating block would serve too, and is not looked for.
    """
    found: set[tuple[BlockId, ValueId]] = set()
    for block in hir.blocks:
        for vid in block.operations:
            node = hir.nodes[vid]
            if isinstance(node, Operation) and isinstance(node.operator, FloatDiv) and _is_unit(hir, node.operands[0]):
                found.add((block.id, node.operands[1]))
    return found


def run(hir: Hir) -> Hir:
    """
    `x/y` becomes `x * (1/y)` where the reciprocal is already computed, and `1/(p*q)` becomes `(1/p)*(1/q)` where
    both factors have one -- both, or it would trade one divide for another divide and a multiply.

    The second is a fastmath license rather than ordinary algebra: at `p=0, q=inf` the left is `1/(0*inf)`, hence
    an infinity, where the right is `inf*0`, hence zero.
    """
    reciprocals = _unit_reciprocals(hir)
    if not reciprocals:
        return hir
    rewrites = 0
    built: dict[tuple[BlockId, ValueId], ValueId] = {}

    def splits(block: BlockId, divisor: ValueId) -> tuple[ValueId, ValueId] | None:
        node = hir.nodes[divisor]
        if isinstance(node, Operation) and isinstance(node.operator, FloatMul):
            p, q = node.operands
            if (block, p) in reciprocals and (block, q) in reciprocals:
                return p, q
        return None

    def build_reciprocal(builder: HirBuilder, divisor: ValueId, remap: dict[ValueId, ValueId]) -> ValueId:
        """
        Every asker builds it the same way, so a consumer rebuilt before the reciprocal itself still lands on the
        one node.

        Iterative and memoized across the pass, all three for real kernels: an unrolled loop nests products past
        the interpreter's recursion limit, repeated squaring shares a factor with itself so an unmemoized walk goes
        exponential, and each of N nested divisors asks for the chain the one below it already built.
        """
        block = builder.current_block
        stack: list[tuple[ValueId, bool]] = [(divisor, False)]
        while stack:
            current, expanded = stack.pop()
            if (block, current) in built:
                continue
            split = splits(block, current)
            if split is None:
                built[block, current] = builder.operation(
                    FloatDiv(), [builder.const_node(FloatConst(1.0)), remap[current]]
                )
            elif expanded:
                built[block, current] = builder.operation(FloatMul(), [built[block, split[0]], built[block, split[1]]])
            else:
                stack.append((current, True))
                stack.extend((factor, False) for factor in split)
        return built[block, divisor]

    def build_value(builder: HirBuilder, node: Node, remap: dict[ValueId, ValueId]) -> ValueId:
        nonlocal rewrites
        if isinstance(node, Operation) and isinstance(node.operator, FloatDiv):
            numerator, divisor = node.operands
            block = builder.current_block
            if _is_unit(hir, numerator):
                if splits(block, divisor) is not None:
                    rewrites += 1
                    return build_reciprocal(builder, divisor, remap)
            elif (block, divisor) in reciprocals:
                rewrites += 1
                return builder.operation(FloatMul(), [remap[numerator], build_reciprocal(builder, divisor, remap)])
        return copy_node(builder, node, remap)

    result = rebuild(hir, build_value)
    if not rewrites:
        return hir
    _logger.info("Reciprocal sharing: %d division(s) answered from a shared reciprocal", rewrites)
    return result
