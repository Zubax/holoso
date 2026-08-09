"""
The machine word, written back into HIR. Answering it at selection instead would strand the constant there, where a
second, poorer copy of the optimizer would have to chase what it enables.
"""

import logging

from .._hir import (
    Hir,
    HirBuilder,
    IntConst,
    IntShiftLeft,
    Node,
    Operation,
    copy_node,
    rebuild,
)
from .._type import IntFormat
from .._util import ValueId

_logger = logging.getLogger(__name__)


def constant_shift_count(hir: Hir, count: ValueId) -> int | None:
    node = hir.nodes[count]
    return node.value if isinstance(node, IntConst) else None


def left_shifts(hir: Hir) -> int:
    """Bounds the caller's fixpoint: every substitution deletes one, and no pass mints one."""
    return sum(
        1 for node in hir.nodes.values() if isinstance(node, Operation) and isinstance(node.operator, IntShiftLeft)
    )


def specialize(hir: Hir, int_format: IntFormat) -> Hir | None:
    """
    ``None`` ends the caller's fixpoint. A rule belongs here only where its answer is independent of every operand,
    since HIR integers are unbounded and a later round can reveal one as a constant no word holds: a left shift past
    the word is zero whatever it shifts, where a right shift past it repeats the sign fill only for a value the word
    already holds -- so that clamp stays at lowering. A negative count and ``imul_pow2`` settle nothing here either.
    """
    width = int_format.width
    assert width >= 2
    substitutions = 0

    def build_value(builder: HirBuilder, node: Node, remap: dict[ValueId, ValueId]) -> ValueId:
        # Counts are read off the ORIGINAL graph: this pass folds nothing, so a constant there is a constant here.
        nonlocal substitutions
        match node:
            case Operation(operator=IntShiftLeft(), operands=(_, count)) if (
                shamt := constant_shift_count(hir, count)
            ) is not None and shamt >= width:
                substitutions += 1
                return builder.const_node(IntConst(0))
            case _:
                return copy_node(builder, node, remap)

    specialized = rebuild(hir, build_value)
    if not substitutions:
        return None
    _logger.info("Shift specialization: %d constant count(s) settled by the %d-bit word", substitutions, width)
    return specialized
