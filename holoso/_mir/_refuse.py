"""The final gate over HIR, immediately before hardware is selected for it."""

import logging

from .._errors import SynthesisError
from .._hir import BoolConst, Branch, Const, Hir, NoNumber, Operation, reverse_postorder, successors
from .._util import BlockId, ValueId

_logger = logging.getLogger(__name__)


def refuse(hir: Hir) -> None:
    """
    Refuse the build for an operation that survived every pass and still names no number: a quotient by zero, a
    logarithm of a non-positive, an indeterminate form. SURVIVING is the criterion -- unrolling and inlining
    substitute values, so the compiler manufactures such expressions itself, and every one it deletes was never part
    of the program. Running it ahead of a reduction would silently stop convicting things.

    Only the blocks the sequencer can ENTER are asked. Pruning leaves exactly one shape where that differs from CFG
    reachability: the never-returning loop, whose exit it declines to orphan.
    """
    known: dict[ValueId, Const] = {vid: node for vid, node in hir.nodes.items() if isinstance(node, Const)}
    blocks = {block.id: block for block in hir.blocks}
    enterable: set[BlockId] = {hir.entry}
    for bid in reverse_postorder(hir):
        if bid not in enterable:
            continue
        block = blocks[bid]
        for vid in block.operations:
            node = hir.nodes[vid]
            assert isinstance(node, Operation)
            operands = [const for o in node.operands if (const := known.get(o)) is not None]
            if len(operands) != len(node.operands):
                # An operand this walk cannot name leaves the operation unnamed, hence unconvicted. Reverse postorder
                # is what makes the walk see as much as it can -- an operand defined by an operation dominates this use
                # and so sits in a block already walked -- but seeing less only ever costs a diagnostic the charter
                # never promised, which is why nothing here asserts how far it reached.
                continue
            try:
                known[vid] = node.operator.evaluate(operands)
            except NoNumber as signal:
                raise SynthesisError(
                    f"{signal.what} names no number, so the build is refused rather than synthesized -- asking what "
                    "the hardware would do with it instead is not the compiler's business. It is refused because it "
                    "SURVIVED optimization: no identity erased it, no guard excluded it, and nothing left it dead, so "
                    "it is part of the program. HIR carries no source positions, so locate the expression in the "
                    "kernel by hand."
                ) from None
        terminator = block.terminator
        if isinstance(terminator, Branch) and isinstance(condition := known.get(terminator.cond), BoolConst):
            enterable.add(terminator.if_true if condition.value else terminator.if_false)
        else:
            enterable.update(successors(block))
    _logger.debug("Survivor refusal sweep: %d of %d block(s) enterable", len(enterable), len(hir.blocks))
