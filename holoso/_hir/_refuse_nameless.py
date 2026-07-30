"""HIR survivor refusal: the operations that outlived optimization and still name no number."""

import logging

from .._errors import SynthesisError
from .._util import BlockId, ValueId
from ._const import BoolConst, Const
from ._copy import reverse_postorder
from ._ir import Branch, Hir, Operation, successors
from ._operators import NoNumber

_logger = logging.getLogger(__name__)


def run(hir: Hir) -> None:
    """
    Refuse the build for an operation that survived every pass and still names no number: a quotient by zero, a
    logarithm of a non-positive, an indeterminate form. It runs last because SURVIVING is the criterion -- unrolling
    and inlining substitute values, so the compiler manufactures such expressions itself, and every one it deletes was
    never part of the program. What is left after the deletions is what the kernel asked for.

    The same question the passes ask, asked once more over the same knowledge, so nothing is proven here that was not
    provable there. That the sweep knows no more than the passes is what keeps it from convicting an expression a
    further round would have erased.
    That it knows ENOUGH is a property of WHERE it runs: on the already-reduced graph every constant an identity
    established is a ``Const`` node, so a single forward walk sees what the fixpoint took several rounds to establish.
    Moving this ahead of a reduction would silently stop convicting things.

    Only the blocks the sequencer can ENTER are asked. A guard that proved its own arm dead must not convict what it
    excluded, and post-DCE block liveness cannot say that: every block left is CFG-reachable, and nothing rewrites a
    proven branch into a jump, so a constant condition can still guard a block liveness calls live. So the walk carries
    its own constants and takes only the successor a proven condition selects.
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
