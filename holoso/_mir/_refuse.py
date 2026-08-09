"""The final gate over HIR, immediately before hardware is selected for it."""

from .._errors import SynthesisError, UnsupportedConstruct
from .._hir import (
    Const,
    Hir,
    IntConst,
    IntShiftLeft,
    IntShiftRight,
    NoNumber,
    Operation,
    reverse_postorder,
)
from .._util import ValueId


def refuse(hir: Hir) -> None:
    """
    Judge what outlived optimization. Every clause here reads the graph alone; what depends on a decision selection
    makes -- whether a constant materializes, which hardware serves an exponent -- is refused where that decision is
    made, or the two would drift.

    SURVIVING is the criterion: unrolling, inlining and substitution manufacture expressions the compiler then
    deletes, and convicting one of those would answer for the compiler's own transformation.
    """
    for slot in hir.state_slots:
        if slot.reset_value.type != hir.nodes[slot.live_out].type:
            raise UnsupportedConstruct(
                f"state slot {slot.name!r} holds {hir.nodes[slot.live_out].type} "
                f"but resets to {slot.reset_value.type}"
            )
    _refuse_operations(hir)


def _refuse_operations(hir: Hir) -> None:
    known: dict[ValueId, Const] = {vid: node for vid, node in hir.nodes.items() if isinstance(node, Const)}
    blocks = {block.id: block for block in hir.blocks}
    for bid in reverse_postorder(hir):
        for vid in blocks[bid].operations:
            node = hir.nodes[vid]
            assert isinstance(node, Operation)
            operands = [const for o in node.operands if (const := known.get(o)) is not None]
            if len(operands) != len(node.operands):
                _refuse_negative_shift(hir, node)
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


def _refuse_negative_shift(hir: Hir, node: Operation) -> None:
    """
    The shifter would silently read a negative count as the other direction -- a wrong answer, not a rail. Asked only
    where the fold could not reach: with both operands known it is the fold that names the verdict.
    """
    if not isinstance(node.operator, (IntShiftLeft, IntShiftRight)):
        return
    count = hir.nodes[node.operands[1]]
    if isinstance(count, IntConst) and count.value < 0:
        raise UnsupportedConstruct(f"shift count {count.value} is negative; Python has no such shift")
