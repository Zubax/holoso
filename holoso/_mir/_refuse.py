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
)


def refuse(hir: Hir) -> None:
    """
    Judge what outlived optimization. Every clause here reads the graph alone; what depends on a decision selection
    makes -- whether a constant materializes, which hardware serves an exponent -- is refused where that decision is
    made, or the two would drift.

    SURVIVING is the criterion: unrolling, inlining and substitution manufacture expressions the compiler then
    deletes, and convicting one of those would answer for the compiler's own transformation.
    """
    for node in hir.nodes.values():
        if not isinstance(node, Operation):
            continue
        operands = [const for o in node.operands if isinstance(const := hir.nodes[o], Const)]
        if len(operands) != len(node.operands):
            # An operand the gate cannot name leaves the operation unnamed, hence unconvicted. The graph is at the
            # optimizer's fixpoint, which folds every operation over known operands that names a number, so no chain
            # of folds is left for the gate to follow.
            _refuse_negative_shift(hir, node)
            continue
        try:
            node.operator.evaluate(operands)
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
