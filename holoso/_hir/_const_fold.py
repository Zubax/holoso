"""HIR constant folding."""

from ._const import Const
from ._copy import copy_node, rebuild
from .._util import ValueId
from ._ir import Hir, HirBuilder, Node, Operation, Phi


def run(hir: Hir) -> Hir:
    """
    Fold every constant expression to a single constant: an operation whose operands are all constant, an operation
    with an absorbing constant operand (``x or True`` -> True, ``x and False`` -> False), and a phi all of whose arms
    are the same constant. Also drop an identity constant operand (``x and True`` -> x, ``x or False`` -> x), which is
    what collapses the residual ``and`` a chained comparison leaves once a statically-true link folds.
    """
    cval: dict[ValueId, Const] = {}

    def uniform_const_arm(arms: tuple[tuple[int, ValueId], ...], remap: dict[ValueId, ValueId]) -> Const | None:
        if not arms:
            return None
        values = [cval.get(remap[arm]) for _, arm in arms]
        first = values[0]
        return first if first is not None and all(value == first for value in values) else None

    def substitute_identity(operation: Operation, new_operands: list[ValueId]) -> ValueId | None:
        """
        An algebraic simplification of a partially-constant operation: drop identity-element constant operands and, if
        exactly one operand survives, return its id (``x and True`` -> x). None when no substitution applies, so the
        caller constant-folds an absorbing operand or copies the node verbatim. Sound for any associative operator that
        declares an identity element: every identity operand drops out and the lone survivor remains.
        """
        identity = operation.operator.identity()
        if identity is None:
            return None
        survivors = [op for op in new_operands if cval.get(op) != identity]
        return survivors[0] if len(survivors) == 1 else None

    def build_value(builder: HirBuilder, vid: ValueId, node: Node, remap: dict[ValueId, ValueId]) -> ValueId:
        folded: Const | None = None
        match node:
            case Const():
                folded = node
            case Operation(operator=operator, operands=operands):
                new_operands = [remap[op] for op in operands]
                operand_consts = [cval.get(op) for op in new_operands]
                if all(const is not None for const in operand_consts):
                    folded = operator.fold_constants([const for const in operand_consts if const is not None])
                elif (absorbing := operator.absorbing()) is not None and absorbing in operand_consts:
                    folded = absorbing  # an absorbing operand fixes the result regardless of the others
                else:
                    survivor = substitute_identity(node, new_operands)
                    if survivor is not None:
                        return survivor
            case Phi(arms=arms):
                folded = uniform_const_arm(arms, remap)
        if folded is None:
            return copy_node(builder, node, remap)
        new_id = builder.const_node(folded)
        cval[new_id] = folded
        return new_id

    return rebuild(hir, build_value)
