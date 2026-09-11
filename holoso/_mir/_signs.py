"""Semantic sign chains, peeled into the MIR sign controls their consumers fold them onto."""

from .._hir import FloatAbs, FloatNeg, Node, Operation
from .._operators import FloatSignControl
from .._util import ValueId


def sign_of(node: Operation) -> FloatSignControl | None:
    match node:
        case Operation(operator=FloatNeg()):
            return FloatSignControl(negate=True)
        case Operation(operator=FloatAbs()):
            return FloatSignControl(absolute=True)
        case _:
            return None


def sign_chain(nodes: dict[ValueId, Node], vid: ValueId) -> tuple[ValueId, FloatSignControl, list[ValueId]]:
    """
    The base value under a chain of semantic sign operations, the one control they amount to, and the chain itself.
    Peeling runs outermost first while composition needs innermost first, which `sign.then(control)` supplies.
    """
    chain: list[ValueId] = []
    control = FloatSignControl()
    node = nodes[vid]
    while isinstance(node, Operation) and (sign := sign_of(node)) is not None:
        chain.append(vid)
        control = sign.then(control)
        (vid,) = node.operands
        node = nodes[vid]
    return vid, control, chain


def collapse_signs(nodes: dict[ValueId, Node], vid: ValueId) -> tuple[ValueId, FloatSignControl]:
    base, control, _ = sign_chain(nodes, vid)
    return base, control
