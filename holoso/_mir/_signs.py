"""Semantic sideband chains, peeled into the MIR conditioners their consumers fold them onto."""

from .._hir import BoolNot, BoolType, FloatAbs, FloatNeg, FloatType, IntType, Node, Operation
from .._operators import BoolInversion, FloatSignControl, IntIdentity, PortConditioner
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


def collapse_bool_inversions(nodes: dict[ValueId, Node], vid: ValueId) -> tuple[ValueId, BoolInversion]:
    """
    Peel a chain of semantic NOT operations, returning the base value and the combined inversion -- the boolean dual
    of collapse_signs. Folding happens on the CONSUMER side only: a NOT over a comparison must never flip
    the producer's tap conditioner (two taps of one comparator port with different inversions cannot fuse and would
    serialize two firings), and consumer-side folding keeps one shared producer for both polarities of a value.
    """
    invert = False
    node = nodes[vid]
    while isinstance(node, Operation) and isinstance(node.operator, BoolNot):
        invert = not invert
        (vid,) = node.operands
        node = nodes[vid]
    return vid, BoolInversion(invert=invert)


def collapse_conditioner(nodes: dict[ValueId, Node], vid: ValueId) -> tuple[ValueId, PortConditioner]:
    """
    Collapse the type's own sideband chain: sign operations over a float value, NOTs over a boolean one. An integer
    has no free sideband, so `ineg`/`iabs` are hardware and nothing collapses.
    """
    ty = nodes[vid].type
    match ty:
        case BoolType():
            return collapse_bool_inversions(nodes, vid)
        case FloatType():
            return collapse_signs(nodes, vid)
        case IntType():
            return vid, IntIdentity()
    raise AssertionError(f"no conditioner for {ty!r}")
