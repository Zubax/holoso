"""
Trivial-phi elimination.

A phi whose arms, ignoring self-references, name a single distinct value is redundant: it always equals that value.
The most common source is a loop-invariant carried through a header -- ``phi(preheader: x, latch: self)`` -- which the
emitter's on-the-fly SSA construction can leave behind when a later user of the phi was already wired to it before the
header sealed and the phi turned out trivial. Such a phi is one extra node and, inside a loop, one extra live value on
the recurrence, so it costs both area and a scheduling slot. This pass replaces every use of a trivial phi with its one
real operand and deletes it, repeating to a fixpoint (collapsing one phi can make a phi that referenced it trivial in
turn). Block structure and predecessors are untouched, so no phi's arm count changes.
"""

import logging

from .._util import ValueId
from ._ir import Block, Branch, Hir, Node, Operation, OutputPort, Phi, StateSlot

_logger = logging.getLogger(__name__)


def _trivial_target(vid: ValueId, node: Phi) -> ValueId | None:
    real = {value for _, value in node.arms if value != vid}
    return next(iter(real)) if len(real) == 1 else None


def _substitute(hir: Hir, victim: ValueId, replacement: ValueId) -> Hir:
    def sub(value: ValueId) -> ValueId:
        return replacement if value == victim else value

    nodes: dict[ValueId, Node] = {}
    for vid, node in hir.nodes.items():
        if vid == victim:
            continue
        match node:
            case Operation(operator=operator, operands=operands):
                nodes[vid] = Operation(operator, tuple(sub(o) for o in operands))
            case Phi(type=phi_type, arms=arms):
                nodes[vid] = Phi(type=phi_type, arms=tuple((pred, sub(value)) for pred, value in arms))
            case _:
                nodes[vid] = node
    blocks = [
        Block(
            block.id,
            tuple(phi for phi in block.phis if phi != victim),
            block.operations,
            (
                Branch(sub(block.terminator.cond), block.terminator.if_true, block.terminator.if_false)
                if isinstance(block.terminator, Branch)
                else block.terminator
            ),
        )
        for block in hir.blocks
    ]
    outputs = [OutputPort(out.name, sub(out.value)) for out in hir.outputs]
    slots = [StateSlot(s.name, s.reset_value, sub(s.live_out)) for s in hir.state_slots]
    return Hir(nodes=nodes, blocks=blocks, input_ids=hir.input_ids, outputs=outputs, state_slots=slots)


def run(hir: Hir) -> Hir:
    removed = 0
    while True:
        target: tuple[ValueId, ValueId] | None = None
        for vid, node in hir.nodes.items():
            if isinstance(node, Phi) and (replacement := _trivial_target(vid, node)) is not None:
                target = (vid, replacement)
                break
        if target is None:
            break
        hir = _substitute(hir, *target)
        removed += 1
    if removed:
        _logger.info("Trivial-phi elimination: %d redundant phi(s) removed", removed)
    return hir
