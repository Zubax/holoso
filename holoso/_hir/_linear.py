"""
Reading a sum as a linear combination and answering it where the terms cancel.

The difference of two close values keeps none of their digits where the remainder the form carries keeps all of them;
a coefficient the answer must round can cost accuracy in return, the trade the fast-math charter makes everywhere.
"""

import logging
import math
from dataclasses import dataclass
from fractions import Fraction

from ._const import Const, FloatConst
from ._copy import copy_node, rebuild, reverse_postorder
from .._util import BlockId, ValueId
from ._ir import Hir, HirBuilder, Node, Operation, StateRead
from ._operators import FloatAdd, FloatMul, FloatMulPow2, FloatNeg
from ._scaling import Rendering, scaled_node, scaling_of_ratio

_logger = logging.getLogger(__name__)

_MAX_TERMS = 64

_MAX_BITS = 4096
"""Term count alone bounds nothing: an unrolled `v = 2**100 * v + y` keeps one term whose numerator grows per trip."""


@dataclass(frozen=True, slots=True)
class _Scaled:
    """In the shape strength reduction gives a constant scaling, so no round restates it."""

    base: ValueId
    rendering: Rendering


@dataclass(frozen=True, slots=True)
class _Constant:
    value: float


type _Answer = _Scaled | _Constant


@dataclass(frozen=True, slots=True)
class _LinearForm:
    """
    Coefficients are exact rationals: `x + 1e-20*x - x` is `1e-20*x`, which a double coefficient, having absorbed the
    small term into 1.0, would answer as zero.
    """

    _terms: tuple[tuple[ValueId, Fraction], ...]
    _constant: Fraction

    def scaled(self, factor: Fraction) -> "_LinearForm":
        # A zero factor is an absorbing element, not a scaling: it would erase every term's identity.
        assert factor
        return _LinearForm(tuple((base, c * factor) for base, c in self._terms), self._constant * factor)

    def plus(self, other: "_LinearForm") -> "_LinearForm":
        merged: dict[ValueId, Fraction] = dict(self._terms)
        for base, c in other._terms:
            total = merged.get(base, Fraction(0)) + c
            if total:
                merged[base] = total
            else:
                merged.pop(base, None)
        return _LinearForm(tuple(sorted(merged.items())), self._constant + other._constant)

    @property
    def oversized(self) -> bool:
        if len(self._terms) > _MAX_TERMS:
            return True
        return any(
            c.numerator.bit_length() > _MAX_BITS or c.denominator.bit_length() > _MAX_BITS
            for _, c in (*self._terms, (0, self._constant))
        )

    def collapsed(self, vid: ValueId) -> _Answer | None:
        """A form the pass gave up on reads as `1 * itself`, which `vid` is here to refuse."""
        if not self._terms:
            if not self._constant:
                return _Constant(0.0)
            # A number the machine must hold, exponent or not, so this one path still asks for the host float.
            named = scaling_of_ratio(self._constant)
            value = None if named is None else named.coefficient()
            return None if value is None else _Constant(value)
        if len(self._terms) == 1 and not self._constant:
            base, coefficient = self._terms[0]
            if base == vid:
                return None
            named = scaling_of_ratio(coefficient)
            rendering = None if named is None else named.rendering()
            return None if rendering is None else _Scaled(base, rendering)
        return None


def _opaque(vid: ValueId) -> _LinearForm:
    return _LinearForm(((vid, Fraction(1)),), Fraction(0))


def _forms(hir: Hir) -> dict[ValueId, _LinearForm]:
    """A non-finite constant is opaque wherever it appears: `Fraction` cannot name one, and `x + inf` folds nowhere."""
    forms: dict[ValueId, _LinearForm] = {}

    def compute(vid: ValueId) -> _LinearForm:
        node = hir.nodes[vid]
        if isinstance(node, FloatConst):
            return _opaque(vid) if not math.isfinite(node.value) else _LinearForm((), Fraction(node.value))
        if not isinstance(node, Operation):
            return _opaque(vid)
        match node.operator:
            case FloatAdd():
                a, b = node.operands
                combined = forms[a].plus(forms[b])
            case FloatNeg():
                combined = forms[node.operands[0]].scaled(Fraction(-1))
            case FloatMulPow2(k=k):
                if abs(k) > _MAX_BITS:
                    return _opaque(vid)
                combined = forms[node.operands[0]].scaled(Fraction(2) ** k)
            case FloatMul():
                a, b = node.operands
                for base, other in ((a, b), (b, a)):  # either way round: this reads the graph it is given
                    constant = hir.nodes[other]
                    # A surviving `inf * 0.0` names no number and is the refusal gate's to convict, not ours to erase.
                    if isinstance(constant, FloatConst) and math.isfinite(constant.value) and constant.value != 0.0:
                        combined = forms[base].scaled(Fraction(constant.value))
                        break
                else:
                    return _opaque(vid)
            case _:
                return _opaque(vid)
        return _opaque(vid) if combined.oversized else combined

    # Dominance order: every operand is defined before its user, so no walk nests; phis are opaque.
    blocks = {block.id: block for block in hir.blocks}
    for vid in hir.input_ids:
        forms[vid] = compute(vid)
    for vid, node in hir.nodes.items():
        if isinstance(node, (Const, StateRead)):
            forms[vid] = compute(vid)
    for bid in reverse_postorder(hir):
        for vid in blocks[bid].phis + blocks[bid].operations:
            forms[vid] = compute(vid)
    assert len(forms) == len(hir.nodes)
    return forms


def run(hir: Hir) -> Hir:
    """
    Every answer is taken: it replaces an addition with at most one operation whatever else reads the terms, so no
    liveness is read to know it does not add work.
    """
    forms = _forms(hir)
    answers: dict[tuple[BlockId, Node], _Answer] = {}
    for block in hir.blocks:
        for vid in block.operations:
            node = hir.nodes[vid]
            if isinstance(node, Operation) and isinstance(node.operator, FloatAdd):
                answer = forms[vid].collapsed(vid)
                if answer is not None:
                    answers[block.id, node] = answer
    if not answers:
        return hir

    def build_value(builder: HirBuilder, node: Node, remap: dict[ValueId, ValueId]) -> ValueId:
        match answers.get((builder.current_block, node)):
            case _Constant(value=value):
                return builder.const_node(FloatConst(value))
            case _Scaled(base=base, rendering=rendering):
                scaled = remap[base]
                operation = scaled_node(rendering.magnitude, scaled, lambda c: builder.const_node(FloatConst(c)))
                if operation is not None:
                    scaled = builder.operation(operation.operator, list(operation.operands))
                return builder.operation(FloatNeg(), [scaled]) if rendering.negative else scaled
            case None:
                return copy_node(builder, node, remap)

    _logger.info("Linear-form cancellation: %d sum(s) answered by what their terms leave", len(answers))
    return rebuild(hir, build_value)
