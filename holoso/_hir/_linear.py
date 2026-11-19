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
from .._util import ValueId
from ._ir import Hir, HirBuilder, Node, Operation, StateRead
from ._operators import FloatAdd, FloatFma, FloatNeg
from ._scaling import Rendering, Scaling, fused_scaling, scaled_node, scaling_layer, scaling_of_ratio

_logger = logging.getLogger(__name__)

_MAX_BITS = 4096
"""An unrolled `v = 2**100 * v + y` keeps one term whose numerator grows per trip."""


@dataclass(frozen=True, slots=True)
class _Scaled:
    """In the shape strength reduction gives a constant scaling, so no round restates it."""

    base: ValueId
    rendering: Rendering


@dataclass(frozen=True, slots=True)
class _Constant:
    value: float


type _Answer = _Scaled | _Constant


class _Form:
    """
    `scale * (sum of coefficient * base + constant)`, with exact rational coefficients: `x + 1e-20*x - x` is `1e-20*x`,
    which a double coefficient, having absorbed the small term into 1.0, would answer as zero.

    A form is grown in place by its last reader and copied for any other, so a sum merges its smaller operand into its
    larger one and a scaling multiplies one number: a chain of any length costs time and memory in proportion to its
    terms, while a sum whose every prefix is read again costs in proportion to the square of its length. `bits` bounds
    every stored coefficient's size, so the bit limit is checked without a walk.
    """

    __slots__ = ("terms", "constant", "scale", "bits")

    def __init__(self, terms: dict[ValueId, Fraction], constant: Fraction) -> None:
        self.terms = terms
        self.constant = constant
        self.scale = Fraction(1)
        self.bits = max((_bits(c) for c in (*terms.values(), constant)), default=0)

    def copy(self) -> "_Form":
        copied = _Form(dict(self.terms), self.constant)
        copied.scale, copied.bits = self.scale, self.bits
        return copied

    def scale_by(self, factor: Fraction) -> None:
        # A zero factor is an absorbing element, not a scaling: it would erase every term's identity.
        assert factor
        self.scale *= factor

    def absorb(self, other: "_Form") -> None:
        """Add `other` into this form, carrying its terms over to this form's scale."""
        ratio = other.scale / self.scale
        for base, c in other.terms.items():
            total = self.terms.get(base, Fraction(0)) + c * ratio
            if total:
                self.terms[base] = total
                self.bits = max(self.bits, _bits(total))
            else:
                self.terms.pop(base, None)
        self.constant += other.constant * ratio
        self.bits = max(self.bits, _bits(self.constant))

    def oversized(self) -> bool:
        """
        Whether a coefficient outgrows the bit limit. The cheap bound overestimates, so where it trips the scale is
        carried into the coefficients and the size taken exactly.
        """
        if self.bits + (0 if self.scale == 1 else _bits(self.scale)) <= _MAX_BITS:
            return False
        self.terms = {base: c * self.scale for base, c in self.terms.items()}
        self.constant *= self.scale
        self.scale = Fraction(1)
        self.bits = max((_bits(c) for c in (*self.terms.values(), self.constant)), default=0)
        return self.bits > _MAX_BITS

    def collapsed(self, vid: ValueId) -> _Answer | None:
        """A form the pass gave up on reads as `1 * itself`, which `vid` is here to refuse."""
        if not self.terms:
            if not self.constant:
                return _Constant(0.0)
            # A number the machine must hold, exponent or not, so this one path still asks for the host float.
            named = scaling_of_ratio(self.scale * self.constant)
            value = None if named is None else named.coefficient()
            return None if value is None else _Constant(value)
        if len(self.terms) == 1 and not self.constant:
            ((base, coefficient),) = self.terms.items()
            if base == vid:
                return None
            named = scaling_of_ratio(self.scale * coefficient)
            rendering = None if named is None else named.rendering()
            return None if rendering is None else _Scaled(base, rendering)
        return None


def _bits(value: Fraction) -> int:
    return max(value.numerator.bit_length(), value.denominator.bit_length())


def _opaque(vid: ValueId) -> _Form:
    return _Form({vid: Fraction(1)}, Fraction(0))


type _Reading = list[tuple[ValueId, Scaling | None]]
"""The operands whose forms a node sums, each under the constant scaling the node gives it, if any."""


def _reading(hir: Hir, vid: ValueId) -> _Reading:
    """How a node reads its operands' forms; empty where it reads none and its own form is a leaf."""

    def constant(operand: ValueId) -> float | None:
        node = hir.nodes[operand]
        return node.value if isinstance(node, FloatConst) else None

    node = hir.nodes[vid]
    if not isinstance(node, Operation):
        return []
    if isinstance(node.operator, FloatAdd):
        a, b = node.operands
        return [(a, None), (b, None)]
    # A surviving `inf * 0.0` names no number and is the refusal gate's to convict, not ours to erase, which the layer
    # reader already declines.
    if (layer := scaling_layer(hir, vid, constant)) is not None:
        return [layer]
    if (fused := fused_scaling(hir, vid, constant)) is not None:
        return [(fused.addend, None), (fused.base, fused.scaling)]
    return []


def _answers(hir: Hir) -> dict[ValueId, _Answer]:
    """
    Each sum's answer, decided as its form is made, since its last reader then grows that form in place. A form read
    more than once is copied for each reader but the last, and a form nothing reads is not kept. A non-finite
    constant is opaque wherever it appears: `Fraction` cannot name one, and `x + inf` folds nowhere.
    """
    readings = {vid: _reading(hir, vid) for vid in hir.nodes}
    pending: dict[ValueId, int] = {vid: 0 for vid in hir.nodes}
    for reading in readings.values():
        for operand, _ in reading:
            pending[operand] += 1
    forms: dict[ValueId, _Form] = {}
    answers: dict[ValueId, _Answer] = {}

    def take(vid: ValueId) -> _Form:
        pending[vid] -= 1
        return forms.pop(vid) if not pending[vid] else forms[vid].copy()

    def sum_of(a: _Form, b: _Form) -> _Form:
        large, small = (a, b) if len(a.terms) >= len(b.terms) else (b, a)
        large.absorb(small)
        return large

    def compute(vid: ValueId) -> _Form:
        node = hir.nodes[vid]
        if not readings[vid]:
            if isinstance(node, FloatConst) and math.isfinite(node.value):
                return _Form({}, Fraction(node.value))
            return _opaque(vid)
        combined: _Form | None = None
        admitted = True
        for operand, scaling in readings[vid]:
            form = take(operand)
            if scaling is not None:
                admitted = admitted and abs(scaling.k) <= _MAX_BITS
                if admitted:
                    form.scale_by(scaling.ratio())
            combined = form if combined is None else sum_of(combined, form)
        assert combined is not None
        return _opaque(vid) if not admitted or combined.oversized() else combined

    def settle(vid: ValueId) -> None:
        form = compute(vid)
        node = hir.nodes[vid]
        if isinstance(node, Operation) and isinstance(node.operator, (FloatAdd, FloatFma)):
            answer = form.collapsed(vid)
            if answer is not None:
                answers[vid] = answer
        if pending[vid]:
            forms[vid] = form

    # Dominance order: every operand is defined before its user, so no walk nests; phis are opaque.
    blocks = {block.id: block for block in hir.blocks}
    for vid in hir.input_ids:
        settle(vid)
    for vid, node in hir.nodes.items():
        if isinstance(node, (Const, StateRead)):
            settle(vid)
    for bid in reverse_postorder(hir):
        for vid in blocks[bid].phis + blocks[bid].operations:
            settle(vid)
    assert not forms and not any(pending.values()), (len(forms), sum(pending.values()))
    return answers


def run(hir: Hir) -> Hir:
    """
    Every answer is taken: it replaces an addition, fused with a product or not, with at most one operation whatever
    else reads the terms, so no liveness is read to know it does not add work.
    """
    answers = _answers(hir)
    if not answers:
        return hir

    def build_value(builder: HirBuilder, vid: ValueId, node: Node, remap: dict[ValueId, ValueId]) -> ValueId:
        match answers.get(vid):
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
