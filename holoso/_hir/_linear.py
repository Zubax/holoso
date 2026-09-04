"""
Linear-form sharing: a sum that is a constant multiple of a sum already computed is that multiple of it.

Strength reduction composes a scaling with an adjacent one, but an intervening addition stops it, so a kernel
needing one quantity in two unit systems materializes the same combination twice.

Rests on distributing a constant across a sum, and relocating the scale a sum is computed at.
The second is the sharp one: a derived sum inherits the keeper's absolute rounding error scaled by the factor between
them, unbounded relative to its own value where the keeper's terms cancel,
and one derived from a keeper that rails is an infinity. Which sum is the keeper follows the order they are written.

Runs on a settled graph, so a form it registers belongs to a value a later pass will keep.
"""

import logging
import math
from dataclasses import dataclass
from fractions import Fraction

from ._const import FloatConst
from ._copy import copy_node, rebuild
from .._util import BlockId, ValueId
from ._ir import Hir, HirBuilder, Node, Operation
from ._operators import FloatAdd, FloatMul, FloatMulPow2, FloatNeg
from ._scaling import Scaling, scaling_of

_logger = logging.getLogger(__name__)

_MAX_TERMS = 64

_MAX_BITS = 4096
"""Term count alone bounds nothing: an unrolled `v = 2**100 * v + y` keeps one term whose numerator grows per trip."""


@dataclass(frozen=True, slots=True)
class _LinearForm:
    """
    Two forms differing by one overall factor denote values exactly one multiplication apart, which is what lets
    the second share the first.

    Coefficients are exact rationals, never host floats: `1e308*x + 1e-300*y` and `1e308*x + 1e-200*y` both collapse
    the small coefficient to zero in doubles, and merging them would answer for a difference the format holds.
    """

    _terms: tuple[tuple[ValueId, Fraction], ...]
    _constant: Fraction

    def scaled(self, factor: Fraction) -> _LinearForm:
        assert factor, "a zero factor is an absorbing element, not a scaling, and would erase every term's identity"
        return _LinearForm(tuple((base, c * factor) for base, c in self._terms), self._constant * factor)

    def plus(self, other: _LinearForm) -> _LinearForm:
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

    def normalized(self) -> tuple[_LinearForm, Fraction] | None:
        """`None` where the terms all cancelled: that denotes a constant, which folding owns rather than sharing."""
        if not self._terms:
            return None
        pivot = self._terms[0][1]
        assert pivot, "a zero coefficient is never kept, so the pivot cannot be one"
        scaled = _LinearForm(tuple((base, c / pivot) for base, c in self._terms), self._constant / pivot)
        return scaled, pivot


def _retirable(hir: Hir, uses: dict[ValueId, int], operand: ValueId, keeper: ValueId) -> bool:
    """A sign op is not an operator but a sideband on its consumer, so it is walked through rather than counted."""
    while operand != keeper and uses[operand] == 1 and isinstance(node := hir.nodes[operand], Operation):
        if not isinstance(node.operator, FloatNeg):
            return True
        (operand,) = node.operands
    return False


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
                for base, other in ((a, b), (b, a)):
                    constant = hir.nodes[other]
                    # Zero and the infinities are absorbing elements rather than scalings, the exclusion
                    # `scaling_of` already makes. Zero matters beyond tidiness: a surviving `inf * 0.0` names no
                    # number and is the refusal gate's to convict, not this pass's to erase.
                    if isinstance(constant, FloatConst) and math.isfinite(constant.value) and constant.value != 0.0:
                        combined = forms[base].scaled(Fraction(constant.value))
                        break
                else:
                    return _opaque(vid)
            case _:
                return _opaque(vid)
        return _opaque(vid) if combined.oversized else combined

    for vid in sorted(hir.nodes):
        node = hir.nodes[vid]
        # Ascending order is what keeps this iterative: a rebuilt graph numbers every operand below its user, so
        # no walk nests. Phis may name a later value, and are opaque anyway.
        assert not isinstance(node, Operation) or all(operand < vid for operand in node.operands)
        forms[vid] = compute(vid)
    return forms


def _scaling(ratio: Fraction) -> Scaling | None:
    """A ratio read as the constant it scales by; `None` where the host arithmetic does not hold it."""
    try:
        return scaling_of(float(ratio))
    except OverflowError:
        return None


def run(hir: Hir) -> Hir:
    """Block-scoped, matching the builder's own interning, so a keeper always dominates the use that adopts it."""
    forms = _forms(hir)
    uses = hir.use_counts()
    rewrites: dict[tuple[BlockId, Node], tuple[ValueId, float]] = {}

    def retires_something(node: Operation, keeper: ValueId) -> bool:
        """
        Where nothing retires, the rewrite trades an addition for a multiplication by a coefficient that need not
        even be exact, so the sum stands. The keeper never counts -- the rewrite READS it, so an operand that is
        the keeper gains a use rather than losing its last one, which is what `x + x + x` turns on.
        """
        return any(_retirable(hir, uses, operand, keeper) for operand in node.operands)

    for block in hir.blocks:
        keepers: dict[_LinearForm, tuple[ValueId, Fraction]] = {}
        for vid in block.operations:
            node = hir.nodes[vid]
            if not (isinstance(node, Operation) and isinstance(node.operator, FloatAdd)):
                continue
            normalized = forms[vid].normalized()
            if normalized is None:
                continue
            unit, pivot = normalized
            keeper = keepers.get(unit)
            if keeper is None:
                keepers[unit] = (vid, pivot)
                continue
            keeper_id, keeper_pivot = keeper
            scaling = _scaling(pivot / keeper_pivot)
            if scaling is None or (coefficient := scaling.coefficient()) is None:
                continue
            # A power-of-two ratio is a scaler once the next reduction round absorbs it, so it pays either way;
            # any other stays a multiplication.
            if scaling.is_power_of_two or retires_something(node, keeper_id):
                # Operations intern per block, so block and node name the value uniquely -- which is how the
                # rebuild driver, given the node and not its id, finds the rewrite.
                rewrites[(block.id, node)] = (keeper_id, coefficient)
    if not rewrites:
        return hir

    def build_value(builder: HirBuilder, node: Node, remap: dict[ValueId, ValueId]) -> ValueId:
        rewrite = rewrites.get((builder.current_block, node))
        if rewrite is None:
            return copy_node(builder, node, remap)
        keeper, coefficient = rewrite
        return builder.operation(FloatMul(), [remap[keeper], builder.const_node(FloatConst(coefficient))])

    result = rebuild(hir, build_value)
    _logger.info("Linear-form sharing: %d sum(s) answered as a scaling of an equivalent sum", len(rewrites))
    return result
