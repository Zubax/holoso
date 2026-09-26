"""FMA contraction: the additions that absorb a product entirely, rewritten into the semantic fma."""

import logging
from collections import defaultdict
from dataclasses import dataclass

from .._hir import (
    FloatAbs,
    FloatAdd,
    FloatConst,
    FloatFma,
    FloatMul,
    FloatMulPow2,
    FloatNeg,
    Hir,
    HirBuilder,
    Node,
    Operation,
    Scaling,
    copy_node,
    eliminate_dead_code,
    rebuild,
    references,
)
from .._operators import FloatSignControl, OpConfig
from .._type import FloatFormat
from .._util import ValueId
from ._signs import sign_chain

_logger = logging.getLogger(__name__)


def _exact_scale(fmt: FloatFormat, k: int) -> float | None:
    """
    `2**k` only where this format holds it exactly: the host rails past k = 1023 while composed exponents are
    unbounded, and a format without subnormals rounds a small power onto a neighbour it is not.
    """
    value = Scaling(1.0, k, False).magnitude()
    return value if value is not None and fmt.round(value) == value else None


@dataclass(frozen=True, slots=True)
class _ValueFmaPlan:
    """
    A contraction of `a*b + c` into one `ffma`: `product` is the FloatMul whose standalone MIR op it suppresses,
    `product_sign` the sign peeled off the product operand.
    """

    product: ValueId
    a: ValueId
    b: ValueId
    c: ValueId
    product_sign: FloatSignControl


@dataclass(frozen=True, slots=True)
class _ScaleFmaPlan:
    """
    The same over an exponent scaling, `a*2**k + c`. The scaler is exact, so the fma may only stand in for it where
    the format holds `2**k` exactly -- checked at planning, since the plan materializes what the scaler never did.
    """

    product: ValueId
    a: ValueId
    scale: float
    c: ValueId
    product_sign: FloatSignControl


type _FmaPlan = _ValueFmaPlan | _ScaleFmaPlan


def _fma_plan(hir: Hir, fmt: FloatFormat, product: ValueId, sign: FloatSignControl, addend: ValueId) -> _FmaPlan | None:
    node = hir.nodes[product]
    if not isinstance(node, Operation):
        return None
    match node.operator:
        case FloatMul():
            a, b = node.operands
            return _ValueFmaPlan(product=product, a=a, b=b, c=addend, product_sign=sign)
        case FloatMulPow2(k=k) if (scale := _exact_scale(fmt, k)) is not None:
            (a,) = node.operands
            return _ScaleFmaPlan(product=product, a=a, scale=scale, c=addend, product_sign=sign)
        case _:
            return None


def contract_fmas(hir: Hir, ops: OpConfig) -> Hir:
    """
    Rewrite every contractible `a*b + c` into the semantic `FloatFma` the kernel could have written itself, so one
    lowering serves both spellings. The product sign rides HIR sign operations, which selection folds onto the
    operand conditioners as it does for any other operand.

    Runs after judgement: the constant an exponent scaling materializes is the machine's own, not the program's.
    """
    plans = _plan_fma_fusions(hir, ops)
    if not plans:
        return hir

    def build_value(builder: HirBuilder, vid: ValueId, node: Node, remap: dict[ValueId, ValueId]) -> ValueId:
        plan = plans.get(vid)
        if plan is None:
            return copy_node(builder, node, remap)
        signed = _signed(builder, remap[plan.a], plan.product_sign)
        match plan:
            case _ValueFmaPlan(b=b):
                # `|a*b|` is `|a||b|`, so an absolute product signs both multipliers; a negation signs only one.
                other = _signed(builder, remap[b], FloatSignControl(absolute=plan.product_sign.absolute))
            case _ScaleFmaPlan(scale=scale):
                other = builder.const_node(FloatConst(scale))
        return builder.operation(FloatFma(), [signed, other, remap[plan.c]])

    _logger.info("FMA contraction: %d addition(s) contracted with their product", len(plans))
    return eliminate_dead_code(rebuild(hir, build_value))


def _signed(builder: HirBuilder, value: ValueId, sign: FloatSignControl) -> ValueId:
    if sign.absolute:
        value = builder.operation(FloatAbs(), [value])
    return builder.operation(FloatNeg(), [value]) if sign.negate else value


def _readers(hir: Hir) -> dict[ValueId, list[ValueId | None]]:
    readers: dict[ValueId, list[ValueId | None]] = defaultdict(list)
    for vid, node in hir.nodes.items():
        for referenced in references(node):
            readers[referenced].append(vid)
    for referenced in hir.external_value_references():
        readers[referenced].append(None)
    return readers


def _plan_fma_fusions(hir: Hir, ops: OpConfig) -> dict[ValueId, _FmaPlan]:
    """
    A product is carried by every add that names it or by none, since one add would single-round a product observed
    elsewhere.
    """
    if ops.options.ffma is None:
        return {}
    fmt = ops.float_format
    readers = _readers(hir)
    position = {vid: index for index, vid in enumerate(vid for block in hir.blocks for vid in block.operations)}
    sites: dict[ValueId, list[tuple[ValueId, _FmaPlan]]] = defaultdict(list)
    signs: dict[ValueId, set[ValueId]] = defaultdict(set)
    for vid, node in hir.nodes.items():
        if not (isinstance(node, Operation) and isinstance(node.operator, FloatAdd)):
            continue
        op0, op1 = node.operands
        for product_operand, addend in ((op0, op1), (op1, op0)):
            base, control, chain = sign_chain(hir.nodes, product_operand)
            plan = _fma_plan(hir, fmt, base, control, addend)
            if plan is not None:
                sites[plan.product].append((vid, plan))
                signs[plan.product].update(chain)

    plans: dict[ValueId, _FmaPlan] = {}
    # Fewest adds first, so an exclusive product -- which nothing else can want -- never loses its add to a shared
    # one that then fails to be claimed whole.
    for product in sorted(sites, key=lambda product: (len(sites[product]), position[product])):
        adds = {add for add, _ in sites[product]}
        cone = {product, *signs[product]}
        reads = [site for member in cone for site in readers[member] if site not in cone]
        # Every read of the product is an absorbing port of a distinct addition, so no add rounds it twice.
        absorbable = bool(reads) and set(reads) <= adds and len(set(reads)) == len(reads)
        if absorbable and not any(add in plans for add in adds):
            plans.update(sites[product])
    return plans
