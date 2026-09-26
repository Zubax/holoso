import math
from collections.abc import Mapping
from dataclasses import replace
from typing import assert_never

from .._mir import Mir, MirBranch, MirConst, MirInput, MirJump, MirOperation, MirPhi, MirRet, MirTerminator
from .._operators import (
    BoolInversion,
    FloatSignControl,
    InlineHardwareOperator,
    IntIdentity,
    PooledHardwareOperator,
    PortConditioner,
    WideConditioner,
)
from .._type import FloatType, IntType
from .._value import FloatValue, IntValue, WideValue
from .._util import ValueId
from ._ir import *
from ._schedule import Schedule
from ._sources import BoolOperandTemplate, OperandTemplate, WideOperandTemplate
from ._build_base import Allocation, ConstPool, PooledConst
from ._mir_facts import mir_operation


def wide_type(mir: Mir, vid: ValueId) -> FloatType | IntType:
    """Which family a wide value belongs to: the bank is physical, so only the value itself names its type."""
    scalar_type = mir.nodes[vid].scalar_type
    assert isinstance(scalar_type, (FloatType, IntType))
    return scalar_type


def bool_operand_template(mir: Mir, vid: ValueId, inversion: PortConditioner) -> BoolOperandTemplate:
    assert isinstance(inversion, BoolInversion)
    node = mir.nodes[vid]
    assert not node.scalar_type.is_wide
    if isinstance(node, MirConst):
        assert type(node.value) is bool
        return BoolOperandTemplate(BoolConstRef(node.value), inversion)
    return BoolOperandTemplate(vid, inversion)


def bool_operand(mir: Mir, vid: ValueId, alloc: Allocation, inversion: PortConditioner) -> BoolOperand:
    return bool_operand_template(mir, vid, inversion).resolve(alloc.bool.assign.__getitem__)


def operand_templates(node: MirOperation, mir: Mir, pool: Mapping[ValueId, PooledConst]) -> list[OperandTemplate]:
    """
    The one normalization of an operation's operands, shared by the LIR build and the allocator's objective so the
    two cannot disagree on a write source's identity. The constant pool stores a float as its magnitude and folds
    the sign onto whoever reads it -- BELOW the MIR normalization that cleared an unconditioned operand, so the sign
    would reappear on a port with nothing to bind it to. This is the last place that still knows the operator, hence
    the last that can erase it.
    """
    templates: list[OperandTemplate] = []
    for position, (vid, conditioner) in enumerate(zip(node.operands, node.operand_conditioners, strict=True)):
        template: OperandTemplate
        if mir.nodes[vid].scalar_type.is_wide:
            template = wide_operand_template(mir, vid, conditioner, pool)
        else:
            template = bool_operand_template(mir, vid, conditioner)
        if position in node.operator.unconditioned_operands:
            assert isinstance(template, WideOperandTemplate)  # the declaration admits float ports alone
            template = replace(template, conditioner=FloatSignControl())
        templates.append(template)
    return templates


def _operands_of(
    node: MirOperation, mir: Mir, alloc: Allocation, pool: dict[ValueId, PooledConst]
) -> list[WideOperand | BoolOperand]:
    return [
        (
            template.resolve(alloc.wide.assign.__getitem__)
            if isinstance(template, WideOperandTemplate)
            else template.resolve(alloc.bool.assign.__getitem__)
        )
        for template in operand_templates(node, mir, pool)
    ]


def _value_dst(mir: Mir, alloc: Allocation, vid: ValueId) -> RegRef | BoolRegRef:
    if mir.nodes[vid].scalar_type.is_wide:
        return RegRef(alloc.wide.assign[vid])
    return BoolRegRef(alloc.bool.assign[vid])


def build_inline_op(
    mir: Mir, vid: ValueId, issue_cycle: int, alloc: Allocation, pool: dict[ValueId, PooledConst]
) -> InlineScheduledOp:
    node = mir_operation(mir, vid)
    assert isinstance(node.operator, InlineHardwareOperator)
    operands = _operands_of(node, mir, alloc, pool)
    return InlineScheduledOp(
        operator=node.operator,
        operands=operands,
        write=PortWrite(port=node.output_port, dst=_value_dst(mir, alloc, vid), conditioner=node.output_conditioner),
        issue_cycle=issue_cycle,
    )


def build_pooled_op(
    mir: Mir,
    members: list[ValueId],
    sched: Schedule,
    alloc: Allocation,
    pool: dict[ValueId, PooledConst],
) -> PooledScheduledOp:
    """
    Build one pooled firing: the members share the operator, operands, and operand conditioners (the fusion key), so
    the operands are resolved once from the leader; each member contributes one PortWrite tapping its output port
    into its own bank's register.
    """
    leader = min(members)
    node = mir_operation(mir, leader)
    assert isinstance(node.operator, PooledHardwareOperator)
    operands = _operands_of(node, mir, alloc, pool)
    swapped = alloc.wide.swap[leader]
    if swapped:  # commutative operator: exchange operands (with their conditioners) to shrink read muxes
        operands.reverse()
    # A swapped firing's taps move with the operands: each member's output port maps through the operator's
    # commutation permutation (the comparator's gt and lt exchange while eq is fixed), so cmp(b,a) tapped at the
    # permuted port yields bit-exactly the member's original value.
    permutation = node.operator.swap_output_permutation

    def tap_port(member: ValueId) -> int:
        port = mir_operation(mir, member).output_port
        if not swapped:
            return port
        assert permutation is not None
        return permutation[port]

    writes = [
        PortWrite(
            port=tap_port(member),
            dst=_value_dst(mir, alloc, member),
            conditioner=mir_operation(mir, member).output_conditioner,
        )
        for member in sorted(members, key=tap_port)
    ]
    return PooledScheduledOp(
        inst=OperatorInstance(node.operator, alloc.wide.instance[leader]),
        operands=operands,
        writes=writes,
        issue_cycle=sched.issue_cycle[leader],
        immediates=node.immediates,
    )


def wide_operand_template(
    mir: Mir, vid: ValueId, conditioner: PortConditioner, pool: Mapping[ValueId, PooledConst]
) -> WideOperandTemplate:
    assert not isinstance(conditioner, BoolInversion)  # negatively, so a new wide conditioner needs no edit
    assert mir.nodes[vid].scalar_type.is_wide
    if isinstance(mir.nodes[vid], MirConst):
        entry = pool[vid]
        match entry.conditioner:
            case FloatSignControl():
                assert isinstance(conditioner, FloatSignControl)
                folded: WideConditioner = entry.conditioner.then(conditioner)
            case IntIdentity():
                assert isinstance(conditioner, IntIdentity)
                folded = conditioner
            case _:
                assert_never(entry.conditioner)
        return WideOperandTemplate(WideConstRef(entry.index), folded)
    return WideOperandTemplate(vid, conditioner)


def wide_operand(
    mir: Mir, vid: ValueId, conditioner: PortConditioner, alloc: Allocation, pool: dict[ValueId, PooledConst]
) -> WideOperand:
    return wide_operand_template(mir, vid, conditioner, pool).resolve(alloc.wide.assign.__getitem__)


def build_outputs(
    mir: Mir, alloc: Allocation, pool: dict[ValueId, PooledConst]
) -> list[WideOutputWire | BoolOutputWire]:
    outputs: list[WideOutputWire | BoolOutputWire] = []
    for out in mir.outputs:
        if mir.nodes[out.value].scalar_type.is_wide:
            tap = wide_operand(mir, out.value, out.conditioner, alloc, pool)
            outputs.append(WideOutputWire(out.name, tap, wide_type(mir, out.value)))
        else:
            outputs.append(BoolOutputWire(out.name, bool_operand(mir, out.value, alloc, out.conditioner)))
    return outputs


def build_terminator(terminator: MirTerminator, alloc: Allocation) -> Terminator:
    match terminator:
        case MirJump(target=target):
            return Jump(target)
        case MirBranch(cond=cond, if_true=if_true, if_false=if_false):
            return Branch(BoolRegRef(alloc.bool.assign[cond]), if_true, if_false)
        case MirRet():
            return Jump(Exit())


def build_const_pool(mir: Mir) -> ConstPool:
    """
    Build the immediate/ROM pool shared by both wide families, interned by the typed encoded value, so
    encoding-equal float literals share a word and class-aware equality keeps `1` and `1.0` distinct where raw
    Python keys would collide. A FLOAT is stored as its nonnegative magnitude, the sign folded into the consumer's
    free sign-control sideband -- value-preserving because `encode(|c|)` with the sign bit set equals `encode(c)`,
    and MIR normalizes `-0.0` and refuses magnitudes degrading to zero, so a folded negate can never emit the `-0`
    ZKF has no room for. An INTEGER has no sideband and is stored whole. A boolean constant rides inline.
    """
    ids: list[ValueId] = []
    seen: set[ValueId] = set()

    def note(vid: ValueId) -> None:
        if isinstance(mir.nodes[vid], MirConst) and mir.nodes[vid].scalar_type.is_wide and vid not in seen:
            seen.add(vid)
            ids.append(vid)

    for node in mir.nodes.values():
        if isinstance(node, MirOperation):
            for operand in node.operands:
                note(operand)
        elif isinstance(node, MirPhi):  # a constant phi arm becomes a copy source, so it must be pooled
            for _, arm, _ in node.arms:
                note(arm)
    for out in mir.outputs:
        note(out.value)
    for slot in mir.state_slots:
        note(slot.live_out)
    values: list[WideValue] = []
    index_of: dict[WideValue, int] = {}
    pool: dict[ValueId, PooledConst] = {}

    def intern(value: WideValue) -> int:
        index = index_of.get(value)
        if index is None:
            index_of[value] = index = len(values)
            values.append(value)
        return index

    for vid in ids:
        const = mir.nodes[vid]
        assert isinstance(const, MirConst)
        value = const.value
        if type(value) is int:
            pool[vid] = PooledConst(intern(IntValue.from_int(mir.int_format, value)), IntIdentity())
            continue
        assert type(value) is float
        magnitude = FloatValue.from_float(mir.float_format, abs(value))
        negate = math.copysign(1.0, value) < 0.0
        pool[vid] = PooledConst(intern(magnitude), FloatSignControl(negate=negate))
    return ConstPool(values, pool)


def build_inputs(mir: Mir, alloc: Allocation) -> list[WideInputLoad | BoolInputLoad]:
    loads: list[WideInputLoad | BoolInputLoad] = []
    for vid in mir.input_ids:
        node = mir.nodes[vid]
        assert isinstance(node, MirInput)
        if node.scalar_type.is_wide:
            loads.append(WideInputLoad(node.name, RegRef(alloc.wide.assign[vid]), wide_type(mir, vid)))
        else:
            loads.append(BoolInputLoad(node.name, BoolRegRef(alloc.bool.assign[vid])))
    return loads
