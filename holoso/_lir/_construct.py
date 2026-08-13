"""Construct LIR operands, scheduled ops, terminators, outputs, inputs, and the constant pool from selected MIR."""

import math
from typing import assert_never

from .._errors import UnsupportedConstruct
from .._mir import (
    Mir,
    MirBoolConst,
    MirBoolInput,
    MirBoolOutput,
    MirBoolView,
    MirBranch,
    MirFloatConst,
    MirFloatInput,
    MirFloatOutput,
    MirIntConst,
    MirIntInput,
    MirIntOutput,
    MirJump,
    MirOperation,
    MirPhi,
    MirRet,
    MirTerminator,
    MirWideView,
)
from .._operators import (
    BoolInversion,
    FloatSignControl,
    InlineHardwareOperator,
    IntIdentity,
    PortConditioner,
    WideConditioner,
)
from .._value import FloatValue, IntValue, WideValue
from .._util import ValueId
from ._ir import *
from ._schedule import Schedule
from ._build_base import Allocation, PooledConst
from ._mir_facts import mir_operation


def bool_operand(bool_mir: MirBoolView, vid: ValueId, alloc: Allocation, inversion: BoolInversion) -> BoolOperand:
    node = bool_mir.nodes[vid]
    if isinstance(node, MirBoolConst):
        return BoolOperand(BoolConstRef(node.value), inversion)  # folds to the negated immediate at construction
    return BoolOperand(BoolRegRef(alloc.bool_reg[vid]), inversion)


def _typed_operand(
    wide_mir: MirWideView,
    bool_mir: MirBoolView,
    vid: ValueId,
    conditioner: PortConditioner,
    alloc: Allocation,
    pool: dict[ValueId, PooledConst],
) -> WideOperand | BoolOperand:
    if vid in bool_mir.nodes:
        assert isinstance(conditioner, BoolInversion)
        return bool_operand(bool_mir, vid, alloc, conditioner)
    assert not isinstance(conditioner, BoolInversion)  # negatively, so a new wide conditioner needs no edit here
    return wide_operand(wide_mir, vid, conditioner, alloc, pool)


def _value_dst(wide_mir: MirWideView, alloc: Allocation, vid: ValueId) -> RegRef | BoolRegRef:
    if vid in wide_mir.operation_nodes:
        return RegRef(alloc.wide_reg[vid])
    return BoolRegRef(alloc.bool_reg[vid])


def build_inline_op(
    mir: Mir,
    wide_mir: MirWideView,
    bool_mir: MirBoolView,
    vid: ValueId,
    issue_cycle: int,
    alloc: Allocation,
    pool: dict[ValueId, PooledConst],
) -> InlineScheduledOp:
    node = mir_operation(mir, vid)
    assert isinstance(node.operator, InlineHardwareOperator)
    operands = [
        _typed_operand(wide_mir, bool_mir, operand, conditioner, alloc, pool)
        for operand, conditioner in zip(node.operands, node.operand_conditioners, strict=True)
    ]
    return InlineScheduledOp(
        operator=node.operator,
        operands=operands,
        write=PortWrite(
            port=node.output_port, dst=_value_dst(wide_mir, alloc, vid), conditioner=node.output_conditioner
        ),
        issue_cycle=issue_cycle,
        latency=node.operator.latency,
    )


def build_pooled_op(
    mir: Mir,
    wide_mir: MirWideView,
    bool_mir: MirBoolView,
    members: list[ValueId],
    sched: Schedule,
    inst_of: dict[ValueId, OperatorInstance],
    alloc: Allocation,
    pool: dict[ValueId, PooledConst],
    swap: dict[ValueId, bool],
) -> PooledScheduledOp:
    """
    Build one pooled firing: the members share the operator, operands, and operand conditioners (the fusion key), so
    the operands are resolved once from the leader; each member contributes one PortWrite tapping its output port
    into its own bank's register.
    """
    leader = min(members)
    node = mir_operation(mir, leader)
    operands = [
        _typed_operand(wide_mir, bool_mir, operand, conditioner, alloc, pool)
        for operand, conditioner in zip(node.operands, node.operand_conditioners, strict=True)
    ]
    swapped = bool(swap.get(leader))
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
            dst=_value_dst(wide_mir, alloc, member),
            conditioner=mir_operation(mir, member).output_conditioner,
        )
        for member in sorted(members, key=tap_port)
    ]
    return PooledScheduledOp(
        inst=inst_of[leader],
        operands=operands,
        writes=writes,
        issue_cycle=sched.issue_cycle[leader],
        latency=node.operator.latency,
        immediates=node.immediates,
    )


def wide_operand(
    wide_mir: MirWideView,
    vid: ValueId,
    conditioner: WideConditioner,
    alloc: Allocation,
    pool: dict[ValueId, PooledConst],
) -> WideOperand:
    node = wide_mir.nodes[vid]
    if isinstance(node, (MirFloatConst, MirIntConst)):
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
        return WideOperand(WideConstRef(entry.index), folded)
    return WideOperand(RegRef(alloc.wide_reg[vid]), conditioner)


def build_outputs(
    mir: Mir,
    wide_mir: MirWideView,
    bool_mir: MirBoolView,
    alloc: Allocation,
    pool: dict[ValueId, PooledConst],
) -> list[WideOutputWire | BoolOutputWire]:
    outputs: list[WideOutputWire | BoolOutputWire] = []
    for out in mir.outputs:
        if isinstance(out, (MirFloatOutput, MirIntOutput)):
            tap = wide_operand(wide_mir, out.value, out.conditioner, alloc, pool)
            outputs.append(WideOutputWire(out.name, tap, wide_mir.scalar_type_of(out.value)))
        elif isinstance(out, MirBoolOutput):
            outputs.append(BoolOutputWire(out.name, bool_operand(bool_mir, out.value, alloc, out.conditioner)))
        else:
            assert False, f"unhandled MIR output {out!r}"
    return outputs


def build_terminator(terminator: MirTerminator, alloc: Allocation) -> Terminator:
    match terminator:
        case MirJump(target=target):
            return Jump(target)
        case MirBranch(cond=cond, if_true=if_true, if_false=if_false):
            return Branch(BoolRegRef(alloc.bool_reg[cond]), if_true, if_false)
        case MirRet():
            return Ret()


def rebase_op(op: PooledScheduledOp, base: int) -> PooledScheduledOp:
    if base == 0:
        return op
    return PooledScheduledOp(
        inst=op.inst,
        operands=op.operands,
        writes=op.writes,
        issue_cycle=op.issue_cycle + base,
        latency=op.latency,
        immediates=op.immediates,
    )


def build_const_pool(
    mir: MirWideView, bool_operations: dict[ValueId, MirOperation] | None = None
) -> tuple[list[WideValue], dict[ValueId, PooledConst]]:
    """
    Build the immediate/ROM pool shared by both wide families, interned by the typed encoded value, so
    encoding-equal float literals share a word and class-aware equality keeps `1` and `1.0` distinct where raw
    Python keys would collide. A FLOAT is stored as its nonnegative magnitude, the sign folded into the consumer's
    free sign-control sideband -- value-preserving because `encode(|c|)` with the sign bit set equals `encode(c)`,
    and MIR normalizes `-0.0` and refuses magnitudes degrading to zero, so a folded negate can never emit the `-0`
    ZKF has no room for. An INTEGER has no sideband and is stored whole. `bool_operations` (the bool-result
    combinational ops) contribute their wide operand constants too.
    """
    ids: list[ValueId] = []
    seen: set[ValueId] = set()

    def note(vid: ValueId) -> None:
        node = mir.nodes.get(vid)  # a bool operand of a bool-result op is not in the wide view; skip it
        if isinstance(node, (MirFloatConst, MirIntConst)) and vid not in seen:
            seen.add(vid)
            ids.append(vid)

    for node in mir.nodes.values():
        if isinstance(node, MirOperation):
            for operand in node.operands:
                note(operand)
        elif isinstance(node, MirPhi):  # a constant phi arm becomes a copy source, so it must be pooled
            for _, arm, _ in node.arms:
                note(arm)
    for operation in (bool_operations or {}).values():
        for operand in operation.operands:
            note(operand)
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
        const = mir.const_nodes[vid]
        if isinstance(const, MirIntConst):
            pool[vid] = PooledConst(intern(IntValue.from_int(mir.int_format, const.value)), IntIdentity())
            continue
        value = const.value
        if math.isnan(value):
            raise UnsupportedConstruct(f"Cannot represent a NaN constant. Only [in]finite numbers are supported.")
        magnitude = FloatValue.from_float(mir.float_format, abs(value))
        negate = math.copysign(1.0, value) < 0.0
        pool[vid] = PooledConst(intern(magnitude), FloatSignControl(negate=negate))
    return values, pool


def tapped_wide_lanes(blocks: list[LirBlock]) -> set[tuple[OperatorInstance, int]]:
    return {
        (op.inst, write.port)
        for block in blocks
        for op in block.ops
        for write in op.writes
        if isinstance(write.dst, RegRef)
    }


def build_inputs(
    mir: Mir, wide_mir: MirWideView, bool_mir: MirBoolView, alloc: Allocation
) -> list[WideInputLoad | BoolInputLoad]:
    loads: list[WideInputLoad | BoolInputLoad] = []
    for vid in mir.input_ids:
        wide_node = wide_mir.nodes.get(vid)
        bool_node = bool_mir.nodes.get(vid)
        if isinstance(wide_node, (MirFloatInput, MirIntInput)):
            loads.append(WideInputLoad(wide_node.name, RegRef(alloc.wide_reg[vid]), wide_node.scalar_type))
        elif isinstance(bool_node, MirBoolInput):
            loads.append(BoolInputLoad(bool_node.name, BoolRegRef(alloc.bool_reg[vid])))
        else:
            assert False, f"unhandled MIR input {vid}"
    return loads
