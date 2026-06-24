"""Construct LIR operands, scheduled ops, terminators, outputs, inputs, and the constant pool from selected MIR."""

import math

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
    MirFloatView,
    MirJump,
    MirOperation,
    MirPhi,
    MirRet,
    MirTerminator,
)
from .._operators import BoolInversion, FloatSignControl, InlineHardwareOperator, PortConditioner
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
    float_mir: MirFloatView,
    bool_mir: MirBoolView,
    vid: ValueId,
    conditioner: PortConditioner,
    alloc: Allocation,
    pool: dict[ValueId, PooledConst],
) -> FloatOperand | BoolOperand:
    if vid in bool_mir.nodes:
        assert isinstance(conditioner, BoolInversion)
        return bool_operand(bool_mir, vid, alloc, conditioner)
    assert isinstance(conditioner, FloatSignControl)
    return operand_signed(float_mir, vid, conditioner, alloc, pool)


def _value_dst(float_mir: MirFloatView, alloc: Allocation, vid: ValueId) -> RegRef | BoolRegRef:
    if vid in float_mir.operation_nodes:
        return RegRef(alloc.float_reg[vid])
    return BoolRegRef(alloc.bool_reg[vid])


def build_inline_op(
    mir: Mir,
    float_mir: MirFloatView,
    bool_mir: MirBoolView,
    vid: ValueId,
    issue_cycle: int,
    alloc: Allocation,
    pool: dict[ValueId, PooledConst],
) -> InlineScheduledOp:
    node = mir_operation(mir, vid)
    assert isinstance(node.operator, InlineHardwareOperator)
    operands = [
        _typed_operand(float_mir, bool_mir, operand, conditioner, alloc, pool)
        for operand, conditioner in zip(node.operands, node.operand_conditioners, strict=True)
    ]
    return InlineScheduledOp(
        operator=node.operator,
        operands=operands,
        write=PortWrite(
            port=node.output_port, dst=_value_dst(float_mir, alloc, vid), conditioner=node.output_conditioner
        ),
        issue_cycle=issue_cycle,
        latency=node.operator.latency,
    )


def build_pooled_op(
    mir: Mir,
    float_mir: MirFloatView,
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
        _typed_operand(float_mir, bool_mir, operand, conditioner, alloc, pool)
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
            dst=_value_dst(float_mir, alloc, member),
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
    )


def operand_signed(
    float_mir: MirFloatView,
    vid: ValueId,
    sign: FloatSignControl,
    alloc: Allocation,
    pool: dict[ValueId, PooledConst],
) -> FloatOperand:
    node = float_mir.nodes[vid]
    if isinstance(node, MirFloatConst):
        entry = pool[vid]
        return FloatOperand(FloatConstRef(entry.index), entry.sign.then(sign))
    return FloatOperand(RegRef(alloc.float_reg[vid]), sign)


def build_outputs(
    mir: Mir,
    float_mir: MirFloatView,
    bool_mir: MirBoolView,
    alloc: Allocation,
    pool: dict[ValueId, PooledConst],
) -> list[FloatOutputWire | BoolOutputWire]:
    outputs: list[FloatOutputWire | BoolOutputWire] = []
    for out in mir.outputs:
        if isinstance(out, MirFloatOutput):
            node = float_mir.nodes[out.value]
            if isinstance(node, MirFloatConst):
                entry = pool[out.value]
                outputs.append(
                    FloatOutputWire(out.name, FloatOperand(FloatConstRef(entry.index), entry.sign.then(out.sign)))
                )
            else:
                outputs.append(FloatOutputWire(out.name, FloatOperand(RegRef(alloc.float_reg[out.value]), out.sign)))
        elif isinstance(out, MirBoolOutput):
            outputs.append(BoolOutputWire(out.name, bool_operand(bool_mir, out.value, alloc, out.inversion)))
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
    )


def build_const_pool(
    mir: MirFloatView, bool_operations: dict[ValueId, MirOperation] | None = None
) -> tuple[list[float], dict[ValueId, PooledConst]]:
    """
    Build the immediate/ROM pool keyed by magnitude: every constant is stored as a nonnegative value, and its sign is
    folded into the consumer's (free) sign-control sideband, so a value and its negation collapse to a single entry.
    This is value-preserving because ``encode(|c|)`` with the sign bit set equals ``encode(c)`` bit-for-bit -- except
    for a magnitude that encodes to zero, where the sign must NOT be folded: ZKF has no negative zero, so a folded
    negate over a zero-encoding magnitude would emit an illegal ``-0`` instead of the canonical ``+0`` that the signed
    value itself encodes to. Such constants therefore keep an identity sign control. ``bool_operations`` (the
    bool-result combinational ops -- comparisons, boolean logic, the float->bool cast) contribute their float operand
    constants too.
    """
    ids: list[ValueId] = []
    seen: set[ValueId] = set()

    def note(vid: ValueId) -> None:
        node = mir.nodes.get(vid)  # a bool operand of a bool-result op is not in the float view; skip it
        if isinstance(node, MirFloatConst) and vid not in seen:
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
    values: list[float] = []
    magnitude_index: dict[float, int] = {}
    pool: dict[ValueId, PooledConst] = {}
    for vid in ids:
        value = mir.const_nodes[vid].value
        if not math.isfinite(value):
            raise UnsupportedConstruct(f"non-finite constant {value!r} is not representable in the ZKF format")
        magnitude = abs(value)
        index = magnitude_index.get(magnitude)
        if index is None:
            index = len(values)
            magnitude_index[magnitude] = index
            values.append(magnitude)
        negate = math.copysign(1.0, value) < 0.0 and mir.fmt.encode(magnitude) != 0
        pool[vid] = PooledConst(index, FloatSignControl(negate=negate))
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
    mir: Mir, float_mir: MirFloatView, bool_mir: MirBoolView, alloc: Allocation
) -> list[FloatInputLoad | BoolInputLoad]:
    loads: list[FloatInputLoad | BoolInputLoad] = []
    for vid in mir.input_ids:
        float_node = float_mir.nodes.get(vid)
        bool_node = bool_mir.nodes.get(vid)
        if isinstance(float_node, MirFloatInput):
            loads.append(FloatInputLoad(float_node.name, RegRef(alloc.float_reg[vid])))
        elif isinstance(bool_node, MirBoolInput):
            loads.append(BoolInputLoad(bool_node.name, BoolRegRef(alloc.bool_reg[vid])))
        else:
            assert False, f"unhandled MIR input {vid}"
    return loads
