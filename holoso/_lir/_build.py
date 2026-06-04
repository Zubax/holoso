"""Build a finished :class:`Lir` from MIR."""

import math
from dataclasses import dataclass

from .._errors import UnsupportedConstruct
from .._hir import ValueId
from .._mir import Mir, MirFloatConst, MirFloatInput, MirFloatOperation, MirFloatView
from .._operators import FloatSignControl
from ._ir import (
    FloatConstRef,
    FloatInputLoad,
    FloatOperand,
    FloatOutputWire,
    FloatRegFileLayout,
    FloatRegRef,
    FloatScheduledOp,
    FloatStateSlot,
    Lir,
)
from ._portassign import assign_commutative_ports
from ._regalloc import FloatAllocation, allocate_float
from ._schedule import Schedule, resolve_pool, schedule_ops


@dataclass(frozen=True, slots=True)
class _PooledConst:
    """A constant's place in the magnitude pool: its index, plus the sign that recovers the original signed value."""

    index: int
    sign: FloatSignControl


def _operation(mir: MirFloatView, vid: ValueId) -> MirFloatOperation:
    return mir.operation_nodes[vid]


def build(mir: Mir, module_name: str) -> Lir:
    """Schedule, bind, and register-allocate selected MIR into a pipelined microprogram."""
    if not mir.outputs:
        raise UnsupportedConstruct("Synthesized kernel must produce at least one output value")
    float_mir = MirFloatView.from_mir(mir)
    pool = resolve_pool(float_mir)
    sched = schedule_ops(float_mir, pool)
    alloc = allocate_float(float_mir, sched.issue_cycle, sched.inst_of, sched.makespan)
    swap = assign_commutative_ports(float_mir, sched, alloc)
    consts, const_pool = _build_const_pool(float_mir)
    lir = Lir(
        module_name=module_name,
        float_instances=sched.instances,
        float_consts=consts,
        float_regfile=FloatRegFileLayout(
            fmt=float_mir.fmt,
            nreg=alloc.nreg,
            nrd=_compute_nrd(sched),
            nwr=_compute_nwr(sched),
            nload=_compute_nload(float_mir),
        ),
        float_inputs=_build_inputs(float_mir, alloc),
        float_ops=_build_ops(float_mir, sched, alloc, const_pool, swap),
        float_outputs=_build_outputs(float_mir, alloc, const_pool),
        float_state_slots=_build_state_slots(float_mir, alloc, const_pool),
        makespan=sched.makespan,
        op_count=len(float_mir.operation_nodes),
        max_chain_len=_max_chain_len(float_mir),
    )
    names = [port.name for port in lir.ports]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise UnsupportedConstruct(f"duplicate port name(s) in the module interface: {', '.join(duplicates)}")
    return lir


def _build_const_pool(mir: MirFloatView) -> tuple[list[float], dict[ValueId, _PooledConst]]:
    """
    Build the immediate/ROM pool keyed by magnitude: every constant is stored as a nonnegative value, and its sign is
    folded into the consumer's (free) sign-control sideband, so a value and its negation collapse to a single entry.
    This is value-preserving because ``encode(|c|)`` with the sign bit set equals ``encode(c)`` bit-for-bit -- except
    for a magnitude that encodes to zero, where the sign must NOT be folded: ZKF has no negative zero, so a folded
    negate over a zero-encoding magnitude would emit an illegal ``-0`` instead of the canonical ``+0`` that the signed
    value itself encodes to. Such constants therefore keep an identity sign control.
    """
    ids: list[ValueId] = []
    seen: set[ValueId] = set()

    def note(vid: ValueId) -> None:
        node = mir.nodes[vid]
        if isinstance(node, MirFloatConst) and vid not in seen:
            seen.add(vid)
            ids.append(vid)

    for node in mir.nodes.values():
        if isinstance(node, MirFloatOperation):
            for operand in node.operands:
                note(operand)
    for out in mir.outputs:
        note(out.value)
    for slot in mir.state_slots:
        note(slot.live_out)
    values: list[float] = []
    magnitude_index: dict[float, int] = {}
    pool: dict[ValueId, _PooledConst] = {}
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
        pool[vid] = _PooledConst(index, FloatSignControl(negate=negate))
    return values, pool


def _operand(
    mir: MirFloatView,
    vid: ValueId,
    sign: FloatSignControl,
    alloc: FloatAllocation,
    pool: dict[ValueId, _PooledConst],
) -> FloatOperand:
    node = mir.nodes[vid]
    if isinstance(node, MirFloatConst):
        entry = pool[vid]
        return FloatOperand(FloatConstRef(entry.index), entry.sign.then(sign))
    return FloatOperand(FloatRegRef(alloc.assign[vid]), sign)


def _build_ops(
    mir: MirFloatView,
    sched: Schedule,
    alloc: FloatAllocation,
    pool: dict[ValueId, _PooledConst],
    swap: dict[ValueId, bool],
) -> list[FloatScheduledOp]:
    ops: list[FloatScheduledOp] = []
    for vid in sorted(sched.issue_cycle, key=lambda v: (sched.issue_cycle[v], v)):
        node = _operation(mir, vid)
        operands = [
            _operand(mir, operand, sign, alloc, pool)
            for operand, sign in zip(node.operands, node.operand_signs, strict=True)
        ]
        if swap.get(vid):  # commutative operator: exchange operands (with their sign sidebands) to shrink read muxes
            operands.reverse()
        ops.append(
            FloatScheduledOp(
                inst=sched.inst_of[vid],
                operands=operands,
                result_sign=node.result_sign,
                dst=FloatRegRef(alloc.assign[vid]),
                issue_cycle=sched.issue_cycle[vid],
                latency=node.operator.latency,
            )
        )
    return ops


def _build_inputs(mir: MirFloatView, alloc: FloatAllocation) -> list[FloatInputLoad]:
    loads: list[FloatInputLoad] = []
    for vid in mir.input_ids:
        node = mir.nodes[vid]
        if not isinstance(node, MirFloatInput):
            continue
        loads.append(FloatInputLoad(node.name, FloatRegRef(alloc.assign[vid])))
    return loads


def _build_outputs(
    mir: MirFloatView, alloc: FloatAllocation, pool: dict[ValueId, _PooledConst]
) -> list[FloatOutputWire]:
    wires: list[FloatOutputWire] = []
    for out in mir.outputs:
        node = mir.nodes[out.value]
        source: FloatRegRef | FloatConstRef
        if isinstance(node, MirFloatConst):
            entry = pool[out.value]
            source = FloatConstRef(entry.index)
            sign = entry.sign.then(out.sign)
        else:
            source = FloatRegRef(alloc.assign[out.value])
            sign = out.sign
        wires.append(FloatOutputWire(out.name, FloatOperand(source, sign)))
    return wires


def _build_state_slots(
    mir: MirFloatView, alloc: FloatAllocation, pool: dict[ValueId, _PooledConst]
) -> list[FloatStateSlot]:
    slots: list[FloatStateSlot] = []
    for slot in mir.state_slots:
        node = mir.nodes[slot.live_out]
        source: FloatRegRef | FloatConstRef
        if isinstance(node, MirFloatConst):
            entry = pool[slot.live_out]
            source = FloatConstRef(entry.index)
            sign = entry.sign.then(slot.sign)
        else:
            source = FloatRegRef(alloc.assign[slot.live_out])
            sign = slot.sign
        reg = FloatRegRef(alloc.state_regs[slot.name])
        tap = FloatOperand(source, sign)
        slots.append(FloatStateSlot(slot.name, reg, slot.reset_value, tap, alloc.install_cycles[slot.name]))
    return slots


def _compute_nrd(sched: Schedule) -> int:
    """
    Combinational read ports: one dedicated port per operator operand (the sum of instance arities).
    Each ``(instance, operand-position)`` reads from its own fixed port, so the controller word carries only the
    per-port register address and the per-cycle operand-routing mux disappears. Floored to >=1 so the regfile
    parameter guard holds when the kernel has no operators.
    """
    return max(1, sum(inst.operator.arity for inst in sched.instances))


def _compute_nwr(sched: Schedule) -> int:
    """
    Synchronous write ports: one dedicated port per operator instance.
    Each instance's result wires straight to its own write port (no write-data routing mux) and the error/commit
    gating is just that port's write-enable. Floored to >=1 so the regfile parameter guard holds with no operators.
    """
    return max(1, len(sched.instances))


def _compute_nload(mir: MirFloatView) -> int:
    """
    Immediate parallel-load lanes: one unique low register per input port.
    Registers 0..nload-1 are exactly the input block in module port order, including unused inputs retained as ports.
    """
    return len(mir.input_ids)


def _max_chain_len(mir: MirFloatView) -> int:
    depth: dict[ValueId, int] = {}
    op_ids = sorted(mir.operation_nodes)
    for vid in op_ids:
        node = _operation(mir, vid)
        operands = [operand for operand in node.operands if operand in mir.operation_nodes]
        depth[vid] = 1 + max((depth[operand] for operand in operands), default=0)
    return max(depth.values(), default=0)
