"""Build a finished :class:`Lir` from MIR."""

import math
from dataclasses import dataclass

from .._errors import UnsupportedConstruct
from .._hir import ValueId
from .._mir import (
    Mir,
    MirBoolConst,
    MirBoolView,
    MirBranch,
    MirBoolOperation,
    MirFloatConst,
    MirFloatInput,
    MirFloatOperation,
    MirFloatView,
    MirJump,
    MirPhi,
    MirRet,
    MirTerminator,
)
from .._operators import FloatHardwareOperator, FloatSignControl
from ._ir import (
    BoolOperand,
    BoolConstRef,
    BoolRegFileLayout,
    BoolRegRef,
    BoolScheduledOp,
    BoolStateSlot,
    BoolWrite,
    boundary_step,
    Branch,
    FloatConstRef,
    FloatCopy,
    FloatInputLoad,
    FloatOperand,
    FloatOperatorInstance,
    FloatOutputWire,
    FloatRegFileLayout,
    FloatRegRef,
    FloatScheduledOp,
    FloatStateSlot,
    Jump,
    Lir,
    LirBlock,
    Ret,
    Terminator,
)
from ._portassign import assign_commutative_ports
from ._regalloc import FloatAllocation, allocate_float
from ._schedule import DEPENDENCY_EDGE, Schedule, resolve_pool, schedule_ops


@dataclass(frozen=True, slots=True)
class _PooledConst:
    """A constant's place in the magnitude pool: its index, plus the sign that recovers the original signed value."""

    index: int
    sign: FloatSignControl


def _operation(mir: MirFloatView, vid: ValueId) -> MirFloatOperation:
    return mir.operation_nodes[vid]


def build(mir: Mir, module_name: str) -> Lir:
    """
    Schedule, bind, and register-allocate selected MIR into a pipelined microprogram.
    """
    if not mir.outputs:
        raise UnsupportedConstruct("Synthesized kernel must produce at least one output value")
    float_mir = MirFloatView.from_mir(mir)
    bool_mir = MirBoolView.from_mir(mir)
    # TODO FIXME: THIS IS TEMPORARY. The straight-line/single-block path will be merged with CFG eventually.
    is_straight_line = (
        len(mir.blocks) == 1
        and isinstance(mir.blocks[0].terminator, MirRet)
        and not bool_mir.state_slots
        and not bool_mir.nodes
    )
    lir = _build_single_block(mir, float_mir, module_name) if is_straight_line else _build_cfg(mir, module_name)
    names = [port.name for port in lir.ports]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise UnsupportedConstruct(f"duplicate port name(s) in the module interface: {', '.join(duplicates)}")
    return lir


def _build_single_block(mir: Mir, float_mir: MirFloatView, module_name: str) -> Lir:
    pool = resolve_pool(float_mir)
    sched = schedule_ops(float_mir, pool)
    alloc = allocate_float(float_mir, sched.issue_cycle, sched.inst_of, sched.makespan)
    swap = assign_commutative_ports(float_mir, sched, alloc)
    consts, const_pool = _build_const_pool(float_mir)
    float_ops = _build_ops(float_mir, sched, alloc, const_pool, swap)
    boundary = boundary_step(sched.makespan)
    return Lir(
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
        float_ops=float_ops,
        float_outputs=_build_outputs(float_mir, alloc, const_pool),
        float_state_slots=_build_state_slots(float_mir, alloc, const_pool),
        makespan=sched.makespan,
        op_count=len(float_mir.operation_nodes),
        max_chain_len=_max_chain_len(float_mir),
        blocks=[LirBlock(0, float_ops, [], [], [], Ret(), sched.makespan)],
        block_base=[0],
        entry=0,
        last_pc=boundary,
        min_initiation_interval=boundary,
        bool_regfile=BoolRegFileLayout(nreg=0),
        bool_state_slots=[],
    )


@dataclass(frozen=True, slots=True)
class _CfgAllocation:
    """The CFG register assignment: float/bool registers per value and slot, plus the per-block phi-arm installs."""

    float_reg: dict[ValueId, int]
    float_slot_reg: dict[str, int]
    nreg: int
    bool_reg: dict[ValueId, int]
    bool_slot_reg: dict[str, int]
    nbreg: int
    float_copies: dict[int, list[tuple[int, ValueId, FloatSignControl]]]  # block -> [(dst reg, source, folded sign)]
    bool_writes: dict[int, list[tuple[int, ValueId]]]  # block index -> [(dst bool register, source value)]


def _mir_rpo(mir: Mir) -> list[int]:
    """Reverse-postorder of the MIR block CFG from the entry (predecessors before successors)."""
    successors: dict[int, list[int]] = {}
    for block in mir.blocks:
        match block.terminator:
            case MirJump(target=target):
                successors[block.id] = [target]
            case MirBranch(if_true=if_true, if_false=if_false):
                successors[block.id] = [if_true, if_false]
            case MirRet():
                successors[block.id] = []
    order: list[int] = []
    visited: set[int] = set()
    # Iterative DFS (explicit stack) rather than recursion: a deep CFG (e.g. nested unrolled loops chaining thousands
    # of blocks) would otherwise exceed Python's recursion limit. A node is emitted once all its successors are done.
    stack: list[tuple[int, int]] = [(mir.entry, 0)]
    visited.add(mir.entry)
    while stack:
        node, index = stack[-1]
        succs = successors[node]
        if index < len(succs):
            stack[-1] = (node, index + 1)
            successor = succs[index]
            if successor not in visited:
                visited.add(successor)
                stack.append((successor, 0))
        else:
            order.append(node)
            stack.pop()
    return order[::-1]


def _build_cfg(mir: Mir, module_name: str) -> Lir:
    """
    Build a multi-block (control-flow) microprogram: schedule each block independently, pool operator instances across
    the mutually-exclusive blocks, allocate registers installing every phi and slot live-out via a copy (no coalescing
    in v1 -- each takes a fresh register), and lay the blocks out in the ROM with the single ``Ret`` as the out_valid
    boundary.
    """
    float_mir = MirFloatView.from_mir(mir)
    bool_mir = MirBoolView.from_mir(mir)
    pool = resolve_pool(float_mir)
    block_sched: dict[int, Schedule] = {
        block.id: schedule_ops(float_mir, pool, schedulable=set(float_mir.block_operations(block)))
        for block in mir.blocks
    }
    inst_of: dict[ValueId, FloatOperatorInstance] = {}
    inst_count: dict[FloatHardwareOperator, int] = {}
    for sched in block_sched.values():
        inst_of.update(sched.inst_of)
        for inst in sched.instances:
            inst_count[inst.operator] = max(inst_count.get(inst.operator, 0), inst.index + 1)
    instances = [FloatOperatorInstance(operator, i) for operator in inst_count for i in range(inst_count[operator])]
    consts, const_pool = _build_const_pool(float_mir, bool_mir.operation_nodes)
    alloc = _allocate_cfg(mir, float_mir, bool_mir)

    blocks: list[LirBlock] = []
    for block in mir.blocks:
        sched = block_sched[block.id]
        float_ops = [
            _build_cfg_op(float_mir, vid, sched, inst_of, alloc, const_pool)
            for vid in sorted(sched.issue_cycle, key=lambda v: (sched.issue_cycle[v], v))
        ]
        op_makespan = max((op.commit_cycle for op in float_ops), default=0)
        # Comparators read float registers, so they issue after the block's float results are readable. Every block
        # holds at most one comparison (its branch condition) and blocks are mutually exclusive, so all comparisons
        # share one pooled holoso_fcmp instance (the emitter PC-muxes its operands); no per-comparison instance.
        bool_issue = op_makespan + DEPENDENCY_EDGE
        bool_ops = [
            BoolScheduledOp(
                operator=bool_mir.operation_nodes[vid].operator,
                operands=[
                    _cfg_operand_signed(float_mir, operand, sign, alloc, const_pool)
                    for operand, sign in zip(
                        bool_mir.operation_nodes[vid].operands, bool_mir.operation_nodes[vid].operand_signs
                    )
                ],
                dst=BoolRegRef(alloc.bool_reg[vid]),
                relation=bool_mir.operation_nodes[vid].relation,
                issue_cycle=bool_issue,
                latency=bool_mir.operation_nodes[vid].operator.latency,
            )
            for vid in bool_mir.block_operations(block)
        ]
        work_makespan = max([op_makespan, *(bop.commit_cycle for bop in bool_ops)])
        install = work_makespan + 1
        copies = [
            FloatCopy(FloatRegRef(dst), _cfg_operand_signed(float_mir, src, sign, alloc, const_pool), install)
            for dst, src, sign in alloc.float_copies.get(block.id, [])
        ]
        bool_writes = [
            BoolWrite(BoolRegRef(dst), _cfg_bool_operand(bool_mir, src, alloc), install)
            for dst, src in alloc.bool_writes.get(block.id, [])
        ]
        has_install = bool(copies or bool_writes)
        block_makespan = install if has_install else work_makespan
        blocks.append(
            LirBlock(
                block.id,
                float_ops,
                bool_ops,
                copies,
                bool_writes,
                _cfg_terminator(block.terminator, alloc),
                block_makespan,
            )
        )

    block_base, last_pc, min_ii = _layout_blocks(mir, blocks)
    flat_ops = [_rebase_op(op, block_base[block.id]) for block in mir.blocks for op in blocks[block.id].float_ops]
    makespan = max((op.commit_cycle for op in flat_ops), default=0)

    # The slot register holds the live-in (read-only in the body); its live-out is a distinct value installed at the
    # Ret boundary (install_cycle is unused on the CFG path -- the emitter installs at LASTPC, the model at Ret).
    float_state_slots = [
        FloatStateSlot(
            slot.name,
            FloatRegRef(alloc.float_slot_reg[slot.name]),
            slot.reset_value,
            _cfg_operand_signed(float_mir, slot.live_out, slot.sign, alloc, const_pool),
            0,
        )
        for slot in float_mir.state_slots
    ]
    bool_state_slots = [
        BoolStateSlot(
            bslot.name,
            BoolRegRef(alloc.bool_slot_reg[bslot.name]),
            bool(bslot.reset_value),
            _cfg_bool_operand(bool_mir, bslot.live_out, alloc),
        )
        for bslot in bool_mir.state_slots
    ]
    return Lir(
        module_name=module_name,
        float_instances=instances,
        float_consts=consts,
        float_regfile=FloatRegFileLayout(
            fmt=float_mir.fmt,
            nreg=max(1, alloc.nreg),
            nrd=max(1, sum(inst.operator.arity for inst in instances)),
            nwr=max(1, len(instances)),
            nload=len(float_mir.input_ids),
        ),
        float_inputs=[
            FloatInputLoad(node.name, FloatRegRef(alloc.float_reg[vid])) for vid, node in float_mir.input_nodes.items()
        ],
        float_ops=flat_ops,
        float_outputs=_build_cfg_outputs(float_mir, alloc, const_pool),
        float_state_slots=float_state_slots,
        makespan=makespan,
        op_count=len(float_mir.operation_nodes),
        max_chain_len=_max_chain_len(float_mir),
        blocks=blocks,
        block_base=block_base,
        entry=mir.entry,
        last_pc=last_pc,
        min_initiation_interval=min_ii,
        bool_regfile=BoolRegFileLayout(nreg=alloc.nbreg),
        bool_state_slots=bool_state_slots,
    )


def _cfg_operand(
    float_mir: MirFloatView, vid: ValueId, alloc: _CfgAllocation, pool: dict[ValueId, _PooledConst]
) -> FloatOperand:
    node = float_mir.nodes[vid]
    if isinstance(node, MirFloatConst):
        entry = pool[vid]
        return FloatOperand(FloatConstRef(entry.index), entry.sign)
    return FloatOperand(FloatRegRef(alloc.float_reg[vid]))


def _cfg_bool_operand(bool_mir: MirBoolView, vid: ValueId, alloc: _CfgAllocation) -> BoolOperand:
    node = bool_mir.nodes[vid]
    if isinstance(node, MirBoolConst):
        return BoolOperand(BoolConstRef(node.value))
    return BoolOperand(BoolRegRef(alloc.bool_reg[vid]))


def _build_cfg_op(
    float_mir: MirFloatView,
    vid: ValueId,
    sched: Schedule,
    inst_of: dict[ValueId, FloatOperatorInstance],
    alloc: _CfgAllocation,
    pool: dict[ValueId, _PooledConst],
) -> FloatScheduledOp:
    node = float_mir.operation_nodes[vid]
    operands = [
        _cfg_operand_signed(float_mir, operand, sign, alloc, pool)
        for operand, sign in zip(node.operands, node.operand_signs, strict=True)
    ]
    return FloatScheduledOp(
        inst=inst_of[vid],
        operands=operands,
        result_sign=node.result_sign,
        dst=FloatRegRef(alloc.float_reg[vid]),
        issue_cycle=sched.issue_cycle[vid],
        latency=node.operator.latency,
    )


def _cfg_operand_signed(
    float_mir: MirFloatView,
    vid: ValueId,
    sign: FloatSignControl,
    alloc: _CfgAllocation,
    pool: dict[ValueId, _PooledConst],
) -> FloatOperand:
    node = float_mir.nodes[vid]
    if isinstance(node, MirFloatConst):
        entry = pool[vid]
        return FloatOperand(FloatConstRef(entry.index), entry.sign.then(sign))
    return FloatOperand(FloatRegRef(alloc.float_reg[vid]), sign)


def _build_cfg_outputs(
    float_mir: MirFloatView, alloc: _CfgAllocation, pool: dict[ValueId, _PooledConst]
) -> list[FloatOutputWire]:
    wires: list[FloatOutputWire] = []
    for out in float_mir.outputs:
        node = float_mir.nodes[out.value]
        if isinstance(node, MirFloatConst):
            entry = pool[out.value]
            wires.append(FloatOutputWire(out.name, FloatOperand(FloatConstRef(entry.index), entry.sign.then(out.sign))))
        else:
            wires.append(FloatOutputWire(out.name, FloatOperand(FloatRegRef(alloc.float_reg[out.value]), out.sign)))
    return wires


def _cfg_terminator(terminator: MirTerminator, alloc: _CfgAllocation) -> Terminator:
    match terminator:
        case MirJump(target=target):
            return Jump(target)
        case MirBranch(cond=cond, if_true=if_true, if_false=if_false):
            return Branch(BoolRegRef(alloc.bool_reg[cond]), if_true, if_false)
        case MirRet():
            return Ret()


def _rebase_op(op: FloatScheduledOp, base: int) -> FloatScheduledOp:
    if base == 0:
        return op
    return FloatScheduledOp(
        inst=op.inst,
        operands=op.operands,
        result_sign=op.result_sign,
        dst=op.dst,
        issue_cycle=op.issue_cycle + base,
        latency=op.latency,
    )


def _layout_blocks(mir: Mir, blocks: list[LirBlock]) -> tuple[list[int], int, int]:
    """
    Lay blocks out in the ROM in reverse-postorder and return (block_base, last_pc, min_initiation_interval). Each
    block spans ``boundary_step(block_makespan) + 1`` fetch steps (its drained body); the single Ret block's boundary
    is the out_valid PC. ``min_initiation_interval`` is the shortest root-to-Ret path's traversed length.
    """
    successors: dict[int, list[int]] = {}
    for b in blocks:
        match b.terminator:
            case Jump(target=t):
                successors[b.index] = [t]
            case Branch(if_true=t, if_false=f):
                successors[b.index] = [t, f]
            case Ret():
                successors[b.index] = []
    # Blocks are laid out linearly in reverse-postorder, but the single Ret block is forced last so its boundary is the
    # highest address (out_valid = pc == LASTPC). A loop body is a DFS leaf (its only edge back to the header is a back
    # edge), so RPO would otherwise place it after the exit; moving Ret last keeps every loop body below the Ret. A
    # back-edge targets an earlier, lower-addressed block, which the next-PC sequencer redirects like any other jump, so
    # the linear layout needs no special case; the frontend emits reducible loops, so a back-edge target dominates it.
    ret_index = next(b.index for b in blocks if isinstance(b.terminator, Ret))
    order = [bid for bid in _mir_rpo(mir) if bid != ret_index] + [ret_index]
    position = {bid: i for i, bid in enumerate(order)}
    length = {b.index: boundary_step(b.block_makespan) + 1 for b in blocks}
    base: dict[int, int] = {}
    cursor = 0
    for index in order:  # reverse-postorder starts at the entry (block 0), so every block's base is assigned here
        base[index] = cursor
        cursor += length[index]
    last_pc = base[ret_index] + boundary_step(next(b for b in blocks if b.index == ret_index).block_makespan)
    # Shortest path latency (traversed fetch steps) from entry to the Ret boundary. Back-edges are skipped: the minimum
    # latency is the path that exits each loop on its first header test (a loop weighted as not-taken), a true lower
    # bound (the model is the authority on the realized, data-dependent count).
    dist: dict[int, int] = {mir.entry: 0}
    for index in order:
        here = dist.get(index)
        if here is None:
            continue
        for successor in successors[index]:
            if position[successor] <= position[index]:
                continue  # a back-edge; not on a shortest forward path
            cand = here + length[index]
            if successor not in dist or cand < dist[successor]:
                dist[successor] = cand
    min_ii = dist.get(ret_index, 0) + boundary_step(next(b for b in blocks if b.index == ret_index).block_makespan)
    block_base = [base[i] for i in range(len(blocks))]
    return block_base, last_pc, min_ii


def _allocate_cfg(mir: Mir, float_mir: MirFloatView, bool_mir: MirBoolView) -> _CfgAllocation:
    """
    Assign float and boolean registers across the CFG. Inputs pin to the low load lanes and each state slot to a
    dedicated register; every operator result and every phi takes a fresh register (no cross-block reuse). A phi is
    resolved by installing each arm's value into the phi's fresh register with a copy at the predecessor's tail; the
    copies are a parallel (simultaneous) bundle, so a swap is read-then-write correct. The state slot register itself
    is READ-ONLY within the transaction: it holds the live-in, every state read of the slot reads it, and it is never
    overwritten in the body. The slot's live-out (a phi register, an operator result, an input, or a constant) is a
    distinct value; the boundary installs it into the slot register at the single Ret exit, read-first, so a later
    read of the live-in (a branch, an arm, or an output such as ``return old``) still sees the old value. This is the
    invariant that makes the conservative coloring correct without liveness; coalescing is a later liveness-aware pass.
    """
    nload = len(float_mir.input_ids)
    float_reg: dict[ValueId, int] = {vid: i for i, vid in enumerate(float_mir.input_ids)}
    float_slot_reg = {slot.name: nload + i for i, slot in enumerate(float_mir.state_slots)}
    for fvid, fnode in float_mir.state_read_nodes.items():
        float_reg[fvid] = float_slot_reg[fnode.name]
    next_free = nload + len(float_mir.state_slots)

    for vid in (*float_mir.phi_nodes, *float_mir.operation_nodes):  # phis and operator results: each a fresh register
        float_reg[vid] = next_free
        next_free += 1
    nreg = next_free

    float_copies: dict[int, list[tuple[int, ValueId, FloatSignControl]]] = {}
    for vid, phi in float_mir.phi_nodes.items():
        for pred, value, sign in phi.arms:
            float_copies.setdefault(pred, []).append((float_reg[vid], value, sign))

    bool_slot_reg = {slot.name: i for i, slot in enumerate(bool_mir.state_slots)}
    bool_reg: dict[ValueId, int] = {}
    for bvid, bnode in bool_mir.state_read_nodes.items():
        bool_reg[bvid] = bool_slot_reg[bnode.name]
    nbreg = len(bool_mir.state_slots)
    for vid in (*bool_mir.phi_nodes, *bool_mir.operation_nodes):  # phis and comparator results: each a fresh register
        bool_reg[vid] = nbreg
        nbreg += 1

    bool_writes: dict[int, list[tuple[int, ValueId]]] = {}
    for vid, phi in bool_mir.phi_nodes.items():
        for pred, value, _sign in phi.arms:  # a boolean arm carries the identity sign (no sign control on bools)
            bool_writes.setdefault(pred, []).append((bool_reg[vid], value))

    # A constant branch condition (e.g. a read-only boolean attribute, or a folded test) has no register of its own;
    # materialize it into a bool register written in the branching block so the next-PC decode can read it. The constant
    # is globally interned, so sibling branches sharing it reuse one register -- but the write must be emitted in EVERY
    # branching block that uses it, else a path reaching the branch through a block that did not write it reads a stale
    # register. (A later static-branch-folding pass would instead drop the dead arm; until then this keeps it correct.)
    for block in mir.blocks:
        terminator = block.terminator
        if isinstance(terminator, MirBranch) and terminator.cond in bool_mir.const_nodes:
            if terminator.cond not in bool_reg:
                bool_reg[terminator.cond] = nbreg
                nbreg += 1
            bool_writes.setdefault(block.id, []).append((bool_reg[terminator.cond], terminator.cond))

    return _CfgAllocation(
        float_reg=float_reg,
        float_slot_reg=float_slot_reg,
        nreg=nreg,
        bool_reg=bool_reg,
        bool_slot_reg=bool_slot_reg,
        nbreg=nbreg,
        float_copies=float_copies,
        bool_writes=bool_writes,
    )


def _build_const_pool(
    mir: MirFloatView, bool_operations: dict[ValueId, MirBoolOperation] | None = None
) -> tuple[list[float], dict[ValueId, _PooledConst]]:
    """
    Build the immediate/ROM pool keyed by magnitude: every constant is stored as a nonnegative value, and its sign is
    folded into the consumer's (free) sign-control sideband, so a value and its negation collapse to a single entry.
    This is value-preserving because ``encode(|c|)`` with the sign bit set equals ``encode(c)`` bit-for-bit -- except
    for a magnitude that encodes to zero, where the sign must NOT be folded: ZKF has no negative zero, so a folded
    negate over a zero-encoding magnitude would emit an illegal ``-0`` instead of the canonical ``+0`` that the signed
    value itself encodes to. Such constants therefore keep an identity sign control. ``bool_operations`` (float
    comparators) contribute their float operand constants too.
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
