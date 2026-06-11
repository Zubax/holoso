"""Build a finished :class:`Lir` from MIR."""

import math
from dataclasses import dataclass

from .._errors import UnsupportedConstruct
from .._hir import ValueId
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
    MirFloatStateRead,
    MirOperation,
    MirFloatView,
    MirJump,
    MirPhi,
    MirRet,
    MirTerminator,
)
from .._operators import FloatHardwareOperator, FloatSignControl
from ._ir import *
from ._liveness import BankLiveness, compute_interference
from ._portassign import assign_commutative_ports
from ._regalloc import ColoringProblem, color
from ._schedule import Schedule, resolve_pool, schedule_ops


@dataclass(frozen=True, slots=True)
class _PooledConst:
    """A constant's place in the magnitude pool: its index, plus the sign that recovers the original signed value."""

    index: int
    sign: FloatSignControl


def _operation(mir: MirFloatView, vid: ValueId) -> MirOperation:
    return mir.operation_nodes[vid]


def build(mir: Mir, module_name: str) -> Lir:
    """
    Schedule, bind, and register-allocate selected MIR into a pipelined microprogram. A straight-line kernel is the
    degenerate single-``Ret``-block control-flow graph, so there is one build path for every kernel.
    """
    if not mir.outputs:
        raise UnsupportedConstruct("Synthesized kernel must produce at least one output value")
    lir = _build_cfg(mir, module_name)
    names = [port.name for port in lir.ports]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise UnsupportedConstruct(f"duplicate port name(s) in the module interface: {', '.join(duplicates)}")
    return lir


@dataclass(frozen=True, slots=True)
class _CfgAllocation:
    """The CFG register assignment: float/bool registers per value and slot, plus the per-block phi-arm installs."""

    float_reg: dict[ValueId, int]
    float_slot_reg: dict[str, int]
    float_install: dict[str, int]  # slot name -> Ret-block-relative scheduler-frame install cycle of its live-out
    nreg: int
    bool_reg: dict[ValueId, int]
    bool_slot_reg: dict[str, int]
    nbreg: int
    copies: dict[int, list[tuple[int, ValueId, FloatSignControl]]]  # block -> [(dst reg, source, folded sign)]
    bool_writes: dict[int, list[tuple[int, ValueId]]]  # block index -> [(dst bool register, source value)]


def _mir_rpo(mir: Mir) -> list[int]:
    """Reverse-postorder of the MIR block CFG from the entry (predecessors before successors)."""
    successors = _succ_map(mir)
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
    Build the microprogram for any kernel (a straight-line kernel is the degenerate single-``Ret``-block graph):
    schedule each block independently, pool operator instances across the mutually-exclusive blocks, color both register
    banks by hardware-frame liveness (reusing registers, coalescing state live-outs), install non-coalesced phi and slot
    live-outs by pc-gated copy, and lay the blocks out in the ROM with the single ``Ret`` as the out_valid boundary.
    """
    float_mir = MirFloatView.from_mir(mir)
    bool_mir = MirBoolView.from_mir(mir)
    pool = resolve_pool(float_mir)
    block_sched: dict[int, Schedule] = {
        block.id: schedule_ops(
            mir.nodes,
            pool,
            schedulable=set(float_mir.block_operations(block)) | set(bool_mir.block_operations(block)),
        )
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
    alloc = _allocate_cfg(mir, float_mir, bool_mir, block_sched, inst_of)
    swap = assign_commutative_ports(float_mir, inst_of, alloc.float_reg)

    blocks: list[LirBlock] = []
    for block in mir.blocks:
        sched = block_sched[block.id]
        # One cross-bank schedule per block. Operations split by category, not by result bank: instance-backed float
        # arithmetic (the scheduler bound it an instance) becomes a FloatScheduledOp; every combinational operator
        # (comparison, boolean logic, and the wide-result ``float(cond)`` cast) becomes
        # a CombScheduledOp. Each issues as soon as its own operands have landed, with no barrier.
        ops = [
            _build_cfg_op(float_mir, vid, sched, inst_of, alloc, const_pool, swap)
            for vid in sorted(
                (v for v in sched.issue_cycle if v in inst_of),
                key=lambda v: (sched.issue_cycle[v], v),
            )
        ]
        comb_ops = [
            _build_cfg_comb_op(float_mir, bool_mir, vid, sched.issue_cycle[vid], alloc, const_pool)
            for vid in sorted(
                (v for v in sched.issue_cycle if v not in inst_of),
                key=lambda v: (sched.issue_cycle[v], v),
            )
        ]
        work_makespan = sched.makespan
        install = work_makespan + 1
        copies = [
            FloatCopy(RegRef(dst), _cfg_operand_signed(float_mir, src, sign, alloc, const_pool), install)
            for dst, src, sign in alloc.copies.get(block.id, [])
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
                ops,
                comb_ops,
                copies,
                bool_writes,
                _cfg_terminator(block.terminator, alloc),
                block_makespan,
            )
        )

    block_base, last_pc, min_ii = _layout_blocks(mir, blocks)
    flat_ops = [_rebase_op(op, block_base[block.id]) for block in mir.blocks for op in blocks[block.id].ops]

    # A coalesced slot's live-out tap resolves to the slot register itself (its operator wrote it directly, no copy); a
    # non-coalesced slot taps the live-out's own register, installed at ``install_cycle`` -- absolutized here by adding
    # the Ret block's base, since the install fires inside the (last-laid-out) Ret block.
    ret_block = next(b.id for b in mir.blocks if isinstance(b.terminator, MirRet))
    float_state_slots = [
        FloatStateSlot(
            slot.name,
            RegRef(alloc.float_slot_reg[slot.name]),
            slot.reset_value,
            _cfg_operand_signed(float_mir, slot.live_out, slot.sign, alloc, const_pool),
            block_base[ret_block] + alloc.float_install[slot.name],
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
    outputs = _build_cfg_outputs(mir, float_mir, bool_mir, alloc, const_pool)
    return Lir(
        module_name=module_name,
        instances=instances,
        float_consts=consts,
        float_format=float_mir.fmt,
        regfile=RegFileLayout(
            width=float_mir.fmt.width,
            nreg=max(1, alloc.nreg),
            nrd=max(1, sum(inst.operator.arity for inst in instances)),
            nwr=max(1, len(instances)),
            nload=len(float_mir.input_ids),
        ),
        inputs=_build_cfg_inputs(mir, float_mir, bool_mir, alloc),
        ops=flat_ops,
        outputs=outputs,
        float_state_slots=float_state_slots,
        blocks=blocks,
        block_base=block_base,
        entry=mir.entry,
        last_pc=last_pc,
        min_initiation_interval=min_ii,
        bool_regfile=BoolRegFileLayout(nreg=alloc.nbreg),
        bool_state_slots=bool_state_slots,
    )


def _cfg_bool_operand(bool_mir: MirBoolView, vid: ValueId, alloc: _CfgAllocation) -> BoolOperand:
    node = bool_mir.nodes[vid]
    if isinstance(node, MirBoolConst):
        return BoolOperand(BoolConstRef(node.value))
    return BoolOperand(BoolRegRef(alloc.bool_reg[vid]))


def _cfg_comb_operand(
    float_mir: MirFloatView,
    bool_mir: MirBoolView,
    vid: ValueId,
    sign: FloatSignControl,
    alloc: _CfgAllocation,
    pool: dict[ValueId, _PooledConst],
) -> FloatOperand | BoolOperand:
    """
    One operand of a combinational op, resolved in its own bank: a boolean value reads the bool bank (no sign), a
    floating-point value reads the wide bank (with its folded sign control).
    """
    if vid in bool_mir.nodes:
        return _cfg_bool_operand(bool_mir, vid, alloc)
    return _cfg_operand_signed(float_mir, vid, sign, alloc, pool)


def _build_cfg_comb_op(
    float_mir: MirFloatView,
    bool_mir: MirBoolView,
    vid: ValueId,
    issue_cycle: int,
    alloc: _CfgAllocation,
    pool: dict[ValueId, _PooledConst],
) -> CombScheduledOp:
    """
    Build one combinational scheduled op. Operands are resolved per bank; the destination follows the result bank --
    a bool-result op (comparison, logic, float->bool) writes a bool register, the float-result ``float(cond)`` a wide
    register.
    """
    node = float_mir.operation_nodes.get(vid) or bool_mir.operation_nodes[vid]
    operands = [
        _cfg_comb_operand(float_mir, bool_mir, operand, sign, alloc, pool)
        for operand, sign in zip(node.operands, node.operand_signs)
    ]
    dst: RegRef | BoolRegRef = (
        RegRef(alloc.float_reg[vid]) if vid in float_mir.operation_nodes else BoolRegRef(alloc.bool_reg[vid])
    )
    return CombScheduledOp(
        operator=node.operator, operands=operands, dst=dst, issue_cycle=issue_cycle, latency=node.operator.latency
    )


def _build_cfg_op(
    float_mir: MirFloatView,
    vid: ValueId,
    sched: Schedule,
    inst_of: dict[ValueId, FloatOperatorInstance],
    alloc: _CfgAllocation,
    pool: dict[ValueId, _PooledConst],
    swap: dict[ValueId, bool],
) -> FloatScheduledOp:
    node = float_mir.operation_nodes[vid]
    operands = [
        _cfg_operand_signed(float_mir, operand, sign, alloc, pool)
        for operand, sign in zip(node.operands, node.operand_signs, strict=True)
    ]
    if swap.get(vid):  # commutative operator: exchange operands (with their sign sidebands) to shrink read muxes
        operands.reverse()
    return FloatScheduledOp(
        inst=inst_of[vid],
        operands=operands,
        result_sign=node.result_sign,
        dst=RegRef(alloc.float_reg[vid]),
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
    return FloatOperand(RegRef(alloc.float_reg[vid]), sign)


def _build_cfg_outputs(
    mir: Mir,
    float_mir: MirFloatView,
    bool_mir: MirBoolView,
    alloc: _CfgAllocation,
    pool: dict[ValueId, _PooledConst],
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
            outputs.append(BoolOutputWire(out.name, _cfg_bool_operand(bool_mir, out.value, alloc)))
        else:
            raise AssertionError(f"unhandled MIR output {out!r}")
    return outputs


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


def _succ_map(mir: Mir) -> dict[int, list[int]]:
    """Successor block ids per block, read off the terminators."""
    succ: dict[int, list[int]] = {}
    for block in mir.blocks:
        match block.terminator:
            case MirJump(target=target):
                succ[block.id] = [target]
            case MirBranch(if_true=if_true, if_false=if_false):
                succ[block.id] = [if_true, if_false]
            case MirRet():
                succ[block.id] = []
    return succ


def _block_has_install(mir: Mir, float_mir: MirFloatView, bool_mir: MirBoolView) -> set[int]:
    """
    Blocks whose drained tail carries a phi-arm install (a float copy, a bool write, or a const branch materialization),
    which lengthens the block by one fetch step. Determinable from the CFG shape alone, before register assignment, so
    the liveness boundary uses the same per-block makespan the layout will.
    """
    has: set[int] = set()
    for phi in (*float_mir.phi_nodes.values(), *bool_mir.phi_nodes.values()):
        for pred, _value, _sign in phi.arms:
            has.add(pred)
    for block in mir.blocks:
        term = block.terminator
        if isinstance(term, MirBranch) and term.cond in bool_mir.const_nodes:
            has.add(block.id)
    return has


def _allocate_float_cfg(
    mir: Mir,
    float_mir: MirFloatView,
    block_sched: dict[int, Schedule],
    inst_of: dict[ValueId, FloatOperatorInstance],
    block_makespan: dict[int, int],
) -> tuple[dict[ValueId, int], dict[str, int], int, dict[str, int]]:
    """
    Color the wide bank across the whole CFG by hardware-frame liveness, reusing registers wherever values do not
    interfere. Inputs pin to the low load lanes and each state live-in to its dedicated slot register. A state slot's
    live-out coalesces onto the slot register when it is an instance-backed operator result whose range does not overlap
    the live-in (the operator writes it directly, no copy); otherwise a pc-gated copy installs it -- as early as the
    live-in is read when the live-out is defined in the Ret block (freeing the source register), the boundary at the
    latest. A coalesced slot register hosts gap tenants; a non-coalesced one is reserved for its read-only live-in. The
    returned ``install`` cycles are Ret-block-relative (absolutized by ``_build_cfg``).
    """
    nload = len(float_mir.input_ids)
    slots = float_mir.state_slots
    float_slot_reg = {slot.name: nload + i for i, slot in enumerate(slots)}
    fresh_start = nload + len(slots)
    op_nodes = float_mir.operation_nodes
    phi_nodes = float_mir.phi_nodes
    state_read_of = {node.name: vid for vid, node in float_mir.state_read_nodes.items()}
    values = {*float_mir.input_ids, *float_mir.state_read_nodes, *op_nodes, *phi_nodes}

    op_block: dict[ValueId, int] = {}
    op_commit: dict[ValueId, int] = {}
    phi_block: dict[ValueId, int] = {}
    for block in mir.blocks:
        sched = block_sched[block.id]
        for vid in block.operations:
            if vid in op_nodes:
                op_block[vid] = block.id
                op_commit[vid] = sched.issue_cycle[vid] + op_nodes[vid].operator.latency
        for vid in block.phis:
            if vid in phi_nodes:
                phi_block[vid] = block.id

    reads: dict[int, list[tuple[ValueId, int]]] = {block.id: [] for block in mir.blocks}
    for block in mir.blocks:
        sched = block_sched[block.id]
        for vid, issue in sched.issue_cycle.items():
            node = mir.nodes.get(vid)
            if not isinstance(node, MirOperation):
                continue
            rc = wide_operand_read_cycle(node.operator, issue)
            for operand in node.operands:
                if operand in values:
                    reads[block.id].append((operand, rc))

    ret_block = next(b.id for b in mir.blocks if isinstance(b.terminator, MirRet))
    arm_out: dict[int, set[ValueId]] = {block.id: set() for block in mir.blocks}
    for phi in phi_nodes.values():
        for pred, arm, _sign in phi.arms:
            if arm in values:
                arm_out[pred].add(arm)
    boundary_outputs: dict[int, set[ValueId]] = {block.id: set() for block in mir.blocks}
    for out in mir.outputs:
        if isinstance(out, MirFloatOutput) and out.value in values:
            boundary_outputs[ret_block].add(out.value)

    def graph(
        boundary: dict[int, set[ValueId]], block_reads: dict[int, list[tuple[ValueId, int]]]
    ) -> dict[ValueId, set[ValueId]]:
        return compute_interference(
            BankLiveness(
                blocks=[b.id for b in mir.blocks],
                entry=mir.entry,
                succ=_succ_map(mir),
                makespan=block_makespan,
                resident=frozenset({*float_mir.input_ids, *float_mir.state_read_nodes}),
                op_commit=op_commit,
                op_block=op_block,
                phi_block=phi_block,
                reads=block_reads,
                boundary_users={b: frozenset(s) for b, s in boundary.items()},
                arm_out={b: frozenset(s) for b, s in arm_out.items()},
            )
        )

    # A slot whose live-in is consumed as ANOTHER slot's live-out (a chained copy, ``self.a = self.b``) must keep its
    # live-in to the boundary, so it can neither coalesce nor early-install. The coalescing oracle reads every live-out
    # at the boundary (it must persist) and every live-in at its actual last read, so a live-out that lands after its
    # live-in is fully read shows as non-interfering and coalesces -- the interference-frame form of the WAR test.
    livein_of = {slot.name: state_read_of.get(slot.name) for slot in slots}
    tapped_by_other: set[str] = set()
    for slot in slots:
        node = float_mir.nodes[slot.live_out]
        if isinstance(node, MirFloatStateRead) and node.name != slot.name:
            tapped_by_other.add(node.name)
    boundary_oracle = {b: set(s) for b, s in boundary_outputs.items()}
    for slot in slots:
        if slot.live_out in values:
            boundary_oracle[ret_block].add(slot.live_out)
    coalesce_graph = graph(boundary_oracle, reads)
    coalesced: dict[str, ValueId] = {}  # slot name -> its live-out, pinned onto the slot register (operator writes it)
    for slot in slots:
        live_out = slot.live_out
        r_in = livein_of[slot.name]
        if slot.sign != FloatSignControl() or live_out not in inst_of or slot.name in tapped_by_other:
            continue  # a folded sign or a non-instance live-out cannot be written into the slot register directly
        if r_in is not None and live_out in coalesce_graph.get(r_in, set()):
            continue  # the live-out's range overlaps the live-in's -- it must be copied, not coalesced
        coalesced[slot.name] = live_out

    # Early install for non-coalesced slots: install the live-out as early as the live-in is fully read and the source
    # is available, freeing the source register -- but only when the live-out is defined in the Ret block (the unique,
    # last, once-per-transaction exit), else at the Ret boundary. On one block (every op is in the Ret block) this is
    # exactly the single-block early install. Cycles are Ret-block-relative in the scheduler frame.
    ret_present = block_makespan[ret_block] + 1
    last_issue_in_ret: dict[ValueId, int] = {}
    for vid, issue in block_sched[ret_block].issue_cycle.items():
        node = mir.nodes.get(vid)
        if isinstance(node, MirOperation):
            for operand in node.operands:
                last_issue_in_ret[operand] = max(last_issue_in_ret.get(operand, 0), issue)
    install: dict[str, int] = {}
    for slot in slots:
        name, live_out, r_in = slot.name, slot.live_out, livein_of[slot.name]
        node = float_mir.nodes[live_out]
        defined_in_ret = isinstance(node, MirFloatInput) or (
            live_out in op_nodes and op_block.get(live_out) == ret_block
        )
        early = name not in coalesced and defined_in_ret and (r_in is None or r_in not in boundary_outputs[ret_block])
        if early:
            cycle = (op_commit[live_out] if live_out in op_nodes else 0) + 1  # read-first: a strictly older commit
            if r_in is not None:
                cycle = max(cycle, last_issue_in_ret.get(r_in, 0))  # do not overwrite the live-in before its last read
            install[name] = min(cycle, ret_present)
        else:
            install[name] = ret_present

    # Final interference. A non-coalesced slot reserves its live-in to the boundary (the install reads it read-first, so
    # the register holds nothing else); a coalesced slot keeps its live-in's actual range, so a gap tenant lands between
    # the live-in's last read and the live-out's landing. A boundary-installed live-out is read at the boundary; an
    # early-installed one is read by its copy at the install step, freeing its source register for a tenant after that.
    boundary_final = {b: set(s) for b, s in boundary_outputs.items()}
    reads_final = {b: list(r) for b, r in reads.items()}
    for slot in slots:
        name, live_out, r_in = slot.name, slot.live_out, livein_of[slot.name]
        if name in coalesced:
            if live_out in values:
                boundary_final[ret_block].add(live_out)  # persists in the slot register to the next initiation
            continue
        if r_in is not None:
            boundary_final[ret_block].add(r_in)
        if live_out not in values:
            continue
        if install[name] < ret_present:
            reads_final[ret_block].append((live_out, copy_step_cycle(install[name])))
        else:
            boundary_final[ret_block].add(live_out)
    interferes = graph(boundary_final, reads_final)

    pinned: dict[ValueId, int] = {vid: i for i, vid in enumerate(float_mir.input_ids)}
    for name, reg in float_slot_reg.items():
        r_in = livein_of[name]
        if r_in is not None:
            pinned[r_in] = reg
    for name, live_out in coalesced.items():  # the coalesced live-out shares its slot register (the operator writes it)
        pinned[live_out] = float_slot_reg[name]
    movable: list[ValueId] = [vid for vid in (*op_nodes, *phi_nodes) if vid not in pinned]

    rpo_pos = {bid: i for i, bid in enumerate(_mir_rpo(mir))}
    block_of = {**op_block, **phi_block}
    movable.sort(key=lambda vid: (rpo_pos[block_of[vid]], landing_cycle(op_commit.get(vid, -3)), vid))

    consumer_ports: dict[ValueId, set[tuple[FloatOperatorInstance, int]]] = {vid: set() for vid in values}
    for vid, inst in inst_of.items():
        node = op_nodes.get(vid)
        if node is None:
            continue
        for pos, operand in enumerate(node.operands):
            if operand in values:
                consumer_ports[operand].add((inst, pos))
    producer_key: dict[ValueId, FloatOperatorInstance | str] = {vid: "input" for vid in float_mir.input_ids}
    producer_key.update({vid: f"state:{node.name}" for vid, node in float_mir.state_read_nodes.items()})
    producer_key.update({vid: (inst_of[vid] if vid in inst_of else f"cast:{vid}") for vid in op_nodes})
    producer_key.update({vid: f"phi:{vid}" for vid in phi_nodes})

    assign, nreg = color(
        ColoringProblem(
            movable=movable,
            pinned=pinned,
            interferes=interferes,
            consumer_ports=consumer_ports,
            producer_key=producer_key,
            fresh_start=fresh_start,
        )
    )
    # Backstop: a non-coalesced slot register must carry nothing but its own live-in (its install copy folds no tenant).
    for slot in slots:
        if slot.name in coalesced:
            continue
        reg = float_slot_reg[slot.name]
        occupants = [vid for vid, r in assign.items() if r == reg and vid != livein_of[slot.name]]
        assert not occupants, f"non-coalesced slot register {reg} ({slot.name!r}) has occupants {occupants}"
    return assign, float_slot_reg, nreg, install


def _allocate_bool_cfg(
    mir: Mir, bool_mir: MirBoolView, block_sched: dict[int, Schedule], block_makespan: dict[int, int]
) -> tuple[dict[ValueId, int], dict[str, int], int]:
    """
    Color the boolean bank across the CFG, reusing 1-bit registers across non-interfering values. Liveness is
    conservative -- every boolean value is held to its block's boundary -- which disables intra-block reuse but lets the
    mutually-exclusive and sequential branch conditions in distinct blocks collapse onto a few registers. Inputs pin to
    the low load lanes and each state live-in to its dedicated slot register (reserved to the boundary).
    """
    nbin = len(bool_mir.input_ids)
    bslots = bool_mir.state_slots
    bool_slot_reg = {slot.name: nbin + i for i, slot in enumerate(bslots)}
    state_read_of = {node.name: vid for vid, node in bool_mir.state_read_nodes.items()}
    fresh_start = nbin + len(bslots)
    op_nodes = bool_mir.operation_nodes
    phi_nodes = bool_mir.phi_nodes
    values = {*bool_mir.input_ids, *bool_mir.state_read_nodes, *op_nodes, *phi_nodes}

    op_block: dict[ValueId, int] = {}
    op_commit: dict[ValueId, int] = {}
    phi_block: dict[ValueId, int] = {}
    for block in mir.blocks:
        sched = block_sched[block.id]
        for vid in block.operations:
            if vid in op_nodes:
                op_block[vid] = block.id
                op_commit[vid] = sched.issue_cycle[vid] + op_nodes[vid].operator.latency
        for vid in block.phis:
            if vid in phi_nodes:
                phi_block[vid] = block.id

    reads: dict[int, list[tuple[ValueId, int]]] = {block.id: [] for block in mir.blocks}
    boundary: dict[int, set[ValueId]] = {block.id: set() for block in mir.blocks}
    for block in mir.blocks:
        bnd = boundary_step(block_makespan[block.id])
        for vid in block_sched[block.id].issue_cycle:
            node = mir.nodes.get(vid)
            if isinstance(node, MirOperation):
                for operand in node.operands:
                    if operand in values:
                        reads[block.id].append((operand, bnd))  # conservative: read at the block boundary
        if isinstance(block.terminator, MirBranch) and block.terminator.cond in values:
            boundary[block.id].add(block.terminator.cond)
    ret_block = next(b.id for b in mir.blocks if isinstance(b.terminator, MirRet))
    for out in mir.outputs:
        if isinstance(out, MirBoolOutput) and out.value in values:
            boundary[ret_block].add(out.value)
    for slot in bslots:
        if slot.live_out in values:
            boundary[ret_block].add(slot.live_out)
        r_in = state_read_of.get(slot.name)
        if r_in is not None:
            boundary[ret_block].add(r_in)  # reserve the slot register (its live-in is held read-first to the boundary)
    arm_out: dict[int, set[ValueId]] = {block.id: set() for block in mir.blocks}
    for phi in phi_nodes.values():
        for pred, arm, _sign in phi.arms:
            if arm in values:
                arm_out[pred].add(arm)

    interferes = compute_interference(
        BankLiveness(
            blocks=[b.id for b in mir.blocks],
            entry=mir.entry,
            succ=_succ_map(mir),
            makespan=block_makespan,
            resident=frozenset({*bool_mir.input_ids, *bool_mir.state_read_nodes}),
            op_commit=op_commit,
            op_block=op_block,
            phi_block=phi_block,
            reads=reads,
            boundary_users={b: frozenset(s) for b, s in boundary.items()},
            arm_out={b: frozenset(s) for b, s in arm_out.items()},
        )
    )

    pinned: dict[ValueId, int] = {vid: i for i, vid in enumerate(bool_mir.input_ids)}
    for name, reg in bool_slot_reg.items():
        r_in = state_read_of.get(name)
        if r_in is not None:
            pinned[r_in] = reg
    rpo_pos = {bid: i for i, bid in enumerate(_mir_rpo(mir))}
    block_of = {**op_block, **phi_block}
    movable = sorted(
        (vid for vid in (*op_nodes, *phi_nodes)),
        key=lambda vid: (rpo_pos[block_of[vid]], landing_cycle(op_commit.get(vid, -3)), vid),
    )
    # One coloring engine for both banks. The boolean bank has no read multiplexer and a one-hot pc-gated write chain,
    # so it carries no read ports and a single uniform producer -- the steering objective then degenerates to register
    # count, and the reach-aware colorer reduces to count-minimizing first-fit.
    assign, nbreg = color(
        ColoringProblem(
            movable=movable,
            pinned=pinned,
            interferes=interferes,
            consumer_ports={vid: set() for vid in values},
            producer_key={vid: "bool" for vid in values},
            fresh_start=fresh_start,
        )
    )
    return assign, bool_slot_reg, nbreg


def _allocate_cfg(
    mir: Mir,
    float_mir: MirFloatView,
    bool_mir: MirBoolView,
    block_sched: dict[int, Schedule],
    inst_of: dict[ValueId, FloatOperatorInstance],
) -> _CfgAllocation:
    """
    Assign wide and boolean registers across the CFG. Both banks are colored by hardware-frame liveness, reusing
    registers across mutually-exclusive and non-overlapping live ranges (:func:`_allocate_float_cfg`,
    :func:`_allocate_bool_cfg`). A phi is resolved by installing each arm's value into the phi's register with a copy at
    the predecessor's tail; the copies are a parallel (simultaneous) bundle, so a swap is read-then-write correct.
    """
    block_makespan = {
        b.id: block_sched[b.id].makespan + (1 if b.id in _block_has_install(mir, float_mir, bool_mir) else 0)
        for b in mir.blocks
    }
    float_reg, float_slot_reg, nreg, float_install = _allocate_float_cfg(
        mir, float_mir, block_sched, inst_of, block_makespan
    )

    copies: dict[int, list[tuple[int, ValueId, FloatSignControl]]] = {}
    for vid, phi in float_mir.phi_nodes.items():
        for pred, value, sign in phi.arms:
            copies.setdefault(pred, []).append((float_reg[vid], value, sign))

    bool_reg, bool_slot_reg, nbreg = _allocate_bool_cfg(mir, bool_mir, block_sched, block_makespan)

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
        float_install=float_install,
        nreg=nreg,
        bool_reg=bool_reg,
        bool_slot_reg=bool_slot_reg,
        nbreg=nbreg,
        copies=copies,
        bool_writes=bool_writes,
    )


def _build_const_pool(
    mir: MirFloatView, bool_operations: dict[ValueId, MirOperation] | None = None
) -> tuple[list[float], dict[ValueId, _PooledConst]]:
    """
    Build the immediate/ROM pool keyed by magnitude: every constant is stored as a nonnegative value, and its sign is
    folded into the consumer's (free) sign-control sideband, so a value and its negation collapse to a single entry.
    This is value-preserving because ``encode(|c|)`` with the sign bit set equals ``encode(c)`` bit-for-bit -- except
    for a magnitude that encodes to zero, where the sign must NOT be folded: ZKF has no negative zero, so a folded
    negate over a zero-encoding magnitude would emit an illegal ``-0`` instead of the canonical ``+0`` that the signed
    value itself encodes to. Such constants therefore keep an identity sign control. ``bool_operations`` (the bool-result
    combinational ops -- comparisons, boolean logic, the float->bool cast) contribute their float operand constants too.
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


def _build_cfg_inputs(
    mir: Mir, float_mir: MirFloatView, bool_mir: MirBoolView, alloc: _CfgAllocation
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
            raise AssertionError(f"unhandled MIR input {vid}")
    return loads
