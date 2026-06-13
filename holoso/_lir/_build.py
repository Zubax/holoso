"""Build a finished :class:`Lir` from MIR."""

import math
from collections.abc import Mapping
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
from .._operators import (
    BoolInversion,
    FloatSignControl,
    InlineHardwareOperator,
    PooledHardwareOperator,
    PortConditioner,
)
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


def build(mir: Mir, module_name: str) -> Lir:
    """
    Schedule, bind, and register-allocate selected MIR into a pipelined microprogram. A straight-line kernel is the
    degenerate single-``Ret``-block control-flow graph, so there is one build path for every kernel.
    """
    if not mir.outputs:
        raise UnsupportedConstruct("Synthesized kernel must produce at least one output value")
    lir = _build_program(mir, module_name)
    names = [port.name for port in lir.ports]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise UnsupportedConstruct(f"duplicate port name(s) in the module interface: {', '.join(duplicates)}")
    return lir


@dataclass(frozen=True, slots=True)
class _FloatArmInstall:
    """A wide phi-arm install at a predecessor's tail: destination register, source value, and the arm's folded sign."""

    dst: int
    source: ValueId
    sign: FloatSignControl


@dataclass(frozen=True, slots=True)
class _BoolArmInstall:
    """A boolean phi-arm install at a predecessor's tail: destination register, source value, and folded inversion."""

    dst: int
    source: ValueId
    inversion: BoolInversion


@dataclass(frozen=True, slots=True)
class _FloatBankAlloc:
    """The wide bank's assignment: register per value, register per state slot, count, and per-slot install cycle."""

    reg: dict[ValueId, int]
    slot_reg: dict[str, int]
    nreg: int
    install: dict[str, int]  # slot name -> Ret-block-relative scheduler-frame install cycle of its live-out


@dataclass(frozen=True, slots=True)
class _BoolBankAlloc:
    """The boolean bank's assignment: register per value, register per state slot, and count."""

    reg: dict[ValueId, int]
    slot_reg: dict[str, int]
    nreg: int


@dataclass(frozen=True, slots=True)
class _Allocation:
    """The register assignment: float/bool registers per value and slot, plus the per-block phi-arm installs."""

    float_reg: dict[ValueId, int]
    float_slot_reg: dict[str, int]
    float_install: dict[str, int]  # slot name -> Ret-block-relative scheduler-frame install cycle of its live-out
    nreg: int
    bool_reg: dict[ValueId, int]
    bool_slot_reg: dict[str, int]
    nbreg: int
    copies: dict[int, list[_FloatArmInstall]]  # block -> wide phi-arm installs at its tail
    bool_writes: dict[int, list[_BoolArmInstall]]  # block -> boolean phi-arm installs at its tail


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


def _build_program(mir: Mir, module_name: str) -> Lir:
    """
    Build the microprogram for any kernel (a straight-line kernel is the degenerate single-``Ret``-block graph):
    schedule each block independently, pool operator instances across the mutually-exclusive blocks, color both register
    banks by hardware-frame liveness (reusing registers, coalescing state live-outs), install non-coalesced phi and slot
    live-outs by pc-gated copy, and lay the blocks out in the ROM with the single ``Ret`` as the out_valid boundary.
    """
    float_mir = MirFloatView.from_mir(mir)
    bool_mir = MirBoolView.from_mir(mir)
    # A branch whose condition is a phi with an arm FROM THE BRANCHING BLOCK cannot be sequenced: the arm's install
    # copy lands in the condition register exactly when the terminator reads it, so the branch would consult the next
    # iteration's value instead of the current one -- and no register assignment can help, since the conflict is the
    # value with itself. The frontend never emits this shape (every arm predecessor is jump-terminated); reject it
    # here so a future pass that creates branch-block arm predecessors fails loudly instead of miscompiling.
    for mir_block in mir.blocks:
        terminator = mir_block.terminator
        if isinstance(terminator, MirBranch):
            cond_node = mir.nodes.get(terminator.cond)
            if isinstance(cond_node, MirPhi) and any(pred == mir_block.id for pred, _, _ in cond_node.arms):
                raise UnsupportedConstruct(
                    f"block {mir_block.id} branches on a phi that takes an arm from the same block; the install "
                    f"would overwrite the condition before the branch reads it"
                )
    pool = resolve_pool(mir.nodes)
    block_sched: dict[int, Schedule] = {
        block.id: schedule_ops(
            mir.nodes,
            pool,
            schedulable=set(float_mir.block_operations(block)) | set(bool_mir.block_operations(block)),
        )
        for block in mir.blocks
    }
    inst_of: dict[ValueId, OperatorInstance] = {}
    inst_count: dict[PooledHardwareOperator, int] = {}
    for sched in block_sched.values():
        inst_of.update(sched.inst_of)
        for inst in sched.instances:
            inst_count[inst.operator] = max(inst_count.get(inst.operator, 0), inst.index + 1)
    instances = [OperatorInstance(operator, i) for operator in inst_count for i in range(inst_count[operator])]
    consts, const_pool = _build_const_pool(float_mir, bool_mir.operation_nodes)
    alloc = _allocate(mir, float_mir, bool_mir, block_sched, inst_of)
    leaders = {leader for sched in block_sched.values() for leader in sched.firings}
    swap = assign_commutative_ports(mir.nodes, inst_of, leaders, alloc.float_reg)

    blocks: list[LirBlock] = []
    for block in mir.blocks:
        sched = block_sched[block.id]
        # One cross-bank schedule per block. Operations split by class, not by result bank: each pooled FIRING (one
        # or more fused taps of one module activation) becomes a PooledScheduledOp with one write per tapped output
        # port; every inline operator (boolean logic, the float<->bool casts) becomes an InlineScheduledOp. Each
        # issues as soon as its own operands have landed, with no barrier.
        ops = [
            _build_pooled_op(mir, float_mir, bool_mir, members, sched, inst_of, alloc, const_pool, swap)
            for _, members in sorted(sched.firings.items(), key=lambda kv: (sched.issue_cycle[kv[0]], kv[0]))
        ]
        inline_ops = [
            _build_inline_op(mir, float_mir, bool_mir, vid, sched.issue_cycle[vid], alloc, const_pool)
            for vid in sorted(
                (v for v in sched.issue_cycle if v not in sched.inst_of),
                key=lambda v: (sched.issue_cycle[v], v),
            )
        ]
        work_makespan = sched.makespan
        install = work_makespan + 1
        copies = [
            FloatCopy(RegRef(c.dst), _operand_signed(float_mir, c.source, c.sign, alloc, const_pool), install)
            for c in alloc.copies.get(block.id, [])
        ]
        bool_writes = [
            BoolWrite(BoolRegRef(w.dst), _bool_operand(bool_mir, w.source, alloc, w.inversion), install)
            for w in alloc.bool_writes.get(block.id, [])
        ]
        has_install = bool(copies or bool_writes)
        block_makespan = install if has_install else work_makespan
        # The latch-free bool bank gives a branch condition exactly one cycle of slack: a bool result committing at the
        # makespan lands one step before the terminator's boundary read. The schedule's makespan covers every commit by
        # construction, so this is a tripwire against a future makespan-computation change only; the emitter-side
        # write-enable placement is guarded by the directed boundary cosim kernel and its white-box twin instead.
        bool_commits = [op.commit_cycle for op in inline_ops if isinstance(op.write.dst, BoolRegRef)] + [
            op.commit_cycle for op in ops if any(isinstance(w.dst, BoolRegRef) for w in op.writes)
        ]
        assert all(
            commit <= block_makespan for commit in bool_commits
        ), f"block {block.id}: a boolean result commits past the block makespan {block_makespan}"
        blocks.append(
            LirBlock(
                block.id,
                ops,
                inline_ops,
                copies,
                bool_writes,
                _build_terminator(block.terminator, alloc),
                block_makespan,
            )
        )

    layout = _layout_blocks(mir, blocks)
    block_base, last_pc, min_ii = layout.block_base, layout.last_pc, layout.min_initiation_interval
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
            _operand_signed(float_mir, slot.live_out, slot.sign, alloc, const_pool),
            block_base[ret_block] + alloc.float_install[slot.name],
        )
        for slot in float_mir.state_slots
    ]
    bool_state_slots = [
        BoolStateSlot(
            bslot.name,
            BoolRegRef(alloc.bool_slot_reg[bslot.name]),
            bool(bslot.reset_value),
            _bool_operand(bool_mir, bslot.live_out, alloc, bslot.inversion),
        )
        for bslot in bool_mir.state_slots
    ]
    outputs = _build_outputs(mir, float_mir, bool_mir, alloc, const_pool)
    return Lir(
        module_name=module_name,
        instances=instances,
        float_consts=consts,
        float_format=float_mir.fmt,
        regfile=RegFileLayout(
            width=float_mir.fmt.width,
            nreg=max(1, alloc.nreg),
            nrd=max(1, sum(inst.operator.arity for inst in instances)),
            nwr=max(1, len(_tapped_wide_lanes(blocks))),
            nload=len(float_mir.input_ids),
        ),
        inputs=_build_inputs(mir, float_mir, bool_mir, alloc),
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


def _bool_operand(bool_mir: MirBoolView, vid: ValueId, alloc: _Allocation, inversion: BoolInversion) -> BoolOperand:
    node = bool_mir.nodes[vid]
    if isinstance(node, MirBoolConst):
        return BoolOperand(BoolConstRef(node.value), inversion)  # folds to the negated immediate at construction
    return BoolOperand(BoolRegRef(alloc.bool_reg[vid]), inversion)


def _typed_operand(
    float_mir: MirFloatView,
    bool_mir: MirBoolView,
    vid: ValueId,
    conditioner: PortConditioner,
    alloc: _Allocation,
    pool: dict[ValueId, _PooledConst],
) -> FloatOperand | BoolOperand:
    """
    One operand resolved in its own bank: a boolean value reads the bool bank with its folded inversion, a
    floating-point value reads the wide bank with its folded sign control.
    """
    if vid in bool_mir.nodes:
        assert isinstance(conditioner, BoolInversion)
        return _bool_operand(bool_mir, vid, alloc, conditioner)
    assert isinstance(conditioner, FloatSignControl)
    return _operand_signed(float_mir, vid, conditioner, alloc, pool)


def _value_dst(float_mir: MirFloatView, alloc: _Allocation, vid: ValueId) -> RegRef | BoolRegRef:
    """The register a value's bank allocated for it: wide for a float-typed tap, boolean otherwise."""
    if vid in float_mir.operation_nodes:
        return RegRef(alloc.float_reg[vid])
    return BoolRegRef(alloc.bool_reg[vid])


def _mir_operation(mir: Mir, vid: ValueId) -> MirOperation:
    node = mir.nodes[vid]
    assert isinstance(node, MirOperation)
    return node


def _build_inline_op(
    mir: Mir,
    float_mir: MirFloatView,
    bool_mir: MirBoolView,
    vid: ValueId,
    issue_cycle: int,
    alloc: _Allocation,
    pool: dict[ValueId, _PooledConst],
) -> InlineScheduledOp:
    """Build one inline firing: operands resolved per bank, the single result written per its tapped port's bank."""
    node = _mir_operation(mir, vid)
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


def _build_pooled_op(
    mir: Mir,
    float_mir: MirFloatView,
    bool_mir: MirBoolView,
    members: list[ValueId],
    sched: Schedule,
    inst_of: dict[ValueId, OperatorInstance],
    alloc: _Allocation,
    pool: dict[ValueId, _PooledConst],
    swap: dict[ValueId, bool],
) -> PooledScheduledOp:
    """
    Build one pooled firing: the members share the operator, operands, and operand conditioners (the fusion key), so
    the operands are resolved once from the leader; each member contributes one PortWrite tapping its output port
    into its own bank's register.
    """
    leader = min(members)
    node = _mir_operation(mir, leader)
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
        port = _mir_operation(mir, member).output_port
        if not swapped:
            return port
        assert permutation is not None
        return permutation[port]

    writes = [
        PortWrite(
            port=tap_port(member),
            dst=_value_dst(float_mir, alloc, member),
            conditioner=_mir_operation(mir, member).output_conditioner,
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


def _operand_signed(
    float_mir: MirFloatView,
    vid: ValueId,
    sign: FloatSignControl,
    alloc: _Allocation,
    pool: dict[ValueId, _PooledConst],
) -> FloatOperand:
    node = float_mir.nodes[vid]
    if isinstance(node, MirFloatConst):
        entry = pool[vid]
        return FloatOperand(FloatConstRef(entry.index), entry.sign.then(sign))
    return FloatOperand(RegRef(alloc.float_reg[vid]), sign)


def _build_outputs(
    mir: Mir,
    float_mir: MirFloatView,
    bool_mir: MirBoolView,
    alloc: _Allocation,
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
            outputs.append(BoolOutputWire(out.name, _bool_operand(bool_mir, out.value, alloc, out.inversion)))
        else:
            raise AssertionError(f"unhandled MIR output {out!r}")
    return outputs


def _build_terminator(terminator: MirTerminator, alloc: _Allocation) -> Terminator:
    match terminator:
        case MirJump(target=target):
            return Jump(target)
        case MirBranch(cond=cond, if_true=if_true, if_false=if_false):
            return Branch(BoolRegRef(alloc.bool_reg[cond]), if_true, if_false)
        case MirRet():
            return Ret()


def _rebase_op(op: PooledScheduledOp, base: int) -> PooledScheduledOp:
    if base == 0:
        return op
    return PooledScheduledOp(
        inst=op.inst,
        operands=op.operands,
        writes=op.writes,
        issue_cycle=op.issue_cycle + base,
        latency=op.latency,
    )


@dataclass(frozen=True, slots=True)
class _BlockLayout:
    """The ROM placement: per-block base PC, the out_valid PC, and the shortest-path initiation interval."""

    block_base: list[int]
    last_pc: int
    min_initiation_interval: int


def _layout_blocks(mir: Mir, blocks: list[LirBlock]) -> _BlockLayout:
    """
    Lay blocks out in the ROM in reverse-postorder, returning their per-block base PCs, the out_valid PC, and the
    shortest-path initiation interval. Each
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
    return _BlockLayout(block_base, last_pc, min_ii)


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


@dataclass(frozen=True, slots=True)
class _PhiInstallFacts:
    """
    One bank's phi-install liveness inputs, per block: the arm values live out of it, and the phi results whose
    install copy writes their register at its tail (one entry per arm predecessor).
    """

    arm_out: dict[int, frozenset[ValueId]]
    installs: dict[int, frozenset[ValueId]]


def _phi_install_facts(mir: Mir, phi_nodes: dict[ValueId, MirPhi], values: set[ValueId]) -> _PhiInstallFacts:
    """
    Per block, one bank's phi-install facts for the liveness inputs: the arm values live out of it (each is read by
    the install copy at its tail) and the phi results whose install copy writes their register there -- one entry
    per arm predecessor.
    """
    arm_out: dict[int, set[ValueId]] = {block.id: set() for block in mir.blocks}
    installs: dict[int, set[ValueId]] = {block.id: set() for block in mir.blocks}
    for vid, phi in phi_nodes.items():
        for pred, arm, _conditioner in phi.arms:
            if arm in values:
                arm_out[pred].add(arm)
            installs[pred].add(vid)  # the phi's register is physically written at this predecessor's tail
    return _PhiInstallFacts(
        {b: frozenset(s) for b, s in arm_out.items()}, {b: frozenset(s) for b, s in installs.items()}
    )


def _block_has_install(mir: Mir, float_mir: MirFloatView, bool_mir: MirBoolView) -> set[int]:
    """
    Blocks whose drained tail carries a phi-arm install (a float copy, a bool write, or a const branch materialization),
    which lengthens the block by one fetch step. Determinable from the CFG shape alone, before register assignment, so
    the liveness boundary uses the same per-block makespan the layout will.
    """
    has: set[int] = set()
    for phi in (*float_mir.phi_nodes.values(), *bool_mir.phi_nodes.values()):
        for pred, _value, _conditioner in phi.arms:
            has.add(pred)
    for block in mir.blocks:
        term = block.terminator
        if isinstance(term, MirBranch) and term.cond in bool_mir.const_nodes:
            has.add(block.id)
    return has


def _allocate_float_bank(
    mir: Mir,
    float_mir: MirFloatView,
    block_sched: dict[int, Schedule],
    inst_of: dict[ValueId, OperatorInstance],
    block_makespan: dict[int, int],
) -> _FloatBankAlloc:
    """
    Color the wide bank across the whole CFG by hardware-frame liveness, reusing registers wherever values do not
    interfere. Inputs pin to the low load lanes and each state live-in to its dedicated slot register. A state slot's
    live-out coalesces onto the slot register when it is an instance-backed operator result whose range does not overlap
    the live-in (the operator writes it directly, no copy); otherwise a pc-gated copy installs it -- as early as the
    live-in is read when the live-out is defined in the Ret block (freeing the source register), the boundary at the
    latest. A coalesced slot register hosts gap tenants; a non-coalesced one is reserved for its read-only live-in. The
    returned ``install`` cycles are Ret-block-relative (absolutized by ``_build_program``).
    """
    nload = len(float_mir.input_ids)
    slots = float_mir.state_slots
    float_slot_reg = {slot.name: nload + i for i, slot in enumerate(slots)}
    fresh_start = nload + len(slots)
    op_nodes = float_mir.operation_nodes
    phi_nodes = float_mir.phi_nodes
    state_read_of = {node.name: vid for vid, node in float_mir.state_read_nodes.items()}
    values = {*float_mir.input_ids, *float_mir.state_read_nodes, *op_nodes, *phi_nodes}
    facts = _bank_liveness_facts(mir, block_sched, op_nodes, phi_nodes, values)
    op_block, op_commit, phi_block, reads = facts.op_block, facts.op_commit, facts.phi_block, facts.reads

    ret_block = next(b.id for b in mir.blocks if isinstance(b.terminator, MirRet))
    phi_facts = _phi_install_facts(mir, phi_nodes, values)
    arm_out, installs = phi_facts.arm_out, phi_facts.installs
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
                op_landing={vid: wide_landing_cycle(commit) for vid, commit in op_commit.items()},
                op_block=op_block,
                phi_block=phi_block,
                reads=block_reads,
                boundary_users={b: frozenset(s) for b, s in boundary.items()},
                arm_out=arm_out,
                installs=installs,
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
    last_read_in_ret: dict[ValueId, int] = {}
    for vid, issue in block_sched[ret_block].issue_cycle.items():
        ret_node = mir.nodes.get(vid)
        if isinstance(ret_node, MirOperation):
            read = operand_read_cycle(ret_node.operator, issue)
            for operand in ret_node.operands:
                last_read_in_ret[operand] = max(last_read_in_ret.get(operand, 0), read)
    install: dict[str, int] = {}
    for slot in slots:
        name, live_out, r_in = slot.name, slot.live_out, livein_of[slot.name]
        node = float_mir.nodes[live_out]
        defined_in_ret = isinstance(node, MirFloatInput) or (
            live_out in op_nodes and op_block.get(live_out) == ret_block
        )
        # A slot whose live-in feeds ANOTHER slot's live-out (a chained copy, ``self.a = self.b``) must hold that
        # live-in until the consuming slot's install reads it -- at the boundary in the worst case -- so it cannot
        # install early: an early install would overwrite the old value before the chained copy captures it.
        early = (
            name not in coalesced
            and name not in tapped_by_other
            and defined_in_ret
            and (r_in is None or r_in not in boundary_outputs[ret_block])
        )
        if early:
            cycle = (op_commit[live_out] if live_out in op_nodes else 0) + 1  # read-first: a strictly older commit
            if r_in is not None:
                # The install fires -- and read-first samples its source -- at ``copy_step_cycle(cycle)``; it must not
                # land before the live-in's last operand read. Deriving the bound from ``operand_read_cycle`` keeps it
                # exact for every consumer class (pooled read latch, inline fire step), present and future.
                cycle = max(cycle, last_read_in_ret.get(r_in, 0) - copy_step_cycle(0))
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
    movable = _movable_order(
        mir, [vid for vid in (*op_nodes, *phi_nodes) if vid not in pinned], op_block, phi_block, op_commit
    )

    # Every pooled use contributes its read ports to the steering objective, whichever bank its TAP lands in: the
    # comparator reads wide registers through real counted read muxes even though its results are boolean.
    consumer_ports: dict[ValueId, set[tuple[OperatorInstance, int]]] = {vid: set() for vid in values}
    for vid, inst in inst_of.items():
        use_node = mir.nodes[vid]
        if not isinstance(use_node, MirOperation):
            continue
        for pos, operand in enumerate(use_node.operands):
            if operand in values:
                consumer_ports[operand].add((inst, pos))
    producer_key: dict[ValueId, OperatorInstance | str] = {vid: "input" for vid in float_mir.input_ids}
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
    return _FloatBankAlloc(assign, float_slot_reg, nreg, install)


@dataclass(frozen=True, slots=True)
class _BankLivenessFacts:
    """
    One bank's per-value liveness inputs: definition block and commit cycle per operation, definition block per phi,
    and the exact per-consumer operand reads (block -> list of (value, read cycle)).
    """

    op_block: dict[ValueId, int]
    op_commit: dict[ValueId, int]
    phi_block: dict[ValueId, int]
    reads: dict[int, list[tuple[ValueId, int]]]


def _bank_liveness_facts(
    mir: Mir,
    block_sched: dict[int, Schedule],
    op_nodes: Mapping[ValueId, MirOperation],
    phi_nodes: Mapping[ValueId, MirPhi],
    values: set[ValueId],
) -> _BankLivenessFacts:
    """
    One bank's per-value liveness facts, identical for both banks so their read-cycle semantics cannot drift:
    definition block and commit cycle per operation, definition block per phi, and EXACT per-consumer operand reads
    (every consumer reads on its own step via the shared cycle helper; the caller adds its bank's boundary users).
    """
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
            rc = operand_read_cycle(node.operator, issue)
            for operand in node.operands:
                if operand in values:
                    reads[block.id].append((operand, rc))
    return _BankLivenessFacts(op_block, op_commit, phi_block, reads)


def _movable_order(
    mir: Mir,
    candidates: list[ValueId],
    op_block: dict[ValueId, int],
    phi_block: dict[ValueId, int],
    op_commit: dict[ValueId, int],
) -> list[ValueId]:
    """
    The deterministic coloring order shared by both banks: reverse-postorder block, then commit cycle, then value id
    (the ``-3`` sentinel sorts a phi -- which has no commit -- ahead of the operations in its block). Value-id last
    keeps the order, and hence the coloring, seed-independent; both banks MUST use this one definition.
    """
    rpo_pos = {bid: i for i, bid in enumerate(_mir_rpo(mir))}
    block_of = {**op_block, **phi_block}
    return sorted(candidates, key=lambda vid: (rpo_pos[block_of[vid]], op_commit.get(vid, -3), vid))


def _allocate_bool_bank(
    mir: Mir, bool_mir: MirBoolView, block_sched: dict[int, Schedule], block_makespan: dict[int, int]
) -> _BoolBankAlloc:
    """
    Color the boolean bank across the CFG, reusing 1-bit registers across non-interfering values. Operand reads are
    EXACT (each consumer reads on its own fire step, via the shared cycle helpers), so a condition consumed mid-block
    frees its register for a later value in the same block -- the select-dense kernels rely on this; only the
    boundary-consumed values (branch conditions, outputs, state live-outs, phi-arm sources) and anything live-out
    into a successor extend to the boundary.
    Inputs pin to the low load lanes and each state live-in to its dedicated slot register (reserved to the boundary).
    """
    nbin = len(bool_mir.input_ids)
    bslots = bool_mir.state_slots
    bool_slot_reg = {slot.name: nbin + i for i, slot in enumerate(bslots)}
    state_read_of = {node.name: vid for vid, node in bool_mir.state_read_nodes.items()}
    fresh_start = nbin + len(bslots)
    op_nodes = bool_mir.operation_nodes
    phi_nodes = bool_mir.phi_nodes
    values = {*bool_mir.input_ids, *bool_mir.state_read_nodes, *op_nodes, *phi_nodes}
    facts = _bank_liveness_facts(mir, block_sched, op_nodes, phi_nodes, values)
    op_block, op_commit, phi_block, reads = facts.op_block, facts.op_commit, facts.phi_block, facts.reads

    boundary: dict[int, set[ValueId]] = {block.id: set() for block in mir.blocks}
    for block in mir.blocks:
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
    phi_facts = _phi_install_facts(mir, phi_nodes, values)
    arm_out, installs = phi_facts.arm_out, phi_facts.installs

    interferes = compute_interference(
        BankLiveness(
            blocks=[b.id for b in mir.blocks],
            entry=mir.entry,
            succ=_succ_map(mir),
            makespan=block_makespan,
            resident=frozenset({*bool_mir.input_ids, *bool_mir.state_read_nodes}),
            op_landing={vid: bool_landing_cycle(commit) for vid, commit in op_commit.items()},
            op_block=op_block,
            phi_block=phi_block,
            reads=reads,
            boundary_users={b: frozenset(s) for b, s in boundary.items()},
            arm_out=arm_out,
            installs=installs,
        )
    )

    pinned: dict[ValueId, int] = {vid: i for i, vid in enumerate(bool_mir.input_ids)}
    for name, reg in bool_slot_reg.items():
        r_in = state_read_of.get(name)
        if r_in is not None:
            pinned[r_in] = reg
    movable = _movable_order(mir, [*op_nodes, *phi_nodes], op_block, phi_block, op_commit)
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
    return _BoolBankAlloc(assign, bool_slot_reg, nbreg)


def _allocate(
    mir: Mir,
    float_mir: MirFloatView,
    bool_mir: MirBoolView,
    block_sched: dict[int, Schedule],
    inst_of: dict[ValueId, OperatorInstance],
) -> _Allocation:
    """
    Assign wide and boolean registers across the CFG. Both banks are colored by hardware-frame liveness, reusing
    registers across mutually-exclusive and non-overlapping live ranges (:func:`_allocate_float_bank`,
    :func:`_allocate_bool_bank`). A phi is resolved by installing each arm's value into the phi's register with a
    copy at the predecessor's tail; the copies are a parallel (simultaneous) bundle, so a swap is read-then-write
    correct.
    """
    block_makespan = {
        b.id: block_sched[b.id].makespan + (1 if b.id in _block_has_install(mir, float_mir, bool_mir) else 0)
        for b in mir.blocks
    }
    float_alloc = _allocate_float_bank(mir, float_mir, block_sched, inst_of, block_makespan)

    copies: dict[int, list[_FloatArmInstall]] = {}
    for vid, phi in float_mir.phi_nodes.items():
        for pred, value, conditioner in phi.arms:
            assert isinstance(conditioner, FloatSignControl)  # a float phi arm carries the float sideband
            copies.setdefault(pred, []).append(_FloatArmInstall(float_alloc.reg[vid], value, conditioner))

    bool_alloc = _allocate_bool_bank(mir, bool_mir, block_sched, block_makespan)

    bool_writes: dict[int, list[_BoolArmInstall]] = {}
    for vid, phi in bool_mir.phi_nodes.items():
        for pred, value, conditioner in phi.arms:
            assert isinstance(conditioner, BoolInversion)
            bool_writes.setdefault(pred, []).append(_BoolArmInstall(bool_alloc.reg[vid], value, conditioner))

    # A constant branch condition (e.g. a read-only boolean attribute, or a folded test) has no register of its own;
    # materialize it into a bool register written in the branching block so the next-PC decode can read it. The constant
    # is globally interned, so sibling branches sharing it reuse one register -- but the write must be emitted in EVERY
    # branching block that uses it, else a path reaching the branch through a block that did not write it reads a stale
    # register. (A later static-branch-folding pass would instead drop the dead arm; until then this keeps it correct.)
    bool_reg, nbreg = bool_alloc.reg, bool_alloc.nreg
    for block in mir.blocks:
        terminator = block.terminator
        if isinstance(terminator, MirBranch) and terminator.cond in bool_mir.const_nodes:
            if terminator.cond not in bool_reg:
                bool_reg[terminator.cond] = nbreg
                nbreg += 1
            bool_writes.setdefault(block.id, []).append(
                _BoolArmInstall(bool_reg[terminator.cond], terminator.cond, BoolInversion())
            )

    return _Allocation(
        float_reg=float_alloc.reg,
        float_slot_reg=float_alloc.slot_reg,
        float_install=float_alloc.install,
        nreg=float_alloc.nreg,
        bool_reg=bool_reg,
        bool_slot_reg=bool_alloc.slot_reg,
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


def _tapped_wide_lanes(blocks: list[LirBlock]) -> set[tuple[OperatorInstance, int]]:
    """The TAPPED wide output-port lanes (one write port each); a never-tapped module output gets no lane."""
    return {
        (op.inst, write.port)
        for block in blocks
        for op in block.ops
        for write in op.writes
        if isinstance(write.dst, RegRef)
    }


def _build_inputs(
    mir: Mir, float_mir: MirFloatView, bool_mir: MirBoolView, alloc: _Allocation
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
