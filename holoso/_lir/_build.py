"""
Build a finished :class:`Lir` from MIR: the top-level orchestration that schedules and lays out the blocks, allocates
both register banks (coalescing phi arms), constructs the per-block LIR, and assembles the final program.
"""

from .._errors import UnsupportedConstruct
from .._mir import Mir, MirBoolView, MirBranch, MirFloatView, MirPhi, MirRet
from ._ir import *
from ._mir_facts import block_has_install, const_branch_conditions
from ._portassign import assign_commutative_ports
from ._schedule import resolve_pool
from ._bankalloc import actual_install_blocks, layout_and_allocate
from ._construct import (
    bool_operand,
    build_inline_op,
    build_inputs,
    build_outputs,
    build_pooled_op,
    build_terminator,
    operand_signed,
    rebase_op,
    tapped_wide_lanes,
)
from ._layout import install_inclusive_makespan, layout_blocks


def build(mir: Mir, module_name: str) -> Lir:
    """
    Schedule, bind, and register-allocate selected MIR into a pipelined microprogram. A straight-line kernel is the
    degenerate single-``Ret``-block control-flow graph, so there is one build path for every kernel.
    """
    if not mir.outputs:
        raise UnsupportedConstruct("Synthesized kernel must produce at least one output value")
    lir = _build_program(mir, module_name)
    _assert_entry_dwell_safe(lir)
    names = [port.name for port in lir.ports]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise UnsupportedConstruct(f"duplicate port name(s) in the module interface: {', '.join(duplicates)}")
    return lir


def _assert_entry_dwell_safe(lir: Lir) -> None:
    """
    Build-time invariant for the accept-dwell contract: no entry-block op issued on cycle 0 writes a persistent-state
    register. The sequencer holds pc 0 while waiting for ``in_valid`` and re-fires ``ucode[0]`` each idle cycle, so a
    cycle-0 write to a state register would be re-driven with stale inputs and corrupt the carried state. In-place state
    commit makes a producer that coalesces onto a slot register write that register directly -- an unconditional
    ``self.x = self.x | a`` (its OR/Select live-out) or an entry-block arm of a conditional update -- so the hazard is
    real, not merely theoretical, and is prevented by the dwell floor in ``schedule_with_overlap`` (the
    ``state_liveouts`` set, which includes both slot live-outs and the arm producers of any phi live-out, is held off
    cycle 0 in the entry block). This assertion is the loud backstop on that floor: the dwell is invisible to BOTH
    validation paths (the cosim bench never delays ``in_valid``; the model asserts it at once and keys cycle-0 ops at
    their read pc, never pc 0), so a regression that let a coalesced result reach a state register on cycle 0 would
    otherwise corrupt state silently.
    """
    entry = lir.blocks[lir.entry]
    state_regs = {slot.reg for slot in lir.float_state_slots} | {slot.reg for slot in lir.bool_state_slots}
    cycle0_writes = [w.dst for op in entry.ops if op.issue_cycle == 0 for w in op.writes]
    cycle0_writes += [op.write.dst for op in entry.inline_ops if op.issue_cycle == 0]
    for dst in cycle0_writes:
        assert dst not in state_regs, f"entry-block cycle-0 op writes persistent-state register {dst.stable_label}"


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
    # Schedule every block in reverse-postorder (a block after its forward-edge predecessors) and lay out each block's
    # terminator offset, with cross-block software pipelining: a block whose successors are all single-predecessor
    # shrinks its terminator below the drained boundary and spills its in-flight results into the successor, which
    # inherits the busy/landing residue. The +1-install drain keeps install-bearing blocks unshrunk, matching makespan.
    #
    # The install set is computed to a fixpoint. ``block_has_install`` marks a block install-bearing from the CFG shape
    # (any phi arm originates in it), but a block whose every arm COALESCES onto the merged register installs nothing,
    # so that +1 drain (and overlap-ineligibility) is spurious. So: lay out and allocate with the conservative CFG set,
    # recompute the install set from the ACTUAL coalesced copies, and re-run until it stops shrinking. Convergence is by
    # monotonicity -- dropping a block's spurious drain frees registers one step earlier, which only enables more
    # coalescing, so the install set is non-increasing over a finite block set and reaches a fixpoint (the assert guards
    # the monotonicity; the iteration count is bounded by the block count). Determinism is preserved: the allocator is
    # seed-fixed and the install set is rebuilt the same way each pass.
    #
    # This install fixpoint NESTS a second one: each ``layout_and_allocate`` round runs the per-bank phi-coalescing /
    # coloring fixpoint in ``coalesce_and_color``. Both are bounded and monotone -- the install set non-increasing here,
    # the inner forbidden-merge set non-decreasing there -- so the composition terminates in at most block-count outer
    # rounds, each a bounded inner fixpoint. The coupling is one-way and cannot deadlock: a shrinking install set only
    # relieves register pressure, enabling more coalescing, never forbidding a merge the inner round already made.
    const_branch_blocks = set(const_branch_conditions(mir, bool_mir))
    has_install_blocks = block_has_install(mir, float_mir, bool_mir)
    for _ in range(len(mir.blocks) + 1):
        result = layout_and_allocate(mir, float_mir, bool_mir, pool, has_install_blocks)
        actual = actual_install_blocks(result.alloc, const_branch_blocks)
        assert actual <= has_install_blocks, "the coalesced-install fixpoint must not grow the install set"
        if actual == has_install_blocks:
            break
        has_install_blocks = actual
    else:
        assert False, "coalesced-install fixpoint did not converge"  # unreachable: monotone over finite blocks
    overlap = result.overlap
    block_sched = overlap.block_sched
    inst_of = result.inst_of
    instances = result.instances
    consts, const_pool = result.consts, result.const_pool
    alloc = result.alloc
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
            build_pooled_op(mir, float_mir, bool_mir, members, sched, inst_of, alloc, const_pool, swap)
            for _, members in sorted(sched.firings.items(), key=lambda kv: (sched.issue_cycle[kv[0]], kv[0]))
        ]
        inline_ops = [
            build_inline_op(mir, float_mir, bool_mir, vid, sched.issue_cycle[vid], alloc, const_pool)
            for vid in sorted(
                (v for v in sched.issue_cycle if v not in sched.inst_of),
                key=lambda v: (sched.issue_cycle[v], v),
            )
        ]
        work_makespan = sched.makespan
        install = work_makespan + 1
        copies = [
            FloatCopy(RegRef(c.dst), operand_signed(float_mir, c.source, c.sign, alloc, const_pool), install)
            for c in alloc.copies.get(block.id, [])
        ]
        bool_writes = [
            BoolWrite(BoolRegRef(w.dst), bool_operand(bool_mir, w.source, alloc, w.inversion), install)
            for w in alloc.bool_writes.get(block.id, [])
        ]
        has_install = block.id in has_install_blocks
        block_makespan = install_inclusive_makespan(work_makespan, has_install)
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
                build_terminator(block.terminator, alloc),
                block_makespan,
                # The terminator offset from the overlap layout: the drained boundary, or shrunk to the issue-side
                # envelope when this block's in-flight results spill into single-predecessor successors.
                overlap.block_term_offset[block.id],
            )
        )

    layout = layout_blocks(mir, blocks)
    block_base, last_pc, min_ii = layout.block_base, layout.last_pc, layout.min_initiation_interval
    flat_ops = [rebase_op(op, block_base[block.id]) for block in mir.blocks for op in blocks[block.id].ops]

    # A coalesced slot's live-out tap resolves to the slot register itself (its operator wrote it directly, no copy); a
    # non-coalesced slot taps the live-out's own register, installed at ``install_cycle`` -- absolutized here by adding
    # the Ret block's base, since the install fires inside the (last-laid-out) Ret block.
    ret_block = next(b.id for b in mir.blocks if isinstance(b.terminator, MirRet))
    float_state_slots = [
        FloatStateSlot(
            slot.name,
            RegRef(alloc.float_slot_reg[slot.name]),
            slot.reset_value,
            operand_signed(float_mir, slot.live_out, slot.sign, alloc, const_pool),
            block_base[ret_block] + alloc.float_install[slot.name],
        )
        for slot in float_mir.state_slots
    ]
    bool_state_slots = [
        BoolStateSlot(
            bslot.name,
            BoolRegRef(alloc.bool_slot_reg[bslot.name]),
            bool(bslot.reset_value),
            bool_operand(bool_mir, bslot.live_out, alloc, bslot.inversion),
        )
        for bslot in bool_mir.state_slots
    ]
    outputs = build_outputs(mir, float_mir, bool_mir, alloc, const_pool)
    lir = Lir(
        module_name=module_name,
        instances=instances,
        float_consts=consts,
        float_format=float_mir.fmt,
        regfile=RegFileLayout(
            width=float_mir.fmt.width,
            nreg=max(1, alloc.nreg),
            nrd=max(1, sum(inst.operator.arity for inst in instances)),
            nwr=max(1, len(tapped_wide_lanes(blocks))),
            nload=len(float_mir.input_ids),
        ),
        inputs=build_inputs(mir, float_mir, bool_mir, alloc),
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
    # A non-coalesced float slot's writeback fires read-first at ``state_copy_step``, at last_pc for a boundary install
    # or below it for an early one. A boundary that collapsed below the install would drop the writeback and freeze the
    # persistent state; the per-block ``term_offset <= wide drain`` invariant in ``schedule_with_overlap`` is the
    # matching guard for the opposite slip (a boundary install degrading into an early one). Backstop, not a live
    # failure.
    for slot in lir.float_state_slots:
        if slot.needs_copy:
            assert (
                lir.state_copy_step(slot) <= last_pc
            ), f"state slot {slot.name!r} writeback at {lir.state_copy_step(slot)} lands past the boundary {last_pc}"
    return lir
