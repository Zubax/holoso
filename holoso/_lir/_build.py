"""
Build a finished :class:`Lir` from MIR: the top-level orchestration that schedules and lays out the blocks, allocates
both register banks (coalescing phi arms), constructs the per-block LIR, and assembles the final program.
"""

from .._errors import UnsupportedConstruct
from .._mir import Mir, MirBoolView, MirBranch, MirFloatView, MirPhi
from .._util import ValueId
from ._ir import *
from ._mir_facts import block_has_install, value_resident_at_entry
from ._portassign import assign_commutative_ports
from ._schedule import resolve_pool
from ._bankalloc import actual_install_blocks, layout_and_allocate
from ._build_base import Allocation, PooledConst
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
    names = [port.name for port in lir.ports]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise UnsupportedConstruct(f"duplicate port name(s) in the module interface: {', '.join(duplicates)}")
    return lir


def _has_state_copy(
    float_mir: MirFloatView, bool_mir: MirBoolView, alloc: Allocation, const_pool: dict[ValueId, PooledConst]
) -> bool:
    """
    Whether the single Ret block's state live-out does NOT coalesce onto its slot register. A non-coalesced slot
    installs by a read-first boundary copy that lands a fetch-pipeline past the live-out, so the Ret block's drain must
    reach it (``boundary_step(makespan)``, bank-independent). A coalesced slot writes its register in place and needs no
    copy (no charge). True iff any slot, float or boolean, is non-coalesced. Recomputed from the allocation each
    coalescing-fixpoint round through the same ``*_liveout_coalesced`` predicates ``build`` applies when it emits the
    install, so the drain charge and the emitted install cannot drift.
    """
    return any(
        not float_liveout_coalesced(
            operand_signed(float_mir, slot.live_out, slot.sign, alloc, const_pool),
            RegRef(alloc.float_slot_reg[slot.name]),
        )
        for slot in float_mir.state_slots
    ) or any(
        not bool_liveout_coalesced(
            bool_operand(bool_mir, bslot.live_out, alloc, bslot.inversion), BoolRegRef(alloc.bool_slot_reg[bslot.name])
        )
        for bslot in bool_mir.state_slots
    )


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
    # the monotonicity; the loop's iteration bound is derived below). Determinism is preserved: the allocator is
    # seed-fixed and the install set is rebuilt the same way each pass.
    #
    # This install fixpoint NESTS a second one: each ``layout_and_allocate`` round runs the per-bank phi-coalescing /
    # coloring fixpoint in ``coalesce_and_color``. Both are bounded and monotone -- the install set non-increasing here,
    # the inner forbidden-merge set non-decreasing there -- so the composition terminates in a bounded number of outer
    # rounds (the loop bound below), each a bounded inner fixpoint. The coupling is one-way and cannot deadlock: a
    # shrinking install set only relieves register pressure, enabling more coalescing, never forbidding a merge the
    # inner round already made.
    #
    # The same fixpoint also sheds the state slot's read-first boundary-copy drain charge once the slot coalesces:
    # ``has_state_copy`` starts conservative (a state slot needs a copy) and clears as coalescing removes it, monotone
    # alongside the install set. It is a single bool: the MIR has one Ret, so the charge is op-wide on that block.
    has_install_blocks = block_has_install(mir, float_mir, bool_mir)
    ret_block = mir.ret_block
    # Conservative seed for the state-copy fixpoint -- the pre-allocation form of ``_has_state_copy``: assume every
    # state slot needs a boundary copy.
    has_state_copy = bool(float_mir.state_slots or bool_mir.state_slots)
    # Iteration bound. All three descending quantities only drop (monotonicity argument at the asserts below): the
    # install set (<= len(blocks) removals as coalescing frees registers), the per-block push bit (<= len(blocks)
    # narrowings), and the single-keyed state-copy charge (height 1). Worst case is their SUM plus a confirming round;
    # the 2*len(blocks)+3 loop bound below leaves a safe margin, with the asserts and the else-clause as loud backstops.
    for _ in range(2 * len(mir.blocks) + 3):
        result = layout_and_allocate(mir, float_mir, bool_mir, pool, has_install_blocks, has_state_copy)
        actual = actual_install_blocks(result.alloc, float_mir, bool_mir, result.overlap.block_sched)
        actual_state = _has_state_copy(float_mir, bool_mir, result.alloc, result.const_pool)
        # The descent is monotone and the fixed point is sound. MONOTONE: a block leaves the install set only by
        # coalescing away its phi-arm install, but a phi arm makes its merge successor multi-predecessor, so such a
        # block can never satisfy ``overlaps`` and never spills -- before or after it leaves -- so no block becomes
        # overlap-eligible across rounds, every schedule is round-invariant, and the push bit and install set only drop.
        # SOUND: a converged classification (actual == has_install_blocks) is self-consistent, and ``landing <=
        # term_offset`` then holds by the drain math (not the asserts), so the layout is correct even under -O. The
        # asserts below pin the narrowing; a never-converging run -- unreachable -- falls to the ``else``, which RAISES.
        assert actual.keys() <= has_install_blocks.keys(), "install fixpoint must not grow"
        assert all(has_install_blocks[b] or not copy for b, copy in actual.items()), "install drain must not widen"
        assert actual_state <= has_state_copy, "state-copy fixpoint must not grow"
        if actual == has_install_blocks and actual_state == has_state_copy:
            break
        has_install_blocks, has_state_copy = actual, actual_state
    else:
        raise AssertionError("coalesced-install fixpoint did not converge")  # survives -O (unlike a bare assert)
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
        # ``resident`` records whether the source is available at block entry (a const, input, or state read) -- it
        # needs no read-first sampling. The placement (``install_issue_cycle``) charges the +1 only when a COMPUTED
        # source is the block's own last work; a resident or earlier-committing computed source pays none.
        copies = []
        for c in alloc.copies.get(block.id, []):
            fsrc = operand_signed(float_mir, c.source, c.sign, alloc, const_pool)
            resident = value_resident_at_entry(float_mir.nodes[c.source])
            iss = install_issue_cycle(work_makespan, resident, sched.commit_or_makespan(c.source))
            copies.append(FloatCopy(RegRef(c.dst), fsrc, iss, resident))
        bool_writes = []
        for w in alloc.bool_writes.get(block.id, []):
            bsrc = bool_operand(bool_mir, w.source, alloc, w.inversion)
            resident = value_resident_at_entry(bool_mir.nodes[w.source])
            iss = install_issue_cycle(work_makespan, resident, sched.commit_or_makespan(w.source))
            bool_writes.append(BoolWrite(BoolRegRef(w.dst), bsrc, iss, resident))
        # The block makespan carries the install +1 only when some install lands past the work makespan (a computed
        # source that is the block's own last work); ``has_install_blocks`` maps each install-bearing block to that bit.
        block_makespan = install_inclusive_makespan(work_makespan, has_install_blocks.get(block.id, False))
        # A branch condition gets exactly one cycle of slack: a bool result committing at the
        # makespan lands one step before the terminator's boundary read. The schedule's makespan covers every commit by
        # construction, so this is a tripwire against a future makespan-computation change only; the emitter-side
        # write-enable placement is guarded by the directed boundary cosim kernel and its white-box twin instead.
        bool_commits = [op.commit_cycle for op in inline_ops if isinstance(op.write.dst, BoolRegRef)] + [
            op.commit_cycle for op in ops if any(isinstance(w.dst, BoolRegRef) for w in op.writes)
        ]
        assert all(
            commit <= block_makespan for commit in bool_commits
        ), f"block {block.id}: a boolean result commits past the block makespan {block_makespan}"
        # Every phi-arm install must LAND within its block (at or before the terminator). An install landing past the
        # terminator is enqueued for a PC the block never reaches: a non-Ret terminator re-keys it onto the taken arm,
        # but a Ret wrap drops it -- a silently dead install. This is the vector-independent structural invariant that a
        # value cosim cannot see (a dead install that does not change outputs passes every value comparison).
        term_offset = overlap.block_term_offset[block.id]
        install_landings = [c.landing for c in copies] + [w.landing for w in bool_writes]
        assert all(
            landing <= term_offset for landing in install_landings
        ), f"block {block.id}: a phi-arm install lands at {max(install_landings)} past the terminator {term_offset}"
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
    # the Ret block's base, since the install fires inside the (last-laid-out) Ret block (``ret_block`` above).
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
    # persistent state; the per-block ``term_offset <= drained boundary`` invariant in ``schedule_with_overlap`` is the
    # matching guard for the opposite slip (a boundary install degrading into an early one). Backstop, not a live
    # failure.
    for slot in lir.float_state_slots:
        if slot.needs_copy:
            assert (
                lir.state_copy_step(slot) <= last_pc
            ), f"state slot {slot.name!r} writeback at {lir.state_copy_step(slot)} lands past the boundary {last_pc}"
    return lir
