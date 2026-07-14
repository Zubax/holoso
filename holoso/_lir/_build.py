"""
Build a finished :class:`Lir` from MIR: the top-level orchestration that schedules and lays out the blocks, allocates
both register banks (coalescing phi arms), constructs the per-block LIR, and assembles the final program.
"""

import logging
from dataclasses import replace

from .._errors import UnsupportedConstruct
from .._mir import (
    Mir,
    MirBoolStateSlot,
    MirBoolView,
    MirBranch,
    MirFloatStateSlot,
    MirFloatView,
    MirPhi,
    MirStateRead,
    MirStateSlot,
)
from .._operators import PortConditioner
from .._util import ValueId
from ._ir import *
from ._mir_facts import block_has_install, pred_count
from ._portassign import assign_commutative_ports
from ._schedule import resolve_pool
from ._bankalloc import actual_install_blocks, install_source_commit, layout_and_allocate
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

_logger = logging.getLogger(__name__)


def build(mir: Mir, module_name: str, fetch_stages: int) -> Lir:
    """
    Schedule, bind, and register-allocate selected MIR into a pipelined microprogram. A straight-line kernel is the
    degenerate single-``Ret``-block control-flow graph, so there is one build path for every kernel. ``fetch_stages``
    is the control-fetch pipeline depth; the datapath lags the fetch by one less than it, the lag threaded throughout.
    """
    if not mir.outputs:
        raise UnsupportedConstruct("Synthesized kernel must produce at least one output value")
    lir = _build_program(mir, module_name, fetch_stages - 1)
    names = [port.name for port in lir.ports]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise UnsupportedConstruct(f"duplicate port name(s) in the module interface: {', '.join(duplicates)}")
    return lir


def _drop_redundant_state_slots(mir: Mir) -> Mir:
    """
    Drop a state slot that is a redundant alias of another: same reset, live-out value id, and conditioner, so by
    induction always equal. The kept representative commits the value; each dropped attribute is write-only as state and
    its ``state_<attr>`` port already taps the shared value, so the duplicate register and its install copy vanish (e.g.
    a phase/frequency detector's public ``up`` aliasing its internal pending latch). A class is left intact when its
    live-out is a phi (a drop would perturb the phi-install placement) or when two members are read at entry (their
    distinct live-ins would need substitution).
    """
    read_names = {node.name for node in mir.nodes.values() if isinstance(node, MirStateRead)}

    def conditioner(slot: MirStateSlot) -> PortConditioner:
        if isinstance(slot, MirFloatStateSlot):
            return slot.sign
        assert isinstance(slot, MirBoolStateSlot)
        return slot.inversion

    classes: dict[tuple[float | bool, ValueId, PortConditioner], list[MirStateSlot]] = {}
    for slot in mir.state_slots:
        classes.setdefault((slot.reset_value, slot.live_out, conditioner(slot)), []).append(slot)

    dropped: set[str] = set()
    for members in classes.values():
        if len(members) < 2:
            continue
        if isinstance(mir.nodes[members[0].live_out], MirPhi):
            continue  # a phi live-out: dropping perturbs its install placement (if-conversion usually elides the phi)
        read_members = [m for m in members if m.name in read_names]
        if len(read_members) >= 2:
            continue  # two read aliases: distinct live-ins, leave intact
        rep = read_members[0] if read_members else members[0]
        assert all(m.name not in read_names for m in members if m is not rep), "a dropped alias must be write-only"
        dropped.update(m.name for m in members if m is not rep)

    if not dropped:
        return mir
    return replace(mir, state_slots=[slot for slot in mir.state_slots if slot.name not in dropped])


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


def _build_program(mir: Mir, module_name: str, fetch_lag: int) -> Lir:
    """
    Build the microprogram for any kernel (a straight-line kernel is the degenerate single-``Ret``-block graph):
    schedule each block independently, pool operator instances across the mutually-exclusive blocks, color both register
    banks by hardware-frame liveness (reusing registers, coalescing state live-outs), install non-coalesced phi and slot
    live-outs by pc-gated copy, and lay the blocks out in the ROM with the single ``Ret`` as the out_valid boundary.
    """
    mir = _drop_redundant_state_slots(mir)
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
    # The phi-residency classification (``value_resident_at_entry``) rests on every phi-bearing block being
    # multi-predecessor, which is what keeps overlap spills out of phi registers; a future pass emitting a phi into a
    # single-predecessor block would silently void that argument (the sibling-install tripwire does not read-gate
    # installs against in-flight landings), so the reliance is machine-checked here.
    pred_edges = pred_count(mir)
    for mir_block in mir.blocks:
        assert (
            not mir_block.phis or pred_edges[mir_block.id] >= 2
        ), f"block {mir_block.id} carries phis with {pred_edges[mir_block.id]} predecessor edge(s)"
    pool = resolve_pool(mir.nodes)
    # Schedule every block in reverse-postorder (a block after its forward-edge predecessors) and lay out each block's
    # terminator offset, with cross-block software pipelining: a block whose successors are all single-predecessor
    # shrinks its terminator below the drained boundary and spills its in-flight results into the successor, which
    # inherits the busy/landing residue. The +1-install drain keeps install-bearing blocks unshrunk, matching makespan.
    #
    # The install set is computed to a fixpoint. ``block_has_install`` marks a block install-bearing from the CFG shape
    # (any phi arm originates in it), but a block whose every arm COALESCES onto the merged register installs nothing,
    # so that +1 drain (and overlap-ineligibility) is spurious. So: lay out and allocate with the conservative CFG
    # seed, recompute the classification from the ACTUAL coalesced copies, and re-run to a fixed point. The movement is
    # TWO-SIDED: a classification mostly narrows (dropping a spurious drain), but the shortened boundary feeds a greedy
    # coalescing that is not monotone in the interference, so a narrowed classification can have to grow back -- and
    # every regrowth is pinned (see the loop body), so each block moves a bounded number of times and the composition
    # with the inner per-bank ``coalesce_and_color`` fixpoint (its forbidden-merge set non-decreasing) terminates.
    # Determinism is preserved: the allocator is seed-fixed and the classification is rebuilt the same way each pass.
    #
    # The same fixpoint also drives the state slot's read-first boundary-copy drain charge: ``has_state_copy`` starts
    # conservative (a state slot needs a copy), usually clears as coalescing removes the copy, and latches back on the
    # regrowth channel noted in the loop. It is a single bool: the MIR has one Ret, so the charge is op-wide there.
    has_install_blocks = block_has_install(mir, float_mir, bool_mir)
    ret_block = mir.ret_block
    # Conservative seed for the state-copy fixpoint -- the pre-allocation form of ``_has_state_copy``: assume every
    # state slot needs a boundary copy.
    has_state_copy = bool(float_mir.state_slots or bool_mir.state_slots)
    # Iteration bound. Every non-final round makes one of the bounded monotone moves per block -- a push-bit narrowing,
    # an install-set key removal, or a pin (a pinned block never moves again), at most three over the run -- or moves
    # the state-copy charge along its drop-then-latch chain (at most two moves). Worst case is their SUM plus a
    # confirming round; the 3*len(blocks)+4 loop bound below leaves a safe margin, with the else-clause as the loud
    # backstop.
    seed_keys = frozenset(has_install_blocks)
    pinned_push: set[int] = set()
    state_copy_latched = False
    for round_index in range(3 * len(mir.blocks) + 4):
        result = layout_and_allocate(mir, float_mir, bool_mir, pool, has_install_blocks, has_state_copy, fetch_lag)
        raw = actual_install_blocks(result.alloc, float_mir, bool_mir, result.overlap.block_sched)
        # The two derivations of install-bearing -- the CFG-shape seed and the post-allocation copies -- must agree on
        # the key universe: a block outside the seed can never install, so a wider ``raw`` means the derivations
        # drifted, which must fail loudly rather than be absorbed as a silent permanent pin. On the FIRST round the
        # bits must agree too: nothing has narrowed yet, so a push the conservative seed missed is the same drift,
        # not legitimate regrowth.
        assert raw.keys() <= seed_keys, "post-allocation installs appeared outside the CFG-shape seed"
        assert round_index > 0 or all(
            has_install_blocks[b] or not bit for b, bit in raw.items()
        ), "a first-round push classification exceeded the conservative CFG-shape seed"
        # A narrowed classification may have to GROW BACK: the shortened boundary feeds the next round's coalescing,
        # whose greedy merge order is not monotone in the interference, so a computed arm that coalesced under the
        # longer boundary can come back residual -- its install then needs the +1 drain the narrowing removed, and a
        # whole dropped KEY can likewise resurface. Any regrowth is PINNED: a pinned block stays install-bearing with a
        # forced +1 drain for the rest of the run, so the two-sided movement still converges (key removals and pins are
        # each monotone, bounded by the block count). An intermediate allocation built on a stale narrower boundary is
        # discarded by the re-run; the converged round has validated every surviving install against a boundary
        # consistent with its own classification.
        regrown = {b for b, bit in raw.items() if b not in has_install_blocks or (bit and not has_install_blocks[b])}
        if regrown - pinned_push:
            _logger.info("Install fixpoint round %d: pinning regrown block(s) %s", round_index, sorted(regrown))
        pinned_push |= regrown
        actual = raw | dict.fromkeys(pinned_push, True)
        # The state-copy charge has its own regrowth channel (a final pin conflict can force a slot back out of
        # coalescing onto a copy), so it is a one-way latch rather than a pure descent.
        raw_state = _has_state_copy(float_mir, bool_mir, result.alloc, result.const_pool)
        if raw_state and not has_state_copy:
            if not state_copy_latched:
                _logger.info("Install fixpoint round %d: latching the state-copy charge", round_index)
            state_copy_latched = True
        actual_state = raw_state or state_copy_latched
        # The moves are monotone and the fixed point is sound. MONOTONE: a block leaves the install set only by
        # coalescing away its phi-arm install, but a phi arm makes its merge successor multi-predecessor, so such a
        # block can never satisfy ``overlaps`` and never spills -- before or after it leaves -- so no block becomes
        # overlap-eligible across rounds and every schedule is round-invariant; keys only shrink except through the
        # growing pin set, and the state latch has height one. SOUND: a converged classification
        # (actual == has_install_blocks) is self-consistent, and ``landing <= term_offset`` then holds by the drain
        # math, so the layout is correct even under -O; every widening is legitimized by the unconditional pin/latch
        # merges above (an assert here would be tautological), and a never-converging run -- unreachable -- falls to
        # the ``else``, which RAISES.
        if actual == has_install_blocks and actual_state == has_state_copy:
            _logger.info(
                "Install fixpoint converged after %d round(s): %d install-bearing block(s), %d pinned, "
                "state copy %s",
                round_index + 1,
                len(actual),
                len(pinned_push),
                "charged" if actual_state else "elided",
            )
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
        # A None commit from ``install_source_commit`` is a block-entry-resident source needing no read-first
        # sampling. The placement (``install_issue_cycle``) charges the +1 only for a computed source committing at
        # the makespan; a resident or earlier-committing computed source pays none.
        copies = []
        for c in alloc.copies.get(block.id, []):
            fsrc = operand_signed(float_mir, c.source, c.sign, alloc, const_pool)
            commit = install_source_commit(sched, float_mir.nodes[c.source], c.source)
            copies.append(FloatCopy(RegRef(c.dst), fsrc, install_issue_cycle(work_makespan, commit), commit is None))
        bool_writes = []
        for w in alloc.bool_writes.get(block.id, []):
            bsrc = bool_operand(bool_mir, w.source, alloc, w.inversion)
            commit = install_source_commit(sched, bool_mir.nodes[w.source], w.source)
            bool_writes.append(
                BoolWrite(BoolRegRef(w.dst), bsrc, install_issue_cycle(work_makespan, commit), commit is None)
            )
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
        installs: list[FloatCopy | BoolWrite] = [*copies, *bool_writes]
        install_landings = [x.landing(fetch_lag) for x in installs]
        assert all(
            landing <= term_offset for landing in install_landings
        ), f"block {block.id}: a phi-arm install lands at {max(install_landings)} past the terminator {term_offset}"
        # A tail install must read its source register strictly before a sibling install's write to that register
        # lands (see ``value_resident_at_entry`` for why placement guarantees this): the structural tripwire for a
        # placement regression, which the value cosim shares with the model and cannot see. Cross-bank pairs are inert
        # (RegRef never equals BoolRegRef), so one check serves both banks.
        for writer in installs:
            for reader in installs:
                if reader.source.source == writer.dst:
                    assert reader.fire_step(fetch_lag) < writer.landing(
                        fetch_lag
                    ), f"block {block.id}: a tail install reads {writer.dst} after a sibling install's write lands"
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
            nreg=alloc.nreg,
            nrd=sum(inst.operator.arity for inst in instances),
            nwr=len(tapped_wide_lanes(blocks)),
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
        fetch_lag=fetch_lag,
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
