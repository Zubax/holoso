"""Block scheduling with cross-block overlap, terminator-offset derivation, and ROM block layout for the builder."""

from collections.abc import Mapping
from dataclasses import dataclass

from .._mir import Mir, MirBlock, MirBoolView, MirBranch, MirFloatView
from .._operators import HardwareOperator, PooledHardwareOperator
from .._util import ValueId
from ._ir import *
from ._schedule import Schedule, schedule_ops
from ._build_base import OverlapLayout
from ._mir_facts import mir_operation, mir_rpo, succ_map


def _value_word_and_landing(mir: Mir, float_mir: MirFloatView, vid: ValueId, issue: int) -> tuple[int, int]:
    """
    For a scheduled value, the (last in-block control WORD, result LANDING) in its block-local frame. The word is the
    latest fetch step the op still drives -- a pooled lane's write-enable (the wide bank one step after commit, the
    boolean bank on it) or an inline op's combinational fire step; the result lands later, after the bank's pipeline.
    Cross-block overlap may place ``term_offset`` between the two: the word stays in the block, the landing spills into
    the (single-predecessor) successor frame.
    """
    operator = mir_operation(mir, vid).operator
    commit = issue + operator.latency
    wide = vid in float_mir.operation_nodes
    if isinstance(operator, PooledHardwareOperator):
        word = pooled_writeback_word(commit, wide)
        landing = landing_cycle(commit, bank_timing(wide))
    else:  # an inline op drives the array write combinationally -- no writeback latch on either bank
        word = inline_fire_cycle(commit)
        landing = inline_landing_cycle(commit)
    return word, landing


def install_inclusive_makespan(work_makespan: int, has_install: bool) -> int:
    """
    The block makespan inclusive of its tail install: an install-bearing block fires its pc-gated phi/slot copies one
    cycle past its last work commit, so its effective makespan is one higher. The single owner of this ``+1`` so the
    overlap layout's boundary derivation and the per-block LirBlock makespan cannot disagree on it.
    """
    return work_makespan + (1 if has_install else 0)


@dataclass(frozen=True, slots=True)
class _SpillCarry:
    """
    The cross-block-overlap residue a block hands each single-predecessor successor: per-instance busy windows still
    in flight at the shrunk terminator (``entry_busy``) and the values whose write spills past it (``livein_landing``,
    the value's landing cycle in the successor-local frame). Both are successor-local cycles in the
    ``absolute_pc = block_base + cycle`` frame the scheduler uses (via ``successor_local_cycle``), so a spill can land
    as early as cycle 0 -- the successor's base PC, available before its first compute cycle.
    """

    entry_busy: dict[tuple[PooledHardwareOperator, int], int]
    livein_landing: dict[ValueId, int]


def _issue_side_envelope(
    mir: Mir, float_mir: MirFloatView, sched: Schedule, block: MirBlock, livein_landing: Mapping[ValueId, int]
) -> int:
    """
    The issue-side floor an OVERLAPPING block's terminator may not precede: the latest control word still driven in the
    block (a pooled write-enable or an inline fire step), padded by the error-latch slack for any err-port op, and the
    branch condition's read floor. The branch condition is the SINGLE owner of that read floor here, derived from where
    the condition becomes readable: a PRODUCED condition lands inside the block at its boolean landing; a SPILLED-IN
    live-in condition (carried past an overlapped predecessor's shrunk terminator) lands at its carried landing cycle
    (``livein_landing``); a RESIDENT live-in condition (an input, persistent state, or a fully-drained prior-block
    result) is available from the block's first cycle and adds nothing. The floor starts at 1.
    """
    floor = 1
    for vid, issue in sched.issue_cycle.items():
        word, _landing = _value_word_and_landing(mir, float_mir, vid, issue)
        floor = max(floor, word)
        operator = mir_operation(mir, vid).operator
        if isinstance(operator, PooledHardwareOperator) and operator.error_ports:
            # The err_pc diagnostic latches ``pc - FETCH_LAG`` when this op's write-enable executes, which is
            # FETCH_LAG fetch steps after its write word. If the terminator redirected by then, err_pc would
            # capture the successor frame's PC instead of this op's step. Keep the latch inside the block: the
            # data writeback still rides the pipeline correctly, but the diagnostic needs the live PC in-frame.
            floor = max(floor, word + FETCH_LAG)
    if isinstance(block.terminator, MirBranch):
        cond = block.terminator.cond
        if cond in sched.issue_cycle:  # produced in-block: readable at its boolean landing
            floor = max(floor, bool_landing_cycle(sched.commit_cycle(cond)))
        elif cond in livein_landing:  # spilled in past an overlapped predecessor: readable at its carried landing
            floor = max(floor, livein_landing[cond])
        # else: a resident live-in condition is available from the block's first cycle and imposes no floor
    return floor


def _block_boundary(
    bid: int,
    makespan: int,
    overlaps: bool,
    envelope: int,
    work_drain: int,
    spill_landings: list[int],
) -> int:
    """
    Derive a block's terminator offset from the classification facts the caller computes (it holds the schedule/CFG
    context): the ISSUE-side ``envelope`` when the block overlaps its single-predecessor successors, else the drained
    boundary ``max(work_drain, *spill_landings)``. The two regimes' physics live in DESIGN.md (cross-block pipelining
    and the drained boundary); the invariant local to here is that the offset never exceeds the block's latched wide
    landing ``boundary_step(makespan, wide_resident=True)``, asserted below.
    """
    if overlaps:
        term_offset = envelope
    else:
        term_offset = max([work_drain, *spill_landings])
    # Within the latched wide landing, a boundary state install placed at the offset stays a read-first last_pc install.
    wide_cap = boundary_step(makespan, wide_resident=True)
    assert term_offset <= wide_cap, f"block {bid}: term_offset {term_offset} exceeds the wide drain {wide_cap}"
    return term_offset


def _spill_local_cycle(bid: int, block_local_cycle: int, term_offset: int) -> int:
    """
    The successor-local cycle of a value spilling past block ``bid``'s shrunk terminator. The callers gate on
    ``block_local_cycle > term_offset``, so a real spill is non-negative (cycle 0 at the successor base is legal).
    """
    local = successor_local_cycle(block_local_cycle, term_offset)
    assert local >= 0, f"block {bid}: spilled landing PC {local} precedes the successor base"
    return local


def schedule_with_overlap(
    mir: Mir,
    float_mir: MirFloatView,
    bool_mir: MirBoolView,
    pool: Mapping[type[HardwareOperator], int],
    has_install_blocks: set[int],
    state_copy_blocks: Mapping[int, bool],
) -> OverlapLayout:
    """
    Schedule every block in reverse-postorder and derive each block's terminator offset, threading cross-block overlap
    forward. A block whose every successor is single-predecessor (so a spill cannot reach a wrong path) and that carries
    no phi/const install shrinks its terminator offset from the drained boundary down to the issue-side envelope -- the
    latest cycle it still drives a control word, plus the branch condition's read floor. The drained boundary is the
    latest cycle a value LANDS in the block's frame, taken per op so it is both bank-aware and inline-aware (a pooled
    result lands through its bank's writeback latch, an inline result a cycle earlier). Its in-flight results then land
    past the terminator, in the uniquely-reached successor frame; the successor inherits that as ``entry_busy`` (the
    predecessor's per-instance busy residue) and ``livein_landing`` (the cycle each spilled value lands), so its
    schedule neither reads a still-in-flight operand nor double-drives a busy instance. Back-edge targets and merge
    blocks are multi-predecessor, so no overlap crosses them: the forward-DAG carry converges in this single pass with
    no fixpoint. Under draining (every block multi-pred-bound or install-bearing) every offset equals its own max op
    landing (plus the tail install) and the carries are empty -- identical to an isolated per-block schedule.
    """
    succ = succ_map(mir)
    pred_count: dict[int, int] = {block.id: 0 for block in mir.blocks}
    for targets in succ.values():
        for target in targets:
            pred_count[target] += 1
    blocks_by_id = {block.id: block for block in mir.blocks}
    # The persistent-state live-out values, plus the arm producers reachable from them through phi chains (a nested
    # conditional/loop update layers phi over phi). A producer of one is dwell-guarded off the ENTRY block's
    # ``ucode[0]``: once a slot live-out -- or any of its (transitive) phi arms -- coalesces onto the slot register, its
    # producer writes the persistent register directly, so re-firing it during the accept dwell would corrupt the
    # carried state; the guard keeps such a producer off cycle 0. For a slot whose live-out does not coalesce the
    # producer writes a temporary and the guard is merely harmless defense-in-depth (see ``_assert_entry_dwell_safe``);
    # it is cost-free unless a guarded value would actually have issued at cycle 0. The transitive walk visits each
    # value once, so loop-carried phi cycles terminate.
    all_phis = {**float_mir.phi_nodes, **bool_mir.phi_nodes}
    state_liveouts_set: set[ValueId] = set()
    worklist = [slot.live_out for slot in (*float_mir.state_slots, *bool_mir.state_slots)]
    while worklist:
        vid = worklist.pop()
        if vid in state_liveouts_set:
            continue
        state_liveouts_set.add(vid)
        phi = all_phis.get(vid)
        if phi is not None:
            worklist.extend(arm for _pred, arm, _conditioner in phi.arms)
    state_liveouts = frozenset(state_liveouts_set)
    block_sched: dict[int, Schedule] = {}
    block_makespan: dict[int, int] = {}
    block_term_offset: dict[int, int] = {}
    block_inflight: dict[int, dict[ValueId, int]] = {}
    # successor block -> the spill carry its single overlapping predecessor hands it (set at most once: a carried-into
    # block is single-predecessor, so only that one predecessor overlaps into it).
    carry: dict[int, _SpillCarry] = {}
    for bid in mir_rpo(mir):
        block = blocks_by_id[bid]
        inherited = carry.get(bid, _SpillCarry({}, {}))
        livein_landing = inherited.livein_landing
        block_inflight[bid] = livein_landing  # the spills this block receives (== its scheduler livein_landing)
        sched = schedule_ops(
            mir.nodes,
            pool,
            schedulable=set(float_mir.block_operations(block)) | set(bool_mir.block_operations(block)),
            entry_busy=inherited.entry_busy,
            livein_landing=livein_landing,
            dwell_guarded=state_liveouts if bid == mir.entry else frozenset(),
        )
        block_sched[bid] = sched
        has_install = bid in has_install_blocks
        makespan = install_inclusive_makespan(sched.makespan, has_install)
        block_makespan[bid] = makespan
        targets = succ[bid]
        overlaps = bool(targets) and not has_install and all(pred_count[target] == 1 for target in targets)
        # The drained boundary is the latest cycle a value LANDS in this block's frame, taken per op so it is both
        # bank-aware AND inline-aware: a pooled result lands through its bank's writeback latch, an inline result writes
        # the array combinationally and lands a cycle earlier. Three landings are INVISIBLE to the op schedule and are
        # added explicitly: (1) a phi/const tail install lands one fetch-pipeline past the work
        # makespan (``boundary_step(makespan, wide)``, ``makespan`` install-inclusive); (2) a NON-coalesced state slot's
        # read-first boundary copy lands at ``boundary_step(sched.makespan, bank)`` -- its source is among the op
        # landings, but the copy adds its bank's fetch-pipeline; ``state_copy_blocks`` carries the Ret blocks that have
        # one (mapped to the wide-vs-bool bank), decided by the coalescing fixpoint -- a coalesced slot writes its
        # register in place and needs no copy, so it is absent; (3) the entry's input loads land on cycle 1.
        work_drain = max(
            (_value_word_and_landing(mir, float_mir, vid, issue)[1] for vid, issue in sched.issue_cycle.items()),
            default=0,
        )
        if has_install:
            work_drain = max(work_drain, boundary_step(makespan, wide_resident=True))
        if bid in state_copy_blocks:
            work_drain = max(work_drain, boundary_step(sched.makespan, wide_resident=state_copy_blocks[bid]))
        if bid == mir.entry:
            work_drain = max(work_drain, 1)
        term_offset = _block_boundary(
            bid,
            makespan,
            overlaps=overlaps,
            envelope=_issue_side_envelope(mir, float_mir, sched, block, livein_landing),
            work_drain=work_drain,
            spill_landings=list(livein_landing.values()),
        )
        block_term_offset[bid] = term_offset
        if overlaps:  # hand the spill residue to the (single-predecessor) successors this block uniquely reaches
            # Both the per-instance busy residue and the value landings cross the shrunk terminator into the successor
            # frame, so both translate through the SAME coordinate map (``successor_local_cycle``) that _trace_landing /
            # Lir.write_landing_pcs and the model's redirect re-keying use -- the scheduler reserves and read-gates each
            # register/instance at the cycle the pipeline truly frees/writes it, on one coordinate contract.
            busy = {
                inst: successor_local_cycle(free, term_offset)
                for inst, free in sched.busy_until.items()
                if successor_local_cycle(free, term_offset) > 0
            }
            landing: dict[ValueId, int] = {}
            for vid, issue in sched.issue_cycle.items():
                _word, land = _value_word_and_landing(mir, float_mir, vid, issue)
                if land > term_offset:
                    landing[vid] = _spill_local_cycle(bid, land, term_offset)
            for vid, land in livein_landing.items():  # a received spill that re-spills past this shrunk terminator
                if land > term_offset:
                    landing[vid] = max(landing.get(vid, 0), _spill_local_cycle(bid, land, term_offset))
            spill = _SpillCarry(busy, landing)
            for target in targets:
                carry[target] = spill
    return OverlapLayout(block_sched, block_makespan, block_term_offset, block_inflight)


@dataclass(frozen=True, slots=True)
class _BlockLayout:
    """The ROM placement: per-block base PC, the out_valid PC, and the shortest-path initiation interval."""

    block_base: list[int]
    last_pc: int
    min_initiation_interval: int


def layout_blocks(mir: Mir, blocks: list[LirBlock]) -> _BlockLayout:
    """
    Lay blocks out in the ROM in reverse-postorder, returning their per-block base PCs, the out_valid PC, and the
    shortest-path initiation interval. Each block spans ``term_offset + 1`` fetch steps (its body up to and including
    the terminator step; the successor frame begins at ``term_pc + 1``); the single Ret block's boundary is the
    out_valid PC.
    ``min_initiation_interval`` is the shortest root-to-Ret path's traversed length.
    """
    successors: dict[int, list[int]] = {b.index: terminator_arms(b.terminator) for b in blocks}
    # Blocks are laid out linearly in reverse-postorder, but the single Ret block is forced last so its boundary is the
    # highest address (out_valid = pc == LASTPC). A loop body is a DFS leaf (its only edge back to the header is a back
    # edge), so RPO would otherwise place it after the exit; moving Ret last keeps every loop body below the Ret. A
    # back-edge targets an earlier, lower-addressed block, which the next-PC sequencer redirects like any other jump, so
    # the linear layout needs no special case; the frontend emits reducible loops, so a back-edge target dominates it.
    ret_index = next(b.index for b in blocks if isinstance(b.terminator, Ret))
    order = [bid for bid in mir_rpo(mir) if bid != ret_index] + [ret_index]
    position = {bid: i for i, bid in enumerate(order)}
    term_offset = {b.index: b.term_offset for b in blocks}
    length = {index: offset + 1 for index, offset in term_offset.items()}
    base: dict[int, int] = {}
    cursor = 0
    for index in order:  # reverse-postorder starts at the entry (block 0), so every block's base is assigned here
        base[index] = cursor
        cursor += length[index]
    last_pc = base[ret_index] + term_offset[ret_index]
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
    min_ii = dist.get(ret_index, 0) + term_offset[ret_index]
    block_base = [base[i] for i in range(len(blocks))]
    return _BlockLayout(block_base, last_pc, min_ii)
