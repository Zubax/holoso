"""Block scheduling with cross-block overlap, terminator-offset derivation, and ROM block layout for the builder."""

from collections.abc import Mapping
from dataclasses import dataclass

from .._mir import Mir, MirBlock, MirBoolView, MirBranch, MirFloatView, MirJump, MirRet
from .._operators import HardwareOperator, PooledHardwareOperator
from .._util import ValueId
from ._ir import *
from ._ir import _bank, _terminator_arms
from ._schedule import Schedule, schedule_ops
from ._build_base import _OverlapLayout
from ._construct import _mir_operation


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


def _value_word_and_landing(mir: Mir, float_mir: MirFloatView, vid: ValueId, issue: int) -> tuple[int, int]:
    """
    For a scheduled value, the (last in-block control WORD, result LANDING) in its block-local frame. The word is the
    latest fetch step the op still drives -- a pooled lane's write-enable (the wide bank one step after commit, the
    boolean bank on it) or an inline op's combinational fire step; the result lands later, after the bank's pipeline.
    Cross-block overlap may place ``term_offset`` between the two: the word stays in the block, the landing spills into
    the (single-predecessor) successor frame.
    """
    operator = _mir_operation(mir, vid).operator
    commit = issue + operator.latency
    wide = vid in float_mir.operation_nodes
    landing = landing_cycle(commit, _bank(wide))
    if isinstance(operator, PooledHardwareOperator):
        word = pooled_writeback_word(commit, wide)
    else:
        word = inline_fire_cycle(commit, wide)
    return word, landing


def _install_inclusive_makespan(work_makespan: int, has_install: bool) -> int:
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


def _issue_side_envelope(mir: Mir, float_mir: MirFloatView, sched: Schedule, block: MirBlock) -> int:
    """
    The issue-side floor an OVERLAPPING block's terminator may not precede: the latest control word still driven in the
    block (a pooled write-enable or an inline fire step), padded by the error-latch slack for any err-port op, and the
    produced branch condition's boolean landing. A live-in branch condition is NOT handled here -- its exact read floor
    is not locally known, so ``_block_boundary`` raises the floor to the wide cap for it. The floor starts at 1.
    """
    floor = 1
    for vid, issue in sched.issue_cycle.items():
        word, _landing = _value_word_and_landing(mir, float_mir, vid, issue)
        floor = max(floor, word)
        operator = _mir_operation(mir, vid).operator
        if isinstance(operator, PooledHardwareOperator) and operator.error_ports:
            # The err_pc diagnostic latches ``pc - FETCH_LAG`` when this op's write-enable executes, which is
            # FETCH_LAG fetch steps after its write word. If the terminator redirected by then, err_pc would
            # capture the successor frame's PC instead of this op's step. Keep the latch inside the block: the
            # data writeback still rides the pipeline correctly, but the diagnostic needs the live PC in-frame.
            floor = max(floor, word + FETCH_LAG)
    if isinstance(block.terminator, MirBranch) and block.terminator.cond in sched.issue_cycle:
        # The produced condition keeps its boolean landing inside the block; a live-in condition is handled in
        # ``_block_boundary`` (its read floor is not locally known, so it pins the floor to the wide cap).
        floor = max(floor, bool_landing_cycle(sched.commit_cycle(block.terminator.cond)))
    return floor


@dataclass(frozen=True, slots=True)
class _BlockBoundary:
    """
    A block's terminator placement: the (possibly overlap-shrunk) ``term_offset`` and the ``wide_cap`` -- the latched
    wide landing of the block's makespan -- that bounds it. ``_block_boundary`` is the SOLE owner of both
    ``boundary_step`` calls (the wide cap and the drained work boundary), the overlap floor clamp, and the
    ``term_offset <= wide_cap`` invariant, so the caller need not recompute the cap for its spill carry.
    """

    term_offset: int
    wide_cap: int


def _block_boundary(
    bid: int,
    makespan: int,
    overlaps: bool,
    envelope: int,
    livein_condition: bool,
    wide_resident: bool,
    does_boundary_work: bool,
    spill_landings: list[int],
) -> _BlockBoundary:
    """
    Derive a block's terminator offset, owning both physical regimes as explicit named branches and the invariants that
    bound them. ``wide_cap`` -- the latched wide landing of this block's makespan -- caps an overlapping block's floor and
    backstops every block's offset.

    The classification facts (``livein_condition``, ``wide_resident``, ``does_boundary_work``) are computed by the caller
    -- which holds the schedule and CFG context -- and passed in as plain booleans, keeping this a pure boundary-physics
    derivation rather than reaching back into the scheduler/MIR to re-derive them. With a single call site that separation
    is deliberate: folding the classification in would only widen this function's input surface, not remove a duplication.

    The two regimes are genuinely different physics and stay separate:
      - OVERLAP: an overlapping block's terminator is the ISSUE-side envelope (the latest control word still driven, the
        produced branch condition's boolean landing). For a live-in condition, whose exact read floor is not locally
        known, the floor is pinned to the conservative wide drain. The landings spill past it into the (single-
        predecessor) successors. The cap is the wide drain, which always accommodates the floor (every control word
        commits by the makespan, so word + the error-latch slack fits within ``wide_cap``); ``min`` is defensive. The
        bank-aware drain below governs only FULLY-DRAINED blocks -- a bool-only block branching on a live-in condition
        legitimately keeps the wide cap here, above its bool drain, so no ``term_offset <= bank-aware drain`` holds.
      - DRAINED: a fully-drained block's terminator offset is the latest cycle a value LANDS in its frame: its own
        bank-aware work drain ``boundary_step(makespan, wide_resident)`` plus any spill-in it cannot forward
        (``max(work_drain, *spill_landings)``). A block with NO boundary work pays no drain -- a non-entry drain-only Ret
        reading already-resident outputs lands its boundary at cycle 0, not at the phantom ``boundary_step(0, ...)`` of a
        value that never commits.
    """
    wide_cap = boundary_step(makespan, wide_resident=True)  # the latched wide landing of this block's makespan
    if overlaps:
        cap = wide_cap
        floor = cap if livein_condition else envelope
        assert floor <= cap, f"block {bid}: a control word sits past the wide drain"  # the cap bounds the floor
        term_offset = min(floor, cap)
    else:
        # ``does_boundary_work`` covers landings INVISIBLE to the op schedule (the entry's cycle-1 input loads and a
        # stateful-float Ret's wide state install) as well as the visible op commits; without any of those the block
        # pays no work drain. ``wide_resident`` selects the drained boundary's bank.
        work_drain = boundary_step(makespan, wide_resident=wide_resident) if does_boundary_work else 0
        term_offset = max([work_drain, *spill_landings])
    # The boundary never exceeds the block's own latched wide landing, so a boundary state install placed there stays a
    # read-first last_pc install (a spill from an overlapping predecessor lands within the successor's wide cap).
    assert term_offset <= wide_cap, f"block {bid}: term_offset {term_offset} exceeds the wide drain {wide_cap}"
    return _BlockBoundary(term_offset, wide_cap)


def _spill_local_cycle(bid: int, block_local_cycle: int, term_offset: int) -> int:
    """
    The successor-local cycle of a value spilling past block ``bid``'s shrunk terminator. The callers gate on
    ``block_local_cycle > term_offset``, so a real spill is non-negative (cycle 0 at the successor base is legal).
    """
    local = successor_local_cycle(block_local_cycle, term_offset)
    assert local >= 0, f"block {bid}: spilled landing PC {local} precedes the successor base"
    return local


def _schedule_with_overlap(
    mir: Mir,
    float_mir: MirFloatView,
    bool_mir: MirBoolView,
    pool: Mapping[type[HardwareOperator], int],
    has_install_blocks: set[int],
) -> _OverlapLayout:
    """
    Schedule every block in reverse-postorder and derive each block's terminator offset, threading cross-block overlap
    forward. A block whose every successor is single-predecessor (so a spill cannot reach a wrong path) and that carries
    no phi/const install shrinks its terminator offset from the drained boundary down to the issue-side envelope -- the
    latest cycle it still drives a control word, plus the branch condition's read floor. The drained boundary is bank-
    aware (``boundary_step(makespan, wide_resident)``): a block carrying any wide value across its boundary pays the
    latched wide landing, an all-boolean boundary drains one step earlier. Its in-flight results then land past the
    terminator, in the uniquely-reached successor frame; the successor inherits that as ``entry_busy`` (the
    predecessor's per-instance busy residue) and ``livein_landing`` (the cycle each spilled value lands), so its
    schedule neither reads a still-in-flight operand nor double-drives a busy instance. Back-edge targets and merge
    blocks are multi-predecessor, so no overlap crosses them: the forward-DAG carry converges in this single pass with
    no fixpoint. Under draining (every block multi-pred-bound or install-bearing) every offset equals its bank-aware
    ``boundary_step(makespan, wide_resident)`` and the carries are empty -- identical to an isolated per-block schedule.
    """
    succ = _succ_map(mir)
    pred_count: dict[int, int] = {block.id: 0 for block in mir.blocks}
    for targets in succ.values():
        for target in targets:
            pred_count[target] += 1
    blocks_by_id = {block.id: block for block in mir.blocks}
    # A wide value that LANDS in a block's frame forces the latched wide drain; a block whose boundary holds only
    # boolean landings (or wide values RESIDENT from a predecessor, already landed) drains one step earlier. A block
    # holds a wide landing iff it defines a float value, installs a float phi arm at its tail (the copy lands wide), a
    # float value spills into it, or -- the conservative case below -- it is the Ret of a stateful float kernel. A float
    # OUTPUT is NOT itself a wide landing: it is a combinational tap, wide-draining only when its value is produced
    # in-frame (caught by ``block_operations``) or spills in, not when it is resident from a predecessor (the
    # octave_index drain-only Ret), so the output does not appear here.
    float_arm_preds = {pred for phi in float_mir.phi_nodes.values() for pred, _arm, _cond in phi.arms}
    # ``ret_boundary_is_wide`` is a DELIBERATE conservative over-approximation, not a precise "lands in-frame" test: a
    # NON-coalesced float state slot installs its live-out wide at the Ret boundary (read-first), but whether a slot
    # coalesces is decided during allocation -- AFTER this layout pass -- so we cannot yet tell a coalesced (resident,
    # no install) slot from a non-coalesced one. Charging the wide drain for every stateful float Ret is always
    # correctness-safe (a later boundary only reads a value later, never wrong) and costs at most one cycle, only when a
    # coalesced slot's live-out is resident at a drain-only Ret -- a case no current kernel hits.
    ret_boundary_is_wide = bool(float_mir.state_slots)
    # The persistent-state live-out values, plus the arm producers reachable from them through phi chains (a nested
    # conditional/loop update layers phi over phi). A producer of one is dwell-guarded off the ENTRY block's ``ucode[0]``:
    # once a slot live-out -- or any of its (transitive) phi arms -- coalesces onto the slot register, its producer writes
    # the persistent register directly, so re-firing it during the accept dwell would corrupt the carried state; the guard
    # keeps such a producer off cycle 0. For a slot whose live-out does not coalesce the producer writes a temporary and
    # the guard is merely harmless defense-in-depth (see ``_assert_entry_dwell_safe``); it is cost-free unless a guarded
    # value would actually have issued at cycle 0. The transitive walk visits each value once, so loop-carried phi cycles
    # terminate.
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
    for bid in _mir_rpo(mir):
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
        makespan = _install_inclusive_makespan(sched.makespan, has_install)
        block_makespan[bid] = makespan
        targets = succ[bid]
        overlaps = bool(targets) and not has_install and all(pred_count[target] == 1 for target in targets)
        # ``ret_state_boundary`` is a DELIBERATE conservative over-approximation: a non-coalesced float state slot
        # installs its live-out wide at the Ret boundary, but whether a slot coalesces is decided AFTER this layout
        # pass, so charging the wide drain for every stateful float Ret is always correctness-safe (a later boundary
        # only reads a value later, never wrong) and costs at most one cycle. ``does_boundary_work`` and ``wide_resident``
        # are pure facts of this block's schedule and CFG shape, computed unconditionally and consumed only on the
        # fully-drained branch inside ``_block_boundary``; ``bid == mir.entry`` and ``ret_state_boundary`` are work
        # terms for landings INVISIBLE to the op schedule (the entry's cycle-1 input loads and a stateful-float Ret's
        # wide state install) -- irreducible block-granularity proxies, since register allocation runs after this pass.
        ret_state_boundary = isinstance(block.terminator, MirRet) and ret_boundary_is_wide
        does_boundary_work = bool(sched.issue_cycle) or has_install or bid == mir.entry or ret_state_boundary
        wide_resident = (
            has_install
            or bool(float_mir.block_operations(block))
            or bid in float_arm_preds
            or any(vid in float_mir.operation_nodes for vid in livein_landing)  # a wide value spills in
            or ret_state_boundary
        )
        livein_condition = isinstance(block.terminator, MirBranch) and block.terminator.cond not in sched.issue_cycle
        boundary = _block_boundary(
            bid,
            makespan,
            overlaps=overlaps,
            envelope=_issue_side_envelope(mir, float_mir, sched, block),
            livein_condition=livein_condition,
            wide_resident=wide_resident,
            does_boundary_work=does_boundary_work,
            spill_landings=list(livein_landing.values()),
        )
        term_offset = boundary.term_offset
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
    return _OverlapLayout(block_sched, block_makespan, block_term_offset, block_inflight)


@dataclass(frozen=True, slots=True)
class _BlockLayout:
    """The ROM placement: per-block base PC, the out_valid PC, and the shortest-path initiation interval."""

    block_base: list[int]
    last_pc: int
    min_initiation_interval: int


def _layout_blocks(mir: Mir, blocks: list[LirBlock]) -> _BlockLayout:
    """
    Lay blocks out in the ROM in reverse-postorder, returning their per-block base PCs, the out_valid PC, and the
    shortest-path initiation interval. Each block spans ``term_offset + 1`` fetch steps (its body up to and including the
    terminator step; the successor frame begins at ``term_pc + 1``); the single Ret block's boundary is the out_valid PC.
    ``min_initiation_interval`` is the shortest root-to-Ret path's traversed length.
    """
    successors: dict[int, list[int]] = {b.index: _terminator_arms(b.terminator) for b in blocks}
    # Blocks are laid out linearly in reverse-postorder, but the single Ret block is forced last so its boundary is the
    # highest address (out_valid = pc == LASTPC). A loop body is a DFS leaf (its only edge back to the header is a back
    # edge), so RPO would otherwise place it after the exit; moving Ret last keeps every loop body below the Ret. A
    # back-edge targets an earlier, lower-addressed block, which the next-PC sequencer redirects like any other jump, so
    # the linear layout needs no special case; the frontend emits reducible loops, so a back-edge target dominates it.
    ret_index = next(b.index for b in blocks if isinstance(b.terminator, Ret))
    order = [bid for bid in _mir_rpo(mir) if bid != ret_index] + [ret_index]
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
