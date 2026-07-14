"""Block scheduling with cross-block overlap, terminator-offset derivation, and ROM block layout for the builder."""

from collections.abc import Mapping
from dataclasses import dataclass

from .._mir import Mir, MirBlock, MirBoolView, MirBranch, MirFloatView
from .._operators import HardwareOperator, PooledHardwareOperator
from .._util import ValueId
from ._ir import *
from ._schedule import Schedule, schedule_ops
from ._build_base import OverlapLayout
from ._mir_facts import mir_operation, mir_rpo, pred_count, succ_map


def _value_word_and_landing(mir: Mir, vid: ValueId, issue: int, fetch_lag: int) -> tuple[int, int, HardwareOperator]:
    """
    For a scheduled value, the (last in-block control WORD, result LANDING) in its block-local frame. The word is the
    latest fetch step the op still drives -- a pooled lane's write opcode on its commit step or an inline op's
    combinational fire step; the result lands later, after the fetch pipeline. Cross-block overlap may place
    ``term_offset`` between the two: the word stays in the block, the landing spills into the (single-predecessor)
    successor frame.
    """
    operator = mir_operation(mir, vid).operator
    commit = issue + operator.latency
    # The control WORD placement still distinguishes the op class: a pooled lane drives its write-opcode word on the
    # commit step, an inline op fires its combinational statement one fetch_lag later. The result LANDING is uniform.
    word = (
        pooled_write_word(commit)
        if isinstance(operator, PooledHardwareOperator)
        else inline_fire_cycle(commit, fetch_lag)
    )
    landing = landing_cycle(commit, fetch_lag)
    return word, landing, operator


def install_inclusive_makespan(work_makespan: int, install_pushes_makespan: bool) -> int:
    """
    The block makespan inclusive of its tail install: a block whose install lands PAST the work makespan is one higher.
    That happens only for a computed source that is the block's own last-committing work, which the install must fire
    one step after to read-first it. A tail whose every install fits at the makespan -- a block-entry-resident source,
    or a computed source committing before the last work -- adds no step. The single owner of this ``+1`` so the overlap
    layout's boundary derivation and the per-block LirBlock makespan cannot disagree on it (the dual of the per-install
    ``install_issue_cycle``).
    """
    return work_makespan + (1 if install_pushes_makespan else 0)


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
    mir: Mir, sched: Schedule, block: MirBlock, livein_landing: Mapping[ValueId, int], fetch_lag: int
) -> int:
    """
    The issue-side floor an OVERLAPPING block's terminator may not precede: the latest control word still driven in the
    block (a pooled write-enable or an inline fire step), the operand-read cycle of any firing (a latch-free wide read
    samples one step past a latency-1 pooled op's control word), padded by the error-latch slack for any err-port op,
    and the branch condition's read floor. The branch condition is the SINGLE owner of that read floor here, derived
    from where the condition becomes readable: a PRODUCED condition lands inside the block at its landing; a SPILLED-IN
    live-in condition (carried past an overlapped predecessor's shrunk terminator) lands at its carried landing cycle
    (``livein_landing``); a RESIDENT live-in condition (an input, persistent state, or a fully-drained prior-block
    result) is available from the block's first cycle and adds nothing. The floor starts at 1 for the ENTRY block only:
    its terminator cannot redirect at PC 0, because the sequencer's accept hold (``pc==0``) precedes the branch
    redirect, so an entry branch must settle at PC>=1. Every other block may redirect at its own base PC, so its floor
    starts at 0 -- an empty resident-condition branch then drains nothing, exactly like a jump.
    """
    floor = 1 if block.id == mir.entry else 0
    for vid, issue in sched.issue_cycle.items():
        word, _landing, operator = _value_word_and_landing(mir, vid, issue, fetch_lag)
        # The block may not end before an op reads its operands: it fires (and samples) at ``operand_read_cycle``. A
        # latch-free wide read samples one step past a latency-1 pooled op's control word, so the read can exceed the
        # word -- without this floor the op would fire past the shrunk terminator and never execute.
        floor = max(floor, word, operand_read_cycle(operator, issue, fetch_lag))
        if isinstance(operator, PooledHardwareOperator) and operator.error_ports:
            # The err_pc diagnostic latches ``pc - fetch_lag`` when this op's write-enable executes, which is
            # fetch_lag fetch steps after its write word. If the terminator redirected by then, err_pc would
            # capture the successor frame's PC instead of this op's step. Keep the latch inside the block: the
            # data write still lands correctly, but the diagnostic needs the live PC in-frame.
            floor = max(floor, word + fetch_lag)
    if isinstance(block.terminator, MirBranch):
        cond = block.terminator.cond
        if cond in sched.issue_cycle:  # produced in-block: readable at its landing
            floor = max(floor, landing_cycle(sched.commit_cycle(cond), fetch_lag))
        elif cond in livein_landing:  # spilled in past an overlapped predecessor: readable at its carried landing
            floor = max(floor, livein_landing[cond])
        # else: a resident live-in condition is available from the block's first cycle and imposes no floor
    return floor


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
    has_install_blocks: Mapping[int, bool],
    has_state_copy: bool,
    fetch_lag: int,
) -> OverlapLayout:
    """
    Schedule every block in reverse-postorder and derive each block's terminator offset, threading cross-block overlap
    forward. A block whose every successor is single-predecessor (so a spill cannot reach a wrong path) and that carries
    no tail install shrinks its terminator offset from the drained boundary down to the issue-side envelope -- the
    latest cycle it still drives a control word, plus the branch condition's read floor. The drained boundary is the
    latest cycle a value LANDS in the block's frame, taken per op (a pooled result and an inline result land at the same
    bank-independent cycle, the combinational landing). Its in-flight results then land
    past the terminator, in the uniquely-reached successor frame; the successor inherits that as ``entry_busy`` (the
    predecessor's per-instance busy residue) and ``livein_landing`` (the cycle each spilled value lands), so its
    schedule neither reads a still-in-flight operand nor double-drives a busy instance. Back-edge targets and merge
    blocks are multi-predecessor, so no overlap crosses them: the forward-DAG carry converges in this single pass with
    no fixpoint. Under draining (every block multi-pred-bound or install-bearing) every offset equals its own max op
    landing (plus the tail install) and the carries are empty -- identical to an isolated per-block schedule.
    """
    succ = succ_map(mir)
    preds = pred_count(mir)
    blocks_by_id = {block.id: block for block in mir.blocks}
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
            fetch_lag=fetch_lag,
            entry_busy=inherited.entry_busy,
            livein_landing=livein_landing,
        )
        block_sched[bid] = sched
        has_install = bid in has_install_blocks
        install_pushes_makespan = has_install_blocks.get(bid, False)
        makespan = install_inclusive_makespan(sched.makespan, install_pushes_makespan)
        block_makespan[bid] = makespan
        targets = succ[bid]
        overlaps = bool(targets) and not has_install and all(preds[target] == 1 for target in targets)
        # The drained boundary is the latest cycle a value LANDS in this block's frame, taken per op -- a pooled result
        # and an inline result both write the array combinationally and land at the same bank-independent cycle. Three
        # landings are INVISIBLE to the op schedule and are added explicitly: (1) a phi tail install -- one whose source
        # is the block's own LAST work lands a step past it at the drain boundary ``boundary_step(makespan)`` (the
        # makespan install-inclusive), while an install fitting at the makespan (a resident source, or a computed
        # source committing before the last work) lands at ``landing_cycle(sched.makespan)`` within the work
        # boundary, paying neither the +1 step nor the later drain; (2) a NON-coalesced state slot's read-first boundary
        # copy lands at ``boundary_step(sched.makespan)`` -- its source is among the op landings, but the copy adds the
        # fetch-pipeline; ``has_state_copy`` flags whether the lone Ret block has one, decided by the coalescing
        # fixpoint -- a coalesced slot writes its register in place and needs no copy, so the charge usually clears
        # (the fixpoint may latch it back on); (3) the entry's input loads land on cycle 1.
        work_drain = max(
            (_value_word_and_landing(mir, vid, issue, fetch_lag)[1] for vid, issue in sched.issue_cycle.items()),
            default=0,
        )
        if install_pushes_makespan:
            work_drain = max(work_drain, boundary_step(makespan, fetch_lag))
        elif has_install:
            work_drain = max(work_drain, landing_cycle(sched.makespan, fetch_lag))
        if bid == mir.ret_block and has_state_copy:
            work_drain = max(work_drain, boundary_step(sched.makespan, fetch_lag))
        if bid == mir.entry:
            work_drain = max(work_drain, 1)
        # The terminator offset: the issue-side envelope when this block overlaps its single-predecessor successors,
        # else the drained boundary max(work_drain, *spill landings). It never exceeds the block's drained landing
        # (a boundary state install placed at the offset stays a read-first last_pc install); the two regimes' physics
        # are in DESIGN.md.
        if overlaps:
            term_offset = _issue_side_envelope(mir, sched, block, livein_landing, fetch_lag)
        else:
            term_offset = max([work_drain, *livein_landing.values()])
        drain_cap = boundary_step(makespan, fetch_lag)
        assert term_offset <= drain_cap, f"block {bid}: term_offset {term_offset} exceeds the drain {drain_cap}"
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
                _word, land, _op = _value_word_and_landing(mir, vid, issue, fetch_lag)
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
