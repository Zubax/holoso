import logging
from collections.abc import Mapping
from dataclasses import dataclass

from .._mir import Mir, MirBlock, MirBoolView, MirBranch, MirWideView, reverse_postorder
from .._operators import HardwareOperator, PooledHardwareOperator
from .._util import ValueId
from ._ir import *
from ._schedule import Schedule, resolve_pool, schedule_ops
from ._build_base import BlockOffsets, BlockSchedules
from ._mir_facts import mir_operation, pred_count, succ_map

_logger = logging.getLogger(__name__)


def _control_word(mir: Mir, vid: ValueId, issue: int, fetch_lag: int) -> tuple[int, HardwareOperator]:
    """
    For a scheduled value, the last in-block control WORD in its block-local frame: the latest fetch step the op
    still drives -- a pooled lane's write opcode on its commit step, or an inline op's combinational fire step one
    fetch lag later. The result LANDS later still, uniformly for both classes, after the fetch pipeline; cross-block
    overlap may place the terminator between the two, the word staying in the block and the landing spilling into
    the single-predecessor successor frame.
    """
    operator = mir_operation(mir, vid).operator
    commit = issue + operator.latency
    if isinstance(operator, PooledHardwareOperator):
        return pooled_write_word(commit), operator
    return inline_fire_cycle(commit, fetch_lag), operator


@dataclass(frozen=True, slots=True)
class _SpillCarry:
    """
    The cross-block-overlap residue a block hands each single-predecessor successor: per-instance busy windows still
    in flight at the shrunk terminator (`entry_busy`) and the values whose write spills past it (`livein_landing`,
    the value's landing cycle in the successor-local frame). Both are successor-local cycles in the
    `absolute_pc = block_base + cycle` frame the scheduler uses (via `successor_local_cycle`), so a spill can land
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
    (`livein_landing`); a RESIDENT live-in condition (an input, persistent state, or a fully-drained prior-block
    result) is available from the block's first cycle and adds nothing. The floor starts at 1 for the ENTRY block only:
    its terminator cannot redirect at PC 0, because the sequencer's accept hold (`pc==0`) precedes the branch
    redirect, so an entry branch must settle at PC>=1. Every other block may redirect at its own base PC, so its floor
    starts at 0 -- an empty resident-condition branch then drains nothing, exactly like a jump.
    """
    floor = 1 if block.id == mir.entry else 0
    for vid, issue in sched.issue_cycle.items():
        word, operator = _control_word(mir, vid, issue, fetch_lag)
        # Without the operand-read floor the op would fire past the shrunk terminator and never execute.
        floor = max(floor, word, operand_read_cycle(operator, issue, fetch_lag))
        if isinstance(operator, PooledHardwareOperator) and operator.error_ports:
            # The err_pc diagnostic latches `pc - fetch_lag` when this op's write-enable executes, which is
            # fetch_lag fetch steps after its write word. If the terminator redirected by then, err_pc would
            # capture the successor frame's PC instead of this op's step. Keep the latch inside the block: the
            # data write still lands correctly, but the diagnostic needs the live PC in-frame.
            floor = max(floor, word + fetch_lag)
    if isinstance(block.terminator, MirBranch):
        cond = block.terminator.cond
        if cond in sched.issue_cycle:
            floor = max(floor, landing_cycle(sched.commit_cycle(cond), fetch_lag))
        elif cond in livein_landing:
            floor = max(floor, livein_landing[cond])
    return floor


def _spill_local_cycle(bid: int, block_local_cycle: int, term_offset: int) -> int:
    """
    The successor-local cycle of a value spilling past block `bid`'s shrunk terminator. The callers gate on
    `block_local_cycle > term_offset`, so a real spill is non-negative (cycle 0 at the successor base is legal).
    """
    local = successor_local_cycle(block_local_cycle, term_offset)
    assert local >= 0, f"block {bid}: spilled landing PC {local} precedes the successor base"
    return local


def schedule_blocks(mir: Mir, wide_mir: MirWideView, bool_mir: MirBoolView, fetch_lag: int) -> BlockSchedules:
    """
    Schedule every block in reverse-postorder, threading cross-block overlap forward, once per build. A block whose
    every successor is single-predecessor (so a spill cannot reach a wrong path) OVERLAPS: its terminator offset is
    fixed here at the issue-side envelope -- the latest cycle it still drives a control word, plus the branch
    condition's read floor -- and its in-flight results land past the terminator, in the uniquely-reached successor
    frame, which inherits `entry_busy` (the predecessor's per-instance busy residue) and `livein_landing` (the cycle
    each spilled value lands), so its schedule neither reads a still-in-flight operand nor double-drives a busy
    instance. Every other block drains, and its offset is derived per install-fixpoint round by `layout_offsets`. A
    block originating a phi arm never overlaps (its phi successor is multi-predecessor, which `_prepare` asserts -- the
    one fact the install set's independence from the schedules rests on), so the install set decides no schedule.
    Back-edge targets and merge blocks are multi-predecessor, so no overlap crosses them: the forward-DAG carry
    converges in this single pass with no fixpoint.
    """
    pool = resolve_pool(mir.nodes)
    succ = succ_map(mir)
    preds = pred_count(mir)
    blocks_by_id = {block.id: block for block in mir.blocks}
    block_sched: dict[int, Schedule] = {}
    block_inflight: dict[int, dict[ValueId, int]] = {}
    block_entry_busy: dict[int, dict[tuple[PooledHardwareOperator, int], int]] = {}
    overlap_term_offset: dict[int, int] = {}
    # successor block -> the spill carry its single overlapping predecessor hands it (set at most once: a carried-into
    # block is single-predecessor, so only that one predecessor overlaps into it).
    carry: dict[int, _SpillCarry] = {}
    for bid in reverse_postorder(mir):
        block = blocks_by_id[bid]
        inherited = carry.get(bid, _SpillCarry({}, {}))
        livein_landing = inherited.livein_landing
        block_inflight[bid] = livein_landing
        block_entry_busy[bid] = inherited.entry_busy
        sched = schedule_ops(
            mir.nodes,
            pool,
            schedulable=set(wide_mir.block_operations(block)) | set(bool_mir.block_operations(block)),
            fetch_lag=fetch_lag,
            entry_busy=inherited.entry_busy,
            livein_landing=livein_landing,
        )
        block_sched[bid] = sched
        targets = succ[bid]
        if not targets or any(preds[target] != 1 for target in targets):
            continue
        term_offset = _issue_side_envelope(mir, sched, block, livein_landing, fetch_lag)
        overlap_term_offset[bid] = term_offset
        # Both the per-instance busy residue and the value landings cross the shrunk terminator into the successor
        # frame, so both translate through the SAME coordinate map (`successor_local_cycle`) that _trace_landing /
        # Lir.write_landing_pcs and the model's redirect re-keying use -- the scheduler reserves and read-gates each
        # register/instance at the cycle the pipeline truly frees/writes it, on one coordinate contract.
        busy = {
            inst: successor_local_cycle(free, term_offset)
            for inst, free in sched.busy_until.items()
            if successor_local_cycle(free, term_offset) > 0
        }
        landing: dict[ValueId, int] = {}
        for vid in sched.issue_cycle:
            land = landing_cycle(sched.commit_cycle(vid), fetch_lag)
            if land > term_offset:
                landing[vid] = _spill_local_cycle(bid, land, term_offset)
        for vid, land in livein_landing.items():  # a received spill that re-spills past this shrunk terminator
            if land > term_offset:
                landing[vid] = max(landing.get(vid, 0), _spill_local_cycle(bid, land, term_offset))
        spill = _SpillCarry(busy, landing)
        for target in targets:
            carry[target] = spill
    instances: dict[PooledHardwareOperator, int] = {}
    for sched in block_sched.values():
        for inst in sched.inst_of.values():
            instances[inst.operator] = max(instances.get(inst.operator, 0), inst.index + 1)
    _logger.info(
        "Operator instances used: %s",
        ", ".join(f"{operator.mnemonic} {count}/{pool[type(operator)]}" for operator, count in instances.items())
        or "none",
    )
    _logger.info(
        "Schedules: %d blocks, %d overlapping, %d receiving spills",
        len(block_sched),
        len(overlap_term_offset),
        len(carry),
    )
    return BlockSchedules(block_sched, block_inflight, block_entry_busy, overlap_term_offset, instances)


def layout_offsets(
    mir: Mir,
    schedules: BlockSchedules,
    has_install_blocks: Mapping[int, bool],
    boundary_install: bool,
    fetch_lag: int,
) -> BlockOffsets:
    """
    One install-fixpoint round's block layout over the once-computed schedules: each block's install-inclusive
    makespan and terminator offset. An overlapping block keeps its fixed envelope. A draining block that computes or
    installs anything ends at the drained boundary of its install-inclusive makespan: every result lands a fixed
    pipeline past its commit, so the latest landing is the makespan's, and a phi tail install lands read-first there
    too (the makespan one past the work only when a source is the block's own last work; see `install_issue_cycle`).
    The lone Ret block is charged that boundary while a slot installs there at the accepted-output edge
    (`boundary_install`), which matters only when it computes nothing. An empty block ends where its received spills
    land, the entry no earlier than its input loads on cycle 1.
    """
    block_makespan: dict[int, int] = {}
    block_term_offset: dict[int, int] = {}
    for bid, sched in schedules.block_sched.items():
        makespan = sched.makespan + (1 if has_install_blocks.get(bid, False) else 0)
        block_makespan[bid] = makespan
        if bid in schedules.overlap_term_offset:
            assert bid not in has_install_blocks, f"block {bid}: an overlapping block cannot carry an install"
            term_offset = schedules.overlap_term_offset[bid]
        else:
            drains = bool(sched.issue_cycle) or bid in has_install_blocks or (bid == mir.ret_block and boundary_install)
            floor = 1 if bid == mir.entry else 0
            drain = boundary_step(makespan, fetch_lag) if drains else floor
            term_offset = max([drain, *schedules.block_inflight[bid].values()])
        assert term_offset <= boundary_step(
            makespan, fetch_lag
        ), f"block {bid}: term_offset {term_offset} past the drain"
        block_term_offset[bid] = term_offset
    return BlockOffsets(block_makespan, block_term_offset)


@dataclass(frozen=True, slots=True)
class _BlockLayout:
    """The ROM placement: per-block base PC, the out_valid PC, and the shortest-path initiation interval."""

    block_base: list[int]
    last_pc: int
    min_initiation_interval: int


def layout_blocks(mir: Mir, blocks: list[LirBlock]) -> _BlockLayout:
    """
    Each block spans `term_offset + 1` fetch steps: its body up to and including the terminator step, the successor
    frame beginning at `term_pc + 1`.
    """
    successors: dict[int, list[int]] = {b.index: terminator_arms(b.terminator) for b in blocks}
    # Blocks are laid out linearly in reverse-postorder, but the single Ret block is forced last so its boundary is the
    # highest address (out_valid = pc == LASTPC). A loop body is a DFS leaf (its only edge back to the header is a back
    # edge), so RPO would otherwise place it after the exit; moving Ret last keeps every loop body below the Ret. A
    # back-edge targets an earlier, lower-addressed block, which the next-PC sequencer redirects like any other jump, so
    # the linear layout needs no special case; the frontend emits reducible loops, so a back-edge target dominates it.
    ret_index = next(b.index for b in blocks if isinstance(b.terminator, Ret))
    order = [bid for bid in reverse_postorder(mir) if bid != ret_index] + [ret_index]
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
                continue
            cand = here + length[index]
            if successor not in dist or cand < dist[successor]:
                dist[successor] = cand
    min_ii = dist.get(ret_index, 0) + term_offset[ret_index]
    block_base = [base[i] for i in range(len(blocks))]
    return _BlockLayout(block_base, last_pc, min_ii)
