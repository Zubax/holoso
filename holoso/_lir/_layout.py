import logging
from collections.abc import Mapping
from dataclasses import replace
from typing import assert_never

from .._mir import Mir, MirBlock, MirBranch, MirRet, predecessors, reverse_postorder, successors
from .._operators import HardwareOperator, PooledHardwareOperator
from .._util import ValueId
from ._ir import *
from ._schedule import Schedule, resolve_pool, schedule_ops
from ._build_base import BlockOffsets, BlockSchedules
from ._mir_facts import mir_operation, succ_map

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


def _terminator_floor(mir: Mir, bid: int) -> int:
    """
    The entry block's terminator cannot redirect at PC 0, because the sequencer's accept hold (`pc==0`) precedes the
    branch redirect and the input loads land on cycle 1. Every other block may redirect at its own base PC, so an
    empty resident-condition branch there drains nothing, exactly like a jump.
    """
    return 1 if bid == mir.entry else 0


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
    result) is available from the block's first cycle and adds nothing.
    """
    floor = _terminator_floor(mir, block.id)
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


def schedule_blocks(mir: Mir, fetch_lag: int) -> BlockSchedules:
    """
    Schedule every block in reverse-postorder, threading cross-block overlap forward, once per graph. A block whose
    every successor is single-predecessor (so a spill cannot reach a wrong path) OVERLAPS: its terminator offset is
    fixed here at the issue-side envelope -- the latest cycle it still drives a control word, plus the branch
    condition's read floor -- and its in-flight results land past the terminator, in the uniquely-reached successor
    frame, which inherits `livein_landing` (the cycle each spilled value lands, successor-local in the
    `absolute_pc = block_base + cycle` frame, so as early as the successor's base PC), so its schedule reads no
    still-in-flight operand. Every other block drains, and its offset is derived per install-fixpoint round by
    `layout_offsets`. A block originating a phi arm never overlaps (its phi successor is multi-predecessor, which
    `_prepare` asserts -- the one fact the install set's independence from the schedules rests on), so the install
    set decides no schedule. Back-edge targets and merge blocks are multi-predecessor, so no overlap crosses them:
    the forward-DAG carry converges in this single pass with no fixpoint.
    """
    pool = resolve_pool(mir.nodes)
    succ = succ_map(mir)
    preds = {bid: len(sources) for bid, sources in predecessors(mir).items()}
    blocks_by_id = {block.id: block for block in mir.blocks}
    block_sched: dict[int, Schedule] = {}
    block_inflight: dict[int, dict[ValueId, int]] = {}
    overlap_term_offset: dict[int, int] = {}
    # successor block -> the landings its single overlapping predecessor spills into it (set at most once: a
    # carried-into block is single-predecessor, so only that one predecessor overlaps into it).
    carry: dict[int, dict[ValueId, int]] = {}
    for bid in reverse_postorder(mir):
        block = blocks_by_id[bid]
        livein_landing = carry.get(bid, {})
        block_inflight[bid] = livein_landing
        sched = schedule_ops(
            mir.nodes,
            pool,
            schedulable=set(block.operations),
            fetch_lag=fetch_lag,
            livein_landing=livein_landing,
        )
        block_sched[bid] = sched
        targets = succ[bid]
        if not targets or any(preds[target] != 1 for target in targets):
            continue
        term_offset = _issue_side_envelope(mir, sched, block, livein_landing, fetch_lag)
        overlap_term_offset[bid] = term_offset
        # The landings cross the shrunk terminator into the successor frame through the SAME coordinate map
        # (`successor_local_cycle`) that Lir.frame_pcs / Lir.write_landing_pcs and the model's redirect re-keying use.
        landing: dict[ValueId, int] = {}
        for vid in sched.issue_cycle:
            land = landing_cycle(sched.commit_cycle(vid), fetch_lag)
            if land > term_offset:
                landing[vid] = _spill_local_cycle(bid, land, term_offset)
        for vid, land in livein_landing.items():  # a received spill that re-spills past this shrunk terminator
            if land > term_offset:
                landing[vid] = max(landing.get(vid, 0), _spill_local_cycle(bid, land, term_offset))
        for target in targets:
            carry[target] = landing
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
    return BlockSchedules(block_sched, block_inflight, overlap_term_offset, instances)


def layout_offsets(
    mir: Mir,
    schedules: BlockSchedules,
    has_install_blocks: Mapping[int, bool],
    fetch_lag: int,
) -> BlockOffsets:
    """
    One install-fixpoint round's block layout over the once-computed schedules: each block's install-inclusive
    makespan and terminator offset. An overlapping block keeps its fixed envelope. A draining block that computes or
    installs anything ends at the drained boundary of its install-inclusive makespan: every result lands a fixed
    pipeline past its commit, so the latest landing is the makespan's, and a phi tail install lands read-first there
    too (the makespan one past the work only when a source is the block's own last work; see `install_issue_cycle`).
    An empty block ends where its received spills land, no earlier than its `_terminator_floor`. A slot's
    boundary install asks for no drain: it samples its source on the exit PC, as an output does.
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
            drains = bool(sched.issue_cycle) or bid in has_install_blocks
            drain = boundary_step(makespan, fetch_lag) if drains else _terminator_floor(mir, bid)
            term_offset = max([drain, *schedules.block_inflight[bid].values()])
        assert term_offset <= boundary_step(
            makespan, fetch_lag
        ), f"block {bid}: term_offset {term_offset} past the drain"
        block_term_offset[bid] = term_offset
    return BlockOffsets(block_makespan, block_term_offset)


def pc_less(mir: Mir, schedules: BlockSchedules, working: set[int]) -> dict[int, Arm]:
    """
    The blocks that take no PC, each mapped to the arm its predecessors take instead: a non-entry block outside
    `working` (the blocks that schedule an operation or carry a copy) that receives no spill and whose arms all resolve,
    through other such blocks, to one target other than itself. Nothing lands in its frame, so what an exit reads there
    is already resident at its predecessor's terminator.
    """
    idle = {
        block.id: [Exit()] if isinstance(block.terminator, MirRet) else successors(block)
        for block in mir.blocks
        if block.id != mir.entry and block.id not in working and not schedules.block_inflight[block.id]
    }
    resolved: dict[int, Arm] = {}

    def resolve(arm: Arm) -> Arm:
        while isinstance(arm, int) and arm in resolved:
            arm = resolved[arm]
        return arm

    changed = True
    while changed:
        changed = False
        for index in sorted(idle.keys() - resolved.keys()):
            targets = {resolve(arm) for arm in idle[index]}
            if len(targets) == 1 and (target := targets.pop()) != index:
                resolved[index] = target
                changed = True
    return {index: resolve(index) for index in resolved}


def _retarget(terminator: Terminator, pc_less: Mapping[int, Arm]) -> Terminator:
    def resolve(arm: Arm) -> Arm:
        return pc_less.get(arm, arm) if isinstance(arm, int) else arm

    match terminator:
        case Jump(target=target):
            return Jump(resolve(target))
        case Branch(cond=cond, if_true=if_true, if_false=if_false):
            arms = resolve(if_true), resolve(if_false)
            return Jump(arms[0]) if arms[0] == arms[1] else Branch(cond, *arms)
        case _:
            assert_never(terminator)


def layout_blocks(mir: Mir, blocks: list[LirBlock], resolved: Mapping[int, Arm]) -> list[LirBlock]:
    """
    The blocks that take PCs, laid out linearly in reverse-postorder; each spans `term_offset + 1` fetch steps, the
    successor frame beginning at `term_pc + 1`. A back-edge targets an earlier, lower-addressed block, which the
    next-PC sequencer redirects like any other jump; the frontend emits reducible loops, so a back-edge target dominates
    it. Arms into a block that takes no PC (`resolved`, see `pc_less`) are resolved through it.
    """
    if resolved:
        _logger.info("Layout: blocks %s take no PC", sorted(resolved))
    blocks = [replace(b, terminator=_retarget(b.terminator, resolved)) for b in blocks if b.index not in resolved]
    by_index = {block.index: block for block in blocks}
    return [by_index[index] for index in reverse_postorder(mir) if index in by_index]
