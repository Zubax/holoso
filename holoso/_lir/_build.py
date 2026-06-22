"""Build a finished :class:`Lir` from MIR."""

import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar, Generic, TypeVar

from .._errors import UnsupportedConstruct
from .._mir import (
    Mir,
    MirBlock,
    MirBoolConst,
    MirBoolInput,
    MirBoolOutput,
    MirBoolStateSlot,
    MirBoolView,
    MirBranch,
    MirFloatConst,
    MirFloatInput,
    MirFloatOutput,
    MirFloatStateSlot,
    MirOperation,
    MirFloatView,
    MirJump,
    MirNode,
    MirPhi,
    MirRet,
    MirStateRead,
    MirStateSlot,
    MirTerminator,
)
from .._operators import (
    BoolInversion,
    FloatSignControl,
    HardwareOperator,
    InlineHardwareOperator,
    PooledHardwareOperator,
    PortConditioner,
)
from .._util import ValueId
from ._ir import *
from ._ir import _terminator_arms  # an underscore helper, so not pulled in by the ``import *`` above
from ._liveness import BankLiveness, compute_interference
from ._portassign import assign_commutative_ports
from ._regalloc import ColoringProblem, color, find_coloring_conflict
from ._schedule import Schedule, resolve_pool, schedule_ops

type _Producer = OperatorInstance | str  # write-source identity for the steering objective (see ._regalloc)
type _Port = tuple[OperatorInstance, int]  # read-port identity (instance + operand position) for the steering objective


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
    real, not merely theoretical, and is prevented by the dwell floor in ``_schedule_with_overlap`` (the ``state_liveouts``
    set, which includes both slot live-outs and the arm producers of any phi live-out, is held off cycle 0 in the entry
    block). This assertion is the loud backstop on that floor: the dwell is invisible to BOTH validation paths (the cosim
    bench never delays ``in_valid``; the model asserts it at once and keys cycle-0 ops at their read pc, never pc 0), so a
    regression that let a coalesced result reach a state register on cycle 0 would otherwise corrupt state silently.
    """
    entry = lir.blocks[lir.entry]
    state_regs = {slot.reg for slot in lir.float_state_slots} | {slot.reg for slot in lir.bool_state_slots}
    cycle0_writes = [w.dst for op in entry.ops if op.issue_cycle == 0 for w in op.writes]
    cycle0_writes += [op.write.dst for op in entry.inline_ops if op.issue_cycle == 0]
    for dst in cycle0_writes:
        assert dst not in state_regs, f"entry-block cycle-0 op writes persistent-state register {dst.stable_label}"


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
class _BankAlloc:
    """One bank's assignment: register per value, register per state slot, count, per-slot install, and coalescing."""

    reg: dict[ValueId, int]
    slot_reg: dict[str, int]
    nreg: int
    install: dict[str, int]  # slot name -> Ret-block-relative scheduler-frame install cycle of its live-out
    coalesced: frozenset[tuple[int, ValueId]]  # (pred, phi) arms coalesced onto the merged register (no copy)


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
    landing = wide_landing_cycle(commit) if wide else bool_landing_cycle(commit)
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
class _OverlapLayout:
    """
    The per-block schedule plus the install-inclusive makespan, the (possibly overlap-shrunk) terminator offset, and
    the spills each block receives -- the predecessor values landing in it past an overlapped terminator, mapped to
    their block-local landing cycle (fed to the allocator's liveness so a spilled register stays reserved in the
    block, and identical to the scheduler's ``livein_landing`` so the two cannot drift). Empty under draining.
    """

    block_sched: dict[int, Schedule]
    block_makespan: dict[int, int]
    block_term_offset: dict[int, int]
    block_inflight: dict[int, dict[ValueId, int]]


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
class _LayoutAllocation:
    """One full layout+allocation pass for a given install set: the overlap layout, pooled instances, the const pool,
    and the register assignment of both banks. Re-run by the coalesced-install fixpoint as the install set shrinks."""

    overlap: _OverlapLayout
    inst_of: dict[ValueId, OperatorInstance]
    instances: list[OperatorInstance]
    consts: list[float]
    const_pool: dict[ValueId, _PooledConst]
    alloc: _Allocation


def _layout_and_allocate(
    mir: Mir,
    float_mir: MirFloatView,
    bool_mir: MirBoolView,
    pool: Mapping[type[HardwareOperator], int],
    has_install_blocks: set[int],
) -> _LayoutAllocation:
    """Lay out the blocks (cross-block overlap) and color both register banks for the given per-block install set."""
    overlap = _schedule_with_overlap(mir, float_mir, bool_mir, pool, has_install_blocks)
    block_sched = overlap.block_sched
    inst_of: dict[ValueId, OperatorInstance] = {}
    inst_count: dict[PooledHardwareOperator, int] = {}
    for sched in block_sched.values():
        inst_of.update(sched.inst_of)
        for inst in sched.instances:
            inst_count[inst.operator] = max(inst_count.get(inst.operator, 0), inst.index + 1)
    instances = [OperatorInstance(operator, i) for operator in inst_count for i in range(inst_count[operator])]
    consts, const_pool = _build_const_pool(float_mir, bool_mir.operation_nodes)
    alloc = _allocate(
        mir,
        float_mir,
        bool_mir,
        block_sched,
        inst_of,
        overlap.block_makespan,
        overlap.block_term_offset,
        overlap.block_inflight,
    )
    return _LayoutAllocation(overlap, inst_of, instances, consts, const_pool, alloc)


def _actual_install_blocks(alloc: _Allocation, const_branch_blocks: set[int]) -> set[int]:
    """
    The blocks that actually install at their tail after coalescing: a real float copy or boolean write, or a const-
    branch materialization (which is not a copy). A CFG-shape phi-arm predecessor whose every arm coalesced installs
    nothing, so it should pay neither the +1 install makespan nor the overlap-ineligibility that ``_block_has_install``
    assigns from the CFG shape alone -- it drops out of the install set here, which the fixpoint feeds back to the next
    layout so the spurious drain is removed.
    """
    blocks = set(const_branch_blocks)
    blocks.update(bid for bid, copies in alloc.copies.items() if copies)
    blocks.update(bid for bid, writes in alloc.bool_writes.items() if writes)
    return blocks


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
    # The install set is computed to a fixpoint. ``_block_has_install`` marks a block install-bearing from the CFG shape
    # (any phi arm originates in it), but a block whose every arm COALESCES onto the merged register installs nothing,
    # so that +1 drain (and overlap-ineligibility) is spurious. So: lay out and allocate with the conservative CFG set,
    # recompute the install set from the ACTUAL coalesced copies, and re-run until it stops shrinking. Convergence is by
    # monotonicity -- dropping a block's spurious drain frees registers one step earlier, which only enables more
    # coalescing, so the install set is non-increasing over a finite block set and reaches a fixpoint (the assert guards
    # the monotonicity; the iteration count is bounded by the block count). Determinism is preserved: the allocator is
    # seed-fixed and the install set is rebuilt the same way each pass.
    #
    # This install fixpoint NESTS a second one: each ``_layout_and_allocate`` round runs the per-bank phi-coalescing /
    # coloring fixpoint in ``_coalesce_and_color``. Both are bounded and monotone -- the install set non-increasing here,
    # the inner forbidden-merge set non-decreasing there -- so the composition terminates in at most block-count outer
    # rounds, each a bounded inner fixpoint. The coupling is one-way and cannot deadlock: a shrinking install set only
    # relieves register pressure, enabling more coalescing, never forbidding a merge the inner round already made.
    const_branch_blocks = set(_const_branch_conditions(mir, bool_mir))
    has_install_blocks = _block_has_install(mir, float_mir, bool_mir)
    for _ in range(len(mir.blocks) + 1):
        result = _layout_and_allocate(mir, float_mir, bool_mir, pool, has_install_blocks)
        actual = _actual_install_blocks(result.alloc, const_branch_blocks)
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
        has_install = block.id in has_install_blocks
        block_makespan = _install_inclusive_makespan(work_makespan, has_install)
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
                # The terminator offset from the overlap layout: the drained boundary, or shrunk to the issue-side
                # envelope when this block's in-flight results spill into single-predecessor successors.
                overlap.block_term_offset[block.id],
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
    lir = Lir(
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
    # A non-coalesced float slot's writeback fires read-first at ``state_copy_step``, at last_pc for a boundary install or
    # below it for an early one. A boundary that collapsed below the install would drop the writeback and freeze the
    # persistent state; the per-block ``term_offset <= wide drain`` invariant in ``_schedule_with_overlap`` is the
    # matching guard for the opposite slip (a boundary install degrading into an early one). Backstop, not a live failure.
    for slot in lir.float_state_slots:
        if slot.needs_copy:
            assert (
                lir.state_copy_step(slot) <= last_pc
            ), f"state slot {slot.name!r} writeback at {lir.state_copy_step(slot)} lands past the boundary {last_pc}"
    return lir


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
            assert False, f"unhandled MIR output {out!r}"
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


def _phi_arm_out(mir: Mir, phi_nodes: dict[ValueId, MirPhi], values: set[ValueId]) -> dict[int, frozenset[ValueId]]:
    """
    Per block, the phi-arm values live out of it (each read by the phi's install copy at the block's tail) -- a liveness
    input for both banks. The residual installs (which phi registers a block writes) are derived separately, per chosen
    coalescing, by :func:`_residual_installs`, so they are not precomputed here.
    """
    arm_out: dict[int, set[ValueId]] = {block.id: set() for block in mir.blocks}
    for _vid, phi in phi_nodes.items():
        for pred, arm, _conditioner in phi.arms:
            if arm in values:
                arm_out[pred].add(arm)
    return {b: frozenset(s) for b, s in arm_out.items()}


def _const_branch_conditions(mir: Mir, bool_mir: MirBoolView) -> dict[int, ValueId]:
    """
    Per block, the constant branch condition it materializes at its tail. A block whose ``MirBranch`` tests a globally
    interned boolean constant has no condition register, so the constant is written into a bool register in the
    branching block. The single source of this CFG-shape fact, shared by ``_block_has_install`` and the install fixpoint
    seed (which must agree, or the monotonicity assert trips) and the allocator's materialization (which also needs the
    condition value to write).
    """
    conditions: dict[int, ValueId] = {}
    for block in mir.blocks:
        term = block.terminator
        if isinstance(term, MirBranch) and term.cond in bool_mir.const_nodes:
            conditions[block.id] = term.cond
    return conditions


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
    has.update(_const_branch_conditions(mir, bool_mir))
    return has


@dataclass(frozen=True, slots=True)
class _PhiCoalescing:
    """One bank's phi-arm coalescing outcome: the class leader of every merged value and the arms that lost their copy."""

    leader: dict[ValueId, ValueId]  # value -> class leader (a value never merged maps to itself, implicitly)
    coalesced: frozenset[tuple[int, ValueId]]  # (pred, phi) arms that share the merged register (no install copy)


def _coalescable_arms(
    phi_nodes: Mapping[ValueId, MirPhi], values: set[ValueId], identity: PortConditioner
) -> dict[ValueId, list[tuple[int, ValueId]]]:
    """
    Per phi, the register-backed, identity-conditioner arms eligible to coalesce (so the arm flows into the merged
    register with no install copy). An arm that ANOTHER arm of the same phi reads under a non-identity conditioner is
    excluded: its residual copy would become a same-step self-conditioned copy (``r <= -r`` / ``b <= ~b``) into the
    merged register, which the install-free oracle cannot see and which the final interference rightly flags. Both banks
    differ only in their identity conditioner (:class:`FloatSignControl` vs :class:`BoolInversion`).
    """
    candidates: dict[ValueId, list[tuple[int, ValueId]]] = {}
    for vid, phi in phi_nodes.items():
        conditioned = {arm for _p, arm, cond in phi.arms if arm in values and cond != identity}
        candidates[vid] = [
            (pred, arm)
            for pred, arm, conditioner in phi.arms
            if arm in values and conditioner == identity and arm not in conditioned
        ]
    return candidates


def _coalesce_phis(
    phi_nodes: Mapping[ValueId, MirPhi],
    phi_order: list[ValueId],
    candidate_arms: dict[ValueId, list[tuple[int, ValueId]]],
    oracle: dict[ValueId, set[ValueId]],
    pinned: dict[ValueId, int],
    reserved_regs: set[int],
    forbidden: set[tuple[int, ValueId]],
) -> _PhiCoalescing:
    """
    Union-find phi-arm coalescing for one bank. Each phi result and its register-backed, identity-conditioner arms
    (``candidate_arms``) merge into one congruence class whenever the merge introduces no interference -- judged on
    ``oracle``, the install-free interference graph, since a coalesced class carries no install copy. The arms and the
    phi then share a register and the install copy vanishes. A class carries at most one pinned register; a class may
    not land on a ``reserved`` register -- the NON-coalesced state-slot registers, whose copy-back/early-install
    machinery owns them. A COALESCED slot's register is NOT reserved: it is seeded by the slot live-out's pin, so the
    phi live-out and the slot live-in (its "unchanged" arm) merge onto it for an in-place commit. ``phi_order`` is the
    deterministic processing order (block reverse-postorder, then value id); arms are processed in their phi-arm order.

    ``oracle`` is only an OVER-APPROXIMATION of coalescability: it omits the residual (non-coalesced) arms' install
    writes, so it can admit a merge the final install-aware interference rejects (a coalesced phi whose residual
    sibling arm's install lands in a register a class member is still live in). The caller (:func:`_coalesce_and_color`)
    corrects this by rebuilding the final interference and re-running with the offending arms in ``forbidden`` -- the
    ``(pred, phi)`` arms this call must skip, never admitting them into a class. The fixpoint converges because
    forbidding only ever grows.
    """
    parent: dict[ValueId, ValueId] = {}
    members: dict[ValueId, set[ValueId]] = {}
    pin: dict[ValueId, int] = {}

    def find(x: ValueId) -> ValueId:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:  # path compression
            parent[x], x = root, parent[x]
        return root

    def ensure(x: ValueId) -> None:
        if x not in parent:
            parent[x] = x
            members[x] = {x}
            if x in pinned:
                pin[x] = pinned[x]

    for phi_vid in phi_order:
        ensure(phi_vid)
        for pred, arm in candidate_arms.get(phi_vid, []):
            if (pred, phi_vid) in forbidden:
                continue  # a prior fixpoint round found this merge unsound under the final interference
            ensure(arm)
            la, lb = find(phi_vid), find(arm)
            if la == lb:
                continue
            pa, pb = pin.get(la), pin.get(lb)
            if pa is not None and pb is not None and pa != pb:
                continue  # the two classes are pinned to different registers
            # Equal pins (pa == pb) merge consistently onto that register; they arise only on a COALESCED slot register
            # (the slot's live-in and its in-place live-out share its slot pin) and SHOULD merge -- that merge is the
            # in-place commit. The other non-reserved pins are the input lanes, each a distinct register, so two distinct
            # non-reserved classes never share a pin. The guard below still rejects the NON-coalesced slot registers.
            merged_pin = pa if pa is not None else pb
            if merged_pin is not None and merged_pin in reserved_regs:
                continue  # a class touching a reserved (non-coalesced state-slot) register may not absorb a phi
            if any(not oracle[m].isdisjoint(members[lb]) for m in members[la]):
                continue  # some member of one class interferes with some member of the other
            lo, hi = (la, lb) if la < lb else (lb, la)  # leader = lowest value id, for determinism
            parent[hi] = lo
            members[lo] |= members.pop(hi)
            if merged_pin is not None:
                pin[lo] = merged_pin
            pin.pop(hi, None)

    leader = {v: find(v) for v in parent}
    coalesced = frozenset(
        (pred, phi_vid)
        for phi_vid in phi_nodes
        for pred, arm in candidate_arms.get(phi_vid, [])
        if leader.get(arm, arm) == leader.get(phi_vid, phi_vid)
    )
    return _PhiCoalescing(leader, coalesced)


@dataclass(frozen=True, slots=True)
class _ColorObjective:
    """
    One bank's steering inputs to quotient coloring, beyond the interference graph and pins: the deterministic movable
    order, the per-value consumer read ports and producers (the write-select objective), and the first freely
    assignable register. Threaded together because the colorer consumes them as a unit.
    """

    movable: list[ValueId]
    consumer_ports: dict[ValueId, set[tuple[OperatorInstance, int]]]
    producer_key: dict[ValueId, frozenset[_Producer]]
    fresh_start: int


def _color_quotient(
    leader: dict[ValueId, ValueId],
    pinned: dict[ValueId, int],
    interferes: dict[ValueId, set[ValueId]],
    objective: _ColorObjective,
) -> tuple[dict[ValueId, int], int, int | None]:
    """
    Color the per-value interference graph after collapsing each coalescing class to its leader, then expand the
    leader's color back onto every member. The quotient unions each class's consumer ports and producers so the
    steering objective stays exact (a coalesced register really is read/written by every member's port/producer). A
    class with a pinned member pins its leader. Reduces to the plain per-value coloring when ``leader`` is the identity
    (every value its own singleton class, e.g. a kernel with no coalescable phi arms). The third return is the register
    of an interfering co-assignment under the FULL (residual-install) interference, or None when the coloring is sound --
    a conflict can only come from the pins, which the caller resolves by backing the offending slot out of coalescing.
    """

    def lead(v: ValueId) -> ValueId:
        return leader.get(v, v)

    leaders = sorted({lead(v) for v in interferes})
    q_pinned: dict[ValueId, int] = {}
    for vid, reg in pinned.items():
        head = lead(vid)
        assert q_pinned.setdefault(head, reg) == reg, f"coalescing class {head} spans two pinned registers"
    q_interferes: dict[ValueId, set[ValueId]] = {head: set() for head in leaders}
    q_ports: dict[ValueId, set[tuple[OperatorInstance, int]]] = {head: set() for head in leaders}
    q_producers: dict[ValueId, set[_Producer]] = {head: set() for head in leaders}
    for vid in sorted(interferes):
        head = lead(vid)
        q_ports[head] |= objective.consumer_ports.get(vid, set())
        q_producers[head] |= objective.producer_key[vid]
        for other in interferes[vid]:
            head_other = lead(other)
            if head_other != head:
                q_interferes[head].add(head_other)
                q_interferes[head_other].add(head)
    q_movable: list[ValueId] = []
    seen: set[ValueId] = set()
    for vid in objective.movable:  # leaders of the movable values, first occurrence, preserving the deterministic order
        head = lead(vid)
        if head in q_pinned or head in seen:
            continue
        seen.add(head)
        q_movable.append(head)
    q_assign, nreg = color(
        ColoringProblem(
            movable=q_movable,
            pinned=q_pinned,
            interferes=q_interferes,
            consumer_ports=q_ports,
            producer_key={head: frozenset(producers) for head, producers in q_producers.items()},
            fresh_start=objective.fresh_start,
        )
    )
    assign = {vid: q_assign[lead(vid)] for vid in interferes}
    # Check the EXPANDED per-value coloring against the FULL (residual-install) interference -- stronger than a check over
    # the collapsed quotient, catching any unsound union or oracle drift. A conflict is returned (not raised) so the
    # slot-coalescing retry can back the offending slot register out and recolor.
    return assign, nreg, find_coloring_conflict(assign, interferes)


def _residual_installs(
    phi_nodes: Mapping[ValueId, MirPhi], coalesced: frozenset[tuple[int, ValueId]]
) -> dict[int, frozenset[ValueId]]:
    """Per predecessor block, the phi dests whose arm did NOT coalesce and so install by a pc-gated copy at its tail."""
    installs: dict[int, set[ValueId]] = {}
    for vid, phi in phi_nodes.items():
        for pred, _arm, _conditioner in phi.arms:
            if (pred, vid) not in coalesced:
                installs.setdefault(pred, set()).add(vid)
    return {pred: frozenset(dests) for pred, dests in installs.items()}


def _coalesce_and_color(
    phi_nodes: Mapping[ValueId, MirPhi],
    phi_order: list[ValueId],
    candidate_arms: dict[ValueId, list[tuple[int, ValueId]]],
    pinned: dict[ValueId, int],
    reserved_regs: set[int],
    build_interferes: Callable[[dict[int, frozenset[ValueId]]], dict[ValueId, set[ValueId]]],
    objective: _ColorObjective,
) -> tuple[dict[ValueId, int], int, _PhiCoalescing, int | None]:
    """
    Coalesce one bank's phi arms and color it, iterated to a soundness fixpoint. ``_coalesce_phis`` judges merges on the
    install-free oracle -- ``build_interferes({})``, the same interference graph with no residual installs -- which
    over-approximates coalescability (see its docstring); the final interference from the actual residual installs can
    therefore show a coalescing class interfering with itself -- a member still live where a residual sibling arm's
    install writes the merged register. When it does, every arm-merge of each offending class is FORBIDDEN and coalescing
    re-runs. Forbidding the whole class (not just the guilty merge) is an intentional sound-but-conservative choice: it
    cannot under-forbid, and the worst case (all arms forbidden) is the copy-everything baseline, which has no
    class-internal interference -- so the loop converges. The returned coalescing is the FINAL one; its ``coalesced``
    arms are exactly the copies the emitter elides. The fourth return is the register of an interfering co-assignment
    (from the pins) or None; the caller backs the offending slot out of in-place coalescing and recolors.
    """
    # The install-free baseline; deriving it here from the same builder keeps it in lockstep with the final graph.
    oracle = build_interferes({})
    forbidden: set[tuple[int, ValueId]] = set()
    arm_budget = sum(len(arms) for arms in candidate_arms.values())
    for _round in range(arm_budget + 1):  # forbidding grows by >= 1 each conflicting round; this bounds the fixpoint
        coalescing = _coalesce_phis(phi_nodes, phi_order, candidate_arms, oracle, pinned, reserved_regs, forbidden)
        interferes = build_interferes(_residual_installs(phi_nodes, coalescing.coalesced))
        bad_leaders: set[ValueId] = set()
        for vid, neighbours in interferes.items():
            head = coalescing.leader.get(vid, vid)
            if any(coalescing.leader.get(other, other) == head for other in neighbours):
                bad_leaders.add(head)  # this class interferes with itself under the final, install-aware graph
        if not bad_leaders:
            assign, nreg, conflict = _color_quotient(coalescing.leader, pinned, interferes, objective)
            return assign, nreg, coalescing, conflict
        forbidden |= {
            (pred, phi_vid)
            for pred, phi_vid in coalescing.coalesced
            if coalescing.leader.get(phi_vid, phi_vid) in bad_leaders
        }
    assert False, "phi-coalescing fixpoint did not converge"  # unreachable: forbidding is monotone and bounded


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
                op_commit[vid] = sched.commit_cycle(vid)
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


type _BankView = MirFloatView | MirBoolView
_SlotT = TypeVar("_SlotT", MirFloatStateSlot, MirBoolStateSlot)  # one bank's concrete state-slot type


@dataclass(frozen=True, slots=True)
class _ObjectiveContext:
    """The loop-invariant inputs to a bank's coloring objective; ``movable`` is layered on per coalescing attempt."""

    mir: Mir
    view: _BankView
    values: set[ValueId]
    op_nodes: Mapping[ValueId, MirOperation]
    phi_nodes: Mapping[ValueId, MirPhi]
    inst_of: Mapping[ValueId, OperatorInstance]


@dataclass(frozen=True, slots=True)
class _ObjectiveTerms:
    """The reusable part of a coloring objective: per-value read ports and write-source identity (loop-invariant)."""

    consumer_ports: dict[ValueId, set[_Port]]
    producer_key: dict[ValueId, frozenset[_Producer]]


@dataclass(frozen=True, slots=True)
class _InstallContext:
    """Inputs to a bank's slot live-out install policy for one coalescing attempt (Ret-block-relative cycles)."""

    slots: Sequence[MirStateSlot]
    coalesced: dict[str, ValueId]  # slot name -> live-out already committed in place (no install copy)
    tapped_by_other: set[str]  # slots whose live-in another slot's live-out reads (a chained copy)
    livein_of: dict[str, ValueId | None]
    op_nodes: Mapping[ValueId, MirOperation]
    op_commit: Mapping[ValueId, int]
    op_block: Mapping[ValueId, int]
    nodes: Mapping[ValueId, MirNode]
    boundary_ret: set[ValueId]  # the Ret block's non-slot boundary users
    last_read_ret: Mapping[ValueId, int]  # last operand-read cycle of each value in the Ret block
    ret_block: int
    ret_present: int


class _Bank(ABC, Generic[_SlotT]):
    """
    The policy surface for one physical register bank. The liveness/coalescing/coloring skeleton in
    :func:`_allocate_bank` is shared; each subclass supplies the wide/boolean specifics -- landing cycle, identity
    conditioner, slot/boundary/objective extraction, and the install policy.
    """

    label: ClassVar[str]
    identity: ClassVar[PortConditioner]  # the no-op conditioner whose absence lets a live-out commit into its slot

    @abstractmethod
    def landing_cycle(self, commit_cycle: int) -> int:
        """The hardware-frame cycle on which a result committed at ``commit_cycle`` lands in its register."""

    @abstractmethod
    def state_slots(self, view: _BankView) -> list[_SlotT]:
        """This bank's state slots, narrowed from the MIR view -- the one place the concrete bank type is asserted."""

    @abstractmethod
    def slot_identity(self, slot: _SlotT) -> bool:
        """Whether the slot folds the identity sideband, so its live-out may commit into the slot register in place."""

    @abstractmethod
    def boundary_base(self, mir: Mir, values: set[ValueId], ret_block: int) -> dict[int, set[ValueId]]:
        """The non-slot boundary users: bank outputs, plus the per-block branch conditions for the boolean bank."""

    @abstractmethod
    def objective_terms(self, ctx: _ObjectiveContext) -> _ObjectiveTerms:
        """The loop-invariant objective terms: the wide read-mux/write-select fan-in, or the boolean's degenerate one."""

    @abstractmethod
    def install_policy(self, ctx: _InstallContext) -> dict[str, int]:
        """
        Each slot's Ret-block-relative live-out install cycle. The wide bank installs early where it can to free the
        source register; the boolean bank installs every live-out at the boundary (it has no early install).
        """


class _WideBank(_Bank[MirFloatStateSlot]):
    label = "wide"
    identity = FloatSignControl()

    def landing_cycle(self, commit_cycle: int) -> int:
        return wide_landing_cycle(commit_cycle)

    def state_slots(self, view: _BankView) -> list[MirFloatStateSlot]:
        assert isinstance(view, MirFloatView)
        return view.state_slots

    def slot_identity(self, slot: MirFloatStateSlot) -> bool:
        return slot.sign == FloatSignControl()

    def boundary_base(self, mir: Mir, values: set[ValueId], ret_block: int) -> dict[int, set[ValueId]]:
        boundary: dict[int, set[ValueId]] = {block.id: set() for block in mir.blocks}
        for out in mir.outputs:
            if isinstance(out, MirFloatOutput) and out.value in values:
                boundary[ret_block].add(out.value)
        return boundary

    def objective_terms(self, ctx: _ObjectiveContext) -> _ObjectiveTerms:
        consumer_ports: dict[ValueId, set[_Port]] = {vid: set() for vid in ctx.values}
        for vid, inst in ctx.inst_of.items():
            use_node = ctx.mir.nodes[vid]
            if not isinstance(use_node, MirOperation):
                continue
            for pos, operand in enumerate(use_node.operands):
                if operand in ctx.values:
                    consumer_ports[operand].add((inst, pos))
        producer_key: dict[ValueId, frozenset[_Producer]] = {vid: frozenset({"input"}) for vid in ctx.view.input_ids}
        producer_key.update({vid: frozenset({f"state:{node.name}"}) for vid, node in ctx.view.state_read_nodes.items()})
        producer_key.update(
            {vid: frozenset({ctx.inst_of[vid] if vid in ctx.inst_of else f"cast:{vid}"}) for vid in ctx.op_nodes}
        )
        producer_key.update({vid: frozenset({f"phi:{vid}"}) for vid in ctx.phi_nodes})
        return _ObjectiveTerms(consumer_ports, producer_key)

    def install_policy(self, ctx: _InstallContext) -> dict[str, int]:
        # Install the live-out as early as the live-in is fully read and the source is available, freeing the source
        # register -- but only when the live-out is produced in the Ret block (a unique, once-per-transaction exit),
        # the live-in is not itself a boundary user, and the slot neither coalesced nor feeds a chained copy. Otherwise
        # the boundary.
        install: dict[str, int] = {}
        for slot in ctx.slots:
            name, live_out, r_in = slot.name, slot.live_out, ctx.livein_of[slot.name]
            node = ctx.nodes[live_out]
            defined_in_ret = isinstance(node, MirFloatInput) or (
                live_out in ctx.op_nodes and ctx.op_block.get(live_out) == ctx.ret_block
            )
            early = (
                name not in ctx.coalesced
                and name not in ctx.tapped_by_other
                and defined_in_ret
                and (r_in is None or r_in not in ctx.boundary_ret)
            )
            if early:
                cycle = (ctx.op_commit[live_out] if live_out in ctx.op_nodes else 0) + 1  # read-first: an older commit
                if r_in is not None:
                    cycle = max(cycle, ctx.last_read_ret.get(r_in, 0) - copy_step_cycle(0))
                install[name] = min(cycle, ctx.ret_present)
            else:
                install[name] = ctx.ret_present
        return install


class _BoolBank(_Bank[MirBoolStateSlot]):
    label = "bool"
    identity = BoolInversion()

    def landing_cycle(self, commit_cycle: int) -> int:
        return bool_landing_cycle(commit_cycle)

    def state_slots(self, view: _BankView) -> list[MirBoolStateSlot]:
        assert isinstance(view, MirBoolView)
        return view.state_slots

    def slot_identity(self, slot: MirBoolStateSlot) -> bool:
        return slot.inversion == BoolInversion()

    def boundary_base(self, mir: Mir, values: set[ValueId], ret_block: int) -> dict[int, set[ValueId]]:
        boundary: dict[int, set[ValueId]] = {block.id: set() for block in mir.blocks}
        for block in mir.blocks:
            if isinstance(block.terminator, MirBranch) and block.terminator.cond in values:
                boundary[block.id].add(block.terminator.cond)
        for out in mir.outputs:
            if isinstance(out, MirBoolOutput) and out.value in values:
                boundary[ret_block].add(out.value)
        return boundary

    def objective_terms(self, ctx: _ObjectiveContext) -> _ObjectiveTerms:
        # The boolean bank has no read multiplexer and a one-hot pc-gated write chain, so it carries no read ports and a
        # single uniform producer -- the objective degenerates to register count.
        return _ObjectiveTerms(
            {vid: set() for vid in ctx.values},
            {vid: frozenset({"bool"}) for vid in ctx.values},
        )

    def install_policy(self, ctx: _InstallContext) -> dict[str, int]:
        return {slot.name: ctx.ret_present for slot in ctx.slots}  # every boolean live-out installs at the boundary


_WIDE = _WideBank()
_BOOL = _BoolBank()


def _allocate_bank(
    bank: _Bank[_SlotT],
    mir: Mir,
    view: _BankView,
    block_sched: dict[int, Schedule],
    inst_of: Mapping[ValueId, OperatorInstance],
    block_makespan: dict[int, int],
    block_term_offset: dict[int, int],
    block_inflight: dict[int, dict[ValueId, int]],
) -> _BankAlloc:
    """
    Color one physical register bank across the CFG. The bank descriptor supplies the landing-cycle, conditioner,
    boundary, objective, and install policies; the liveness/coalescing/coloring skeleton is shared.
    """
    nload = len(view.input_ids)
    slots = bank.state_slots(view)
    slot_reg = {slot.name: nload + i for i, slot in enumerate(slots)}
    fresh_start = nload + len(slots)
    # Explicit bindings: the union-view property types are undecidable under the constrained-TypeVar reanalysis.
    op_nodes: dict[ValueId, MirOperation] = view.operation_nodes
    phi_nodes: dict[ValueId, MirPhi] = view.phi_nodes
    state_read_nodes: Mapping[ValueId, MirStateRead] = view.state_read_nodes
    state_read_of = {node.name: vid for vid, node in state_read_nodes.items()}
    values = {*view.input_ids, *state_read_nodes, *op_nodes, *phi_nodes}
    facts = _bank_liveness_facts(mir, block_sched, op_nodes, phi_nodes, values)
    op_block, op_commit, phi_block, reads = facts.op_block, facts.op_commit, facts.phi_block, facts.reads

    ret_block = next(b.id for b in mir.blocks if isinstance(b.terminator, MirRet))
    arm_out = _phi_arm_out(mir, phi_nodes, values)
    boundary_base = bank.boundary_base(mir, values, ret_block)

    def graph(
        boundary: dict[int, set[ValueId]],
        block_reads: dict[int, list[tuple[ValueId, int]]],
        install_facts: dict[int, frozenset[ValueId]],
    ) -> dict[ValueId, set[ValueId]]:
        return compute_interference(
            BankLiveness(
                blocks=[b.id for b in mir.blocks],
                entry=mir.entry,
                succ=_succ_map(mir),
                makespan=block_makespan,
                term_offset=block_term_offset,
                resident=frozenset({*view.input_ids, *state_read_nodes}),
                op_landing={vid: bank.landing_cycle(commit) for vid, commit in op_commit.items()},
                op_block=op_block,
                phi_block=phi_block,
                reads=block_reads,
                boundary_users={b: frozenset(s) for b, s in boundary.items()},
                arm_out=arm_out,
                installs=install_facts,
                inflight_defs=block_inflight,
            )
        )

    # A slot whose live-in is consumed as ANOTHER slot's live-out (a chained copy, ``self.a = self.b``) must keep its
    # live-in to the boundary, so it can neither coalesce nor early-install. The coalescing oracle reads every live-out
    # at the boundary (it must persist) and every live-in at its actual last read, so a live-out that lands after its
    # live-in is fully read shows as non-interfering and coalesces -- the interference-frame form of the WAR test.
    livein_of = {slot.name: state_read_of.get(slot.name) for slot in slots}
    tapped_by_other: set[str] = set()
    for slot in slots:
        node = view.nodes[slot.live_out]  # the view holds only this bank's nodes, so a state-read here is this bank's
        read_name = node.name if isinstance(node, MirStateRead) else None
        if read_name is not None and read_name != slot.name:
            tapped_by_other.add(read_name)
    boundary_oracle = {b: set(s) for b, s in boundary_base.items()}
    for slot in slots:
        if slot.live_out in values:
            boundary_oracle[ret_block].add(slot.live_out)
    coalesce_graph = graph(boundary_oracle, reads, {})
    candidate_arms = _coalescable_arms(phi_nodes, values, bank.identity)
    phi_order = _movable_order(mir, list(phi_nodes), {}, phi_block, {})
    ret_present = block_makespan[ret_block] + 1
    # Last operand-read cycle of each value in the Ret block, from the shared liveness facts so read-cycle semantics
    # cannot drift; it bounds how early a slot may install over its source. Loop-invariant -- only an early install
    # reads it, and it is keyed only by state live-ins (always in ``values``), so the facts' value filter drops nothing.
    last_read_in_ret: dict[ValueId, int] = {}
    for vid, rc in reads[ret_block]:
        last_read_in_ret[vid] = max(last_read_in_ret.get(vid, 0), rc)
    obj_terms = bank.objective_terms(_ObjectiveContext(mir, view, values, op_nodes, phi_nodes, inst_of))
    slot_by_reg = {reg: name for name, reg in slot_reg.items()}

    # Slot-coalescing with validate-and-retry. Each eligible live-out is optimistically committed in place; if the
    # colorer then forces two interfering pinned values onto a slot register -- an in-place commit the install-free
    # oracle wrongly admitted -- the offending slot is backed out to a copy-back and the bank is recolored. Each round
    # forces at least one more slot to copy back, so it converges; the all-copy-back floor is always sound.
    forced_copy: set[str] = set()
    for _attempt in range(len(slots) + 1):
        coalesced: dict[str, ValueId] = {}  # slot name -> live-out, pinned onto the slot register (written in-place)
        for slot in slots:
            live_out = slot.live_out
            r_in = livein_of[slot.name]
            producible = live_out in op_nodes or live_out in phi_nodes
            if not bank.slot_identity(slot) or not producible or slot.name in tapped_by_other:
                continue  # a folded sideband, a non-producible live-out, or a chained copy cannot be written in-place
            if slot.name in forced_copy:
                continue  # demoted to copy-back by a prior retry round (its in-place commit was unsound)
            if r_in is not None and live_out in coalesce_graph.get(r_in, set()):
                continue  # the live-out's range overlaps the live-in's -- it must be copied, not coalesced
            coalesced[slot.name] = live_out

        install = bank.install_policy(
            _InstallContext(
                slots,
                coalesced,
                tapped_by_other,
                livein_of,
                op_nodes,
                op_commit,
                op_block,
                view.nodes,
                boundary_base[ret_block],
                last_read_in_ret,
                ret_block,
                ret_present,
            )
        )

        # Final interference. A non-coalesced slot reserves its live-in to the boundary (the install reads it read-first,
        # so the register holds nothing else); a coalesced slot keeps its live-in's actual range, so a gap tenant lands
        # between the live-in's last read and the live-out's landing. A boundary-installed live-out is read at the
        # boundary; an early-installed one is read by its copy at the install step, freeing its source for a later tenant.
        boundary_final = {b: set(s) for b, s in boundary_base.items()}
        early_reads: list[tuple[ValueId, int]] = []  # extra Ret-block reads from early-installed slot copies
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
                early_reads.append((live_out, copy_step_cycle(install[name])))
            else:
                boundary_final[ret_block].add(live_out)
        # Only an early install adds a Ret-block read; with none (always so for the boolean bank) the shared facts'
        # reads are reused as-is, so no per-attempt copy is made.
        if early_reads:
            reads_final = {b: list(r) for b, r in reads.items()}
            reads_final[ret_block].extend(early_reads)
        else:
            reads_final = reads

        pinned: dict[ValueId, int] = {vid: i for i, vid in enumerate(view.input_ids)}
        for name, reg in slot_reg.items():
            r_in = livein_of[name]
            if r_in is not None:
                pinned[r_in] = reg
        for name, live_out in coalesced.items():  # the coalesced live-out shares its slot register (written in place)
            pinned[live_out] = slot_reg[name]

        movable = _movable_order(
            mir, [vid for vid in (*op_nodes, *phi_nodes) if vid not in pinned], op_block, phi_block, op_commit
        )
        assign, nreg, coalescing, conflict = _coalesce_and_color(
            phi_nodes,
            phi_order,
            candidate_arms,
            pinned,
            {slot_reg[s.name] for s in slots if s.name not in coalesced},
            lambda residual: graph(boundary_final, reads_final, residual),
            _ColorObjective(movable, obj_terms.consumer_ports, obj_terms.producer_key, fresh_start),
        )
        if conflict is not None:
            demoted = slot_by_reg.get(conflict)
            assert (
                demoted is not None and demoted not in forced_copy
            ), f"coloring conflict on register {conflict} not resolvable by backing a slot out of coalescing"
            forced_copy.add(demoted)
            continue
        # Dwell hazard: a coalesced slot register written by an entry-block cycle-0 op would be re-driven each idle cycle
        # during the accept dwell, corrupting carried state. Demote the slot to a copy-back -- reserving its register so
        # no tenant lands there -- and retry; _assert_entry_dwell_safe backstops.
        entry_cycle0 = {vid for vid, c in block_sched[mir.entry].issue_cycle.items() if c == 0}
        dwell = {
            slot.name
            for slot in slots
            if slot.name in coalesced and slot_reg[slot.name] in {assign[v] for v in entry_cycle0 if v in assign}
        } - forced_copy
        if dwell:
            forced_copy |= dwell
            continue
        break
    else:  # pragma: no cover -- the all-copy-back floor is conflict-free, so the loop always breaks first
        assert False, f"{bank.label} slot-coalescing retry did not converge"

    # Backstop: a non-coalesced slot register must carry nothing but its own live-in. A coalesced slot register IS shared
    # by its in-place live-out (and any phi arms merged onto it) and is skipped here.
    for slot in slots:
        if slot.name in coalesced:
            continue
        reg = slot_reg[slot.name]
        occupants = [vid for vid, r in assign.items() if r == reg and vid != livein_of[slot.name]]
        assert (
            not occupants
        ), f"non-coalesced {bank.label} slot register {reg} ({slot.name!r}) has occupants {occupants}"
    return _BankAlloc(assign, slot_reg, nreg, install, coalescing.coalesced)


def _allocate(
    mir: Mir,
    float_mir: MirFloatView,
    bool_mir: MirBoolView,
    block_sched: dict[int, Schedule],
    inst_of: dict[ValueId, OperatorInstance],
    block_makespan: dict[int, int],
    block_term_offset: dict[int, int],
    block_inflight: dict[int, dict[ValueId, int]],
) -> _Allocation:
    """
    Assign wide and boolean registers across the CFG. Both banks are colored by hardware-frame liveness, reusing
    registers across mutually-exclusive and non-overlapping live ranges. A phi is resolved by installing each arm's
    value into the phi's register with a copy at the predecessor's tail; the copies are a parallel (simultaneous)
    bundle, so a swap is read-then-write correct. ``block_makespan`` (install-inclusive) and ``block_term_offset`` (the
    drained boundary, or the overlap-shrunk terminator) come from the overlap layout, so the liveness boundary matches
    the laid-out block spans exactly. ``block_inflight`` carries each block's received cross-block spills (split per
    bank), reserving a spilled value's register across every successor frame it lands in even where the value is
    dataflow-dead.
    """
    float_inflight = {
        bid: {vid: land for vid, land in spills.items() if vid in float_mir.operation_nodes}
        for bid, spills in block_inflight.items()
    }
    bool_inflight = {
        bid: {vid: land for vid, land in spills.items() if vid in bool_mir.operation_nodes}
        for bid, spills in block_inflight.items()
    }
    float_alloc = _allocate_bank(
        _WIDE, mir, float_mir, block_sched, inst_of, block_makespan, block_term_offset, float_inflight
    )
    bool_alloc = _allocate_bank(
        _BOOL, mir, bool_mir, block_sched, inst_of, block_makespan, block_term_offset, bool_inflight
    )

    # A phi arm coalesced onto the merged register needs no install copy: the arm value already resides in the phi's
    # register (they share a coloring class). Only the residual (non-coalesced) arms install by a pc-gated copy.
    copies: dict[int, list[_FloatArmInstall]] = {}
    for vid, phi in float_mir.phi_nodes.items():
        for pred, value, sign in phi.arms:
            assert isinstance(sign, FloatSignControl)
            if (pred, vid) in float_alloc.coalesced:
                continue
            copies.setdefault(pred, []).append(_FloatArmInstall(float_alloc.reg[vid], value, sign))

    bool_writes: dict[int, list[_BoolArmInstall]] = {}
    for vid, phi in bool_mir.phi_nodes.items():
        for pred, value, inversion in phi.arms:
            assert isinstance(inversion, BoolInversion)
            if (pred, vid) in bool_alloc.coalesced:
                continue
            bool_writes.setdefault(pred, []).append(_BoolArmInstall(bool_alloc.reg[vid], value, inversion))

    # A constant branch condition (e.g. a read-only boolean attribute, or a folded test) has no register of its own;
    # materialize it into a bool register written in the branching block so the next-PC decode can read it. The constant
    # is globally interned, so sibling branches sharing it reuse one register -- but the write must be emitted in EVERY
    # branching block that uses it, else a path reaching the branch through a block that did not write it reads a stale
    # register. (A later static-branch-folding pass would instead drop the dead arm; until then this keeps it correct.)
    bool_reg, nbreg = bool_alloc.reg, bool_alloc.nreg
    for block_id, cond in _const_branch_conditions(mir, bool_mir).items():
        if cond not in bool_reg:
            bool_reg[cond] = nbreg
            nbreg += 1
        bool_writes.setdefault(block_id, []).append(_BoolArmInstall(bool_reg[cond], cond, BoolInversion()))

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
            assert False, f"unhandled MIR input {vid}"
    return loads
