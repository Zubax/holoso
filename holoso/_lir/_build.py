import logging

from .._errors import UnsupportedConstruct
from .._mir import (
    Mir,
    MirBranch,
    MirConst,
    MirStateRead,
    predecessors,
    reverse_postorder,
    successors,
    thread_arm,
    threadable_arms,
)
from .._operators import InlineHardwareOperator
from .._type import BoolType, FloatType, IntType
from .._util import BlockId
from .._value import FloatValue, IntValue, WideValue, coerce_scalar
from ._ir import *
from ._mir_facts import mir_operation
from ._bankalloc import CoalescedLayout, allocate, converge
from ._build_base import BuildContext
from ._regalloc import RegallocTuning
from ._construct import (
    bool_operand,
    build_const_pool,
    build_inline_op,
    build_inputs,
    build_outputs,
    build_pooled_op,
    build_terminator,
    wide_operand,
    wide_type,
)
from ._layout import layout_blocks, pc_less, schedule_blocks
from ._sources import steering

_logger = logging.getLogger(__name__)


def build(mir: Mir, module_name: str, fetch_stages: int, tuning: RegallocTuning) -> Lir:
    """
    Schedule, bind, and register-allocate selected MIR into a pipelined microprogram. A straight-line kernel is the
    degenerate single-`Ret`-block control-flow graph, so there is one build path for every kernel. `fetch_stages`
    is the control-fetch pipeline depth; the datapath lags the fetch by one less than it, the lag threaded throughout.
    """
    if fetch_stages != 3:
        raise ValueError(f"only the 3-stage control fetch is implemented, got {fetch_stages}")
    if not mir.outputs:
        raise UnsupportedConstruct("Synthesized kernel must produce at least one output value")
    lir = _build_program(mir, module_name, fetch_stages - 1, tuning)
    names = [port.name for port in lir.ports]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise UnsupportedConstruct(f"duplicate port names in the module interface: {', '.join(duplicates)}")
    return lir


def _prepare(mir: Mir, fetch_lag: int) -> BuildContext:
    # HIR dead-code elimination drops every slot nothing reads, so a slot's register always opens on its live-in; the
    # allocator relies on that.
    read = {node.name for node in mir.nodes.values() if isinstance(node, MirStateRead)}
    assert all(slot.name in read for slot in mir.state_slots), "a state slot nothing reads reached the LIR"
    machine = {FloatType(mir.float_format), IntType(mir.int_format), BoolType()}
    assert all(node.scalar_type in machine for node in mir.nodes.values())
    # A branching block installs its phi arms at its tail, on every edge out of it, which is sound only where the merge
    # phis are dead on the other edges: across a forward edge, whose merge does not dominate the block, every use of
    # a merge phi lies past the merge. A back edge would overwrite a phi the loop still reads -- the branch condition
    # itself among them. The frontend's latches all jump, and arm threading threads forward only. A branch on a
    # constant never reaches here either: HIR pruning settles every decided branch.
    by_id = {block.id: block for block in mir.blocks}
    position = {bid: index for index, bid in enumerate(reverse_postorder(mir))}
    for mir_block in mir.blocks:
        if isinstance(mir_block.terminator, MirBranch):
            assert not isinstance(mir.nodes[mir_block.terminator.cond], MirConst), f"block {mir_block.id}"
            assert all(position[succ] > position[mir_block.id] for succ in successors(mir_block) if by_id[succ].phis)
    # The phi-residency premise -- a phi's register settles before any frame that reads it begins, so a phi source
    # is always settled (`install_source_commit`) -- rests on every phi-bearing block being multi-predecessor, which
    # is what keeps overlap spills out of phi registers; a future pass emitting a phi into a single-predecessor block
    # would silently void that argument (the copy-ordering check in `Lir._check_block` does not read-gate installs
    # against in-flight landings), so the reliance is machine-checked here. It is also what keeps the schedules
    # independent of the install set (see `schedule_blocks`).
    preds = predecessors(mir)
    assert all(len(preds[block.id]) >= 2 for block in mir.blocks if block.phis)
    return BuildContext(mir, fetch_lag, schedule_blocks(mir, fetch_lag), build_const_pool(mir))


def _pc_less(layout: CoalescedLayout) -> dict[BlockId, Arm]:
    ctx = layout.ctx
    working = {bid for bid, sched in ctx.schedules.block_sched.items() if sched.issue_cycle} | layout.copy_blocks()
    return pc_less(ctx.mir, ctx.schedules, working)


def _pc_spans(layout: CoalescedLayout) -> dict[BlockId, int]:
    """Each block's PC count in the program the layout lays out, zero for a block that takes none."""
    resolved, offsets = _pc_less(layout), layout.offsets.block_term_offset
    return {block.id: 0 if block.id in resolved else offsets[block.id] + 1 for block in layout.ctx.mir.blocks}


def _converge_threaded(mir: Mir, fetch_lag: int) -> CoalescedLayout:
    """
    The install fixpoint converged with every arm block threaded out whose threading shortens its own path and
    lengthens none. A path's length is the sum of its blocks' PC spans, so a threading is kept only while no remaining
    block's span grows; that also catches what threading costs elsewhere, a merge phi now occupying its register across
    the predecessor's boundary, where an arm of another path may no longer coalesce onto it. Scheduling and the
    fixpoint are cheap next to the coloring, which runs once on the result.
    """
    layout = converge(_prepare(mir, fetch_lag))
    spans = _pc_spans(layout)
    tried: set[BlockId] = set()
    while candidates := [arm for arm in threadable_arms(layout.ctx.mir) if arm not in tried and spans[arm]]:
        arm = candidates[0]
        tried.add(arm)
        trial = converge(_prepare(thread_arm(layout.ctx.mir, arm), fetch_lag))
        trial_spans = _pc_spans(trial)
        if grown := sorted(bid for bid, span in trial_spans.items() if span > spans[bid]):
            _logger.info("Threading: arm block %d kept, as threading it would grow blocks %s", arm, grown)
            continue
        _logger.info("Threading: arm block %d taken out, %d PCs off its path", arm, spans[arm])
        layout, spans = trial, trial_spans
    return layout


def _build_program(mir: Mir, module_name: str, fetch_lag: int, tuning: RegallocTuning) -> Lir:
    """
    Schedule each block, pool operator instances across the mutually-exclusive blocks, color both register banks by
    hardware-frame liveness (reusing registers, coalescing state live-outs, orienting commutative firings), install
    non-coalesced phi and slot live-outs by pc-gated copy, and lay the blocks out in the ROM, the transaction ending on
    every terminator arm that exits.
    """
    coalesced = _converge_threaded(mir, fetch_lag)
    ctx = coalesced.ctx
    mir = ctx.mir
    alloc = allocate(coalesced, tuning)
    offsets = coalesced.offsets
    block_sched = ctx.schedules.block_sched
    instances = alloc.instances
    consts, const_pool = ctx.const_pool.values, ctx.const_pool.entries

    blocks: list[LirBlock] = []
    for block in mir.blocks:
        sched = block_sched[block.id]
        # Operations split by operator class, not by result bank; each issues as soon as its own operands have landed,
        # with no barrier.
        ops = [
            build_pooled_op(mir, members, sched, alloc, const_pool)
            for _, members in sorted(sched.firings.items(), key=lambda kv: (sched.issue_cycle[kv[0]], kv[0]))
        ]
        inline_ops = [
            build_inline_op(mir, vid, sched.issue_cycle[vid], alloc, const_pool)
            for vid in sorted(
                (v for v in sched.issue_cycle if isinstance(mir_operation(mir, v).operator, InlineHardwareOperator)),
                key=lambda v: (sched.issue_cycle[v], v),
            )
        ]
        blocks.append(
            LirBlock(
                block.id,
                ops,
                inline_ops,
                alloc.copies.get(block.id, []),
                build_terminator(block.terminator, alloc),
                offsets.block_term_offset[block.id],
            )
        )

    assert {block.index for block in blocks if block.ops or block.inline_ops or block.copies} == {
        bid for bid, sched in block_sched.items() if sched.issue_cycle
    } | coalesced.copy_blocks()
    laid_out = layout_blocks(mir, blocks, _pc_less(coalesced))

    def encoded_reset(slot_name: str, scalar_type: FloatType | IntType, raw: float | int | bool) -> WideValue:
        value = coerce_scalar(scalar_type, raw, f"state slot {slot_name!r} reset")
        assert isinstance(value, (FloatValue, IntValue))
        return value

    wide_state_slots = [
        WideStateSlot(
            slot.name,
            RegRef(alloc.wide_slot_reg[slot.name]),
            encoded_reset(slot.name, wide_type(mir, slot.live_out), slot.reset_value),
            wide_operand(mir, slot.live_out, slot.conditioner, alloc, const_pool),
            alloc.wide_install[slot.name],
        )
        for slot in mir.state_slots
        if mir.nodes[slot.live_out].scalar_type.is_wide
    ]
    bool_state_slots = [
        BoolStateSlot(
            slot.name,
            BoolRegRef(alloc.bool_slot_reg[slot.name]),
            bool(slot.reset_value),
            bool_operand(mir, slot.live_out, alloc, slot.conditioner),
            alloc.bool_install[slot.name],
        )
        for slot in mir.state_slots
        if not mir.nodes[slot.live_out].scalar_type.is_wide
    ]
    outputs = build_outputs(mir, alloc, const_pool)
    lir = Lir(
        module_name=module_name,
        instances=instances,
        wide_consts=consts,
        float_format=mir.float_format,
        int_format=mir.int_format,
        regfile=RegFileLayout(nreg=alloc.wide.nreg),
        inputs=build_inputs(mir, alloc),
        outputs=outputs,
        wide_state_slots=wide_state_slots,
        blocks=laid_out,
        bool_regfile=RegFileLayout(nreg=alloc.bool.nreg),
        bool_state_slots=bool_state_slots,
        fetch_lag=fetch_lag,
    )
    # The wide bank was allocated against the steering the emitter builds, arm for arm.
    emitted = steering(lir)
    assert (alloc.wide.read_arms, alloc.wide.write_arms) == (emitted.read, emitted.wide_write)
    _logger.info(
        "LIR: %d blocks, last PC %d, exits %s, min II %d, %d instances",
        len(lir.blocks),
        lir.last_pc,
        lir.exit_pcs,
        lir.min_initiation_interval,
        len(instances),
    )
    return lir
