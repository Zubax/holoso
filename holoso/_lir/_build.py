import logging
from dataclasses import replace

from .._errors import UnsupportedConstruct
from .._mir import Mir, MirBoolView, MirBranch, MirPhi, MirStateRead, MirStateSlot, MirWideView
from .._operators import InlineHardwareOperator, PortConditioner
from .._type import FloatType, IntType
from .._util import ValueId
from .._value import FloatValue, IntValue, WideValue, coerce_scalar
from ._ir import *
from ._mir_facts import mir_operation, pred_count
from ._bankalloc import allocate, converge
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
    tapped_wide_lanes,
    wide_operand,
)
from ._layout import layout_blocks, schedule_blocks
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


def _drop_redundant_state_slots(mir: Mir) -> Mir:
    """
    Drop a state slot that is a redundant alias of another: same reset, live-out value id, and conditioner, so by
    induction always equal. The kept representative commits the value; each dropped attribute is write-only as state and
    its `state_<attr>` port already taps the shared value, so the duplicate register and its install copy vanish (e.g.
    a phase/frequency detector's public `up` aliasing its internal pending latch). A class is left intact when its
    live-out is a phi (a drop would perturb the phi-install placement) or when two members are read at entry (their
    distinct live-ins would need substitution).
    """
    read_names = {node.name for node in mir.nodes.values() if isinstance(node, MirStateRead)}

    classes: dict[tuple[float | int | bool, ValueId, PortConditioner], list[MirStateSlot]] = {}
    for slot in mir.state_slots:
        classes.setdefault((slot.reset_value, slot.live_out, slot.conditioner), []).append(slot)

    dropped: set[str] = set()
    for members in classes.values():
        if len(members) < 2:
            continue
        if isinstance(mir.nodes[members[0].live_out], MirPhi):
            continue  # rare: if-conversion usually elides the phi
        read_members = [m for m in members if m.name in read_names]
        if len(read_members) >= 2:
            continue
        rep = read_members[0] if read_members else members[0]
        assert all(m.name not in read_names for m in members if m is not rep), "a dropped alias must be write-only"
        dropped.update(m.name for m in members if m is not rep)

    if not dropped:
        return mir
    _logger.info("State slots: %d redundant aliases dropped: %s", len(dropped), sorted(dropped))
    return replace(mir, state_slots=[slot for slot in mir.state_slots if slot.name not in dropped])


def _prepare(mir: Mir, fetch_lag: int) -> BuildContext:
    mir = _drop_redundant_state_slots(mir)
    wide_mir = MirWideView.from_mir(mir)
    bool_mir = MirBoolView.from_mir(mir)
    # A branch whose condition is a phi with an arm FROM THE BRANCHING BLOCK cannot be sequenced: the arm's install
    # copy lands in the condition register exactly when the terminator reads it, so the branch would consult the next
    # iteration's value instead of the current one -- and no register assignment can help, since the conflict is the
    # value with itself. The frontend never emits this shape (every arm predecessor is jump-terminated); reject it
    # here so a future pass that creates branch-block arm predecessors fails loudly instead of miscompiling. A branch on
    # a constant never reaches here either: HIR pruning settles every decided branch.
    for mir_block in mir.blocks:
        terminator = mir_block.terminator
        if isinstance(terminator, MirBranch):
            assert (
                terminator.cond not in bool_mir.const_nodes
            ), f"block {mir_block.id} branches on a constant pruning missed"
            cond_node = mir.nodes.get(terminator.cond)
            if isinstance(cond_node, MirPhi) and any(pred == mir_block.id for pred, _, _ in cond_node.arms):
                raise UnsupportedConstruct(
                    f"block {mir_block.id} branches on a phi that takes an arm from the same block; the install "
                    f"would overwrite the condition before the branch reads it"
                )
    # The phi-residency premise -- a phi's register settles before any frame that reads it begins, so a phi source
    # is always settled (`install_source_commit`) -- rests on every phi-bearing block being multi-predecessor, which
    # is what keeps overlap spills out of phi registers; a future pass emitting a phi into a single-predecessor block
    # would silently void that argument (the sibling-install tripwire does not read-gate installs against in-flight
    # landings), so the reliance is machine-checked here. It is also what keeps the schedules independent of the
    # install set (see `schedule_blocks`).
    pred_edges = pred_count(mir)
    for mir_block in mir.blocks:
        assert (
            not mir_block.phis or pred_edges[mir_block.id] >= 2
        ), f"block {mir_block.id} carries phis with {pred_edges[mir_block.id]} predecessor edges"
    schedules = schedule_blocks(mir, wide_mir, bool_mir, fetch_lag)
    const_pool = build_const_pool(wide_mir, bool_mir.operation_nodes)
    return BuildContext(mir, wide_mir, bool_mir, fetch_lag, schedules, const_pool)


def _build_program(mir: Mir, module_name: str, fetch_lag: int, tuning: RegallocTuning) -> Lir:
    """
    Schedule each block, pool operator instances across the mutually-exclusive blocks, color both register banks by
    hardware-frame liveness (reusing registers, coalescing state live-outs, orienting commutative firings), install
    non-coalesced phi and slot live-outs by pc-gated copy, and lay the blocks out in the ROM, the transaction ending on
    every terminator arm that exits.
    """
    ctx = _prepare(mir, fetch_lag)
    mir, wide_mir, bool_mir = ctx.mir, ctx.wide_mir, ctx.bool_mir
    coalesced = converge(ctx)
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
            build_pooled_op(mir, wide_mir, bool_mir, members, sched, alloc, const_pool)
            for _, members in sorted(sched.firings.items(), key=lambda kv: (sched.issue_cycle[kv[0]], kv[0]))
        ]
        inline_ops = [
            build_inline_op(mir, wide_mir, bool_mir, vid, sched.issue_cycle[vid], alloc, const_pool)
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
                offsets.block_makespan[block.id],
                offsets.block_term_offset[block.id],
            )
        )

    laid_out = layout_blocks(mir, ctx.schedules, blocks)

    def encoded_reset(slot_name: str, scalar_type: FloatType | IntType, raw: float | int | bool) -> WideValue:
        value = coerce_scalar(scalar_type, raw, f"state slot {slot_name!r} reset")
        assert isinstance(value, (FloatValue, IntValue))
        return value

    wide_state_slots = [
        WideStateSlot(
            slot.name,
            RegRef(alloc.wide_slot_reg[slot.name]),
            encoded_reset(slot.name, wide_mir.scalar_type_of(slot.live_out), slot.reset_value),
            wide_operand(wide_mir, slot.live_out, slot.conditioner, alloc, const_pool),
            alloc.wide_install[slot.name],
        )
        for slot in wide_mir.state_slots
    ]
    bool_state_slots = [
        BoolStateSlot(
            slot.name,
            BoolRegRef(alloc.bool_slot_reg[slot.name]),
            bool(slot.reset_value),
            bool_operand(bool_mir, slot.live_out, alloc, slot.conditioner),
            alloc.bool_install[slot.name],
        )
        for slot in bool_mir.state_slots
    ]
    outputs = build_outputs(mir, wide_mir, bool_mir, alloc, const_pool)
    lir = Lir(
        module_name=module_name,
        instances=instances,
        wide_consts=consts,
        float_format=wide_mir.float_format,
        int_format=wide_mir.int_format,
        regfile=RegFileLayout(
            nreg=alloc.wide.nreg,
            nrd=sum(inst.operator.signature.arity for inst in instances),
            nwr=len(tapped_wide_lanes(laid_out)),
            nload=len(wide_mir.input_ids),
        ),
        inputs=build_inputs(mir, wide_mir, bool_mir, alloc),
        outputs=outputs,
        wide_state_slots=wide_state_slots,
        blocks=laid_out,
        bool_regfile=BoolRegFileLayout(nreg=alloc.bool.nreg),
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
