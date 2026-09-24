"""
Register-bank allocation: liveness facts, the wide and boolean bank policies, the layout and coalescing pass the
install fixpoint iterates, and the coloring of both banks' coalescing classes once it has converged.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import ClassVar, assert_never

from .._mir import (
    Mir,
    reverse_postorder,
    MirBoolView,
    MirBranch,
    MirOperation,
    MirPhi,
    MirStateRead,
    MirStateSlot,
    MirWideView,
)
from .._operators import (
    InlineHardwareOperator,
    PooledHardwareOperator,
    PortConditioner,
)
from .._util import ValueId
from ._ir import *
from ._mir_facts import block_has_install, mir_operation, phi_arm_out, succ_map
from ._liveness import BankLiveness, compute_interference
from ._schedule import Schedule
from ._regalloc import (
    Coloring,
    ColoringProblem,
    Firing,
    FixedProducer,
    InlineWriter,
    InputWriter,
    InstanceSlot,
    MoveWriter,
    RegallocTuning,
    SlotWriter,
    color,
)
from ._sources import BoolOperandTemplate, WideOperandTemplate
from ._build_base import (
    Allocation,
    BuildContext,
    BlockOffsets,
)
from ._construct import bool_operand_template, operand_templates, wide_operand_template
from ._coalesce import PhiCoalescing, coalescable_arms, coalesce, find_coloring_conflict
from ._layout import layout_offsets

_logger = logging.getLogger(__name__)


type _BankView = MirWideView | MirBoolView


class _Bank[D: _Placement | Boundary](ABC):
    """
    What differs between the two physical register banks; `prepare_bank`, `_coalesce_bank` and `_color_bank` are
    shared.
    """

    label: ClassVar[str]

    @abstractmethod
    def view(self, wide_mir: MirWideView, bool_mir: MirBoolView) -> _BankView: ...

    @abstractmethod
    def boundary_base(self, mir: Mir, values: set[ValueId], ret_block: int) -> dict[int, set[ValueId]]:
        """The non-slot boundary users: bank outputs, plus the per-block branch conditions for the boolean bank."""

    @abstractmethod
    def fixed_producers(
        self, layout: CoalescedLayout, coalesced: _CoalescedBank[D], bool_reg: Mapping[ValueId, int]
    ) -> dict[ValueId, list[FixedProducer]]:
        """
        Each value's writers beside the pooled lanes, as the objective prices them: the wide bank's input loads,
        inline results and residual arms, the boolean bank's residual arms only (a one-bit mux is not priced).
        `bool_reg` is the boolean bank's assignment, decided first, so the wide bank's inline results name the
        boolean registers they read (empty while the boolean bank itself is colored).
        """

    @abstractmethod
    def firings(self, layout: CoalescedLayout, coalesced: _CoalescedBank[D]) -> list[Firing]:
        """The pooled firings the objective prices: every one for the wide bank, none for the boolean bank."""

    @abstractmethod
    def install_policy(self, facts: _BankFacts[D], coalesced: dict[str, ValueId]) -> dict[str, D]:
        """
        Where each slot not committed in place installs its live-out, for one attempt whose in-place set is
        `coalesced` (slot name -> live-out). The wide bank installs early where it can to free the source register;
        the boolean bank installs every live-out at the boundary.
        """


@dataclass(frozen=True, slots=True)
class _Placement:
    """
    Where a copy fires in its block's frame -- a phi arm's tail install in its predecessor's, a slot's early install in
    the Ret block's: the scheduler-frame issue cycle, and whether the source is settled before the block's first step
    (a constant, an input, a state read, a phi, or a foreign result already landed), which needs no read-first
    sampling. Stamped once per install, the placement the interference residence, the install classification and the
    emitted copy all read.
    """

    issue: int
    settled: bool


@dataclass(frozen=True, slots=True)
class _BankFacts[D: _Placement | Boundary]:
    """
    One physical register bank's facts that no install-fixpoint round changes, computed once per build. The
    read-cycle semantics live here once, so the rounds cannot drift from them.
    """

    bank: _Bank[D]
    view: _BankView
    values: set[ValueId]
    op_nodes: Mapping[ValueId, MirOperation]
    phi_nodes: Mapping[ValueId, MirPhi]
    slots: Sequence[MirStateSlot]
    slot_reg: dict[str, int]
    fresh_start: int
    livein_of: dict[str, ValueId | None]
    tapped_by_other: set[str]  # slots whose live-in another slot's live-out reads (a chained copy)
    # The Ret-block copy placement each early-install candidate source is ready at. A candidate is a result scheduled in
    # the Ret block or an input; availability alone does not make an early copy cheaper, so the domain stays that
    # conservative.
    ret_install_ready: dict[ValueId, _Placement]
    reads: dict[int, list[tuple[ValueId, int]]]
    boundary_base: dict[int, set[ValueId]]
    liveness: BankLiveness  # the prototype: what no attempt changes, the rest left empty
    candidate_arms: dict[ValueId, list[tuple[int, ValueId]]]
    placement: dict[tuple[int, ValueId], _Placement]  # (pred, phi) -> where the arm's install fires
    order: list[ValueId]  # every operation and phi in the deterministic coloring order
    last_read_in_ret: dict[ValueId, int]
    ret_block: int
    ret_makespan: int | None  # None when the Ret block schedules nothing, leaving no drain to land an early install in
    fetch_lag: int

    def interference(
        self,
        term_offset: dict[int, int],
        boundary: dict[int, set[ValueId]],
        reads: dict[int, list[tuple[ValueId, int]]],
        residual: dict[int, frozenset[ValueId]],
    ) -> dict[ValueId, set[ValueId]]:
        """
        The interference graph of one coalescing attempt over the prototype. A residual install fires where its
        placement says, and MAY transiently land past the current terminator: after a push-bit narrowing, an
        intermediate attempt can hold a computed arm de-coalesced whose install no longer fits the shortened boundary
        -- the outer install fixpoint then re-widens (pins) the classification and re-runs, and only the converged
        round's placements are emitted.
        """
        installs = {
            block: {vid: inline_fire_cycle(self.placement[(block, vid)].issue, self.fetch_lag) for vid in vids}
            for block, vids in residual.items()
        }
        return compute_interference(
            replace(
                self.liveness,
                term_offset=term_offset,
                reads=reads,
                boundary_users={b: frozenset(users) for b, users in boundary.items()},
                installs=installs,
            )
        )


@dataclass(frozen=True, slots=True)
class _CoalescedBank[D: _Placement | Boundary]:
    """
    One bank coalesced and pinned for one round but not colored. Nothing here depends on the register a movable value
    takes.
    """

    facts: _BankFacts[D]
    install: dict[str, InPlace | D]
    coalescing: PhiCoalescing
    pinned: dict[ValueId, int]
    reserved: set[int]  # the non-coalesced slot registers, which no movable value may join
    interferes: dict[ValueId, set[ValueId]]
    movable: list[ValueId]

    def residual_arms(self) -> list[tuple[int, ValueId, ValueId, PortConditioner]]:
        """Every `(pred, phi, arm value, conditioner)` not coalesced, so installing by a copy at `pred`'s tail."""
        return [
            (pred, vid, value, conditioner)
            for vid, phi in self.facts.phi_nodes.items()
            for pred, value, conditioner in phi.arms
            if (pred, vid) not in self.coalescing.coalesced
        ]


@dataclass(frozen=True, slots=True)
class CoalescedLayout:
    """
    One install-fixpoint round: the terminator offsets its install classification implies, and both banks coalesced
    and pinned on them but not colored. Nothing it reads depends on the coloring, which `allocate` runs once on the
    converged round.
    """

    ctx: BuildContext
    offsets: BlockOffsets
    bool_bank: _CoalescedBank[Boundary]
    wide_bank: _CoalescedBank[_Placement | Boundary]

    def install_blocks(self) -> dict[int, bool]:
        """
        The post-coalescing install classification, refining `block_has_install`'s conservative one: each block that
        installs a residual phi arm at its tail mapped to whether any of its installs issues past the work makespan.
        """
        block_sched = self.ctx.schedules.block_sched
        install: dict[int, bool] = {}
        for bank in (self.wide_bank, self.bool_bank):
            for pred, vid, _value, _conditioner in bank.residual_arms():
                pushes = bank.facts.placement[(pred, vid)].issue > block_sched[pred].makespan
                install[pred] = install.get(pred, False) or pushes
        return install


class _WideBank(_Bank[_Placement | Boundary]):
    label = "wide"

    def view(self, wide_mir: MirWideView, bool_mir: MirBoolView) -> MirWideView:
        return wide_mir

    def boundary_base(self, mir: Mir, values: set[ValueId], ret_block: int) -> dict[int, set[ValueId]]:
        boundary: dict[int, set[ValueId]] = {block.id: set() for block in mir.blocks}
        for out in mir.outputs:
            if out.value in values:
                boundary[ret_block].add(out.value)
        return boundary

    def fixed_producers(
        self, layout: CoalescedLayout, coalesced: _CoalescedBank[_Placement | Boundary], bool_reg: Mapping[ValueId, int]
    ) -> dict[ValueId, list[FixedProducer]]:
        ctx, facts = layout.ctx, coalesced.facts
        producers: dict[ValueId, list[FixedProducer]] = {vid: [] for vid in facts.values}
        producers.update({vid: [InputWriter(vid)] for vid in ctx.wide_mir.input_ids})
        for vid, node in facts.op_nodes.items():
            if isinstance(node.operator, PooledHardwareOperator):
                continue
            assert isinstance(node.operator, InlineHardwareOperator)
            operands = tuple(
                t.resolve(bool_reg.__getitem__) if isinstance(t, BoolOperandTemplate) else t
                for t in operand_templates(node, ctx.wide_mir, ctx.bool_mir, ctx.const_pool.entries)
            )
            producers[vid] = [InlineWriter(node.operator, operands, node.output_conditioner)]
        for _pred, vid, value, conditioner in coalesced.residual_arms():
            template = wide_operand_template(ctx.wide_mir, value, conditioner, ctx.const_pool.entries)
            producers[vid].append(MoveWriter(template))
        return producers

    def firings(self, layout: CoalescedLayout, coalesced: _CoalescedBank[_Placement | Boundary]) -> list[Firing]:
        # Every pooled firing reads wide operands, so every one is listed, a comparator whose taps are all boolean
        # included. A firing is bindable when its class has more than one realized instance and its busy window ends
        # inside the block (`issue + II <= term_offset + 1`, the scheduler's carry condition), so it leaves no residue
        # a successor could inherit.
        ctx, facts = layout.ctx, coalesced.facts
        values, op_nodes = facts.values, facts.op_nodes
        block_sched = ctx.schedules.block_sched
        firings: list[Firing] = []
        for bid in sorted(block_sched):
            sched = block_sched[bid]
            for leader in sorted(sched.firings, key=lambda vid: (sched.issue_cycle[vid], vid)):
                node = mir_operation(ctx.mir, leader)
                operator = node.operator
                assert isinstance(operator, PooledHardwareOperator)
                issue = sched.issue_cycle[leader]
                bindable = (
                    ctx.schedules.instances[operator] > 1
                    and issue + operator.initiation_interval <= layout.offsets.block_term_offset[bid] + 1
                )
                reads: list[ValueId | WideConstRef] = []
                for template in operand_templates(node, ctx.wide_mir, ctx.bool_mir, ctx.const_pool.entries):
                    assert isinstance(template, WideOperandTemplate) and (
                        template.hole is None or template.hole in values
                    )
                    reads.append(template.source)
                writes = [(op_nodes[m].output_port, m) for m in sched.firings[leader] if m in values]
                seed = sched.inst_of[leader].index
                firings.append(Firing(leader, operator, bid, issue, seed, bindable, reads, writes))
        return firings

    def install_policy(
        self, facts: _BankFacts[_Placement | Boundary], coalesced: dict[str, ValueId]
    ) -> dict[str, _Placement | Boundary]:
        # Install the live-out as early as the live-in is fully read and the source is available, freeing the source
        # register -- but only from an eligible source (see `ret_install_ready`), where the live-in is not itself a
        # boundary user, and the slot neither coalesced nor feeds a chained copy. Otherwise the boundary.
        install: dict[str, _Placement | Boundary] = {}
        boundary_ret = facts.boundary_base[facts.ret_block]
        for slot in facts.slots:
            name, live_out, r_in = slot.name, slot.live_out, facts.livein_of[slot.name]
            ready = facts.ret_install_ready.get(live_out)
            if ready is None:
                install[name] = Boundary()
                continue
            cycle = ready.issue
            if r_in is not None:
                cycle = max(cycle, facts.last_read_in_ret.get(r_in, 0) - inline_fire_cycle(0, facts.fetch_lag))
            early = (
                facts.ret_makespan is not None
                and cycle <= facts.ret_makespan
                and name not in coalesced
                and name not in facts.tapped_by_other
                and (r_in is None or r_in not in boundary_ret)
            )
            install[name] = _Placement(cycle, ready.settled) if early else Boundary()
        return install


class _BoolBank(_Bank[Boundary]):
    label = "bool"

    def view(self, wide_mir: MirWideView, bool_mir: MirBoolView) -> MirBoolView:
        return bool_mir

    def boundary_base(self, mir: Mir, values: set[ValueId], ret_block: int) -> dict[int, set[ValueId]]:
        boundary: dict[int, set[ValueId]] = {block.id: set() for block in mir.blocks}
        for block in mir.blocks:
            if isinstance(block.terminator, MirBranch):
                assert block.terminator.cond in values
                boundary[block.id].add(block.terminator.cond)
        for out in mir.outputs:
            if out.value in values:
                boundary[ret_block].add(out.value)
        return boundary

    def fixed_producers(
        self, layout: CoalescedLayout, coalesced: _CoalescedBank[Boundary], bool_reg: Mapping[ValueId, int]
    ) -> dict[ValueId, list[FixedProducer]]:
        # One-bit muxes are not priced: no firings (a comparator's lanes), no inline results, no input loads. The
        # objective is the register count plus the residual phi arms and the slot installs the bank-independent code
        # adds.
        producers: dict[ValueId, list[FixedProducer]] = {vid: [] for vid in coalesced.facts.values}
        for _pred, vid, value, inversion in coalesced.residual_arms():
            producers[vid].append(MoveWriter(bool_operand_template(layout.ctx.bool_mir, value, inversion)))
        return producers

    def firings(self, layout: CoalescedLayout, coalesced: _CoalescedBank[Boundary]) -> list[Firing]:
        return []

    def install_policy(self, facts: _BankFacts[Boundary], coalesced: dict[str, ValueId]) -> dict[str, Boundary]:
        return {slot.name: Boundary() for slot in facts.slots}


_WIDE = _WideBank()

_BOOL = _BoolBank()


def install_source_commit(
    sched: Schedule, source: ValueId, inflight: Mapping[ValueId, int], fetch_lag: int
) -> int | None:
    """
    A tail install source's commit cycle in the installing block's own frame, None for a source SETTLED before the
    block's first step -- exactly `install_issue_cycle`'s contract. A source scheduled in this block commits at its
    own commit cycle. One spilled in past an overlapped predecessor boundary (`inflight`, the scheduler's own
    `livein_landing`) is charged the virtual commit whose landing is its carried landing cycle, so the install's fire
    step cannot precede that landing; the spill bound (a received spill lands at successor-local <= fetch_lag,
    enforced by the issue-side envelope's word floors) keeps that virtual commit negative, so a spilled source never
    pushes the install past the work makespan -- asserted, so a future envelope change that breaks the bound fails
    loudly instead of firing an install before its source lands. Everything else -- a constant, an input, a state
    read, a phi, or a foreign result already landed -- is settled at entry.
    """
    if source in sched.issue_cycle:
        return sched.commit_cycle(source)
    landing = inflight.get(source)
    if landing is not None:
        commit = landing - fetch_lag - READ_FIRST_EDGE
        assert commit < 0, f"received spill lands at {landing}, past the fetch lag: the spill bound is broken"
        return commit
    return None


@dataclass(frozen=True, slots=True)
class _BankLivenessFacts:
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
    fetch_lag: int,
) -> _BankLivenessFacts:
    """
    Identical for both banks so their read-cycle semantics cannot drift. The operand reads are EXACT per consumer, each
    on its own step; the caller adds its bank's boundary users.
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
            rc = operand_read_cycle(node.operator, issue, fetch_lag)
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
    The `-3` sentinel sorts a phi -- which has no commit -- ahead of the operations in its block. Value-id last keeps
    the order, and hence the coloring, seed-independent; both banks MUST use this one definition.
    """
    rpo_pos = {bid: i for i, bid in enumerate(reverse_postorder(mir))}
    block_of = {**op_block, **phi_block}
    return sorted(candidates, key=lambda vid: (rpo_pos[block_of[vid]], op_commit.get(vid, -3), vid))


def prepare_bank[D: _Placement | Boundary](bank: _Bank[D], ctx: BuildContext) -> _BankFacts[D]:
    """
    One bank's facts for the whole build (see `_BankFacts`): everything coalescing it needs except the round's
    terminator offsets, which enter through the interference builder.
    """
    mir, wide_mir, bool_mir, fetch_lag = ctx.mir, ctx.wide_mir, ctx.bool_mir, ctx.fetch_lag
    block_sched = ctx.schedules.block_sched
    view = bank.view(wide_mir, bool_mir)
    nload = len(view.input_ids)
    slots: Sequence[MirStateSlot] = view.state_slots
    slot_reg = {slot.name: nload + i for i, slot in enumerate(slots)}
    op_nodes = view.operation_nodes
    phi_nodes = view.phi_nodes
    state_read_nodes = view.state_read_nodes
    state_read_of = {node.name: vid for vid, node in state_read_nodes.items()}
    values = {*view.input_ids, *state_read_nodes, *op_nodes, *phi_nodes}
    facts = _bank_liveness_facts(mir, block_sched, op_nodes, phi_nodes, values, fetch_lag)
    # The spills this bank receives, reserving a spilled value's register across every successor frame it lands in
    # even where the value is dataflow-dead.
    inflight = {
        bid: {vid: land for vid, land in spills.items() if vid in op_nodes}
        for bid, spills in ctx.schedules.block_inflight.items()
    }
    ret_block = mir.ret_block
    # Each phi arm's install is placed in the PREDECESSOR's own frame (where the install fires, not the source's home
    # block).
    placement: dict[tuple[int, ValueId], _Placement] = {}
    for vid, phi in phi_nodes.items():
        for pred, value, _conditioner in phi.arms:
            commit = install_source_commit(block_sched[pred], value, inflight[pred], fetch_lag)
            issue = install_issue_cycle(block_sched[pred].makespan, commit)
            placement[(pred, vid)] = _Placement(issue, commit is None)
    liveness = BankLiveness(
        blocks=[b.id for b in mir.blocks],
        entry=mir.entry,
        succ=succ_map(mir),
        term_offset={},
        resident=frozenset({*view.input_ids, *state_read_nodes}),
        # Every result -- pooled or inline, wide or boolean -- lands at the one bank-independent landing.
        op_landing={vid: landing_cycle(commit, fetch_lag) for vid, commit in facts.op_commit.items()},
        op_block=facts.op_block,
        phi_block=facts.phi_block,
        reads={},
        boundary_users={},
        arm_out=phi_arm_out(mir, phi_nodes, values),
        installs={},
        inflight_defs=inflight,
    )
    # A slot whose live-in is consumed as ANOTHER slot's live-out (a chained copy, `self.a = self.b`) must keep its
    # live-in to the boundary, so it can neither coalesce nor early-install.
    livein_of = {slot.name: state_read_of.get(slot.name) for slot in slots}
    tapped_by_other: set[str] = set()
    for slot in slots:
        node = view.nodes[slot.live_out]  # the view holds only this bank's nodes, so a state-read here is this bank's
        read_name = node.name if isinstance(node, MirStateRead) else None
        if read_name is not None and read_name != slot.name:
            tapped_by_other.add(read_name)
    # Bounds how early a slot may install over its source. Keyed only by state live-ins (always in `values`), so the
    # facts' value filter drops nothing.
    last_read_in_ret: dict[ValueId, int] = {}
    for vid, rc in facts.reads[ret_block]:
        last_read_in_ret[vid] = max(last_read_in_ret.get(vid, 0), rc)
    ret_install_ready: dict[ValueId, _Placement] = {}
    for slot in slots:
        if slot.live_out in block_sched[ret_block].issue_cycle or slot.live_out in view.input_ids:
            commit = install_source_commit(block_sched[ret_block], slot.live_out, inflight[ret_block], fetch_lag)
            ret_install_ready[slot.live_out] = _Placement(install_ready_cycle(commit), commit is None)
    return _BankFacts(
        bank=bank,
        view=view,
        values=values,
        op_nodes=op_nodes,
        phi_nodes=phi_nodes,
        slots=slots,
        slot_reg=slot_reg,
        fresh_start=nload + len(slots),
        livein_of=livein_of,
        tapped_by_other=tapped_by_other,
        ret_install_ready=ret_install_ready,
        reads=facts.reads,
        boundary_base=bank.boundary_base(mir, values, ret_block),
        liveness=liveness,
        candidate_arms=coalescable_arms(phi_nodes, values),
        placement=placement,
        order=_movable_order(mir, [*op_nodes, *phi_nodes], facts.op_block, facts.phi_block, facts.op_commit),
        last_read_in_ret=last_read_in_ret,
        ret_block=ret_block,
        ret_makespan=block_sched[ret_block].makespan if block_sched[ret_block].issue_cycle else None,
        fetch_lag=fetch_lag,
    )


def converge(ctx: BuildContext) -> CoalescedLayout:
    """
    Lay out and coalesce to the install fixpoint over the once-computed schedules. The rounds decide the draining
    blocks' terminator offsets, the coalescing, the pins and the reserved slot registers, none of which depends on
    the coloring; the schedules, the overlap carry and the instance counts depend on none of the rounds (see
    `schedule_blocks`).
    """
    mir, wide_mir, bool_mir = ctx.mir, ctx.wide_mir, ctx.bool_mir
    wide_facts, bool_facts = prepare_bank(_WIDE, ctx), prepare_bank(_BOOL, ctx)
    # The install set is computed to a fixpoint. `block_has_install` marks a block install-bearing from the CFG shape
    # (any phi arm originates in it), but a block whose every arm COALESCES onto the merged register installs nothing,
    # so that +1 drain is spurious. So: lay out and coalesce with the conservative CFG seed, recompute the
    # classification from the ACTUAL coalesced copies, and re-run to a fixed point. The movement is TWO-SIDED (a
    # classification mostly narrows, but can have to grow back; see the loop body), yet every regrowth is pinned, so
    # each block moves a bounded number of times and the composition with the inner per-bank `coalesce` fixpoint (its
    # forbidden-merge set non-decreasing) terminates.
    has_install_blocks = block_has_install(mir, wide_mir, bool_mir)
    # Iteration bound. Every non-final round makes one of the bounded monotone moves per block -- a push-bit narrowing,
    # an install-set key removal, or a pin (a pinned block never moves again), at most three over the run. Worst case
    # is their SUM plus a confirming round; the 3*len(blocks)+4 loop bound below leaves a safe margin, with the
    # trailing raise as the loud backstop.
    seed_keys = frozenset(has_install_blocks)
    pinned_push: set[int] = set()
    for round_index in range(3 * len(mir.blocks) + 4):
        offsets = layout_offsets(mir, ctx.schedules, has_install_blocks, ctx.fetch_lag)
        _logger.info(
            "Layout: terminator offsets %s", [offsets.block_term_offset[b] for b in sorted(offsets.block_term_offset)]
        )
        layout = CoalescedLayout(ctx, offsets, _coalesce_bank(bool_facts, offsets), _coalesce_bank(wide_facts, offsets))
        raw = layout.install_blocks()
        # The two derivations of install-bearing -- the CFG-shape seed and the post-coalescing copies -- must agree on
        # the key universe: a block outside the seed can never install, so a wider `raw` means the derivations
        # drifted, which must fail loudly rather than be absorbed as a silent permanent pin. On the FIRST round the
        # bits must agree too: nothing has narrowed yet, so a push the conservative seed missed is the same drift,
        # not legitimate regrowth.
        assert raw.keys() <= seed_keys, "post-coalescing installs appeared outside the CFG-shape seed"
        assert round_index > 0 or all(
            has_install_blocks[b] or not bit for b, bit in raw.items()
        ), "a first-round push classification exceeded the conservative CFG-shape seed"
        # A narrowed classification may have to GROW BACK: the shortened boundary feeds the next round's coalescing,
        # whose greedy merge order is not monotone in the interference, so a computed arm that coalesced under the
        # longer boundary can come back residual -- its install then needs the +1 drain the narrowing removed, and a
        # whole dropped KEY can likewise resurface. Any regrowth is PINNED: a pinned block stays install-bearing with a
        # forced +1 drain for the rest of the run, so the two-sided movement still converges (key removals and pins are
        # each monotone, bounded by the block count). An intermediate layout built on a stale narrower boundary is
        # discarded by the re-run; the converged round has validated every surviving install against a boundary
        # consistent with its own classification.
        regrown = {b for b, bit in raw.items() if b not in has_install_blocks or (bit and not has_install_blocks[b])}
        if regrown - pinned_push:
            _logger.info("Install fixpoint round %d: pinning regrown blocks %s", round_index, sorted(regrown))
        pinned_push |= regrown
        actual = raw | dict.fromkeys(pinned_push, True)
        # The moves are monotone and the fixed point is sound. MONOTONE: keys only shrink except through the growing
        # pin set (a round moves only the draining blocks' terminator offsets and the coalescing; the schedules sit
        # outside the loop). SOUND: a converged classification (actual == has_install_blocks) is self-consistent, and
        # `landing <= term_offset` then holds by the drain math, so the layout is correct even under -O; every
        # widening is legitimized by the unconditional pin merge above (an assert here would be tautological), and a
        # never-converging run -- unreachable -- falls through to the trailing raise.
        if actual == has_install_blocks:
            _logger.info(
                "Install fixpoint converged after %d rounds: %d install-bearing blocks, %d pinned",
                round_index + 1,
                len(actual),
                len(pinned_push),
            )
            return layout
        has_install_blocks = actual
    raise AssertionError("coalesced-install fixpoint did not converge")  # survives -O (unlike a bare assert)


def _coalesce_bank[D: _Placement | Boundary](facts: _BankFacts[D], offsets: BlockOffsets) -> _CoalescedBank[D]:
    """
    Coalesce one physical register bank across the CFG on one round's terminator offsets: phi-arm coalescing, slot
    in-place commits with validate-and-retry, pins and install decisions. The coloring is `_color_bank`'s, once the
    layout has converged.
    """
    bank, slots, values = facts.bank, facts.slots, facts.values
    op_nodes, phi_nodes = facts.op_nodes, facts.phi_nodes
    livein_of, slot_reg, ret_block, fetch_lag = facts.livein_of, facts.slot_reg, facts.ret_block, facts.fetch_lag
    term_offset = offsets.block_term_offset
    assert facts.ret_makespan is None or offsets.block_makespan[ret_block] == facts.ret_makespan
    phi_order = [vid for vid in facts.order if vid in phi_nodes]
    # The coalescing oracle reads every live-out at the boundary (it must persist) and every live-in at its actual
    # last read, so a live-out that lands after its live-in is fully read shows as non-interfering and coalesces --
    # the interference-frame form of the WAR test.
    boundary_oracle = {b: set(s) for b, s in facts.boundary_base.items()}
    for slot in slots:
        if slot.live_out in values:
            boundary_oracle[ret_block].add(slot.live_out)
    coalesce_graph = facts.interference(term_offset, boundary_oracle, facts.reads, {})
    slot_by_reg = {reg: name for name, reg in slot_reg.items()}

    # Slot-coalescing with validate-and-retry. Each eligible live-out is optimistically committed in place; if two
    # interfering pinned values then land on a slot register -- an in-place commit the install-free oracle wrongly
    # admitted -- the offending slot is backed out to a copy-back and the bank is re-coalesced. Each round forces at
    # least one more slot to copy back, so it converges; the all-copy-back floor is always sound.
    forced_copy: set[str] = set()
    for _attempt in range(len(slots) + 1):
        coalesced: dict[str, ValueId] = {}  # slot name -> live-out, pinned onto the slot register (written in-place)
        for slot in slots:
            live_out = slot.live_out
            r_in = livein_of[slot.name]
            producible = live_out in op_nodes or live_out in phi_nodes
            if not slot.conditioner.is_identity or not producible or slot.name in facts.tapped_by_other:
                continue  # a folded sideband, a non-producible live-out, or a chained copy cannot be written in-place
            if slot.name in forced_copy:
                continue
            if r_in is not None and live_out in coalesce_graph.get(r_in, set()):
                continue
            coalesced[slot.name] = live_out

        install = bank.install_policy(facts, coalesced)

        # Final interference. A non-coalesced slot reserves its live-in to the boundary (the install reads it
        # read-first, so the register holds nothing else); a coalesced slot keeps its live-in's actual range, so a gap
        # tenant lands between the live-in's last read and the live-out's landing. A boundary-installed live-out is read
        # at the boundary; an early-installed one is read by its copy at the install step, freeing its source for a
        # later tenant.
        boundary_final = {b: set(s) for b, s in facts.boundary_base.items()}
        early_reads: list[tuple[ValueId, int]] = []
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
            match install[name]:
                case _Placement(issue=issue):
                    early_reads.append((live_out, inline_fire_cycle(issue, fetch_lag)))
                case Boundary():
                    boundary_final[ret_block].add(live_out)
                case _:
                    assert_never(install[name])
        if early_reads:
            reads_final = {b: list(r) for b, r in facts.reads.items()}
            reads_final[ret_block].extend(early_reads)
        else:
            reads_final = facts.reads

        pinned: dict[ValueId, int] = {vid: i for i, vid in enumerate(facts.view.input_ids)}
        for name, reg in slot_reg.items():
            r_in = livein_of[name]
            if r_in is not None:
                pinned[r_in] = reg
        # A coalesced live-out shares its slot register (written in place). Two slots ending on one value: the last
        # holds it, and the others copy from its register at the boundary; such a slot's own register holds only
        # its live-in. Reserved are the slot registers not written in place and those nothing occupies.
        for name, live_out in coalesced.items():
            pinned[live_out] = slot_reg[name]
        reserved = {slot_reg[s.name] for s in slots if s.name not in coalesced}
        reserved |= {reg for reg in slot_reg.values() if reg not in pinned.values()}
        coalescing, interferes, conflict = coalesce(
            phi_nodes,
            phi_order,
            facts.candidate_arms,
            pinned,
            reserved,
            lambda residual: facts.interference(term_offset, boundary_final, reads_final, residual),
        )
        if conflict is not None:
            demoted = slot_by_reg.get(conflict)
            assert (
                demoted is not None and demoted not in forced_copy
            ), f"coloring conflict on register {conflict} not resolvable by backing a slot out of coalescing"
            forced_copy.add(demoted)
            continue
        break
    else:  # pragma: no cover -- the all-copy-back floor is conflict-free, so the loop always breaks first
        assert False, f"{bank.label} slot-coalescing retry did not converge"
    _logger.info(
        "%s bank: %d of %d phi arms coalesced, %d of %d slots in place, %d demoted to copy-back",
        bank.label,
        len(coalescing.coalesced),
        sum(len(arms) for arms in facts.candidate_arms.values()),
        len(coalesced),
        len(slots),
        len(forced_copy),
    )

    movable = [vid for vid in facts.order if vid not in pinned]
    # A slot is in place when its live-out ended up PINNED to the slot register under the identity -- decided by the
    # pins, not by the attempt set: an unwritten slot is never attempted (its live-out is its own live-in) yet its
    # pin puts it in place, and a shared live-out is attempted by every slot ending on it yet pinned to one.
    decisions: dict[str, InPlace | D] = {
        slot.name: (
            InPlace()
            if slot.conditioner.is_identity and pinned.get(slot.live_out) == slot_reg[slot.name]
            else install[slot.name]
        )
        for slot in slots
    }
    return _CoalescedBank(facts, decisions, coalescing, pinned, reserved, interferes, movable)


def _color_quotient[D: _Placement | Boundary](
    coalesced: _CoalescedBank[D],
    producers: dict[ValueId, list[FixedProducer]],
    firings: list[Firing],
    layout: CoalescedLayout,
    tuning: RegallocTuning,
) -> Coloring:
    """
    Color the per-value interference graph after collapsing each coalescing class to its leader, then expand the
    leader's color back onto every member. The quotient maps every firing's read and write endpoints and every
    producer's holes to leaders and concatenates each class's fixed producers, so the steering objective is the
    merged register's actual read and write fan-in. A class with a pinned member pins its leader.
    """
    coalescing, interferes = coalesced.coalescing, coalesced.interferes

    lead = coalescing.lead

    leaders = sorted({lead(v) for v in interferes})
    q_pinned = coalescing.pinned
    q_interferes: dict[ValueId, set[ValueId]] = {head: set() for head in leaders}
    q_producers: dict[ValueId, list[FixedProducer]] = {head: [] for head in leaders}
    for vid in sorted(interferes):
        head = lead(vid)
        q_producers[head] += [producer.substitute(lead) for producer in producers[vid]]
        for other in interferes[vid]:
            head_other = lead(other)
            if head_other != head:
                q_interferes[head].add(head_other)
                q_interferes[head_other].add(head)
    # Two members' identical results, or two arms moving one operand, are one writer of the merged register.
    q_producers = {head: list(dict.fromkeys(ps)) for head, ps in q_producers.items()}
    q_movable: list[ValueId] = []
    seen: set[ValueId] = set()
    for vid in coalesced.movable:
        head = lead(vid)
        if head in q_pinned or head in seen:
            continue
        seen.add(head)
        q_movable.append(head)
    q_firings = [
        replace(
            firing,
            reads=[source if isinstance(source, WideConstRef) else lead(source) for source in firing.reads],
            writes=[(port, lead(value)) for port, value in firing.writes],
        )
        for firing in firings
    ]
    entry_busy = {
        InstanceSlot(bid, operator, index): busy
        for bid, residue in layout.ctx.schedules.block_entry_busy.items()
        for (operator, index), busy in residue.items()
    }
    coloring = color(
        ColoringProblem(
            movable=q_movable,
            pinned=q_pinned,
            interferes=q_interferes,
            fixed_producers=q_producers,
            reserved=frozenset(coalesced.reserved),
            fresh_start=coalesced.facts.fresh_start,
            firings=q_firings,
            instances=layout.ctx.schedules.instances,
            entry_busy=entry_busy,
            tuning=tuning,
        )
    )
    assign = {vid: coloring.assign[lead(vid)] for vid in interferes}
    # The EXPANDED per-value coloring against the FULL (residual-install) interference, stronger than a check over
    # the collapsed quotient: it catches an unsound union or oracle drift. The pins were validated by `coalesce`.
    assert find_coloring_conflict(assign, interferes) is None
    return replace(coloring, assign=assign)


def _color_bank[D: _Placement | Boundary](
    coalesced: _CoalescedBank[D],
    layout: CoalescedLayout,
    bool_reg: Mapping[ValueId, int],
    tuning: RegallocTuning,
) -> Coloring:
    bank = coalesced.facts.bank
    _logger.info(
        "Coloring %s register bank: %d values, %d pinned, %d movable, %d reserved",
        bank.label,
        len(coalesced.facts.values),
        len(coalesced.pinned),
        len(coalesced.movable),
        len(coalesced.reserved),
    )
    producers = bank.fixed_producers(layout, coalesced, bool_reg)
    # A slot installing its live-out by a copy is one more writer of its slot register. On a reserved register that
    # writer is alone; a read slot whose live-out sits in ANOTHER slot's register keeps its own open to other values,
    # and its boundary install is an arm among theirs.
    for slot in coalesced.facts.slots:
        r_in = coalesced.facts.livein_of[slot.name]
        if r_in is not None and not isinstance(coalesced.install[slot.name], InPlace):
            producers[r_in].append(SlotWriter(slot.name))
    coloring = _color_quotient(coalesced, producers, bank.firings(layout, coalesced), layout, tuning)
    # Backstop: a reserved (non-coalesced) slot register must carry nothing but its own live-in. A coalesced slot
    # register IS shared by its in-place live-out (and any phi arms merged onto it).
    for slot in coalesced.facts.slots:
        reg = coalesced.facts.slot_reg[slot.name]
        if reg not in coalesced.reserved:
            continue
        occupants = [
            vid for vid, r in coloring.assign.items() if r == reg and vid != coalesced.facts.livein_of[slot.name]
        ]
        assert (
            not occupants
        ), f"non-coalesced {bank.label} slot register {reg} ({slot.name!r}) has occupants {occupants}"
    return coloring


def allocate(layout: CoalescedLayout, tuning: RegallocTuning) -> Allocation:
    """
    Color both banks of a converged layout. Both are colored by hardware-frame liveness, reusing registers across
    mutually-exclusive and non-overlapping live ranges. A phi is resolved by installing each arm's value into the
    phi's register with a copy at the predecessor's tail; copies sharing a fire step are a parallel (read-then-write)
    bundle, and placement keeps a swap correct (see `install_issue_cycle`).
    """
    # The boolean bank first: its allocation depends on nothing the wide bank decides, and the wide bank's inline
    # results are keyed by the boolean registers they read.
    bool_coloring = _color_bank(layout.bool_bank, layout, {}, tuning)
    wide = _color_bank(layout.wide_bank, layout, bool_coloring.assign, tuning)

    # A phi arm coalesced onto the merged register needs no install copy: the arm value already resides in the phi's
    # register (they share a coloring class). Only the residual (non-coalesced) arms install by a pc-gated copy.
    ctx = layout.ctx
    copies: dict[int, list[WideCopy | BoolCopy]] = {}
    for pred, vid, value, conditioner in layout.wide_bank.residual_arms():
        placement = layout.wide_bank.facts.placement[(pred, vid)]
        source = wide_operand_template(ctx.wide_mir, value, conditioner, ctx.const_pool.entries)
        copy = WideCopy(
            RegRef(wide.assign[vid]), source.resolve(wide.assign.__getitem__), placement.issue, placement.settled
        )
        copies.setdefault(pred, []).append(copy)
    for pred, vid, value, inversion in layout.bool_bank.residual_arms():
        placement = layout.bool_bank.facts.placement[(pred, vid)]
        bool_source = bool_operand_template(ctx.bool_mir, value, inversion).resolve(bool_coloring.assign.__getitem__)
        bool_copy = BoolCopy(BoolRegRef(bool_coloring.assign[vid]), bool_source, placement.issue, placement.settled)
        copies.setdefault(pred, []).append(bool_copy)
    wide_facts = layout.wide_bank.facts
    wide_install: dict[str, InPlace | Early | Boundary] = {}
    for slot in wide_facts.slots:
        match decision := layout.wide_bank.install[slot.name]:
            case _Placement(issue=issue, settled=settled):
                source = wide_operand_template(ctx.wide_mir, slot.live_out, slot.conditioner, ctx.const_pool.entries)
                reg = RegRef(wide_facts.slot_reg[slot.name])
                early = WideCopy(reg, source.resolve(wide.assign.__getitem__), issue, settled)
                copies.setdefault(wide_facts.ret_block, []).append(early)
                wide_install[slot.name] = Early()
            case InPlace() | Boundary():
                wide_install[slot.name] = decision
            case _:
                assert_never(decision)

    instances = [
        OperatorInstance(operator, i)
        for operator, count in layout.ctx.schedules.instances.items()
        for i in range(count)
    ]
    # First-free binding takes instance k only while 0..k-1 are busy, so no rebinding can close an instance the
    # scheduler realized; the labels are dense.
    bound = {
        OperatorInstance(seed.operator, wide.instance[leader])
        for sched in layout.ctx.schedules.block_sched.values()
        for leader, seed in sched.inst_of.items()
    }
    assert bound == set(instances)
    _logger.info(
        "Allocation: %d wide + %d bool registers, %d copies",
        wide.nreg,
        bool_coloring.nreg,
        sum(len(block_copies) for block_copies in copies.values()),
    )
    return Allocation(
        wide=wide,
        wide_slot_reg=layout.wide_bank.facts.slot_reg,
        wide_install=wide_install,
        bool=bool_coloring,
        bool_slot_reg=layout.bool_bank.facts.slot_reg,
        bool_install=layout.bool_bank.install,
        copies=copies,
        instances=instances,
    )
