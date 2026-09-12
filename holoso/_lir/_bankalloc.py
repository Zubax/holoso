"""
Register-bank allocation: liveness facts, the wide and boolean bank policies, the layout and coalescing pass the
install fixpoint iterates, and the coloring of both banks' coalescing classes once it has converged.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import ClassVar

from .._mir import (
    Mir,
    reverse_postorder,
    MirBoolOutput,
    MirBoolView,
    MirBranch,
    MirFloatInput,
    MirFloatOutput,
    MirIntInput,
    MirIntOutput,
    MirNode,
    MirOperation,
    MirPhi,
    MirStateRead,
    MirStateSlot,
    MirWideView,
)
from .._operators import (
    BoolInversion,
    HardwareOperator,
    InlineHardwareOperator,
    PooledHardwareOperator,
    PortConditioner,
)
from .._value import WideValue
from .._util import ValueId
from ._ir import *
from ._mir_facts import const_branch_conditions, mir_operation, phi_arm_out, succ_map
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
    PoolWord,
    RegallocTuning,
    SlotWriter,
    color,
    find_coloring_conflict,
)
from ._sources import BoolOperandTemplate
from ._build_base import Allocation, BoolArmInstall, OverlapLayout, PooledConst, WideArmInstall
from ._construct import bool_operand_template, build_const_pool, operand_templates, wide_operand_template
from ._coalesce import PhiCoalescing, coalescable_arms, coalesce
from ._layout import schedule_with_overlap

_logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _CoalescedBank:
    """
    One bank coalesced and pinned but not colored: its values and nodes, its slots with their registers, live-ins
    and install cycles, the phi-arm coalescing, the pins, the reserved registers, the final install-aware
    interference and the movable order. Nothing here depends on the register a movable value takes.
    """

    values: set[ValueId]
    op_nodes: Mapping[ValueId, MirOperation]
    phi_nodes: Mapping[ValueId, MirPhi]
    slots: Sequence[MirStateSlot]
    livein_of: dict[str, ValueId | None]
    slot_reg: dict[str, int]
    install: dict[str, int]  # slot name -> Ret-block-relative scheduler-frame install cycle of its live-out
    coalescing: PhiCoalescing
    pinned: dict[ValueId, int]
    reserved: set[int]  # the non-coalesced slot registers, which no movable value may join
    interferes: dict[ValueId, set[ValueId]]
    movable: list[ValueId]
    fresh_start: int

    def needs_copy(self, slot: MirStateSlot) -> bool:
        """Whether the slot installs its live-out by a copy: it does not sit in the slot register under the identity."""
        return not (slot.conditioner.is_identity and self.pinned.get(slot.live_out) == self.slot_reg[slot.name])

    def residual_arms(self) -> list[tuple[int, ValueId, ValueId, PortConditioner]]:
        """Every `(pred, phi, arm value, conditioner)` not coalesced, so installing by a copy at `pred`'s tail."""
        return [
            (pred, vid, value, conditioner)
            for vid, phi in self.phi_nodes.items()
            for pred, value, conditioner in phi.arms
            if (pred, vid) not in self.coalescing.coalesced
        ]


@dataclass(frozen=True, slots=True)
class CoalescedLayout:
    """
    One layout pass for a given install set, both banks coalesced and pinned but not colored: what the install
    fixpoint iterates on. Nothing it reads depends on the coloring, which `allocate` runs once on the converged
    layout. `instances` is the instance count the scheduler realized per operator.
    """

    mir: Mir
    wide_mir: MirWideView
    bool_mir: MirBoolView
    overlap: OverlapLayout
    instances: dict[PooledHardwareOperator, int]
    consts: list[WideValue]
    const_pool: dict[ValueId, PooledConst]
    fetch_lag: int
    bool_bank: _CoalescedBank
    wide_bank: _CoalescedBank

    def install_blocks(self) -> dict[int, bool]:
        """
        The post-coalescing install classification, refining `block_has_install`'s conservative one: each block that
        installs at its tail -- a residual phi arm, or the constant branch condition it materializes -- mapped to
        whether any of its installs issues past the work makespan, decided through the same `install_source_commit`
        + `install_issue_cycle` pair as the LIR placement and the interference residence, so the three cannot drift.
        """
        overlap, fetch_lag = self.overlap, self.fetch_lag
        install: dict[int, bool] = {}
        for bank in (self.wide_bank, self.bool_bank):
            for pred, _vid, value, _conditioner in bank.residual_arms():
                sched = overlap.block_sched[pred]
                commit = install_source_commit(sched, value, overlap.block_inflight[pred], fetch_lag)
                pushes = install_issue_cycle(sched.makespan, commit) > sched.makespan
                install[pred] = install.get(pred, False) or pushes
        for block_id in const_branch_conditions(self.mir, self.bool_mir):
            install.setdefault(block_id, False)  # a literal is a settled source, never pushing
        return install

    def has_state_copy(self) -> bool:
        """
        Whether some slot of either bank installs its live-out by a copy. A non-coalesced slot installs by a
        read-first boundary copy that lands a fetch pipeline past the live-out, so the Ret block's drain must reach
        it (`boundary_step(makespan)`, bank-independent); a coalesced slot writes its register in place.
        """
        return any(bank.needs_copy(slot) for bank in (self.wide_bank, self.bool_bank) for slot in bank.slots)


def layout_and_coalesce(
    mir: Mir,
    wide_mir: MirWideView,
    bool_mir: MirBoolView,
    pool: Mapping[type[HardwareOperator], int],
    has_install_blocks: Mapping[int, bool],
    has_state_copy: bool,
    fetch_lag: int,
) -> CoalescedLayout:
    overlap = schedule_with_overlap(mir, wide_mir, bool_mir, pool, has_install_blocks, has_state_copy, fetch_lag)
    inst_count: dict[PooledHardwareOperator, int] = {}
    for sched in overlap.block_sched.values():
        for inst in sched.inst_of.values():
            inst_count[inst.operator] = max(inst_count.get(inst.operator, 0), inst.index + 1)
    _logger.info(
        "Operator instances used: %s",
        ", ".join(f"{operator.mnemonic} {count}/{pool[type(operator)]}" for operator, count in inst_count.items())
        or "none",
    )
    consts, const_pool = build_const_pool(wide_mir, bool_mir.operation_nodes)
    bool_bank = _coalesce_bank(_BOOL, mir, wide_mir, bool_mir, overlap, fetch_lag)
    wide_bank = _coalesce_bank(_WIDE, mir, wide_mir, bool_mir, overlap, fetch_lag)
    return CoalescedLayout(
        mir, wide_mir, bool_mir, overlap, inst_count, consts, const_pool, fetch_lag, bool_bank, wide_bank
    )


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
    read, a phi, or a foreign result already landed -- is settled at entry. The lone encoding of the residency rule,
    shared by the push classification, the interference residence, and the LIR copy placement, so the three cannot
    drift apart (a drift here recreates the placement-disagreement class of the phi-swap miscompile).
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
    The deterministic coloring order shared by both banks: reverse-postorder block, then commit cycle, then value id
    (the `-3` sentinel sorts a phi -- which has no commit -- ahead of the operations in its block). Value-id last
    keeps the order, and hence the coloring, seed-independent; both banks MUST use this one definition.
    """
    rpo_pos = {bid: i for i, bid in enumerate(reverse_postorder(mir))}
    block_of = {**op_block, **phi_block}
    return sorted(candidates, key=lambda vid: (rpo_pos[block_of[vid]], op_commit.get(vid, -3), vid))


type _BankView = MirWideView | MirBoolView


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
    fetch_lag: int


class _Bank(ABC):
    """
    What differs between the two physical register banks: `_coalesce_bank` and `_color_bank` are shared, and each
    subclass supplies the wide or boolean specifics: the view, the boundary users, the priced writers and firings,
    and the install policy.
    """

    label: ClassVar[str]

    @abstractmethod
    def view(self, wide_mir: MirWideView, bool_mir: MirBoolView) -> _BankView: ...

    @abstractmethod
    def boundary_base(self, mir: Mir, values: set[ValueId], ret_block: int) -> dict[int, set[ValueId]]:
        """The non-slot boundary users: bank outputs, plus the per-block branch conditions for the boolean bank."""

    @abstractmethod
    def fixed_producers(
        self, layout: CoalescedLayout, coalesced: _CoalescedBank, bool_reg: Mapping[ValueId, int]
    ) -> dict[ValueId, list[FixedProducer]]:
        """
        Each value's writers beside the pooled lanes, as the objective prices them: the wide bank's input loads,
        inline results and residual arms, the boolean bank's residual arms only (a one-bit mux is not priced).
        `bool_reg` is the boolean bank's assignment, decided first, so the wide bank's inline results name the
        boolean registers they read (empty while the boolean bank itself is colored).
        """

    @abstractmethod
    def firings(self, layout: CoalescedLayout, coalesced: _CoalescedBank) -> list[Firing]:
        """The pooled firings the objective prices: every one for the wide bank, none for the boolean bank."""

    @abstractmethod
    def install_policy(self, ctx: _InstallContext) -> dict[str, int]:
        """
        Each slot's Ret-block-relative live-out install cycle. The wide bank installs early where it can to free the
        source register; the boolean bank installs every live-out at the boundary (it has no early install).
        """


class _WideBank(_Bank):
    label = "wide"

    def view(self, wide_mir: MirWideView, bool_mir: MirBoolView) -> MirWideView:
        return wide_mir

    def boundary_base(self, mir: Mir, values: set[ValueId], ret_block: int) -> dict[int, set[ValueId]]:
        boundary: dict[int, set[ValueId]] = {block.id: set() for block in mir.blocks}
        for out in mir.outputs:
            if isinstance(out, (MirFloatOutput, MirIntOutput)) and out.value in values:
                boundary[ret_block].add(out.value)
        return boundary

    def fixed_producers(
        self, layout: CoalescedLayout, coalesced: _CoalescedBank, bool_reg: Mapping[ValueId, int]
    ) -> dict[ValueId, list[FixedProducer]]:
        producers: dict[ValueId, list[FixedProducer]] = {vid: [] for vid in coalesced.values}
        producers.update({vid: [InputWriter(vid)] for vid in layout.wide_mir.input_ids})
        for vid, node in coalesced.op_nodes.items():
            if isinstance(node.operator, PooledHardwareOperator):
                continue
            assert isinstance(node.operator, InlineHardwareOperator)
            operands = tuple(
                t.resolve(bool_reg.__getitem__) if isinstance(t, BoolOperandTemplate) else t
                for t in operand_templates(node, layout.wide_mir, layout.bool_mir, layout.const_pool)
            )
            producers[vid] = [InlineWriter(node.operator, operands, node.output_conditioner)]
        for _pred, vid, value, conditioner in coalesced.residual_arms():
            assert not isinstance(conditioner, BoolInversion)
            template = wide_operand_template(layout.wide_mir, value, conditioner, layout.const_pool)
            producers[vid].append(MoveWriter(template))
        return producers

    def firings(self, layout: CoalescedLayout, coalesced: _CoalescedBank) -> list[Firing]:
        # Every pooled firing reads wide operands, so every one is listed, a comparator whose taps are all boolean
        # included. A firing is bindable when its class has more than one realized instance and its busy window ends
        # inside the block (`issue + II <= term_offset + 1`, the scheduler's carry condition), so it leaves no residue
        # a successor could inherit.
        values, op_nodes = coalesced.values, coalesced.op_nodes
        block_sched = layout.overlap.block_sched
        firings: list[Firing] = []
        for bid in sorted(block_sched):
            sched = block_sched[bid]
            for leader in sorted(sched.firings, key=lambda vid: (sched.issue_cycle[vid], vid)):
                node = mir_operation(layout.mir, leader)
                operator = node.operator
                assert isinstance(operator, PooledHardwareOperator)
                issue = sched.issue_cycle[leader]
                bindable = (
                    layout.instances[operator] > 1
                    and issue + operator.initiation_interval <= layout.overlap.block_term_offset[bid] + 1
                )
                reads: list[ValueId | PoolWord] = []
                for operand in node.operands:
                    if operand in values:
                        reads.append(operand)
                    else:
                        entry = layout.const_pool.get(operand)
                        assert entry is not None, f"operand {operand} of firing {leader} is neither wide nor pooled"
                        reads.append(PoolWord(entry.index))
                writes = [(op_nodes[m].output_port, m) for m in sched.firings[leader] if m in values]
                seed = sched.inst_of[leader].index
                firings.append(Firing(leader, operator, bid, issue, seed, bindable, reads, writes))
        return firings

    def install_policy(self, ctx: _InstallContext) -> dict[str, int]:
        # Install the live-out as early as the live-in is fully read and the source is available, freeing the source
        # register -- but only when the live-out is produced in the Ret block (a unique, once-per-transaction exit),
        # the live-in is not itself a boundary user, and the slot neither coalesced nor feeds a chained copy. Otherwise
        # the boundary.
        install: dict[str, int] = {}
        for slot in ctx.slots:
            name, live_out, r_in = slot.name, slot.live_out, ctx.livein_of[slot.name]
            node = ctx.nodes[live_out]
            defined_in_ret = isinstance(node, (MirFloatInput, MirIntInput)) or (
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
                    cycle = max(cycle, ctx.last_read_ret.get(r_in, 0) - inline_fire_cycle(0, ctx.fetch_lag))
                install[name] = min(cycle, ctx.ret_present)
            else:
                install[name] = ctx.ret_present
        return install


class _BoolBank(_Bank):
    label = "bool"

    def view(self, wide_mir: MirWideView, bool_mir: MirBoolView) -> MirBoolView:
        return bool_mir

    def boundary_base(self, mir: Mir, values: set[ValueId], ret_block: int) -> dict[int, set[ValueId]]:
        boundary: dict[int, set[ValueId]] = {block.id: set() for block in mir.blocks}
        for block in mir.blocks:
            if isinstance(block.terminator, MirBranch) and block.terminator.cond in values:
                boundary[block.id].add(block.terminator.cond)
        for out in mir.outputs:
            if isinstance(out, MirBoolOutput) and out.value in values:
                boundary[ret_block].add(out.value)
        return boundary

    def fixed_producers(
        self, layout: CoalescedLayout, coalesced: _CoalescedBank, bool_reg: Mapping[ValueId, int]
    ) -> dict[ValueId, list[FixedProducer]]:
        # One-bit muxes are not priced: no firings (a comparator's lanes), no inline results, no input loads. The
        # objective is the register count plus the residual phi arms and the slot installs the bank-independent code
        # adds.
        producers: dict[ValueId, list[FixedProducer]] = {vid: [] for vid in coalesced.values}
        for _pred, vid, value, inversion in coalesced.residual_arms():
            assert isinstance(inversion, BoolInversion)
            producers[vid].append(MoveWriter(bool_operand_template(layout.bool_mir, value, inversion)))
        return producers

    def firings(self, layout: CoalescedLayout, coalesced: _CoalescedBank) -> list[Firing]:
        return []

    def install_policy(self, ctx: _InstallContext) -> dict[str, int]:
        return {slot.name: ctx.ret_present for slot in ctx.slots}  # every boolean live-out installs at the boundary


_WIDE = _WideBank()

_BOOL = _BoolBank()


@dataclass(frozen=True, slots=True)
class _InterferenceBuilder:
    """
    Builds a bank's interference graph: the loop-invariant liveness facts (residency, result landings, definition
    blocks, phi-arm live-outs, in-flight defs) are fixed at construction, and the held MIR supplies the block CFG, so
    each coalescing attempt produces a graph by passing only what varies -- the boundary users, the per-block reads,
    and the residual installs.
    """

    mir: Mir
    work_makespan: dict[int, int]
    term_offset: dict[int, int]
    resident: frozenset[ValueId]
    op_landing: dict[ValueId, int]
    op_block: dict[ValueId, int]
    phi_block: dict[ValueId, int]
    arm_out: dict[int, frozenset[ValueId]]
    inflight_defs: dict[int, dict[ValueId, int]]
    # block -> phi dest -> its arm source's commit cycle in THIS block's frame, None for a source settled before the
    # block's first step (`install_issue_cycle`'s contract), so the interference residence matches the LIR copy's
    # placement and cannot drift onto a foreign block's frame.
    install_source: dict[int, dict[ValueId, int | None]]
    fetch_lag: int

    def _install_fire(self, block: int, vid: ValueId) -> int:
        """
        The install's block-local fire step, via the same helpers as the LIR install so residence matches exactly.
        It MAY transiently land past the current terminator: after a push-bit narrowing, an intermediate coalescing
        attempt can hold a computed arm de-coalesced whose install no longer fits the shortened boundary -- the outer
        install fixpoint then re-widens (pins) the classification and re-runs, and only the converged round's
        placements are emitted (the build-side landing assert guards those). Frame confinement itself is guaranteed by
        `install_issue_cycle`'s own precondition, so there is nothing falsifiable to assert here.
        """
        issue = install_issue_cycle(self.work_makespan[block], self.install_source[block][vid])
        return inline_fire_cycle(issue, self.fetch_lag)

    def build(
        self,
        boundary: dict[int, set[ValueId]],
        block_reads: dict[int, list[tuple[ValueId, int]]],
        install_facts: dict[int, frozenset[ValueId]],
    ) -> dict[ValueId, set[ValueId]]:
        return compute_interference(
            BankLiveness(
                blocks=[b.id for b in self.mir.blocks],
                entry=self.mir.entry,
                succ=succ_map(self.mir),
                term_offset=self.term_offset,
                resident=self.resident,
                op_landing=self.op_landing,
                op_block=self.op_block,
                phi_block=self.phi_block,
                reads=block_reads,
                boundary_users={b: frozenset(s) for b, s in boundary.items()},
                arm_out=self.arm_out,
                installs={b: {vid: self._install_fire(b, vid) for vid in vids} for b, vids in install_facts.items()},
                inflight_defs=self.inflight_defs,
            )
        )


def _coalesce_bank(
    bank: _Bank, mir: Mir, wide_mir: MirWideView, bool_mir: MirBoolView, overlap: OverlapLayout, fetch_lag: int
) -> _CoalescedBank:
    """
    Coalesce one physical register bank across the CFG: its liveness, phi-arm coalescing, slot in-place commits with
    validate-and-retry, pins and install placement. The bank descriptor supplies the boundary and install policies;
    the rest is bank-independent. The coloring is `_color_bank`'s, once the layout has converged.
    """
    block_sched = overlap.block_sched
    view = bank.view(wide_mir, bool_mir)
    nload = len(view.input_ids)
    slots: Sequence[MirStateSlot] = view.state_slots
    slot_reg = {slot.name: nload + i for i, slot in enumerate(slots)}
    fresh_start = nload + len(slots)
    op_nodes = view.operation_nodes
    phi_nodes = view.phi_nodes
    state_read_nodes = view.state_read_nodes
    state_read_of = {node.name: vid for vid, node in state_read_nodes.items()}
    values = {*view.input_ids, *state_read_nodes, *op_nodes, *phi_nodes}
    facts = _bank_liveness_facts(mir, block_sched, op_nodes, phi_nodes, values, fetch_lag)
    op_block, op_commit, phi_block, reads = facts.op_block, facts.op_commit, facts.phi_block, facts.reads
    # The spills this bank receives, reserving a spilled value's register across every successor frame it lands in
    # even where the value is dataflow-dead.
    inflight = {
        bid: {vid: land for vid, land in spills.items() if vid in op_nodes}
        for bid, spills in overlap.block_inflight.items()
    }

    ret_block = mir.ret_block
    arm_out = phi_arm_out(mir, phi_nodes, values)
    boundary_base = bank.boundary_base(mir, values, ret_block)

    # Per block, each phi dest whose arm originates there -> its source's commit cycle in the PREDECESSOR's own frame
    # (where the install fires, not the source's home block), None for a source settled before the block's first
    # step. The same rule the build and `install_blocks` apply, so the interference residence cannot drift onto a
    # foreign block's frame.
    install_source: dict[int, dict[ValueId, int | None]] = {}
    for vid, phi in phi_nodes.items():
        for pred, value, _conditioner in phi.arms:
            install_source.setdefault(pred, {})[vid] = install_source_commit(
                block_sched[pred], value, inflight[pred], fetch_lag
            )

    interference = _InterferenceBuilder(
        mir=mir,
        work_makespan={bid: sched.makespan for bid, sched in block_sched.items()},
        term_offset=overlap.block_term_offset,
        resident=frozenset({*view.input_ids, *state_read_nodes}),
        # Every result -- pooled or inline, wide or boolean -- lands at the one bank-independent landing.
        op_landing={vid: landing_cycle(commit, fetch_lag) for vid, commit in op_commit.items()},
        op_block=op_block,
        phi_block=phi_block,
        arm_out=arm_out,
        inflight_defs=inflight,
        install_source=install_source,
        fetch_lag=fetch_lag,
    )

    # A slot whose live-in is consumed as ANOTHER slot's live-out (a chained copy, `self.a = self.b`) must keep its
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
    coalesce_graph = interference.build(boundary_oracle, reads, {})
    candidate_arms = coalescable_arms(phi_nodes, values)
    phi_order = _movable_order(mir, list(phi_nodes), {}, phi_block, {})
    ret_present = overlap.block_makespan[ret_block] + 1
    # Last operand-read cycle of each value in the Ret block, from the shared liveness facts so read-cycle semantics
    # cannot drift; it bounds how early a slot may install over its source. Loop-invariant -- only an early install
    # reads it, and it is keyed only by state live-ins (always in `values`), so the facts' value filter drops nothing.
    last_read_in_ret: dict[ValueId, int] = {}
    for vid, rc in reads[ret_block]:
        last_read_in_ret[vid] = max(last_read_in_ret.get(vid, 0), rc)
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
            if not slot.conditioner.is_identity or not producible or slot.name in tapped_by_other:
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
                fetch_lag,
            )
        )

        # Final interference. A non-coalesced slot reserves its live-in to the boundary (the install reads it
        # read-first, so the register holds nothing else); a coalesced slot keeps its live-in's actual range, so a gap
        # tenant lands between the live-in's last read and the live-out's landing. A boundary-installed live-out is read
        # at the boundary; an early-installed one is read by its copy at the install step, freeing its source for a
        # later tenant.
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
                early_reads.append((live_out, inline_fire_cycle(install[name], fetch_lag)))
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
            candidate_arms,
            pinned,
            reserved,
            lambda residual: interference.build(boundary_final, reads_final, residual),
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
        sum(len(arms) for arms in candidate_arms.values()),
        len(coalesced),
        len(slots),
        len(forced_copy),
    )

    movable = _movable_order(
        mir, [vid for vid in (*op_nodes, *phi_nodes) if vid not in pinned], op_block, phi_block, op_commit
    )
    return _CoalescedBank(
        values,
        op_nodes,
        phi_nodes,
        slots,
        livein_of,
        slot_reg,
        install,
        coalescing,
        pinned,
        reserved,
        interferes,
        movable,
        fresh_start,
    )


def _color_quotient(
    coalesced: _CoalescedBank,
    producers: dict[ValueId, list[FixedProducer]],
    firings: list[Firing],
    layout: CoalescedLayout,
    tuning: RegallocTuning,
) -> Coloring:
    """
    Color the per-value interference graph after collapsing each coalescing class to its leader, then expand the
    leader's color back onto every member. The quotient maps every firing's read and write endpoints and every
    producer's holes to leaders and concatenates each class's fixed producers, so the steering objective is the
    merged register's actual read and write fan-in. A class with a pinned member pins its leader. Reduces to the
    plain per-value coloring when the coalescing is the identity (every value its own singleton class, e.g. a kernel
    with no coalescable phi arms).
    """
    coalescing, interferes = coalesced.coalescing, coalesced.interferes

    def lead(v: ValueId) -> ValueId:
        return coalescing.leader.get(v, v)

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
    for vid in coalesced.movable:  # leaders of the movable values, first occurrence, preserving the deterministic order
        head = lead(vid)
        if head in q_pinned or head in seen:
            continue
        seen.add(head)
        q_movable.append(head)
    q_firings = [
        replace(
            firing,
            reads=[source if isinstance(source, PoolWord) else lead(source) for source in firing.reads],
            writes=[(port, lead(value)) for port, value in firing.writes],
        )
        for firing in firings
    ]
    entry_busy = {
        InstanceSlot(bid, operator, index): busy
        for bid, residue in layout.overlap.block_entry_busy.items()
        for (operator, index), busy in residue.items()
    }
    coloring = color(
        ColoringProblem(
            movable=q_movable,
            pinned=q_pinned,
            interferes=q_interferes,
            fixed_producers=q_producers,
            reserved=frozenset(coalesced.reserved),
            fresh_start=coalesced.fresh_start,
            firings=q_firings,
            instances=layout.instances,
            entry_busy=entry_busy,
            tuning=tuning,
        )
    )
    assign = {vid: coloring.assign[lead(vid)] for vid in interferes}
    # The EXPANDED per-value coloring against the FULL (residual-install) interference, stronger than a check over
    # the collapsed quotient: it catches an unsound union or oracle drift. The pins were validated by `coalesce`.
    assert find_coloring_conflict(assign, interferes) is None
    return replace(coloring, assign=assign)


def _color_bank(
    bank: _Bank,
    coalesced: _CoalescedBank,
    layout: CoalescedLayout,
    bool_reg: Mapping[ValueId, int],
    tuning: RegallocTuning,
) -> Coloring:
    """
    Color one coalesced bank: every value's register, and the orientation, binding and mux arms the annealer chose.
    """
    _logger.info(
        "Coloring %s register bank: %d values, %d pinned, %d movable, %d reserved",
        bank.label,
        len(coalesced.values),
        len(coalesced.pinned),
        len(coalesced.movable),
        len(coalesced.reserved),
    )
    producers = bank.fixed_producers(layout, coalesced, bool_reg)
    # A slot installing its live-out by a copy is one more writer of its slot register. On a reserved register that
    # writer is alone; a read slot whose live-out sits in ANOTHER slot's register keeps its own open to other values,
    # and its boundary install is an arm among theirs.
    for slot in coalesced.slots:
        r_in = coalesced.livein_of[slot.name]
        if r_in is not None and coalesced.needs_copy(slot):
            producers[r_in].append(SlotWriter(slot.name))
    coloring = _color_quotient(coalesced, producers, bank.firings(layout, coalesced), layout, tuning)
    # Backstop: a reserved (non-coalesced) slot register must carry nothing but its own live-in. A coalesced slot
    # register IS shared by its in-place live-out (and any phi arms merged onto it).
    for slot in coalesced.slots:
        reg = coalesced.slot_reg[slot.name]
        if reg not in coalesced.reserved:
            continue
        occupants = [vid for vid, r in coloring.assign.items() if r == reg and vid != coalesced.livein_of[slot.name]]
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
    bool_coloring = _color_bank(_BOOL, layout.bool_bank, layout, {}, tuning)
    bool_reg = dict(bool_coloring.assign)
    wide = _color_bank(_WIDE, layout.wide_bank, layout, bool_reg, tuning)

    # A phi arm coalesced onto the merged register needs no install copy: the arm value already resides in the phi's
    # register (they share a coloring class). Only the residual (non-coalesced) arms install by a pc-gated copy.
    wide_copies: dict[int, list[WideArmInstall]] = {}
    for pred, vid, value, conditioner in layout.wide_bank.residual_arms():
        assert not isinstance(conditioner, BoolInversion)
        wide_copies.setdefault(pred, []).append(WideArmInstall(wide.assign[vid], value, conditioner))
    bool_writes: dict[int, list[BoolArmInstall]] = {}
    for pred, vid, value, inversion in layout.bool_bank.residual_arms():
        assert isinstance(inversion, BoolInversion)
        bool_writes.setdefault(pred, []).append(BoolArmInstall(bool_reg[vid], value, inversion))

    # A constant branch condition (e.g. a read-only boolean attribute, or a folded test) has no register of its own;
    # materialize it into a bool register written in the branching block so the next-PC decode can read it. The constant
    # is globally interned, so sibling branches sharing it reuse one register -- but the write must be emitted in EVERY
    # branching block that uses it, else a path reaching the branch through a block that did not write it reads a stale
    # register. (A later static-branch-folding pass would instead drop the dead arm; until then this keeps it correct.)
    nbreg = bool_coloring.nreg
    for block_id, cond in const_branch_conditions(layout.mir, layout.bool_mir).items():
        if cond not in bool_reg:
            bool_reg[cond] = nbreg
            nbreg += 1
        bool_writes.setdefault(block_id, []).append(BoolArmInstall(bool_reg[cond], cond, BoolInversion()))

    instances = [OperatorInstance(operator, i) for operator, count in layout.instances.items() for i in range(count)]
    # First-free binding takes instance k only while 0..k-1 are busy, so no rebinding can close an instance the
    # scheduler realized; the labels are dense.
    bound = {
        OperatorInstance(seed.operator, wide.instance[leader])
        for sched in layout.overlap.block_sched.values()
        for leader, seed in sched.inst_of.items()
    }
    assert bound == set(instances)
    _logger.info(
        "Allocation: %d wide + %d bool registers, %d wide copies, %d bool writes",
        wide.nreg,
        nbreg,
        sum(len(copies) for copies in wide_copies.values()),
        sum(len(writes) for writes in bool_writes.values()),
    )
    return Allocation(
        wide=wide,
        wide_slot_reg=layout.wide_bank.slot_reg,
        wide_install=layout.wide_bank.install,
        bool_reg=bool_reg,
        bool_slot_reg=layout.bool_bank.slot_reg,
        nbreg=nbreg,
        wide_copies=wide_copies,
        bool_writes=bool_writes,
        instances=instances,
    )
