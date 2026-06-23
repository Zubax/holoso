"""Register-bank allocation: liveness facts, the wide/bool bank policy surface, and the layout+allocation pass."""

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar, Generic, TypeVar

from .._mir import (
    Mir,
    MirBoolOutput,
    MirBoolStateSlot,
    MirBoolView,
    MirBranch,
    MirFloatInput,
    MirFloatOutput,
    MirFloatStateSlot,
    MirFloatView,
    MirNode,
    MirOperation,
    MirPhi,
    MirStateRead,
    MirStateSlot,
)
from .._operators import BoolInversion, FloatSignControl, HardwareOperator, PooledHardwareOperator, PortConditioner
from .._util import ValueId
from ._ir import *
from ._mir_facts import const_branch_conditions, mir_rpo, phi_arm_out, succ_map
from ._liveness import BankLiveness, compute_interference
from ._schedule import Schedule
from ._regalloc import Producer
from ._build_base import (
    Allocation,
    BoolArmInstall,
    ColorObjective,
    FloatArmInstall,
    OverlapLayout,
    PooledConst,
)
from ._construct import build_const_pool
from ._coalesce import coalescable_arms, coalesce_and_color
from ._layout import schedule_with_overlap


@dataclass(frozen=True, slots=True)
class _BankAlloc:
    """One bank's assignment: register per value, register per state slot, count, per-slot install, and coalescing."""

    reg: dict[ValueId, int]
    slot_reg: dict[str, int]
    nreg: int
    install: dict[str, int]  # slot name -> Ret-block-relative scheduler-frame install cycle of its live-out
    coalesced: frozenset[tuple[int, ValueId]]  # (pred, phi) arms coalesced onto the merged register (no copy)


@dataclass(frozen=True, slots=True)
class _LayoutAllocation:
    """
    One full layout+allocation pass for a given install set: the overlap layout, pooled instances, the const pool,
    and the register assignment of both banks. Re-run by the coalesced-install fixpoint as the install set shrinks.
    """

    overlap: OverlapLayout
    inst_of: dict[ValueId, OperatorInstance]
    instances: list[OperatorInstance]
    consts: list[float]
    const_pool: dict[ValueId, PooledConst]
    alloc: Allocation


def layout_and_allocate(
    mir: Mir,
    float_mir: MirFloatView,
    bool_mir: MirBoolView,
    pool: Mapping[type[HardwareOperator], int],
    has_install_blocks: set[int],
    state_copy_blocks: Mapping[int, bool],
) -> _LayoutAllocation:
    """Lay out the blocks (cross-block overlap) and color both register banks for the given per-block install set."""
    overlap = schedule_with_overlap(mir, float_mir, bool_mir, pool, has_install_blocks, state_copy_blocks)
    block_sched = overlap.block_sched
    inst_of: dict[ValueId, OperatorInstance] = {}
    inst_count: dict[PooledHardwareOperator, int] = {}
    for sched in block_sched.values():
        inst_of.update(sched.inst_of)
        for inst in sched.instances:
            inst_count[inst.operator] = max(inst_count.get(inst.operator, 0), inst.index + 1)
    instances = [OperatorInstance(operator, i) for operator in inst_count for i in range(inst_count[operator])]
    consts, const_pool = build_const_pool(float_mir, bool_mir.operation_nodes)
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


def actual_install_blocks(alloc: Allocation, const_branch_blocks: set[int]) -> set[int]:
    """
    The blocks that actually install at their tail after coalescing: a real float copy or boolean write, or a const-
    branch materialization (which is not a copy). A CFG-shape phi-arm predecessor whose every arm coalesced installs
    nothing, so it should pay neither the +1 install makespan nor the overlap-ineligibility that ``block_has_install``
    assigns from the CFG shape alone -- it drops out of the install set here, which the fixpoint feeds back to the next
    layout so the spurious drain is removed.
    """
    blocks = set(const_branch_blocks)
    blocks.update(bid for bid, copies in alloc.copies.items() if copies)
    blocks.update(bid for bid, writes in alloc.bool_writes.items() if writes)
    return blocks


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
    rpo_pos = {bid: i for i, bid in enumerate(mir_rpo(mir))}
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

    consumer_ports: dict[ValueId, set[ReadPort]]
    producer_key: dict[ValueId, frozenset[Producer]]


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
        """
        The loop-invariant objective terms: the wide read-mux/write-select fan-in, or the boolean's degenerate one.
        """

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
        consumer_ports: dict[ValueId, set[ReadPort]] = {vid: set() for vid in ctx.values}
        for vid, inst in ctx.inst_of.items():
            use_node = ctx.mir.nodes[vid]
            if not isinstance(use_node, MirOperation):
                continue
            for pos, operand in enumerate(use_node.operands):
                if operand in ctx.values:
                    consumer_ports[operand].add((inst, pos))
        producer_key: dict[ValueId, frozenset[Producer]] = {vid: frozenset({"input"}) for vid in ctx.view.input_ids}
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


@dataclass(frozen=True, slots=True)
class _InterferenceBuilder:
    """
    Builds a bank's interference graph: the loop-invariant liveness facts (residency, result landings, definition
    blocks, phi-arm live-outs, in-flight defs) are fixed at construction, and the held MIR supplies the block CFG, so
    each coalescing attempt produces a graph by passing only what varies -- the boundary users, the per-block reads,
    and the residual installs.
    """

    mir: Mir
    makespan: dict[int, int]
    term_offset: dict[int, int]
    resident: frozenset[ValueId]
    op_landing: dict[ValueId, int]
    op_block: dict[ValueId, int]
    phi_block: dict[ValueId, int]
    arm_out: dict[int, frozenset[ValueId]]
    inflight_defs: dict[int, dict[ValueId, int]]

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
                makespan=self.makespan,
                term_offset=self.term_offset,
                resident=self.resident,
                op_landing=self.op_landing,
                op_block=self.op_block,
                phi_block=self.phi_block,
                reads=block_reads,
                boundary_users={b: frozenset(s) for b, s in boundary.items()},
                arm_out=self.arm_out,
                installs=install_facts,
                inflight_defs=self.inflight_defs,
            )
        )


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

    ret_block = mir.ret_block
    arm_out = phi_arm_out(mir, phi_nodes, values)
    boundary_base = bank.boundary_base(mir, values, ret_block)

    interference = _InterferenceBuilder(
        mir=mir,
        makespan=block_makespan,
        term_offset=block_term_offset,
        resident=frozenset({*view.input_ids, *state_read_nodes}),
        op_landing={
            vid: (
                bank.landing_cycle(commit)  # pooled: through the bank's writeback latch
                if isinstance(op_nodes[vid].operator, PooledHardwareOperator)
                else inline_landing_cycle(commit)  # inline: combinational array write, no writeback latch
            )
            for vid, commit in op_commit.items()
        },
        op_block=op_block,
        phi_block=phi_block,
        arm_out=arm_out,
        inflight_defs=block_inflight,
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
    coalesce_graph = interference.build(boundary_oracle, reads, {})
    candidate_arms = coalescable_arms(phi_nodes, values, bank.identity)
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
        assign, nreg, coalescing, conflict = coalesce_and_color(
            phi_nodes,
            phi_order,
            candidate_arms,
            pinned,
            {slot_reg[s.name] for s in slots if s.name not in coalesced},
            lambda residual: interference.build(boundary_final, reads_final, residual),
            ColorObjective(movable, obj_terms.consumer_ports, obj_terms.producer_key, fresh_start),
        )
        if conflict is not None:
            demoted = slot_by_reg.get(conflict)
            assert (
                demoted is not None and demoted not in forced_copy
            ), f"coloring conflict on register {conflict} not resolvable by backing a slot out of coalescing"
            forced_copy.add(demoted)
            continue
        # Dwell hazard: a coalesced slot register written by an entry-block cycle-0 op would be re-driven each idle
        # cycle during the accept dwell, corrupting carried state. Demote the slot to a copy-back -- reserving its
        # register so no tenant lands there -- and retry; _assert_entry_dwell_safe backstops.
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

    # Backstop: a non-coalesced slot register must carry nothing but its own live-in. A coalesced slot register IS
    # shared by its in-place live-out (and any phi arms merged onto it) and is skipped here.
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
) -> Allocation:
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
    copies: dict[int, list[FloatArmInstall]] = {}
    for vid, phi in float_mir.phi_nodes.items():
        for pred, value, sign in phi.arms:
            assert isinstance(sign, FloatSignControl)
            if (pred, vid) in float_alloc.coalesced:
                continue
            copies.setdefault(pred, []).append(FloatArmInstall(float_alloc.reg[vid], value, sign))

    bool_writes: dict[int, list[BoolArmInstall]] = {}
    for vid, phi in bool_mir.phi_nodes.items():
        for pred, value, inversion in phi.arms:
            assert isinstance(inversion, BoolInversion)
            if (pred, vid) in bool_alloc.coalesced:
                continue
            bool_writes.setdefault(pred, []).append(BoolArmInstall(bool_alloc.reg[vid], value, inversion))

    # A constant branch condition (e.g. a read-only boolean attribute, or a folded test) has no register of its own;
    # materialize it into a bool register written in the branching block so the next-PC decode can read it. The constant
    # is globally interned, so sibling branches sharing it reuse one register -- but the write must be emitted in EVERY
    # branching block that uses it, else a path reaching the branch through a block that did not write it reads a stale
    # register. (A later static-branch-folding pass would instead drop the dead arm; until then this keeps it correct.)
    bool_reg, nbreg = bool_alloc.reg, bool_alloc.nreg
    for block_id, cond in const_branch_conditions(mir, bool_mir).items():
        if cond not in bool_reg:
            bool_reg[cond] = nbreg
            nbreg += 1
        bool_writes.setdefault(block_id, []).append(BoolArmInstall(bool_reg[cond], cond, BoolInversion()))

    return Allocation(
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
