"""
CFG liveness and hardware-frame register interference for the unified register allocator.

Register sharing across a control-flow graph is decided by an interference graph: two register-needing values may share
a register exactly when they are never simultaneously live on any execution. The graph is built in two stages.

First, classic backward dataflow over the block CFG yields per-block live-in / live-out sets, with SSA phi semantics: a
phi result is defined at the head of its block, and each phi arm value is live out of the *predecessor* it arrives from
(a phi is lowered as a parallel copy at the predecessor's tail). The fixpoint converges over back-edges, so loop-carried
values stay live across the whole loop body.

Second, within each block every live value is given a half-open residence interval in that block's executing-step
(hardware) frame -- the same frame as :attr:`Lir.reg_liveness`, the write timeline, and the numerical model -- using the
shared cycle helpers. A value resident from a predecessor (live-in, or a phi result) lands on the block's first step; a
value defined by an in-block operator lands on its bank's landing cycle (the wide bank pays the writeback latch, the
boolean bank only the read-first edge); a value that is live out of the block (or read by the block's boundary -- an
output, a branch condition, a state live-out, or a phi-arm copy) stays resident through the block boundary; and a phi
result additionally occupies its register at the tail of every arm predecessor, where its install copy physically
writes it one step before the boundary -- deliberately also foreclosing the phi sharing a register with its own arm
values, although that same-step read-first self-copy would be harmless (conservatism a future copy-coalescing pass
can reclaim, not a safety invariant). Two values interfere when their intervals overlap in *some* block under the
read-first rule ``R(a) < W(b)``. Values that live entirely within mutually-exclusive blocks (the two arms of an
``if``) share no block, so they never interfere -- path-awareness falls out of the per-block quantification with no
explicit reasoning about which arms are exclusive.

This module is bank-agnostic: the caller supplies a :class:`BankLiveness` describing one register family (the wide float
bank or the 1-bit boolean bank) and receives the symmetric interference adjacency over that bank's values.
"""

from dataclasses import dataclass, field

from .._hir import ValueId
from ._ir import boundary_step, copy_step_cycle


@dataclass(frozen=True, slots=True)
class BankLiveness:
    """
    One register family's liveness inputs, in the per-block-drained schedule (every block's results land before the
    next block fetches, so a cross-block live-in is resident from its block's first step).

    All cycles are block-local in the executing-step frame (block start is step 1). ``op_landing`` is each in-block
    definition's bank-true landing cycle (the caller computes it with the shared ``_ir`` cycle helper of its bank:
    the latched wide bank pays the writeback latch, the latch-free boolean bank only the read-first edge). ``reads``
    carries every in-block operand read with its hardware read cycle; ``boundary_users`` carries values consumed at a
    block's boundary that are not otherwise live out (outputs and state live-outs at the Ret block, a branch
    condition); ``arm_out`` carries, per block, the phi-arm values a successor's phi takes from that block (each is
    live out of it); ``installs`` carries, per block, the phi-result values whose install copy WRITES their register
    at this block's tail (one entry per arm predecessor) -- the install is a real write the emitter performs one step
    before the block boundary, so the phi register must be modeled as occupied there or it could clobber a value
    still read at the boundary (e.g. the very branch condition selecting the successor).
    """

    blocks: list[int]
    entry: int
    succ: dict[int, list[int]]
    # Block makespan, INCLUSIVE of the +1 install step when the block carries any install -- the same value the
    # layout uses -- so an install's copy step is exactly ``copy_step_cycle(makespan[block])``.
    makespan: dict[int, int]
    resident: frozenset[ValueId]  # inputs and state live-ins: resident from the start, defined at the entry
    op_landing: dict[ValueId, int]  # op-result value -> its bank-true landing cycle in its def block (block-local)
    op_block: dict[ValueId, int]  # op-result value -> its def block
    phi_block: dict[ValueId, int]  # phi-result value -> the block whose head defines it
    reads: dict[int, list[tuple[ValueId, int]]] = field(default_factory=dict)  # block -> [(value, read cycle)]
    boundary_users: dict[int, frozenset[ValueId]] = field(default_factory=dict)  # block -> boundary-read values
    arm_out: dict[int, frozenset[ValueId]] = field(default_factory=dict)  # block -> phi-arm values live out of it
    installs: dict[int, frozenset[ValueId]] = field(default_factory=dict)  # block -> phi dests installed at its tail


@dataclass(frozen=True, slots=True)
class _Live:
    live_in: dict[int, set[ValueId]]
    live_out: dict[int, set[ValueId]]


def _defs(bank: BankLiveness) -> dict[int, set[ValueId]]:
    """Values defined in each block: in-block operator results, the block's phi results, plus residents at the entry."""
    defs: dict[int, set[ValueId]] = {b: set() for b in bank.blocks}
    for vid, block in bank.op_block.items():
        defs[block].add(vid)
    for vid, block in bank.phi_block.items():
        defs[block].add(vid)
    defs[bank.entry].update(bank.resident)
    return defs


def _uses(bank: BankLiveness, defs: dict[int, set[ValueId]]) -> dict[int, set[ValueId]]:
    """Upward-exposed uses per block: values read in the block (or at its boundary) that the block does not define."""
    uses: dict[int, set[ValueId]] = {b: set() for b in bank.blocks}
    for block in bank.blocks:
        local = defs[block]
        for vid, _cycle in bank.reads.get(block, []):
            if vid not in local:
                uses[block].add(vid)
        for vid in bank.boundary_users.get(block, frozenset()):
            if vid not in local:
                uses[block].add(vid)
    return uses


def _dataflow(bank: BankLiveness, defs: dict[int, set[ValueId]], uses: dict[int, set[ValueId]]) -> _Live:
    """Backward liveness fixpoint with phi semantics (arm values live out of their predecessor)."""
    phi_defs: dict[int, set[ValueId]] = {b: set() for b in bank.blocks}
    for vid, block in bank.phi_block.items():
        phi_defs[block].add(vid)
    live_in: dict[int, set[ValueId]] = {b: set() for b in bank.blocks}
    live_out: dict[int, set[ValueId]] = {b: set() for b in bank.blocks}
    changed = True
    while changed:
        changed = False
        for block in reversed(bank.blocks):  # a postorder-ish sweep converges quickly; the fixpoint guarantees it
            out: set[ValueId] = set(bank.arm_out.get(block, frozenset()))
            for succ in bank.succ[block]:
                out |= live_in[succ] - phi_defs[succ]
            new_in = uses[block] | (out - defs[block])
            if out != live_out[block] or new_in != live_in[block]:
                live_out[block] = out
                live_in[block] = new_in
                changed = True
    return _Live(live_in=live_in, live_out=live_out)


def compute_interference(bank: BankLiveness) -> dict[ValueId, set[ValueId]]:
    """
    Build the symmetric register-interference adjacency for one bank from per-block hardware-frame residence intervals.
    """
    defs = _defs(bank)
    uses = _uses(bank, defs)
    live = _dataflow(bank, defs, uses)

    all_values: set[ValueId] = set(bank.resident) | set(bank.op_block) | set(bank.phi_block)
    interferes: dict[ValueId, set[ValueId]] = {vid: set() for vid in all_values}

    for block in bank.blocks:
        boundary = boundary_step(bank.makespan[block])
        installed = bank.installs.get(block, frozenset())
        live_set = live.live_in[block] | defs[block] | installed
        # A value's residence in this block: it lands on the block's first step when it is resident, a phi result, or a
        # live-in carried from a predecessor; on its operator's landing cycle when defined here; and on the install
        # step (one before the boundary) when its only presence is a phi install at this block's tail. It dies on its
        # last in-block read, extended to the boundary when it is live out, consumed at the boundary, or installed
        # here (the installed value must survive into the successor).
        write_at: dict[ValueId, int] = {}
        read_at: dict[ValueId, int] = {}
        for vid in live_set:
            if vid in bank.op_block and bank.op_block[vid] == block and vid not in live.live_in[block]:
                w = bank.op_landing[vid]
            elif vid in installed and vid not in live.live_in[block] and vid not in defs[block]:
                # The install fires on the copy step of the block's install-inclusive makespan, one step before the
                # boundary (see the ``makespan`` field contract).
                w = copy_step_cycle(bank.makespan[block])
            else:
                w = 1
            write_at[vid] = w
            read_at[vid] = w
        for vid, cycle in bank.reads.get(block, []):
            if vid in read_at:
                read_at[vid] = max(read_at[vid], cycle)
        boundary_users = bank.boundary_users.get(block, frozenset())
        for vid in live_set:
            if vid in live.live_out[block] or vid in boundary_users or vid in installed:
                read_at[vid] = max(read_at[vid], boundary)
        members = sorted(live_set)
        for i, a in enumerate(members):
            wa, ra = write_at[a], read_at[a]
            for b in members[i + 1 :]:
                # Read-first: a and b may share a register iff one's last read strictly precedes the other's landing.
                if not (ra < write_at[b] or read_at[b] < wa):
                    interferes[a].add(b)
                    interferes[b].add(a)
    return interferes
