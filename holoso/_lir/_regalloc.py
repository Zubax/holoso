"""
Reach-aware register allocation over the software-pipelined (cycle-accurate) schedule.

Register-needing values are the input ports, persistent state-slot live-ins, and operator results (constants are
immediates, not registers). Whether two values may share a register is decided in the executing-step (hardware) frame,
the same frame as ``Lir.float_liveness`` and the write timeline: a value *lands* -- becomes readable in the array -- at
its write cycle (an operator result ``FETCH_LAG + 2`` after its commit, an input or a state live-in on cycle 1) and is
*last read* at its read cycle (an operand ``FETCH_LAG - 1`` after the consumer's issue, an output on the boundary step,
a non-coalesced live-out source on its writeback step). Two values may share a register when the older one's last read
strictly precedes the newer one's landing, ``R(a) < W(b)``; the read-first edge is folded into the landing, so a read
on a value's landing cycle resolves to that value, which is why the boundary is strict. This hardware-accurate rule
shares registers the older scheduler-frame rule ``last_use <= def_cycle`` left apart -- that rule was several cycles too
conservative because the read and write latches widen the real separation -- and is verified bit-exact against the RTL
by cosimulation. (The state-slot install scheduling below stays in the scheduler frame, which is frame-invariant for
deciding when a writeback may fire.)

Persistent state is a loop-carried dependence: each slot owns a dedicated register that is reset to its snapshot, read
for the live-in, and must hold the slot's live-out at the initiation boundary. When the live-out is an operator result
whose live range does not overlap the live-in, it is *coalesced* onto the slot register (the operator writes it
directly, no copy); otherwise the backend copies the live-out into the slot register at its install cycle: as early as
the old live-in is fully read and the source is available when the source is an ordinary register (which then frees for
reuse), the boundary at the latest. A coalesced slot register is itself reusable for unrelated temporaries during its
dead gap -- after the live-in is last read and before the live-out operator returns, with the tenant folded into the
slot's write select -- which sheds registers (and so narrows the regfile muxes) on stateful kernels without disturbing
the loop-carried value. A non-coalesced slot, installed by a standalone copy the backend cannot fold a tenant into,
stays reserved.

Unlike a CPU register allocator, the primary objective here is NOT to minimize the register count: flip-flops are
abundant on an FPGA and interconnect is scarce, so the cost that matters most is *steering* -- the fan-in of the
per-port read muxes and the per-register write selects of the sparse register file synthesized in the backend. The
primary objective is therefore total mux fan-in: ``sum_p max(0, |read-set(p)| - 1) + sum_r max(0, |writers(r)| - 1)``,
where a read port ``p`` is one operator ``(instance, operand-position)`` and ``writers(r)`` are the distinct producers
(operator instances plus the input-load) of the values placed in register ``r``. Two values read by the same port that
do not interfere are best placed in the same register so that port reaches one register, not two; values produced by
the same instance likewise want to share a register so its write port fans into one place.

Register count is a bounded *secondary* objective. A register costs flip-flops but no steering, so it is worth shedding
only when doing so widens a write select modestly: the allocator colors twice -- once reach-minimal, once compacting
dead registers up to a write-select cap -- and keeps whichever minimizes ``reach + _REG_PRICE * registers`` (see those
constants). The reach-minimal coloring is always a candidate, so trading for fewer registers never raises steering by
more than the price paid per register freed.

The allocator is a port-affinity-biased graph coloring (a linear scan whose register choice minimizes the marginal
increase in total mux fan-in), refined by simulated annealing. Input ports are pinned to the unique low registers
``0..nload-1`` so the step-0 parallel-load lanes map one-to-one onto module input ports; the state-slot registers sit
directly above them; operation results may reuse an input register once its value is dead, or a coalesced slot register
during its dead gap. There is no spilling: a value the cap cannot place by reuse simply opens a new register.
"""

from collections import Counter
from dataclasses import dataclass
import logging
import os

import numpy as np
from scipy.optimize import dual_annealing

from .._hir import ValueId
from .._mir import MirFloatConst, MirFloatInput, MirFloatOperation, MirFloatStateRead, MirFloatView
from .._operators import FloatSignControl
from ._ir import FloatOperatorInstance, boundary_step, copy_step_cycle, landing_cycle, read_latch_cycle

# Read port identity (operator instance + operand position) and write-source identity (an instance, the input load, or
# a per-slot state writer -- the writeback copy / reset that drives a slot register).
type _Port = tuple[FloatOperatorInstance, int]
type _Producer = FloatOperatorInstance | str
_INPUT_LOAD: _Producer = "input_load"


def _state_writer(name: str) -> _Producer:
    return f"state:{name}"


def _admits_tenant(slot_gaps: dict[int, tuple[int, int]], reg: int, read_cycle: int) -> bool:
    """
    Whether ``reg`` admits a tenant whose last read is ``read_cycle``. A non-slot register (``gap is None``) always
    admits; a coalesced slot admits only a tenant dying before its live-out lands (``read_cycle < gap[1]``); a reserved
    non-coalesced slot carries an empty gap (``gap[1] == gap[0]``) that admits nothing.
    """
    gap = slot_gaps.get(reg)
    return gap is None or read_cycle < gap[1]


# Budget for the SciPy dual-annealing refinement. It only polishes an already-valid greedy seed (and is a no-op when
# the seed is already at the reach floor), so the function-evaluation cap keeps build time bounded; raise it to trade
# build time for a deeper search. The environment override is for testing only; eventually we might add an API handle.
_REFINE_MAXITER = int(os.getenv("HOLOSO_REGALLOC_EFFORT", "5000"))

# Balance of reach against register count, layered on the hardware-accurate liveness. ``_WRITE_SELECT_CAP`` bounds how
# wide a per-register write select the compaction may build (the ":1" of the select -- the number of distinct producers
# sharing a register); reuse never widens a read mux beyond a fresh register, so the write select is the only mux a
# compacted coloring can grow. ``_REG_PRICE`` is what one freed register is worth in mux-arm units: the allocator keeps
# whichever coloring minimizes ``reach + _REG_PRICE * registers``. The price bounds the spectrum -- price 0 stays
# reach-minimal (registers only break a reach tie), price -> inf takes every register the cap can free, and a fractional
# price compacts only when a register comes near reach-free. The default 2.0 sheds one register off each bundled small
# kernel with f_max held and selects no wider than 2:1.
#
# These two are EXPERIMENTAL / ADVANCED knobs with no user-facing surface; they are read once from the environment for
# tuning and may be promoted to proper parameters if a real need arises.
_WRITE_SELECT_CAP = int(os.getenv("HOLOSO_WRITE_SELECT_CAP", "2"))
_REG_PRICE = float(os.getenv("HOLOSO_REG_PRICE", "2.0"))
_NO_CAP = 1 << 30  # an effectively unbounded write-select budget, used for the reach-minimal coloring
_INFEASIBLE_COST = 1e18  # annealing penalty for an undecodable point (far above any real mux-fan-in objective)

_logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FloatAllocation:
    assign: dict[ValueId, int]  # register-needing value -> register index
    nreg: int
    state_regs: dict[str, int]  # state-slot name -> its dedicated persistent register index
    install_cycles: dict[str, int]  # state-slot name -> scheduler-frame cycle its live-out lands in the slot register


def _operation(mir: MirFloatView, vid: ValueId) -> MirFloatOperation:
    return mir.operation_nodes[vid]


def _objective(
    assign: dict[ValueId, int],
    consumer_ports: dict[ValueId, set[_Port]],
    producer_key: dict[ValueId, _Producer],
) -> int:
    """Total sparse-regfile mux fan-in: read-mux fan-in across ports plus write-select fan-in across registers."""
    members: dict[int, list[ValueId]] = {}
    for vid, reg in assign.items():
        members.setdefault(reg, []).append(vid)
    port_regs: dict[_Port, set[int]] = {}
    write = 0
    for reg, vids in members.items():
        writers: set[_Producer] = set()
        for vid in vids:
            writers.add(producer_key[vid])
            for port in consumer_ports[vid]:
                port_regs.setdefault(port, set()).add(reg)
        write += max(0, len(writers) - 1)
    read = sum(max(0, len(regs) - 1) for regs in port_regs.values())
    return read + write


def _assert_no_interference(
    assign: dict[ValueId, int], write_hw: dict[ValueId, int], read_hw: dict[ValueId, int]
) -> None:
    """Backstop: two values sharing a register must be disjoint in hardware (read-first: ``R(a) < W(b)`` is OK)."""
    members: dict[int, list[ValueId]] = {}
    for vid, reg in assign.items():
        members.setdefault(reg, []).append(vid)
    for reg, vids in members.items():
        for i, a in enumerate(vids):
            for b in vids[i + 1 :]:
                if not (read_hw[a] < write_hw[b] or read_hw[b] < write_hw[a]):
                    raise AssertionError(f"register {reg} shared by interfering values {a} and {b}")


def allocate_float(
    mir: MirFloatView,
    issue_cycle: dict[ValueId, int],
    inst_of: dict[ValueId, FloatOperatorInstance],
    makespan: int,
) -> FloatAllocation:
    present_cycle = makespan + 1

    def def_cycle_of(vid: ValueId) -> int:
        node = mir.nodes[vid]
        if isinstance(node, MirFloatOperation):
            return issue_cycle[vid] + node.operator.latency
        return 0  # an input port (accept edge) or a state read (already resident from the previous initiation)

    input_values = [vid for vid in mir.input_ids if isinstance(mir.nodes[vid], MirFloatInput)]
    state_values = list(mir.state_read_nodes)  # live-in reads, one per slot that the method reads before writing
    operation_values = [vid for vid in issue_cycle if isinstance(mir.nodes[vid], MirFloatOperation)]
    reg_values: list[ValueId] = [*input_values, *state_values, *operation_values]
    def_cycle = {vid: def_cycle_of(vid) for vid in reg_values}
    last_use: dict[ValueId, int] = {vid: def_cycle[vid] for vid in reg_values}

    # Per-value consumer read ports (which operator operand positions read it) and its producer. Outputs and persisted
    # state live-outs read the register array directly (not through a read port), so they extend liveness but add no
    # port reach.
    consumer_ports: dict[ValueId, set[_Port]] = {vid: set() for vid in reg_values}
    producer_key: dict[ValueId, _Producer] = {vid: _INPUT_LOAD for vid in input_values}
    producer_key.update({vid: _state_writer(_state_read_name(mir, vid)) for vid in state_values})
    producer_key.update({vid: inst_of[vid] for vid in operation_values})
    for vid in operation_values:
        op = _operation(mir, vid)
        for pos, operand in enumerate(op.operands):
            if isinstance(mir.nodes[operand], MirFloatConst):
                continue
            last_use[operand] = max(last_use[operand], issue_cycle[vid])
            consumer_ports[operand].add((inst_of[vid], pos))
    for out in mir.outputs:
        if out.value in last_use and not isinstance(mir.nodes[out.value], MirFloatConst):
            last_use[out.value] = max(last_use[out.value], present_cycle)

    nload = len(input_values)
    read_of_slot = {_state_read_name(mir, vid): vid for vid in state_values}
    state_regs: dict[str, int] = {slot.name: nload + i for i, slot in enumerate(mir.state_slots)}

    # Pin inputs to the low load lanes and each state slot's live-in to its dedicated persistent register.
    pinned: dict[ValueId, int] = {vid: reg for reg, vid in enumerate(input_values)}
    for name, reg in state_regs.items():
        r_in = read_of_slot.get(name)
        if r_in is not None:
            pinned[r_in] = reg

    # Coalesce a slot's live-out onto its register when it is an unconditioned operator result whose live range does
    # not overlap the live-in; otherwise leave it for the backend to copy (and sign-condition) at its install cycle.
    for slot in mir.state_slots:
        live_out = slot.live_out
        if slot.sign != FloatSignControl() or live_out in pinned:
            continue
        if not isinstance(mir.nodes[live_out], MirFloatOperation):
            continue
        r_in = read_of_slot.get(slot.name)
        if r_in is None or last_use[r_in] <= def_cycle[live_out]:
            pinned[live_out] = state_regs[slot.name]

    # Schedule each slot's writeback and extend its source's last use to that install cycle, after which the source
    # register is free for unrelated values. A non-coalesced live-out is copied at the boundary by default; but when its
    # source is an ordinary (non-slot) register and no other slot reads this slot's register, the copy can fire as soon
    # as the old live-in is fully read and the source is available (read-first lets the copy's write share the cycle of
    # the live-in's last read). Coalesced, constant, chained (source is another slot), and read-by-another slots stay at
    # the boundary, where holding the source to the present cycle is required and frees nothing.
    tapped_by_other = {
        _state_read_name(mir, slot.live_out)
        for slot in mir.state_slots
        if isinstance(mir.nodes[slot.live_out], MirFloatStateRead) and _state_read_name(mir, slot.live_out) != slot.name
    }
    install_cycles: dict[str, int] = {}
    for slot in mir.state_slots:
        live_out = slot.live_out
        node = mir.nodes[live_out]
        r_in = read_of_slot.get(slot.name)
        coalesced = pinned.get(live_out) == state_regs[slot.name]
        early = (
            not coalesced and isinstance(node, (MirFloatInput, MirFloatOperation)) and slot.name not in tapped_by_other
        )
        if early:
            cycle = def_cycle[live_out] + 1  # read-first: the copy reads a value committed strictly before its read
            if r_in is not None:
                cycle = max(cycle, last_use[r_in])  # do not overwrite the old live-in before its last read
            cycle = min(cycle, present_cycle)
        else:
            cycle = present_cycle
        install_cycles[slot.name] = cycle
        if not isinstance(node, MirFloatConst):
            last_use[live_out] = max(last_use[live_out], cycle)

    # WAR backstop: each slot's new value must land no earlier than its live-in's last read, so the old value is fully
    # consumed first (read-first allows equality). This holds by construction -- coalescing requires it (above) and the
    # install cycle is computed as >= last_use[r_in] -- so the assert is a guard that trips loudly if a future change
    # weakens either path, rather than letting a copy scheduled too early silently corrupt the carried-over state.
    for slot in mir.state_slots:
        r_in = read_of_slot.get(slot.name)
        if r_in is None or slot.live_out == r_in:
            continue  # write-only (no live-in), or a no-op writeback of the live-in itself: no new value lands
        coalesced = pinned.get(slot.live_out) == state_regs[slot.name]
        new_value_cycle = def_cycle[slot.live_out] if coalesced else install_cycles[slot.name]
        assert last_use[r_in] <= new_value_cycle, (
            f"state slot {slot.name!r} write-after-read violated: live-in last read at {last_use[r_in]} "
            f"exceeds new-value write cycle {new_value_cycle}"
        )

    # Hardware-frame liveness for register interference. The scheduler-frame def_cycle/last_use above drive only the
    # state-slot install scheduling (which is frame-invariant); register sharing is decided here in the executing-step
    # frame, mirroring Lir.float_liveness and float_write_timeline. A value lands -- becomes readable in the array -- at
    # its write cycle and is last read at its read cycle: an operator result lands FETCH_LAG+2 after its commit and an
    # operand is read FETCH_LAG-1 after issue; inputs and state live-ins are resident from cycle 1; an output stays
    # resident through the boundary; a non-coalesced slot live-out source is read by the writeback copy on its install
    # step. Two values may share a register when the older one's last read strictly precedes the newer one's landing,
    # R(a) < W(b) -- the read-first edge is folded into the landing, so the model resolves a read on a landing cycle to
    # that value, hence the strict boundary. This is less conservative than the scheduler-frame rule it replaces and
    # frees registers the hardware can actually share.
    present_hw = boundary_step(makespan)  # initiation interval: the last result lands here, outputs are resident here
    write_hw: dict[ValueId, int] = {
        vid: (landing_cycle(def_cycle[vid]) if isinstance(mir.nodes[vid], MirFloatOperation) else 1)
        for vid in reg_values
    }
    read_hw: dict[ValueId, int] = dict(write_hw)  # a value with no reads frees its register on its own landing cycle
    for vid in operation_values:
        op_read = read_latch_cycle(issue_cycle[vid])
        for operand in _operation(mir, vid).operands:
            if operand in read_hw:  # constants are immediates, not register reads
                read_hw[operand] = max(read_hw[operand], op_read)
    for out in mir.outputs:
        if out.value in read_hw and not isinstance(mir.nodes[out.value], MirFloatConst):
            read_hw[out.value] = max(read_hw[out.value], present_hw)
    for slot in mir.state_slots:
        src = slot.live_out
        if src in read_hw and not isinstance(mir.nodes[src], MirFloatConst):
            read_hw[src] = max(read_hw[src], copy_step_cycle(install_cycles[slot.name]))

    # Each slot register is reusable for temporaries during its dead gap -- after the live-in is last read and before
    # the live-out re-occupies it. Only a COALESCED slot is opened: the backend folds a tenant into the slot register's
    # write select (the operator results writing it share one select), so the live-out landing is the reblock. A
    # non-coalesced slot is installed by a standalone copy the backend cannot fold a tenant into, so it stays reserved
    # (an empty gap, ``reblock == gap_start``, admits no tenant). A write-only slot's register is free from the start.
    # ``slot_gaps[reg] = (gap_start, reblock)``; every slot register appears so it is excluded from generic reuse.
    slot_gaps: dict[int, tuple[int, int]] = {}
    for slot in mir.state_slots:
        reg = state_regs[slot.name]
        r_in = read_of_slot.get(slot.name)
        gap_start = read_hw[r_in] if r_in is not None else 0
        coalesced = pinned.get(slot.live_out) == reg
        reblock = write_hw[slot.live_out] if coalesced else gap_start
        slot_gaps[reg] = (gap_start, reblock)

    movable = [vid for vid in operation_values if vid not in pinned]
    fresh_start = nload + len(mir.state_slots)
    cap, price = _WRITE_SELECT_CAP, _REG_PRICE

    def greedy_seed(compact: bool, budget: int) -> tuple[dict[ValueId, int], int]:
        seed = _greedy(
            movable, pinned, slot_gaps, fresh_start, write_hw, read_hw, consumer_ports, producer_key, compact, budget
        )
        return seed, max((max(seed.values()) + 1) if seed else 0, fresh_start)

    def refined(seed: dict[ValueId, int], nreg: int, budget: int) -> dict[ValueId, int]:
        return _refine(seed, nreg, movable, slot_gaps, write_hw, read_hw, consumer_ports, producer_key, budget)

    base_seed, base_nreg = greedy_seed(compact=False, budget=_NO_CAP)
    comp_seed, comp_nreg = greedy_seed(compact=True, budget=cap)
    # Balance reach against register count. The reach-minimal coloring is the default, so the result never regresses on
    # the proxy; the compacted coloring (write selects widened up to the budget to free flip-flops) uses fewer registers
    # but more reach, so it is adopted only when it strictly lowers reach + price*registers (fewer registers breaking a
    # tie). At price 0 it wins only by matching the reach floor with fewer registers, so the result stays reach-minimal;
    # as price grows it buys registers back at up to that many mux arms each.
    assign, nreg = refined(base_seed, base_nreg, _NO_CAP), base_nreg
    if comp_nreg < base_nreg:
        comp_assign = refined(comp_seed, comp_nreg, cap)
        base_score = (_objective(assign, consumer_ports, producer_key) + price * base_nreg, base_nreg)
        comp_score = (_objective(comp_assign, consumer_ports, producer_key) + price * comp_nreg, comp_nreg)
        if comp_score < base_score:
            assign, nreg = comp_assign, comp_nreg
    _assert_no_interference(assign, write_hw, read_hw)
    # The backend can only fold a tenant into a coalesced slot's write select; a non-coalesced slot register, installed
    # by a standalone copy, must carry nothing but its own live-in. The empty gap above guarantees this -- the assert
    # trips loudly if a future change opens a non-coalesced gap the emitter would silently drop.
    for slot in mir.state_slots:
        reg = state_regs[slot.name]
        if pinned.get(slot.live_out) == reg:
            continue
        r_in = read_of_slot.get(slot.name)
        occupants = [vid for vid, r in assign.items() if r == reg and vid != r_in]
        assert (
            not occupants
        ), f"non-coalesced slot register {reg} ({slot.name!r}) has non-reserved occupants {occupants}"
    _logger.info(
        "Float regalloc: values=%d input_pins=%d state_slots=%d cap=%d price=%g base_reg=%d comp_reg=%d registers=%d",
        len(reg_values),
        len(input_values),
        len(mir.state_slots),
        cap,
        price,
        base_nreg,
        comp_nreg,
        nreg,
    )
    return FloatAllocation(assign=assign, nreg=nreg, state_regs=state_regs, install_cycles=install_cycles)


def _state_read_name(mir: MirFloatView, vid: ValueId) -> str:
    node = mir.nodes[vid]
    assert isinstance(node, MirFloatStateRead)
    return node.name


def _greedy(
    movable: list[ValueId],
    pinned: dict[ValueId, int],
    slot_gaps: dict[int, tuple[int, int]],
    fresh_start: int,
    write_hw: dict[ValueId, int],
    read_hw: dict[ValueId, int],
    consumer_ports: dict[ValueId, set[_Port]],
    producer_key: dict[ValueId, _Producer],
    compact: bool,
    cap: int,
) -> dict[ValueId, int]:
    """
    Port-affinity-biased linear scan. With ``compact`` false it is reach-minimal: each value takes the register (a dead
    one or a fresh one) of least marginal mux growth, opening a register only when reuse would actually cost steering.
    With ``compact`` true it instead prefers any dead register whose write select stays within ``cap`` over a fresh one,
    freeing flip-flops at the cost of a wider write select; ``cap`` is ignored when ``compact`` is false.
    """
    assign: dict[ValueId, int] = {}
    reg_ports: dict[int, set[_Port]] = {}
    reg_writers: dict[int, set[_Producer]] = {}
    port_reach: Counter[_Port] = Counter()  # registers each read port currently reaches

    def place(vid: ValueId, reg: int) -> None:
        assign[vid] = reg
        ports = reg_ports.setdefault(reg, set())
        for port in consumer_ports[vid]:
            if port not in ports:
                ports.add(port)
                port_reach[port] += 1
        reg_writers.setdefault(reg, set()).add(producer_key[vid])

    def marginal_cost(vid: ValueId, reg: int) -> int:
        # Adding a register to a port that already reaches >=1 register grows that port's mux by one (the first
        # register a port reaches is free); likewise the first writer of a register is free, each further one costs one.
        ports: frozenset[_Port] | set[_Port] = reg_ports.get(reg, frozenset())
        writers: frozenset[_Producer] | set[_Producer] = reg_writers.get(reg, frozenset())
        read = sum(1 for port in consumer_ports[vid] if port not in ports and port_reach[port] >= 1)
        write = 1 if (producer_key[vid] not in writers and len(writers) >= 1) else 0
        return read + write

    def writers_after(reg: int, producer: _Producer) -> int:
        return len(reg_writers.get(reg, frozenset()) | {producer})

    def candidate_key(vid: ValueId, reg: int, is_fresh: int) -> tuple[int, int, int]:
        # Compaction sorts reuse ahead of marginal mux growth (is_fresh leads), so any admissible dead register beats a
        # fresh one; reach minimization sorts mux growth first and uses reuse only to break ties. ``is_fresh`` (0 dead,
        # 1 fresh) then ``reg`` keep the order total either way.
        cost = marginal_cost(vid, reg)
        return (is_fresh, cost, reg) if compact else (cost, is_fresh, reg)

    # Pinned values (inputs, state live-ins, coalesced live-outs) take their fixed registers first.
    for vid, reg in sorted(pinned.items(), key=lambda item: (item[1], item[0])):
        place(vid, reg)
    # An input lane frees once its value is dead; a slot register frees during its dead gap (seeded at the gap start;
    # the per-candidate reblock check below stops a tenant from overlapping the returning live-out).
    active: list[tuple[int, int]] = [(read_hw[vid], reg) for vid, reg in pinned.items() if reg not in slot_gaps]
    active.extend((gap_start, reg) for reg, (gap_start, _) in slot_gaps.items())
    free: list[int] = []
    next_reg = fresh_start

    for vid in sorted(movable, key=lambda v: (write_hw[v], v)):
        w = write_hw[vid]
        retained: list[tuple[int, int]] = []
        for r, reg in active:
            if r < w:  # the occupant's last read precedes vid's landing, so the register is free for vid (read-first)
                free.append(reg)
            else:
                retained.append((r, reg))
        active = retained
        producer = producer_key[vid]
        # Compaction and reach minimization share one ranking and differ only in what the register choice optimizes
        # first (see ``candidate_key``). Reach-minimal opens a register only when reuse would cost steering; compaction
        # admits only dead registers whose write select stays within the budget and takes any of them over a fresh one,
        # so it frees flip-flops at the cost of a wider write select. A fresh register is always the fallback.
        best_reg, best_fresh = next_reg, True
        best_key = candidate_key(vid, next_reg, 1)
        for reg in free:
            if compact and writers_after(reg, producer) > cap:
                continue
            if not _admits_tenant(slot_gaps, reg, read_hw[vid]):  # tenant must die before the slot's live-out returns
                continue
            key = candidate_key(vid, reg, 0)
            if key < best_key:
                best_key, best_reg, best_fresh = key, reg, False
        if best_fresh:
            next_reg += 1
        else:
            free.remove(best_reg)
        place(vid, best_reg)
        active.append((read_hw[vid], best_reg))
    return assign


def _refine(
    seed: dict[ValueId, int],
    nreg: int,
    movable: list[ValueId],
    slot_gaps: dict[int, tuple[int, int]],
    write_hw: dict[ValueId, int],
    read_hw: dict[ValueId, int],
    consumer_ports: dict[ValueId, set[_Port]],
    producer_key: dict[ValueId, _Producer],
    cap: int,
) -> dict[ValueId, int]:
    """
    Refine the greedy assignment with SciPy's simulated annealing (``scipy.optimize.dual_annealing``).

    Each movable operation value gets a continuous coordinate in ``[0, nreg)``; a decode in landing order maps it to its
    preferred register, repairing interference by scanning to the next register whose occupant's last read precedes this
    value's landing and whose write select stays within ``cap`` (a last-resort fallback may exceed ``cap`` when no other
    register is free, since decode cannot open a fresh one; see the decode body). Pinned values (inputs, state live-ins,
    coalesced live-outs) are held fixed; a slot register is reusable only inside its dead gap (free from the gap start,
    re-blocked before its live-out returns). Every evaluated point is thus a valid (interference-free) coloring reusing
    only the ``nreg`` seed registers, and the annealer minimizes the mux-fan-in objective. The greedy seed is the
    starting point and the best point seen is kept, so the pass can only improve on the seed.
    """
    order = sorted(movable, key=lambda vid: (write_hw[vid], vid))
    if len(order) < 2 or nreg <= 1:
        return seed
    op_set = set(order)
    pinned = [(vid, reg) for vid, reg in seed.items() if vid not in op_set]

    def decode(coords: np.ndarray) -> dict[ValueId, int] | None:
        assign: dict[ValueId, int] = {}
        # free_after[reg] = max last read of any value placed in reg; the register is free for a value landing at w iff
        # it is < w (read-first). writers[reg] tracks the running producer set so the decode honors the write-select
        # budget the greedy did, rather than re-growing a select past the cap to shave a read arm. Landing order keeps
        # the free check O(1).
        free_after = [-1] * nreg
        writers: list[set[_Producer]] = [set() for _ in range(nreg)]
        for vid, reg in pinned:
            assign[vid] = reg
            free_after[reg] = max(free_after[reg], read_hw[vid])
            writers[reg].add(producer_key[vid])
        for reg, (gap_start, _) in slot_gaps.items():  # a slot register is free for tenants from its gap start, not
            free_after[reg] = gap_start  # the live-out's read; the reblock (per candidate) guards the gap's end
        for index, vid in enumerate(order):
            w = write_hw[vid]
            producer = producer_key[vid]
            pref = min(nreg - 1, max(0, int(coords[index])))
            # Primary scan anchored at pref: the first free register whose write select stays within budget, so x0
            # decodes back to the greedy seed. Decode must place every value within the fixed nreg registers -- it
            # cannot open a fresh one as the greedy can -- so when none is within budget it falls back to the free
            # register whose select grows least; this is the only path that may push a write select past the cap.
            chosen = -1
            for offset in range(nreg):  # primary: a register free at w, within budget, and within its gap if a slot
                reg = (pref + offset) % nreg
                admissible = free_after[reg] < w and _admits_tenant(slot_gaps, reg, read_hw[vid])
                if admissible and len(writers[reg] | {producer}) <= cap:
                    chosen = reg
                    break
            if chosen < 0:
                free = [r for r in range(nreg) if free_after[r] < w and _admits_tenant(slot_gaps, r, read_hw[vid])]
                if not free:
                    # An infeasible annealer point: an earlier placement consumed the only register this value could
                    # occupy. With gap admission the feasible set the unconstrained scan assumed nonempty can be empty
                    # (a free register may be a slot register the value's gap rejects). Reject the point; the seed and
                    # every accepted point stay valid, so the search still only improves.
                    return None
                chosen = min(free, key=lambda reg: (len(writers[reg] | {producer}), reg))
            assign[vid] = chosen
            free_after[chosen] = read_hw[vid]
            writers[chosen].add(producer)
        return assign

    _logger.info("Regalloc refinement effort: %d", _REFINE_MAXITER)
    if _REFINE_MAXITER <= 0:
        return seed

    best = seed
    best_cost = _objective(seed, consumer_ports, producer_key)
    if best_cost == 0:  # reach floor is structurally 0: the seed is globally optimal, so the anneal cannot improve it
        return best

    def cost(coords: np.ndarray) -> float:
        nonlocal best, best_cost
        candidate = decode(coords)
        if candidate is None:
            return _INFEASIBLE_COST  # an undecodable point: penalize so the annealer steers away; never beats `best`
        value = _objective(candidate, consumer_ports, producer_key)
        if value < best_cost:
            _logger.debug(
                "Annealing new best: cost=%d previous=%d registers=%d movable_values=%d",
                value,
                best_cost,
                nreg,
                len(order),
            )
            best, best_cost = candidate, value
        return float(value)

    x0 = np.array([float(seed[vid]) for vid in order])
    bounds = [(0.0, nreg - 1e-6)] * len(order)
    dual_annealing(cost, bounds, x0=x0, seed=0, maxiter=_REFINE_MAXITER, maxfun=_REFINE_MAXITER, no_local_search=True)
    return best
