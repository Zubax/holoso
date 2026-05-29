"""
Reach-aware register allocation over the software-pipelined (cycle-accurate) schedule.

Register-needing values are the input ports and operator results (constants are immediates, not registers). A value
is *defined* (written into its register) at its commit cycle -- ``issue_cycle + latency`` for an op, cycle 0 for an
input (the accept edge) -- and *last used* at the latest cycle it is read: the issue cycle of its last consuming op,
or the output-presentation cycle ``makespan + 1`` if it drives an output. Two values may share a register when the
older one's last use is no later than the newer one's definition cycle (``last_use <= def_cycle``); this is sound
because the register file is read-first (a read on the definition cycle still returns the old value) and the read and
write latches only widen that separation, so the rule is conservative.

Unlike a CPU register allocator, the objective here is NOT to minimize the register count: flip-flops are abundant on
an FPGA and interconnect is scarce, so the cost that matters is *steering* -- the fan-in of the per-port read muxes and
the per-register write selects of the sparse register file synthesized in the backend. We therefore minimize total
mux fan-in: ``sum_p max(0, |read-set(p)| - 1) + sum_r max(0, |writers(r)| - 1)``, where a read port ``p`` is one
operator ``(instance, operand-position)`` and ``writers(r)`` are the distinct producers (operator instances plus the
input-load) of the values placed in register ``r``. Two values read by the same port that do not interfere are best
placed in the same register so that port reaches one register, not two; values produced by the same instance likewise
want to share a register so its write port fans into one place.

The allocator is a port-affinity-biased graph coloring (a linear scan whose register choice minimizes the marginal
increase in total mux fan-in), refined by a bounded, deterministic simulated-annealing pass. Input ports are pinned to
the unique low registers ``0..nload-1`` so the step-0 parallel-load lanes map one-to-one onto module input ports;
operation results may still reuse those registers once the input value is dead. The register count simply grows; we
never spill.
"""

import math
import random
from collections import Counter
from dataclasses import dataclass

from .._hir import ValueId
from .._mir import MirFloatConst, MirFloatInput, MirFloatOperation, MirFloatView
from ._ir import FloatOperatorInstance

# Read port identity (operator instance + operand position) and write-source identity (an instance, or the input load).
type _Port = tuple[FloatOperatorInstance, int]
type _Producer = FloatOperatorInstance | str
_INPUT_LOAD: _Producer = "input_load"

# Annealing schedule.
_ANNEAL_ITERS_PER_VALUE = 100
_ANNEAL_MAX_ITERS = 1000000
_ANNEAL_T0 = 2.0


@dataclass(frozen=True, slots=True)
class FloatAllocation:
    assign: dict[ValueId, int]  # register-needing value -> register index
    nreg: int


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
    assign: dict[ValueId, int], def_cycle: dict[ValueId, int], last_use: dict[ValueId, int]
) -> None:
    """Backstop: no two values sharing a register may have overlapping live ranges (read-first: last_use<=def is OK)."""
    members: dict[int, list[ValueId]] = {}
    for vid, reg in assign.items():
        members.setdefault(reg, []).append(vid)
    for reg, vids in members.items():
        for i, a in enumerate(vids):
            for b in vids[i + 1 :]:
                if not (last_use[a] <= def_cycle[b] or last_use[b] <= def_cycle[a]):
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
        return 0  # an input port, written at the accept edge

    input_values = [vid for vid in mir.input_ids if isinstance(mir.nodes[vid], MirFloatInput)]
    operation_values = [vid for vid in issue_cycle if isinstance(mir.nodes[vid], MirFloatOperation)]
    reg_values: list[ValueId] = [*input_values, *operation_values]
    def_cycle = {vid: def_cycle_of(vid) for vid in reg_values}
    last_use: dict[ValueId, int] = {vid: def_cycle[vid] for vid in reg_values}

    # Per-value consumer read ports (which operator operand positions read it) and its producer. Outputs read the
    # register array directly (not through a read port), so they extend liveness but add no port reach.
    consumer_ports: dict[ValueId, set[_Port]] = {vid: set() for vid in reg_values}
    producer_key: dict[ValueId, _Producer] = {vid: _INPUT_LOAD for vid in input_values}
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

    assign = _greedy(input_values, operation_values, def_cycle, last_use, consumer_ports, producer_key)
    nreg = (max(assign.values()) + 1) if assign else 0
    assign = _anneal(assign, nreg, operation_values, def_cycle, last_use, consumer_ports, producer_key)
    _assert_no_interference(assign, def_cycle, last_use)
    return FloatAllocation(assign=assign, nreg=nreg)


def _greedy(
    input_values: list[ValueId],
    operation_values: list[ValueId],
    def_cycle: dict[ValueId, int],
    last_use: dict[ValueId, int],
    consumer_ports: dict[ValueId, set[_Port]],
    producer_key: dict[ValueId, _Producer],
) -> dict[ValueId, int]:
    """Linear scan whose register choice minimizes the marginal increase in total mux fan-in (port-affinity bias)."""
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

    # Inputs are pinned to the low registers backing the parallel-load lanes.
    for reg, vid in enumerate(input_values):
        place(vid, reg)
    active: list[tuple[int, int]] = [(last_use[vid], assign[vid]) for vid in input_values]  # (last_use, reg)
    free: list[int] = []
    next_reg = len(input_values)

    for vid in sorted(operation_values, key=lambda v: (def_cycle[v], v)):
        d = def_cycle[vid]
        retained: list[tuple[int, int]] = []
        for lu, reg in active:
            if lu <= d:  # read-first: a read on cycle d still sees the old value, so the register is free for vid
                free.append(reg)
            else:
                retained.append((lu, reg))
        active = retained
        # Choose the candidate (a freed register or a brand-new one) with the least marginal mux growth; on a tie
        # prefer reusing an existing register (key flag 0 < 1) and the lowest index, so the count grows only when
        # reuse would actually cost steering.
        best_key: tuple[int, int, int] | None = None
        best_reg = next_reg
        best_fresh = True
        for reg in free:
            key = (marginal_cost(vid, reg), 0, reg)
            if best_key is None or key < best_key:
                best_key, best_reg, best_fresh = key, reg, False
        fresh_key = (marginal_cost(vid, next_reg), 1, next_reg)
        if best_key is None or fresh_key < best_key:
            best_reg, best_fresh = next_reg, True
        if best_fresh:
            next_reg += 1
        else:
            free.remove(best_reg)
        place(vid, best_reg)
        active.append((last_use[vid], best_reg))
    return assign


def _anneal(
    assign: dict[ValueId, int],
    nreg: int,
    operation_values: list[ValueId],
    def_cycle: dict[ValueId, int],
    last_use: dict[ValueId, int],
    consumer_ports: dict[ValueId, set[_Port]],
    producer_key: dict[ValueId, _Producer],
) -> dict[ValueId, int]:
    """
    Simulated-annealing refinement of the greedy assignment.
    Moves relocate one operation value to a different, interference-free, already-allocated register (inputs stay
    pinned, the register count never grows), accepted by the Metropolis criterion on the mux-fan-in objective with a
    linear cooling schedule. The best assignment seen is returned, so the pass can only improve on the greedy seed.
    """
    if not operation_values or nreg <= 1:
        return assign
    members: dict[int, set[ValueId]] = {}
    for vid, reg in assign.items():
        members.setdefault(reg, set()).add(vid)

    def fits(vid: ValueId, target: int) -> bool:
        for other in members.get(target, ()):
            if other != vid and not (last_use[vid] <= def_cycle[other] or last_use[other] <= def_cycle[vid]):
                return False
        return True

    rng = random.Random(0)
    current = dict(assign)
    current_cost = _objective(current, consumer_ports, producer_key)
    best, best_cost = dict(current), current_cost
    iterations = min(_ANNEAL_MAX_ITERS, _ANNEAL_ITERS_PER_VALUE * len(operation_values))
    for i in range(iterations):
        temperature = _ANNEAL_T0 * (1.0 - i / iterations)
        vid = operation_values[rng.randrange(len(operation_values))]
        target = rng.randrange(nreg)
        source = current[vid]
        if target == source or not fits(vid, target):
            continue
        current[vid] = target
        cost = _objective(current, consumer_ports, producer_key)
        delta = cost - current_cost
        if delta <= 0 or (temperature > 0.0 and rng.random() < math.exp(-delta / temperature)):
            members[source].discard(vid)
            members.setdefault(target, set()).add(vid)
            current_cost = cost
            if cost < best_cost:
                best, best_cost = dict(current), cost
        else:
            current[vid] = source
    return best
