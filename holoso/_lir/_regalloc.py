"""
Reach-aware register coloring over an explicit interference graph.

The engine is bank- and timeline-agnostic: register sharing is decided entirely by the symmetric interference graph the
caller supplies (built in :mod:`._liveness` from per-block hardware-frame residence, the same executing-step frame as
``Lir.reg_liveness``), so the one ``color`` routine colors a straight-line block or a whole control-flow graph, and
either the wide or the boolean bank. The interference graph already encodes the read-first ``R(a) < W(b)`` rule and the
path-awareness of mutually-exclusive arms; this module only places values onto registers given those constraints.

Unlike a CPU register allocator, the primary objective is NOT to minimize the register count: flip-flops are abundant on
an FPGA and interconnect is scarce, so the cost that matters most is *steering* -- the fan-in of the per-port read muxes
and the per-register write selects of the sparse register file synthesized in the backend. The primary objective is
therefore total mux fan-in: ``sum_p max(0, |read-set(p)| - 1) + sum_r max(0, |writers(r)| - 1)``, where a read port
``p`` is one operator ``(instance, operand-position)`` and ``writers(r)`` are the distinct producers of the values
placed in register ``r``. Two values read by the same port that do not interfere are best placed in the same register so
that port reaches one register, not two; values produced by the same instance likewise want to share a register so its
write port fans into one place.

Register count is a bounded *secondary* objective. A register costs flip-flops but no steering, so it is worth shedding
only when doing so widens a write select modestly: the allocator colors twice -- once reach-minimal, once compacting
into shared registers up to a write-select cap -- and keeps whichever minimizes ``reach + _REG_PRICE * registers`` (see
those constants). The reach-minimal coloring is always a candidate, so trading for fewer registers never raises steering
by more than the price paid per register freed.

The allocator is a port-affinity-biased graph coloring (each value takes the same-interference-free register of least
marginal mux growth), refined by simulated annealing. Pinned values (input ports on the low load lanes, state live-ins
and coalesced live-outs on their slot registers) are fixed by the caller; everything else is movable, reusing a pinned
or earlier register wherever the interference graph allows. There is no spilling: a value the cap cannot place by reuse
simply opens a new register.
"""

from collections import Counter
from dataclasses import dataclass
import os

import numpy as np
from scipy.optimize import dual_annealing

from .._hir import ValueId
from ._ir import FloatOperatorInstance

# Read port identity (operator instance + operand position) and write-source identity: an operator instance, the input
# load, or a per-slot state writer -- opaque keys for grouping the read-mux and write-select fan-in objective.
type _Port = tuple[FloatOperatorInstance, int]
type _Producer = FloatOperatorInstance | str


# Budget for the SciPy dual-annealing refinement. It only polishes an already-valid greedy seed (and is a no-op when
# the seed is already at the reach floor), so the function-evaluation cap keeps build time bounded; raise it to trade
# build time for a deeper search. The environment override is for testing only; eventually we might add an API handle.
_REFINE_MAXITER = int(os.getenv("HOLOSO_REGALLOC_EFFORT", "5000"))

# Balance of reach against register count, layered on the hardware-accurate liveness. ``_REG_REUSE_WRITE_CAP`` bounds how
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
_REG_REUSE_WRITE_CAP = int(os.getenv("HOLOSO_REG_REUSE_WRITE_CAP", "2"))
_REG_PRICE = float(os.getenv("HOLOSO_REG_PRICE", "2.0"))
_NO_CAP = 1 << 30  # an effectively unbounded write-select budget, used for the reach-minimal coloring
_INFEASIBLE_COST = 1e18  # annealing penalty for an undecodable point (far above any real mux-fan-in objective)


@dataclass(frozen=True, slots=True)
class ColoringProblem:
    """
    A register-coloring instance decoupled from any single timeline: register sharing is decided by an explicit
    interference graph (see :mod:`._liveness`), so the same engine colors a straight-line block or a whole CFG, and
    either register bank.

    ``movable`` are the values to place (in a stable order); ``pinned`` fixes inputs and state live-ins to their
    registers; ``interferes`` is the symmetric adjacency; ``consumer_ports`` and ``producer_key`` drive the mux-fan-in
    objective; ``fresh_start`` is the first register index above the pinned block. There is no write-path restriction:
    the emitter drives every register with a single priority chain over all its writers, so any two non-interfering
    values may share a register regardless of whether they are produced by an operator, a phi-arm copy, or a cast.
    """

    movable: list[ValueId]
    pinned: dict[ValueId, int]
    interferes: dict[ValueId, set[ValueId]]
    consumer_ports: dict[ValueId, set[_Port]]
    producer_key: dict[ValueId, _Producer]
    fresh_start: int


def color(problem: ColoringProblem) -> tuple[dict[ValueId, int], int]:
    """
    Color one bank by the reach-aware objective: a port-affinity greedy seed (reach-minimal and cap-compacted), the
    cheaper of the two by ``reach + price * registers``, refined by simulated annealing. Reduces to the straight-line
    coloring on a single block because the interference graph there is exactly the interval-overlap graph.
    """
    cap, price = _REG_REUSE_WRITE_CAP, _REG_PRICE

    def greedy_seed(compact: bool, budget: int) -> tuple[dict[ValueId, int], int]:
        seed = _color_greedy(problem, compact, budget)
        return seed, max((max(seed.values()) + 1) if seed else 0, problem.fresh_start)

    base_seed, base_nreg = greedy_seed(compact=False, budget=_NO_CAP)
    comp_seed, comp_nreg = greedy_seed(compact=True, budget=cap)
    assign, nreg = _color_refine(problem, base_seed, base_nreg, _NO_CAP), base_nreg
    if comp_nreg < base_nreg:
        comp_assign = _color_refine(problem, comp_seed, comp_nreg, cap)
        base_score = (_objective(assign, problem.consumer_ports, problem.producer_key) + price * base_nreg, base_nreg)
        comp_score = (
            _objective(comp_assign, problem.consumer_ports, problem.producer_key) + price * comp_nreg,
            comp_nreg,
        )
        if comp_score < base_score:
            assign, nreg = comp_assign, comp_nreg
    _assert_graph_coloring(assign, problem.interferes)
    return assign, nreg


def _color_greedy(problem: ColoringProblem, compact: bool, cap: int) -> dict[ValueId, int]:
    """
    Port-affinity-biased graph coloring. Each value takes a register whose occupants it does not interfere with, of
    least marginal mux growth (``compact`` ranks any admissible reused register ahead of a fresh one); a fresh register
    is the fallback. With one block this reproduces the straight-line linear scan exactly.
    """
    assign: dict[ValueId, int] = {}
    reg_ports: dict[int, set[_Port]] = {}
    reg_writers: dict[int, set[_Producer]] = {}
    reg_members: dict[int, set[ValueId]] = {}
    port_reach: Counter[_Port] = Counter()

    def place(vid: ValueId, reg: int) -> None:
        assign[vid] = reg
        ports = reg_ports.setdefault(reg, set())
        for port in problem.consumer_ports[vid]:
            if port not in ports:
                ports.add(port)
                port_reach[port] += 1
        reg_writers.setdefault(reg, set()).add(problem.producer_key[vid])
        reg_members.setdefault(reg, set()).add(vid)

    def marginal_cost(vid: ValueId, reg: int) -> int:
        ports: frozenset[_Port] | set[_Port] = reg_ports.get(reg, frozenset())
        writers: frozenset[_Producer] | set[_Producer] = reg_writers.get(reg, frozenset())
        read = sum(1 for port in problem.consumer_ports[vid] if port not in ports and port_reach[port] >= 1)
        write = 1 if (problem.producer_key[vid] not in writers and len(writers) >= 1) else 0
        return read + write

    def candidate_key(vid: ValueId, reg: int, is_fresh: int) -> tuple[int, int, int]:
        cost = marginal_cost(vid, reg)
        return (is_fresh, cost, reg) if compact else (cost, is_fresh, reg)

    def admissible(vid: ValueId, reg: int) -> bool:
        if reg not in reg_members:  # a reserved register below fresh_start with no occupant (e.g. a write-only slot)
            return False
        if not problem.interferes[vid].isdisjoint(reg_members[reg]):
            return False
        if compact and len(reg_writers[reg] | {problem.producer_key[vid]}) > cap:
            return False
        return True

    for vid, reg in sorted(problem.pinned.items(), key=lambda item: (item[1], item[0])):
        place(vid, reg)
    next_reg = problem.fresh_start
    for vid in problem.movable:
        best_reg, best_fresh = next_reg, True
        best_key = candidate_key(vid, next_reg, 1)
        for reg in range(next_reg):
            if not admissible(vid, reg):
                continue
            key = candidate_key(vid, reg, 0)
            if key < best_key:
                best_key, best_reg, best_fresh = key, reg, False
        if best_fresh:
            next_reg += 1
        place(vid, best_reg)
    return assign


def _color_refine(problem: ColoringProblem, seed: dict[ValueId, int], nreg: int, cap: int) -> dict[ValueId, int]:
    """
    Refine the greedy coloring with ``scipy.optimize.dual_annealing``. Each movable value gets a continuous coordinate
    in ``[0, nreg)``; a decode repairs it to the nearest same-pool register free of interference and within the
    write-select budget. Every evaluated point is a valid coloring and the seed is the start, so the pass only improves.
    """
    order = problem.movable
    if len(order) < 2 or nreg <= 1:
        return seed
    op_set = set(order)
    pinned = [(vid, reg) for vid, reg in seed.items() if vid not in op_set]
    seed_regs = set(seed.values())  # the registers the seed actually uses; a reserved (unplaced) register is excluded

    def decode(coords: np.ndarray) -> dict[ValueId, int] | None:
        assign: dict[ValueId, int] = {}
        members: list[set[ValueId]] = [set() for _ in range(nreg)]
        writers: list[set[_Producer]] = [set() for _ in range(nreg)]
        for vid, reg in pinned:
            assign[vid] = reg
            members[reg].add(vid)
            writers[reg].add(problem.producer_key[vid])

        def fits(vid: ValueId, reg: int, within_cap: bool) -> bool:
            if reg not in seed_regs:  # a reserved register with no seed occupant (e.g. a write-only state slot)
                return False
            if not problem.interferes[vid].isdisjoint(members[reg]):
                return False
            return not within_cap or len(writers[reg] | {problem.producer_key[vid]}) <= cap

        for index, vid in enumerate(order):
            pref = min(nreg - 1, max(0, int(coords[index])))
            chosen = -1
            for offset in range(nreg):
                reg = (pref + offset) % nreg
                if fits(vid, reg, within_cap=True):
                    chosen = reg
                    break
            if chosen < 0:
                free = [r for r in range(nreg) if fits(vid, r, within_cap=False)]
                if not free:
                    return None
                chosen = min(free, key=lambda reg: (len(writers[reg] | {problem.producer_key[vid]}), reg))
            assign[vid] = chosen
            members[chosen].add(vid)
            writers[chosen].add(problem.producer_key[vid])
        return assign

    if _REFINE_MAXITER <= 0:
        return seed
    best = seed
    best_cost = _objective(seed, problem.consumer_ports, problem.producer_key)
    if best_cost == 0:
        return best

    def cost(coords: np.ndarray) -> float:
        nonlocal best, best_cost
        candidate = decode(coords)
        if candidate is None:
            return _INFEASIBLE_COST
        value = _objective(candidate, problem.consumer_ports, problem.producer_key)
        if value < best_cost:
            best, best_cost = candidate, value
        return float(value)

    x0 = np.array([float(seed[vid]) for vid in order])
    bounds = [(0.0, nreg - 1e-6)] * len(order)
    dual_annealing(cost, bounds, x0=x0, seed=0, maxiter=_REFINE_MAXITER, maxfun=_REFINE_MAXITER, no_local_search=True)
    return best


def _assert_graph_coloring(assign: dict[ValueId, int], interferes: dict[ValueId, set[ValueId]]) -> None:
    """Backstop: two values sharing a register must not interfere."""
    by_reg: dict[int, list[ValueId]] = {}
    for vid, reg in assign.items():
        by_reg.setdefault(reg, []).append(vid)
    for reg, vids in by_reg.items():
        members = set(vids)
        for vid in vids:
            clash = interferes[vid] & members
            if clash:
                raise AssertionError(f"register {reg} shared by interfering values {vid} and {sorted(clash)}")


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
