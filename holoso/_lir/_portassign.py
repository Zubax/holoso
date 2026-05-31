"""
Commutative operand port assignment.

The per-operand read mux is the dominant fabric cost, and its fan-in is the operand's read-set: the distinct registers
that operand position reads across the schedule. For a commutative operator (``a op b == b op a`` bit-for-bit) the two
operands may be freely swapped, which moves a register from one operand position's read-set to the other's. Choosing
each use's orientation to minimise the total read-set size therefore shrinks the read muxes at zero hardware and zero
latency cost -- it is a pure relabelling of which physical port reads which register (the Chen & Cong port-assignment
lever; ASP-DAC 2004).

The choice is made after register allocation, over the realised register assignment, and is seeded from the current
(source) orientation so it can only reduce read-mux fan-in, never increase it. Minimising the total distinct-register
count over operand orientations is an instance of graph bipartisation (NP-hard in general), so a deterministic
seeded local search is used; the instances here are tiny.
"""

from collections import defaultdict

from .._hir import ValueId
from .._mir import MirFloatOperation, MirFloatView
from ._ir import FloatOperatorInstance
from ._regalloc import FloatAllocation
from ._schedule import Schedule

# One commutative use: its value id and the registers its two operands occupy (``None`` for a constant operand, which
# is sourced from the immediate path and never enters a read-set).
type _Use = tuple[ValueId, int | None, int | None]


def assign_commutative_ports(mir: MirFloatView, sched: Schedule, alloc: FloatAllocation) -> dict[ValueId, bool]:
    """
    Per commutative operator instance, orient each use's operands to minimise the total read-set size across its two
    read ports. Returns ``{use value id: swap?}`` -- ``True`` means the emitter should exchange the two operands.
    Seeded from the current orientation, so total read-mux fan-in never increases.
    """
    uses_by_instance: dict[FloatOperatorInstance, list[_Use]] = defaultdict(list)
    for vid in sched.issue_cycle:
        node = mir.nodes[vid]
        if not (isinstance(node, MirFloatOperation) and node.operator.is_commutative):
            continue
        first, second = (alloc.assign.get(operand) for operand in node.operands)
        uses_by_instance[sched.inst_of[vid]].append((vid, first, second))
    swap: dict[ValueId, bool] = {}
    for uses in uses_by_instance.values():
        swap.update(_minimise_fan_in(uses))
    return swap


def _fan_in(uses: list[_Use], orientation: list[bool]) -> int:
    """Total distinct registers read across the two operand ports under the given per-use orientation."""
    port_a: set[int] = set()
    port_b: set[int] = set()
    for (_, first, second), swapped in zip(uses, orientation, strict=True):
        left, right = (second, first) if swapped else (first, second)
        if left is not None:
            port_a.add(left)
        if right is not None:
            port_b.add(right)
    return len(port_a) + len(port_b)


def _local_minimum(uses: list[_Use], seed: list[bool]) -> list[bool]:
    """Flip individual orientations while that strictly lowers fan-in, until no single flip helps."""
    orientation = list(seed)
    improved = True
    while improved:
        improved = False
        for index in range(len(orientation)):
            before = _fan_in(uses, orientation)
            orientation[index] = not orientation[index]
            if _fan_in(uses, orientation) < before:
                improved = True
            else:
                orientation[index] = not orientation[index]
    return orientation


def _greedy_seed(uses: list[_Use]) -> list[bool]:
    """A first-fit orientation: place each use's operands on the side that adds the fewest new registers."""
    port_a: set[int] = set()
    port_b: set[int] = set()
    orientation: list[bool] = []
    for _, first, second in uses:
        keep = (first is not None and first not in port_a) + (second is not None and second not in port_b)
        flip = (second is not None and second not in port_a) + (first is not None and first not in port_b)
        swapped = flip < keep
        left, right = (second, first) if swapped else (first, second)
        if left is not None:
            port_a.add(left)
        if right is not None:
            port_b.add(right)
        orientation.append(swapped)
    return orientation


def _minimise_fan_in(uses: list[_Use]) -> dict[ValueId, bool]:
    # The all-False seed is the current orientation, so the chosen result is never worse than the input. The greedy
    # and all-True seeds give the local search additional deterministic starting points.
    best = _local_minimum(uses, [False] * len(uses))
    for seed in (_greedy_seed(uses), [True] * len(uses)):
        candidate = _local_minimum(uses, seed)
        if _fan_in(uses, candidate) < _fan_in(uses, best):
            best = candidate
    return {use[0]: swapped for use, swapped in zip(uses, best, strict=True)}
