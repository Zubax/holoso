"""
Commutative operand port assignment.

The per-operand read mux is the dominant fabric cost, and its fan-in is the operand's read-set: the distinct registers
that operand position reads across the schedule. For a commutative operator (``a op b == b op a`` bit-for-bit) the two
operands may be freely swapped, which moves a register from one operand position's read-set to the other's. Choosing
each use's orientation to minimise the total read-set size therefore shrinks the read muxes at zero hardware and zero
latency cost -- it is a pure relabelling of which physical port reads which register (the Chen & Cong port-assignment
lever; ASP-DAC 2004).

The choice is made after register allocation, over the realised register assignment. Minimising the total
distinct-register count over operand orientations is an instance of graph bipartisation (NP-hard in general); a plain
local search gets trapped well above the optimum (on ekf1_stateless's multiplier it stalls at 50 register-arms where the
optimum is 46). It is solved exactly as a small MILP (HiGHS via ``scipy.optimize.milp``): orientation variables ``o_i``
and port-reach indicators ``y_{port,reg}`` linked so ``y`` is forced on wherever an orientation places a register, with
the objective summing ``y``. A deterministic local search seeded from the source orientation is the fallback when the
MILP does not prove optimality in the time budget, so the result never increases read-mux fan-in.
"""

from collections import defaultdict
import logging

import numpy as np
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint, milp

from .._util import ValueId
from .._mir import MirNode, MirOperation
from ._ir import OperatorInstance

# One commutative use: its value id and the registers its two operands occupy (``None`` for a constant operand, which
# is sourced from the immediate path and never enters a read-set).
type _Use = tuple[ValueId, int | None, int | None]

# Generous budget for the exact solve. If not solved in time, the deterministic local-search fallback is used instead.
# Currently, the timeout is so large that the fallback is effectively disabled; this is intentional (may revisit later).
_MILP_TIME_LIMIT_S = 3600.0

_logger = logging.getLogger(__name__)


def assign_commutative_ports(
    nodes: dict[ValueId, MirNode],
    inst_of: dict[ValueId, OperatorInstance],
    leaders: set[ValueId],
    assign: dict[ValueId, int],
) -> dict[ValueId, bool]:
    """
    Per commutative operator instance, orient each FIRING's operands to minimise the total read-set size across its
    two read ports. Returns ``{firing leader: swap?}`` -- ``True`` means the build exchanges the two operands and
    permutes the firing's output-port taps through the operator's ``swap_output_permutation``. Solved exactly per
    instance; total read-mux fan-in is minimised (and never exceeds the source orientation). It is cycle-agnostic: it
    depends only on which wide registers each commutative firing's two operands occupy and which physical instance it
    binds, so one call orients every commutative firing across the whole flattened CFG -- the comparator's firings
    included, whose taps are boolean but whose operand muxes are ordinary wide read ports.
    """
    uses_by_instance: dict[OperatorInstance, list[_Use]] = defaultdict(list)
    for vid in sorted(leaders):
        node = nodes[vid]
        if not (isinstance(node, MirOperation) and node.operator.is_commutative):
            continue
        first, second = (assign.get(operand) for operand in node.operands)
        uses_by_instance[inst_of[vid]].append((vid, first, second))
    swap: dict[ValueId, bool] = {}
    fan_in_before = 0
    fan_in_after = 0
    for inst, uses in uses_by_instance.items():
        before = _fan_in(uses, [False] * len(uses))
        orientation = _optimal_orientation(uses)
        if orientation is None:
            _logger.warning(
                "Commutative port assignment fallback: instance=%s index=%d uses=%d fan_in_before=%d",
                inst.operator.instance_stem,
                inst.index,
                len(uses),
                before,
            )
            orientation = _local_search(uses)
        after = _fan_in(uses, orientation)
        fan_in_before += before
        fan_in_after += after
        swap.update({use[0]: swapped for use, swapped in zip(uses, orientation, strict=True)})
    _logger.info(
        "Commutative port assignment: uses=%d instances=%d fan_in=%d->%d",
        sum(len(uses) for uses in uses_by_instance.values()),
        len(uses_by_instance),
        fan_in_before,
        fan_in_after,
    )
    return swap


def _fan_in(uses: list[_Use], orientation: list[bool]) -> int:
    port_a: set[int] = set()
    port_b: set[int] = set()
    for (_, first, second), swapped in zip(uses, orientation, strict=True):
        left, right = (second, first) if swapped else (first, second)
        if left is not None:
            port_a.add(left)
        if right is not None:
            port_b.add(right)
    return len(port_a) + len(port_b)


def _optimal_orientation(uses: list[_Use]) -> list[bool] | None:
    """
    Minimum-total-read-set orientation, solved exactly with a MILP. Variables: one binary orientation per use plus a
    binary ``read[port][register]`` indicator; each use forces the indicator of whichever port its operand lands on,
    and the objective sums the indicators. Returns the per-use swap flags, or ``None`` if optimality is not proven
    within the time budget.
    """
    registers = sorted({reg for _, first, second in uses for reg in (first, second) if reg is not None})
    if not registers:
        return [False] * len(uses)
    index_of = {reg: i for i, reg in enumerate(registers)}
    n_uses, n_reg = len(uses), len(registers)

    def read(port: int, reg: int) -> int:
        return n_uses + port * n_reg + index_of[reg]

    n_vars = n_uses + 2 * n_reg
    rows: list[np.ndarray] = []
    lower: list[float] = []

    def require(terms: list[tuple[int, float]], at_least: float) -> None:
        row = np.zeros(n_vars)
        for variable, coefficient in terms:
            row[variable] = coefficient
        rows.append(row)
        lower.append(at_least)

    for use, (_, first, second) in enumerate(uses):
        # orientation 0 = no swap: operand 0 -> port 0, operand 1 -> port 1; orientation 1 swaps them.
        if first is not None:
            require([(read(0, first), 1.0), (use, 1.0)], 1.0)  # read[0][first] >= 1 - o
            require([(read(1, first), 1.0), (use, -1.0)], 0.0)  # read[1][first] >= o
        if second is not None:
            require([(read(1, second), 1.0), (use, 1.0)], 1.0)  # read[1][second] >= 1 - o
            require([(read(0, second), 1.0), (use, -1.0)], 0.0)  # read[0][second] >= o

    cost = np.zeros(n_vars)
    cost[n_uses:] = 1.0  # minimise the number of (port, register) reads = total read-set size
    constraint = LinearConstraint(sp.csr_matrix(np.array(rows)), lower, np.inf)
    result = milp(
        cost,
        constraints=[constraint],
        integrality=np.ones(n_vars),
        bounds=Bounds(0, 1),
        options={"time_limit": _MILP_TIME_LIMIT_S, "mip_rel_gap": 0.0},
    )
    if result.status != 0 or result.x is None:  # 0 == proven optimal; otherwise let the caller fall back
        _logger.warning("Port assignment MILP not optimal: uses=%d status=%s", len(uses), result.status)
        return None
    return [round(result.x[use]) > 0 for use in range(n_uses)]


def _local_search(uses: list[_Use]) -> list[bool]:
    """
    Deterministic seeded local search fallback: flip individual orientations while that lowers fan-in. The all-False
    seed is the source orientation, so the result is never worse than the input; the greedy and all-True seeds add
    deterministic starting points. This is the last resort fallback after MILP.
    """

    def local_minimum(seed: list[bool]) -> list[bool]:
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

    best = local_minimum([False] * len(uses))
    for seed in (_greedy_seed(uses), [True] * len(uses)):
        candidate = local_minimum(seed)
        if _fan_in(uses, candidate) < _fan_in(uses, best):
            best = candidate
    return best


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
