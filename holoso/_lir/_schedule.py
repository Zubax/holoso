"""
Software-pipelined (zero-bubble) list scheduling over selected MIR.

The hardware operators are fully pipelined (throughput 1) and their latencies are static and data-independent, so the
whole schedule is computed here at compile time and the backend controller is just a cycle counter replaying it.
"""

from collections.abc import Mapping
from dataclasses import dataclass

from .._hir import ValueId
from .._mir import MirFloatInput, MirFloatOperation, MirFloatStateRead, MirFloatView
from .._operators import FloatHardwareOperator, HardwareOperator
from ._ir import FloatOperatorInstance

# The extra cycles a consumer must wait beyond a producer's commit before it may read the result: one for the
# register-file write-to-read edge (read-first), plus one per latch traversed (write latch then read latch).
DEPENDENCY_EDGE = 3

# Inputs are the exception. They load straight into the register array at the accept edge (bypassing the write latch),
# and the 3-stage microcode fetch lands that load before the first control word reaches the datapath, so neither the
# write latch nor the read-first edge applies to an input-reading op -- only the read latch remains. (It would regain
# the read-first cycle only if the fetch were ever made shallow enough not to hide the load.)
INPUT_DEPENDENCY_EDGE = 1


@dataclass(frozen=True, slots=True)
class Schedule:
    """The scheduler's output: per-op issue cycle and bound instance, the full instance set, and the makespan."""

    issue_cycle: dict[ValueId, int]
    inst_of: dict[ValueId, FloatOperatorInstance]
    instances: list[FloatOperatorInstance]
    makespan: int  # max commit cycle (issue_cycle + latency), or 0 if there are no ops


def _operation(mir: MirFloatView, vid: ValueId) -> MirFloatOperation:
    return mir.operation_nodes[vid]


def _op_ids(mir: MirFloatView) -> list[ValueId]:
    return list(mir.operation_nodes)


def _operator_operands(mir: MirFloatView, vid: ValueId) -> list[ValueId]:
    """Operand values that are themselves operators; inputs/consts are ready at cycle 1."""
    return [operand for operand in _operation(mir, vid).operands if operand in mir.operation_nodes]


def _present_classes(mir: MirFloatView) -> set[type[HardwareOperator]]:
    return {type(_operation(mir, vid).operator) for vid in _op_ids(mir)}


def resolve_pool(mir: MirFloatView) -> dict[type[HardwareOperator], int]:
    """
    The per-class instance budget: at least one of every operator class present in the graph.

    The budget is applied per distinct hardware operator, so ``fmul_ilog2_const`` gets the requested number of
    instances for each distinct ``K``.
    """
    pool: dict[type[HardwareOperator], int] = {}
    for cls in _present_classes(mir):
        requested = 1  # TODO: we can add heuristics for determining how many operator instances to use.
        pool[cls] = max(1, requested)
    return pool


def _critical_path(mir: MirFloatView, op_ids: list[ValueId]) -> dict[ValueId, int]:
    """Priority height: longest latency-weighted path to a sink, counting DEPENDENCY_EDGE per dependency edge."""
    consumers: dict[ValueId, list[ValueId]] = {vid: [] for vid in op_ids}
    for vid in op_ids:
        for operand in _operator_operands(mir, vid):
            consumers[operand].append(vid)
    height: dict[ValueId, int] = {}
    for vid in sorted(op_ids, reverse=True):  # consumers have larger IDs; process them first
        node = _operation(mir, vid)
        height[vid] = node.operator.latency + max((DEPENDENCY_EDGE + height[c] for c in consumers[vid]), default=0)
    return height


def schedule_ops(mir: MirFloatView, pool: Mapping[type[HardwareOperator], int]) -> Schedule:
    """Place every selected operation on the earliest cycle its operands are ready and a free instance exists."""
    op_ids = _op_ids(mir)
    if not op_ids:
        return Schedule(issue_cycle={}, inst_of={}, instances=[], makespan=0)

    height = _critical_path(mir, op_ids)
    issue_cycle: dict[ValueId, int] = {}
    inst_count: dict[FloatHardwareOperator, int] = {}
    slot_of: dict[ValueId, tuple[FloatHardwareOperator, int]] = {}

    def commit_cycle(vid: ValueId) -> int:
        return issue_cycle[vid] + _operation(mir, vid).operator.latency

    def is_ready(vid: ValueId, cycle: int) -> bool:
        # A consumer may read an operator producer only DEPENDENCY_EDGE cycles after it commits (the read-first write
        # edge plus the write and read latches). An input loads directly into the array at the accept edge and the
        # fetch lag hides it, so an input-reading op needs only INPUT_DEPENDENCY_EDGE; constants are immediates with no
        # read-timing constraint.
        for operand in _operation(mir, vid).operands:
            if operand in mir.operation_nodes:
                if operand not in issue_cycle or cycle < commit_cycle(operand) + DEPENDENCY_EDGE:
                    return False
            elif isinstance(mir.nodes[operand], (MirFloatInput, MirFloatStateRead)) and cycle < INPUT_DEPENDENCY_EDGE:
                # A state read is already resident in its register (like a preloaded input), so only the read latch
                # applies; the value carried over from the previous initiation is read through the same path.
                return False
        return True

    unscheduled = set(op_ids)
    cap = sum(_operation(mir, vid).operator.latency for vid in op_ids) + DEPENDENCY_EDGE * len(op_ids) + 64
    cycle = 1
    while unscheduled:
        if cycle > cap:
            raise RuntimeError("scheduler made no progress")
        ready = sorted((vid for vid in unscheduled if is_ready(vid, cycle)), key=lambda vid: (-height[vid], vid))
        used: dict[FloatHardwareOperator, int] = {}
        for vid in ready:
            operator = _operation(mir, vid).operator
            slot = used.get(operator, 0)
            if slot >= pool[type(operator)]:
                continue
            used[operator] = slot + 1
            inst_count[operator] = max(inst_count.get(operator, 0), slot + 1)
            slot_of[vid] = (operator, slot)
            issue_cycle[vid] = cycle
            unscheduled.discard(vid)
        cycle += 1

    inst_of, instances = _bind_instances(inst_count, slot_of)
    makespan = max((commit_cycle(vid) for vid in op_ids), default=0)
    return Schedule(issue_cycle=issue_cycle, inst_of=inst_of, instances=instances, makespan=makespan)


def _bind_instances(
    inst_count: dict[FloatHardwareOperator, int], slot_of: dict[ValueId, tuple[FloatHardwareOperator, int]]
) -> tuple[dict[ValueId, FloatOperatorInstance], list[FloatOperatorInstance]]:
    """
    Bind each operation to a physical instance. Instance indices are local to one concrete hardware operator value.
    """
    inst_of = {vid: FloatOperatorInstance(operator, slot) for vid, (operator, slot) in slot_of.items()}
    instances = [
        FloatOperatorInstance(operator, slot) for operator in inst_count for slot in range(inst_count[operator])
    ]
    return inst_of, instances
