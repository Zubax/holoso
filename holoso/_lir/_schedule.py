"""
Software-pipelined (zero-bubble) list scheduling over selected MIR.

The hardware operators are fully pipelined (throughput 1) and their latencies are static and data-independent, so the
whole schedule is computed here at compile time and the backend controller is just a cycle counter replaying it.
"""

from collections.abc import Mapping
from dataclasses import dataclass

from .._hir import ValueId
from .._mir import Mir, MirOperation
from .._operators import HardwareOperator
from ._ir import OperatorInstance


@dataclass(frozen=True, slots=True)
class Schedule:
    """The scheduler's output: per-op issue cycle and bound instance, the full instance set, and the makespan."""

    issue_cycle: dict[ValueId, int]
    inst_of: dict[ValueId, OperatorInstance]
    instances: list[OperatorInstance]
    makespan: int  # max commit cycle (issue_cycle + latency), or 0 if there are no ops


def _operation(mir: Mir, vid: ValueId) -> MirOperation:
    node = mir.nodes[vid]
    assert isinstance(node, MirOperation)
    return node


def _op_ids(mir: Mir) -> list[ValueId]:
    return [vid for vid, node in mir.nodes.items() if isinstance(node, MirOperation)]


def _operator_operands(mir: Mir, vid: ValueId) -> list[ValueId]:
    """Operand values that are themselves operators; inputs/consts are ready at cycle 1."""
    return [operand for operand in _operation(mir, vid).operands if isinstance(mir.nodes[operand], MirOperation)]


def _present_classes(mir: Mir) -> set[type[HardwareOperator]]:
    return {type(_operation(mir, vid).operator) for vid in _op_ids(mir)}


def resolve_pool(mir: Mir, instances: Mapping[type[HardwareOperator], int] | None) -> dict[type[HardwareOperator], int]:
    """
    The per-class instance budget: at least one of every operator class present in the graph.

    The budget is applied per distinct hardware operator, so ``fmul_ilog2_const`` gets the requested number of
    instances for each distinct ``K``.
    """
    pool: dict[type[HardwareOperator], int] = {}
    for cls in _present_classes(mir):
        requested = 1 if instances is None else instances.get(cls, 1)
        pool[cls] = max(1, requested)
    return pool


def _critical_path(mir: Mir, op_ids: list[ValueId]) -> dict[ValueId, int]:
    """Priority height: the latency-weighted longest path to a sink, counting the +1 writeback per dependency edge."""
    consumers: dict[ValueId, list[ValueId]] = {vid: [] for vid in op_ids}
    for vid in op_ids:
        for operand in _operator_operands(mir, vid):
            consumers[operand].append(vid)
    height: dict[ValueId, int] = {}
    for vid in sorted(op_ids, reverse=True):  # consumers have larger IDs; process them first
        node = _operation(mir, vid)
        height[vid] = node.operator.latency + max((1 + height[c] for c in consumers[vid]), default=0)
    return height


def schedule_ops(mir: Mir, pool: Mapping[type[HardwareOperator], int]) -> Schedule:
    """Place every selected operation on the earliest cycle its operands are ready and a free instance exists."""
    op_ids = _op_ids(mir)
    if not op_ids:
        return Schedule(issue_cycle={}, inst_of={}, instances=[], makespan=0)

    height = _critical_path(mir, op_ids)
    issue_cycle: dict[ValueId, int] = {}
    inst_count: dict[HardwareOperator, int] = {}
    slot_of: dict[ValueId, tuple[HardwareOperator, int]] = {}

    def commit_cycle(vid: ValueId) -> int:
        return issue_cycle[vid] + _operation(mir, vid).operator.latency

    unscheduled = set(op_ids)
    cap = sum(_operation(mir, vid).operator.latency for vid in op_ids) + 2 * len(op_ids) + 64
    cycle = 1
    while unscheduled:
        assert cycle <= cap, "scheduler made no progress"
        ready = [
            vid
            for vid in unscheduled
            if all(x in issue_cycle and cycle >= commit_cycle(x) + 1 for x in _operator_operands(mir, vid))
        ]
        ready.sort(key=lambda vid: (-height[vid], vid))
        used: dict[HardwareOperator, int] = {}
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
    inst_count: dict[HardwareOperator, int], slot_of: dict[ValueId, tuple[HardwareOperator, int]]
) -> tuple[dict[ValueId, OperatorInstance], list[OperatorInstance]]:
    """
    Bind each operation to a physical instance. Instance indices are local to one concrete hardware operator value.
    """
    inst_of = {vid: OperatorInstance(operator, slot) for vid, (operator, slot) in slot_of.items()}
    instances = [OperatorInstance(operator, slot) for operator in inst_count for slot in range(inst_count[operator])]
    return inst_of, instances
