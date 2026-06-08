"""
Software-pipelined (zero-bubble) list scheduling over selected MIR.

The hardware operators are fully pipelined (throughput 1) and their latencies are static and data-independent, so the
whole schedule is computed here at compile time and the backend controller is just a cycle counter replaying it.
"""

from collections.abc import Mapping
from dataclasses import dataclass

from .._hir import ValueId
from .._mir import MirBoolConst, MirFloatConst, MirNode, MirOperation, MirFloatView
from .._operators import FComparisonOperator, FloatHardwareOperator, HardwareOperator
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


def _op(nodes: dict[ValueId, MirNode], vid: ValueId) -> MirOperation:
    node = nodes[vid]
    assert isinstance(node, MirOperation)
    return node


def _operator_operands(nodes: dict[ValueId, MirNode], vid: ValueId, schedulable: set[ValueId]) -> list[ValueId]:
    """Operand values scheduled alongside ``vid`` (same block); all other operands are resident at block start."""
    return [operand for operand in _op(nodes, vid).operands if operand in schedulable]


def resolve_pool(mir: MirFloatView) -> dict[type[HardwareOperator], int]:
    """
    The per-class instance budget: at least one of every float operator class present in the graph.

    The budget is applied per distinct hardware operator, so ``fmul_ilog2_const`` gets the requested number of
    instances for each distinct ``K``. Only the instance-backed float operators are pooled; the combinational
    comparison/logic/cast operators carry no physical instance and need no budget.
    """
    pool: dict[type[HardwareOperator], int] = {}
    for node in mir.operation_nodes.values():
        if isinstance(node.operator, FloatHardwareOperator):  # only instance-backed float arithmetic is pooled
            requested = 1  # TODO: we can add heuristics for determining how many operator instances to use.
            pool[type(node.operator)] = max(1, requested)
    return pool


def _critical_path(
    nodes: dict[ValueId, MirNode], op_ids: list[ValueId], schedulable: set[ValueId]
) -> dict[ValueId, int]:
    """Priority height: longest latency-weighted path to a sink, counting DEPENDENCY_EDGE per dependency edge."""
    consumers: dict[ValueId, list[ValueId]] = {vid: [] for vid in op_ids}
    for vid in op_ids:
        for operand in _operator_operands(nodes, vid, schedulable):
            consumers[operand].append(vid)
    height: dict[ValueId, int] = {}
    for vid in sorted(op_ids, reverse=True):  # consumers have larger IDs; process them first
        node = _op(nodes, vid)
        height[vid] = node.operator.latency + max((DEPENDENCY_EDGE + height[c] for c in consumers[vid]), default=0)
    return height


def schedule_ops(
    nodes: dict[ValueId, MirNode], pool: Mapping[type[HardwareOperator], int], schedulable: set[ValueId]
) -> Schedule:
    """
    Place every operation in ``schedulable`` (one block's operations, across both register banks) on the earliest cycle
    its operands are ready and a free instance exists. A single dependency-aware pass spans float and boolean-result
    operations: a comparison (or cast) issues as soon as its operands have landed, independently of the block's float
    arithmetic, so cross-bank dependency chains (a value feeding a comparison feeding a cast) schedule correctly without
    a barrier. Operands outside ``schedulable`` are block live-ins resident at the block start (a prior block's drained
    result, a state read, an input, or a phi); constants are immediates. Only instance-backed float operators consume a
    pool slot and bind a physical instance; the combinational comparison/logic/cast operators issue inline with none.
    """
    op_ids = sorted(schedulable)
    if not op_ids:
        return Schedule(issue_cycle={}, inst_of={}, instances=[], makespan=0)
    schedulable_set = set(op_ids)

    height = _critical_path(nodes, op_ids, schedulable_set)
    issue_cycle: dict[ValueId, int] = {}
    inst_count: dict[FloatHardwareOperator, int] = {}
    slot_of: dict[ValueId, tuple[FloatHardwareOperator, int]] = {}

    def commit_cycle(vid: ValueId) -> int:
        return issue_cycle[vid] + _op(nodes, vid).operator.latency

    def is_ready(vid: ValueId, cycle: int) -> bool:
        # A consumer may read a same-block operator producer only DEPENDENCY_EDGE cycles after it commits (the
        # read-first write edge plus the write and read latches). Every other operand -- a state read, an input, a
        # phi, or a result drained in from a prior block -- is resident at the block start, so it needs only
        # INPUT_DEPENDENCY_EDGE (the read latch); constants are immediates with no read-timing constraint.
        for operand in _op(nodes, vid).operands:
            if operand in schedulable_set:
                if operand not in issue_cycle or cycle < commit_cycle(operand) + DEPENDENCY_EDGE:
                    return False
            elif not isinstance(nodes[operand], (MirFloatConst, MirBoolConst)) and cycle < INPUT_DEPENDENCY_EDGE:
                return False
        return True

    unscheduled = set(op_ids)
    cap = sum(_op(nodes, vid).operator.latency for vid in op_ids) + DEPENDENCY_EDGE * len(op_ids) + 64
    cycle = 1
    while unscheduled:
        if cycle > cap:
            raise RuntimeError("scheduler made no progress")
        ready = sorted((vid for vid in unscheduled if is_ready(vid, cycle)), key=lambda vid: (-height[vid], vid))
        used: dict[FloatHardwareOperator, int] = {}
        used_comparator = 0  # the single pooled holoso_fcmp serves one comparison per cycle (throughput 1, PC-muxed)
        for vid in ready:
            operator = _op(nodes, vid).operator
            if isinstance(operator, FloatHardwareOperator):
                # Instance-backed float arithmetic contends for a pooled physical instance and binds one.
                slot = used.get(operator, 0)
                if slot >= pool[type(operator)]:
                    continue
                used[operator] = slot + 1
                inst_count[operator] = max(inst_count.get(operator, 0), slot + 1)
                slot_of[vid] = (operator, slot)
            elif isinstance(operator, FComparisonOperator):
                # Every relation time-shares one comparator; serialize to one comparison per cycle so each gets a
                # distinct in_valid PC (the emitter's operand mux is keyed on that PC). It binds no instance.
                if used_comparator >= 1:
                    continue
                used_comparator += 1
            # Other combinational operators (boolean logic, casts) are independent inline gates: no contention.
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
