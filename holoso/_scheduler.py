"""
Software-pipelined (zero-bubble) list scheduling over a lowered HIR.

The operators are fully pipelined (throughput 1) and their latencies are static and data-independent, so the whole
schedule is computed here at compile time and the backend controller is just a cycle counter replaying it. We issue
each operator as soon as its operands are *ready* -- without waiting for unrelated ops to finish (no barrier). The
register file is read-first: a result written on cycle ``c+L`` lands in the FF on the next edge and is
readable from ``c+L+1``, so a data-dependent consumer is held one extra cycle past the producer's latency.

Cycle 0 is the input-load/accept cycle; inputs are readable from cycle 1. Compute cycles start at 1. An op issued at
cycle ``c`` reads its register operands at ``c`` and commits at ``c + L``. The dependency constraint is therefore
``c_consumer >= c_producer + L_producer + 1``.

Instances are pooled by operator instance: equal ops time-share ``pool[type(op)]`` copies, at most one issue per copy
per cycle.
"""

from collections.abc import Mapping
from dataclasses import astuple, dataclass

from ._hir import Hir, OpNode, ValueId
from ._lir import OperatorInstance
from ._operators import ALL_OP_CLASSES, Op

_CLASS_ORDER = {cls: index for index, cls in enumerate(ALL_OP_CLASSES)}


@dataclass(frozen=True, slots=True)
class Schedule:
    """The scheduler's output: per-op issue cycle and bound instance, the full instance set, and the makespan."""

    issue_cycle: dict[ValueId, int]
    inst_of: dict[ValueId, OperatorInstance]
    instances: list[OperatorInstance]
    makespan: int  # max commit cycle (issue_cycle + latency), or 0 if there are no ops


def _opnode(hir: Hir, vid: ValueId) -> OpNode:
    node = hir.nodes[vid]
    assert isinstance(node, OpNode)
    return node


def _op_ids(hir: Hir) -> list[ValueId]:
    return [vid for vid, node in hir.nodes.items() if isinstance(node, OpNode)]


def _operands(hir: Hir, vid: ValueId) -> list[ValueId]:
    op = _opnode(hir, vid)
    return [op.a] if op.b is None else [op.a, op.b]


def _operator_operands(hir: Hir, vid: ValueId) -> list[ValueId]:
    """Operand values that are themselves operators (the scheduling dependencies). Inputs/consts are ready at cycle 1."""
    return [x for x in _operands(hir, vid) if isinstance(hir.nodes[x], OpNode)]


def _present_classes(hir: Hir) -> set[type[Op]]:
    return {type(_opnode(hir, vid).op) for vid in _op_ids(hir)}


def resolve_pool(hir: Hir, instances: Mapping[type[Op], int] | None) -> dict[type[Op], int]:
    """
    The per-class instance budget: at least one of every operator class present in the graph (default 1).

    The budget is applied per distinct operator, so ``fmul_ilog2_const`` gets ``pool[FMulILog2Op]`` instances per
    distinct ``K``. Raising a class's budget lets that many equal ops co-issue.
    """
    pool: dict[type[Op], int] = {}
    for cls in _present_classes(hir):
        requested = 1 if instances is None else instances.get(cls, 1)
        pool[cls] = max(1, requested)
    return pool


def _critical_path(hir: Hir, op_ids: list[ValueId]) -> dict[ValueId, int]:
    """Priority height: the latency-weighted longest path to a sink, counting the +1 writeback per dependency edge."""
    consumers: dict[ValueId, list[ValueId]] = {vid: [] for vid in op_ids}
    for vid in op_ids:
        for operand in _operator_operands(hir, vid):
            consumers[operand].append(vid)
    height: dict[ValueId, int] = {}
    for vid in sorted(op_ids, reverse=True):  # consumers have larger ids; process them first
        op = _opnode(hir, vid)
        height[vid] = op.op.latency + max((1 + height[c] for c in consumers[vid]), default=0)
    return height


def schedule_ops(hir: Hir, pool: Mapping[type[Op], int]) -> Schedule:
    """Place every operator on the earliest cycle its operands are ready and a free instance exists."""
    op_ids = _op_ids(hir)
    if not op_ids:
        return Schedule(issue_cycle={}, inst_of={}, instances=[], makespan=0)

    height = _critical_path(hir, op_ids)
    issue_cycle: dict[ValueId, int] = {}
    inst_count: dict[Op, int] = {}  # operator instance -> copies needed (peak concurrent use of that module)
    slot_of: dict[ValueId, tuple[Op, int]] = {}  # op -> (its operator instance, its 0-based copy slot)

    def commit_cycle(vid: ValueId) -> int:
        op = _opnode(hir, vid)
        return issue_cycle[vid] + op.op.latency

    unscheduled = set(op_ids)
    cap = sum(_opnode(hir, vid).op.latency for vid in op_ids) + 2 * len(op_ids) + 64
    cycle = 1
    while unscheduled:
        assert cycle <= cap, "scheduler made no progress"
        ready = [
            vid
            for vid in unscheduled
            if all(x in issue_cycle and cycle >= commit_cycle(x) + 1 for x in _operator_operands(hir, vid))
        ]
        ready.sort(key=lambda vid: (-height[vid], vid))
        used: dict[Op, int] = {}  # operator instance -> copies busy this cycle
        for vid in ready:
            op = _opnode(hir, vid)
            key = op.op
            slot = used.get(key, 0)
            if slot >= pool[type(key)]:  # every copy of this module is busy this cycle -> the op waits
                continue
            used[key] = slot + 1
            inst_count[key] = max(inst_count.get(key, 0), slot + 1)
            slot_of[vid] = (key, slot)
            issue_cycle[vid] = cycle
            unscheduled.discard(vid)
        cycle += 1

    inst_of, instances = _bind_instances(inst_count, slot_of)
    makespan = max((commit_cycle(vid) for vid in op_ids), default=0)
    return Schedule(issue_cycle=issue_cycle, inst_of=inst_of, instances=instances, makespan=makespan)


def _bind_instances(
    inst_count: dict[Op, int], slot_of: dict[ValueId, tuple[Op, int]]
) -> tuple[dict[ValueId, OperatorInstance], list[OperatorInstance]]:
    """
    Give each operator a contiguous block of instance indices within its class, then bind every op.

    Indices are 0-based within a class and run over the class's distinct operators in a deterministic order (class
    order, then the operator's field tuple), so a class with several variants (e.g. ``fmul_ilog2_const`` with
    multiple ``K``) gets uniquely-named modules.
    """
    keys = sorted(inst_count, key=lambda op: (_CLASS_ORDER[type(op)], astuple(op)))
    base: dict[Op, int] = {}
    per_class_next: dict[type[Op], int] = {}
    for op in keys:
        base[op] = per_class_next.get(type(op), 0)
        per_class_next[type(op)] = base[op] + inst_count[op]
    inst_of = {vid: OperatorInstance(op, base[op] + slot) for vid, (op, slot) in slot_of.items()}
    instances = [OperatorInstance(op, base[op] + s) for op in keys for s in range(inst_count[op])]
    return inst_of, instances
