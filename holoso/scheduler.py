"""Barrier-step multi-issue list scheduling over a lowered HIR.

Each step issues every ready operator it can onto a distinct free instance of the right kind, ordered by critical-path
(latency) priority. The controller waits for all issued operators in a step to complete before advancing, so an
operator is ready for the next step once all of its operator operands have been scheduled.
"""

from __future__ import annotations

from collections.abc import Mapping

from .hir import Hir, OpNode, ValueId
from .operators import OpKind


def _opnode(hir: Hir, vid: ValueId) -> OpNode:
    node = hir.nodes[vid]
    assert isinstance(node, OpNode)
    return node


def _op_ids(hir: Hir) -> list[ValueId]:
    return [vid for vid, node in hir.nodes.items() if isinstance(node, OpNode)]


def _operator_operands(hir: Hir, vid: ValueId) -> list[ValueId]:
    """Operand values that are themselves operators (the scheduling dependencies). Inputs/consts are always ready."""
    op = _opnode(hir, vid)
    operands = [op.a] if op.b is None else [op.a, op.b]
    return [x for x in operands if isinstance(hir.nodes[x], OpNode)]


def present_kinds(hir: Hir) -> set[OpKind]:
    return {_opnode(hir, vid).kind for vid in _op_ids(hir)}


def resolve_pool(hir: Hir, instances: Mapping[OpKind, int] | None) -> dict[OpKind, int]:
    """The per-kind instance budget for scheduling: at least one of every kind present in the graph."""
    pool: dict[OpKind, int] = {}
    for kind in present_kinds(hir):
        requested = 1 if instances is None else instances.get(kind, 1)
        pool[kind] = max(1, requested)
    return pool


def _critical_path(hir: Hir, op_ids: list[ValueId]) -> dict[ValueId, int]:
    """Latency-weighted longest path from each operator to any sink (priority: schedule the long pole first)."""
    consumers: dict[ValueId, list[ValueId]] = {vid: [] for vid in op_ids}
    for vid in op_ids:
        for operand in _operator_operands(hir, vid):
            consumers[operand].append(vid)
    height: dict[ValueId, int] = {}
    for vid in sorted(op_ids, reverse=True):  # consumers have larger ids; process them first
        op = _opnode(hir, vid)
        height[vid] = op.latency + max((height[c] for c in consumers[vid]), default=0)
    return height


def schedule_ops(hir: Hir, pool: Mapping[OpKind, int]) -> list[list[ValueId]]:
    """Return the schedule as a list of steps, each a list of operator value-ids issued in parallel."""
    op_ids = _op_ids(hir)
    height = _critical_path(hir, op_ids)
    scheduled: set[ValueId] = set()
    unscheduled: set[ValueId] = set(op_ids)
    steps: list[list[ValueId]] = []
    while unscheduled:
        ready = [vid for vid in unscheduled if all(x in scheduled for x in _operator_operands(hir, vid))]
        ready.sort(key=lambda vid: (-height[vid], vid))
        free = dict(pool)
        issue: list[ValueId] = []
        for vid in ready:
            kind = _opnode(hir, vid).kind
            if free.get(kind, 0) > 0:
                free[kind] -= 1
                issue.append(vid)
        assert issue, "no progress: a ready operator must always be issuable since every present kind has an instance"
        steps.append(issue)
        scheduled.update(issue)
        unscheduled.difference_update(issue)
    return steps
