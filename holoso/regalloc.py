"""Linear-scan register allocation over a barrier-step schedule.

Register-needing values are the input ports and operator results (constants are immediates, not registers). A value is
written into its register at the end of the step that produces it (inputs: before step 0) and read during the steps
that consume it (outputs: at the final DONE step). Two values may share a register when the older one's last read is
no later than the newer one's definition step. We never spill -- the register count simply grows.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

from .hir import Const, Hir, OpNode, ValueId


@dataclass(frozen=True, slots=True)
class Allocation:
    assign: dict[ValueId, int]  # register-needing value -> register index
    nreg: int


def _opnode(hir: Hir, vid: ValueId) -> OpNode:
    node = hir.nodes[vid]
    assert isinstance(node, OpNode)
    return node


def allocate(hir: Hir, op_steps: list[list[ValueId]]) -> Allocation:
    done_step = len(op_steps)
    step_of: dict[ValueId, int] = {vid: k for k, issue in enumerate(op_steps) for vid in issue}

    reg_values: list[ValueId] = [*hir.input_ids, *(vid for issue in op_steps for vid in issue)]
    def_step: dict[ValueId, int] = {vid: -1 for vid in hir.input_ids}
    def_step.update(step_of)
    last_use: dict[ValueId, int] = {vid: def_step[vid] for vid in reg_values}

    for vid, step in step_of.items():
        op = _opnode(hir, vid)
        for operand in (op.a, op.b):
            if operand is None or isinstance(hir.nodes[operand], Const):
                continue
            last_use[operand] = max(last_use[operand], step)
    for out in hir.outputs:
        if not isinstance(hir.nodes[out.value], Const):
            last_use[out.value] = max(last_use.get(out.value, -1), done_step)

    assign: dict[ValueId, int] = {}
    free: list[int] = []
    active: list[tuple[int, int]] = []  # (last_use, reg)
    next_reg = 0
    for vid in sorted(reg_values, key=lambda v: (def_step[v], v)):
        d = def_step[vid]
        retained: list[tuple[int, int]] = []
        for lu, reg in active:
            if lu <= d:
                heapq.heappush(free, reg)
            else:
                retained.append((lu, reg))
        active = retained
        if free:
            reg = heapq.heappop(free)
        else:
            reg = next_reg
            next_reg += 1
        assign[vid] = reg
        active.append((last_use[vid], reg))
    return Allocation(assign=assign, nreg=next_reg)
