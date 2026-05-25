"""
Linear-scan register allocation over the software-pipelined (cycle-accurate) schedule.

Register-needing values are the input ports and operator results (constants are immediates, not registers). A value
is *defined* (written into its register) at its commit cycle -- ``issue_cycle + latency`` for an op, cycle 0 for an
input (the accept edge) -- and *last used* at the latest cycle it is read: the issue cycle of its last consuming op,
or the output-presentation cycle ``makespan + 1`` if it drives an output.

Two values share a register when the older one's last use is no later than the newer one's definition cycle
(``last_use <= def_cycle``). This aggressive rule is sound ONLY because the register file is read-first (``RWPASS=0``,
see ``backend_verilog._emit_regfile``): when a consumer reads value V on cycle X while value W is committed to the
same register on cycle X, the combinational read returns the old (stored) V and W lands on the next edge -- no
corruption. Under a write-through file this would forward W into the read and break. We never spill -- the register
count simply grows.
"""

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


def allocate(hir: Hir, issue_cycle: dict[ValueId, int], makespan: int) -> Allocation:
    present_cycle = makespan + 1

    def def_cycle(vid: ValueId) -> int:
        node = hir.nodes[vid]
        if isinstance(node, OpNode):
            return issue_cycle[vid] + node.op.latency(hir.fmt)
        return 0  # an input port, written at the accept edge

    reg_values: list[ValueId] = [*hir.input_ids, *issue_cycle.keys()]
    last_use: dict[ValueId, int] = {vid: def_cycle(vid) for vid in reg_values}

    for vid in issue_cycle:
        op = _opnode(hir, vid)
        for operand in (op.a, op.b):
            if operand is None or isinstance(hir.nodes[operand], Const):
                continue
            last_use[operand] = max(last_use[operand], issue_cycle[vid])
    for out in hir.outputs:
        if not isinstance(hir.nodes[out.value], Const):
            last_use[out.value] = max(last_use.get(out.value, 0), present_cycle)

    assign: dict[ValueId, int] = {}
    free: list[int] = []
    active: list[tuple[int, int]] = []  # (last_use, reg)
    next_reg = 0
    for vid in sorted(reg_values, key=lambda v: (def_cycle(v), v)):
        d = def_cycle(vid)
        retained: list[tuple[int, int]] = []
        for lu, reg in active:
            if lu <= d:  # read-first (RWPASS=0): a read on cycle d still sees the old value, so sharing is sound
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
