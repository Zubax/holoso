"""
Software-pipelined (zero-bubble) list scheduling over a lowered HIR.

The operators are fully pipelined (throughput 1) and their latencies are static and data-independent, so the whole
schedule is computed here at compile time and the backend controller is just a cycle counter replaying it. We issue
each operator as soon as its operands are *ready* -- without waiting for unrelated ops to finish (no barrier). The
register file is read-first (``RWPASS=0``): a result written on cycle ``c+L`` lands in the FF on the next edge and is
readable from ``c+L+1``, so a data-dependent consumer is held one extra cycle past the producer's latency.

Cycle 0 is the input-load/accept cycle; inputs are readable from cycle 1. Compute cycles start at 1. An op issued at
cycle ``c`` reads its register operands at ``c`` and commits at ``c + L``. The dependency constraint is therefore
``c_consumer >= c_producer + L_producer + 1``.

A fast op co-issued with a slow one is no longer gated on the slow one: each consumer advances on its own producer's
commit. Instances are pooled by *resource key* -- ``(kind, elaboration params)``: ops sharing a key (all ``fadd``, or
all ``fmul_ilog2_const`` with the same ``K``) time-share ``pool[kind]`` instances, at most one issue per instance per
cycle. Optional read/write port budgets (default unbounded) further gate admission and lengthen the makespan to fit.
"""

from collections.abc import Mapping
from dataclasses import dataclass

from .hir import Const, Hir, OpNode, ValueId
from .lir import OperatorInstance
from .operators import OpKind, ResourceKey

_KIND_ORDER = {kind: index for index, kind in enumerate(OpKind)}


@dataclass(frozen=True, slots=True)
class Schedule:
    """The scheduler's output: per-op issue cycle and bound instance, the full instance set, and the makespan."""

    issue_cycle: dict[ValueId, int]
    inst_of: dict[ValueId, OperatorInstance]
    instances: tuple[OperatorInstance, ...]
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


def _register_operands(hir: Hir, vid: ValueId) -> list[ValueId]:
    """Operand values backed by a register (inputs or operator results); constants are immediates, no read port."""
    return [x for x in _operands(hir, vid) if not isinstance(hir.nodes[x], Const)]


def present_kinds(hir: Hir) -> set[OpKind]:
    return {_opnode(hir, vid).kind for vid in _op_ids(hir)}


def resolve_pool(hir: Hir, instances: Mapping[OpKind, int] | None) -> dict[OpKind, int]:
    """
    The per-kind instance budget: at least one of every kind present in the graph (default 1).

    The budget is applied per *resource key* of that kind, so ``fmul_ilog2_const`` gets ``pool[FMUL_ILOG2]`` instances
    per distinct ``K``. Raising a kind's budget lets that many ops of any one resource key co-issue.
    """
    pool: dict[OpKind, int] = {}
    for kind in present_kinds(hir):
        requested = 1 if instances is None else instances.get(kind, 1)
        pool[kind] = max(1, requested)
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
        height[vid] = op.latency + max((1 + height[c] for c in consumers[vid]), default=0)
    return height


def _check_port_budget(hir: Hir, op_ids: list[ValueId], nrd: int | None, nwr: int | None) -> None:
    """
    Reject read/write port budgets no schedule can meet, with a clear message rather than a 'no progress' stall.

    A binary op must read its (up to two) distinct register operands in one cycle, and the output-presentation cycle
    reads every distinct output register at once -- both are hard floors on ``nrd``. The single-cycle input load is the
    floor on ``nwr``. (Operator commits can always be spread across cycles, so they impose no extra floor.)
    """
    max_op_reads = max((len(set(_register_operands(hir, vid))) for vid in op_ids), default=0)
    output_regs = len({out.value for out in hir.outputs if not isinstance(hir.nodes[out.value], Const)})
    floor_nrd = max(max_op_reads, output_regs, 1)
    floor_nwr = max(len(hir.input_ids), 1)
    if nrd is not None and nrd < floor_nrd:
        raise ValueError(
            f"nrd={nrd} read ports is infeasible: need >= {floor_nrd} "
            f"(an operator reads up to {max_op_reads} registers; output presentation reads {output_regs})"
        )
    if nwr is not None and nwr < floor_nwr:
        raise ValueError(f"nwr={nwr} write ports is infeasible: need >= {floor_nwr} for the single-cycle input load")


def schedule_ops(hir: Hir, pool: Mapping[OpKind, int], *, nrd: int | None = None, nwr: int | None = None) -> Schedule:
    """
    Greedily place every operator on the earliest cycle its operands are ready and a free instance/port exists.

    ``nrd``/``nwr`` are optional read/write port budgets; ``None`` means unbounded (sized to the schedule's peak
    afterwards). With budgets set, an op that cannot claim a read slot at its issue cycle or a write slot at its
    commit cycle waits, lengthening the makespan.
    """
    op_ids = _op_ids(hir)
    if nrd is not None or nwr is not None:
        _check_port_budget(hir, op_ids, nrd, nwr)
    if not op_ids:
        return Schedule(issue_cycle={}, inst_of={}, instances=(), makespan=0)

    height = _critical_path(hir, op_ids)
    issue_cycle: dict[ValueId, int] = {}
    inst_count: dict[ResourceKey, int] = {}  # resource key -> instances needed (peak concurrent use of that module)
    slot_of: dict[ValueId, tuple[ResourceKey, int]] = {}  # op -> (its resource key, its 0-based instance slot)
    writes_used: dict[int, int] = {}  # commit cycle -> write ports already claimed (forward-indexed; budget only)

    def commit_cycle(vid: ValueId) -> int:
        return issue_cycle[vid] + _opnode(hir, vid).latency

    unscheduled = set(op_ids)
    cap = sum(_opnode(hir, vid).latency for vid in op_ids) + 2 * len(op_ids) + 64
    cycle = 1
    while unscheduled:
        assert cycle <= cap, "scheduler made no progress (infeasible port budget?)"
        ready = [
            vid
            for vid in unscheduled
            if all(x in issue_cycle and cycle >= commit_cycle(x) + 1 for x in _operator_operands(hir, vid))
        ]
        ready.sort(key=lambda vid: (-height[vid], vid))
        used: dict[ResourceKey, int] = {}  # resource key -> instances busy this cycle
        cycle_reads: set[int] = set()
        for vid in ready:
            op = _opnode(hir, vid)
            rk = ResourceKey.of(op.kind, op.k)
            new_reads = {r for r in _register_operands(hir, vid) if r not in cycle_reads}
            commit = cycle + op.latency
            if nrd is not None and len(cycle_reads) + len(new_reads) > nrd:
                continue
            if nwr is not None and writes_used.get(commit, 0) + 1 > nwr:
                continue
            slot = used.get(rk, 0)
            if slot >= pool[op.kind]:  # every copy of this module is busy this cycle -> the op waits
                continue
            used[rk] = slot + 1
            inst_count[rk] = max(inst_count.get(rk, 0), slot + 1)
            slot_of[vid] = (rk, slot)
            issue_cycle[vid] = cycle
            cycle_reads |= new_reads
            writes_used[commit] = writes_used.get(commit, 0) + 1
            unscheduled.discard(vid)
        cycle += 1

    inst_of, instances = _bind_instances(inst_count, slot_of)
    makespan = max((commit_cycle(vid) for vid in op_ids), default=0)
    return Schedule(issue_cycle=issue_cycle, inst_of=inst_of, instances=instances, makespan=makespan)


def _bind_instances(
    inst_count: dict[ResourceKey, int], slot_of: dict[ValueId, tuple[ResourceKey, int]]
) -> tuple[dict[ValueId, OperatorInstance], tuple[OperatorInstance, ...]]:
    """
    Give each resource key a contiguous block of instance indices within its kind, then bind every op.

    Indices are 0-based within a kind and run over the kind's resource keys in a deterministic order, so a kind with
    several elaboration variants (e.g. ``fmul_ilog2_const`` with multiple ``K``) gets uniquely-named modules.
    """
    keys = sorted(inst_count, key=lambda rk: (_KIND_ORDER[rk.kind], rk.params))
    base: dict[ResourceKey, int] = {}
    per_kind_next: dict[OpKind, int] = {}
    for rk in keys:
        base[rk] = per_kind_next.get(rk.kind, 0)
        per_kind_next[rk.kind] = base[rk] + inst_count[rk]
    inst_of = {vid: OperatorInstance(rk, base[rk] + slot) for vid, (rk, slot) in slot_of.items()}
    instances = tuple(OperatorInstance(rk, base[rk] + s) for rk in keys for s in range(inst_count[rk]))
    return inst_of, instances
