"""Orchestrate scheduling + register allocation into a finished :class:`Lir`, and derive interface/metrics from it."""

from __future__ import annotations

from collections.abc import Mapping

from .hir import Const, Hir, InPort, OpNode, ValueId
from .lir import (
    ConstRef,
    InputLoad,
    Lir,
    Operand,
    OutputWire,
    RegFileLayout,
    RegRef,
    ScheduledOp,
)
from .operators import DEFAULT_STAGES, OpKind, Sgnop, StageConfig
from .regalloc import Allocation, allocate
from .result import Direction, IIModel, ModuleInterface, Port, PortRole, SynthesisMetrics
from .scheduler import Schedule, resolve_pool, schedule_ops


def _opnode(hir: Hir, vid: ValueId) -> OpNode:
    node = hir.nodes[vid]
    assert isinstance(node, OpNode)
    return node


def build(
    hir: Hir,
    module_name: str,
    instances: Mapping[OpKind, int] | None = None,
    stages: StageConfig = DEFAULT_STAGES,
) -> Lir:
    """Schedule, bind, and register-allocate a lowered HIR into a pipelined microprogram.

    ``stages`` must match the configuration used to annotate latencies in :func:`passes.run`, so the schedule and the
    emitted ``STAGE_*`` instance params agree; it is recorded on the :class:`Lir` for the backend and report.
    """
    pool = resolve_pool(hir, instances)
    sched = schedule_ops(hir, pool)
    alloc = allocate(hir, sched.issue_cycle, sched.makespan)
    consts, const_index = _build_const_pool(hir)
    return Lir(
        fmt=hir.fmt,
        stages=stages,
        module_name=module_name,
        instances=sched.instances,
        consts=consts,
        regfile=RegFileLayout(
            nreg=alloc.nreg,
            nrd=_compute_nrd(hir, sched, alloc),
            nwr=_compute_nwr(hir, sched),
        ),
        inputs=_build_inputs(hir, alloc),
        ops=_build_ops(hir, sched, alloc, const_index),
        outputs=_build_outputs(hir, alloc, const_index),
        makespan=sched.makespan,
        op_count=sum(1 for node in hir.nodes.values() if isinstance(node, OpNode)),
        max_chain_len=_max_chain_len(hir),
    )


def _build_const_pool(hir: Hir) -> tuple[tuple[float, ...], dict[ValueId, int]]:
    ids: list[ValueId] = []
    seen: set[ValueId] = set()

    def note(vid: ValueId) -> None:
        node = hir.nodes[vid]
        if isinstance(node, Const) and vid not in seen:
            seen.add(vid)
            ids.append(vid)

    for node in hir.nodes.values():
        if isinstance(node, OpNode):
            note(node.a)
            if node.b is not None:
                note(node.b)
    for out in hir.outputs:
        note(out.value)
    values: list[float] = []
    for vid in ids:
        node = hir.nodes[vid]
        assert isinstance(node, Const)
        values.append(node.value)
    return tuple(values), {vid: index for index, vid in enumerate(ids)}


def _operand(hir: Hir, vid: ValueId, sgnop: Sgnop, alloc: Allocation, const_index: dict[ValueId, int]) -> Operand:
    node = hir.nodes[vid]
    if isinstance(node, Const):
        return Operand(ConstRef(const_index[vid]), sgnop)
    return Operand(RegRef(alloc.assign[vid]), sgnop)


def _build_ops(
    hir: Hir, sched: Schedule, alloc: Allocation, const_index: dict[ValueId, int]
) -> tuple[ScheduledOp, ...]:
    ops: list[ScheduledOp] = []
    for vid in sorted(sched.issue_cycle, key=lambda v: (sched.issue_cycle[v], v)):
        op = _opnode(hir, vid)
        operand_b = None if op.b is None else _operand(hir, op.b, op.b_sgnop, alloc, const_index)
        ops.append(
            ScheduledOp(
                inst=sched.inst_of[vid],
                a=_operand(hir, op.a, op.a_sgnop, alloc, const_index),
                b=operand_b,
                y_sgnop=op.y_sgnop,
                k=op.k,
                dst=RegRef(alloc.assign[vid]),
                issue_cycle=sched.issue_cycle[vid],
                latency=op.latency,
            )
        )
    return tuple(ops)


def _build_inputs(hir: Hir, alloc: Allocation) -> tuple[InputLoad, ...]:
    loads: list[InputLoad] = []
    for vid in hir.input_ids:
        node = hir.nodes[vid]
        assert isinstance(node, InPort)
        loads.append(InputLoad(node.name, RegRef(alloc.assign[vid])))
    return tuple(loads)


def _build_outputs(hir: Hir, alloc: Allocation, const_index: dict[ValueId, int]) -> tuple[OutputWire, ...]:
    wires: list[OutputWire] = []
    for out in hir.outputs:
        node = hir.nodes[out.value]
        source: RegRef | ConstRef
        if isinstance(node, Const):
            source = ConstRef(const_index[out.value])
        else:
            source = RegRef(alloc.assign[out.value])
        wires.append(OutputWire(out.name, source, out.sgnop))
    return tuple(wires)


def _compute_nrd(hir: Hir, sched: Schedule, alloc: Allocation) -> int:
    """Combinational read ports: the peak distinct register reads in any issue cycle, or the output presentation."""
    per_cycle: dict[int, set[int]] = {}
    for vid, cycle in sched.issue_cycle.items():
        op = _opnode(hir, vid)
        regs = per_cycle.setdefault(cycle, set())
        for operand in (op.a, op.b):
            if operand is not None and not isinstance(hir.nodes[operand], Const):
                regs.add(alloc.assign[operand])
    max_reads = max((len(regs) for regs in per_cycle.values()), default=0)
    output_regs = {alloc.assign[o.value] for o in hir.outputs if not isinstance(hir.nodes[o.value], Const)}
    return max(max_reads, len(output_regs), 1)


def _compute_nwr(hir: Hir, sched: Schedule) -> int:
    """Synchronous write ports: the single-cycle input load, and the peak operator commits landing on one cycle."""
    per_commit: dict[int, int] = {}
    for vid, cycle in sched.issue_cycle.items():
        commit = cycle + _opnode(hir, vid).latency
        per_commit[commit] = per_commit.get(commit, 0) + 1
    peak_commits = max(per_commit.values(), default=0)
    return max(len(hir.input_ids), peak_commits, 1)


def _max_chain_len(hir: Hir) -> int:
    depth: dict[ValueId, int] = {}
    op_ids = sorted(vid for vid, node in hir.nodes.items() if isinstance(node, OpNode))
    for vid in op_ids:
        op = _opnode(hir, vid)
        operands = [x for x in (op.a, op.b) if x is not None and isinstance(hir.nodes[x], OpNode)]
        depth[vid] = 1 + max((depth[x] for x in operands), default=0)
    return max(depth.values(), default=0)


def cycle_count(lir: Lir) -> int:
    """Exact in_valid->out_valid latency: the schedule makespan (last commit cycle) plus one cycle to present.

    Cycle 0 accepts and writes the inputs; compute cycles 1..makespan run the pipelined schedule (the last operator
    commits on the makespan cycle); the result lands in the register file on the next edge and is presented on cycle
    makespan+1. Data-independent, so this is exact. Zero-op (pure passthrough) modules present on cycle 1.
    """
    return lir.makespan + 1


def _ii_model(lir: Lir) -> IIModel:
    formula = f"makespan {lir.makespan} + 1 present cycle"
    return IIModel(makespan=lir.makespan, cycles=cycle_count(lir), formula=formula)


def interface_of(lir: Lir) -> ModuleInterface:
    fmt = lir.fmt
    ports: list[Port] = [
        Port("clk", Direction.IN, PortRole.CONTROL, 1),
        Port("rst", Direction.IN, PortRole.CONTROL, 1),
        Port("in_valid", Direction.IN, PortRole.CONTROL, 1),
        Port("in_ready", Direction.OUT, PortRole.CONTROL, 1),
        Port("out_valid", Direction.OUT, PortRole.CONTROL, 1),
        Port("out_ready", Direction.IN, PortRole.CONTROL, 1),
    ]
    ports.extend(Port(f"in_{load.name}", Direction.IN, PortRole.DATA, fmt.width) for load in lir.inputs)
    ports.extend(Port(wire.name, Direction.OUT, PortRole.DATA, fmt.width) for wire in lir.outputs)
    ports.append(Port("err_cyc", Direction.OUT, PortRole.CONTROL, lir.cyc_width))
    return ModuleInterface(lir.module_name, fmt, tuple(ports), _ii_model(lir))


def metrics_of(lir: Lir) -> SynthesisMetrics:
    counts: dict[str, int] = {}
    for inst in lir.instances:
        counts[inst.kind.value] = counts.get(inst.kind.value, 0) + 1
    return SynthesisMetrics(
        operator_instances=counts,
        n_float_regs=lir.regfile.nreg,
        n_bool_regs=0,
        read_ports=lir.regfile.nrd,
        write_ports=lir.regfile.nwr,
        makespan=lir.makespan,
        ii_cycles=cycle_count(lir),
        op_count=lir.op_count,
        max_chain_len=lir.max_chain_len,
    )
