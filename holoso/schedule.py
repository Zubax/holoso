"""Orchestrate scheduling + register allocation into a finished :class:`Lir`, and derive interface/metrics from it."""

from __future__ import annotations

from collections.abc import Mapping

from .hir import Const, Hir, InPort, OpNode, ValueId
from .lir import (
    ConstRef,
    InputLoad,
    Issue,
    Lir,
    Operand,
    OperatorInstance,
    OutputWire,
    RegFileLayout,
    RegRef,
    Step,
)
from .operators import OpKind, Sgnop
from .regalloc import Allocation, allocate
from .result import Direction, IIModel, ModuleInterface, Port, PortRole, SynthesisMetrics
from .scheduler import resolve_pool, schedule_ops


def _opnode(hir: Hir, vid: ValueId) -> OpNode:
    node = hir.nodes[vid]
    assert isinstance(node, OpNode)
    return node


def build(hir: Hir, module_name: str, instances: Mapping[OpKind, int] | None = None) -> Lir:
    """Schedule, bind, and register-allocate a lowered HIR into a microprogram."""
    pool = resolve_pool(hir, instances)
    op_steps = schedule_ops(hir, pool)
    alloc = allocate(hir, op_steps)
    consts, const_index = _build_const_pool(hir)
    ilog2_index = _ilog2_index(hir, op_steps)
    return Lir(
        fmt=hir.fmt,
        module_name=module_name,
        instances=_build_instances(hir, op_steps, ilog2_index),
        consts=consts,
        regfile=RegFileLayout(
            nreg=alloc.nreg,
            nrd=_compute_nrd(hir, op_steps, alloc),
            # Write ports serve both the single-cycle input load and per-step result commits.
            nwr=max(len(hir.input_ids), max((len(issue) for issue in op_steps), default=0), 1),
        ),
        inputs=_build_inputs(hir, alloc),
        steps=_build_steps(hir, op_steps, alloc, const_index, ilog2_index),
        outputs=_build_outputs(hir, alloc, const_index),
        op_count=sum(1 for node in hir.nodes.values() if isinstance(node, OpNode)),
        max_chain_len=_max_chain_len(hir),
    )


def _ilog2_index(hir: Hir, op_steps: list[list[ValueId]]) -> dict[ValueId, int]:
    """Assign each FMUL_ILOG2 op its own dedicated instance index (its K is an elaboration-time parameter)."""
    index: dict[ValueId, int] = {}
    for issue in op_steps:
        for vid in issue:
            if _opnode(hir, vid).kind is OpKind.FMUL_ILOG2:
                index[vid] = len(index)
    return index


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


def _build_instances(
    hir: Hir, op_steps: list[list[ValueId]], ilog2_index: dict[ValueId, int]
) -> tuple[OperatorInstance, ...]:
    peak: dict[OpKind, int] = {}
    for issue in op_steps:
        per_step: dict[OpKind, int] = {}
        for vid in issue:
            kind = _opnode(hir, vid).kind
            if kind is OpKind.FMUL_ILOG2:
                continue  # one dedicated instance per op (see below)
            per_step[kind] = per_step.get(kind, 0) + 1
        for kind, count in per_step.items():
            peak[kind] = max(peak.get(kind, 0), count)
    instances: list[OperatorInstance] = []
    for kind in OpKind:  # deterministic definition order
        if kind is OpKind.FMUL_ILOG2:
            continue
        for index in range(peak.get(kind, 0)):
            instances.append(OperatorInstance(kind, index))
    for vid in sorted(ilog2_index, key=lambda v: ilog2_index[v]):
        instances.append(OperatorInstance(OpKind.FMUL_ILOG2, ilog2_index[vid], _opnode(hir, vid).k))
    return tuple(instances)


def _operand(hir: Hir, vid: ValueId, sgnop: Sgnop, alloc: Allocation, const_index: dict[ValueId, int]) -> Operand:
    node = hir.nodes[vid]
    if isinstance(node, Const):
        return Operand(ConstRef(const_index[vid]), sgnop)
    return Operand(RegRef(alloc.assign[vid]), sgnop)


def _build_steps(
    hir: Hir,
    op_steps: list[list[ValueId]],
    alloc: Allocation,
    const_index: dict[ValueId, int],
    ilog2_index: dict[ValueId, int],
) -> tuple[Step, ...]:
    steps: list[Step] = []
    for index, issue in enumerate(op_steps):
        per_kind: dict[OpKind, int] = {}
        issues: list[Issue] = []
        latency = 0
        for vid in issue:
            op = _opnode(hir, vid)
            if op.kind is OpKind.FMUL_ILOG2:
                inst = OperatorInstance(OpKind.FMUL_ILOG2, ilog2_index[vid], op.k)
            else:
                local = per_kind.get(op.kind, 0)
                per_kind[op.kind] = local + 1
                inst = OperatorInstance(op.kind, local)
            operand_b = None if op.b is None else _operand(hir, op.b, op.b_sgnop, alloc, const_index)
            issues.append(
                Issue(
                    inst=inst,
                    a=_operand(hir, op.a, op.a_sgnop, alloc, const_index),
                    b=operand_b,
                    y_sgnop=op.y_sgnop,
                    k=op.k,
                    dst=RegRef(alloc.assign[vid]),
                )
            )
            latency = max(latency, op.latency)
        steps.append(Step(index, tuple(issues), latency))
    return tuple(steps)


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


def _compute_nrd(hir: Hir, op_steps: list[list[ValueId]], alloc: Allocation) -> int:
    max_reads = 0
    for issue in op_steps:
        regs: set[int] = set()
        for vid in issue:
            op = _opnode(hir, vid)
            for operand in (op.a, op.b):
                if operand is not None and not isinstance(hir.nodes[operand], Const):
                    regs.add(alloc.assign[operand])
        max_reads = max(max_reads, len(regs))
    output_regs = {alloc.assign[o.value] for o in hir.outputs if not isinstance(hir.nodes[o.value], Const)}
    return max(max_reads, len(output_regs), 1)


def _max_chain_len(hir: Hir) -> int:
    depth: dict[ValueId, int] = {}
    op_ids = sorted(vid for vid, node in hir.nodes.items() if isinstance(node, OpNode))
    for vid in op_ids:
        op = _opnode(hir, vid)
        operands = [x for x in (op.a, op.b) if x is not None and isinstance(hir.nodes[x], OpNode)]
        depth[vid] = 1 + max((depth[x] for x in operands), default=0)
    return max(depth.values(), default=0)


def cycle_count(lir: Lir) -> int:
    """Exact in_valid->out_valid latency (== II; data-independent for a combinational module).

    One cycle to accept the inputs, then for each FSM step one launch cycle plus the step's barrier latency (the
    slowest issued operator). The barrier means a step costs its max operator latency regardless of co-issued faster
    operators, so the total is fully determined by the schedule.
    """
    return 1 + sum(step.latency + 1 for step in lir.steps)


def _ii_model(lir: Lir) -> IIModel:
    steps = len(lir.steps)
    compute = sum(step.latency for step in lir.steps)
    formula = f"1 accept + {steps} step launches + {compute} operator cycles"
    return IIModel(step_count=steps, cycle_estimate=cycle_count(lir), formula=formula)


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
    ports.append(Port("diag_error", Direction.OUT, PortRole.CONTROL, 1))
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
        step_count=len(lir.steps),
        ii_estimate=cycle_count(lir),
        op_count=lir.op_count,
        max_chain_len=lir.max_chain_len,
    )
