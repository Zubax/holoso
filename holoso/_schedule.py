"""Build a finished :class:`Lir` and derive its interface."""

import math
from collections.abc import Mapping

from ._errors import UnsupportedConstruct
from ._hir import Const, Hir, InPort, OpNode, ValueId
from ._lir import (
    ConstRef,
    FloatRegFileLayout,
    InputLoad,
    Lir,
    Operand,
    OutputWire,
    RegRef,
    ScheduledOp,
)
from ._operators import FloatOp, Op, Sgnop
from ._regalloc import Allocation, allocate
from ._interface import ControlInputPort, ControlOutputPort, DataInputPort, DataOutputPort, ModuleInterface, Port
from ._scheduler import Schedule, resolve_pool, schedule_ops
from ._type import FloatFormat, FloatType


def _opnode(hir: Hir, vid: ValueId) -> OpNode:
    node = hir.nodes[vid]
    assert isinstance(node, OpNode)
    return node


def build(hir: Hir, module_name: str, *, fmt: FloatFormat, instances: Mapping[type[Op], int] | None = None) -> Lir:
    """
    Schedule, bind, and register-allocate a lowered HIR into a pipelined microprogram.
    """
    _check_float_ops_match(hir, fmt)
    pool = resolve_pool(hir, instances)
    sched = schedule_ops(hir, pool)
    alloc = allocate(hir, sched.issue_cycle, sched.makespan)
    consts, const_index = _build_const_pool(hir)
    return Lir(
        module_name=module_name,
        instances=sched.instances,
        consts=consts,
        regfile=FloatRegFileLayout(
            fmt=fmt,
            nreg=alloc.nreg,
            nrd=_compute_nrd(hir, sched, alloc),
            nwr=_compute_nwr(hir, sched),
            nload=_compute_nload(hir, alloc),
        ),
        inputs=_build_inputs(hir, alloc),
        ops=_build_ops(hir, sched, alloc, const_index),
        outputs=_build_outputs(hir, alloc, const_index),
        makespan=sched.makespan,
        op_count=sum(1 for node in hir.nodes.values() if isinstance(node, OpNode)),
        max_chain_len=_max_chain_len(hir),
    )


def _check_float_ops_match(hir: Hir, fmt: FloatFormat) -> None:
    for node in hir.nodes.values():
        if isinstance(node, OpNode) and isinstance(node.op, FloatOp) and node.op.fmt != fmt:
            raise ValueError(f"operator {node.op.mnemonic} uses {node.op.fmt}, but the float regfile uses {fmt}")


def _build_const_pool(hir: Hir) -> tuple[list[float], dict[ValueId, int]]:
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
        if not math.isfinite(node.value):
            raise UnsupportedConstruct(f"non-finite constant {node.value!r} is not representable in the ZKF format")
        values.append(node.value)
    return values, {vid: index for index, vid in enumerate(ids)}


def _operand(hir: Hir, vid: ValueId, sgnop: Sgnop, alloc: Allocation, const_index: dict[ValueId, int]) -> Operand:
    node = hir.nodes[vid]
    if isinstance(node, Const):
        return Operand(ConstRef(const_index[vid]), sgnop)
    return Operand(RegRef(alloc.assign[vid]), sgnop)


def _build_ops(hir: Hir, sched: Schedule, alloc: Allocation, const_index: dict[ValueId, int]) -> list[ScheduledOp]:
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
                dst=RegRef(alloc.assign[vid]),
                issue_cycle=sched.issue_cycle[vid],
                latency=op.op.latency,
            )
        )
    return ops


def _build_inputs(hir: Hir, alloc: Allocation) -> list[InputLoad]:
    loads: list[InputLoad] = []
    for vid in hir.input_ids:
        node = hir.nodes[vid]
        assert isinstance(node, InPort)
        loads.append(InputLoad(node.name, RegRef(alloc.assign[vid])))
    return loads


def _build_outputs(hir: Hir, alloc: Allocation, const_index: dict[ValueId, int]) -> list[OutputWire]:
    wires: list[OutputWire] = []
    for out in hir.outputs:
        node = hir.nodes[out.value]
        source: RegRef | ConstRef
        if isinstance(node, Const):
            source = ConstRef(const_index[out.value])
        else:
            source = RegRef(alloc.assign[out.value])
        wires.append(OutputWire(out.name, source, out.sgnop))
    return wires


def _compute_nrd(hir: Hir, sched: Schedule, alloc: Allocation) -> int:
    """
    Combinational read ports: the peak distinct register reads in any issue cycle.
    Outputs use the register file's passive ``view`` bus, so only operand reads count here.
    """
    per_cycle: dict[int, set[int]] = {}
    for vid, cycle in sched.issue_cycle.items():
        op = _opnode(hir, vid)
        regs = per_cycle.setdefault(cycle, set())
        for operand in (op.a, op.b):
            if operand is not None and not isinstance(hir.nodes[operand], Const):
                regs.add(alloc.assign[operand])
    max_reads = max((len(regs) for regs in per_cycle.values()), default=0)
    return max(max_reads, 1)


def _compute_nwr(hir: Hir, sched: Schedule) -> int:
    """
    Synchronous write ports: the peak operator commits landing on one cycle.
    Inputs use the register file's immediate ``load`` port on cycle 0.
    """
    per_commit: dict[int, int] = {}
    for vid, cycle in sched.issue_cycle.items():
        commit = cycle + _opnode(hir, vid).op.latency
        per_commit[commit] = per_commit.get(commit, 0) + 1
    peak_commits = max(per_commit.values(), default=0)
    return max(peak_commits, 1)


def _compute_nload(hir: Hir, alloc: Allocation) -> int:
    """
    Immediate parallel-load lanes: enough to cover the highest register any input occupies (registers 0..nload-1).

    Inputs commit at cycle 0 and read back from cycle 1; they are loaded in one shot via the register file's ``load``
    port rather than through write ports. Sized from the max occupied input register, not ``len(inputs)``: an unused
    input (never read) is freed at cycle 0 by the linear-scan allocator and may share a low register, so the input
    count can exceed the highest input register index. A *used* input is never freed at cycle 0, so it uniquely owns
    its register; registers 0..nload-1 are exactly the input block.
    """
    return max((alloc.assign[vid] for vid in hir.input_ids), default=-1) + 1


def _max_chain_len(hir: Hir) -> int:
    depth: dict[ValueId, int] = {}
    op_ids = sorted(vid for vid, node in hir.nodes.items() if isinstance(node, OpNode))
    for vid in op_ids:
        op = _opnode(hir, vid)
        operands = [x for x in (op.a, op.b) if x is not None and isinstance(hir.nodes[x], OpNode)]
        depth[vid] = 1 + max((depth[x] for x in operands), default=0)
    return max(depth.values(), default=0)


def interface_of(lir: Lir) -> ModuleInterface:
    scalar_type = FloatType(lir.regfile.fmt)
    ports: list[Port] = [
        ControlInputPort("clk", 1),
        ControlInputPort("rst", 1),
        ControlInputPort("in_valid", 1),
        ControlOutputPort("in_ready", 1),
        ControlOutputPort("out_valid", 1),
        ControlInputPort("out_ready", 1),
    ]
    ports.extend(DataInputPort(f"in_{load.name}", scalar_type) for load in lir.inputs)
    ports.extend(DataOutputPort(wire.name, scalar_type) for wire in lir.outputs)
    ports.append(ControlOutputPort("err_cyc", lir.cyc_width))
    return ModuleInterface(lir.module_name, ports)
