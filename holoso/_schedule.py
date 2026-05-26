"""Build a finished :class:`Lir` from MIR and derive its interface."""

import math
from collections.abc import Mapping

from ._errors import UnsupportedConstruct
from ._hir import ValueId
from ._interface import ControlInputPort, ControlOutputPort, DataInputPort, DataOutputPort, ModuleInterface, Port
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
from ._mir import Mir, MirConst, MirInput, MirOperation
from ._operators import FloatHardwareOperator, HardwareOperator, SignControl
from ._regalloc import Allocation, allocate
from ._scheduler import Schedule, resolve_pool, schedule_ops
from ._type import FloatFormat, FloatType


def _operation(mir: Mir, vid: ValueId) -> MirOperation:
    node = mir.nodes[vid]
    assert isinstance(node, MirOperation)
    return node


def build(
    mir: Mir,
    module_name: str,
    *,
    fmt: FloatFormat,
    instances: Mapping[type[HardwareOperator], int] | None = None,
) -> Lir:
    """Schedule, bind, and register-allocate selected MIR into a pipelined microprogram."""
    _check_float_ops_match(mir, fmt)
    pool = resolve_pool(mir, instances)
    sched = schedule_ops(mir, pool)
    alloc = allocate(mir, sched.issue_cycle, sched.makespan)
    consts, const_index = _build_const_pool(mir)
    return Lir(
        module_name=module_name,
        instances=sched.instances,
        consts=consts,
        regfile=FloatRegFileLayout(
            fmt=fmt,
            nreg=alloc.nreg,
            nrd=_compute_nrd(mir, sched, alloc),
            nwr=_compute_nwr(mir, sched),
            nload=_compute_nload(mir, alloc),
        ),
        inputs=_build_inputs(mir, alloc),
        ops=_build_ops(mir, sched, alloc, const_index),
        outputs=_build_outputs(mir, alloc, const_index),
        makespan=sched.makespan,
        op_count=sum(1 for node in mir.nodes.values() if isinstance(node, MirOperation)),
        max_chain_len=_max_chain_len(mir),
    )


def _check_float_ops_match(mir: Mir, fmt: FloatFormat) -> None:
    for node in mir.nodes.values():
        if (
            isinstance(node, MirOperation)
            and isinstance(node.operator, FloatHardwareOperator)
            and node.operator.fmt != fmt
        ):
            raise ValueError(
                f"operator {node.operator.mnemonic} uses {node.operator.fmt}, but the float regfile uses {fmt}"
            )


def _build_const_pool(mir: Mir) -> tuple[list[float], dict[ValueId, int]]:
    ids: list[ValueId] = []
    seen: set[ValueId] = set()

    def note(vid: ValueId) -> None:
        node = mir.nodes[vid]
        if isinstance(node, MirConst) and vid not in seen:
            seen.add(vid)
            ids.append(vid)

    for node in mir.nodes.values():
        if isinstance(node, MirOperation):
            for operand in node.operands:
                note(operand)
    for out in mir.outputs:
        note(out.value)
    values: list[float] = []
    for vid in ids:
        node = mir.nodes[vid]
        assert isinstance(node, MirConst)
        if not math.isfinite(node.value):
            raise UnsupportedConstruct(f"non-finite constant {node.value!r} is not representable in the ZKF format")
        values.append(node.value)
    return values, {vid: index for index, vid in enumerate(ids)}


def _operand(mir: Mir, vid: ValueId, sign: SignControl, alloc: Allocation, const_index: dict[ValueId, int]) -> Operand:
    node = mir.nodes[vid]
    if isinstance(node, MirConst):
        return Operand(ConstRef(const_index[vid]), sign)
    return Operand(RegRef(alloc.assign[vid]), sign)


def _build_ops(mir: Mir, sched: Schedule, alloc: Allocation, const_index: dict[ValueId, int]) -> list[ScheduledOp]:
    ops: list[ScheduledOp] = []
    for vid in sorted(sched.issue_cycle, key=lambda v: (sched.issue_cycle[v], v)):
        node = _operation(mir, vid)
        operands = [
            _operand(mir, operand, sign, alloc, const_index)
            for operand, sign in zip(node.operands, node.operand_signs, strict=True)
        ]
        ops.append(
            ScheduledOp(
                inst=sched.inst_of[vid],
                operands=operands,
                result_sign=node.result_sign,
                dst=RegRef(alloc.assign[vid]),
                issue_cycle=sched.issue_cycle[vid],
                latency=node.operator.latency,
            )
        )
    return ops


def _build_inputs(mir: Mir, alloc: Allocation) -> list[InputLoad]:
    loads: list[InputLoad] = []
    for vid in mir.input_ids:
        node = mir.nodes[vid]
        assert isinstance(node, MirInput)
        loads.append(InputLoad(node.name, RegRef(alloc.assign[vid])))
    return loads


def _build_outputs(mir: Mir, alloc: Allocation, const_index: dict[ValueId, int]) -> list[OutputWire]:
    wires: list[OutputWire] = []
    for out in mir.outputs:
        node = mir.nodes[out.value]
        source: RegRef | ConstRef
        if isinstance(node, MirConst):
            source = ConstRef(const_index[out.value])
        else:
            source = RegRef(alloc.assign[out.value])
        wires.append(OutputWire(out.name, source, out.sign))
    return wires


def _compute_nrd(mir: Mir, sched: Schedule, alloc: Allocation) -> int:
    """
    Combinational read ports: the peak distinct register reads in any issue cycle.
    Outputs use the register file's passive ``view`` bus, so only operand reads count here.
    """
    per_cycle: dict[int, set[int]] = {}
    for vid, cycle in sched.issue_cycle.items():
        node = _operation(mir, vid)
        regs = per_cycle.setdefault(cycle, set())
        for operand in node.operands:
            if not isinstance(mir.nodes[operand], MirConst):
                regs.add(alloc.assign[operand])
    max_reads = max((len(regs) for regs in per_cycle.values()), default=0)
    return max(max_reads, 1)


def _compute_nwr(mir: Mir, sched: Schedule) -> int:
    """Synchronous write ports: the peak operator commits landing on one cycle."""
    per_commit: dict[int, int] = {}
    for vid, cycle in sched.issue_cycle.items():
        commit = cycle + _operation(mir, vid).operator.latency
        per_commit[commit] = per_commit.get(commit, 0) + 1
    peak_commits = max(per_commit.values(), default=0)
    return max(peak_commits, 1)


def _compute_nload(mir: Mir, alloc: Allocation) -> int:
    """
    Immediate parallel-load lanes: enough to cover the highest register any input occupies.

    Registers 0..nload-1 are exactly the input block for used inputs; unused inputs may share low registers.
    """
    return max((alloc.assign[vid] for vid in mir.input_ids), default=-1) + 1


def _max_chain_len(mir: Mir) -> int:
    depth: dict[ValueId, int] = {}
    op_ids = sorted(vid for vid, node in mir.nodes.items() if isinstance(node, MirOperation))
    for vid in op_ids:
        node = _operation(mir, vid)
        operands = [operand for operand in node.operands if isinstance(mir.nodes[operand], MirOperation)]
        depth[vid] = 1 + max((depth[operand] for operand in operands), default=0)
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
