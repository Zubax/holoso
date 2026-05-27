"""Build a finished :class:`Lir` from MIR and derive its interface."""

import math
from collections.abc import Mapping

from .._errors import UnsupportedConstruct
from .._hir import ValueId
from .._interface import ControlInputPort, ControlOutputPort, DataInputPort, DataOutputPort, ModuleInterface, Port
from .._mir import (
    Mir,
    MirConst,
    MirFloatConst,
    MirFloatInput,
    MirFloatOperation,
    MirFloatOutput,
    MirInput,
    MirOperation,
)
from .._operators import FloatHardwareOperator, FloatSignControl, HardwareOperator
from .._type import FloatFormat, FloatType
from ._ir import (
    FloatConstRef,
    FloatInputLoad,
    FloatOperand,
    FloatOperatorInstance,
    FloatOutputWire,
    FloatRegFileLayout,
    FloatRegRef,
    FloatScheduledOp,
    Lir,
    OperatorInstance,
)
from ._regalloc import FloatAllocation, allocate_float
from ._schedule import Schedule, resolve_pool, schedule_ops


def _operation(mir: Mir, vid: ValueId) -> MirFloatOperation:
    node = mir.nodes[vid]
    assert isinstance(node, MirFloatOperation)
    return node


def build(
    mir: Mir,
    module_name: str,
    *,
    instances: Mapping[type[HardwareOperator], int] | None = None,
) -> Lir:
    """Schedule, bind, and register-allocate selected MIR into a pipelined microprogram."""
    _check_supported_domains(mir)
    fmt = _float_format_of(mir)
    pool = resolve_pool(mir, instances)
    sched = schedule_ops(mir, pool)
    alloc = allocate_float(mir, sched.issue_cycle, sched.makespan)
    consts, const_index = _build_const_pool(mir)
    float_instances, float_instance_of = _float_instances(sched)
    return Lir(
        module_name=module_name,
        float_instances=float_instances,
        float_consts=consts,
        float_regfile=FloatRegFileLayout(
            fmt=fmt,
            nreg=alloc.nreg,
            nrd=_compute_nrd(sched),
            nwr=_compute_nwr(sched),
            nload=_compute_nload(mir, alloc),
        ),
        float_inputs=_build_inputs(mir, alloc),
        float_ops=_build_ops(mir, sched, alloc, const_index, float_instance_of),
        float_outputs=_build_outputs(mir, alloc, const_index),
        makespan=sched.makespan,
        op_count=sum(1 for node in mir.nodes.values() if isinstance(node, MirFloatOperation)),
        max_chain_len=_max_chain_len(mir),
    )


def _check_supported_domains(mir: Mir) -> None:
    for vid, node in mir.nodes.items():
        if isinstance(node, MirInput) and not isinstance(node, MirFloatInput):
            raise UnsupportedConstruct(f"LIR construction does not support non-float MIR input {vid}")
        if isinstance(node, MirConst) and not isinstance(node, MirFloatConst):
            raise UnsupportedConstruct(f"LIR construction does not support non-float MIR constant {vid}")
        if isinstance(node, MirOperation) and not isinstance(node, MirFloatOperation):
            raise UnsupportedConstruct(f"LIR construction does not support non-float MIR operation {vid}")
    for out in mir.outputs:
        if not isinstance(out, MirFloatOutput):
            raise UnsupportedConstruct(f"LIR construction does not support non-float MIR output {out.name!r}")


def _float_instances(
    sched: Schedule,
) -> tuple[list[FloatOperatorInstance], dict[OperatorInstance, FloatOperatorInstance]]:
    instances: list[FloatOperatorInstance] = []
    instance_of: dict[OperatorInstance, FloatOperatorInstance] = {}
    for inst in sched.instances:
        if not isinstance(inst.operator, FloatHardwareOperator):
            raise UnsupportedConstruct(
                f"LIR construction does not support non-float hardware operator {inst.operator.mnemonic!r}"
            )
        float_inst = FloatOperatorInstance(inst.operator, inst.index)
        instances.append(float_inst)
        instance_of[inst] = float_inst
    return instances, instance_of


def _float_format_of(mir: Mir) -> FloatFormat:
    formats: set[FloatFormat] = set()
    for node in mir.nodes.values():
        match node:
            case MirFloatInput(scalar_type=scalar_type):
                formats.add(scalar_type.fmt)
            case MirFloatConst(scalar_type=scalar_type):
                formats.add(scalar_type.fmt)
            case MirFloatOperation(scalar_type=scalar_type):
                formats.add(scalar_type.fmt)
    if len(formats) != 1:
        ordered = ", ".join(str(fmt) for fmt in sorted(formats, key=lambda fmt: (fmt.wexp, fmt.wman)))
        raise ValueError(f"LIR requires exactly one floating-point format; got {ordered or 'none'}")
    return next(iter(formats))


def _build_const_pool(mir: Mir) -> tuple[list[float], dict[ValueId, int]]:
    ids: list[ValueId] = []
    seen: set[ValueId] = set()

    def note(vid: ValueId) -> None:
        node = mir.nodes[vid]
        if isinstance(node, MirFloatConst) and vid not in seen:
            seen.add(vid)
            ids.append(vid)

    for node in mir.nodes.values():
        if isinstance(node, MirFloatOperation):
            for operand in node.operands:
                note(operand)
    for out in mir.outputs:
        note(out.value)
    values: list[float] = []
    for vid in ids:
        node = mir.nodes[vid]
        assert isinstance(node, MirFloatConst)
        if not math.isfinite(node.value):
            raise UnsupportedConstruct(f"non-finite constant {node.value!r} is not representable in the ZKF format")
        values.append(node.value)
    return values, {vid: index for index, vid in enumerate(ids)}


def _operand(
    mir: Mir, vid: ValueId, sign: FloatSignControl, alloc: FloatAllocation, const_index: dict[ValueId, int]
) -> FloatOperand:
    node = mir.nodes[vid]
    if isinstance(node, MirFloatConst):
        return FloatOperand(FloatConstRef(const_index[vid]), sign)
    return FloatOperand(FloatRegRef(alloc.assign[vid]), sign)


def _build_ops(
    mir: Mir,
    sched: Schedule,
    alloc: FloatAllocation,
    const_index: dict[ValueId, int],
    float_instance_of: dict[OperatorInstance, FloatOperatorInstance],
) -> list[FloatScheduledOp]:
    ops: list[FloatScheduledOp] = []
    for vid in sorted(sched.issue_cycle, key=lambda v: (sched.issue_cycle[v], v)):
        node = _operation(mir, vid)
        operands = [
            _operand(mir, operand, sign, alloc, const_index)
            for operand, sign in zip(node.operands, node.operand_signs, strict=True)
        ]
        ops.append(
            FloatScheduledOp(
                inst=float_instance_of[sched.inst_of[vid]],
                operands=operands,
                result_sign=node.result_sign,
                dst=FloatRegRef(alloc.assign[vid]),
                issue_cycle=sched.issue_cycle[vid],
                latency=node.operator.latency,
            )
        )
    return ops


def _build_inputs(mir: Mir, alloc: FloatAllocation) -> list[FloatInputLoad]:
    loads: list[FloatInputLoad] = []
    for vid in mir.input_ids:
        node = mir.nodes[vid]
        assert isinstance(node, MirFloatInput)
        loads.append(FloatInputLoad(node.name, FloatRegRef(alloc.assign[vid])))
    return loads


def _build_outputs(mir: Mir, alloc: FloatAllocation, const_index: dict[ValueId, int]) -> list[FloatOutputWire]:
    wires: list[FloatOutputWire] = []
    for out in mir.outputs:
        assert isinstance(out, MirFloatOutput)
        node = mir.nodes[out.value]
        source: FloatRegRef | FloatConstRef
        if isinstance(node, MirFloatConst):
            source = FloatConstRef(const_index[out.value])
        else:
            source = FloatRegRef(alloc.assign[out.value])
        wires.append(FloatOutputWire(out.name, source, out.sign))
    return wires


def _compute_nrd(sched: Schedule) -> int:
    """
    Combinational read ports: one dedicated port per operator operand (the sum of instance arities).

    Each ``(instance, operand-position)`` reads from its own fixed port, so the controller word carries only the
    per-port register address and the per-cycle operand-routing mux disappears. Floored to >=1 so the regfile
    parameter guard holds when the kernel has no operators.
    """
    return max(1, sum(inst.operator.arity for inst in sched.instances))


def _compute_nwr(sched: Schedule) -> int:
    """
    Synchronous write ports: one dedicated port per operator instance.

    Each instance's result wires straight to its own write port (no write-data routing mux) and the error/commit
    gating is just that port's write-enable. Floored to >=1 so the regfile parameter guard holds with no operators.
    """
    return max(1, len(sched.instances))


def _compute_nload(mir: Mir, alloc: FloatAllocation) -> int:
    """
    Immediate parallel-load lanes: enough to cover the highest register any input occupies.

    Registers 0..nload-1 are exactly the input block for used inputs; unused inputs may share low registers.
    """
    return max((alloc.assign[vid] for vid in mir.input_ids if vid in alloc.assign), default=-1) + 1


def _max_chain_len(mir: Mir) -> int:
    depth: dict[ValueId, int] = {}
    op_ids = sorted(vid for vid, node in mir.nodes.items() if isinstance(node, MirFloatOperation))
    for vid in op_ids:
        node = _operation(mir, vid)
        operands = [operand for operand in node.operands if isinstance(mir.nodes[operand], MirFloatOperation)]
        depth[vid] = 1 + max((depth[operand] for operand in operands), default=0)
    return max(depth.values(), default=0)


def interface_of(lir: Lir) -> ModuleInterface:
    scalar_type = FloatType(lir.float_regfile.fmt)
    ports: list[Port] = [
        ControlInputPort("clk", 1),
        ControlInputPort("rst", 1),
        ControlInputPort("in_valid", 1),
        ControlOutputPort("in_ready", 1),
        ControlOutputPort("out_valid", 1),
        ControlInputPort("out_ready", 1),
    ]
    ports.extend(DataInputPort(f"in_{load.name}", scalar_type) for load in lir.float_inputs)
    ports.extend(DataOutputPort(wire.name, scalar_type) for wire in lir.float_outputs)
    ports.append(ControlOutputPort("err_pc", lir.cyc_width))
    return ModuleInterface(lir.module_name, ports)
