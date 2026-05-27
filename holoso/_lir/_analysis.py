"""Shared analysis helpers over finished float LIR."""

from dataclasses import dataclass

from ._ir import FloatConstRef, FloatRegRef, FloatScheduledOp, Lir


@dataclass(frozen=True, slots=True)
class InputProducer:
    index: int


@dataclass(frozen=True, slots=True)
class OperationProducer:
    index: int


type FloatProducer = InputProducer | OperationProducer


def group_by_cycle(lir: Lir) -> tuple[dict[int, list[FloatScheduledOp]], dict[int, list[FloatScheduledOp]]]:
    """Group the schedule into per-cycle issues and commits, canonically ordered."""
    issues: dict[int, list[FloatScheduledOp]] = {}
    commits: dict[int, list[FloatScheduledOp]] = {}
    for op in lir.float_ops:
        issues.setdefault(op.issue_cycle, []).append(op)
        commits.setdefault(op.commit_cycle, []).append(op)
    for group in (issues, commits):
        for ops in group.values():
            ops.sort(key=lambda op: (op.inst.operator.instance_stem, op.inst.index, op.dst.index, op.issue_cycle))
    return issues, commits


def float_liveness(lir: Lir) -> dict[FloatRegRef, set[int]]:
    """
    Map each float register to the clock cycles on which it holds a live value.

    A value is written on its definition cycle and read on each consumer issue cycle, or on the output-present cycle if
    it drives an output. The returned rows include the final visible compute row when a value remains live for output.
    """
    present = lir.makespan + 1
    defs: dict[FloatRegRef, list[int]] = {}
    uses: dict[FloatRegRef, list[int]] = {}
    for load in lir.float_inputs:
        defs.setdefault(load.dst, []).append(0)
    for op in lir.float_ops:
        defs.setdefault(op.dst, []).append(op.commit_cycle)
        for operand in op.operands:
            if isinstance(operand.source, FloatRegRef):
                uses.setdefault(operand.source, []).append(op.issue_cycle)
    for wire in lir.float_outputs:
        if isinstance(wire.source, FloatRegRef):
            uses.setdefault(wire.source, []).append(present)
    live: dict[FloatRegRef, set[int]] = {}
    for reg in defs.keys() | uses.keys():
        writes = sorted(defs.get(reg, []))
        reads = sorted(uses.get(reg, []))
        rows: set[int] = set()
        for i, start in enumerate(writes):
            nxt = writes[i + 1] if i + 1 < len(writes) else present + 1
            last = max((use for use in reads if start <= use < nxt), default=start)
            rows.update(range(start, last + 1))
        live[reg] = rows
    return live


def float_write_timeline(lir: Lir) -> dict[FloatRegRef, list[tuple[int, FloatProducer]]]:
    """Per-register write timeline used to resolve a physical register source at a specific read cycle."""
    writes: dict[FloatRegRef, list[tuple[int, FloatProducer]]] = {}
    for i, load in enumerate(lir.float_inputs):
        writes.setdefault(load.dst, []).append((0, InputProducer(i)))
    for j, op in enumerate(lir.float_ops):
        writes.setdefault(op.dst, []).append((op.commit_cycle, OperationProducer(j)))
    for events in writes.values():
        events.sort()
    return writes


def latest_producer_before(
    writes: dict[FloatRegRef, list[tuple[int, FloatProducer]]], source: FloatRegRef, read_cycle: int
) -> FloatProducer:
    """
    Return the producer of the value a register holds when read at ``read_cycle``.

    The register file is read-first, so writes committed at the read cycle are not visible until the next cycle.
    """
    chosen: FloatProducer | None = None
    for commit_cycle, producer in writes[source]:
        if commit_cycle < read_cycle:
            chosen = producer
        else:
            break
    if chosen is None:
        raise RuntimeError("operand read resolves to no prior writer; the schedule is inconsistent")
    return chosen


def float_source_label(source: FloatRegRef | FloatConstRef) -> str:
    return source.stable_label
