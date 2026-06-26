"""
Test-only write-timeline analysis: a per-register ``(landing cycle, producer)`` timeline and a resolver from a
hardware read cycle back to the value's producer.

Lives in the test suite because it is consumed only by tests (the emitter and the numerical model never need it). The
resolver reads a flat per-register timeline, which is path-exact only on a SINGLE-BLOCK kernel: after mutually-exclusive
arms rejoin at a merge, one flat ordering cannot attribute a read to the arm that actually executed.
``build_write_timeline`` asserts the kernel is single-block so the resolver is never applied where it would be unsound.
"""

from dataclasses import dataclass

from holoso._lir import Lir, RegRef


@dataclass(frozen=True, slots=True)
class InputProducer:
    """A write to a register that came from an input-load lane ``index`` (in module-port order)."""

    index: int


@dataclass(frozen=True, slots=True)
class OperationProducer:
    """A write to a register that came from operation ``index`` in ``Lir.ops``."""

    index: int


@dataclass(frozen=True, slots=True)
class StateProducer:
    """A state register's live-in: the value it carries over from the previous initiation (or the reset snapshot)."""

    index: int  # index into Lir.float_state_slots


@dataclass(frozen=True, slots=True)
class InlineProducer:
    """A wide-register write that came from an inline firing: ``Lir.blocks[block].inline_ops[index]``."""

    block: int
    index: int


type Producer = InputProducer | OperationProducer | StateProducer | InlineProducer


def build_write_timeline(lir: Lir) -> dict[RegRef, list[tuple[int, Producer]]]:
    """
    Per-register write timeline ``(landing cycle, producer)`` in the hardware/executing-step frame, used to resolve a
    register source at a hardware read cycle. A value is readable from the cycle it lands in the array: inputs and state
    live-ins on cycle 1, an operator result on its ``write_landing_pcs`` landing.

    Single-block only: a multi-block kernel would need a path-aware resolver (see the module docstring), so the timeline
    -- and the flat resolver below -- are restricted to the one-block case where every write lands exactly once.
    """
    assert len(lir.blocks) == 1, "the flat write timeline is path-exact only on a single-block kernel"
    writes: dict[RegRef, list[tuple[int, Producer]]] = {}
    for i, load in enumerate(lir.float_inputs):
        writes.setdefault(load.dst, []).append((1, InputProducer(i)))
    # A slot register starts each initiation holding its live-in; a coalesced operator may then overwrite it later in
    # the same initiation via its own OperationProducer entry.
    for s, slot in enumerate(lir.float_state_slots):
        writes.setdefault(slot.reg, []).append((1, StateProducer(s)))
    # ``Lir.ops`` is the per-block ``ops`` flattened in block order, so this index matches OperationProducer. Inline
    # firings write the wide bank too (the bool->float cast); without them a cast-fed operand resolves to no producer.
    op_index = 0
    for block in lir.blocks:
        for op in block.ops:
            for write in op.writes:
                if isinstance(write.dst, RegRef):
                    for pc in lir.write_landing_pcs(block, op, write):
                        writes.setdefault(write.dst, []).append((pc, OperationProducer(op_index)))
            op_index += 1
        for k, inline_op in enumerate(block.inline_ops):
            if isinstance(inline_op.write.dst, RegRef):
                for pc in lir.write_landing_pcs(block, inline_op, inline_op.write):
                    writes.setdefault(inline_op.write.dst, []).append((pc, InlineProducer(block.index, k)))
    for events in writes.values():
        events.sort(key=lambda event: event[0])
    return writes


def latest_producer_before(
    writes: dict[RegRef, list[tuple[int, Producer]]], source: RegRef, read_cycle: int
) -> Producer:
    """
    Return the producer of the value a register holds when read at hardware-frame ``read_cycle``.

    The timeline is keyed by each value's landing cycle -- the cycle it becomes readable in the array, which already
    folds in the fetch lag and the read-first edge -- so a read on a value's landing cycle returns that value.
    """
    chosen: Producer | None = None
    for landing_cycle, producer in writes[source]:
        if landing_cycle <= read_cycle:
            chosen = producer
        else:
            break
    if chosen is None:
        raise RuntimeError("operand read resolves to no prior writer; the schedule is inconsistent")
    return chosen
