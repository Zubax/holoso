"""Resolution helper over an LIR wide-register write timeline (see ``Lir.write_timeline``)."""

from ._ir import Producer, RegRef


def latest_producer_before(
    writes: dict[RegRef, list[tuple[int, Producer]]], source: RegRef, read_cycle: int
) -> Producer:
    """
    Return the producer of the value a register holds when read at hardware-frame ``read_cycle``.

    The timeline is keyed by each value's landing cycle -- the cycle it becomes readable in the array, which already
    folds in the write latch and the read-first edge -- so a read on a value's landing cycle returns that value.
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
