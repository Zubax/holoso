"""Resolution helper over a float LIR write timeline (see ``Lir.float_write_timeline``)."""

from ._ir import FloatProducer, FloatRegRef


def latest_producer_before(
    writes: dict[FloatRegRef, list[tuple[int, FloatProducer]]], source: FloatRegRef, read_cycle: int
) -> FloatProducer:
    """
    Return the producer of the value a register holds when read at hardware-frame ``read_cycle``.

    The timeline is keyed by each value's landing cycle -- the cycle it becomes readable in the array, which already
    folds in the write latch and the read-first edge -- so a read on a value's landing cycle returns that value.
    """
    chosen: FloatProducer | None = None
    for landing_cycle, producer in writes[source]:
        if landing_cycle <= read_cycle:
            chosen = producer
        else:
            break
    if chosen is None:
        raise RuntimeError("operand read resolves to no prior writer; the schedule is inconsistent")
    return chosen
