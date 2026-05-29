"""Resolution helper over a float LIR write timeline (see ``Lir.float_write_timeline``)."""

from ._ir import FloatProducer, FloatRegRef


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
