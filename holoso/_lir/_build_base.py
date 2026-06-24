"""
Shared carrier types for the LIR builder, referenced by more than one builder stage: the constant-pool entry, the
wide/bool phi-arm installs, the full register allocation, the cross-block overlap layout, and the coloring objective.
They sit at the base of the builder stages' dependency DAG (construct/coalesce -> layout -> bankalloc all import from
here) to keep those stage modules acyclic.
"""

from dataclasses import dataclass

from .._operators import BoolInversion, FloatSignControl
from .._util import ValueId
from ._ir import ReadPort
from ._schedule import Schedule
from ._regalloc import Producer


@dataclass(frozen=True, slots=True)
class PooledConst:
    """A constant's place in the magnitude pool: its index, plus the sign that recovers the original signed value."""

    index: int
    sign: FloatSignControl


@dataclass(frozen=True, slots=True)
class FloatArmInstall:
    """A wide phi-arm install at a predecessor's tail: destination register, source value, and the arm's folded sign."""

    dst: int
    source: ValueId
    sign: FloatSignControl


@dataclass(frozen=True, slots=True)
class BoolArmInstall:
    """A boolean phi-arm install at a predecessor's tail: destination register, source value, and folded inversion."""

    dst: int
    source: ValueId
    inversion: BoolInversion


@dataclass(frozen=True, slots=True)
class Allocation:
    float_reg: dict[ValueId, int]
    float_slot_reg: dict[str, int]
    float_install: dict[str, int]  # slot name -> Ret-block-relative scheduler-frame install cycle of its live-out
    nreg: int
    bool_reg: dict[ValueId, int]
    bool_slot_reg: dict[str, int]
    nbreg: int
    copies: dict[int, list[FloatArmInstall]]  # block -> wide phi-arm installs at its tail
    bool_writes: dict[int, list[BoolArmInstall]]  # block -> boolean phi-arm installs at its tail


@dataclass(frozen=True, slots=True)
class OverlapLayout:
    """
    The per-block schedule plus the install-inclusive makespan, the (possibly overlap-shrunk) terminator offset, and
    the spills each block receives -- the predecessor values landing in it past an overlapped terminator, mapped to
    their block-local landing cycle (fed to the allocator's liveness so a spilled register stays reserved in the
    block, and identical to the scheduler's ``livein_landing`` so the two cannot drift). Empty under draining.
    """

    block_sched: dict[int, Schedule]
    block_makespan: dict[int, int]
    block_term_offset: dict[int, int]
    block_inflight: dict[int, dict[ValueId, int]]


@dataclass(frozen=True, slots=True)
class ColorObjective:
    """
    One bank's steering inputs to quotient coloring, beyond the interference graph and pins: the deterministic movable
    order, the per-value consumer read ports and producers (the write-select objective), and the first freely
    assignable register. Threaded together because the colorer consumes them as a unit.
    """

    movable: list[ValueId]
    consumer_ports: dict[ValueId, set[ReadPort]]
    producer_key: dict[ValueId, frozenset[Producer]]
    fresh_start: int
