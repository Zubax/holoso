"""
Shared carrier types for the LIR builder, referenced by more than one builder stage: the constant-pool entry, the
wide/bool phi-arm installs, the full register allocation, and the cross-block overlap layout.
They sit at the base of the builder stages' dependency DAG (construct/coalesce -> layout -> bankalloc all import from
here) to keep those stage modules acyclic.
"""

from dataclasses import dataclass

from .._operators import BoolInversion, PooledHardwareOperator, WideConditioner
from .._util import ValueId
from ._ir import OperatorInstance
from ._schedule import Schedule
from ._regalloc import Coloring


@dataclass(frozen=True, slots=True)
class PooledConst:
    """
    A constant's place in the pool, with the conditioner its consumers compose onto. Only the float half folds
    anything there -- the magnitude/sign split rides the free `holoso_fsgnop` sideband, which two's complement has
    no counterpart for -- so an integer entry is stored whole and carries the identity.
    """

    index: int
    conditioner: WideConditioner


@dataclass(frozen=True, slots=True)
class WideArmInstall:
    """A wide phi-arm install at a predecessor's tail: destination register, source value, folded conditioner."""

    dst: int
    source: ValueId
    conditioner: WideConditioner


@dataclass(frozen=True, slots=True)
class BoolArmInstall:
    """A boolean phi-arm install at a predecessor's tail: destination register, source value, and folded inversion."""

    dst: int
    source: ValueId
    inversion: BoolInversion


@dataclass(frozen=True, slots=True)
class Allocation:
    """Every decision of the register allocator: both banks' registers, the installs, and the pooled binding."""

    wide: Coloring  # every wide value's register, the orientation and instance per firing, the steering as counted
    wide_slot_reg: dict[str, int]
    wide_install: dict[str, int]  # slot name -> Ret-block-relative scheduler-frame install cycle of its live-out
    bool_reg: dict[ValueId, int]
    bool_slot_reg: dict[str, int]
    nbreg: int
    wide_copies: dict[int, list[WideArmInstall]]  # block -> wide phi-arm installs at its tail
    bool_writes: dict[int, list[BoolArmInstall]]  # block -> boolean phi-arm installs at its tail
    instances: list[OperatorInstance]  # the pooled instances realized, `wide.instance` labeling them per firing


@dataclass(frozen=True, slots=True)
class OverlapLayout:
    """
    The per-block schedule plus the install-inclusive makespan, the (possibly overlap-shrunk) terminator offset, and
    the residue each block receives from an overlapping predecessor: the values landing in it past the overlapped
    terminator, mapped to their block-local landing cycle (fed to the allocator's liveness so a spilled register
    stays reserved in the block, and identical to the scheduler's `livein_landing` so the two cannot drift), and per
    instance slot the block-local cycle before which it is still busy. Both empty under draining.
    """

    block_sched: dict[int, Schedule]
    block_makespan: dict[int, int]
    block_term_offset: dict[int, int]
    block_inflight: dict[int, dict[ValueId, int]]
    block_entry_busy: dict[int, dict[tuple[PooledHardwareOperator, int], int]]
