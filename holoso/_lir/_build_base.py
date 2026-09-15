"""
Carrier types shared by more than one LIR builder stage. They sit at the base of the stages' dependency DAG
(construct/coalesce -> layout -> bankalloc all import from here) to keep those stage modules acyclic.
"""

from dataclasses import dataclass

from .._mir import Mir, MirBoolView, MirWideView
from .._operators import PooledHardwareOperator, WideConditioner
from .._util import ValueId
from .._value import WideValue
from ._ir import BoolCopy, Boundary, Early, InPlace, OperatorInstance, WideCopy
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
class ConstPool:
    values: list[WideValue]
    entries: dict[ValueId, PooledConst]


@dataclass(frozen=True, slots=True)
class Allocation:
    """Every decision of the register allocator: both banks' registers, the installs, and the pooled binding."""

    wide: Coloring  # every wide value's register, the orientation and instance per firing, the steering as counted
    wide_slot_reg: dict[str, int]
    wide_install: dict[str, InPlace | Early | Boundary]
    bool: Coloring
    bool_slot_reg: dict[str, int]
    bool_install: dict[str, InPlace | Boundary]
    # block -> its residual phi-arm installs, and the Ret block's early ones
    copies: dict[int, list[WideCopy | BoolCopy]]
    instances: list[OperatorInstance]  # the pooled instances realized, `wide.instance` labeling them per firing


@dataclass(frozen=True, slots=True)
class BlockSchedules:
    """
    Every block scheduled once per build, with the cross-block overlap threaded through: the per-block schedule;
    the residue each block receives from an overlapping predecessor -- the values landing in it past the overlapped
    terminator, mapped to their block-local landing cycle (fed to the allocator's liveness so a spilled register stays
    reserved in the block, and identical to the scheduler's `livein_landing` so the two cannot drift), and per
    instance slot the block-local cycle before which it is still busy; the terminator offset of every OVERLAPPING
    block, fixed at its issue-side envelope (a block absent here drains, and its offset is the install fixpoint's to
    decide each round); and the instance count the scheduler realized per operator.
    """

    block_sched: dict[int, Schedule]
    block_inflight: dict[int, dict[ValueId, int]]
    block_entry_busy: dict[int, dict[tuple[PooledHardwareOperator, int], int]]
    overlap_term_offset: dict[int, int]
    instances: dict[PooledHardwareOperator, int]


@dataclass(frozen=True, slots=True)
class BlockOffsets:
    """One install-fixpoint round's block layout: each block's install-inclusive makespan and terminator offset."""

    block_makespan: dict[int, int]
    block_term_offset: dict[int, int]


@dataclass(frozen=True, slots=True)
class BuildContext:
    """What every install-fixpoint round shares."""

    mir: Mir
    wide_mir: MirWideView
    bool_mir: MirBoolView
    fetch_lag: int
    schedules: BlockSchedules
    const_pool: ConstPool
