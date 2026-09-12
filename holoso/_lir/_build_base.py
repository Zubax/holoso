"""
Shared carrier types for the LIR builder, referenced by more than one builder stage: the constant pool, the
wide/bool phi-arm installs, the full register allocation, the once-computed block schedules, and the per-round block
layout. They sit at the base of the builder stages' dependency DAG (construct/coalesce -> layout -> bankalloc all
import from here) to keep those stage modules acyclic.
"""

from dataclasses import dataclass

from .._mir import Mir, MirBoolView, MirWideView
from .._operators import BoolInversion, PooledHardwareOperator, WideConditioner
from .._util import ValueId
from .._value import WideValue
from ._ir import InPlace, OperatorInstance
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
    """The wide constant pool: index -> encoded machine word, and per pooled constant value its entry."""

    values: list[WideValue]
    entries: dict[ValueId, PooledConst]


@dataclass(frozen=True, slots=True)
class ArmPlacement:
    """
    Where a phi arm's tail install fires in its predecessor's frame: the scheduler-frame issue cycle, and whether the
    source is settled before the block's first step (a constant, an input, a state read, a phi, or a foreign result
    already landed), which needs no read-first sampling. Stamped once per arm, the placement the interference
    residence, the install classification and the emitted copy all read.
    """

    issue: int
    settled: bool


@dataclass(frozen=True, slots=True)
class WideArmInstall:
    """A wide phi-arm install at a predecessor's tail: destination register, source value, folded conditioner."""

    dst: int
    source: ValueId
    conditioner: WideConditioner
    placement: ArmPlacement


@dataclass(frozen=True, slots=True)
class BoolArmInstall:
    """A boolean phi-arm install at a predecessor's tail: destination register, source value, and folded inversion."""

    dst: int
    source: ValueId
    inversion: BoolInversion
    placement: ArmPlacement


@dataclass(frozen=True, slots=True)
class Early:
    """Install a slot's live-out by a pc-gated copy at this Ret-relative scheduler-frame cycle, ahead of the boundary."""

    ret_cycle: int


@dataclass(frozen=True, slots=True)
class Boundary:
    """Install a slot's live-out by a read-first copy at the accepted-output edge."""


type WideSlotInstall = InPlace | Early | Boundary
type BoolSlotInstall = InPlace | Boundary


@dataclass(frozen=True, slots=True)
class Allocation:
    """Every decision of the register allocator: both banks' registers, the installs, and the pooled binding."""

    wide: Coloring  # every wide value's register, the orientation and instance per firing, the steering as counted
    wide_slot_reg: dict[str, int]
    wide_install: dict[str, WideSlotInstall]
    bool_reg: dict[ValueId, int]
    bool_slot_reg: dict[str, int]
    bool_install: dict[str, BoolSlotInstall]
    nbreg: int
    wide_copies: dict[int, list[WideArmInstall]]  # block -> wide phi-arm installs at its tail
    bool_writes: dict[int, list[BoolArmInstall]]  # block -> boolean phi-arm installs at its tail
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
