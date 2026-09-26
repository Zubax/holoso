"""Thin API for the low-level IR consumer contract."""

from ._build import build as build
from ._regalloc import RegallocTuning as RegallocTuning
from ._ir import Lir as Lir
from ._ir import successor_blocks as successor_blocks
from ._ir import (
    Arm as Arm,
    BoolConstRef as BoolConstRef,
    BoolInputLoad as BoolInputLoad,
    BoolOperand as BoolOperand,
    BoolRegRef as BoolRegRef,
    BoolStateSlot as BoolStateSlot,
    Boundary as Boundary,
    Branch as Branch,
    Exit as Exit,
    InlineScheduledOp as InlineScheduledOp,
    Jump as Jump,
    LirBlock as LirBlock,
    OperatorInstance as OperatorInstance,
    PooledScheduledOp as PooledScheduledOp,
    RegRef as RegRef,
    ScheduledOp as ScheduledOp,
    Terminator as Terminator,
    WideConstRef as WideConstRef,
    WideInputLoad as WideInputLoad,
    WideOperand as WideOperand,
    WideStateSlot as WideStateSlot,
    install_landing as install_landing,
    landing_cycle as landing_cycle,
    operand_read_cycle as operand_read_cycle,
)
from ._sources import (
    HandshakeArm as HandshakeArm,
    InlineWriteSource as InlineWriteSource,
    MoveWriteSource as MoveWriteSource,
    OpWriteSource as OpWriteSource,
    ReadSource as ReadSource,
    handshake_arms as handshake_arms,
    steering as steering,
    write_arms as write_arms,
    WriteEvent as WriteEvent,
    WriteSource as WriteSource,
    read_sources_per_port as read_sources_per_port,
    write_events as write_events,
    write_sources_per_register as write_sources_per_register,
)
from ._ports import (
    ControlInputPort as ControlInputPort,
    ControlOutputPort as ControlOutputPort,
    ControlPort as ControlPort,
    DataInputPort as DataInputPort,
    DataOutputPort as DataOutputPort,
    DataPort as DataPort,
    Direction as Direction,
    Port as Port,
)
