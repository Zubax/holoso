"""Thin API for the low-level IR consumer contract."""

from ._build import build as build
from ._ir import Lir as Lir
from ._ir import (
    BoolConstRef as BoolConstRef,
    BoolOperand as BoolOperand,
    BoolRegRef as BoolRegRef,
    BoolSource as BoolSource,
    BoolWrite as BoolWrite,
    Branch as Branch,
    CombScheduledOp as CombScheduledOp,
    FloatConstRef as FloatConstRef,
    FloatCopy as FloatCopy,
    FloatOperand as FloatOperand,
    FloatOperatorInstance as FloatOperatorInstance,
    FloatRegRef as FloatRegRef,
    FloatScheduledOp as FloatScheduledOp,
    FloatStateSlot as FloatStateSlot,
    FETCH_LAG as FETCH_LAG,
    FETCH_STAGES as FETCH_STAGES,
    boundary_step as boundary_step,
    copy_step_cycle as copy_step_cycle,
    landing_cycle as landing_cycle,
    read_latch_cycle as read_latch_cycle,
    InputProducer as InputProducer,
    Jump as Jump,
    LirBlock as LirBlock,
    OperationProducer as OperationProducer,
    OperatorInstance as OperatorInstance,
    Ret as Ret,
    StateProducer as StateProducer,
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
from ._analysis import latest_producer_before as latest_producer_before
