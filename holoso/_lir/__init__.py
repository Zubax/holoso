"""Thin API for the low-level IR consumer contract."""

from ._build import build as build
from ._ir import Lir as Lir
from ._ir import (
    FloatConstRef as FloatConstRef,
    FloatOperand as FloatOperand,
    FloatOperatorInstance as FloatOperatorInstance,
    FloatRegRef as FloatRegRef,
    FloatScheduledOp as FloatScheduledOp,
    FETCH_LAG as FETCH_LAG,
    FETCH_STAGES as FETCH_STAGES,
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
from ._analysis import (
    group_by_cycle as group_by_cycle,
    float_liveness as float_liveness,
    InputProducer as InputProducer,
    OperationProducer as OperationProducer,
    float_write_timeline as float_write_timeline,
    latest_producer_before as latest_producer_before,
)
