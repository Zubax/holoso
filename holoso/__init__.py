"""Holoso: a narrow Python-to-Verilog synthesizer for numeric kernels."""

from ._api import synthesize as synthesize, SynthesisResult as SynthesisResult
from ._interface import (
    ControlInputPort as ControlInputPort,
    ControlOutputPort as ControlOutputPort,
    ControlPort as ControlPort,
    DataInputPort as DataInputPort,
    DataOutputPort as DataOutputPort,
    DataPort as DataPort,
    Direction as Direction,
    ModuleInterface as ModuleInterface,
    Port as Port,
)
from ._type import FloatFormat as FloatFormat, FloatType as FloatType, ScalarType as ScalarType
from ._errors import (
    HolosoError as HolosoError,
    MissingIntrinsic as MissingIntrinsic,
    SourceUnavailable as SourceUnavailable,
    SynthesisError as SynthesisError,
    UnsupportedConstruct as UnsupportedConstruct,
)

from ._backend.cocotb import CocotbOutput as CocotbOutput
from ._backend.html import HtmlOutput as HtmlOutput
from ._backend.numerical import NumericalModel as NumericalModel
from ._backend.verilog import VerilogOutput as VerilogOutput

from ._operators import (
    FAddOperator as FAddOperator,
    FDivOperator as FDivOperator,
    FMulILog2OperatorFamily as FMulILog2OperatorFamily,
    FMulOperator as FMulOperator,
    OpConfig as OpConfig,
)

__version__ = "0.1.0"
