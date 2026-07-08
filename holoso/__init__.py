"""Holoso: a narrow Python-to-Verilog synthesizer for numeric kernels."""

from ._api import synthesize as synthesize, SynthesisResult as SynthesisResult
from ._lir import (
    ControlInputPort as ControlInputPort,
    ControlOutputPort as ControlOutputPort,
    ControlPort as ControlPort,
    DataInputPort as DataInputPort,
    DataOutputPort as DataOutputPort,
    DataPort as DataPort,
    Direction as Direction,
    Port as Port,
)
from ._type import (
    BoolType as BoolType,
    FloatFormat as FloatFormat,
    FloatType as FloatType,
    LogicalPort as LogicalPort,
    ScalarType as ScalarType,
)
from ._value import FloatValue as FloatValue
from ._errors import (
    HolosoError as HolosoError,
    MissingIntrinsic as MissingIntrinsic,
    SourceUnavailable as SourceUnavailable,
    SynthesisError as SynthesisError,
    UnsupportedConstruct as UnsupportedConstruct,
)

from ._backend.cocotb import CocotbOutput as CocotbOutput
from ._backend.html import HtmlOutput as HtmlOutput
from ._backend.numerical import (
    NumericalModel as NumericalModel,
    NumericalSimulator as NumericalSimulator,
)
from ._backend.verilog import VerilogOutput as VerilogOutput

from ._operators import (
    FAddOperator as FAddOperator,
    FAtan2Operator as FAtan2Operator,
    FDivOperator as FDivOperator,
    FExp2Operator as FExp2Operator,
    FLog2Operator as FLog2Operator,
    FMulILog2OperatorFamily as FMulILog2OperatorFamily,
    FMulOperator as FMulOperator,
    FCmpOperator as FCmpOperator,
    FFmaOperator as FFmaOperator,
    FRoundOperator as FRoundOperator,
    FSincosOperator as FSincosOperator,
    FSortOperator as FSortOperator,
    OpConfig as OpConfig,
)

__version__ = "0.1.12"
__url__ = "https://holoso.digital"
