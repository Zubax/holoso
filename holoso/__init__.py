"""Holoso: a narrow Python-to-Verilog synthesizer for numeric kernels."""

from ._api import (
    OperatorOptions as OperatorOptions,
    Options as Options,
    SynthesisResult as SynthesisResult,
    synthesize as synthesize,
)
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
    IntFormat as IntFormat,
)
from ._value import FloatValue as FloatValue
from ._errors import (
    HolosoError as HolosoError,
    SourceUnavailable as SourceUnavailable,
    SynthesisError as SynthesisError,
    UnsupportedConstruct as UnsupportedConstruct,
)

from ._backend.cocotb import CocotbOutput as CocotbOutput
from ._backend.html import HtmlOutput as HtmlOutput
from ._backend.numerical import NumericalModel as NumericalModel, NumericalSimulator as NumericalSimulator
from ._backend.verilog import VerilogOutput as VerilogOutput

from . import _operators

# The user names an operator's knobs without naming the operator type, which is not part of the public API.
FAddOptions = _operators.FAddOperator.Options
FAtan2Options = _operators.FAtan2Operator.Options
FCmpOptions = _operators.FCmpOperator.Options
FDivOptions = _operators.FDivOperator.Options
FExp2Options = _operators.FExp2Operator.Options
FFmaOptions = _operators.FFmaOperator.Options
FFromIntOptions = _operators.FFromIntOperator.Options
FLog2Options = _operators.FLog2Operator.Options
FMulILog2Options = _operators.FMulILog2Operator.Options
FMulOptions = _operators.FMulOperator.Options
FRoundOptions = _operators.FRoundOperator.Options
FSincosOptions = _operators.FSincosOperator.Options
FSortOptions = _operators.FSortOperator.Options
FToIntOptions = _operators.FToIntOperator.Options
IMulOptions = _operators.IMulOperator.Options

__version__ = "0.2.0"
__url__ = "https://holoso.digital"
