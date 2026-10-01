"""Holoso: a narrow Python-to-Verilog synthesizer for numeric kernels."""

from ._api import (
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
    IntType as IntType,
)
from ._value import FloatValue as FloatValue, IntValue as IntValue
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

from ._operators import (
    FAddOptions as FAddOptions,
    FCmpOptions as FCmpOptions,
    FCordicOptions as FCordicOptions,
    FDivOptions as FDivOptions,
    FExp2Options as FExp2Options,
    FFmaOptions as FFmaOptions,
    FFromIntOptions as FFromIntOptions,
    FILog2Options as FILog2Options,
    FLog2Options as FLog2Options,
    FMulILog2Options as FMulILog2Options,
    FMulOptions as FMulOptions,
    FRoundOptions as FRoundOptions,
    FSortOptions as FSortOptions,
    FSqrtOptions as FSqrtOptions,
    FToIntOptions as FToIntOptions,
    IAbsOptions as IAbsOptions,
    IAddOptions as IAddOptions,
    ICmpOptions as ICmpOptions,
    IDivOptions as IDivOptions,
    IMulOptions as IMulOptions,
    IPopcntOptions as IPopcntOptions,
    IShlOptions as IShlOptions,
    IShrOptions as IShrOptions,
    ISubOptions as ISubOptions,
    OperatorOptions as OperatorOptions,
)

__version__ = "0.6.2"
__url__ = "https://holoso.digital"
