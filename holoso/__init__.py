"""Holoso: a narrow Python-to-Verilog synthesizer for numeric kernels."""

from ._api import synthesize as synthesize
from ._backend_verilog import VerilogOutput as VerilogOutput
from ._errors import (
    HolosoError as HolosoError,
    MissingIntrinsic as MissingIntrinsic,
    SourceUnavailable as SourceUnavailable,
    SynthesisError as SynthesisError,
    UnsupportedConstruct as UnsupportedConstruct,
)
from ._format import FloatFormat as FloatFormat
from ._operators import (
    FAddOp as FAddOp,
    FDivOp as FDivOp,
    FMulILog2GenericOp as FMulILog2GenericOp,
    FMulOp as FMulOp,
    OpConfig as OpConfig,
)
from ._result import (
    IIModel as IIModel,
    ModuleInterface as ModuleInterface,
    Port as Port,
    SynthesisMetrics as SynthesisMetrics,
    SynthesisResult as SynthesisResult,
)

__version__ = "0.1.0"
