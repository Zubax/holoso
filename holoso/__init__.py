"""Holoso: a narrow Python-to-Verilog synthesizer for numeric kernels."""

from .api import synthesize as synthesize
from .errors import (
    HolosoError as HolosoError,
    MissingIntrinsic as MissingIntrinsic,
    SourceUnavailable as SourceUnavailable,
    SynthesisError as SynthesisError,
    UnsupportedConstruct as UnsupportedConstruct,
)
from .format import FloatFormat as FloatFormat
from .operators import (
    FAddOp as FAddOp,
    FDivOp as FDivOp,
    FMulILog2GenericOp as FMulILog2GenericOp,
    FMulOp as FMulOp,
    OpConfig as OpConfig,
)
from .result import (
    IIModel as IIModel,
    ModuleInterface as ModuleInterface,
    Port as Port,
    SynthesisMetrics as SynthesisMetrics,
    SynthesisResult as SynthesisResult,
    write_artifacts as write_artifacts,
)

__version__ = "0.1.0"
