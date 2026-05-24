"""Holoso: a narrow Python-to-Verilog synthesizer for numeric kernels."""

from __future__ import annotations

from .api import synthesize
from .errors import (
    HolosoError,
    MissingIntrinsic,
    SourceUnavailable,
    SynthesisError,
    UnsupportedConstruct,
)
from .format import FloatFormat
from .operators import OpKind, StageConfig
from .result import (
    IIModel,
    ModuleInterface,
    Port,
    SynthesisMetrics,
    SynthesisResult,
    write_artifacts,
)

__version__ = "0.1.0"

__all__ = [
    "FloatFormat",
    "HolosoError",
    "IIModel",
    "MissingIntrinsic",
    "ModuleInterface",
    "OpKind",
    "Port",
    "SourceUnavailable",
    "StageConfig",
    "SynthesisError",
    "SynthesisMetrics",
    "SynthesisResult",
    "UnsupportedConstruct",
    "__version__",
    "synthesize",
    "write_artifacts",
]
