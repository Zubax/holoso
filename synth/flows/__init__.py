"""
Tool-specific synthesis flows for the OOC evaluation harness, plus the abstract flow interface.
"""

from abc import ABC, abstractmethod
from pathlib import Path

from holoso.result import SynthesisResult

from .._synth import SynthArtifact


class Flow(ABC):
    """
    A synthesis flow: configured with a chip target and frequency, it prepares a runnable artifact.
    Concrete flows are frozen dataclasses carrying their own (tool-specific) device descriptor, a
    ``target_frequency_MHz``, and a tool-native ``options`` mapping.
    """

    @abstractmethod
    def available(self) -> bool:
        """Whether the underlying tool(s) can be found (PATH, /opt, /usr, /home)."""

    @abstractmethod
    def prepare(self, result: SynthesisResult, extra_rtl: list[Path]) -> SynthArtifact:
        """
        Build the OOC wrapper and emit a self-contained recipe whose ``synthesize`` runs the flow.
        ``extra_rtl`` are the additional Verilog files the DUT needs.
        """
