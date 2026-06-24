from abc import ABC, abstractmethod

from holoso import SynthesisResult

from .._synth import SynthArtifact


class Flow(ABC):
    @abstractmethod
    def available(self) -> bool:
        """Whether the underlying tool(s) can be found (PATH, /opt, /usr, /home)."""

    @abstractmethod
    def prepare(self, result: SynthesisResult) -> SynthArtifact:
        """The generated module's only dependency is the bundled support library, so no external RTL is required."""
