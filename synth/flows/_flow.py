from abc import ABC, abstractmethod

from holoso import SynthesisResult

from .._synth import SynthArtifact


class Flow(ABC):
    @abstractmethod
    def available(self) -> bool: ...

    @abstractmethod
    def prepare(self, result: SynthesisResult) -> SynthArtifact: ...
