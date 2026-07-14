from abc import ABC, abstractmethod

from .._synth import OocDesign, SynthArtifact


class Flow(ABC):
    @abstractmethod
    def available(self) -> bool: ...

    @abstractmethod
    def prepare(self, design: OocDesign) -> SynthArtifact: ...
