from abc import ABC, abstractmethod

from holoso import Options

from .._synth import OocDesign, SynthArtifact


class Flow(ABC):
    @abstractmethod
    def available(self) -> bool: ...

    @abstractmethod
    def attributed(self, options: Options) -> Options:
        """
        `options` with this flow's synthesis attributes where they leave an attribute mapping empty: an operator whose
        options carry a mapping of their own keeps it, so a configuration overrides the flow, operator by operator.
        Every synthesis entry point passes its options through this before synthesizing for the flow.
        """

    @abstractmethod
    def prepare(self, design: OocDesign) -> SynthArtifact: ...
