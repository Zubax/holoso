"""The composition contract for a synthesized module."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import enum

from ._type import ScalarType


class Direction(enum.StrEnum):
    """Module port direction."""

    IN = "in"
    OUT = "out"


@dataclass(frozen=True, slots=True)
class Port(ABC):
    """One I/O port on a generated module."""

    name: str

    @property
    @abstractmethod
    def direction(self) -> Direction:
        pass

    @property
    @abstractmethod
    def width(self) -> int:
        """Port width in bits."""


@dataclass(frozen=True, slots=True)
class DataPort(Port, ABC):
    scalar_type: ScalarType

    @property
    def width(self) -> int:
        return self.scalar_type.width


@dataclass(frozen=True, slots=True)
class DataInputPort(DataPort):
    @property
    def direction(self) -> Direction:
        return Direction.IN


@dataclass(frozen=True, slots=True)
class DataOutputPort(DataPort):
    @property
    def direction(self) -> Direction:
        return Direction.OUT


@dataclass(frozen=True, slots=True)
class ControlPort(Port, ABC):
    bit_width: int

    @property
    def width(self) -> int:
        return self.bit_width


@dataclass(frozen=True, slots=True)
class ControlInputPort(ControlPort):
    @property
    def direction(self) -> Direction:
        return Direction.IN


@dataclass(frozen=True, slots=True)
class ControlOutputPort(ControlPort):
    @property
    def direction(self) -> Direction:
        return Direction.OUT


@dataclass(frozen=True, slots=True)
class ModuleInterface:
    module_name: str
    ports: list[Port]

    @property
    def input_ports(self) -> list[DataInputPort]:
        return [p for p in self.ports if isinstance(p, DataInputPort)]

    @property
    def output_ports(self) -> list[DataOutputPort]:
        return [p for p in self.ports if isinstance(p, DataOutputPort)]

    @property
    def control_ports(self) -> list[ControlPort]:
        return [p for p in self.ports if isinstance(p, ControlPort)]
