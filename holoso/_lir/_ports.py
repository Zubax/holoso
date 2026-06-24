"""Generated module port descriptions owned by LIR."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import enum

from .._type import ScalarType


class Direction(enum.StrEnum):
    IN = "in"
    OUT = "out"


@dataclass(frozen=True, slots=True)
class Port(ABC):
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
