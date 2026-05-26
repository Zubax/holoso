"""The composition contract for a synthesized module."""

import enum
from dataclasses import dataclass

from ._format import FloatFormat


class Direction(enum.Enum):
    IN = "in"
    OUT = "out"


class PortRole(enum.Enum):
    DATA = "data"  # a float scalar carried in/out of the module
    CONTROL = "control"  # clock, reset, the valid/ready handshake, diagnostics


@dataclass(frozen=True, slots=True)
class Port:
    """One Verilog port on a generated module."""

    name: str
    direction: Direction
    role: PortRole
    width: int


@dataclass(frozen=True, slots=True)
class ModuleInterface:
    """The generated module's ports and float format."""

    module_name: str
    float_format: FloatFormat
    ports: list[Port]

    @property
    def input_ports(self) -> list[Port]:
        return [p for p in self.ports if p.role is PortRole.DATA and p.direction is Direction.IN]

    @property
    def output_ports(self) -> list[Port]:
        return [p for p in self.ports if p.role is PortRole.DATA and p.direction is Direction.OUT]

    @property
    def control_ports(self) -> list[Port]:
        return [p for p in self.ports if p.role is PortRole.CONTROL]
