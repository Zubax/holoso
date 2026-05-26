"""The composition contract for a synthesized module: its ports, timing, and resource/timing metrics."""

import enum
from collections.abc import Mapping
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
    width: int  # bits; 1 for control ports


@dataclass(frozen=True, slots=True)
class IIModel:
    """
    The module's initiation interval -- an exact, statically known cycle count, not an estimate.

    v0 operator latencies are data-independent, so the scheduler computes the schedule's length precisely. ``makespan``
    is the last commit cycle; ``cycles`` is the exact in_valid->out_valid latency (``makespan + 1``). ``formula`` renders
    how that count arises; it is fixed once ``WEXP``/``WMAN`` and the stage knobs pin each operator's latency.
    """

    makespan: int
    cycles: int
    formula: str


@dataclass(frozen=True, slots=True)
class ModuleInterface:
    """The generated module's ports and timing -- the contract for composing it with other RTL."""

    module_name: str
    float_format: FloatFormat
    ports: list[Port]
    ii: IIModel

    @property
    def input_ports(self) -> list[Port]:
        return [p for p in self.ports if p.role is PortRole.DATA and p.direction is Direction.IN]

    @property
    def output_ports(self) -> list[Port]:
        return [p for p in self.ports if p.role is PortRole.DATA and p.direction is Direction.OUT]

    @property
    def control_ports(self) -> list[Port]:
        return [p for p in self.ports if p.role is PortRole.CONTROL]


@dataclass(frozen=True, slots=True)
class SynthesisMetrics:
    """Resource and timing figures for a synthesized module."""

    operator_instances: Mapping[str, int]
    n_float_regs: int
    n_bool_regs: int
    read_ports: int  # register-file combinational read ports (NRD)
    write_ports: int  # register-file synchronous write ports (NWR)
    makespan: int  # schedule's last commit cycle
    ii_cycles: int  # exact in_valid->out_valid latency (makespan + 1)
    op_count: int
    max_chain_len: int
