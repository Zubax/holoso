"""The in-memory result of a synthesis run, plus the only filesystem-touching helpers."""

import enum
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .format import FloatFormat


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
    ports: tuple[Port, ...]
    ii: IIModel

    @property
    def input_ports(self) -> tuple[Port, ...]:
        return tuple(p for p in self.ports if p.role is PortRole.DATA and p.direction is Direction.IN)

    @property
    def output_ports(self) -> tuple[Port, ...]:
        return tuple(p for p in self.ports if p.role is PortRole.DATA and p.direction is Direction.OUT)

    @property
    def control_ports(self) -> tuple[Port, ...]:
        return tuple(p for p in self.ports if p.role is PortRole.CONTROL)


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


@dataclass(frozen=True, slots=True)
class SynthesisResult:
    """Everything produced by a synthesis run, held in memory. Nothing is written to disk unless requested."""

    module_name: str
    interface: ModuleInterface
    verilog: str
    support: str
    support_header: str
    testbench: str
    report_html: str
    metrics: SynthesisMetrics
    hir: object  # holoso.hir.Hir -- tightened once the IR lands (M2/M4)
    lir: object  # holoso.lir.Lir

    def write(self, out_dir: Path | str) -> dict[str, Path]:
        """Write the artifacts to ``out_dir`` and return the written paths."""
        return write_artifacts(self, out_dir)


def write_artifacts(result: SynthesisResult, out_dir: Path | str) -> dict[str, Path]:
    """Write the generated artifacts to ``out_dir`` (the only Holoso operation that touches the filesystem)."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    name = result.module_name
    files: dict[str, tuple[str, str]] = {
        "verilog": (f"{name}.v", result.verilog),
        "support": ("holoso_support.v", result.support),
        "support_header": ("holoso_support.vh", result.support_header),
        "testbench": (f"test_{name}.py", result.testbench),
        "report": (f"{name}.html", result.report_html),
    }
    written: dict[str, Path] = {}
    for key, (filename, content) in files.items():
        path = directory / filename
        path.write_text(content, encoding="utf-8")
        written[key] = path
    return written
