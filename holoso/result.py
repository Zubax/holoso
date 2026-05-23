"""The in-memory result of a synthesis run, plus the only filesystem-touching helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .format import FloatFormat

Direction = Literal["in", "out", "ctrl"]


@dataclass(frozen=True, slots=True)
class Port:
    """One Verilog port on a generated module."""

    name: str
    direction: Direction
    width: int  # bits; 1 for control ports


@dataclass(frozen=True, slots=True)
class IIModel:
    """The module's initiation interval. For a combinational v0 module it is a fixed cycle count.

    ``formula`` is a human-readable expression of the cycle count in terms of operator latencies, since the true
    figure depends on the operators' instantiation parameters (``WEXP``/``WMAN``/stage knobs).
    """

    step_count: int
    cycle_estimate: int
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
        return tuple(p for p in self.ports if p.direction == "in")

    @property
    def output_ports(self) -> tuple[Port, ...]:
        return tuple(p for p in self.ports if p.direction == "out")


@dataclass(frozen=True, slots=True)
class SynthesisMetrics:
    """Resource and timing figures for a synthesized module."""

    operator_instances: Mapping[str, int]
    n_float_regs: int
    n_bool_regs: int
    step_count: int
    ii_estimate: int
    op_count: int
    max_chain_len: int


@dataclass(frozen=True, slots=True)
class SynthesisResult:
    """Everything produced by a synthesis run, held in memory. Nothing is written to disk unless requested."""

    module_name: str
    interface: ModuleInterface
    verilog: str
    support: str
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
        "testbench": (f"test_{name}.py", result.testbench),
        "report": (f"{name}.html", result.report_html),
    }
    written: dict[str, Path] = {}
    for key, (filename, content) in files.items():
        path = directory / filename
        path.write_text(content, encoding="utf-8")
        written[key] = path
    return written
