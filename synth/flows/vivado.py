"""
AMD/Xilinx Vivado out-of-context synthesis verification flow.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from holoso.result import SynthesisResult

from .._detect import find_tool, require_tool
from .._ooc import build_ooc_wrapper
from .._synth import CommandSpec, ResourceUse, SourceFile, SynthArtifact, SynthReport, assemble, run_logged
from . import Flow

_TCL = "run_vivado.tcl"
_XDC = "ooc.xdc"
_LOG = "vivado_run.log"  # our captured stdout; Vivado writes its own vivado.log in the same dir
_UTIL = "utilization.rpt"
_TIMING = "worst_path.rpt"


@dataclass(frozen=True, slots=True)
class XilinxPart:
    """An AMD/Xilinx part for the Vivado flow, e.g. ``xc7a35tcsg324-1``."""

    name: str = "xc7a35tcsg324-1"


@dataclass(frozen=True, slots=True)
class VivadoFlow(Flow):
    """Vivado synthesis + place-and-route, out of context."""

    part: XilinxPart = field(default_factory=XilinxPart)
    target_frequency_MHz: float = 100.0
    retiming: bool = True
    options: dict[str, Any] = field(default_factory=dict)

    def available(self) -> bool:
        return find_tool("vivado") is not None

    def prepare(self, result: SynthesisResult, extra_rtl: list[Path]) -> SynthArtifact:
        wrapper = build_ooc_wrapper(result)
        top = wrapper.top
        src = assemble(result, wrapper, extra_rtl)
        verilog_paths = [sf.path for sf in src if sf.path.suffix == ".v"]

        period_ns = 1000.0 / self.target_frequency_MHz
        xdc = SourceFile(Path(_XDC), f"create_clock -name clk -period {period_ns:.4f} [get_ports clk]\n")
        tcl = SourceFile(Path(_TCL), _tcl(top, self.part.name, verilog_paths, self.retiming))
        commands = [CommandSpec(["vivado", "-mode", "batch", "-source", _TCL, "-nojournal"])]

        def runner(directory: Path) -> SynthReport:
            run_logged(
                [require_tool("vivado"), "-mode", "batch", "-source", _TCL, "-nojournal"],
                directory / _LOG,
                cwd=directory,
            )
            return _parse(self, directory)

        return SynthArtifact(flow="vivado", top=top, files=[*src, xdc, tcl], commands=commands, runner=runner)


def _tcl(top: str, part: str, verilog_paths: list[Path], retiming: bool) -> str:
    read_list = " ".join(path.as_posix() for path in verilog_paths)
    retiming_arg = " -retiming" if retiming else ""
    return "\n".join(
        [
            f"read_verilog [list {read_list}]",
            f"read_xdc {_XDC}",
            f"synth_design -top {top} -part {part} -mode out_of_context{retiming_arg}",
            "opt_design",
            "place_design",
            "route_design",
            f"report_utilization -file {_UTIL}",
            f"report_timing -delay_type max -max_paths 1 -nworst 1 -file {_TIMING}",
            "",
        ]
    )


def _parse_utilization(text: str) -> dict[str, ResourceUse]:
    # Each resource row is "| <Site Type> | <Used> | <Fixed> | [<Prohibited> |] <Available> | <Util%> |".
    # Available is the cell just before the trailing Util% column, which is robust to the optional Prohibited column.
    resources: dict[str, ResourceUse] = {}
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.split("|")]
        if len(cells) >= 6 and cells[1] and cells[2].isdigit() and cells[-3].isdigit():
            resources[cells[1]] = ResourceUse(cells[1], int(cells[2]), int(cells[-3]))
    return resources


def _parse(flow: VivadoFlow, directory: Path) -> SynthReport:
    timing = (directory / _TIMING).read_text(errors="replace")
    slack_match = re.search(r"Slack\s*\((?:MET|VIOLATED)\)\s*:\s*(-?[0-9.]+)\s*ns", timing)
    if not slack_match:
        raise RuntimeError(f"Vivado timing report had no slack value; see {directory / _TIMING}")
    slack_ns = float(slack_match.group(1))
    target = flow.target_frequency_MHz
    achieved_period_ns = 1000.0 / target - slack_ns
    fmax_MHz = 1000.0 / achieved_period_ns if achieved_period_ns > 0 else float("inf")

    resources = _parse_utilization((directory / _UTIL).read_text(errors="replace"))

    return SynthReport(
        flow="vivado",
        target_frequency_MHz=target,
        fmax_MHz=fmax_MHz,
        slack_ns=slack_ns,
        resources=resources,
        artifact_dir=directory,
        logs=[directory / _LOG, directory / _TIMING, directory / _UTIL],
    )
