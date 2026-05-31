"""
Yosys + nextpnr out-of-context synthesis verification flow.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from holoso import SynthesisResult

from .._detect import find_tool, require_tool
from .._ooc import build_ooc_wrapper
from .._synth import RTL_SUBDIR, CommandSpec, ResourceUse, SourceFile, SynthArtifact, SynthReport, assemble, run_logged
from . import Flow

_SCRIPT = "synth.ys"
_NETLIST = "ooc.json"
_ROUTED = "routed.json"
_REPORT = "report.json"
_YOSYS_LOG = "yosys.log"
_NEXTPNR_LOG = "nextpnr.log"


@dataclass(frozen=True, slots=True)
class Ecp5Device:
    """A Lattice ECP5 target as nextpnr-ecp5 wants it: a size flag plus a package and speed grade."""

    size: str = "25k"  # nextpnr-ecp5 --<size>
    package: str = "CABGA381"
    speed_grade: int = 6


@dataclass(frozen=True, slots=True)
class YosysEcp5Flow(Flow):
    """Yosys synthesis + nextpnr-ecp5 place-and-route, out of context."""

    device: Ecp5Device = field(default_factory=Ecp5Device)
    target_frequency_MHz: float = 100.0
    options: dict[str, Any] = field(default_factory=dict)

    def available(self) -> bool:
        return find_tool("yosys") is not None and find_tool("nextpnr-ecp5") is not None

    def prepare(self, result: SynthesisResult, extra_rtl: list[Path]) -> SynthArtifact:
        wrapper = build_ooc_wrapper(result)
        top = wrapper.top
        src = assemble(result, wrapper, extra_rtl)
        script = SourceFile(Path(_SCRIPT), _yosys_script(top, result.module_name))

        nextpnr_args: list[str] = [
            f"--{self.device.size}",
            "--package",
            self.device.package,
            "--speed",
            str(self.device.speed_grade),
            "--freq",
            f"{self.target_frequency_MHz:g}",
            "--out-of-context",
            "--timing-allow-fail",
            "--json",
            _NETLIST,
            "--write",
            _ROUTED,
            "--report",
            _REPORT,
            *(str(a) for a in self.options.get("nextpnr_extra", ())),
        ]
        commands = [
            CommandSpec(["yosys", "-s", _SCRIPT]),
            CommandSpec(["nextpnr-ecp5", *nextpnr_args]),
        ]

        def runner(directory: Path) -> SynthReport:
            run_logged([require_tool("yosys"), "-s", _SCRIPT], directory / _YOSYS_LOG, cwd=directory)
            run_logged([require_tool("nextpnr-ecp5"), *nextpnr_args], directory / _NEXTPNR_LOG, cwd=directory)
            return _parse(self, directory)

        return SynthArtifact(flow="yosys-ecp5", top=top, files=[*src, script], commands=commands, runner=runner)


def _yosys_script(top: str, dut_module: str) -> str:
    # Read only the files we generate; pull the instantiated primitives from the bundled rtl/ libdir on demand.
    libdir = RTL_SUBDIR.as_posix()
    return "\n".join(
        [
            "read_verilog -I . holoso_support.v",
            f"read_verilog -I . {dut_module}.v",
            f"read_verilog -I . {top}.v",
            f"hierarchy -check -top {top} -libdir {libdir}",
            # Retiming underperforms on many OOC targets today, apparently around the fmul DSP/packer boundary.
            # Revisit this if future Yosys/nextpnr versions or RTL changes make retiming consistently beneficial.
            # ABC9 also appears to be sucky at least on ECP5.
            f"synth_ecp5 -top {top} -noiopad -dff -noabc9 -run begin:check",
            "clean",
            f"hierarchy -check -top {top}",
            "stat",
            "check -noinit",
            "blackbox =A:whitebox",
            f"write_json {_NETLIST}",
            "",
        ]
    )


def _parse(flow: YosysEcp5Flow, directory: Path) -> SynthReport:
    report = json.loads((directory / _REPORT).read_text())

    fmax_map = report.get("fmax", {})
    achieved = [
        float(item["achieved"])
        for item in fmax_map.values()
        if isinstance(item, dict) and isinstance(item.get("achieved"), int | float)
    ]
    if not achieved:
        raise RuntimeError(f"nextpnr did not report an achieved fmax; see {directory / _NEXTPNR_LOG}")
    fmax_MHz = min(achieved)
    target = flow.target_frequency_MHz

    resources: dict[str, ResourceUse] = {}
    utilization = report.get("utilization", {})
    if isinstance(utilization, dict):
        for name, item in utilization.items():
            if isinstance(item, dict) and isinstance(item.get("used"), int):
                available = item.get("available")
                resources[name] = ResourceUse(name, item["used"], available if isinstance(available, int) else None)

    return SynthReport(
        flow="yosys-ecp5",
        target_frequency_MHz=target,
        fmax_MHz=fmax_MHz,
        slack_ns=1000.0 / target - 1000.0 / fmax_MHz,
        resources=resources,
        artifact_dir=directory,
        logs=[directory / _YOSYS_LOG, directory / _NEXTPNR_LOG],
    )
