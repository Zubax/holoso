"""Shared helpers for Yosys/nextpnr-based synthesis tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import json
import os
import shlex
import shutil
import subprocess

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_ROOT = REPO_ROOT / "build" / "synth"
YOSYS_TIMEOUT_S = float(os.environ.get("YOSYS_SYNTH_TIMEOUT_S", "1800"))
NEXTPNR_TIMEOUT_S = float(os.environ.get("YOSYS_NEXTPNR_TIMEOUT_S", "1800"))

# Keep the measurement boundary flops present so primary I/O paths are represented as register-to-register timing.
SYNTH_REG_ATTR = '(* keep = "true", syn_preserve = "true", dont_touch = "true" *)'


@dataclass(frozen=True)
class ClockTiming:
    name: str
    achieved_mhz: float
    constraint_mhz: float

    @property
    def slack_ns(self) -> float:
        return (1000.0 / self.constraint_mhz) - (1000.0 / self.achieved_mhz)


@dataclass(frozen=True)
class ResourceUtilization:
    name: str
    used: int
    available: int | None

    @property
    def percent(self) -> float | None:
        if self.available is None or self.available <= 0:
            return None
        return 100.0 * self.used / self.available


def clean_build_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def require_executable(name: str) -> Path:
    path = shutil.which(name)
    if path is None:
        raise AssertionError(f"required executable {name!r} was not found on PATH")
    return Path(path)


def _format_yosys_path(path: Path) -> str:
    resolved = str(path)
    if any(char.isspace() for char in resolved):
        return '"' + resolved.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return resolved


def write_synthesis_script(
    path: Path,
    *,
    include_dirs: Sequence[Path],
    sources: Sequence[Path],
    top: str,
    commands: Sequence[str],
) -> None:
    include_args = " ".join(f"-I {_format_yosys_path(item)}" for item in include_dirs)
    read_args = f"{include_args} " if include_args else ""
    path.write_text(
        "\n".join(
            [f"read_verilog {read_args}{_format_yosys_path(source)}" for source in sources]
            + [
                f"hierarchy -check -top {top}",
            ]
            + list(commands)
            + [""]
        )
    )


def run_logged(command: Sequence[str | Path], log_path: Path, *, timeout_s: float) -> None:
    rendered = [str(item) for item in command]
    print("$ " + " ".join(shlex.quote(item) for item in rendered), flush=True)
    print(f"  log: {log_path}", flush=True)
    with log_path.open("w") as log:
        command_line = "$ " + " ".join(shlex.quote(item) for item in rendered) + "\n\n"
        log.write(command_line)
        log.flush()
        try:
            subprocess.run(
                rendered,
                cwd=REPO_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            log.write(f"\n\n[holoso synth] command exceeded {timeout_s:g}s timeout and was killed\n")
            raise AssertionError(f"{rendered[0]} timed out after {timeout_s:g}s; see {log_path}") from exc
        except subprocess.CalledProcessError as exc:
            raise AssertionError(f"{rendered[0]} failed with exit code {exc.returncode}; see {log_path}") from exc


def run_yosys(script: Path, log_path: Path) -> None:
    run_logged([require_executable("yosys"), "-s", script], log_path, timeout_s=YOSYS_TIMEOUT_S)


def run_nextpnr_ecp5(args: Sequence[str | Path], log_path: Path) -> None:
    run_logged([require_executable("nextpnr-ecp5"), *args], log_path, timeout_s=NEXTPNR_TIMEOUT_S)


def read_json(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise AssertionError(f"{path} does not contain a JSON object")
    return data


def clock_timings(report: dict[str, object]) -> list[ClockTiming]:
    fmax = report.get("fmax")
    if not isinstance(fmax, dict):
        return []
    timings: list[ClockTiming] = []
    for name, item in fmax.items():
        if not isinstance(name, str) or not isinstance(item, dict):
            continue
        achieved = item.get("achieved")
        constraint = item.get("constraint")
        if isinstance(achieved, int | float) and isinstance(constraint, int | float):
            timings.append(ClockTiming(name, float(achieved), float(constraint)))
    return timings


def timing_summary(timings: Sequence[ClockTiming]) -> str:
    return ", ".join(
        f"{item.name}: achieved {item.achieved_mhz:.2f} MHz, target {item.constraint_mhz:.2f} MHz, "
        f"slack {item.slack_ns:.3f} ns"
        for item in timings
    )


def utilization_used(report: dict[str, object], resource: str) -> int | None:
    utilization = report.get("utilization")
    if not isinstance(utilization, dict):
        return None
    item = utilization.get(resource)
    if not isinstance(item, dict):
        return None
    used = item.get("used")
    return int(used) if isinstance(used, int) else None


def resource_utilization(report: dict[str, object], resources: Sequence[str]) -> list[ResourceUtilization]:
    utilization = report.get("utilization")
    if not isinstance(utilization, dict):
        return []

    result: list[ResourceUtilization] = []
    for resource in resources:
        item = utilization.get(resource)
        if not isinstance(item, dict):
            continue
        used = item.get("used")
        available = item.get("available")
        if isinstance(used, int):
            result.append(
                ResourceUtilization(
                    resource,
                    used,
                    available if isinstance(available, int) else None,
                )
            )
    return result


def utilization_summary(report: dict[str, object], resources: Sequence[str]) -> str:
    lines = []
    for item in resource_utilization(report, resources):
        if item.percent is None:
            lines.append(f"{item.name}: {item.used}")
        else:
            lines.append(f"{item.name}: {item.used}/{item.available} ({item.percent:.2f}%)")
    return "\n".join(lines) if lines else "not reported"


def read_cell_counts(netlist: Path, top: str) -> dict[str, int]:
    data = read_json(netlist)
    modules = data.get("modules")
    if not isinstance(modules, dict):
        return {}
    module = modules.get(top)
    if not isinstance(module, dict):
        return {}
    cells = module.get("cells")
    if not isinstance(cells, dict):
        return {}
    counts: dict[str, int] = {}
    for cell in cells.values():
        if isinstance(cell, dict) and isinstance(cell.get("type"), str):
            cell_type = cell["type"]
            counts[cell_type] = counts.get(cell_type, 0) + 1
    return counts
