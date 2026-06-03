"""
Core building blocks for the synthesis-evaluation harness.
"""

import os
import shlex
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from holoso import SynthesisResult

from ._ooc import OocWrapper

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_ROOT = REPO_ROOT / "build" / "synth"
DEFAULT_TIMEOUT_S = float(os.environ.get("HOLOSO_SYNTH_TIMEOUT_S", "3600"))

# Caller-supplied RTL is bundled under this subdirectory so a tool's library search can target just those files.
RTL_SUBDIR = Path("rtl")


def new_build_dir(prefix: str) -> Path:
    """Create and return a fresh build directory under ``BUILD_ROOT`` for one synthesis run."""
    BUILD_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return Path(tempfile.mkdtemp(prefix=f"{prefix}_{stamp}_", dir=BUILD_ROOT))


def run_logged(argv: list[str | Path], log_path: Path, *, cwd: Path, timeout_s: float = DEFAULT_TIMEOUT_S) -> None:
    """Run ``argv`` in ``cwd``, teeing stdout+stderr to ``log_path``; raise on nonzero exit or timeout."""
    rendered = [str(item) for item in argv]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("$ " + " ".join(shlex.quote(item) for item in rendered), flush=True)
    print(f"  cwd: {cwd}\n  log: {log_path}", flush=True)
    with log_path.open("w") as log:
        log.write("$ " + " ".join(shlex.quote(item) for item in rendered) + "\n\n")
        log.flush()
        try:
            subprocess.run(rendered, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            log.write(f"\n\n[holoso synth] command exceeded {timeout_s:g}s timeout and was killed\n")
            raise


@dataclass(frozen=True, slots=True)
class ResourceUse:
    """One fabric-resource figure: how many were used out of how many available (when the tool reports it)."""

    name: str
    used: int
    available: int | None = None

    @property
    def fraction(self) -> float | None:
        """Used divided by available, nominally in [0, 1]; may exceed 1.0 when the design did not fit."""
        if not self.available:
            return None
        return self.used / self.available


@dataclass(frozen=True, slots=True)
class SynthReport:
    """The parsed outcome of one synthesis + place-and-route run."""

    flow: str
    target_frequency_MHz: float
    fmax_MHz: float
    slack_ns: float
    resources: Mapping[str, ResourceUse]
    artifact_dir: Path
    logs: list[Path]


@dataclass(frozen=True, slots=True)
class SourceFile:
    """One file to materialize, addressed by a path relative to the artifact directory."""

    path: Path
    content: str


@dataclass(frozen=True, slots=True)
class CommandSpec:
    """One tool invocation, run with the artifact directory as the working directory (for manual reproduction)."""

    argv: list[str | Path]


@dataclass(frozen=True, slots=True)
class SynthArtifact:
    """A self-contained synthesis recipe plus a bound runner that executes and parses it."""

    flow: str
    top: str
    files: list[SourceFile]
    commands: list[CommandSpec]
    runner: Callable[[Path], SynthReport]

    def write(self, directory: Path) -> None:
        """Materialize every file into ``directory`` (creating nested directories as needed)."""
        directory.mkdir(parents=True, exist_ok=True)
        for source in self.files:
            target = directory / source.path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(source.content, encoding="utf-8")

    def synthesize(self, directory: Path | None = None) -> SynthReport:
        """Write the recipe (to ``directory`` or a fresh build dir) and run the flow, returning the parsed result."""
        target = directory if directory is not None else new_build_dir(self.flow)
        self.write(target)
        return self.runner(target)


def assemble(result: SynthesisResult, wrapper: OocWrapper, extra_rtl: list[Path]) -> list[SourceFile]:
    """Bundle the wrapper, the generated module, the support files, and the caller-supplied RTL into one source set.

    A generated module instantiates backend-provided support files plus additional RTL primitives (the Kulibin float
    modules) that the caller supplies -- the harness does not go looking for them. Everything is bundled as in-memory
    :class:`SourceFile`s so a recipe directory is self-contained.
    """
    files = [SourceFile(Path(name), content) for name, content in result.verilog_output.support_files.items()] + [
        SourceFile(Path(f"{result.module_name}.v"), result.verilog_output.verilog),
        SourceFile(Path(f"{wrapper.top}.v"), wrapper.verilog),
    ]
    files += [SourceFile(RTL_SUBDIR / path.name, path.read_text(encoding="utf-8")) for path in extra_rtl]
    return files
