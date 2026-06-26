import os
import shlex
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from holoso import SynthesisResult

from ._flow_id import FlowId
from ._ooc import OocWrapper

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_ROOT = REPO_ROOT / "build" / "synth"
DEFAULT_TIMEOUT_S = float(os.environ.get("HOLOSO_SYNTH_TIMEOUT_S", "3600"))


def new_build_dir(prefix: str) -> Path:
    BUILD_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return Path(tempfile.mkdtemp(prefix=f"{prefix}_{stamp}_", dir=BUILD_ROOT))


def run_logged(argv: list[str | Path], log_path: Path, *, cwd: Path, timeout_s: float = DEFAULT_TIMEOUT_S) -> None:
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
    name: str
    used: int
    available: int | None = None

    @property
    def fraction(self) -> float | None:
        """May exceed 1.0 when the design did not fit."""
        if not self.available:
            return None
        return self.used / self.available


@dataclass(frozen=True, slots=True)
class SynthReport:
    flow: FlowId
    target_frequency_MHz: float
    fmax_MHz: float
    slack_ns: float
    resources: Mapping[str, ResourceUse]
    artifact_dir: Path
    logs: list[Path]


@dataclass(frozen=True, slots=True)
class SourceFile:
    path: Path  # relative to the artifact directory
    content: str


@dataclass(frozen=True, slots=True)
class CommandSpec:
    """The recipe's tool invocations are recorded only for manual reproduction; the runner executes them itself."""

    argv: list[str | Path]


@dataclass(frozen=True, slots=True)
class SynthArtifact:
    flow: FlowId
    top: str
    files: list[SourceFile]
    commands: list[CommandSpec]
    runner: Callable[[Path], SynthReport]

    def write(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for source in self.files:
            target = directory / source.path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(source.content, encoding="utf-8")

    def synthesize(self, directory: Path | None = None) -> SynthReport:
        target = directory if directory is not None else new_build_dir(self.flow)
        self.write(target)
        return self.runner(target)


def assemble(result: SynthesisResult, wrapper: OocWrapper) -> list[SourceFile]:
    """
    A generated module instantiates only support-library modules, all of which live inside the single bundled
    ``holoso_support.v``; nothing else is needed. Everything is bundled in memory so a recipe directory is
    self-contained.
    """
    return [SourceFile(Path(name), content) for name, content in result.verilog_output.support_files.items()] + [
        SourceFile(Path(f"{result.module_name}.v"), result.verilog_output.verilog),
        SourceFile(Path(f"{wrapper.top}.v"), wrapper.verilog),
    ]
