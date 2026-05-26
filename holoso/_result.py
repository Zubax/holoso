"""The in-memory result of a synthesis run, plus the only filesystem-touching helper."""

from dataclasses import dataclass
from pathlib import Path

from ._backend.cocotb import CocotbOutput
from ._backend.html import HtmlOutput
from ._backend.numerical import NumericalModel
from ._backend.verilog import VerilogOutput
from ._interface import ModuleInterface, SynthesisMetrics


@dataclass(frozen=True, slots=True)
class SynthesisResult:
    """Everything produced by a synthesis run, held in memory. Nothing is written to disk unless requested."""

    module_name: str
    interface: ModuleInterface
    verilog_output: VerilogOutput
    model: NumericalModel
    cocotb_output: CocotbOutput
    html_output: HtmlOutput
    metrics: SynthesisMetrics

    def write(self, out_dir: Path | str) -> dict[str, Path]:
        """
        Write every artifact to ``out_dir`` and return the written paths keyed by filename.

        This is the only Holoso operation that touches the filesystem.
        """
        directory = Path(out_dir)
        directory.mkdir(parents=True, exist_ok=True)
        files: dict[str, str] = {
            f"{self.module_name}.v": self.verilog_output.verilog,
            **self.verilog_output.support_files,
            f"test_{self.module_name}.py": self.cocotb_output.testbench,
            f"{self.module_name}.html": self.html_output.html,
        }
        written: dict[str, Path] = {}
        for filename, content in files.items():
            path = directory / filename
            path.write_text(content, encoding="utf-8")
            written[filename] = path
        return written
