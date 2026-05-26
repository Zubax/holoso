"""The public synthesis entry point."""

from collections.abc import Callable, Mapping
from typing import Any
from dataclasses import dataclass
from pathlib import Path

from ._backend.cocotb import generate as generate_testbench, CocotbOutput
from ._backend.html import generate as generate_html, HtmlOutput
from ._backend.numerical import generate as generate_model, NumericalModel
from ._backend.verilog import generate as generate_verilog, VerilogOutput

from ._format import FloatFormat
from ._frontend import lower
from ._operators import Op, OpConfig
from ._passes import run
from ._schedule import build, interface_of
from ._interface import ModuleInterface

type Target = Callable[..., Any] | type[object]


@dataclass(frozen=True, slots=True)
class SynthesisResult:
    """Everything produced by a synthesis run, held in memory. Nothing is written to disk unless requested."""

    module_name: str
    interface: ModuleInterface

    verilog_output: VerilogOutput
    numerical_model: NumericalModel
    cocotb_output: CocotbOutput
    html_output: HtmlOutput

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


def synthesize(
    target: Target,
    *,
    float_format: FloatFormat,
    ops: OpConfig,
    parameters: Mapping[str, object] | None = None,
    entry: str = "__call__",
    name: str | None = None,
    operator_instances: Mapping[type[Op], int] | None = None,
) -> SynthesisResult:
    """
    Synthesize ``target`` (a function or class object) into a Verilog ZISC FSM, returned in memory.

    ``ops`` is the operator configuration, constructed explicitly by the caller: each field fixes one operator's
    parameters, including any pipeline-stage knobs that lengthen its latency to ease timing closure. ``parameters``
    overrides a class's keyword-only ``__init__`` defaults; ``entry`` selects the analyzed method for a class
    (default ``__call__``); ``name`` overrides the generated module name; ``operator_instances`` sets the number of
    instances per operator class for scheduling (default one each).
    """
    hir = run(lower(target, float_format), ops)
    module_name: str = name if name is not None else str(getattr(target, "__name__", "holoso_module"))
    lir = build(hir, module_name, instances=operator_instances)
    interface = interface_of(lir)
    verilog_output = generate_verilog(lir)
    html_output = generate_html(lir, interface, verilog_output)
    model = generate_model(lir)
    cocotb_output = generate_testbench(model, interface)
    return SynthesisResult(
        module_name=module_name,
        interface=interface,
        verilog_output=verilog_output,
        numerical_model=model,
        cocotb_output=cocotb_output,
        html_output=html_output,
    )
