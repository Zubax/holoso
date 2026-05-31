"""The public synthesis entry point."""

from collections.abc import Callable
from typing import Any
from dataclasses import dataclass, fields
from pathlib import Path
import logging

from ._backend.cocotb import generate as generate_testbench, CocotbOutput
from ._backend.html import generate as generate_html, HtmlOutput
from ._backend.numerical import generate as generate_model, NumericalModel
from ._backend.verilog import generate as generate_verilog, VerilogOutput

from ._frontend import lower as lower_frontend
from ._hir import optimize
from ._lir import ControlPort, DataInputPort, DataOutputPort, Port, build
from ._mir import lower as lower_to_mir
from ._operators import OpConfig

type Target = Callable[..., Any] | type[object]

_logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SynthesisResult:
    """Everything produced by a synthesis run, held in memory. Nothing is written to disk unless requested."""

    module_name: str

    ports: list[Port]
    input_ports: list[DataInputPort]
    output_ports: list[DataOutputPort]
    control_ports: list[ControlPort]

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


def synthesize(target: Target, *, ops: OpConfig, entry: str = "__call__", name: str | None = None) -> SynthesisResult:
    """
    Synthesize ``target`` (a function or class object) into a Verilog ZISC FSM, returned in memory.

    ``ops`` is the operator configuration, constructed explicitly by the caller: each field fixes one operator's
    float format and parameters, including any pipeline-stage knobs that lengthen its latency to ease timing closure.
    ``entry`` selects the analyzed method for a class (default ``__call__``);
    ``name`` overrides the generated module name.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)-5.5s %(name)s: %(message)s")  # no-op if already setup
    module_name: str = name if name is not None else str(getattr(target, "__name__", "holoso_module"))
    _logger.info("Synthesis start module=%r entry=%r target=%r", module_name, entry, target)
    _logger.info("Configured operators:")
    for field in fields(ops):
        _logger.info("\t%s: %s", field.name, getattr(ops, field.name))

    hir = optimize(lower_frontend(target))
    _logger.info("HIR:\n\tinputs=%s\n\toutputs=%s\n\thir_nodes=%d", hir.input_ids, hir.outputs, len(hir.nodes))

    mir = lower_to_mir(hir, ops)

    lir = build(mir, module_name)
    _logger.info("LIR ports:\n\t%s", "\n\t".join(f"{port}" for port in lir.ports))

    verilog_output = generate_verilog(lir)
    html_output = generate_html(lir, verilog_output)
    model = generate_model(lir)
    cocotb_output = generate_testbench(model)

    _logger.info("Generated Verilog: %s", verilog_output)
    return SynthesisResult(
        module_name=module_name,
        ports=lir.ports,
        input_ports=lir.input_ports,
        output_ports=lir.output_ports,
        control_ports=lir.control_ports,
        verilog_output=verilog_output,
        numerical_model=model,
        cocotb_output=cocotb_output,
        html_output=html_output,
    )
