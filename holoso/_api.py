"""The public synthesis entry point."""

from collections.abc import Callable
from typing import Any
from dataclasses import dataclass, fields
from pathlib import Path
import inspect
import logging
import re

from ._backend.cocotb import generate as generate_testbench, CocotbOutput
from ._backend.html import generate as generate_html, HtmlOutput
from ._backend.numerical import generate as generate_model, NumericalModel
from ._backend.verilog import generate as generate_verilog, VerilogOutput

from ._frontend import lower as lower_frontend
from ._hir import optimize
from ._lir import ControlPort, DataInputPort, DataOutputPort, Port, build
from ._mir import lower as lower_to_mir
from ._operators import OpConfig

type Target = Callable[..., Any]
"""
Currently supported targets are:
- A plain stateless function or lambda.
- A bound method of a class instance -- stateful. Public attributes become additional output ports.
- Later on we may potentially add support for multiple methods per class, where the generated module will provide
  a selector port to choose which method to execute, all sharing the same state. In this case we would accept
  a tuple containing the class type and a list of its unbound methods. This remains to be seen.
"""

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


def synthesize(target: Target, /, ops: OpConfig, *, name: str | None = None) -> SynthesisResult:
    """
    Synthesize ``target`` (a plain function or a bound method of a constructed instance) into RTL.
    ``ops`` is the operator configuration, constructed explicitly by the caller: each field fixes one operator's
    float format and parameters, including any pipeline-stage knobs that lengthen its latency to ease timing closure.
    ``name`` overrides the generated module name (inferred from target by default).
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)-5.5s %(name)s: %(message)s")  # no-op if already setup
    module_name: str = name or _default_module_name(target)
    _validate_module_name(module_name)
    _logger.info("Synthesis start: module=%r target=%r", module_name, target)
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


def _default_module_name(target: Target) -> str:
    if inspect.ismethod(target):
        n = type(target.__self__).__name__
        if "__" not in target.__name__:
            n += f"_{target.__name__}"
        return n
    return str(getattr(target, "__name__", "kernel"))


_MODULE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# Keywords from supported HDLs etc. that are not valid module names.
_BLACKLIST = frozenset("""
always and assign automatic begin buf bufif0 bufif1 case casex casez cell cmos config deassign default defparam
design disable edge else end endcase endconfig endfunction endgenerate endmodule endprimitive endspecify endtable
endtask event for force forever fork function generate genvar highz0 highz1 if ifnone incdir include initial inout
input instance integer join large liblist library localparam macromodule medium module nand negedge nmos nor
noshowcancelled not notif0 notif1 or output parameter pmos posedge primitive pull0 pull1 pulldown pullup
pulsestyle_onevent pulsestyle_ondetect rcmos real realtime reg release repeat rnmos rpmos rtran rtranif0 rtranif1
scalared showcancelled signed small specify specparam strong0 strong1 supply0 supply1 table task time tran tranif0
tranif1 tri tri0 tri1 triand trior trireg unsigned use uwire vectored wait wand weak0 weak1 while wire wor xnor xor
""".split())


def _validate_module_name(name: str) -> None:
    if _MODULE_NAME.fullmatch(name) is None:
        raise ValueError(f"module name {name!r} is not a valid identifier; expected [A-Za-z_][A-Za-z0-9_]*")
    if name in _BLACKLIST:
        raise ValueError(f"module name {name!r} is a reserved keyword; choose another name")
    if name.lower().startswith("holoso"):
        raise ValueError(f"module name {name!r} uses the reserved 'holoso' prefix; choose another name")
