"""The public synthesis entry point."""

from collections.abc import Callable
from typing import Any
from dataclasses import dataclass, fields
from pathlib import Path
import inspect
import logging
import os
import re

from ._backend.cocotb import generate as generate_testbench, CocotbOutput
from ._backend.html import generate as generate_html, HtmlOutput
from ._backend.numerical import generate as generate_model, NumericalModel
from ._backend.verilog import generate as generate_verilog, VerilogOutput

from ._eel import lower as lower_frontend
from ._lir import Branch, ControlPort, DataInputPort, DataOutputPort, Port, RegallocTuning, build
from ._mir import MirOptions, lower as lower_to_mir
from ._operators import OperatorOptions
from ._type import FloatFormat, IntFormat

type Target = Callable[..., Any]
"""
Currently supported targets are:
- A plain stateless function. It must be importable, so a lambda is refused: its source cannot be recovered.
- A bound method of a class instance -- stateful. An attribute any reachable method writes becomes a state register,
  and a public one additionally gets a `state_...` output port; an attribute that is only read folds to its value.
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

    int_format: IntFormat
    """
    The integer format (machine word) chosen for this kernel.
    Guaranteed to be at least Options.wint_min bits wide.
    Defines the range of the (saturating) integer arithmetics.
    """

    initiation_interval: tuple[int, int | None]  # (min II, max II or None when not statically determined)
    verilog_output: VerilogOutput
    numerical_model: NumericalModel
    cocotb_output: CocotbOutput
    html_output: HtmlOutput

    frontend_ir: list[str]
    """
    The front-end intermediate representation after each pass, earliest first (raw), final last (most refined).
    """

    def write(self, out_dir: Path | str) -> dict[str, Path]:
        """
        Write every artifact to `out_dir` and return the written paths keyed by filename.
        This is the only Holoso operation that touches the filesystem.
        """
        directory = Path(out_dir)
        directory.mkdir(parents=True, exist_ok=True)
        files: dict[str, str] = {
            f"{self.module_name}.v": self.verilog_output.verilog,
            **self.verilog_output.support_files,
            f"test_{self.module_name}.py": self.cocotb_output.testbench,
            f"{self.module_name}.html": self.html_output.html,
            **{f"{self.module_name}.pass{i}.fir": text for i, text in enumerate(self.frontend_ir)},
        }
        written: dict[str, Path] = {}
        for filename, content in files.items():
            path = directory / filename
            path.write_text(content, encoding="utf-8")
            written[filename] = path
        return written


@dataclass(frozen=True, slots=True)
class Options:
    """Everything configurable that controls how the ZISC machine and its microcode are built."""

    operator: OperatorOptions

    ffmt: FloatFormat = FloatFormat(6, 18)
    """wexp is usually 6..11 bits; wman is usually a multiple of the DSP tile operand width, 18 bits on most FPGAs."""

    wint_min: int = 16
    """
    Lower bound on the native integer bit width.
    The actual integer width may be greater if the kernel uses floats and the floats are wider than this minimum.
    Integers saturate, so the settled word sets their rails; it is reported as SynthesisResult.int_format.
    """

    wmultiplier: int | None = None
    """
    The native DSP slice width, if known: lets the RTL split wide products along it rather than into equal halves,
    usually saving DSP tiles and timing margin.
    """

    ifconv_max_ops: int = int(os.getenv("HOLOSO_IFCONV_MAX_OPS", "8"))
    """Per-arm operation budget for diamond if-conversion; 0 converts only the operation-free diamonds."""

    unroll_max_trips: int = int(os.getenv("HOLOSO_UNROLL_MAX_TRIPS", "1024"))
    """
    A counted `for range(...)` loop with more trips than this lowers to a back-edge loop with a runtime
    counter instead of unrolling; 0 never unrolls. `while` loops and aggregate iteration are budget-bounded.
    """

    ucode_fetch_stages: int = 3
    """Controller fmax/latency trade-off: a deeper fetch raises fmax but costs idle refills on a mispredicted branch."""

    regalloc_effort: int = int(os.getenv("HOLOSO_REGALLOC_EFFORT", "5000"))
    """How hard to search for the best register allocation. Better machines take longer to build."""

    regalloc_reuse_write_cap: int = int(os.getenv("HOLOSO_REG_REUSE_WRITE_CAP", "2"))
    """
    How wide a per-register write select the regfile compaction may build: more compact register file, larger and
    slower steering fabric. A penalty rather than a hard cap, since some registers need an irreducibly wider select.
    """

    regalloc_register_price: float = float(os.getenv("HOLOSO_REG_PRICE", "2.0"))
    """What one register is worth in steering mux arms. Greater values buy fewer registers with heavier steering."""

    def __post_init__(self) -> None:
        if self.wint_min < 2:
            raise ValueError(f"wint_min must be >= 2, got {self.wint_min}")
        if self.wmultiplier is not None and self.wmultiplier < 2:
            raise ValueError(f"wmultiplier must be >= 2 when set, got {self.wmultiplier}")
        if self.ifconv_max_ops < 0:
            raise ValueError(f"ifconv_max_ops must be >= 0, got {self.ifconv_max_ops}")
        if self.unroll_max_trips < 0:
            raise ValueError(f"unroll_max_trips must be >= 0, got {self.unroll_max_trips}")
        if self.ucode_fetch_stages < 1:
            raise ValueError(f"ucode_fetch_stages must be >= 1, got {self.ucode_fetch_stages}")
        if self.regalloc_effort < 0:
            raise ValueError(f"regalloc_effort must be >= 0, got {self.regalloc_effort}")
        if self.regalloc_reuse_write_cap < 1:
            raise ValueError(f"regalloc_reuse_write_cap must be >= 1, got {self.regalloc_reuse_write_cap}")
        if self.regalloc_register_price <= 0:
            raise ValueError(f"regalloc_register_price must be > 0, got {self.regalloc_register_price}")


def _mir_options(options: Options) -> MirOptions:
    """What selection needs of the configuration; the integer width is absent because it is an answer, not a knob."""
    return MirOptions(
        operator=options.operator,
        float_format=options.ffmt,
        wint_min=options.wint_min,
        wmultiplier=options.wmultiplier or 0,
        ifconv_max_ops=options.ifconv_max_ops,
    )


def synthesize(target: Target, /, options: Options, *, name: str | None = None) -> SynthesisResult:
    """
    Synthesize `target` (a plain function or a bound method of a constructed instance) into RTL.
    `options` configures the machine; `name` overrides the generated module name (inferred from target by default).
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)-5.5s %(name)s: %(message)s")  # no-op if already setup
    module_name: str = name or _default_module_name(target)
    _validate_module_name(module_name)
    _logger.info("Synthesis start: module=%r target=%r", module_name, target)
    _logger.info("Options:")
    for field in fields(options):
        value = getattr(options, field.name)
        if field.name == "operator":
            for op_field in fields(value):
                if (configured := getattr(value, op_field.name)) is not None:
                    _logger.info("\toperator.%s: %s", op_field.name, configured)
        else:
            _logger.info("\t%s: %s", field.name, value)

    frontend = lower_frontend(target, options.unroll_max_trips)
    mir = lower_to_mir(frontend.hir, _mir_options(options))
    lir = build(
        mir,
        module_name,
        options.ucode_fetch_stages,
        RegallocTuning(
            effort=options.regalloc_effort,
            reuse_write_cap=options.regalloc_reuse_write_cap,
            register_price=options.regalloc_register_price,
        ),
    )
    _logger.info("LIR ports:\n\t%s", "\n\t".join(f"{port}" for port in lir.ports))

    verilog_output = generate_verilog(lir)
    html_output = generate_html(lir, verilog_output)
    model = generate_model(lir)
    cocotb_output = generate_testbench(model)

    # Only a branch makes the path data-dependent. Counting blocks instead would call a pruned kernel inexact for
    # the jump chain pruning leaves behind, which every transaction walks identically.
    latency_is_exact = not any(isinstance(block.terminator, Branch) for block in lir.blocks)
    ii = (lir.min_initiation_interval, lir.min_initiation_interval if latency_is_exact else None)
    _logger.info("Generated Verilog: %s; II [min,max]: %s cycles", verilog_output, ii)
    return SynthesisResult(
        module_name=module_name,
        ports=lir.ports,
        input_ports=lir.input_ports,
        output_ports=lir.output_ports,
        control_ports=lir.control_ports,
        int_format=lir.int_format,
        initiation_interval=ii,
        verilog_output=verilog_output,
        numerical_model=model,
        cocotb_output=cocotb_output,
        html_output=html_output,
        frontend_ir=frontend.passes,
    )


def _default_module_name(target: Target) -> str:
    if inspect.ismethod(target):
        n = type(target.__self__).__name__
        if "__" not in target.__name__:
            n += f"_{target.__name__}"
        return n
    return str(getattr(target, "__name__", "kernel"))


def _validate_module_name(name: str) -> None:
    if _MODULE_NAME.fullmatch(name) is None:
        raise ValueError(f"module name {name!r} is not a valid identifier; expected [A-Za-z_][A-Za-z0-9_]*")
    if name in _BLACKLIST:
        raise ValueError(f"module name {name!r} is a reserved keyword; choose another name")
    if name.lower().startswith("holoso"):
        raise ValueError(f"module name {name!r} uses the reserved 'holoso' prefix; choose another name")


_MODULE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# Keywords from supported HDLs etc. that are not valid module names. Includes Verilog and VHDL keywords.
_BLACKLIST = frozenset("""
always and assign automatic begin buf bufif0 bufif1 case casex casez cell cmos config deassign default defparam
design disable edge else end endcase endconfig endfunction endgenerate endmodule endprimitive endspecify endtable
endtask event for force forever fork function generate genvar highz0 highz1 if ifnone incdir include initial inout
input instance integer join large liblist library localparam macromodule medium module nand negedge nmos nor
noshowcancelled not notif0 notif1 or output parameter pmos posedge primitive pull0 pull1 pulldown pullup
pulsestyle_onevent pulsestyle_ondetect rcmos real realtime reg release repeat rnmos rpmos rtran rtranif0 rtranif1
scalared showcancelled signed small specify specparam strong0 strong1 supply0 supply1 table task time tran tranif0
tranif1 tri tri0 tri1 triand trior trireg unsigned use uwire vectored wait wand weak0 weak1 while wire wor xnor xor
abs access after alias all architecture array assert attribute block body buffer bus component configuration constant
context disconnect downto elsif entity exit file generic group guarded impure in inertial is label linkage literal
loop map mod new next null of on open others out package port postponed procedure process protected pure
range record register reject rem report return rol ror select severity signal shared sla sll sra srl subtype
then to transport type unaffected units until variable when with
accept_on always_comb always_ff always_latch assume before bind bins binsof bit break byte chandle checker
class clocking const constraint continue cover covergroup coverpoint cross dist do endchecker endclass
endclocking endgroup endinterface endpackage endprogram endproperty endsequence enum expect export extends extern
final first_match foreach forkjoin global iff ignore_bins illegal_bins implements implies import inside int interface
intersect join_any join_none let logic longint matches modport nettype packed priority program property rand randc
randcase randsequence ref reject_on restrict s_always s_eventually s_nexttime s_until s_until_with sequence shortint
shortreal soft solve static string strong struct super sync_accept_on sync_reject_on tagged this throughout
timeprecision timeunit typedef union unique unique0 var virtual void wait_order weak wildcard within
assume_guarantee eventually fairness interconnect local nexttime restrict_guarantee untyped until_with vmode vprop vunit
false none true as async await def del elif except finally from lambda nonlocal pass raise try yield
""".split())
