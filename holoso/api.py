"""The public synthesis entry point."""

from collections.abc import Callable, Mapping
from typing import Any

from .backend_support import support_header, support_verilog
from .backend_verilog import generate
from .format import FloatFormat
from .frontend import lower
from .operators import Op, OpConfig
from .passes import run
from .report import build_report_html
from .result import SynthesisResult
from .schedule import build, interface_of, metrics_of
from .verify.testbench import render_testbench

type Target = Callable[..., Any] | type[object]


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
    metrics = metrics_of(lir)
    verilog = generate(lir)
    testbench = render_testbench(lir, float_format, target) if callable(target) else ""
    return SynthesisResult(
        module_name=module_name,
        interface=interface,
        verilog=verilog,
        support=support_verilog(),
        support_header=support_header(),
        testbench=testbench,
        report_html=build_report_html(lir, interface, metrics, verilog),
        metrics=metrics,
        hir=hir,
        lir=lir,
    )
