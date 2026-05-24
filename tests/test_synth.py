"""End-to-end tests of the public synthesize() API, the report, artifact writing, and the generated testbench."""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path

import pytest

import holoso
from holoso import FloatFormat, StageConfig


def _kernel(a, b):  # type: ignore[no-untyped-def]  # module-level so inspect.getsource works
    return (a - b) * 0.25 + a * b


def test_synthesize_small_kernel_result() -> None:
    result = holoso.synthesize(_kernel, float_format=FloatFormat(8, 24))
    assert result.module_name == "_kernel"
    assert "module _kernel" in result.verilog
    assert "holoso_regfile" in result.support
    assert '`include "holoso_support.vh"' in result.support
    assert "`ifndef HOLOSO_REGFILE_VH" not in result.support
    assert "`HOLOSO_REGFILE_LANE" in result.support_header
    assert "@cocotb.test()" in result.testbench
    assert "<html" in result.report_html.lower()
    assert result.metrics.op_count >= 3
    names = [p.name for p in result.interface.ports]
    assert "in_a" in names and "out_0" in names and "err_cyc" in names


def test_synthesize_threads_pipeline_stages() -> None:
    base = holoso.synthesize(_kernel, float_format=FloatFormat(8, 24))
    staged = holoso.synthesize(
        _kernel, float_format=FloatFormat(8, 24), stages=StageConfig(fadd_decode=1, fmul_product=1)
    )
    assert "STAGE_" not in base.verilog  # default stages emit no STAGE_* instance params
    assert ".STAGE_DECODE(1)" in staged.verilog and ".STAGE_PRODUCT(1)" in staged.verilog
    assert staged.metrics.ii_cycles > base.metrics.ii_cycles  # the added stages lengthen the schedule


def test_rejects_non_finite_constants() -> None:
    def overflow(a):  # type: ignore[no-untyped-def]
        return a + 1e400  # overflows to +inf, not representable in the ZKF format

    def folds_to_nan(a):  # type: ignore[no-untyped-def]
        return a + (1e400 - 1e400)  # inf - inf const-folds to NaN

    for fn in (overflow, folds_to_nan):
        with pytest.raises(holoso.UnsupportedConstruct):
            holoso.synthesize(fn, float_format=FloatFormat(8, 24))


def test_generated_testbench_is_valid_python() -> None:
    result = holoso.synthesize(_kernel, float_format=FloatFormat(8, 24))
    compile(result.testbench, "<generated-testbench>", "exec")


def test_write_artifacts(tmp_path: Path) -> None:
    result = holoso.synthesize(_kernel, float_format=FloatFormat(8, 24))
    paths = result.write(tmp_path)
    assert set(paths) == {"verilog", "support", "support_header", "testbench", "report"}
    assert (tmp_path / "_kernel.v").exists()
    assert (tmp_path / "test__kernel.py").exists()
    assert (tmp_path / "_kernel.html").exists()
    assert (tmp_path / "holoso_support.v").exists()
    assert (tmp_path / "holoso_support.vh").exists()


def test_report_has_expected_sections() -> None:
    result = holoso.synthesize(_kernel, float_format=FloatFormat(8, 24))
    report = result.report_html
    header_html = report.split("<pre class='modhdr'", 1)[1].split("</code></pre>", 1)[0]
    header_html = header_html.split("<code>", 1)[1]
    header_text = html.unescape(re.sub(r"<[^>]+>", "", header_html))

    for token in ("Metrics", "Module Header", "Interface", "Schedule", "_kernel"):
        assert token in report
    for token in ("// CONTROL PORTS", "// INPUT PORTS", "// OUTPUT PORTS", "// DIAGNOSTIC PORTS"):
        assert token in result.verilog
        assert token in header_text
    for token in (
        "module _kernel #(",
        "parameter WEXP =  8,  // Float exponent bits",
        "parameter WMAN = 24   // Float mantissa bits",
        "input  wire [WEXP+WMAN-1:0] in_a",
        "output wire [WEXP+WMAN-1:0] out_0",
        "output reg  [4:0] err_cyc",
    ):
        assert token in result.verilog
        assert token in header_text
    assert report.index("<h2>Interface</h2>") < report.index("<h2>Module Header</h2>")
    assert "--modhdr-width:" not in report
    assert "Runtime diagnostics available while the module is running." in header_text
    assert "vh-keyword" in report and "vh-comment" in report and "vh-number" in report


def test_report_schedule_displays_exact_ii_cycle_rows() -> None:
    result = holoso.synthesize(_kernel, float_format=FloatFormat(8, 24))
    grid = result.report_html.split("<table class='grid'>", 1)[1].split("</table>", 1)[0]
    cycle_labels = re.findall(r"<td class='clk'>([^<]+)</td>", grid)

    assert len(cycle_labels) == result.metrics.ii_cycles
    assert cycle_labels[0] == "in"
    assert cycle_labels[-1] == str(result.metrics.makespan)
    assert "out" not in cycle_labels


def test_synthesize_ekf1() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    import ekf1

    result = holoso.synthesize(ekf1.update_x_P, float_format=FloatFormat(6, 18))
    assert result.module_name == "update_x_P"
    assert len(result.interface.output_ports) == 9
    assert result.metrics.operator_instances.get("fdiv") == 1
    compile(result.testbench, "<generated-testbench>", "exec")


def test_class_target_is_unsupported() -> None:
    class Stateful:
        def __call__(self, x):  # type: ignore[no-untyped-def]
            return x

    with pytest.raises(holoso.UnsupportedConstruct):
        holoso.synthesize(Stateful, float_format=FloatFormat(6, 18))
