"""End-to-end tests of the public synthesize() API, the report, artifact writing, and the generated testbench."""

import html
import re
import sys
from pathlib import Path

import pytest

import holoso
from holoso import FAddOp, FDivOp, FloatFormat, FMulILog2GenericOp, FMulOp, OpConfig


def _kernel(a, b):  # type: ignore[no-untyped-def]  # module-level so inspect.getsource works
    return (a - b) * 0.25 + a * b


OPS = OpConfig(FAddOp(), FMulOp(), FDivOp(), FMulILog2GenericOp())


def test_synthesize_small_kernel_result() -> None:
    result = holoso.synthesize(_kernel, float_format=FloatFormat(8, 24), ops=OPS)
    assert result.module_name == "_kernel"
    assert "module _kernel" in result.verilog_output.verilog
    assert "holoso_regfile" in result.verilog_output.support_files["holoso_support.v"]
    assert '`include "holoso_support.vh"' in result.verilog_output.support_files["holoso_support.v"]
    assert "`ifndef HOLOSO_REGFILE_VH" not in result.verilog_output.support_files["holoso_support.v"]
    assert "`HOLOSO_REGFILE_LANE" in result.verilog_output.support_files["holoso_support.vh"]
    assert "@cocotb.test()" in result.cocotb_output.testbench
    assert "<html" in result.html_output.html.lower()
    names = [p.name for p in result.interface.ports]
    assert "in_a" in names and "out_0" in names and "err_cyc" in names


def test_synthesize_threads_pipeline_stages() -> None:
    base = holoso.synthesize(_kernel, float_format=FloatFormat(8, 24), ops=OPS)
    staged = holoso.synthesize(
        _kernel,
        float_format=FloatFormat(8, 24),
        ops=OpConfig(FAddOp(decode=1), FMulOp(product=1), FDivOp(), FMulILog2GenericOp()),
    )
    assert "STAGE_" not in base.verilog_output.verilog  # default stages emit no STAGE_* instance params
    assert ".STAGE_DECODE(1)" in staged.verilog_output.verilog and ".STAGE_PRODUCT(1)" in staged.verilog_output.verilog


def test_rejects_non_finite_constants() -> None:
    def overflow(a):  # type: ignore[no-untyped-def]
        return a + 1e400  # overflows to +inf, not representable in the ZKF format

    def folds_to_nan(a):  # type: ignore[no-untyped-def]
        return a + (1e400 - 1e400)  # inf - inf const-folds to NaN

    for fn in (overflow, folds_to_nan):
        with pytest.raises(holoso.UnsupportedConstruct):
            holoso.synthesize(fn, float_format=FloatFormat(8, 24), ops=OPS)


def test_generated_testbench_is_valid_python() -> None:
    result = holoso.synthesize(_kernel, float_format=FloatFormat(8, 24), ops=OPS)
    compile(result.cocotb_output.testbench, "<generated-testbench>", "exec")


def test_write_artifacts(tmp_path: Path) -> None:
    result = holoso.synthesize(_kernel, float_format=FloatFormat(8, 24), ops=OPS)
    paths = result.write(tmp_path)
    assert set(paths) == {"_kernel.v", "holoso_support.v", "holoso_support.vh", "test__kernel.py", "_kernel.html"}
    assert (tmp_path / "_kernel.v").exists()
    assert (tmp_path / "test__kernel.py").exists()
    assert (tmp_path / "_kernel.html").exists()
    assert (tmp_path / "holoso_support.v").exists()
    assert (tmp_path / "holoso_support.vh").exists()


def test_report_has_expected_sections() -> None:
    result = holoso.synthesize(_kernel, float_format=FloatFormat(8, 24), ops=OPS)
    report = result.html_output.html
    header_html = report.split("<pre class='modhdr'", 1)[1].split("</code></pre>", 1)[0]
    header_html = header_html.split("<code>", 1)[1]
    header_text = html.unescape(re.sub(r"<[^>]+>", "", header_html))

    for token in ("Metrics", "Module Header", "Interface", "Schedule", "_kernel"):
        assert token in report
    for token in ("// CONTROL PORTS", "// INPUT PORTS", "// OUTPUT PORTS", "// DIAGNOSTIC PORTS"):
        assert token in result.verilog_output.verilog
        assert token in header_text
    for token in (
        "module _kernel (",
        "input  wire [31:0] in_a",
        "output wire [31:0] out_0",
        "output reg  [4:0] err_cyc",
    ):
        assert token in result.verilog_output.verilog
        assert token in header_text
    assert "module _kernel #(" not in result.verilog_output.verilog
    assert "parameter WEXP" not in result.verilog_output.verilog
    assert "parameter WMAN" not in result.verilog_output.verilog
    assert "localparam WEXP  = 8;" in result.verilog_output.verilog
    assert "localparam WMAN  = 24;" in result.verilog_output.verilog
    assert report.index("<h2>Interface</h2>") < report.index("<h2>Module Header</h2>")
    assert "--modhdr-width:" not in report
    assert "Runtime diagnostics available while the module is running." in header_text
    assert "vh-keyword" in report and "vh-comment" in report and "vh-number" in report


def test_report_schedule_displays_exact_ii_cycle_rows() -> None:
    result = holoso.synthesize(_kernel, float_format=FloatFormat(8, 24), ops=OPS)
    grid = result.html_output.html.split("<table class='grid'>", 1)[1].split("</table>", 1)[0]
    cycle_labels = re.findall(r"<td class='clk'>([^<]+)</td>", grid)

    assert cycle_labels[0] == "in"
    assert cycle_labels[1:] == [str(i) for i in range(1, len(cycle_labels))]
    assert "out" not in cycle_labels


def test_synthesize_ekf1() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    import ekf1

    result = holoso.synthesize(ekf1.update_x_P, float_format=FloatFormat(6, 18), ops=OPS)
    assert result.module_name == "update_x_P"
    assert len(result.interface.output_ports) == 9
    compile(result.cocotb_output.testbench, "<generated-testbench>", "exec")


def test_class_target_is_unsupported() -> None:
    class Stateful:
        def __call__(self, x):  # type: ignore[no-untyped-def]
            return x

    with pytest.raises(holoso.UnsupportedConstruct):
        holoso.synthesize(Stateful, float_format=FloatFormat(6, 18), ops=OPS)
