"""End-to-end tests of the public synthesize() API, the report, artifact writing, and the generated testbench."""

import html
import re
import sys
from pathlib import Path

import pytest

import holoso
from holoso import FAddOperator, FDivOperator, FloatFormat, FMulILog2OperatorFamily, FMulOperator, OpConfig


def _kernel(a, b):  # type: ignore[no-untyped-def]  # module-level so inspect.getsource works
    return (a - b) * 0.25 + a * b


FMT32 = FloatFormat(8, 24)


def _ops(fmt: FloatFormat = FMT32) -> OpConfig:
    return OpConfig(FAddOperator(fmt), FMulOperator(fmt), FDivOperator(fmt), FMulILog2OperatorFamily(fmt))


def _has_localparam(verilog: str, name: str, value: int) -> bool:
    return (
        re.search(rf"^localparam\s+(?:\[[^\]]+\]\s+)?{re.escape(name)}\s*=\s*{value};", verilog, re.MULTILINE)
        is not None
    )


def test_synthesize_small_kernel_result() -> None:
    result = holoso.synthesize(_kernel, ops=_ops())
    assert result.module_name == "_kernel"
    assert "module _kernel" in result.verilog_output.verilog
    # The full-crossbar register file is gone; storage is emitted inline as a sparse register array.
    assert "holoso_regfile" not in result.verilog_output.support_files["holoso_support.v"]
    assert "holoso_fadd" in result.verilog_output.support_files["holoso_support.v"]
    assert "reg  [W-1:0] regs [0:NREG-1];" in result.verilog_output.verilog
    assert '`include "holoso_support.vh"' in result.verilog_output.support_files["holoso_support.v"]
    assert "`define HOLOSO_FSGNOP_NEG" in result.verilog_output.support_files["holoso_support.vh"]
    assert "HOLOSO_REGFILE_LANE" not in result.verilog_output.support_files["holoso_support.vh"]
    assert "@cocotb.test()" in result.cocotb_output.testbench
    assert "<html" in result.html_output.html.lower()
    names = [p.name for p in result.ports]
    assert "in_a" in names and "out_0" in names and "err_pc" in names
    assert all(isinstance(p, holoso.DataInputPort) for p in result.input_ports)
    assert all(isinstance(p, holoso.DataOutputPort) for p in result.output_ports)
    assert any(isinstance(p, holoso.ControlOutputPort) and p.name == "err_pc" for p in result.control_ports)
    assert all(isinstance(p.scalar_type, holoso.FloatType) and p.scalar_type.fmt == FMT32 for p in result.input_ports)


def test_op_config_rejects_mixed_float_formats() -> None:
    fmt24 = FloatFormat(6, 18)
    ops = OpConfig(FAddOperator(FMT32), FMulOperator(fmt24), FDivOperator(FMT32), FMulILog2OperatorFamily(FMT32))
    with pytest.raises(ValueError, match="same format"):
        _ = ops.float_format


def test_constant_only_module_keeps_operator_configured_format() -> None:
    def const_only():  # type: ignore[no-untyped-def]
        return 3.5

    fmt = FloatFormat(6, 18)
    result = holoso.synthesize(const_only, ops=_ops(fmt))
    assert result.numerical_model.float_format == fmt
    assert _has_localparam(result.verilog_output.verilog, "WEXP", 6)
    assert _has_localparam(result.verilog_output.verilog, "WMAN", 18)
    assert all(p.width == fmt.width for p in result.output_ports)


def test_synthesize_threads_pipeline_stages() -> None:
    base = holoso.synthesize(_kernel, ops=_ops())
    staged = holoso.synthesize(
        _kernel,
        ops=OpConfig(
            FAddOperator(FMT32, stage_decode=1),
            FMulOperator(FMT32, stage_product=1),
            FDivOperator(FMT32),
            FMulILog2OperatorFamily(FMT32),
        ),
    )
    # Every STAGE_* is emitted explicitly (defaults as 0), so the instantiation is self-describing and threading is
    # visible in both directions: default operators show 0, configured ones show 1.
    assert ".STAGE_DECODE(0)" in base.verilog_output.verilog
    assert ".STAGE_DECODE(1)" in staged.verilog_output.verilog and ".STAGE_PRODUCT(1)" in staged.verilog_output.verilog


def test_rejects_non_finite_constants() -> None:
    def overflow(a):  # type: ignore[no-untyped-def]
        return a + 1e400  # overflows to +inf, not representable in the ZKF format

    def folds_to_nan(a):  # type: ignore[no-untyped-def]
        return a + (1e400 - 1e400)  # inf - inf const-folds to NaN

    for fn in (overflow, folds_to_nan):
        with pytest.raises(holoso.UnsupportedConstruct):
            holoso.synthesize(fn, ops=_ops())


def test_generated_testbench_is_valid_python() -> None:
    result = holoso.synthesize(_kernel, ops=_ops())
    assert "holoso.FloatValue.from_bits(_FMT, bits)" in result.cocotb_output.testbench
    assert "exp_bits = [value.bits for value in expected]" in result.cocotb_output.testbench
    assert "_FMT.decode(bits)" not in result.cocotb_output.testbench
    compile(result.cocotb_output.testbench, "<generated-testbench>", "exec")


def test_write_artifacts(tmp_path: Path) -> None:
    result = holoso.synthesize(_kernel, ops=_ops())
    paths = result.write(tmp_path)
    assert set(paths) == {"_kernel.v", "holoso_support.v", "holoso_support.vh", "test__kernel.py", "_kernel.html"}
    assert (tmp_path / "_kernel.v").exists()
    assert (tmp_path / "test__kernel.py").exists()
    assert (tmp_path / "_kernel.html").exists()
    assert (tmp_path / "holoso_support.v").exists()
    assert (tmp_path / "holoso_support.vh").exists()


def test_report_has_expected_sections() -> None:
    result = holoso.synthesize(_kernel, ops=_ops())
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
        "output wire [4:0] err_pc",
    ):
        assert token in result.verilog_output.verilog
        assert token in header_text
    assert "module _kernel #(" not in result.verilog_output.verilog
    assert "parameter WEXP" not in result.verilog_output.verilog
    assert "parameter WMAN" not in result.verilog_output.verilog
    assert "reg  [CYCW-1:0] err_pc_q;" in result.verilog_output.verilog
    assert "assign err_pc    = err_pc_q;" in result.verilog_output.verilog
    assert _has_localparam(result.verilog_output.verilog, "WEXP", 8)
    assert _has_localparam(result.verilog_output.verilog, "WMAN", 24)
    assert report.index("<h2>Interface</h2>") < report.index("<h2>Module Header</h2>")
    assert "--modhdr-width:" not in report
    assert "Runtime diagnostics available while the module is running." in header_text
    assert "vh-keyword" in report and "vh-comment" in report and "vh-number" in report


def test_report_schedule_displays_exact_ii_cycle_rows() -> None:
    result = holoso.synthesize(_kernel, ops=_ops())
    grid = result.html_output.html.split("<table class='grid'>", 1)[1].split("</table>", 1)[0]
    cycle_labels = re.findall(r"<td class='clk'>([^<]+)</td>", grid)

    assert cycle_labels[0] == "in"
    assert cycle_labels[1:] == [str(i) for i in range(1, len(cycle_labels))]
    assert "out" not in cycle_labels


def test_synthesize_ekf1() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    import ekf1

    fmt = FloatFormat(6, 18)
    result = holoso.synthesize(ekf1.update_x_P, ops=_ops(fmt))
    assert result.module_name == "update_x_P"
    assert len(result.output_ports) == 9
    compile(result.cocotb_output.testbench, "<generated-testbench>", "exec")


def test_class_target_is_unsupported() -> None:
    class Stateful:
        def __call__(self, x):  # type: ignore[no-untyped-def]
            return x

    with pytest.raises(holoso.UnsupportedConstruct):
        holoso.synthesize(Stateful, ops=_ops(FloatFormat(6, 18)))
