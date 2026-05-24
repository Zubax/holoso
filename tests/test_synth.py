"""End-to-end tests of the public synthesize() API, the report, artifact writing, and the generated testbench."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

import holoso
from holoso import FloatFormat


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
    assert "in_a" in names and "out_0" in names and "diag_error" in names


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
    report = holoso.synthesize(_kernel, float_format=FloatFormat(8, 24)).report_html
    for token in ("Metrics", "Interface", "Schedule", "_kernel"):
        assert token in report


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
