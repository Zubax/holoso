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
    assert ".LATENCY(4)" in base.verilog_output.verilog and ".LATENCY(1)" in base.verilog_output.verilog
    assert ".LATENCY(5)" in staged.verilog_output.verilog and ".LATENCY(2)" in staged.verilog_output.verilog


def test_rejects_non_finite_constants() -> None:
    def overflow(a):  # type: ignore[no-untyped-def]
        return a + 1e400  # overflows to +inf, not representable in the ZKF format

    def folds_to_nan(a):  # type: ignore[no-untyped-def]
        return a + (1e400 - 1e400)  # inf - inf const-folds to NaN

    for fn in (overflow, folds_to_nan):
        with pytest.raises(holoso.UnsupportedConstruct):
            holoso.synthesize(fn, ops=_ops())


def test_write_artifacts(tmp_path: Path) -> None:
    result = holoso.synthesize(_kernel, ops=_ops())
    paths = result.write(tmp_path)
    assert set(paths) == {"_kernel.v", "holoso_support.v", "test__kernel.py", "_kernel.html"}
    assert (tmp_path / "_kernel.v").exists()
    assert (tmp_path / "test__kernel.py").exists()
    assert (tmp_path / "_kernel.html").exists()
    assert (tmp_path / "holoso_support.v").exists()


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
