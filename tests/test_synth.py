"""End-to-end tests of the public synthesize() API, the report, artifact writing, and the generated testbench."""

import html
import math
import re
import sys
from pathlib import Path

import pytest

import holoso
from holoso import (
    FAddOperator,
    FCmpOperator,
    FDivOperator,
    FloatFormat,
    FMulILog2OperatorFamily,
    FMulOperator,
    OpConfig,
)


def _kernel(a: float, b: float) -> float:  # module-level so inspect.getsource works
    return (a - b) * 0.25 + a * b


FMT32 = FloatFormat(8, 24)
_NAN = float("nan")


def _ops(fmt: FloatFormat = FMT32) -> OpConfig:
    return OpConfig(
        FAddOperator(fmt), FMulOperator(fmt), FDivOperator(fmt), FMulILog2OperatorFamily(fmt), FCmpOperator(fmt)
    )


def _has_localparam(verilog: str, name: str, value: int) -> bool:
    return (
        re.search(rf"^localparam\s+(?:\[[^\]]+\]\s+)?{re.escape(name)}\s*=\s*{value};", verilog, re.MULTILINE)
        is not None
    )


def test_op_config_rejects_mixed_float_formats() -> None:
    fmt24 = FloatFormat(6, 18)
    ops = OpConfig(
        FAddOperator(FMT32),
        FMulOperator(fmt24),
        FDivOperator(FMT32),
        FMulILog2OperatorFamily(FMT32),
        FCmpOperator(FMT32),
    )
    with pytest.raises(ValueError, match="same format"):
        _ = ops.float_format


def test_constant_only_module_keeps_operator_configured_format() -> None:
    def const_only() -> float:
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
            FMulOperator(FMT32, stage_product=2),
            FDivOperator(FMT32),
            FMulILog2OperatorFamily(FMT32),
            FCmpOperator(FMT32),
        ),
    )
    # Every STAGE_* is emitted explicitly (defaults as 0), so the instantiation is self-describing and configured
    # values are visible.
    assert ".STAGE_DECODE(0)" in base.verilog_output.verilog
    assert ".STAGE_DECODE(1)" in staged.verilog_output.verilog and ".STAGE_PRODUCT(2)" in staged.verilog_output.verilog
    assert ".LATENCY(4)" in base.verilog_output.verilog and ".LATENCY(1)" in base.verilog_output.verilog
    assert ".LATENCY(5)" in staged.verilog_output.verilog and ".LATENCY(3)" in staged.verilog_output.verilog


def test_rejects_nan_constant_data_but_defers_nan_producing_folds() -> None:
    # Literal NaN DATA cannot exist in the (NaN-free) target format and refuses up front; an expression whose
    # fold WOULD be NaN instead stays a runtime operation with the hardware's own NaN-free answer.
    def nan_global(a: float) -> float:
        return a + _NAN

    with pytest.raises(holoso.UnsupportedConstruct):
        holoso.synthesize(nan_global, ops=_ops())

    def folds_to_nan(a: float) -> float:
        return a + (1e400 - 1e400)

    holoso.synthesize(folds_to_nan, ops=_ops())


def test_infinity_constants_are_allowed() -> None:
    def overflow(a: float) -> float:
        return a + 1e400

    def hidden_by_fast_math(a: float) -> tuple[float, float, float]:
        t = a + 1e400
        return 0.0 * t, 0.0 / t, t / t

    out = holoso.synthesize(overflow, ops=_ops()).numerical_model.elaborate().run(1.0)[0]
    assert math.isinf(float(out)) and float(out) > 0.0

    folded = holoso.synthesize(hidden_by_fast_math, ops=_ops()).numerical_model.elaborate().run(1.0)
    assert [float(value) for value in folded] == [0.0, 0.0, 1.0]


def test_write_artifacts(tmp_path: Path) -> None:
    result = holoso.synthesize(_kernel, ops=_ops())
    paths = result.write(tmp_path)
    assert set(paths) == {"_kernel.v", "holoso_support.v", "test__kernel.py", "_kernel.html"}
    assert (tmp_path / "_kernel.v").exists()
    assert (tmp_path / "test__kernel.py").exists()
    assert (tmp_path / "_kernel.html").exists()
    assert (tmp_path / "holoso_support.v").exists()


def test_rejects_invalid_and_reserved_module_names() -> None:
    # An empty name is falsy, so it is not "invalid" -- it just falls back to the target-derived default.
    for bad in ("1bad", "bad-name", "a/b", "has space", "../escape"):
        with pytest.raises(ValueError, match="valid identifier"):
            holoso.synthesize(_kernel, ops=_ops(), name=bad)
    for reserved in ("holoso", "Holoso_x", "holoso_support", "HOLOSOmod"):
        with pytest.raises(ValueError, match="reserved"):
            holoso.synthesize(_kernel, ops=_ops(), name=reserved)
    # Reserved words would emit unparsable RTL (`module module (`); a same-spelled non-keyword is still fine.
    for keyword in ("module", "reg", "wire", "assign", "always"):
        with pytest.raises(ValueError, match="reserved keyword"):
            holoso.synthesize(_kernel, ops=_ops(), name=keyword)
    assert (
        holoso.synthesize(_kernel, ops=_ops(), name="Module").module_name == "Module"
    )  # case-sensitive: not a keyword


def test_accepts_valid_module_name(tmp_path: Path) -> None:
    result = holoso.synthesize(_kernel, ops=_ops(), name="good_name")
    assert result.module_name == "good_name"
    paths = result.write(tmp_path)
    assert set(paths) == {
        "good_name.v",
        "holoso_support.v",
        "test_good_name.py",
        "good_name.html",
    }


def test_synthesize_ekf1_stateless() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    import ekf1_stateless

    fmt = FloatFormat(6, 18)
    result = holoso.synthesize(ekf1_stateless.update_x_P, ops=_ops(fmt))
    assert result.module_name == "update_x_P"
    assert len(result.output_ports) == 9
    compile(result.cocotb_output.testbench, "<generated-testbench>", "exec")


def test_class_target_is_unsupported() -> None:
    class Stateful:
        def __call__(self, x: float) -> float:
            return x

    # The front-end has no single-function source for a class target, so it rejects it as SourceUnavailable.
    with pytest.raises(holoso.SourceUnavailable):
        holoso.synthesize(Stateful, ops=_ops(FloatFormat(6, 18)))
