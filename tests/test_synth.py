"""End-to-end tests of the public synthesize() API, the report, artifact writing, and the generated testbench."""

import math
import re
from pathlib import Path

import pytest

import holoso
from holoso import (
    FAddOptions,
    FCmpOptions,
    FDivOptions,
    FMulILog2Options,
    FMulOptions,
    FloatFormat,
    OperatorOptions,
    Options,
)


def _kernel(a: float, b: float) -> float:  # module-level so inspect.getsource works
    return (a - b) * 0.25 + a * b


FMT32 = FloatFormat(8, 24)
_NAN = float("nan")


def _ops(fmt: FloatFormat = FMT32) -> Options:
    return Options(
        OperatorOptions(
            fadd=FAddOptions(),
            fmul=FMulOptions(),
            fdiv=FDivOptions(),
            fmul_ilog2=FMulILog2Options(),
            fcmp=FCmpOptions(),
        ),
        ffmt=fmt,
    )


def _has_localparam(verilog: str, name: str, value: int) -> bool:
    return (
        re.search(rf"^localparam\s+(?:\[[^\]]+\]\s+)?{re.escape(name)}\s*=\s*{value};", verilog, re.MULTILINE)
        is not None
    )


def test_constant_only_module_keeps_operator_configured_format() -> None:
    def const_only() -> float:
        return 3.5

    fmt = FloatFormat(6, 18)
    result = holoso.synthesize(const_only, _ops(fmt))
    assert result.numerical_model.float_format == fmt
    assert _has_localparam(result.verilog_output.verilog, "WEXP", 6)
    assert _has_localparam(result.verilog_output.verilog, "WMAN", 18)
    assert all(p.width == fmt.width for p in result.output_ports)


def test_synthesize_threads_pipeline_stages() -> None:
    base = holoso.synthesize(_kernel, _ops())
    staged = holoso.synthesize(
        _kernel,
        Options(
            OperatorOptions(
                fadd=FAddOptions(stage_decode=1),
                fmul=FMulOptions(stage_product=2),
                fdiv=FDivOptions(),
                fmul_ilog2=FMulILog2Options(),
                fcmp=FCmpOptions(),
            ),
            ffmt=FMT32,
        ),
    )
    # Every STAGE_* is emitted explicitly (defaults as 0), so the instantiation is self-describing and configured
    # values are visible.
    assert ".STAGE_DECODE(0)" in base.verilog_output.verilog
    assert ".STAGE_DECODE(1)" in staged.verilog_output.verilog and ".STAGE_PRODUCT(2)" in staged.verilog_output.verilog
    assert ".LATENCY(4)" in base.verilog_output.verilog and ".LATENCY(1)" in base.verilog_output.verilog
    assert ".LATENCY(5)" in staged.verilog_output.verilog and ".LATENCY(3)" in staged.verilog_output.verilog


def test_rejects_nan_constant_data_and_nan_producing_folds_alike() -> None:
    # A literal NaN is not a number, so it cannot be written as data; an expression of constants that denotes no
    # number is the same defect reached by arithmetic, and is refused rather than handed to the datapath.
    def nan_global(a: float) -> float:
        return a + _NAN

    with pytest.raises(holoso.UnsupportedConstruct):
        holoso.synthesize(nan_global, _ops())

    def folds_to_nan(a: float) -> float:
        return a + (1e400 - 1e400)  # inf + -inf: an indeterminate form, so it names nothing

    with pytest.raises(holoso.SynthesisError, match="names no number"):
        holoso.synthesize(folds_to_nan, _ops())


def test_infinity_constants_are_allowed() -> None:
    def overflow(a: float) -> float:
        return a + 1e400

    def hidden_by_fast_math(a: float) -> tuple[float, float, float]:
        t = a + 1e400
        return 0.0 * t, 0.0 / t, t / t

    out = holoso.synthesize(overflow, _ops()).numerical_model.elaborate().run(1.0)[0]
    assert math.isinf(float(out)) and float(out) > 0.0

    folded = holoso.synthesize(hidden_by_fast_math, _ops()).numerical_model.elaborate().run(1.0)
    assert [float(value) for value in folded] == [0.0, 0.0, 1.0]


def test_write_artifacts(tmp_path: Path) -> None:
    result = holoso.synthesize(_kernel, _ops())
    paths = result.write(tmp_path)
    assert set(paths) == {
        "_kernel.v",
        "holoso_support.v",
        "test__kernel.py",
        "_kernel.html",
        "_kernel.pass0.fir",
        "_kernel.pass1.fir",
    }
    for name in paths:
        assert (tmp_path / name).read_text(encoding="utf-8")


def test_frontend_ir_records_the_passes(tmp_path: Path) -> None:
    """The written documents are the front end's own passes in order, not two prints of one program."""
    result = holoso.synthesize(_kernel, _ops())
    desugared, refined = result.frontend_ir
    # The desugarer knows no types and keeps the source spelling; partial evaluation types the boundary and
    # resolves the operators, so the subtraction has become a negate-and-add pair by the second document.
    assert "fn _kernel(a, b):" in desugared and "a - b" in desugared
    assert "fn _kernel(a: float, b: float):" in refined and "intrinsic fneg(b)" in refined
    # Locations are on, which is the whole point of shipping these next to the RTL.
    assert all(f"# {Path(__file__).name}:" in doc for doc in result.frontend_ir)
    result.write(tmp_path)
    assert (tmp_path / "_kernel.pass0.fir").read_text(encoding="utf-8") == desugared
    assert (tmp_path / "_kernel.pass1.fir").read_text(encoding="utf-8") == refined


def test_rejects_invalid_and_reserved_module_names() -> None:
    # One representative per validation class. An empty name is falsy, so it is not "invalid" -- it just falls back
    # to the target-derived default.
    with pytest.raises(ValueError, match="valid identifier"):
        holoso.synthesize(_kernel, _ops(), name="1bad")
    with pytest.raises(ValueError, match="reserved"):
        holoso.synthesize(_kernel, _ops(), name="Holoso_x")
    # A reserved word would emit unparsable RTL (`module module (`); a same-spelled non-keyword is still fine.
    with pytest.raises(ValueError, match="reserved keyword"):
        holoso.synthesize(_kernel, _ops(), name="module")
    assert holoso.synthesize(_kernel, _ops(), name="Module").module_name == "Module"  # case-sensitive: not a keyword


def test_accepts_valid_module_name(tmp_path: Path) -> None:
    # The full default-manifest sweep is owned by test_write_artifacts; here only that the explicit name threads in.
    result = holoso.synthesize(_kernel, _ops(), name="good_name")
    assert result.module_name == "good_name"
    assert "good_name.v" in result.write(tmp_path)


def test_generated_testbench_is_valid_python() -> None:
    compile(holoso.synthesize(_kernel, _ops()).cocotb_output.testbench, "<generated-testbench>", "exec")


def test_class_target_is_unsupported() -> None:
    class Stateful:
        def __call__(self, x: float) -> float:
            return x

    with pytest.raises(holoso.UnsupportedConstruct, match="is not a plain function"):
        holoso.synthesize(Stateful, _ops(FloatFormat(6, 18)))
