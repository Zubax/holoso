"""Elaboration tests for the generated Verilog backend (structural correctness under Icarus)."""

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from holoso import (
    FAddOperator,
    FDivOperator,
    FloatFormat,
    FMulILog2OperatorFamily,
    FMulOperator,
    OpConfig,
    UnsupportedConstruct,
)
from holoso._backend.verilog import generate
from holoso._frontend import lower
from holoso._hir import optimize
from holoso._lir import build
from holoso._mir import lower as lower_to_mir

from .hdl.hdl_float_oracle import HDL_DIR, sources

requires_iverilog = pytest.mark.skipif(shutil.which("iverilog") is None, reason="iverilog not installed")


def _ops(fmt: FloatFormat) -> OpConfig:
    return OpConfig(FAddOperator(fmt), FMulOperator(fmt), FDivOperator(fmt), FMulILog2OperatorFamily(fmt))


def _run(target, ops: OpConfig):  # type: ignore[no-untyped-def]
    return lower_to_mir(optimize(lower(target)), ops)


def _compile(name: str, verilog: str, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    vpath = tmp_path / f"{name}.v"
    vpath.write_text(verilog)
    cmd = [
        "iverilog",
        "-g2012",
        "-I",
        str(HDL_DIR),
        "-s",
        name,
        "-o",
        str(tmp_path / f"{name}.out"),
        str(vpath),
        *(str(s) for s in sources()),
    ]
    return subprocess.run(cmd, capture_output=True, text=True)


def _elaborate(name: str, verilog: str, tmp_path: Path) -> None:
    result = _compile(name, verilog, tmp_path)
    assert result.returncode == 0, result.stderr


def test_operator_instance_names_include_hardware_identity() -> None:
    def scale(a, b):  # type: ignore[no-untyped-def]
        return a * 4.0 + b * 8.0

    fmt = FloatFormat(6, 18)
    lir = build(_run(scale, _ops(fmt)), "scale")
    names = re.findall(
        r"\bholoso_fmul_ilog2_const\s+#\([^;]+?\)\s+u_([A-Za-z_][A-Za-z0-9_]*)\s+\(", generate(lir).verilog
    )

    assert len(names) == len(set(names))
    assert all(re.fullmatch(r"fmul_ilog2_const_[0-9a-f]{8}_0", name) for name in names)
    assert all("stage_decode" not in name and "e6_m18" not in name and "_k_" not in name for name in names)
    assert all(name == name.lower() for name in names)


@requires_iverilog
def test_streaming_wrapper_requires_latency(tmp_path: Path) -> None:
    verilog = """
module missing_latency;
    wire clk = 1'b0;
    wire rst = 1'b0;
    wire in_valid = 1'b0;
    wire [31:0] a = 32'h0;
    wire [31:0] b = 32'h0;
    wire out_valid;
    wire a_gt_b;
    wire a_eq_b;
    wire a_lt_b;

    holoso_fcmp #(.WEXP(8), .WMAN(24)) u_cmp (
        .clk(clk), .rst(rst), .in_valid(in_valid),
        .a_sgnop(2'd0), .b_sgnop(2'd0), .a(a), .b(b),
        .out_valid(out_valid), .a_gt_b(a_gt_b), .a_eq_b(a_eq_b), .a_lt_b(a_lt_b)
    );
endmodule
"""
    result = _compile("missing_latency", verilog, tmp_path)
    assert result.returncode != 0
    assert "_zkf_invalid_latency_mismatch" in result.stderr


@requires_iverilog
def test_small_kernel_elaborates(tmp_path: Path) -> None:
    def kernel(a, b):  # type: ignore[no-untyped-def]
        return (a - b) * 0.25 + a * b

    fmt = FloatFormat(8, 24)
    lir = build(_run(kernel, _ops(fmt)), "kernel")
    _elaborate("kernel", generate(lir).verilog, tmp_path)


@requires_iverilog
def test_kernel_with_division_elaborates(tmp_path: Path) -> None:
    def blend(a, b, c):  # type: ignore[no-untyped-def]
        return a / b + c * 2.0

    fmt = FloatFormat(6, 18)
    lir = build(_run(blend, _ops(fmt)), "blend")
    _elaborate("blend", generate(lir).verilog, tmp_path)


@requires_iverilog
def test_constant_only_module_elaborates(tmp_path: Path) -> None:
    # No inputs and an all-constant output => zero registers; NREG must floor to >=1 so the regfile parameter
    # guard does not instantiate its error stub (BUG1 regression).
    def const_only():  # type: ignore[no-untyped-def]
        return 3.5

    fmt = FloatFormat(8, 24)
    lir = build(_run(const_only, _ops(fmt)), "const_only")
    _elaborate("const_only", generate(lir).verilog, tmp_path)


def test_parameter_name_colliding_with_control_port_is_rejected() -> None:
    # A parameter named 'valid'/'ready' becomes data port in_valid/in_ready, colliding with the control ports and
    # producing un-elaboratable Verilog; LIR construction must reject it instead of emitting duplicate ports.
    def collide(valid, ready):  # type: ignore[no-untyped-def]
        return valid + ready

    fmt = FloatFormat(6, 18)
    with pytest.raises(UnsupportedConstruct, match="duplicate port"):
        build(_run(collide, _ops(fmt)), "collide")


def test_kernel_without_outputs_is_rejected() -> None:
    def empty(x):  # type: ignore[no-untyped-def]
        return ()

    fmt = FloatFormat(6, 18)
    with pytest.raises(UnsupportedConstruct, match="at least one output"):
        build(_run(empty, _ops(fmt)), "empty")


@requires_iverilog
def test_state_port_name_does_not_collide_with_internal_sign_wire(tmp_path: Path) -> None:
    # A public attribute `y_d` becomes the port state_y_d; a sibling slot `y` whose boundary copy carries a folded sign
    # used to emit an internal wire also named state_y_d, producing a duplicate (multiply-driven) identifier. The
    # internal sign wire must live outside the state_<attr> port namespace, so the module elaborates cleanly.
    class Collide:
        def __init__(self) -> None:
            self.y = 0.0
            self.y_d = 0.0
            self._p = 0.0

        def __call__(self, x):  # type: ignore[no-untyped-def]
            self.y_d = self._p
            self.y = -self._p  # sign-flipped boundary copy -> internal sign-conditioning wire for slot y
            self._p = x
            return self.y

    fmt = FloatFormat(8, 24)
    lir = build(_run(Collide().__call__, _ops(fmt)), "collide_state")
    _elaborate("collide_state", generate(lir).verilog, tmp_path)


@requires_iverilog
def test_ekf1_elaborates(tmp_path: Path) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    import ekf1

    fmt = FloatFormat(6, 18)
    lir = build(_run(ekf1.update_x_P, _ops(fmt)), "update_x_P")
    _elaborate("update_x_P", generate(lir).verilog, tmp_path)
