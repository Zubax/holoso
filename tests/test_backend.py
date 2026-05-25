"""Elaboration tests for the generated Verilog backend (structural correctness under Icarus)."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from holoso.backend_verilog import generate
from holoso.format import FloatFormat
from holoso.frontend import lower
from holoso.operators import FAddOp, FDivOp, FMulILog2GenericOp, FMulOp, OpConfig
from holoso.passes import run
from holoso.schedule import build

from hdl_float_oracle import HDL_DIR, sources

requires_iverilog = pytest.mark.skipif(shutil.which("iverilog") is None, reason="iverilog not installed")
OPS = OpConfig(FAddOp(), FMulOp(), FDivOp(), FMulILog2GenericOp())


def _elaborate(name: str, verilog: str, tmp_path: Path) -> None:
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
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@requires_iverilog
def test_small_kernel_elaborates(tmp_path: Path) -> None:
    def kernel(a, b):  # type: ignore[no-untyped-def]
        return (a - b) * 0.25 + a * b

    lir = build(run(lower(kernel, FloatFormat(8, 24)), OPS), "kernel")
    _elaborate("kernel", generate(lir), tmp_path)


@requires_iverilog
def test_kernel_with_division_elaborates(tmp_path: Path) -> None:
    def blend(a, b, c):  # type: ignore[no-untyped-def]
        return a / b + c * 2.0

    lir = build(run(lower(blend, FloatFormat(6, 18)), OPS), "blend")
    _elaborate("blend", generate(lir), tmp_path)


@requires_iverilog
def test_constant_only_module_elaborates(tmp_path: Path) -> None:
    # No inputs and an all-constant output => zero registers; NREG must floor to >=1 so the regfile parameter
    # guard does not instantiate its error stub (BUG1 regression).
    def const_only():  # type: ignore[no-untyped-def]
        return 3.5

    lir = build(run(lower(const_only, FloatFormat(8, 24)), OPS), "const_only")
    _elaborate("const_only", generate(lir), tmp_path)


@requires_iverilog
def test_ekf1_elaborates(tmp_path: Path) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    import ekf1

    lir = build(run(lower(ekf1.update_x_P, FloatFormat(6, 18)), OPS), "update_x_P")
    _elaborate("update_x_P", generate(lir), tmp_path)
