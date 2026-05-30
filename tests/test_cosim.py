"""Functional cosimulation: drive generated modules and check their outputs bit-for-bit against the model backend."""

import sys
from collections.abc import Callable
from pathlib import Path

import pytest
from cocotb_tools.runner import get_runner

from holoso import FAddOperator, FDivOperator, FloatFormat, FMulILog2OperatorFamily, FMulOperator, OpConfig
from holoso._backend.cocotb import generate as generate_testbench
from holoso._backend.numerical import generate as build_model
from holoso._backend.verilog import generate as generate_verilog
from holoso._frontend import lower
from holoso._hir import optimize
from holoso._lir import build
from holoso._mir import lower as lower_to_mir

from .hdl.hdl_float_oracle import HDL_DIR, REPO_ROOT, SIMULATORS, build_args, sources


def _ops(fmt: FloatFormat) -> OpConfig:
    return OpConfig(FAddOperator(fmt), FMulOperator(fmt), FDivOperator(fmt), FMulILog2OperatorFamily(fmt))


def _run_cosim(sim: str, fn: Callable[..., object], fmt: FloatFormat, name: str, ops: OpConfig | None = None) -> None:
    ops = _ops(fmt) if ops is None else ops
    lir = build(lower_to_mir(optimize(lower(fn)), ops), name)
    model = build_model(lir)
    # Generated sources live outside the cocotb build dir, which the runner wipes on clean=True.
    gen_dir = REPO_ROOT / "build" / "holoso_gen" / f"{name}_w{fmt.wexp}_{fmt.wman}"
    gen_dir.mkdir(parents=True, exist_ok=True)
    build_dir = REPO_ROOT / "build" / "cocotb" / sim / f"synth_{name}_w{fmt.wexp}_{fmt.wman}"
    verilog_path = gen_dir / f"{name}.v"
    verilog_path.write_text(generate_verilog(lir).verilog)
    # The generated bench embeds the bit-exact model and checks the DUT's output bits exactly.
    test_module = f"test_{name}"
    (gen_dir / f"{test_module}.py").write_text(generate_testbench(model).testbench)

    runner = get_runner(sim)
    runner.build(
        sources=[verilog_path, *sources()],
        includes=[HDL_DIR],
        hdl_toplevel=name,
        build_args=build_args(sim),
        build_dir=str(build_dir),
        clean=True,
        timescale=("1ns", "1ps"),
    )
    runner.test(
        hdl_toplevel=name,
        test_module=test_module,
        test_dir=str(gen_dir),
        build_dir=str(build_dir),
        results_xml=str(build_dir / "results.xml"),
    )


@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_small_kernel(sim: str) -> None:
    def kernel(a, b):  # type: ignore[no-untyped-def]
        return (a - b) * 0.25 + a * b

    _run_cosim(sim, kernel, FloatFormat(8, 24), "kernel")


@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_division(sim: str) -> None:
    def blend(a, b, c):  # type: ignore[no-untyped-def]
        return a / b + c * 2.0

    _run_cosim(sim, blend, FloatFormat(6, 18), "blend")


@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_ekf1(sim: str) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    import ekf1

    _run_cosim(sim, ekf1.update_x_P, FloatFormat(6, 18), "update_x_P")


@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_ekf1_pow2_width(sim: str) -> None:
    # W = 8 + 24 = 32 is a power of two, so the read-mux gather stride equals W and the elements are not zero-extended
    # (pad = 0). This exercises that branch of the gather emission, which the W=24 kernels do not.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    import ekf1

    _run_cosim(sim, ekf1.update_x_P, FloatFormat(8, 24), "update_x_P_w32")


@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_staged_kernel(sim: str) -> None:
    def kernel(a, b):  # type: ignore[no-untyped-def]
        return (a - b) * 0.25 + a * b

    # Exercise staged operator parameters end-to-end through synthesis and cosim.
    fmt = FloatFormat(8, 24)
    ops = OpConfig(
        FAddOperator(fmt, stage_decode=1),
        FMulOperator(fmt, stage_product=1),
        FDivOperator(fmt),
        FMulILog2OperatorFamily(fmt, stage_decode=1),
    )
    _run_cosim(sim, kernel, fmt, "kernel_staged", ops=ops)


@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_staged_division(sim: str) -> None:
    def blend(a, b, c):  # type: ignore[no-untyped-def]
        return a / b + (a - c)

    # Exercise the STAGE_ALIGN (fadd) and STAGE_INPUT (fdiv) knobs end-to-end -- the combos the staged-kernel misses.
    fmt = FloatFormat(6, 18)
    ops = OpConfig(
        FAddOperator(fmt, stage_decode=1, stage_align=1),
        FMulOperator(fmt),
        FDivOperator(fmt, stage_input=1),
        FMulILog2OperatorFamily(fmt),
    )
    _run_cosim(sim, blend, fmt, "blend_staged", ops=ops)


@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_poly3(sim: str) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    import poly3

    _run_cosim(sim, poly3.poly3, FloatFormat(6, 18), "poly3")


@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_madd(sim: str) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    import madd

    _run_cosim(sim, madd.madd, FloatFormat(6, 18), "madd")


@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_new_operator_stages(sim: str) -> None:
    def kernel(a, b, c):  # type: ignore[no-untyped-def]
        return (a - b) / c + a * b * 0.25  # fadd, fdiv, fmul, and fmul_ilog2 (the 2^-2 scale) all in one kernel

    # Exercise the newly-shipped ZKF knobs end-to-end: fadd STAGE_INPUT/STAGE_NORMALIZE/STAGE_PACK, fmul STAGE_PACK,
    # fdiv STAGE_PACK, and fmul_ilog2 STAGE_INPUT -- all folded into the latency model and the latched datapath.
    fmt = FloatFormat(8, 24)
    ops = OpConfig(
        FAddOperator(fmt, stage_input=1, stage_normalize=2, stage_pack=1),
        FMulOperator(fmt, stage_input=1, stage_pack=1),
        FDivOperator(fmt, stage_pack=1),
        FMulILog2OperatorFamily(fmt, stage_input=1),
    )
    _run_cosim(sim, kernel, fmt, "new_stages", ops=ops)
