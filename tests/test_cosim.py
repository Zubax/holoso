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
def test_cosim_ekf1_stateless(sim: str) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    import ekf1_stateless

    _run_cosim(sim, ekf1_stateless.update_x_P, FloatFormat(6, 18), "update_x_P")


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
def test_cosim_trapezoidal_integrator(sim: str) -> None:
    # A stateful class: the bound method becomes a streaming module whose persistent state (the leaky accumulator y and
    # the one-sample delay _x_prev) is exercised across the whole random input sequence, bit-for-bit against the model.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    from trapezoidal_leaky_streaming_integrator import TrapezoidalLeakyStreamingIntegrator

    _run_cosim(sim, TrapezoidalLeakyStreamingIntegrator(k=2**-22).__call__, FloatFormat(6, 18), "trapz_integrator")


class _ShiftRegister2:
    """A two-deep delay line returning the input from two steps ago; both state slots are non-coalesced copy slots."""

    def __init__(self) -> None:
        self._a = 0.0
        self._b = 0.0

    def __call__(self, x: float) -> float:
        out = self._b
        self._b = self._a
        self._a = x
        return out


@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_shift_register_backpressure(sim: str) -> None:
    # The returned value taps a copy-slot register and the chain advances every accept, so together with the testbench's
    # random back-pressure this pins down that the boundary copy fires exactly once per accepted transaction -- no
    # mid-handshake output mutation and no state over-advance while out_ready is held low.
    _run_cosim(sim, _ShiftRegister2().__call__, FloatFormat(6, 18), "shift2")


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


# The generated bench only checks err_pc == 0 over a bounded input range, so it never exercises the div0 -> err_pc
# path. This custom bench drives an exact zero divisor and asserts the diagnostic is set, then cleared on the next
# accepted transaction. It cannot reuse the generated bench because the numerical model does not predict errors.
_ERR_BENCH = """
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge, Timer
import holoso

_FMT = holoso.FloatFormat(@@WEXP@@, @@WMAN@@)


async def _transact(dut, a, b):
    while int(dut.in_ready.value) != 1:
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
    dut.in_a.value = int(_FMT.encode(a))
    dut.in_b.value = int(_FMT.encode(b))
    dut.in_valid.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")
    dut.in_valid.value = 0
    while int(dut.out_valid.value) != 1:
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
    err = int(dut.err_pc.value)
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")
    return err


@cocotb.test()
async def div0_errpc(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await FallingEdge(dut.clk)
    dut.rst.value = 1
    dut.in_valid.value = 0
    for _ in range(4):
        await RisingEdge(dut.clk)
    dut.rst.value = 0
    await FallingEdge(dut.clk)
    dut.out_ready.value = 1

    assert await _transact(dut, 6.0, 2.0) == 0, "clean divide spuriously flagged err_pc"
    assert await _transact(dut, 1.0, 0.0) != 0, "divide-by-zero did not set err_pc"
    assert await _transact(dut, 6.0, 2.0) == 0, "err_pc was not cleared on the next transaction"
"""


@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_div0_error(sim: str) -> None:
    def kdiv(a, b):  # type: ignore[no-untyped-def]
        return a / b

    fmt = FloatFormat(6, 18)
    name = "kdiv"
    lir = build(lower_to_mir(optimize(lower(kdiv)), _ops(fmt)), name)
    gen_dir = REPO_ROOT / "build" / "holoso_gen" / f"{name}_err_w{fmt.wexp}_{fmt.wman}"
    gen_dir.mkdir(parents=True, exist_ok=True)
    build_dir = REPO_ROOT / "build" / "cocotb" / sim / f"err_{name}_w{fmt.wexp}_{fmt.wman}"
    (gen_dir / f"{name}.v").write_text(generate_verilog(lir).verilog)
    test_module = f"test_{name}_err"
    (gen_dir / f"{test_module}.py").write_text(
        _ERR_BENCH.replace("@@WEXP@@", str(fmt.wexp)).replace("@@WMAN@@", str(fmt.wman))
    )

    runner = get_runner(sim)
    runner.build(
        sources=[gen_dir / f"{name}.v", *sources()],
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
