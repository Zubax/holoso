"""Functional cosimulation: drive generated modules and check their outputs bit-for-bit against the model backend."""

import sys
from pathlib import Path

import numpy as np
import pytest
from cocotb_tools.runner import get_runner

from holoso import (
    FAddOperator,
    FCmpOperator,
    FDivOperator,
    FloatFormat,
    FMulILog2OperatorFamily,
    FMulOperator,
    OpConfig,
)
from holoso._backend.verilog import generate as generate_verilog
from holoso._frontend import lower
from holoso._hir import optimize
from holoso._lir import build, pooled_write_word
from holoso._mir import lower as lower_to_mir

from ._cosim import run_cosim
from ._modelref import (
    ChainedSlots,
    COMPARATOR_OP_CASES,
    OperatorCase,
    PIPELINE_OP_CASES,
    SelectHold,
    branch_boundary_kernel,
    const_branch_kernel,
    diamond_then_loop_kernel,
    overlap_dead_arm_spill_kernel,
    overlap_div_err_kernel,
    overlap_spill_kernel,
)
from .hdl.hdl_float_oracle import HDL_DIR, REPO_ROOT, SIMULATORS, build_args, sources

pytestmark = pytest.mark.cosim


@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_small_kernel(sim: str) -> None:
    def kernel(a, b):  # type: ignore[no-untyped-def]
        return (a - b) * 0.25 + a * b

    run_cosim(sim, kernel, FloatFormat(8, 24), "kernel")


@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_division(sim: str) -> None:
    def blend(a, b, c):  # type: ignore[no-untyped-def]
        return a / b + c * 2.0

    run_cosim(sim, blend, FloatFormat(6, 18), "blend")


@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_ekf1_stateless(sim: str) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    import ekf1_stateless

    run_cosim(sim, ekf1_stateless.update_x_P, FloatFormat(6, 18), "update_x_P")


@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_ekf1_stateful(sim: str) -> None:
    # The stateful filter inlines the stateless kernel and threads its vector state (x, P_urt) across the random
    # transaction sequence, bit-for-bit against the model -- exercising the aggregates, inlining, and per-element slots.
    # Large measurement noise keeps the kernel's 1/x21 divisor dominated by the constant R_ct*R_shunt and the Kalman
    # gain tiny, so the random 64-step sequence cannot drive that divisor to an exact zero (err_pc) however the state
    # wanders. The DUT-vs-model bits agree regardless of stability; this config only keeps the err_pc check meaningful.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    import ekf1_stateful

    filt = ekf1_stateful.Ekf1(
        x=[0.0, 0.0, 0.0],
        P_urt=[1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
        R_diag=[1.0e3, 1.0e3],
        Q_diag=np.array([1.0e-6, 1.0e-6, 1.0e-6]),
    )
    run_cosim(sim, filt.update, FloatFormat(6, 18), "ekf1_stateful")


@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_staged_kernel(sim: str) -> None:
    def kernel(a, b):  # type: ignore[no-untyped-def]
        return (a - b) * 0.25 + a * b

    fmt = FloatFormat(8, 24)
    ops = OpConfig(
        FAddOperator(fmt, stage_decode=1),
        FMulOperator(fmt, stage_product=1),
        FDivOperator(fmt),
        FMulILog2OperatorFamily(fmt, stage_decode=1),
        FCmpOperator(fmt),
    )
    run_cosim(sim, kernel, fmt, "kernel_staged", ops=ops)


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
        FCmpOperator(fmt),
    )
    run_cosim(sim, blend, fmt, "blend_staged", ops=ops)


@pytest.mark.parametrize("config", COMPARATOR_OP_CASES, ids=lambda config: config.label)
@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_comparison_at_branch_boundary(sim: str, config: OperatorCase) -> None:
    # The boundary-slack corner kernel (see _modelref.branch_boundary_kernel), exercised at both comparator-only and
    # full-pipeline latency points. The white-box twin in test_schedule.py
    # (test_branch_comparison_commits_at_block_makespan) pins that this kernel actually hits the corner.
    fmt = FloatFormat(6, 18)
    run_cosim(sim, branch_boundary_kernel, fmt, f"cmp_branch_{config.label}", ops=config.make_ops(fmt))


@pytest.mark.parametrize("config", COMPARATOR_OP_CASES, ids=lambda config: config.label)
@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_overlap_spill(sim: str, config: OperatorCase) -> None:
    # Cross-block software pipelining (M7): the entry block shrinks its terminator to a wide chain's write word, and
    # that result spills past the terminator into BOTH single-predecessor arms, which read it. If the arm read did not
    # wait for the in-flight landing in the successor frame -- or the spill mis-aligned by even one frame -- the RTL
    # would diverge from the cycle-accurate model here. The white-box twin
    # (test_schedule.py test_overlap_spilled_result_lands_in_successor_frame) pins that the spill actually triggers.
    # See _modelref.overlap_spill_kernel. The staged cases move the early condition's landing relative to the spilling
    # chain, and the full staged case also moves the wide chain itself.
    fmt = FloatFormat(6, 18)
    run_cosim(sim, overlap_spill_kernel, fmt, f"overlap_spill_{config.label}", ops=config.make_ops(fmt))


@pytest.mark.parametrize("config", COMPARATOR_OP_CASES, ids=lambda config: config.label)
@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_const_branch(sim: str, config: OperatorCase) -> None:
    # Drained-boundary (round-5 fix): an empty const-branch block's condition is a pc-gated install read AT the
    # terminator, landing at the drained boundary; the drain must not shrink below it or the branch reads
    # the condition one PC before it lands. The model crashes (KeyError) on the first transaction, but a stale-register
    # branch misdirect is a SILENT RTL miscompile only cosim discriminates (RTL vs model from one LIR). The white-box
    # twin (test_schedule.py test_const_branch_install_block_drains_to_its_inline_landing) pins the block stays at the
    # drained boundary.
    fmt = FloatFormat(6, 18)
    run_cosim(sim, const_branch_kernel, fmt, f"const_branch_{config.label}", ops=config.make_ops(fmt))


@pytest.mark.parametrize("config", COMPARATOR_OP_CASES, ids=lambda config: config.label)
@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_diamond_then_loop(sim: str, config: OperatorCase) -> None:
    # Empty merge-block elimination (B4): threading the non-convertible diamond's empty merge onto its arms composes
    # its phi arms into the loop header, producing a THREE-arm loop-header phi (two forward init arms from the diamond
    # arms plus the back-edge) -- a phi shape no other kernel pushes through the RTL emitter. The model and RTL are
    # generated by different paths from one LIR, so only cosim discriminates a wrong emitter assumption (e.g. a loop-
    # header phi having exactly two arms) from a self-consistent model. The white-box twin
    # (test_schedule.py test_empty_merge_block_is_threaded_into_its_successor) pins that the merge was actually removed.
    fmt = FloatFormat(6, 18)
    run_cosim(sim, diamond_then_loop_kernel, fmt, f"diamond_then_loop_{config.label}", ops=config.make_ops(fmt))


@pytest.mark.parametrize("config", COMPARATOR_OP_CASES, ids=lambda config: config.label)
@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_overlap_dead_arm_spill(sim: str, config: OperatorCase) -> None:
    # P3a tightened the cross-block spill carry to the model's true landing PC (land - term_offset - 1), changing the
    # emitted Verilog for this kernel. cosim is the orthogonal gate to the source-semantic dead-arm test: it proves the
    # RTL's register write timing stays lockstep with the model's _pending re-keying under the tighter read-gate (the
    # model and RTL are generated by different paths from one LIR). See _modelref.overlap_dead_arm_spill_kernel; the
    # silent clobber it guards against is caught by test_schedule.py, not here (model and RTL share the register file).
    fmt = FloatFormat(6, 18)
    run_cosim(
        sim,
        overlap_dead_arm_spill_kernel,
        fmt,
        f"overlap_dead_arm_spill_{config.label}",
        ops=config.make_ops(fmt),
    )


@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_poly3(sim: str) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    import poly3

    run_cosim(sim, poly3.poly3, FloatFormat(6, 18), "poly3")


@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_madd(sim: str) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    import madd

    run_cosim(sim, madd.madd, FloatFormat(6, 18), "madd")


@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_trapezoidal_integrator(sim: str) -> None:
    # A stateful class: the bound method becomes a streaming module whose persistent state (the leaky accumulator y and
    # the one-sample delay _x_prev) is exercised across the whole random input sequence, bit-for-bit against the model.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    from trapezoidal_leaky_streaming_integrator import TrapezoidalLeakyStreamingIntegrator

    run_cosim(sim, TrapezoidalLeakyStreamingIntegrator(k=2**-22).__call__, FloatFormat(6, 18), "trapz_integrator")


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


class _UnusedBoolInputAccumulator:
    def __init__(self) -> None:
        self.y = 0.0

    def __call__(self, flag: bool, x):  # type: ignore[no-untyped-def]
        self.y = self.y + x + 1.0
        return self.y


@pytest.mark.parametrize("config", PIPELINE_OP_CASES, ids=lambda config: config.label)
@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_shift_register_backpressure(sim: str, config: OperatorCase) -> None:
    # The returned value taps a copy-slot register and the chain advances every accept, so together with the testbench's
    # random back-pressure this pins down that the boundary copy fires exactly once per accepted transaction -- no
    # mid-handshake output mutation and no state over-advance while out_ready is held low.
    fmt = FloatFormat(6, 18)
    run_cosim(sim, _ShiftRegister2().__call__, fmt, f"shift2_{config.label}", ops=config.make_ops(fmt))


@pytest.mark.parametrize("config", PIPELINE_OP_CASES, ids=lambda config: config.label)
@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_unused_bool_input_keeps_cfg_state_timing(sim: str, config: OperatorCase) -> None:
    fmt = FloatFormat(6, 18)
    vectors = [
        {"flag": 0, "x": fmt.encode(2.0)},
        {"flag": 1, "x": fmt.encode(4.0)},
    ]
    run_cosim(
        sim,
        _UnusedBoolInputAccumulator().__call__,
        fmt,
        f"unused_bool_state_{config.label}",
        ops=config.make_ops(fmt),
        vectors=vectors,
    )


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
        FCmpOperator(fmt),
    )
    run_cosim(sim, kernel, fmt, "new_stages", ops=ops)


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


@pytest.mark.parametrize("config", PIPELINE_OP_CASES, ids=lambda config: config.label)
@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_div0_error(sim: str, config: OperatorCase) -> None:
    def kdiv(a, b):  # type: ignore[no-untyped-def]
        return a / b

    fmt = FloatFormat(6, 18)
    name = f"kdiv_{config.label}"
    lir = build(lower_to_mir(optimize(lower(kdiv)), config.make_ops(fmt)), name, fetch_stages=3)
    gen_dir = REPO_ROOT / "build" / "holoso_gen" / sim / f"{name}_err_w{fmt.wexp}_{fmt.wman}"
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


# A 3-input variant of the err bench for the cross-block-overlap err_pc corner: it asserts the EXACT latched step
# (not merely nonzero), since the regression set err_pc to a wrong-but-nonzero value (the redirected successor frame).
_ERR_BENCH3 = """
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge, Timer
import holoso

_FMT = holoso.FloatFormat(@@WEXP@@, @@WMAN@@)


async def _transact(dut, x, y, z):
    while int(dut.in_ready.value) != 1:
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
    dut.in_x.value = int(_FMT.encode(x))
    dut.in_y.value = int(_FMT.encode(y))
    dut.in_z.value = int(_FMT.encode(z))
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
async def overlap_div0_errpc(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await FallingEdge(dut.clk)
    dut.rst.value = 1
    dut.in_valid.value = 0
    for _ in range(4):
        await RisingEdge(dut.clk)
    dut.rst.value = 0
    await FallingEdge(dut.clk)
    dut.out_ready.value = 1

    # x<z takes the NON-fall-through (true) arm; y==0 errs the entry-block division, whose result spills past the
    # shrunk terminator. err_pc must still latch the division's own step (@@ERRPC@@), not the redirected successor PC.
    assert await _transact(dut, 1.0, 2.0, 3.0) == 0, "clean divide on the overlapped arm spuriously flagged err_pc"
    assert await _transact(dut, 0.0, 0.0, 1.0) == @@ERRPC@@, "div0 latched the wrong err_pc step across the redirect"
    assert await _transact(dut, 1.0, 2.0, 3.0) == 0, "err_pc was not cleared after the erroring transaction"
"""


@pytest.mark.parametrize("config", COMPARATOR_OP_CASES, ids=lambda config: config.label)
@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_overlap_div0_errpc(sim: str, config: OperatorCase) -> None:
    # M7 regression (review round 3, Codex P1): an error-bearing division whose result spills past a SHRUNK
    # terminator must still latch err_pc to its OWN step, not the redirected non-fall-through successor frame. The data
    # is correct regardless (model == RTL), so only this step-exact err_pc cosim catches the regression. White-box
    # twin: test_schedule.py::test_overlap_keeps_error_op_diagnostic_latch_in_frame. See
    # _modelref.overlap_div_err_kernel.
    fmt = FloatFormat(6, 18)
    name = f"overlap_div_err_{config.label}"
    lir = build(lower_to_mir(optimize(lower(overlap_div_err_kernel)), config.make_ops(fmt)), name, fetch_stages=3)
    entry = next(block for block in lir.blocks if block.index == lir.entry)
    (fdiv,) = [op for op in entry.ops if op.inst.operator.error_ports]
    err_pc = lir.block_base[entry.index] + pooled_write_word(fdiv.commit_cycle)
    gen_dir = REPO_ROOT / "build" / "holoso_gen" / sim / f"{name}_errpc_w{fmt.wexp}_{fmt.wman}"
    gen_dir.mkdir(parents=True, exist_ok=True)
    build_dir = REPO_ROOT / "build" / "cocotb" / sim / f"errpc_{name}_w{fmt.wexp}_{fmt.wman}"
    (gen_dir / f"{name}.v").write_text(generate_verilog(lir).verilog)
    test_module = f"test_{name}_errpc"
    (gen_dir / f"{test_module}.py").write_text(
        _ERR_BENCH3.replace("@@WEXP@@", str(fmt.wexp))
        .replace("@@WMAN@@", str(fmt.wman))
        .replace("@@ERRPC@@", str(err_pc))
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


@pytest.mark.parametrize("config", COMPARATOR_OP_CASES, ids=lambda config: config.label)
@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_mirrored_comparisons_swap_orientation(sim: str, config: OperatorCase) -> None:
    # RTL twin of test_schedule.test_commutative_comparator_swap_permutes_output_taps: mirrored comparisons over one
    # operand pair make the port assignment orient one comparator firing swapped (its lt tap moving to gt), so the
    # emitted module must still produce both relations bit-exactly through the permuted lane.
    def kernel(a, b):  # type: ignore[no-untyped-def]
        below = a < b
        above = b < a
        return [float(below), float(above)]

    fmt = FloatFormat(6, 18)
    run_cosim(sim, kernel, fmt, f"mirrored_cmp_{config.label}", ops=config.make_ops(fmt))


@pytest.mark.parametrize("config", PIPELINE_OP_CASES, ids=lambda config: config.label)
@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_chained_slots_keep_the_old_value_across_the_install(sim: str, config: OperatorCase) -> None:
    # RTL twin of test_schedule.test_chained_slot_live_in_blocks_early_install: before the fix, "_b"'s early install
    # clobbered the value "_a"'s boundary copy reads, so the DUT diverged from the model on the second transaction.
    fmt = FloatFormat(6, 18)
    run_cosim(sim, ChainedSlots().__call__, fmt, f"chained_slots_{config.label}", ops=config.make_ops(fmt))


@pytest.mark.parametrize("config", COMPARATOR_OP_CASES, ids=lambda config: config.label)
@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_select_kernels(sim: str, config: OperatorCase) -> None:
    # If-converted selects in RTL: both polarities of a max kernel, an arm-sign select (the negation rides the
    # operand conditioner), and a comparison -> select -> arithmetic cross-bank chain, in one kernel.
    def kernel(a, b):  # type: ignore[no-untyped-def]
        m = a if a > b else b
        s = a if b > 0.0 else -a
        return m * 2.0 + s

    fmt = FloatFormat(6, 18)
    run_cosim(sim, kernel, fmt, f"select_mix_{config.label}", ops=config.make_ops(fmt))


@pytest.mark.parametrize("config", COMPARATOR_OP_CASES, ids=lambda config: config.label)
@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_select_reads_state_live_in_before_early_install(sim: str, config: OperatorCase) -> None:
    # RTL twin of test_schedule.test_state_early_install_respects_a_select_reader: the slot's early install must not
    # fire before the Ret-block select reads the OLD live-in value.
    fmt = FloatFormat(6, 18)
    run_cosim(sim, SelectHold().step, fmt, f"select_hold_{config.label}", ops=config.make_ops(fmt))


@pytest.mark.parametrize("config", COMPARATOR_OP_CASES, ids=lambda config: config.label)
@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_not_folding_sinks(sim: str, config: OperatorCase) -> None:
    # NOT-folding in RTL: inverted logic operands, an inverted bool output path (via the casts), both polarities of
    # one comparison, and a self-toggling inverted boolean state slot, across several transactions.
    class Toggle:
        def __init__(self) -> None:
            self._flip = False

        def step(self, a, b):  # type: ignore[no-untyped-def]
            old = self._flip
            self._flip = not self._flip
            gate = not (a > b)
            both = gate and (not (b > a))
            return float(gate) + 2.0 * float(both) + (4.0 if old else 0.0)

    fmt = FloatFormat(6, 18)
    run_cosim(sim, Toggle().step, fmt, f"not_sinks_{config.label}", ops=config.make_ops(fmt))


@pytest.mark.parametrize("config", COMPARATOR_OP_CASES, ids=lambda config: config.label)
@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_inverted_bool_phi_arm(sim: str, config: OperatorCase) -> None:
    # RTL twin of test_schedule.test_inverted_bool_phi_arm_installs_with_opposite_polarities: the conditional flag
    # negation rides the phi-arm install's inversion.
    def kernel(a, b, c):  # type: ignore[no-untyped-def]
        flag = a > b
        if c > 0.0:
            flag = not flag
            d = a / (c * c + 1.0)
        else:
            d = b
        return [float(flag), d]

    fmt = FloatFormat(6, 18)
    run_cosim(sim, kernel, fmt, f"inverted_arm_{config.label}", ops=config.make_ops(fmt))


@pytest.mark.parametrize("config", COMPARATOR_OP_CASES, ids=lambda config: config.label)
@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_phi_coalescing_residual_install_conflict(sim: str, config: OperatorCase) -> None:
    # RTL twin of test_schedule.test_phi_coalescing_residual_install_conflict_is_resolved: phi ``a`` would coalesce onto
    # input ``x``'s register, but ``x`` is still live as sibling phi ``z``'s identity arm (``z = x``) where ``a``'s
    # residual sign-folded else-arm install writes that register, so the soundness fixpoint de-coalesces ``a``. This
    # proves the de-coalesced residual install is bit-exact in RTL, not only against the Python cycle model. The
    # division keeps the diamond a real branch (un-if-converted), which is what creates the phi merge.
    def kernel(x, b, cc):  # type: ignore[no-untyped-def]
        if b < cc:
            a = x
            z = 1.0
            d = b
        else:
            a = -(x + 1.0)
            z = x
            d = x / b
        return [a, z, d]

    fmt = FloatFormat(6, 18)
    run_cosim(sim, kernel, fmt, f"coal_conflict_{config.label}", ops=config.make_ops(fmt))


@pytest.mark.parametrize("config", COMPARATOR_OP_CASES, ids=lambda config: config.label)
@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_not_over_loop_phi_and_inverted_public_state(sim: str, config: OperatorCase) -> None:
    # Two NOT-folding corners under RTL: a flag negated per trip of a while loop (the self-arm inversion install
    # fires once per iteration through the back edge), and a PUBLIC boolean state attribute whose live-out is a
    # negation (the state_<attr> port and the install both ride inversions).
    class LoopToggle:
        def __init__(self) -> None:
            self.armed = False

        def step(self, x):  # type: ignore[no-untyped-def]
            flag = x > 0.0
            w = x
            while w > 1.0:
                flag = not flag
                w = w - 1.0
            self.armed = not (x > 2.0)
            return float(flag)

    fmt = FloatFormat(6, 18)
    run_cosim(sim, LoopToggle().step, fmt, f"loop_toggle_{config.label}", ops=config.make_ops(fmt))
