"""Functional cosimulation: drive generated modules and check their outputs bit-for-bit against the model backend."""

import dataclasses
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from jaxtyping import Float64
import pytest
import zkf

import holoso
from holoso import FAddOptions, FCordicOptions, FloatFormat, OperatorOptions, Options
from holoso._backend.verilog import generate as generate_verilog
from holoso._eel import lower
from holoso._lir._ir import OperatorInstance, exits, successor_blocks
from holoso._mir import lower as lower_to_mir

from holoso._value import ScalarLike
from ._cosim import generated_dir, rendered_instance, run_bench, run_cosim
from ._public import instance_count
from ._modelref import (
    branch_boundary_kernel,
    build_lir,
    mir_options,
    ChainedSlots,
    COMPARATOR_OPTIONS_CASES,
    const_branch_kernel,
    default_options,
    DEFAULT_UNROLL_MAX_TRIPS,
    diamond_then_loop_kernel,
    instantiated_modules,
    EMPTY_RET_KERNELS,
    EMPTY_RET_VECTORS,
    EXIT_ARM_KERNELS,
    EXIT_ARM_VECTORS,
    OptionsCase,
    overlap_dead_arm_spill_kernel,
    overlap_div_err_kernel,
    overlap_spill_kernel,
    phi_swap_computed_loop,
    PIPELINE_OPTIONS_CASES,
    InputLatchSelect,
    SelectHold,
    SharedLiveOut,
    SharedLiveOutBool,
    staged_options,
)
from .hdl.hdl_float_oracle import SIMULATORS
from . import _options as lean

pytestmark = pytest.mark.cosim


@dataclass(frozen=True, slots=True)
class _Specialization:
    """Of the ports of a block built for every primitive, those a specialization to some of them keeps and drops."""

    kept: frozenset[str]
    dropped: frozenset[str]

    def holds(self, ports: frozenset[str]) -> bool:
        return self.kept <= ports and not self.dropped & ports


_QUOTIENT = _Specialization(frozenset({"a", "b", "y"}), frozenset({"op_sqrt"}))
_ROOT = _Specialization(frozenset({"a", "y"}), frozenset({"op_sqrt", "b"}))
_ROTATION = _Specialization(frozenset({"a", "r0", "r1", "quadrant"}), frozenset({"vectoring", "b"}))
_VECTORING = _Specialization(frozenset({"a", "b", "r0", "r1"}), frozenset({"vectoring", "quadrant"}))
_ANGLE = _Specialization(frozenset({"a", "b", "r0"}), frozenset({"vectoring", "quadrant", "r1"}))
_ROUNDING = _Specialization(frozenset({"a", "y_float", "y_int"}), frozenset({"round_mode"}))


def _lean_options(**operators: Any) -> Options:
    """The lean configuration of every float operator, the conversions included, with `operators` replaced."""
    options = default_options(FloatFormat(6, 18))
    conversions: dict[str, Any] = {"frint": lean.frint(), "ffromint": lean.ffromint()}
    return dataclasses.replace(options, operator=dataclasses.replace(options.operator, **(conversions | operators)))


@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_small_kernel(sim: str) -> None:
    def kernel(a: float, b: float) -> float:
        return (a - b) * 0.25 + a * b

    run_cosim(sim, holoso.synthesize(kernel, default_options(FloatFormat(8, 24)), name="kernel"))


@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_greek_identifiers(sim: str) -> None:
    # The interface names come from Python identifiers, which admit Greek; Verilog does not. Only a real elaboration
    # proves the spelled-out names reach the DUT, since the bench addresses every port by that same name.
    def kernel(ω: float, Δt: float) -> float:
        return ω * Δt

    run_cosim(sim, holoso.synthesize(kernel, default_options(FloatFormat(8, 24)), name="greek_kernel"))


@pytest.mark.parametrize("config", COMPARATOR_OPTIONS_CASES, ids=lambda config: config.label)
@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_comparison_at_branch_boundary(sim: str, config: OptionsCase) -> None:
    # The boundary-slack corner kernel (see _modelref.branch_boundary_kernel), exercised at both comparator-only and
    # full-pipeline latency points. The white-box twin in test_schedule.py
    # (test_branch_comparison_commits_at_block_makespan) pins that this kernel actually hits the corner.
    fmt = FloatFormat(6, 18)
    result = holoso.synthesize(branch_boundary_kernel, config.make_options(fmt), name=f"cmp_branch_{config.label}")
    run_cosim(sim, result)


@pytest.mark.parametrize("config", COMPARATOR_OPTIONS_CASES, ids=lambda config: config.label)
@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_overlap_spill(sim: str, config: OptionsCase) -> None:
    # Cross-block software pipelining: the entry block shrinks its terminator to a wide chain's write word, and
    # that result spills past the terminator into BOTH single-predecessor arms, which read it. If the arm read did not
    # wait for the in-flight landing in the successor frame -- or the spill mis-aligned by even one frame -- the RTL
    # would diverge from the cycle-accurate model here. The white-box twin
    # (test_schedule.py test_overlap_spilled_result_lands_in_successor_frame) pins that the spill actually triggers.
    # See _modelref.overlap_spill_kernel. The staged cases move the early condition's landing relative to the spilling
    # chain, and the full staged case also moves the wide chain itself.
    fmt = FloatFormat(6, 18)
    result = holoso.synthesize(overlap_spill_kernel, config.make_options(fmt), name=f"overlap_spill_{config.label}")
    run_cosim(sim, result)


@pytest.mark.parametrize("config", COMPARATOR_OPTIONS_CASES, ids=lambda config: config.label)
@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_const_branch(sim: str, config: OptionsCase) -> None:
    fmt = FloatFormat(6, 18)
    result = holoso.synthesize(const_branch_kernel, config.make_options(fmt), name=f"const_branch_{config.label}")
    run_cosim(sim, result)


@pytest.mark.parametrize("config", COMPARATOR_OPTIONS_CASES, ids=lambda config: config.label)
@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_diamond_then_loop(sim: str, config: OptionsCase) -> None:
    # Empty merge-block elimination: threading the non-convertible diamond's empty merge onto its arms composes
    # its phi arms into the loop header, producing a THREE-arm loop-header phi (two forward init arms from the diamond
    # arms plus the back-edge) -- a phi shape no other kernel pushes through the RTL emitter. The model and RTL are
    # generated by different paths from one LIR, so only cosim discriminates a wrong emitter assumption (e.g. a loop-
    # header phi having exactly two arms) from a self-consistent model. The white-box twin
    # (test_schedule.py test_empty_merge_block_is_threaded_into_its_successor) pins that the merge was actually removed.
    fmt = FloatFormat(6, 18)
    result = holoso.synthesize(
        diamond_then_loop_kernel, config.make_options(fmt), name=f"diamond_then_loop_{config.label}"
    )
    run_cosim(sim, result)


@pytest.mark.parametrize("config", COMPARATOR_OPTIONS_CASES, ids=lambda config: config.label)
@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_overlap_dead_arm_spill(sim: str, config: OptionsCase) -> None:
    # Cosim is the orthogonal gate to the source-semantic dead-arm test: the cross-block spill carry reaches the model's
    # true landing PC (land - term_offset - 1), and the RTL's register write timing must stay in lockstep with the
    # model's re-keying (the model and RTL are generated by different paths from one LIR). See
    # _modelref.overlap_dead_arm_spill_kernel; the silent clobber it guards against is caught by
    # test_overlap_behavior.py, not here (model and RTL share the register file).
    fmt = FloatFormat(6, 18)
    result = holoso.synthesize(
        overlap_dead_arm_spill_kernel, config.make_options(fmt), name=f"overlap_dead_arm_spill_{config.label}"
    )
    run_cosim(sim, result)


@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_min_max(sim: str) -> None:
    # Binary min/max lower to the comparator. Taking BOTH min and max of one pair fuses to a SINGLE firing that writes
    # two wide registers at once, so this proves the generated RTL lands both results bit-exactly against the model.
    # The |a|,|b| sort in a second firing exercises the operand sign conditioners.
    def kernel(a: float, b: float) -> tuple[float, ...]:
        return min(a, b), max(a, b), min(abs(a), abs(b))

    options = Options(
        operator=OperatorOptions(
            fadd=lean.fadd(),
            fmul=lean.fmul(),
            fdivsqrt=lean.fdivsqrt(),
            fmul_ilog2=lean.fmul_ilog2(),
            fcmp=lean.fcmp(),
        ),
        ffmt=FloatFormat(8, 24),
        wmultiplier=18,
    )
    run_cosim(sim, holoso.synthesize(kernel, options, name="min_max"))


@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_relations_beside_min_max(sim: str) -> None:
    # One comparator orders a pair for its min, its max and two relations in a single firing, one relation tapped
    # mirrored, and compares another pair in a firing of its own; only the RTL's port wiring can get those taps wrong.
    def kernel(a: float, b: float, c: float) -> tuple[float, float, bool, bool, bool]:
        return min(a, -b), max(a, -b), a < -b, -b < a, a < c

    result = holoso.synthesize(kernel, default_options(FloatFormat(8, 24)), name="min_max_flags")
    assert instance_count(result.verilog_output.verilog, "fcmp") == 1, "the premise of this test"
    values = (0.0, 1.5, -1.5, 2.0, -7.25, float("inf"), float("-inf"))
    run_cosim(sim, result, vectors=[{"a": a, "b": b, "c": c} for a in values for b in values for c in values[:3]])


def _min_beside_a_flag_twice(a: float, b: float) -> tuple[float, bool, bool]:
    return min(a, b), a < b, a >= b


def _max_beside_two_flags_twice(a: float, b: float) -> tuple[float, bool, bool, bool]:
    return max(a, b), b > a, a == b, a != b


def _difference_beside_a_flag_twice(a: int, b: int) -> tuple[int, bool, bool, bool]:
    return a - b, a < b, b < a, a >= b


@pytest.mark.parametrize("instances", [1, 2])
@pytest.mark.parametrize(
    "kernel", [_min_beside_a_flag_twice, _max_beside_two_flags_twice, _difference_beside_a_flag_twice]
)
@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_a_second_tap_of_a_flag_beside_a_sort_or_a_difference(
    sim: str, kernel: Callable[..., object], instances: int
) -> None:
    # A flag tapped twice under opposite inversions takes a firing of its own beside the one that sorts or subtracts;
    # tapping flags alone, that firing may exchange its operands, which moves its taps to their mirrors.
    options = default_options(FloatFormat(8, 24))
    operator = dataclasses.replace(
        options.operator,
        fcmp=lean.fcmp(instances=instances),
        iadds=holoso.IAddsOptions(instances=instances),
    )
    options = dataclasses.replace(options, operator=operator, wint_min=16)
    result = holoso.synthesize(kernel, options, name=f"second_tap_{kernel.__name__.strip('_')}_x{instances}")
    if kernel is _difference_beside_a_flag_twice:
        values: tuple[float | int, ...] = (0, 1, -1, 7, -7, 32767, -32768)
    else:
        values = (0.0, -0.0, 1.5, -1.5, 2.0, float("inf"), float("-inf"))
    run_cosim(sim, result, vectors=[{"a": a, "b": b} for a in values for b in values])


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

    def __call__(self, flag: bool, x: float) -> float:
        self.y = self.y + x + 1.0
        return self.y


class _BoolShiftRegister2:
    def __init__(self) -> None:
        self._a = False
        self._b = False

    def __call__(self, x: bool) -> bool:
        out = self._b
        self._b = self._a
        self._a = x
        return out


@pytest.mark.parametrize("bank", ["wide", "bool"])
@pytest.mark.parametrize("config", PIPELINE_OPTIONS_CASES, ids=lambda config: config.label)
@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_shift_register_backpressure(sim: str, config: OptionsCase, bank: str) -> None:
    # The returned value taps a copy-slot register and the chain advances every accept, so together with the testbench's
    # random back-pressure this pins down that the boundary copy fires exactly once per accepted transaction -- no
    # mid-handshake output mutation and no state over-advance while out_ready is held low.
    fmt = FloatFormat(6, 18)
    kernel = _ShiftRegister2().__call__ if bank == "wide" else _BoolShiftRegister2().__call__
    result = holoso.synthesize(kernel, config.make_options(fmt), name=f"shift2_{bank}_{config.label}")
    run_cosim(sim, result)


@pytest.mark.parametrize("label", list(EMPTY_RET_KERNELS))
@pytest.mark.parametrize("config", PIPELINE_OPTIONS_CASES, ids=lambda config: config.label)
@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_boundary_install_from_an_empty_ret(sim: str, config: OptionsCase, label: str) -> None:
    fmt = FloatFormat(6, 18)
    result = holoso.synthesize(
        EMPTY_RET_KERNELS[label](), config.make_options(fmt), name=f"empty_ret_{label}_{config.label}"
    )
    run_cosim(sim, result, vectors=[{"x": x, "y": y} for x, y in EMPTY_RET_VECTORS])


@pytest.mark.parametrize("label", list(EXIT_ARM_KERNELS))
@pytest.mark.parametrize("config", PIPELINE_OPTIONS_CASES, ids=lambda config: config.label)
@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_branch_arm_ending_the_transaction(sim: str, config: OptionsCase, label: str) -> None:
    fmt = FloatFormat(6, 18)
    result = holoso.synthesize(EXIT_ARM_KERNELS[label], config.make_options(fmt), name=f"{label}_{config.label}")
    run_cosim(sim, result, vectors=[{"x": x, "y": y} for x, y in EXIT_ARM_VECTORS])


@pytest.mark.parametrize("config", PIPELINE_OPTIONS_CASES, ids=lambda config: config.label)
@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_unused_bool_input_keeps_cfg_state_timing(sim: str, config: OptionsCase) -> None:
    fmt = FloatFormat(6, 18)
    vectors: list[Mapping[str, ScalarLike]] = [
        {"flag": False, "x": 2.0},
        {"flag": True, "x": 4.0},
    ]
    result = holoso.synthesize(
        _UnusedBoolInputAccumulator().__call__, config.make_options(fmt), name=f"unused_bool_state_{config.label}"
    )
    run_cosim(sim, result, vectors=vectors)


@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_new_operator_stages(sim: str) -> None:
    def kernel(a: float, b: float, c: float) -> float:
        return (a - b) / c + a * b * 0.25  # fadd, fdivsqrt, fmul, and fmul_ilog2 (the 2^-2 scale) all in one kernel

    # Nondefault pipeline stages on fadd, fmul, fdivsqrt and fmul_ilog2, so the latency model must match the staged RTL
    # end to end.
    options = Options(
        operator=OperatorOptions(
            fadd=lean.fadd(stage_input=1, stage_core=FAddOptions.CoreStages.NORMALIZE | FAddOptions.CoreStages.PACK),
            fmul=lean.fmul(stage_input=1),
            fdivsqrt=lean.fdivsqrt(stage_output=1),
            fmul_ilog2=lean.fmul_ilog2(stage_input=1),
            fcmp=lean.fcmp(),
        ),
        ffmt=FloatFormat(8, 24),
        wmultiplier=18,
    )
    run_cosim(sim, holoso.synthesize(kernel, options, name="new_stages"))


# The generated bench only checks err_pc == 0 over a bounded input range, so it never exercises the zero-divisor ->
# err_pc path. This custom bench drives an exact zero divisor and asserts the diagnostic is set, then cleared on the
# next accepted transaction. It cannot reuse the generated bench because the numerical model does not predict errors.
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
    dut.in_a.value = _FMT.encode(a).bits
    dut.in_b.value = _FMT.encode(b).bits
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
    Clock(dut.clk, 10, unit="ns").start()
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


def _run_err_bench(sim: str, name: str, fmt: FloatFormat, verilog: str, bench_text: str, suffix: str = "err") -> None:
    label = f"{name}_{suffix}_w{fmt.wexp}_{fmt.wman}"
    verilog_path = generated_dir(sim, label) / f"{name}.v"
    verilog_path.write_text(verilog)
    run_bench(sim, label, [verilog_path], name, bench_text)


@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_div0_error(sim: str) -> None:
    # Staged only: this is the one public-entry row where the error rides the divider's full stage pipeline.
    def kdiv(a: float, b: float) -> float:
        return a / b

    fmt = FloatFormat(6, 18)
    name = "kdiv_staged"
    verilog = holoso.synthesize(kdiv, staged_options(fmt), name=name).verilog_output.verilog
    bench = _ERR_BENCH.replace("@@WEXP@@", str(fmt.wexp)).replace("@@WMAN@@", str(fmt.wman))
    _run_err_bench(sim, name, fmt, verilog, bench)


# log2 has two error ports (domain_error for x<0, pole for x==0), both ORed into the single err_pc. As with a zero
# divisor, the generated bench never reaches the error path (it asserts err_pc == 0 and the numerical model does not
# predict errors), so this custom bench drives both error operands and asserts err_pc latches for each, then clears.
_LOG2_ERR_BENCH = """
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge, Timer
import holoso

_FMT = holoso.FloatFormat(@@WEXP@@, @@WMAN@@)


async def _transact(dut, x):
    while int(dut.in_ready.value) != 1:
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
    dut.in_x.value = _FMT.encode(x).bits
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
async def log2_errpc(dut):
    Clock(dut.clk, 10, unit="ns").start()
    await FallingEdge(dut.clk)
    dut.rst.value = 1
    dut.in_valid.value = 0
    for _ in range(4):
        await RisingEdge(dut.clk)
    dut.rst.value = 0
    await FallingEdge(dut.clk)
    dut.out_ready.value = 1

    assert await _transact(dut, 8.0) == 0, "clean log2 spuriously flagged err_pc"
    assert await _transact(dut, 0.0) != 0, "log2(+0) pole did not set err_pc"
    assert await _transact(dut, 8.0) == 0, "err_pc was not cleared after the pole"
    assert await _transact(dut, -4.0) != 0, "log2(x<0) domain error did not set err_pc"
    assert await _transact(dut, 8.0) == 0, "err_pc was not cleared after the domain error"
"""


@pytest.mark.parametrize("config", PIPELINE_OPTIONS_CASES, ids=lambda config: config.label)
@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_log2_error(sim: str, config: OptionsCase) -> None:
    def klog2(x: float) -> float:
        return math.log2(x)

    fmt = FloatFormat(6, 18)
    name = f"klog2_{config.label}"
    verilog = holoso.synthesize(klog2, config.make_options(fmt), name=name).verilog_output.verilog
    bench = _LOG2_ERR_BENCH.replace("@@WEXP@@", str(fmt.wexp)).replace("@@WMAN@@", str(fmt.wman))
    _run_err_bench(sim, name, fmt, verilog, bench)


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
    dut.in_x.value = _FMT.encode(x).bits
    dut.in_y.value = _FMT.encode(y).bits
    dut.in_z.value = _FMT.encode(z).bits
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
    Clock(dut.clk, 10, unit="ns").start()
    await FallingEdge(dut.clk)
    dut.rst.value = 1
    dut.in_valid.value = 0
    for _ in range(4):
        await RisingEdge(dut.clk)
    dut.rst.value = 0
    await FallingEdge(dut.clk)
    dut.out_ready.value = 1

    # x<z takes the true arm and y==0 errs a division across a redirect; err_pc must latch the division's own step
    # (@@ERRPC@@), not a successor's.
    assert await _transact(dut, 1.0, 2.0, 3.0) == 0, "a clean transaction spuriously flagged err_pc"
    assert await _transact(dut, 0.0, 0.0, 1.0) == @@ERRPC@@, "div0 latched the wrong err_pc step across the redirect"
    assert await _transact(dut, 1.0, 2.0, 3.0) == 0, "err_pc was not cleared after the erroring transaction"
"""


@pytest.mark.parametrize("config", COMPARATOR_OPTIONS_CASES, ids=lambda config: config.label)
@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_overlap_div0_errpc(sim: str, config: OptionsCase) -> None:
    # Overlap regression: an error-bearing division whose result spills past a SHRUNK
    # terminator must still latch err_pc to its OWN step, not the redirected non-fall-through successor frame. The data
    # is correct regardless (model == RTL), so only this step-exact err_pc cosim catches the regression. White-box
    # twin: test_schedule.py::test_overlap_keeps_error_op_diagnostic_latch_in_frame. See
    # _modelref.overlap_div_err_kernel. The expected err_pc step has no public spelling, so this test stays on the
    # private LIR chain and emits the RTL from the same LIR it reads the step from.
    fmt = FloatFormat(6, 18)
    name = f"overlap_div_err_{config.label}"
    options = config.make_options(fmt)
    lir = build_lir(
        lower_to_mir(lower(overlap_div_err_kernel, DEFAULT_UNROLL_MAX_TRIPS).hir, mir_options(options)),
        name,
    )
    entry = lir.blocks[0]
    (fdiv,) = [op for op in entry.ops if op.inst.operator.error_ports]
    err_pc = lir.block_base[entry.index] + fdiv.commit_cycle
    bench = (
        _ERR_BENCH3.replace("@@WEXP@@", str(fmt.wexp))
        .replace("@@WMAN@@", str(fmt.wman))
        .replace("@@ERRPC@@", str(err_pc))
    )
    _run_err_bench(sim, name, fmt, generate_verilog(lir, {}).verilog, bench, suffix="errpc")


def _threaded_div_err_kernel(x: float, y: float, z: float) -> float:
    if x < z:
        r = 1.0  # threaded into the comparing block, so every later block starts four steps earlier
    else:
        r = x / z  # an error-bearing division keeps the branch real
    return (r + x) / y


@pytest.mark.parametrize("config", COMPARATOR_OPTIONS_CASES, ids=lambda config: config.label)
@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_threaded_arm_div0_errpc(sim: str, config: OptionsCase) -> None:
    # The model does not predict err_pc, so a merge division erring past a threaded arm is checked step-exact in RTL:
    # the true arm is threaded out, and err_pc must name the merge division's step in the layout without it.
    fmt = FloatFormat(6, 18)
    name = f"threaded_div_err_{config.label}"
    options = config.make_options(fmt)
    lir = build_lir(
        lower_to_mir(lower(_threaded_div_err_kernel, DEFAULT_UNROLL_MAX_TRIPS).hir, mir_options(options)), name
    )
    assert len(lir.blocks) == 3, "the constant arm must be threaded out"
    (merge,) = [block for block in lir.blocks if exits(block.terminator)]
    (fdiv,) = [op for op in merge.ops if op.inst.operator.error_ports]
    err_pc = lir.block_base[merge.index] + fdiv.commit_cycle
    bench = (
        _ERR_BENCH3.replace("@@WEXP@@", str(fmt.wexp))
        .replace("@@WMAN@@", str(fmt.wman))
        .replace("@@ERRPC@@", str(err_pc))
    )
    _run_err_bench(sim, name, fmt, generate_verilog(lir, {}).verilog, bench, suffix="errpc")


@pytest.mark.parametrize(
    "config",
    [config for config in COMPARATOR_OPTIONS_CASES if config.label != "staged"],
    ids=lambda config: config.label,
)
@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_mirrored_comparisons_swap_orientation(sim: str, config: OptionsCase) -> None:
    # RTL twin of test_schedule.test_commutative_comparator_swap_permutes_output_taps: mirrored comparisons over one
    # operand pair make the port assignment orient one comparator firing swapped (its lt tap moving to gt), so the
    # emitted module must still produce both relations bit-exactly through the permuted lane. The default row checks
    # the port permutation and fcmp_s1 the delayed comparator taps; full staging adds no new mapping here.
    def kernel(a: float, b: float) -> tuple[float, ...]:
        below = a < b
        above = b < a
        return float(below), float(above)

    fmt = FloatFormat(6, 18)
    run_cosim(sim, holoso.synthesize(kernel, config.make_options(fmt), name=f"mirrored_cmp_{config.label}"))


@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_two_multiplier_instances_co_issue(sim: str) -> None:
    # Two co-issued products realize the second multiplier; two firings driven into one instance would leave one of
    # them with the other's product, which the lockstep comparison against the model catches.
    def kernel(a: float, b: float, c: float, d: float) -> float:
        return a * b + c * d

    fmt = FloatFormat(6, 18)
    options = default_options(fmt)
    operators = dataclasses.replace(options.operator, fmul=lean.fmul(instances=2))
    options = dataclasses.replace(options, operator=operators)
    lir = build_lir(lower_to_mir(lower(kernel, DEFAULT_UNROLL_MAX_TRIPS).hir, mir_options(options)), "fmul_pair")
    products = [op for block in lir.blocks for op in block.ops if op.inst.operator.name == "fmul"]
    assert {op.inst.name for op in products} == {"fmul_0", "fmul_1"}
    assert len({op.issue_cycle for op in products}) == 1, "the products must co-issue for the pool to bind both"
    run_cosim(sim, holoso.synthesize(kernel, options, name="fmul_pair"))


def _sincos_seam(x: float, c: bool) -> float:
    t = math.sin(2.0 * math.pi * x)
    if c:
        r = math.sin(2.0 * math.pi * t)
    else:
        r = t * 2.0
    return r


def _atan2_seam(x: float, y: float, c: bool) -> float:
    t = math.atan2(y, x) / (2.0 * math.pi)
    if c:
        r = math.atan2(t, x) / (2.0 * math.pi)
    else:
        r = t * 2.0
    return r


def _rotation_to_vectoring_seam(x: float, y: float, c: bool) -> float:
    t = math.sin(2.0 * math.pi * x)
    if c:
        r = math.atan2(t, y) / (2.0 * math.pi)
    else:
        r = t * 2.0
    return r


def _vectoring_to_rotation_seam(x: float, y: float, c: bool) -> float:
    t = math.atan2(y, x) / (2.0 * math.pi)
    if c:
        r = math.sin(2.0 * math.pi * t)
    else:
        r = t * 2.0
    return r


@pytest.mark.parametrize(
    "kernel, switches_vectoring",
    [
        (_sincos_seam, False),
        (_atan2_seam, False),
        (_rotation_to_vectoring_seam, True),
        (_vectoring_to_rotation_seam, True),
    ],
    ids=lambda value: value.__name__.strip("_") if callable(value) else None,
)
@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_a_core_reissues_at_the_first_step_past_a_seam(
    sim: str, kernel: Callable[..., float], switches_vectoring: bool
) -> None:
    # A CORDIC core stays busy for its whole initiation interval, which is its latency, and no busy window is carried
    # across a block seam: the entry's terminator sits at the core's write word, the entry firing's latency after its
    # issue, and the arm reissues on the same instance at its first step, the step after the terminator, whether the
    # arm's firing rotates or vectors. The kernel's over-issue `$fatal` is the oracle, which the model cannot see.
    fmt = FloatFormat(6, 18)
    options = dataclasses.replace(default_options(fmt), ifconv_max_ops=0)
    name = f"seam_{kernel.__name__.strip('_')}"
    lir = build_lir(lower_to_mir(lower(kernel, DEFAULT_UNROLL_MAX_TRIPS).hir, mir_options(options)), name)
    entry = lir.blocks[0]
    by_index = {block.index: block for block in lir.blocks}
    (first,) = [op for op in entry.ops if op.initiation_interval > 1]
    reissues = [
        (entry.term_offset + 1 + op.issue_cycle - first.issue_cycle, op.drive.immediates != first.drive.immediates)
        for arm in successor_blocks(entry.terminator)
        for op in by_index[arm].ops
        if op.inst == first.inst
    ]
    assert first.initiation_interval == first.latency
    expected = [(first.latency + 1, switches_vectoring)]
    assert reissues == expected, "the premise needs the reissue right at the seam"
    run_cosim(sim, holoso.synthesize(kernel, options, name=name))


@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_two_cordic_instances_with_contending_windows(sim: str) -> None:
    # Three sin/cos products contend for two CORDIC instances whose busy windows are long (II well above 1), so a
    # wrong rebinding would double-issue; the kernel's over-issue `$fatal` is the oracle.
    def kernel(a: float, b: float, c: float) -> tuple[float, float, float]:
        return math.sin(a) * math.cos(b), math.sin(b) + math.cos(c), math.sin(c) * math.sin(a)

    fmt = FloatFormat(6, 18)
    options = default_options(fmt)
    assert options.operator.fcordic is not None
    operators = dataclasses.replace(options.operator, fcordic=lean.fcordic(instances=2), fmul=lean.fmul(instances=2))
    options = dataclasses.replace(options, operator=operators)
    lir = build_lir(lower_to_mir(lower(kernel, DEFAULT_UNROLL_MAX_TRIPS).hir, mir_options(options)), "sincos_pair")
    assert {inst.name for inst in lir.instances} >= {"fcordic_0", "fcordic_1"}
    run_cosim(sim, holoso.synthesize(kernel, options, name="sincos_pair"))


@pytest.mark.parametrize(
    "config",
    [
        None,
        lean.fcordic(stage_core_mul=1, stage_core=FCordicOptions.CoreStages.NORMALIZE | FCordicOptions.CoreStages.PACK),
    ],
)
@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_one_cordic_both_rotates_and_vectors(sim: str, config: FCordicOptions | None) -> None:
    # Rotations and vectorings share one CORDIC instance, a rotation followed by a vectoring and a vectoring by a
    # rotation at the tightest spacing each allows, the earlier firing's own initiation interval, while each commits at
    # its own latency; the vectoring's magnitude feeds a fused hypot, and the kernel's over-issue `$fatal`
    # guards the spacing.
    def kernel(a: float, b: float, x: float, y: float, z: float) -> tuple[float, float, float]:
        s = math.sin(a)
        theta = math.atan2(s, x)
        return math.cos(theta) * math.hypot(s, x), math.atan2(y, z), math.sin(b)

    options = default_options(FloatFormat(6, 18))
    if config is not None:
        options = dataclasses.replace(options, operator=dataclasses.replace(options.operator, fcordic=config))
    name = "cordic_both" if config is None else "cordic_both_staged"
    lir = build_lir(lower_to_mir(lower(kernel, DEFAULT_UNROLL_MAX_TRIPS).hir, mir_options(options)), name)
    assert {inst.name for inst in lir.instances if inst.operator.name == "fcordic"} == {"fcordic_0"}
    (block,) = lir.blocks
    firings = sorted((op for op in block.ops if op.inst.name == "fcordic_0"), key=lambda op: op.issue_cycle)
    tight = {
        (dict(earlier.drive.immediates)["vectoring"], dict(later.drive.immediates)["vectoring"])
        for earlier, later in zip(firings, firings[1:])
        if later.issue_cycle - earlier.issue_cycle == earlier.initiation_interval
    }
    switches = {(False, True), (True, False)}
    assert tight >= switches, firings
    run_cosim(sim, holoso.synthesize(kernel, options, name=name))


@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_two_cordic_instances_each_specialized_to_its_own_drive(sim: str) -> None:
    # A rotation and a vectoring issued together take an instance each, so each instance renders its block specialized
    # to its own drive alone, the other drive's datapath dropped; the vectoring's magnitude also feeds a fused hypot.
    def kernel(a: float, y: float, x: float) -> tuple[float, float, float]:
        return math.sin(a), math.atan2(y, x), math.hypot(y, x)

    options = default_options(FloatFormat(6, 18))
    options = dataclasses.replace(
        options, operator=dataclasses.replace(options.operator, fcordic=lean.fcordic(instances=2))
    )
    lir = build_lir(lower_to_mir(lower(kernel, DEFAULT_UNROLL_MAX_TRIPS).hir, mir_options(options)), "cordic_pair")
    vectorings: dict[OperatorInstance, set[int]] = {}
    for op in (op for block in lir.blocks for op in block.ops if op.inst.operator.name == "fcordic"):
        vectorings.setdefault(op.inst, set()).add(dict(op.drive.immediates)["vectoring"])
    assert sorted(tuple(values) for values in vectorings.values()) == [(False,), (True,)], "the premise of this test"
    result = holoso.synthesize(kernel, options, name="cordic_pair")
    ports = [rendered_instance(result, f"fcordic_{index}").ports for index in range(2)]
    assert any(map(_ROTATION.holds, ports)) and any(map(_VECTORING.holds, ports))
    run_cosim(sim, result)


@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_two_comparator_instances_swap_and_rebind(sim: str) -> None:
    # Six comparisons over two shared operands, two of them mirrored, on two comparator instances: the allocator
    # both flips the mirrored firings (their lt tap moving to gt) and rebinds firings across the instances so each
    # instance's ports read one operand register. Every relation must survive the permuted boolean lanes bit-exactly.
    def kernel(a: float, b: float, c: float, d: float, e: float, f: float, g: float, h: float) -> tuple[float, ...]:
        return float(d < c), float(a < b), float(a < e), float(c < f), float(h < c), float(a < g)

    fmt = FloatFormat(6, 18)
    options = default_options(fmt)
    options = dataclasses.replace(options, operator=dataclasses.replace(options.operator, fcmp=lean.fcmp(instances=2)))
    lir = build_lir(lower_to_mir(lower(kernel, DEFAULT_UNROLL_MAX_TRIPS).hir, mir_options(options)), "cmp_pair")
    assert {inst.name for inst in lir.instances} == {"fcmp_0", "fcmp_1"}
    run_cosim(sim, holoso.synthesize(kernel, options, name="cmp_pair"))


@pytest.mark.parametrize("config", PIPELINE_OPTIONS_CASES, ids=lambda config: config.label)
@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_chained_slots_keep_the_old_value_across_the_install(sim: str, config: OptionsCase) -> None:
    # RTL twin of test_schedule.test_chained_slot_live_in_blocks_early_install: before the fix, "_b"'s early install
    # clobbered the value "_a"'s boundary copy reads, so the DUT diverged from the model on the second transaction.
    fmt = FloatFormat(6, 18)
    result = holoso.synthesize(ChainedSlots().__call__, config.make_options(fmt), name=f"chained_slots_{config.label}")
    run_cosim(sim, result)


@pytest.mark.parametrize("config", COMPARATOR_OPTIONS_CASES, ids=lambda config: config.label)
@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_select_kernels(sim: str, config: OptionsCase) -> None:
    # If-converted selects in RTL: both polarities of a max kernel, an arm-sign select (the negation rides the
    # operand conditioner), and a comparison -> select -> arithmetic cross-bank chain, in one kernel.
    def kernel(a: float, b: float) -> float:
        m = a if a > b else b
        s = a if b > 0.0 else -a
        return m * 2.0 + s

    fmt = FloatFormat(6, 18)
    run_cosim(sim, holoso.synthesize(kernel, config.make_options(fmt), name=f"select_mix_{config.label}"))


@pytest.mark.parametrize("config", COMPARATOR_OPTIONS_CASES, ids=lambda config: config.label)
@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_select_reads_state_live_in_before_early_install(sim: str, config: OptionsCase) -> None:
    # RTL twin of test_schedule.test_state_early_install_respects_a_select_reader: the slot's early install must not
    # fire before the Ret-block select reads the OLD live-in value.
    fmt = FloatFormat(6, 18)
    run_cosim(sim, holoso.synthesize(SelectHold().step, config.make_options(fmt), name=f"select_hold_{config.label}"))


@pytest.mark.parametrize("config", PIPELINE_OPTIONS_CASES, ids=lambda config: config.label)
@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_input_latch_installs_on_the_ret_blocks_first_cycle(sim: str, config: OptionsCase) -> None:
    fmt = FloatFormat(6, 18)
    kernel = InputLatchSelect().__call__
    run_cosim(sim, holoso.synthesize(kernel, config.make_options(fmt), name=f"input_latch_select_{config.label}"))


@pytest.mark.parametrize("config", COMPARATOR_OPTIONS_CASES, ids=lambda config: config.label)
@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_not_folding_sinks(sim: str, config: OptionsCase) -> None:
    # NOT-folding in RTL: inverted logic operands, an inverted bool output path (via the casts), both polarities of
    # one comparison, and a self-toggling inverted boolean state slot, across several transactions.
    class Toggle:
        def __init__(self) -> None:
            self._flip = False

        def step(self, a: float, b: float) -> float:
            old = self._flip
            self._flip = not self._flip
            gate = not (a > b)
            both = gate and (not (b > a))
            return float(gate) + 2.0 * float(both) + (4.0 if old else 0.0)

    fmt = FloatFormat(6, 18)
    run_cosim(sim, holoso.synthesize(Toggle().step, config.make_options(fmt), name=f"not_sinks_{config.label}"))


@pytest.mark.parametrize("config", COMPARATOR_OPTIONS_CASES, ids=lambda config: config.label)
@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_inverted_bool_phi_arm(sim: str, config: OptionsCase) -> None:
    # RTL twin of test_schedule.test_inverted_bool_phi_arm_installs_with_opposite_polarities: the conditional flag
    # negation rides the phi-arm install's inversion.
    def kernel(a: float, b: float, c: float) -> tuple[float, ...]:
        flag = a > b
        if c > 0.0:
            flag = not flag
            d = a / (c * c + 1.0)
        else:
            d = b
        return float(flag), d

    fmt = FloatFormat(6, 18)
    run_cosim(sim, holoso.synthesize(kernel, config.make_options(fmt), name=f"inverted_arm_{config.label}"))


@pytest.mark.parametrize("config", COMPARATOR_OPTIONS_CASES, ids=lambda config: config.label)
@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_phi_coalescing_residual_install_conflict(sim: str, config: OptionsCase) -> None:
    # RTL twin of test_allocation_behavior.test_phi_coalescing_residual_install_conflict_is_resolved: phi `a` would
    # coalesce onto input `x`'s register, but `x` is still live as sibling phi `z`'s identity arm (`z = x`) where `a`'s
    # residual sign-folded else-arm install writes that register, so the soundness fixpoint de-coalesces `a`. This
    # proves the de-coalesced residual install is bit-exact in RTL, not only against the Python cycle model. The
    # division keeps the diamond a real branch (un-if-converted), which is what creates the phi merge.
    def kernel(x: float, b: float, cc: float) -> tuple[float, ...]:
        if b < cc:
            a = x
            z = 1.0
            d = b
        else:
            a = -(x + 1.0)
            z = x
            d = x / b
        return a, z, d

    fmt = FloatFormat(6, 18)
    run_cosim(sim, holoso.synthesize(kernel, config.make_options(fmt), name=f"coal_conflict_{config.label}"))


@pytest.mark.parametrize("config", COMPARATOR_OPTIONS_CASES, ids=lambda config: config.label)
@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_not_over_loop_phi_and_inverted_public_state(sim: str, config: OptionsCase) -> None:
    # Two NOT-folding corners under RTL: a flag negated per trip of a while loop (the self-arm inversion install
    # fires once per iteration through the back edge), and a PUBLIC boolean state attribute whose live-out is a
    # negation (the state_<attr> port and the install both ride inversions).
    class LoopToggle:
        def __init__(self) -> None:
            self.armed = False

        def step(self, x: float) -> float:
            was_armed = self.armed
            flag = x > 0.0
            w = x
            while w > 1.0:
                flag = not flag
                w = w - 1.0
            self.armed = not (x > 2.0)
            return float(flag) + float(was_armed)

    fmt = FloatFormat(6, 18)
    run_cosim(sim, holoso.synthesize(LoopToggle().step, config.make_options(fmt), name=f"loop_toggle_{config.label}"))


@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_two_sincos_share_ii_instance(sim: str) -> None:
    # Two independent sincos time-share one II>1 CORDIC; the schedule must space issues by the initiation interval.
    # Under-spacing drops the second in_valid (SIMULATION $fatal) or diverges from the model -- end-to-end II>1 proof.
    def kernel(a: float, b: float) -> tuple[float, float, float, float]:
        return math.sin(a), math.cos(a), math.sin(b), math.cos(b)

    run_cosim(sim, holoso.synthesize(kernel, default_options(FloatFormat(8, 24)), name="cs_two_sincos"))


@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_two_atan2_share_ii_instance(sim: str) -> None:
    # The vectoring twin of test_cosim_two_sincos_share_ii_instance: deeper latency than rotation, so a distinct
    # re-accept boundary the schedule must space by the initiation interval.
    def kernel(a: float, b: float, c: float, d: float) -> tuple[float, float]:
        return math.atan2(a, b), math.atan2(c, d)

    run_cosim(sim, holoso.synthesize(kernel, default_options(FloatFormat(8, 24)), name="cs_two_atan2"))


@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_atan2_hypot_fused(sim: str) -> None:
    # hypot and atan2 over one operand pair fuse into a single CORDIC firing (theta + magnitude).
    def kernel(y: float, x: float) -> tuple[float, float]:
        return math.hypot(y, x), math.atan2(y, x)

    run_cosim(sim, holoso.synthesize(kernel, default_options(FloatFormat(8, 24)), name="cs_atan2_hypot"))


@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_root_and_lone_hypot(sim: str) -> None:
    # The native root inside a real design, and the lone hypot's expansion around it -- two exponent extractions,
    # an integer max, three exact scalings and a sum of squares. ZKF's own suite proves the blocks; this proves
    # the machine built around them, including the integer datapath the expansion reaches into. No division
    # and no comparator: the expansion needs neither.
    def kernel(x: float, y: float) -> tuple[float, float]:
        return math.sqrt(abs(x)), math.hypot(x, y)

    options = Options(
        operator=OperatorOptions(
            fadd=lean.fadd(),
            fmul=lean.fmul(),
            fmul_ilog2=lean.fmul_ilog2(),
            filog2=lean.filog2(),
            fdivsqrt=lean.fdivsqrt(),
        ),
        ffmt=FloatFormat(8, 24),
        wmultiplier=18,
    )
    run_cosim(sim, holoso.synthesize(kernel, options, name="cs_root_hypot"))


@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_divider_shared_by_quotient_and_root(sim: str) -> None:
    # The instance choosing its operation per firing on `op_sqrt`, the one divider no single-operation kernel renders.
    def kernel(a: float, b: float) -> tuple[float, float]:
        return a / b, math.sqrt(abs(a))

    result = holoso.synthesize(kernel, default_options(FloatFormat(8, 24)), name="cs_quotient_and_root")
    assert {"op_sqrt", "b"} <= rendered_instance(result, "fdivsqrt_0").ports
    run_cosim(sim, result)


@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_n_ary_norm(sim: str) -> None:
    # The Euclidean norm of a four-vector: one exponent extraction per leg, a compare/select tree over the integer
    # bank, five exact scalings and a pairwise sum of squares under one root -- the n-ary shape the two-legged
    # expansion above never exercises, driven through real RTL.
    def kernel(v: Float64[np.ndarray, "4"]) -> float:
        return float(np.linalg.norm(v))

    options = Options(
        operator=OperatorOptions(
            fadd=lean.fadd(),
            fmul=lean.fmul(),
            fmul_ilog2=lean.fmul_ilog2(),
            filog2=lean.filog2(),
            fdivsqrt=lean.fdivsqrt(),
        ),
        ffmt=FloatFormat(8, 24),
        wmultiplier=18,
    )
    run_cosim(sim, holoso.synthesize(kernel, options, name="cs_norm4"))


@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_library_composites(sim: str) -> None:
    # Composite library stubs end-to-end: cbrt's sign/exp2/log2 expansion behind its zero guard, tan's sincos+div,
    # the pow rung ladder under a static exponent, sinh's exp difference, and tanh's stable sigmoid form -- all
    # bit-exact against the model backend.
    def kernel(x: float, y: float) -> tuple[float, float, float, float, float]:
        return math.cbrt(x), math.tan(y), pow(x, 3.0), math.sinh(x), math.tanh(y)

    run_cosim(sim, holoso.synthesize(kernel, default_options(FloatFormat(8, 24)), name="cs_lib_composites"))


@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_tan_pole_latches_no_error(sim: str) -> None:
    # At the format-nearest pi/2, cos rounds to exactly zero. tan's c==0 guard is a real branch (the divide is
    # unspeculatable, so if-conversion cannot fold it into an always-executed select), which skips the divide at the
    # pole -- so no divide-by-zero error is latched. The bench asserts err_pc==0 on every vector, so an unguarded
    # sin/cos would fail here; a normal input in the same run confirms the divide path still works.
    def kernel(x: float) -> float:
        return math.tan(x)

    fmt = FloatFormat(8, 24)
    pole = float(np.float32(math.pi / 2))
    vectors: list[Mapping[str, ScalarLike]] = [
        {"x": pole},
        {"x": -pole},
        {"x": 0.5},
    ]
    run_cosim(sim, holoso.synthesize(kernel, default_options(fmt), name="cs_tan_pole"), vectors=vectors)


@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_phi_swap_computed_arm(sim: str) -> None:
    # The RTL leg of the computed-arm swap regression (see test_interpret): the DUT replays the same LIR as the model,
    # so this pins the emitted install words against the bench's bit-exact model across multi-trip loop executions.
    # Explicit vectors keep the while trip count small; the bench's own sweep would draw huge counts.
    fmt = FloatFormat(6, 18)
    vectors: list[Mapping[str, ScalarLike]] = [{"x": x, "n": n} for x in (1.0, 2.0, -1.5) for n in (1.0, 2.0, 4.0)]
    result = holoso.synthesize(phi_swap_computed_loop, default_options(fmt), name="cs_phi_swap_computed")
    run_cosim(sim, result, vectors=vectors)


@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_shared_live_out_state_slots(sim: str) -> None:
    # The install arm outranks the opcode arm in the same statement, so if the two could ever be enabled on one PC the
    # DUT would diverge from the model here -- across a persistent recurrence, on the very first transaction.
    # _modelref.SharedLiveOut is the one home of this kernel; its elaboration/premise twin lives in test_backend.
    result = holoso.synthesize(SharedLiveOut().step, default_options(FloatFormat(6, 18)), name="shared_live_out")
    run_cosim(sim, result)


@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_shared_live_out_bool_state_slots(sim: str) -> None:
    # The boolean bank leans on the same priority argument as the wide one, and boolean installs are boundary-only, so
    # a shadowed write here would diverge from the model on the recurrence rather than on a single transaction.
    result = holoso.synthesize(
        SharedLiveOutBool().step, default_options(FloatFormat(6, 18)), name="shared_bool_live_out"
    )
    run_cosim(sim, result)


def _quotient(a: float, b: float) -> float:
    return a / b


def _root(a: float) -> float:
    return math.sqrt(abs(a))


def _sincos(a: float) -> tuple[float, float]:
    return math.sin(a), math.cos(a)


def _atan2_and_hypot(y: float, x: float) -> tuple[float, float]:
    return math.atan2(y, x), math.hypot(y, x)


def _atan2(y: float, x: float) -> float:
    return math.atan2(y, x)


def _floor(x: float) -> int:
    return math.floor(x)


@pytest.mark.parametrize(
    "kernel,instance,specialization",
    [
        pytest.param(_quotient, "fdivsqrt_0", _QUOTIENT, id="quotient"),
        pytest.param(_root, "fdivsqrt_0", _ROOT, id="root"),
        pytest.param(_sincos, "fcordic_0", _ROTATION, id="rotation"),
        pytest.param(_atan2_and_hypot, "fcordic_0", _VECTORING, id="vectoring"),
        pytest.param(_atan2, "fcordic_0", _ANGLE, id="vectoring_without_magnitude"),
        pytest.param(_floor, "frint_0", _ROUNDING, id="rounding"),
    ],
)
@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_an_instance_running_one_primitive_renders_its_block_specialized_to_it(
    sim: str, kernel: Callable[..., object], instance: str, specialization: _Specialization
) -> None:
    # The operator is built for every primitive, but this kernel's one instance runs one, so it renders the block
    # specialized to that primitive: no immediate, none of the ports the primitive leaves unused, and for an angle whose
    # magnitude nothing reads, no `r1`. The model evaluates the specialized block too, so what this catches beyond the
    # fast session's model-against-interpreter comparison is a connection the specialization broke.
    result = holoso.synthesize(kernel, _lean_options(), name=f"specialized{kernel.__name__}")
    assert specialization.holds(rendered_instance(result, instance).ports)
    run_cosim(sim, result)


def _quotient_beside_root(a: float, b: float, c: float) -> tuple[float, float]:
    return a / b, math.sqrt(abs(c))


def _floor_beside_ceil(x: float, y: float) -> tuple[int, int]:
    return math.floor(x), math.ceil(y)


@pytest.mark.parametrize(
    "kernel,operator,specializations",
    [
        pytest.param(_quotient_beside_root, "fdivsqrt", (_QUOTIENT, _ROOT), id="quotient_and_root"),
        pytest.param(_floor_beside_ceil, "frint", (_ROUNDING, _ROUNDING), id="floor_and_ceil"),
    ],
)
@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_primitives_issued_together_take_an_instance_each_specialized_to_its_own(
    sim: str, kernel: Callable[..., object], operator: str, specializations: tuple[_Specialization, ...]
) -> None:
    # Two firings of different primitives issued together take an instance each once two are allowed, so each
    # instance renders a block of its own, specialized to its own primitive alone, under a module name of its own.
    options = _lean_options(**{operator: dataclasses.replace(getattr(_lean_options().operator, operator), instances=2)})
    result = holoso.synthesize(kernel, options, name=f"separated_{operator}")
    rendered = [rendered_instance(result, f"{operator}_{index}") for index in range(2)]
    assert len({instance.module for instance in rendered}) == 2
    assert all(any(spec.holds(instance.ports) for instance in rendered) for spec in specializations)
    run_cosim(sim, result)


def _rotation_beside_atan2(a: float, y: float, x: float) -> tuple[float, float, float]:
    return math.sin(a), math.cos(a), math.atan2(y, x)


@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_a_cordic_rotating_beside_a_lone_angle_keeps_r1_without_the_magnitude(sim: str) -> None:
    # One instance both rotates and vectors and no vectoring firing reads the magnitude, so its block is built without
    # the length but keeps `r1`, which carries the rotation's cosine; its ports are those of the full block, so the
    # knob table tells it apart.
    result = holoso.synthesize(_rotation_beside_atan2, _lean_options(), name="rotation_beside_angle")
    rendered = rendered_instance(result, "fcordic_0")
    assert {"vectoring", "a", "b", "r0", "r1", "quadrant"} <= rendered.ports
    assert rendered.knobs["_vectoring"] == "None" and rendered.knobs["_length"] == "False"
    run_cosim(sim, result)


def _twin(a: float, b: float) -> float:
    return a * b + a


def _declared_modules(verilog: str) -> set[str]:
    return set(re.findall(r"^module (\S+?)\s*\(", verilog, re.MULTILINE))


def _lockstep_top(name: str, first: holoso.SynthesisResult, second: holoso.SynthesisResult) -> str:
    """
    A module with the kernels' interface that drives both with the same inputs and presents the first; the simulation
    stops where the second departs from the first, in its handshake at any cycle or in its results while they are
    valid. Registers start unknown until the first reset, so nothing is compared before it.
    """
    assert [(port.name, port.direction, port.width) for port in first.ports] == [
        (port.name, port.direction, port.width) for port in second.ports
    ]
    outputs = [port for port in first.ports if port.direction == holoso.Direction.OUT]
    handshake = [port.name for port in outputs if isinstance(port, holoso.ControlOutputPort)]
    results = [port.name for port in outputs if isinstance(port, holoso.DataOutputPort)]

    def concatenated(names: list[str], prefix: str = "") -> str:
        return "{" + ", ".join(prefix + name for name in names) + "}"

    def instance(module: str, prefix: str) -> str:
        connections = ",\n".join(
            f"    .{port.name}({prefix if port.direction == holoso.Direction.OUT else ''}{port.name})"
            for port in first.ports
        )
        return f"{module} u_{module} (\n{connections}\n);"

    declarations = ",\n".join(
        f"    {'input ' if port.direction == holoso.Direction.IN else 'output'} wire [{port.width - 1}:0] {port.name}"
        for port in first.ports
    )
    second_outputs = "\n".join(f"wire [{port.width - 1}:0] second_{port.name};" for port in outputs)
    return f"""
module {name} (
{declarations}
);
{second_outputs}
{instance(first.module_name, "")}
{instance(second.module_name, "second_")}
reg reset_seen = 1'b0;
always @(posedge clk) begin
    reset_seen <= reset_seen | rst;
    if (reset_seen && !rst && {concatenated(handshake)} !== {concatenated(handshake, "second_")})
        $fatal(1, "the second kernel's handshake departed from the first's");
    if (reset_seen && !rst && out_valid && {concatenated(results)} !== {concatenated(results, "second_")})
        $fatal(1, "the second kernel's results departed from the first's");
end
endmodule
"""


@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_two_kernels_with_identical_operators_in_one_simulation(sim: str) -> None:
    # A design may hold several kernels built with the same options, whose operators are then the same configurations
    # under the same digests; the kernel's prefix on each private module is what keeps the two files apart in one
    # build. A simulator may resolve a name declared twice to one of the declarations, so the disjointness is asserted
    # outright as well.
    options = _lean_options()
    first = holoso.synthesize(_twin, options, name="twin_first")
    second = holoso.synthesize(_twin, options, name="twin_second")
    texts = [result.verilog_output.verilog for result in (first, second)]
    blocks = [
        {
            module.removeprefix("\\").removeprefix(f"_{result.module_name}_")
            for module in _declared_modules(text)
            if "_zkf_" in module
        }
        for result, text in zip((first, second), texts)
    ]
    assert blocks[0] == blocks[1] and len(blocks[0]) >= 2, "the premise: the same blocks under the same digests"
    assert not _declared_modules(texts[0]) & _declared_modules(texts[1])
    gen_dir = generated_dir(sim, "twin_kernels")
    sources = [gen_dir / f"{result.module_name}.v" for result in (first, second)] + [gen_dir / "twin_kernels.v"]
    for path, text in zip(sources, [*texts, _lockstep_top("twin_kernels", first, second)]):
        path.write_text(text)
    run_bench(sim, "twin_kernels", sources, "twin_kernels", first.cocotb_output.testbench)


def _float_and_integer_sums(x: float, n: int) -> tuple[float, int]:
    return x + 1.0, n + 1


@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_a_kernel_renders_no_operator_it_does_not_instantiate(sim: str) -> None:
    # Every float operator is configured and every integer one available, but only what the kernel instantiates may
    # reach its file, as a module or otherwise.
    result = holoso.synthesize(_float_and_integer_sums, _lean_options(), name="operator_subset")
    assert instantiated_modules(result) == {"add", "iadds"}
    declared = re.findall(rf"^module _operator_subset_zkf_(\w+?)_[0-9a-f]{{12}}\(", result.verilog_output.verilog, re.M)
    assert sorted(declared) == ["add", "iadds"]
    run_cosim(sim, result)


@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_the_microcode_rom_attributes_reach_its_read_register(sim: str) -> None:
    options = dataclasses.replace(_lean_options(), ucode_rom_attrs={"rom_style": "block"})
    result = holoso.synthesize(_twin, options, name="ucode_rom_attrs")
    assert re.search(r'^\(\* rom_style = "block" \*\)\nreg \[UCW-1:0\] ucode_q;', result.verilog_output.verilog, re.M)
    run_cosim(sim, result)


def _exp2(x: float) -> float:
    return math.exp2(x)


@pytest.mark.parametrize("sim", SIMULATORS)
def test_cosim_an_exp2_carries_its_table_attributes_on_its_block(sim: str) -> None:
    attributes = {"rom_style": "distributed"}
    options = _lean_options(fexp2=lean.fexp2(rom_attrs=attributes))
    name = "exp2_rom_attrs"
    lir = build_lir(lower_to_mir(lower(_exp2, DEFAULT_UNROLL_MAX_TRIPS).hir, mir_options(options)), name)
    (instance,) = [instance for instance in lir.instances if instance.operator.name == "fexp2"]
    block = lir.instances[instance]
    assert isinstance(block, zkf.Exp2) and block.rom_attrs == attributes
    run_cosim(sim, holoso.synthesize(_exp2, options, name=name))
