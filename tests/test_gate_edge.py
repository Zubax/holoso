"""
Gate verification for the `transacting` issue/install qualifier.

Three structural guards (no simulator) keep a later refactor from silently dropping the gate: every operator
`iv`, every ucode-driven constant install, and every pooled commit write-enable must be ANDed with `transacting`.
Two cosims (Icarus, which exposes the internal `transacting` wire and the `regs` array; Verilator may optimize
them away) drive a swept idle dwell:
`test_transacting_edge_pins_at_accept_plus_fetch_lag` certifies the qualifier rises on exactly the genuine step-0 (a
late rise drops step-0; an early rise fires a spurious fill-window issue, cosim-invisible -- the hazard the gate exists
to stop), and `test_state_slot_inert_during_dwell` certifies a cycle-0 inline write does not commit to its
persistent-state register while the PC dwells idle at pc 0 (the held `ucode[0]` commits nothing).
`test_combinational_state_slot_inert_during_dwell` certifies the same of a combinational pooled firing, whose issue and
commit both ride `ucode[0]`.
"""

import re
import shutil
from collections.abc import Callable
from pathlib import Path

import pytest
from cocotb_tools.runner import get_runner

import holoso
from holoso import FloatFormat
from holoso._backend.verilog import generate as generate_verilog
from holoso._eel import lower
from holoso._lir import Lir
from holoso._mir import lower as lower_to_mir

from ._modelref import build_lir, combinational_options, default_mir, default_options, mir_options
from ._modelref import DEFAULT_UNROLL_MAX_TRIPS
from .hdl.hdl_float_oracle import REPO_ROOT, build_args

_HDL_DIR = Path(__file__).resolve().parent / "hdl"
_FMT = FloatFormat(8, 24)


def _cycle0_kernel(a: float, b: float) -> float:
    # Leads with a pooled fadd (a - b) and fmul (a * b) on cycle 0, so the iv gate is exercised on a real cycle-0 issue.
    return (a - b) * 0.25 + a * b


class _ConstInstallLoop:
    # The attribute's old value is never read, so it keeps no register: the loop counter's preheader arm is an
    # entry-block constant install writing 0.0 on cycle 0 -- it lands on ucode[0], so it must be transacting-gated to
    # stay inert while the PC dwells idle.
    def __init__(self) -> None:
        self._s = 5.0

    def __call__(self, n: float) -> float:
        self._s = 0.0
        while self._s < n:
            self._s = self._s + 1.0
        return self._s


class _CycleZeroStateWrite:
    # A persistent-state slot reset to 5.0 whose live-out is a select over its own live-in, committed in place by an
    # inline write on cycle 0 -- it lands on ucode[0], so it must be transacting-gated to leave the slot alone while the
    # PC dwells idle.
    def __init__(self) -> None:
        self._s = 5.0

    def __call__(self, n: float, keep: bool) -> float:
        self._s = self._s if keep else n
        return self._s * 2.0


class _CombinationalAccumulator:
    def __init__(self) -> None:
        self._s = 5.0

    def __call__(self, x: float) -> float:
        self._s = self._s + x
        return self._s


class _CombinationalIntAccumulator:
    # The integer adder takes no issue strobe, so the write opcode is the only thing the gate masks.
    def __init__(self) -> None:
        self._s = 5

    def __call__(self, x: int) -> int:
        self._s = self._s + x
        return self._s


def _verilog(fn: Callable[..., object], name: str) -> str:
    return holoso.synthesize(fn, default_options(_FMT), name=name).verilog_output.verilog


def _assert_effect_trigger_gated(fn: Callable[..., object], name: str, pattern: str) -> None:
    # Every decode wire for an effect-trigger field (named by `pattern`) must AND `transacting`, so a held ucode[0]
    # dwell, a fill bubble, or a stale pre-reset word triggers nothing.
    arms = [
        line.strip()
        for line in _verilog(fn, name).splitlines()
        if line.lstrip().startswith("wire") and re.search(pattern, line)
    ]
    assert arms, f"kernel produced no {pattern} wires to check"
    for line in arms:
        # A 1-bit trigger masks as `transacting & ...`; a wider write opcode as `{W{transacting}} & ...`.
        assert "transacting" in line and " & " in line, f"{pattern} decode is not gated by transacting: {line}"


def test_every_operator_iv_is_gated_by_transacting() -> None:
    # Operator issue can ride ucode[0] on a cycle-0-leading kernel, so the decoded in_valid field must AND transacting
    # to stay inert during the idle dwell; the operator's in_valid port then reads the gated field directly.
    _assert_effect_trigger_gated(_cycle0_kernel, "gate_iv", r"\buc_\w+_in_valid\b")


def test_const_install_is_gated_by_transacting() -> None:
    # A cycle-0 const-install sits on ucode[0]; its per-register write opcode must AND transacting so the held dwell
    # decodes to the NOP code (0) and installs nothing.
    _assert_effect_trigger_gated(_ConstInstallLoop().__call__, "gate_cwe", r"\buc_op_")


def test_pooled_write_enable_is_gated_by_transacting() -> None:
    # A pooled commit rides the executing word, so a held ucode[0] (accept dwell) or a stale pre-reset commit word must
    # commit nothing. Every write opcode ANDs transacting; the single decode-point gate covers both the register write
    # cases and the err path that reads those opcodes.
    _assert_effect_trigger_gated(_cycle0_kernel, "gate_we", r"\buc_op_")


def _run_bench(name: str, lir: Lir, testcase: str, env: dict[str, int], monkeypatch: pytest.MonkeyPatch) -> None:
    if shutil.which("iverilog") is None:
        pytest.skip("icarus (iverilog) not available")
    sim = "icarus"
    gen_dir = REPO_ROOT / "build" / "holoso_gen" / name
    gen_dir.mkdir(parents=True, exist_ok=True)
    (gen_dir / f"{name}.v").write_text(generate_verilog(lir, {}).verilog)
    build_dir = REPO_ROOT / "build" / "cocotb" / sim / name
    shutil.rmtree(build_dir, ignore_errors=True)
    runner = get_runner(sim)
    runner.build(
        sources=[gen_dir / f"{name}.v"],
        hdl_toplevel=name,
        build_args=build_args(sim),
        build_dir=str(build_dir),
        clean=True,
        timescale=("1ns", "1ps"),
    )
    monkeypatch.setenv("HOLOSO_FETCH_LAG", str(lir.fetch_lag))
    for key, value in env.items():
        monkeypatch.setenv(key, str(value))
    runner.test(
        hdl_toplevel=name,
        test_module="gate_edge_bench",
        testcase=testcase,
        test_dir=str(_HDL_DIR),
        build_dir=str(build_dir),
        results_xml=str(build_dir / "results.xml"),
    )


@pytest.mark.cosim
@pytest.mark.parametrize("k", [0, 1, 2, 3, 5])
def test_transacting_edge_pins_at_accept_plus_fetch_lag(k: int, monkeypatch: pytest.MonkeyPatch) -> None:
    name = f"gate_edge_k{k}"
    lir = build_lir(
        lower_to_mir(lower(_cycle0_kernel, DEFAULT_UNROLL_MAX_TRIPS).hir, default_mir(_FMT)),
        name,
    )
    assert any(op.issue_cycle == 0 for op in lir.blocks[0].ops), "kernel must issue a pooled op on cycle 0"
    _run_bench(name, lir, "transacting_edge", {"HOLOSO_DWELL_K": k}, monkeypatch)


@pytest.mark.cosim
@pytest.mark.parametrize("k", [1, 2, 3, 5])
def test_state_slot_inert_during_dwell(k: int, monkeypatch: pytest.MonkeyPatch) -> None:
    name = f"gate_state_k{k}"
    lir = build_lir(
        lower_to_mir(lower(_CycleZeroStateWrite().__call__, DEFAULT_UNROLL_MAX_TRIPS).hir, default_mir(_FMT)),
        name,
    )
    (slot,) = lir.wide_state_slots
    assert any(
        op.issue_cycle == 0 and slot.reg in (write.dst for write in op.writes) for op in lir.blocks[0].inline_ops
    ), "kernel must write its state slot on cycle 0"
    env = {
        "HOLOSO_DWELL_K": k,
        "HOLOSO_SLOT_IDX": slot.reg.index,
        "HOLOSO_SLOT_RESET_BITS": slot.reset_value.bits,
    }
    _run_bench(name, lir, "state_inert_during_dwell", env, monkeypatch)


@pytest.mark.cosim
@pytest.mark.parametrize("k", [1, 2, 3, 5])
@pytest.mark.parametrize(
    "label,make",
    [
        pytest.param("float", lambda: _CombinationalAccumulator().__call__, id="float"),
        pytest.param("int", lambda: _CombinationalIntAccumulator().__call__, id="int"),
    ],
)
def test_combinational_state_slot_inert_during_dwell(
    label: str, make: Callable[[], Callable[..., object]], k: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    name = f"gate_comb_{label}_k{k}"
    mir = lower_to_mir(lower(make(), DEFAULT_UNROLL_MAX_TRIPS).hir, mir_options(combinational_options(_FMT)))
    lir = build_lir(mir, name)
    (slot,) = lir.wide_state_slots
    assert any(
        op.latency == 0 and op.issue_cycle == 0 and slot.reg in (write.dst for write in op.writes)
        for op in lir.blocks[0].ops
    ), "kernel must commit its state slot from a combinational firing on cycle 0"
    env = {
        "HOLOSO_DWELL_K": k,
        "HOLOSO_SLOT_IDX": slot.reg.index,
        "HOLOSO_SLOT_RESET_BITS": slot.reset_value.bits,
    }
    _run_bench(name, lir, "state_inert_during_dwell", env, monkeypatch)
