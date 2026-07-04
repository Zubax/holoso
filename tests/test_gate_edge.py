"""
Gate verification for the ``transacting`` issue/install qualifier.

Three structural guards (no simulator) keep a later refactor from silently dropping the gate: every pooled operator
``iv``, every ucode-driven constant install, and every pooled commit write-enable must be ANDed with ``transacting``.
Two cosims (Icarus, which exposes the internal ``transacting`` wire and the ``regs`` array; Verilator may optimize
them away) drive a swept idle dwell:
``test_transacting_edge_pins_at_accept_plus_fetch_lag`` certifies the qualifier rises on exactly the genuine step-0 (a
late rise drops step-0; an early rise fires a spurious fill-window issue, inert for today's feed-forward operators and
thus cosim-invisible -- the hazard the gate exists to stop once iterative operators land), and
``test_state_slot_inert_during_dwell`` certifies a cycle-0 constant install does not commit to its persistent-state
register while the PC dwells idle at pc 0 (the held ``ucode[0]`` commits nothing).
"""

import shutil
from collections.abc import Callable
from pathlib import Path

import pytest
from cocotb_tools.runner import get_runner

from holoso import FloatFormat
from holoso._backend.verilog import generate as generate_verilog
from holoso._frontend import lower
from holoso._hir import optimize
from holoso._lir import Lir, build
from holoso._mir import lower as lower_to_mir

from ._modelref import default_ops
from .hdl.hdl_float_oracle import HDL_DIR, REPO_ROOT, build_args, sources

_HDL_DIR = Path(__file__).resolve().parent / "hdl"
_FMT = FloatFormat(8, 24)


def _cycle0_kernel(a: float, b: float):  # type: ignore[no-untyped-def]
    # Leads with a pooled fadd (a - b) and fmul (a * b) on cycle 0, so the iv gate is exercised on a real cycle-0 issue.
    return (a - b) * 0.25 + a * b


class _ConstInstallState:
    # A persistent-state slot reset to 5.0 that an entry-block constant install overwrites with 0.0 on cycle 0 -- the
    # install lands on ucode[0], so it must be transacting-gated to stay inert while the PC dwells idle.
    def __init__(self) -> None:
        self._s = 5.0

    def __call__(self, n: float):  # type: ignore[no-untyped-def]
        self._s = 0.0
        while self._s < n:
            self._s = self._s + 1.0
        return self._s


def _verilog(fn: Callable[..., object], name: str) -> str:
    return generate_verilog(build(lower_to_mir(optimize(lower(fn)), default_ops(_FMT)), name, fetch_stages=3)).verilog


def _assert_effect_trigger_gated(fn: Callable[..., object], name: str, prefix: str) -> None:
    # Every decode wire for an effect-trigger field (named by ``prefix``) must AND ``transacting``, so a held ucode[0]
    # dwell, a fill bubble, or a stale pre-reset word triggers nothing.
    arms = [
        line.strip() for line in _verilog(fn, name).splitlines() if line.lstrip().startswith("wire") and prefix in line
    ]
    assert arms, f"kernel produced no {prefix} wires to check"
    for line in arms:
        # A 1-bit trigger masks as ``transacting & ...``; a wider write opcode as ``{W{transacting}} & ...``.
        assert "transacting" in line and " & " in line, f"{prefix} decode is not gated by transacting: {line}"


def test_every_operator_iv_is_gated_by_transacting() -> None:
    # Operator issue can ride ucode[0] on a cycle-0-leading kernel, so the decoded uc_issue field must AND transacting
    # to stay inert during the idle dwell; the operator's in_valid port then reads the gated field directly.
    _assert_effect_trigger_gated(_cycle0_kernel, "gate_iv", "uc_issue_")


def test_const_install_is_gated_by_transacting() -> None:
    # A cycle-0 const-install sits on ucode[0]; its per-register write opcode must AND transacting so the held dwell
    # decodes to the NOP code (0) and installs nothing. This float-slot kernel covers the mechanism.
    _assert_effect_trigger_gated(_ConstInstallState().__call__, "gate_cwe", "uc_op_")


def test_pooled_write_enable_is_gated_by_transacting() -> None:
    # A pooled commit rides the executing word, so a held ucode[0] (accept dwell) or a stale pre-reset commit word must
    # commit nothing. Every write opcode ANDs transacting; the single decode-point gate covers both the register write
    # cases and the err path that reads those opcodes.
    _assert_effect_trigger_gated(_cycle0_kernel, "gate_we", "uc_op_")


def _run_bench(name: str, lir: Lir, testcase: str, env: dict[str, int], monkeypatch: pytest.MonkeyPatch) -> None:
    if shutil.which("iverilog") is None:
        pytest.skip("icarus (iverilog) not available")
    sim = "icarus"
    gen_dir = REPO_ROOT / "build" / "holoso_gen" / name
    gen_dir.mkdir(parents=True, exist_ok=True)
    (gen_dir / f"{name}.v").write_text(generate_verilog(lir).verilog)
    build_dir = REPO_ROOT / "build" / "cocotb" / sim / name
    shutil.rmtree(build_dir, ignore_errors=True)
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
    lir = build(lower_to_mir(optimize(lower(_cycle0_kernel)), default_ops(_FMT)), name, fetch_stages=3)
    assert any(op.issue_cycle == 0 for op in lir.blocks[lir.entry].ops), "kernel must issue a pooled op on cycle 0"
    _run_bench(name, lir, "transacting_edge", {"HOLOSO_DWELL_K": k}, monkeypatch)


@pytest.mark.cosim
@pytest.mark.parametrize("k", [1, 2, 3, 5])
def test_state_slot_inert_during_dwell(k: int, monkeypatch: pytest.MonkeyPatch) -> None:
    name = f"gate_state_k{k}"
    lir = build(lower_to_mir(optimize(lower(_ConstInstallState().__call__)), default_ops(_FMT)), name, fetch_stages=3)
    slots = lir.float_state_slots
    assert slots, "kernel must have a float state slot with a cycle-0 const install"
    slot = slots[0]
    env = {
        "HOLOSO_DWELL_K": k,
        "HOLOSO_SLOT_IDX": slot.reg.index,
        "HOLOSO_SLOT_RESET_BITS": _FMT.encode(slot.reset_value),
    }
    _run_bench(name, lir, "state_inert_during_dwell", env, monkeypatch)
