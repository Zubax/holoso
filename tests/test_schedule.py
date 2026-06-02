"""Unit tests for pipelined scheduling, register allocation, and LIR construction."""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import pytest

from holoso import FAddOperator, FDivOperator, FloatFormat, FMulILog2OperatorFamily, FMulOperator, OpConfig
from holoso._errors import UnsupportedConstruct
from holoso._frontend import lower
from holoso._hir import optimize
from holoso._lir import FloatRegRef
from holoso._lir._schedule import DEPENDENCY_EDGE
from holoso._mir import (
    lower as lower_to_mir,
    Mir,
    MirBuilder,
    MirFloatConst,
    MirFloatInput,
    MirFloatOperation,
    MirFloatOutput,
    MirFloatView,
    MirInput,
    MirOperation,
)
from holoso._operators import FMulILog2Operator, FloatSignControl, HardwareOperator
from holoso._lir import build
from holoso._lir._schedule import resolve_pool, schedule_ops
from holoso._type import FloatType, ScalarSignature, ScalarType

FMT = FloatFormat(6, 18)
OPS = OpConfig(FAddOperator(FMT), FMulOperator(FMT), FDivOperator(FMT), FMulILog2OperatorFamily(FMT))


@dataclass(frozen=True)
class _TestHardwareOperator(HardwareOperator):
    @property
    def latency(self) -> int:
        return 1

    @property
    def signature(self) -> ScalarSignature:
        ty = FloatType(FMT)
        return ScalarSignature((ty,), ty)

    def render(self, *operands: str) -> str:
        (operand,) = operands
        return f"{self.mnemonic}({operand})"

    def hdl_params(self) -> dict[str, int]:
        return {}


@dataclass(frozen=True)
class ATestHardwareOperator(_TestHardwareOperator):
    mnemonic: ClassVar[str] = "atest"


@dataclass(frozen=True, slots=True)
class OtherScalarType(ScalarType):
    @property
    def width(self) -> int:
        return 1


@dataclass(frozen=True, slots=True)
class OtherMirInput(MirInput):
    pass


def _run(target, ops: OpConfig = OPS) -> Mir:  # type: ignore[no-untyped-def]
    return lower_to_mir(optimize(lower(target)), ops)


def _view(mir: Mir) -> MirFloatView:
    return MirFloatView.from_mir(mir)


def _schedule(mir: Mir):
    view = _view(mir)
    return schedule_ops(view, resolve_pool(view))


def _muls(mir: Mir) -> list[int]:
    return [vid for vid, n in mir.nodes.items() if isinstance(n, MirOperation) and isinstance(n.operator, FMulOperator)]


def test_schedule_respects_dependencies() -> None:
    def f(a, b):  # type: ignore[no-untyped-def]
        return (a - b) * 0.25 + a * b

    mir = _run(f)
    sched = _schedule(mir)
    for vid, cycle in sched.issue_cycle.items():
        op = mir.nodes[vid]
        assert isinstance(op, MirOperation)
        assert cycle >= 1  # nothing issues on the accept cycle
        for operand in op.operands:
            node = mir.nodes[operand]
            if isinstance(node, MirOperation):
                # A consumer issues no earlier than the producer's commit plus the register-file traversal edge
                # (the read-first write edge plus the read and write latches).
                assert cycle >= sched.issue_cycle[operand] + node.operator.latency + DEPENDENCY_EDGE


def test_pipelined_issue_overlaps_a_slow_op() -> None:
    # A fast chain advances while an unrelated slow divide is still in flight -- the barrier model could not do this.
    def f(a, b, c):  # type: ignore[no-untyped-def]
        return a / b + (a + b + c)

    mir = _run(f)
    sched = _schedule(mir)
    div = next(
        vid for vid, n in mir.nodes.items() if isinstance(n, MirOperation) and isinstance(n.operator, FDivOperator)
    )
    div_node = mir.nodes[div]
    assert isinstance(div_node, MirOperation)
    div_commit = sched.issue_cycle[div] + div_node.operator.latency
    adds = [vid for vid, n in mir.nodes.items() if isinstance(n, MirOperation) and isinstance(n.operator, FAddOperator)]
    # Some fadd of the independent (a+b+c) chain issues before the divide commits -- genuine overlap, no barrier.
    assert any(sched.issue_cycle[vid] < div_commit for vid in adds)


def _ilog2(mir: Mir) -> list[int]:
    return [
        vid for vid, n in mir.nodes.items() if isinstance(n, MirOperation) and isinstance(n.operator, FMulILog2Operator)
    ]


def test_fmul_ilog2_same_k_shares_one_instance() -> None:
    # Two K=2 scalings that never run on the same cycle (the second waits on a multiply) pool onto one instance.
    def f(a, b):  # type: ignore[no-untyped-def]
        return (a * b) * 4.0, b * 4.0

    mir = _run(f)
    il = _ilog2(mir)
    assert len(il) == 2
    sched = _schedule(mir)
    assert sched.issue_cycle[il[0]] != sched.issue_cycle[il[1]]  # not concurrent
    assert sched.inst_of[il[0]] == sched.inst_of[il[1]]  # ...so they share the one instance
    assert sum(1 for i in sched.instances if isinstance(i.operator, FMulILog2Operator)) == 1


def test_fmul_ilog2_same_k_serializes_by_default_parallelizes_with_budget() -> None:
    # Two independent K=2 scalings are both ready at cycle 1; the per-kind budget governs them like any other kind.
    def f(a, b):  # type: ignore[no-untyped-def]
        return a * 4.0, b * 4.0

    mir = _run(f)
    il = _ilog2(mir)
    assert len(il) == 2

    one = _schedule(mir)  # default budget 1 -> serialize onto a single instance
    assert one.issue_cycle[il[0]] != one.issue_cycle[il[1]]
    assert sum(1 for i in one.instances if isinstance(i.operator, FMulILog2Operator)) == 1


def test_fmul_ilog2_different_k_never_shares() -> None:
    def f(a, b):  # type: ignore[no-untyped-def]
        return a * 4.0 + b * 8.0  # K=2 and K=3 -- distinct hardware modules

    mir = _run(f)
    il = _ilog2(mir)
    assert len(il) == 2
    sched = _schedule(mir)
    assert sched.inst_of[il[0]] != sched.inst_of[il[1]]  # different K -> different instances
    assert {sched.inst_of[v].operator.k for v in il} == {2, 3}
    assert {sched.inst_of[v].index for v in il} == {0}  # indices are local to each concrete operator value


def test_build_lir_small_kernel() -> None:
    def f(a, b):  # type: ignore[no-untyped-def]
        return (a - b) * 0.25 + a * b

    lir = build(_run(f), "kernel")
    assert lir.module_name == "kernel"
    assert lir.float_regfile.fmt == FMT
    assert lir.float_regfile.nreg >= 1
    assert {i.name for i in lir.float_inputs} == {"a", "b"}
    assert lir.float_regfile.nload == 2  # both inputs are preloaded via the regfile load port (registers 0..1)
    assert [o.name for o in lir.float_outputs] == ["out_0"]
    assert all(isinstance(o.tap.source, FloatRegRef) for o in lir.float_outputs)
    assert lir.makespan == max(op.commit_cycle for op in lir.float_ops)

    names = [p.name for p in lir.ports]
    for expected in (
        "clk",
        "rst",
        "in_valid",
        "in_ready",
        "out_valid",
        "out_ready",
        "in_a",
        "in_b",
        "out_0",
        "err_pc",
    ):
        assert expected in names


def test_state_writeback_installs_early_and_is_first_class() -> None:
    from holoso._lir import FETCH_LAG, FloatOperand

    class LeakyDelay:
        def __init__(self) -> None:
            self._p = 0.0

        def __call__(self, x):  # type: ignore[no-untyped-def]
            out = self._p + x  # reads the old _p; the fadd result is the only output
            self._p = x  # a non-coalesced writeback whose source (the input x) is an ordinary register
            return out

    lir = build(_run(LeakyDelay().__call__), "leaky_delay")
    (slot,) = lir.float_state_slots
    assert lir.has_state and slot.needs_copy and isinstance(slot.tap, FloatOperand)
    # The non-coalesced writeback is a first-class event in the liveness model: the slot register holds a live value on
    # its install step (previously absent, which is why the report could not render it).
    assert lir.state_copy_step(slot) in lir.float_liveness[slot.reg]
    assert lir.state_copy_step(slot) == slot.install_cycle + FETCH_LAG + 1
    # Nothing reads _p's register after the old live-in and its source is an ordinary register, so the copy installs
    # before the boundary -- freeing the source register for the rest of the initiation rather than pinning it there.
    assert lir.state_copy_step(slot) < lir.initiation_interval
    # The carried live-out must survive to the boundary even though nothing reads it again this frame, so the slot
    # register stays live from its install step through the boundary -- an early install is not the value's death.
    assert set(range(lir.state_copy_step(slot), lir.initiation_interval + 1)) <= lir.float_liveness[slot.reg]
    # Output wires carry the same FloatOperand tap primitive as state slots.
    assert all(isinstance(w.tap, FloatOperand) for w in lir.float_outputs)
    # The hardware-frame cycle helpers are the single source of truth the report reuses (no duplicated formulas).
    op = lir.float_ops[0]
    assert lir.result_landing_cycle(op) == op.commit_cycle + FETCH_LAG + 2
    assert lir.operand_read_cycle(op) == op.issue_cycle + FETCH_LAG - 1


def test_state_war_backstop_allows_noop_writeback() -> None:
    # A no-op writeback (live-out is the live-in value itself) writes no new value, so the write-after-read backstop
    # must not trip -- this previously aborted a legal build.
    class Hold:
        def __init__(self) -> None:
            self.s = 0.0

        def __call__(self, x):  # type: ignore[no-untyped-def]
            out = self.s + x
            self.s = self.s
            return out

    lir = build(_run(Hold().__call__), "hold")  # must not raise AssertionError
    assert {s.name for s in lir.float_state_slots} == {"s"}


def test_copy_slot_residence_unbroken_when_tapped_at_boundary() -> None:
    # When an output taps a copy slot's register at the boundary, read-first means that read returns the live-in, so the
    # live-in residence must stay continuous through the boundary (no false dead gap from the new boundary def).
    class Delay:
        def __init__(self) -> None:
            self._d = 0.0

        def __call__(self, x):  # type: ignore[no-untyped-def]
            prev = self._d
            self._d = x
            return prev

    lir = build(_run(Delay().__call__), "delay")
    (slot,) = lir.float_state_slots
    assert sorted(lir.float_liveness[slot.reg]) == list(range(1, lir.initiation_interval + 1))


def test_state_early_copy_frees_source_register() -> None:
    # The trapezoidal integrator's update is `_x_prev = in_x`. in_x's only late use is feeding that writeback, so the
    # copy installs in_x into the _x_prev slot register early; in_x's register is then reused by a later operation
    # instead of being pinned to the boundary -- the register-efficiency win this enables.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    from trapezoidal_leaky_streaming_integrator import TrapezoidalLeakyStreamingIntegrator

    lir = build(_run(TrapezoidalLeakyStreamingIntegrator(k=2**-22).__call__), "trapz")
    (xprev,) = [s for s in lir.float_state_slots if s.name == "_x_prev"]
    (in_x,) = lir.float_inputs
    assert xprev.needs_copy and in_x.dst == xprev.tap.source  # the copy's source is the input register
    assert xprev.install_cycle <= lir.makespan  # installs before the boundary (present cycle == makespan + 1)
    # The freed input register is reused: a later operation's result is assigned to it as well.
    assert any(op.dst == in_x.dst for op in lir.float_ops)


def test_build_lir_ekf1() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    import ekf1

    lir = build(_run(ekf1.update_x_P), "update_x_P")
    assert len(lir.float_inputs) == 17
    assert len(lir.float_outputs) == 9
    fdivs = [inst for inst in lir.float_instances if isinstance(inst.operator, FDivOperator)]
    assert len(fdivs) == 1
    # The two K=1 power-of-two scalings are non-concurrent, so they pool onto a single shared instance.
    assert sum(1 for inst in lir.float_instances if isinstance(inst.operator, FMulILog2Operator)) == 1
    # Register reuse: not every distinct value occupies its own register.
    assert lir.float_regfile.nreg < lir.op_count + len(lir.float_inputs)
    # Inputs preload through the regfile's load port (registers 0..nload-1), so nload spans the input block.
    assert lir.float_regfile.nload == 17
    # Dedicated ports: one read port per operator operand (sum of arities = 2+2+1+2), one write port per instance.
    assert lir.float_regfile.nwr == 4
    assert lir.float_regfile.nrd == 7
    # The 1/x21 numerator survives as a constant immediate.
    assert any(abs(c - 1.0) < 1e-12 for c in lir.float_consts)


def test_build_rejects_mir_with_mixed_float_formats() -> None:
    other = FloatFormat(8, 24)
    mir = Mir(
        nodes={
            0: MirFloatInput("a", FloatType(FMT)),
            1: MirFloatOperation(
                FAddOperator(other),
                [0, 0],
                [FloatSignControl(), FloatSignControl()],
                FloatSignControl(),
            ),
        },
        input_ids=[0],
        outputs=[MirFloatOutput("out_0", 1)],
        state_slots=[],
    )
    with pytest.raises(ValueError, match="exactly one floating-point format"):
        build(mir, "mixed")


def test_mir_builder_rejects_mixed_float_operand_formats() -> None:
    other = FloatFormat(8, 24)
    builder = MirBuilder()
    a = builder.float_input("a", FloatType(FMT))
    b = builder.float_input("b", FloatType(other))
    with pytest.raises(ValueError, match="expects operands"):
        builder.float_operation(
            FAddOperator(FMT),
            [a, b],
            [FloatSignControl(), FloatSignControl()],
        )


def test_mir_float_subclasses_validate_float_invariants() -> None:
    with pytest.raises(TypeError, match="scalar_type"):
        MirFloatInput("a", OtherScalarType())
    with pytest.raises(TypeError, match="scalar_type"):
        MirFloatConst(OtherScalarType(), 1.0)
    with pytest.raises(TypeError, match="operator"):
        MirFloatOperation(
            ATestHardwareOperator(),
            [0],
            [FloatSignControl()],
            FloatSignControl(),
        )
    with pytest.raises(ValueError, match="operand"):
        MirFloatOperation(
            FAddOperator(FMT),
            [0],
            [FloatSignControl(), FloatSignControl()],
            FloatSignControl(),
        )
    with pytest.raises(ValueError, match="sign control"):
        MirFloatOperation(
            FAddOperator(FMT),
            [0, 0],
            [FloatSignControl()],
            FloatSignControl(),
        )
    with pytest.raises(TypeError, match="sign"):
        MirFloatOutput("out_0", 0, object())


def test_float_view_rejects_non_float_mir_before_scheduling() -> None:
    mir = Mir(
        nodes={0: OtherMirInput("a", OtherScalarType())},
        input_ids=[0],
        outputs=[MirFloatOutput("out_0", 0)],
        state_slots=[],
    )
    with pytest.raises(UnsupportedConstruct, match="non-float MIR input"):
        MirFloatView.from_mir(mir)


def test_float_view_rejects_non_input_input_id() -> None:
    mir = Mir(
        nodes={0: MirFloatConst(FloatType(FMT), 1.0)},
        input_ids=[0],
        outputs=[MirFloatOutput("out_0", 0)],
        state_slots=[],
    )
    with pytest.raises(ValueError, match="must reference a MirFloatInput"):
        MirFloatView.from_mir(mir)


def test_float_view_rejects_missing_input_id() -> None:
    mir = Mir(
        nodes={0: MirFloatConst(FloatType(FMT), 1.0)},
        input_ids=[1],
        outputs=[MirFloatOutput("out_0", 0)],
        state_slots=[],
    )
    with pytest.raises(ValueError, match="does not reference a MIR node"):
        MirFloatView.from_mir(mir)


def test_fmul_ilog2_operator_rejects_out_of_range_k() -> None:
    limit = (1 << FMT.wexp) - 2
    assert FMulILog2Operator(FMT, k=-limit).k == -limit
    assert FMulILog2Operator(FMT, k=limit - 1).k == limit - 1
    with pytest.raises(ValueError, match="k must satisfy"):
        FMulILog2Operator(FMT, k=limit)
    with pytest.raises(ValueError, match="k must satisfy"):
        FMulILog2Operator(FMT, k=-limit - 1)


def _read_mux_fan_in(lir) -> int:  # type: ignore[no-untyped-def]
    return sum(max(0, len(regs) - 1) for regs in lir.read_set_per_port.values())


def test_marked_commutative_operators_are_bit_exact_commutative() -> None:
    # The port-assignment pass swaps a commutative operator's operands, which is only sound if the operator is
    # exactly symmetric. Guard the FAddOperator/FMulOperator markings against a future non-commutative slip-up.
    import random

    from holoso._value import FloatValue, add_float_values, mul_float_values

    rng = random.Random(0)
    assert FAddOperator(FMT).is_commutative and FMulOperator(FMT).is_commutative
    assert not FDivOperator(FMT).is_commutative
    for evaluate in (add_float_values, mul_float_values):
        for _ in range(5000):
            a = FloatValue.from_float(FMT, rng.uniform(-2.0, 2.0) * 2.0 ** rng.randint(-22, 22))
            b = FloatValue.from_float(FMT, rng.uniform(-2.0, 2.0) * 2.0 ** rng.randint(-22, 22))
            assert evaluate(a, b).bits == evaluate(b, a).bits


def test_commutative_port_assignment_never_increases_read_mux_fan_in(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import holoso._lir._build as build_module

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    import ekf1

    cfg = OpConfig(
        FAddOperator(FMT, stage_decode=1),
        FMulOperator(FMT, stage_input=1),
        FDivOperator(FMT),
        FMulILog2OperatorFamily(FMT),
    )
    monkeypatch.setattr(build_module, "assign_commutative_ports", lambda *args, **kwargs: {})
    baseline = build(_run(ekf1.update_x_P, cfg), "ekf1")
    monkeypatch.undo()
    optimized = build(_run(ekf1.update_x_P, cfg), "ekf1")

    assert _read_mux_fan_in(optimized) <= _read_mux_fan_in(baseline)
    assert _read_mux_fan_in(optimized) < _read_mux_fan_in(baseline)  # ekf1 has commutative reach to reclaim
