"""Unit tests for pipelined scheduling, register allocation, and LIR construction."""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import pytest

from holoso import FAddOperator, FDivOperator, FloatFormat, FMulILog2OperatorFamily, FMulOperator, OpConfig
from holoso._frontend import lower
from holoso._hir import optimize
from holoso._lir import FloatRegRef
from holoso._mir import (
    lower as lower_to_mir,
    Mir,
    MirBuilder,
    MirFloatConst,
    MirFloatInput,
    MirFloatOperation,
    MirFloatOutput,
    MirOperation,
)
from holoso._operators import FMulILog2Operator, FloatSignControl, HardwareOperator
from holoso._lir import build, interface_of
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


def _run(target, ops: OpConfig = OPS) -> Mir:  # type: ignore[no-untyped-def]
    return lower_to_mir(optimize(lower(target)), ops)


def _muls(mir: Mir) -> list[int]:
    return [vid for vid, n in mir.nodes.items() if isinstance(n, MirOperation) and isinstance(n.operator, FMulOperator)]


def test_schedule_respects_dependencies() -> None:
    def f(a, b):  # type: ignore[no-untyped-def]
        return (a - b) * 0.25 + a * b

    mir = _run(f)
    sched = schedule_ops(mir, resolve_pool(mir, None))
    for vid, cycle in sched.issue_cycle.items():
        op = mir.nodes[vid]
        assert isinstance(op, MirOperation)
        assert cycle >= 1  # nothing issues on the accept cycle; inputs are readable from cycle 1
        for operand in op.operands:
            node = mir.nodes[operand]
            if isinstance(node, MirOperation):
                # A consumer issues no earlier than the producer's commit + 1 (read-first writeback latency).
                assert cycle >= sched.issue_cycle[operand] + node.operator.latency + 1


def test_multi_issue_packs_independent_ops() -> None:
    def f(a, b, c):  # type: ignore[no-untyped-def]
        return a * b + b * c

    mir = _run(f)
    muls = _muls(mir)
    assert len(muls) == 2

    two = schedule_ops(mir, resolve_pool(mir, {FMulOperator: 2}))
    assert two.issue_cycle[muls[0]] == two.issue_cycle[muls[1]]  # two instances -> both multiplies issue together
    assert two.inst_of[muls[0]].index != two.inst_of[muls[1]].index  # ...on distinct instances

    one = schedule_ops(mir, resolve_pool(mir, {FMulOperator: 1}))
    assert one.issue_cycle[muls[0]] != one.issue_cycle[muls[1]]  # one instance forces them onto consecutive cycles


def test_pipelined_issue_overlaps_a_slow_op() -> None:
    # A fast chain advances while an unrelated slow divide is still in flight -- the barrier model could not do this.
    def f(a, b, c):  # type: ignore[no-untyped-def]
        return a / b + (a + b + c)

    mir = _run(f)
    sched = schedule_ops(mir, resolve_pool(mir, None))
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
    sched = schedule_ops(mir, resolve_pool(mir, None))
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

    one = schedule_ops(mir, resolve_pool(mir, None))  # default budget 1 -> serialize onto a single instance
    assert one.issue_cycle[il[0]] != one.issue_cycle[il[1]]
    assert sum(1 for i in one.instances if isinstance(i.operator, FMulILog2Operator)) == 1

    two = schedule_ops(mir, resolve_pool(mir, {FMulILog2Operator: 2}))  # budget 2 -> co-issue on two instances
    assert two.issue_cycle[il[0]] == two.issue_cycle[il[1]]
    assert two.inst_of[il[0]].index != two.inst_of[il[1]].index


def test_fmul_ilog2_different_k_never_shares() -> None:
    def f(a, b):  # type: ignore[no-untyped-def]
        return a * 4.0 + b * 8.0  # K=2 and K=3 -- distinct hardware modules

    mir = _run(f)
    il = _ilog2(mir)
    assert len(il) == 2
    sched = schedule_ops(mir, resolve_pool(mir, None))
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
    assert all(isinstance(o.source, FloatRegRef) for o in lir.float_outputs)
    assert lir.makespan == max(op.commit_cycle for op in lir.float_ops)

    iface = interface_of(lir)
    names = [p.name for p in iface.ports]
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
        "err_cyc",
    ):
        assert expected in names


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
    # Port counts track internal parallelism, not I/O width.
    assert lir.float_regfile.nwr == 3
    assert lir.float_regfile.nrd == 5
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


def test_fmul_ilog2_operator_rejects_out_of_range_k() -> None:
    with pytest.raises(ValueError, match="k must satisfy"):
        FMulILog2Operator(FMT, k=1 << (FMT.wexp - 1))
