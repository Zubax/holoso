"""Unit tests for pipelined scheduling, register allocation, and LIR construction."""

import sys
from pathlib import Path

from holoso import FAddOperator, FDivOperator, FloatFormat, FMulILog2OperatorFamily, FMulOperator, OpConfig
from holoso._frontend import lower
from holoso._hir import optimize
from holoso._lir import RegRef
from holoso._lower import lower as lower_to_mir
from holoso._mir import Mir, MirOperation
from holoso._operators import FMulILog2Operator
from holoso._schedule import build, interface_of
from holoso._scheduler import resolve_pool, schedule_ops

FMT = FloatFormat(6, 18)
OPS = OpConfig(FAddOperator(FMT), FMulOperator(FMT), FDivOperator(FMT), FMulILog2OperatorFamily(FMT))


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


def test_build_lir_small_kernel() -> None:
    def f(a, b):  # type: ignore[no-untyped-def]
        return (a - b) * 0.25 + a * b

    lir = build(_run(f), "kernel", fmt=FMT)
    assert lir.module_name == "kernel"
    assert lir.regfile.nreg >= 1
    assert {i.name for i in lir.inputs} == {"a", "b"}
    assert lir.regfile.nload == 2  # both inputs are preloaded via the regfile load port (registers 0..1)
    assert [o.name for o in lir.outputs] == ["out_0"]
    assert all(isinstance(o.source, RegRef) for o in lir.outputs)
    assert lir.makespan == max(op.commit_cycle for op in lir.ops)

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

    lir = build(_run(ekf1.update_x_P), "update_x_P", fmt=FMT)
    assert len(lir.inputs) == 17
    assert len(lir.outputs) == 9
    fdivs = [inst for inst in lir.instances if isinstance(inst.operator, FDivOperator)]
    assert len(fdivs) == 1
    # The two K=1 power-of-two scalings are non-concurrent, so they pool onto a single shared instance.
    assert sum(1 for inst in lir.instances if isinstance(inst.operator, FMulILog2Operator)) == 1
    # Register reuse: not every distinct value occupies its own register.
    assert lir.regfile.nreg < lir.op_count + len(lir.inputs)
    # Inputs preload through the regfile's load port (registers 0..nload-1), so nload spans the input block.
    assert lir.regfile.nload == 17
    # Port counts track internal parallelism, not I/O width.
    assert lir.regfile.nwr == 3
    assert lir.regfile.nrd == 5
    # The 1/x21 numerator survives as a constant immediate.
    assert any(abs(c - 1.0) < 1e-12 for c in lir.consts)
