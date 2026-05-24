"""Unit tests for pipelined scheduling, register allocation, and LIR construction."""

from __future__ import annotations

import sys
from pathlib import Path

from holoso.format import FloatFormat
from holoso.frontend import lower
from holoso.hir import OpNode
from holoso.lir import RegRef
from holoso.operators import OpKind
from holoso.passes import run
from holoso.schedule import build, cycle_count, interface_of, metrics_of
from holoso.scheduler import resolve_pool, schedule_ops

FMT = FloatFormat(6, 18)


def _muls(hir):  # type: ignore[no-untyped-def]
    return [vid for vid, n in hir.nodes.items() if isinstance(n, OpNode) and n.kind is OpKind.FMUL]


def test_schedule_respects_dependencies() -> None:
    def f(a, b):  # type: ignore[no-untyped-def]
        return (a - b) * 0.25 + a * b

    hir = run(lower(f, FMT))
    sched = schedule_ops(hir, resolve_pool(hir, None))
    for vid, cycle in sched.issue_cycle.items():
        op = hir.nodes[vid]
        assert isinstance(op, OpNode)
        assert cycle >= 1  # nothing issues on the accept cycle; inputs are readable from cycle 1
        for operand in (op.a, op.b):
            node = hir.nodes[operand] if operand is not None else None
            if isinstance(node, OpNode):
                # A consumer issues no earlier than the producer's commit + 1 (read-first writeback latency).
                assert cycle >= sched.issue_cycle[operand] + node.latency + 1


def test_multi_issue_packs_independent_ops() -> None:
    def f(a, b, c):  # type: ignore[no-untyped-def]
        return a * b + b * c

    hir = run(lower(f, FMT))
    muls = _muls(hir)
    assert len(muls) == 2

    two = schedule_ops(hir, resolve_pool(hir, {OpKind.FMUL: 2}))
    assert two.issue_cycle[muls[0]] == two.issue_cycle[muls[1]]  # two instances -> both multiplies issue together
    assert two.inst_of[muls[0]].index != two.inst_of[muls[1]].index  # ...on distinct instances

    one = schedule_ops(hir, resolve_pool(hir, {OpKind.FMUL: 1}))
    assert one.issue_cycle[muls[0]] != one.issue_cycle[muls[1]]  # one instance forces them onto consecutive cycles


def test_pipelined_issue_overlaps_a_slow_op() -> None:
    # A fast chain advances while an unrelated slow divide is still in flight -- the barrier model could not do this.
    def f(a, b, c):  # type: ignore[no-untyped-def]
        return a / b + (a + b + c)

    hir = run(lower(f, FMT))
    sched = schedule_ops(hir, resolve_pool(hir, None))
    div = next(vid for vid, n in hir.nodes.items() if isinstance(n, OpNode) and n.kind is OpKind.FDIV)
    div_commit = sched.issue_cycle[div] + hir.nodes[div].latency  # type: ignore[union-attr]
    adds = [vid for vid, n in hir.nodes.items() if isinstance(n, OpNode) and n.kind is OpKind.FADD]
    # Some fadd of the independent (a+b+c) chain issues before the divide commits -- genuine overlap, no barrier.
    assert any(sched.issue_cycle[vid] < div_commit for vid in adds)


def test_fmul_ilog2_instances_are_dedicated_and_unlimited() -> None:
    # Power-of-two scalings get one dedicated instance each and may co-issue freely (never pool-capped).
    def f(a, b):  # type: ignore[no-untyped-def]
        return a * 4.0 + b * 8.0  # two independent fmul_ilog2_const

    hir = run(lower(f, FMT))
    ilog2 = [vid for vid, n in hir.nodes.items() if isinstance(n, OpNode) and n.kind is OpKind.FMUL_ILOG2]
    assert len(ilog2) == 2
    sched = schedule_ops(hir, resolve_pool(hir, None))
    assert sched.issue_cycle[ilog2[0]] == sched.issue_cycle[ilog2[1]]  # co-issue despite the default pool of 1
    assert sched.inst_of[ilog2[0]].index != sched.inst_of[ilog2[1]].index  # ...on dedicated instances


def test_build_lir_small_kernel() -> None:
    def f(a, b):  # type: ignore[no-untyped-def]
        return (a - b) * 0.25 + a * b

    lir = build(run(lower(f, FMT)), "kernel")
    assert lir.module_name == "kernel"
    assert lir.regfile.nreg >= 1
    assert {i.name for i in lir.inputs} == {"a", "b"}
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
        "diag_error",
    ):
        assert expected in names
    assert iface.ii.cycle_estimate == lir.makespan + 1 == cycle_count(lir)

    metrics = metrics_of(lir)
    assert metrics.makespan == lir.makespan
    assert metrics.n_float_regs == lir.regfile.nreg


def test_build_lir_ekf1() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    import ekf1

    lir = build(run(lower(ekf1.update_x_P, FMT)), "update_x_P")
    assert len(lir.inputs) == 17
    assert len(lir.outputs) == 9
    fdivs = [inst for inst in lir.instances if inst.kind is OpKind.FDIV]
    assert len(fdivs) == 1
    # Register reuse: not every distinct value occupies its own register.
    assert lir.regfile.nreg < lir.op_count + len(lir.inputs)
    # The 1/x21 numerator survives as a constant immediate.
    assert any(abs(c - 1.0) < 1e-12 for c in lir.consts)

    metrics = metrics_of(lir)
    assert metrics.operator_instances.get("fdiv") == 1
    assert metrics.op_count == lir.op_count
    assert metrics.max_chain_len >= 1
