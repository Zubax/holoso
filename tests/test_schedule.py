"""Unit tests for scheduling, register allocation, and LIR construction."""

from __future__ import annotations

import sys
from pathlib import Path

from holoso.format import FloatFormat
from holoso.frontend import lower
from holoso.hir import OpNode
from holoso.lir import RegRef
from holoso.operators import OpKind
from holoso.passes import run
from holoso.schedule import build, interface_of, metrics_of
from holoso.scheduler import resolve_pool, schedule_ops

FMT = FloatFormat(6, 18)


def _muls(hir):  # type: ignore[no-untyped-def]
    return [vid for vid, n in hir.nodes.items() if isinstance(n, OpNode) and n.kind is OpKind.FMUL]


def test_schedule_respects_dependencies() -> None:
    def f(a, b):  # type: ignore[no-untyped-def]
        return (a - b) * 0.25 + a * b

    hir = run(lower(f, FMT))
    steps = schedule_ops(hir, resolve_pool(hir, None))
    step_of = {vid: k for k, issue in enumerate(steps) for vid in issue}
    for k, issue in enumerate(steps):
        for vid in issue:
            op = hir.nodes[vid]
            assert isinstance(op, OpNode)
            for operand in (op.a, op.b):
                if operand is not None and isinstance(hir.nodes[operand], OpNode):
                    assert step_of[operand] < k


def test_multi_issue_packs_independent_ops() -> None:
    def f(a, b, c):  # type: ignore[no-untyped-def]
        return a * b + b * c

    hir = run(lower(f, FMT))
    muls = _muls(hir)
    assert len(muls) == 2

    two = schedule_ops(hir, resolve_pool(hir, {OpKind.FMUL: 2}))
    assert all(m in two[0] for m in muls)  # both multiplies issue together

    one = schedule_ops(hir, resolve_pool(hir, {OpKind.FMUL: 1}))
    assert not all(m in one[0] for m in muls)  # one instance forces serialization


def test_build_lir_small_kernel() -> None:
    def f(a, b):  # type: ignore[no-untyped-def]
        return (a - b) * 0.25 + a * b

    lir = build(run(lower(f, FMT)), "kernel")
    assert lir.module_name == "kernel"
    assert lir.regfile.nreg >= 1
    assert {i.name for i in lir.inputs} == {"a", "b"}
    assert [o.name for o in lir.outputs] == ["out_0"]
    assert all(isinstance(o.source, RegRef) for o in lir.outputs)

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
    assert iface.ii.cycle_estimate > 0

    metrics = metrics_of(lir)
    assert metrics.step_count == len(lir.steps)
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
