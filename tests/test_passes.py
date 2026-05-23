"""Unit tests for the HIR optimization/lowering passes."""

from __future__ import annotations

import sys
from pathlib import Path

from holoso.format import FloatFormat
from holoso.frontend import lower
from holoso.hir import Const, Hir, InPort, OpNode
from holoso.operators import SGNOP_NEG, OpKind
from holoso.passes import run

FMT = FloatFormat(6, 18)


def _ops(hir: Hir) -> list[OpNode]:
    return [n for n in hir.nodes.values() if isinstance(n, OpNode)]


def _consts(hir: Hir) -> list[float]:
    return [n.value for n in hir.nodes.values() if isinstance(n, Const)]


def test_mul_by_pow2_const_becomes_ilog2() -> None:
    def f(a):  # type: ignore[no-untyped-def]
        return a * 0.25

    ops = _ops(run(lower(f, FMT)))
    assert len(ops) == 1
    assert ops[0].kind is OpKind.FMUL_ILOG2 and ops[0].k == -2


def test_left_const_mul_pow2_is_commutative() -> None:
    def f(a):  # type: ignore[no-untyped-def]
        return 2 * a

    ops = _ops(run(lower(f, FMT)))
    assert len(ops) == 1
    assert ops[0].kind is OpKind.FMUL_ILOG2 and ops[0].k == 1


def test_div_by_pow2_becomes_ilog2() -> None:
    def f(a):  # type: ignore[no-untyped-def]
        return a / 4.0

    ops = _ops(run(lower(f, FMT)))
    assert len(ops) == 1
    assert ops[0].kind is OpKind.FMUL_ILOG2 and ops[0].k == -2


def test_div_by_nonpow2_const_becomes_reciprocal_multiply() -> None:
    def f(a):  # type: ignore[no-untyped-def]
        return a / 3.0

    hir = run(lower(f, FMT))
    ops = _ops(hir)
    assert [o.kind for o in ops] == [OpKind.FMUL]
    assert any(abs(c - 1.0 / 3.0) < 1e-12 for c in _consts(hir))


def test_true_division_stays_fdiv() -> None:
    def f(a, b):  # type: ignore[no-untyped-def]
        return a / b

    assert [o.kind for o in _ops(run(lower(f, FMT)))] == [OpKind.FDIV]


def test_subtraction_folds_into_b_sgnop_neg() -> None:
    def f(a, b):  # type: ignore[no-untyped-def]
        return a - b

    ops = _ops(run(lower(f, FMT)))
    assert len(ops) == 1
    assert ops[0].kind is OpKind.FADD and ops[0].b_sgnop == SGNOP_NEG


def test_operand_negation_folds_into_operator() -> None:
    def f(a, b):  # type: ignore[no-untyped-def]
        return a * (-b)

    ops = _ops(run(lower(f, FMT)))
    assert len(ops) == 1
    assert ops[0].kind is OpKind.FMUL and ops[0].b_sgnop == SGNOP_NEG


def test_lowered_hir_has_only_inport_const_opnode() -> None:
    def f(a, b):  # type: ignore[no-untyped-def]
        return (a - b) * 0.25 + a * b

    hir = run(lower(f, FMT))
    assert all(isinstance(n, (InPort, Const, OpNode)) for n in hir.nodes.values())


def test_ekf1_lowering() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    import ekf1

    hir = run(lower(ekf1.update_x_P, FMT))
    assert all(isinstance(n, (InPort, Const, OpNode)) for n in hir.nodes.values())
    assert hir.op_count(OpKind.FDIV) == 1  # only x22 = 1 / x21
    assert hir.op_count(OpKind.FMUL_ILOG2) >= 1  # the "2 * ..." terms
    assert len(hir.input_ids) == 17
    assert len(hir.outputs) == 9
