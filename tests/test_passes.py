"""Unit tests for the HIR optimization/lowering passes."""

import sys
from pathlib import Path

from holoso import FAddOp, FDivOp, FloatFormat, FMulILog2GenericOp, FMulOp, OpConfig
from holoso._frontend import lower
from holoso._hir import Const, Hir, InPort, OpNode
from holoso._operators import FMulILog2Op, Sgnop
from holoso._passes import run

FMT = FloatFormat(6, 18)
OPS = OpConfig(FAddOp(), FMulOp(), FDivOp(), FMulILog2GenericOp())


def _op_count(hir: Hir, cls: type) -> int:
    return sum(1 for n in hir.nodes.values() if isinstance(n, OpNode) and isinstance(n.op, cls))


def _ops(hir: Hir) -> list[OpNode]:
    return [n for n in hir.nodes.values() if isinstance(n, OpNode)]


def _consts(hir: Hir) -> list[float]:
    return [n.value for n in hir.nodes.values() if isinstance(n, Const)]


def test_mul_by_pow2_const_becomes_ilog2() -> None:
    def f(a):  # type: ignore[no-untyped-def]
        return a * 0.25

    ops = _ops(run(lower(f, FMT), OPS))
    assert len(ops) == 1
    assert isinstance(ops[0].op, FMulILog2Op) and ops[0].op.k == -2


def test_left_const_mul_pow2_is_commutative() -> None:
    def f(a):  # type: ignore[no-untyped-def]
        return 2 * a

    ops = _ops(run(lower(f, FMT), OPS))
    assert len(ops) == 1
    assert isinstance(ops[0].op, FMulILog2Op) and ops[0].op.k == 1


def test_div_by_pow2_becomes_ilog2() -> None:
    def f(a):  # type: ignore[no-untyped-def]
        return a / 4.0

    ops = _ops(run(lower(f, FMT), OPS))
    assert len(ops) == 1
    assert isinstance(ops[0].op, FMulILog2Op) and ops[0].op.k == -2


def test_div_by_nonpow2_const_becomes_reciprocal_multiply() -> None:
    def f(a):  # type: ignore[no-untyped-def]
        return a / 3.0

    hir = run(lower(f, FMT), OPS)
    ops = _ops(hir)
    assert [type(o.op) for o in ops] == [FMulOp]
    assert any(abs(c - 1.0 / 3.0) < 1e-12 for c in _consts(hir))


def test_true_division_stays_fdiv() -> None:
    def f(a, b):  # type: ignore[no-untyped-def]
        return a / b

    assert [type(o.op) for o in _ops(run(lower(f, FMT), OPS))] == [FDivOp]


def test_subtraction_folds_into_b_sgnop_neg() -> None:
    def f(a, b):  # type: ignore[no-untyped-def]
        return a - b

    ops = _ops(run(lower(f, FMT), OPS))
    assert len(ops) == 1
    assert isinstance(ops[0].op, FAddOp) and ops[0].b_sgnop is Sgnop.NEG


def test_operand_negation_folds_into_operator() -> None:
    def f(a, b):  # type: ignore[no-untyped-def]
        return a * (-b)

    ops = _ops(run(lower(f, FMT), OPS))
    assert len(ops) == 1
    assert isinstance(ops[0].op, FMulOp) and ops[0].b_sgnop is Sgnop.NEG


def test_lowered_hir_has_only_inport_const_opnode() -> None:
    def f(a, b):  # type: ignore[no-untyped-def]
        return (a - b) * 0.25 + a * b

    hir = run(lower(f, FMT), OPS)
    assert all(isinstance(n, (InPort, Const, OpNode)) for n in hir.nodes.values())


def test_ekf1_lowering() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    import ekf1

    hir = run(lower(ekf1.update_x_P, FMT), OPS)
    assert all(isinstance(n, (InPort, Const, OpNode)) for n in hir.nodes.values())
    assert _op_count(hir, FDivOp) == 1  # only x22 = 1 / x21
    assert _op_count(hir, FMulILog2Op) >= 1  # the "2 * ..." terms
    assert len(hir.input_ids) == 17
    assert len(hir.outputs) == 9
