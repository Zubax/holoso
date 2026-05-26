"""Unit tests for HIR optimization and MIR selection passes."""

import sys
from pathlib import Path

from holoso import FAddOperator, FDivOperator, FloatFormat, FMulILog2OperatorFamily, FMulOperator, OpConfig
from holoso._frontend import lower
from holoso._hir import optimize
from holoso._lower import lower as lower_to_mir
from holoso._mir import Mir, MirConst, MirInput, MirOperation
from holoso._operators import FMulILog2Operator, SignControl

FMT = FloatFormat(6, 18)
OPS = OpConfig(FAddOperator(FMT), FMulOperator(FMT), FDivOperator(FMT), FMulILog2OperatorFamily(FMT))


def _run(target, ops: OpConfig = OPS) -> Mir:  # type: ignore[no-untyped-def]
    return lower_to_mir(optimize(lower(target)), ops)


def _op_count(mir: Mir, cls: type) -> int:
    return sum(1 for n in mir.nodes.values() if isinstance(n, MirOperation) and isinstance(n.operator, cls))


def _ops(mir: Mir) -> list[MirOperation]:
    return [n for n in mir.nodes.values() if isinstance(n, MirOperation)]


def _consts(mir: Mir) -> list[float]:
    return [n.value for n in mir.nodes.values() if isinstance(n, MirConst)]


def test_mul_by_pow2_const_becomes_ilog2() -> None:
    def f(a):  # type: ignore[no-untyped-def]
        return a * 0.25

    ops = _ops(_run(f))
    assert len(ops) == 1
    assert isinstance(ops[0].operator, FMulILog2Operator) and ops[0].operator.k == -2


def test_left_const_mul_pow2_is_commutative() -> None:
    def f(a):  # type: ignore[no-untyped-def]
        return 2 * a

    ops = _ops(_run(f))
    assert len(ops) == 1
    assert isinstance(ops[0].operator, FMulILog2Operator) and ops[0].operator.k == 1


def test_div_by_pow2_becomes_ilog2() -> None:
    def f(a):  # type: ignore[no-untyped-def]
        return a / 4.0

    ops = _ops(_run(f))
    assert len(ops) == 1
    assert isinstance(ops[0].operator, FMulILog2Operator) and ops[0].operator.k == -2


def test_div_by_nonpow2_const_becomes_reciprocal_multiply() -> None:
    def f(a):  # type: ignore[no-untyped-def]
        return a / 3.0

    mir = _run(f)
    ops = _ops(mir)
    assert [type(o.operator) for o in ops] == [FMulOperator]
    assert any(abs(c - 1.0 / 3.0) < 1e-12 for c in _consts(mir))


def test_infeasible_pow2_falls_back_to_multiply() -> None:
    def f(a):  # type: ignore[no-untyped-def]
        return a * 16.0

    fmt = FloatFormat(3, 4)
    ops = OpConfig(FAddOperator(fmt), FMulOperator(fmt), FDivOperator(fmt), FMulILog2OperatorFamily(fmt))
    mir = _run(f, ops)
    selected = _ops(mir)
    assert [type(o.operator) for o in selected] == [FMulOperator]
    assert any(c == 16.0 for c in _consts(mir))


def test_true_division_stays_fdiv() -> None:
    def f(a, b):  # type: ignore[no-untyped-def]
        return a / b

    assert [type(o.operator) for o in _ops(_run(f))] == [FDivOperator]


def test_subtraction_folds_into_second_operand_sign() -> None:
    def f(a, b):  # type: ignore[no-untyped-def]
        return a - b

    ops = _ops(_run(f))
    assert len(ops) == 1
    assert isinstance(ops[0].operator, FAddOperator) and ops[0].operand_signs[1] == SignControl(negate=True)


def test_operand_negation_folds_into_operator() -> None:
    def f(a, b):  # type: ignore[no-untyped-def]
        return a * (-b)

    ops = _ops(_run(f))
    assert len(ops) == 1
    assert isinstance(ops[0].operator, FMulOperator) and ops[0].operand_signs[1] == SignControl(negate=True)


def test_pure_sign_output_adds_no_operation() -> None:
    def f(a):  # type: ignore[no-untyped-def]
        return -abs(a)

    mir = _run(f)
    assert _ops(mir) == []
    assert mir.outputs[0].sign == SignControl(absolute=True).then(SignControl(negate=True))


def test_selected_mir_has_only_input_const_operation_nodes() -> None:
    def f(a, b):  # type: ignore[no-untyped-def]
        return (a - b) * 0.25 + a * b

    mir = _run(f)
    assert all(isinstance(n, (MirInput, MirConst, MirOperation)) for n in mir.nodes.values())


def test_ekf1_lowering() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    import ekf1

    mir = _run(ekf1.update_x_P)
    assert all(isinstance(n, (MirInput, MirConst, MirOperation)) for n in mir.nodes.values())
    assert _op_count(mir, FDivOperator) == 1  # only x22 = 1 / x21
    assert _op_count(mir, FMulILog2Operator) >= 1  # the "2 * ..." terms
    assert len(mir.input_ids) == 17
    assert len(mir.outputs) == 9
