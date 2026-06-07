"""Unit tests for HIR optimization and MIR selection passes."""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import pytest

from holoso import (
    FAddOperator,
    FCmpOperator,
    FDivOperator,
    FloatFormat,
    FMulILog2OperatorFamily,
    FMulOperator,
    OpConfig,
)
from holoso._errors import UnsupportedConstruct
from holoso._frontend import lower
from holoso._hir import (
    Const,
    FloatAdd,
    FloatConst,
    FloatType as HirFloatType,
    HirBuilder,
    InPort,
    Operation,
    Operator,
    Signature,
    Type,
    optimize,
)
from holoso._hir._const_fold import run as fold_constants
from holoso._lir import build
from holoso._mir import lower as lower_to_mir, Mir, MirFloatConst, MirFloatInput, MirOperation
from holoso._operators import FMulILog2Operator, FloatSignControl
from holoso._backend.numerical import generate as build_model

FMT = FloatFormat(6, 18)
OPS = OpConfig(FAddOperator(FMT), FMulOperator(FMT), FDivOperator(FMT), FMulILog2OperatorFamily(FMT), FCmpOperator(FMT))


@dataclass(frozen=True, slots=True)
class OtherType(Type):
    pass


@dataclass(frozen=True, slots=True)
class OtherConst(Const):
    value: int

    @property
    def type(self) -> OtherType:
        return OtherType()


@dataclass(frozen=True, slots=True)
class OtherFold(Operator):
    mnemonic: ClassVar[str] = "other_fold"

    @property
    def signature(self) -> Signature:
        ty = OtherType()
        return Signature((ty,), ty)

    def fold_constants(self, operands: list[Const]) -> Const:
        (operand,) = operands
        assert isinstance(operand, OtherConst)
        return OtherConst(operand.value + 1)

    def render(self, *operands: str) -> str:
        (operand,) = operands
        return f"other_fold({operand})"


def _run(target, ops: OpConfig = OPS) -> Mir:  # type: ignore[no-untyped-def]
    return lower_to_mir(optimize(lower(target)), ops)


def _op_count(mir: Mir, cls: type) -> int:
    return sum(1 for n in mir.nodes.values() if isinstance(n, MirOperation) and isinstance(n.operator, cls))


def _ops(mir: Mir) -> list[MirOperation]:
    return [n for n in mir.nodes.values() if isinstance(n, MirOperation)]


def _consts(mir: Mir) -> list[float]:
    return [n.value for n in mir.nodes.values() if isinstance(n, MirFloatConst)]


def test_hir_nodes_carry_float_type() -> None:
    builder = HirBuilder()
    builder.block()
    a = builder.float_input("a")
    one = builder.float_const(1.0)
    y = builder.operation(FloatAdd(), [a, one])
    builder.output("out_0", y)
    builder.ret()
    hir = builder.finish()

    input_node = hir.nodes[a]
    op_node = hir.nodes[y]
    assert isinstance(input_node, InPort)
    assert isinstance(op_node, Operation)
    assert input_node.type == HirFloatType()
    assert op_node.type == HirFloatType()


def test_hir_builder_rejects_wrong_semantic_operand_type() -> None:
    builder = HirBuilder()
    builder.block()
    a = builder.input("a", OtherType())
    b = builder.float_input("b")
    try:
        builder.operation(FloatAdd(), [a, b])
    except ValueError as ex:
        assert "expects operands" in str(ex)
    else:
        raise AssertionError("expected a semantic type mismatch")


def test_lower_rejects_non_float_hir_input_type() -> None:
    builder = HirBuilder()
    builder.block()
    a = builder.input("a", OtherType())
    builder.output("out_0", a)
    builder.ret()
    hir = builder.finish()

    try:
        lower_to_mir(hir, OPS)
    except UnsupportedConstruct as ex:
        assert "no MIR lowering rule" in str(ex)
    else:
        raise AssertionError("expected HIR-to-MIR lowering to reject non-float semantic input")


def test_hir_constant_folding_returns_float_const() -> None:
    def f():  # type: ignore[no-untyped-def]
        return 1.25 + 2.0

    hir = optimize(lower(f))
    node = hir.nodes[hir.outputs[0].value]
    assert isinstance(node, FloatConst)
    assert node.value == 3.25


def test_hir_constant_folding_preserves_const_subclass() -> None:
    builder = HirBuilder()
    builder.block()
    x = builder.const_node(OtherConst(10))
    y = builder.operation(OtherFold(), [x])
    builder.output("out_0", y)
    builder.ret()

    hir = fold_constants(builder.finish())
    node = hir.nodes[hir.outputs[0].value]
    assert isinstance(node, OtherConst)
    assert node.value == 11


def test_mir_constant_only_node_carries_float_type() -> None:
    def f():  # type: ignore[no-untyped-def]
        return 3.5

    mir = _run(f)
    const = mir.nodes[mir.outputs[0].value]
    assert isinstance(const, MirFloatConst)
    assert const.scalar_type.fmt == FMT


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


def test_wide_supported_pow2_uses_ilog2_operator() -> None:
    def f(a):  # type: ignore[no-untyped-def]
        return a * 16.0

    fmt = FloatFormat(3, 4)
    ops = OpConfig(
        FAddOperator(fmt), FMulOperator(fmt), FDivOperator(fmt), FMulILog2OperatorFamily(fmt), FCmpOperator(fmt)
    )
    mir = _run(f, ops)
    selected = _ops(mir)
    assert [type(o.operator) for o in selected] == [FMulILog2Operator]
    assert selected[0].operator.k == 4
    assert _consts(mir) == []


def test_unsupported_pow2_shift_is_rejected() -> None:
    def f(a):  # type: ignore[no-untyped-def]
        return a * 64.0

    fmt = FloatFormat(3, 4)
    ops = OpConfig(
        FAddOperator(fmt), FMulOperator(fmt), FDivOperator(fmt), FMulILog2OperatorFamily(fmt), FCmpOperator(fmt)
    )
    try:
        _run(f, ops)
    except UnsupportedConstruct as ex:
        assert "unsupported power-of-two float scale" in str(ex)
    else:
        raise AssertionError("expected an unsupported power-of-two shift")


def test_true_division_stays_fdiv() -> None:
    def f(a, b):  # type: ignore[no-untyped-def]
        return a / b

    assert [type(o.operator) for o in _ops(_run(f))] == [FDivOperator]


def test_subtraction_folds_into_second_operand_sign() -> None:
    def f(a, b):  # type: ignore[no-untyped-def]
        return a - b

    ops = _ops(_run(f))
    assert len(ops) == 1
    assert isinstance(ops[0].operator, FAddOperator) and ops[0].operand_signs[1] == FloatSignControl(negate=True)


def test_operand_negation_folds_into_operator() -> None:
    def f(a, b):  # type: ignore[no-untyped-def]
        return a * (-b)

    ops = _ops(_run(f))
    assert len(ops) == 1
    assert isinstance(ops[0].operator, FMulOperator) and ops[0].operand_signs[1] == FloatSignControl(negate=True)


def test_pure_sign_output_adds_no_operation() -> None:
    def f(a):  # type: ignore[no-untyped-def]
        return -abs(a)

    mir = _run(f)
    assert _ops(mir) == []
    assert mir.outputs[0].sign == FloatSignControl(absolute=True).then(FloatSignControl(negate=True))


def test_selected_mir_has_only_input_const_operation_nodes() -> None:
    def f(a, b):  # type: ignore[no-untyped-def]
        return (a - b) * 0.25 + a * b

    mir = _run(f)
    assert all(isinstance(n, (MirFloatInput, MirFloatConst, MirOperation)) for n in mir.nodes.values())


def test_ekf1_stateless_lowering() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    import ekf1_stateless

    mir = _run(ekf1_stateless.update_x_P)
    assert all(isinstance(n, (MirFloatInput, MirFloatConst, MirOperation)) for n in mir.nodes.values())
    assert _op_count(mir, FDivOperator) == 1  # only x22 = 1 / x21
    assert _op_count(mir, FMulILog2Operator) >= 1  # the "2 * ..." terms
    assert len(mir.input_ids) == 17
    assert len(mir.outputs) == 9


def test_unclosed_loop_phi_is_rejected() -> None:
    # HirBuilder.finish validates that every phi has one arm per CFG predecessor: a loop-header phi opened (open_phi)
    # but never closed (its back-edge arm missing) is a construction bug and must be caught, not emitted malformed.
    builder = HirBuilder()
    entry = builder.block()
    header = builder.block()
    x = builder.float_input("x")
    builder.position_at(entry)
    builder.jump(header)
    builder.position_at(header)
    builder.open_phi(HirFloatType(), (entry, x))  # only the preheader arm; the latch arm is never supplied
    builder.jump(header)  # back-edge: the header now has two predecessors (entry, header) but the phi carries one arm
    with pytest.raises(RuntimeError, match="predecessor"):
        builder.finish()


def _deep_cfg_kernel(p0):  # type: ignore[no-untyped-def]
    # A doubly-nested unrolled loop (each trip count well under the unroll threshold) with a per-iteration branch.
    # Unrolling chains ~2700 basic blocks, so the CFG reverse-postorder DFS recurses far deeper than Python's default
    # recursion limit. The accumulator stays >=0, so every iteration takes the add arm: the result is the input + 900.
    acc = p0
    for _i in range(30):
        for _j in range(30):
            if acc > 0.0:
                acc = acc + 1.0
            else:
                acc = acc - 1.0
    return acc


def test_deep_cfg_does_not_overflow_recursion() -> None:
    # Regression: the HIR/MIR/LIR reverse-postorder traversals walked the block CFG recursively, so a deep CFG -- here
    # nested unrolled loops chaining thousands of blocks -- overflowed Python's recursion limit with a RecursionError
    # in _copy.reverse_postorder (and the symmetric _lir._build._mir_rpo). With recursion in place, optimize() raises;
    # the iterative DFS compiles cleanly. Exercise the whole front-to-back pipeline (optimize, MIR lowering, LIR build)
    # since each contains a CFG DFS, and check the bit-exact model against the plain-Python reference.
    hir = lower(_deep_cfg_kernel)
    assert len(hir.blocks) > 1000  # the CFG is genuinely deep (otherwise the regression would not bite)
    model = build_model(build(_run(_deep_cfg_kernel), "deep"))
    for x in (0.5, 2.0, 8.0):  # acc stays positive -> +900 every time; 0.5/2.0/8.0 are exact in ZKF
        assert float(model(x)[0]) == _deep_cfg_kernel(x)
