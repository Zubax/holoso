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
    BoolAnd,
    BoolConst,
    BoolOr,
    BoolType,
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
from holoso._hir import BoolSelect, FloatDiv as HirFloatDiv, Phi, Select
from holoso._hir import _if_convert as if_convert_pass
from holoso._hir._const_fold import run as fold_constants
from holoso._lir import build
from holoso._mir import lower as lower_to_mir, Mir, MirFloatConst, MirFloatInput, MirOperation
from holoso._operators import FMulILog2Operator, FloatSignControl
from ._importguard import forbidden_imports
from ._modelref import build_model

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
        assert False, "expected a semantic type mismatch"


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
        assert False, "expected HIR-to-MIR lowering to reject non-float semantic input"


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
        assert False, "expected an unsupported power-of-two shift"


def test_true_division_stays_fdiv() -> None:
    def f(a, b):  # type: ignore[no-untyped-def]
        return a / b

    assert [type(o.operator) for o in _ops(_run(f))] == [FDivOperator]


def test_subtraction_folds_into_second_operand_sign() -> None:
    def f(a, b):  # type: ignore[no-untyped-def]
        return a - b

    ops = _ops(_run(f))
    assert len(ops) == 1
    assert isinstance(ops[0].operator, FAddOperator) and ops[0].operand_conditioners[1] == FloatSignControl(negate=True)


def test_operand_negation_folds_into_operator() -> None:
    def f(a, b):  # type: ignore[no-untyped-def]
        return a * (-b)

    ops = _ops(_run(f))
    assert len(ops) == 1
    assert isinstance(ops[0].operator, FMulOperator) and ops[0].operand_conditioners[1] == FloatSignControl(negate=True)


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
    # in _copy.reverse_postorder (and the symmetric _lir._layout.mir_rpo). With recursion in place, optimize() raises;
    # the iterative DFS compiles cleanly. Exercise the whole front-to-back pipeline (optimize, MIR lowering, LIR build)
    # since each contains a CFG DFS, and check the bit-exact model against the plain-Python reference.
    hir = lower(_deep_cfg_kernel)
    assert len(hir.blocks) > 1000  # the CFG is genuinely deep (otherwise the regression would not bite)
    model = build_model(build(_run(_deep_cfg_kernel), "deep"))
    for x in (0.5, 2.0, 8.0):  # acc stays positive -> +900 every time; 0.5/2.0/8.0 are exact in ZKF
        assert float(model.run(x)[0]) == _deep_cfg_kernel(x)


def test_const_fold_handles_absorbing_and_identity_boolean_connectives() -> None:
    # Regression (user): the constant folder must fold every constant expression, including a partially-constant
    # connective via its absorbing element (``x or True`` -> True, ``x and False`` -> False), and must drop the
    # identity element (``x or False`` -> x, ``x and True`` -> x), which is what collapses the residual ``and`` a
    # chained comparison leaves once a statically-true link folds.
    builder = HirBuilder()
    builder.block()
    x = builder.input("x", BoolType())  # a dynamic boolean operand
    true_, false_ = builder.bool_const(True), builder.bool_const(False)
    builder.output("or_abs", builder.operation(BoolOr(), [x, true_]))
    builder.output("and_abs", builder.operation(BoolAnd(), [x, false_]))
    builder.output("or_id", builder.operation(BoolOr(), [x, false_]))
    builder.output("and_id", builder.operation(BoolAnd(), [x, true_]))
    builder.ret()
    folded = fold_constants(builder.finish())
    out = {o.name: folded.nodes[o.value] for o in folded.outputs}
    assert out["or_abs"] == BoolConst(True)  # x or True  -> True   (absorbing)
    assert out["and_abs"] == BoolConst(False)  # x and False -> False  (absorbing)
    assert isinstance(out["or_id"], InPort) and out["or_id"].name == "x"  # x or False -> x  (identity dropped)
    assert isinstance(out["and_id"], InPort) and out["and_id"].name == "x"  # x and True -> x  (identity dropped)


def _hir_of(target):  # type: ignore[no-untyped-def]
    return optimize(lower(target))


def test_if_conversion_collapses_a_pure_diamond() -> None:
    # A small pure diamond becomes one straight-line block: the merge phi is replaced by a select over the branch
    # condition, the branch disappears, and the arms' operations run unconditionally.
    def f(a, b):  # type: ignore[no-untyped-def]
        if a > b:
            y = a + b
        else:
            y = a - b
        return y

    hir = _hir_of(f)
    assert len(hir.blocks) == 1
    selects = [n for n in hir.nodes.values() if isinstance(n, Operation) and isinstance(n.operator, Select)]
    assert len(selects) == 1


def test_if_conversion_refuses_an_unspeculatable_arm() -> None:
    # Division must not be speculated: a div-by-zero on the not-taken path would assert the module error flag.
    def f(a, b):  # type: ignore[no-untyped-def]
        if a > b:
            y = a + b
        else:
            y = a / b
        return y

    hir = _hir_of(f)
    assert len(hir.blocks) == 4  # the diamond survives as a real branch
    assert not any(isinstance(n, Operation) and isinstance(n.operator, Select) for n in hir.nodes.values())


def test_if_conversion_respects_the_arm_size_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(if_convert_pass, "_IFCONV_MAX_OPS", 1)

    def f(a, b):  # type: ignore[no-untyped-def]
        if a > b:
            y = (a + b) * a + b  # three operations: over the per-arm budget of one
        else:
            y = a - b
        return y

    hir = _hir_of(f)
    assert len(hir.blocks) == 4


def test_if_conversion_knob_zero_disables_the_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(if_convert_pass, "_IFCONV_MAX_OPS", 0)

    def f(a, b):  # type: ignore[no-untyped-def]
        if a > b:
            y = a + b
        else:
            y = a - b
        return y

    hir = _hir_of(f)
    assert len(hir.blocks) == 4 and not any(
        isinstance(n, Operation) and isinstance(n.operator, Select) for n in hir.nodes.values()
    )


def test_if_conversion_converts_a_boolean_phi_merge() -> None:
    # Bool-phi if-conversion: a diamond merging a boolean collapses to one block, the merge becoming a bool_select
    # (a float select is the wide dual). Both arms here are dynamic comparisons, so strength reduction keeps the mux.
    def f(a, b, c):  # type: ignore[no-untyped-def]
        if a > b:
            flag = b > c
        else:
            flag = a > c
        return float(flag)

    hir = _hir_of(f)
    assert len(hir.blocks) == 1
    assert any(isinstance(n, Operation) and isinstance(n.operator, BoolSelect) for n in hir.nodes.values())
    assert not any(isinstance(n, Operation) and isinstance(n.operator, Select) for n in hir.nodes.values())


def test_if_conversion_reduces_constant_armed_boolean_select() -> None:
    # The state-machine merge shape: arms are boolean constants, so the bool_select reduces to and/or/not via strength
    # reduction (no select node survives), exactly the schmitt/pfd collapse to a single straight-line block.
    def f(a, b, hold: bool):  # type: ignore[no-untyped-def]
        if a > b:
            flag = True
        else:
            flag = hold  # passthrough arm
        return float(flag)

    hir = _hir_of(f)
    assert len(hir.blocks) == 1
    # bool_select(a>b, True, hold) == (a>b) or hold -- reduced away, no select of either flavor remains.
    assert not any(
        isinstance(n, Operation) and isinstance(n.operator, (BoolSelect, Select)) for n in hir.nodes.values()
    )


def test_bool_select_reductions_are_truth_table_correct() -> None:
    # The bool-mux strength-reduction identities (a bool_select with constant arms collapses to and/or/not/passthrough)
    # must be bit-exact. Each shape is run through the numerical model over every boolean input combination and checked
    # against its Python reference; a wrong identity -- e.g. (c,False,True) reduced to c not ~c -- mismatches here.
    import itertools

    def s_tf(c: bool):  # type: ignore[no-untyped-def]   # (c, True, False) -> c
        if c:
            y = True
        else:
            y = False
        return y

    def s_ft(c: bool):  # type: ignore[no-untyped-def]   # (c, False, True) -> not c
        if c:
            y = False
        else:
            y = True
        return y

    def s_t_dyn(c: bool, f: bool):  # type: ignore[no-untyped-def]   # (c, True, f) -> c or f
        if c:
            y = True
        else:
            y = f
        return y

    def s_f_dyn(c: bool, f: bool):  # type: ignore[no-untyped-def]   # (c, False, f) -> (not c) and f
        if c:
            y = False
        else:
            y = f
        return y

    def s_dyn_t(c: bool, t: bool):  # type: ignore[no-untyped-def]   # (c, t, True) -> (not c) or t
        if c:
            y = t
        else:
            y = True
        return y

    def s_dyn_f(c: bool, t: bool):  # type: ignore[no-untyped-def]   # (c, t, False) -> c and t
        if c:
            y = t
        else:
            y = False
        return y

    def s_dyn_dyn(c: bool, t: bool, f: bool):  # type: ignore[no-untyped-def]   # (c, t, f) -> bool_select kept
        if c:
            y = t
        else:
            y = f
        return y

    def s_dyn_not_dyn(c: bool, t: bool, f: bool):  # type: ignore[no-untyped-def]  # (c, t, ~f) -> kept, arm inverted
        if c:
            y = t
        else:
            y = not f
        return y

    cases = [
        (s_tf, lambda c: c, 1, False),
        (s_ft, lambda c: not c, 1, False),
        (s_t_dyn, lambda c, f: c or f, 2, False),
        (s_f_dyn, lambda c, f: (not c) and f, 2, False),
        (s_dyn_t, lambda c, t: (not c) or t, 2, False),
        (s_dyn_f, lambda c, t: c and t, 2, False),
        (s_dyn_dyn, lambda c, t, f: t if c else f, 3, True),
        # A surviving bool_select whose arm carries a NOT-folded inversion: the inversion rides the arm conditioner
        # (the generic inline-operand inversion path), distinct from the constant-arm reductions above.
        (s_dyn_not_dyn, lambda c, t, f: t if c else (not f), 3, True),
    ]
    for fn, ref, arity, keeps_select in cases:
        hir = _hir_of(fn)
        has_select = any(isinstance(n, Operation) and isinstance(n.operator, BoolSelect) for n in hir.nodes.values())
        assert (
            has_select == keeps_select
        ), f"{fn.__name__}: bool_select presence {has_select} != expected {keeps_select}"
        model = build_model(build(lower_to_mir(hir, OPS), fn.__name__))
        for combo in itertools.product([False, True], repeat=arity):
            got = bool(model.run(*combo)[0])
            assert got == bool(ref(*combo)), f"{fn.__name__}{combo}: got {got}, want {ref(*combo)}"


def test_if_conversion_collapses_nested_chains_to_one_block() -> None:
    # Sequential diamonds collapse one after another, recompacting block ids each time, leaving a single block.
    def f(x, y):  # type: ignore[no-untyped-def]
        if x > 0.0:
            a = x + y
        else:
            a = x - y
        if y > 0.0:
            b = a * 2.0
        else:
            b = a * 4.0
        return b

    hir = _hir_of(f)
    assert len(hir.blocks) == 1
    selects = [n for n in hir.nodes.values() if isinstance(n, Operation) and isinstance(n.operator, Select)]
    assert len(selects) == 2


def test_if_conversion_repoints_loop_header_phi_arms() -> None:
    # A diamond inside a while body: the dissolved merge block fed the loop-header phis, whose arms must repoint to
    # the spliced block (the localized pin for the repoint path; the examples exercise it only end-to-end).
    def f(x):  # type: ignore[no-untyped-def]
        w = x
        while w > 0.0:
            if w > 2.0:
                step = 2.0
            else:
                step = 1.0
            w = w - step
        return w

    hir = _hir_of(f)
    selects = [n for n in hir.nodes.values() if isinstance(n, Operation) and isinstance(n.operator, Select)]
    assert len(selects) == 1
    block_ids = {b.id for b in hir.blocks}
    for node in hir.nodes.values():
        if isinstance(node, Phi):
            assert all(pred in block_ids for pred, _ in node.arms), "phi arms must reference surviving blocks only"


def test_speculatable_hir_operators_map_to_error_free_hardware() -> None:
    # The speculation flag and the hardware error sideband are two declarations of one fact: division is the only
    # error-bearing operator today, and it must stay unspeculatable. A future error-bearing operator must declare
    # speculatable=False (the default) on its HIR side, or if-conversion would assert the module error flag for a
    # never-taken path.
    assert FDivOperator(FMT).error_ports and not HirFloatDiv.speculatable


def test_dead_diamond_frees_its_condition_cone() -> None:
    # Conversion turns control dependence into data dependence: when a diamond's merged results are entirely unused,
    # its condition cone becomes ordinary dead code -- INCLUDING an error-bearing division feeding only the
    # condition, which then reports nothing (exactly as an unused division without a branch around it reports
    # nothing today). This pins the documented semantics of the error sideband: executed operators only.
    def f(a, b, x):  # type: ignore[no-untyped-def]
        if bool(a / b):
            y = x + 1.0
        else:
            y = x - 1.0
        _ = y  # the merged result is never returned: the whole diamond, condition cone included, is dead
        return x

    hir = _hir_of(f)
    assert len(hir.blocks) == 1
    assert not any(
        isinstance(n, Operation) and isinstance(n.operator, HirFloatDiv) for n in hir.nodes.values()
    ), "the unused condition cone (division included) is dead code after conversion"


def test_operator_layer_does_not_import_hir() -> None:
    """
    The hardware operator models are a base vocabulary layer below the IR pipeline; they must never reach back into the
    semantic HIR -- the smell W12 removed (importing ``RelationalOp`` from ``_hir``). Locks the severed edge
    transitively.
    """
    offenders = forbidden_imports("holoso._operators", "holoso._hir")
    assert not offenders, f"the operator layer transitively imports HIR: {offenders}"
