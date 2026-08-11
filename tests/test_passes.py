"""Unit tests for HIR optimization and MIR selection passes."""

import dataclasses
import math
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import pytest

import holoso
from holoso import (
    FAddOptions,
    FCmpOptions,
    FDivOptions,
    FMulILog2Options,
    FMulOptions,
    FloatFormat,
    FloatValue,
    OperatorOptions,
    Options,
)
from holoso._operators import FAddOperator, FDivOperator, FMulOperator, OpConfig
from holoso._errors import SynthesisError, UnsupportedConstruct
from holoso._util import ValueId
from holoso._eel import lower
from holoso._hir import (
    BoolAnd,
    BoolConst,
    BoolNot,
    BoolOr,
    BoolType,
    Const,
    FloatAdd,
    FloatConst,
    FloatMul,
    FloatType as HirFloatType,
    Hir,
    HirBuilder,
    InPort,
    Operation,
    Operator,
    Signature,
    Type,
    optimize,
)
from holoso._hir import Branch, BoolSelect, FloatDiv as HirFloatDiv, Phi, FloatSelect, StateRead
from holoso._hir import (
    FloatAbs,
    FloatAtan2,
    FloatCos,
    FloatExp2,
    FloatFloor,
    FloatFma,
    FloatHypot2,
    FloatIsFinite,
    FloatIsInf,
    FloatIsNegInf,
    FloatIsPosInf,
    FloatLog2,
    FloatRound,
    FloatSin,
    FloatSqrt,
    FloatToBool,
    FloatTrunc,
)
from holoso._hir import (
    BoolToInt,
    BoolXor,
    FloatToInt,
    IntAbs,
    IntAdd,
    IntBwAnd,
    IntBwNot,
    IntBwOr,
    IntBwXor,
    IntConst,
    IntDivFloor,
    IntEqual,
    IntGreater,
    IntGreaterOrEqual,
    IntLess,
    IntLessOrEqual,
    IntMod,
    IntMul,
    IntMulPow2,
    IntNeg,
    IntNotEqual,
    IntSelect,
    IntShiftLeft,
    IntShiftRight,
    IntSub,
    IntToBool,
    IntToFloat,
    IntType,
)
from ._modelref import DEFAULT_IFCONV_MAX_OPS, default_ifmt, build_lir
from holoso._mir._refuse import refuse
from holoso._mir import (
    lower as lower_to_mir,
    Mir,
    MirFloatConst,
    MirFloatInput,
    MirFloatOutput,
    MirIntConst,
    MirInterpreter,
    MirOperation,
)
from holoso._operators import FMulILog2Operator, FloatSignControl
from ._importguard import forbidden_imports
from ._modelref import (
    branch_boundary_kernel,
    build_model,
    build_ops,
    const_branch_kernel,
    diamond_then_loop_kernel,
    overlap_spill_kernel,
    phi_swap_loop,
)
from ._examples import equal_temperament

FMT = FloatFormat(6, 18)
OPS = build_ops(
    Options(
        OperatorOptions(
            fadd=FAddOptions(),
            fmul=FMulOptions(),
            fdiv=FDivOptions(),
            fmul_ilog2=FMulILog2Options(),
            fcmp=FCmpOptions(),
        ),
        ffmt=FMT,
    )
)


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
        return Signature((OtherType(),), OtherType())

    def evaluate(self, operands: list[Const]) -> Const:
        (operand,) = operands
        assert isinstance(operand, OtherConst)
        return OtherConst(operand.value + 1)


def _run(target: Callable[..., object], ops: OpConfig = OPS, fmt: FloatFormat = FMT) -> Mir:
    return lower_to_mir(lower(target).hir, ops, fmt, default_ifmt(fmt), DEFAULT_IFCONV_MAX_OPS)


def _op_count(mir: Mir, cls: type) -> int:
    return sum(1 for n in mir.nodes.values() if isinstance(n, MirOperation) and isinstance(n.operator, cls))


def _ops(mir: Mir) -> list[MirOperation]:
    return [n for n in mir.nodes.values() if isinstance(n, MirOperation)]


def _consts(mir: Mir) -> list[float]:
    return [n.value for n in mir.nodes.values() if isinstance(n, MirFloatConst)]


def _exponent_of(mir: Mir, op: MirOperation) -> int:
    assert isinstance(op.operator, FMulILog2Operator)
    exponent = mir.nodes[op.operands[1]]
    assert isinstance(exponent, MirIntConst)
    return exponent.value


def test_hir_nodes_carry_float_type() -> None:
    builder = HirBuilder()
    builder.block()
    a = builder.input("a", HirFloatType())
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


def test_lower_rejects_non_float_hir_input_type() -> None:
    builder = HirBuilder()
    builder.block()
    a = builder.input("a", OtherType())
    builder.output("out_0", a)
    builder.ret()
    hir = builder.finish()

    try:
        lower_to_mir(hir, OPS, FMT, default_ifmt(FMT), DEFAULT_IFCONV_MAX_OPS)
    except UnsupportedConstruct as ex:
        assert "no MIR lowering rule" in str(ex)
    else:
        assert False, "expected HIR-to-MIR lowering to reject non-float semantic input"


def test_hir_constant_folding_returns_float_const() -> None:
    def f() -> float:
        return 1.25 + 2.0

    hir = optimize(lower(f).hir, DEFAULT_IFCONV_MAX_OPS)
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

    hir = optimize(builder.finish(), DEFAULT_IFCONV_MAX_OPS)
    node = hir.nodes[hir.outputs[0].value]
    assert isinstance(node, OtherConst)
    assert node.value == 11


def test_mir_constant_only_node_carries_float_type() -> None:
    def f() -> float:
        return 3.5

    mir = _run(f)
    const = mir.nodes[mir.outputs[0].value]
    assert isinstance(const, MirFloatConst)
    assert const.scalar_type.fmt == FMT


def test_mul_by_pow2_const_becomes_ilog2() -> None:
    def f(a: float) -> float:
        return a * 0.25

    mir = _run(f)
    ops = _ops(mir)
    assert len(ops) == 1
    assert _exponent_of(mir, ops[0]) == -2


def test_left_const_fmul_pow2_is_commutative() -> None:
    def f(a: float) -> float:
        return 2 * a

    mir = _run(f)
    ops = _ops(mir)
    assert len(ops) == 1
    assert _exponent_of(mir, ops[0]) == 1


def test_div_by_pow2_becomes_ilog2() -> None:
    def f(a: float) -> float:
        return a / 4.0

    mir = _run(f)
    ops = _ops(mir)
    assert len(ops) == 1
    assert _exponent_of(mir, ops[0]) == -2


def test_div_by_nonpow2_const_becomes_reciprocal_multiply() -> None:
    def f(a: float) -> float:
        return a / 3.0

    mir = _run(f)
    ops = _ops(mir)
    assert [type(o.operator) for o in ops] == [FMulOperator]
    assert any(abs(c - 1.0 / 3.0) < 1e-12 for c in _consts(mir))


def test_wide_supported_pow2_uses_ilog2_operator() -> None:
    def f(a: float) -> float:
        return a * 16.0

    fmt = FloatFormat(3, 4)
    ops = build_ops(
        Options(
            OperatorOptions(
                fadd=FAddOptions(),
                fmul=FMulOptions(),
                fdiv=FDivOptions(),
                fmul_ilog2=FMulILog2Options(),
                fcmp=FCmpOptions(),
            ),
            ffmt=fmt,
        )
    )
    mir = _run(f, ops, fmt)
    selected = _ops(mir)
    assert [type(o.operator) for o in selected] == [FMulILog2Operator]
    assert _exponent_of(mir, selected[0]) == 4
    assert _consts(mir) == []


def test_a_scale_past_the_float_range_builds_and_rails() -> None:
    """The exponent is an integer constant, so no scale is refused; past the format's range the scaler rails."""

    def f(a: float) -> float:
        return a * 64.0

    fmt = FloatFormat(3, 4)
    ops = build_ops(
        Options(
            OperatorOptions(
                fadd=FAddOptions(),
                fmul=FMulOptions(),
                fdiv=FDivOptions(),
                fmul_ilog2=FMulILog2Options(),
                fcmp=FCmpOptions(),
            ),
            ffmt=fmt,
        )
    )
    mir = _run(f, ops, fmt)
    selected = _ops(mir)
    assert [type(o.operator) for o in selected] == [FMulILog2Operator]
    assert _exponent_of(mir, selected[0]) == 6
    for value in [1.0, -0.5, 0.0]:
        assert MirInterpreter(mir).run(value) == [FloatValue.from_float(fmt, value).scale_pow2(6)]


def test_an_exponent_past_the_int_format_clamps_where_the_scaler_rails() -> None:
    """
    A count past the int word lies far beyond the float's dynamic range, so the clamped exponent rails identically.
    """

    def f(a: float) -> tuple[float, float]:
        return a * 2.0**1000, a / 2.0**1000

    options = Options(
        OperatorOptions(
            fadd=FAddOptions(),
            fmul=FMulOptions(),
            fdiv=FDivOptions(),
            fmul_ilog2=FMulILog2Options(),
            fcmp=FCmpOptions(),
        ),
        ffmt=FloatFormat(4, 5),
        wint_min=2,
    )
    mir = lower_to_mir(lower(f).hir, build_ops(options), options.ffmt, options.ifmt, DEFAULT_IFCONV_MAX_OPS)
    scales = [op for op in _ops(mir) if isinstance(op.operator, FMulILog2Operator)]
    assert sorted(_exponent_of(mir, op) for op in scales) == [options.ifmt.min, options.ifmt.max]
    for value in [1.0, -1.5, 0.0]:
        loaded = FloatValue.from_float(options.ffmt, value)
        assert MirInterpreter(mir).run(value) == [loaded.scale_pow2(1000), loaded.scale_pow2(-1000)]


def test_true_division_stays_fdiv() -> None:
    def f(a: float, b: float) -> float:
        return a / b

    assert [type(o.operator) for o in _ops(_run(f))] == [FDivOperator]


def test_subtraction_folds_into_second_operand_sign() -> None:
    def f(a: float, b: float) -> float:
        return a - b

    ops = _ops(_run(f))
    assert len(ops) == 1
    assert isinstance(ops[0].operator, FAddOperator) and ops[0].operand_conditioners[1] == FloatSignControl(negate=True)


def test_operand_negation_folds_into_operator() -> None:
    def f(a: float, b: float) -> float:
        return a * (-b)

    ops = _ops(_run(f))
    assert len(ops) == 1
    assert isinstance(ops[0].operator, FMulOperator) and ops[0].operand_conditioners[1] == FloatSignControl(negate=True)


def test_pure_sign_output_adds_no_operation() -> None:
    def f(a: float) -> float:
        return -abs(a)

    mir = _run(f)
    assert _ops(mir) == []
    assert isinstance(mir.outputs[0], MirFloatOutput)
    assert mir.outputs[0].conditioner == FloatSignControl(absolute=True).then(FloatSignControl(negate=True))


def test_selected_mir_has_only_input_const_operation_nodes() -> None:
    def f(a: float, b: float) -> float:
        return (a - b) * 0.25 + a * b

    mir = _run(f)
    assert all(isinstance(n, (MirFloatInput, MirFloatConst, MirIntConst, MirOperation)) for n in mir.nodes.values())


def test_ekf1_stateless_lowering() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    import ekf1_stateless

    mir = _run(ekf1_stateless.update_x_P)
    assert all(isinstance(n, (MirFloatInput, MirFloatConst, MirIntConst, MirOperation)) for n in mir.nodes.values())
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
    x = builder.input("x", HirFloatType())
    builder.position_at(entry)
    builder.jump(header)
    builder.position_at(header)
    builder.open_phi(HirFloatType(), (entry, x))  # only the preheader arm; the latch arm is never supplied
    builder.jump(header)  # back-edge: the header now has two predecessors (entry, header) but the phi carries one arm
    with pytest.raises(RuntimeError, match="predecessor"):
        builder.finish()


def _deep_cfg_kernel(p0: float) -> float:
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
    # in _copy.reverse_postorder (and the symmetric _lir._mir_facts.mir_rpo). With recursion in place, optimize()
    # raises; the iterative DFS compiles cleanly. Exercise the whole front-to-back pipeline (optimize, MIR lowering,
    # LIR build) since each contains a CFG DFS, and check the bit-exact model against the plain-Python reference.
    hir = lower(_deep_cfg_kernel).hir
    assert len(hir.blocks) > 1000  # the CFG is genuinely deep (otherwise the regression would not bite)
    model = build_model(build_lir(_run(_deep_cfg_kernel), "deep"))
    for x in (0.5, 2.0, 8.0):  # acc stays positive -> +900 every time; 0.5/2.0/8.0 are exact in ZKF
        assert float(model.run(x)[0]) == _deep_cfg_kernel(x)


def test_absorbing_and_identity_boolean_connectives_reduce() -> None:
    # Regression (user): a partially-constant connective reduces through its absorbing element (``x or True`` -> True,
    # ``x and False`` -> False), and its identity element drops out (``x or False`` -> x, ``x and True`` -> x), which
    # is what collapses the residual ``and`` a chained comparison leaves once a statically-true link folds.
    builder = HirBuilder()
    builder.block()
    x = builder.input("x", BoolType())  # a dynamic boolean operand
    true_, false_ = builder.bool_const(True), builder.bool_const(False)
    builder.output("or_abs", builder.operation(BoolOr(), [x, true_]))
    builder.output("and_abs", builder.operation(BoolAnd(), [x, false_]))
    builder.output("or_id", builder.operation(BoolOr(), [x, false_]))
    builder.output("and_id", builder.operation(BoolAnd(), [x, true_]))
    builder.ret()
    reduced = optimize(builder.finish(), DEFAULT_IFCONV_MAX_OPS)
    out = {o.name: reduced.nodes[o.value] for o in reduced.outputs}
    assert out["or_abs"] == BoolConst(True)  # x or True  -> True   (absorbing)
    assert out["and_abs"] == BoolConst(False)  # x and False -> False  (absorbing)
    assert isinstance(out["or_id"], InPort) and out["or_id"].name == "x"  # x or False -> x  (identity dropped)
    assert isinstance(out["and_id"], InPort) and out["and_id"].name == "x"  # x and True -> x  (identity dropped)


def _hir_of(target: object, ifconv_max_ops: int = DEFAULT_IFCONV_MAX_OPS) -> Hir:
    return optimize(lower(target).hir, ifconv_max_ops)


def test_if_conversion_collapses_a_pure_diamond() -> None:
    def f(a: float, b: float) -> float:
        if a > b:
            y = a + b
        else:
            y = a - b
        return y

    hir = _hir_of(f)
    assert len(hir.blocks) == 1
    selects = [n for n in hir.nodes.values() if isinstance(n, Operation) and isinstance(n.operator, FloatSelect)]
    assert len(selects) == 1


def test_if_conversion_refuses_an_unspeculatable_arm() -> None:
    # Division must not be speculated: a div-by-zero on the not-taken path would assert the module error flag.
    def f(a: float, b: float) -> float:
        if a > b:
            y = a + b
        else:
            y = a / b
        return y

    hir = _hir_of(f)
    assert len(hir.blocks) == 4  # the diamond survives as a real branch
    assert not any(isinstance(n, Operation) and isinstance(n.operator, FloatSelect) for n in hir.nodes.values())


def test_if_conversion_respects_the_arm_size_budget() -> None:
    def f(a: float, b: float) -> float:
        if a > b:
            y = (a + b) * a + b  # three operations, one over the budget below
        else:
            y = a - b  # a negate and an add: exactly at the budget, so only the then-arm can refuse
        return y

    assert len(_hir_of(f, 2).blocks) == 4
    assert len(_hir_of(f, 3).blocks) == 1


def test_zero_budget_still_converts_the_operation_free_diamond() -> None:
    def constant_arms(a: float, b: float) -> float:
        if a > b:
            y = 1.0
        else:
            y = 2.0
        return y

    def one_op_arm(a: float, b: float) -> float:
        if a > b:
            y = a + b
        else:
            y = 1.0
        return y

    hir = _hir_of(constant_arms, 0)
    assert len(hir.blocks) == 1
    assert any(isinstance(n, Operation) and isinstance(n.operator, FloatSelect) for n in hir.nodes.values())
    assert len(_hir_of(one_op_arm, 0).blocks) == 4


def test_synthesize_honors_the_ifconv_option() -> None:
    def f(a: float, b: float) -> float:
        if a > b:
            y = a + b
        else:
            y = a - b
        return y

    # A converted diamond leaves one straight-line block, so the II is exact; a surviving branch makes it data-dependent.
    options = Options(OperatorOptions(fadd=FAddOptions(), fcmp=FCmpOptions()), ffmt=FMT, ifconv_max_ops=2)
    assert holoso.synthesize(f, options, name="ifconv_on").initiation_interval[1] is not None
    disabled = dataclasses.replace(options, ifconv_max_ops=0)
    assert holoso.synthesize(f, disabled, name="ifconv_off").initiation_interval[1] is None


def test_ifconv_budget_changes_the_schedule_but_never_the_result() -> None:
    # The knob's whole contract. The II inequality is what keeps it honest: without it the comparison would still
    # pass if the budget stopped having any effect at all.
    def f(a: float, b: float, c: float) -> float:
        if a > b:
            y = a + b
        else:
            y = a * c
        if y > c:
            z = y + 1.0
        else:
            z = y - c
        return z

    options = Options(OperatorOptions(fadd=FAddOptions(), fmul=FMulOptions(), fcmp=FCmpOptions()), ffmt=FMT)
    branchy = holoso.synthesize(f, dataclasses.replace(options, ifconv_max_ops=0), name="ifconv_branchy")
    converted = holoso.synthesize(f, dataclasses.replace(options, ifconv_max_ops=64), name="ifconv_converted")
    assert branchy.initiation_interval != converted.initiation_interval, "the budget no longer moves the schedule"
    left, right = branchy.numerical_model.elaborate(), converted.numerical_model.elaborate()
    for a, b, c in [(1.0, 0.5, 2.0), (0.5, 1.0, -2.0), (-3.0, -4.0, 0.25), (0.0, 0.0, 1.0), (2.0, 2.0, -1.0)]:
        assert left.run(a, b, c) == right.run(a, b, c), f"if-conversion changed the result at ({a}, {b}, {c})"


def test_if_conversion_converts_a_boolean_phi_merge() -> None:
    # Bool-phi if-conversion: a diamond merging a boolean collapses to one block, the merge becoming a bselect
    # (an fselect is the wide dual). Both arms here are dynamic comparisons, so strength reduction keeps the mux.
    def f(a: float, b: float, c: float) -> float:
        if a > b:
            flag = b > c
        else:
            flag = a > c
        return float(flag)

    hir = _hir_of(f)
    assert len(hir.blocks) == 1
    assert any(isinstance(n, Operation) and isinstance(n.operator, BoolSelect) for n in hir.nodes.values())


def test_if_conversion_converts_an_integer_phi_merge() -> None:
    # The integer dual of the two above, so min/max/sign cost one mux rather than a branch.
    def f(a: int, b: int) -> int:
        if a > b:
            y = a + b
        else:
            y = a - b
        return y

    hir = _hir_of(f)
    assert len(hir.blocks) == 1
    assert any(isinstance(n, Operation) and isinstance(n.operator, IntSelect) for n in hir.nodes.values())


def test_an_error_bearing_integer_arm_stays_behind_its_guard() -> None:
    # Floor-division and modulo assert the div-by-zero flag, so a never-taken arm holding one must not be speculated.
    def f(a: int, b: int) -> int:
        if a > b:
            y = a + b
        else:
            y = a // b
        return y

    hir = _hir_of(f)
    assert len(hir.blocks) == 4
    assert not any(isinstance(n, Operation) and isinstance(n.operator, IntSelect) for n in hir.nodes.values())
    assert not any(isinstance(n, Operation) and isinstance(n.operator, FloatSelect) for n in hir.nodes.values())


def test_if_conversion_reduces_constant_armed_boolean_select() -> None:
    # The state-machine merge shape: arms are boolean constants, so the bselect reduces to and/or/not via strength
    # reduction (no select node survives), exactly the schmitt/pfd collapse to a single straight-line block.
    def f(a: float, b: float, hold: bool) -> float:
        if a > b:
            flag = True
        else:
            flag = hold  # passthrough arm
        return float(flag)

    hir = _hir_of(f)
    assert len(hir.blocks) == 1
    # bselect(a>b, True, hold) == (a>b) or hold -- reduced away, no select of either flavor remains.
    assert not any(
        isinstance(n, Operation) and isinstance(n.operator, (BoolSelect, FloatSelect)) for n in hir.nodes.values()
    )


def test_bselect_reductions_are_truth_table_correct() -> None:
    # The bool-mux strength-reduction identities (a bselect with constant arms collapses to and/or/not/passthrough)
    # must be bit-exact. Each shape is run through the numerical model over every boolean input combination and checked
    # against its Python reference; a wrong identity -- e.g. (c,False,True) reduced to c not ~c -- mismatches here.
    import itertools

    def s_tf(c: bool) -> bool:  # (c, True, False) -> c
        if c:
            y = True
        else:
            y = False
        return y

    def s_ft(c: bool) -> bool:  # (c, False, True) -> not c
        if c:
            y = False
        else:
            y = True
        return y

    def s_t_dyn(c: bool, f: bool) -> bool:  # (c, True, f) -> c or f
        if c:
            y = True
        else:
            y = f
        return y

    def s_f_dyn(c: bool, f: bool) -> bool:  # (c, False, f) -> (not c) and f
        if c:
            y = False
        else:
            y = f
        return y

    def s_dyn_t(c: bool, t: bool) -> bool:  # (c, t, True) -> (not c) or t
        if c:
            y = t
        else:
            y = True
        return y

    def s_dyn_f(c: bool, t: bool) -> bool:  # (c, t, False) -> c and t
        if c:
            y = t
        else:
            y = False
        return y

    def s_dyn_dyn(c: bool, t: bool, f: bool) -> bool:  # (c, t, f) -> bselect kept
        if c:
            y = t
        else:
            y = f
        return y

    def s_dyn_not_dyn(c: bool, t: bool, f: bool) -> bool:  # (c, t, ~f) -> kept, arm inverted
        if c:
            y = t
        else:
            y = not f
        return y

    cases: list[tuple[Callable[..., bool], Callable[..., bool], int, bool]] = [
        (s_tf, lambda c: c, 1, False),
        (s_ft, lambda c: not c, 1, False),
        (s_t_dyn, lambda c, f: c or f, 2, False),
        (s_f_dyn, lambda c, f: (not c) and f, 2, False),
        (s_dyn_t, lambda c, t: (not c) or t, 2, False),
        (s_dyn_f, lambda c, t: c and t, 2, False),
        (s_dyn_dyn, lambda c, t, f: t if c else f, 3, True),
        # A surviving bselect whose arm carries a NOT-folded inversion: the inversion rides the arm conditioner
        # (the generic inline-operand inversion path), distinct from the constant-arm reductions above.
        (s_dyn_not_dyn, lambda c, t, f: t if c else (not f), 3, True),
    ]
    for fn, ref, arity, keeps_select in cases:
        hir = _hir_of(fn)
        has_select = any(isinstance(n, Operation) and isinstance(n.operator, BoolSelect) for n in hir.nodes.values())
        assert has_select == keeps_select, f"{fn.__name__}: bselect presence {has_select} != expected {keeps_select}"
        model = build_model(
            build_lir(lower_to_mir(lower(fn).hir, OPS, FMT, default_ifmt(FMT), DEFAULT_IFCONV_MAX_OPS), fn.__name__)
        )
        for combo in itertools.product([False, True], repeat=arity):
            got = bool(model.run(*combo)[0])
            assert got == bool(ref(*combo)), f"{fn.__name__}{combo}: got {got}, want {ref(*combo)}"


def test_identical_mux_arms_collapse_whatever_the_selector() -> None:
    # ``mux(c, X, X) == X`` is the universal mux identity the constant-arm reductions above depend on: they read a
    # ``True``/``True`` pair as the True/False table entry and answer ``c``, which is only never wrong because
    # identical arms have already been reduced away. The shape cannot be witnessed before if-conversion -- while a
    # block boundary separates the arms, their copies of ``x*2.0`` are distinct values and the phi is not trivial --
    # so it appears only once the splice interns them into one block and each arm's comparison becomes a
    # self-comparison. Both kernels are constant functions in Python and must be constant in hardware too.
    def identical_bool_arms(x: float) -> bool:
        doubled = x * 2.0
        if x > 0.0:
            same = (x * 2.0) == doubled
        else:
            same = (x * 2.0) == doubled
        return same

    def identical_float_arms(x: float) -> float:
        doubled = x * 2.0
        if x > 0.0:
            picked = x * 2.0
        else:
            picked = doubled
        return picked

    for kernel, read in ((identical_bool_arms, bool), (identical_float_arms, float)):
        hir = _hir_of(kernel)
        assert not any(
            isinstance(n, Operation) and isinstance(n.operator, (BoolSelect, FloatSelect)) for n in hir.nodes.values()
        ), f"{kernel.__name__}: a mux over identical arms survived"
        model = build_model(
            build_lir(
                lower_to_mir(lower(kernel).hir, OPS, FMT, default_ifmt(FMT), DEFAULT_IFCONV_MAX_OPS), kernel.__name__
            )
        )
        for x in (-8.0, -1.0, 0.0, 0.5, 3.0):
            got, want = read(model.run(x)[0]), read(kernel(x))
            assert got == want, f"{kernel.__name__}({x}): got {got}, want {want}"


def test_if_conversion_collapses_nested_chains_to_one_block() -> None:
    def f(x: float, y: float) -> float:
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
    selects = [n for n in hir.nodes.values() if isinstance(n, Operation) and isinstance(n.operator, FloatSelect)]
    assert len(selects) == 2


def test_if_conversion_repoints_loop_header_phi_arms() -> None:
    # A diamond inside a while body: the dissolved merge block fed the loop-header phis, whose arms must repoint to
    # the spliced block (the localized pin for the repoint path; the examples exercise it only end-to-end).
    def f(x: float) -> float:
        w = x
        while w > 0.0:
            if w > 2.0:
                step = 2.0
            else:
                step = 1.0
            w = w - step
        return w

    hir = _hir_of(f)
    selects = [n for n in hir.nodes.values() if isinstance(n, Operation) and isinstance(n.operator, FloatSelect)]
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
    assert FDivOperator(FMT, FDivOptions()).error_ports and not HirFloatDiv.speculatable


def test_dead_diamond_frees_its_condition_cone() -> None:
    # Conversion turns control dependence into data dependence: when a diamond's merged results are entirely unused,
    # its condition cone becomes ordinary dead code -- INCLUDING an error-bearing division feeding only the
    # condition, which then reports nothing (exactly as an unused division without a branch around it reports
    # nothing today). This pins the documented semantics of the error sideband: executed operators only.
    def f(a: float, b: float, x: float) -> float:
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
    semantic HIR -- the smell W12 removed (importing a relation enum from ``_hir``). Locks the severed edge
    transitively.
    """
    offenders = forbidden_imports("holoso._operators", "holoso._hir")
    assert not offenders, f"the operator layer transitively imports HIR: {offenders}"


def test_the_operator_families_are_leaves_over_one_shared_vocabulary() -> None:
    """
    The families answer only to ``_common``: were one to import another, the shared vocabulary would have to live in
    whichever family happened to be lowest, which is how a family module becomes everyone's dumping ground.
    """
    families = ["holoso._operators._float", "holoso._operators._int", "holoso._operators._bool"]
    for family in families:
        for other in families:
            if family != other:
                assert not forbidden_imports(family, other), f"{family} reaches into {other}"
        assert not forbidden_imports("holoso._operators._common", family), f"the shared vocabulary reaches {family}"


def test_a_reduction_minted_constant_does_not_reach_the_datapath() -> None:
    # A reduction can mint what another rule would erase: the one-shot latch below reduces to ``first and False``,
    # which is the constant False written the long way. Every connective a reduction mints therefore passes through
    # the declared algebra in the same walk, or the latch live-out ships as a live boolean operation.
    class Primed:
        def __init__(self) -> None:
            self.y: float = 0.0
            self._first: bool = True

        def __call__(self, x: float) -> float:
            if self._first:
                self._first = False
                self.y = x
            else:
                self.y = self.y + 0.5 * (x - self.y)
            return self.y

    hir = _hir_of(Primed().__call__)
    first = next(slot for slot in hir.state_slots if slot.name == "_first")
    assert isinstance(hir.nodes[first.live_out], BoolConst)


# Exercised at the builder level so every operator/type pair is pinned directly, independent of which spellings the
# frontend happens to emit for it.
_INT_OPERATORS: list[tuple[Operator, list[Type], Type]] = [
    (IntAdd(), [IntType(), IntType()], IntType()),
    (IntSub(), [IntType(), IntType()], IntType()),
    (IntMul(), [IntType(), IntType()], IntType()),
    (IntMulPow2(3), [IntType()], IntType()),
    (IntDivFloor(), [IntType(), IntType()], IntType()),
    (IntMod(), [IntType(), IntType()], IntType()),
    (IntShiftLeft(), [IntType(), IntType()], IntType()),
    (IntShiftRight(), [IntType(), IntType()], IntType()),
    (IntBwAnd(), [IntType(), IntType()], IntType()),
    (IntBwOr(), [IntType(), IntType()], IntType()),
    (IntBwXor(), [IntType(), IntType()], IntType()),
    (IntNeg(), [IntType()], IntType()),
    (IntAbs(), [IntType()], IntType()),
    (IntBwNot(), [IntType()], IntType()),
    (IntLess(), [IntType(), IntType()], BoolType()),
    (IntLessOrEqual(), [IntType(), IntType()], BoolType()),
    (IntEqual(), [IntType(), IntType()], BoolType()),
    (IntNotEqual(), [IntType(), IntType()], BoolType()),
    (IntGreaterOrEqual(), [IntType(), IntType()], BoolType()),
    (IntGreater(), [IntType(), IntType()], BoolType()),
    (IntSelect(), [BoolType(), IntType(), IntType()], IntType()),
    (IntToFloat(), [IntType()], HirFloatType()),
    (FloatToInt(), [HirFloatType()], IntType()),
    (IntToBool(), [IntType()], BoolType()),
    (BoolToInt(), [BoolType()], IntType()),
]


@pytest.mark.parametrize("operator, operand_types, result_type", _INT_OPERATORS)
def test_integer_operator_signature(operator: Operator, operand_types: list[Type], result_type: Type) -> None:
    builder = HirBuilder()
    builder.block()
    operands = [
        {
            IntType(): builder.int_const(1),
            BoolType(): builder.bool_const(True),
            HirFloatType(): builder.float_const(1.0),
        }[operand_type]
        for operand_type in operand_types
    ]
    vid = builder.operation(operator, operands)
    assert builder.type_of(vid) == result_type


def test_integer_identity_and_absorbing_operands_simplify_against_a_runtime_value() -> None:
    # The declared integer identities and absorbing elements simplify against an operand the folder cannot see:
    # nothing but the input survives the identity chain, and an absorbing operand fixes the result outright.
    builder = HirBuilder()
    builder.block()
    n = builder.input("n", IntType())
    zero, one, all_ones = builder.int_const(0), builder.int_const(1), builder.int_const(-1)
    value = builder.operation(IntAdd(), [n, zero])
    value = builder.operation(IntMul(), [value, one])
    value = builder.operation(IntBwOr(), [value, zero])
    value = builder.operation(IntBwXor(), [value, zero])
    value = builder.operation(IntBwAnd(), [value, all_ones])
    builder.output("identities", value)
    builder.output("killed", builder.operation(IntMul(), [n, zero]))
    builder.output("saturated", builder.operation(IntBwOr(), [n, all_ones]))
    builder.output("masked_off", builder.operation(IntBwAnd(), [n, zero]))
    builder.ret()

    hir = optimize(builder.finish(), DEFAULT_IFCONV_MAX_OPS)
    outputs = {out.name: hir.nodes[out.value] for out in hir.outputs}
    assert isinstance(outputs["identities"], InPort), "every identity must drop, leaving the input itself"
    assert outputs["killed"] == outputs["masked_off"] == IntConst(0)
    assert outputs["saturated"] == IntConst(-1)
    assert not [node for node in hir.nodes.values() if isinstance(node, Operation)]


def test_a_constant_integer_expression_folds_away_entirely() -> None:
    # Folding is exact at arbitrary precision -- no width, no saturation -- so a fully static integer expression
    # disappears before MIR ever has to hold it in a machine word.
    builder = HirBuilder()
    builder.block()
    value = builder.operation(IntAdd(), [builder.int_const(2), builder.int_const(3)])
    value = builder.operation(IntMul(), [value, builder.int_const(2**24)])  # past the machine word, within the float
    builder.output("y", builder.operation(IntToFloat(), [value]))
    builder.ret()

    raw = builder.finish()
    hir = optimize(raw, DEFAULT_IFCONV_MAX_OPS)
    assert not [node for node in hir.nodes.values() if isinstance(node, Operation)]
    (out,) = hir.outputs
    assert hir.nodes[out.value] == FloatConst(float(5 * 2**24))
    lower_to_mir(raw, OPS, FMT, default_ifmt(FMT), DEFAULT_IFCONV_MAX_OPS)  # nothing integer is left


def test_integer_folding_is_exact_across_the_vocabulary() -> None:
    huge = 3**500  # far past any hardware width, and past the binary64 carrier
    cases: list[tuple[Operator, list[int], int]] = [
        (IntAdd(), [huge, 1], huge + 1),
        (IntSub(), [1, huge], 1 - huge),
        (IntMul(), [huge, -huge], -(huge**2)),
        (IntDivFloor(), [7, 2], 3),
        (IntDivFloor(), [-7, 2], -4),
        (IntDivFloor(), [7, -2], -4),
        (IntDivFloor(), [-7, -2], 3),
        (IntMod(), [7, 2], 1),
        (IntMod(), [-7, 2], 1),
        (IntMod(), [7, -2], -1),
        (IntMod(), [-7, -2], -1),
        (IntShiftLeft(), [huge, 64], huge << 64),
        (IntShiftRight(), [-huge, 64], -huge >> 64),
        (IntBwAnd(), [0b1100, 0b1010], 0b1000),
        (IntBwOr(), [0b1100, 0b1010], 0b1110),
        (IntBwXor(), [0b1100, 0b1010], 0b0110),
        (IntNeg(), [huge], -huge),
        (IntMulPow2(64), [huge], huge << 64),
        (IntAbs(), [-huge], huge),
        (IntBwNot(), [huge], ~huge),
    ]
    for operator, operands, expected in cases:
        builder = HirBuilder()
        builder.block()
        builder.output("y", builder.operation(operator, [builder.int_const(operand) for operand in operands]))
        builder.ret()
        hir = optimize(builder.finish(), DEFAULT_IFCONV_MAX_OPS)
        assert not [node for node in hir.nodes.values() if isinstance(node, Operation)], operator
        assert hir.nodes[hir.outputs[0].value] == IntConst(expected), operator


def test_integer_folding_has_no_size_limit() -> None:
    # A big shift is computed, not declined: an expression the user wrote is one the user asked for, and folding it
    # is exactly what the same program does in Python. Nothing about the compiler's convenience stops a fold, so an
    # all-constant integer expression always disappears -- no machine word ever has to hold it.
    builder = HirBuilder()
    builder.block()
    shifted = builder.operation(IntShiftLeft(), [builder.int_const(1), builder.int_const(20_000)])
    builder.output(
        "y", builder.operation(IntToFloat(), [builder.operation(IntBwAnd(), [shifted, builder.int_const(0)])])
    )
    builder.ret()

    raw = builder.finish()
    hir = optimize(raw, DEFAULT_IFCONV_MAX_OPS)
    assert not [node for node in hir.nodes.values() if isinstance(node, Operation)]
    assert hir.nodes[hir.outputs[0].value] == FloatConst(0.0)
    lower_to_mir(raw, OPS, FMT, default_ifmt(FMT), DEFAULT_IFCONV_MAX_OPS)  # nothing integer survives


def _reduced_int_outputs(populate: Callable[[HirBuilder, ValueId], None]) -> dict[str, object]:
    """One integer input ``n`` through ``optimize``; each named output resolved to its surviving node."""
    builder = HirBuilder()
    builder.block()
    populate(builder, builder.input("n", IntType()))
    builder.ret()
    hir = optimize(builder.finish(), DEFAULT_IFCONV_MAX_OPS)
    return {out.name: hir.nodes[out.value] for out in hir.outputs}


def test_the_integer_subtraction_rules_the_shared_algebra_cannot_state() -> None:
    # The declared algebra drops an identity operand from either side, which for subtraction is only true of the
    # right one: ``x - 0`` is ``x`` while ``0 - x`` is the negation, so each direction is its own rule.
    def populate(builder: HirBuilder, n: ValueId) -> None:
        zero = builder.int_const(0)
        builder.output("dropped", builder.operation(IntSub(), [n, zero]))
        builder.output("negated", builder.operation(IntSub(), [zero, n]))
        builder.output("cancelled", builder.operation(IntSub(), [n, n]))

    outputs = _reduced_int_outputs(populate)
    assert isinstance(outputs["dropped"], InPort)
    assert isinstance(outputs["negated"], Operation) and outputs["negated"].operator == IntNeg()
    assert outputs["cancelled"] == IntConst(0)


def test_a_power_of_two_integer_product_mints_the_saturating_scaling() -> None:
    # The exponent is absorbed into the operator from either side, and the constant -- even one no machine word
    # holds -- goes dead with it, so it is never asked to materialize.
    def populate(builder: HirBuilder, n: ValueId) -> None:
        builder.output("right", builder.operation(IntMul(), [n, builder.int_const(8)]))
        builder.output("left", builder.operation(IntMul(), [builder.int_const(2**40), n]))

    outputs = _reduced_int_outputs(populate)
    assert isinstance(outputs["right"], Operation) and outputs["right"].operator == IntMulPow2(3)
    assert isinstance(outputs["left"], Operation) and outputs["left"].operator == IntMulPow2(40)


def test_integer_negations_share_one_tracking_across_their_spellings() -> None:
    # ``x * -1``, ``x // -1`` and the written negation all name one node, ``-(-x)`` returns the base, and a sum
    # of a value with its own negation is zero without anything firing.
    def populate(builder: HirBuilder, n: ValueId) -> None:
        neg = builder.operation(IntNeg(), [n])
        builder.output("restored", builder.operation(IntNeg(), [neg]))
        builder.output("cancelled", builder.operation(IntAdd(), [n, neg]))
        builder.output("by_product", builder.operation(IntMul(), [n, builder.int_const(-1)]))
        builder.output("by_quotient", builder.operation(IntDivFloor(), [n, builder.int_const(-1)]))

    outputs = _reduced_int_outputs(populate)
    assert isinstance(outputs["restored"], InPort)
    assert outputs["cancelled"] == IntConst(0)
    assert outputs["by_product"] == outputs["by_quotient"]
    assert isinstance(outputs["by_product"], Operation) and outputs["by_product"].operator == IntNeg()


def test_integer_division_and_remainder_reduce_against_their_constants() -> None:
    def populate(builder: HirBuilder, n: ValueId) -> None:
        builder.output("q_one", builder.operation(IntDivFloor(), [n, builder.int_const(1)]))
        builder.output("q_pow2", builder.operation(IntDivFloor(), [n, builder.int_const(8)]))
        builder.output("q_self", builder.operation(IntDivFloor(), [n, n]))
        builder.output("q_zero", builder.operation(IntDivFloor(), [builder.int_const(0), n]))
        builder.output("r_one", builder.operation(IntMod(), [n, builder.int_const(1)]))
        builder.output("r_neg_one", builder.operation(IntMod(), [n, builder.int_const(-1)]))
        builder.output("r_pow2", builder.operation(IntMod(), [n, builder.int_const(8)]))
        builder.output("r_self", builder.operation(IntMod(), [n, n]))
        builder.output("r_zero", builder.operation(IntMod(), [builder.int_const(0), n]))

    outputs = _reduced_int_outputs(populate)
    assert isinstance(outputs["q_one"], InPort)
    assert isinstance(outputs["q_pow2"], Operation) and outputs["q_pow2"].operator == IntShiftRight()
    assert outputs["q_self"] == IntConst(1)
    assert outputs["q_zero"] == outputs["r_one"] == outputs["r_neg_one"] == IntConst(0)
    assert outputs["r_self"] == outputs["r_zero"] == IntConst(0)
    mask = outputs["r_pow2"]
    assert isinstance(mask, Operation) and mask.operator == IntBwAnd()


def test_bitwise_value_equality_and_complement_rules() -> None:
    def populate(builder: HirBuilder, n: ValueId) -> None:
        inverted = builder.operation(IntBwNot(), [n])
        builder.output("xor_self", builder.operation(IntBwXor(), [n, n]))
        builder.output("and_self", builder.operation(IntBwAnd(), [n, n]))
        builder.output("or_self", builder.operation(IntBwOr(), [n, n]))
        builder.output("complement", builder.operation(IntBwXor(), [n, builder.int_const(-1)]))
        builder.output("restored", builder.operation(IntBwNot(), [inverted]))
        builder.output("annihilated", builder.operation(IntBwAnd(), [n, inverted]))
        builder.output("saturated", builder.operation(IntBwOr(), [n, inverted]))
        builder.output("disagreed", builder.operation(IntBwXor(), [n, inverted]))

    outputs = _reduced_int_outputs(populate)
    assert outputs["xor_self"] == outputs["annihilated"] == IntConst(0)
    assert isinstance(outputs["and_self"], InPort) and isinstance(outputs["or_self"], InPort)
    assert isinstance(outputs["complement"], Operation) and outputs["complement"].operator == IntBwNot()
    assert isinstance(outputs["restored"], InPort)
    assert outputs["saturated"] == outputs["disagreed"] == IntConst(-1)


@pytest.mark.parametrize(
    "relation, expected",
    [
        (IntEqual(), True),
        (IntLessOrEqual(), True),
        (IntGreaterOrEqual(), True),
        (IntNotEqual(), False),
        (IntLess(), False),
        (IntGreater(), False),
    ],
)
def test_a_reflexive_integer_comparison_folds_to_its_truth(relation: Operator, expected: bool) -> None:
    # No integer is a NaN, so every relation is decided over equal operands without seeing their value.
    def populate(builder: HirBuilder, n: ValueId) -> None:
        builder.output("y", builder.operation(relation, [n, n]))

    assert _reduced_int_outputs(populate)["y"] == BoolConst(expected)


def test_the_boolean_connectives_fold_over_equal_operands() -> None:
    builder = HirBuilder()
    builder.block()
    b = builder.input("b", BoolType())
    builder.output("and_self", builder.operation(BoolAnd(), [b, b]))
    builder.output("or_self", builder.operation(BoolOr(), [b, b]))
    builder.output("xor_self", builder.operation(BoolXor(), [b, b]))
    builder.output("inverted", builder.operation(BoolXor(), [b, builder.bool_const(True)]))
    builder.ret()
    hir = optimize(builder.finish(), DEFAULT_IFCONV_MAX_OPS)
    outputs = {out.name: hir.nodes[out.value] for out in hir.outputs}
    assert isinstance(outputs["and_self"], InPort) and isinstance(outputs["or_self"], InPort)
    assert outputs["xor_self"] == BoolConst(False)
    assert isinstance(outputs["inverted"], Operation) and outputs["inverted"].operator == BoolNot()


def test_an_integer_self_division_erases_an_operand_that_names_no_number() -> None:
    # The integer dual of the float rule above: ``5 // 0`` has no value for the fold, so it is an operand the
    # compiler cannot see, and ``q // q`` and ``q % q`` speak for it whatever it turns out to be -- the division
    # reduces, its operand goes dead, and nothing is left for the survivor sweep to convict.
    builder = HirBuilder()
    builder.block()
    q = builder.operation(IntDivFloor(), [builder.int_const(5), builder.int_const(0)])
    builder.output("quotient", builder.operation(IntDivFloor(), [q, q]))
    builder.output("remainder", builder.operation(IntMod(), [q, q]))
    builder.ret()
    hir = optimize(builder.finish(), DEFAULT_IFCONV_MAX_OPS)
    assert hir.nodes[hir.outputs[0].value] == IntConst(1)
    assert hir.nodes[hir.outputs[1].value] == IntConst(0)
    assert not [node for node in hir.nodes.values() if isinstance(node, Operation)]


@pytest.mark.parametrize(
    "operator, operands, expected",
    [
        (FloatFma(), (2.0, 3.0, 4.0), 10.0),
        (FloatExp2(), (math.inf,), math.inf),
        (FloatExp2(), (-math.inf,), 0.0),
        (FloatExp2(), (1e10,), math.inf),  # past the carrier, upward only
        (FloatLog2(), (math.inf,), math.inf),
        (FloatSqrt(), (math.inf,), math.inf),
        (FloatHypot2(), (math.inf, 1.0), math.inf),
        (FloatHypot2(), (1.5e308, 1.5e308), math.inf),  # math.hypot saturates rather than raising, so the fold does
        (FloatHypot2(), (-math.inf, 1.0), math.inf),  # a magnitude is never negative
        (FloatAtan2(), (math.inf, math.inf), math.pi / 4.0),
        (FloatFloor(), (2.7,), 2.0),
        (FloatTrunc(), (-2.7,), -2.0),
        (FloatSin(), (0.0,), 0.0),
    ],
)
def test_a_float_fold_takes_the_ideal_result_over_the_extended_domain(
    operator: Operator, operands: tuple[float, ...], expected: float
) -> None:
    # The transcendental and rounding folds take the ideal result over the whole extended domain, infinities included;
    # nothing here consults a numeric format, and overflow becomes the mathematically right infinity.
    hir = _optimized_constant_operation(operator, *operands)
    assert hir.nodes[hir.outputs[0].value] == FloatConst(expected)


def _wrapped_infinity_times_zero(wrapper: Operator) -> Hir:
    builder = HirBuilder()
    builder.block()
    wrapped = builder.operation(wrapper, [builder.float_const(math.inf)])
    builder.output("y", builder.operation(FloatMul(), [wrapped, builder.float_const(0.0)]))
    builder.ret()
    hir = optimize(builder.finish(), DEFAULT_IFCONV_MAX_OPS)
    refuse(hir)
    return hir


def test_an_identity_cannot_be_dodged_by_spelling_its_operand_as_an_expression() -> None:
    # "Known" has to mean known, not "spelled as a constant node". ``abs(inf)``, ``floor(inf)``, ``trunc(inf)``
    # each IS the infinity, however written, so the product is the same indeterminate form as ``inf * 0.0`` and
    # survives to be refused.
    for spelled in (FloatAbs(), FloatFloor(), FloatTrunc()):
        with pytest.raises(SynthesisError, match="names no number"):
            _wrapped_infinity_times_zero(spelled)
    # The other side of the same rule, and the reason it is not a dodge: ``sin(inf)`` names NO number, so it is an
    # operand the compiler cannot see, and the absorbing zero claims it exactly as it claims a runtime one. The wrapper
    # is deleted by that rewrite, so nothing naming no number survives -- a refusal missed under the charter's license,
    # never a wrong answer, and the alternative would be an identity that consults how its operand was produced.
    hir = _wrapped_infinity_times_zero(FloatSin())
    assert hir.nodes[hir.outputs[0].value] == FloatConst(0.0)


def test_a_self_division_erases_an_operand_that_names_no_number() -> None:
    # ``x/x == 1`` speaks for an operand the compiler cannot see, and ``inf + -inf`` is one: the fold has no value for
    # it. So the quotient reduces and the sum goes with it, leaving nothing for the survivor sweep to convict.
    builder = HirBuilder()
    builder.block()
    settled = builder.operation(FloatAdd(), [builder.float_const(math.inf), builder.float_const(-math.inf)])
    builder.output("y", builder.operation(HirFloatDiv(), [settled, settled]))
    builder.ret()
    hir = optimize(builder.finish(), DEFAULT_IFCONV_MAX_OPS)
    assert hir.nodes[hir.outputs[0].value] == FloatConst(1.0)
    assert not [node for node in hir.nodes.values() if isinstance(node, Operation)]


def _merge_of(builder: HirBuilder, arm: Callable[[HirBuilder], ValueId]) -> ValueId:
    """A phi merging two separately-built arms behind a runtime condition -- the shape a plain ``if/else`` leaves."""
    entry = builder.current_block
    then_block, else_block, merge = builder.block(), builder.block(), builder.block()
    builder.position_at(entry)
    builder.branch(builder.input("c", BoolType()), then_block, else_block)
    builder.position_at(then_block)
    taken = arm(builder)
    builder.jump(merge)
    builder.position_at(else_block)
    other = arm(builder)
    builder.jump(merge)
    builder.position_at(merge)
    return builder.phi(builder.type_of(taken), [(then_block, taken), (else_block, other)])


@pytest.mark.parametrize(
    "operator, other",
    [(FloatAdd(), -math.inf), (HirFloatDiv(), math.inf)],
    ids=["inf_minus_inf", "inf_over_inf"],
)
def test_an_operand_known_only_through_a_merge_still_blocks_the_identity(operator: Operator, other: float) -> None:
    # An identity may only claim an operand the compiler cannot see, and a merge of two arms that agree is one it can:
    # the folder names the merge, and the round after that the identity no longer applies. Which spellings it gets to
    # in time is a matter of how far the analysis reaches, and the charter promises no particular reach -- an identity
    # winning a race against the folder costs an unfolded diagnosis, never a wrong answer.
    builder = HirBuilder()
    builder.block()
    merged = _merge_of(builder, lambda b: b.operation(FloatMul(), [b.input("x", HirFloatType()), b.float_const(0.0)]))
    infinite = builder.operation(FloatAdd(), [merged, builder.float_const(math.inf)])  # 0.0 + inf, known to be inf
    builder.output("y", builder.operation(operator, [infinite, builder.float_const(other)]))
    builder.ret()
    with pytest.raises(SynthesisError, match="names no number"):
        refuse(optimize(builder.finish(), DEFAULT_IFCONV_MAX_OPS))


@pytest.mark.parametrize(
    "operator, expected",
    [
        (FloatIsFinite(), [True, True, False, False]),
        (FloatIsInf(), [False, False, True, True]),
        (FloatIsPosInf(), [False, False, True, False]),
        (FloatIsNegInf(), [False, False, False, True]),
    ],
    ids=["isfinite", "isinf", "isposinf", "isneginf"],
)
def test_float_classification_folds_at_host_precision(operator: Operator, expected: list[bool]) -> None:
    # Left to MIR these fold AFTER conversion into the target format, so 1e100 classified as non-finite in a narrow
    # build -- a constant whose value depends on the selected hardware, which the charter forbids. Folding in HIR
    # settles them in the compiler's own arithmetic instead.
    for operand, want in zip([0.0, 1e100, math.inf, -math.inf], expected, strict=True):
        hir = _optimized_constant_operation(operator, operand)
        assert hir.nodes[hir.outputs[0].value] == BoolConst(want), (operator, operand)


def test_a_float_to_bool_cast_folds_at_host_precision() -> None:
    # A magnitude the configured format would flush to zero is nonzero to the compiler, and the compiler's answer is
    # the one that stands: the cast folds True.
    hir = _optimized_constant_operation(FloatToBool(), 2.0**-200)
    assert hir.nodes[hir.outputs[0].value] == BoolConst(True)


def test_a_constant_condition_selects_a_mux_arm_in_every_scalar_family() -> None:
    # A mux whose select line never moves is not a mux. The three families are the same identity, so leaving any one
    # of them out would be exactly the special-casing the design forbids.
    builder = HirBuilder()
    builder.block()
    x = builder.input("x", HirFloatType())
    runtime_int = builder.operation(FloatToInt(), [x])
    runtime_bool = builder.operation(FloatToBool(), [x])
    true, false = builder.bool_const(True), builder.bool_const(False)
    builder.output("float", builder.operation(FloatSelect(), [true, builder.float_const(3.0), x]))
    builder.output("bool", builder.operation(BoolSelect(), [false, runtime_bool, builder.bool_const(True)]))
    builder.output(
        "int",
        builder.operation(IntToFloat(), [builder.operation(IntSelect(), [true, builder.int_const(7), runtime_int])]),
    )
    builder.ret()

    hir = optimize(builder.finish(), DEFAULT_IFCONV_MAX_OPS)
    chosen = {out.name: hir.nodes[out.value] for out in hir.outputs}
    assert chosen["float"] == FloatConst(3.0)
    assert chosen["bool"] == BoolConst(True)
    assert chosen["int"] == FloatConst(7.0)
    assert not [node for node in hir.nodes.values() if isinstance(node, Operation)]


def test_the_integer_float_round_trip_is_not_an_identity() -> None:
    # ``int(float(n)) == n`` is NOT an axiom here, so the nest stands over an operand the compiler cannot see. The
    # carrier may be coarser than the integer, and then the round trip ROUNDS -- exactness is promised only where a
    # float holds the integer, and an identity may claim nothing weaker than every value. The constant case shows the
    # rounding the axiom would have had to deny.
    builder = HirBuilder()
    builder.block()
    n = builder.input("n", IntType())
    builder.output("y", builder.operation(FloatToInt(), [builder.operation(IntToFloat(), [n])]))
    builder.ret()
    hir = optimize(builder.finish(), DEFAULT_IFCONV_MAX_OPS)
    assert [node.operator for node in hir.nodes.values() if isinstance(node, Operation)] == [IntToFloat(), FloatToInt()]

    def constant_round_trip(value: int) -> Hir:
        builder = HirBuilder()
        builder.block()
        builder.output(
            "y", builder.operation(FloatToInt(), [builder.operation(IntToFloat(), [builder.int_const(value)])])
        )
        builder.ret()
        return optimize(builder.finish(), DEFAULT_IFCONV_MAX_OPS)

    assert constant_round_trip(5).nodes[constant_round_trip(5).outputs[0].value] == IntConst(5)
    rounded = constant_round_trip(2**53 + 1)
    assert rounded.nodes[rounded.outputs[0].value] == IntConst(2**53)  # exactly what Python answers
    assert int(float(2**53 + 1)) == 2**53


def test_the_float_integer_round_trip_is_a_truncation() -> None:
    # float(int(x)) == trunc(x) for every x, and a rounding over an already-integral value is the identity, so the
    # nest collapses to one truncation and a truncation of a floor collapses to the floor.
    builder = HirBuilder()
    builder.block()
    x = builder.input("x", HirFloatType())
    builder.output("round_trip", builder.operation(IntToFloat(), [builder.operation(FloatToInt(), [x])]))
    builder.output("idempotent", builder.operation(FloatTrunc(), [builder.operation(FloatFloor(), [x])]))
    builder.ret()

    hir = optimize(builder.finish(), DEFAULT_IFCONV_MAX_OPS)
    operators = sorted((n.operator for n in hir.nodes.values() if isinstance(n, Operation)), key=lambda op: op.mnemonic)
    assert operators == [FloatFloor(), FloatTrunc()]
    assert hir.nodes[hir.outputs[0].value] == Operation(FloatTrunc(), (hir.input_ids[0],))
    assert hir.nodes[hir.outputs[1].value] == Operation(FloatFloor(), (hir.input_ids[0],))


def _optimized_constant_operation(operator: Operator, *operands: float | int) -> Hir:
    """``operator(consts...)`` wired to an output so DCE keeps it, run through the whole HIR pipeline."""
    builder = HirBuilder()
    builder.block()
    values = [
        builder.int_const(operand) if isinstance(operand, int) else builder.float_const(operand) for operand in operands
    ]
    result = builder.operation(operator, values)
    if builder.type_of(result) == IntType():
        result = builder.operation(IntToFloat(), [result])
    builder.output("y", result)
    builder.ret()
    hir = optimize(builder.finish(), DEFAULT_IFCONV_MAX_OPS)
    refuse(hir)  # the gate is the boundary's, so a refusal expectation must reach past optimization for it
    return hir


def test_selection_cannot_be_reached_without_passing_the_gate() -> None:
    # Were selection callable on its own, a caller could optimize, judge, substitute, and lower -- convicting what
    # its own later round would have erased.
    builder = HirBuilder()
    builder.block()
    builder.output("y", builder.operation(HirFloatDiv(), [builder.float_const(1.0), builder.float_const(0.0)]))
    builder.ret()
    with pytest.raises(SynthesisError, match="names no number"):
        lower_to_mir(builder.finish(), OPS, FMT, default_ifmt(FMT), DEFAULT_IFCONV_MAX_OPS)


@pytest.mark.parametrize(
    "operator, operands",
    [
        (FloatLog2(), (-2.0,)),
        (FloatSqrt(), (-1.0,)),
        (FloatSin(), (math.inf,)),
        (FloatCos(), (-math.inf,)),
        (FloatAdd(), (math.inf, -math.inf)),
        (FloatMul(), (0.0, math.inf)),
        (HirFloatDiv(), (math.inf, math.inf)),
        (HirFloatDiv(), (0.0, 0.0)),
        (FloatFma(), (math.inf, 0.0, 1.0)),
        (FloatFma(), (math.inf, 1.0, -math.inf)),
        (IntDivFloor(), (7, 0)),
        (IntMod(), (7, 0)),
        (IntShiftLeft(), (1, -1)),
        (IntShiftRight(), (1, -1)),
        (FloatFma(), (1e300, 1e300, 0.0)),
    ],
    ids=[
        "log2_negative",
        "sqrt_negative",
        "sin_infinite",
        "cos_infinite",
        "inf_minus_inf",
        "zero_times_inf",
        "inf_over_inf",
        "zero_over_zero",
        "fma_inf_times_zero",
        "fma_inf_minus_inf",
        "int_div_zero",
        "int_mod_zero",
        "shift_left_negative",
        "shift_right_negative",
        "fma_past_the_carrier",
    ],
)
def test_an_operation_outside_its_mathematical_domain_is_refused(
    operator: Operator, operands: tuple[object, ...]
) -> None:
    # The criterion is first-principles arithmetic: none of these expressions names a number, whatever the datapath
    # would answer if asked. How the host signals it -- a raised exception for some, a NaN for others -- is an accident
    # of the host and no part of the rule.
    with pytest.raises(SynthesisError, match="names no number"):
        _optimized_constant_operation(operator, *operands)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "operator, operand, expected",
    [
        (FloatLog2(), 0.0, -math.inf),
        (FloatFloor(), math.inf, math.inf),
        (FloatRound(), -math.inf, -math.inf),
    ],
    ids=["log2_zero", "floor_of_infinity", "round_of_infinity"],
)
def test_a_pole_the_operator_reference_answers_folds_to_that_value(
    operator: Operator, operand: float, expected: float
) -> None:
    # Each evaluate follows its own registered np reference: np.log2 answers the
    # one-sided -inf limit at the pole like the RTL, and an infinity passes through the rounding family.
    # These DO name the value the hardware computes, so the fold hands it back instead of refusing.
    hir = _optimized_constant_operation(operator, operand)
    assert hir.nodes[hir.outputs[0].value] == FloatConst(expected)


def _never_returns(x: float) -> float:
    y = x
    while (x * 0.0) <= 0.0:
        y = y + 1.0
    return y


def _never_returns_through_a_loop_phi(x: float) -> float:
    # The guard reads a loop-header phi, so only the round after the latch arm is rebuilt can decide it.
    y = x * 0.0
    while y == 0.0:
        y = x * 0.0
    return x


@pytest.mark.parametrize("kernel", [_never_returns, _never_returns_through_a_loop_phi], ids=lambda fn: fn.__name__[1:])
def test_a_kernel_that_provably_never_returns_is_refused(kernel: Callable[..., float]) -> None:
    with pytest.raises(UnsupportedConstruct, match="never returns"):
        optimize(lower(kernel).hir, DEFAULT_IFCONV_MAX_OPS)


def test_an_arm_a_proven_guard_excludes_is_deleted_rather_than_merely_unconvicted() -> None:
    # The dual of the above: the quotient is not spared a conviction, it is not there. A division is unspeculatable,
    # so nothing but pruning can remove it.
    def excluded_by_a_guard(x: float) -> float:
        r = x
        if (x * 0.0) > 1.0:
            r = 1.0 / (x * 0.0)
        return r

    hir = optimize(lower(excluded_by_a_guard).hir, DEFAULT_IFCONV_MAX_OPS)
    assert not [
        node for node in hir.nodes.values() if isinstance(node, Operation) and isinstance(node.operator, HirFloatDiv)
    ], "the excluded arm's quotient survived the pruning that proves its guard"
    assert not [block for block in hir.blocks if isinstance(block.terminator, Branch)], "the guard itself survived"
    assert hir.nodes[hir.outputs[0].value] == hir.nodes[hir.input_ids[0]], "the surviving path must return x itself"


def test_a_loop_whose_test_is_proven_false_dissolves_entirely() -> None:
    # Only the graph's ``x*0 == 0`` identity decides this test, so the front end residualizes a real loop.
    def never_enters(x: float) -> float:
        y = x
        while (y * 0.0) > 1.0:
            y = y + 1.0
        return y

    hir = optimize(lower(never_enters).hir, DEFAULT_IFCONV_MAX_OPS)
    assert not [block for block in hir.blocks if isinstance(block.terminator, Branch)], "the loop test survived"
    assert not [
        node for node in hir.nodes.values() if isinstance(node, Operation)
    ], "the body's addition survived a loop that is never entered"
    assert hir.nodes[hir.outputs[0].value] == hir.nodes[hir.input_ids[0]]


def test_pruning_one_guard_settles_the_next() -> None:
    # Why reduction and pruning are a mutual fixpoint and not a sequence: the second guard is undecidable until the
    # first arm is gone, so one pass of each leaves it standing.
    def cascade(x: float) -> float:
        r = 1.0
        if (x * 0.0) > 1.0:
            r = 2.0
        if r > 1.5:
            return 3.0
        return 4.0

    hir = optimize(lower(cascade).hir, DEFAULT_IFCONV_MAX_OPS)
    assert not [
        block for block in hir.blocks if isinstance(block.terminator, Branch)
    ], "both guards must be gone, not merely the first"
    assert hir.nodes[hir.outputs[0].value] == FloatConst(4.0)


def test_a_state_slot_live_out_follows_a_merge_pruning_collapses() -> None:
    # A slot's live-out is the one reference outside the value DAG entirely, so a collapse reaches it only by hand.
    class HeldByADeadGuard:
        def __init__(self) -> None:
            self.s = 0.0

        def __call__(self, x: float) -> float:
            if (x * 0.0) > 1.0:
                self.s = x
            return self.s

    hir = optimize(lower(HeldByADeadGuard().__call__).hir, DEFAULT_IFCONV_MAX_OPS)
    (slot,) = hir.state_slots
    assert hir.nodes[slot.live_out] == StateRead("s", HirFloatType()), "the slot must carry its own live-in forward"
    assert not [block for block in hir.blocks if isinstance(block.terminator, Branch)]


def test_a_proven_break_kills_the_back_edge_and_collapses_the_carried_merges() -> None:
    # The latch becomes unreachable, so the header's loop-carried phis lose their latch arm and collapse. Distinct
    # from a loop deleted whole, where no merge has to be repaired at all.
    def breaks_on_the_first_trip(x: float, n: float) -> float:
        y = x
        t = n
        while t > 0.0:
            y = y + 1.0
            if (x * 0.0) <= 1.0:
                break
            t = t - 1.0
        return y

    hir = optimize(lower(breaks_on_the_first_trip).hir, DEFAULT_IFCONV_MAX_OPS)
    assert (
        len([block for block in hir.blocks if isinstance(block.terminator, Branch)]) == 1
    ), "only the loop's own runtime test may survive; the break's guard is decided"
    (phi,) = [node for node in hir.nodes.values() if isinstance(node, Phi)]
    arms = sorted((hir.nodes[arm] for _pred, arm in phi.arms), key=lambda node: isinstance(node, Operation))
    assert [type(arm) for arm in arms] == [InPort, Operation], "the merge must be the untaken test against one trip"


def test_a_decided_branch_whose_arms_share_a_target_repairs_by_dropping_an_arm() -> None:
    # The untaken edge's successor survives via another edge, so nothing is deleted and the repair is the arm filter
    # alone. Built directly: the front end gives a merge two predecessors, and three is what leaves a phi behind.
    builder = HirBuilder()
    entry, dead, live, other, merge = (builder.block() for _ in range(5))
    builder.position_at(entry)
    x = builder.input("x", HirFloatType())
    flag = builder.input("flag", BoolType())
    builder.branch(builder.bool_const(False), dead, live)
    builder.position_at(dead)
    builder.jump(merge)
    builder.position_at(live)
    builder.branch(flag, other, merge)
    builder.position_at(other)
    builder.jump(merge)
    builder.position_at(merge)
    builder.output(
        "y",
        builder.phi(HirFloatType(), [(dead, builder.float_const(1.0)), (live, x), (other, builder.float_const(3.0))]),
    )
    builder.ret()

    hir = optimize(builder.finish(), 0)  # a zero budget keeps the surviving diamond a real branch, so a phi remains
    (phi,) = [node for node in hir.nodes.values() if isinstance(node, Phi)]
    assert [hir.nodes[arm] for _pred, arm in phi.arms] == [
        InPort("x", HirFloatType()),
        FloatConst(3.0),
    ], "the dead predecessor's arm must go and the other two must stay"


def never_uniform_until_the_latch_arm_is_rebuilt(x: float, n: float) -> float:
    step = x * 0.0 + 1.0
    t = n
    while t > 0.0:
        step = x * 0.0 + 1.0
        t = t - 1.0
    return x + step


@pytest.mark.parametrize(
    "kernel",
    [
        branch_boundary_kernel,
        const_branch_kernel,
        diamond_then_loop_kernel,
        overlap_spill_kernel,
        phi_swap_loop,
        equal_temperament,  # pruning leaves a merge only threading normalizes into a diamond, so one round misses it
        never_uniform_until_the_latch_arm_is_rebuilt,
    ],
    ids=lambda fn: fn.__name__,
)
def test_optimization_reaches_a_fixpoint_in_one_call(kernel: Callable[..., object]) -> None:
    # Compared whole, value ids included, since those drive the tie-breaks downstream.
    once = optimize(lower(kernel).hir, DEFAULT_IFCONV_MAX_OPS)
    assert optimize(once, DEFAULT_IFCONV_MAX_OPS) == once


def test_a_loop_phi_is_folded_once_its_latch_arm_has_been_rebuilt() -> None:
    # A phi is opened before its latch arm exists, so the round that emits it cannot see that every arm is one value.
    hir = optimize(lower(never_uniform_until_the_latch_arm_is_rebuilt).hir, DEFAULT_IFCONV_MAX_OPS)
    assert [node.type for node in hir.nodes.values() if isinstance(node, Phi)] == [HirFloatType()]  # the counter's
    out = hir.nodes[hir.outputs[0].value]
    assert isinstance(out, Operation) and [hir.nodes[o] for o in out.operands] == [
        InPort("x", HirFloatType()),
        FloatConst(1.0),
    ]


def test_a_collapsed_merge_is_substituted_into_a_later_merge() -> None:
    # Every later reference to a collapsed phi must follow, including an arm of a DIFFERENT block's phi -- the one
    # reference kind that is neither operand nor terminator. Built directly: the front end never emits this shape.
    builder = HirBuilder()
    entry, dead, live, first, left, right, second = (builder.block() for _ in range(7))
    builder.position_at(entry)
    x = builder.input("x", HirFloatType())
    flag = builder.input("flag", BoolType())
    builder.branch(builder.bool_const(False), dead, live)
    for arm in (dead, live):
        builder.position_at(arm)
        builder.jump(first)
    builder.position_at(first)
    merged = builder.phi(HirFloatType(), [(dead, builder.float_const(1.0)), (live, x)])
    builder.branch(flag, left, right)
    for arm in (left, right):
        builder.position_at(arm)
        builder.jump(second)
    builder.position_at(second)
    builder.output("y", builder.phi(HirFloatType(), [(left, merged), (right, builder.float_const(2.0))]))
    builder.ret()

    hir = optimize(builder.finish(), DEFAULT_IFCONV_MAX_OPS)
    out = hir.nodes[hir.outputs[0].value]
    assert isinstance(out, Operation) and isinstance(out.operator, FloatSelect)
    condition, chosen, other = out.operands
    assert [hir.nodes[condition], hir.nodes[chosen], hir.nodes[other]] == [
        InPort("flag", BoolType()),
        InPort("x", HirFloatType()),
        FloatConst(2.0),
    ], "the collapsed merge must reach the later merge as the value it stood for, not as a dangling reference"


@pytest.mark.parametrize(
    "value,degrades",
    [
        (1e-12, True),
        (-1e-12, True),
        (1e30, True),
        (-1e30, True),
        (0.0, False),
        (math.inf, False),
        (-math.inf, False),
        (1.0, False),
        (9.313225746154785e-10, False),  # the smallest normal
        (4.656612873077393e-10, False),  # half of it, which ties upward to the smallest normal
        (math.nextafter(4.656612873077393e-10, 0.0), True),  # one ulp below that tie, which encodes to nothing
    ],
    ids=lambda value: repr(value),
)
def test_a_literal_the_format_cannot_hold_is_refused_rather_than_silently_degraded(
    value: float, degrades: bool
) -> None:
    # The float dual of the integer range refusal. An infinity is representable and so is accepted, where a finite
    # value that encodes to one -- or to zero -- is a literal the machine cannot hold, and substituting what it
    # encodes to would answer for a number the kernel did not write.
    builder = HirBuilder()
    builder.block()
    builder.output("y", builder.operation(FloatAdd(), [builder.input("x", HirFloatType()), builder.float_const(value)]))
    builder.ret()
    if degrades:
        with pytest.raises(UnsupportedConstruct, match="degrades"):
            lower_to_mir(builder.finish(), OPS, FMT, default_ifmt(FMT), DEFAULT_IFCONV_MAX_OPS)
    else:
        lower_to_mir(builder.finish(), OPS, FMT, default_ifmt(FMT), DEFAULT_IFCONV_MAX_OPS)


def test_a_divisor_whose_reciprocal_degrades_is_never_silently_applied() -> None:
    # 3e9 is representable but its reciprocal is not, and the reciprocal is what HIR's ``x/c -> x*(1/c)`` hands the
    # machine, which would multiply by zero and answer zero for every input.
    builder = HirBuilder()
    builder.block()
    x = builder.input("x", HirFloatType())
    builder.output("y", builder.operation(HirFloatDiv(), [x, builder.float_const(3e9)]))
    builder.ret()
    with pytest.raises(UnsupportedConstruct, match="degrades"):
        lower_to_mir(builder.finish(), OPS, FMT, default_ifmt(FMT), DEFAULT_IFCONV_MAX_OPS)


def test_a_state_slot_resetting_to_a_value_the_format_cannot_hold_is_refused() -> None:
    # A reset snapshot never becomes a pooled constant, so the node-level rule never sees it.
    builder = HirBuilder()
    builder.block()
    x = builder.input("x", HirFloatType())
    builder.state_slot("s", FloatConst(1e-12), x)
    builder.output("y", x)
    builder.ret()
    with pytest.raises(UnsupportedConstruct, match="degrades"):
        lower_to_mir(builder.finish(), OPS, FMT, default_ifmt(FMT), DEFAULT_IFCONV_MAX_OPS)


def test_a_float_slot_with_an_integer_reset_is_refused() -> None:
    # The slot's live-out is a float while its reset snapshot is an integer: a slot register holds one family, and
    # only a sweep over the slots themselves sees the mismatch, since no node in the graph carries it.
    builder = HirBuilder()
    builder.block()
    x = builder.input("x", HirFloatType())
    builder.state_slot("s", IntConst(0), x)
    builder.output("y", x)
    builder.ret()
    with pytest.raises(UnsupportedConstruct, match="holds FloatType.. but resets to IntType"):
        lower_to_mir(builder.finish(), OPS, FMT, default_ifmt(FMT), DEFAULT_IFCONV_MAX_OPS)


def test_a_bselect_repeating_its_condition_reduces_to_a_gate() -> None:
    # ``if c: r = a`` over a boolean leaves bselect(c, a, c) -- Python's eager ``and`` shape written as a branch --
    # and its dual leaves bselect(c, c, b). Both are pure gates, so the mux must not reach the hardware.
    def and_shape(c: bool, a: bool) -> bool:
        r = c
        if c:
            r = a
        return r

    def or_shape(c: bool, b: bool) -> bool:
        r = b
        if c:
            r = c
        return r

    for kernel, gate in ((and_shape, BoolAnd), (or_shape, BoolOr)):
        operators = [node.operator for node in _hir_of(kernel).nodes.values() if isinstance(node, Operation)]
        assert not any(isinstance(op, BoolSelect) for op in operators), f"{kernel.__name__}: the mux survived"
        assert any(isinstance(op, gate) for op in operators), f"{kernel.__name__}: expected a {gate.__name__}"
        # The shape alone cannot tell an `and` rewritten as an `or`; only the truth table can.
        sim = build_model(build_lir(_run(kernel), kernel.__name__))
        for c in (False, True):
            for other in (False, True):
                assert sim.run(c, other)[0] is kernel(c, other), f"{kernel.__name__}({c}, {other})"
