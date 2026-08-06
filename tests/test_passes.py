"""Unit tests for HIR optimization and MIR selection passes."""

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
from holoso._hir import BoolSelect, FloatDiv as HirFloatDiv, NoNumber, Phi, Ret, FloatSelect
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
from holoso._hir import _if_convert as if_convert_pass
from ._modelref import DEFAULT_IFMT, build_lir
from holoso._mir import lower as lower_to_mir, Mir, MirFloatConst, MirFloatInput, MirFloatOutput, MirOperation
from holoso._operators import FMulILog2Operator, FloatSignControl
from ._importguard import forbidden_imports
from ._modelref import build_model, build_ops

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
    return lower_to_mir(optimize(lower(target).hir), ops, fmt, DEFAULT_IFMT)


def _op_count(mir: Mir, cls: type) -> int:
    return sum(1 for n in mir.nodes.values() if isinstance(n, MirOperation) and isinstance(n.operator, cls))


def _ops(mir: Mir) -> list[MirOperation]:
    return [n for n in mir.nodes.values() if isinstance(n, MirOperation)]


def _consts(mir: Mir) -> list[float]:
    return [n.value for n in mir.nodes.values() if isinstance(n, MirFloatConst)]


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


def test_hir_builder_rejects_wrong_semantic_operand_type() -> None:
    builder = HirBuilder()
    builder.block()
    a = builder.input("a", OtherType())
    b = builder.input("b", HirFloatType())
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
        lower_to_mir(hir, OPS, FMT, DEFAULT_IFMT)
    except UnsupportedConstruct as ex:
        assert "no MIR lowering rule" in str(ex)
    else:
        assert False, "expected HIR-to-MIR lowering to reject non-float semantic input"


def test_hir_constant_folding_returns_float_const() -> None:
    def f() -> float:
        return 1.25 + 2.0

    hir = optimize(lower(f).hir)
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

    hir = optimize(builder.finish())
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

    ops = _ops(_run(f))
    assert len(ops) == 1
    assert isinstance(ops[0].operator, FMulILog2Operator) and ops[0].operator.k == -2


def test_left_const_fmul_pow2_is_commutative() -> None:
    def f(a: float) -> float:
        return 2 * a

    ops = _ops(_run(f))
    assert len(ops) == 1
    assert isinstance(ops[0].operator, FMulILog2Operator) and ops[0].operator.k == 1


def test_div_by_pow2_becomes_ilog2() -> None:
    def f(a: float) -> float:
        return a / 4.0

    ops = _ops(_run(f))
    assert len(ops) == 1
    assert isinstance(ops[0].operator, FMulILog2Operator) and ops[0].operator.k == -2


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
    assert isinstance(selected[0].operator, FMulILog2Operator) and selected[0].operator.k == 4
    assert _consts(mir) == []


def test_unsupported_pow2_shift_is_rejected() -> None:
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
    try:
        _run(f, ops, fmt)
    except UnsupportedConstruct as ex:
        assert "unsupported power-of-two float scale" in str(ex)
    else:
        assert False, "expected an unsupported power-of-two shift"


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
    assert mir.outputs[0].sign == FloatSignControl(absolute=True).then(FloatSignControl(negate=True))


def test_selected_mir_has_only_input_const_operation_nodes() -> None:
    def f(a: float, b: float) -> float:
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
    reduced = optimize(builder.finish())
    out = {o.name: reduced.nodes[o.value] for o in reduced.outputs}
    assert out["or_abs"] == BoolConst(True)  # x or True  -> True   (absorbing)
    assert out["and_abs"] == BoolConst(False)  # x and False -> False  (absorbing)
    assert isinstance(out["or_id"], InPort) and out["or_id"].name == "x"  # x or False -> x  (identity dropped)
    assert isinstance(out["and_id"], InPort) and out["and_id"].name == "x"  # x and True -> x  (identity dropped)


def _hir_of(target: object) -> Hir:
    return optimize(lower(target).hir)


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


def test_if_conversion_respects_the_arm_size_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(if_convert_pass, "_IFCONV_MAX_OPS", 1)

    def f(a: float, b: float) -> float:
        if a > b:
            y = (a + b) * a + b  # three operations: over the per-arm budget of one
        else:
            y = a - b
        return y

    hir = _hir_of(f)
    assert len(hir.blocks) == 4


def test_if_conversion_knob_zero_disables_the_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(if_convert_pass, "_IFCONV_MAX_OPS", 0)

    def f(a: float, b: float) -> float:
        if a > b:
            y = a + b
        else:
            y = a - b
        return y

    hir = _hir_of(f)
    assert len(hir.blocks) == 4 and not any(
        isinstance(n, Operation) and isinstance(n.operator, FloatSelect) for n in hir.nodes.values()
    )


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
        model = build_model(build_lir(lower_to_mir(hir, OPS, FMT, DEFAULT_IFMT), fn.__name__))
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
        model = build_model(build_lir(lower_to_mir(hir, OPS, FMT, DEFAULT_IFMT), kernel.__name__))
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


# The integer vocabulary is deliberately unreachable through the frontend, so it is exercised at the builder level:
# HIR declares the types and operators, and MIR refuses every one of them until the integer backend lands.
_INT_OPERATORS: list[tuple[Operator, list[Type], Type]] = [
    (IntAdd(), [IntType(), IntType()], IntType()),
    (IntSub(), [IntType(), IntType()], IntType()),
    (IntMul(), [IntType(), IntType()], IntType()),
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
    # The signature is enforced, so a wrong-typed operand cannot silently build an ill-typed graph.
    wrong = builder.float_const(2.0) if operand_types[0] != HirFloatType() else builder.int_const(2)
    with pytest.raises(ValueError):
        builder.operation(operator, [wrong, *operands[1:]])


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

    hir = optimize(builder.finish())
    outputs = {out.name: hir.nodes[out.value] for out in hir.outputs}
    assert isinstance(outputs["identities"], InPort), "every identity must drop, leaving the input itself"
    assert outputs["killed"] == outputs["masked_off"] == IntConst(0)
    assert outputs["saturated"] == IntConst(-1)
    assert not [node for node in hir.nodes.values() if isinstance(node, Operation)]


def test_a_constant_integer_expression_folds_away_entirely() -> None:
    # Folding is exact at arbitrary precision -- no width, no saturation -- so a fully static integer expression
    # disappears before MIR, which is what keeps MIR's integer refusal from rejecting programs the design accepts.
    builder = HirBuilder()
    builder.block()
    value = builder.operation(IntAdd(), [builder.int_const(2), builder.int_const(3)])
    value = builder.operation(IntMul(), [value, builder.int_const(2**200)])
    builder.output("y", builder.operation(IntToFloat(), [value]))
    builder.ret()

    hir = optimize(builder.finish())
    assert not [node for node in hir.nodes.values() if isinstance(node, Operation)]
    (out,) = hir.outputs
    assert hir.nodes[out.value] == FloatConst(float(5 * 2**200))
    lower_to_mir(hir, OPS, FMT, DEFAULT_IFMT)  # nothing integer is left, so MIR accepts it


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
        (IntAbs(), [-huge], huge),
        (IntBwNot(), [huge], ~huge),
    ]
    for operator, operands, expected in cases:
        builder = HirBuilder()
        builder.block()
        builder.output("y", builder.operation(operator, [builder.int_const(operand) for operand in operands]))
        builder.ret()
        hir = optimize(builder.finish())
        assert not [node for node in hir.nodes.values() if isinstance(node, Operation)], operator
        assert hir.nodes[hir.outputs[0].value] == IntConst(expected), operator


def test_integer_folding_has_no_size_limit() -> None:
    # A big shift is computed, not declined: an expression the user wrote is one the user asked for, and folding it
    # is exactly what the same program does in Python. Nothing about the compiler's convenience stops a fold, so an
    # all-constant integer expression always disappears -- which is what keeps it away from MIR's integer refusal.
    builder = HirBuilder()
    builder.block()
    shifted = builder.operation(IntShiftLeft(), [builder.int_const(1), builder.int_const(20_000)])
    builder.output(
        "y", builder.operation(IntToFloat(), [builder.operation(IntBwAnd(), [shifted, builder.int_const(0)])])
    )
    builder.ret()

    hir = optimize(builder.finish())
    assert not [node for node in hir.nodes.values() if isinstance(node, Operation)]
    assert hir.nodes[hir.outputs[0].value] == FloatConst(0.0)
    lower_to_mir(hir, OPS, FMT, DEFAULT_IFMT)  # nothing integer survives, however wide the intermediate was


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
    return optimize(builder.finish())


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
    hir = optimize(builder.finish())
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
        optimize(builder.finish())


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

    hir = optimize(builder.finish())
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
    hir = optimize(builder.finish())
    assert [node.operator for node in hir.nodes.values() if isinstance(node, Operation)] == [IntToFloat(), FloatToInt()]

    def constant_round_trip(value: int) -> Hir:
        builder = HirBuilder()
        builder.block()
        builder.output(
            "y", builder.operation(FloatToInt(), [builder.operation(IntToFloat(), [builder.int_const(value)])])
        )
        builder.ret()
        return optimize(builder.finish())

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

    hir = optimize(builder.finish())
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
    return optimize(builder.finish())


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


def test_a_block_the_sequencer_cannot_enter_is_never_convicted() -> None:
    # Nothing rewrites a proven branch into a jump, so a proven-dead arm survives into the survivor sweep, and in a
    # never-returning loop it is still CFG-reachable -- the shape where post-DCE block liveness and enterability
    # disagree. The guard proves its own arm dead, so the sweep must not convict what that arm holds.
    # Every constant here is opaque to the front end (an operation over a parameter) and constant to HIR (``x*0 == 0``):
    # the loop condition, so the loop residualizes rather than unrolling and only HIR sees that it never exits; the
    # guard, so its arm is proven dead; and both operands of the quotient, because a fold names a quotient only where
    # it knows BOTH of them, so an unknown numerator would go unconvicted wherever it sat and would witness nothing.
    def never_returns(x: float) -> float:
        y = x
        while (x * 0.0) <= 0.0:
            gate = y * 0.0
            if gate > 0.0:
                y = (y * 0.0) / (y * 0.0)
            y = y + 1.0
        return y

    hir = optimize(lower(never_returns).hir)
    assert any(isinstance(block.terminator, Ret) for block in hir.blocks)
    quotients = [
        node for node in hir.nodes.values() if isinstance(node, Operation) and isinstance(node.operator, HirFloatDiv)
    ]
    assert quotients, "the excluded arm's quotient must still be present -- otherwise the skip rule is untested"
    assert all(
        all(isinstance(hir.nodes[operand], FloatConst) for operand in node.operands) for node in quotients
    ), "both operands must be constant, or the sweep would decline to convict it wherever it sat"


def _int_in_port(builder: HirBuilder) -> None:
    builder.output("y", builder.operation(IntToFloat(), [builder.input("n", IntType())]))


def _int_const(builder: HirBuilder) -> None:
    # A runtime selector between two integer constants: the constants survive folding and reach MIR as themselves.
    selected = builder.operation(
        IntSelect(), [builder.input("c", BoolType()), builder.int_const(3), builder.int_const(4)]
    )
    builder.output("y", builder.operation(IntToFloat(), [selected]))


def _int_operation(builder: HirBuilder) -> None:
    # Rooted at FLOAT inputs, so no integer constant, port, or state read precedes the operation in lowering order.
    a, b = builder.input("a", HirFloatType()), builder.input("b", HirFloatType())
    total = builder.operation(IntAdd(), [builder.operation(FloatToInt(), [a]), builder.operation(FloatToInt(), [b])])
    builder.output("y", builder.operation(IntToFloat(), [total]))


def _int_state_read(builder: HirBuilder) -> None:
    value = builder.state_read("s", IntType())
    builder.state_slot("s", IntConst(0), value)
    builder.output("y", builder.operation(IntToFloat(), [value]))


def _int_phi(builder: HirBuilder) -> None:
    # A merge of two integer values: the sweep sees it directly, ahead of any lowering-order effect.
    entry = builder.current_block
    then_block, else_block, merge = builder.block(), builder.block(), builder.block()
    builder.branch(builder.input("c", BoolType()), then_block, else_block)
    builder.position_at(then_block)
    builder.jump(merge)
    builder.position_at(else_block)
    builder.jump(merge)
    builder.position_at(merge)
    phi = builder.phi(IntType(), [(then_block, builder.int_const(1)), (else_block, builder.int_const(2))])
    builder.output("y", builder.operation(IntToFloat(), [phi]))
    assert entry == 0


def _int_state_reset(builder: HirBuilder) -> None:
    # An integer reset carried by an otherwise float slot: only a sweep over the slots themselves sees this one.
    x = builder.input("x", HirFloatType())
    builder.state_slot("s", IntConst(0), x)
    builder.output("y", x)


@pytest.mark.parametrize(
    "make",
    [_int_in_port, _int_const, _int_operation, _int_state_read, _int_state_reset, _int_phi],
    ids=["in_port", "const", "operation", "state_read", "state_reset", "phi"],
)
def test_integer_nodes_are_refused_by_mir_lowering(make: Callable[[HirBuilder], None]) -> None:
    # Integers are contained at the HIR/MIR seam until the integer backend lands. Each node kind takes a different
    # lowering path, and a conversion operator is integer-OPERANDED while being float- or bool-RESULTED, so the
    # refusal cannot key on the result type alone.
    builder = HirBuilder()
    builder.block()
    make(builder)
    builder.ret()
    with pytest.raises(UnsupportedConstruct, match="not yet lowerable to hardware"):
        lower_to_mir(builder.finish(), OPS, FMT, DEFAULT_IFMT)


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
