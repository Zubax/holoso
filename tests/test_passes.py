"""
HIR optimization and MIR selection: public synthesize-level behavior (values against CPython or independent
literals, operator-module instantiation presence, the initiation-interval branch discriminator, verbatim refusal
diagnostics) plus the white-box contracts that have no public spelling.
"""

import dataclasses
import itertools
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
    FFromIntOptions,
    FMulILog2Options,
    FMulOptions,
    FRoundOptions,
    FToIntOptions,
    FloatFormat,
    FloatType,
    FloatValue,
    OperatorOptions,
    Options,
    SynthesisError,
    UnsupportedConstruct,
)
from holoso._operators import FDivOperator
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
from holoso._hir import Branch, BoolSelect, FloatDiv as HirFloatDiv, Phi, FloatSelect
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
    IntPopcount,
    IntSelect,
    IntShiftLeft,
    IntShiftRight,
    IntSub,
    IntToBool,
    IntToFloat,
    IntType,
)
from holoso._mir._refuse import refuse
from holoso._mir import lower as lower_to_mir
from ._importguard import forbidden_imports
from ._modelref import (
    branch_boundary_kernel,
    mir_options,
    const_branch_kernel,
    DEFAULT_IFCONV_MAX_OPS,
    default_tolerance,
    DEFAULT_UNROLL_MAX_TRIPS,
    diamond_then_loop_kernel,
    instantiated_modules as _instantiated,
    overlap_spill_kernel,
    phi_swap_loop,
    within,
)
from ._examples import equal_temperament

FMT = FloatFormat(6, 18)
OPTIONS = Options(
    OperatorOptions(
        fadd=FAddOptions(),
        fmul=FMulOptions(),
        fdiv=FDivOptions(),
        fmul_ilog2=FMulILog2Options(),
        fcmp=FCmpOptions(),
    ),
    ffmt=FMT,
)
INT_OPTIONS = Options(OperatorOptions())
OPS = mir_options(OPTIONS)


def _synth(
    target: Callable[..., object], options: Options = OPTIONS, name: str | None = None
) -> holoso.SynthesisResult:
    return holoso.synthesize(target, options, name=name or target.__name__.strip("_"))


def _ints(values: list[FloatValue | holoso.IntValue | bool]) -> list[int | bool]:
    out: list[int | bool] = []
    for value in values:
        assert not isinstance(value, FloatValue)
        out.append(value if isinstance(value, bool) else int(value))
    return out


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
        lower_to_mir(hir, OPS)
    except UnsupportedConstruct as ex:
        assert "no MIR lowering rule" in str(ex)
    else:
        assert False, "expected HIR-to-MIR lowering to reject non-float semantic input"


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


def test_a_constant_only_kernel_lowers_to_its_exact_value() -> None:
    def f() -> float:
        return 3.5

    result = _synth(f, name="const_only")
    assert _instantiated(result) == set()
    assert float(result.numerical_model.elaborate().run()[0]) == 3.5


def test_a_power_of_two_scale_selects_the_exponent_operator() -> None:
    def quarter(a: float) -> float:
        return a * 0.25

    def doubled(a: float) -> float:
        return 2 * a

    def fourth(a: float) -> float:
        return a / 4.0

    for kernel in (quarter, doubled, fourth):
        result = _synth(kernel)
        assert _instantiated(result) == {"holoso_fmul_ilog2"}, kernel.__name__
        sim = result.numerical_model.elaborate()
        for a in (1.0, -0.5, 0.0, 6.0):
            assert float(sim.run(a)[0]) == kernel(a), f"{kernel.__name__}({a})"


def test_div_by_nonpow2_const_becomes_reciprocal_multiply() -> None:
    def f(a: float) -> float:
        return a / 3.0

    result = _synth(f, name="div3")
    assert _instantiated(result) == {"holoso_fmul"}
    sim = result.numerical_model.elaborate()
    for a in (3.0, -1.5, 0.75):
        want = a / 3.0
        assert within(float(sim.run(a)[0]), want, *default_tolerance(FMT, 2, magnitude=max(1.0, abs(want)))), a


_WIDE_OPTIONS = dataclasses.replace(OPTIONS, ffmt=FloatFormat(12, 24))
"""An exponent field wide enough to hold values the compiler's own binary64 cannot, which is what makes the host-side
declinations below reachable through the public API at all."""


def test_a_reciprocal_the_host_cannot_hold_leaves_the_division_standing() -> None:
    # ``x/c == x*(1/c)`` only where the reciprocal is a number: a subnormal divisor has none the host can hold, so
    # minting it answers infinity for a quotient that is an ordinary, representable value.
    def f(a: float) -> float:
        return a / 1e-320

    result = _synth(f, _WIDE_OPTIONS, name="div_subnormal")
    assert _instantiated(result) == {"holoso_fdiv"}
    got = float(result.numerical_model.elaborate().run(1e-20)[0])
    want = 1e-20 / 1e-320
    assert within(got, want, *default_tolerance(_WIDE_OPTIONS.ffmt, 2, magnitude=want))


def test_adjacent_constant_scalings_compose_into_one() -> None:
    # Associativity is chartered, so a value scaled twice is scaled once -- and costs exactly what the collapsed
    # spelling costs, whether the two scales are ordinary factors, exact exponents, or one of each.
    def factors(a: float) -> float:
        return (a * 3.0) * 5.0

    def one_factor(a: float) -> float:
        return a * 15.0

    def exponents(a: float) -> float:
        return (((a * 2.0) * 4.0) * 8.0) * 16.0

    def one_exponent(a: float) -> float:
        return a * 1024.0

    def mixed(a: float) -> float:
        return (a * 3.0) * 4.0

    def one_mixed(a: float) -> float:
        return a * 12.0

    for composed, collapsed in ((factors, one_factor), (exponents, one_exponent), (mixed, one_mixed)):
        result, reference = _synth(composed), _synth(collapsed)
        assert result.initiation_interval == reference.initiation_interval, composed.__name__
        assert _instantiated(result) == _instantiated(reference), composed.__name__
        sim = result.numerical_model.elaborate()
        for a in (1.0, -0.5, 0.0, 6.0):
            assert float(sim.run(a)[0]) == collapsed(a), f"{composed.__name__}({a})"


def test_a_composition_landing_on_a_negative_power_of_two_keeps_its_exponent() -> None:
    # The sign is part of a scaling, not part of its constant: a composition landing on a NEGATIVE power of two must
    # stay an exponent plus a free sign sideband, exactly as the pair it replaces was. Materializing it as a constant
    # instead subjects an unbounded exact scale to the format, which refuses whatever the format cannot hold.
    def negated(a: float) -> float:
        return (a * 2.0**40) * -1.0

    def negated_and_doubled(a: float) -> float:
        return (a * 2.0**40) * -2.0

    for kernel in (negated, negated_and_doubled):
        result = _synth(kernel)
        assert _instantiated(result) == {"holoso_fmul_ilog2"}, kernel.__name__

    # The same fact where the format CAN hold the constant, so nothing refuses and only the selection shows it.
    wide = _synth(negated, dataclasses.replace(OPTIONS, ffmt=FloatFormat(8, 36)), name="negated_wide")
    assert _instantiated(wide) == {"holoso_fmul_ilog2"}
    sim = wide.numerical_model.elaborate()
    for a in (1.0, -0.5, 0.0, 6.0):
        assert float(sim.run(a)[0]) == -(a * 2.0**40), a

    # And the scaler alone suffices: a kernel needing no general multiplier must not acquire one.
    scaler_only = Options(OperatorOptions(fmul_ilog2=FMulILog2Options()), ffmt=FMT)
    assert _instantiated(_synth(negated, scaler_only, name="negated_scaler_only")) == {"holoso_fmul_ilog2"}


def test_a_constant_multiplier_the_format_cannot_hold_is_carried_apart() -> None:
    # A constant the optimizer minted faces the format alone, and the significand of any scaling lies in [1, 2),
    # which every format holds exactly. So a scale that fits no single constant always fits two, and the kernels
    # below build rather than being refused over a number appearing nowhere in their source.
    def composed(a: float) -> float:
        return (a * 2.0**40) * 3.0

    def tiny_constant(a: float) -> float:
        return a * math.cos(math.pi / 2)  # the front end folds this to 6.1e-17, which e6m18 cannot hold

    for kernel, probes in ((composed, (1e-9, 3e-9)), (tiny_constant, (1e9, 4e9))):
        result = _synth(kernel)
        assert _instantiated(result) == {"holoso_fmul", "holoso_fmul_ilog2"}, kernel.__name__
        sim = result.numerical_model.elaborate()
        for a in probes:
            want = kernel(a)
            assert within(float(sim.run(a)[0]), want, *default_tolerance(FMT, 2, magnitude=want)), f"{kernel} {a}"


def test_a_scale_past_the_formats_reach_is_refused_rather_than_split() -> None:
    # Splitting pays only where some operand the machine holds reaches a result it holds. Past the format's own
    # exponent span none does, so the split would buy an extra operation and answer zero anyway -- and the constant,
    # which names the trouble and the way out, is worth refusing over after all.
    def f(a: float) -> float:
        return a * 1e-40  # e6m18 tops out near 4.3e9, so every product is far under the 9.3e-10 floor

    with pytest.raises(UnsupportedConstruct) as exc:
        _synth(f, name="past_the_reach")
    assert exc.value.message == (
        "constant 1e-40 degrades to 0.0 in FloatFormat(wexp=6, wman=18); widen wexp or rescale"
    )


def test_without_the_scaler_the_constant_is_still_what_is_refused() -> None:
    # The split needs the exponent scaler. Without it the kernel must be refused over the constant it cannot hold,
    # which names both the trouble and the way out, rather than over an operator the kernel never asked for.
    options = Options(OperatorOptions(fmul=FMulOptions(), fdiv=FDivOptions()), ffmt=FMT)

    def f(x: float) -> float:
        return x / 3e9

    with pytest.raises(UnsupportedConstruct) as exc:
        _synth(f, options, name="no_scaler_to_split_into")
    assert exc.value.message == (
        "constant 3.333333333333333e-10 degrades to 0.0 in FloatFormat(wexp=6, wman=18); widen wexp or rescale"
    )


def test_an_ordinary_scaling_pair_is_not_split() -> None:
    # The split is for what one constant cannot carry; a pair that composes to an ordinary constant must still
    # collapse to the single multiply, not acquire a second operation.
    def f(a: float) -> float:
        return (a * 4.0) * 3.0

    result = _synth(f, name="ordinary_pair")
    assert _instantiated(result) == {"holoso_fmul"}
    assert float(result.numerical_model.elaborate().run(2.0)[0]) == 24.0


def test_a_composed_scaling_declines_where_the_host_arithmetic_rails() -> None:
    # The composition is the associativity identity, whose precondition is that the compiler's OWN arithmetic holds
    # the product -- not that the target format does. A railed product is no product, so the exact exponents stand,
    # where folding them would scale by a materialized infinity instead.
    def f(a: float) -> float:
        return (a * 2.0**1023) * 4.0

    result = _synth(f, name="railed_pair")
    assert _instantiated(result) == {"holoso_fmul_ilog2"}
    assert float(result.numerical_model.elaborate().run(0.0)[0]) == 0.0


def _two_chained_multiplies(a: float, b: float) -> float:
    """A chain no composition can shorten -- the cost reference for a declined one."""
    return (a * 3.0) * b


def test_a_composed_scaling_declines_where_the_host_arithmetic_collapses() -> None:
    # The other side of the same precondition. Both constants survive this format, and so does their true product;
    # only the host loses it, and folding there would answer zero for every input.
    def f(a: float) -> float:
        return (a * 1e-200) * 1e-200

    result = _synth(f, _WIDE_OPTIONS, name="collapsed_pair")
    assert _instantiated(result) == {"holoso_fmul"}
    assert result.initiation_interval == _synth(_two_chained_multiplies, _WIDE_OPTIONS).initiation_interval
    got = float(result.numerical_model.elaborate().run(1e100)[0])
    assert within(got, 1e-300, *default_tolerance(FloatFormat(12, 24), 2, magnitude=1e-300))


def test_a_composed_scaling_declines_where_the_product_is_subnormal() -> None:
    # Gradual underflow is a partial collapse, and the guard is about what the compiler's arithmetic can carry, not
    # about whether it returned something: a product landing in the host's subnormals keeps only the bits underflow
    # leaves it, so materializing it would round a constant this format holds exactly.
    def f(a: float) -> float:
        return (a * 1e-200) * 1e-120

    result = _synth(f, _WIDE_OPTIONS, name="subnormal_product")
    assert _instantiated(result) == {"holoso_fmul"}
    assert result.initiation_interval == _synth(_two_chained_multiplies, _WIDE_OPTIONS).initiation_interval
    # 1e-320 is a host subnormal carrying ~13 significand bits; folding it lands ~185 ulp off the true product.
    got = float(result.numerical_model.elaborate().run(1.0)[0])
    assert within(got, 1e-200 * 1e-120, *default_tolerance(_WIDE_OPTIONS.ffmt, 2, magnitude=1e-320))


def test_a_signed_power_of_two_scales_by_its_exponent_however_it_is_spelled() -> None:
    # One value, three spellings, one selection: the sign of an exact power of two rides the free sideband, so
    # nothing here needs a general multiplier and a kernel configured without one still builds.
    def scaled(a: float) -> float:
        return a * -16.0

    def divided(a: float) -> float:
        return a / -8.0

    scaler_only = Options(OperatorOptions(fmul_ilog2=FMulILog2Options()), ffmt=FMT)
    for kernel in (scaled, divided):
        result = _synth(kernel, scaler_only)
        assert _instantiated(result) == {"holoso_fmul_ilog2"}, kernel.__name__
        sim = result.numerical_model.elaborate()
        for a in (1.0, -2.5, 0.0, 6.0):
            assert float(sim.run(a)[0]) == kernel(a), f"{kernel.__name__}({a})"


def test_the_absorbing_zero_outranks_a_composition() -> None:
    # ``x*0 == 0`` holds for the non-finite operand too, so the pair must not be combined into the indeterminate form
    # first: composing ``inf`` with ``0.0`` names no number and would refuse a build that has a defined answer.
    def f(a: float) -> float:
        return (a * math.inf) * 0.0

    result = _synth(f, name="inf_then_zero")
    assert _instantiated(result) == set()
    assert float(result.numerical_model.elaborate().run(3.0)[0]) == 0.0


def test_a_composition_can_migrate_the_multiplier_the_kernel_needs() -> None:
    # The composed constant faces operator availability on its own: a product landing exactly on a power of two
    # selects the exponent scaler, so a kernel spelling only general products can be refused over an operator it
    # never asked for. Documented under the optimizer's DEFERRED note rather than prevented.
    options = Options(OperatorOptions(fmul=FMulOptions()), ffmt=FMT)

    def f(a: float) -> float:
        return (a * 3.0) * (2.0 / 3.0)

    with pytest.raises(UnsupportedConstruct, match="fmul_ilog2"):
        _synth(f, options, name="migrated_multiplier")


def _narrow_options() -> Options:
    return dataclasses.replace(OPTIONS, ffmt=FloatFormat(3, 4))


def test_wide_supported_pow2_uses_ilog2_operator() -> None:
    def f(a: float) -> float:
        return a * 16.0

    result = _synth(f, _narrow_options(), name="x16")
    assert _instantiated(result) == {"holoso_fmul_ilog2"}
    sim = result.numerical_model.elaborate()
    # FloatFormat(3, 4) tops out at 2**3 * 1.9375 == 15.5, so 16.0 rails to the infinity while -8.0 is exact.
    for value, want in [(1.0, math.inf), (-0.5, -8.0), (0.0, 0.0)]:
        assert float(sim.run(value)[0]) == want, value


def test_a_scale_past_the_float_range_builds_and_rails() -> None:
    """The exponent is an integer constant, so no scale is refused; past the format's range the scaler rails."""

    def f(a: float) -> float:
        return a * 64.0

    result = _synth(f, _narrow_options(), name="x64")
    assert _instantiated(result) == {"holoso_fmul_ilog2"}
    sim = result.numerical_model.elaborate()
    # 64.0 and -32.0 both lie past FloatFormat(3, 4)'s 15.5 ceiling, so each rails to its signed infinity.
    for value, want in [(1.0, math.inf), (-0.5, -math.inf), (0.0, 0.0)]:
        assert float(sim.run(value)[0]) == want, value


def test_an_exponent_past_the_int_format_clamps_where_the_scaler_rails() -> None:
    """
    A count past the int word lies far beyond the float's dynamic range, so the clamped exponent rails identically.
    """

    def f(a: float) -> tuple[float, float]:
        return a * 2.0**1000, a / 2.0**1000

    options = dataclasses.replace(OPTIONS, ffmt=FloatFormat(4, 5), wint_min=2)
    result = _synth(f, options, name="clamped_scale")
    assert _instantiated(result) == {"holoso_fmul_ilog2"}
    sim = result.numerical_model.elaborate()
    # Any nonzero value scaled by 2**±1000 leaves FloatFormat(4, 5) entirely: signed infinity up, zero down.
    for value, want in [(1.0, [math.inf, 0.0]), (-1.5, [-math.inf, 0.0]), (0.0, [0.0, 0.0])]:
        assert [float(v) for v in sim.run(value)] == want, value


def test_subtraction_folds_into_second_operand_sign() -> None:
    def f(a: float, b: float) -> float:
        return a - b

    result = _synth(f, name="sub")
    assert _instantiated(result) == {"holoso_fadd"}
    assert result.verilog_output.verilog.count("holoso_fadd #") == 1
    sim = result.numerical_model.elaborate()
    for a, b in [(2.0, 0.5), (0.5, 2.0), (-1.0, -3.0)]:
        assert float(sim.run(a, b)[0]) == a - b


def test_operand_negation_folds_into_operator() -> None:
    def f(a: float, b: float) -> float:
        return a * (-b)

    result = _synth(f, name="mulneg")
    assert _instantiated(result) == {"holoso_fmul"}
    assert result.verilog_output.verilog.count("holoso_fmul #") == 1
    sim = result.numerical_model.elaborate()
    for a, b in [(2.0, 0.5), (-4.0, 0.25), (3.0, 0.0)]:
        assert float(sim.run(a, b)[0]) == a * (-b)


def test_pure_sign_output_adds_no_operation() -> None:
    def f(a: float) -> float:
        return -abs(a)

    result = _synth(f, name="negabs")
    assert _instantiated(result) == set()
    sim = result.numerical_model.elaborate()
    for a in (-2.0, 0.0, 1.5):
        assert float(sim.run(a)[0]) == -abs(a)


def test_a_mixed_expression_selects_only_real_operators() -> None:
    def f(a: float, b: float) -> float:
        return (a - b) * 0.25 + a * b

    result = _synth(f, name="mixed_expr")
    assert _instantiated(result) == {"holoso_fadd", "holoso_fmul", "holoso_fmul_ilog2"}
    sim = result.numerical_model.elaborate()
    for a, b in [(2.0, 0.5), (1.0, 1.0), (-4.0, 0.25)]:
        assert float(sim.run(a, b)[0]) == f(a, b)


def test_ekf1_stateless_synthesis() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    import ekf1_stateless

    result = _synth(ekf1_stateless.update_x_P, name="ekf1_stateless")
    assert len(result.input_ports) == 17
    assert len(result.output_ports) == 9
    verilog = result.verilog_output.verilog
    assert verilog.count("holoso_fdiv #") == 1  # the source's only division, x22 = 1 / x21, on one pooled divider
    assert verilog.count("holoso_fmul_ilog2 #") >= 1  # the "2 * ..." terms


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
    # nested unrolled loops chaining thousands of blocks -- overflowed Python's recursion limit with a RecursionError;
    # the iterative DFS compiles cleanly. The whole front-to-back pipeline runs (each layer contains a CFG DFS), and
    # the bit-exact model is checked against the plain-Python reference.
    options = Options(OperatorOptions(fadd=FAddOptions(), fcmp=FCmpOptions()), ffmt=FMT)
    sim = _synth(_deep_cfg_kernel, options, name="deep_cfg").numerical_model.elaborate()
    for x in (0.5, 2.0, 8.0):  # acc stays positive -> +900 every time; 0.5/2.0/8.0 are exact in ZKF
        assert float(sim.run(x)[0]) == _deep_cfg_kernel(x)


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
    return optimize(lower(target, DEFAULT_UNROLL_MAX_TRIPS).hir, ifconv_max_ops)


# A converted diamond leaves straight-line code whose latency is exact, so initiation_interval[1] is a number;
# a surviving real branch makes the latency data-dependent and initiation_interval[1] is None.


def test_if_conversion_collapses_a_pure_diamond() -> None:
    def f(a: float, b: float) -> float:
        if a > b:
            y = a + b
        else:
            y = a - b
        return y

    result = _synth(f, name="pure_diamond")
    assert result.initiation_interval[1] is not None
    sim = result.numerical_model.elaborate()
    for a, b in [(2.0, 0.5), (0.5, 2.0), (1.0, 1.0)]:
        assert float(sim.run(a, b)[0]) == f(a, b)


def test_if_conversion_refuses_an_unspeculatable_arm() -> None:
    # Division must not be speculated: a div-by-zero on the not-taken path would assert the module error flag.
    def f(a: float, b: float) -> float:
        if a > b:
            y = a + b
        else:
            y = a / b
        return y

    result = _synth(f, name="unspec_arm")
    assert result.initiation_interval[1] is None  # the diamond survives as a real branch
    assert "holoso_fdiv" in _instantiated(result)
    sim = result.numerical_model.elaborate()
    for a, b in [(3.0, 1.0), (1.0, 2.0), (1.0, 4.0)]:
        assert float(sim.run(a, b)[0]) == f(a, b)


def test_if_conversion_respects_the_arm_size_budget() -> None:
    def f(a: float, b: float) -> float:
        if a > b:
            y = (a + b) * a + b  # three operations, one over the budget below
        else:
            y = a - b  # a negate and an add: exactly at the budget, so only the then-arm can refuse
        return y

    over = _synth(f, dataclasses.replace(OPTIONS, ifconv_max_ops=2), name="budget_over")
    within_budget = _synth(f, dataclasses.replace(OPTIONS, ifconv_max_ops=3), name="budget_within")
    assert over.initiation_interval[1] is None
    assert within_budget.initiation_interval[1] is not None
    for result in (over, within_budget):
        sim = result.numerical_model.elaborate()
        for a, b in [(2.0, 0.5), (0.5, 2.0), (1.0, 1.0)]:
            assert float(sim.run(a, b)[0]) == f(a, b)


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

    zero_budget = dataclasses.replace(OPTIONS, ifconv_max_ops=0)
    converted = _synth(constant_arms, zero_budget, name="const_arms")
    refused = _synth(one_op_arm, zero_budget, name="one_op_arm")
    assert converted.initiation_interval[1] is not None
    assert refused.initiation_interval[1] is None
    for result, kernel in ((converted, constant_arms), (refused, one_op_arm)):
        sim = result.numerical_model.elaborate()
        for a, b in [(2.0, 0.5), (0.5, 2.0), (1.0, 1.0)]:
            assert float(sim.run(a, b)[0]) == kernel(a, b)


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
        want = f(a, b, c)  # exact at these vectors, so both configurations answer the CPython value
        got_branchy, got_converted = float(left.run(a, b, c)[0]), float(right.run(a, b, c)[0])
        assert got_branchy == want, f"branchy diverged from Python at ({a}, {b}, {c})"
        assert got_converted == want, f"converted diverged from Python at ({a}, {b}, {c})"


def test_if_conversion_converts_a_boolean_phi_merge() -> None:
    # A diamond merging a boolean collapses to straight-line code; both arms are dynamic comparisons.
    def f(a: float, b: float, c: float) -> float:
        if a > b:
            flag = b > c
        else:
            flag = a > c
        return float(flag)

    result = _synth(f, name="bool_phi_merge")
    assert result.initiation_interval[1] is not None
    sim = result.numerical_model.elaborate()
    for a, b, c in [(3.0, 2.0, 1.0), (3.0, 2.0, 2.5), (1.0, 2.0, 0.5), (1.0, 2.0, 1.5)]:
        assert float(sim.run(a, b, c)[0]) == f(a, b, c)


def test_if_conversion_converts_an_integer_phi_merge() -> None:
    # The integer dual, so min/max/sign cost one mux rather than a branch.
    def f(a: int, b: int) -> int:
        if a > b:
            y = a + b
        else:
            y = a - b
        return y

    result = _synth(f, INT_OPTIONS, name="int_phi_merge")
    assert result.initiation_interval[1] is not None
    sim = result.numerical_model.elaborate()
    for a, b in [(3, 1), (1, 3), (2, 2), (-5, -9)]:
        assert _ints(sim.run(a, b)) == [f(a, b)]


def test_an_error_bearing_integer_arm_stays_behind_its_guard() -> None:
    # Floor-division and modulo assert the div-by-zero flag, so a never-taken arm holding one must not be speculated.
    def f(a: int, b: int) -> int:
        if a > b:
            y = a + b
        else:
            y = a // b
        return y

    result = _synth(f, INT_OPTIONS, name="int_guarded_div")
    assert result.initiation_interval[1] is None
    assert "holoso_idivs" in _instantiated(result)
    sim = result.numerical_model.elaborate()
    for a, b in [(3, 1), (1, 3), (-7, 2)]:
        assert _ints(sim.run(a, b)) == [f(a, b)]


def test_if_conversion_reduces_constant_armed_boolean_select() -> None:
    # The state-machine merge shape: arms are boolean constants, so the merge reduces to plain gates and the whole
    # kernel stays straight-line -- exactly the schmitt/pfd collapse.
    def f(a: float, b: float, hold: bool) -> float:
        if a > b:
            flag = True
        else:
            flag = hold  # passthrough arm
        return float(flag)

    result = _synth(f, name="const_armed_select")
    assert result.initiation_interval[1] is not None
    hir = _hir_of(f)  # the collapse itself is invisible publicly (inline selects produce no branch): white-box
    assert len(hir.blocks) == 1
    assert not any(
        isinstance(n, Operation) and isinstance(n.operator, (BoolSelect, FloatSelect)) for n in hir.nodes.values()
    ), "the constant-armed bselect must reduce to gates, not survive as a mux"
    sim = result.numerical_model.elaborate()
    for a, b in [(2.0, 1.0), (1.0, 2.0)]:
        for hold in (False, True):
            assert float(sim.run(a, b, hold)[0]) == f(a, b, hold)


def test_bselect_reductions_are_truth_table_correct() -> None:
    # The bool-mux strength-reduction identities (a bselect with constant arms collapses to and/or/not/passthrough)
    # must be bit-exact. Every shape runs over the full boolean input cube against its CPython reference; a wrong
    # identity -- e.g. (c,False,True) reduced to c not ~c -- mismatches here.
    def shapes(c: bool, t: bool, f: bool) -> tuple[bool, bool, bool, bool, bool, bool, bool, bool]:
        if c:  # (c, True, False) -> c
            y0 = True
        else:
            y0 = False
        if c:  # (c, False, True) -> not c
            y1 = False
        else:
            y1 = True
        if c:  # (c, True, f) -> c or f
            y2 = True
        else:
            y2 = f
        if c:  # (c, False, f) -> (not c) and f
            y3 = False
        else:
            y3 = f
        if c:  # (c, t, True) -> (not c) or t
            y4 = t
        else:
            y4 = True
        if c:  # (c, t, False) -> c and t
            y5 = t
        else:
            y5 = False
        if c:  # (c, t, f) -> bselect kept
            y6 = t
        else:
            y6 = f
        if c:  # (c, t, ~f) -> kept, the inversion riding the arm conditioner
            y7 = t
        else:
            y7 = not f
        return y0, y1, y2, y3, y4, y5, y6, y7

    result = _synth(shapes, name="bselect_shapes")
    assert result.initiation_interval[1] is not None
    sim = result.numerical_model.elaborate()
    for combo in itertools.product([False, True], repeat=3):
        assert list(sim.run(*combo)) == list(shapes(*combo)), combo


def test_identical_mux_arms_collapse_whatever_the_selector() -> None:
    # ``mux(c, X, X) == X`` is the universal mux identity the constant-arm reductions depend on. The shape appears
    # only once if-conversion interns both arms into one block, so each arm's comparison becomes a self-comparison;
    # the mux must be gone from the graph (white-box: inline selects are publicly invisible), and the kernels must
    # answer exactly as CPython does.
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
        result = _synth(kernel)
        assert result.initiation_interval[1] is not None, kernel.__name__
        sim = result.numerical_model.elaborate()
        for x in (-8.0, -1.0, 0.0, 0.5, 3.0):
            got, want = read(sim.run(x)[0]), read(kernel(x))
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

    result = _synth(f, name="nested_chain")
    assert result.initiation_interval[1] is not None
    sim = result.numerical_model.elaborate()
    for x, y in [(1.0, 2.0), (-1.0, 2.0), (1.0, -2.0), (-1.5, -0.5)]:
        assert float(sim.run(x, y)[0]) == f(x, y)


def test_if_conversion_repoints_loop_header_phi_arms() -> None:
    # A diamond inside a while body: the dissolved merge block fed the loop-header phis, whose arms must repoint to
    # the spliced block (the localized pin for the repoint path; the examples exercise it only end-to-end). The outer
    # runtime loop keeps the public II data-dependent either way, so the discriminator is blind here: white-box.
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

    result = _synth(f, name="dead_diamond")
    assert result.initiation_interval[1] is not None
    assert _instantiated(result) == set(), "the unused condition cone (division included) is dead code"
    sim = result.numerical_model.elaborate()
    for a, b, x in [(1.0, 0.0, 2.5), (0.0, 1.0, -3.0)]:  # the dead division never runs, so b == 0 is inert
        assert float(sim.run(a, b, x)[0]) == x


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
    (IntPopcount(), [IntType()], IntType()),
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
    # The declared integer identities and absorbing elements simplify against an operand the folder cannot see.
    # Only the arithmetic links (``+ 0``, ``* 1``, ``* 0``) would instantiate modules, so the empty set pins their
    # folds; the bitwise links are inline either way, and their reductions are pinned white-box on the graph.
    def f(n: int) -> tuple[int, int, int, int]:
        return ((((n + 0) * 1) | 0) ^ 0) & -1, n * 0, n | -1, n & 0

    result = _synth(f, INT_OPTIONS, name="int_identities")
    assert _instantiated(result) == set()
    sim = result.numerical_model.elaborate()
    for n in (-9, -1, 0, 1, 7, 1000):
        assert _ints(sim.run(n)) == list(f(n))


def test_a_constant_integer_expression_folds_away_entirely() -> None:
    # Folding is exact at arbitrary precision -- no width, no saturation -- so a fully static integer expression
    # disappears before MIR ever has to hold it in a machine word, and no converter or multiplier is instantiated.
    def f() -> float:
        k = 2 + 3
        return float(k * 2**24)  # past the machine word, within the float

    result = _synth(f, name="const_int_fold")
    assert _instantiated(result) == set()
    assert float(result.numerical_model.elaborate().run()[0]) == float(5 * 2**24)


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
        (IntPopcount(), [-huge], huge.bit_count()),
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
    lower_to_mir(raw, OPS)  # nothing integer survives


def test_the_integer_subtraction_rules_the_shared_algebra_cannot_state() -> None:
    # ``x - 0`` is ``x`` while ``0 - x`` is the negation, so each direction is its own rule; the negation is the
    # only direction that needs the subtractor. Pooling hides op counts, so the artifact pins one subtractor
    # instance and no other module class.
    def f(n: int) -> tuple[int, int, int]:
        return n - 0, 0 - n, n - n

    result = _synth(f, INT_OPTIONS, name="int_sub_rules")
    assert _instantiated(result) == {"holoso_isubs"}
    assert result.verilog_output.verilog.count("holoso_isubs #") == 1
    sim = result.numerical_model.elaborate()
    for n in (-9, -1, 0, 1, 7, 1000):
        assert _ints(sim.run(n)) == list(f(n))


def test_a_power_of_two_integer_product_mints_the_saturating_scaling() -> None:
    # The exponent is absorbed into the operator from either side, and the constant -- even one no machine word
    # holds -- goes dead with it, so it is never asked to materialize and no multiplier is instantiated.
    def f(n: int) -> tuple[int, int]:
        return n * 8, (2**40) * n

    result = _synth(f, INT_OPTIONS, name="int_pow2_product")
    assert "holoso_imuls" not in result.verilog_output.verilog
    sim = result.numerical_model.elaborate()
    word = result.int_format
    for n in (-9, -1, 0, 1, 7, 1000):
        # 2**40 lies far past any machine word, so every nonzero product rails by sign whatever the kernel settled on.
        railed = word.min if n < 0 else word.max if n > 0 else 0
        assert _ints(sim.run(n)) == [n * 8, railed]


def test_integer_negations_share_one_tracking_across_their_spellings() -> None:
    # ``x * -1`` and ``x // -1`` are negations rather than a product and a quotient, so no multiplier or divider
    # module appears; ``-(-x)`` returns the base and ``n + (-n)`` folds to zero, so no adder appears either. The
    # surviving negations bind the one pooled subtractor instance.
    def f(n: int) -> tuple[int, int, int, int]:
        return -(-n), n + (-n), n * -1, n // -1

    result = _synth(f, INT_OPTIONS, name="int_negations")
    verilog = result.verilog_output.verilog
    assert verilog.count("holoso_isubs #") == 1
    assert "holoso_imuls" not in verilog and "holoso_idivs" not in verilog and "holoso_iadds" not in verilog
    sim = result.numerical_model.elaborate()
    for n in (-9, -1, 0, 1, 7, 1000):
        assert _ints(sim.run(n)) == list(f(n))


def test_integer_division_and_remainder_reduce_against_their_constants() -> None:
    # Every quotient and remainder here reduces -- identity, power-of-two shift/mask, self, zero -- so no divider
    # is instantiated; where CPython raises (n == 0), the reductions answer for the operand they erased.
    def f(n: int) -> tuple[int, int, int, int, int, int, int, int, int]:
        return n // 1, n // 8, n // n, 0 // n, n % 1, n % -1, n % 8, n % n, 0 % n

    result = _synth(f, INT_OPTIONS, name="int_divmod_rules")
    assert "holoso_idivs" not in result.verilog_output.verilog
    sim = result.numerical_model.elaborate()
    for n in (-9, -1, 1, 7, 1000):
        assert _ints(sim.run(n)) == list(f(n))
    assert _ints(sim.run(0)) == [0, 0, 1, 0, 0, 0, 0, 0, 0]  # q // q == 1 and r % r == 0, whatever q turns out to be


def test_bitwise_value_equality_and_complement_rules() -> None:
    def f(n: int) -> tuple[int, int, int, int, int, int, int, int]:
        return n ^ n, n & n, n | n, n ^ -1, ~(~n), n & ~n, n | ~n, n ^ ~n

    sim = _synth(f, INT_OPTIONS, name="int_bitwise_rules").numerical_model.elaborate()
    for n in (-9, -1, 0, 1, 7, 1000):
        assert _ints(sim.run(n)) == list(f(n))


def test_the_bitwise_reductions_fold_on_the_graph() -> None:
    # The bitwise operators are inline, so no public artifact can witness these folds: white-box over the optimizer.
    builder = HirBuilder()
    builder.block()
    n = builder.input("n", IntType())
    inverted = builder.operation(IntBwNot(), [n])
    builder.output("xor_self", builder.operation(IntBwXor(), [n, n]))
    builder.output("annihilated", builder.operation(IntBwAnd(), [n, inverted]))
    builder.output("saturated", builder.operation(IntBwOr(), [n, inverted]))
    builder.output("disagreed", builder.operation(IntBwXor(), [n, inverted]))
    builder.output("complement", builder.operation(IntBwXor(), [n, builder.int_const(-1)]))
    builder.ret()
    hir = optimize(builder.finish(), DEFAULT_IFCONV_MAX_OPS)
    outputs = {out.name: hir.nodes[out.value] for out in hir.outputs}
    assert outputs["xor_self"] == outputs["annihilated"] == IntConst(0)
    assert outputs["saturated"] == outputs["disagreed"] == IntConst(-1)
    assert isinstance(outputs["complement"], Operation) and outputs["complement"].operator == IntBwNot()


def test_a_reflexive_integer_comparison_folds_to_its_truth() -> None:
    # No integer is a NaN, so every relation is decided over equal operands without seeing their value, and no
    # comparator is instantiated.
    def f(n: int) -> tuple[bool, bool, bool, bool, bool, bool]:
        return n == n, n <= n, n >= n, n != n, n < n, n > n

    result = _synth(f, INT_OPTIONS, name="int_reflexive_cmp")
    assert _instantiated(result) == set()
    sim = result.numerical_model.elaborate()
    for n in (-9, 0, 7):
        assert list(sim.run(n)) == list(f(n))


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
    # The integer dual of the float rule below: ``5 // 0`` has no value for the fold, so it is an operand the
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


def test_the_composed_cast_laws_are_observable_where_the_formats_distinguish_them() -> None:
    # The public value rows for the two cast laws above. FloatFormat(6, 18) carries 19 significand bits, so
    # 1048577 == 2**20 + 1 rounds to 2**20 through the float while wint_min=34 lets the integer word hold both
    # sides -- the round trip is visibly not an identity; 3.75 truncates to 3 either side of zero.
    options = Options(
        OperatorOptions(fround=FRoundOptions(), ffromint=FFromIntOptions(), ftoint=FToIntOptions()),
        ffmt=FMT,
        wint_min=34,
    )

    def int_float_int(n: int) -> int:
        return int(float(n))

    def float_int_float(x: float) -> float:
        return float(int(x))

    int_sim = _synth(int_float_int, options).numerical_model.elaborate()
    assert _ints(int_sim.run(1048577)) == [1048576]
    float_sim = _synth(float_int_float, options).numerical_model.elaborate()
    assert [float(v) for v in float_sim.run(3.75)] == [3.0]
    assert [float(v) for v in float_sim.run(-3.75)] == [-3.0]


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
        lower_to_mir(builder.finish(), OPS)


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
    with pytest.raises(UnsupportedConstruct) as exc:
        holoso.synthesize(kernel, OPTIONS, name="never_returns")
    assert exc.value.message == "the kernel provably never returns, so no output of it is ever raised"


def test_an_arm_a_proven_guard_excludes_is_deleted_rather_than_merely_unconvicted() -> None:
    # The dual of the refusal above: the quotient is not spared a conviction, it is not there. A division is
    # unspeculatable, so nothing but pruning can remove it -- and with it the guard, leaving an exact schedule.
    def excluded_by_a_guard(x: float) -> float:
        r = x
        if (x * 0.0) > 1.0:
            r = 1.0 / (x * 0.0)
        return r

    result = _synth(excluded_by_a_guard)
    assert result.initiation_interval[1] is not None, "the guard itself survived"
    assert _instantiated(result) == set(), "the excluded arm's quotient survived the pruning that proves its guard"
    sim = result.numerical_model.elaborate()
    for x in (3.0, -1.0, 0.0):
        assert float(sim.run(x)[0]) == x


def test_a_loop_whose_test_is_proven_false_dissolves_entirely() -> None:
    # Only the graph's ``x*0 == 0`` identity decides this test, so the front end residualizes a real loop.
    def never_enters(x: float) -> float:
        y = x
        while (y * 0.0) > 1.0:
            y = y + 1.0
        return y

    result = _synth(never_enters)
    assert result.initiation_interval[1] is not None, "the loop test survived"
    assert _instantiated(result) == set(), "the body's addition survived a loop that is never entered"
    sim = result.numerical_model.elaborate()
    for x in (3.0, -1.0, 0.0):
        assert float(sim.run(x)[0]) == x


def test_pruning_one_guard_settles_the_next() -> None:
    # Why reduction and pruning are a mutual fixpoint and not a sequence: the second guard is undecidable until the
    # first arm is gone, so one pass of each leaves it standing. Both guards must be gone, not merely the first.
    def cascade(x: float) -> float:
        r = 1.0
        if (x * 0.0) > 1.0:
            r = 2.0
        if r > 1.5:
            return 3.0
        return 4.0

    result = _synth(cascade)
    assert result.initiation_interval[1] is not None
    sim = result.numerical_model.elaborate()
    for x in (3.0, -1.0, 0.0):
        assert float(sim.run(x)[0]) == cascade(x) == 4.0


def test_a_state_slot_live_out_follows_a_merge_pruning_collapses() -> None:
    # A slot's live-out is the one reference outside the value DAG entirely, so a collapse reaches it only by hand:
    # the slot must carry its own live-in forward, and the schedule stays exact once the guard is pruned.
    class HeldByADeadGuard:
        def __init__(self) -> None:
            self.s = 0.0

        def __call__(self, x: float) -> float:
            if (x * 0.0) > 1.0:
                self.s = x
            return self.s

    result = _synth(HeldByADeadGuard().__call__, name="held_by_dead_guard")
    assert result.initiation_interval[1] is not None
    assert [(port.name, port.scalar_type) for port in result.output_ports] == [("state_s", FloatType(FMT))]
    sim = result.numerical_model.elaborate()
    reference = HeldByADeadGuard()
    for x in (3.0, -1.0, 0.0, 7.5):
        assert [float(value) for value in sim.run(x)] == [reference(x)] == [0.0]


def test_a_proven_break_kills_the_back_edge_and_collapses_the_carried_merges() -> None:
    # The latch becomes unreachable, so the header's loop-carried phis lose their latch arm and collapse. Distinct
    # from a loop deleted whole, where no merge has to be repaired at all. The outer runtime loop keeps the public II
    # data-dependent whether or not the break is decided, so the discriminator is blind here: white-box.
    def breaks_on_the_first_trip(x: float, n: float) -> float:
        y = x
        t = n
        while t > 0.0:
            y = y + 1.0
            if (x * 0.0) <= 1.0:
                break
            t = t - 1.0
        return y

    hir = optimize(lower(breaks_on_the_first_trip, DEFAULT_UNROLL_MAX_TRIPS).hir, DEFAULT_IFCONV_MAX_OPS)
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
    once = optimize(lower(kernel, DEFAULT_UNROLL_MAX_TRIPS).hir, DEFAULT_IFCONV_MAX_OPS)
    assert optimize(once, DEFAULT_IFCONV_MAX_OPS) == once


def test_a_loop_phi_is_folded_once_its_latch_arm_has_been_rebuilt() -> None:
    # A phi is opened before its latch arm exists, so the round that emits it cannot see that every arm is one value.
    hir = optimize(
        lower(never_uniform_until_the_latch_arm_is_rebuilt, DEFAULT_UNROLL_MAX_TRIPS).hir, DEFAULT_IFCONV_MAX_OPS
    )
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
    "value,degraded",
    [
        (1e-12, "0.0"),
        (-1e-12, "0.0"),
        (1e30, "inf"),
        (-1e30, "-inf"),
        (0.0, None),
        (math.inf, None),
        (-math.inf, None),
        (1.0, None),
        (9.313225746154785e-10, None),  # the smallest normal
        (4.656612873077393e-10, None),  # half of it, which ties upward to the smallest normal
        (math.nextafter(4.656612873077393e-10, 0.0), "0.0"),  # one ulp below that tie, which encodes to nothing
    ],
    ids=lambda value: repr(value),
)
def test_a_literal_the_format_cannot_hold_is_refused_rather_than_silently_degraded(
    value: float, degraded: str | None
) -> None:
    # The float dual of the integer range refusal. An infinity is representable and so is accepted, where a finite
    # value that encodes to one -- or to zero -- is a literal the machine cannot hold, and substituting what it
    # encodes to would answer for a number the kernel did not write.
    def f(x: float) -> float:
        return x + value

    if degraded is None:
        holoso.synthesize(f, OPTIONS, name="pinned_literal")
    else:
        with pytest.raises(UnsupportedConstruct) as exc:
            holoso.synthesize(f, OPTIONS, name="pinned_literal")
        assert exc.value.message == (
            f"constant {value} degrades to {degraded} in FloatFormat(wexp=6, wman=18); widen wexp or rescale"
        )


def test_a_divisor_whose_reciprocal_degrades_is_carried_apart() -> None:
    # 3e9 is representable but its reciprocal is not, and the reciprocal is what HIR's ``x/c -> x*(1/c)`` hands the
    # machine. Multiplying by it would answer zero for every input; carrying it as a significand this format holds
    # and an exponent no format bounds costs one operation and answers to the format's own precision.
    def f(x: float) -> float:
        return x / 3e9

    result = _synth(f, name="degrading_reciprocal")
    assert _instantiated(result) == {"holoso_fmul", "holoso_fmul_ilog2"}
    sim = result.numerical_model.elaborate()
    for x in (3e9, 1.5e9, 3e8):
        assert within(float(sim.run(x)[0]), x / 3e9, *default_tolerance(FMT, 2, magnitude=x / 3e9)), x


def test_a_state_slot_resetting_to_a_value_the_format_cannot_hold_is_refused() -> None:
    # A reset snapshot never becomes a pooled constant, so the node-level rule never sees it.
    class TinyReset:
        def __init__(self) -> None:
            self.s = 1e-12

        def __call__(self, x: float) -> float:
            self.s = x
            return x

    with pytest.raises(UnsupportedConstruct) as exc:
        holoso.synthesize(TinyReset().__call__, OPTIONS, name="tiny_reset")
    assert exc.value.message == (
        "state slot 's' reset 1e-12 degrades to 0.0 in FloatFormat(wexp=6, wman=18); widen wexp or rescale"
    )


def test_a_float_slot_with_an_integer_reset_is_refused() -> None:
    # The slot's live-out is a float while its reset snapshot is an integer: a slot register holds one family, and
    # only a sweep over the slots themselves sees the mismatch, since no node in the graph carries it. The frontend
    # coerces such a literal, so the mismatched snapshot is only spellable at the builder level.
    builder = HirBuilder()
    builder.block()
    x = builder.input("x", HirFloatType())
    builder.state_slot("s", IntConst(0), x)
    builder.output("y", x)
    builder.ret()
    with pytest.raises(UnsupportedConstruct, match="holds FloatType.. but resets to IntType"):
        lower_to_mir(builder.finish(), OPS)


def test_a_bselect_repeating_its_condition_reduces_to_a_gate() -> None:
    # ``if c: r = a`` over a boolean leaves bselect(c, a, c) -- Python's eager ``and`` shape written as a branch --
    # and its dual leaves bselect(c, c, b). The shape alone cannot tell an ``and`` rewritten as an ``or``; only the
    # truth table can, so both are scored against CPython over the full input cube.
    def gate_shapes(c: bool, a: bool, b: bool) -> tuple[bool, bool]:
        r1 = c
        if c:
            r1 = a
        r2 = b
        if c:
            r2 = c
        return r1, r2

    result = _synth(gate_shapes)
    assert result.initiation_interval[1] is not None
    operators = [node.operator for node in _hir_of(gate_shapes).nodes.values() if isinstance(node, Operation)]
    assert not any(isinstance(op, BoolSelect) for op in operators), "a condition-repeating bselect survived"
    assert any(isinstance(op, BoolAnd) for op in operators) and any(isinstance(op, BoolOr) for op in operators)
    sim = result.numerical_model.elaborate()
    for combo in itertools.product([False, True], repeat=3):
        assert list(sim.run(*combo)) == list(gate_shapes(*combo)), combo
