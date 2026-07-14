"""
Integer frontend + HIR readiness (stage 8). Integer kernels lower Python -> FIR -> HIR with the integer operator
vocabulary; the integer backend (MIR/model/Verilog) is a later wiring milestone, so a runtime-integer kernel is a
LOCATED MIR rejection, not a runnable model. These tests use the three layers Codex prescribed, none a substitute for
another:

  1. Typed HIR topology -- ``optimize(lower(kernel))`` produces the expected operators AND types/wiring (catches a
     swapped operand, a missing promotion, a wrong port type -- things an operator-set check alone would miss).
  2. CPython static-fold oracle -- an all-constant integer kernel folds to an ``IntConst`` whose value equals running
     the kernel in CPython, exactly, including huge integers and every ``//``/``%`` sign quadrant.
  3. MIR containment -- a runtime-integer kernel raises the located "not yet lowerable" rejection at MIR with the
     exact operator mnemonic, proving nothing integer silently reaches the backend.
"""

from math import floor

import numpy as np
import pytest

import holoso
from holoso import (
    FAddOperator,
    FCmpOperator,
    FDivOperator,
    FloatFormat,
    FMulILog2OperatorFamily,
    FMulOperator,
    FRoundOperator,
    FSortOperator,
    OpConfig,
    UnsupportedConstruct,
)
from holoso._frontend import lower
from holoso._hir import (
    FloatAdd,
    FloatToInt,
    Hir,
    InPort,
    IntAbs,
    IntAdd,
    IntConst,
    IntDivFloor,
    IntMod,
    IntMul,
    IntNeg,
    IntRelational,
    IntSelect,
    IntSub,
    IntToFloat,
    IntType,
    Operation,
    optimize,
)
from holoso._mir import MirInterpreter, lower as lower_to_mir
from holoso._value import FloatValue

FMT = FloatFormat(6, 18)


def _ops() -> OpConfig:
    return OpConfig(
        FAddOperator(FMT),
        FMulOperator(FMT),
        FDivOperator(FMT),
        FMulILog2OperatorFamily(FMT),
        FCmpOperator(FMT),
        fround=FRoundOperator(FMT),
        fsort=FSortOperator(FMT),
    )


def _hir(kernel: object) -> Hir:
    return optimize(lower(kernel))


def _run_model(kernel: object, *args: float | bool) -> list[float | bool]:
    """One transaction through the numerical model at FMT; the e6m18 rounding of promoted integers is the point."""
    interpreter = MirInterpreter(lower_to_mir(_hir(kernel), _ops()))
    encoded = [a if type(a) is bool else FloatValue.from_float(FMT, float(a)) for a in args]
    return [v if isinstance(v, bool) else float(v) for v in interpreter.run(*encoded)]


def _op_names(hir: Hir) -> set[str]:
    return {type(n.operator).__name__ for n in hir.nodes.values() if isinstance(n, Operation)}


def _int_consts(hir: Hir) -> list[int]:
    return [n.value for n in hir.nodes.values() if isinstance(n, IntConst)]


# ----------------------------------- 1. typed HIR topology -----------------------------------


def test_integer_arithmetic_lowers_to_integer_operators() -> None:
    def kernel(a: int, b: int) -> int:
        return a * b - a // b + a % b

    hir = _hir(kernel)
    assert {IntMul.__name__, IntSub.__name__, IntDivFloor.__name__, IntMod.__name__} <= _op_names(hir)
    assert all(isinstance(n.type, IntType) for n in hir.nodes.values() if isinstance(n, InPort))  # int input ports
    assert IntToFloat.__name__ not in _op_names(hir)  # a pure-integer kernel never touches the float datapath


def test_integer_parameter_and_return_ports_are_typed_integer() -> None:
    def kernel(a: int, b: int) -> int:
        return a + b

    hir = _hir(kernel)
    inputs = [n for n in hir.nodes.values() if isinstance(n, InPort)]
    assert {n.name for n in inputs} == {"a", "b"} and all(isinstance(n.type, IntType) for n in inputs)
    assert _op_names(hir) == {IntAdd.__name__}


def test_integer_comparison_lowers_to_integer_relational() -> None:
    def kernel(a: int, b: int) -> bool:
        return a < b

    assert _op_names(_hir(kernel)) == {IntRelational.__name__}


def test_integer_negation_lowers_to_integer_negate() -> None:
    def kernel(a: int) -> int:
        return -a

    assert _op_names(_hir(kernel)) == {IntNeg.__name__}


def test_int_plus_float_promotes_the_integer_edge_then_adds_in_float() -> None:
    def kernel(a: int, x: float) -> float:
        return a + x  # Python promotes the int to float

    ops = _op_names(_hir(kernel))
    assert IntToFloat.__name__ in ops and FloatAdd.__name__ in ops


def test_true_division_of_integers_promotes_to_float() -> None:
    def kernel(a: int, b: int) -> float:
        return a / b  # Python ``/`` is always float, even on two integers

    ops = _op_names(_hir(kernel))
    assert IntToFloat.__name__ in ops and "FloatDiv" in ops and IntDivFloor.__name__ not in ops


def test_runtime_float_to_int_and_int_to_float_casts_lower_to_conversions() -> None:
    def to_int(x: float) -> int:
        return int(x)  # truncation toward zero

    def to_float(a: int) -> float:
        return float(a)

    assert _op_names(_hir(to_int)) == {FloatToInt.__name__}
    assert _op_names(_hir(to_float)) == {IntToFloat.__name__}


# ----------------------------------- 2. CPython static-fold oracle -----------------------------------


def _fold_oracle(kernel: object) -> None:
    hir = _hir(kernel)
    assert _int_consts(hir) == [kernel()], f"HIR {_int_consts(hir)} vs CPython {kernel()}"  # type: ignore[operator]


def test_huge_integers_fold_exactly_without_rounding() -> None:
    def a() -> int:
        return 2**53 + 1  # inexact in float64; the exact MetaInt fold must not round it to 2**53

    def b() -> int:
        return 10**30 - 1

    def c() -> int:
        return 123456789 * 987654321

    for kernel in (a, b, c):
        _fold_oracle(kernel)


def test_floor_division_folds_toward_negative_infinity_in_every_sign_quadrant() -> None:
    def pp() -> int:
        return 7 // 2

    def np_() -> int:
        return (-7) // 2  # rounds toward -inf (-4), unlike C truncation (-3)

    def pn() -> int:
        return 7 // (-2)

    def nn() -> int:
        return (-7) // (-2)

    for kernel in (pp, np_, pn, nn):
        _fold_oracle(kernel)


def test_modulo_takes_the_sign_of_the_divisor_in_every_quadrant() -> None:
    def pp() -> int:
        return 7 % 2

    def np_() -> int:
        return (-7) % 2  # Python: +1, the sign follows the divisor

    def pn() -> int:
        return 7 % (-2)

    def nn() -> int:
        return (-7) % (-2)

    for kernel in (pp, np_, pn, nn):
        _fold_oracle(kernel)


def test_unary_and_builtin_integer_folds_match_cpython() -> None:
    def neg() -> int:
        return -(2**40)

    def absv() -> int:
        return abs(-(2**60))

    def mixed() -> int:
        return 2 + 3 * 4 - 5

    for kernel in (neg, absv, mixed):
        _fold_oracle(kernel)


def test_a_signed_integer_return_is_an_integer_output_port() -> None:
    def kernel() -> int:
        return 42

    # A scalar integer return leaves the wide bank as an integer-typed value (leaf 0 of the return bundle).
    assert _int_consts(_hir(kernel)) == [42]


# ----------------------------------- 3. MIR containment -----------------------------------


def _runtime_int_kernels() -> list[object]:
    def add(a: int, b: int) -> int:
        return a + b

    def sub(a: int, b: int) -> int:
        return a - b

    def mul(a: int, b: int) -> int:
        return a * b

    def floordiv(a: int, b: int) -> int:
        return a // b

    def mod(a: int, b: int) -> int:
        return a % b

    def neg(a: int) -> int:
        return -a

    def cmp(a: int, b: int) -> bool:
        return a <= b

    def to_int(x: float) -> int:
        return int(x)

    return [add, sub, mul, floordiv, mod, neg, cmp, to_int]


@pytest.mark.parametrize("kernel", _runtime_int_kernels(), ids=lambda k: k.__name__)
def test_runtime_integer_kernel_is_a_located_mir_rejection(kernel: object) -> None:
    # HIR readiness only: every integer operator family lowers through FIR to HIR, and MIR refuses it cleanly (the
    # integer backend is the wiring milestone). The refusal is a located SynthesisError, never a raw crash.
    hir = _hir(kernel)
    with pytest.raises(UnsupportedConstruct):
        lower_to_mir(hir, _ops())


def test_runtime_integer_rejection_is_reachable_through_public_synthesize() -> None:
    def counter(a: int, b: int) -> int:
        return a * b + b

    with pytest.raises(UnsupportedConstruct):
        holoso.synthesize(counter, _ops(), name="int_counter")


# ----------------------------------- bitwise, shifts, and the boolean bank -----------------------------------


def test_integer_bitwise_and_shift_operators_lower() -> None:
    def kernel(a: int, b: int) -> int:
        return ((a & b) | (a ^ b)) + (a << 2) + (a >> 1)

    assert {"IntAnd", "IntOr", "IntXor", "IntShiftLeft", "IntShiftRight"} <= _op_names(_hir(kernel))


def test_boolean_bitwise_stays_in_the_boolean_bank() -> None:
    def kernel(a: bool, b: bool) -> bool:
        return (a ^ b) & (a | b)

    names = _op_names(_hir(kernel))
    assert {"BoolXor", "BoolAnd", "BoolOr"} <= names
    assert not any(name.startswith("Int") for name in names)  # booleans never leak into the integer bank


def test_compile_time_negative_shift_count_is_rejected() -> None:
    def kernel(a: int) -> int:
        return a << -1  # Python raises; a compile-time-known negative count is a located refusal

    with pytest.raises(UnsupportedConstruct, match="negative shift count"):
        lower(kernel)


def test_boolean_shift_and_mixed_int_bool_bitwise_are_rejected() -> None:
    def bool_shift(a: bool) -> int:
        return a << 2  # a boolean shift is not modelled; an explicit cast is required

    def mixed(a: bool, n: int) -> int:
        return a | n  # bool-as-int promotion is not modelled

    for fn in (bool_shift, mixed):
        with pytest.raises(UnsupportedConstruct, match="two integers"):
            lower(fn)


def test_a_huge_static_left_shift_does_not_hang_the_compiler() -> None:
    def zero_shift() -> int:
        return 0 << (10**9)  # folds to zero regardless of the count

    def one_shift() -> int:
        return 1 << (10**9)  # defers to a runtime shift rather than materializing an astronomical integer

    assert _int_consts(_hir(zero_shift)) == [0]
    assert "IntShiftLeft" in _op_names(_hir(one_shift))


# ----------------------------------- conversion elisions -----------------------------------


def test_float_of_int_truncation_elides_the_integer_round_trip() -> None:
    def truncate(x: float) -> float:
        return float(int(x))  # i2f(f2i(x)) canonicalizes to FloatTrunc(x); no integer node survives

    # The integer round-trip is gone -- a pure float truncation remains, so no FloatToInt reaches MIR to reject.
    assert _op_names(_hir(truncate)) == {"FloatTrunc"}


def test_float_of_int_of_an_integer_valued_float_elides_entirely() -> None:
    def via_floor(x: float) -> float:
        return float(int(floor(x)))  # floor is already integral, so the integer round trip vanishes

    names = _op_names(_hir(via_floor))
    assert "FloatFloor" in names and "FloatToInt" not in names and "IntToFloat" not in names


# ----------------------------------- runtime power to exp2 -----------------------------------


def test_power_of_two_with_a_runtime_exponent_lowers_to_exp2() -> None:
    def int_base(x: float) -> float:
        return 2**x

    def float_base(x: float) -> float:
        return 2.0**x  # type: ignore[no-any-return]  # typeshed types float**float as Any (negative base -> complex)

    for kernel in (int_base, float_base):
        assert _op_names(_hir(kernel)) == {"FloatExp2"}


def test_a_non_two_base_with_a_runtime_exponent_lowers_via_exp2_log2() -> None:
    # The general runtime-exponent power is the direct fastmath identity exp2(e * log2(b)); the constant base's log2
    # folds statically, so 10**x costs one multiply and one exp2.
    def ten_base(x: float) -> float:
        return 10**x

    names = _op_names(_hir(ten_base))
    assert "FloatExp2" in names and "FloatLog2" not in names  # log2(10) folded to a constant


# ----------------------------------- integer truthiness and persistent state -----------------------------------


def test_integer_truthiness_lowers_to_a_format_agnostic_int_to_bool() -> None:
    # Regression: an integer in a truthiness position used to emit FloatToBool on an integer value and crash the
    # builder; it now lowers to IntToBool, contained at the integer boundary.
    def kernel(x: float) -> float:
        return 1.0 if int(x) else 2.0

    assert "IntToBool" in _op_names(_hir(kernel))


def test_a_known_integer_stored_into_integer_state_stays_typed() -> None:
    # Regression: a Known integer written to an integer state slot materialized as a float and mismatched the integer
    # phi, crashing the builder; it now stays a typed integer.
    class Counter:
        def __init__(self) -> None:
            self.n = 0

        def step(self, tick: bool) -> float:
            if tick:
                self.n = 1
            return float(self.n)

    hir = lower(Counter().step)  # emits without a phi-type crash
    (slot,) = hir.state_slots
    assert isinstance(hir.nodes[slot.live_out].type, IntType)


def test_integer_state_counter_is_a_located_mir_rejection() -> None:
    class Counter:
        def __init__(self) -> None:
            self.n = 0

        def step(self, tick: bool) -> float:
            if tick:
                self.n = self.n + 1
            return float(self.n)

    with pytest.raises(UnsupportedConstruct):
        holoso.synthesize(Counter().step, _ops(), name="int_state_counter")


# ------------------------ regressions: integer values must never float-promote and round ------------------------


def test_abs_of_an_integer_operand_is_contained_not_float_promoted() -> None:
    # abs preserves the operand kind: abs(int(x)) is an integer (IntAbs), so the following integer arithmetic stays
    # exact and is contained at MIR, never promoted to float and rounded (2**53 + 1 -> 2**53).
    def kernel(x: float) -> float:
        return float(abs(int(x)) + 1 + 1)

    assert IntAbs.__name__ in _op_names(_hir(kernel))  # abs(int) is IntAbs, not a float promotion
    with pytest.raises(UnsupportedConstruct, match="not yet lowerable"):
        lower_to_mir(_hir(kernel), _ops())


def test_math_floor_is_integer_returning() -> None:
    # math.floor(x) is an int in Python, so integer arithmetic on it stays exact (an integer-backend rejection) rather
    # than rounding in float; a pure float use elides the int round-trip back to the float rounding operator.
    def int_context(x: float) -> float:
        return float(floor(x) + 1 + 1)

    def float_context(x: float) -> float:
        return floor(x) * 2.0

    with pytest.raises(UnsupportedConstruct):
        lower_to_mir(_hir(int_context), _ops())
    names = _op_names(_hir(float_context))
    assert "FloatFloor" in names and "FloatToInt" not in names  # the integer round-trip elided, metrics preserved


def test_integer_base_raised_to_a_known_power_stays_integer_and_is_contained() -> None:
    def kernel(x: float) -> float:
        return float(int(x) ** 3)  # exact integer exponentiation (an IntMul chain), not a rounding float chain

    hir = _hir(kernel)
    assert IntMul.__name__ in _op_names(hir) and "FloatMul" not in _op_names(hir)
    with pytest.raises(UnsupportedConstruct, match="not yet lowerable"):
        lower_to_mir(hir, _ops())


def test_numpy_sign_of_an_integer_is_rejected() -> None:
    # np.sign is int-polymorphic (np.sign of an integer is an integer); its float composite would round subsequent
    # integer arithmetic, so an integer operand is a located rejection.
    def kernel(x: float) -> float:
        return float(np.sign(int(x)) + int(x) + 1)

    with pytest.raises(UnsupportedConstruct, match="np.sign"):
        lower(kernel)


def test_an_integer_beyond_the_binary64_carrier_is_a_clean_rejection() -> None:
    # A finite inexact integer merged with a float rounds (fastmath), but 10**400 is beyond the binary64 carrier
    # entirely, so the promotion at the merge is a located rejection, never a raw OverflowError.
    def kernel(flag: bool, x: float) -> float:
        if flag:
            y = 10**400
        else:
            y = x  # type: ignore[assignment]  # deliberately mixes int and float across the merge
        return y

    with pytest.raises(UnsupportedConstruct):
        lower_to_mir(_hir(kernel), _ops())


def test_integer_and_or_lowers_to_int_select_without_crashing() -> None:
    # Both a runtime-int arm and a Known-int arm must stay integer: a Known integer materialized as a float would make
    # the select FLOAT while the analyzer typed it INT, crashing the integer phi (regression).
    def runtime_arms(a: int, b: int) -> int:
        return a or b

    def known_arm(a: int) -> int:
        return a or 5

    for kernel in (runtime_arms, known_arm):
        assert "IntSelect" in _op_names(_hir(kernel))  # IntSelect, then the integer-backend rejection, never a crash


def test_a_known_integer_stored_into_a_float_state_slot_stays_float() -> None:
    # A Known integer written to a FLOAT-typed leaf materializes as a float, matching the slot, not an IntConst.
    class K:
        def __init__(self) -> None:
            self.x = 0.0

        def step(self) -> float:
            self.x = 1
            return self.x

    (slot,) = lower(K().step).state_slots
    assert type(lower(K().step).nodes[slot.live_out].type).__name__ == "FloatType"


def test_min_max_of_a_known_integer_and_a_runtime_float_promotes_to_float_min_max() -> None:
    # min/max mixing an integer with a float promote the integer operand and select in the float datapath; the
    # winning operand's Python type is not preserved (the documented C-style deviation from builtin min/max).
    def kernel(x: float) -> float:
        return float(min(2**24, x) + 1 + 1)

    hir = _hir(kernel)
    assert "FloatMin" in _op_names(hir) and IntSelect.__name__ not in _op_names(hir)
    assert _run_model(kernel, 0.5) == [2.5]


def test_integer_base_to_the_zeroth_power_is_the_integer_one_and_is_contained() -> None:
    # int(x)**0 is the integer 1 in Python, so the following arithmetic stays exact integer (contained at MIR),
    # never a floated 1.0 (regression: the power==0 shortcut once fired before the integer-base check).
    def kernel(x: float) -> float:
        return float(int(x) ** 0 + int(x) + 1)

    hir = _hir(kernel)
    assert IntAdd.__name__ in _op_names(hir)
    with pytest.raises(UnsupportedConstruct, match="not yet lowerable"):
        lower_to_mir(hir, _ops())


def test_a_constant_boolean_bitwise_guard_folds_and_lowers() -> None:
    # Regression: True & False folds to a Known bool in the analyzer, but emission replayed it as a runtime bitwise and
    # hit "boolean reaches a float operation"; the guard must fold so the branch resolves.
    def kernel(x: float) -> float:
        return x if True & False else -x

    assert _op_names(_hir(kernel)) == {"FloatNeg"}  # the guard folded to False -> the -x arm, no bitwise/select


def test_a_runtime_integer_stored_into_a_float_slot_promotes() -> None:
    # Regression: math.floor(v) is an integer; storing it to a float leaf must promote on that edge, not leave a
    # FloatToInt that the slot's float reset mismatches.
    class K:
        def __init__(self) -> None:
            self.x = 0.0

        def step(self, v: float) -> float:
            self.x = floor(v)
            return self.x

    (slot,) = lower(K().step).state_slots
    assert type(lower(K().step).nodes[slot.live_out].type).__name__ == "FloatType"


# --------- an int/float control-flow merge promotes the integer path to float, C-style (accepted rounding) ---------


def test_a_runtime_integer_merge_promotes_to_float_and_lowers() -> None:
    # int(x) merged with a float at a phi (if/else) or a conditional select promotes the integer path to float on its
    # own edge, so the following arithmetic runs in the float datapath. Python keeps each path's runtime kind; the
    # promotion is the documented C-style deviation, its rounding accepted under the fastmath charter. The promoted
    # int(x) arm collapses to FloatTrunc (i2f(f2i(x))), keeping the truncation inside the float datapath.
    def phi_join(flag: bool, x: float) -> float:
        if flag:
            y = int(x)
        else:
            y = x  # type: ignore[assignment]  # deliberately mixes int and float across the merge
        return y + 1 + 1

    def sel_mixed(flag: bool, x: float) -> float:
        y = int(x) if flag else x
        return y + 1 + 1

    for kernel in (phi_join, sel_mixed):
        assert "FloatTrunc" in _op_names(_hir(kernel))
        assert _run_model(kernel, True, 2.5) == [4.0]  # trunc(2.5) + 2
        assert _run_model(kernel, False, 2.5) == [4.5]


def test_a_constant_integer_merge_promotes_and_rounds_in_the_target_format() -> None:
    # A Known integer arm merged with a float promotes to a float constant; a value the target format cannot hold
    # exactly rounds (2**18 + 1 -> 2**18 in e6m18) -- accepted C-style precision loss, not a rejection. The two
    # datapath +1 additions round back onto 2**18 each time (ties-to-even), pinning the promoted semantics.
    def kernel(flag: bool, x: float) -> float:
        v = 2**18 + 1 if flag else x
        return v + 1 + 1

    assert _run_model(kernel, True, 0.0) == [float(2**18)]
    assert _run_model(kernel, False, 1.0) == [3.0]


def test_a_mixed_merge_promotes_in_arithmetic_and_comparison_alike() -> None:
    # Arithmetic against a definite float operand promotes every path (as Python does); a comparison likewise promotes
    # the integer path and compares in float -- no exactness obligation remains.
    def promote(flag: bool, x: float) -> float:
        y = int(x) if flag else x
        return y + 1.5

    def compare(flag: bool, x: float) -> float:
        y = int(x) if flag else x
        return 1.0 if y > 2.0 else 0.0

    assert _run_model(promote, True, 2.5) == [3.5]
    assert _run_model(compare, True, 2.9) == [0.0]  # trunc(2.9) == 2.0 is not > 2.0
    assert _run_model(compare, False, 2.9) == [1.0]


def test_a_runtime_integer_state_join_promotes_the_slot_to_float() -> None:
    # A conditional runtime-integer store into a float-reset slot promotes the slot to float: the integer store edge
    # casts on its own edge, the live-in stays float, and reads run in the float datapath.
    class K:
        def __init__(self) -> None:
            self.y = 0.0

        def step(self, flag: bool, x: float) -> float:
            if flag:
                self.y = int(x)
            return self.y + 1 + 1

    hir = lower(K().step)
    slot = next(s for s in hir.state_slots if s.name == "y")
    assert type(hir.nodes[slot.live_out].type).__name__ == "FloatType"
    assert _run_model(K().step, True, 3.5) == [5.0, 3.0]  # return trunc(3.5)+2, then the public state port y


def test_an_int_float_comparison_promotes_and_rounds_in_the_target_format() -> None:
    # Python compares an integer and a float exactly; Holoso instead promotes the integer into the float datapath,
    # where 2**18 + 1 rounds onto 2**18 in e6m18 -- so the comparison sees the rounded constant. Accepted C-style
    # deviation (the plain-Python oracle legitimately diverges here); a representable integer compares unchanged.
    def rounded(x: float) -> float:
        return 1.0 if (2**18 + 1) == x else 0.0

    def representable(x: float) -> float:
        return 1.0 if x > 5 else 0.0

    assert _run_model(rounded, float(2**18)) == [1.0]  # False in Python, True after the accepted rounding
    assert _run_model(rounded, float(2**18 + 2)) == [0.0]
    assert _run_model(representable, 6.0) == [1.0]


def test_min_max_of_a_known_integer_operand_stays_integer_and_is_contained() -> None:
    # Regression (review round 4): max(int(x), 2**18 + 1) typed the result Residual(INT), but the emitter read the Known
    # integer constant as float (value_of floats it via _const) and emitted a rounding FloatMax that lowered and
    # miscompiled (262144.0 vs CPython 262145.0). A Known-integer min/max operand must materialize as an IntConst
    # (arm_value), keeping the operation an IntSelect that is contained at MIR.
    def kernel(x: float) -> float:
        return max(int(x), 2**18 + 1)

    ops = _op_names(_hir(kernel))
    assert IntSelect.__name__ in ops and "FloatMax" not in ops  # an integer select, not a rounding float max
    with pytest.raises(UnsupportedConstruct, match="not yet lowerable"):
        lower_to_mir(_hir(kernel), _ops())


def test_a_promoted_merge_flows_through_every_use_as_a_plain_float() -> None:
    # A promoted int/float merge is an ordinary float everywhere: unary negation, an intrinsic (np.minimum, abs), and
    # an aggregate element all lower and compute in the float datapath (review round 4 rejected each of these).
    def unary(flag: bool, x: float) -> float:
        y = 2**18 + 1 if flag else x
        return 1.0 if -y == -float(2**18) else 0.0  # the promoted constant rounds onto 2**18, then negates

    def numpy_min(flag: bool, x: float) -> float:
        y = 2**18 if flag else x
        return float(np.minimum(y, 2**18) + 1 + 1)

    def in_abs(flag: bool, x: float) -> float:
        y = 5 if flag else x
        return 1.0 if abs(y) > 2.0 else 0.0

    def in_list(flag: bool, x: float) -> float:
        y = 2**18 + 1 if flag else x
        return 1.0 if [y][0] == float(2**18) else 0.0

    assert _run_model(unary, True, 0.0) == [1.0]
    assert _run_model(numpy_min, False, 5.0) == [7.0]  # min(5.0, 2**18) + 2
    assert _run_model(in_abs, True, 0.0) == [1.0]  # abs(5) promoted: 5.0 > 2.0
    assert _run_model(in_list, True, 0.0) == [1.0]


def test_a_known_integer_store_to_a_float_slot_folds_the_rounded_value() -> None:
    # Review round 1 (Codex): the store into a float slot rounds the integer into the binary64 carrier, so the fact a
    # read-back sees must be the ROUNDED float -- folding the comparison with the exact 2**53 + 1 would disagree with
    # the value the datapath actually stored (observed: model returned 0.0 where the RTL stores 2**53).
    class K:
        def __init__(self) -> None:
            self.y = 0.0

        def step(self) -> float:
            self.y = 2**53 + 1
            return 1.0 if self.y == float(2**53) else 0.0

    hir = _hir(K().step)
    from holoso._hir import FloatConst, Operation

    (out,) = [o for o in hir.outputs if o.name == "out_0"]
    result = hir.nodes[out.value]
    assert isinstance(result, FloatConst) and result.value == 1.0  # folded on the rounded store, matching the datapath
    assert not any(isinstance(n, Operation) for n in hir.nodes.values())


def test_a_nonpositive_known_base_with_a_runtime_exponent_is_rejected() -> None:
    # Review round 1 (Codex): 0.0**e through the general exp2(e * log2(b)) path would assert the log2 pole error on
    # every transaction (Python: plain 0.0). A compile-time nonpositive base is a located rejection instead; a runtime
    # base keeps the documented C-style log2 domain-error behavior.
    def zero_base(e: float) -> float:
        return 0.0**e  # type: ignore[no-any-return]  # typeshed types float**float as Any

    def negative_base(e: float) -> float:
        return (-2.0) ** e  # type: ignore[no-any-return]

    for kernel in (zero_base, negative_base):
        with pytest.raises(UnsupportedConstruct, match="positive base"):
            lower(kernel)


def test_a_known_integer_rejoining_a_pure_int_phi_stays_integer() -> None:
    # Review round 1: a diamond whose arms both bind the same Known integer used to type the rejoin phi as float,
    # crashing a later pure-int phi with a raw ValueError instead of lowering to a typed integer merge.
    def kernel(x: float, a: bool, b: bool) -> float:
        if a:
            i = 0
            u = x + 1.0
        else:
            i = 0
            u = x - 1.0
        if b:
            i = int(u)
        return float(i + 1)

    hir = _hir(kernel)
    assert IntAdd.__name__ in _op_names(hir)
    with pytest.raises(UnsupportedConstruct, match="not yet lowerable"):
        lower_to_mir(hir, _ops())


def test_an_integer_element_of_an_aggregate_stays_integer() -> None:
    # Review round 1: an integer element flowing through a tuple/list must keep its integer typing (it used to be
    # reclassified float by the aggregate handle, escaping the MIR containment).
    def kernel(x: float) -> float:
        return 1.0 if [int(x)][0] == 3 else 0.0

    hir = _hir(kernel)
    assert IntRelational.__name__ in _op_names(hir) and "FloatRelational" not in _op_names(hir)
    with pytest.raises(UnsupportedConstruct, match="not yet lowerable"):
        lower_to_mir(hir, _ops())


def test_a_known_integral_float_exponent_expands_to_a_chain() -> None:
    # x ** 3.0 is the same monomial as x ** 3 under fastmath: it expands to the multiply chain instead of paying the
    # exp2/log2 pair (which would also drag in a domain-error cone for a negative base).
    def kernel(x: float) -> float:
        return x**3.0  # type: ignore[no-any-return]

    names = _op_names(_hir(kernel))
    assert "FloatMul" in names and "FloatExp2" not in names and "FloatLog2" not in names
    assert _run_model(kernel, 2.0) == [8.0]


def test_int_float_int_round_trip_collapses_to_the_identity() -> None:
    # The fastmath charter: int -> float -> int collapses away completely, precision loss ignored.
    def kernel(a: int) -> int:
        return int(float(a))

    assert _op_names(_hir(kernel)) == set()


def test_an_inexact_integer_constant_in_a_float_position_rounds() -> None:
    # 2**53 + 1 is not binary64-exact; the promotion rounds it under fastmath instead of rejecting.
    from holoso._hir import FloatConst

    def kernel(x: float) -> float:
        return x + (2**53 + 1)

    hir = _hir(kernel)
    assert float(2**53) in [n.value for n in hir.nodes.values() if isinstance(n, FloatConst)]
