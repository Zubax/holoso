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

import pytest

import holoso
from holoso import (
    FAddOperator,
    FCmpOperator,
    FDivOperator,
    FloatFormat,
    FMulILog2OperatorFamily,
    FMulOperator,
    OpConfig,
    UnsupportedConstruct,
)
from holoso._frontend import lower
from holoso._hir import (
    FloatAdd,
    FloatToInt,
    Hir,
    InPort,
    IntAdd,
    IntConst,
    IntDivFloor,
    IntMod,
    IntMul,
    IntNeg,
    IntRelational,
    IntSub,
    IntToFloat,
    IntType,
    Operation,
    optimize,
)
from holoso._mir import lower as lower_to_mir

FMT = FloatFormat(6, 18)


def _ops() -> OpConfig:
    return OpConfig(
        FAddOperator(FMT), FMulOperator(FMT), FDivOperator(FMT), FMulILog2OperatorFamily(FMT), FCmpOperator(FMT)
    )


def _hir(kernel: object) -> Hir:
    return optimize(lower(kernel))


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
