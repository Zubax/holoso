"""
The partial-evaluator scalar core, driven black-box through the differential oracle wherever the behavior is
observable (kernels defined here, lowered through ``holoso._eel`` and compared against CPython), with residual
pins on the location-stripped public ``frontend_ir[-1]`` where the behavior is invisible to the oracle
(static-control folding, orphan dropping). Survivor refusals are pinned publicly as bare ``SynthesisError``
(never ``UnsupportedConstruct``: the kernel residualized rather than being refused early). Rejections pin one
located diagnostic per family.
"""

import math
import types
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pytest

import holoso
from holoso import (
    FAddOptions,
    FCmpOptions,
    FDivOptions,
    FMulILog2Options,
    FMulOptions,
    OperatorOptions,
    Options,
    SynthesisError,
    UnsupportedConstruct,
)
from holoso._eel import lower
from holoso._eel._desugar import desugar
from holoso._eel._lower import resolve_target
from holoso._eel._pe import partial_evaluate

from ._modelref import DEFAULT_UNROLL_MAX_TRIPS
from holoso._eel._print import print_eel
from holoso._hir import HirEvaluator

from ._eeloracle import assert_hir_matches_reference
from ._public import strip_locations

type _Row = Mapping[str, float | bool | int]

_INT_ONLY = Options(OperatorOptions())
_FADD = Options(OperatorOptions(fadd=FAddOptions()))
_FMUL = Options(OperatorOptions(fmul=FMulOptions()))
_FCMP = Options(OperatorOptions(fcmp=FCmpOptions()))
_FADD_FMUL = Options(OperatorOptions(fadd=FAddOptions(), fmul=FMulOptions()))
_FADD_FDIV = Options(OperatorOptions(fadd=FAddOptions(), fdiv=FDivOptions()))
_FMUL_ILOG2 = Options(OperatorOptions(fmul_ilog2=FMulILog2Options()))

_NEVER = False
_ENABLED = True
_SCALE = 4
_GAIN = 2.5
_BIG_INT = 2**54 + 1
_HUGE_INT = 10**400
_NAN_VALUE = float("nan")
_NP_GAIN = np.float64(1.5)
_BOX = types.SimpleNamespace(value=0.0)
_XS = (1.0, 2.0)


def _oracle(fn: Callable[..., object], vectors: Sequence[_Row]) -> None:
    compared = assert_hir_matches_reference(lower(fn, DEFAULT_UNROLL_MAX_TRIPS).hir, fn, vectors, label=fn.__name__)
    assert compared == len(vectors)


def _rejects(fn: object, match: str) -> None:
    assert callable(fn)
    with pytest.raises(UnsupportedConstruct, match=match):
        holoso.synthesize(fn, _INT_ONLY, name="k")


def _residual_text(fn: Callable[..., object]) -> str:
    """For the kernels public synthesis refuses; everything that synthesizes pins text via :func:`_residual`."""
    assert isinstance(fn, types.FunctionType)
    return print_eel(partial_evaluate(desugar(fn), fn, None, DEFAULT_UNROLL_MAX_TRIPS))


def _residual(fn: Callable[..., object], options: Options) -> str:
    return strip_locations(holoso.synthesize(fn, options, name="k").frontend_ir[-1])


def _has_residual_if(text: str) -> bool:
    return any(line.lstrip().startswith("if ") for line in text.splitlines())


# ---------------------------------------------------------------------- arithmetic, promotion, casts


def _mixed(a: float, n: int) -> float:
    return (a - n) * _SCALE + n / 2 - a / n + float(n) * a


def test_mixed_arithmetic_and_promotion() -> None:
    _oracle(_mixed, [{"a": 1.5, "n": 2}, {"a": -0.25, "n": 7}, {"a": 3.0, "n": -3}, {"a": 0.5, "n": 1}])


def _compare_mix(n: int, x: float) -> tuple[bool, bool, bool]:
    return n < x, x <= n, n == 2


def test_mixed_comparisons_promote() -> None:
    _oracle(_compare_mix, [{"n": 2, "x": 2.5}, {"n": 3, "x": 3.0}, {"n": -1, "x": -1.5}, {"n": 2, "x": 1.0}])


def _big_promotion(x: float) -> float:
    return x * _BIG_INT


def test_static_int_meets_float_as_its_image() -> None:
    _oracle(_big_promotion, [{"x": 1.0}, {"x": -0.5}, {"x": 3.25}])


def _int_return_promotes(n: int) -> float:
    return n


def test_declared_float_accepts_int_lane() -> None:
    _oracle(_int_return_promotes, [{"n": 3}, {"n": -17}, {"n": 0}])


def _gate_with_static(x: float) -> bool:
    return _ENABLED and x > 0.0


def test_gate_with_static_operand_stays_residual() -> None:
    _oracle(_gate_with_static, [{"x": 1.0}, {"x": -1.0}, {"x": 0.0}])
    assert "band(True," in _residual(_gate_with_static, _FCMP)


# ---------------------------------------------------------------------- control, joins, definite assignment


def _ternary_promotes(c: bool, x: float) -> float:
    return 1 if c else x


def test_join_promotes_static_int_arm() -> None:
    _oracle(_ternary_promotes, [{"c": True, "x": 5.0}, {"c": False, "x": 5.0}, {"c": False, "x": -2.5}])


def _augmented(x: float, n: int) -> tuple[float, int]:
    y = x
    y += 2.5
    y *= x
    m = n
    m //= 2
    m <<= 1
    return y, m


def test_augmented_assignment_rebinds() -> None:
    _oracle(_augmented, [{"x": 1.5, "n": 9}, {"x": -0.5, "n": -9}])


def _static_taken(x: float) -> float:
    if _ENABLED:
        return x * 2.0
    return x / 0.0


def test_static_condition_folds_the_branch_away() -> None:
    _oracle(_static_taken, [{"x": 1.5}, {"x": -2.0}])
    assert not _has_residual_if(_residual(_static_taken, _FMUL_ILOG2))


def _dead_arm_is_lazy(x: float) -> float:
    if _NEVER:
        bad = NEVER_DEFINED + True  # type: ignore[name-defined]  # noqa: F821
        return bad  # type: ignore[no-any-return]
    return x + 1.0


def test_dead_arm_semantics_are_never_judged() -> None:
    """
    The split: the desugar whitelist judges syntax even in dead arms, while the partial evaluator's
    semantics (name resolution, typing) run only where evaluation reaches -- exactly CPython's laziness.
    """
    _oracle(_dead_arm_is_lazy, [{"x": 1.0}, {"x": -1.0}])


def _equal_static_arms(c: bool) -> float:
    return 1 if c else 1.0


def test_equal_arms_join_statically_and_drop_the_branch() -> None:
    _oracle(_equal_static_arms, [{"c": True}, {"c": False}])
    assert not _has_residual_if(_residual(_equal_static_arms, _INT_ONLY))


def _rebind_across_branch(c: bool, x: float) -> float:
    y = x * 2.0
    if c:
        y = y + 1.0
    return y


def test_prior_binding_joins_with_branch_arm() -> None:
    _oracle(_rebind_across_branch, [{"c": True, "x": 1.0}, {"c": False, "x": 1.0}])


def _incompatible_arms(c: bool) -> float:
    return 1 if c else True


def test_incompatible_arm_types_reject() -> None:
    _rejects(_incompatible_arms, "incompatible types")


def _one_sided_read(c: bool) -> float:
    if c:
        y = 1.0
    return y


def test_one_sided_binding_read_rejects() -> None:
    _rejects(_one_sided_read, "not bound on every path")


def _read_before_binding(x: float) -> float:
    r: float = q * x  # type: ignore[has-type, used-before-def]
    q = 1.0
    return r + q


def test_read_before_binding_rejects() -> None:
    _rejects(_read_before_binding, "not bound on every path")


def _truthy_condition(x: float) -> float:
    if x:
        return 1.0
    return 0.0


def _static_truthy_condition(x: float) -> float:
    if 1:
        return x
    return -x


def test_conditions_must_be_bool_at_both_binding_times() -> None:
    _rejects(_truthy_condition, "condition must be a bool")
    _rejects(_static_truthy_condition, "condition must be a bool")


def _early_return(x: float) -> float:
    if x > 0.0:
        return x
    return -x


def test_an_early_return_joins_at_the_exit() -> None:
    _oracle(_early_return, [{"x": 3.0}, {"x": -2.0}, {"x": 0.0}])


# ---------------------------------------------------------------------- powers and the budget


def _pow_chains(x: float) -> tuple[float, float, float, float]:
    return x**3, x**1, x**-2, (-2.0) ** 3 * x


def test_power_chains_and_reciprocal() -> None:
    _oracle(_pow_chains, [{"x": 1.5}, {"x": -2.0}, {"x": 0.25}])


def _pow_zero_orphans(a: float, b: float) -> float:
    return (a * b) ** 0


def test_zero_power_orphans_its_base() -> None:
    _oracle(_pow_zero_orphans, [{"a": 2.0, "b": 3.0}])
    text = _residual(_pow_zero_orphans, _INT_ONLY)
    assert "fmul" not in text
    assert "return 1.0" in text


def _exp2_float(note: float) -> float:
    return 2 ** ((note - 69.0) / 12.0)


def _exp2_int_exponent(n: int) -> float:
    return 2.0**n


def test_exponentials_of_two() -> None:
    _oracle(_exp2_float, [{"note": 69.0}, {"note": 57.0}, {"note": 81.5}])
    _oracle(_exp2_int_exponent, [{"n": 0}, {"n": 5}, {"n": -3}])


def _static_pow_folds(x: float) -> float:
    return 7**30 * x + 2**-3


def test_fully_static_powers_fold() -> None:
    _oracle(_static_pow_folds, [{"x": 1.0}, {"x": 1e-20}])


def _pow_residual_exponent(a: float, b: float) -> float:
    return a**b  # type: ignore[no-any-return]


def _pow_float_exponent(x: float) -> float:
    return x**0.5  # type: ignore[no-any-return]


def _pow_int_base_runtime_exponent(n: int) -> float:
    return 2**n  # type: ignore[no-any-return]


def _pow_int_pair(b: int, n: int) -> float:
    return b**n  # type: ignore[no-any-return]


def _pow_negative_base_high_exponent(x: float) -> float:
    return x**7.0  # type: ignore[no-any-return]


def _zero_base_pow(e: float) -> float:
    return 0.0**e  # type: ignore[no-any-return]


def test_residual_powers_lower_through_the_pow_stub() -> None:
    """Runtime exponents C-promote and inline the registry pow_ stub, ints and negative bases included."""
    _oracle(
        _pow_residual_exponent,
        [{"a": 3.0, "b": 2.5}, {"a": 2.0, "b": 2.0}, {"a": 0.5, "b": -1.5}, {"a": -1.0, "b": math.inf}],
    )
    _oracle(_pow_float_exponent, [{"x": 4.0}, {"x": 2.0}, {"x": 0.25}])
    _oracle(_pow_int_base_runtime_exponent, [{"n": 0}, {"n": 5}, {"n": -3}])
    _oracle(
        _pow_int_pair, [{"b": 3, "n": 2}, {"b": 2, "n": 5}, {"b": -2, "n": 3}, {"b": -3, "n": 7}, {"b": 2, "n": -2}]
    )
    _oracle(_pow_negative_base_high_exponent, [{"x": -2.0}, {"x": -1.5}, {"x": 3.0}])
    _oracle(_pow_float_exponent, [{"x": 0.0}])
    _oracle(_zero_base_pow, [{"e": 2.5}, {"e": 0.5}])
    assert "flog2" not in _residual_text(_zero_base_pow)


def _zero_to_negative_power(x: float) -> float:
    return x + 0.0**-1


def _overflowing_static_power(x: float) -> float:
    return x + 1e300**3


def _zero_to_negative_float_power(x: float) -> float:
    return x + 0.0**-1.0  # type: ignore[no-any-return]


def _overflowing_static_float_power(x: float) -> float:
    return x + 1e300**3.0  # type: ignore[no-any-return]


def test_a_static_power_the_host_refuses_saturates_like_the_datapath() -> None:
    """
    The lowering owns the answer at every binding time: a fold runs the very stub the hardware runs, so an
    overflow saturates to inf as ``exp2`` already does, and each exponent domain answers its pole the way its own
    datapath does -- the composite's guarded +inf under a float exponent, a division that names no number under an
    integer one, judged by the survivor sweep rather than predicted.
    """
    for fn in (_overflowing_static_power, _overflowing_static_float_power, _zero_to_negative_float_power):
        assert "inf" in _residual(fn, _FADD), fn
    with pytest.raises(SynthesisError, match="names no number") as info:
        holoso.synthesize(_zero_to_negative_power, _FADD_FDIV, name="k")
    assert not isinstance(info.value, UnsupportedConstruct)


def _dead_pole(x: float) -> float:
    y = 0.0**-1  # noqa: F841
    return x


def test_a_dead_static_pole_is_not_convicted() -> None:
    """Unlike the old host-raise oracle, an unused pole is no longer a diagnostic: only a survivor is judged."""
    assert "fdiv" not in _residual(_dead_pole, _INT_ONLY)


def _nonnegative_int_fold_stays_int(x: float) -> float:
    return float((3**5) // 2) * x


def _negative_base_fold_stays_int(x: float) -> float:
    return float(((-3) ** 5) // 2) * x


def test_the_exact_fold_is_the_sign_blind_one_python_gives() -> None:
    """The base's sign plays no part: only a negative EXPONENT leaves the integers, as in CPython."""
    _oracle(_nonnegative_int_fold_stays_int, [{"x": 2.0}])
    _oracle(_negative_base_fold_stays_int, [{"x": 2.0}])
    assert "-122" in _residual(_negative_base_fold_stays_int, _FMUL)


def _pow_huge_exponent(x: float) -> float:
    return x ** (10**9)


def test_a_huge_exponent_expands_logarithmically() -> None:
    """
    The TODO.md huge-exponent hang class: square-and-multiply spends one trip per exponent BIT, so an exponent no
    linear chain could ever expand costs a few dozen multiplies instead of exhausting the budget -- the whole
    kernel synthesizes promptly on a multiplier-only machine.
    """
    holoso.synthesize(_pow_huge_exponent, _FMUL, name="k")


# ---------------------------------------------------------------------- environment snapshots


def _global_scalar(x: float) -> float:
    return x * _GAIN + _SCALE


def _quoted_annotations(x: "float") -> "float":
    # The quotes are the fixture: a kernel may still spell its annotations as strings, which the front end resolves
    # with eval_str. Unquoting them here would leave the test unable to fail for the reason it exists.
    return x * 2.0


def test_global_scalars_fold() -> None:
    _oracle(_global_scalar, [{"x": 1.0}, {"x": -2.0}])


def test_quoted_annotations_are_accepted() -> None:
    _oracle(_quoted_annotations, [{"x": 1.5}])


def _make_closure_kernel() -> Callable[[float], float]:
    gain = 3.5

    def kernel(x: float) -> float:
        return x * gain

    return kernel


def test_closure_cells_snapshot() -> None:
    _oracle(_make_closure_kernel(), [{"x": 1.0}, {"x": -2.0}])


def _make_unbound_cell_kernel() -> Callable[[float], float]:
    def kernel(x: float) -> float:
        return x * late

    if _NEVER:
        late = 1.0
    return kernel


def test_unbound_closure_cell_rejects() -> None:
    _rejects(_make_unbound_cell_kernel(), "unbound in its enclosing scope")


def _missing_global(x: float) -> float:
    return x * NEVER_DEFINED  # type: ignore[name-defined, no-any-return]  # noqa: F821


def _nan_global(x: float) -> float:
    return x * _NAN_VALUE


def _numpy_global(x: float) -> float:
    return x * _NP_GAIN


def _object_global(x: float) -> float:
    return x * _BOX  # type: ignore[operator]


def test_environment_rejections() -> None:
    _rejects(_missing_global, "is not defined")
    _rejects(_nan_global, "is NaN")
    _rejects(_object_global, "is not a bool, int, or float scalar")


def test_numpy_scalars_snapshot_as_their_exact_values() -> None:
    _oracle(_numpy_global, [{"x": 2.0}, {"x": -0.5}])
    assert "env" not in _residual(_numpy_global, _FMUL)


# ---------------------------------------------------------------------- the module boundary


def _returns_nothing(x: float) -> None:
    y = x * 2.0
    assert y == y


def _returns_bare_none(x: float) -> None:
    y = x * 2.0
    assert y == y
    return None


def test_none_kernels_have_no_outputs() -> None:
    _oracle(_returns_nothing, [{"x": 1.0}])
    _oracle(_returns_bare_none, [{"x": 1.0}])
    assert lower(_returns_nothing, DEFAULT_UNROLL_MAX_TRIPS).hir.outputs == []


def _mixed_tuple(n: int, x: float) -> tuple[int, float, bool]:
    return n * 2, x / 2.0, x > n


def test_flat_tuple_return() -> None:
    _oracle(_mixed_tuple, [{"n": 3, "x": 1.5}, {"n": -1, "x": -0.5}])


def _unannotated_param(x) -> float:  # type: ignore[no-untyped-def]
    return float(x)


def _str_param(x: str) -> float:
    return 1.0


def _no_return_annotation(x: float):  # type: ignore[no-untyped-def]
    return x


def _none_returns_value(x: float) -> None:
    return x  # type: ignore[return-value]


def _value_returns_none(x: float) -> float:
    if x > 0.0:
        y = x
    return None  # type: ignore[return-value]


def _int_from_float(x: float) -> int:
    return x  # type: ignore[return-value]


def _bool_from_float(x: float) -> bool:
    return x  # type: ignore[return-value]


def _arity_mismatch(x: float) -> tuple[float, float]:
    return x, x, x  # type: ignore[return-value]


def _scalar_where_tuple(x: float) -> tuple[float, float]:
    return x  # type: ignore[return-value]


def _nested_tuple(x: float) -> tuple[tuple[float, float], float]:
    return (x, 2.0 * x), x + 1.0


def _falls_off_the_end(x: float) -> float:  # type: ignore[return]
    if _NEVER:
        return x


def test_interface_annotation_rejections() -> None:
    _rejects(_unannotated_param, "requires a type annotation")
    _rejects(_str_param, "annotation of parameter 'x' is not supported")
    _rejects(_no_return_annotation, "return type annotation is required")
    _rejects(_none_returns_value, "annotation does not match the returned scalar")
    _rejects(_value_returns_none, "returns no value but its annotation declares one")
    _rejects(_int_from_float, "type float where the annotation declares int")
    _rejects(_bool_from_float, "type float where the annotation declares bool")
    _rejects(_arity_mismatch, r"has 3 element\(s\) where the annotation declares 2")
    _rejects(_scalar_where_tuple, "is not a sequence")
    _rejects(_falls_off_the_end, "can complete without returning a value")


def test_nested_tuple_returns_flatten_row_major() -> None:
    _oracle(_nested_tuple, [{"x": 1.5}, {"x": -2.0}])


# ---------------------------------------------------------------------- type-model rejections


def _bool_arithmetic(b: bool) -> int:
    return b + 1


def _bool_negate(b: bool) -> int:
    return -b


def _bool_invert(b: bool) -> bool:
    return ~b  # type: ignore[return-value]


def _bool_ordering(a: bool, b: bool) -> bool:
    return a < b


def _bool_vs_number(b: bool) -> bool:
    return b == 0


def _float_floordiv(a: float, b: float) -> float:
    return a // b


def _float_mod(a: float, b: float) -> float:
    return a % b


def _float_shift(a: float, n: int) -> float:
    return a << n  # type: ignore[operator]


def _mixed_bitwise(n: int, b: bool) -> int:
    return n & b


def _float_truthiness(a: float, b: float) -> bool:
    return a and b  # type: ignore[return-value]


def _not_on_float(a: float) -> bool:
    return not a


def test_type_model_rejections() -> None:
    for fn, match in [
        (_bool_arithmetic, "booleans take no part in arithmetic"),
        (_bool_negate, "booleans take no part in arithmetic"),
        (_bool_invert, "integer-only"),
        (_bool_ordering, "ordering comparisons of booleans"),
        (_bool_vs_number, "cannot be compared with a number"),
        (_float_floordiv, "`//` is integer-only"),
        (_float_mod, "`%` is integer-only"),
        (_float_shift, "`<<` is integer-only"),
        (_mixed_bitwise, "requires two ints or two bools"),
        (_float_truthiness, "truthiness is not supported"),
        (_not_on_float, "requires a bool operand"),
    ]:
        _rejects(fn, match)


# ---------------------------------------------------------------------- the captured environment


def _attr_gap(x: float) -> float:
    return _BOX.value * x  # type: ignore[no-any-return]


def _store_gap(x: float) -> float:
    _BOX.value = x
    return x


def _index_gap(x: float) -> float:
    return _XS[0] * x


def _unpack_gap(x: float) -> float:
    a, b = x, x
    return a + b


def test_a_captured_object_store_is_observable_outside_and_rejects() -> None:
    _rejects(_store_gap, "cannot mutate '_BOX': it was captured from outside the kernel")


def test_captured_aggregates_and_instance_attributes_fold() -> None:
    for fn in (_attr_gap, _index_gap, _unpack_gap):
        _oracle(fn, [{"x": 2.0}, {"x": -1.5}])


_sin = math.sin
_cos = math.cos


def _select_callee_ternary(c: bool, x: float) -> float:
    f = _sin if c else _cos
    return f(x)


def _select_callee_branch(c: bool, x: float) -> float:
    if c:
        f = _sin
    else:
        f = _cos
    return f(x)


def _select_tuple_ternary(c: bool, x: float) -> tuple[float, float]:
    y = (x, 1.0) if c else (2.0, x)
    return y


def _select_shape_mismatch(c: bool, x: float) -> float:
    y = (x, 1.0) if c else (x, 1.0, 2.0)
    return y[0]


def test_unjoinable_branch_values_reject_located_at_the_read() -> None:
    """
    Reading a binding whose branch values cannot merge is a located rejection like any other
    definite-assignment failure, in every spelling -- never a bare internal assertion.
    """
    for fn in (_select_callee_ternary, _select_callee_branch, _select_shape_mismatch):
        _rejects(fn, "cannot merge")


def test_same_shape_aggregates_join_leafwise() -> None:
    _oracle(_select_tuple_ternary, [{"c": True, "x": 3.0}, {"c": False, "x": 3.0}])


class _Stateful:
    def __init__(self) -> None:
        self.y = 0.0

    def step(self, x: float) -> float:
        self.y = self.y + x
        return self.y


def test_a_state_writing_bound_method_accumulates_across_transactions() -> None:
    _oracle(_Stateful().step, [{"x": 1.5}, {"x": -0.25}, {"x": 4.0}])
    with pytest.raises(SynthesisError, match="not a plain function"):
        resolve_target(3)


# ---------------------------------------------------------------------- fault residualization and hygiene


def _static_fault_reaches_output(x: float) -> float:
    return x + 1.0 / 0.0


def test_static_fault_residualizes_for_survivor_refusal() -> None:
    """CPython raises ZeroDivisionError; the compiler builds, and the fault surfaces only if it survives."""
    with pytest.raises(SynthesisError, match="names no number") as info:
        holoso.synthesize(_static_fault_reaches_output, _FADD_FDIV, name="k")
    assert not isinstance(info.value, UnsupportedConstruct)


def _huge_int_promotion(x: float) -> float:
    return x * _HUGE_INT


def _huge_int_ratio() -> float:
    return _HUGE_INT / (2 * _HUGE_INT)


def test_unrepresentable_promotion_residualizes() -> None:
    """
    float(10**400) overflows: CPython raises at the multiply, the compiler refuses survivor-based. The rule
    extends to fully static division: `/` promotes BOTH operands to their float images first, per the
    C-promotion model, even though CPython's int/int true division would answer 0.5 without any float
    conversion -- folding it Python's way would let the same expression answer differently by binding time.
    """
    assert "int_to_float" in _residual_text(_huge_int_promotion)
    for fn in (_huge_int_promotion, _huge_int_ratio):
        with pytest.raises(SynthesisError, match="names no number") as info:
            holoso.synthesize(fn, _FADD_FMUL, name="k")
        assert not isinstance(info.value, UnsupportedConstruct), fn


# 2**53 + 1 is the first integer a binary64 cannot hold, so any lane that routes it through a float is visible.
_UNHOLDABLE = 9007199254740993


def _abs_of_an_unholdable_int() -> bool:
    return abs(-_UNHOLDABLE) == 9007199254740992


def _round_of_an_unholdable_int() -> bool:
    return round(_UNHOLDABLE) == 9007199254740992


def _floor_of_an_unholdable_int() -> bool:
    return math.floor(_UNHOLDABLE) == 9007199254740992


def _min_of_unholdable_ints() -> int:
    return min(_UNHOLDABLE, 9007199254740992)


def _abs_stays_integer(n: int) -> int:
    return abs(-3) + n


def test_a_static_integer_folds_in_the_integer_domain() -> None:
    """
    Regression: every entry named a float operator, so a static integer was rounded before the fold --
    ``abs(-(2**53+1))`` compared equal to 2**53 -- and gone before the gate below HIR could refuse it.
    """
    for kernel in (_abs_of_an_unholdable_int, _round_of_an_unholdable_int, _floor_of_an_unholdable_int):
        result = holoso.synthesize(kernel, _INT_ONLY, name="k")
        assert result.numerical_model.elaborate().run() == [kernel()], kernel.__name__
    # The min answers an INT past the machine word, so it stays on the evaluator (no public spelling holds it).
    assert HirEvaluator(lower(_min_of_unholdable_ints, DEFAULT_UNROLL_MAX_TRIPS).hir).run() == [
        _min_of_unholdable_ints()
    ]
    # The result keeps the integer TYPE too, so it composes with the integer operators rather than poisoning them.
    _oracle(_abs_stays_integer, [{"n": 4}, {"n": -1}])


def _abs_at_the_int64_boundary() -> int:
    return abs(-(2**63))


def _np_abs_at_the_int64_boundary() -> int:
    return int(np.abs(-(2**63)))


def test_the_integer_abs_is_exact_at_arbitrary_precision() -> None:
    """Both spellings share the one IntAbs entry, where the host has numpy wrapping under int64 and ``abs`` not."""
    for kernel in (_abs_at_the_int64_boundary, _np_abs_at_the_int64_boundary):
        assert HirEvaluator(lower(kernel, DEFAULT_UNROLL_MAX_TRIPS).hir).run() == [2**63], kernel.__name__


def _log2_of_an_integer_zero(x: float) -> float:
    return x + math.log2(0)


def _exp2_of_an_integer_overflow(x: float) -> float:
    return x + math.exp2(2000)


def _floor_of_a_runtime_integer(a: int) -> int:
    return math.floor(a)


def _rounding_chain_of_a_runtime_integer(a: int) -> int:
    return (round(a) % 3) + math.trunc(a) + math.ceil(a)


def test_rounding_an_integer_is_the_identity_at_either_binding_time() -> None:
    # Their integer entry is the identity stub, so they lower to nothing and the operand stays an integer.
    assert _residual(_floor_of_a_runtime_integer, _INT_ONLY).splitlines()[-1].strip() == "return a"
    _oracle(_floor_of_a_runtime_integer, [{"a": -7}, {"a": 0}, {"a": 5}])
    _oracle(_rounding_chain_of_a_runtime_integer, [{"a": 7}, {"a": -4}])


def _sign_of_a_static_integer(x: float) -> float:
    return x + float(np.sign(-7) % 3)


def test_an_integer_entry_can_be_a_composite_as_well_as_an_operator() -> None:
    # np.sign's integer entry is a composite where abs above is one operator; both must select the integer domain.
    _oracle(_sign_of_a_static_integer, [{"x": 1.5}, {"x": -0.25}])


def test_a_symbol_with_no_integer_entry_promotes_rather_than_asking_the_host() -> None:
    """
    Regression: a static integer used to fold by CALLING the named callee, and a raise there is not an answer --
    ``math.log2(0)`` raises where the operator's reference computes -inf. Promotion cannot differ for 0 and 0.0.
    """
    for kernel, want in ((_log2_of_an_integer_zero, -math.inf), (_exp2_of_an_integer_overflow, math.inf)):
        result = holoso.synthesize(kernel, _FADD, name="k")
        assert [float(v) for v in result.numerical_model.elaborate().run(0.0)] == [want], kernel.__name__
