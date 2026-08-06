"""
The call machinery, driven black-box through the differential oracle wherever the behavior is observable:
registry identity dispatch (every spelling, and the typed lowering each operand domain selects), user-function
inlining with CPython signature binding, boundary conformance, the pow one-behavior rule, raise semantics, and the
re-attribution of every rejection arising inside an inlined callee to the user's call site.
"""

import inspect
import math
import types
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pytest

from holoso._eel import lower
from holoso._eel._desugar import desugar
from holoso._eel._pe import partial_evaluate
from holoso._eel._print import print_eel
from holoso._errors import SynthesisError, UnsupportedConstruct
from holoso._hir import IntSelect, Operation, optimize

from ._eeloracle import assert_hir_matches_reference

type _Row = Mapping[str, float | bool | int]

_NEVER = False
_ENABLED = True
_SCALE = 4


def _oracle(fn: Callable[..., object], vectors: Sequence[_Row]) -> None:
    compared = assert_hir_matches_reference(lower(fn).hir, fn, vectors, label=fn.__name__)
    assert compared == len(vectors)


def _rejects(fn: object, match: str) -> None:
    with pytest.raises(UnsupportedConstruct, match=match):
        lower(fn)


def _residual_text(fn: Callable[..., object]) -> str:
    assert isinstance(fn, types.FunctionType)
    return print_eel(partial_evaluate(desugar(fn), fn))


def _lowered_ops(fn: Callable[..., object]) -> list[str]:
    hir = optimize(lower(fn).hir)
    return [type(node.operator).__name__ for node in hir.nodes.values() if isinstance(node, Operation)]


def _line_of(fn: Callable[..., object], needle: str) -> int:
    source, start = inspect.getsourcelines(fn)
    for offset, line in enumerate(source):
        if needle in line:
            return start + offset
    raise AssertionError(needle)


# ---------------------------------------------------------------------- intrinsics


def _intrinsic_mix(x: float, y: float) -> float:
    return math.sqrt(abs(x)) + min(x, y) * max(x, y) + np.hypot(x, y) + math.atan2(y, x)  # type: ignore[no-any-return]


def test_intrinsics_under_every_spelling() -> None:
    _oracle(_intrinsic_mix, [{"x": 1.5, "y": -2.0}, {"x": -0.25, "y": 0.5}, {"x": 3.0, "y": 4.0}])


_sqrt_alias = math.sqrt


def _aliased_intrinsic(x: float) -> float:
    return _sqrt_alias(x)


def test_callee_dispatch_is_identity_not_spelling() -> None:
    _oracle(_aliased_intrinsic, [{"x": 4.0}, {"x": 2.25}])
    assert "fsqrt" in _residual_text(_aliased_intrinsic)


def sqrt(v: float) -> float:
    return v + 1.0


def _shadowing_user_function(x: float) -> float:
    return sqrt(x)


def test_a_user_function_spelled_like_a_library_one_inlines_as_user_code() -> None:
    _oracle(_shadowing_user_function, [{"x": 4.0}, {"x": -1.0}])
    assert "fsqrt" not in _residual_text(_shadowing_user_function)


def _sqrt_of_int(n: int) -> float:
    return math.sqrt(n)


def test_intrinsic_operands_promote_per_signature() -> None:
    _oracle(_sqrt_of_int, [{"n": 4}, {"n": 2}, {"n": 0}])


# ---------------------------------------------------------------------- typed library dispatch


def _abs_of_int(n: int) -> int:
    return abs(n)


def _abs_of_int_composes_with_the_integer_operators(n: int) -> int:
    return abs(n) % 3


def _min_max_of_ints(a: int, b: int) -> int:
    return min(a, b) * 10 + max(a, b)


def _sign_of_int(n: int) -> int:
    return int(np.sign(n))


def test_a_symbol_whose_python_answer_is_an_integer_lowers_over_integers() -> None:
    """Regression: every entry named a float operator, so a runtime integer promoted and came back a float."""
    _oracle(_abs_of_int, [{"n": -3}, {"n": 0}, {"n": 7}])
    _oracle(_abs_of_int_composes_with_the_integer_operators, [{"n": -8}, {"n": 5}])
    _oracle(_min_max_of_ints, [{"a": 3, "b": 5}, {"a": 5, "b": 3}, {"a": -4, "b": -4}])
    _oracle(_sign_of_int, [{"n": -9}, {"n": 0}, {"n": 2}])
    # The compare-and-select entries cost one mux rather than control flow, which is if-conversion's doing.
    hir = optimize(lower(_min_max_of_ints).hir)
    assert sum(isinstance(node, Operation) and isinstance(node.operator, IntSelect) for node in hir.nodes.values()) == 2
    assert len(hir.blocks) == 1


def _fabs_of_int(n: int) -> float:
    return math.fabs(n)


def _fabs_of_int_declared_int(n: int) -> int:
    return math.fabs(n)  # type: ignore[return-value]


def _rint_of_int_declared_int(n: int) -> int:
    return np.rint(n)  # type: ignore[no-any-return]


def _mixed_min(a: int, x: float) -> float:
    return min(a, x)


def test_a_float_answering_spelling_keeps_its_own_domain() -> None:
    """Each is its own symbol; the int return annotation is what makes the float visible, a float one converting."""
    _oracle(_fabs_of_int, [{"n": -3}, {"n": 4}])
    for kernel in (_fabs_of_int_declared_int, _rint_of_int_declared_int):
        _rejects(kernel, "the returned value has type float where the annotation declares int")
    _oracle(_mixed_min, [{"a": 3, "x": 5.5}, {"a": 7, "x": -1.25}])  # one float operand takes the whole call to floats


def _math_roundings_of_a_float(x: float) -> tuple[int, int, int, int]:
    return math.floor(x), math.ceil(x), math.trunc(x), round(x)


def _numpy_roundings_of_a_float(x: float) -> tuple[float, float, float, float]:
    return float(np.floor(x)), float(np.ceil(x)), float(np.trunc(x)), float(np.round(x))


def _floor_consumed_as_a_float(x: float) -> float:
    return math.floor(x)


def test_a_rounding_answers_what_its_own_spelling_answers() -> None:
    """math.floor/ceil/trunc and round of a float answer an int; the numpy spellings of the same answer a float."""
    vectors = [{"x": 2.5}, {"x": -2.5}, {"x": 3.7}, {"x": -0.5}, {"x": 4.0}]
    _oracle(_math_roundings_of_a_float, vectors)
    _oracle(_numpy_roundings_of_a_float, vectors)
    # The cast costs nothing where a float consumes the result: it reduces back to the bare rounder.
    hir = optimize(lower(_floor_consumed_as_a_float).hir)
    lowered = [type(node.operator).__name__ for node in hir.nodes.values() if isinstance(node, Operation)]
    assert lowered == ["FloatFloor"]


def _dot_of_static_ints(x: float) -> float:
    return float(np.dot(2, 3)) + x


def test_an_array_composite_judges_scalars_by_its_own_rank_guard() -> None:
    # Regression: a static-integer fold ran the host callee, so np.dot(2, 3) answered 6 where np.dot(2.0, 3.0) was refused.
    _rejects(_dot_of_static_ints, "does not accept scalar operands")


# ---------------------------------------------------------------------- composite stubs


def _composites(x: float) -> float:
    return math.exp(x) * math.log10(x + 10.0) + math.sinh(x / 4.0)


def test_composite_stubs_inline_and_match_the_host() -> None:
    _oracle(_composites, [{"x": 0.5}, {"x": 1.0}, {"x": -0.5}, {"x": 2.0}])


def _tan(x: float) -> float:
    return math.tan(x)


def test_a_stub_opening_with_tuple_unpacking_inlines() -> None:
    # tan opens with ``s, c = sin(x), cos(x)``, so it exercises tuple unpacking inside an inlined stub.
    _oracle(_tan, [{"x": 0.5}, {"x": -1.0}, {"x": 2.0}, {"x": 0.0}])


def _cbrt(x: float) -> float:
    return math.cbrt(x)


def test_sibling_library_stub_calls_inline_through_the_user_path() -> None:
    # cbrt calls sign_float (reached by name here, not through its own registry key) plus the exp2/log2 intrinsics.
    _oracle(_cbrt, [{"x": 8.0}, {"x": -27.0}, {"x": 0.5}, {"x": 0.0}])


def _pi_scaled(x: float) -> float:
    return math.pi * x + math.tau


def test_module_constants_snapshot_through_attribute_reads() -> None:
    _oracle(_pi_scaled, [{"x": 1.0}, {"x": -2.5}])
    assert "env" not in _residual_text(_pi_scaled)


# ---------------------------------------------------------------------- pow: one behavior under every spelling


def _pow_op(a: float, b: float) -> float:
    return a**b  # type: ignore[no-any-return]


def _pow_builtin(a: float, b: float) -> float:
    return pow(a, b)  # type: ignore[no-any-return]


def _pow_math(a: float, b: float) -> float:
    return math.pow(a, b)


def _pow_np(a: float, b: float) -> float:
    return np.power(a, b)  # type: ignore[no-any-return]


def _body(fn: Callable[..., object]) -> str:
    return _residual_text(fn).split("\n", 1)[1]


def test_every_pow_spelling_is_one_behavior() -> None:
    reference = _body(_pow_op)
    assert reference == _body(_pow_builtin) == _body(_pow_math) == _body(_pow_np)
    _oracle(_pow_op, [{"a": 3.0, "b": 2.5}, {"a": 2.0, "b": 5.0}, {"a": 0.5, "b": -2.0}])


def _int_pair_pow(a: int, b: int) -> float:
    return pow(a, b)  # type: ignore[no-any-return]


def _int_pair_np_power(a: int, b: int) -> float:
    return np.power(a, b)  # type: ignore[no-any-return]


def test_spelled_int_pairs_ride_the_same_float_lane_as_the_operator() -> None:
    _oracle(_int_pair_pow, [{"a": 3, "b": 2}, {"a": 2, "b": 5}, {"a": 2, "b": -1}])
    _oracle(_int_pair_np_power, [{"a": 3, "b": 2}, {"a": 2, "b": 5}])


def _static_pow_conforms(x: float) -> float:
    return math.pow(2, 3) + x


def test_static_int_operands_of_a_float_stub_fold_to_the_float_image() -> None:
    _oracle(_static_pow_conforms, [{"x": 0.0}, {"x": 1.5}])
    assert "8.0" in _residual_text(_static_pow_conforms)


def _neg_base_float_exponent(x: float) -> float:
    return (-2.0) ** 6.0 + x  # type: ignore[no-any-return]


def _neg_base_int_exponent(x: float) -> float:
    return (-2.0) ** 6 + x


def _dead_static_pow_fault(x: float) -> float:
    y = (-2.0) ** 6.0  # noqa: F841
    return x


def _neg_base_odd_exponent(x: float) -> float:
    return (-2.0) ** 7.0 + x  # type: ignore[no-any-return]


def _neg_base_fractional_exponent(x: float) -> float:
    return (-8.0) ** 0.5 + x  # type: ignore[no-any-return]


def test_static_negative_base_powers_fold_through_the_stub_parity_lane() -> None:
    """The stub honors integer exponents of negative bases (sign by parity); the fractional case is complex."""
    _oracle(_neg_base_float_exponent, [{"x": 0.0}, {"x": 1.0}])
    assert "64.0" in _residual_text(_neg_base_float_exponent)
    _oracle(_neg_base_int_exponent, [{"x": 0.0}, {"x": 1.0}])
    assert "64.0" in _residual_text(_neg_base_int_exponent)
    _oracle(_neg_base_odd_exponent, [{"x": 0.0}, {"x": 1.0}])
    assert "-128.0" in _residual_text(_neg_base_odd_exponent)
    _oracle(_dead_static_pow_fault, [{"x": 3.5}])
    with pytest.raises(SynthesisError, match="names no number"):
        optimize(lower(_neg_base_fractional_exponent).hir)


def _pow_static_int_pair(n: int) -> int:
    return (2**3) // 3 + n


def _pow_static_negative_exponent(x: float) -> float:
    return 2**-1 + x


def _pow_runtime_base_static_exponent(x: float) -> float:
    return x**3


def _pow_runtime_base_negative_exponent(x: float) -> float:
    return x**-2


def _pow_runtime_int_base(n: int) -> int:
    return n**2


def _pow_runtime_exponent(x: float, n: int) -> float:
    return x**n


def test_the_pow_table_selects_a_lowering_per_operand_position() -> None:
    """One entry, four lowerings, chosen by the type AND the binding time/sign of each position independently."""
    _oracle(_pow_static_int_pair, [{"n": 0}, {"n": 5}])
    assert "2" in _residual_text(_pow_static_int_pair)  # the fold stayed integral, so `//` accepted it
    _oracle(_pow_static_negative_exponent, [{"x": 0.0}, {"x": 1.5}])
    assert "0.5" in _residual_text(_pow_static_negative_exponent)
    # On the optimized HIR, so the chain's leading multiply by one is elided as it is in hardware.
    assert _lowered_ops(_pow_runtime_base_static_exponent) == ["FloatMul", "FloatMul"]
    assert _lowered_ops(_pow_runtime_base_negative_exponent) == ["FloatMul", "FloatDiv"]
    assert _lowered_ops(_pow_runtime_int_base) == ["IntMul"]  # a runtime int base stays integral, as in CPython
    general = _lowered_ops(_pow_runtime_exponent)
    assert "FloatLog2" in general and "FloatExp2" in general
    _oracle(_pow_runtime_base_static_exponent, [{"x": 2.0}, {"x": -1.5}])
    _oracle(_pow_runtime_base_negative_exponent, [{"x": 2.0}, {"x": -1.5}])
    _oracle(_pow_runtime_int_base, [{"n": 3}, {"n": -4}])
    _oracle(_pow_runtime_exponent, [{"x": 2.0, "n": 3}, {"x": 0.5, "n": -2}])


def _bool_pow_operator(flag: bool) -> float:
    return flag**2


def _bool_pow_spelled(flag: bool) -> float:
    return pow(flag, 2)


def test_a_boolean_power_rejects_the_same_way_under_the_operator_and_its_spelling() -> None:
    """The operator resolves the entry the spelling resolves, so a domain refusal cannot differ between them."""
    messages = []
    for fn in (_bool_pow_operator, _bool_pow_spelled):
        with pytest.raises(UnsupportedConstruct) as excinfo:
            lower(fn)
        messages.append(excinfo.value.message.split("() ", 1)[1])
    assert messages[0] == messages[1]
    assert "boolean is not a number" in messages[0]


# ---------------------------------------------------------------------- user-function inlining


def _helper(a: float, b: float = 2.0, *, gain: float = 3.0) -> float:
    return (a + b) * gain


def _uses_helper(x: float) -> float:
    return _helper(x) + _helper(x, 1.5) + _helper(x, gain=0.5) + _helper(b=x, a=1.0)


def test_user_inlining_binds_positional_keyword_and_default_arguments() -> None:
    _oracle(_uses_helper, [{"x": 1.0}, {"x": -0.5}, {"x": 2.25}])


def _inner(u: float) -> float:
    return math.sqrt(u * u + 1.0)


def _outer(u: float, v: float) -> float:
    return _inner(u) * _inner(v)


def _uses_nested(x: float) -> float:
    return _outer(x, 2.0 * x)


def test_nested_user_inlining() -> None:
    _oracle(_uses_nested, [{"x": 0.5}, {"x": -3.0}])


def _make_closure_kernel() -> Callable[[float], float]:
    def helper(z: float) -> float:
        return z * 2.0 + 1.0

    def kernel(x: float) -> float:
        return helper(x) - helper(-x)

    return kernel


def test_a_callee_reached_through_a_closure_cell_inlines() -> None:
    _oracle(_make_closure_kernel(), [{"x": 1.25}, {"x": -0.5}])


def _pair(x: float) -> tuple[float, float]:
    return x + 1.0, x - 1.0


def _returns_helper_tuple(x: float) -> tuple[float, float]:
    return _pair(x * 2.0)


def test_a_tuple_returning_helper_feeds_the_kernel_return() -> None:
    _oracle(_returns_helper_tuple, [{"x": 1.0}, {"x": -2.5}])


def _exotic_default(a: float, junk: object = math) -> float:
    return a * 2.0


def _uses_exotic_default(x: float) -> float:
    return _exotic_default(x)


def test_an_unused_non_scalar_default_is_admitted_lazily() -> None:
    _oracle(_uses_exotic_default, [{"x": 3.0}])


def _nan_sentinel(a: float, marker: object = math.nan) -> float:
    return a * 2.0


def _uses_nan_sentinel(x: float) -> float:
    return _nan_sentinel(x)


def _nan_reader(a: float, marker: object = math.nan) -> float:
    return a * marker  # type: ignore[operator]  # honest at runtime only when a caller overrides the sentinel


def _reads_nan_sentinel(x: float) -> float:
    return _nan_reader(x)


def test_a_nan_default_is_judged_at_use_not_at_binding() -> None:
    _oracle(_uses_nan_sentinel, [{"x": 1.5}, {"x": -0.25}])
    _rejects(_reads_nan_sentinel, "the captured value of 'marker' is NaN")


class _MathBase:
    @staticmethod
    def scale(v: float) -> float:
        return v * 3.0


class _MathKit(_MathBase):
    @staticmethod
    def shift(v: float) -> float:
        return v + 1.0

    def plain(v: float) -> float:  # type: ignore[misc]  # deliberately not a method: accessed via the class
        return v * 0.5


def _uses_class_functions(x: float) -> float:
    return _MathKit.scale(x) + _MathKit.shift(x) + _MathKit.plain(x)


def test_staticmethods_and_plain_functions_resolve_through_class_metadata() -> None:
    _oracle(_uses_class_functions, [{"x": 2.0}, {"x": -1.0}])


def _broken_helper(*xs: float) -> float:
    return 0.0


def _dead_arm_call(x: float) -> float:
    if _NEVER:
        return _broken_helper(x)
    return x


def test_a_callee_in_a_statically_dead_arm_is_never_desugared() -> None:
    _oracle(_dead_arm_call, [{"x": 1.5}])


# ---------------------------------------------------------------------- binding and conformance rejections


def _kw_to_intrinsic(x: float) -> float:
    return math.atan2(y=x, x=1.0)  # type: ignore[call-arg]  # rejected on the host too: math.atan2 is positional-only


def _kw_to_stub(x: float) -> float:
    return math.tan(x=x)  # type: ignore[call-arg]  # rejected on the host too: math.tan is positional-only


def _wrong_arity(x: float) -> float:
    return math.sqrt(x, x)  # type: ignore[call-arg]


def _po_helper(a: float, /) -> float:
    return a * 2.0


def _kw_to_positional_only(x: float) -> float:
    return _po_helper(a=x)  # type: ignore[call-arg]


def _missing_argument(x: float) -> float:
    return _helper()  # type: ignore[call-arg]


def _bool_to_stub(x: float) -> float:
    return math.sqrt(True)


def _bool_to_helper(x: float) -> float:
    return _helper(x > 0.0)


def test_binding_and_conformance_rejections() -> None:
    for fn, match in [
        (_kw_to_intrinsic, r"math.atan2\(\) takes no keyword arguments"),
        (_kw_to_stub, r"math.tan\(\) takes no keyword arguments"),
        (_wrong_arity, r"math.sqrt\(\) takes 1 argument\(s\), got 2"),
        (_kw_to_positional_only, "the arguments do not bind: .*positional-only"),
        (_missing_argument, "the arguments do not bind: missing a required argument: 'a'"),
        (_bool_to_stub, r"math\.sqrt\(\) takes float operands, got bool"),
        (_bool_to_helper, "the argument 'a' has type bool where the annotation declares float"),
    ]:
        _rejects(fn, match)


def _recurse(x: float) -> float:
    return _recurse(x - 1.0)


def _mutual_a(x: float) -> float:
    return _mutual_b(x)


def _mutual_b(x: float) -> float:
    return _mutual_a(x)


def test_recursive_inlining_rejects() -> None:
    _rejects(_recurse, "recursive inlining is not supported")
    _rejects(_mutual_a, "recursive inlining is not supported")


def _procedure(x: float) -> None:
    y = x  # noqa: F841


def _uses_procedure_value(x: float) -> float:
    return _procedure(x)  # type: ignore[func-returns-value,return-value]


def test_a_value_less_callee_cannot_be_used_as_an_expression() -> None:
    _rejects(_uses_procedure_value, "returns no value")


def _lying_helper(v: float) -> bool:
    return v * 2.0  # type: ignore[return-value]  # deliberately mis-annotated


def _calls_lying_helper(x: float) -> bool:
    return _lying_helper(x)


def test_a_return_annotation_mismatch_points_into_the_callee() -> None:
    pattern = r"in _lying_helper\(\): the returned value has type float where the annotation declares bool \(at "
    with pytest.raises(UnsupportedConstruct, match=pattern) as info:
        lower(_calls_lying_helper)
    location = info.value.location
    assert location is not None
    assert location.lineno == _line_of(_calls_lying_helper, "_lying_helper(x)")
    assert f":{_line_of(_lying_helper, 'return v * 2.0')})" in str(info.value)


class _Accumulator:
    def __init__(self) -> None:
        self.total = 0.0

    def add(self, x: float) -> float:
        self.total = self.total + x
        return self.total


_BOUND = _Accumulator().add


def _calls_bound_method(x: float) -> float:
    return _BOUND(x)


def test_a_state_writing_bound_callee_rejects_at_the_helper_write() -> None:
    _rejects(_calls_bound_method, r"in _BOUND\(\): an attribute store is only supported on the entry method's receiver")


def _unimplemented_library(x: float) -> float:
    return math.erf(x)


def _unimplemented_python_library(x: float) -> float:
    y = np.identity(2)  # noqa: F841
    return x


def test_unregistered_callables_are_uniform_no_library_is_special() -> None:
    """
    Substitution is the _lib registry's decision alone; the interpreter knows no library names.
    A plain-Python callable is ingested like user code wherever it lives, rejecting on its first
    unsupported construct chain-attributed to the user's call site; a source-less callable rejects generically.
    """
    _rejects(_unimplemented_library, "calls to 'math.erf' are not supported yet")
    with pytest.raises(UnsupportedConstruct, match=r"in np.identity\(\): ") as info:
        lower(_unimplemented_python_library)
    location = info.value.location
    assert location is not None
    assert location.lineno == _line_of(_unimplemented_python_library, "np.identity(2)")


# ---------------------------------------------------------------------- raise semantics


def _static_raise(x: float) -> float:
    if _ENABLED:
        raise RuntimeError(f"gain {_SCALE}")
    return x


def _empty_raise(x: float) -> float:
    raise ValueError()


def _guarded_raise(x: float) -> float:
    if x > 0.0:
        raise ValueError("positive")
    return x


def _residual_message(x: float) -> float:
    raise ValueError(f"bad {x}")


_MODE = "fast"


def _string_interpolation(x: float) -> float:
    raise ValueError(f"mode {_MODE} at gain {_SCALE}")


def _literal_interpolation(x: float) -> float:
    raise ValueError(f"{'bad'} input")


def test_raise_semantics() -> None:
    _rejects(_static_raise, "RuntimeError: gain 4")
    _rejects(_empty_raise, "ValueError\n")
    _rejects(_guarded_raise, "data-dependent path")
    _rejects(_residual_message, "interpolates a value that is not a compile-time constant")
    _rejects(_string_interpolation, "ValueError: mode fast at gain 4")
    _rejects(_literal_interpolation, "ValueError: bad input")


def _always_raises(v: float) -> float:
    raise ValueError("stub domain")


def _guarded_call_to_raiser(c: bool, x: float) -> float:
    if c:
        y = _always_raises(x)
    else:
        y = x
    return y


def test_a_callee_raise_is_judged_where_written_not_at_the_guarded_call_site() -> None:
    """DESIGN.md: unconditional within the writing function fires even under the caller's residual arm."""
    with pytest.raises(UnsupportedConstruct, match=r"in _always_raises\(\): ValueError: stub domain") as info:
        lower(_guarded_call_to_raiser)
    location = info.value.location
    assert location is not None
    assert location.lineno == _line_of(_guarded_call_to_raiser, "_always_raises(x)")


# ---------------------------------------------------------------------- re-attribution


def _bad_helper(v: float) -> float:
    f = lambda: 0.0  # noqa: E731
    return v


def _calls_bad_helper(x: float) -> float:
    return _bad_helper(x)


def test_a_callee_desugar_failure_reports_the_user_call_site_with_the_chain() -> None:
    with pytest.raises(UnsupportedConstruct, match=r"in _bad_helper\(\): lambdas are not supported \(at ") as info:
        lower(_calls_bad_helper)
    location = info.value.location
    assert location is not None
    assert location.lineno == _line_of(_calls_bad_helper, "_bad_helper(x)")


def _deep_reject(v: float) -> float:
    while v > 0.0:
        break
    return v


def _middle(v: float) -> float:
    return _deep_reject(v)


def _calls_deep(x: float) -> float:
    return _middle(x)


def test_nested_rejections_report_the_outermost_call_site_with_the_full_chain() -> None:
    with pytest.raises(
        UnsupportedConstruct, match=r"in _middle\(\): in _deep_reject\(\): the body of this data-dependent loop"
    ) as info:
        lower(_calls_deep)
    location = info.value.location
    assert location is not None
    assert location.lineno == _line_of(_calls_deep, "_middle(x)")


# ---------------------------------------------------------------------- registry-bound array methods


def _method_transpose(x: float) -> float:
    m = np.asarray(((x, 1.0), (2.0, 3.0), (4.0, 5.0)))
    t = m.transpose()
    return float(t[0][1] + t[1][2] + t.shape[0] - t.shape[1])


def _method_ravel(x: float) -> float:
    m = np.asarray(((x, 1.0), (2.0, 3.0)))
    return float(m.ravel()[1] + np.ravel(m)[2] + m.flatten()[3])


def _method_dot(x: float) -> float:
    v = np.asarray((x, 2.0))
    return float(v.dot(np.asarray((3.0, 4.0))))


def _method_trace(x: float) -> float:
    m = np.asarray(((x, 1.0), (2.0, 3.0)))
    return float(m.trace())


def _explicit_descriptor_spelling(x: float) -> float:
    v = np.asarray((x, 2.0, 3.0))
    return float(np.ndarray.flatten(v)[2])


def test_registry_bound_array_methods() -> None:
    _oracle(_method_transpose, [{"x": 7.0}, {"x": -1.5}])
    _oracle(_method_ravel, [{"x": 7.0}, {"x": 0.5}])
    _oracle(_method_dot, [{"x": 2.0}, {"x": -3.0}])
    _oracle(_method_trace, [{"x": 1.0}, {"x": 4.5}])
    _oracle(_explicit_descriptor_spelling, [{"x": 9.0}])


def _ravel_read_then_store(x: float) -> float:
    v = np.asarray((x, 2.0))
    f = v.ravel
    v[0] = x + 1.0
    return float(f()[0])


def _ravel_result_derives_storage(x: float) -> float:
    v = np.asarray((x, 2.0))
    r = v.ravel()
    v[0] = x + 1.0
    return float(r[0])


def test_a_method_read_and_a_derived_result_both_share_the_receiver() -> None:
    _rejects(_ravel_read_then_store, "cannot store into v.0.: it is shared")
    _rejects(_ravel_result_derives_storage, "cannot store into v.0.: it is shared")


def _method_arity_excess(x: float) -> float:
    m = np.asarray(((x, 1.0), (2.0, 3.0)))
    return float(m.flatten(1.0)[0])  # type: ignore[arg-type]


def _int_trace_stays_exact() -> bool:
    m = np.asarray(((9007199254740993, 0), (0, 0)))
    return bool(m.trace() == 9007199254740992)


def test_method_arity_messages_exclude_the_receiver() -> None:
    _rejects(_method_arity_excess, r"\.flatten\(\) takes 0 argument\(s\), got 1")


def test_an_integer_trace_keeps_exact_integers() -> None:
    _oracle(_int_trace_stays_exact, [{}])
