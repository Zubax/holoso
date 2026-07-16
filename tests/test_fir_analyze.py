"""
Analyzer tests, anchored by a differential oracle: wherever analysis claims Known(value), executing the same
kernel in real Python must produce exactly that value. Structural assertions are kept to what the design pins
(executability, promotion, rejection); everything else is behavioral.
"""

from collections.abc import Callable

import pytest

from holoso._frontend._fir._analyze import AnalysisRejection, Analyzer, ResidualUnit
from holoso._frontend._fir._fact import Known, Residual
from holoso._frontend._fir._value import SemType
from holoso._frontend._fir._ir import ReturnPlace, StateLeaf
from holoso._frontend._fir._value import StaticValue, as_python

_GAINS = (1.0, 2.0, 4.0)


def _analyzed_return(kernel: Callable[..., object]) -> object:
    result = Analyzer(kernel).fixpoint()
    return result.block_in[result.unit.exit].get(ReturnPlace())


def _assert_known_matches_python(kernel: Callable[..., object], *args: object) -> None:
    fact = _analyzed_return(kernel)
    assert isinstance(fact, Known), fact
    assert as_python(fact.value) == kernel(*args)


def test_static_arithmetic_folds_to_python_exact_values() -> None:
    def kernel() -> float:
        base = 3.0
        return (base * 2.0 + 1.0) / 7.0 - 0.5

    _assert_known_matches_python(kernel)


def test_static_loop_accumulation_matches_python() -> None:
    def kernel() -> float:
        acc = 0.0
        for gain in _GAINS:
            if gain > 1.5:
                acc = acc + gain * gain
        return acc

    _assert_known_matches_python(kernel)


def test_static_comprehension_matches_python() -> None:
    def kernel() -> float:
        squares = [v * v for v in (1.0, 2.0, 3.0) if v > 1.0]
        return squares[0] + squares[1]

    _assert_known_matches_python(kernel)


def test_static_call_chain_matches_python() -> None:
    def offset(v: float, delta: float = 0.25) -> float:
        return v + delta

    def scale(v: float, factor: float) -> float:
        return offset(v * factor)

    def kernel() -> float:
        return scale(3.0, factor=4.0) + scale(1.0, 2.0)

    _assert_known_matches_python(kernel)


def test_runtime_parameters_stay_residual_but_static_parts_fold() -> None:
    def kernel(x: float) -> float:
        table = (10.0, 20.0)
        return x * table[1]

    fact = _analyzed_return(kernel)
    assert fact == Residual(SemType.FLOAT)


def test_known_bool_drives_edge_selection() -> None:
    def kernel(x: float) -> float:
        enabled = False
        if enabled:
            return x * 1000.0
        return x

    fact = _analyzed_return(kernel)
    assert fact == Residual(SemType.FLOAT)  # only the untaken arm multiplies; the taken path is the identity


def test_runtime_branch_joins_to_residual() -> None:
    def kernel(x: float) -> float:
        if x > 0.0:
            y = 1.0
        else:
            y = 2.0
        return y

    assert _analyzed_return(kernel) == Residual(SemType.FLOAT)


def test_possibly_unbound_local_is_a_located_rejection() -> None:
    def kernel(x: float) -> float:
        if x > 0.0:
            y = 1.0
        return y  # noqa: F821

    with pytest.raises(AnalysisRejection, match="may be unbound"):
        Analyzer(kernel).fixpoint()


def test_unbound_read_in_dead_branch_is_fine() -> None:
    def kernel(x: float) -> float:
        run_extra = False
        if run_extra:
            return extra_gain * x  # type: ignore[name-defined,no-any-return]  # noqa: F821
        return x

    assert _analyzed_return(kernel) == Residual(SemType.FLOAT)


def test_recursion_is_a_located_rejection() -> None:
    def spiral(v: float) -> float:
        return spiral(v - 1.0) if v > 0.0 else v

    def kernel(x: float) -> float:
        return spiral(x)

    with pytest.raises(AnalysisRejection, match="recursive"):
        Analyzer(kernel).fixpoint()


def test_runtime_element_loop_binds_trips_through_projections() -> None:
    # The trip count of a fixed-layout aggregate IS static, and each runtime element binds through a
    # synthesized projection prelude, so the loop unrolls exactly like one over constants.
    def kernel(x: float) -> float:
        acc = 0.0
        for step in (x, x + 1.0):
            acc = acc + step
        return acc

    result = Analyzer(kernel).fixpoint()
    assert result.binding_facts is not None  # analysis completes; the value itself is pinned in test_matrix


def test_assert_is_dropped_wholesale() -> None:
    # An assert is accepted and ignored (its test is never lowered), mirroring Python under -O, so even a
    # data-dependent assert has no effect and the kernel lowers to just its body.
    def kernel(x: float) -> float:
        assert x > 0.0, "input must be positive"
        return x

    assert _analyzed_return(kernel) == Residual(SemType.FLOAT)


def test_statically_true_assert_folds_away() -> None:
    def kernel(x: float) -> float:
        tolerance = 0.5
        assert tolerance > 0.0, "tolerance must be positive"
        return x * tolerance

    assert _analyzed_return(kernel) == Residual(SemType.FLOAT)


def test_live_state_promotes_and_dead_state_folds() -> None:
    class Filter:
        def __init__(self) -> None:
            self.accumulator = 0.0
            self.legacy_mode = False
            self.legacy_gain = 5.0

        def step(self, x: float) -> float:
            if self.legacy_mode:
                self.legacy_gain = 0.0
            self.accumulator = self.accumulator + x
            return self.accumulator * self.legacy_gain

    component = Filter()
    result = Analyzer(component.step).fixpoint()
    exit_env = result.block_in[result.unit.exit]
    assert exit_env.get(StateLeaf(component, ("accumulator",))) == Residual(SemType.FLOAT)
    assert exit_env.get(StateLeaf(component, ("legacy_gain",))) == Known(as_admitted(5.0))


def as_admitted(value: object) -> "StaticValue":
    from holoso._frontend._fir._value import admit

    admitted = admit(value)
    assert admitted is not None
    return admitted


def test_state_livein_joins_reset_with_exit_values() -> None:
    class Toggler:
        def __init__(self) -> None:
            self.mode = 0.0

        def step(self, x: float) -> float:
            previous = self.mode
            self.mode = 1.0
            return previous + x

    component = Toggler()
    result = Analyzer(component.step).fixpoint()
    # The live-in of mode is join(reset 0.0, exit 1.0) = Residual: the first transaction reads 0.0, later ones 1.0.
    assert result.block_in[result.unit.exit].get(StateLeaf(component, ("mode",))) == Known(as_admitted(1.0))
    facts = [fact for fact in result.block_in[result.unit.entry].facts.values()]
    assert Residual(SemType.FLOAT) in facts  # the promoted live-in is visible at entry


def test_member_call_expands_through_dunder_call() -> None:
    class Lpf:
        def __init__(self) -> None:
            self.alpha = 0.25

        def __call__(self, v: float) -> float:
            return v * self.alpha

    lpf = Lpf()

    def kernel(x: float) -> float:
        return lpf(x) + lpf(2.0)

    assert _analyzed_return(kernel) == Residual(SemType.FLOAT)


def test_expansion_origins_point_at_the_user_call_site() -> None:
    def failing_helper(v: float) -> float:
        return v & 1  # type: ignore[operator]  # a bitwise operator on a float: rejected inside the inlined callee

    def kernel(x: float) -> float:
        return failing_helper(x)

    with pytest.raises(AnalysisRejection) as info:
        Analyzer(kernel).fixpoint()
    assert "requires integer operands" in str(info.value)
    assert any(frame.function.endswith("kernel") for frame in info.value.origin)  # re-attributed via origin stack


def test_eager_boolean_values_fold_like_python() -> None:
    def kernel() -> float:
        limit = 3.0
        fallback = 7.0
        return limit and fallback

    _assert_known_matches_python(kernel)


def test_chained_comparison_folds_like_python() -> None:
    def kernel() -> bool:
        low, mid, high = 1.0, 2.0, 3.0
        return low < mid < high

    _assert_known_matches_python(kernel)


def test_unpack_arity_guard_folds_away_for_correct_code() -> None:
    def kernel(x: float) -> float:
        a, b = x + 1.0, x + 2.0
        return a + b

    assert _analyzed_return(kernel) == Residual(SemType.FLOAT)


def test_unpack_arity_mismatch_is_a_located_rejection() -> None:
    def kernel(x: float) -> float:
        a, b = x + 1.0, x + 2.0, x + 3.0  # type: ignore[misc]
        return a + b  # type: ignore[has-type,no-any-return]

    with pytest.raises(AnalysisRejection, match="unpack"):
        Analyzer(kernel).fixpoint()


def test_string_equality_branches_fold() -> None:
    def kernel() -> float:
        mode = "fast"
        if mode == "fast":
            return 1.0
        return 2.0

    _assert_known_matches_python(kernel)


def test_range_loops_unroll_via_concrete_builtin_calls() -> None:
    def kernel() -> float:
        n = 3
        total = 0.0
        for _ in range(n):
            total = total + 1.0
        return total

    _assert_known_matches_python(kernel)


def test_static_while_stays_a_runtime_loop() -> None:
    def kernel() -> float:
        x = 10.0
        while x > 1.0:
            x = x / 2.0
        return x

    # A while is a real CFG loop by design, never unrolled: the join at its header keeps the carried value residual
    # even when a concrete execution would terminate statically.
    assert _analyzed_return(kernel) == Residual(SemType.FLOAT)


def test_return_inside_static_loop_reaches_the_canonical_exit() -> None:
    def kernel(x: float) -> float:
        for gate in (1.0, 2.0):
            if x > gate:
                return 5.0
        return 0.0

    result = Analyzer(kernel).fixpoint()
    fact = result.block_in[result.unit.exit].get(ReturnPlace())
    assert fact == Residual(SemType.FLOAT)  # 5.0 and 0.0 join: the early return is visible at the ONE exit

    def all_return_inside(x: float) -> float:
        for gate in (0.0,):
            return x + gate
        return -1.0

    inner = Analyzer(all_return_inside).fixpoint()
    assert inner.unit.exit in inner.block_in  # the canonical exit is reached even when every path returns in-loop


def test_nested_and_aliased_state_promotes() -> None:
    class Inner:
        def __init__(self) -> None:
            self.count = 0.0

    class Outer:
        def __init__(self) -> None:
            self.inner = Inner()

        def step_nested(self, x: float) -> float:
            self.inner.count = self.inner.count + 1.0
            return self.inner.count

    component = Outer()
    result = Analyzer(component.step_nested).fixpoint()
    assert result.block_in[result.unit.exit].get(StateLeaf(component.inner, ("count",))) == Residual(SemType.FLOAT)


def test_inplace_list_mutation_is_rejected() -> None:
    def kernel(x: float) -> float:
        history = [x]
        history += [x + 1.0]  # in-place: an alias of history would observe this, unlike a rebinding
        return history[1]

    with pytest.raises(AnalysisRejection, match="in-place list mutation"):
        Analyzer(kernel).fixpoint()


def test_signed_zero_join_stays_residual() -> None:
    def kernel(x: float) -> float:
        y = 0.0 if x > 0.0 else -0.0
        return y

    assert _analyzed_return(kernel) == Residual(SemType.FLOAT)  # +0.0 and -0.0 are DIFFERENT: no false Known


def test_one_armed_aggregate_binding_joins_cleanly() -> None:
    def kernel(x: float) -> float:
        if x > 0.0:
            pair = (x, 1.0)
            return pair[0]
        return 0.0

    assert _analyzed_return(kernel) == Residual(SemType.FLOAT)


def test_known_sequences_join_elementwise() -> None:
    def kernel(x: float) -> float:
        coefficients = (1.0, 2.0) if x > 0.0 else (3.0, 2.0)
        return coefficients[0] * x + coefficients[1]

    assert _analyzed_return(kernel) == Residual(SemType.FLOAT)


def test_one_armed_state_read_still_folds() -> None:
    class Config:
        def __init__(self) -> None:
            self.gain = 5.0

        def step(self, x: float) -> float:
            if x > 0.0:
                boosted = self.gain * x
                return boosted
            return self.gain  # the unconditional read after a one-armed read must stay Known(5.0)

    component = Config()
    result = Analyzer(component.step).fixpoint()
    assert result.block_in[result.unit.exit].get(StateLeaf(component, ("gain",))) == Known(as_admitted(5.0))


def test_huge_range_is_refused_before_materialization() -> None:
    def kernel(x: float) -> float:
        total = 0.0
        for _ in range(10**9):
            total = total + x
        return total

    with pytest.raises(AnalysisRejection, match="unroll threshold"):
        Analyzer(kernel).fixpoint()


def test_recursion_inside_an_unrolled_loop_is_detected() -> None:
    def dive(v: float) -> float:
        return dive(v * 0.5)

    def kernel(x: float) -> float:
        total = 0.0
        for _ in (1.0, 2.0):
            total = total + dive(x)
        return total

    with pytest.raises(AnalysisRejection, match="recursive"):
        Analyzer(kernel).fixpoint()


def test_string_concatenation_folds() -> None:
    def kernel() -> float:
        mode = "fa" + "st"
        if mode == "fast":
            return 1.0
        return 2.0

    _assert_known_matches_python(kernel)


def test_intrinsic_folds_survive_argument_widening() -> None:
    def kernel(x: float) -> float:
        y = 2.0
        result = 0.0
        while result < x:  # the loop widens y after the first abstract pass over the body
            result = result + abs(y)
            y = y - x
        return result

    # abs is a registered intrinsic: a first pass sees Known(2.0), but once y widens to a runtime value the fold is
    # non-destructively redone as an HIR FloatAbs operation -- never left stale at Known(2.0).
    result = Analyzer(kernel).fixpoint()
    assert result.block_in[result.unit.exit].get(ReturnPlace()) == Residual(SemType.FLOAT)


def test_keyword_only_defaults_bind_like_python() -> None:
    def affine(value: float = 1.0, gain: float = 2.0, *, bias: float = 3.0) -> float:
        return value * gain + bias

    def kernel() -> float:
        return affine(10.0)

    _assert_known_matches_python(kernel)


def test_component_property_getter_inlines_and_recomputes() -> None:
    # A @property getter on a component is inlined (desugared to a bound call), so it recomputes from the current state
    # each read: after `self.value = x`, `self.doubled` reads the just-stored value.
    class Scaler:
        def __init__(self) -> None:
            self.value = 0.0

        @property
        def doubled(self) -> float:
            return self.value * 2.0

        def step(self, x: float) -> float:
            self.value = x
            return self.doubled

    assert _analyzed_return(Scaler().step) == Residual(SemType.FLOAT)


def test_conditional_state_store_keeps_the_read_bound() -> None:
    class Latch:
        def __init__(self) -> None:
            self.value = 0.0

        def step(self, update: float) -> float:
            if update > 0.0:
                self.value = update
            return self.value  # Python always reads a bound value; no MaybeUnbound may appear here

    component = Latch()
    result = Analyzer(component.step).fixpoint()
    assert result.block_in[result.unit.exit].get(StateLeaf(component, ("value",))) == Residual(SemType.FLOAT)


def test_invariant_bool_state_keeps_folding_dead_guards() -> None:
    class Mode:
        def __init__(self) -> None:
            self.enabled = False
            self.gain = 7.0

        def step(self, x: float) -> float:
            self.enabled = False  # stored every transaction, yet invariantly False
            if self.enabled:
                self.gain = 0.0
            return self.gain * x

    component = Mode()
    result = Analyzer(component.step).fixpoint()
    assert result.block_in[result.unit.exit].get(StateLeaf(component, ("gain",))) == Known(as_admitted(7.0))


def test_property_setters_are_a_located_rejection() -> None:
    class Doubling:
        def __init__(self) -> None:
            self._backing = 1.0

        @property
        def x(self) -> float:
            return self._backing

        @x.setter
        def x(self, value: float) -> None:
            self._backing = value * 2.0

        def step(self, v: float) -> float:
            self.x = v + 1.0
            return self._backing

    with pytest.raises(AnalysisRejection, match="property"):
        Analyzer(Doubling().step).fixpoint()


def test_list_method_calls_are_a_located_rejection() -> None:
    def kernel(x: float) -> float:
        values = [10.0, 20.0]
        values.append(x)  # mutates a disposable reconstruction; the fact would go stale
        return values[0]

    with pytest.raises(AnalysisRejection, match="list method"):
        Analyzer(kernel).fixpoint()


def test_double_del_is_a_located_rejection() -> None:
    def kernel(x: float) -> float:
        value = x
        del value
        del value  # Python raises UnboundLocalError here
        return 20.0

    with pytest.raises(AnalysisRejection, match="unbound at this del"):
        Analyzer(kernel).fixpoint()


def test_ambiguous_array_truth_is_a_located_rejection() -> None:
    def kernel(x: float) -> float:
        if _TABLE_FOR_TRUTH:  # numpy raises: the truth of a multi-element array is ambiguous
            return x
        return 7.0

    with pytest.raises(AnalysisRejection, match="truth"):
        Analyzer(kernel).fixpoint()


import numpy as _np

_TABLE_FOR_TRUTH = _np.array([1.0, 2.0])


def test_positional_only_parameters_refuse_keywords() -> None:
    def helper(value: float, /) -> float:
        return value + 1.0

    def kernel() -> float:
        return helper(value=10.0)  # type: ignore[call-arg]  # Python raises TypeError

    with pytest.raises(AnalysisRejection, match="missing argument"):
        Analyzer(kernel).fixpoint()


def test_wrapped_helpers_are_analyzed_as_wrappers() -> None:
    import functools
    from collections.abc import Callable

    def traced(fn: Callable[[float], float]) -> Callable[[float], float]:
        @functools.wraps(fn)
        def wrapper(*args: float) -> float:
            return fn(*args)

        return wrapper

    @traced
    def offset(v: float) -> float:
        return v + 0.5

    def kernel() -> float:
        return offset(2.0)

    # A wrapper may add behavior, so it is analyzed as-is; this variadic one is a located rejection, never a
    # silent unwrap to the wrapped function's semantics.
    with pytest.raises(AnalysisRejection, match="variadic parameters"):
        Analyzer(kernel).fixpoint()


def test_slotted_components_are_not_properties() -> None:
    class Accumulator:
        __slots__ = ("total",)

        def __init__(self) -> None:
            self.total = 0.0

        def step(self, x: float) -> float:
            self.total = self.total + x
            return self.total

    component = Accumulator()
    result = Analyzer(component.step).fixpoint()
    assert result.block_in[result.unit.exit].get(StateLeaf(component, ("total",))) == Residual(SemType.FLOAT)


def test_branch_arm_method_calls_join_cleanly() -> None:
    class Comp:
        def __init__(self) -> None:
            self.offset = 1.0

        def helper(self, v: float) -> float:
            return v + self.offset

        def step(self, x: float) -> float:
            if x > 0.0:
                y = self.helper(x)
            else:
                y = self.helper(-x)
            return y

    result = Analyzer(Comp().step).fixpoint()
    assert result.block_in[result.unit.exit].get(ReturnPlace()) == Residual(SemType.FLOAT)


def test_record_method_calls_are_a_located_rejection() -> None:
    import dataclasses

    @dataclasses.dataclass
    class Acc:
        total: float = 0.0

        def bumped(self, v: float) -> float:
            return self.total + v

    def kernel(x: float) -> float:
        acc = Acc()
        return acc.bumped(x)

    with pytest.raises(AnalysisRejection, match="record attribute"):
        Analyzer(kernel).fixpoint()


def test_bool_arithmetic_is_a_located_rejection() -> None:
    def kernel() -> float:
        flag = 3.0 > 1.0
        return -flag  # the charter demands an explicit conversion; np-bool provenance makes folds unsound

    with pytest.raises(AnalysisRejection, match="explicit conversion"):
        Analyzer(kernel).fixpoint()


def test_join_rejections_are_located() -> None:
    def kernel(x: float) -> float:
        if x > 0.0:
            y = 1.0
        else:
            y = "text"  # type: ignore[assignment]  # honest type confusion; the message must locate it
        return y

    with pytest.raises(AnalysisRejection) as info:
        Analyzer(kernel).fixpoint()
    assert info.value.origin[0].line > 0


def test_unary_minus_on_a_runtime_tuple_is_a_located_rejection() -> None:
    def kernel(x: float) -> float:
        pair = (x, 2.0)
        neg = -pair  # type: ignore[operator]  # Python raises TypeError
        return neg[1]  # type: ignore[no-any-return]

    with pytest.raises(AnalysisRejection, match="aggregate"):
        Analyzer(kernel).fixpoint()


def test_lazily_created_state_is_a_located_rejection() -> None:
    class Lazy:
        def step(self, x: float) -> float:
            self.stash = x  # never assigned in __init__: no reset value exists
            return self.stash

    with pytest.raises(AnalysisRejection, match="does not exist on the component"):
        Analyzer(Lazy().step).fixpoint()


def test_classmethod_helpers_join_cleanly() -> None:
    class Comp:
        gain = 2.0

        @classmethod
        def scaled(cls, v: float) -> float:
            return v * cls.gain

        def __init__(self) -> None:
            self.offset = 1.0

        def step(self, x: float) -> float:
            if x > 0.0:
                y = self.scaled(x)
            else:
                y = self.scaled(-x)
            return y + self.offset

    result = Analyzer(Comp().step).fixpoint()
    assert result.block_in[result.unit.exit].get(ReturnPlace()) == Residual(SemType.FLOAT)


def test_tuple_repetition_folds_like_python() -> None:
    def kernel() -> float:
        window = (0.0,) * 3
        return window[2] + 1.0

    _assert_known_matches_python(kernel)


def test_instance_callbacks_shadow_class_methods() -> None:
    def triple(v: float) -> float:
        return v * 3.0

    class Comp:
        def transform(self, v: float) -> float:
            return v * 2.0

        def __init__(self) -> None:
            self.transform = triple  # type: ignore[method-assign]  # honest callback slot shadows the method

        def step(self, x: float) -> float:
            return self.transform(2.0)

    result = Analyzer(Comp().step).fixpoint()
    fact = result.block_in[result.unit.exit].get(ReturnPlace())
    assert isinstance(fact, Known) and as_python(fact.value) == 6.0  # Python binds the instance attribute


def test_cached_property_is_a_located_rejection() -> None:
    import functools

    class Comp:
        def __init__(self) -> None:
            self.base = 2.0

        @functools.cached_property
        def gain(self) -> float:
            return self.base * 2.0

        def step(self, x: float) -> float:
            self.base = 3.0
            return self.gain * x

    with pytest.raises(AnalysisRejection, match="descriptor attribute"):
        Analyzer(Comp().step).fixpoint()


def test_custom_setattr_components_are_a_located_rejection() -> None:
    class Clamped:
        def __init__(self) -> None:
            object.__setattr__(self, "gain", 0.5)

        def __setattr__(self, name: str, value: float) -> None:
            object.__setattr__(self, name, min(1.0, max(0.0, value)))

        def step(self, x: float) -> float:
            self.gain = 3.0  # Python clamps to 1.0; a direct leaf write would record 3.0
            return self.gain * x

    with pytest.raises(AnalysisRejection, match="custom __setattr__"):
        Analyzer(Clamped().step).fixpoint()


def test_bound_method_keywords_bind_without_positional_only_underflow() -> None:
    class Comp:
        def affine(self, value: float, gain: float) -> float:
            return value * gain

        def step(self, x: float) -> float:
            return self.affine(value=3.0, gain=4.0) + x

    result = Analyzer(Comp().step).fixpoint()
    assert result.block_in[result.unit.exit].get(ReturnPlace()) == Residual(SemType.FLOAT)


def test_value_method_identities_join_across_arms() -> None:
    # str.count keeps the value-method machinery exercised now that sequence .count/.index reject (their
    # identity-and-equality semantics are not reconstruction-safe); substring counting is value-determined.
    def kernel(x: float) -> float:
        table = "aab"
        if x > 0.0:
            n = table.count("a")  # 2
        else:
            n = table.count("b")  # 1
        return n * 1.0  # the join must reconcile the two method fetches, not reject them

    result = Analyzer(kernel).fixpoint()
    assert result.block_in[result.unit.exit].get(ReturnPlace()) == Residual(SemType.FLOAT)


def test_residual_true_division_types_as_float() -> None:
    class Counter:
        def __init__(self) -> None:
            self.n = 0.0

        def step(self, x: float) -> float:
            self.n = self.n + 1.0
            return self.n / 2.0

    result = Analyzer(Counter().step).fixpoint()
    assert result.block_in[result.unit.exit].get(ReturnPlace()) == Residual(SemType.FLOAT)

    def int_division(x: float) -> float:
        n = 1 if x > 0.0 else 2
        return n / 2  # Python / yields float unconditionally, int operands included

    assert _analyzed_return(int_division) == Residual(SemType.FLOAT)


def test_scalar_matmul_is_a_located_rejection() -> None:
    def kernel(x: float) -> float:
        return x @ 2.0  # type: ignore[operator,no-any-return]  # Python: TypeError, unconditionally

    with pytest.raises(AnalysisRejection, match="not defined for scalars"):
        Analyzer(kernel).fixpoint()


def test_bool_state_joining_with_float_is_a_located_rejection() -> None:
    class Flagged:
        def __init__(self) -> None:
            self.mode = False

        def step(self, x: float) -> float:
            if x > 0.0:
                self.mode = 1.0  # type: ignore[assignment]  # honest type drift: bool state must not enter the float bank
            return 1.0 if self.mode else 0.0

    with pytest.raises(AnalysisRejection, match="irreconcilable"):
        Analyzer(Flagged().step).fixpoint()
