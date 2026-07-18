"""
Frontend tests: the fixed storage schema (B1). A variable is strongly typed for the function's lifetime: its
first definition establishes the schema (independent first definitions on different paths join, int promoting
to float), and once established a store may only keep the kind -- bool accepts bool, int accepts int, float
accepts float or int (the integer converts on the store edge: a statically Known one exact-or-reject, a runtime
one with the hardware conversion's rounding). Every other store is a located rejection at the store site,
resolved over the stabilized graph in CFG preorder, outranking any downstream rejection the carried fact
provokes. The schema sees SemType kinds only: aggregate-valued stores to locals are fact-only (a reshape or
reflavor is a representation change, not a type change), as are references, strings, and ranges; ``del`` does
not erase a schema. Persistent state slots instead take their full schema -- flavor, geometry, per-cell kinds
-- from the reset value.
"""

import math
from collections.abc import Callable

import numpy as np
import pytest

import holoso
from holoso import FloatFormat, UnsupportedConstruct

from ._modelref import default_ops


def _synthesize(fn: Callable[..., object], name: str) -> holoso.SynthesisResult:
    return holoso.synthesize(fn, default_ops(FloatFormat(11, 52)), name=name)


def _reject(fn: Callable[..., object], match: str) -> UnsupportedConstruct:
    with pytest.raises(UnsupportedConstruct, match=match) as excinfo:
        _synthesize(fn, "rejected")
    return excinfo.value


# ---------------------------------------- locals ----------------------------------------


def test_local_int_rebound_to_float_rejects_at_the_store() -> None:
    def kernel(v: float) -> float:
        x = 0
        x = v  # type: ignore[assignment]
        return x

    error = _reject(kernel, "variables are strongly typed")
    assert error.location is not None and error.location.line is not None
    assert "x = v" in error.location.line


def test_local_float_rebound_to_bool_rejects_at_the_store() -> None:
    def kernel(v: float) -> float:
        y = v * 2.0
        y = v > 0.0
        return float(y)

    error = _reject(kernel, "variables are strongly typed")
    assert error.location is not None and error.location.line is not None
    assert "y = v > 0.0" in error.location.line


def test_augmented_assignment_cannot_change_the_kind() -> None:
    def kernel(v: float) -> float:
        x = 1
        x += v  # type: ignore[assignment]
        return x

    error = _reject(kernel, "variables are strongly typed")
    assert error.location is not None and error.location.line is not None
    assert "x += v" in error.location.line


def test_loop_carried_type_change_rejects_at_the_store() -> None:
    def kernel(v: float) -> float:
        x = 1
        while x < 4:
            x = x + v  # type: ignore[assignment]
        return float(x)

    error = _reject(kernel, "variables are strongly typed")
    assert error.location is not None and error.location.line is not None
    assert "x = x + v" in error.location.line


def test_root_parameter_annotation_establishes_the_schema() -> None:
    def kernel(x: float) -> float:
        x = True
        return float(x)

    _reject(kernel, "variables are strongly typed")


def test_del_does_not_erase_an_established_schema() -> None:
    def kernel(v: float) -> float:
        x = 0
        del x
        x = v  # type: ignore[assignment]
        return x

    _reject(kernel, "variables are strongly typed")


def test_loop_counter_rebinding_to_a_float_rejects() -> None:
    def kernel(v: float) -> float:
        for i in range(1):
            pass
        i = v  # type: ignore[assignment]  # noqa: PLW2901
        return i

    _reject(kernel, "variables are strongly typed")


def test_aggregate_rebinding_is_a_representation_change_not_a_type_change() -> None:
    # The finite-set controller reshapes a parameter in place and the in-place-mutation rejection tells users to
    # grow lists by rebinding, so aggregate stores to locals stay outside the schema: shape, flavor, and arity
    # may all change, and the leaf kinds ride the fact flow exactly as before.
    from collections.abc import Sequence

    def kernel(a: float, b: float) -> float:
        v = np.array([a, b])
        v = v.reshape((2, 1))
        acc: list[float] = []
        acc = acc + [a]
        acc = acc + [b]
        t: Sequence[float] = (acc[0], acc[1])
        t = [t[0], t[1], b]
        return float(v[1, 0]) + t[0] * t[2]

    model = _synthesize(kernel, "agg_rebind").numerical_model.elaborate()
    assert float(model.run(2.0, 3.0)[0]) == 3.0 + 2.0 * 3.0


def test_local_array_dtype_rebinding_follows_the_fact_flow() -> None:
    def kernel(a: float) -> float:
        v = np.array([4, 2])
        v = v * a
        return float(v[0])

    model = _synthesize(kernel, "local_dtype").numerical_model.elaborate()
    assert float(model.run(0.5)[0]) == 2.0


def test_a_dead_store_does_not_violate_the_schema() -> None:
    def kernel(v: float) -> float:
        x = v
        if False:
            x = True
        return x

    model = _synthesize(kernel, "dead_store").numerical_model.elaborate()
    assert float(model.run(2.5)[0]) == 2.5


def test_non_datapath_values_neither_establish_nor_violate() -> None:
    def kernel(v: float) -> float:
        x = "note"
        x = 1.0  # type: ignore[assignment]  # the string neither established nor blocks the schema
        x = v  # type: ignore[assignment]
        return x  # type: ignore[return-value]

    model = _synthesize(kernel, "fact_only").numerical_model.elaborate()
    assert float(model.run(3.25)[0]) == 3.25


# ---------------------------------------- the store-edge conversion ----------------------------------------


def test_int_store_into_a_float_variable_converts_on_the_store_edge() -> None:
    def kernel(value: float) -> float:
        x = value
        x = 3
        return x / 2

    model = _synthesize(kernel, "store_edge_exact").numerical_model.elaborate()
    assert float(model.run(0.0)[0]) == 1.5


def test_inexact_int_store_into_a_float_variable_rejects_on_the_carrier_rule() -> None:
    # Without the conversion the int fact survived inside the float variable and the subtraction folded as EXACT
    # integer arithmetic, returning 1.0 where the explicit float() spelling returns 0.0 -- a silent value
    # divergence between equivalent spellings.
    def implicit(value: float) -> float:
        current = value
        current = 2**53 + 1
        return current - 2**53

    error = _reject(implicit, "not exactly representable in the binary64 carrier")
    assert error.location is not None and error.location.line is not None
    assert "current = 2**53 + 1" in error.location.line

    def explicit(value: float) -> float:
        current = value
        current = float(2**53 + 1)
        return current - 2**53

    model = _synthesize(explicit, "store_edge_spelled_rounding").numerical_model.elaborate()
    assert float(model.run(0.0)[0]) == 0.0


def test_dead_oversized_int_store_into_a_float_variable_rejects() -> None:
    def kernel(value: float) -> float:
        x = 1.5
        x = 2**7000  # noqa: F841
        return value

    _reject(kernel, "beyond the binary64 carrier range")


def test_store_edge_conversion_verdict_is_arm_order_independent() -> None:
    # The exactness verdict is derived from the STABILIZED facts, never from a transient pre-join Known: the
    # inexact constant sits in one branch arm, the other arm holds 0, and both spellings must agree -- each is
    # a legal runtime integer store that converts on the store edge with the merge's float promotion.
    def else_arm(x: float) -> float:
        acc = 0.0
        n = 0 if x > 0.0 else 2**53 + 1
        acc = n
        return acc * 0.5

    def then_arm(x: float) -> float:
        acc = 0.0
        n = (2**53 + 1) if x > 0.0 else 0
        acc = n
        return acc * 0.5

    rounded_half = float(2**53 + 1) * 0.5
    else_model = _synthesize(else_arm, "conv_else_arm").numerical_model.elaborate()
    then_model = _synthesize(then_arm, "conv_then_arm").numerical_model.elaborate()
    assert float(else_model.run(1.0)[0]) == 0.0 and float(else_model.run(-1.0)[0]) == rounded_half
    assert float(then_model.run(1.0)[0]) == rounded_half and float(then_model.run(-1.0)[0]) == 0.0


def test_state_store_edge_conversion_verdict_is_arm_order_independent() -> None:
    class ElseArm:
        def __init__(self) -> None:
            self.y = 0.0

        def step(self, x: float) -> float:
            n = 0 if x > 0.0 else 2**53 + 1
            self.y = n
            return self.y

    class ThenArm:
        def __init__(self) -> None:
            self.y = 0.0

        def step(self, x: float) -> float:
            n = (2**53 + 1) if x > 0.0 else 0
            self.y = n
            return self.y

    rounded = float(2**53 + 1)
    else_model = _synthesize(ElseArm().step, "state_conv_else").numerical_model.elaborate()
    then_model = _synthesize(ThenArm().step, "state_conv_then").numerical_model.elaborate()
    assert float(else_model.run(1.0)[0]) == 0.0 and float(else_model.run(-1.0)[0]) == rounded
    assert float(then_model.run(1.0)[0]) == rounded and float(then_model.run(-1.0)[0]) == 0.0


def test_the_conversion_survives_del_like_the_schema_does() -> None:
    def kernel(value: float) -> float:
        x = 1.5
        del x
        x = 2**53 + 1
        return x - 2**53

    _reject(kernel, "not exactly representable in the binary64 carrier")


def _fround_ops() -> holoso.OpConfig:
    import dataclasses

    fmt = FloatFormat(11, 52)
    return dataclasses.replace(default_ops(fmt), fround=holoso.FRoundOperator(fmt))


def test_runtime_int_store_into_a_float_local_emits_the_conversion() -> None:
    # The analyzer converts the FACT on the store edge; the datapath must convert too, or downstream consumers
    # read an integer cell under a float fact (the truth test picked IntToBool and the dangling float_to_int
    # rejected at MIR). Both spellings must synthesize and agree with Python.
    def implicit(value: float) -> float:
        current = value
        current = int(value)
        return 1.0 if current else 0.0

    def explicit(value: float) -> float:
        current = value
        current = float(int(value))
        return 1.0 if current else 0.0

    for name, kernel in (("store_conv_implicit", implicit), ("store_conv_explicit", explicit)):
        model = holoso.synthesize(kernel, _fround_ops(), name=name).numerical_model.elaborate()
        for value in (-2.5, -1.0, -0.5, 0.0, 0.5, 1.0, 2.5):
            assert float(model.run(value)[0]) == kernel(value), f"{name}({value})"


def test_runtime_int_store_into_a_float_slot_emits_the_conversion() -> None:
    class Acc:
        def __init__(self) -> None:
            self.acc = 0.0

        def step(self, value: float) -> float:
            self.acc = int(value)
            return 1.0 if self.acc else 0.0

    model = holoso.synthesize(Acc().step, _fround_ops(), name="state_store_conv").numerical_model.elaborate()
    reference = Acc()
    for value in (-2.5, -0.5, 0.0, 0.5, 2.5):
        assert float(model.run(value)[0]) == reference.step(value), f"step({value})"


def test_all_integer_phi_store_converts_at_runtime_like_the_explicit_spelling() -> None:
    # A genuinely runtime integer (an all-constant int phi included) converts AT RUNTIME with the hardware
    # conversion's round-to-nearest; only a statically Known stored value is exact-or-reject. Any constant
    # folding of that conversion must match the runtime operator bit-for-bit, so the phi spelling and the
    # explicit float() spelling must produce identical outputs on both paths.
    def phi(flag: bool) -> float:
        x = (2**53 + 1) if flag else 1
        y = 0.0
        y = x
        return y

    def explicit(flag: bool) -> float:
        x = (2**53 + 1) if flag else 1
        y = 0.0
        y = float(x)
        return y

    phi_model = _synthesize(phi, "phi_store_conv").numerical_model.elaborate()
    explicit_model = _synthesize(explicit, "explicit_store_conv").numerical_model.elaborate()
    for flag in (True, False):
        phi_out = phi_model.run(flag)[0]
        explicit_out = explicit_model.run(flag)[0]
        assert float(phi_out) == float(explicit_out) == float(2**53 + 1 if flag else 1), f"flag={flag}"


# ---------------------------------------- per-execution-scope freshness ----------------------------------------


def test_comprehension_target_schema_is_fresh_per_execution() -> None:
    def int_first(v: float) -> float:
        acc = 0.0
        for sequence in ((1,), (2.0,)):
            converted = [float(item) for item in sequence]
            acc = acc + converted[0] * v
        return acc

    def float_first(v: float) -> float:
        acc = 0.0
        for sequence in ((2.0,), (1,)):
            converted = [float(item) for item in sequence]
            acc = acc + converted[0] * v
        return acc

    for name, kernel in (("comp_int_first", int_first), ("comp_float_first", float_first)):
        model = _synthesize(kernel, name).numerical_model.elaborate()
        assert float(model.run(2.0)[0]) == kernel(2.0) == 6.0


def test_unrolled_loop_target_kind_is_python_faithful_per_trip() -> None:
    def int_first(v: float) -> float:
        acc = 0.0
        for c in (1, 2.5):
            acc = acc + c * v
        return acc

    def float_first(v: float) -> float:
        acc = 0.0
        for c in (2.5, 1):
            acc = acc + c * v
        return acc

    for name, kernel in (("loop_int_first", int_first), ("loop_float_first", float_first)):
        model = _synthesize(kernel, name).numerical_model.elaborate()
        assert float(model.run(2.0)[0]) == kernel(2.0) == 7.0


def test_loop_target_remains_usable_after_the_loop() -> None:
    def kernel(v: float) -> float:
        for c in (1, 2.5):
            pass
        return c * v  # the last trip's binding survives the loop, as in Python

    model = _synthesize(kernel, "post_loop_target").numerical_model.elaborate()
    assert float(model.run(2.0)[0]) == 5.0


def test_literal_float_store_into_an_int_variable_still_rejects() -> None:
    def kernel(v: float) -> float:
        x = 0
        x = 1.5  # type: ignore[assignment]  # noqa: F841
        return v

    _reject(kernel, "variables are strongly typed")


def test_local_store_violation_outranks_its_downstream_secondary_rejection() -> None:
    # The violating store carries its float fact onward, making the downstream shift ill-typed; the causal
    # store rejection must be the one reported, never the secondary operator rejection it provoked.
    def kernel(x: float) -> float:
        count = 0
        count = x  # type: ignore[assignment]
        return float(count << 1)

    error = _reject(kernel, "variable 'count' is an int and cannot be rebound")
    assert error.location is not None and error.location.line is not None
    assert "count = x" in error.location.line


def test_the_first_violating_store_in_preorder_wins_over_a_later_conversion_failure() -> None:
    # A mid-flight raise at the binary64 store would preempt the earlier rebinding; the resolution walk ranks
    # every violation kind uniformly, so the first store in CFG preorder reports.
    def kernel(x: float) -> float:
        flag = True
        flag = x  # type: ignore[assignment]  # noqa: F841
        acc = 0.0
        acc = 2**53 + 1
        return acc - 2**53

    error = _reject(kernel, "variable 'flag' is a bool and cannot be rebound")
    assert error.location is not None and error.location.line is not None
    assert "flag = x" in error.location.line


# ---------------------------------------- calibration: what stays legal ----------------------------------------


def test_ifexp_int_cast_arm_stays_a_legal_float_phi() -> None:
    import dataclasses

    def kernel(c: bool, v: float) -> float:
        x = int(v) if c else v
        return float(x)

    ops = dataclasses.replace(default_ops(FloatFormat(11, 52)), fround=holoso.FRoundOperator(FloatFormat(11, 52)))
    model = holoso.synthesize(kernel, ops, name="ifexp_phi").numerical_model.elaborate()
    assert float(model.run(True, 2.75)[0]) == 2.0
    assert float(model.run(False, 2.75)[0]) == 2.75


def test_independent_first_definitions_join_with_promotion() -> None:
    def kernel(c: bool, v: float) -> float:
        if c:
            x = 1
        else:
            x = v  # type: ignore[assignment]
        return float(x)

    model = _synthesize(kernel, "arm_join").numerical_model.elaborate()
    assert float(model.run(True, 3.5)[0]) == 1.0
    assert float(model.run(False, 3.5)[0]) == 3.5


def test_float_variable_accepts_an_int_store_with_conversion() -> None:
    def kernel(v: float) -> float:
        x = 2.5
        if v > 0.0:
            x = 1  # int into an established float: converts on the store edge
        return x

    model = _synthesize(kernel, "float_accepts_int").numerical_model.elaborate()
    assert float(model.run(1.0)[0]) == 1.0
    assert float(model.run(-1.0)[0]) == 2.5


# ---------------------------------------- persistent state ----------------------------------------


def test_int_slot_stored_a_float_that_reaches_the_exit_rejects() -> None:
    class Drift:
        def __init__(self) -> None:
            self.n = 0

        def step(self, v: float) -> float:
            self.n = v  # type: ignore[assignment]
            return self.n

    error = _reject(Drift().step, "stores an incompatible type")
    assert error.location is not None and error.location.line is not None
    assert "self.n = v" in error.location.line


def test_int_slot_stored_a_float_then_restored_rejects_at_the_bad_store() -> None:
    class Restore:
        def __init__(self) -> None:
            self.n = 0

        def step(self, v: float) -> float:
            self.n = v * 2.0  # type: ignore[assignment]
            out = self.n
            self.n = 0
            return out

    error = _reject(Restore().step, "stores an incompatible type")
    assert error.location is not None and error.location.line is not None
    assert "self.n = v * 2.0" in error.location.line


def test_bool_slot_stored_a_float_rejects_at_the_store() -> None:
    class Flag:
        def __init__(self) -> None:
            self.armed = False

        def step(self, v: float) -> float:
            self.armed = v  # type: ignore[assignment]
            return v

    error = _reject(Flag().step, "stores an incompatible type")
    assert error.location is not None and error.location.line is not None
    assert "self.armed = v" in error.location.line


def test_int_reset_array_slot_stored_float_cells_rejects() -> None:
    class Decay:
        def __init__(self) -> None:
            self.v = np.array([4, 2])

        def step(self, a: float) -> float:
            self.v = self.v * a
            return float(self.v[0])

    _reject(Decay().step, "stores an incompatible type at cell")


def test_state_store_violation_outranks_its_downstream_secondary_rejection() -> None:
    # The violating store carries its float fact onward, making the downstream shift ill-typed; the causal store
    # rejection must be the one reported, never the secondary operator rejection it provoked.
    class Counter:
        def __init__(self) -> None:
            self.count = 0

        def step(self, value: float) -> float:
            self.count = value  # type: ignore[assignment]
            return float(self.count << 1)

    error = _reject(Counter().step, "state attribute 'count' stores an incompatible type")
    assert error.location is not None and error.location.line is not None
    assert "self.count = value" in error.location.line


def test_competing_state_violations_report_the_then_arm_first() -> None:
    # Two independent violations in opposite branch arms rank in CFG preorder (then-arm first) regardless of
    # the order the worklist happened to discover the arms, and regardless of whether a downstream secondary
    # rejection forces the deferral path: both shapes must report then_count.
    class Counter:
        def __init__(self) -> None:
            self.then_count = 0
            self.else_count = 0

        def step(self, v: float) -> float:
            if v > 0.0:
                self.then_count = v  # type: ignore[assignment]
            else:
                self.else_count = v  # type: ignore[assignment]
            return v

    class CounterShift:
        def __init__(self) -> None:
            self.then_count = 0
            self.else_count = 0

        def step(self, v: float) -> float:
            if v > 0.0:
                self.then_count = v  # type: ignore[assignment]
            else:
                self.else_count = v  # type: ignore[assignment]
            return float(self.else_count << 1)

    for kernel in (Counter().step, CounterShift().step):
        error = _reject(kernel, "state attribute 'then_count' stores an incompatible type")
        assert error.location is not None and error.location.line is not None
        assert "self.then_count = v" in error.location.line


def test_competing_state_conversion_failures_report_the_then_arm_first() -> None:
    class TwoStores:
        def __init__(self) -> None:
            self.a = 0.0
            self.b = 0.0

        def step(self, v: float) -> float:
            if v > 0.0:
                self.a = 2**53 + 1
            else:
                self.b = 2**53 + 1
            return v

    error = _reject(TwoStores().step, "state attribute 'a' is a float; the stored integer is not exactly")
    assert error.location is not None and error.location.line is not None
    assert "self.a = 2**53 + 1" in error.location.line


def test_state_store_violation_outranks_a_provoked_loop_rejection() -> None:
    # The carried float defers the range() call, which leaves the loop iterable unresolved at the TERMINATOR;
    # the causal store must still outrank that provoked failure, exactly as it outranks a provoked op failure.
    class LoopBound:
        def __init__(self) -> None:
            self.n = 0

        def step(self, v: float) -> float:
            self.n = v  # type: ignore[assignment]
            acc = 0.0
            for _ in range(self.n):
                acc = acc + 1.0
            return acc

    error = _reject(LoopBound().step, "state attribute 'n' stores an incompatible type")
    assert error.location is not None and error.location.line is not None
    assert "self.n = v" in error.location.line


def test_float_slot_stored_an_inexact_int_rejects_at_the_store() -> None:
    class Acc:
        def __init__(self) -> None:
            self.total = 0.0

        def step(self, v: float) -> float:
            self.total = 2**53 + 1
            return self.total - v

    error = _reject(Acc().step, "state attribute 'total' is a float; the stored integer is not exactly representable")
    assert error.location is not None and error.location.line is not None
    assert "self.total = 2**53 + 1" in error.location.line


def test_float_array_slot_stored_an_inexact_int_cell_rejects_at_the_store() -> None:
    class Vec:
        def __init__(self) -> None:
            self.w = np.array([0.0, 0.0])

        def step(self, v: float) -> float:
            self.w = np.array([1, 2**53 + 1])
            return float(self.w[1]) - v

    error = _reject(
        Vec().step, "state attribute 'w' cell 1 is a float; the stored integer is not exactly representable"
    )
    assert error.location is not None and error.location.line is not None
    assert "self.w = np.array" in error.location.line


def test_int_slot_kept_integer_stays_an_integer_slot() -> None:
    from holoso._frontend import lower
    from holoso._hir import IntType

    class Count:
        def __init__(self) -> None:
            self.n = 0

        def step(self, up: bool) -> float:
            if up:
                self.n = self.n + 1
            return float(self.n)

    hir = lower(Count().step)
    (slot,) = hir.state_slots
    assert isinstance(hir.nodes[slot.live_out].type, IntType)


def test_float_slot_accepts_int_stores_with_conversion() -> None:
    class Timer:
        def __init__(self) -> None:
            self.t = 0.0

        def step(self, v: float) -> float:
            if v > 0.0:
                self.t = 0  # uart-style: the reset fixes float, the int converts on the store edge
            else:
                self.t = self.t + v
            return self.t

    model = _synthesize(Timer().step, "float_slot_int_store").numerical_model.elaborate()
    reference = Timer()
    for v in (1.0, -2.0, -3.0, 4.0):
        assert float(model.run(v)[0]) == reference.step(v)


def test_float_vector_slot_accepts_int_cells_with_conversion() -> None:
    class Vec:
        def __init__(self) -> None:
            self.v = [0.0, 0.0]

        def step(self, a: float) -> float:
            self.v = [1, self.v[1] + a]
            return self.v[0] + self.v[1]

    model = _synthesize(Vec().step, "float_vec_int_cell").numerical_model.elaborate()
    reference = Vec()
    for a in (2.0, 3.0):
        assert float(model.run(a)[0]) == reference.step(a)


# ------------------------------- the deferral net across layers and rounds -------------------------------


def test_local_violation_outranks_the_join_rejection_its_carried_fact_provokes() -> None:
    # The violating store carries its float onward, which meets the established bool at the branch merge; the
    # join-layer rejection must defer like any provoked failure, so the branchy spelling reports exactly what
    # the straight-line spelling does.
    def kernel(x: float) -> float:
        flag = True
        if x > 0.0:
            flag = x  # type: ignore[assignment]
        return x

    error = _reject(kernel, "variable 'flag' is a bool and cannot be rebound to a float")
    assert error.location is not None and error.location.line is not None
    assert "flag = x" in error.location.line


def test_state_violation_outranks_the_join_rejection_its_carried_fact_provokes() -> None:
    class Meter:
        def __init__(self) -> None:
            self.n = 0

        def step(self, x: float) -> float:
            if x > 0.0:
                t = True
            else:
                self.n = x  # type: ignore[assignment]
                t = self.n  # type: ignore[assignment]
            return float(t)

    error = _reject(Meter().step, "state attribute 'n' stores an incompatible type")
    assert error.location is not None and error.location.line is not None
    assert "self.n = x" in error.location.line


def test_local_violation_outranks_a_provoked_library_rejection() -> None:
    # The library refusal is a sibling of the analysis rejection under the shared located mixin; the net must
    # hold back both, or the missing-primitive diagnostic masks the causal store.
    def kernel(x: float) -> float:
        n = 0
        n = x  # type: ignore[assignment]
        return math.gamma(x)

    error = _reject(kernel, "variable 'n' is an int and cannot be rebound to a float")
    assert error.location is not None and error.location.line is not None
    assert "n = x" in error.location.line


def test_state_join_failure_defers_to_stabilization_instead_of_a_transient_exactness_verdict() -> None:
    # The sum is a Known integer only in round one (the leaf's live-in stabilizes to a runtime integer), so an
    # exactness verdict drawn when the unrelated state-merge failure surfaces would be false; the merge failure
    # must instead ride to stabilization, where both spellings report it identically.
    class Implicit:
        def __init__(self) -> None:
            self.n = 0
            self.tag = False

        def step(self, x: float) -> float:
            out = 0.0
            n = self.n + (2**53 + 1)
            out = n
            self.n = 1
            self.tag = "s"  # type: ignore[assignment]
            return out

    class Explicit:
        def __init__(self) -> None:
            self.n = 0
            self.tag = False

        def step(self, x: float) -> float:
            out = 0.0
            n = self.n + (2**53 + 1)
            out = float(n)
            self.n = 1
            self.tag = "s"  # type: ignore[assignment]
            return out

    for kernel in (Implicit().step, Explicit().step):
        error = _reject(kernel, "values of irreconcilable kinds merge here")
        assert "not exactly representable" not in str(error)
        assert error.location is not None and error.location.line is not None
        assert 'self.tag = "s"' in error.location.line


def test_state_obligation_survives_the_round_boundary() -> None:
    # The shift reads the leaf's live-in, so round two rejects it before re-reaching the violating store; the
    # obligation recorded in round one must survive the round reset, or the induced operator rejection reports.
    class Counter:
        def __init__(self) -> None:
            self.count = 0

        def step(self, value: float) -> float:
            out = float(self.count << 1)
            self.count = value  # type: ignore[assignment]
            return out

    error = _reject(Counter().step, "state attribute 'count' stores an incompatible type")
    assert error.location is not None and error.location.line is not None
    assert "self.count = value" in error.location.line


def test_stale_store_obligation_expires_when_the_cascade_cuts_the_stores_own_value() -> None:
    # Stranded-vs-stale doctrine (round 6, adjudication in round 7): a bridge origin whose store SURVIVES in a
    # live block but executed only with an Unbound value is STALE -- the deferral that cut it is the true
    # stable rejection, so the entry expires and the first deferral in executable preorder surfaces. This
    # REVERSES the round-4 expectation, which re-attached the bridged obligation and reported the store.
    class Counter:
        def __init__(self) -> None:
            self.count = 0

        def step(self, value: float) -> float:
            out = float(self.count << 1)
            self.count = out  # type: ignore[assignment]
            return value

    error = _reject(Counter().step, "bitwise/shift operator << requires integer operands")
    assert error.location is not None and error.location.line is not None
    assert "out = float(self.count << 1)" in error.location.line


def test_stale_store_obligation_expires_when_the_cascade_cuts_a_mixed_store_value() -> None:
    # Same stale expiry with the unbound value also feeding the return: the provoked may-be-unbound rejection
    # still defers to stabilization (the round never aborts mid-flight), and the shift deferral that cut the
    # store outranks it in executable preorder. Reversed from round 4 like the test above; round 7 adjudicates.
    class Counter:
        def __init__(self) -> None:
            self.count = 0

        def step(self, value: float) -> float:
            out = float(self.count << 1)
            self.count = value + out  # type: ignore[assignment]
            return out

    error = _reject(Counter().step, "bitwise/shift operator << requires integer operands")
    assert error.location is not None and error.location.line is not None
    assert "out = float(self.count << 1)" in error.location.line


def test_stranded_transient_exactness_verdict_reports_when_the_loop_head_dies() -> None:
    # Stranded-vs-stale doctrine (round 6, adjudication in round 7): the Implicit store violates in round one
    # and its block then falls behind the rejected loop head, leaving the executable graph entirely -- a
    # STRANDED origin, which reports its bridged message ahead of any deferred rejection (nothing downstream
    # can testify once the store's own cascade removed the block). This REVERSES the round-5 expectation,
    # which silently dropped the entry and let the range() deferral surface, and it splits the two spellings:
    # the Explicit float(...) cast never violates, so its round runs violation-free and the range() rejection
    # reports directly.
    class Implicit:
        def __init__(self) -> None:
            self.total = 0.0
            self.k = 0

        def step(self, x: float) -> float:
            acc = 0.0
            for _ in range(self.k):
                acc = acc + x
            self.total = self.k + (2**53 + 1)
            self.k = self.k + 1
            return acc

    error = _reject(Implicit().step, "not exactly representable")
    assert error.location is not None and error.location.line is not None
    assert "self.total = self.k + (2**53 + 1)" in error.location.line

    class Explicit:
        def __init__(self) -> None:
            self.total = 0.0
            self.k = 0

        def step(self, x: float) -> float:
            acc = 0.0
            for _ in range(self.k):
                acc = acc + x
            self.total = float(self.k + (2**53 + 1))
            self.k = self.k + 1
            return acc

    error = _reject(Explicit().step, "call to range with runtime arguments")
    assert "not exactly representable" not in str(error)
    assert error.location is not None and error.location.line is not None
    assert "for _ in range(self.k):" in error.location.line


def test_violating_unroll_clone_reports_despite_a_conforming_sibling() -> None:
    # Unroll clones of one store share an origin: trip 0 conforms while trip 1 violates, and the deferred float
    # shift leaves ``out`` unbound in round two. The conforming clone must not open a window in the deferral net
    # for the provoked may-be-unbound rejection; the violating clone's verdict reports at stabilization.
    class Ternary:
        def __init__(self) -> None:
            self.n = 0

        def step(self, x: float) -> float:
            out = 0.0
            for v in range(2):
                out = out + float(self.n << 1)
                self.n = x if v == 1 else 0  # type: ignore[assignment]
            return out

    class Subscript:
        def __init__(self) -> None:
            self.n = 0

        def step(self, x: float) -> float:
            out = 0.0
            for v in range(2):
                out = out + float(self.n << 1)
                self.n = (0, x)[v]  # type: ignore[assignment]
            return out

    for kernel, store_line in (
        (Ternary().step, "self.n = x if v == 1 else 0"),
        (Subscript().step, "self.n = (0, x)[v]"),
    ):
        error = _reject(kernel, "state attribute 'n' stores an incompatible type")
        assert error.location is not None and error.location.line is not None
        assert store_line in error.location.line


def test_the_surviving_violation_wins_when_a_stale_sibling_expires() -> None:
    # Stranded-vs-stale doctrine (round 6, adjudication in round 7): an obligation whose store survives in a
    # live block but executed only unbound is STALE and expires, so the violation still standing in the stable
    # graph reports instead, regardless of source order. This REVERSES the round-5 expectation, where the
    # bridge re-attached the cut store's obligation at its own preorder rank and the source-earlier line won.
    class StateFirst:
        def __init__(self) -> None:
            self.a = 0

        def step(self, x: float) -> float:
            out = float(self.a << 1)
            self.a = out  # type: ignore[assignment]
            b = 1
            b = x  # type: ignore[assignment]
            return b

    error = _reject(StateFirst().step, "variable 'b' is an int and cannot be rebound to a float")
    assert error.location is not None and error.location.line is not None
    assert "b = x" in error.location.line

    class LocalFirst:
        def __init__(self) -> None:
            self.k = 0

        def step(self, x: float) -> float:
            b = True
            b = self.k << 1  # type: ignore[assignment]
            self.k = x  # type: ignore[assignment]
            return x

    error = _reject(LocalFirst().step, "state attribute 'k' stores an incompatible type")
    assert error.location is not None and error.location.line is not None
    assert "self.k = x" in error.location.line


def test_cascade_unbound_does_not_violate_an_aggregate_slot() -> None:
    # The deferred shift leaves its consumer chain unbound; the Unbound reaching the aggregate slot must not
    # manufacture a scalar-store violation that outranks the causal store in preorder.
    class Vec:
        def __init__(self) -> None:
            self.n = 0
            self.v = [0.0, 0.0]

        def step(self, x: float) -> float:
            if x > 0.0:
                y = x << 1  # type: ignore[operator]
                self.v = [y, 0.0]
            else:
                self.n = x  # type: ignore[assignment]
            return x

    error = _reject(Vec().step, "state attribute 'n' stores an incompatible type")
    assert error.location is not None and error.location.line is not None
    assert "self.n = x" in error.location.line


def test_stale_store_entry_is_not_self_fulfilling() -> None:
    # Round-6 regression (Codex): the stale entry's own pendency defers the real stable rejection, keeps its
    # own store's value Unbound, and the old stabilization re-attach then restored the stale verdict -- the
    # implicit spelling reported "not exactly representable" where the stable truth is the range() rejection.
    # Under stranded-vs-stale, the surviving-but-unbound store's entry expires and both spellings agree.
    class Implicit:
        def __init__(self) -> None:
            self.total = 0.0
            self.k = 0

        def step(self, x: float) -> float:
            n = len(range(self.k))
            self.total = n + (2**53 + 1)
            self.k = self.k + 1
            return x

    class Explicit:
        def __init__(self) -> None:
            self.total = 0.0
            self.k = 0

        def step(self, x: float) -> float:
            n = len(range(self.k))
            self.total = float(n + (2**53 + 1))
            self.k = self.k + 1
            return x

    for kernel in (Implicit().step, Explicit().step):
        error = _reject(kernel, "call to range with runtime arguments")
        assert "not exactly representable" not in str(error)
        assert error.location is not None and error.location.line is not None
        assert "n = len(range(self.k))" in error.location.line


def test_stranded_store_obligation_reports_over_the_provoked_cascade() -> None:
    # Round-6 regression (Claude): the violating store's carried float poisons the live-in, the provoked &
    # rejection defers, the loop head dies on the unbound list cell, and the post-loop store block leaves the
    # executable graph. The old silent expiry then reported the innocent `t = self.a & 1` line; a STRANDED
    # origin must report its own store instead -- nothing in the graph can testify for it.
    class Meter:
        def __init__(self) -> None:
            self.a = 3

        def step(self, x: float) -> float:
            t = self.a & 1
            acc = 0.0
            for v in [t, 1]:
                acc = acc + float(v)
            self.a = x  # type: ignore[assignment]
            return acc

    error = _reject(Meter().step, "state attribute 'a' stores an incompatible type")
    assert error.location is not None and error.location.line is not None
    assert "self.a = x" in error.location.line


def test_conforming_clone_does_not_pop_the_shared_origin_on_incomplete_evidence() -> None:
    # Round-6 regression (Claude): trip 0 conforms bound while trip 1 executes with the cut `out`, recording
    # nothing, and the guarded p-store is discovered only in round two, forcing a third round. The old boundary
    # reconcile popped the shared origin on trip 0's conforming verdict alone, and round three ran with an open
    # deferral net, raising the innocent shift mid-round. An origin that also executed unbound is exempt from
    # the pop, so the obligation survives to stabilization and the stranded store reports.
    class Meter:
        def __init__(self) -> None:
            self.n = 0
            self.p = 0.0
            self.q = 0.0

        def step(self, x: float) -> float:
            out = float(self.n << 1)
            t = (0.0, x)[int(self.p)]
            for v in [t, 1]:
                self.n = out if v == 1 else 0  # type: ignore[assignment]
            if self.q > 0.0:
                self.p = x
            self.q = x
            return x

    error = _reject(Meter().step, "state attribute 'n' stores an incompatible type")
    assert error.location is not None and error.location.line is not None
    assert "self.n = out if v == 1 else 0" in error.location.line


def test_unroll_restart_carries_the_bridge_unchanged() -> None:
    # Round-6 regression (Codex): an _UnrollRestart is a mid-round event, so the partial round's verdicts are
    # no evidence -- the old restart path reconciled them into the bridge, whose leaked pendency deferred the
    # loop-head rejection and let the stranded transient exactness verdicts of the first-arm unroll report.
    # With the bridge carried through unchanged, the reseeded round is violation-free and the loop-head
    # rejection surfaces directly.
    class Meter:
        def __init__(self) -> None:
            self.total = 0.0

        def step(self, x: float) -> float:
            if x > 0.0:
                m = 2
            else:
                m = 3
            for v in range(m):
                self.total = v + (2**53 + 1)
            return x

    error = _reject(Meter().step, "loop trip count is not static here")
    assert "not exactly representable" not in str(error)
    assert error.location is not None and error.location.line is not None
    assert "for v in range(m):" in error.location.line


def test_a_conforming_visit_of_a_cut_store_op_does_not_settle_its_origin() -> None:
    # Round-6 regression (Codex): one CFG store op executes bound-and-conforming through the else arm and
    # unbound through the then arm's join, so the retained conforming verdict used to pop the origin at the
    # third-round boundary and the reopened net raised the else-arm shift first (worklist encounter order).
    # With the executed-unbound exemption the net holds, and stabilization surfaces the first deferral in
    # executable preorder: the then arm.
    class Meter:
        def __init__(self) -> None:
            self.count = 0
            self.p = 0.0
            self.q = 0.0

        def step(self, value: float) -> float:
            if value > 0.0:
                b = float(self.count << 2)
                t = b
            else:
                b = float(self.count << 1)
                t = 0
            self.count = t  # type: ignore[assignment]
            if self.q > 0.0:
                self.p = value
            self.q = value
            return value

    error = _reject(Meter().step, "runtime operation reads an aggregate or unbound value")
    assert error.location is not None and error.location.line is not None
    assert "b = float(self.count << 2)" in error.location.line


def test_stranded_clone_messages_fold_earliest_first() -> None:
    # Round-6 regression (Codex): unroll clones of one store share an origin, and when several violate with
    # different messages the bridge entry must keep the earliest-recorded one (trip order), not the last
    # overwrite -- under last-overwrite the trip-1 float message displaced the trip-0 aggregate message once
    # the origin stranded.
    class Meter:
        def __init__(self) -> None:
            self.n = 0

        def step(self, x: float) -> float:
            t = self.n & 1
            acc = 0.0
            for v in [[0.0, 1.0], t + x]:
                self.n = v  # type: ignore[assignment]
                acc = acc + 1.0
            return acc

    error = _reject(Meter().step, "state attribute 'n' persists a scalar; an aggregate cannot be stored into it")
    assert error.location is not None and error.location.line is not None
    assert "self.n = v" in error.location.line
