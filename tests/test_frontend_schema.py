"""
Frontend tests: the fixed storage schema (B1). A variable is strongly typed for the function's lifetime: its
first definition establishes the schema (independent first definitions on different paths join, int promoting
to float), and once established a store may only keep the kind -- bool accepts bool, int accepts int, float
accepts float or int (the integer converts on the store edge). Every other store is a located rejection at the
store site. The schema sees SemType kinds only: aggregate-valued stores to locals are fact-only (a reshape or
reflavor is a representation change, not a type change), as are references, strings, and ranges; ``del`` does
not erase a schema. Persistent state slots instead take their full schema -- flavor, geometry, per-cell kinds
-- from the reset value.
"""

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
