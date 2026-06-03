"""Unit tests for the Python-to-HIR frontend."""

import dataclasses
import math
import sys
from pathlib import Path

import pytest

from holoso import MissingIntrinsic, UnsupportedConstruct
from holoso._frontend import lower
from holoso._frontend._lower import _port_name
from holoso._hir import FloatAbs, FloatAdd, FloatDiv, FloatMul, FloatNeg, Operation, StateRead, optimize

from ._modelref import flatten_value, output_names


def _arith_count(hir, op_type):  # type: ignore[no-untyped-def]
    return sum(1 for n in hir.nodes.values() if isinstance(n, Operation) and type(n.operator) is op_type)


def test_scalar_is_output_zero() -> None:
    assert output_names(3.14) == ["out_0"]


def test_flat_sequence_is_positional() -> None:
    assert output_names((1.0, 2.0, 3.0)) == ["out_0", "out_1", "out_2"]


def test_nested_list_row_major_like_ekf1_stateless() -> None:
    # ekf1_stateless's update_x_P returns a 9x1 nested list -> out_0_0 .. out_8_0
    matrix = [[float(i)] for i in range(9)]
    assert output_names(matrix) == [f"out_{i}_0" for i in range(9)]


def test_matrix_n_by_m() -> None:
    matrix = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    assert output_names(matrix) == ["out_0_0", "out_0_1", "out_0_2", "out_1_0", "out_1_1", "out_1_2"]


def test_dataclass_fields_and_nesting() -> None:
    @dataclasses.dataclass
    class Foo:
        bar: float

    @dataclasses.dataclass
    class Baz:
        foo: Foo

    assert output_names((Baz(Foo(1.0)), 2.0)) == ["out_0_foo_bar", "out_1"]


def test_bare_dataclass_uses_field_names() -> None:
    @dataclasses.dataclass
    class Out:
        x: float
        y: float

    assert output_names(Out(1.0, 2.0)) == ["out_x", "out_y"]


def test_port_name_paths() -> None:
    assert _port_name([0]) == "out_0"
    assert _port_name([0, "foo", "bar"]) == "out_0_foo_bar"
    assert _port_name([3, 1]) == "out_3_1"


def test_flatten_value_returns_leaves() -> None:
    leaves = flatten_value([[1.5], [2.5]])
    assert [value for _, value in leaves] == [1.5, 2.5]


def test_small_kernel_inputs_outputs_and_ops() -> None:
    def kernel(a, b):  # type: ignore[no-untyped-def]
        return (a - b) * 0.25 + a * b

    hir = lower(kernel)
    assert hir.input_names() == ["a", "b"]
    assert [o.name for o in hir.outputs] == ["out_0"]
    assert _arith_count(hir, FloatMul) == 2  # (a-b)*0.25 and a*b
    assert _arith_count(hir, FloatAdd) == 2  # subtraction (add+neg) and the final add
    assert _arith_count(hir, FloatNeg) == 1  # the negation introduced by subtraction


def test_pow_expands_to_multiply_chain() -> None:
    def cube(a):  # type: ignore[no-untyped-def]
        return a**3

    hir = lower(cube)
    assert _arith_count(hir, FloatMul) == 2  # a*a*a


def test_abs_lowers_to_semantic_operation() -> None:
    def f(a):  # type: ignore[no-untyped-def]
        return abs(a)

    hir = lower(f)
    abs_ops = [n for n in hir.nodes.values() if isinstance(n, Operation) and type(n.operator) is FloatAbs]
    assert len(abs_ops) == 1


def test_division_lowers_to_div() -> None:
    def f(a, b):  # type: ignore[no-untyped-def]
        return a / b

    hir = lower(f)
    assert _arith_count(hir, FloatDiv) == 1
    divs = [n for n in hir.nodes.values() if isinstance(n, Operation) and type(n.operator) is FloatDiv]
    assert len(divs) == 1


def test_ekf1_stateless_structure() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    import ekf1_stateless

    hir = lower(ekf1_stateless.update_x_P)
    assert len(hir.input_ids) == 17
    assert [o.name for o in hir.outputs] == [f"out_{i}_0" for i in range(9)]
    assert _arith_count(hir, FloatDiv) == 1  # only x22 = 1 / x21


def test_for_loop_is_unsupported() -> None:
    def f(a):  # type: ignore[no-untyped-def]
        x = a
        for _ in range(3):
            x = x + a
        return x

    with pytest.raises(UnsupportedConstruct):
        lower(f)


def test_unknown_global_is_unsupported() -> None:
    def f(a):  # type: ignore[no-untyped-def]
        return a + UNDEFINED_GLOBAL  # type: ignore[name-defined]  # noqa: F821

    with pytest.raises(UnsupportedConstruct):
        lower(f)


def test_missing_intrinsic_message() -> None:
    def f(a):  # type: ignore[no-untyped-def]
        return math.sqrt(a)

    with pytest.raises(MissingIntrinsic, match="sqrt"):
        lower(f)


def _integrator_class():  # type: ignore[no-untyped-def]
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    from trapezoidal_leaky_streaming_integrator import TrapezoidalLeakyStreamingIntegrator

    return TrapezoidalLeakyStreamingIntegrator


def test_stateful_method_state_slots_and_dedup() -> None:
    integrator = _integrator_class()(k=2**-22)
    hir = lower(integrator.__call__)
    assert hir.input_names() == ["x"]  # self is dropped; remaining parameters become inputs
    # `return self.y` is deduped onto the public state port state_y; the private _x_prev gets no port, so the output
    # list alone distinguishes public from private. Both slots reset to 0.
    assert [o.name for o in hir.outputs] == ["state_y"]
    slots = {s.name: s for s in hir.state_slots}
    assert set(slots) == {"y", "_x_prev"}
    assert slots["y"].reset_value == 0.0 and slots["_x_prev"].reset_value == 0.0
    assert {n.slot for n in hir.nodes.values() if isinstance(n, StateRead)} == {"y", "_x_prev"}


def test_returned_public_state_alias_is_deduped() -> None:
    # The dedup is by dataflow, not spelling: returning a public attribute through an alias must still collapse onto its
    # state_<attr> port rather than emitting a second positional output for the same value.
    class Aliased:
        def __init__(self) -> None:
            self.y = 0.0

        def __call__(self, x):  # type: ignore[no-untyped-def]
            self.y = x
            y = self.y
            return y

    hir = lower(Aliased().__call__)
    assert [o.name for o in hir.outputs] == ["state_y"]


def test_mixed_return_dedupes_public_alias_keeps_distinct_leaf() -> None:
    class Mixed:
        def __init__(self) -> None:
            self.y = 0.0

        def __call__(self, x):  # type: ignore[no-untyped-def]
            self.y = x * 2.0
            a = self.y
            return (a, x)  # a aliases public self.y (deduped to state_y); x is distinct (keeps its positional out_1)

    hir = lower(Mixed().__call__)
    assert [o.name for o in hir.outputs] == ["out_1", "state_y"]


def test_return_value_equal_to_public_state_is_deduped_even_without_aliasing() -> None:
    # Dedup keys on the value, not provenance: returning x while x is also a public slot's live-out collapses onto that
    # slot's port even though the return never names the attribute. This is safe -- state_last carries the very same
    # wire, so the value stays observable; a separate out_0 would only duplicate it.
    class Passthrough:
        def __init__(self) -> None:
            self.last = 0.0

        def __call__(self, x):  # type: ignore[no-untyped-def]
            self.last = x
            return x

    hir = lower(Passthrough().__call__)
    assert [o.name for o in hir.outputs] == ["state_last"]


def test_unreachable_state_write_is_ignored() -> None:
    # A state write after the return is unreachable and never lowered; collecting it must not be attempted (it used to
    # crash with a KeyError). The method synthesizes as if the dead line were not there.
    class Dead:
        def __init__(self) -> None:
            self.y = 0.0

        def __call__(self, x):  # type: ignore[no-untyped-def]
            return x
            self.y = x  # unreachable

    hir = lower(Dead().__call__)
    assert [o.name for o in hir.outputs] == ["out_0"]
    assert hir.state_slots == []


def test_attribute_written_only_in_dead_code_reads_as_constant() -> None:
    # An attribute whose only assignment is unreachable is not state: a reachable read of it folds to its snapshot
    # constant, so it gets no slot and no out_<attr> port (whether it is state depends on its write being reachable).
    class Stale:
        def __init__(self) -> None:
            self.y = 5.0

        def __call__(self, x):  # type: ignore[no-untyped-def]
            r = x + self.y  # y folds to its snapshot 5.0 -- its only write is dead
            return r
            self.y = x  # unreachable

    hir = lower(Stale().__call__)
    assert hir.state_slots == []
    assert [o.name for o in hir.outputs] == ["out_0"]
    assert all(not (isinstance(n, StateRead) and n.slot == "y") for n in hir.nodes.values())


def test_stateful_readonly_attribute_is_folded_constant() -> None:
    integrator = _integrator_class()(k=2**-22)
    hir = optimize(lower(integrator.__call__))
    # k is only read, so it is a folded constant, not a persistent slot or a state read.
    assert "k" not in {s.name for s in hir.state_slots}
    assert all(not (isinstance(n, StateRead) and n.slot == "k") for n in hir.nodes.values())


def test_stateful_reset_state_is_the_instance_snapshot() -> None:
    # The reset value is whatever the instance holds at synthesis time, including post-construction mutation.
    integrator = _integrator_class()(k=2**-22)
    integrator.y = 1.5  # type: ignore[attr-defined]
    slots = {s.name: s for s in lower(integrator.__call__).state_slots}
    assert slots["y"].reset_value == 1.5


def test_init_method_target_is_rejected() -> None:
    integrator = _integrator_class()(k=2**-22)
    with pytest.raises(UnsupportedConstruct, match="__init__"):
        lower(integrator.__init__)


def test_class_object_target_is_rejected() -> None:
    with pytest.raises(UnsupportedConstruct, match="bound method"):
        lower(_integrator_class())


def test_method_without_return_exposes_public_state() -> None:
    class Accumulator:
        def __init__(self) -> None:
            self.total = 0.0

        def update(self, x: float) -> None:
            self.total = self.total + x

    hir = lower(Accumulator().update)
    assert [o.name for o in hir.outputs] == ["state_total"]
    assert {s.name for s in hir.state_slots} == {"total"}


def test_assigning_uninitialized_attribute_is_rejected() -> None:
    class Bad:
        def __init__(self) -> None:
            self.y = 0.0

        def __call__(self, x: float) -> float:
            self.scratch = x  # never initialized on the instance
            return self.y

    with pytest.raises(UnsupportedConstruct, match="not initialized"):
        lower(Bad().__call__)


def test_nested_attribute_access_is_rejected() -> None:
    class Bad:
        def __init__(self) -> None:
            self.y = 0.0

        def __call__(self, x: float) -> float:
            return x + self.y.real  # nested attribute access on self.y

    with pytest.raises(UnsupportedConstruct, match="direct self"):
        lower(Bad().__call__)
