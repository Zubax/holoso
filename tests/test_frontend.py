"""Unit tests for the Python-to-HIR frontend."""

import dataclasses
import math
import sys
import types
from pathlib import Path

import numpy as np
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


# --- Compile-time aggregates -----------------------------------------------------------------------------------------


def test_tuple_build_and_index() -> None:
    def f(a, b):  # type: ignore[no-untyped-def]
        z = a, b
        return [z[1], z[0]]  # swapped

    assert [o.name for o in lower(f).outputs] == ["out_0", "out_1"]


def test_list_slice() -> None:
    def f(a, b, c):  # type: ignore[no-untyped-def]
        v = [a, b, c]
        return v[1:3]

    assert [o.name for o in lower(f).outputs] == ["out_0", "out_1"]


def test_vector_scalar_broadcast() -> None:
    def f(a, b):  # type: ignore[no-untyped-def]
        v = [a, b]
        return v * 0.5  # elementwise: one multiply per leaf

    hir = lower(f)
    assert _arith_count(hir, FloatMul) == 2
    assert [o.name for o in hir.outputs] == ["out_0", "out_1"]


def test_flatten_collapses_nesting() -> None:
    def f(a, b):  # type: ignore[no-untyped-def]
        m = [[a], [b]]
        return m.flatten()

    assert [o.name for o in lower(f).outputs] == ["out_0", "out_1"]


def test_index_out_of_range_is_rejected() -> None:
    def f(a):  # type: ignore[no-untyped-def]
        v = [a]
        return v[3]

    with pytest.raises(UnsupportedConstruct, match="out of range"):
        lower(f)


def test_indexing_a_scalar_is_rejected() -> None:
    def f(a):  # type: ignore[no-untyped-def]
        return a[0]

    with pytest.raises(UnsupportedConstruct, match="index or slice a scalar"):
        lower(f)


def test_star_unpacking_a_scalar_is_rejected() -> None:
    def f(a):  # type: ignore[no-untyped-def]
        return [*a]

    with pytest.raises(UnsupportedConstruct, match="unpack"):
        lower(f)


# --- Tuple-unpacking assignment --------------------------------------------------------------------------------------


def test_tuple_unpacking_routes_values() -> None:
    # The right-hand side is built once before any binding, so a swap reads both sources first (no clobber).
    def swap(a, b):  # type: ignore[no-untyped-def]
        x, y = b, a
        return [x, y]

    hir = lower(swap)
    assert hir.input_names() == ["a", "b"]
    assert [o.value for o in hir.outputs] == [hir.input_ids[1], hir.input_ids[0]]


def test_starred_and_nested_unpacking_route_values() -> None:
    def f(a, b, c):  # type: ignore[no-untyped-def]
        first, *rest = [a, b, c]  # rest binds the surplus as an aggregate
        r0, r1 = rest  # nested unpacking of that aggregate
        return [first, r0, r1]

    hir = lower(f)
    assert [o.value for o in hir.outputs] == list(hir.input_ids)


def test_chained_assignment_binds_every_target() -> None:
    def f(a):  # type: ignore[no-untyped-def]
        x = y = a + a
        return [x, y]

    hir = lower(f)
    out = [o.value for o in hir.outputs]
    assert out[0] == out[1]  # both targets name the same single value


def test_unpacking_a_scalar_source_is_rejected() -> None:
    def f(a):  # type: ignore[no-untyped-def]
        x, y = a
        return x + y

    with pytest.raises(UnsupportedConstruct, match="unpack a scalar"):
        lower(f)


def test_unpacking_arity_mismatch_is_rejected() -> None:
    def f(a, b, c):  # type: ignore[no-untyped-def]
        x, y = [a, b, c]
        return x + y

    with pytest.raises(UnsupportedConstruct, match="unpack 3 values into 2"):
        lower(f)


def test_stateful_tuple_assignment_to_attributes() -> None:
    # Unpacking into self attributes must register both as persistent state; the swap reads the live-ins first.
    class Rotate:
        def __init__(self) -> None:
            self.x = 1.0
            self.y = 2.0

        def step(self, k):  # type: ignore[no-untyped-def]
            self.x, self.y = self.y, self.x + k
            return self.x

    hir = lower(Rotate().step)
    assert {s.name for s in hir.state_slots} == {"x", "y"}
    assert "state_x" in {o.name for o in hir.outputs}


def test_unpacked_name_shadows_global_callable() -> None:
    # A name bound only via tuple unpacking is local, so a same-named global function is not inlined at a call site;
    # this exercises _collect_local_names descending into unpacking targets.
    def f(a):  # type: ignore[no-untyped-def]
        _addmul, b = a, a  # _addmul is now a local value (Python would raise 'float not callable' when called)
        return _addmul(b)

    with pytest.raises(UnsupportedConstruct, match="not a callable"):
        lower(f)


# --- Importing and inlining a pure function --------------------------------------------------------------------------


def _addmul(p, q):  # type: ignore[no-untyped-def]
    return [p + q, p * q]


def test_inlined_global_function() -> None:
    def f(a, b):  # type: ignore[no-untyped-def]
        return _addmul(a, b)

    hir = lower(f)
    assert [o.name for o in hir.outputs] == ["out_0", "out_1"]
    assert _arith_count(hir, FloatAdd) == 1 and _arith_count(hir, FloatMul) == 1


def test_inlined_global_with_star_args() -> None:
    def f(a, b):  # type: ignore[no-untyped-def]
        v = [a, b]
        return _addmul(*v)

    hir = lower(f)
    assert _arith_count(hir, FloatAdd) == 1 and _arith_count(hir, FloatMul) == 1


def test_inline_arity_mismatch_is_rejected() -> None:
    def f(a):  # type: ignore[no-untyped-def]
        return _addmul(a)  # _addmul takes two positional arguments

    with pytest.raises(UnsupportedConstruct, match="positional arguments"):
        lower(f)


def cbrt(x):  # type: ignore[no-untyped-def]
    return x * x  # a user-defined global whose name collides with the same-named intrinsic placeholder


def test_user_global_function_shadows_intrinsic_name() -> None:
    # A module-level def named like an intrinsic is the caller's own function; Python would call it, so it is inlined.
    def f(a):  # type: ignore[no-untyped-def]
        return cbrt(a)

    assert _arith_count(lower(f), FloatMul) == 1  # the inlined x * x, not a MissingIntrinsic rejection


def test_local_name_shadows_global_callable() -> None:
    # A parameter named like a global function refers to the parameter (a value), which is not callable.
    def f(_addmul, a):  # type: ignore[no-untyped-def]
        return _addmul(a)

    with pytest.raises(UnsupportedConstruct, match="not a callable"):
        lower(f)


def test_flatten_on_a_scalar_is_rejected() -> None:
    def f(a):  # type: ignore[no-untyped-def]
        return a.flatten()

    with pytest.raises(UnsupportedConstruct, match="aggregate"):
        lower(f)


def test_negative_boolean_literal_is_rejected() -> None:
    def f():  # type: ignore[no-untyped-def]
        return -True

    with pytest.raises(UnsupportedConstruct, match="boolean"):
        lower(f)


def test_abs_accepts_a_star_unpacked_argument() -> None:
    # Call-argument unpacking applies uniformly: abs(*v) on a one-element aggregate is abs of that single element.
    def f(a):  # type: ignore[no-untyped-def]
        v = [a]
        return abs(*v)

    assert _arith_count(lower(f), FloatAbs) == 1


def test_unary_plus_is_scalar_identity_and_rejects_aggregates() -> None:
    def scalar_ok(a):  # type: ignore[no-untyped-def]
        return +a  # identity on a scalar

    assert [o.name for o in lower(scalar_ok).outputs] == ["out_0"]

    def aggregate_bad(a, b):  # type: ignore[no-untyped-def]
        v = [a, b]
        return +v  # unary plus does not apply to an aggregate

    with pytest.raises(UnsupportedConstruct, match="scalar"):
        lower(aggregate_bad)


def test_method_style_abs_call_is_rejected() -> None:
    # Only a bare-name abs(...) is the builtin; a method-style a.abs(b) must not be silently treated as it (which would
    # drop the receiver) -- there is no supported scalar method, so it is an unsupported call.
    def f(a, b):  # type: ignore[no-untyped-def]
        return a.abs(b)

    with pytest.raises(UnsupportedConstruct, match="abs"):
        lower(f)


def _rebind_globals(fn, **overrides):  # type: ignore[no-untyped-def]
    """A copy of ``fn`` whose module globals carry ``overrides`` (its source stays retrievable via the shared code)."""
    return types.FunctionType(
        fn.__code__, {**fn.__globals__, **overrides}, fn.__name__, fn.__defaults__, fn.__closure__
    )


def test_noncallable_global_shadowing_builtin_is_rejected() -> None:
    # A non-callable global shadows the built-in (Python raises TypeError on the call), so the name is not the builtin
    # it spells; holoso must reject rather than silently emitting FloatAbs / the list-tuple identity.
    def use_abs(a):  # type: ignore[no-untyped-def]
        return abs(a)

    def use_list(a):  # type: ignore[no-untyped-def]
        return list((a, a))

    def use_tuple(a):  # type: ignore[no-untyped-def]
        return tuple((a, a))

    for fn, shadow in ((use_abs, {"abs": 5}), (use_list, {"list": 5}), (use_tuple, {"tuple": 5})):
        with pytest.raises(UnsupportedConstruct, match="non-callable"):
            lower(_rebind_globals(fn, **shadow))


def test_callable_global_shadowing_abs_is_inlined_not_floatabs() -> None:
    # A callable global named ``abs`` is the caller's own function; Python would call it, so holoso inlines it instead
    # of emitting the FloatAbs builtin -- the non-callable guard must not disturb this legitimate shadow.
    def use_abs(a):  # type: ignore[no-untyped-def]
        return abs(a)

    hir = lower(_rebind_globals(use_abs, abs=cbrt))  # cbrt is a module-level def returning x * x
    assert _arith_count(hir, FloatAbs) == 0 and _arith_count(hir, FloatMul) == 1


# --- numpy-array aggregates and executable-numpy interop --------------------------------------------------------------


def test_numpy_array_state_decomposes_like_a_list() -> None:
    import numpy.typing as npt

    @dataclasses.dataclass
    class Filt:
        v: npt.NDArray[np.float64]  # shape-less annotation: holoso infers the length from the reset value

        def step(self, a):  # type: ignore[no-untyped-def]
            self.v = self.v * a

    hir = lower(Filt(np.array([1.0, 2.0, 3.0])).step)
    assert {s.name for s in hir.state_slots} == {"v_0", "v_1", "v_2"}
    assert [o.name for o in hir.outputs] == ["state_v_0", "state_v_1", "state_v_2"]


def test_jaxtyping_array_field_lowers_and_is_validated() -> None:
    from jaxtyping import Float64

    @dataclasses.dataclass
    class Filt:
        v: Float64[np.ndarray, "3"]

        def step(self, a):  # type: ignore[no-untyped-def]
            self.v = self.v * a

    assert {s.name for s in lower(Filt(np.array([1.0, 2.0, 3.0])).step).state_slots} == {"v_0", "v_1", "v_2"}
    with pytest.raises(UnsupportedConstruct, match="declared array type"):
        lower(Filt(np.array([1.0, 2.0, 3.0, 4.0])).step)  # value shape (4,) violates the declared "3"


def test_numpy_integer_array_values_coerce_to_real() -> None:
    @dataclasses.dataclass
    class Filt:
        v: np.ndarray  # type: ignore[type-arg]

        def step(self, a):  # type: ignore[no-untyped-def]
            self.v = self.v * a

    assert {s.name for s in lower(Filt(np.array([2, 3])).step).state_slots} == {"v_0", "v_1"}


def test_numpy_asarray_is_identity_on_an_aggregate() -> None:
    def f(a, b):  # type: ignore[no-untyped-def]
        return np.asarray([a, b]).flatten()  # asarray of an array-like is identity in this compile-time model

    assert [o.name for o in lower(f).outputs] == ["out_0", "out_1"]


def test_list_is_identity_on_an_aggregate() -> None:
    def f(a, b, c):  # type: ignore[no-untyped-def]
        v = [a, b, c]
        return list(v[0:2])  # list() of a slice carries the same elements -- identity here

    assert [o.name for o in lower(f).outputs] == ["out_0", "out_1"]


def test_list_of_a_scalar_is_rejected() -> None:
    def f(a):  # type: ignore[no-untyped-def]
        return list(a)  # Python: list(scalar) is a TypeError -- a scalar is not iterable

    with pytest.raises(UnsupportedConstruct, match="list"):
        lower(f)


def test_tuple_is_identity_on_an_aggregate() -> None:
    def f(a, b, c):  # type: ignore[no-untyped-def]
        v = [a, b, c]
        return tuple(v[0:2])  # tuple() of a slice is identity here, co-equal with list()

    assert [o.name for o in lower(f).outputs] == ["out_0", "out_1"]


def test_numpy_alias_shadowed_by_a_local_is_not_numpy() -> None:
    # ``np`` is rebound to a local value, so ``np.asarray`` is a method call on that value, not the numpy function.
    def f(a):  # type: ignore[no-untyped-def]
        np = [a]
        return np.asarray([a])

    with pytest.raises(UnsupportedConstruct, match="asarray"):
        lower(f)


def test_name_assigned_later_is_local_before_its_assignment() -> None:
    # A name assigned anywhere in a function is local throughout (Python's rule); using it as a global/builtin/numpy
    # before that assignment is invalid Python (UnboundLocalError), so holoso rejects it rather than seeing the global.
    def shadows_numpy(a):  # type: ignore[no-untyped-def]
        y = np.asarray([a])
        np = [a]  # noqa: F841  # makes np local for the whole body
        return y

    with pytest.raises(UnsupportedConstruct):
        lower(shadows_numpy)

    def shadows_builtin(a):  # type: ignore[no-untyped-def]
        y = abs(a)
        abs = [a]  # noqa: F841  # makes abs local for the whole body
        return y

    with pytest.raises(UnsupportedConstruct, match="local name"):
        lower(shadows_builtin)


def test_multidimensional_array_state_is_rejected() -> None:
    @dataclasses.dataclass
    class Filt:
        m: np.ndarray  # type: ignore[type-arg]

        def step(self, a):  # type: ignore[no-untyped-def]
            self.m = self.m * a

    with pytest.raises(UnsupportedConstruct, match="1-D"):
        lower(Filt(np.array([[1.0, 2.0], [3.0, 4.0]])).step)


def test_ekf1_stateful_structure() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    import ekf1_stateful

    filt = ekf1_stateful.Ekf1(
        x=[0.0, 0.0, 0.0], P_urt=[1.0, 0.0, 0.0, 1.0, 0.0, 1.0], R_diag=[1.0, 1.0], Q_diag=np.array([1.0, 1.0, 1.0])
    )
    hir = lower(filt.update)
    assert hir.input_names() == ["dt", "u_shunt", "di_dt"]  # self dropped; keyword-only params become inputs
    assert [o.name for o in hir.outputs] == ["state_x_0", "state_x_1", "state_x_2"] + [
        f"state_P_urt_{i}" for i in range(6)
    ]
    assert {s.name for s in hir.state_slots} == {f"x_{i}" for i in range(3)} | {f"P_urt_{i}" for i in range(6)}
    assert _arith_count(hir, FloatDiv) == 1  # the inlined kernel's single 1/x21


# --- Vector-valued state and keyword-only inputs ---------------------------------------------------------------------


def test_vector_state_decomposes_to_per_element_slots() -> None:
    class Vec:
        def __init__(self) -> None:
            self.v = [1.0, 2.0, 3.0]

        def update(self, a):  # type: ignore[no-untyped-def]
            self.v = [self.v[0] + a, self.v[1], self.v[2]]

    hir = lower(Vec().update)
    assert {s.name: s.reset_value for s in hir.state_slots} == {"v_0": 1.0, "v_1": 2.0, "v_2": 3.0}
    assert [o.name for o in hir.outputs] == ["state_v_0", "state_v_1", "state_v_2"]


def test_vector_state_shape_mismatch_is_rejected() -> None:
    class Vec:
        def __init__(self) -> None:
            self.v = [0.0, 0.0]

        def update(self, a):  # type: ignore[no-untyped-def]
            self.v = [a]  # the slot holds two scalars, but one is assigned

    with pytest.raises(UnsupportedConstruct, match="2-element vector"):
        lower(Vec().update)


def test_vector_state_nested_shape_is_rejected() -> None:
    # A nested aggregate has the right leaf count (2) but the wrong shape: the slot layout is a flat 2-vector, so the
    # next transaction would reconstruct a flat shape that disagrees with the one written this transaction.
    class Vec:
        def __init__(self) -> None:
            self.v = [0.0, 0.0]

        def update(self, a, b):  # type: ignore[no-untyped-def]
            self.v = [[a, b]]

    with pytest.raises(UnsupportedConstruct, match="incompatible shape"):
        lower(Vec().update)


def test_vector_state_slot_name_collision_is_rejected() -> None:
    # The vector ``v`` decomposes into slot ``v_0``, which would alias the distinct scalar attribute ``v_0``.
    class Vec:
        def __init__(self) -> None:
            self.v = [1.0]
            self.v_0 = 2.0

        def update(self, a):  # type: ignore[no-untyped-def]
            self.v = [a]
            self.v_0 = a + 1.0

    with pytest.raises(UnsupportedConstruct, match="aliasing collision"):
        lower(Vec().update)


def test_keyword_only_params_become_inputs() -> None:
    def f(a, *, b, c):  # type: ignore[no-untyped-def]
        return a + b + c

    assert lower(f).input_names() == ["a", "b", "c"]


def test_dataclass_instance_is_stateful() -> None:
    @dataclasses.dataclass
    class Acc:
        total: float
        gain: list  # type: ignore[type-arg]

        def step(self, x):  # type: ignore[no-untyped-def]
            self.total = self.total + x * self.gain[0]

    hir = lower(Acc(0.0, [2.0]).step)
    assert {s.name for s in hir.state_slots} == {"total"}  # gain is read-only config, not state
    assert [o.name for o in hir.outputs] == ["state_total"]
