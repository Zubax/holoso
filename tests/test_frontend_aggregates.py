"""Frontend tests: aggregates -- tuple/list/record/array construction, projection, slicing, unpacking, iteration."""

import dataclasses
import math
import types
from pathlib import Path

import numpy as np
import pytest

import holoso
from holoso import FloatFormat, UnsupportedConstruct
from holoso._frontend import lower
from holoso._hir import Branch, FloatAdd, FloatMul, FloatNeg, FloatRelational, optimize

from ._frontend_common import _assert_shape_kernel_matches_python as _assert_shape_kernel_matches_python
from ._modelref import arith_count as _arith_count, default_ops


def _lower_generated_kernel(tmp_path: Path, name: str, source: str):  # type: ignore[no-untyped-def]
    import importlib.util

    module_path = tmp_path / f"_{name}.py"
    module_path.write_text(source)
    spec = importlib.util.spec_from_file_location(f"_{name}", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_comprehension_with_too_many_generators_is_a_located_rejection(tmp_path: Path) -> None:
    # Regression (Codex): each generator adds one `_expand_comprehension` frame, so a comprehension with hundreds of
    # generators used to leak a bare RecursionError. It now rejects with a located error; CPython runs it normally.
    clauses = " ".join(f"for a{i} in [{float(i)!r}]" for i in range(200))
    module = _lower_generated_kernel(
        tmp_path, "many_generators", f"def kernel(x: float) -> float:\n    return [x {clauses}][0]\n"
    )
    assert module.kernel(6.5) == 6.5  # valid, runnable Python
    with pytest.raises(UnsupportedConstruct, match="comprehension nesting expands"):
        lower(module.kernel)


def test_deeply_nested_comprehensions_are_a_located_rejection(tmp_path: Path) -> None:
    # Regression (Codex): a single comprehension's generators are bounded, but nested comprehensions accumulate
    # expansion frames across levels, so deep nesting used to leak a bare RecursionError; it is now a located error.
    inner = "x"
    for depth in range(15):
        clauses = " ".join(f"for a{depth}_{i} in [0.0]" for i in range(64))
        inner = f"[{inner} {clauses}][0]"
    module = _lower_generated_kernel(
        tmp_path, "nested_comprehensions", f"def kernel(x: float) -> float:\n    return {inner}\n"
    )
    assert module.kernel(3.25) == 3.25  # valid, runnable Python (below CPython's own compiler limit)
    with pytest.raises(UnsupportedConstruct, match="comprehension nesting expands"):
        lower(module.kernel)


@pytest.mark.skip(reason="FIR_PARITY_PENDING: runtime aggregate value — stage 9")
def test_tuple_build_and_index() -> None:
    def f(a: float, b: float) -> list[float]:
        z = a, b
        return [z[1], z[0]]

    assert [o.name for o in lower(f).outputs] == ["out_0", "out_1"]


@pytest.mark.skip(reason="FIR_PARITY_PENDING: aggregate slicing — stage 9")
def test_list_slice() -> None:
    def f(a: float, b: float, c: float) -> list[float]:
        v = [a, b, c]
        return v[1:3]

    assert [o.name for o in lower(f).outputs] == ["out_0", "out_1"]


@pytest.mark.skip(reason="FIR_PARITY_PENDING: np.array/asarray of runtime values — stage 9")
def test_vector_scalar_broadcast() -> None:
    def f(a: float, b: float) -> list[float]:
        v = np.array([a, b])
        return v * 0.5  # type: ignore[return-value]

    hir = lower(f)
    assert _arith_count(hir, FloatMul) == 2
    assert [o.name for o in hir.outputs] == ["out_0", "out_1"]


@pytest.mark.skip(reason="FIR_PARITY_PENDING: np.array/asarray of runtime values — stage 9")
def test_flatten_collapses_nesting() -> None:
    def f(a: float, b: float) -> list[float]:
        m = np.array([[a], [b]])
        return m.flatten()  # type: ignore[return-value]

    assert [o.name for o in lower(f).outputs] == ["out_0", "out_1"]


def test_index_out_of_range_is_rejected() -> None:
    def f(a: float) -> float:
        v = [a]
        return v[3]

    with pytest.raises(UnsupportedConstruct, match="out of range"):
        lower(f)


@pytest.mark.skip(reason="FIR_PARITY_PENDING: runtime subscript/indexing — stage 9")
def test_indexing_a_scalar_is_rejected() -> None:
    def f(a: float) -> float:
        return a[0]  # type: ignore[no-any-return, index]

    with pytest.raises(UnsupportedConstruct, match="index or slice a scalar"):
        lower(f)


def test_star_unpacking_a_scalar_is_rejected() -> None:
    def f(a: float) -> float:
        return [*a]  # type: ignore[misc, return-value]

    with pytest.raises(UnsupportedConstruct, match="unpack"):
        lower(f)


def test_tuple_unpacking_routes_values() -> None:
    # The right-hand side is built once before any binding, so a swap reads both sources first (no clobber).
    def swap(a: float, b: float) -> list[float]:
        x, y = b, a
        return [x, y]

    hir = lower(swap)
    assert hir.input_names() == ["a", "b"]
    assert [o.value for o in hir.outputs] == [hir.input_ids[1], hir.input_ids[0]]


@pytest.mark.skip(reason="FIR_PARITY_PENDING: starred unpacking targets — stage 9")
def test_starred_and_nested_unpacking_route_values() -> None:
    def f(a: float, b: float, c: float) -> list[float]:
        first, *rest = [a, b, c]
        r0, r1 = rest
        return [first, r0, r1]

    hir = lower(f)
    assert [o.value for o in hir.outputs] == list(hir.input_ids)


def test_chained_assignment_binds_every_target() -> None:
    def f(a: float) -> list[float]:
        x = y = a + a
        return [x, y]

    hir = lower(f)
    out = [o.value for o in hir.outputs]
    assert out[0] == out[1]


def test_unpacking_a_scalar_source_is_rejected() -> None:
    def f(a: float) -> float:
        x, y = a  # type: ignore[misc]
        return x + y  # type: ignore[no-any-return, has-type]

    with pytest.raises(UnsupportedConstruct, match="length of a runtime value"):
        lower(f)


def test_unpacking_arity_mismatch_is_rejected() -> None:
    def f(a: float, b: float, c: float) -> float:
        x, y = [a, b, c]  # type: ignore[misc]
        return x + y  # type: ignore[no-any-return, has-type]

    with pytest.raises(UnsupportedConstruct, match="cannot unpack: expected 2 values"):
        lower(f)


@pytest.mark.skip(reason="FIR_PARITY_PENDING: shape query (.ndim/.shape/.T/.flatten) — stage 9")
def test_flatten_on_a_scalar_is_rejected() -> None:
    def f(a: float) -> float:
        return a.flatten()  # type: ignore[no-any-return, attr-defined]

    with pytest.raises(UnsupportedConstruct, match="aggregate"):
        lower(f)


@pytest.mark.skip(reason="FIR_PARITY_PENDING: np.array/asarray of runtime values — stage 9")
def test_unary_plus_and_minus_apply_elementwise_to_aggregates() -> None:
    def scalar_ok(a: float) -> float:
        return +a

    assert [o.name for o in lower(scalar_ok).outputs] == ["out_0"]

    def aggregate_ok(a: float, b: float) -> list[float]:
        v = np.array([a, b])
        return +v  # type: ignore[return-value]

    assert [o.name for o in lower(aggregate_ok).outputs] == ["out_0", "out_1"]

    def negated(a: float, b: float) -> list[float]:
        v = np.array([a, b])
        return -v  # type: ignore[return-value]

    hir = lower(negated)
    assert [o.name for o in hir.outputs] == ["out_0", "out_1"]
    assert _arith_count(hir, FloatNeg) == 2


@pytest.mark.skip(reason="FIR_PARITY_PENDING: np.array/asarray of runtime values — stage 9")
def test_numpy_asarray_is_identity_on_an_aggregate() -> None:
    def f(a: float, b: float) -> list[float]:
        return np.asarray([a, b]).flatten()  # type: ignore[return-value]  # identity in this compile-time model

    assert [o.name for o in lower(f).outputs] == ["out_0", "out_1"]


@pytest.mark.skip(reason="FIR_PARITY_PENDING: aggregate slicing — stage 9")
def test_list_is_identity_on_an_aggregate() -> None:
    def f(a: float, b: float, c: float) -> list[float]:
        v = [a, b, c]
        return list(v[0:2])

    assert [o.name for o in lower(f).outputs] == ["out_0", "out_1"]


def test_list_of_a_scalar_is_rejected() -> None:
    def f(a: float) -> float:
        return list(a)  # type: ignore[no-any-return, call-overload]  # list(scalar) is a Python TypeError

    with pytest.raises(UnsupportedConstruct, match="list"):
        lower(f)


@pytest.mark.skip(reason="FIR_PARITY_PENDING: aggregate slicing — stage 9")
def test_tuple_is_identity_on_an_aggregate() -> None:
    def f(a: float, b: float, c: float) -> list[float]:
        v = [a, b, c]
        return tuple(v[0:2])  # type: ignore[return-value]

    assert [o.name for o in lower(f).outputs] == ["out_0", "out_1"]


# ---------------------------------------------------------------- compile-time shape queries


@pytest.mark.skip(reason="FIR_PARITY_PENDING: runtime subscript/indexing — stage 9")
def test_static_shape_queries_in_index_range_and_branch_positions() -> None:
    from jaxtyping import Float64

    def kernel(v: Float64[np.ndarray, "3"], m: Float64[np.ndarray, "2 3"]) -> float:
        acc = v[0]
        for i in range(1, len(v)):  # len() bounds an unrolled range
            acc = acc + v[i]
        acc = acc + m[m.ndim - 2][m.shape[-1] - 3]  # .ndim and .shape[k] are compile-time integers
        if v.ndim == 1:  # a shape comparison folds, so only the taken arm is lowered
            acc = acc * 2.0
        return acc  # type: ignore[no-any-return]

    hir = lower(kernel)
    assert [o.name for o in hir.outputs] == ["out_0"]
    assert _arith_count(hir, FloatAdd) == 3  # v[0]+v[1]+v[2] then +m[0][0]; the ndim branch adds no add
    assert _arith_count(hir, FloatMul) == 1


@pytest.mark.skip(reason="FIR_PARITY_PENDING: len() of a runtime aggregate — stage 9")
def test_len_follows_python_and_accepts_a_ragged_list() -> None:
    # len() is a Python builtin, not a numpy one, so it counts the items of any aggregate; only .ndim/.shape are
    # numpy-only and rejected on a sequence.
    def ragged(a: float, b: float) -> float:
        rows = [[a, b], [a]]
        acc = 0.0
        for i in range(len(rows)):
            for j in range(len(rows[i])):
                acc = acc + rows[i][j]
        return acc

    assert _arith_count(lower(ragged), FloatAdd) == 3


@pytest.mark.skip(reason="FIR_PARITY_PENDING: shape query (.ndim/.shape/.T/.flatten) — stage 9")
def test_shape_queries_are_rejected_outside_a_static_position() -> None:
    from jaxtyping import Float64

    def ndim_as_value(v: Float64[np.ndarray, "3"]) -> float:
        return float(v.ndim)

    def shape_as_value(v: Float64[np.ndarray, "3"]) -> float:
        return float(v.shape[0])

    def len_as_value(v: Float64[np.ndarray, "3"]) -> float:
        return float(len(v))

    for kernel in (ndim_as_value, shape_as_value, len_as_value):
        with pytest.raises(UnsupportedConstruct, match="compile-time integer"):
            lower(kernel)

    def ndim_of_list(a: float, b: float) -> float:
        rows = [a, b]
        return a if rows.ndim == 1 else b  # type: ignore[attr-defined]

    with pytest.raises(UnsupportedConstruct, match="Python list/tuple"):
        lower(ndim_of_list)

    def len_of_scalar(a: float) -> float:
        acc = 0.0
        for _ in range(len(a)):  # type: ignore[arg-type]
            acc = acc + a
        return acc

    with pytest.raises(UnsupportedConstruct, match="len\\(\\) of a scalar"):
        lower(len_of_scalar)

    def bad_axis(v: Float64[np.ndarray, "3"]) -> float:
        return v[v.shape[2]]  # type: ignore[return-value]

    with pytest.raises(UnsupportedConstruct, match="axis 2 is out of range"):
        lower(bad_axis)


@pytest.mark.skip(reason="FIR_PARITY_PENDING: shape query (.ndim/.shape/.T/.flatten) — stage 9")
def test_numpy_only_shape_queries_are_rejected_on_a_sequence_however_it_is_spelled() -> None:
    # A shape query never lowers its receiver, so the list/tuple rejection has to walk the receiver expression itself.
    # Reaching a list through a subscript, a transpose, or a state attribute must not hand it .ndim/.shape/.T, none of
    # which Python gives a list -- otherwise Holoso would accept a kernel that is not runnable Python.
    def ndim_through_a_subscript(a: float, b: float) -> float:
        rows = [[a, b], [b, a]]
        return a if rows[0].ndim == 1 else b  # type: ignore[attr-defined]

    with pytest.raises(UnsupportedConstruct, match="ndim on a Python list/tuple"):
        lower(ndim_through_a_subscript)

    def transpose_of_a_sequence_in_a_static_position(a: float, b: float) -> float:
        rows = [a, b]
        acc = 0.0
        for i in range(len(rows.T)):  # type: ignore[attr-defined]
            acc = acc + rows[i]
        return acc

    with pytest.raises(UnsupportedConstruct, match="transpose on a Python list/tuple"):
        lower(transpose_of_a_sequence_in_a_static_position)

    class ListState:
        def __init__(self) -> None:
            self.vec = [1.0, 2.0]

        def __call__(self, a: float) -> float:
            return a if self.vec.ndim == 1 else -a  # type: ignore[attr-defined]

    with pytest.raises(UnsupportedConstruct, match="ndim on a Python list/tuple"):
        lower(ListState().__call__)


@pytest.mark.skip(reason="FIR_PARITY_PENDING: aggregate slicing — stage 9")
def test_empty_slice_has_no_shape_but_still_has_a_length() -> None:
    # An empty aggregate is not an array (array_shape says so), so the static shape probe must not report a zero-length
    # axis that lowering can never produce. Iteration and len() still follow Python, which give an empty slice a length.
    from jaxtyping import Float64

    def iterate_an_empty_slice(v: Float64[np.ndarray, "5"]) -> float:
        acc = v[0]
        for x in v[2:2]:
            acc = acc + x
        return acc  # type: ignore[no-any-return]

    assert iterate_an_empty_slice(np.arange(5.0)) == 0.0  # the kernel is runnable Python, and the loop is empty
    assert _arith_count(lower(iterate_an_empty_slice), FloatAdd) == 0

    def ndim_of_an_empty_slice(v: Float64[np.ndarray, "5"]) -> float:
        return v[0] if v[2:2].ndim == 1 else v[1]  # type: ignore[no-any-return]

    with pytest.raises(UnsupportedConstruct, match="rectangular"):
        lower(ndim_of_an_empty_slice)

    def matmul_of_an_empty_slice(v: Float64[np.ndarray, "5"], w: Float64[np.ndarray, "5"]) -> float:
        return v[2:2] @ w  # type: ignore[no-any-return]

    # The stub's own shape guard surfaces the same diagnostic through the operator, at the user's call site.
    with pytest.raises(UnsupportedConstruct, match="rectangular"):
        lower(matmul_of_an_empty_slice)


@pytest.mark.skip(reason="FIR_PARITY_PENDING: runtime subscript/indexing — stage 9")
def test_shape_queries_still_reach_arrays_through_the_same_spellings() -> None:
    # The complement of the rejection above: an array receiver keeps .ndim/.shape/.T through a subscript or a state
    # attribute, and len() keeps working on a list attribute, where Python does give it a length.
    class Mixed:
        def __init__(self) -> None:
            self.m = np.array([[1.0, 2.0], [3.0, 4.0]])
            self.v = [1.0, 2.0]

        def __call__(self, a: float) -> float:
            acc = a
            for i in range(len(self.v)):
                acc = acc + self.v[i]
            for i in range(self.m.ndim):
                acc = acc + self.m[i][i]
            return acc + self.m.T[0][1]  # type: ignore[no-any-return]

    assert [o.name for o in lower(Mixed().__call__).outputs] == ["out_0"]

    from jaxtyping import Float64

    def array_row_has_ndim(m: Float64[np.ndarray, "2 3"]) -> float:
        return m[0][0] if m[0].ndim == 1 else m[1][0]  # type: ignore[no-any-return]

    assert [o.name for o in lower(array_row_has_ndim).outputs] == ["out_0"]


# ---------------------------------------------------------------- comprehensions and aggregate iteration


@pytest.mark.skip(reason="FIR_PARITY_PENDING: len() of a runtime aggregate — stage 9")
def test_list_comprehension_unrolls_and_scopes_its_target() -> None:
    from jaxtyping import Float64

    def scaled(v: Float64[np.ndarray, "3"], s: float) -> Float64[np.ndarray, "3"]:
        return np.array([v[i] * s for i in range(len(v))])

    hir = lower(scaled)
    assert [o.name for o in hir.outputs] == ["out_0", "out_1", "out_2"]
    assert _arith_count(hir, FloatMul) == 3

    def nested(m: Float64[np.ndarray, "2 3"]) -> Float64[np.ndarray, "3 2"]:
        return np.array([[m[i][j] for i in range(2)] for j in range(3)])

    assert [o.name for o in lower(nested).outputs] == [f"out_{i}_{j}" for i in range(3) for j in range(2)]

    def filtered(m: Float64[np.ndarray, "3 3"]) -> Float64[np.ndarray, "3"]:
        return np.array([m[i][j] for i in range(3) for j in range(3) if i < j])

    assert [o.name for o in lower(filtered).outputs] == ["out_0", "out_1", "out_2"]

    def target_does_not_leak(v: Float64[np.ndarray, "3"]) -> float:
        rows = [v[k] for k in range(3)]
        return rows[0] + k  # type: ignore[name-defined, no-any-return]  # noqa: F821

    # Unlike a ``for`` counter, a comprehension target is confined to the comprehension, exactly as in Python.
    with pytest.raises(UnsupportedConstruct, match="unknown name 'k'"):
        lower(target_does_not_leak)


@pytest.mark.skip(reason="FIR_PARITY_PENDING: runtime subscript/indexing — stage 9")
def test_comprehension_yields_a_python_list_not_an_array() -> None:
    from jaxtyping import Float64

    def arithmetic_on_a_comprehension(v: Float64[np.ndarray, "2"]) -> float:
        rows = [v[i] for i in range(2)]
        return (rows * 2.0)[0]  # type: ignore[no-any-return, operator]

    # A comprehension is a Python list, so numpy-only operations need np.array(...) around it, as in Python.
    with pytest.raises(UnsupportedConstruct, match="Python list/tuple"):
        lower(arithmetic_on_a_comprehension)


@pytest.mark.skip(reason="FIR_PARITY_PENDING: runtime subscript/indexing — stage 9")
def test_comprehension_rejections() -> None:
    from jaxtyping import Float64

    def dynamic_filter(v: Float64[np.ndarray, "3"]) -> Float64[np.ndarray, "3"]:
        return np.array([v[i] for i in range(3) if v[i] > 0.0])

    with pytest.raises(UnsupportedConstruct, match="compile-time condition"):
        lower(dynamic_filter)

    def walrus_inside(v: Float64[np.ndarray, "2"]) -> Float64[np.ndarray, "2"]:
        return np.array([(t := v[i]) + t for i in range(2)])

    with pytest.raises(UnsupportedConstruct, match="walrus"):
        lower(walrus_inside)

    def tuple_target(v: Float64[np.ndarray, "2"]) -> float:
        pairs = [[v[0], v[1]]]
        return [a + b for a, b in pairs][0]  # type: ignore[no-any-return]

    with pytest.raises(UnsupportedConstruct, match="comprehension target must be a plain name"):
        lower(tuple_target)

    def over_threshold(a: float) -> float:
        return [a for _ in range(1000)][0]

    with pytest.raises(UnsupportedConstruct, match="unroll threshold"):
        lower(over_threshold)


@pytest.mark.skip(reason="FIR_PARITY_PENDING: iteration over an aggregate — stage 9")
def test_for_loop_iterates_an_aggregate() -> None:
    from jaxtyping import Float64

    def sum_rows(m: Float64[np.ndarray, "2 3"]) -> float:
        acc = 0.0
        for row in m:
            for x in row:
                acc = acc + x
        return acc

    hir = lower(sum_rows)
    assert _arith_count(hir, FloatAdd) == 6  # the 0.0 seed folds away only later, in the optimizer
    assert [o.name for o in hir.outputs] == ["out_0"]

    def iterate_a_scalar(a: float) -> float:
        acc = 0.0
        for x in a:  # type: ignore[attr-defined]
            acc = acc + x
        return acc

    with pytest.raises(UnsupportedConstruct, match="range|aggregate"):
        lower(iterate_a_scalar)


_COMPREHENSION_SHADOW = 1  # a module-level integer constant a comprehension target below deliberately shadows


@pytest.mark.skip(reason="FIR_PARITY_PENDING: iteration over an aggregate — stage 9")
def test_comprehension_target_shadows_a_same_named_module_constant() -> None:
    # A comprehension is its own scope in Python, so its target shadows a same-named global while it is bound. If the
    # target were not registered as a local for its extent, the static evaluators would resolve the global integer
    # behind the binding and fold the comparison to a compile-time answer -- a silent miscompile: the kernel would
    # return all ones instead of a one-hot vector.
    from jaxtyping import Float64

    def one_hot(v: Float64[np.ndarray, "3"]) -> Float64[np.ndarray, "3"]:
        return np.array([1.0 if _COMPREHENSION_SHADOW == 1 else 0.0 for _COMPREHENSION_SHADOW in v])

    hir = lower(one_hot)
    assert _arith_count(hir, FloatRelational) == 3  # three runtime comparisons, not a folded constant

    inputs = np.array([5.0, 1.0, 9.0])
    sim = holoso.synthesize(one_hot, default_ops(FloatFormat(11, 52)), name="onehot").numerical_model.elaborate()
    assert [float(x) for x in sim.run(*inputs.tolist())] == pytest.approx(list(np.asarray(one_hot(inputs))))

    def indexed(v: Float64[np.ndarray, "3"]) -> Float64[np.ndarray, "3"]:
        # The complement: a range-bound target IS a compile-time integer, so its comparison still folds away.
        return np.array([v[i] * 2.0 if i == 1 else v[i] for i in range(3)])

    assert _arith_count(lower(indexed), FloatRelational) == 0 and _arith_count(lower(indexed), FloatMul) == 1


_COMPREHENSION_BOUND = 2  # a module-level integer the outermost generator below reads from the enclosing scope


@pytest.mark.skip(reason="FIR_PARITY_PENDING: np.array/asarray of runtime values — stage 9")
def test_comprehension_scoping_follows_python_exactly() -> None:
    # Python evaluates the OUTERMOST iterable in the enclosing scope, before the comprehension's scope exists, so the
    # range bound below is the module constant, not the (as yet unbound) target that shadows it.
    from jaxtyping import Float64

    def outermost_iterable_reads_the_enclosing_scope(x: float) -> Float64[np.ndarray, "2"]:
        return np.array([x for _COMPREHENSION_BOUND in range(_COMPREHENSION_BOUND)])

    assert [o.name for o in lower(outermost_iterable_reads_the_enclosing_scope).outputs] == ["out_0", "out_1"]
    assert len(outermost_iterable_reads_the_enclosing_scope(1.0)) == 2  # and it is runnable Python

    def inner_generator_sees_the_unbound_comprehension_local(v: Float64[np.ndarray, "2"]) -> float:
        y = v
        return [y for x in range(1) for y in y][0]  # type: ignore[no-any-return]  # Python: UnboundLocalError on the inner y

    with pytest.raises(UnboundLocalError):
        inner_generator_sees_the_unbound_comprehension_local(np.array([1.0, 2.0]))
    with pytest.raises(UnsupportedConstruct, match="unknown name 'y'"):
        lower(inner_generator_sees_the_unbound_comprehension_local)


def test_comprehension_filter_is_lowered_before_it_is_folded() -> None:
    # The filter decides which items exist, so it must fold -- but it is lowered first, exactly as an ``if`` test is,
    # so its operands are type-checked. A fold that never looked at the condition would wave an unsupported operand
    # through whenever the other side of an ``or`` happened to be statically true.
    def unsupported_operand(x: float) -> float:
        return [x for i in range(1) if _returns_a_dict(x) or True][0]

    # The rejection surfaces from BUILDING the filter's callee (its dict literal is unsupported), which is the
    # point: the filter was lowered and expanded rather than being folded away by the statically-true ``or`` arm.
    with pytest.raises(UnsupportedConstruct, match="Dict is not supported"):
        lower(unsupported_operand)


def _returns_a_dict(v: object) -> object:
    return {"not": v}


@pytest.mark.skip(reason="FIR_PARITY_PENDING: shape query (.ndim/.shape/.T/.flatten) — stage 9")
def test_a_shape_query_cannot_slip_past_a_rejection_the_stub_makes() -> None:
    # ``.T`` is rejected on a scalar, so asking for the shape of one must be rejected identically -- otherwise a
    # static position would quietly accept an expression a value position rejects, and neither is runnable Python.
    def transpose_of_a_scalar_as_a_value(a: float) -> float:
        return a.T  # type: ignore[attr-defined, no-any-return]

    def transpose_of_a_scalar_in_a_shape_query(a: float) -> float:
        return a if a.T.ndim == 0 else -a  # type: ignore[attr-defined]

    for kernel in (transpose_of_a_scalar_as_a_value, transpose_of_a_scalar_in_a_shape_query):
        with pytest.raises(UnsupportedConstruct, match="transpose a scalar"):
            lower(kernel)


@pytest.mark.skip(reason="FIR_PARITY_PENDING: aggregate slicing — stage 9")
def test_iteration_and_shape_queries_reach_every_aggregate_spelling() -> None:
    # A `for` iterates whatever Python iterates: a slice, a transpose, a flattened matrix, a comprehension. The shape
    # queries reach the same values. Each kernel is runnable Python, so a construct Holoso accepts but Python rejects
    # would fail here rather than pass as a spurious positive.
    from jaxtyping import Float64

    def over_a_slice(m: Float64[np.ndarray, "2 3"]) -> float:
        acc = 0.0
        for e in m[0][1:]:
            acc = acc + e
        return acc

    def over_a_transpose(m: Float64[np.ndarray, "2 3"]) -> float:
        acc = 0.0
        for row in m.T:
            acc = acc + row[0]
        return acc

    def over_a_flatten(m: Float64[np.ndarray, "2 2"]) -> float:
        acc = 0.0
        for e in m.flatten():
            acc = acc + e
        return acc

    def over_a_comprehension(v: Float64[np.ndarray, "3"]) -> float:
        acc = 0.0
        for e in [v[i] * 2.0 for i in range(3)]:
            acc = acc + e
        return acc

    def negative_axis_on_a_vector(v: Float64[np.ndarray, "3"]) -> float:
        return v[v.shape[-1] - 1]  # type: ignore[no-any-return]

    m23, m22, v3 = np.arange(6.0).reshape(2, 3), np.arange(4.0).reshape(2, 2), np.arange(3.0)
    for kernel, args in (
        (over_a_slice, (m23,)),
        (over_a_transpose, (m23,)),
        (over_a_flatten, (m22,)),
        (over_a_comprehension, (v3,)),
        (negative_axis_on_a_vector, (v3,)),
    ):
        assert [o.name for o in lower(kernel).outputs] == ["out_0"]
        kernel(*args)  # runnable Python


@pytest.mark.skip(reason="FIR_PARITY_PENDING: aggregate slicing — stage 9")
def test_multi_axis_indexing_validates_its_axes_against_the_shape() -> None:
    from jaxtyping import Float64

    def out_of_range_behind_an_empty_slice(m: Float64[np.ndarray, "2 2"]) -> float:
        if len(m[:0, 99]) == 0:  # Python: IndexError, axis 1 has size 2
            return 17.0
        return -17.0

    with pytest.raises(IndexError):
        out_of_range_behind_an_empty_slice(np.array([[1.0, 2.0], [3.0, 4.0]]))

    # An empty leading slice selects no item, so a per-item bounds check never fires: the axes need an up-front probe.
    with pytest.raises(UnsupportedConstruct, match="invalid index"):
        lower(out_of_range_behind_an_empty_slice)

    def too_many_axes_behind_an_empty_slice(m: Float64[np.ndarray, "2 2"]) -> float:
        if len(m[:0, 0, 0]) == 0:  # Python: IndexError, too many indices
            return 17.0
        return -17.0

    with pytest.raises(IndexError):
        too_many_axes_behind_an_empty_slice(np.array([[1.0, 2.0], [3.0, 4.0]]))
    with pytest.raises(UnsupportedConstruct, match="too many indices"):
        lower(too_many_axes_behind_an_empty_slice)

    def multi_axis_on_an_empty_slice(m: Float64[np.ndarray, "2 2"]) -> float:
        acc = 2.0
        for x in m[0:0, :][:, 0]:
            acc = acc + x
        return acc

    # An empty aggregate is not an array, so it has no axes to index; this must be a located error, not an assertion.
    with pytest.raises(UnsupportedConstruct, match="rectangular"):
        lower(multi_axis_on_an_empty_slice)


@pytest.mark.skip(reason="FIR_PARITY_PENDING: aggregate slicing — stage 9")
def test_a_sequence_stays_a_sequence_through_a_subscript() -> None:
    class ListState:
        def __init__(self) -> None:
            self.values = [1.0, 2.0]

        def step(self, x: float) -> float:
            return x if self.values[0:1].ndim == 1 else -x  # type: ignore[attr-defined]

    with pytest.raises(AttributeError):
        ListState().step(3.0)  # a Python list slice has no .ndim
    with pytest.raises(UnsupportedConstruct, match="ndim on a Python list/tuple"):
        lower(ListState().step)


@pytest.mark.skip(reason="FIR_PARITY_PENDING: aggregate slicing — stage 9")
def test_an_empty_aggregate_makes_no_check_vacuous() -> None:
    # An empty aggregate has no leaves, so a per-leaf type check and a per-item shape check both prove nothing.
    from jaxtyping import Float64

    def negate_an_empty_boolean_slice(c: bool) -> float:
        flags = np.array([c])
        invalid = -flags[:0]  # Python: TypeError, numpy cannot negate booleans
        return 17.0 if len(invalid) == 0 else -17.0

    with pytest.raises(TypeError):
        negate_an_empty_boolean_slice(True)
    with pytest.raises(UnsupportedConstruct, match="empty aggregate"):
        lower(negate_an_empty_boolean_slice)

    def add_empty_slices_of_different_widths(a: Float64[np.ndarray, "2 3"], b: Float64[np.ndarray, "2 2"]) -> float:
        invalid = a[:0, :] + b[:0, :]  # Python: ValueError, shapes (0,3) and (0,2)
        return 17.0 if len(invalid) == 0 else -17.0

    with pytest.raises(ValueError):
        add_empty_slices_of_different_widths(np.zeros((2, 3)), np.zeros((2, 2)))
    with pytest.raises(UnsupportedConstruct, match="empty aggregate"):
        lower(add_empty_slices_of_different_widths)


@pytest.mark.skip(reason="FIR_PARITY_PENDING: shape query (.ndim/.shape/.T/.flatten) — stage 9")
def test_indexing_a_sequence_of_arrays_yields_an_array() -> None:
    from jaxtyping import Float64

    def row_of_a_list(a: Float64[np.ndarray, "2"], x: float) -> float:
        rows = [a]
        return x if rows[0].ndim == 1 else -x  # the element is the ndarray, not a list

    assert row_of_a_list(np.zeros(2), 3.0) == 3.0
    assert [o.name for o in lower(row_of_a_list).outputs] == ["out_0"]


@pytest.mark.skip(reason="FIR_PARITY_PENDING: runtime subscript/indexing — stage 9")
def test_a_scalar_takes_an_empty_tuple_key_as_identity_like_a_numpy_scalar() -> None:
    # A Holoso scalar is rank zero, as a numpy scalar is, so `x[()]` yields the scalar itself -- while `x[0]` and a
    # slice, which a numpy scalar also rejects, stay rejected.
    from jaxtyping import Float64

    def element_full_index(v: Float64[np.ndarray, "3"]) -> float:
        return v[0][()]  # type: ignore[no-any-return]  # numpy: v[0], a 0-D identity

    assert element_full_index(np.array([2.0, 4.0, 6.0])) == 2.0
    _assert_shape_kernel_matches_python(element_full_index, np.array([2.0, 4.0, 6.0]))

    def index_a_scalar(v: Float64[np.ndarray, "3"]) -> float:
        return v[0][0]  # type: ignore[no-any-return]  # numpy: IndexError, too many indices for a scalar

    with pytest.raises(IndexError):
        index_a_scalar(np.array([2.0, 4.0, 6.0]))
    with pytest.raises(UnsupportedConstruct, match="cannot index or slice a scalar"):
        lower(index_a_scalar)


def test_a_runtime_aggregate_local_lowers_and_computes() -> None:
    # The structural spine: a tuple of runtime leaves bound to a NAMED local (previously "a runtime aggregate in a
    # local is not supported yet") flows leafwise through SSA and indexes back out.
    def kernel(x: float, y: float) -> float:
        pair = (x * 2.0, y + 1.0)
        return pair[0] + pair[1]

    model = holoso.synthesize(kernel, default_ops(FloatFormat(11, 52)), name="agg_local").numerical_model.elaborate()
    assert float(model.run(3.0, 4.0)[0]) == 11.0


def test_an_aggregate_local_joins_across_a_diamond_per_leaf() -> None:
    # A diamond whose arms bind different tuples to one local merges leafwise: the differing leaf gets a phi, the
    # equal Known leaf stays a constant, and a Known-int leaf merging with a float leaf promotes C-style.
    def kernel(c: bool, x: float) -> float:
        if c:
            pair = (x, 1.0)
        else:
            pair = (2, 1.0)  # the first leaf is a Known INTEGER on this arm: the merge promotes it
        return pair[0] * 10.0 + pair[1]

    model = holoso.synthesize(kernel, default_ops(FloatFormat(11, 52)), name="agg_diamond").numerical_model.elaborate()
    assert float(model.run(True, 3.5)[0]) == 36.0
    assert float(model.run(False, 3.5)[0]) == 21.0


def test_an_aggregate_conditional_selection_selects_per_leaf() -> None:
    # ``t if c else u`` over tuples emits one typed select per differing residual leaf (previously rejected).
    def kernel(c: bool, x: float, y: float) -> float:
        chosen = (x, y) if c else (y, x)
        return chosen[0] - chosen[1]

    model = holoso.synthesize(kernel, default_ops(FloatFormat(11, 52)), name="agg_select").numerical_model.elaborate()
    assert float(model.run(True, 7.0, 2.0)[0]) == 5.0
    assert float(model.run(False, 7.0, 2.0)[0]) == -5.0


def test_aggregate_concat_and_repeat_of_runtime_leaves_compute() -> None:
    # ``+`` and ``*`` on tuple/list values route leaves without hardware; the elements still compute.
    def kernel(x: float, y: float) -> float:
        joined = (x,) + (y,)
        tripled = [x] * 3
        return joined[1] * 100.0 + tripled[2]

    model = holoso.synthesize(kernel, default_ops(FloatFormat(11, 52)), name="agg_concat").numerical_model.elaborate()
    assert float(model.run(3.0, 4.0)[0]) == 403.0


def test_a_maybe_unbound_aggregate_read_is_a_located_rejection() -> None:
    # Boundness stays at the aggregate ROOT: a tuple bound on one arm only is maybe-unbound at the join and its
    # read rejects exactly as a scalar's would (Python would raise UnboundLocalError).
    def kernel(c: bool, x: float) -> float:
        if c:
            pair = (x, x)
        return pair[0]  # noqa: F821  # deliberately maybe-unbound: the kernel under test

    with pytest.raises(UnsupportedConstruct, match="may be unbound"):
        lower(kernel)


def test_static_string_and_record_locals_lower_because_every_use_folds() -> None:
    # Review round 2: a fully-static string or record bound to a NAMED local never reaches the datapath (every use
    # folds), so the store must not try to materialize it.
    def string_mode(x: float) -> float:
        mode = "fast"
        return x * 2.0 if mode == "fast" else x

    @dataclasses.dataclass(frozen=True)
    class Params:
        gain: float

    def record_local(x: float) -> float:
        p = Params(gain=2.0)
        return x * p.gain

    for kernel, argument, expected in ((string_mode, 3.0, 6.0), (record_local, 3.0, 6.0)):
        hir = optimize(lower(kernel))
        assert all(not isinstance(b.terminator, Branch) for b in hir.blocks)  # the static guard folded away
        model = holoso.synthesize(kernel, default_ops(FloatFormat(11, 52)), name=kernel.__name__)
        assert float(model.numerical_model.elaborate().run(argument)[0]) == expected == kernel(argument)


# ---------------------------------------- spine review round (round 5) ----------------------------------------


def test_a_flavor_degraded_join_computes_per_leaf() -> None:
    # Review round 5: a tuple-meets-list diamond degrades to a flavor-erased structural layout; the per-leaf SSA
    # cells must stay aligned across the arms' differing container flavors (crashed: "read of an undefined place").
    def flat(c: bool, x: float) -> float:
        if c:
            pair = (x, 1.0)
        else:
            pair = [2.0, x]  # type: ignore[assignment]
        return pair[0] + pair[1]

    def nested(c: bool, x: float) -> float:
        if c:
            v = ((x,), 1.0)
        else:
            v = ([2.0], x)  # type: ignore[assignment]
        return v[0][0] + v[1]

    def looped(x: float) -> float:
        acc = (x, 0.0)
        while acc[0] < 10.0:
            acc = [acc[0] * 2.0, acc[1] + 1.0]  # type: ignore[assignment]
        return acc[0] + acc[1]

    for kernel, arguments in (
        (flat, (True, 3.0)),
        (flat, (False, 3.0)),
        (nested, (True, 3.0)),
        (nested, (False, 3.0)),
        (looped, (0.5,)),
        (looped, (11.0,)),
    ):
        model = holoso.synthesize(kernel, default_ops(FloatFormat(11, 52)), name=kernel.__name__)
        assert float(model.numerical_model.elaborate().run(*arguments)[0]) == kernel(*arguments)


def test_a_residual_record_field_projection_computes() -> None:
    # Review round 5: a record with residual leaves (a select of two records) projects a field through the leaf
    # cells (crashed: PyAttr emission assumed a component ObjectRef). Nested case: an aggregate-valued field.
    @dataclasses.dataclass(frozen=True)
    class Params:
        gain: float
        offset: float

    def scalar_field(c: bool, x: float) -> float:
        p = Params(2.0, 1.0) if c else Params(3.0, -1.0)
        return x * p.gain + p.offset

    @dataclasses.dataclass(frozen=True)
    class Gains:
        pair: tuple[float, float]
        scale: float

    def aggregate_field(c: bool, x: float) -> float:
        g = Gains((1.5, 2.0), 3.0) if c else Gains((2.5, 4.0), 5.0)
        return g.pair[0] * g.scale + g.pair[1] * x

    for kernel in (scalar_field, aggregate_field):
        model = holoso.synthesize(kernel, default_ops(FloatFormat(11, 52)), name=kernel.__name__)
        elaborated = model.numerical_model.elaborate()
        for c in (True, False):
            assert float(elaborated.run(c, 3.0)[0]) == kernel(c, 3.0)


def test_a_known_condition_selection_of_aggregates_routes_leaves() -> None:
    # Review round 5: ``and``/``or`` picking an aggregate under a Known condition must route leaves, not
    # materialize the aggregate as a scalar (crashed: "an aggregate value reaches a scalar operand position").
    def kernel(x: float, y: float) -> float:
        pair = True and (x, 1.0)
        other = (y, 2.0) or (x, 0.0)
        return pair[0] + other[0]

    model = holoso.synthesize(kernel, default_ops(FloatFormat(11, 52)), name="known_cond_agg").numerical_model
    assert float(model.elaborate().run(3.0, 4.0)[0]) == 7.0


def test_aggregate_static_leaves_outside_the_datapath_fold_at_every_use() -> None:
    # Review round 5: a str/function leaf inside an aggregate must stay fact-only (its every use folds); eager
    # materialization rejected kernels whose non-datapath leaves never reach hardware.
    def tag_guard(x: float) -> float:
        pair = ("gain", x)
        return pair[1] if pair[0] == "gain" else 0.0

    def names(x: float) -> float:
        mode = ("fast", "slow")
        return x * 2.0 if mode[0] == "fast" else x

    @dataclasses.dataclass(frozen=True)
    class Named:
        label: str
        value: float

    def record_with_str(x: float) -> float:
        n = Named("boost", 2.0)
        return x * n.value

    def helper(v: float) -> float:
        return v + 1.0

    def dispatch(x: float) -> float:
        table = (helper, helper)
        return table[0](x)

    for kernel, expected in ((tag_guard, 3.0), (names, 6.0), (record_with_str, 6.0), (dispatch, 4.0)):
        model = holoso.synthesize(kernel, default_ops(FloatFormat(11, 52)), name=kernel.__name__)
        assert float(model.numerical_model.elaborate().run(3.0)[0]) == expected == kernel(3.0)


def test_component_aggregate_attributes_normalize_at_admission() -> None:
    # Review round 5: attribute-sourced aggregates must enter the fact domain normalized (Known(StaticSeq) is
    # banned), so concat and selects treat them exactly like locally-built sequences.
    class Comp:
        def __init__(self) -> None:
            self.gains = (1.0, 2.0)

        def __call__(self, c: bool, x: float) -> float:
            extended = self.gains + (3.0, 4.0)
            chosen = self.gains if c else (5.0, 6.0)
            return x * extended[3] + chosen[0]

    model = holoso.synthesize(Comp().__call__, default_ops(FloatFormat(11, 52)), name="attr_agg").numerical_model
    elaborated = model.elaborate()
    assert float(elaborated.run(True, 2.0)[0]) == 9.0
    assert float(elaborated.run(False, 2.0)[0]) == 13.0


def test_list_mutation_through_a_component_attribute_is_a_located_rejection() -> None:
    # Review round 5 MISCOMPILE: ``.append`` on an attribute-sourced list mutated a disposable reconstruction of
    # the snapshot (returned 4.0 where Python returns 5.0); ``+=`` lost its dedicated in-place message.
    class Appender:
        def __init__(self) -> None:
            self.config = [1.0]

        def __call__(self, x: float) -> float:
            self.config.append(2.0)
            return x + float(len(self.config))

    class Augmenter:
        def __init__(self) -> None:
            self.buf = [0.0]

        def __call__(self, x: float) -> float:
            self.buf += [x]
            return self.buf[0]

    with pytest.raises(UnsupportedConstruct, match="list method 'append'"):
        lower(Appender().__call__)
    with pytest.raises(UnsupportedConstruct, match="in-place list mutation"):
        lower(Augmenter().__call__)


def test_record_truth_is_python_object_truth() -> None:
    # Review round 5 MISCOMPILE: a zero-field record is truthy in Python; arity-based truth chose the else arm.
    @dataclasses.dataclass(frozen=True)
    class Marker:
        pass

    marker = Marker()

    def kernel(x: float) -> float:
        return x if marker else 0.0

    model = holoso.synthesize(kernel, default_ops(FloatFormat(11, 52)), name="record_truth").numerical_model
    assert float(model.elaborate().run(3.0)[0]) == 3.0


def test_flavor_erased_truth_with_an_array_side_stays_ambiguous() -> None:
    # Review round 5: a join carrying the ARRAY flavor cannot fold truth by arity (numpy raises on multi-element
    # truth), so it inherits the array ambiguity rejection.
    def kernel(c: bool, x: float) -> float:
        v = np.array([1.0, 2.0]) if c else (1.0, 2.0)
        return x if v else 0.0

    with pytest.raises(UnsupportedConstruct, match="truth value of an array"):
        lower(kernel)


def test_a_record_is_not_positionally_subscriptable() -> None:
    # Review round 5 MISCOMPILE: positional subscript of a record projected field 0 where Python raises TypeError.
    @dataclasses.dataclass(frozen=True)
    class Params:
        gain: float

    p = Params(2.0)

    def global_record(x: float) -> float:
        return x + p[0]  # type: ignore[index,no-any-return]

    def local_record(x: float) -> float:
        q = Params(3.0)
        return x + q[0]  # type: ignore[index,no-any-return]

    for kernel in (global_record, local_record):
        with pytest.raises(UnsupportedConstruct, match="record is not subscriptable"):
            lower(kernel)


def test_a_boolean_index_into_an_array_is_a_located_rejection() -> None:
    # Review round 5 MISCOMPILE: ``table[True]`` applied operator.index (True == 1) where numpy prepends an axis
    # (boolean advanced indexing). Python's tuple semantics genuinely accept booleans as indices, so only the
    # array flavor rejects.
    table = np.array([10.0, 20.0])

    def array_bool_index(x: float) -> float:
        return x + table[True]  # type: ignore[no-any-return]

    def tuple_bool_index(x: float) -> float:
        pair = (x, x * 2.0)
        return pair[True]

    with pytest.raises(UnsupportedConstruct, match="boolean index"):
        lower(array_bool_index)
    model = holoso.synthesize(tuple_bool_index, default_ops(FloatFormat(11, 52)), name="tuple_bool").numerical_model
    assert float(model.elaborate().run(3.0)[0]) == 6.0


def test_a_zero_dimensional_array_folds_and_rejects_navigation() -> None:
    # Review rounds 5+6: a 0-d ndarray crashed structural navigation with a raw IndexError; scalarizing it at
    # admission was tried and reverted (it broke static type identity: isinstance(z, np.ndarray) folded False).
    # It stays an array -- concrete folds work through materialization, navigation is a located rejection.
    z = np.array(3.0)

    def scales(x: float) -> float:
        return x * float(z)

    def type_sensitive(x: float) -> float:
        return x if isinstance(z, np.ndarray) else 0.0

    def subscripted(x: float) -> float:
        return x * z[0]  # type: ignore[no-any-return]  # numpy raises IndexError: too many indices

    def sized(x: float) -> float:
        return x + float(len(z))  # numpy raises TypeError: len() of unsized object

    fmt = FloatFormat(11, 52)
    assert float(holoso.synthesize(scales, default_ops(fmt), name="p").numerical_model.elaborate().run(2.0)[0]) == 6.0
    model = holoso.synthesize(type_sensitive, default_ops(fmt), name="p").numerical_model
    assert float(model.elaborate().run(3.0)[0]) == 3.0
    with pytest.raises(UnsupportedConstruct, match="0-dimensional array cannot be indexed"):
        lower(subscripted)
    with pytest.raises(UnsupportedConstruct, match=r"len\(\) of"):  # the concrete fold surfaces numpy's own message
        lower(sized)


# ---------------------------------------- spine review round 6 ----------------------------------------


def test_record_truth_with_a_custom_bool_is_a_located_rejection() -> None:
    # Review rounds 6+7: a record class defining __bool__ folded truthy by default (round 6 MISCOMPILE); folding
    # the override concretely was then found unsound too -- it runs on a value-faithful but not TYPE-faithful
    # reconstruction (np.bool_ admits as plain bool, flipping ``a + b == 2``), so any class-dictionary
    # __bool__/__len__ entry rejects, ``__bool__ = None`` included (Python raises TypeError on its truth).
    @dataclasses.dataclass(frozen=True)
    class Disabled:
        def __bool__(self) -> bool:
            return False

    @dataclasses.dataclass(frozen=True)
    class Sized:
        def __len__(self) -> int:
            return 0

    @dataclasses.dataclass(frozen=True)
    class Unbooled:
        __bool__ = None

    for instance in (Disabled(), Sized(), Unbooled()):

        def kernel(x: float) -> float:
            return x if instance else 0.0  # noqa: B023

        with pytest.raises(UnsupportedConstruct, match="custom __bool__"):
            lower(kernel)

    @dataclasses.dataclass(frozen=True)
    class Gated:
        level: float

        def __bool__(self) -> bool:
            return self.level > 0.0

    def residual(c: bool, x: float) -> float:
        g = Gated(1.0) if c else Gated(-1.0)
        return x if g else 0.0

    with pytest.raises(UnsupportedConstruct, match="custom __bool__"):
        lower(residual)


def test_a_flavor_divergent_return_is_a_contract_rejection() -> None:
    # Review round 6: one arm returns a tuple, the other a list; strict contracts refuse the flavor-erased join
    # instead of blessing the diverging path with the declared flavor.
    def diverges(flag: bool, x: float) -> tuple[float, float]:
        if flag:
            return x, 1.0
        return [x, 2.0]  # type: ignore[return-value]

    def joined_local(flag: bool, x: float) -> list[float]:
        v = (x, 1.0) if flag else [x, 2.0]
        return v  # type: ignore[return-value]

    for kernel in (diverges, joined_local):
        with pytest.raises(UnsupportedConstruct, match="container flavor diverges across paths"):
            lower(kernel)


def test_concretely_folded_aggregate_subscripts_need_no_cells() -> None:
    # Review round 6: a tuple key or a slice OBJECT folds concretely at analysis; emission must not enter the
    # positional projection with a non-integer key (crashed: AssertionError / TypeError at operator.index).
    def tuple_key(x: float) -> float:
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        row = a[(1,)]
        return float(row[1] + x)

    def slice_object(x: float) -> float:
        t = (1.0, 2.0, 3.0)
        s = slice(0, 2)
        u = t[s]
        return u[0] + u[1] + x  # type: ignore[index,no-any-return]  # mypy picks the int overload for t[s]

    fmt = FloatFormat(11, 52)
    for kernel, expected in ((tuple_key, 5.0), (slice_object, 4.0)):
        model = holoso.synthesize(kernel, default_ops(fmt), name=kernel.__name__).numerical_model
        assert float(model.elaborate().run(1.0)[0]) == expected == kernel(1.0)


def test_sequence_repeat_counts_follow_python() -> None:
    # Review rounds 6-8: a negative count yields the empty sequence and a plain-bool count repeats 0/1 times,
    # exactly as Python -- sound since the NpBool split keeps np.bool_ provenance, so the np spelling (which
    # numpy 2 stripped of __index__, a Python TypeError) rejects instead of miscompiling. A count beyond the
    # ssize_t index range (Python: OverflowError) rejects rather than clamping.
    def repeat_negative(x: float) -> float:
        rest = [x] * -1
        return x + float(len(rest))

    def repeat_python_bool(x: float) -> float:
        pair = (x,) * True
        return pair[0]

    fmt = FloatFormat(11, 52)
    for kernel in (repeat_negative, repeat_python_bool):
        model = holoso.synthesize(kernel, default_ops(fmt), name=kernel.__name__).numerical_model
        assert float(model.elaborate().run(3.0)[0]) == 3.0 == kernel(3.0)

    numpy_true = np.bool_(True)

    def repeat_numpy_bool(x: float) -> float:
        pair = (x,) * numpy_true  # type: ignore[operator]  # numpy 2: TypeError, np.bool_ has no __index__
        return pair[0]  # type: ignore[no-any-return]

    def repeat_beyond_index_range(x: float) -> float:
        rest = (x,) * -(1 << 100)
        return x + float(len(rest))

    for kernel in (repeat_numpy_bool, repeat_beyond_index_range):
        with pytest.raises(UnsupportedConstruct, match="arithmetic on an aggregate value"):
            lower(kernel)


# ---------------------------------------- spine review round 7 ----------------------------------------


def test_concretely_folded_array_attributes_need_no_cells() -> None:
    # Review round 7: attribute navigation of an all-Known array (.shape, .T, a 0-d array's .real) folds at
    # analysis; emission asserted the aggregate source was a record and crashed. Same invariant as the folded
    # subscript: an all-Known destination needs no cells.
    zero_d = np.array(3.0)
    vector = np.array([1.0, 2.0, 3.0])
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]])

    def real_of_zero_d(x: float) -> float:
        return x + float(zero_d.real)

    def shape_len(x: float) -> float:
        return x + float(len(zero_d.shape))

    def shape_element(x: float) -> float:
        return x + float(vector.shape[0])

    def transposed_element(x: float) -> float:
        return x + float(matrix.T[0][1])

    fmt = FloatFormat(11, 52)
    for kernel in (real_of_zero_d, shape_len, shape_element, transposed_element):
        model = holoso.synthesize(kernel, default_ops(fmt), name=kernel.__name__).numerical_model
        assert float(model.elaborate().run(2.0)[0]) == kernel(2.0)


def test_an_index_able_aggregate_key_projects_like_a_known_integer() -> None:
    # Review round 7: a 0-d integer array is a valid Python sequence index (operator.index accepts it); the
    # analyzer projected the child but emission asserted the key fact was Known and crashed on the aggregate.
    zero_d_index = np.array(0)

    def kernel(x: float) -> float:
        t = (x, 2.0)
        return t[zero_d_index] * 4.0

    model = holoso.synthesize(kernel, default_ops(FloatFormat(11, 52)), name="idx_agg").numerical_model
    assert float(model.elaborate().run(3.0)[0]) == 12.0 == kernel(3.0)


# ---------------------------------------- spine review round 8 ----------------------------------------


def test_numpy_bool_provenance_survives_admission() -> None:
    # Review round 8: np.bool_ admitted as a plain StaticBool, so reconstructions and index/repeat semantics
    # silently took the Python-bool meaning. The NpBool variant keeps provenance exactly like NpInt/NpFloat: the
    # np spelling of a subscript index rejects (numpy 2 removed __index__) while the plain bool keeps Python's
    # bool-as-int semantics, and a concrete operator.index(np.True_) surfaces numpy's own TypeError.
    numpy_true = np.bool_(True)

    def np_key(x: float) -> float:
        return (x, x + 1.0)[numpy_true]  # type: ignore[call-overload]

    import operator

    index_of = operator.index

    def np_index_call(x: float) -> float:
        return ((x,) * index_of(numpy_true))[0]  # type: ignore[arg-type,no-any-return]

    with pytest.raises(UnsupportedConstruct, match="np.bool_ subscript index"):
        lower(np_key)
    with pytest.raises(UnsupportedConstruct, match="cannot be interpreted as an integer"):
        lower(np_index_call)

    def py_key(x: float) -> float:
        pair = (x, x * 2.0)
        return pair[True]

    model = holoso.synthesize(py_key, default_ops(FloatFormat(11, 52)), name="py_key").numerical_model
    assert float(model.elaborate().run(3.0)[0]) == 6.0 == py_key(3.0)


def test_identity_dependent_array_attributes_are_a_located_rejection() -> None:
    # Review round 8 MISCOMPILE: .base observed the admitted snapshot's storage, not the user's view (returned the
    # backing array's element). Only the value-determined navigation set folds.
    backing = np.array([1.0, 2.0, 3.0])
    view = backing[1:]

    def kernel(x: float) -> float:
        return x + float(view.base[0])  # type: ignore[index]

    with pytest.raises(UnsupportedConstruct, match="array attribute 'base' is not supported"):
        lower(kernel)


# ---------------------------------------- spine review round 9 ----------------------------------------


def test_derived_numpy_booleans_keep_provenance() -> None:
    # Review round 9: comparisons, bitops, and numpy classifier folds produced plain StaticBool even with numpy
    # operands, laundering the provenance the NpBool split introduced -- the derived value then passed the
    # bool-as-int gates that a spelled np.bool_ fails. Boolean-producing folds now keep numpy provenance.
    derived_key = np.True_ == np.True_
    derived_count = np.int64(1) == np.int64(1)

    def np_compare_key(x: float) -> float:
        return (x, -x)[derived_key]

    def np_compare_count(x: float) -> float:
        return ((x,) * derived_count)[0]  # type: ignore[no-any-return]

    def np_bitop_count(x: float) -> float:
        return ((x,) * (np.True_ & np.True_))[0]  # type: ignore[operator,no-any-return]

    def np_classifier_count(x: float) -> float:
        return ((x,) * np.isfinite(1))[0]  # type: ignore[no-any-return]

    with pytest.raises(UnsupportedConstruct, match="np.bool_ subscript index"):
        lower(np_compare_key)
    for kernel in (np_compare_count, np_bitop_count, np_classifier_count):
        with pytest.raises(UnsupportedConstruct, match="arithmetic on an aggregate value"):
            lower(kernel)

    def python_compare_key(x: float) -> float:
        pair = (x, x * 2.0)
        return pair[1 == 1]

    model = holoso.synthesize(python_compare_key, default_ops(FloatFormat(11, 52)), name="py_cmp").numerical_model
    assert float(model.elaborate().run(3.0)[0]) == 6.0 == python_compare_key(3.0)


def test_flatten_with_an_order_argument_keeps_rejecting() -> None:
    # Review round 9 established that flatten(order="K"/"A") observes the memory layout the C-contiguous
    # snapshot discarded (a demonstrated wrong value on a Fortran-ordered original). Flatten is back as a
    # structural relayout, but ONLY in the default C order, which is layout-independent; any order argument
    # keeps rejecting.
    fortran = np.asfortranarray([[1.0, 2.0], [3.0, 4.0]])

    def kernel(x: float) -> float:
        return x + float(fortran.flatten(order="K")[1])

    with pytest.raises(UnsupportedConstruct, match="default C order"):
        lower(kernel)

    def default_order(x: float) -> float:
        # C-order flattening is memory-layout-independent, so it is faithful even on a Fortran-ordered original.
        return x + float(fortran.flatten()[1])

    model = holoso.synthesize(default_order, default_ops(FloatFormat(11, 52)), name="k").numerical_model
    assert float(model.elaborate().run(1.0)[0]) == default_order(1.0)


# ---------------------------------------- spine review round 13 ----------------------------------------


def test_dataclass_construction_admits_only_the_generated_machinery() -> None:
    # Review round 13: a user __new__ and a field default_factory are the remaining compile-time code hooks in
    # construction -- a factory popped the user's live list once per analysis round and baked the replay's draw
    # into the emitted model. Construction admits only generated-__init__ classes with no hooks or factories.
    import enum
    import itertools

    class Mode(enum.IntEnum):
        A = 1

    @dataclasses.dataclass
    class NewBox:
        mode: Mode
        scale: float = dataclasses.field(init=False)

        def __new__(cls, mode: Mode) -> "NewBox":
            obj = super().__new__(cls)
            obj.scale = 10.0 if isinstance(mode, Mode) else 20.0
            return obj

    mode = Mode.A

    def dunder_new(x: float) -> float:
        return x * NewBox(mode).scale

    live = [9.0, 7.0, 5.0]

    @dataclasses.dataclass
    class FactoryBox:
        v: float
        junk: float = dataclasses.field(default_factory=live.pop)

    def factory(x: float) -> float:
        return x * FactoryBox(2.0).v

    counter = itertools.count(10)

    @dataclasses.dataclass
    class CounterBox:
        v: float = dataclasses.field(default_factory=lambda: float(next(counter)))

    def drawing_factory(x: float) -> float:
        return x + CounterBox().v

    for kernel in (dunder_new, factory, drawing_factory):
        with pytest.raises(UnsupportedConstruct, match="is not supported in a kernel"):
            lower(kernel)
    assert live == [9.0, 7.0, 5.0]  # compilation must never mutate the user's live objects


def test_oversized_range_unpacking_is_a_located_rejection() -> None:
    # len() of range(10**30) raises OverflowError (past ssize_t), which escaped raw through the builder's
    # unpacking arity check; it is a located rejection now, like the unroll path's.
    def kernel(x: float) -> float:
        a, b = range(10**30)
        return x * float(a + b)

    with pytest.raises(UnsupportedConstruct, match="oversized range"):
        lower(kernel)


def test_dataclass_construction_rejects_data_descriptor_fields_before_running_them() -> None:
    # A field backed by a data descriptor would route field assignment (and reads) through user code, which
    # structural construction cannot reproduce; the schema predicate refuses before anything is modeled.
    events: list[object] = []

    class LoggedField:
        def __get__(self, instance: object, owner: object = None) -> object:
            return self if instance is None else 0

        def __set__(self, instance: object, value: object) -> None:
            events.append(value)

    @dataclasses.dataclass
    class Box:
        value: int = LoggedField()  # type: ignore[assignment]

    def kernel(x: float) -> float:
        box = Box(7)
        return x + float(box.value)

    with pytest.raises(UnsupportedConstruct, match="record class 'Box' has a descriptor-backed field"):
        lower(kernel)
    assert events == [], "the descriptor setter must never run at compile time"


# ------------------------------ aggregate conversion rules (migration phase 6) ------------------------------


def test_list_and_tuple_convert_aggregates_as_layout_operations() -> None:
    # list()/tuple() over an aggregate re-flavors the SAME leaves -- runtime ones included -- without any
    # evaluation; concrete containers (a range, a string) still fold through the vetted constructor.
    def runtime_conversion(x: float, y: float) -> float:
        items = list((x, y))
        pair = tuple([y, x])
        return items[0] * 10.0 + pair[0]

    model = holoso.synthesize(runtime_conversion, default_ops(FloatFormat(11, 52)), name="conv").numerical_model
    assert float(model.elaborate().run(3.0, 4.0)[0]) == runtime_conversion(3.0, 4.0) == 34.0

    def concatenates_after_conversion(x: float) -> float:
        grown = list((x, 2.0)) + [3.0]
        return grown[0] + grown[2]

    model = holoso.synthesize(
        concatenates_after_conversion, default_ops(FloatFormat(11, 52)), name="convcat"
    ).numerical_model
    assert float(model.elaborate().run(1.0)[0]) == concatenates_after_conversion(1.0) == 4.0

    def concrete_sources(x: float) -> float:
        return x * float(list(range(4))[2] + len(tuple("ab")))

    model = holoso.synthesize(concrete_sources, default_ops(FloatFormat(11, 52)), name="convrng").numerical_model
    assert float(model.elaborate().run(2.0)[0]) == concrete_sources(2.0) == 8.0

    @dataclasses.dataclass
    class Point:
        x: float
        y: float

    def record_conversion(x: float) -> float:
        return x + list(Point(1.0, 2.0))[0]  # type: ignore[call-overload,no-any-return]

    with pytest.raises(UnsupportedConstruct, match="a record cannot cross into a concrete call"):
        lower(record_conversion)


# ------------------------------ structural records (migration phase 5) ------------------------------


def test_record_construction_with_runtime_arguments_is_structural() -> None:
    # Phase 5: a record built from runtime values is a layout operation -- argument cells install into
    # per-field windows -- so construction, projection, and branch joins ride the aggregate spine.
    @dataclasses.dataclass(frozen=True)
    class Decision:
        drive: float
        ok: bool
        scale: float = 2.0

    def round_trip(x: float, p: bool) -> float:
        d = Decision(x + 1.0, p)
        return d.drive * d.scale if d.ok else -d.drive

    def joined(c: bool, x: float) -> float:
        d = Decision(x, True) if c else Decision(-x, True, 5.0)
        return d.drive * d.scale

    for kernel, argsets in ((round_trip, [(3.0, True), (3.0, False)]), (joined, [(True, 3.0), (False, 3.0)])):
        model = holoso.synthesize(kernel, default_ops(FloatFormat(11, 52)), name=kernel.__name__).numerical_model
        elaborated = model.elaborate()
        for argset in argsets:
            assert float(elaborated.run(*argset)[0]) == kernel(*argset)


def test_record_construction_keywords_defaults_and_kw_only() -> None:
    @dataclasses.dataclass(frozen=True)
    class Gains:
        a: float
        b: float = dataclasses.field(default=7.0, kw_only=True)
        c: float = 1.0

    def kernel(x: float) -> float:
        full = Gains(x, 3.0, b=5.0)
        defaulted = Gains(c=x, a=2.0)
        return full.a + full.b + full.c + defaulted.a * defaulted.b + defaulted.c

    model = holoso.synthesize(kernel, default_ops(FloatFormat(11, 52)), name="kwrec").numerical_model
    assert float(model.elaborate().run(2.0)[0]) == kernel(2.0)


def test_record_construction_nests_and_carries_nonvalue_leaves() -> None:
    # A record field may be another runtime record, and a field may hold what the datapath cannot: an
    # unadmittable default (None) or an explicit object reference stays a fact-only leaf as in tuples.
    @dataclasses.dataclass(frozen=True)
    class Inner:
        v: float
        s: float

    @dataclasses.dataclass(frozen=True)
    class Outer:
        inner: Inner
        bias: float

    def nested(x: float) -> float:
        o = Outer(Inner(x, 4.0), 0.5)
        return o.inner.v * o.inner.s + o.bias

    @dataclasses.dataclass(frozen=True)
    class Tagged:
        v: float
        tag: object = None

    def reference_leaves(x: float) -> float:
        own = Tagged(x, math)
        defaulted = Tagged(x * 2.0)
        return own.v + defaulted.v

    for kernel, expected in ((nested, 8.5), (reference_leaves, 6.0)):
        model = holoso.synthesize(kernel, default_ops(FloatFormat(11, 52)), name=kernel.__name__).numerical_model
        assert float(model.elaborate().run(2.0)[0]) == kernel(2.0) == expected


def test_record_construction_slots_and_bare_subclass() -> None:
    # slots=True fields are member descriptors (the fields themselves, not user code); an undecorated subclass
    # constructs through the parent's generated schema and keeps its OWN class identity in the layout.
    @dataclasses.dataclass(frozen=True, slots=True)
    class Slotted:
        v: float
        w: float

    @dataclasses.dataclass(frozen=True)
    class Base:
        v: float

    class Sub(Base):
        pass

    def kernel(x: float) -> float:
        s = Slotted(x, 3.0)
        u = Sub(x * 2.0)
        picked = 1.0 if isinstance(u, Sub) else 0.0
        return s.v * s.w + u.v + picked

    model = holoso.synthesize(kernel, default_ops(FloatFormat(11, 52)), name="slotsub").numerical_model
    assert float(model.elaborate().run(2.0)[0]) == kernel(2.0) == 11.0


def test_record_construction_never_runs_hooks_and_names_schema_mismatches() -> None:
    # Structural construction executes NO class machinery; a class whose construction would run user code
    # refuses by schema, and the call-to-field mapping errors are located and named.
    events: list[object] = []

    @dataclasses.dataclass
    class Hooked:
        v: float

        def __post_init__(self) -> None:
            events.append(self.v)

    def hooked(x: float) -> float:
        return Hooked(x).v

    @dataclasses.dataclass
    class WithInitVar:
        v: float
        tweak: dataclasses.InitVar[float] = 0.0

    def initvar(x: float) -> float:
        return WithInitVar(x).v

    @dataclasses.dataclass
    class Reader:
        v: float

        def __getattribute__(self, name: str) -> object:
            return object.__getattribute__(self, name)

    def warped_reads(x: float) -> float:
        return Reader(x).v

    @dataclasses.dataclass(frozen=True)
    class Plain:
        drive: float
        ok: bool
        scale: float = 2.0

    def missing(x: float) -> float:
        return Plain(x).drive  # type: ignore[call-arg]

    def excess(x: float) -> float:
        return Plain(x, True, 1.0, 2.0).drive  # type: ignore[call-arg]

    def unknown(x: float) -> float:
        return Plain(x, True, nope=1.0).drive  # type: ignore[call-arg]

    def duplicate(x: float) -> float:
        return Plain(x, drive=2.0, ok=True).drive  # type: ignore[misc]

    for kernel, pattern in (
        (hooked, "runs user code in construction"),
        (initvar, "init-only or init=False fields"),
        (warped_reads, "runs user code in construction"),
        (missing, "missing the required field 'ok'"),
        (excess, "takes 3 positional argument"),
        (unknown, "has no field 'nope'"),
        (duplicate, "got multiple values for field 'drive'"),
    ):
        with pytest.raises(UnsupportedConstruct, match=pattern):
            lower(kernel)
    assert events == [], "construction hooks must never run at compile time"


def test_record_field_provenance_rides_construction() -> None:
    # Children of a structural construction are the argument facts THEMSELVES: a retained enum member keeps
    # answering mixin-base isinstance after a field round trip, and a LOST leaf keeps refusing.
    import enum

    class Marker:
        pass

    class Mode(Marker, enum.IntEnum):
        B = 4

    @dataclasses.dataclass(frozen=True)
    class WithEnum:
        mode: Mode
        gain: float

    def faithful(x: float) -> float:
        w = WithEnum(Mode.B, x)
        return w.gain * 2.0 if isinstance(w.mode, Marker) else w.gain

    model = holoso.synthesize(faithful, default_ops(FloatFormat(11, 52)), name="provrec").numerical_model
    assert float(model.elaborate().run(3.0)[0]) == faithful(3.0) == 6.0

    @dataclasses.dataclass(frozen=True)
    class Holder:
        value: int

    def lost(x: float, p: bool) -> float:
        h = Holder(Mode.B if p else 4)
        return x if isinstance(h.value, Marker) else -x

    with pytest.raises(UnsupportedConstruct, match="not decidable"):
        lower(lost)


def test_record_default_snapshots_are_admitted_once() -> None:
    # A mutable field default (an eq=False record is identity-hashable, so dataclasses accepts it) must pin
    # its FIRST admitted value: a permitted module hook mutating the default mid-analysis must not move the
    # folded constant between the fixpoint and the emission replay.
    @dataclasses.dataclass(eq=False)
    class Knob:
        v: float

    knob_default = Knob(5.0)

    @dataclasses.dataclass(frozen=True)
    class Cfg:
        scale: float
        knob: Knob = knob_default

    module = types.ModuleType("lazy_default")

    def module_getattr(name: str) -> float:
        if name != "trigger":
            raise AttributeError(name)
        knob_default.v = 9.0
        return 0.0

    module.__getattr__ = module_getattr  # type: ignore[method-assign]

    def kernel(x: float) -> float:
        c = Cfg(x)
        ignored = float(getattr(module, "trigger"))
        return c.knob.v * x + ignored

    model = holoso.synthesize(kernel, default_ops(FloatFormat(11, 52)), name="defsnap").numerical_model
    knob_default.v = 5.0
    assert float(model.elaborate().run(2.0)[0]) == 10.0, "the first admitted default must be the folded one"


def test_record_construction_certifies_the_live_initializer() -> None:
    # Design consult (Codex gpt-5.6-sol ultra): schema purity is certified against the LIVE class, not just
    # decoration-time shape -- a __post_init__ deleted after decoration still sits in the generated bytecode
    # (Python raises AttributeError), mutated initializer defaults diverge from Field metadata, a __del__ is an
    # implicit lifetime hook, and a callable-object __setattr__ is user code the source check must not miss.
    @dataclasses.dataclass
    class Ghost:
        v: float

        def __post_init__(self) -> None:
            pass

    del Ghost.__post_init__

    def ghost(x: float) -> float:
        return Ghost(x).v

    @dataclasses.dataclass
    class Moved:
        v: float
        gain: float = 1.0

    Moved.__init__.__defaults__ = (2.0,)

    def moved(x: float) -> float:
        return Moved(x).gain * x

    @dataclasses.dataclass
    class Finalized:
        v: float

        def __del__(self) -> None:
            pass

    def finalized(x: float) -> float:
        return Finalized(x).v

    events: list[object] = []

    class RecordingSetter:
        def __call__(self, instance: object, name: str, value: object) -> None:
            events.append((name, value))

    @dataclasses.dataclass
    class Trapped:
        v: float

    Trapped.__setattr__ = RecordingSetter()  # type: ignore[method-assign]

    def trapped(x: float) -> float:
        return Trapped(x).v

    class Donor:
        __slots__ = ("v",)

    @dataclasses.dataclass
    class Aliased:
        v: float

    Aliased.v = Donor.v  # type: ignore[attr-defined]

    def aliased(x: float) -> float:
        return Aliased(x).v

    for kernel, pattern in (
        (ghost, "runs user code in construction"),
        (moved, "initializer defaults diverging"),
        (finalized, "runs user code in construction"),
        (trapped, "runs user code in construction"),
        (aliased, "descriptor-backed field"),
    ):
        with pytest.raises(UnsupportedConstruct, match=pattern):
            lower(kernel)
    assert events == [], "the callable setter must never run at compile time"


def test_record_defaults_admit_lazily_at_first_omission() -> None:
    # Design consult (Codex gpt-5.6-sol ultra): Python never observes an overridden default, so the snapshot is
    # taken at the FIRST construction that actually omits the field -- after a permitted module hook mutated the
    # default object, the omitting construction (and Python) see the mutated value, while an eager schema-time
    # snapshot would fold the stale one.
    @dataclasses.dataclass(eq=False)
    class Knob:
        v: float

    lazy_knob = Knob(5.0)

    @dataclasses.dataclass(frozen=True)
    class Cfg:
        scale: float
        knob: Knob = lazy_knob

    module = types.ModuleType("lazy_default_two")

    def module_getattr(name: str) -> float:
        if name != "trigger":
            raise AttributeError(name)
        lazy_knob.v = 9.0
        return 0.0

    module.__getattr__ = module_getattr  # type: ignore[method-assign]

    def kernel(x: float) -> float:
        overridden = Cfg(x, Knob(1.0))
        ignored = float(getattr(module, "trigger"))
        defaulted = Cfg(x)
        return defaulted.knob.v * x + overridden.knob.v + ignored

    expected = kernel(2.0)
    assert expected == 19.0, "Python reads the mutated default"
    lazy_knob.v = 5.0
    model = holoso.synthesize(kernel, default_ops(FloatFormat(11, 52)), name="lazydef").numerical_model
    lazy_knob.v = 5.0
    assert float(model.elaborate().run(2.0)[0]) == expected


def test_record_carrying_sequences_never_rebuild_through_subscript_fallback() -> None:
    # Review round (Claude ultrathink): a slice/tuple key on a record-CARRYING tuple slipped past the key and
    # top-level-object guards into the concrete subscript fallback, which rebuilt real instances at compile
    # time (a user __del__ fired during analysis; hook-free records crossed silently) -- the one record path
    # that still reached host evaluation. The fallback now refuses before materializing.
    destroyed: list[object] = []

    @dataclasses.dataclass(frozen=True)
    class Watched:
        lo: float
        hi: float

        def __del__(self) -> None:
            destroyed.append(self)

    bands = (Watched(1.0, 2.0), Watched(3.0, 4.0))

    def admitted(x: float) -> float:
        pair = bands[slice(0, 2)]
        return x + pair[0].lo  # type: ignore[no-any-return,index]

    destroyed.clear()
    with pytest.raises(UnsupportedConstruct, match="record-carrying sequence"):
        lower(admitted)
    assert destroyed == [], "no instance may be rebuilt (and collected) at compile time"

    @dataclasses.dataclass(frozen=True)
    class Plain:
        v: float

    def hook_free(x: float) -> float:
        t = (Plain(1.0), Plain(x))
        return x + t[slice(0, 2)][0].v  # type: ignore[index,no-any-return]

    with pytest.raises(UnsupportedConstruct, match="record-carrying sequence"):
        lower(hook_free)


# ------------------------------ M2 step 3a: slices and starred targets ------------------------------


def test_slices_of_positional_containers_are_window_operations() -> None:
    # Slice syntax desugars to the vetted slice() constructor and the subscript transfer re-aggregates the
    # SAME children -- runtime leaves included -- so windows, strides, reversals, and open bounds all ride
    # the aggregate spine with no evaluation.
    def runtime_window(x: float, y: float) -> float:
        t = (x, y, x + y, 4.0)
        mid = t[1:3]
        return mid[0] * 10.0 + mid[1]

    def negative_step(x: float, y: float) -> float:
        r = (x, y, 3.0)[::-1]
        return r[0] + r[2] * 100.0

    def stride_and_tail(x: float) -> float:
        items = [x, 2.0, x * 3.0, 4.0, 5.0]
        return items[::2][1] + items[-2:][0]

    def nested_window(x: float) -> float:
        t = ((x, 1.0), (2.0, x), (3.0, 4.0))
        pair = t[0:2][1]
        return pair[0] + pair[1]

    def string_slice(x: float) -> float:
        return x * 2.0 if "gain"[0:2] == "ga" else x

    for kernel, argsets in (
        (runtime_window, [(3.0, 4.0)]),
        (negative_step, [(1.0, 2.0)]),
        (stride_and_tail, [(2.0,)]),
        (nested_window, [(7.0,)]),
        (string_slice, [(3.0,)]),
    ):
        model = holoso.synthesize(kernel, default_ops(FloatFormat(11, 52)), name=kernel.__name__).numerical_model
        elaborated = model.elaborate()
        for argset in argsets:
            assert float(elaborated.run(*argset)[0]) == kernel(*argset)

    def zero_step(x: float) -> float:
        return (x, 2.0)[::0][0]  # type: ignore[misc,no-any-return]

    with pytest.raises(UnsupportedConstruct, match="slice step cannot be zero"):
        lower(zero_step)

    def runtime_bounds(x: float, n: int) -> float:
        return (x, 2.0, 3.0)[0:n][0]

    with pytest.raises(UnsupportedConstruct, match="call to slice with runtime arguments"):
        lower(runtime_bounds)


def test_starred_assignment_targets_desugar_to_windows() -> None:
    # a, *rest, b = v splits into integer projections plus a list()-converted slice window, with the arity
    # check relaxed to at-least; the star may land anywhere and may absorb zero elements.
    def star_middle(x: float, y: float) -> float:
        a, *mid, b = (x, y, x + y, 4.0)
        return a + mid[0] * 10.0 + mid[1] * 100.0 + b

    def star_first(x: float) -> float:
        *init, last = (1.0, 2.0, x)
        return last * 10.0 + init[1]

    def star_empty(x: float) -> float:
        t: tuple[float, ...] = (x, 2.0)
        a, *rest, b = t
        return a + b + len(rest)

    def star_concat(x: float) -> float:
        t: tuple[float, ...] = (x, 2.0, 3.0)
        a, *rest = t
        grown = rest + [4.0]
        return grown[0] + grown[1]

    for kernel, argsets in (
        (star_middle, [(1.0, 2.0)]),
        (star_first, [(7.0,)]),
        (star_empty, [(3.0,)]),
        (star_concat, [(1.5,)]),
    ):
        model = holoso.synthesize(kernel, default_ops(FloatFormat(11, 52)), name=kernel.__name__).numerical_model
        elaborated = model.elaborate()
        for argset in argsets:
            assert float(elaborated.run(*argset)[0]) == kernel(*argset)

    def star_arity_fail(x: float) -> float:
        t: tuple[float, ...] = (x,)
        a, b, *rest = t
        return a + b

    with pytest.raises(UnsupportedConstruct, match="expected at least 2 values"):
        lower(star_arity_fail)
