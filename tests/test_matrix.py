"""
Statically-shaped matrix/vector support: the ``@`` operator, elementwise aggregate arithmetic, transpose, numpy-style
subscripts, jaxtyping-annotated parameters/returns, matrix state, and ndarray module constants. Structure and
diagnostics are checked on the lowered HIR; numerical behavior is checked black-box through the public API against
numpy executing the very same kernel.
"""

import dataclasses
import sys
import warnings
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from numpy import matmul as _matmul
from jaxtyping import Bool, Float, Float64, Int, Shaped

import holoso
from holoso import FFmaOperator, FloatFormat, UnsupportedConstruct
from holoso._frontend import lower
from holoso._hir import FloatAdd, FloatMul, optimize
from holoso._mir import lower as lower_to_mir

from ._modelref import arith_count as _arith_count, default_ops

# Wide enough that the model's arithmetic coincides with float64 up to the final rounding, so kernels can be compared
# against their own native numpy execution with a tight tolerance.
_FMT = FloatFormat(11, 52)

GAIN = np.array([[0.5, -0.25], [0.125, 1.0]])
COEFFS = np.array([2.0, -1.0, 0.5])
INT_TAPS = np.array([1, 2, 3])

# ndarray module constants for the self-contained stateful filter kernel below.
PROC_NOISE = np.array([[1.0e-4, 0.0], [0.0, 1.0e-2]])
OBS = np.eye(2)
MEAS_VAR = np.array([4.0e-2, 2.5e-1])


class TrackingFilter:
    """
    A self-contained 2-state Kalman-style filter exercising the full matrix feature surface in one stateful kernel:
    matrix/vector parameters and state, ndarray module constants, ``@`` in every shape, transpose, elementwise scalar
    broadcast, an annotated local, a static row loop, and a shaped return. It is ordinary executable numpy, so its own
    native execution is the reference.
    """

    x: Float64[np.ndarray, "2"]
    P: Float64[np.ndarray, "2 2"]

    def __init__(self) -> None:
        self.x = np.zeros(2)
        self.P = np.eye(2) * 10.0

    def update(self, F: Float64[np.ndarray, "2 2"], z: Float64[np.ndarray, "2"]) -> Float64[np.ndarray, "2"]:
        x = F @ self.x
        P = F @ self.P @ F.T + PROC_NOISE
        prediction: Float64[np.ndarray, "2"] = x  # annotated local carrying the a-priori forecast to the return port
        for i in range(2):
            h = OBS[i]
            y = z[i] - h @ x
            s = h @ P @ h + MEAS_VAR[i]  # innovation variance: a runtime scalar divisor
            k = (P @ h) / s
            x = x + k * y
            hp = h @ P
            P = P - np.array([k[0] * hp, k[1] * hp])
        self.x = x
        self.P = P
        return prediction


def _sim(fn: Callable[..., object]) -> holoso.NumericalSimulator:
    return holoso.synthesize(fn, default_ops(_FMT), name="kernel").numerical_model.elaborate()


def _run(sim: holoso.NumericalSimulator, *arrays: np.ndarray | float) -> np.ndarray:
    flat: list[float] = []
    for a in arrays:
        flat += [float(a)] if isinstance(a, float) else np.asarray(a, dtype=np.float64).flatten().tolist()
    return np.array([float(v) for v in sim.run(*flat)])


def _assert_python_matches_holoso(fn: Callable[..., object], *inputs: np.ndarray | float) -> None:
    # Runs the kernel as plain Python and asserts it agrees with Holoso; the Python call also proves the kernel is
    # genuinely valid, runnable Python, so a construct Holoso accepts but Python rejects fails here instead of passing
    # as a spurious "positive" (e.g. ``mat + [1, 2]`` sneaking into a success test).
    want = np.asarray(fn(*inputs)).flatten()
    got = _run(_sim(fn), *inputs)
    assert np.allclose(got, want, rtol=1e-9, atol=1e-300), fn.__name__


# ---------------------------------------------------------------- structure


def test_matmul_shapes_and_port_layout() -> None:
    def mat_vec(a: Float64[np.ndarray, "2 3"], x: Float64[np.ndarray, "3"]) -> Float64[np.ndarray, "2"]:
        return a @ x  # type: ignore[no-any-return]

    hir = lower(mat_vec)
    assert hir.input_names() == ["a_0_0", "a_0_1", "a_0_2", "a_1_0", "a_1_1", "a_1_2", "x_0", "x_1", "x_2"]
    assert [o.name for o in hir.outputs] == ["out_0", "out_1"]
    assert _arith_count(hir, FloatMul) == 6 and _arith_count(hir, FloatAdd) == 4
    _assert_python_matches_holoso(mat_vec, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), np.array([1.0, 0.0, -1.0]))
    _assert_python_matches_holoso(mat_vec, np.array([[0.5, -1.0, 2.0], [3.0, -2.0, 0.25]]), np.array([2.0, -1.0, 0.5]))

    def vec_mat(x: Float64[np.ndarray, "2"], a: Float64[np.ndarray, "2 3"]) -> Float64[np.ndarray, "3"]:
        return x @ a  # type: ignore[no-any-return]

    assert [o.name for o in lower(vec_mat).outputs] == ["out_0", "out_1", "out_2"]
    _assert_python_matches_holoso(vec_mat, np.array([1.0, -2.0]), np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    _assert_python_matches_holoso(vec_mat, np.array([0.5, 2.0]), np.array([[-1.0, 0.25, 3.0], [2.0, -2.0, 1.0]]))

    def dot(v: Float64[np.ndarray, "3"], w: Float64[np.ndarray, "3"]) -> float:
        return v @ w  # type: ignore[no-any-return]

    assert [o.name for o in lower(dot).outputs] == ["out_0"]
    _assert_python_matches_holoso(dot, np.array([1.0, 2.0, 3.0]), np.array([4.0, -5.0, 6.0]))
    _assert_python_matches_holoso(dot, np.array([0.5, -1.0, 2.0]), np.array([2.0, 3.0, -1.0]))

    def mat_mat(a: Float64[np.ndarray, "2 3"], b: Float64[np.ndarray, "3 2"]) -> Float64[np.ndarray, "2 2"]:
        return a @ b  # type: ignore[no-any-return]

    assert [o.name for o in lower(mat_mat).outputs] == ["out_0_0", "out_0_1", "out_1_0", "out_1_1"]
    _assert_python_matches_holoso(
        mat_mat, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    )
    _assert_python_matches_holoso(
        mat_mat, np.array([[0.5, -1.0, 2.0], [3.0, -2.0, 0.25]]), np.array([[2.0, -1.0], [0.5, 3.0], [-2.0, 1.0]])
    )


def test_matmul_rejections() -> None:
    def scalar_operand(a: float, x: Float64[np.ndarray, "2"]) -> Float64[np.ndarray, "2"]:
        return a @ x  # type: ignore[operator, unused-ignore]

    with pytest.raises(UnsupportedConstruct, match="scalar"):
        lower(scalar_operand)

    def dim_mismatch(a: Float64[np.ndarray, "2 3"], x: Float64[np.ndarray, "2"]) -> Float64[np.ndarray, "2"]:
        return a @ x  # type: ignore[no-any-return]

    with pytest.raises(UnsupportedConstruct, match="mismatch"):
        lower(dim_mismatch)

    def ragged(a: float, b: float) -> float:
        # A bare Python list has no ``@`` (a TypeError in Python), so the matrix product is rejected as a list operation
        # before rectangularity is even considered; the ragged literal cannot be wrapped in np.array either.
        return [[a, b], [a]] @ [a, b]  # type: ignore[operator, no-any-return]

    with pytest.raises(UnsupportedConstruct, match="matrix semantics"):
        lower(ragged)

    def three_dee(a: float) -> float:
        return np.array([[[a]]]) @ np.array([a])  # type: ignore[no-any-return]

    with pytest.raises(UnsupportedConstruct, match="1-D or 2-D"):
        lower(three_dee)

    def boolean(v: Float64[np.ndarray, "2"], flag: bool) -> float:
        return v @ np.array([flag, flag])  # type: ignore[no-any-return]

    with pytest.raises(UnsupportedConstruct, match="explicit conversion"):
        lower(boolean)  # the dot's first product hits the scalar boolean-arithmetic doctrine


def test_dot_product_left_fold_contracts_to_fma_chain() -> None:
    # The documented reason for the left-fold dot expansion: with ffma configured, an n-element dot must lower to one
    # fmul plus n-1 ffma (each running-sum add fuses the next single-use product). A balanced tree or product reuse
    # would silently void this, so pin the exact MIR operator population in both configurations.
    def dot(v: Float64[np.ndarray, "3"], w: Float64[np.ndarray, "3"]) -> float:
        return v @ w  # type: ignore[no-any-return]

    def mnemonic_counts(with_fma: bool) -> dict[str, int]:
        ops = default_ops(_FMT)
        if with_fma:
            ops = dataclasses.replace(ops, ffma=FFmaOperator(_FMT))
        mir = lower_to_mir(optimize(lower(dot)), ops)
        counts: dict[str, int] = {}
        for node in mir.nodes.values():
            operator = getattr(node, "operator", None)
            if operator is not None:
                stem = operator.mnemonic.split("_")[0]
                counts[stem] = counts.get(stem, 0) + 1
        return counts

    assert mnemonic_counts(with_fma=True) == {"fmul": 1, "ffma": 2}
    assert mnemonic_counts(with_fma=False) == {"fmul": 3, "fadd": 2}
    _assert_python_matches_holoso(dot, np.array([1.0, 2.0, 3.0]), np.array([4.0, -5.0, 6.0]))
    _assert_python_matches_holoso(dot, np.array([0.5, -1.0, 2.0]), np.array([2.0, 3.0, -1.0]))


def test_np_matmul_call_is_the_operator() -> None:
    def with_operator(a: Float64[np.ndarray, "2 2"], x: Float64[np.ndarray, "2"]) -> Float64[np.ndarray, "2"]:
        return a @ x  # type: ignore[no-any-return]

    def with_call(a: Float64[np.ndarray, "2 2"], x: Float64[np.ndarray, "2"]) -> Float64[np.ndarray, "2"]:
        return np.matmul(a, x)  # type: ignore[no-any-return]

    ops = (_arith_count(lower(with_operator), FloatMul), _arith_count(lower(with_operator), FloatAdd))
    assert ops == (_arith_count(lower(with_call), FloatMul), _arith_count(lower(with_call), FloatAdd))
    for kernel in (with_operator, with_call):
        _assert_python_matches_holoso(kernel, np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([1.0, -1.0]))
        _assert_python_matches_holoso(kernel, np.array([[0.5, -1.0], [2.0, 0.25]]), np.array([2.0, 3.0]))

    def keywords(a: Float64[np.ndarray, "2 2"], x: Float64[np.ndarray, "2"]) -> Float64[np.ndarray, "2"]:
        return np.matmul(a, x, casting="no")  # type: ignore[no-any-return]

    with pytest.raises(UnsupportedConstruct, match="keyword"):
        lower(keywords)


def test_np_matmul_bare_name_import_resolves() -> None:
    def with_bare_name(a: Float64[np.ndarray, "2 2"], x: Float64[np.ndarray, "2"]) -> Float64[np.ndarray, "2"]:
        return _matmul(a, x)  # type: ignore[no-any-return]

    assert [o.name for o in lower(with_bare_name).outputs] == ["out_0", "out_1"]
    _assert_python_matches_holoso(with_bare_name, np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([1.0, -1.0]))
    _assert_python_matches_holoso(with_bare_name, np.array([[0.5, -1.0], [2.0, 0.25]]), np.array([2.0, 3.0]))


def test_augmented_assignment_to_array_is_rejected() -> None:
    # Regression: numpy '+=' / '@=' mutate in place while the frontend rebinds, so an alias would diverge; the array
    # augmented forms must be rejected in favor of the explicit 'x = x + ...' rebind.
    def name_target(v: Float64[np.ndarray, "2"], s: float) -> Float64[np.ndarray, "2"]:
        v += s
        return v

    with pytest.raises(UnsupportedConstruct, match="rebind instead"):
        lower(name_target)

    @dataclasses.dataclass
    class State:
        P: Float64[np.ndarray, "2 2"]

        def step(self, f: Float64[np.ndarray, "2 2"]) -> None:
            self.P @= f

    with pytest.raises(UnsupportedConstruct, match="rebind instead"):
        lower(State(np.eye(2)).step)  # the in-place rejection precedes any matmul question, exactly like *=

    def scalar_ok(a: float, s: float) -> float:
        a += s
        return a

    assert [o.name for o in lower(scalar_ok).outputs] == ["out_0"]


def test_unary_minus_on_boolean_aggregate_is_rejected_with_location() -> None:
    def f(a: bool, b: bool) -> float:
        v = np.array([a, b])
        return (-v)[0]  # type: ignore[no-any-return]

    with pytest.raises(UnsupportedConstruct, match="boolean"):
        lower(f)


def test_elementwise_arithmetic_and_broadcast() -> None:
    def combos(v: Float64[np.ndarray, "2"], w: Float64[np.ndarray, "2"], s: float) -> Float64[np.ndarray, "2"]:
        return (v + w) * s - w / 2.0 + (s - v) * w  # type: ignore[no-any-return]

    assert [o.name for o in lower(combos).outputs] == ["out_0", "out_1"]

    def length_mismatch(v: Float64[np.ndarray, "2"], w: Float64[np.ndarray, "3"]) -> Float64[np.ndarray, "2"]:
        return v + w  # type: ignore[no-any-return]

    with pytest.raises(UnsupportedConstruct, match="mismatched shapes"):
        lower(length_mismatch)

    def structure_mismatch(v: Float64[np.ndarray, "2"], m: Float64[np.ndarray, "2 2"]) -> Float64[np.ndarray, "2"]:
        return v + m  # type: ignore[no-any-return]

    with pytest.raises(UnsupportedConstruct, match="mismatched shapes"):
        lower(structure_mismatch)


def test_python_list_arithmetic_keeps_python_semantics() -> None:
    # A Python list/tuple is never given numpy semantics: list ``+`` means concatenation (implemented structurally,
    # exactly as Python evaluates it), while list ``-`` is a TypeError in Python and list ``*`` by a float count is
    # too, so both are rejected rather than silently reinterpreted. The idiomatic elementwise fix is np.array([...]).
    def list_add(a: float, b: float, c: float, d: float) -> float:
        return ([a, b] + [c, d])[0]  # a valid Python list concatenation, folded structurally

    _assert_python_matches_holoso(list_add, 1.5, -2.0, 0.25, 3.0)

    def list_sub(a: float, b: float, c: float, d: float) -> float:
        return ([a, b] - [c, d])[0]  # type: ignore[operator, no-any-return]

    def list_scale(a: float, b: float, s: float) -> float:
        return ([a, b] * s)[0]  # type: ignore[operator]

    for kernel in (list_sub, list_scale):
        with pytest.raises(UnsupportedConstruct, match="aggregate value"):
            lower(kernel)


def test_np_array_of_ragged_list_is_rejected() -> None:
    # The np.array/asarray/asanyarray factory mirrors numpy, which rejects a ragged array literal.
    def ragged(a: float, b: float) -> float:
        return np.array([[a, b], [a]])[0, 0]  # type: ignore[no-any-return]

    with pytest.raises(UnsupportedConstruct, match="rectangular"):
        lower(ragged)


def test_numpy_only_methods_on_a_list_are_rejected() -> None:
    # `.T`, `.flatten()`, and multi-axis indexing are numpy-array operations undefined on a Python list, so they are
    # rejected on a list literal; wrapping in np.array([...]) makes them valid.
    def transpose(a: float, b: float) -> float:
        return ([a, b].T)[0]  # type: ignore[attr-defined, no-any-return]

    with pytest.raises(UnsupportedConstruct, match="lists are immutable"):
        lower(transpose)

    def flatten(a: float, b: float) -> float:
        return ([[a, b]].flatten())[0]  # type: ignore[attr-defined, no-any-return]

    with pytest.raises(UnsupportedConstruct, match="lists are immutable"):
        lower(flatten)

    def multi_axis(a: float, b: float) -> float:
        m = [[a, b], [b, a]]
        return m[0, 1]  # type: ignore[call-overload, no-any-return]

    with pytest.raises(UnsupportedConstruct, match="multi-axis"):
        lower(multi_axis)


def test_list_of_array_demotes_to_sequence_for_arithmetic() -> None:
    # list(arr)/tuple(arr) produce Python sequences (as in Python), so arithmetic on the result is rejected even though
    # the argument was an array -- guards against the builtins accidentally keeping array semantics.
    def via_list(v: Float64[np.ndarray, "2"], s: float) -> float:
        return (list(v) * s)[0]  # type: ignore[operator, no-any-return]

    with pytest.raises(UnsupportedConstruct, match="aggregate value"):
        lower(via_list)


def test_star_unpack_remainder_is_a_python_list() -> None:
    # Regression: a starred target binds a plain list even when unpacking an array (PEP 3132), so arithmetic on the
    # remainder must be rejected -- it must not inherit the source array's semantics.
    def spread(v: Float64[np.ndarray, "3"]) -> float:
        first, *rest = v
        return (rest + rest)[0]  # type: ignore[no-any-return]

    _assert_python_matches_holoso(spread, np.array([5.0, 6.0, 7.0]))


def test_mismatched_branch_flavor_merge_rejects_array_ops() -> None:
    # Regression: a value that is an array in one arm and a list in the other must not silently gain array semantics
    # (that made acceptance depend on arm order). A numpy op on the merged value is rejected; structural use stays fine.
    def arithmetic(c: bool, a: float, b: float) -> Float64[np.ndarray, "2"]:
        if c:
            v = np.array([a, b])
        else:
            v = [a, b]  # type: ignore[assignment]
        return v * 2.0

    with pytest.raises(UnsupportedConstruct, match="aggregate value"):
        lower(arithmetic)

    def structural(c: bool, a: float, b: float) -> float:
        if c:
            v = np.array([a, b])
        else:
            v = [a, b]  # type: ignore[assignment]
        return v[0]  # type: ignore[no-any-return]

    assert [o.name for o in lower(structural).outputs] == ["out_0"]


def test_state_assignment_flavor_must_match_reset() -> None:
    # Regression: the reset snapshot fixes an attribute's read-back flavor, so storing the other flavor (a list into an
    # ndarray-reset slot) is rejected -- otherwise it would round-trip back as an array and diverge from Python.
    @dataclasses.dataclass
    class ListIntoArray:
        v: np.ndarray

        def step(self, s: float) -> None:
            w = self.v * s
            self.v = [w[0], w[1]]  # type: ignore[assignment]

    with pytest.raises(UnsupportedConstruct, match="numpy array"):
        lower(ListIntoArray(np.array([1.0, 2.0])).step)


def test_boolean_bitwise_operators_lower_in_the_boolean_bank() -> None:
    def bitor(a: bool, b: bool) -> bool:
        return a | b

    def bitand(a: bool, b: bool) -> bool:
        return a & b

    def bitxor(a: bool, b: bool) -> bool:
        return a ^ b

    for kernel in (bitor, bitand, bitxor):
        sim = _sim(kernel)
        for a in (False, True):
            for b in (False, True):
                assert bool(sim.run(a, b)[0]) == kernel(a, b)

    def float_bitor(a: float, b: float) -> float:
        return a | b  # type: ignore[operator,no-any-return]

    with pytest.raises(UnsupportedConstruct, match=r"bitwise/shift operator \| requires integer operands"):
        lower(float_bitor)


def test_unsupported_operator_diagnostic_names_the_operator() -> None:
    # An unsupported operator must be named even when the operands' shapes mismatch, rather than being masked by the
    # shape-mismatch diagnostic.
    def modulo(v: Float64[np.ndarray, "2"], w: Float64[np.ndarray, "3"]) -> Float64[np.ndarray, "2"]:
        return v % w  # type: ignore[no-any-return]

    with pytest.raises(UnsupportedConstruct, match="operator '%' is not supported on arrays"):
        lower(modulo)


def test_transpose_structure() -> None:
    def t(m: Float64[np.ndarray, "2 3"]) -> Float64[np.ndarray, "3 2"]:
        return m.T

    hir = lower(t)
    assert [o.name for o in hir.outputs] == ["out_0_0", "out_0_1", "out_1_0", "out_1_1", "out_2_0", "out_2_1"]
    assert _arith_count(hir, FloatMul) == 0  # a pure reindexing, no hardware
    _assert_python_matches_holoso(t, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    _assert_python_matches_holoso(t, np.array([[-1.0, 0.5, 2.0], [3.0, -2.0, 0.25]]))

    def vector_identity(v: Float64[np.ndarray, "2"]) -> Float64[np.ndarray, "2"]:
        return v.T

    assert [o.name for o in lower(vector_identity).outputs] == ["out_0", "out_1"]
    _assert_python_matches_holoso(vector_identity, np.array([1.0, -2.0]))
    _assert_python_matches_holoso(vector_identity, np.array([0.5, 3.0]))

    def scalar_t(a: float) -> float:
        return a.T  # type: ignore[attr-defined, no-any-return]

    with pytest.raises(UnsupportedConstruct, match="transpose"):
        lower(scalar_t)


def test_state_attribute_named_t_shadows_transpose() -> None:
    @dataclasses.dataclass
    class Holder:
        T: float

        def step(self, a: float) -> float:
            self.T = self.T + a
            return self.T

    hir = lower(Holder(0.0).step)
    assert [s.name for s in hir.state_slots] == ["T"]


def test_state_attributes_named_shape_and_ndim_shadow_the_shape_queries() -> None:
    # ``.shape``/``.ndim`` on the instance keep Python's own attribute-resolution priority, exactly as ``.T`` does:
    # they are state reads, not compile-time shape queries.
    @dataclasses.dataclass
    class Holder:
        shape: float
        ndim: float
        T: float

        def step(self, a: float) -> float:
            self.shape = self.shape + a
            self.ndim = self.ndim * 2.0
            self.T = self.T - a
            return self.shape + self.ndim + self.T

    hir = lower(Holder(0.0, 1.0, 2.0).step)
    assert [s.name for s in hir.state_slots] == ["shape", "ndim", "T"]

    # The reset snapshot is read at synthesis time, so the reference instance must stay untouched until then.
    sim = _sim(Holder(0.5, 1.0, 2.0).step)
    reference = Holder(0.5, 1.0, 2.0)
    for a in (0.25, -1.5, 3.0):
        want = reference.step(a)
        returned, state_shape, state_ndim, state_t = _run(sim, a)
        assert returned == pytest.approx(want)
        assert (state_shape, state_ndim, state_t) == pytest.approx((reference.shape, reference.ndim, reference.T))


def test_numpy_subscripts() -> None:
    def picks(m: Float64[np.ndarray, "2 3"]) -> tuple[float, float, float, float]:
        column = m[:, 2]
        return m[0, 1], m[1][2], column[0], m[1:, 0][0]

    assert [o.name for o in lower(picks).outputs] == ["out_0", "out_1", "out_2", "out_3"]
    _assert_python_matches_holoso(picks, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    _assert_python_matches_holoso(picks, np.array([[-1.0, 0.5, 2.0], [3.0, -2.0, 0.25]]))

    def too_many(m: Float64[np.ndarray, "2 2"]) -> float:
        return m[0, 1, 0]  # type: ignore[no-any-return]

    with pytest.raises(UnsupportedConstruct, match="too many indices"):
        lower(too_many)


def test_multi_axis_index_on_list_is_rejected() -> None:
    # Multi-axis m[i, j] is a numpy-array operation with no meaning on a Python list (list[i, j] is a tuple key, a
    # TypeError), so it must be rejected as a list operation rather than silently indexed. Chained m[i][j] on the same
    # (even ragged) list stays valid (plain list indexing).
    def list_multi_axis(a: float, b: float) -> float:
        m = [[a, b], [a]]
        return m[0, 1]  # type: ignore[call-overload,no-any-return]

    with pytest.raises(UnsupportedConstruct, match="multi-axis"):
        lower(list_multi_axis)

    def ragged_chained(a: float, b: float) -> float:
        m = [[a, b], [a]]
        return m[0][1]

    assert [o.name for o in lower(ragged_chained).outputs] == ["out_0"]


def test_shaped_parameter_annotation_rejections() -> None:
    def symbolic(v: Float64[np.ndarray, "n"]) -> float:
        return v[0]  # type: ignore[no-any-return]

    with pytest.raises(UnsupportedConstruct, match="fixed"):
        lower(symbolic)

    def broadcastable(v: Float64[np.ndarray, "#3"]) -> float:
        return v[0]  # type: ignore[no-any-return]

    with pytest.raises(UnsupportedConstruct, match="fixed"):
        lower(broadcastable)

    def three_dee(v: Float64[np.ndarray, "2 2 2"]) -> float:
        return v[0, 0, 0]  # type: ignore[no-any-return]

    with pytest.raises(UnsupportedConstruct, match="1-D and 2-D"):
        lower(three_dee)

    def boolean(v: Bool[np.ndarray, "2"]) -> float:
        return v[0]  # type: ignore[no-any-return]

    with pytest.raises(UnsupportedConstruct, match="floating-point"):
        lower(boolean)

    def integer(v: Int[np.ndarray, "2"]) -> float:
        return v[0]  # type: ignore[no-any-return]

    with pytest.raises(UnsupportedConstruct, match="floating-point"):
        lower(integer)

    def shape_only(v: Shaped[np.ndarray, "2"]) -> float:
        return v[0]  # type: ignore[no-any-return]

    with pytest.raises(UnsupportedConstruct, match="floating-point"):
        lower(shape_only)

    def shapeless(v: np.ndarray) -> float:
        return v[0]  # type: ignore[no-any-return]

    with pytest.raises(UnsupportedConstruct, match="jaxtyping"):
        lower(shapeless)

    class _FakeArray:  # carries a bare ``dims`` attribute: not a jaxtyping annotation, so not an array at all
        dims = None

    def fake(v: _FakeArray) -> float:
        return 1.0

    with pytest.raises(UnsupportedConstruct, match="expected float, bool, int, a fixed-shape jaxtyping array"):
        lower(fake)


def test_wide_float_dtype_annotation_is_accepted() -> None:
    def f(v: Float[np.ndarray, "2"]) -> float:
        return v[0] + v[1]  # type: ignore[no-any-return]

    assert lower(f).input_names() == ["v_0", "v_1"]


def test_array_annotations_must_be_over_ndarray() -> None:
    # Float64[list, "2"] carries dims like any jaxtyping form; seeding it as an array would swap LIST
    # semantics (duplication on *) for elementwise arithmetic.
    def listy(v: Float64[list[float], "2"], s: float) -> float:
        return v[0] * s

    with pytest.raises(UnsupportedConstruct, match="over np.ndarray"):
        lower(listy)


def test_array_port_budget_is_enforced() -> None:
    def huge(v: Float64[np.ndarray, "8192"]) -> float:
        return v[0]  # type: ignore[no-any-return]

    with pytest.raises(UnsupportedConstruct, match="beyond"):
        lower(huge)


def test_decomposed_parameter_port_collision_is_rejected() -> None:
    def collides(v: Float64[np.ndarray, "2"], v_0: float) -> float:
        return v[1] * v_0  # type: ignore[no-any-return]

    with pytest.raises(UnsupportedConstruct, match="already claims"):
        lower(collides)


def test_array_return_annotation_is_validated() -> None:
    def good(v: Float64[np.ndarray, "2"]) -> Float64[np.ndarray, "2"]:
        return v * 2.0

    assert [o.name for o in lower(good).outputs] == ["out_0", "out_1"]

    def nested(v: Float64[np.ndarray, "2"], flag: bool) -> tuple[Float64[np.ndarray, "2"], bool]:
        return v * 2.0, flag

    assert [o.name for o in lower(nested).outputs] == ["out_0_0", "out_0_1", "out_1"]

    def wrong_shape(v: Float64[np.ndarray, "2"]) -> Float64[np.ndarray, "3"]:
        return v * 2.0

    with pytest.raises(UnsupportedConstruct, match="shape mismatch"):
        lower(wrong_shape)

    def scalar_returned(v: Float64[np.ndarray, "2"]) -> Float64[np.ndarray, "2"]:
        return v[0]  # type: ignore[no-any-return]

    with pytest.raises(UnsupportedConstruct, match="shape mismatch"):
        lower(scalar_returned)

    def boolean_leaves(flag: bool) -> Float64[np.ndarray, "1"]:
        return np.array([flag])

    with pytest.raises(UnsupportedConstruct, match="declared float"):
        lower(boolean_leaves)

    def reflavored(flag: bool) -> Float64[np.ndarray, "1"]:
        # The annotation promises the caller an ndarray; a LIST of matching geometry is an observable
        # reflavoring, not RTL plumbing -- np.array([...]) is the explicit conversion.
        return [1.0]  # type: ignore[return-value]

    with pytest.raises(UnsupportedConstruct, match="shape mismatch"):
        lower(reflavored)


def test_matrix_state_annotation_and_assignment_validation() -> None:
    @dataclasses.dataclass
    class Declared:
        P: Float64[np.ndarray, "2 2"]

        def step(self, a: float) -> None:
            self.P = self.P * a

    with pytest.raises(UnsupportedConstruct, match="declared array type"):
        lower(Declared(np.zeros((3, 3))).step)

    @dataclasses.dataclass
    class Reshaped:
        P: Float64[np.ndarray, "2 2"]

        def step(self, a: float) -> None:
            self.P = self.P[0] * a

    with pytest.raises(UnsupportedConstruct, match="incompatible shape"):
        lower(Reshaped(np.zeros((2, 2))).step)

    @dataclasses.dataclass
    class Empty:
        v: np.ndarray

        def step(self, a: float) -> None:
            self.v = self.v * a

    with pytest.raises(UnsupportedConstruct, match="empty"):
        lower(Empty(np.zeros(0)).step)


def test_state_assignment_element_type_mismatch_is_rejected() -> None:
    # Regression: a bool-leaved value assigned to a float attribute must be rejected, not stored as a float-reset slot
    # whose live-out is a boolean -- which would leave the slot's live-in and live-out at different types.
    @dataclasses.dataclass
    class FloatMatrix:
        P: Float64[np.ndarray, "2 2"]

        def step(self, flag: bool) -> None:
            # A bool-valued array (flavor matches the ndarray slot) so the check reached is the leaf-type one.
            self.P = np.array([[flag, flag], [flag, flag]])

    with pytest.raises(UnsupportedConstruct, match="incompatible type"):
        lower(FloatMatrix(np.zeros((2, 2))).step)

    @dataclasses.dataclass
    class FloatScalar:
        y: float

        def step(self, flag: bool) -> None:
            self.y = flag

    with pytest.raises(UnsupportedConstruct, match="incompatible type"):
        lower(FloatScalar(0.0).step)


def test_arithmetic_on_boolean_operands_is_rejected_with_location() -> None:
    # Regression: bool arithmetic reached HIR construction and raised a bare, location-less ValueError; it must be a
    # source-located UnsupportedConstruct, in both scalar and elementwise-aggregate positions.
    def scalar(a: bool, b: bool) -> float:
        return a + b

    with pytest.raises(UnsupportedConstruct, match="boolean"):
        lower(scalar)

    def aggregate(v: Float64[np.ndarray, "2"], flag: bool) -> Float64[np.ndarray, "2"]:
        return v + np.array([flag, flag])  # type: ignore[no-any-return]

    with pytest.raises(UnsupportedConstruct, match="boolean"):
        lower(aggregate)


def test_matrix_carried_across_while_loop_matches_numpy() -> None:
    # The per-leaf phi spine carries an aggregate around a runtime back-edge like any scalars; the legacy
    # engine rejected this, the structural one computes it.
    def f(m: Float64[np.ndarray, "2 2"], n: float) -> Float64[np.ndarray, "2 2"]:
        x = n
        while x > 0.0:
            m = m * 0.5
            x = x - 1.0
        return m

    _assert_python_matches_holoso(f, np.array([[1.0, 2.0], [3.0, 4.0]]), 3.0)


def test_ndarray_constant_element_folds_in_static_position() -> None:
    # An ndarray-constant element is statically known, so it must fold a branch condition (and serve as a static index)
    # exactly as it folds in value position -- otherwise the branch reads as dynamic and a single-arm return is
    # wrongly rejected.
    def gated(a: float) -> float:
        if _GATE_CONST[1] > 0.0:  # statically true
            return a * 2.0
        return a  # statically dead

    sim = _sim(gated)
    assert float(sim.run(3.0)[0]) == 6.0

    def indexed(v: Float64[np.ndarray, "3"]) -> float:
        return v[_INDEX_CONST[0]]  # type: ignore[no-any-return]  # constant int-array element as a static index

    assert [o.name for o in lower(indexed).outputs] == ["out_0"]

    def chained(a: float) -> float:
        if _GATE_CONST2[0][1] > 0.0:  # chained indexing of a 2-D constant, statically true
            return a * 2.0
        return a

    assert float(_sim(chained).run(3.0)[0]) == 6.0


def test_readonly_ndarray_attribute_element_folds_a_branch() -> None:
    # Regression: a read-only ndarray instance attribute's element must fold a static branch, exactly as a module
    # constant does -- otherwise the guarded write reads as dynamic and wrongly becomes a spurious persistent state slot
    # (changing the synthesized interface).
    @dataclasses.dataclass
    class Filter:
        gain: np.ndarray  # read-only 2-D configuration, never assigned in the method

        def step(self, a: float) -> float:
            out = a
            if self.gain[0, 1] < 0.0:  # statically false: gain[0, 1] == 1.0
                out = a * 2.0  # statically dead
            return out

    hir = lower(Filter(np.array([[0.0, 1.0]])).step)
    assert [s.name for s in hir.state_slots] == []  # no spurious state from the statically-dead write
    assert [o.name for o in hir.outputs] == ["out_0"]


def test_sliced_and_transposed_constant_folds_in_static_position() -> None:
    # Regression: the static evaluator must fold every constant-array operation the value lowerer supports -- including
    # slicing and transpose -- or a statically-known guard reads as dynamic and creates a spurious state slot.
    @dataclasses.dataclass
    class WithSlice:
        y: float

        def step(self, a: float) -> float:
            if _GATE_CONST[0:2][1] > 0.0:  # statically true
                return a * 2.0
            self.y = a  # statically dead
            return self.y

    hir_slice = lower(WithSlice(0.0).step)
    assert [s.name for s in hir_slice.state_slots] == []
    assert [o.name for o in hir_slice.outputs] == ["out_0"]

    @dataclasses.dataclass
    class WithTranspose:
        y: float

        def step(self, a: float) -> float:
            if _GATE_CONST2.T[1, 0] < 0.0:  # _GATE_CONST2.T[1, 0] == _GATE_CONST2[0, 1] == 1.0, statically false
                self.y = a  # statically dead
            return a

    hir_t = lower(WithTranspose(0.0).step)
    assert [s.name for s in hir_t.state_slots] == []
    assert [o.name for o in hir_t.outputs] == ["out_0"]

    @dataclasses.dataclass
    class WithFlatten:
        y: float

        def step(self, a: float) -> float:
            if _GATE_CONST2.flatten()[1] > 0.0:  # flatten()[1] == 1.0, statically true
                return a * 2.0
            self.y = a  # statically dead
            return self.y

    hir_f = lower(WithFlatten(0.0).step)
    assert [s.name for s in hir_f.state_slots] == []
    assert [o.name for o in hir_f.outputs] == ["out_0"]

    def via_identity(a: float) -> float:
        if np.asarray(_GATE_CONST2)[0, 1] > 0.0:  # array-identity wrapper then index, statically true
            return a * 2.0
        return a

    assert float(_sim(via_identity).run(3.0)[0]) == 6.0


def test_transpose_of_matrix_state_attribute() -> None:
    # Coverage: ``self.P.T`` transposes state (the chained case of the ``.T``-vs-``self.T`` resolution), distinct from
    # the ``self.T`` state-read carve-out.
    @dataclasses.dataclass
    class Holder:
        P: Float64[np.ndarray, "2 2"]

        def step(self, s: float) -> Float64[np.ndarray, "2 2"]:
            return self.P.T * s

    sim = holoso.synthesize(
        Holder(np.array([[1.0, 2.0], [3.0, 4.0]])).step, default_ops(_FMT), name="pt"
    ).numerical_model.elaborate()
    got = _run(sim, 2.0).reshape(2, 2)
    assert np.allclose(got, np.array([[1.0, 2.0], [3.0, 4.0]]).T * 2.0)


def test_unary_plus_rejects_boolean_but_is_identity_on_floats() -> None:
    # Regression: unary plus skipped the boolean guard the other arithmetic operators apply, silently passing a bool
    # through (Python's +True is int 1, which has no runtime type here).
    def scalar(flag: bool) -> float:
        return +flag

    with pytest.raises(UnsupportedConstruct, match="boolean"):
        lower(scalar)

    def aggregate(a: bool, b: bool) -> Float64[np.ndarray, "2"]:
        return +np.array([a, b])

    with pytest.raises(UnsupportedConstruct, match="boolean"):
        lower(aggregate)

    def floats(v: Float64[np.ndarray, "2"]) -> Float64[np.ndarray, "2"]:
        return +v

    assert [o.name for o in lower(floats).outputs] == ["out_0", "out_1"]


def test_ndarray_module_constant_behavior() -> None:
    # A boolean constant element is a bool VALUE (the width collapse admits the array); returning it as a
    # declared float is an honest contract mismatch. A 3-D constant is ordinary bookkeeping: its elements
    # fold through the rank-N gather exactly as numpy indexes them.
    def boolean(a: float) -> float:
        return _BOOL_CONST[0]  # type: ignore[no-any-return]

    with pytest.raises(UnsupportedConstruct, match="returns bool"):
        lower(boolean)

    def three_dee(a: float) -> float:
        return a + float(_CUBE_CONST[0, 0, 0])

    assert float(_sim(three_dee).run(2.0)[0]) == 2.0  # the cube is zeros


def test_ndarray_subclass_constant_and_state_are_rejected() -> None:
    # Regression: an ndarray subclass (np.matrix) redefines operators (``*`` is matmul), so folding it as a plain array
    # would silently diverge from its own Python semantics; it must be rejected, both as a module constant (an
    # unadmitted reference, so its subscript refuses) and as a state reset.
    def constant(a: float) -> float:
        return _MATRIX_CONST[0, 1] + a  # type: ignore[no-any-return]

    with pytest.raises(UnsupportedConstruct, match="subscript of an object"):
        lower(constant)

    @dataclasses.dataclass
    class Stateful:
        P: np.ndarray

        def step(self, a: float) -> None:
            self.P = self.P * a

    with pytest.raises(UnsupportedConstruct, match="plain numpy array"):
        lower(Stateful(_np_matrix()).step)


def test_power_of_boolean_is_rejected_with_location() -> None:
    # Regression: '**' bypassed the boolean-operand guard applied to the other arithmetic operators, raising a bare
    # ValueError (flag**2) or silently returning the bool (flag**1) instead of a source-located UnsupportedConstruct.
    def squared(flag: bool) -> float:
        return flag**2

    with pytest.raises(UnsupportedConstruct, match="boolean"):
        lower(squared)

    def first_power(flag: bool) -> float:
        return flag**1

    with pytest.raises(UnsupportedConstruct, match="boolean"):
        lower(first_power)


def test_boolean_operand_to_float_builtin_or_intrinsic_is_rejected_with_location() -> None:
    # Regression: abs/min/max/round and the math/numpy float intrinsics passed a boolean operand straight to HIR
    # construction, raising a bare location-less ValueError; each must reject it with a source-located error.
    def with_abs(flag: bool) -> float:
        return abs(flag)

    def with_min(flag: bool, x: float) -> float:
        return min(flag, x)

    def with_round(flag: bool) -> float:
        return round(flag)

    def with_floor(flag: bool) -> float:
        return np.floor(flag)  # type: ignore[no-any-return]

    for kernel in (with_abs, with_min, with_round, with_floor):
        with pytest.raises(UnsupportedConstruct, match="boolean"):
            lower(kernel)


def _np_matrix() -> np.ndarray:
    # np.matrix is an ndarray subclass with different operator semantics; construct it under a warning filter since it
    # is deliberately used to check that such subclasses are rejected.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PendingDeprecationWarning)
        return np.matrix([[1.0, 2.0], [3.0, 4.0]])


_BOOL_CONST = np.array([True, False])
_ZERO_D_CONST = np.array(5.0)
_EMPTY_CONST = np.array([])
_CUBE_CONST = np.zeros((2, 2, 2))
_MATRIX_CONST = _np_matrix()
_GATE_CONST = np.array([0.0, 1.0])
_GATE_CONST2 = np.array([[0.0, 1.0], [1.0, 0.0]])
_INDEX_CONST = np.array([2, 0, 1])


# ---------------------------------------------------------------- elementwise arithmetic (stage 9a-1)
# Array parameters await stage 9b, so these kernels source their arrays from module constants and derive
# runtime-leaf arrays by chaining an elementwise op onto a runtime scalar.


def test_module_constant_elementwise_matches_numpy() -> None:
    def scaled(s: float) -> tuple[float, float, float]:
        v = COEFFS * s
        return v[0], v[1], v[2]

    def reflected(s: float) -> tuple[float, float, float]:
        v = s - COEFFS
        return v[0], v[1], v[2]

    def chained(s: float, t: float) -> tuple[float, float, float]:
        v = (COEFFS * s) * t  # the left operand carries runtime leaves, not a constant snapshot
        return v[0], v[1], v[2]

    def paired(s: float, t: float) -> tuple[float, float, float]:
        v = (COEFFS * s) + (COEFFS - t)  # array x array with runtime leaves on both sides
        return v[0], v[1], v[2]

    def divided(s: float) -> tuple[float, float, float, float, float, float]:
        v = COEFFS / s
        w = s / COEFFS
        return v[0], v[1], v[2], w[0], w[1], w[2]

    def matrix(s: float) -> tuple[float, float, float, float]:
        m = GAIN * s
        return m[0][0], m[0][1], m[1][0], m[1][1]

    _assert_python_matches_holoso(scaled, 2.5)
    _assert_python_matches_holoso(reflected, -1.25)
    _assert_python_matches_holoso(chained, 2.0, -0.5)
    _assert_python_matches_holoso(paired, 3.0, 0.75)
    _assert_python_matches_holoso(divided, 4.0)
    _assert_python_matches_holoso(matrix, -2.0)


def test_elementwise_known_operands_fold_statically() -> None:
    # A fully static elementwise result folds leafwise, so it drives branches exactly like a module constant:
    # the guarded write below is statically dead and must not become a spurious state slot.
    @dataclasses.dataclass
    class Gated:
        y: float

        def step(self, a: float) -> float:
            if (COEFFS * 2.0)[1] > 0.0:  # (-1.0) * 2.0: statically false
                self.y = a  # statically dead
            return a

    hir = lower(Gated(0.0).step)
    assert [s.name for s in hir.state_slots] == []
    assert [o.name for o in hir.outputs] == ["out_0"]


def test_integer_elementwise_folds_and_promotes() -> None:
    # Runtime integer products await the integer sprint (no integer hardware yet), so the runtime-leaf coverage
    # here is the promoting form: / yields float even on all-integer operands, as in Python and numpy.
    def scaled_by_known(s: float) -> float:
        v = INT_TAPS * 2  # all-Known integer leaves fold leafwise, exactly as numpy computes them
        return s + float(v[2])

    _assert_python_matches_holoso(scaled_by_known, 7.0)

    def true_division(s: float) -> tuple[float, float, float]:
        v = INT_TAPS / int(s)
        return v[0], v[1], v[2]

    ops = dataclasses.replace(default_ops(_FMT), fround=holoso.FRoundOperator(_FMT))
    sim = holoso.synthesize(true_division, ops, name="kernel").numerical_model.elaborate()
    got = np.array([float(v) for v in sim.run(8.0)])
    assert np.allclose(got, np.asarray(true_division(8.0)), rtol=1e-12, atol=1e-300)


def test_elementwise_unary_matches_numpy() -> None:
    def negated(s: float) -> tuple[float, float, float]:
        v = -(COEFFS * s)
        return v[0], v[1], v[2]

    def positive(s: float) -> tuple[float, float, float]:
        v = +(COEFFS * s)
        return v[0], v[1], v[2]

    _assert_python_matches_holoso(negated, 1.5)
    _assert_python_matches_holoso(positive, -0.25)


def test_zero_dimensional_operands_yield_the_scalar_sort() -> None:
    # numpy arithmetic on a 0-d array returns np.float64/np.int64 (the scalar sort, unary included), never a 0-d
    # array; the float() below is the observable -- it lowers only if the result is a genuine scalar fact.
    def scaled(s: float) -> float:
        return float(_ZERO_D_CONST * s)

    def negated(s: float) -> float:
        return s + float(-_ZERO_D_CONST)

    def paired(s: float) -> float:
        return float(_ZERO_D_CONST + _ZERO_D_CONST) * s

    _assert_python_matches_holoso(scaled, 2.5)
    _assert_python_matches_holoso(negated, -1.0)
    _assert_python_matches_holoso(paired, 0.5)


def test_elementwise_rejections_are_located() -> None:
    def mismatched(s: float) -> float:
        v = COEFFS + _GATE_CONST  # (3,) + (2,)
        return v[0] * s  # type: ignore[no-any-return]

    with pytest.raises(UnsupportedConstruct, match="mismatched shapes"):
        lower(mismatched)

    def mixed_rank(s: float) -> float:
        v = GAIN + _GATE_CONST  # (2, 2) + (2,): numpy would broadcast; only a scalar broadcasts here
        return v[0][0] * s  # type: ignore[no-any-return]

    with pytest.raises(UnsupportedConstruct, match="mismatched shapes"):
        lower(mixed_rank)

    def augmented(s: float) -> float:
        v = COEFFS
        v = v * s
        v *= s
        return v[0]  # type: ignore[no-any-return]

    with pytest.raises(UnsupportedConstruct, match="rebind"):
        lower(augmented)

    def modulo(s: float) -> float:
        v = COEFFS % s
        return v[0]  # type: ignore[no-any-return]

    with pytest.raises(UnsupportedConstruct, match="'%'"):
        lower(modulo)

    def power(s: float) -> float:
        v = COEFFS**s
        return v[0]  # type: ignore[no-any-return]

    with pytest.raises(UnsupportedConstruct, match=r"'\*\*'"):
        lower(power)

    def product(s: float) -> float:
        v = COEFFS @ [1.0, 2.0, s]  # a list operand never acquires matrix semantics
        return float(v * s)

    with pytest.raises(UnsupportedConstruct, match="matrix semantics"):
        lower(product)

    def flag_scalar(s: float) -> float:
        # Boolean ARRAY operands cannot exist yet (bool ndarrays are outside the admitted domain until the
        # np.array factory lands), so the reachable boolean-rejection corner is the scalar side.
        v = COEFFS * (s > 0.0)
        return v[0]  # type: ignore[no-any-return]

    with pytest.raises(UnsupportedConstruct, match="explicit conversion"):
        lower(flag_scalar)

    def runtime_integer(s: float) -> float:
        v = INT_TAPS * int(s)  # the scalar integer datapath saturates where numpy int64 wraps
        return float(v[0])

    with pytest.raises(UnsupportedConstruct, match="cast to float"):
        lower(runtime_integer)

    def oversized_int_constant(s: float) -> float:
        v = (COEFFS * s) + 10**400  # numpy raises OverflowError converting the constant, before any element
        return v[0]  # type: ignore[no-any-return]

    def oversized_on_empty(s: float) -> float:
        v = _EMPTY_CONST + 10**400  # the conversion is array-wide: numpy raises even with zero elements
        return s + float(len(v))

    def oversized_for_int64(s: float) -> float:
        v = INT_TAPS + 2**63  # numpy: "Python int too large to convert to C long"
        return s + float(v[0])

    for kernel in (oversized_int_constant, oversized_on_empty, oversized_for_int64):
        with pytest.raises(UnsupportedConstruct, match="OverflowError"):
            lower(kernel)


# ---------------------------------------------------------------- array factories (stage 9a-2a)


def test_np_array_factory_over_runtime_values_matches_numpy() -> None:
    def vector(a: float, b: float, s: float) -> tuple[float, float]:
        v = np.array([a, b]) * s
        return v[0], v[1]

    def identity(s: float) -> tuple[float, float, float]:
        v = np.asarray(COEFFS * s)  # asarray over a runtime-leaf array is the identity copy
        return v[0], v[1], v[2]

    def nested(a: float, b: float) -> tuple[float, float]:
        m = np.asarray([[a], [b]])  # ListLayout(ListLayout(ATOM)) becomes a (2, 1) array
        return m[0][0], m[1][0]

    def mixed_flavor(a: float, b: float, c: float, d: float) -> float:
        m = np.array([(a, b), [c, d]])  # numpy accepts mixed tuple/list nesting as one rectangular block
        return m[1][0]  # type: ignore[no-any-return]

    def known_int_leaf(a: float) -> tuple[float, float]:
        v = np.array([a, 1]) * 2.0  # the Known integer leaf becomes np.float64(1.0), as numpy discovers
        return v[0], v[1]

    _assert_python_matches_holoso(vector, 1.5, -2.0, 3.0)
    _assert_python_matches_holoso(identity, 0.5)
    _assert_python_matches_holoso(nested, 2.0, -1.0)
    _assert_python_matches_holoso(mixed_flavor, 1.0, 2.0, 3.0, 4.0)
    _assert_python_matches_holoso(known_int_leaf, 2.5)


def test_np_array_factory_subset_rejections() -> None:
    def oversized_known_int(a: float) -> float:
        v = np.array([a, 10**100])  # numpy would build an OBJECT array here
        return v[0]  # type: ignore[no-any-return]

    def uint64_range_int(a: float) -> float:
        v = np.array([a, 2**63])  # numpy would silently promote through uint64 to float64
        return v[0]  # type: ignore[no-any-return]

    for kernel in (oversized_known_int, uint64_range_int):
        with pytest.raises(UnsupportedConstruct, match="64-bit"):
            lower(kernel)

    def runtime_integers(a: float, b: float) -> float:
        v = np.array([int(a), int(b)])  # an integer array with runtime leaves cannot lower yet
        return float(v[0])

    with pytest.raises(UnsupportedConstruct, match="cast to float"):
        lower(runtime_integers)

    def bool_mix(a: float, b: float) -> float:
        v = np.array([a > 0.0, b])  # numpy would promote the bool; the subset rejects the mix
        return v[1]  # type: ignore[no-any-return]

    with pytest.raises(UnsupportedConstruct, match="mixes boolean"):
        lower(bool_mix)


def test_bool_array_construction_and_arithmetic_rejections() -> None:
    # A boolean-dtype array is constructible from runtime comparisons; its arithmetic keeps the scalar
    # doctrine's explicit-conversion rejection, unary included (numpy itself refuses unary +/- on bools).
    def scaled(a: float, b: float) -> float:
        v = np.array([a > 0.0, b > 0.0]) * 2.0
        return v[0]  # type: ignore[no-any-return]

    def negated(a: float, b: float) -> float:
        v = -np.array([a > 0.0, b > 0.0])
        return float(v[0])

    for kernel in (scaled, negated):
        with pytest.raises(UnsupportedConstruct, match="explicit conversion"):
            lower(kernel)

    def truth(a: float, b: float) -> float:
        if np.array([a > 0.0, b > 0.0]):
            return a
        return b

    with pytest.raises(UnsupportedConstruct, match="ambiguous"):
        lower(truth)


def test_runtime_array_shape_metadata_folds_statically() -> None:
    # .ndim/.shape/.size are layout-determined, so they fold on RUNTIME-leaf arrays exactly as on constants:
    # the guarded write below is statically dead and must not become a spurious state slot.
    @dataclasses.dataclass
    class Gated:
        y: float

        def step(self, a: float) -> float:
            g = COEFFS * a
            if g.ndim != 1 or g.shape[0] != 3 or g.size != 3 or len(g) != 3:  # statically false
                self.y = a  # statically dead
            return a * 2.0

    hir = lower(Gated(0.0).step)
    assert [s.name for s in hir.state_slots] == []
    assert [o.name for o in hir.outputs] == ["out_0"]


# ---------------------------------------------------------------- gathers, unpacking, iteration (stage 9a-2b)


def test_array_slices_and_gathers_match_numpy() -> None:
    def window(s: float) -> tuple[float, float]:
        v = (COEFFS * s)[0:2]
        return v[0], v[1]

    def reversed_view(s: float) -> tuple[float, float, float]:
        v = (COEFFS * s)[::-1]
        return v[0], v[1], v[2]

    def column(s: float) -> tuple[float, float]:
        c = (GAIN * s)[:, 1]
        return c[0], c[1]

    def tail_rows(s: float) -> float:
        t = (GAIN * s)[1:, 0]
        return t[0]  # type: ignore[no-any-return]

    def element(s: float) -> float:
        return (GAIN * s)[0, 1]  # type: ignore[no-any-return]

    def rows_kept(s: float) -> tuple[float, float]:
        m = (GAIN * s)[0:1]  # a leading-axis window of a matrix keeps the trailing axis
        return m[0][0], m[0][1]

    for kernel, args in (
        (window, (2.5,)),
        (reversed_view, (-1.5,)),
        (column, (3.0,)),
        (tail_rows, (0.5,)),
        (element, (2.0,)),
        (rows_kept, (-0.25,)),
    ):
        _assert_python_matches_holoso(kernel, *args)


def test_array_gather_rejections_are_located() -> None:
    def too_many(s: float) -> float:
        return (GAIN * s)[0, 1, 0]  # type: ignore[no-any-return]

    with pytest.raises(UnsupportedConstruct, match="too many indices"):
        lower(too_many)

    def bool_key(s: float) -> float:
        # numpy treats a boolean inside a tuple key as ADVANCED indexing, never as m[1, 0].
        return (GAIN * s)[True, 0]  # type: ignore[no-any-return]

    with pytest.raises(UnsupportedConstruct, match="boolean index"):
        lower(bool_key)

    def out_of_range(s: float) -> float:
        return (GAIN * s)[0, 5]  # type: ignore[no-any-return]

    with pytest.raises(UnsupportedConstruct, match="out of range"):
        lower(out_of_range)


def test_star_unpack_of_arrays() -> None:
    def helper(a: float, b: float, c: float) -> float:
        return a * 100.0 + b * 10.0 + c

    def spread(s: float) -> float:
        return helper(*(COEFFS * s))

    _assert_python_matches_holoso(spread, 2.0)

    def zero_d(s: float) -> float:
        # NOTE arithmetic on a 0-d array yields the scalar sort, so only the direct constant exercises this.
        return s + helper(1.0, 2.0, *_ZERO_D_CONST)

    with pytest.raises(UnsupportedConstruct, match="0-dimensional"):
        lower(zero_d)


def test_iteration_over_runtime_aggregates_matches_numpy() -> None:
    def over_vector(s: float) -> float:
        acc = 0.0
        for x in COEFFS * s:
            acc = acc + x * x
        return acc

    def over_rows(s: float) -> float:
        acc = 0.0
        for row in GAIN * s:
            acc = acc + row[0] - row[1]
        return acc

    def over_tuple(a: float, b: float) -> float:
        acc = 1.0
        for x in (a * 2.0, b + 1.0, a - b):
            acc = acc * x
        return acc

    _assert_python_matches_holoso(over_vector, 1.5)
    _assert_python_matches_holoso(over_rows, -2.0)
    _assert_python_matches_holoso(over_tuple, 0.5, 3.0)


# ---------------------------------------------------------------- flatten/ravel (stage 9a-2c)


def test_flatten_and_ravel_relayout_runtime_arrays() -> None:
    def flattened(a: float, b: float, c: float) -> tuple[float, float, float]:
        m = np.asarray([[a], [b], [c]]).flatten()  # the ekf1 shape: (3, 1) -> (3,) in C order
        return m[0], m[1], m[2]

    def raveled(s: float) -> tuple[float, float, float, float]:
        m = (GAIN * s).ravel()
        return m[0], m[1], m[2], m[3]

    def stored_method(s: float) -> float:
        pick = (COEFFS * s).flatten  # a bound method is a value; calling it later is ordinary Python
        v = pick()
        return v[2]  # type: ignore[no-any-return]

    def known_leaves(s: float) -> float:
        return s + float(GAIN.flatten()[2])  # an all-Known flatten folds leafwise

    _assert_python_matches_holoso(flattened, 1.5, -2.0, 0.25)
    _assert_python_matches_holoso(raveled, 2.0)
    _assert_python_matches_holoso(stored_method, -1.0)
    _assert_python_matches_holoso(known_leaves, 3.0)


def test_flatten_order_arguments_are_rejected() -> None:
    def fortran(s: float) -> float:
        v = (GAIN * s).flatten("F")  # a non-C order observes strides the layout deliberately discards
        return v[0]  # type: ignore[no-any-return]

    with pytest.raises(UnsupportedConstruct, match="default C order"):
        lower(fortran)


# ---------------------------------------------------------------- comparisons, reductions, reshape (9d-1)


def test_array_comparisons_yield_boolean_masks() -> None:
    def masked(s: float) -> tuple[float, float, float]:
        mask = (COEFFS * s) >= 0.5  # elementwise comparison with a scalar broadcast
        return float(mask[0]), float(mask[1]), float(mask[2])

    def paired(s: float) -> float:
        hits = (COEFFS * s) > (COEFFS * 0.5)  # array-with-array comparison, same shape
        if hits[1]:
            return s
        return -s

    _assert_python_matches_holoso(masked, 2.0)
    _assert_python_matches_holoso(paired, 1.0)

    def mismatched(s: float) -> float:
        wrong = (COEFFS * s) > _GATE_CONST  # (3,) vs (2,)
        return float(wrong[0])

    with pytest.raises(UnsupportedConstruct, match="mismatched shapes"):
        lower(mismatched)


def test_array_reductions_match_numpy() -> None:
    def peak(s: float) -> float:
        return float(np.max(COEFFS * s))

    def centered(s: float) -> tuple[float, float, float]:
        v = COEFFS * s
        w = v - float(np.mean(v))  # the controller's zero-mean shape
        return w[0], w[1], w[2]

    # The mean's constant division strength-reduces to a reciprocal multiply under the fastmath doctrine, so
    # an exactly-zero centered element carries one ulp of reciprocal rounding; the tolerance names that.
    sorted_ops = dataclasses.replace(default_ops(_FMT), fsort=holoso.FSortOperator(_FMT))
    for kernel, argument in ((peak, -2.0), (peak, 3.0), (centered, 2.0)):
        sim = holoso.synthesize(kernel, sorted_ops, name="kernel").numerical_model.elaborate()
        got = np.array([float(v) for v in sim.run(argument)])
        assert np.allclose(got, np.asarray(kernel(argument)).flatten(), rtol=1e-12, atol=1e-12)

    def matrix_reduction(s: float) -> float:
        return float(np.max(GAIN * s))  # only the 1-D form is supported

    with pytest.raises(UnsupportedConstruct, match="1-D"):
        lower(matrix_reduction)


def test_reshape_is_a_static_relayout() -> None:
    def widen(a: float, b: float, c: float) -> tuple[float, float, float]:
        column = np.array([a, b, c]).reshape((3, 1))  # the controller's output shape
        flat = column.reshape(3)  # and the no-op round trip
        return flat[0], column[1][0], flat[2]

    _assert_python_matches_holoso(widen, 1.5, -2.0, 0.25)

    def wrong_count(s: float) -> float:
        v = (COEFFS * s).reshape((2, 2))
        return v[0][0]  # type: ignore[no-any-return]

    with pytest.raises(UnsupportedConstruct, match="cannot reshape"):
        lower(wrong_count)

    def inferred(s: float) -> float:
        v = (COEFFS * s).reshape((-1, 1))
        return v[0][0]  # type: ignore[no-any-return]

    with pytest.raises(UnsupportedConstruct, match="spell the shape"):
        lower(inferred)


def test_reshape_argument_arity_follows_python() -> None:
    # reshape() with no arguments is a TypeError in Python even for a one-cell array (math.prod(()) == 1
    # made the size check pass); reshape(()) on a one-cell array is VALID and yields the 0-d scalar sort
    # on extraction.
    def no_arguments(s: float) -> float:
        v = np.array([s]).reshape()  # type: ignore[call-overload]
        return v[0]  # type: ignore[no-any-return]

    with pytest.raises(UnsupportedConstruct, match="requires a shape"):
        lower(no_arguments)

    def empty_shape(s: float) -> float:
        v = np.array([s]).reshape(())
        return float(v)

    _assert_python_matches_holoso(empty_shape, 2.5)


def test_helper_annotations_are_documentation() -> None:
    # Doctrine: a callee's annotations are documentation, never a lowering directive -- an UNANNOTATED
    # helper must inline from its call-site facts (parameter and return validation is root-only).
    def bare_helper(v, gain):  # type: ignore[no-untyped-def]
        return v * gain + 1.0

    def kernel(x: float) -> float:
        return bare_helper(x, 2.0)  # type: ignore[no-any-return, no-untyped-call]

    _assert_python_matches_holoso(kernel, 3.0)


def test_dtype_float_casts_leaves_explicitly() -> None:
    def from_flags(a: float, b: float, c: float) -> tuple[float, float, float]:
        switches = (a > 0.0, b > 0.0, c > 0.0)
        v = np.array(switches, dtype=float) * 2.0  # the controller's balance-step shape
        return v[0], v[1], v[2]

    def from_mixed(a: float) -> tuple[float, float]:
        v = np.asarray([a > 0.0, 3], dtype=float)  # an explicit dtype IS the conversion
        return v[0], v[1]

    _assert_python_matches_holoso(from_flags, 1.0, -1.0, 2.0)
    _assert_python_matches_holoso(from_mixed, -0.5)

    def implicit(a: float) -> float:
        v = np.array([a > 0.0, 3.0])  # NO dtype: the implicit widening keeps rejecting
        return v[0]  # type: ignore[no-any-return]

    with pytest.raises(UnsupportedConstruct, match="mixes boolean"):
        lower(implicit)


# ---------------------------------------------------------------- record ports (9d-2) and stage 10


def test_record_ports_decompose_by_field_path() -> None:
    @dataclasses.dataclass(frozen=True)
    class Gains:
        p: float
        taps: Float64[np.ndarray, "2"]
        enabled: bool

    @dataclasses.dataclass(frozen=True)
    class Report:
        drive: float
        flags: tuple[bool, bool]

    def kernel(g: Gains, x: float) -> Report:
        scale = g.p * x if g.enabled else x
        return Report(drive=scale + float(g.taps[0]) - float(g.taps[1]), flags=(scale > 0.0, g.enabled))

    hir = lower(kernel)
    assert hir.input_names() == ["g_p", "g_taps_0", "g_taps_1", "g_enabled", "x"]
    assert [o.name for o in hir.outputs] == ["out_drive", "out_flags_0", "out_flags_1"]


def test_record_contract_rejections_are_located() -> None:
    with pytest.raises(UnsupportedConstruct, match="recursively contains itself"):
        lower(_recursive_record_kernel)

    @dataclasses.dataclass(frozen=True)
    class Wide:
        value: str

    def stringly(v: Wide) -> float:
        return 1.0

    with pytest.raises(UnsupportedConstruct, match="expected float, bool, int"):
        lower(stringly)

    @dataclasses.dataclass(frozen=True)
    class Pair:
        a: float
        b: float

    @dataclasses.dataclass(frozen=True)
    class Other:
        a: float
        b: float

    def wrong_record(x: float) -> Pair:
        return Other(x, x)  # type: ignore[return-value]

    with pytest.raises(UnsupportedConstruct, match="declared record 'Pair'"):
        lower(wrong_record)


def test_ambiguous_record_port_paths_refuse_at_build() -> None:
    # The underscore join is not injective: a field 'a_b' beside a nested record field 'a.b' renders the
    # same cell name. Until the injective path codec lands, the ambiguity is a located refusal, never a
    # pair of duplicate ports failing deep in synthesis.
    with pytest.raises(UnsupportedConstruct, match="already claims"):
        lower(_ambiguous_record_kernel)


def test_parity_registry_is_empty() -> None:
    # Stage 10: every example the catalogue and the off-catalogue suites know lowers through the new front
    # end; the registry exists only as the (empty) assertion point.
    from tests._examples import FIR_PARITY_PENDING

    assert FIR_PARITY_PENDING == {}


# ---------------------------------------------------------------- numeric width collapse


def test_narrow_numpy_widths_collapse_to_category_carriers() -> None:
    # Width is immaterial to the domain: narrow scalars/arrays admit by exact embedding into bool/int64/float64.
    # The float32 values below are exactly representable, so plain-Python numpy execution agrees bit-for-bit.
    def f32_array(s: float) -> tuple[float, float, float]:
        v = _F32_GAINS * s
        return v[0], v[1], v[2]

    def i16_fold(s: float) -> float:
        v = _I16_TAPS * 2  # all-Known integer leaves fold exactly through the int64 carrier
        return s + float(v[1])

    def u64_small(s: float) -> float:
        return s + float(_U64_SMALL[0])

    def narrow_scalars(s: float) -> float:
        return float(s * _F32_SCALAR + _I8_SCALAR)

    _assert_python_matches_holoso(f32_array, 2.5)
    _assert_python_matches_holoso(i16_fold, 7.0)
    _assert_python_matches_holoso(u64_small, 1.0)
    _assert_python_matches_holoso(narrow_scalars, -0.5)

    def bool_const_array(s: float) -> float:
        v = _BOOL_CONST * s  # a bool ndarray constant now admits, and its arithmetic keeps the scalar doctrine
        return v[0]  # type: ignore[no-any-return]

    with pytest.raises(UnsupportedConstruct, match="explicit conversion"):
        lower(bool_const_array)


def test_unembeddable_numpy_values_stay_non_static() -> None:
    # No exact 64-bit embedding exists, so these stay outside the value domain (references, rejected at use).
    def u64_huge(s: float) -> float:
        return s + float(_U64_HUGE[0])

    def longdouble(s: float) -> float:
        return s + float(_LONGDOUBLE[0])

    for kernel in (u64_huge, longdouble):
        with pytest.raises(UnsupportedConstruct, match="subscript of an object"):
            lower(kernel)


_F32_GAINS = np.array([0.5, -0.25, 2.0], dtype=np.float32)
_I16_TAPS = np.array([3, -2, 5], dtype=np.int16)
_U64_SMALL = np.array([7, 2], dtype=np.uint64)
_U64_HUGE = np.array([2**63, 1], dtype=np.uint64)
_LONGDOUBLE = np.array([1.0], dtype=np.longdouble)
_F32_SCALAR = np.float32(0.25)
_I8_SCALAR = np.int8(-3)


# ---------------------------------------------------------------- 9a review-round regressions


def test_datetime_dtypes_stay_non_static() -> None:
    # timedelta64 satisfies np.issubdtype(_, np.integer); admitting it as a plain integer would drop the unit
    # semantics (numpy rounds a scaled duration to integral nanoseconds where the int carrier keeps fractions).
    def kernel(scale: float) -> float:
        return float((_DURATION_CONST * scale)[0])

    with pytest.raises(UnsupportedConstruct, match="dtype timedelta64"):
        lower(kernel)


def test_byte_swapped_unsigned_boundary_holds() -> None:
    # dtype equality misses the big-endian spelling of uint64, so 2**63 wrapped negative through astype.
    def oversized(x: float) -> float:
        return x + float(_BE_U64_HUGE[0])

    with pytest.raises(UnsupportedConstruct, match="subscript of an object"):
        lower(oversized)

    def in_range(x: float) -> float:
        return x + float(_BE_U64_SMALL[0])

    _assert_python_matches_holoso(in_range, 1.0)


def test_list_subscript_keys_are_advanced_indexing() -> None:
    # numpy: m[[0]] is FANCY indexing (shape (1, 2)); reading it as the basic m[0] silently changes geometry.
    def known(s: float) -> float:
        return s * float(len(GAIN[[0]]))  # all-Known folds concretely through numpy itself: len == 1

    _assert_python_matches_holoso(known, 3.0)

    def runtime(s: float) -> float:
        v = (GAIN * s)[[0]]
        return v[0][0]  # type: ignore[no-any-return]

    with pytest.raises(UnsupportedConstruct, match="runtime aggregate"):
        lower(runtime)


def test_type_checking_annotations_do_not_block_state() -> None:
    # PEP 649 evaluates annotations lazily; a TYPE_CHECKING-only name on ANY class attribute must not crash
    # aggregate-state analysis (the annotation is typing documentation, not a schema).
    class Tracker:
        helper: SomeCheckOnlyType  # type: ignore[name-defined]  # noqa: F821

        def __init__(self) -> None:
            self.v = [1.0, 2.0]

        def step(self, a: float) -> None:
            self.v = [self.v[1], a]

    hir = lower(Tracker().step)
    assert {s.name for s in hir.state_slots} == {"v_0", "v_1"}


def test_scalar_reset_still_validates_the_field_contract() -> None:
    @dataclasses.dataclass
    class Declared:
        v: Float64[np.ndarray, "2"]

        def step(self, a: float) -> None:
            self.v = self.v * a

    with pytest.raises(UnsupportedConstruct, match="declared array type"):
        lower(Declared(1.0).step)  # type: ignore[arg-type]  # a scalar reset against a declared 2-vector


def test_zero_dimensional_array_state_is_rejected() -> None:
    class Zero:
        def __init__(self) -> None:
            self.v = np.array(1.0)

        def step(self, a: float) -> None:
            self.v = self.v * a

    with pytest.raises(UnsupportedConstruct, match="0-dimensional"):
        lower(Zero().step)


def test_zero_d_operand_broadcasts_with_arrays() -> None:
    def kernel(s: float) -> tuple[float, float, float]:
        v = _ZERO_D_CONST * (COEFFS * s)  # numpy broadcasts a 0-d array like a scalar
        return v[0], v[1], v[2]

    _assert_python_matches_holoso(kernel, 2.0)


def test_platform_alias_scalars_admit_like_their_dtypes() -> None:
    def kernel(s: float) -> float:
        return s + float(_LONGLONG_CONST)

    _assert_python_matches_holoso(kernel, 1.5)


def test_float_store_into_int_reset_array_reads_back_float() -> None:
    # The stored cells fix the read-back dtype: an int-reset slot holding promoted float cells must not
    # reject later arithmetic as "runtime integer" (the values already matched Python; the reason was wrong).
    class Decay:
        def __init__(self) -> None:
            self.v = np.array([4, 2])

        def step(self, a: float) -> float:
            self.v = self.v * a
            w = self.v * 2  # an INTEGER scalar: a stale int layout would misread this as runtime-int math
            return w[0]  # type: ignore[no-any-return]

    sim = _sim(Decay().step)
    reference = Decay()
    for _ in range(3):
        want = reference.step(0.5)
        got = _run(sim, 0.5)
        assert got[0] == pytest.approx(want)
        assert np.allclose(got[1:], reference.v.astype(np.float64), rtol=1e-12)


def test_width_collapse_extends_to_type_identity() -> None:
    # SANCTIONED deviation, part of the width collapse: an embedded narrow scalar is OBSERVATIONALLY its
    # 64-bit carrier, type identity included -- isinstance answers for the carrier (np.float64 IS a float
    # subclass) where plain numpy would answer for np.float32. Values, arithmetic, and type identity follow
    # the carrier together; distinguishing them would require width provenance the domain deliberately drops.
    def kernel(x: float) -> float:
        return x if isinstance(_F32_GAINS[0], float) else -x

    model = holoso.synthesize(kernel, default_ops(_FMT), name="kernel").numerical_model
    assert float(model.elaborate().run(3.0)[0]) == 3.0  # the carrier verdict; plain Python returns -3.0


@dataclasses.dataclass(frozen=True)
class _SelfLoop:
    inner: _SelfLoop  # noqa: F821  # legal under PEP 649: evaluated lazily, when the name exists


@dataclasses.dataclass(frozen=True)
class _AmbiguousInner:
    b: float


@dataclasses.dataclass(frozen=True)
class _AmbiguousOuter:
    a: _AmbiguousInner
    a_b: float


def _ambiguous_record_kernel(x: _AmbiguousOuter) -> float:
    return x.a_b


def _recursive_record_kernel(v: _SelfLoop) -> float:
    return 1.0


_DURATION_CONST = np.array([3], dtype="timedelta64[ns]")
_BE_U64_HUGE = np.array([2**63], dtype=">u8")
_BE_U64_SMALL = np.array([7], dtype=">u8")
_LONGLONG_CONST = np.longlong(7)


# ---------------------------------------------------------------- behavior (model vs numpy)


def test_matmul_matches_numpy() -> None:
    def transform(a: Float64[np.ndarray, "3 3"], x: Float64[np.ndarray, "3"]) -> Float64[np.ndarray, "3"]:
        return a @ x  # type: ignore[no-any-return]

    def chained(
        a: Float64[np.ndarray, "2 3"], b: Float64[np.ndarray, "3 3"], x: Float64[np.ndarray, "3"]
    ) -> Float64[np.ndarray, "2"]:
        return a @ b @ x  # type: ignore[no-any-return]

    def row_form(x: Float64[np.ndarray, "3"], a: Float64[np.ndarray, "3 2"]) -> Float64[np.ndarray, "2"]:
        return x @ a  # type: ignore[no-any-return]

    def dot(v: Float64[np.ndarray, "4"], w: Float64[np.ndarray, "4"]) -> float:
        return v @ w  # type: ignore[no-any-return]

    def quadratic_form(m: Float64[np.ndarray, "2 2"], v: Float64[np.ndarray, "2"]) -> float:
        return v @ m @ v  # type: ignore[no-any-return]

    def gram(a: Float64[np.ndarray, "2 3"]) -> Float64[np.ndarray, "3 3"]:
        return a.T @ a  # type: ignore[no-any-return]

    rng = np.random.default_rng(0xA11CE)
    cases: list[tuple[Callable[..., object], list[np.ndarray]]] = [
        (transform, [rng.normal(size=(3, 3)), rng.normal(size=3)]),
        (chained, [rng.normal(size=(2, 3)), rng.normal(size=(3, 3)), rng.normal(size=3)]),
        (row_form, [rng.normal(size=3), rng.normal(size=(3, 2))]),
        (dot, [rng.normal(size=4), rng.normal(size=4)]),
        (quadratic_form, [rng.normal(size=(2, 2)), rng.normal(size=2)]),
        (gram, [rng.normal(size=(2, 3))]),
    ]
    for fn, arrays in cases:
        got = _run(_sim(fn), *arrays)
        want = np.asarray(fn(*arrays)).flatten()
        assert np.allclose(got, want, rtol=1e-12, atol=1e-300), fn.__name__


def test_np_array_factory_converts_list_and_matches_numpy() -> None:
    # np.array([...]) converts a Python list/tuple into a numpy array on which arithmetic, the matrix product, and
    # elementwise combination with another array are all defined; the results match numpy executing the same kernel.
    def vec_add(a: float, b: float, c: float, d: float) -> Float64[np.ndarray, "2"]:
        return np.array([a, b]) + np.array([c, d])  # type: ignore[no-any-return]

    def dot(a: float, b: float, c: float, d: float) -> float:
        return np.array([a, b]) @ np.array([c, d])  # type: ignore[no-any-return]

    def mat_minus_rows(m: Float64[np.ndarray, "2 2"], a: float, b: float) -> Float64[np.ndarray, "2 2"]:
        row = np.array([a, b])
        return m - np.array([row, row])  # type: ignore[no-any-return]

    a, b, c, d = 1.5, -2.0, 0.25, 3.0
    assert np.allclose(_run(_sim(vec_add), a, b, c, d), vec_add(a, b, c, d), rtol=1e-12, atol=1e-300)
    assert np.allclose(_run(_sim(dot), a, b, c, d), np.asarray(dot(a, b, c, d)), rtol=1e-12, atol=1e-300)

    m = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert np.allclose(_run(_sim(mat_minus_rows), m, a, b), np.asarray(mat_minus_rows(m, a, b)).flatten(), rtol=1e-12)


def test_elementwise_and_globals_match_numpy() -> None:
    def kernel(x: Float64[np.ndarray, "2"], s: float) -> Float64[np.ndarray, "2"]:
        y = GAIN @ (x + COEFFS[0:2]) - x / 4.0
        return (y * s + GAIN[1]) @ GAIN  # type: ignore[no-any-return]

    rng = np.random.default_rng(0xB0B)
    x, s = rng.normal(size=2), float(rng.normal())
    got = _run(_sim(kernel), x, s)
    assert np.allclose(got, np.asarray(kernel(x, s)), rtol=1e-12, atol=1e-300)


def test_integer_dtype_module_constant_folds_to_floats() -> None:
    def kernel(v: Float64[np.ndarray, "3"]) -> float:
        return v @ INT_TAPS  # type: ignore[no-any-return]

    v = np.array([0.5, -1.5, 2.0])
    got = _run(_sim(kernel), v)
    assert got[0] == float(v @ INT_TAPS)


def test_matrix_state_update_matches_numpy_across_transactions() -> None:
    @dataclasses.dataclass
    class Decay:
        P: Float64[np.ndarray, "2 2"]

        def step(self, f: Float64[np.ndarray, "2 2"]) -> None:
            self.P = f @ self.P @ f.T

    sim = holoso.synthesize(Decay(np.eye(2)).step, default_ops(_FMT), name="decay").numerical_model.elaborate()
    assert [p.name for p in sim.outputs] == ["state_P_0_0", "state_P_0_1", "state_P_1_0", "state_P_1_1"]
    reference = Decay(np.eye(2))
    f = np.array([[1.0, 0.125], [-0.25, 0.9375]])
    for _ in range(4):
        got = _run(sim, f)
        reference.step(f)
        assert np.allclose(got, reference.P.flatten(), rtol=1e-12, atol=1e-300)


def test_annotated_local_assignment_matches_numpy() -> None:
    # Locks in the annotated local-assignment statement ``name: T = value`` (both array- and scalar-annotated forms) --
    # the annotation is decorative and the value binds like a plain assignment; a frontend branch no other kernel hits.
    def kernel(a: Float64[np.ndarray, "2 2"], x: Float64[np.ndarray, "2"]) -> Float64[np.ndarray, "2"]:
        y: Float64[np.ndarray, "2"] = a @ x  # array-annotated local bound to an expression
        t: float = y[0] - y[1]  # scalar-annotated local
        return y * t

    rng = np.random.default_rng(0xDEC1)
    a, x = rng.normal(size=(2, 2)), rng.normal(size=2)
    got = _run(_sim(kernel), a, x)
    assert np.allclose(got, np.asarray(kernel(a, x)), rtol=1e-12, atol=1e-300)


def test_runtime_divisor_division_matches_numpy() -> None:
    # Locks in float division by a RUNTIME divisor -- the strength reducer's fallthrough to a real FloatDiv, the fdiv
    # operator's execution, and the value model's exact divide -- none of which a constant/power-of-two divisor reaches.
    def scalar_div(a: float, b: float) -> float:
        return a / b

    def vector_over_scalar(v: Float64[np.ndarray, "3"], s: float) -> Float64[np.ndarray, "3"]:
        return v / s

    def kalman_gain(P: Float64[np.ndarray, "2 2"], h: Float64[np.ndarray, "2"], s: float) -> Float64[np.ndarray, "2"]:
        return (P @ h) / s  # type: ignore[no-any-return]  # matrix-vector product over a runtime scalar

    scalar_cases = [(6.0, 3.0), (1.0, -4.0), (2.5, 0.5), (0.0, 7.0)]  # each divides exactly in both formats; last: 0/b
    for fmt in (_FMT, FloatFormat(6, 18)):  # a second, narrower datapath width also exercises the divide
        sim = holoso.synthesize(scalar_div, default_ops(fmt), name="div").numerical_model.elaborate()
        for a, b in scalar_cases:
            assert np.allclose(_run(sim, a, b), a / b, rtol=1e-12, atol=1e-300), (fmt, a, b)

    rng = np.random.default_rng(0x0D17)
    v, s = rng.normal(size=3), float(rng.uniform(0.5, 2.0))
    assert np.allclose(_run(_sim(vector_over_scalar), v, s), v / s, rtol=1e-12, atol=1e-300)

    P, h, s = rng.normal(size=(2, 2)), rng.normal(size=2), float(rng.uniform(0.5, 2.0))
    assert np.allclose(_run(_sim(kalman_gain), P, h, s), (P @ h) / s, rtol=1e-12, atol=1e-300)


def test_stateful_kalman_style_filter_matches_numpy_across_transactions() -> None:
    # Locks in the whole matrix feature surface composed in one stateful kernel across transactions: matrix/vector
    # parameters and carried state, ndarray module constants, ``@`` in every shape with transpose, elementwise scalar
    # broadcast, an annotated local, a static row loop, a shaped return, and the runtime-divisor Kalman gain.
    sim = holoso.synthesize(TrackingFilter().update, default_ops(_FMT), name="tracker").numerical_model.elaborate()
    assert [p.name for p in sim.outputs] == [
        "out_0", "out_1", "state_x_0", "state_x_1", "state_P_0_0", "state_P_0_1", "state_P_1_0", "state_P_1_1",
    ]  # fmt: skip
    reference = TrackingFilter()
    rng = np.random.default_rng(0xF117E5)
    F = np.array([[1.0, 0.1], [0.0, 1.0]])
    for step in range(6):
        z = np.array([float(rng.uniform(-1.0, 1.0)), float(rng.uniform(-1.0, 1.0))])
        got = _run(sim, F, z)
        prediction = reference.update(F, z)
        want = np.array([float(v) for v in (*prediction, *reference.x, *reference.P.flatten())])
        assert np.all(np.isfinite(want))
        assert np.allclose(got, want, rtol=1e-9, atol=1e-12), step


def test_imu_frame_transform_example_matches_numpy() -> None:
    # The bundled 3D rigid-body / IMU frame transform example must lower and agree with its own plain-numpy execution,
    # confirming the matmul/transpose/broadcast composition it demonstrates is valid, runnable Python.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    import imu_frame_transform

    yaw90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    roll90 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    for rotation in (yaw90, roll90):
        _assert_python_matches_holoso(
            imu_frame_transform.transform,
            rotation,
            np.array([1.0, 2.0, 3.0]),
            np.array([0.1, -0.2, 9.9]),
            np.array([2.0, 0.0, -1.0]),
        )


def test_imu_frame_transform_fma_matches_numpy() -> None:
    # The ffma-contracted datapath the synth matrix's FMA rows exercise (each dot-product multiply-accumulate fused into
    # a single-rounded a*b+c) must compute the same transform: FMA changes only the rounding, not the result.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    import imu_frame_transform

    ops = dataclasses.replace(default_ops(_FMT), ffma=FFmaOperator(_FMT))
    sim = holoso.synthesize(imu_frame_transform.transform, ops, name="imu_fma").numerical_model.elaborate()
    yaw90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    roll90 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    for rotation in (yaw90, roll90):
        inputs = (rotation, np.array([1.0, 2.0, 3.0]), np.array([0.1, -0.2, 9.9]), np.array([2.0, 0.0, -1.0]))
        want = np.asarray(imu_frame_transform.transform(*inputs)).flatten()
        assert np.allclose(_run(sim, *inputs), want, rtol=1e-9, atol=1e-300)


# ---------------------------------------------------------------- linear algebra library functions


def test_operators_are_the_library_functions() -> None:
    # ``@`` and ``.T`` lower by resolving np.matmul / np.transpose in the registry, so the operator and its spelled
    # call cannot drift apart: identical HIR, not merely identical values.
    def with_operators(a: Float64[np.ndarray, "2 3"], b: Float64[np.ndarray, "2 3"]) -> Float64[np.ndarray, "2 2"]:
        return a @ b.T  # type: ignore[no-any-return]

    def with_calls(a: Float64[np.ndarray, "2 3"], b: Float64[np.ndarray, "2 3"]) -> Float64[np.ndarray, "2 2"]:
        return np.matmul(a, np.transpose(b))  # type: ignore[no-any-return]

    counts = [
        (_arith_count(lower(k), FloatMul), _arith_count(lower(k), FloatAdd)) for k in (with_operators, with_calls)
    ]
    assert counts[0] == counts[1] == (12, 8)
    for kernel in (with_operators, with_calls):
        _assert_python_matches_holoso(
            kernel, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), np.array([[0.5, -1.0, 2.0], [3.0, -2.0, 0.25]])
        )


def test_np_dot_is_the_matrix_product() -> None:
    def dot_kernel(a: Float64[np.ndarray, "2 2"], x: Float64[np.ndarray, "2"]) -> Float64[np.ndarray, "2"]:
        return np.dot(a, x)  # type: ignore[no-any-return]

    assert [o.name for o in lower(dot_kernel).outputs] == ["out_0", "out_1"]
    _assert_python_matches_holoso(dot_kernel, np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([1.0, -1.0]))

    def scalar_dot(a: float, x: Float64[np.ndarray, "2"]) -> Float64[np.ndarray, "2"]:
        return np.dot(a, x)  # type: ignore[no-any-return]

    # numpy would multiply here; Holoso rejects rather than silently reinterpreting the matrix product as a broadcast.
    with pytest.raises(UnsupportedConstruct, match="scalar"):
        lower(scalar_dot)


def test_np_trace_and_np_outer() -> None:
    def tr(m: Float64[np.ndarray, "3 3"]) -> float:
        return np.trace(m)  # type: ignore[no-any-return]

    assert [o.name for o in lower(tr).outputs] == ["out_0"]
    assert _arith_count(lower(tr), FloatMul) == 0  # a fold of the diagonal, no multiplies
    _assert_python_matches_holoso(tr, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]))

    def outer(u: Float64[np.ndarray, "2"], v: Float64[np.ndarray, "3"]) -> Float64[np.ndarray, "2 3"]:
        return np.outer(u, v)

    assert _arith_count(lower(outer), FloatMul) == 6 and _arith_count(lower(outer), FloatAdd) == 0
    _assert_python_matches_holoso(outer, np.array([1.0, -2.0]), np.array([0.5, 3.0, -1.0]))

    def rect_trace(m: Float64[np.ndarray, "2 3"]) -> float:
        return np.trace(m)  # type: ignore[no-any-return]

    # numpy walks the shorter diagonal; Holoso rejects rather than reinterpreting.
    with pytest.raises(UnsupportedConstruct, match="square"):
        lower(rect_trace)

    def outer_of_matrix(m: Float64[np.ndarray, "2 2"]) -> Float64[np.ndarray, "2 2"]:
        return np.outer(m, m)

    with pytest.raises(UnsupportedConstruct, match="1-D"):
        lower(outer_of_matrix)


def test_trace_of_a_1x1_boolean_matrix_is_rejected_like_a_larger_one() -> None:
    # The diagonal fold is seeded at 0.0, so even a 1x1 trace contracts through an addition and rejects a boolean
    # diagonal, rather than passing the boolean through where numpy would widen it to an integer.
    def bool_trace(flag: bool) -> bool:
        return np.trace(np.array([[flag]]))  # type: ignore[no-any-return]

    with pytest.raises(UnsupportedConstruct, match="explicit conversion"):
        lower(bool_trace)


@pytest.mark.skip(
    reason="FIR_PARITY_PENDING: blocked by E1 call-site attribution; enables at S2.11 (scalar-.T sub-case rewritten there)"
)
def test_library_shape_rejection_is_attributed_to_the_user_call_site() -> None:
    # A stub validates its own operands with a ``raise`` on a statically taken path; the error must name the user's
    # spelling and point at the user's line, never into the stub source.
    def bad(a: Float64[np.ndarray, "2 3"], x: Float64[np.ndarray, "2"]) -> Float64[np.ndarray, "2"]:
        return a @ x  # type: ignore[no-any-return]

    with pytest.raises(UnsupportedConstruct, match=r"in matmul\(\).*mismatch") as excinfo:
        lower(bad)
    assert excinfo.value.location is not None
    assert excinfo.value.location.line is not None and "a @ x" in excinfo.value.location.line

    def bad_t(a: float) -> float:
        return a.T  # type: ignore[attr-defined, no-any-return]

    with pytest.raises(UnsupportedConstruct, match=r"in transpose\(\).*transpose a scalar"):
        lower(bad_t)


def test_matrix_state_transposed_under_a_shape_guard_across_transactions() -> None:
    # The reset snapshot fixes the shape, so the guard folds identically in the scan and in lowering, while the
    # attribute itself is reassigned every transaction. Compare every port by NAME: the returned leaf is deduped onto
    # the public state port that already carries it, so a positional comparison would read the wrong wire.
    class Flip:
        def __init__(self) -> None:
            self.P = np.array([[1.0, 2.0], [3.0, 4.0]])
            self.s = 0.0

        def step(self, x: float) -> float:
            self.P = self.P.T
            if self.P.ndim == 2:
                self.s = self.s + self.P[0][1] * x
            return self.s

    sim = _sim(Flip().step)
    ports = [p.name for p in sim.outputs]
    assert ports == ["state_P_0_0", "state_P_0_1", "state_P_1_0", "state_P_1_1", "state_s"]
    reference = Flip()
    for _ in range(4):
        want = reference.step(2.0)
        got = dict(zip(ports, [float(v) for v in sim.run(2.0)]))
        assert got["state_s"] == pytest.approx(want)
        assert [got[f"state_P_{i}_{j}"] for i in range(2) for j in range(2)] == pytest.approx(
            list(reference.P.flatten())
        )


def test_matrix_product_inside_a_comprehension_inside_a_loop() -> None:
    # The stub is inlined from inside a comprehension element, itself inside an unrolled loop, and its result feeds
    # persistent state. Exercises the interaction of aggregate iteration, comprehension scoping, and stub inlining.
    class Accumulate:
        def __init__(self) -> None:
            self.acc = 0.0

        def step(self, a: Float64[np.ndarray, "2 2"], v: Float64[np.ndarray, "2"]) -> float:
            for _ in range(2):
                w = [(a @ v)[k] + (a.T @ v)[k] for k in range(2)]
                self.acc = self.acc + w[0] + w[1]
            return self.acc

    a, v = np.array([[1.0, 0.5], [-0.5, 2.0]]), np.array([3.0, -1.0])
    sim = _sim(Accumulate().step)
    reference = Accumulate()
    for _ in range(3):
        want = reference.step(a, v)
        assert _run(sim, a, v)[0] == pytest.approx(want)


def test_for_over_an_aggregate_inside_a_while_loop() -> None:
    # A target first bound inside the loop is not loop-carried, so aggregate iteration composes with a back-edge loop.
    class SumRows:
        def __init__(self) -> None:
            self.s = 0.0

        def step(self, m: Float64[np.ndarray, "2 2"], n: float) -> float:
            j = 0.0
            while j < n:
                for row in m:
                    self.s = self.s + row[0]
                j = j + 1.0
            return self.s

    m = np.array([[1.0, 2.0], [3.0, 4.0]])
    sim = _sim(SumRows().step)
    reference = SumRows()
    for _ in range(3):
        want = reference.step(m, 2.0)
        assert _run(sim, m, 2.0)[0] == pytest.approx(want)
