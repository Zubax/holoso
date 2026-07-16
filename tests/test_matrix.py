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


@pytest.mark.skip(reason="FIR_PARITY_PENDING: matmul on array parameters — stage 9")
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


@pytest.mark.skip(reason="FIR_PARITY_PENDING: matmul rejections on array/aggregate operands — stage 9")
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

    with pytest.raises(UnsupportedConstruct, match="Python list/tuple"):
        lower(ragged)

    def three_dee(a: float) -> float:
        return np.array([[[a]]]) @ np.array([a])  # type: ignore[no-any-return]

    with pytest.raises(UnsupportedConstruct, match="1-D or 2-D"):
        lower(three_dee)

    def boolean(v: Float64[np.ndarray, "2"], flag: bool) -> float:
        return v @ np.array([flag, flag])  # type: ignore[no-any-return]

    with pytest.raises(UnsupportedConstruct, match="floating-point"):
        lower(boolean)


@pytest.mark.skip(reason="FIR_PARITY_PENDING: dot product on array parameters — stage 9")
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


@pytest.mark.skip(reason="FIR_PARITY_PENDING: np.matmul on array parameters — stage 9")
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


@pytest.mark.skip(reason="FIR_PARITY_PENDING: np.matmul on array parameters — stage 9")
def test_np_matmul_bare_name_import_resolves() -> None:
    def with_bare_name(a: Float64[np.ndarray, "2 2"], x: Float64[np.ndarray, "2"]) -> Float64[np.ndarray, "2"]:
        return _matmul(a, x)  # type: ignore[no-any-return]

    assert [o.name for o in lower(with_bare_name).outputs] == ["out_0", "out_1"]
    _assert_python_matches_holoso(with_bare_name, np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([1.0, -1.0]))
    _assert_python_matches_holoso(with_bare_name, np.array([[0.5, -1.0], [2.0, 0.25]]), np.array([2.0, 3.0]))


@pytest.mark.skip(reason="FIR_PARITY_PENDING: augmented assignment to array — stage 9")
def test_augmented_assignment_to_array_is_rejected() -> None:
    # Regression: numpy '+=' / '@=' mutate in place while the frontend rebinds, so an alias would diverge; the array
    # augmented forms must be rejected in favor of the explicit 'x = x + ...' rebind.
    def name_target(v: Float64[np.ndarray, "2"], s: float) -> Float64[np.ndarray, "2"]:
        v += s
        return v

    with pytest.raises(UnsupportedConstruct, match="augmented assignment to a list or array"):
        lower(name_target)

    @dataclasses.dataclass
    class State:
        P: Float64[np.ndarray, "2 2"]

        def step(self, f: Float64[np.ndarray, "2 2"]) -> None:
            self.P @= f

    with pytest.raises(UnsupportedConstruct, match="augmented assignment to a list or array"):
        lower(State(np.eye(2)).step)

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


@pytest.mark.skip(reason="FIR_PARITY_PENDING: elementwise array arithmetic — stage 9")
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


@pytest.mark.skip(reason="FIR_PARITY_PENDING: numpy methods on runtime list/array — stage 9")
def test_numpy_only_methods_on_a_list_are_rejected() -> None:
    # `.T`, `.flatten()`, and multi-axis indexing are numpy-array operations undefined on a Python list, so they are
    # rejected on a list literal; wrapping in np.array([...]) makes them valid.
    def transpose(a: float, b: float) -> float:
        return ([a, b].T)[0]  # type: ignore[attr-defined, no-any-return]

    with pytest.raises(UnsupportedConstruct, match="transpose"):
        lower(transpose)

    def flatten(a: float, b: float) -> float:
        return ([[a, b]].flatten())[0]  # type: ignore[attr-defined, no-any-return]

    with pytest.raises(UnsupportedConstruct, match="flattening"):
        lower(flatten)

    def multi_axis(a: float, b: float) -> float:
        m = [[a, b], [b, a]]
        return m[0, 1]  # type: ignore[call-overload, no-any-return]

    with pytest.raises(UnsupportedConstruct, match="multi-axis"):
        lower(multi_axis)


@pytest.mark.skip(reason="FIR_PARITY_PENDING: runtime aggregate (list) arithmetic — stage 9")
def test_list_of_array_demotes_to_sequence_for_arithmetic() -> None:
    # list(arr)/tuple(arr) produce Python sequences (as in Python), so arithmetic on the result is rejected even though
    # the argument was an array -- guards against the builtins accidentally keeping array semantics.
    def via_list(v: Float64[np.ndarray, "2"], s: float) -> float:
        return (list(v) * s)[0]  # type: ignore[operator, no-any-return]

    with pytest.raises(UnsupportedConstruct, match="Python list/tuple"):
        lower(via_list)


@pytest.mark.skip(reason="FIR_PARITY_PENDING: starred unpacking of aggregate — stage 9")
def test_star_unpack_remainder_is_a_python_list() -> None:
    # Regression: a starred target binds a plain list even when unpacking an array (PEP 3132), so arithmetic on the
    # remainder must be rejected -- it must not inherit the source array's semantics.
    def spread(v: Float64[np.ndarray, "3"]) -> float:
        first, *rest = v
        return (rest + rest)[0]  # type: ignore[no-any-return]

    with pytest.raises(UnsupportedConstruct, match="Python list/tuple"):
        lower(spread)


@pytest.mark.skip(reason="FIR_PARITY_PENDING: array/list arithmetic on merged value — stage 9")
def test_mismatched_branch_flavor_merge_rejects_array_ops() -> None:
    # Regression: a value that is an array in one arm and a list in the other must not silently gain array semantics
    # (that made acceptance depend on arm order). A numpy op on the merged value is rejected; structural use stays fine.
    def arithmetic(c: bool, a: float, b: float) -> Float64[np.ndarray, "2"]:
        if c:
            v = np.array([a, b])
        else:
            v = [a, b]  # type: ignore[assignment]
        return v * 2.0

    with pytest.raises(UnsupportedConstruct, match="Python list/tuple"):
        lower(arithmetic)

    def structural(c: bool, a: float, b: float) -> float:
        if c:
            v = np.array([a, b])
        else:
            v = [a, b]  # type: ignore[assignment]
        return v[0]  # type: ignore[no-any-return]

    assert [o.name for o in lower(structural).outputs] == ["out_0"]


@pytest.mark.skip(reason="FIR_PARITY_PENDING: array state assignment — stage 9")
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


@pytest.mark.skip(reason="FIR_PARITY_PENDING: array modulo operator — stage 9 aggregate ops")
def test_unsupported_operator_diagnostic_names_the_operator() -> None:
    # An unsupported operator must be named even when its operands are boolean (its own diagnostic wins over the
    # float-operand check), rather than being misreported as a boolean-operand error.
    def bitor(a: bool, b: bool) -> bool:
        return a | b

    with pytest.raises(UnsupportedConstruct, match="unsupported binary operator BitOr"):
        lower(bitor)

    # Also on the aggregate path: an unsupported operator must be named even when the operands' shapes mismatch, rather
    # than being masked by the shape-mismatch diagnostic.
    def modulo(v: Float64[np.ndarray, "2"], w: Float64[np.ndarray, "3"]) -> Float64[np.ndarray, "2"]:
        return v % w  # type: ignore[no-any-return]

    with pytest.raises(UnsupportedConstruct, match="unsupported binary operator Mod"):
        lower(modulo)


@pytest.mark.skip(reason="FIR_PARITY_PENDING: array transpose (.T) — stage 9")
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


@pytest.mark.skip(reason="FIR_PARITY_PENDING: array subscript/slice — stage 9")
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


@pytest.mark.skip(reason="FIR_PARITY_PENDING: runtime aggregate (list) indexing — stage 9")
def test_multi_axis_index_on_list_is_rejected() -> None:
    # Multi-axis m[i, j] is a numpy-array operation with no meaning on a Python list (list[i, j] is a tuple key, a
    # TypeError), so it must be rejected as a list operation rather than silently indexed. Chained m[i][j] on the same
    # (even ragged) list stays valid (plain list indexing).
    def list_multi_axis(a: float, b: float) -> float:
        m = [[a, b], [a]]
        return m[0, 1]  # type: ignore[call-overload,no-any-return]

    with pytest.raises(UnsupportedConstruct, match="Python list/tuple"):
        lower(list_multi_axis)

    def ragged_chained(a: float, b: float) -> float:
        m = [[a, b], [a]]
        return m[0][1]

    assert [o.name for o in lower(ragged_chained).outputs] == ["out_0"]


@pytest.mark.skip(reason="FIR_PARITY_PENDING: shaped array parameter annotations — stage 9")
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

    class _FakeArray:  # structurally array-like (has ``dims``) but its dims is not a real jaxtyping tuple
        dims = None

    def fake(v: _FakeArray) -> float:
        return 1.0

    with pytest.raises(UnsupportedConstruct, match="not a valid fixed-shape array annotation"):
        lower(fake)


@pytest.mark.skip(reason="FIR_PARITY_PENDING: array parameter subscript — stage 9")
def test_wide_float_dtype_annotation_is_accepted() -> None:
    def f(v: Float[np.ndarray, "2"]) -> float:
        return v[0] + v[1]  # type: ignore[no-any-return]

    assert lower(f).input_names() == ["v_0", "v_1"]


def test_decomposed_parameter_port_collision_is_rejected() -> None:
    def collides(v: Float64[np.ndarray, "2"], v_0: float) -> float:
        return v[1] * v_0  # type: ignore[no-any-return]

    with pytest.raises(UnsupportedConstruct, match="collides"):
        lower(collides)


@pytest.mark.skip(reason="FIR_PARITY_PENDING: array return values — stage 9")
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
        return [flag]  # type: ignore[return-value]

    with pytest.raises(UnsupportedConstruct, match="floating-point"):
        lower(boolean_leaves)


@pytest.mark.skip(reason="FIR_PARITY_PENDING: matrix state annotation/assignment — stage 9")
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


@pytest.mark.skip(reason="FIR_PARITY_PENDING: array state assignment — stage 9")
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


@pytest.mark.skip(reason="FIR_PARITY_PENDING: matrix carried across loop — stage 9")
def test_matrix_carried_across_while_loop_is_rejected() -> None:
    def f(m: Float64[np.ndarray, "2 2"], n: float) -> Float64[np.ndarray, "2 2"]:
        x = n
        while x > 0.0:
            m = m * 0.5
            x = x - 1.0
        return m

    with pytest.raises(UnsupportedConstruct, match="aggregate"):
        lower(f)


@pytest.mark.skip(reason="FIR_PARITY_PENDING: array parameter subscript — stage 9")
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


@pytest.mark.skip(reason="FIR_PARITY_PENDING: constant slicing/transpose folding — stage 9")
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


@pytest.mark.skip(reason="FIR_PARITY_PENDING: matrix state transpose — stage 9")
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


@pytest.mark.skip(reason="FIR_PARITY_PENDING: unary plus on array — stage 9")
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


@pytest.mark.skip(
    reason="FIR_PARITY_PENDING: runtime array/matrix state and its constant-index rejection policy — stage 9"
)
def test_ndarray_module_constant_rejections() -> None:
    def boolean(a: float) -> float:
        return _BOOL_CONST[0]  # type: ignore[no-any-return]

    with pytest.raises(UnsupportedConstruct, match="real numbers"):
        lower(boolean)

    def three_dee(a: float) -> float:
        return _CUBE_CONST[0, 0, 0]  # type: ignore[no-any-return]

    with pytest.raises(UnsupportedConstruct, match="1-D or 2-D"):
        lower(three_dee)


@pytest.mark.skip(
    reason="FIR_PARITY_PENDING: runtime array/matrix state and its constant-index rejection policy — stage 9"
)
def test_ndarray_subclass_constant_and_state_are_rejected() -> None:
    # Regression: an ndarray subclass (np.matrix) redefines operators (``*`` is matmul), so folding it as a plain array
    # would silently diverge from its own Python semantics; it must be rejected, both as a module constant and a reset.
    def constant(a: float) -> float:
        return _MATRIX_CONST[0, 1] + a  # type: ignore[no-any-return]

    with pytest.raises(UnsupportedConstruct, match="plain numpy array"):
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
        v = COEFFS @ COEFFS
        return v * s  # type: ignore[no-any-return]

    with pytest.raises(UnsupportedConstruct, match="matrix product"):
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


# ---------------------------------------------------------------- behavior (model vs numpy)


@pytest.mark.skip(reason="FIR_PARITY_PENDING: matmul on array parameters — stage 9")
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


@pytest.mark.skip(reason="FIR_PARITY_PENDING: np.array with runtime arguments — stage 9")
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


@pytest.mark.skip(reason="FIR_PARITY_PENDING: array slicing/elementwise — stage 9")
def test_elementwise_and_globals_match_numpy() -> None:
    def kernel(x: Float64[np.ndarray, "2"], s: float) -> Float64[np.ndarray, "2"]:
        y = GAIN @ (x + COEFFS[0:2]) - x / 4.0
        return (y * s + GAIN[1]) @ GAIN  # type: ignore[no-any-return]

    rng = np.random.default_rng(0xB0B)
    x, s = rng.normal(size=2), float(rng.normal())
    got = _run(_sim(kernel), x, s)
    assert np.allclose(got, np.asarray(kernel(x, s)), rtol=1e-12, atol=1e-300)


@pytest.mark.skip(reason="FIR_PARITY_PENDING: matmul on array parameters — stage 9")
def test_integer_dtype_module_constant_folds_to_floats() -> None:
    def kernel(v: Float64[np.ndarray, "3"]) -> float:
        return v @ INT_TAPS  # type: ignore[no-any-return]

    v = np.array([0.5, -1.5, 2.0])
    got = _run(_sim(kernel), v)
    assert got[0] == float(v @ INT_TAPS)


@pytest.mark.skip(reason="FIR_PARITY_PENDING: matmul on array state — stage 9")
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


@pytest.mark.skip(reason="FIR_PARITY_PENDING: matmul on array parameters — stage 9")
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


@pytest.mark.skip(reason="FIR_PARITY_PENDING: array (vector/matrix) division — stage 9")
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


@pytest.mark.skip(reason="FIR_PARITY_PENDING: matrix feature surface — stage 9")
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


@pytest.mark.skip(reason="FIR_PARITY_PENDING: matmul on array parameters — stage 9")
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


@pytest.mark.skip(reason="FIR_PARITY_PENDING: matmul on array parameters — stage 9")
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


@pytest.mark.skip(reason="FIR_PARITY_PENDING: matmul/transpose on array parameters — stage 9")
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


@pytest.mark.skip(reason="FIR_PARITY_PENDING: np.dot on array parameters — stage 9")
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


@pytest.mark.skip(reason="FIR_PARITY_PENDING: np.trace/np.outer on array parameters — stage 9")
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


@pytest.mark.skip(reason="FIR_PARITY_PENDING: np.trace on runtime array — stage 9")
def test_trace_of_a_1x1_boolean_matrix_is_rejected_like_a_larger_one() -> None:
    # The diagonal fold is seeded at 0.0, so even a 1x1 trace contracts through an addition and rejects a boolean
    # diagonal, rather than passing the boolean through where numpy would widen it to an integer.
    def bool_trace(flag: bool) -> bool:
        return np.trace(np.array([[flag]]))  # type: ignore[no-any-return]

    with pytest.raises(UnsupportedConstruct, match="floating-point"):
        lower(bool_trace)


@pytest.mark.skip(reason="FIR_PARITY_PENDING: matmul on array parameters — stage 9")
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


@pytest.mark.skip(reason="FIR_PARITY_PENDING: matrix state — stage 9")
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


@pytest.mark.skip(reason="FIR_PARITY_PENDING: matmul in comprehension/loop — stage 9")
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


@pytest.mark.skip(reason="FIR_PARITY_PENDING: iteration over an aggregate — stage 9")
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
