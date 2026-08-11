"""
Statically-shaped matrix/vector support: the ``@`` operator, elementwise aggregate arithmetic, transpose, numpy-style
subscripts, jaxtyping-annotated parameters/returns, matrix state, and ndarray module constants. Diagnostics and
structure are checked through the public synthesis artifacts (ports, initiation interval, emitted Verilog, residual
``frontend_ir``); numerical behavior is checked black-box through the public API against numpy executing the very
same kernel. The binding-time tests pinning what folds statically stay on the lowered HIR, and the FMA-chain test
keeps a compact MIR population sentinel for the exact operator count that module pooling erases from the Verilog.
"""

import dataclasses
import warnings
from collections.abc import Callable

import numpy as np
import pytest
from jaxtyping import Bool, Float, Float64, Int, Shaped

import holoso
from holoso._operators import FFmaOperator
from holoso import FFmaOptions, FloatFormat, UnsupportedConstruct
from holoso._eel import lower
from holoso._mir import lower as lower_to_mir

from ._examples import imu_frame_transform
from ._modelref import DEFAULT_IFCONV_MAX_OPS, default_ops, default_options
from ._public import strip_inline_prelude, strip_locations

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


def _synth(fn: Callable[..., object]) -> holoso.SynthesisResult:
    return holoso.synthesize(fn, default_options(_FMT), name="kernel")


def _refused(fn: Callable[..., object], match: str) -> None:
    with pytest.raises(UnsupportedConstruct, match=match):
        _synth(fn)


def _sim(fn: Callable[..., object]) -> holoso.NumericalSimulator:
    return _synth(fn).numerical_model.elaborate()


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

    result = _synth(mat_vec)
    assert [p.name for p in result.input_ports] == [
        "in_a_0_0", "in_a_0_1", "in_a_0_2", "in_a_1_0", "in_a_1_1", "in_a_1_2", "in_x_0", "in_x_1", "in_x_2",
    ]  # fmt: skip
    assert [p.name for p in result.output_ports] == ["out_0", "out_1"]
    a, x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), np.array([1.0, 0.0, -1.0])
    got = _run(result.numerical_model.elaborate(), a, x)
    assert np.allclose(got, a @ x, rtol=1e-12, atol=1e-300)

    def vec_mat(x: Float64[np.ndarray, "2"], a: Float64[np.ndarray, "2 3"]) -> Float64[np.ndarray, "3"]:
        return x @ a  # type: ignore[no-any-return]

    assert [p.name for p in _synth(vec_mat).output_ports] == ["out_0", "out_1", "out_2"]

    def dot(v: Float64[np.ndarray, "3"], w: Float64[np.ndarray, "3"]) -> float:
        return v @ w  # type: ignore[no-any-return]

    assert [p.name for p in _synth(dot).output_ports] == ["out_0"]

    def mat_mat(a: Float64[np.ndarray, "2 3"], b: Float64[np.ndarray, "3 2"]) -> Float64[np.ndarray, "2 2"]:
        return a @ b  # type: ignore[no-any-return]

    assert [p.name for p in _synth(mat_mat).output_ports] == ["out_0_0", "out_0_1", "out_1_0", "out_1_1"]


def test_matmul_rejections() -> None:
    def scalar_operand(a: float, x: Float64[np.ndarray, "2"]) -> Float64[np.ndarray, "2"]:
        return a @ x  # type: ignore[operator, unused-ignore]

    _refused(scalar_operand, "scalar")

    def dim_mismatch(a: Float64[np.ndarray, "2 3"], x: Float64[np.ndarray, "2"]) -> Float64[np.ndarray, "2"]:
        return a @ x  # type: ignore[no-any-return]

    _refused(dim_mismatch, "mismatch")

    def ragged(a: float, b: float) -> float:
        # A bare Python list has no ``@`` (a TypeError in Python), so the matrix product is rejected as a list operation
        # before rectangularity is even considered; the ragged literal cannot be wrapped in np.array either.
        return [[a, b], [a]] @ [a, b]  # type: ignore[operator, no-any-return]

    _refused(ragged, "Python list/tuple")

    def three_dee(a: float) -> float:
        return np.array([[[a]]]) @ np.array([a])  # type: ignore[no-any-return]

    _refused(three_dee, "1-D and 2-D")

    def boolean(v: Float64[np.ndarray, "2"], flag: bool) -> float:
        return v @ np.array([flag, flag])  # type: ignore[no-any-return]

    _refused(boolean, "must hold numbers, not booleans")


def test_dot_product_left_fold_contracts_to_fma_chain() -> None:
    # The documented reason for the left-fold dot expansion: with ffma configured, an n-element dot must lower to one
    # fmul plus n-1 ffma (each running-sum add fuses the next single-use product). At n=4 any balanced tree must keep a
    # real fadd (its final add sums two ffma results), so the pooled module set pins the chain publicly, and the
    # residual text pins the left-fold association; the MIR population sentinel keeps the exact count that module
    # pooling erases from the Verilog.
    def dot(v: Float64[np.ndarray, "4"], w: Float64[np.ndarray, "4"]) -> float:
        return v @ w  # type: ignore[no-any-return]

    def mnemonic_counts(with_fma: bool) -> dict[str, int]:
        ops = default_ops(_FMT)
        if with_fma:
            ops = dataclasses.replace(ops, ffma=FFmaOperator(_FMT, FFmaOptions(), 0))
        mir = lower_to_mir(lower(dot).hir, ops, DEFAULT_IFCONV_MAX_OPS)
        counts: dict[str, int] = {}
        for node in mir.nodes.values():
            operator = getattr(node, "operator", None)
            if operator is not None:
                stem = operator.mnemonic.split("_")[0]
                counts[stem] = counts.get(stem, 0) + 1
        return counts

    assert mnemonic_counts(with_fma=True) == {"fmul": 1, "ffma": 3}
    assert mnemonic_counts(with_fma=False) == {"fmul": 4, "fadd": 3}

    options = default_options(_FMT)
    options = dataclasses.replace(options, operator=dataclasses.replace(options.operator, ffma=FFmaOptions()))
    result = holoso.synthesize(dot, options, name="kernel")
    verilog = result.verilog_output.verilog
    assert "holoso_ffma #(" in verilog and "holoso_fadd #(" not in verilog
    residual = strip_locations(result.frontend_ir[-1])
    assert "    %2: float = intrinsic fadd(%0, %1)\n" in residual
    assert "    %4: float = intrinsic fadd(%2, %3)\n" in residual
    assert "    %6: float = intrinsic fadd(%4, %5)\n" in residual


def test_np_matmul_keyword_arguments_are_rejected() -> None:
    def keywords(a: Float64[np.ndarray, "2 2"], x: Float64[np.ndarray, "2"]) -> Float64[np.ndarray, "2"]:
        return np.matmul(a, x, subok=True)  # type: ignore[no-any-return]

    _refused(keywords, "keyword")


def test_augmented_assignment_to_array_is_rejected() -> None:
    # Regression: numpy '+=' / '@=' mutate in place while the frontend rebinds, so an alias would diverge; the array
    # augmented forms must be rejected in favor of the explicit 'x = x + ...' rebind.
    def name_target(v: Float64[np.ndarray, "2"], s: float) -> Float64[np.ndarray, "2"]:
        v += s
        return v

    _refused(name_target, "cannot update 'v' in place")

    @dataclasses.dataclass
    class State:
        P: Float64[np.ndarray, "2 2"]

        def step(self, f: Float64[np.ndarray, "2 2"]) -> None:
            self.P @= f

    _refused(State(np.eye(2)).step, "`@=` is not supported on arrays")

    def scalar_ok(a: float, s: float) -> float:
        a += s
        return a

    assert [p.name for p in _synth(scalar_ok).output_ports] == ["out_0"]


def test_unary_minus_on_boolean_aggregate_is_rejected_with_location() -> None:
    def f(a: bool, b: bool) -> float:
        v = np.array([a, b])
        return (-v)[0]  # type: ignore[no-any-return]

    _refused(f, "boolean")


def test_unsupported_operator_diagnostic_names_the_operator() -> None:
    # On the aggregate path an unsupported operator must be named even when the operands' shapes mismatch, rather than
    # being masked by the shape-mismatch diagnostic.
    def modulo(v: Float64[np.ndarray, "2"], w: Float64[np.ndarray, "3"]) -> Float64[np.ndarray, "2"]:
        return v % w  # type: ignore[no-any-return]

    _refused(modulo, "`%` is not supported on arrays")


def test_transpose_structure() -> None:
    def t(m: Float64[np.ndarray, "2 3"]) -> Float64[np.ndarray, "3 2"]:
        return m.T

    result = _synth(t)
    assert [p.name for p in result.output_ports] == ["out_0_0", "out_0_1", "out_1_0", "out_1_1", "out_2_0", "out_2_1"]
    # A pure reindexing: no hardware at all -- no pooled operator instantiations and no inline call sites remain after
    # stripping the unconditional support prelude -- at the combinational II.
    assert result.initiation_interval == (1, 1)
    assert "holoso_" not in strip_inline_prelude(result.verilog_output.verilog)
    m = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert np.array_equal(_run(result.numerical_model.elaborate(), m), m.T.flatten())

    def vector_identity(v: Float64[np.ndarray, "2"]) -> Float64[np.ndarray, "2"]:
        return v.T

    result_v = _synth(vector_identity)
    assert [p.name for p in result_v.output_ports] == ["out_0", "out_1"]
    assert np.array_equal(_run(result_v.numerical_model.elaborate(), np.array([1.0, -2.0])), [1.0, -2.0])

    def scalar_t(a: float) -> float:
        return a.T  # type: ignore[attr-defined, no-any-return]

    _refused(scalar_t, "a scalar has no supported attribute")


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

    # The reset snapshot is read at synthesis time, so the reference instance must stay untouched until then.
    sim = _sim(Holder(0.5, 1.0, 2.0).step)
    assert [p.name for p in sim.outputs] == ["out_0", "state_T", "state_ndim", "state_shape"]
    reference = Holder(0.5, 1.0, 2.0)
    for a in (0.25, -1.5, 3.0):
        want = reference.step(a)
        returned, state_t, state_ndim, state_shape = _run(sim, a)
        assert returned == pytest.approx(want)
        assert (state_shape, state_ndim, state_t) == pytest.approx((reference.shape, reference.ndim, reference.T))


def test_numpy_subscripts() -> None:
    def picks(m: Float64[np.ndarray, "2 3"]) -> tuple[float, float, float, float]:
        column = m[:, 2]
        return m[0, 1], m[1][2], column[0], m[1:, 0][0]

    result = _synth(picks)
    assert [p.name for p in result.output_ports] == ["out_0", "out_1", "out_2", "out_3"]
    m = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert np.array_equal(_run(result.numerical_model.elaborate(), m), np.asarray(picks(m)))

    def too_many(m: Float64[np.ndarray, "2 2"]) -> float:
        return m[0, 1, 0]  # type: ignore[no-any-return]

    _refused(too_many, "must name every axis")


def test_shaped_parameter_annotation_rejections() -> None:
    def symbolic(v: Float64[np.ndarray, "n"]) -> float:
        return v[0]  # type: ignore[no-any-return]

    _refused(symbolic, "fixed")

    def broadcastable(v: Float64[np.ndarray, "#3"]) -> float:
        return v[0]  # type: ignore[no-any-return]

    _refused(broadcastable, "fixed")

    def three_dee(v: Float64[np.ndarray, "2 2 2"]) -> float:
        return v[0, 0, 0]  # type: ignore[no-any-return]

    _refused(three_dee, "1-D and 2-D")

    def boolean(v: Bool[np.ndarray, "2"]) -> float:
        return v[0]  # type: ignore[no-any-return]

    _refused(boolean, "float or integer family")

    def integer(v: Int[np.ndarray, "2"]) -> int:
        return v[0]  # type: ignore[no-any-return]

    # The integer family is supported alongside the float one.
    assert [p.name for p in _synth(integer).input_ports] == ["in_v_0", "in_v_1"]

    def shape_only(v: Shaped[np.ndarray, "2"]) -> float:
        return v[0]  # type: ignore[no-any-return]

    _refused(shape_only, "float or integer family")

    def shapeless(v: np.ndarray) -> float:
        return v[0]  # type: ignore[no-any-return]

    _refused(shapeless, "annotation of parameter 'v' is not supported")

    class _FakeArray:  # structurally array-like (has ``dims``) but its dims is not a real jaxtyping tuple
        dims = None

    def fake(v: _FakeArray) -> float:
        return 1.0

    _refused(fake, "only numpy array containers are supported")


def test_wide_float_dtype_annotation_is_accepted() -> None:
    def f(v: Float[np.ndarray, "2"]) -> float:
        return v[0] + v[1]  # type: ignore[no-any-return]

    assert [p.name for p in _synth(f).input_ports] == ["in_v_0", "in_v_1"]


def test_array_return_annotation_is_validated() -> None:
    def good(v: Float64[np.ndarray, "2"]) -> Float64[np.ndarray, "2"]:
        return v * 2.0

    assert [p.name for p in _synth(good).output_ports] == ["out_0", "out_1"]

    def nested(v: Float64[np.ndarray, "2"], flag: bool) -> tuple[Float64[np.ndarray, "2"], bool]:
        return v * 2.0, flag

    assert [p.name for p in _synth(nested).output_ports] == ["out_0_0", "out_0_1", "out_1"]

    def wrong_shape(v: Float64[np.ndarray, "2"]) -> Float64[np.ndarray, "3"]:
        return v * 2.0

    _refused(wrong_shape, "the annotation declares")

    def scalar_returned(v: Float64[np.ndarray, "2"]) -> Float64[np.ndarray, "2"]:
        return v[0]  # type: ignore[no-any-return]

    _refused(scalar_returned, "the returned value is not an array")

    def boolean_leaves(flag: bool) -> Float64[np.ndarray, "1"]:
        return [flag]  # type: ignore[return-value]

    _refused(boolean_leaves, "the returned value is not an array")


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

    assert [o.name for o in lower(indexed).hir.outputs] == ["out_0"]
    assert float(_sim(indexed).run(1.0, 2.0, 3.0)[0]) == 3.0  # _INDEX_CONST[0] == 2, so the pick is v[2]

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

    hir = lower(Filter(np.array([[0.0, 1.0]])).step).hir
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

    hir_slice = lower(WithSlice(0.0).step).hir
    assert [s.name for s in hir_slice.state_slots] == []
    assert [o.name for o in hir_slice.outputs] == ["out_0"]

    @dataclasses.dataclass
    class WithTranspose:
        y: float

        def step(self, a: float) -> float:
            if _GATE_CONST2.T[1, 0] < 0.0:  # _GATE_CONST2.T[1, 0] == _GATE_CONST2[0, 1] == 1.0, statically false
                self.y = a  # statically dead
            return a

    hir_t = lower(WithTranspose(0.0).step).hir
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

    hir_f = lower(WithFlatten(0.0).step).hir
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
        Holder(np.array([[1.0, 2.0], [3.0, 4.0]])).step, default_options(_FMT), name="pt"
    ).numerical_model.elaborate()
    got = _run(sim, 2.0).reshape(2, 2)
    assert np.allclose(got, np.array([[1.0, 2.0], [3.0, 4.0]]).T * 2.0)


def test_unary_plus_rejects_boolean_but_is_identity_on_floats() -> None:
    # Regression: unary plus skipped the boolean guard the other arithmetic operators apply, silently passing a bool
    # through (Python's +True is int 1, which has no runtime type here).
    def scalar(flag: bool) -> float:
        return +flag

    _refused(scalar, "boolean")

    def aggregate(a: bool, b: bool) -> Float64[np.ndarray, "2"]:
        return +np.array([a, b])

    _refused(aggregate, "boolean")

    def floats(v: Float64[np.ndarray, "2"]) -> Float64[np.ndarray, "2"]:
        return +v

    result = _synth(floats)
    assert [p.name for p in result.output_ports] == ["out_0", "out_1"]
    assert np.array_equal(_run(result.numerical_model.elaborate(), np.array([1.5, -2.0])), [1.5, -2.0])


def test_ndarray_module_constant_rejections() -> None:
    def boolean(a: float) -> float:
        return _BOOL_CONST[0]  # type: ignore[no-any-return]

    _refused(boolean, "not a supported aggregate")

    def three_dee(a: float) -> float:
        return _CUBE_CONST[0, 0, 0]  # type: ignore[no-any-return]

    _refused(three_dee, "works only on an array")


def test_ndarray_subclass_constant_and_state_are_rejected() -> None:
    # Regression: an ndarray subclass (np.matrix) redefines operators (``*`` is matmul), so folding it as a plain array
    # would silently diverge from its own Python semantics; it must be rejected, both as a module constant and a reset.
    def constant(a: float) -> float:
        return _MATRIX_CONST[0, 1] + a  # type: ignore[no-any-return]

    _refused(constant, "not a captured object")

    @dataclasses.dataclass
    class Stateful:
        P: np.ndarray

        def step(self, a: float) -> None:
            self.P = self.P * a

    _refused(Stateful(_np_matrix()).step, "numpy subclass")


def test_power_of_boolean_is_rejected_with_location() -> None:
    # Regression: 'flag**1' bypassed the boolean-operand guard applied to the other arithmetic operators, silently
    # returning the bool instead of a source-located UnsupportedConstruct.
    def first_power(flag: bool) -> float:
        return flag**1

    _refused(first_power, "boolean")


def _np_matrix() -> np.ndarray:
    # np.matrix is an ndarray subclass with different operator semantics; construct it under a warning filter since it
    # is deliberately used to check that such subclasses are rejected.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PendingDeprecationWarning)
        return np.matrix([[1.0, 2.0], [3.0, 4.0]])


_BOOL_CONST = np.array([True, False])
_CUBE_CONST = np.zeros((2, 2, 2))
_MATRIX_CONST = _np_matrix()
_GATE_CONST = np.array([0.0, 1.0])
_GATE_CONST2 = np.array([[0.0, 1.0], [1.0, 0.0]])
_INDEX_CONST = np.array([2, 0, 1])


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


def test_stateful_kalman_style_filter_matches_numpy_across_transactions() -> None:
    # Locks in the whole matrix feature surface composed in one stateful kernel across transactions: matrix/vector
    # parameters and carried state, ndarray module constants, ``@`` in every shape with transpose, elementwise scalar
    # broadcast, an annotated local, a static row loop, a shaped return, and the runtime-divisor Kalman gain.
    sim = holoso.synthesize(TrackingFilter().update, default_options(_FMT), name="tracker").numerical_model.elaborate()
    assert [p.name for p in sim.outputs] == [
        "out_0", "out_1", "state_P_0_0", "state_P_0_1", "state_P_1_0", "state_P_1_1", "state_x_0", "state_x_1",
    ]  # fmt: skip
    reference = TrackingFilter()
    rng = np.random.default_rng(0xF117E5)
    F = np.array([[1.0, 0.1], [0.0, 1.0]])
    for step in range(6):
        z = np.array([float(rng.uniform(-1.0, 1.0)), float(rng.uniform(-1.0, 1.0))])
        got = _run(sim, F, z)
        prediction = reference.update(F, z)
        want = np.array([float(v) for v in (*prediction, *reference.P.flatten(), *reference.x)])
        assert np.all(np.isfinite(want))
        assert np.allclose(got, want, rtol=1e-9, atol=1e-12), step


def test_imu_frame_transform_example_matches_numpy() -> None:
    # The bundled 3D rigid-body / IMU frame transform example must lower and agree with its own plain-numpy execution,
    # confirming the matmul/transpose/broadcast composition it demonstrates is valid, runnable Python.
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
    options = default_options(_FMT)
    options = dataclasses.replace(options, operator=dataclasses.replace(options.operator, ffma=FFmaOptions()))
    sim = holoso.synthesize(imu_frame_transform.transform, options, name="imu_fma").numerical_model.elaborate()
    yaw90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    roll90 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    for rotation in (yaw90, roll90):
        inputs = (rotation, np.array([1.0, 2.0, 3.0]), np.array([0.1, -0.2, 9.9]), np.array([2.0, 0.0, -1.0]))
        want = np.asarray(imu_frame_transform.transform(*inputs)).flatten()
        assert np.allclose(_run(sim, *inputs), want, rtol=1e-9, atol=1e-300)


# ---------------------------------------------------------------- linear algebra library functions


def test_operators_are_the_library_functions() -> None:
    # Two keys on one entry, so an operator and its spelled call give byte-identical Verilog under one module name --
    # identical RTL, not merely identical values.
    def with_operators(a: Float64[np.ndarray, "2 3"], b: Float64[np.ndarray, "2 3"]) -> Float64[np.ndarray, "2 2"]:
        return a @ b.T  # type: ignore[no-any-return]

    def with_calls(a: Float64[np.ndarray, "2 3"], b: Float64[np.ndarray, "2 3"]) -> Float64[np.ndarray, "2 2"]:
        return np.matmul(a, np.transpose(b))  # type: ignore[no-any-return]

    spelled = [_synth(k).verilog_output.verilog for k in (with_operators, with_calls)]
    assert spelled[0] == spelled[1]
    _assert_python_matches_holoso(
        with_operators, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), np.array([[0.5, -1.0, 2.0], [3.0, -2.0, 0.25]])
    )


def test_np_trace_and_np_outer() -> None:
    def tr(m: Float64[np.ndarray, "3 3"]) -> float:
        return np.trace(m)  # type: ignore[no-any-return]

    result = _synth(tr)
    assert [p.name for p in result.output_ports] == ["out_0"]
    verilog = result.verilog_output.verilog
    assert "holoso_fadd #(" in verilog and "holoso_fmul #(" not in verilog  # a fold of the diagonal, no multiplies
    m = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    assert _run(result.numerical_model.elaborate(), m)[0] == np.trace(m)

    def outer(u: Float64[np.ndarray, "2"], v: Float64[np.ndarray, "3"]) -> Float64[np.ndarray, "2 3"]:
        return np.outer(u, v)

    result_o = _synth(outer)
    verilog_o = result_o.verilog_output.verilog
    assert "holoso_fmul #(" in verilog_o and "holoso_fadd #(" not in verilog_o  # products only, no sums
    u, w = np.array([1.0, -2.0]), np.array([0.5, 3.0, -1.0])
    assert np.array_equal(_run(result_o.numerical_model.elaborate(), u, w), np.outer(u, w).flatten())

    def rect_trace(m: Float64[np.ndarray, "2 3"]) -> float:
        return np.trace(m)  # type: ignore[no-any-return]

    # numpy walks the shorter diagonal; Holoso rejects rather than reinterpreting.
    _refused(rect_trace, "square")

    def vec_trace(v: Float64[np.ndarray, "3"]) -> float:
        return np.trace(v)  # type: ignore[no-any-return]

    _refused(vec_trace, r"in np\.trace\(\): ValueError: trace requires a matrix, got a 1-D value")

    def outer_of_matrix(m: Float64[np.ndarray, "2 2"]) -> Float64[np.ndarray, "2 2"]:
        return np.outer(m, m)

    _refused(outer_of_matrix, "1-D")


def test_trace_of_a_1x1_boolean_matrix_is_rejected_like_a_larger_one() -> None:
    # The diagonal fold is seeded at 0.0, so even a 1x1 trace contracts through an addition and rejects a boolean
    # diagonal, rather than passing the boolean through where numpy would widen it to an integer.
    def bool_trace(flag: bool) -> bool:
        return np.trace(np.array([[flag]]))  # type: ignore[no-any-return]

    _refused(bool_trace, "must hold numbers, not booleans")


def test_library_shape_rejection_is_attributed_to_the_user_call_site() -> None:
    # A stub validates its own operands with a ``raise`` on a statically taken path; the error must name the user's
    # spelling and point at the user's line, never into the stub source.
    def bad(a: Float64[np.ndarray, "2 3"], x: Float64[np.ndarray, "2"]) -> Float64[np.ndarray, "2"]:
        return a @ x  # type: ignore[no-any-return]

    with pytest.raises(UnsupportedConstruct, match=r"in @\(\).*mismatch") as excinfo:
        _synth(bad)
    assert excinfo.value.location is not None
    assert excinfo.value.location.line is not None and "a @ x" in excinfo.value.location.line

    def bad_t(a: float) -> float:
        return np.transpose(a)  # type: ignore[return-value]

    _refused(bad_t, r"in np\.transpose\(\).*transpose a scalar")


def test_matrix_state_transposed_under_a_shape_guard_across_transactions() -> None:
    # The reset snapshot fixes the shape, so the guard folds identically in the scan and in lowering, while the
    # attribute itself is reassigned every transaction. Compare every port by NAME: the returned leaf is deduped onto
    # the public state port that already carries it, so a positional comparison would read the wrong wire.
    class Flip:
        def __init__(self) -> None:
            self.P = np.array([[1.0, 2.0], [3.0, 4.0]])
            self.s = 0.0

        def step(self, x: float) -> float:
            self.P = np.array(
                self.P.T
            )  # an explicit copy: the bare view shares the storage of the slot it installs into
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
