"""
Acceptance gate + independence guard for the MIR interpreter (``holoso._mir.MirInterpreter``).

The interpreter is the schedule-independent bit-exact oracle: it evaluates the MIR dataflow graph directly, sharing the
front/mid-end and ``operator.evaluate`` with the numerical model but NONE of the LIR scheduling/binding/regalloc/overlap
machinery. Before it can be trusted as an oracle it must agree bit-for-bit with the numerical model on kernels that are
already known correct -- every bundled example (validated against Python in ``test_example_reference``) and every
scheduling corner kernel (validated against RTL in the cosim suite). A disagreement here means the interpreter is wrong,
not the compiler; only once this gate is green does an interpreter-vs-model divergence elsewhere indict the LIR layer.

The independence guard asserts the interpreter's TRANSITIVE import closure excludes ``holoso._lir`` -- the layer it
exists to verify -- so the oracle can never silently re-couple to the artifact under test, even through an intermediary.
"""

from collections.abc import Callable

import numpy as np
import pytest

from holoso._backend.numerical import NumericalSimulator
from holoso._mir import MirInterpreter
from holoso._operators import OpConfig
from holoso._type import BoolType, FloatFormat
from holoso._value import FloatValue

from ._examples import SPECS, ExampleSpec, parity_marks
from ._importguard import forbidden_imports
from ._modelref import (
    ChainedSlots,
    SelectHold,
    SlotSwap,
    Vector,
    assert_model_equals_interpreter,
    bool_phi_swap_computed_loop,
    branch_boundary_kernel,
    branchy_swap_mixed_arm_loop,
    build_model_and_interpreter,
    const_branch_kernel,
    default_ops,
    diamond_then_loop_kernel,
    overlap_dead_arm_spill_kernel,
    overlap_div_err_kernel,
    overlap_drained_passthrough_kernel,
    overlap_livein_branch_arm_kernel,
    overlap_spill_kernel,
    phi_swap_computed_loop,
    phi_swap_loop,
    random_legal_bits,
    staged_ops,
)


def _decode_spec_vector(model: NumericalSimulator, fmt: FloatFormat, row: dict[str, int]) -> Vector:
    vector: Vector = []
    for port in model.inputs:
        bits = row[port.name]
        vector.append(bool(bits) if isinstance(port.scalar_type, BoolType) else FloatValue.from_bits(fmt, bits))
    return vector


_EXAMPLE_CASES = [
    pytest.param(spec, fmt, id=f"{spec.name}-e{fmt.wexp}m{fmt.wman}", marks=parity_marks(spec.name))
    for spec in SPECS
    for fmt in spec.formats
]


@pytest.mark.parametrize("spec,fmt", _EXAMPLE_CASES)
def test_interpreter_matches_model_on_examples(spec: ExampleSpec, fmt: FloatFormat) -> None:
    model, interpreter = build_model_and_interpreter(spec.make_kernel(), default_ops(fmt), spec.name)
    vectors = [_decode_spec_vector(model, fmt, row) for row in spec.vectors(fmt)]
    assert_model_equals_interpreter(model, interpreter, vectors, spec.name)


# The scheduling corner kernels, driven with bounded random inputs (magnitudes kept modest so the data-dependent
# diamond-then-loop trip count stays small and no operation overflows the coarse format). Each is exercised at both the
# minimum-latency and the deeply-pipelined operator configs, since the model's timing -- but never the interpreter's --
# changes with latency, so this cross-checks the model across schedules against the one fixed reference.
_CORNER_KERNELS: list[tuple[str, Callable[[], Callable[..., object]]]] = [
    ("branch_boundary", lambda: branch_boundary_kernel),
    ("overlap_spill", lambda: overlap_spill_kernel),
    ("overlap_dead_arm_spill", lambda: overlap_dead_arm_spill_kernel),
    ("const_branch", lambda: const_branch_kernel),
    ("diamond_then_loop", lambda: diamond_then_loop_kernel),
    ("overlap_div_err", lambda: overlap_div_err_kernel),
    ("overlap_drained_passthrough", lambda: overlap_drained_passthrough_kernel),
    ("overlap_livein_branch_arm", lambda: overlap_livein_branch_arm_kernel),
    ("chained_slots", lambda: ChainedSlots().__call__),
    ("select_hold", lambda: SelectHold().step),
]


def _bounded_vectors(model: NumericalSimulator, fmt: FloatFormat, rng: np.random.Generator, count: int) -> list[Vector]:
    vectors: list[Vector] = []
    for _ in range(count):
        vector: Vector = []
        for port in model.inputs:
            if isinstance(port.scalar_type, BoolType):
                vector.append(bool(rng.integers(0, 2)))
            else:
                vector.append(FloatValue.from_float(fmt, float(rng.uniform(0.5, 3.5))))
        vectors.append(vector)
    return vectors


@pytest.mark.parametrize("ops_factory", [default_ops, staged_ops], ids=["default", "staged"])
@pytest.mark.parametrize("label,make_kernel", _CORNER_KERNELS, ids=[name for name, _ in _CORNER_KERNELS])
def test_interpreter_matches_model_on_corners(
    label: str, make_kernel: Callable[[], Callable[..., object]], ops_factory: Callable[[FloatFormat], OpConfig]
) -> None:
    fmt = FloatFormat(6, 18)
    model, interpreter = build_model_and_interpreter(make_kernel(), ops_factory(fmt), label)
    rng = np.random.default_rng(0xC0FFEE)
    vectors = _bounded_vectors(model, fmt, rng, 64)
    assert_model_equals_interpreter(model, interpreter, vectors, f"{label}-{ops_factory.__name__}")


def test_interpreter_matches_model_on_edge_bits() -> None:
    fmt = FloatFormat(6, 18)
    model, interpreter = build_model_and_interpreter(branch_boundary_kernel, default_ops(fmt), "branch_boundary")
    rng = np.random.default_rng(0x5EED)
    vectors: list[Vector] = [
        [FloatValue.from_bits(fmt, random_legal_bits(fmt, rng)) for _ in model.inputs] for _ in range(256)
    ]
    assert_model_equals_interpreter(model, interpreter, vectors, "branch_boundary-edgebits")


def test_loop_header_phi_swap_resolves_in_parallel() -> None:
    """
    A loop-header phi swap (``a, b = b, a``) must resolve its cross-referencing phis as a parallel snapshot. Checked
    against the float64 Python reference (which swaps correctly) AND interp==model, so a sequential-phi regression in
    either oracle -- or one shared by both -- surfaces as a divergence. Integer-valued inputs keep every output exact.
    """
    fmt = FloatFormat(6, 18)
    model, interpreter = build_model_and_interpreter(phi_swap_loop, default_ops(fmt), "phi_swap_loop")
    for x in (2.0, -3.0, 0.5, 5.0):
        for n in (1.0, 2.0, 3.0, 4.0):
            vector = [FloatValue.from_float(fmt, x), FloatValue.from_float(fmt, n)]
            model_out = model.run(*vector)
            interp_out = interpreter.run(*vector)
            assert model_out == interp_out, f"interp != model at x={x} n={n}"
            reference = phi_swap_loop(x, n)
            assert (
                float(model_out[0]) == reference
            ), f"model != python at x={x} n={n}: {float(model_out[0])} vs {reference}"


def test_loop_header_phi_swap_with_computed_arm_resolves_in_parallel() -> None:
    """
    The computed-arm swap (``a, b = b, a + x``): the latch mixes a phi-sourced install with a computed-source install,
    so a placement that fires one tail install after a sibling has overwritten its source register miscompiles even
    though the pure swap (same-placement installs) stays correct. Python is the oracle; the model and the RTL replay
    the same LIR, and interp==model is asserted so a divergence localizes the guilty layer.
    """
    fmt = FloatFormat(6, 18)
    model, interpreter = build_model_and_interpreter(phi_swap_computed_loop, default_ops(fmt), "phi_swap_computed")
    for x in (1.0, 2.0, -1.5):
        for n in (1.0, 2.0, 3.0, 4.0):
            vector = [FloatValue.from_float(fmt, x), FloatValue.from_float(fmt, n)]
            model_out = model.run(*vector)
            interp_out = interpreter.run(*vector)
            reference = phi_swap_computed_loop(x, n)
            assert (
                float(interp_out[0]) == reference
            ), f"interp != python at x={x} n={n}: {float(interp_out[0])} vs {reference}"
            assert model_out == interp_out, f"interp != model at x={x} n={n}"


def test_bool_loop_header_phi_swap_with_computed_arm_resolves_in_parallel() -> None:
    """The boolean-bank twin of the computed-arm swap: the latch installs are BoolWrites, not FloatCopys."""
    fmt = FloatFormat(6, 18)
    model, interpreter = build_model_and_interpreter(
        bool_phi_swap_computed_loop, default_ops(fmt), "bool_phi_swap_computed"
    )
    for x in (False, True):
        for n in (1.0, 2.0, 3.0, 4.0):
            vector: Vector = [x, FloatValue.from_float(fmt, n)]
            model_out = model.run(*vector)
            interp_out = interpreter.run(*vector)
            reference = bool_phi_swap_computed_loop(x, n)
            assert (
                bool(interp_out[0]),
                bool(interp_out[1]),
            ) == reference, f"interp != python at x={x} n={n}: {interp_out} vs {reference}"
            assert model_out == interp_out, f"interp != model at x={x} n={n}"


def test_mixed_arm_swap_diamond_builds_and_matches_python() -> None:
    """
    Pins transient tolerance in the install fixpoint: under a narrowing classification with a strict interference
    residence bound this kernel does not even BUILD (the boundary shrinks below a transiently de-coalesced computed
    arm's install and the residence assert trips), so the value grid is secondary to synthesis itself.
    """
    fmt = FloatFormat(6, 18)
    model, interpreter = build_model_and_interpreter(branchy_swap_mixed_arm_loop, default_ops(fmt), "mixed_arm_swap")
    for x in (1.0, -1.0):
        for n in (1.0, 2.0, 3.0):
            vector = [FloatValue.from_float(fmt, x), FloatValue.from_float(fmt, 4.0), FloatValue.from_float(fmt, n)]
            model_out = model.run(*vector)
            interp_out = interpreter.run(*vector)
            reference = branchy_swap_mixed_arm_loop(x, 4.0, n)
            assert (float(interp_out[0]), float(interp_out[1])) == reference, f"interp != python at x={x} n={n}"
            assert model_out == interp_out, f"interp != model at x={x} n={n}"


def test_state_slot_swap_writeback_is_parallel() -> None:
    """
    Two persistent slots that swap (``self._a, self._b = self._b, self._a``) force the parallel, read-first state
    writeback to exchange their registers from OLD values. Checked against the float64 Python reference (which swaps
    correctly) AND interp==model, so a sequential-writeback regression in either oracle -- or one shared by both --
    surfaces. Integer inputs keep the output exact.
    """
    fmt = FloatFormat(6, 18)
    model, interpreter = build_model_and_interpreter(SlotSwap().step, default_ops(fmt), "slot_swap")
    reference = SlotSwap()
    for x in (0.0, 1.0, -2.0, 3.0, -4.0, 5.0):
        vector = [FloatValue.from_float(fmt, x)]
        model_out = model.run(*vector)
        interp_out = interpreter.run(*vector)
        assert model_out == interp_out, f"interp != model at x={x}"
        expected = reference.step(x)
        assert float(model_out[0]) == expected, f"model != python at x={x}: {float(model_out[0])} vs {expected}"


def test_interpreter_imports_nothing_from_lir() -> None:
    """
    The oracle must never re-couple to the layer it verifies: its TRANSITIVE import closure excludes ``holoso._lir``.
    """
    offenders = forbidden_imports(MirInterpreter.__module__, "holoso._lir")
    assert not offenders, f"interpreter transitively imports the LIR layer it verifies: {offenders}"
