"""
Behavioral validation of every compilable example against its ORIGINAL Python execution.

The cosimulation suite (``test_cosim_examples.py``) checks the emitted RTL against the kernel's EMBEDDED numerical
model -- but both descend from the same front-end lowering, so a front-end miscompile poisons the RTL and the model
identically and the bit-for-bit check still passes. That suite proves ``RTL == compiler-model``; it cannot prove
``compiler-model == Python semantics``. This module closes that gap: it drives each example's numerical model AND a
fresh plain-Python instance of the same kernel over ``reference_vectors()`` (the manual sequence then the random draw)
and asserts they agree. Boolean lanes and float lanes without a budget must match bit-for-bit; a lane whose arithmetic
accumulates rounding carries a spec-owned independent ``OutputTolerance`` (scaled by its own reference magnitude and
growing with the recurrence age for carried state), so a compiler defect cannot loosen its own oracle.
Inputs are quantized into the format first, so the model and the reference see the same operands and only the
per-operation rounding differs.
The per-input format-edge sweep is excluded here -- the model legitimately diverges from float64 at the format extremes
(an operation overflowing to the format's infinity stays finite in float64), which the cosim suite covers instead.

The example specs are shared with the cosimulation suite via ``_examples``: the cosim suite drives the full
``raw_vectors()`` (manual + random + edges), this suite the ``reference_vectors()`` subset, over one source of truth.
"""

import dataclasses
from collections.abc import Callable, Mapping

import numpy as np
import pytest

import holoso
from holoso import BoolType, FloatFormat, FloatType, IntType
from ._examples import SPECS, ExampleSpec, InputVector, OutputTolerance
from ._modelref import default_options, flatten_value, port_name, within

_STATE_PREFIX = "state_"

_CASES = [
    pytest.param(spec, spec.formats[0], id=f"{spec.name}-e{spec.formats[0].wexp}m{spec.formats[0].wman}")
    for spec in SPECS
    if spec.reference is not None
]


def _quantize(value: float | bool | int, fmt: FloatFormat) -> float | bool | int:
    """A float rounded into ``fmt``, so the model and the float64 reference get one operand; a bool/int is exact."""
    return value if isinstance(value, (bool, int)) else fmt.decode(fmt.encode(value))


def _declared_family(scalar_type: object) -> type:
    match scalar_type:
        case BoolType():
            return bool
        case IntType():
            return int
        case FloatType():
            return float
    raise AssertionError(scalar_type)


def _family_of(value: object) -> type:
    """The family a value actually carries, on either side: the model's typed scalars and Python's own numbers."""
    if isinstance(value, (bool, np.bool_)):
        return bool
    if isinstance(value, (int, np.integer, holoso.IntValue)):
        return int
    return float


def _assert_model_matches_reference(
    label: str,
    model: holoso.NumericalSimulator,
    reference: Callable[..., object],
    inputs: tuple[str, ...],
    vectors: list[InputVector],
    fmt: FloatFormat,
    tolerances: Mapping[str, OutputTolerance],
) -> None:
    """
    Advance the model and the plain-Python reference in lockstep, matching outputs BY NAME: a returned leaf maps to
    its ``out_<path>`` port via ``port_name``, a ``state_<attr>`` port reads the reference instance's attribute, and
    a returned leaf without a port must equal a public state live-out the model folded into its ``state_`` port.
    Both sides must carry the port's declared scalar family before any coercion, so a bool/float swap cannot hide.
    """
    instance = getattr(reference, "__self__", None)
    assert [port.name for port in model.inputs] == list(inputs), f"{label}: input ports differ from the spec"
    port_names = {port.name for port in model.outputs}
    assert set(tolerances) <= port_names, f"{label}: budgets for nonexistent ports {set(tolerances) - port_names}"
    for age, row in enumerate(vectors):
        quantized = {name: _quantize(value, fmt) for name, value in row.items()}
        got = model.run(*[quantized[port.name] for port in model.inputs])
        result = reference(*[quantized[name] for name in inputs])
        return_leaves = {port_name(path): leaf for path, leaf in flatten_value(result)}
        expected: dict[str, float | bool | int] = {}
        for port in model.outputs:
            if port.name.startswith(_STATE_PREFIX):
                value = getattr(instance, port.name[len(_STATE_PREFIX) :])
                assert not isinstance(value, (list, tuple, np.ndarray)), f"{label}: unexpected vector public state"
                expected[port.name] = value
            else:
                assert port.name in return_leaves, f"{label}: port {port.name} has no leaf in {sorted(return_leaves)}"
                expected[port.name] = return_leaves[port.name]
        state_live_outs = [value for name, value in expected.items() if name.startswith(_STATE_PREFIX)]
        for leaf_name, leaf in return_leaves.items():
            if leaf_name not in port_names:
                assert any(
                    isinstance(leaf, bool) == isinstance(value, bool) and leaf == value for value in state_live_outs
                ), f"{label} {row}: return leaf {leaf_name} has no port and matches no public state live-out"
        for port, got_value in zip(model.outputs, got, strict=True):
            want = expected[port.name]
            family = _declared_family(port.scalar_type)
            assert (
                _family_of(got_value) is family
            ), f"{label} {row} {port.name}: model value {got_value!r} is not of the declared {port.scalar_type}"
            assert (
                _family_of(want) is family
            ), f"{label} {row} {port.name}: reference value {want!r} is not of the declared {port.scalar_type}"
            if family is bool:
                assert bool(got_value) == bool(want), f"{label} {row} {port.name}: {bool(got_value)} != {bool(want)}"
            elif family is int:
                assert isinstance(got_value, holoso.IntValue) and isinstance(want, int)
                assert int(got_value) == want, f"{label} {row} {port.name}: {int(got_value)} != {want}"
            else:
                budget = tolerances.get(port.name)
                atol = 0.0 if budget is None else budget.allowance(fmt, float(want), age)
                assert within(
                    float(got_value), float(want), 0.0, atol
                ), f"{label} {row} {port.name}: {float(got_value)} vs {float(want)} (atol={atol:g})"


@pytest.mark.parametrize("spec,fmt", _CASES)
def test_example_matches_python_reference(spec: ExampleSpec, fmt: FloatFormat) -> None:
    model = holoso.synthesize(spec.make_kernel(), spec.options(fmt), name=spec.name).numerical_model.elaborate()
    assert spec.reference is not None
    _assert_model_matches_reference(
        spec.name, model, spec.make_kernel(), spec.inputs, spec.reference_vectors(), fmt, spec.reference
    )


class _StateLeafAheadOfComputed:
    def __init__(self) -> None:
        self.acc = 0.0

    def __call__(self, x: float) -> tuple[float, float]:
        self.acc = self.acc + x
        return self.acc, x * 4.0


def test_a_folded_state_leaf_ahead_of_a_computed_leaf_maps_by_name() -> None:
    """
    The FIRST returned leaf is public state (folded into ``state_acc``; only the second leaf keeps an ``out_`` port),
    so a mapper pairing ``out_`` ports with the leading return leaves positionally would compare the computed port
    against the state leaf and fail. All values are dyadic, so the comparison is bit-exact with no budget.
    """
    fmt = FloatFormat(6, 18)
    model = holoso.synthesize(
        _StateLeafAheadOfComputed().__call__, default_options(fmt), name="state_leaf_first"
    ).numerical_model.elaborate()
    _assert_model_matches_reference(
        "state_leaf_first",
        model,
        _StateLeafAheadOfComputed().__call__,
        ("x",),
        [{"x": 1.0}, {"x": 2.0}, {"x": -0.5}],
        fmt,
        {},
    )
