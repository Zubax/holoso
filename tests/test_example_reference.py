"""
Behavioral validation of every compilable example against its ORIGINAL Python execution.

The cosimulation suite (`test_cosim_examples.py`) checks the emitted RTL against the kernel's EMBEDDED numerical
model -- but both descend from the same front-end lowering, so a front-end miscompile poisons the RTL and the model
identically and the bit-for-bit check still passes. That suite proves `RTL == compiler-model`; it cannot prove
`compiler-model == Python semantics`. This module closes that gap: it drives each example's numerical model AND a
fresh plain-Python instance of the same kernel over `reference_vectors()` (the manual sequence then the random draw)
and asserts they agree. Boolean lanes and float lanes without a budget must match bit-for-bit; a lane whose arithmetic
accumulates rounding carries a spec-owned independent `OutputTolerance` (scaled by its own reference magnitude and
growing with the recurrence age for carried state), so a compiler defect cannot loosen its own oracle.
Inputs are quantized into the format first, so the model and the reference see the same operands and only the
per-operation rounding differs.
The per-input format-edge sweep is excluded here -- the model legitimately diverges from float64 at the format extremes
(an operation overflowing to the format's infinity stays finite in float64), which the cosim suite covers instead.

The example specs are shared with the cosimulation suite via `_examples`: the cosim suite drives the full
`raw_vectors()` (manual + random + edges), this suite the `reference_vectors()` subset, over one source of truth.
"""

import dataclasses
import itertools
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pytest

import holoso
from holoso import BoolType, FloatFormat, FloatType, IntType
from ._examples import SPECS, ExampleSpec, InputVector, OutputTolerance
from majority_voter import MajorityVoter  # noqa: E402  # _examples put examples/ on the path
from uart import UartTx  # noqa: E402
from ._eeloracle import walk_instance_leaves
from ._modelref import default_options, flatten_value, port_name, within

_STATE_PREFIX = "state_"

_CASES = [
    pytest.param(spec, spec.formats[0], id=f"{spec.name}-e{spec.formats[0].wexp}m{spec.formats[0].wman}")
    for spec in SPECS
    if spec.reference is not None
]


def _quantize(value: float | bool | int, fmt: FloatFormat) -> float | bool | int:
    """A float rounded into `fmt`, so the model and the float64 reference get one operand; a bool/int is exact."""
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


def _state_leaves(instance: object) -> dict[str, Any]:
    """
    Public attribute leaves named exactly as the compiler decomposes state slots (a scalar keeps its name, an
    aggregate flattens row-major, a nested component prefixes its own leaves) -- the forward direction of
    `_eeloracle.walk_instance_leaves`. The name alone cannot prove which attribute a port belongs to (aggregate
    `q` and scalar `q_0` both mint `q_0`), so a name minted twice is dropped and its lookup fails loudly instead
    of comparing against a guess.
    """
    leaves: dict[str, Any] = {}
    ambiguous: set[str] = set()
    for name, private, leaf in walk_instance_leaves(instance):
        if private:
            continue
        if name in leaves:
            ambiguous.add(name)
        leaves[name] = leaf
    return {name: leaf for name, leaf in leaves.items() if name not in ambiguous}


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
    its `out_<path>` port via `port_name`, a `state_<attr>` port reads the reference instance's attribute, and
    a returned leaf without a port must equal a public state live-out the model folded into its `state_` port.
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
        return_leaves = {} if result is None else {port_name(path): leaf for path, leaf in flatten_value(result)}
        state_leaves = {} if instance is None else _state_leaves(instance)
        expected: dict[str, float | bool | int] = {}
        for port in model.outputs:
            if port.name.startswith(_STATE_PREFIX):
                name = port.name[len(_STATE_PREFIX) :]
                assert name in state_leaves, f"{label}: port {port.name} matches no unambiguous public state leaf"
                expected[port.name] = state_leaves[name]
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
                assert isinstance(got_value, holoso.IntValue) and isinstance(want, (int, np.integer))
                assert int(got_value) == int(want), f"{label} {row} {port.name}: {int(got_value)} != {int(want)}"
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


def test_flux_observer_matches_python_reference() -> None:
    """
    The generic suite skips flux_observer (`reference=None`: its carried flux/i_last are VECTOR state the scalar
    attribute reader cannot address), so this bespoke twin maps each `state_<attr>_<k>` port onto the reference
    instance's array element. The `i_last` lanes are an identity install of the quantized input, so they must match
    bit-for-bit; the rounded lanes carry explicit budgets (the atan2 CORDIC is faithful rather than exact, and the
    flux recurrence accumulates rounding through carried state at the flux-linkage scale).
    """
    spec = next(spec for spec in SPECS if spec.name == "flux_observer")
    fmt = spec.formats[0]
    model = holoso.synthesize(spec.make_kernel(), spec.options(fmt), name=spec.name).numerical_model.elaborate()
    reference = spec.make_kernel()
    instance = reference.__self__  # type: ignore[attr-defined]
    budgets = {
        "out_0": OutputTolerance(ulps=64),
        "state_flux_0": OutputTolerance(ulps=64, growth_ulps=1, floor=0.005),
        "state_flux_1": OutputTolerance(ulps=64, growth_ulps=1, floor=0.005),
    }
    for age, row in enumerate(spec.reference_vectors()):
        quantized = {name: _quantize(value, fmt) for name, value in row.items()}
        got = {
            port.name: float(value)
            for port, value in zip(model.outputs, model.run(*[quantized[port.name] for port in model.inputs]))
        }
        theta = reference(
            quantized["dt"],
            np.array([quantized["u_ab_0"], quantized["u_ab_1"]]),
            np.array([quantized["i_ab_0"], quantized["i_ab_1"]]),
        )
        assert isinstance(theta, float)
        expected = {
            "out_0": theta,
            "state_flux_0": float(instance.flux[0]),
            "state_flux_1": float(instance.flux[1]),
            "state_i_last_0": float(instance.i_last[0]),
            "state_i_last_1": float(instance.i_last[1]),
        }
        assert got.keys() == expected.keys()
        for name, want in expected.items():
            budget = budgets.get(name)
            atol = 0.0 if budget is None else budget.allowance(fmt, want, age)
            assert within(got[name], want, 0.0, atol), f"flux_observer[{age}] {name}: {got[name]} vs {want} ({atol=:g})"


def test_imu_fusion_matches_python_reference() -> None:
    """
    The generic suite skips imu_fusion (`reference=None`: its gyro/accel VECTOR parameters cannot bind from scalar
    rows), so this bespoke twin reconstructs the vectors and maps every port onto the reference instance over the
    full catalogue sequence. Budgets: the q lanes are an isometric recurrence (linear growth at the per-step
    rounding count, unit-norm floor); the b lanes are bounded absolutely by the catalogue clamp, whose rails
    resynchronize them bit-exactly (floor = the clamp); the out lanes inherit the carried q error at the ~1 g
    operand scale. The valid and gyro_clip bool lanes must match bit-for-bit.
    """
    spec = next(spec for spec in SPECS if spec.name == "imu_fusion")
    fmt = spec.formats[0]
    model = holoso.synthesize(spec.make_kernel(), spec.options(fmt), name=spec.name).numerical_model.elaborate()
    reference = spec.make_kernel()
    instance = reference.__self__  # type: ignore[attr-defined]
    budgets = {
        **{f"state_attitude_{i}": OutputTolerance(ulps=64, growth_ulps=32, floor=1.0) for i in range(4)},
        **{f"state_bias_{i}": OutputTolerance(ulps=16, growth_ulps=4, floor=instance.bias_limit) for i in range(3)},
        **{f"out_0_{i}": OutputTolerance(ulps=64, growth_ulps=16, floor=10.0) for i in range(3)},
    }
    for age, row in enumerate(spec.reference_vectors()):
        quantized = {name: _quantize(value, fmt) for name, value in row.items()}
        got = {
            port.name: value
            for port, value in zip(model.outputs, model.run(*[quantized[port.name] for port in model.inputs]))
        }
        result = reference(
            np.array([quantized[f"gyro_{k}"] for k in range(3)]),
            np.array([quantized[f"accel_{k}"] for k in range(3)]),
            np.array([[quantized[f"gyro_cal_{i}_{j}"] for j in range(3)] for i in range(3)]),
            np.array([[quantized[f"accel_cal_{i}_{j}"] for j in range(3)] for i in range(3)]),
            quantized["temperature"],
            quantized["dt"],
        )
        assert isinstance(result, tuple)
        out, valid = result
        assert bool(got["out_1"]) == bool(valid) and bool(got["state_gyro_clip"]) == bool(instance.gyro_clip), age
        if age == 0:
            # The catalogue opens with an accepted coarse-alignment row -- the property the frozen sequence exists
            # to cover. Acceptance takes the first-sample arm (the bias integrator must not run), and the attitude
            # must snap to the measured gravity direction: with these rates one propagation step from identity can
            # reach at most |w|*dt < 0.6 rad, so a larger angle proves the alignment arm fired.
            assert bool(valid), "the opening catalogue row must be accepted"
            assert all(b == 0.0 for b in instance.bias)
            assert 2.0 * np.arccos(min(1.0, abs(float(instance.attitude[0])))) > 0.7
        expected = {
            **{f"out_0_{k}": float(out[k]) for k in range(3)},
            **{f"state_bias_{k}": float(instance.bias[k]) for k in range(3)},
            **{f"state_attitude_{k}": float(instance.attitude[k]) for k in range(4)},
        }
        for name, want in expected.items():
            atol = budgets[name].allowance(fmt, want, age)
            assert within(
                float(got[name]), want, 0.0, atol
            ), f"imu_fusion[{age}] {name}: {float(got[name])} vs {want} ({atol=:g})"


class _StateLeafAheadOfComputed:
    def __init__(self) -> None:
        self.acc = 0.0

    def __call__(self, x: float) -> tuple[float, float]:
        self.acc = self.acc + x
        return self.acc, x * 4.0


def test_a_folded_state_leaf_ahead_of_a_computed_leaf_maps_by_name() -> None:
    """
    The FIRST returned leaf is public state (folded into `state_acc`; only the second leaf keeps an `out_` port),
    so a mapper pairing `out_` ports with the leading return leaves positionally would compare the computed port
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


def test_the_popcount_witnesses_match_an_independent_reduction() -> None:
    """
    Both suites above run the rewritten kernel against itself, so a rewrite that is wrong in both directions passes
    them. These oracles are spelled independently of the kernels: a bit-string count for the parity and a plain sum
    for the vote, driven over the values the shared vectors never reach.
    """
    for odd in (False, True):
        for char in [*range(-300, 301), 0x55, 0xC3, 0x7F, 0x01, 256, 511, -(2**33), 2**33 - 1]:
            expected = odd ^ (bin(char & 0xFF).count("1") % 2 == 1)
            assert UartTx(parity=odd)._parity_bit(char) == expected, (char, odd)
    for bits in itertools.product((False, True), repeat=5):
        assert MajorityVoter._majority(*bits) == (sum(bits) >= 3), bits
