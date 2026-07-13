"""
Behavioral validation of every compilable example against its ORIGINAL Python execution.

The cosimulation suite (``test_cosim_examples.py``) checks the emitted RTL against the kernel's EMBEDDED numerical
model -- but both descend from the same front-end lowering, so a front-end miscompile poisons the RTL and the model
identically and the bit-for-bit check still passes. That suite proves ``RTL == compiler-model``; it cannot prove
``compiler-model == Python semantics``. This module closes that gap: it drives each example's numerical model AND a
fresh plain-Python instance of the same kernel over ``reference_vectors()`` (the manual sequence then the random draw)
and asserts they agree. Boolean lanes and exact (integer/Sterbenz) float lanes must match bit-for-bit; a kernel whose
float outputs accumulate rounding (``ReferenceComparison.APPROXIMATE``) is compared within a format-derived tolerance.
Inputs are quantized into the format first, so the model and the reference see the same operands and only the
per-operation rounding differs.
The per-input format-edge sweep is excluded here -- the model legitimately diverges from float64 at the format extremes
(an operation overflowing to the format's infinity stays finite in float64), which the cosim suite covers instead.

The example specs are shared with the cosimulation suite via ``_examples``: the cosim suite drives the full
``raw_vectors()`` (manual + random + edges), this suite the ``reference_vectors()`` subset, over one source of truth.
"""

import numpy as np
import pytest

import holoso
from holoso import BoolType, FloatFormat
from ._examples import SPECS, ExampleSpec, ReferenceComparison, parity_marks
from ._modelref import default_ops, default_tolerance, flatten_value, within

# A kernel the generic scalar-lane harness cannot drive (``ReferenceComparison.EXCLUDED``) is skipped here: it has
# public VECTOR state this harness would read by a non-existent per-element attribute (``ekf1_stateful``); its
# aggregate-state read-back is validated against the Python reference in ``test_verify.py`` instead. Every other example
# returns its outputs (optionally alongside scalar public state), which this harness compares directly.
#
# A public state attribute drives an output port named ``state_<attr>``; a return value drives ``out_<n>``. The model
# emits the surviving return leaves as the leading ``out_`` ports (a returned public attribute is folded into its
# ``state_`` port), so the two are matched by walking the outputs and consuming return leaves in order. This positional
# walk assumes the surviving ``out_`` leaves form a leading prefix of the flattened return -- true for every current
# kernel; a future kernel that returned a folded public attribute AHEAD of a computed value would need keyed mapping.
_STATE_PREFIX = "state_"

_CASES = [
    pytest.param(
        spec,
        spec.formats[0],
        id=f"{spec.name}-e{spec.formats[0].wexp}m{spec.formats[0].wman}",
        marks=parity_marks(spec.name),
    )
    for spec in SPECS
    if spec.reference is not ReferenceComparison.EXCLUDED
]


def _quantize(value: float | bool, fmt: FloatFormat) -> float | bool:
    """A float rounded into ``fmt`` (so the model and float64 reference get an identical operand); a bool unchanged."""
    return value if isinstance(value, bool) else fmt.decode(fmt.encode(value))


def _model_for(spec: ExampleSpec, fmt: FloatFormat) -> holoso.NumericalSimulator:
    # Built through the public facade, exactly as a user would; ``_lir.ops`` (read once below for the tolerance op
    # count) is the only internal datum the result does not expose publicly.
    return holoso.synthesize(spec.make_kernel(), default_ops(fmt), name=spec.name).numerical_model.elaborate()


@pytest.mark.parametrize("spec,fmt", _CASES)
def test_example_matches_python_reference(spec: ExampleSpec, fmt: FloatFormat) -> None:
    model = _model_for(spec, fmt)
    reference = spec.make_kernel()  # a fresh plain-Python instance, advanced in lockstep with the model
    instance = getattr(reference, "__self__", None)  # the bound receiver, for reading scalar public-state live-outs
    op_count = max(len(model._lir.ops), 1)
    for row in spec.reference_vectors():  # manual then random; the model and float64 reference agree on this subset
        quantized = {name: _quantize(value, fmt) for name, value in row.items()}
        got = model.run(*[quantized[port.name] for port in model.inputs])
        leaves = [leaf for _, leaf in flatten_value(reference(*[quantized[name] for name in spec.inputs]))]
        expected: list[float | bool] = []
        return_index = 0
        for port in model.outputs:
            if port.name.startswith(_STATE_PREFIX):
                value = getattr(instance, port.name[len(_STATE_PREFIX) :])
                assert not isinstance(value, (list, tuple, np.ndarray)), f"{spec.name}: unexpected vector public state"
                expected.append(value)
            else:
                expected.append(leaves[return_index])
                return_index += 1
        floats = [abs(float(v)) for v in (*quantized.values(), *expected) if not isinstance(v, bool)]
        # Exact (0, 0) unless the kernel accumulates rounding: a discrete/Sterbenz float output must match bit-for-bit,
        # so a stuck or off-by-one value cannot hide under a loose relative tolerance (acute in the coarse byte format).
        approximate = spec.reference is ReferenceComparison.APPROXIMATE
        rtol, atol = default_tolerance(fmt, op_count, max([1.0, *floats])) if approximate else (0.0, 0.0)
        for port, got_value, want in zip(model.outputs, got, expected):
            if isinstance(port.scalar_type, BoolType):
                assert bool(got_value) == bool(
                    want
                ), f"{spec.name} {row} {port.name}: {bool(got_value)} != {bool(want)}"
            else:
                assert within(
                    float(got_value), float(want), rtol, atol
                ), f"{spec.name} {row} {port.name}: {float(got_value)} vs {float(want)}"
