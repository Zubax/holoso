"""
End-to-end cosimulation of every catalogued scalar-drivable example: each is driven with hand-built sensible vectors, a
frozen random sweep, and format edge cases, then checked bit-for-bit against its embedded model under a lean (no
optional stages) and a deeply pipelined operator configuration at each spec's datapath formats -- e8m36 for most,
e6m18 for the kernels that build no float operator at all (where the format only sizes the integer word, so this is
the width their own scripts generate), and octave_index at both.

This proves ``RTL == embedded numerical model``; it does NOT prove ``model == Python semantics`` (both descend from the
same lowering). ``test_example_reference.py`` covers that second half, driving the same example specs against a fresh
plain-Python instance of each kernel.

``iir1_lpf`` exercises real control flow: a boolean first-sample state and a data-dependent if/else, synthesized
through the CFG/branch backend (the first sample takes ``y = x``, every later sample the IIR update). ``pid`` and
``schmitt_trigger`` exercise float comparisons (``holoso_fcmp``) driving branches: a PID with three-way saturation +
anti-windup, a derivative-on-error channel, and a boolean ``_started`` state that suppresses the first-update
derivative spike; and two-threshold hysteresis (a state held untouched across the deadband).

``signal_window`` exercises the Phase 1 expression forms: boolean connectives, a chained comparison, nested
conditional (ternary) expressions (branch + phi), and both float<->bool casts, including a cross-domain
comparison -> bool -> float-cast -> float-multiply chain. ``remainder`` is a pure function computing the IEEE 754
remainder by data-dependent iterative reduction (two magnitude-ratio-bounded back-edge loops, no division).

Non-catalogue examples are frontend feature gaps or non-scalar interfaces, not verification scope:
  - iir1_hpf: ``UnsupportedConstruct: cannot call 'self.lpf': it is a separate component instance (IIR1LPF);
    hierarchical state is not supported yet``.
  - finite_set_current_controller: ``UnsupportedConstruct: the annotation of parameter 'kin' is not supported yet``
    (a dataclass-typed parameter).
  - imu_frame_transform: ndarray-typed inputs, driven by the oracle and metrics layers instead.
"""

import pytest

import holoso
from holoso import FloatFormat
from ._cosim import run_cosim
from ._examples import SPECS, ExampleSpec
from ._modelref import PIPELINE_OPTIONS_CASES, OptionsCase
from .hdl.hdl_float_oracle import SIMULATORS

pytestmark = pytest.mark.cosim

# Each example is exercised at the lean default schedule and a deeply pipelined one, to explore the schedule and
# handshake at two latency points; both are bit-exact against the same model.
_OP_CONFIGS = PIPELINE_OPTIONS_CASES

# One case per (spec, datapath format), at each spec's declared formats.
_SPEC_FORMATS = [
    pytest.param(spec, fmt, id=f"{spec.name}-e{fmt.wexp}m{fmt.wman}") for spec in SPECS for fmt in spec.formats
]


@pytest.mark.parametrize("sim", SIMULATORS)
@pytest.mark.parametrize("config", _OP_CONFIGS, ids=lambda c: c.label)
@pytest.mark.parametrize("spec,fmt", _SPEC_FORMATS)
def test_example_cosim(spec: ExampleSpec, fmt: FloatFormat, config: OptionsCase, sim: str) -> None:
    name = f"{spec.name}_{config.label}_e{fmt.wexp}m{fmt.wman}"
    options = spec.configured(config.make_options(fmt))
    result = holoso.synthesize(spec.make_kernel(), options, name=name)
    run_cosim(sim, result, vectors=spec.raw_vectors())
