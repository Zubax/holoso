"""
End-to-end cosimulation of every compilable example kernel: each is driven with hand-built sensible vectors, a frozen
random sweep, and format edge cases, then checked bit-for-bit against its embedded model under a lean (no optional
stages) and a deeply pipelined operator configuration at the wide e8m36 datapath.

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

Still-excluded examples are frontend feature gaps (not verification scope), confirmed by an in-memory compile probe:
  - iir1_hpf: ``UnsupportedConstruct: call to 'lpf'`` -- a foreign call on an instance-attribute sub-filter (the
    frontend inlines only global functions, not a nested object); ``float()`` itself is now supported.
  - finite_set_current_controller: ``UnsupportedConstruct`` -- nested/foreign attribute access.
"""

import pytest

from holoso import FloatFormat
from ._cosim import run_cosim
from ._examples import SPECS, ExampleSpec
from ._modelref import PIPELINE_OP_CASES, OperatorCase
from .hdl.hdl_float_oracle import SIMULATORS

pytestmark = pytest.mark.cosim

# Each example is exercised at the lean default schedule and a deeply pipelined one, to explore the schedule and
# handshake at two latency points; both are bit-exact against the same model.
_OP_CONFIGS = PIPELINE_OP_CASES

# One case per (spec, datapath format): every spec runs at e8m36, and a spec that lists a second format (octave_index
# adds the shallow e6m18) also runs there -- exercising the merge-threaded loop at both pipeline depths.
_SPEC_FORMATS = [
    pytest.param(spec, fmt, id=f"{spec.name}-e{fmt.wexp}m{fmt.wman}") for spec in SPECS for fmt in spec.formats
]


@pytest.mark.parametrize("sim", SIMULATORS)
@pytest.mark.parametrize("config", _OP_CONFIGS, ids=lambda c: c.label)
@pytest.mark.parametrize("spec,fmt", _SPEC_FORMATS)
def test_example_cosim(spec: ExampleSpec, fmt: FloatFormat, config: OperatorCase, sim: str) -> None:
    name = f"{spec.name}_{config.label}_e{fmt.wexp}m{fmt.wman}"
    run_cosim(sim, spec.make_kernel(), fmt, name, ops=config.make_ops(fmt), vectors=spec.vectors(fmt))
