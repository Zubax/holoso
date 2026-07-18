"""
End-to-end out-of-context synthesis of the example matrix: every ``SynthTarget`` is synthesized in-process and its
achieved f_max is asserted to meet the target frequency on its tool. This is the timing-closure regression guard for
RTL-generation changes -- the functional guarantee (RTL == model) lives in the cosimulation suite, and the
deterministic scheduling guard in the golden corpus (``test_golden``); this layer owns the physical timing only.

The whole module is ``synth``-marked (it needs an FPGA toolchain): ``nox -s synth_examples`` runs it and the normal
suite skips it. A target whose flow's tool is absent skips individually, so a Yosys-only CI still exercises every Yosys
row and the on-prem Diamond/Vivado rows skip cleanly, while ``test_some_target_flow_is_available`` fails loudly if no
tool is present at all, so a fully-missing toolchain cannot pass green.
"""

import shutil

import pytest

import holoso
from synth._synth import BUILD_ROOT, build_compiler_ooc_design
from synth.flows import make_flow

from . import _impact
from ._synth_targets import TARGETS, SynthTarget

pytestmark = pytest.mark.synth


def test_some_target_flow_is_available() -> None:
    # A safety net for the safety net: under ``-m synth`` an absent tool skips its targets, so with NO tool installed
    # every parametrized case would skip and the session would pass while verifying nothing. Fail loudly instead, so a
    # misconfigured CI (a lost toolchain) is caught rather than reported green.
    flows = {target.flow for target in TARGETS}
    assert any(
        make_flow(flow, 100.0).available() for flow in flows
    ), "no synthesis tool available; the matrix would pass while verifying nothing"


# Heaviest-first so xdist starts the long wide-datapath rows immediately instead of scheduling them last and tailing.
_BY_COST = sorted(TARGETS, key=lambda t: t.ops.float_format.wman, reverse=True)
_TARGET_PARAMS = [pytest.param(target, id=target.label) for target in _BY_COST]


@pytest.mark.parametrize("target", _TARGET_PARAMS)
def test_target_closes_timing(target: SynthTarget) -> None:
    flow = make_flow(target.flow, target.target_frequency_MHz)
    if not flow.available():
        pytest.skip(f"{target.flow.value} tool not available")

    # The label keys the row (example, format, stage knobs, flow, and frequency target all name it), so a matching
    # Verilog digest under a recorded pass means the identical netlist met the identical bar.
    row = f"synth:{target.label}"
    digest = _impact.verilog_digest(target.kernel(), target.ops, target.name) if _impact.enabled() else ""
    if digest and (head := _impact.cached_pass(row, digest)):
        pytest.skip(f"impact-cache: Verilog unchanged since {head}")

    result = holoso.synthesize(target.kernel(), target.ops, name=target.name)
    directory = BUILD_ROOT / "examples" / target.label
    shutil.rmtree(directory, ignore_errors=True)
    report = flow.prepare(build_compiler_ooc_design(result)).synthesize(directory)

    assert report.fmax_MHz >= target.target_frequency_MHz, (
        f"{target.label}: f_max {report.fmax_MHz:.2f} MHz < target {target.target_frequency_MHz:.2f} MHz "
        f"(slack {report.slack_ns:+.3f} ns); logs in {report.artifact_dir}"
    )
    if digest:
        _impact.record_pass(row, digest)
