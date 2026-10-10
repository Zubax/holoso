"""
End-to-end out-of-context synthesis of the example matrix: every `SynthTarget` is synthesized in-process and its
achieved f_max is asserted to meet the target frequency on its tool. This is the timing-closure regression guard for
RTL-generation changes -- the functional guarantee (RTL == model) lives in the cosimulation suite, and the
deterministic scheduling guard in `test_latency_freeze`; this layer owns the physical timing only.

The whole module is `synth`-marked (it needs an FPGA toolchain): `nox -s synth_examples` runs it and the normal
suite skips it. A target whose flow's tool is absent skips individually, so a Yosys-only CI still exercises every Yosys
row and the on-prem Diamond/Vivado rows skip cleanly, while `test_some_target_flow_is_available` fails loudly if no
tool is present at all, so a fully-missing toolchain cannot pass green.
"""

import re
import shutil

import pytest

import holoso
from synth._synth import BUILD_ROOT, build_compiler_ooc_design
from synth.flows import FlowId, make_flow

from ._synth_targets import TARGETS, SynthTarget

pytestmark = pytest.mark.synth


def test_some_target_flow_is_available() -> None:
    # A safety net for the safety net: under `-m synth` an absent tool skips its targets, so with NO tool installed
    # every parametrized case would skip and the session would pass while verifying nothing. Fail loudly instead, so a
    # misconfigured CI (a lost toolchain) is caught rather than reported green.
    flows = {target.flow for target in TARGETS}
    assert any(
        make_flow(flow, 100.0).available() for flow in flows
    ), "no synthesis tool available; the matrix would pass while verifying nothing"


# Heaviest-first so xdist starts the long wide-datapath rows immediately instead of scheduling them last and tailing.
_BY_COST = sorted(TARGETS, key=lambda t: t.ops.ffmt.wman, reverse=True)


@pytest.mark.parametrize("target", _BY_COST, ids=lambda t: t.label)
def test_target_closes_timing(target: SynthTarget) -> None:
    flow = make_flow(target.flow, target.target_frequency_MHz, target.device_class)
    if not flow.available():
        pytest.skip(f"{target.flow.value} tool not available")

    result = holoso.synthesize(target.kernel(), target.ops, name=target.name)
    directory = BUILD_ROOT / "examples" / target.label
    shutil.rmtree(directory, ignore_errors=True)
    report = flow.prepare(build_compiler_ooc_design(result)).synthesize(directory)

    assert report.fmax_MHz >= target.target_frequency_MHz, (
        f"{target.label}: f_max {report.fmax_MHz:.2f} MHz < target {target.target_frequency_MHz:.2f} MHz "
        f"(slack {report.slack_ns:+.3f} ns); logs in {report.artifact_dir}"
    )
    if target.flow == FlowId.VIVADO_ARTIX7:
        # The flow's ROM attribute hook is what keeps the microcode ROM out of LUT logic (5-10 percent of a large
        # machine, DESIGN.md's fabric-area exploration).
        block_rams = report.resources["RAMB36/FIFO*"].used + report.resources["RAMB18"].used
        assert block_rams >= 1, f"{target.label}: the microcode ROM is not in block RAM"
    if target.flow == FlowId.DIAMOND_ECP5:
        # LSE maps a lookup table into block RAM only with resource sharing enabled; left in LUT logic the tables
        # still close at a narrow format, so timing alone would not notice the flow losing them (DESIGN.md's
        # fabric-area exploration). A total cannot tell a table from the microcode ROM, hence the owners. This sees a
        # table that is not mapped at all, not one mapped in part.
        tables = len(re.findall(r"^holoso_f(?:exp2|log2) #\(", result.verilog_output.verilog, re.MULTILINE))
        owners = report.block_ram_owners.items()
        owning = [path for path, n in owners if n and re.search(r"/u_f(?:exp2|log2)_\d+$", path)]
        assert len(owning) == tables, f"{target.label}: a lookup table is not in block RAM: {owning}"
