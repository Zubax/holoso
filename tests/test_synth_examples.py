"""
End-to-end out-of-context synthesis of the example matrix: every ``SynthTarget`` is synthesized in-process and its
achieved f_max is asserted to meet the target frequency on its tool. This is the timing-closure regression guard for
RTL-generation changes -- the functional guarantee (RTL == model) lives in the cosimulation suite, and the
deterministic scheduling guard in ``test_latency_freeze``; this layer owns the physical timing only.

The ``synth`` marker gates tool-dependence, not slowness: every test that needs an FPGA toolchain carries it (the
parametrized matrix and the fast ``test_some_target_flow_is_available`` guard), so ``nox -s synth_examples`` runs them
while the normal suite never invokes a tool. A target whose flow's tool is absent skips, so a Yosys-only CI still
exercises every Yosys row and the on-prem Diamond/Vivado rows skip cleanly; the availability guard fails if no tool is
present at all, so a fully-missing toolchain cannot pass green. The one tool-free guard
(``test_ambient_target_env_does_not_leak_into_lean_rows``) is unmarked, so it runs in the normal suite. Any further
tool-dependent test added here must therefore carry its own ``@pytest.mark.synth``.
"""

import os
import shutil

import pytest

import holoso
from synth._synth import BUILD_ROOT
from synth.flows import make_flow

from ._synth_targets import TARGET_ENV_KEYS, TARGETS, SynthTarget


def _apply_env(monkeypatch: pytest.MonkeyPatch, target: SynthTarget) -> None:
    """
    Set exactly this target's env, normalized: every key any target uses is cleared first so an ambient value (e.g. a
    shell ``HOLOSO_DIAMOND_HARD=1``) cannot leak into a lean row and run the hard strategy, masking a closure
    regression. The values are read inside ``flow.prepare()`` (``HOLOSO_DIAMOND_HARD`` in diamond ``_strategy``).
    """
    for key in TARGET_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    for key, value in target.env.items():
        monkeypatch.setenv(key, value)


@pytest.mark.synth
def test_some_target_flow_is_available() -> None:
    # A safety net for the safety net: under ``-m synth`` an absent tool skips its targets, so with NO tool installed
    # every parametrized case would skip and the session would pass while verifying nothing. Fail loudly instead, so a
    # misconfigured CI (a lost toolchain) is caught rather than reported green.
    flows = {target.flow for target in TARGETS}
    assert any(
        make_flow(flow, 100.0).available() for flow in flows
    ), "no synthesis tool available; the matrix would pass while verifying nothing"


@pytest.mark.synth
@pytest.mark.parametrize("target", TARGETS, ids=lambda t: t.label)
def test_target_closes_timing(target: SynthTarget, monkeypatch: pytest.MonkeyPatch) -> None:
    flow = make_flow(target.flow, target.target_frequency_MHz)
    if not flow.available():
        pytest.skip(f"{target.flow.value} tool not available")
    _apply_env(monkeypatch, target)

    result = holoso.synthesize(target.kernel(), target.ops, name=target.name)
    directory = BUILD_ROOT / "examples" / target.label
    shutil.rmtree(directory, ignore_errors=True)
    report = flow.prepare(result).synthesize(directory)

    assert report.fmax_MHz >= target.target_frequency_MHz, (
        f"{target.label}: f_max {report.fmax_MHz:.2f} MHz < target {target.target_frequency_MHz:.2f} MHz "
        f"(slack {report.slack_ns:+.3f} ns); logs in {report.artifact_dir}"
    )


def test_ambient_target_env_does_not_leak_into_lean_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    # Regression: per-target env is normalized, not merely overlaid. With HOLOSO_DIAMOND_HARD=1 in the ambient shell, a
    # lean target (empty env) must run with it CLEARED -- otherwise the hard Diamond strategy would silently rescue a
    # lean closure regression. A hard target still sets it. Exercises the real _apply_env, no synthesis tool needed.
    assert "HOLOSO_DIAMOND_HARD" in TARGET_ENV_KEYS
    monkeypatch.setenv("HOLOSO_DIAMOND_HARD", "1")  # ambient shell value
    _apply_env(monkeypatch, next(t for t in TARGETS if not t.env))
    assert "HOLOSO_DIAMOND_HARD" not in os.environ
    _apply_env(monkeypatch, next(t for t in TARGETS if t.env.get("HOLOSO_DIAMOND_HARD")))
    assert os.environ["HOLOSO_DIAMOND_HARD"] == "1"
