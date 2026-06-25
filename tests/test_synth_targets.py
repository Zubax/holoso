"""
Fast guards on the synthesis target matrix, runnable without any synthesis tool (so they live in the normal suite,
not the ``synth``-marked one): every catalogued example is represented, target labels are unique per flow, and every
target carries a single-format operator configuration.
"""

import holoso

from ._examples import SPECS, ekf1_stateful
from ._synth_targets import F_e6m18, TARGETS, UNSYNTHESIZED, op_config

_CATALOGUE = {spec.name for spec in SPECS}


def test_every_example_has_a_synth_target() -> None:
    assert (
        UNSYNTHESIZED <= _CATALOGUE
    ), f"stale exclusions (not catalogued examples): {sorted(UNSYNTHESIZED - _CATALOGUE)}"
    covered = {target.example for target in TARGETS if target.example is not None}
    missing = _CATALOGUE - covered - UNSYNTHESIZED
    assert not missing, f"catalogued examples absent from the synth matrix: {sorted(missing)}"


def test_target_labels_unique() -> None:
    labels = [target.label for target in TARGETS]
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    assert not duplicates, f"duplicate (name, flow) targets -- give a more descriptive name: {duplicates}"


def test_targets_carry_single_format_ops() -> None:
    # Each row's OpConfig is built at import time, so an out-of-range stage already fails collection; this additionally
    # confirms every target's operators share one float format (OpConfig.float_format raises otherwise).
    for target in TARGETS:
        _ = target.ops.float_format


def test_ekf1_stateful_targets_synthesize_the_bundled_example_kernel(monkeypatch) -> None:
    # Regression: the matrix must verify the kernel the bundled example ships -- ekf1_stateful's default Ekf1().update
    # -- not the cosim/reference SPEC factory (a divisor-safe reset that folds different constants into different RTL).
    # All ekf1_stateful rows must therefore share one synthesis kernel, distinct from the SPEC's, else the synth suite
    # would silently guard the wrong circuit.
    monkeypatch.setenv("HOLOSO_REGALLOC_EFFORT", "10")  # the structure, not the schedule, is what we compare
    ops = op_config(F_e6m18)
    kernels = {target.kernel for target in TARGETS if target.example == "ekf1_stateful"}
    assert len(kernels) == 1, "all ekf1_stateful synth rows must share one synthesis kernel factory"
    (kernel,) = kernels

    # The bundled default and the cosim SPEC factory (spec.make_kernel) synthesize to different RTL -- verified -- so
    # the rows must match the former. Two compiles suffice: equality to the bundled kernel implies inequality to cosim.
    got = holoso.synthesize(kernel(), ops, name="ekf1_stateful").verilog_output.verilog
    bundled = holoso.synthesize(ekf1_stateful.Ekf1().update, ops, name="ekf1_stateful").verilog_output.verilog
    assert got == bundled, "ekf1_stateful synth rows must synthesize the bundled Ekf1().update, not the cosim factory"
