"""
Fast guards on the synthesis target matrix, runnable without any synthesis tool: every catalogued example is
represented in the matrix, and target labels are unique per flow.
"""

from ._examples import SPECS
from ._synth_targets import TARGETS

_CATALOGUE = {spec.name for spec in SPECS}


def test_every_example_has_a_synth_target() -> None:
    covered = {target.example for target in TARGETS if target.example is not None}
    missing = _CATALOGUE - covered
    assert not missing, f"catalogued examples absent from the synth matrix: {sorted(missing)}"


def test_target_labels_unique() -> None:
    labels = [target.label for target in TARGETS]
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    assert not duplicates, f"duplicate (name, flow) targets -- give a more descriptive name: {duplicates}"
