"""
Structural sanity for the shared example catalogue. The ``ExampleSpec.__post_init__`` asserts already pin the
static fields (``nominal``/``manual``/``edge_overrides`` keys against ``inputs``) at import time; the one invariant
it cannot check without an rng is that ``draw_random`` actually produces exactly the declared inputs, which every
downstream suite then relies on. This module closes that gap cheaply, with no synthesis or simulation.
"""

import numpy as np
import pytest

from ._examples import SPECS, ExampleSpec


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.name)
def test_draw_random_yields_declared_inputs(spec: ExampleSpec) -> None:
    row = spec.draw_random(np.random.default_rng(0xC0FFEE))
    assert set(row) == set(spec.inputs), f"{spec.name}: draw_random keys {set(row)} != inputs {set(spec.inputs)}"
