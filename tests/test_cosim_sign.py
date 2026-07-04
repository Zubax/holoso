"""
Cosim of the two inlined-``holoso_fsgnop`` sites the bundled examples miss: a folded sign on a state writeback and on
output taps. The ``remainder``/``pid`` specs already cover the inline-firing and phi-arm-install sites.
"""

import pytest

from holoso import FloatFormat
from ._cosim import run_cosim
from .hdl.hdl_float_oracle import SIMULATORS


class SignHold:
    """Negated-input hold: writes ``-a`` to persistent state and outputs the negated previous state."""

    def __init__(self) -> None:
        self.acc = 0.0

    def __call__(self, a: float) -> float:
        prev = self.acc
        self.acc = -a
        return -prev


@pytest.mark.cosim
@pytest.mark.parametrize("sim", SIMULATORS)
def test_sign_conditioning_cosim(sim: str) -> None:
    fmt = FloatFormat(wexp=6, wman=18)
    run_cosim(sim, SignHold().__call__, fmt, "sign_hold")
