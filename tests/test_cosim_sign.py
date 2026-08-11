"""
Cosim of the two inlined-``holoso_fsgnop`` sites the bundled examples miss: a folded sign on a state writeback and on
output taps. The ``remainder``/``pid`` specs already cover the inline-firing and phi-arm-install sites.
"""

import dataclasses

import pytest

import holoso
from holoso import FloatFormat
from ._cosim import run_cosim
from ._modelref import default_options
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
@pytest.mark.parametrize("wint_min", (None, 33))
@pytest.mark.parametrize("sim", SIMULATORS)
def test_sign_conditioning_cosim(sim: str, wint_min: int | None) -> None:
    options = default_options(FloatFormat(wexp=6, wman=18))
    if wint_min is not None:
        options = dataclasses.replace(options, wint_min=wint_min)
    # The module name carries the integer width because the two parametrizations build distinct machines.
    result = holoso.synthesize(SignHold().__call__, options, name=f"sign_hold_r{options.ifmt.width}")
    run_cosim(sim, result)
