"""
Coverage for the multi-distinct-constant install path. A register that receives two or more DISTINCT constant phi-arm
installs selects among them through its per-register write opcode, one ``case`` arm per constant ``const_N`` net. No
bundled example exercises this -- each installs at most one distinct constant per register -- so this kernel pins it: a
comparison-gated branch (a division per arm blocks if-conversion, so the merge stays a real phi) assigns two distinct
constants to one merged variable.
"""

import re

import pytest

import holoso
from holoso import FloatFormat

from ._cosim import run_cosim
from ._modelref import default_options
from .hdl.hdl_float_oracle import SIMULATORS

_FMT = FloatFormat(wexp=8, wman=36)


def _multi_const_install(x: float) -> float:
    if x > 0.0:
        a = 1.0 / x
        w = 3.0
    else:
        a = 1.0 / (x + 1.0)
        w = 5.0
    return a + w


def test_multi_distinct_const_install_selects_among_constants() -> None:
    """A register must select among >=2 distinct constants; otherwise the cosim below would not exercise that mux."""
    verilog = holoso.synthesize(_multi_const_install, default_options(_FMT), name="multi_const").verilog_output.verilog
    per_reg: dict[str, set[str]] = {}
    for reg, const in re.findall(r"regs\[(\d+)\] <= const_(\d+);", verilog):
        per_reg.setdefault(reg, set()).add(const)
    assert any(len(consts) >= 2 for consts in per_reg.values()), "no register selects among >=2 distinct constants"


@pytest.mark.cosim
@pytest.mark.parametrize("sim", SIMULATORS)
def test_multi_distinct_const_install_cosim(sim: str) -> None:
    """RTL == model for a register installing two distinct constants: one write-opcode case arm per const."""
    run_cosim(sim, holoso.synthesize(_multi_const_install, default_options(_FMT), name="multi_const"))
