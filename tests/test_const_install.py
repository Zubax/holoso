"""
Coverage for the multi-distinct-constant install path. A register that receives two or more DISTINCT constant phi-arm
installs is driven by a microcode const-pool selector (``uc_ccidx``) muxing among its constants, rather than the bare
``const_N`` net a single-constant register uses. No bundled example exercises this -- each installs at most one distinct
constant per register -- so this kernel pins it: a comparison-gated branch (a division per arm blocks if-conversion, so
the merge stays a real phi) assigns two distinct constants to one merged variable.
"""

import pytest

from holoso import FloatFormat
from holoso._backend.verilog import generate as generate_verilog
from holoso._frontend import lower
from holoso._hir import optimize
from holoso._lir import build
from holoso._mir import lower as lower_to_mir

from ._cosim import run_cosim
from ._modelref import default_ops
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


def test_multi_distinct_const_install_emits_selector() -> None:
    """The kernel must emit a const-pool selector field; otherwise the cosim below would not exercise the mux at all."""
    lir = build(lower_to_mir(optimize(lower(_multi_const_install)), default_ops(_FMT)), "multi_const", fetch_stages=3)
    assert "uc_ccidx_" in generate_verilog(lir).verilog


@pytest.mark.cosim
@pytest.mark.parametrize("sim", SIMULATORS)
def test_multi_distinct_const_install_cosim(sim: str) -> None:
    """RTL == model for a register installing two distinct constants: the ``uc_ccidx`` selector picks the right one."""
    run_cosim(sim, _multi_const_install, _FMT, "multi_const")
