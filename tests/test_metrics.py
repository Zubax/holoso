"""
Allocator repeatability probe. The per-example steering/area baseline rows that used to live here moved into
the golden corpus: ``tests/_golden_cases.py`` freezes the exact figures per case in its ABI manifests and
carries the former ceilings as legacy non-regression rows, both asserted by ``tests/test_golden.py``.

What remains is the same-process repeatability check the corpus cannot express: the allocator's annealing is
``seed=0``, so two builds of the same kernel in one interpreter must agree byte-for-byte -- the property that
makes any frozen baseline stable in the first place (the cross-process, cross-hash-seed side is certified by
``tools/refreeze_golden.py --check-determinism`` and sampled by ``tests/test_determinism.py``).
"""

import pytest

from ._golden_cases import CASES, DEFAULT_REGALLOC, build_artifacts


@pytest.fixture(autouse=True)
def _pinned_regalloc_knobs(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Pin every register-allocation tuning knob to its shipped default so the comparison is reproducible
    regardless of the developer's environment (``HOLOSO_REGALLOC_EFFORT`` speed-ups, write-cap/price
    experiments). The knobs are env-read-once at import, so the module attributes are patched.
    """
    import holoso._lir._regalloc as regalloc

    monkeypatch.setattr(regalloc, "_REFINE_MAXITER", DEFAULT_REGALLOC.refine_maxiter)
    monkeypatch.setattr(regalloc, "_REG_REUSE_WRITE_CAP", DEFAULT_REGALLOC.reg_reuse_write_cap)
    monkeypatch.setattr(regalloc, "_REG_PRICE", DEFAULT_REGALLOC.reg_price)


def test_build_is_deterministic() -> None:
    case = next(case for case in CASES if case.case_id == "ekf1_stateless-e8m36")
    first = build_artifacts(case)
    second = build_artifacts(case)
    assert first.metrics == second.metrics
    assert first.verilog == second.verilog
    assert first.abi_json == second.abi_json
    assert first.hir_dump == second.hir_dump
