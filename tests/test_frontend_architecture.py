"""
Architectural guards for the resolution-totality restructure (docs/campaign.md Stage 4, docs/decisions/
arch-ruling.md). These do not test behavior; they pin the SHAPE the restructure is supposed to change, so
that progress is measured rather than asserted and regressions cannot pass unnoticed.
"""

import ast
from pathlib import Path

import pytest

from ._importguard import transitive_holoso_imports

_EMITTER = "holoso._frontend._fir._emit"

# The frontend decision modules the emitter reaches TODAY. The restructure exists to empty this set: emission
# is to consume a closed typed plan surface, deciding nothing. It is therefore a RATCHET -- every entry is a
# debt, the set may only shrink, and the test fails in both directions so that a removal is recorded in the
# commit that earns it rather than drifting silently.
#
# Deliberately NOT a ban on reaching `holoso._errors`: `_hir._const` constructs `UnsupportedConstruct` for a
# NaN constant, so nothing that emits HIR can satisfy that, and the spike's SC2 failed on exactly this reading
# (docs/decisions/arch-ruling.md). The debt this measures is the FRONTEND DECISION LAYER.
_EMITTER_DECISION_DEBT = frozenset(
    {
        "holoso._frontend._ast_support",
        "holoso._frontend._fir._analysis_support",
        "holoso._frontend._fir._analyze",
        "holoso._frontend._fir._build",
        "holoso._frontend._fir._fact",
        "holoso._frontend._fir._fold",
        "holoso._frontend._fir._ir",
        "holoso._frontend._fir._opsem",
        "holoso._frontend._fir._resolve",
        "holoso._frontend._fir._signature",
        "holoso._frontend._fir._value",
        "holoso._frontend._lib",
        "holoso._frontend._lib._intrinsics",
        "holoso._frontend._lib._linalg",
        "holoso._frontend._lib._numpy",
        "holoso._frontend._lib._registry",
    }
)


def test_emitter_decision_layer_debt_only_shrinks() -> None:
    reached = {m for m in transitive_holoso_imports(_EMITTER) if m.startswith("holoso._frontend")}
    added = sorted(reached - _EMITTER_DECISION_DEBT)
    removed = sorted(_EMITTER_DECISION_DEBT - reached)
    assert not added, (
        f"emission reaches new frontend decision modules {added}; the restructure removes these dependencies, "
        "it does not add them"
    )
    assert not removed, (
        f"emission no longer reaches {removed} -- that is the point, so delete those entries from "
        "_EMITTER_DECISION_DEBT in this same commit and let the ratchet hold the ground"
    )


def test_the_ratchet_is_measured_against_real_modules() -> None:
    # `from pkg import Name` yields `pkg.Name`, which is not a module. Counting those inflates the debt and can
    # name a class where a module belongs, which would make the ratchet's own numbers untrustworthy.
    reached = transitive_holoso_imports(_EMITTER)
    unresolved = sorted(m for m in reached if not _module_source(m))
    assert not unresolved, f"closure contains non-modules: {unresolved}"


def test_the_guard_root_fails_loudly_when_it_does_not_resolve() -> None:
    # A typo in the root would otherwise yield an empty closure and a permanently green guard.
    with pytest.raises(ValueError):
        transitive_holoso_imports("holoso._frontend._fir._emit_that_does_not_exist")


def test_emission_rejection_sites_only_shrink() -> None:
    # M5 retires EmissionRejection: every refusal moves upstream, diagnostic-identical. The count is the debt,
    # and the corpus pins the messages, so this only has to stop the number from growing.
    source = ast.parse(Path(_module_source(_EMITTER)).read_text(encoding="utf-8"))
    raises = [
        node
        for node in ast.walk(source)
        if isinstance(node, ast.Raise)
        and isinstance(node.exc, ast.Call)
        and isinstance(node.exc.func, ast.Name)
        and node.exc.func.id == "EmissionRejection"
    ]
    assert len(raises) <= 42, f"emission gained refusal sites ({len(raises)} > 42); M5 moves them upstream"


def _module_source(module: str) -> str:
    import importlib.util

    spec = importlib.util.find_spec(module)
    origin = spec.origin if spec is not None else None
    return origin if origin is not None and origin.endswith(".py") else ""


def test_a_missing_call_plan_is_a_verifier_error_not_a_walk_time_crash() -> None:
    # Emission reaches its plan tables with a bare subscript, so an analyzer that failed to record one would
    # surface as a KeyError from deep inside the walk, naming neither the op nor the block. The verifier runs
    # before the walk and names both. Checked by removing a plan the analyzer did record.
    import numpy as np

    from holoso._frontend._fir._analyze import Analyzer, verify_plan_totality

    class Kernel:
        def __init__(self) -> None:
            self.s = 0.0

        def step(self, x: float) -> float:
            self.s = float(np.array([x, x]).shape[0])
            return self.s

    result = Analyzer(Kernel().step).fixpoint()
    verify_plan_totality(result)  # the real analysis is total

    assert result.call_plans, "the kernel must exercise at least one call plan for this test to mean anything"
    result.call_plans.pop(next(iter(result.call_plans)))
    with pytest.raises(AssertionError, match="has no call plan"):
        verify_plan_totality(result)
