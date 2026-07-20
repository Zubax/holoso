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

# The frontend decision modules `_emit.py` imports DIRECTLY. The restructure drives this set toward empty; an
# empty set is NECESSARY for "emission decides nothing" and nowhere near sufficient, since a decision can also
# arrive through the plan (see the blind spot below). It is a RATCHET -- every entry is a debt, the set may
# only shrink, and the test fails in both directions so a removal is spent in the commit that earns it.
#
# DIRECT imports, not the transitive closure, and the difference is the whole value of the meter. The emitter's
# closure is every module under `holoso/_frontend` that exists, and it is implied entirely by the single import
# of `_analyze`: measured, the emitter contributes NOTHING to its own closure beyond that one edge. So a closure
# ratchet cannot grow -- three decision-carrying imports (`_registry.resolve`, `_fold.admit_call`,
# `_opsem.static_binop`) added straight into the emitter leave it green -- and cannot shrink either until
# `_emit` stops importing `_analyze`, which is M7, which the ruling made optional. It would read 16 through all
# of M1-M6 by construction. The direct set is 8, moves one import at a time, and a new decision dependency
# lands in it immediately.
#
# Deliberately NOT a ban on reaching `holoso._errors`: `_hir._const` constructs `UnsupportedConstruct` for a
# NaN constant, so nothing that emits HIR can satisfy that, and the spike's SC2 failed on exactly this reading
# (docs/decisions/arch-ruling.md).
#
# KNOWN BLIND SPOT, recorded because a guard that implies more coverage than it has is worse than none: a
# decision can reach the emitter through the PLAN rather than through an import, and one does today --
# `CallPlan.intrinsic` hands over a live registry `Intrinsic`, and `_emit_intrinsic` branches on its result
# rule. No import-based guard can see that. It is the J6 class from the ruling, and closing it is M2/M3 work
# still outstanding (docs/campaign.md).
_EMITTER_DIRECT_DEBT = frozenset(
    {
        "holoso._frontend._ast_support",
        "holoso._frontend._fir._analyze",
        "holoso._frontend._fir._fact",
        "holoso._frontend._fir._ir",
        "holoso._frontend._fir._opsem",
        "holoso._frontend._fir._signature",
        "holoso._frontend._fir._value",
        "holoso._frontend._lib",
    }
)


def test_emitter_direct_decision_debt_only_shrinks() -> None:
    imported = {m for m in _direct_imports(_EMITTER) if m.startswith("holoso._frontend")}
    added = sorted(imported - _EMITTER_DIRECT_DEBT)
    removed = sorted(_EMITTER_DIRECT_DEBT - imported)
    assert not added, (
        f"emission directly imports new frontend decision modules {added}; the restructure removes these "
        "dependencies, it does not add them"
    )
    assert not removed, (
        f"emission no longer imports {removed} -- that is the point, so delete those entries from "
        "_EMITTER_DIRECT_DEBT in this same commit and let the ratchet hold the ground"
    )


def _direct_imports(module: str) -> set[str]:
    """The modules `module` itself names in an import statement, resolving relative imports against its package."""
    path = Path(_module_source(module))
    package = module.rpartition(".")[0]
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        match node:
            case ast.Import(names=names):
                found |= {alias.name for alias in names}
            case ast.ImportFrom(module=name, level=0):
                if name:
                    found.add(name)
            case ast.ImportFrom(module=name, level=level):
                anchor = package if level == 1 else package.rsplit(".", level - 1)[0]
                found.add(f"{anchor}.{name}" if name else anchor)
    return found


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
    #
    # CONSTRUCTIONS, not `raise` statements: hoisting the raises into a `_reject(...)` helper would drop a
    # raise-shaped count to zero with byte-identical diagnostics and nothing actually moved upstream, so the
    # guard would read as M5 complete while measuring a refactor.
    source = ast.parse(Path(_module_source(_EMITTER)).read_text(encoding="utf-8"))
    built = [
        node
        for node in ast.walk(source)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "EmissionRejection"
    ]
    # EXACT, not a ceiling: `<= 42` would let the count fall to 41 and grow back to 42 unnoticed, which is the
    # regrowth a ratchet exists to prevent. A reduction is progress and updates this number in its own commit.
    assert len(built) == 42, (
        f"emission constructs {len(built)} refusals, recorded 42: fewer means M5 progress, so update the number "
        "here in the same commit; more means a refusal was added where the restructure removes them"
    )


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
