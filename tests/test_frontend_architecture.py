"""
Architectural guards for the resolution-totality restructure (docs/campaign.md Stage 4, docs/decisions/
arch-ruling.md). These do not test behavior; they pin the SHAPE the restructure is supposed to change, so
that progress is measured rather than asserted and regressions cannot pass unnoticed.
"""

import ast
from pathlib import Path

import pytest

from ._importguard import direct_imports, transitive_holoso_imports

_EMITTER = "holoso._frontend._fir._emit"

# WHAT `_emit.py` IMPORTS FROM THE FRONTEND, BY SYMBOL. This is the debt the restructure pays down, and it is
# recorded per symbol rather than per module because a module-level meter cannot see the imports that matter:
# `_opsem`, `_lib` and `_analyze` are already listed, so `static_binop`, `resolve` and any analyzer symbol
# could be added to the emitter without moving a module-level set at all. Measured, not argued -- two of the
# three probes that motivated abandoning the closure meter also pass a module-level direct meter.
#
# It is a RATCHET: the test fails in both directions, so a symbol that leaves is spent in the commit that earns
# it rather than drifting, and a symbol that arrives is a regression.
#
# NO CLASSIFICATION IS ASSERTED HERE. The set mixes the decision surface the restructure removes (the analyzer
# handle, the library registry types, the fact vocabulary emission re-reasons over) with structural vocabulary
# it plausibly keeps (FIR op and place types it walks, `executable_rpo`, port-name helpers). Only the former is
# expected to reach empty; which symbol is which has not been established, so this file does not pretend to
# know. It counts, and the count moving is the evidence.
#
# BLIND SPOT, named because a guard implying more than it measures is worse than none: a decision can reach the
# emitter through the PLAN rather than through any import, and one does -- `CallPlan.intrinsic` carries a live
# registry `Intrinsic` whose `result_rule` and `integer_implementation` the emitter branches on. Nothing
# import-shaped can see that. It is the J6 class from docs/decisions/arch-ruling.md, and it is outstanding.
_EMITTER_FRONTEND_DEBT: dict[str, frozenset[str]] = {
    "holoso._frontend._ast_support": frozenset(
        {
            "indexed_names",
            "port_name",
            "state_port_name",
        }
    ),
    "holoso._frontend._fir._analyze": frozenset(
        {
            "Analyzer",
            "CallLowering",
            "CallPlan",
            "ResidualUnit",
            "verify_plan_totality",
        }
    ),
    "holoso._frontend._fir._fact": frozenset(
        {
            "AggregateFact",
            "AggregateLayout",
            "ArrayDType",
            "ArrayIndex",
            "ArrayLayout",
            "AtomicFact",
            "ContainerFlavor",
            "Fact",
            "Known",
            "LeafPath",
            "ListIndex",
            "ListLayout",
            "RecordField",
            "RecordLayout",
            "Reference",
            "Residual",
            "StructuralIndex",
            "StructuralLayout",
            "TupleIndex",
            "TupleLayout",
            "ValueLayout",
            "child_layouts",
            "child_slice",
            "leaf_count",
            "leaf_paths",
            "materialize_static",
            "normalize_static",
            "outer_arity",
        }
    ),
    "holoso._frontend._fir._ir": frozenset(
        {
            "BindingId",
            "BlockId",
            "Branch",
            "BuildList",
            "BuildTuple",
            "Jump",
            "LoadConst",
            "LoadPlace",
            "LoadRef",
            "Local",
            "LocatedRejection",
            "Op",
            "OriginStack",
            "Place",
            "PyAttr",
            "PyBin",
            "PyCall",
            "PyCompare",
            "PyLen",
            "PyNot",
            "PySelect",
            "PyStoreAttr",
            "PySubscript",
            "PyTruth",
            "PyUn",
            "ReturnPlace",
            "SelectMode",
            "StateLeaf",
            "StorePlace",
            "UnbindPlace",
            "UnitExit",
            "executable_rpo",
            "source_position",
        }
    ),
    "holoso._frontend._fir._opsem": frozenset(
        {
            "BinOp",
            "UnOp",
        }
    ),
    "holoso._frontend._fir._signature": frozenset(
        {
            "ArrayReturn",
            "ListReturn",
            "RecordReturn",
            "ReturnContract",
            "ScalarReturn",
            "TupleReturn",
            "VariadicTupleReturn",
            "VoidReturn",
        }
    ),
    "holoso._frontend._fir._value": frozenset(
        {
            "MetaInt",
            "NpBool",
            "NpFloat",
            "NpInt",
            "SemType",
            "StaticBool",
            "StaticFloat",
            "StaticSlice",
            "StaticValue",
            "admit",
            "as_python",
        }
    ),
    "holoso._frontend._lib": frozenset(
        {
            "IntegerImplementation",
            "Intrinsic",
            "IntrinsicResultRule",
        }
    ),
}


def test_emitter_frontend_debt_only_shrinks() -> None:
    imported = {name for name in direct_imports(_EMITTER) if name.startswith("holoso._frontend")}
    recorded = {owner for owner in _EMITTER_FRONTEND_DEBT} | {
        f"{owner}.{symbol}" for owner, symbols in _EMITTER_FRONTEND_DEBT.items() for symbol in symbols
    }
    added = sorted(imported - recorded)
    removed = sorted(recorded - imported)
    assert not added, (
        f"emission imports {added} from the frontend; the restructure removes these dependencies, it does not "
        "add them"
    )
    assert not removed, (
        f"emission no longer imports {removed} -- that is the point, so delete those entries from "
        "_EMITTER_FRONTEND_DEBT in this same commit and let the ratchet hold the ground"
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
    # and the corpus pins the messages, so this only has to stop the number from drifting either way.
    #
    # BOTH numbers, and exactly, because each alone is defeatable and a ceiling is not a ratchet:
    #   - counting `raise` statements alone: hoisting them into a `_reject(...)` helper drops it to zero with
    #     byte-identical diagnostics and nothing moved upstream;
    #   - counting constructions alone: a helper that constructs internally lets a NEW refusal be added while
    #     the construction count stays put -- measured, a helper plus one added refusal reads 42 constructions
    #     and 41 direct raises;
    #   - `<=` on either would permit 42 -> 41 -> 42 regrowth.
    # A genuine upstream move changes both together, and updates both here in the commit that earns it.
    source = ast.parse(Path(_module_source(_EMITTER)).read_text(encoding="utf-8"))
    constructed = sum(
        1
        for node in ast.walk(source)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "EmissionRejection"
    )
    raised = sum(
        1
        for node in ast.walk(source)
        if isinstance(node, ast.Raise)
        and isinstance(node.exc, ast.Call)
        and isinstance(node.exc.func, ast.Name)
        and node.exc.func.id == "EmissionRejection"
    )
    assert (constructed, raised) == (42, 42), (
        f"emission constructs {constructed} refusals and raises {raised} directly, recorded (42, 42): fewer of "
        "both is M5 progress and updates the numbers here in the same commit; anything else means a refusal was "
        "added, or routed through a helper where the count can no longer see it"
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


def test_lower_fir_actually_runs_the_plan_verifier(monkeypatch: pytest.MonkeyPatch) -> None:
    # The verifier cannot fail for any result today (see its docstring), so nothing else in the suite would
    # notice if the call were deleted as dead code -- and it exists precisely to be in place BEFORE M1 rewrites
    # recording. Pin the call site itself.
    import holoso._frontend._fir._emit as emit_module

    calls: list[object] = []
    monkeypatch.setattr(emit_module, "verify_plan_totality", calls.append)

    class Kernel:
        def __init__(self) -> None:
            self.s = 0.0

        def step(self, x: float) -> float:
            self.s = x + 1.0
            return self.s

    emit_module.lower_fir(Kernel().step)
    assert len(calls) == 1, "lower_fir must run the plan verifier before emission walks"
