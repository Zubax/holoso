"""
Architectural guards for the resolution-totality restructure (docs/campaign.md Stage 4, docs/decisions/
arch-ruling.md). These do not test behavior; they pin the SHAPE the restructure is supposed to change, so
that progress is measured rather than asserted and regressions cannot pass unnoticed.
"""

import ast
from pathlib import Path

import pytest

from ._importguard import direct_imports

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
# TWO LIMITS, measured and stated rather than implied away. This counts IMPORT EDGES, not dependencies: an
# import that is dead weight counts as debt, so deleting one is ordinary lint that the ratchet reads as
# progress -- two such (`indexed_names`, `StaticSlice`) were found and removed when this was written. And it
# is scoped to `_emit.py` alone, while the refusal counter below is package-scoped: if emission code moves to
# a sibling module, its frontend imports leave this file and the ratchet will invite deleting entries for debt
# that merely moved. Both fail loudly rather than silently; the hazard is in following the message blindly.
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
        f"emission no longer imports {removed} -- if the dependency is GONE that is the point, so delete those "
        "entries here in the same commit; if it merely moved to a sibling module, or the import was dead "
        "weight, the debt did not shrink and this ledger's scope is what needs revisiting"
    )


def test_emission_reaches_the_frontend_only_by_named_symbols() -> None:
    # The ledger above records what `from X import Y` names. A plain `import holoso._frontend...` records only
    # the module and lets every symbol behind it -- `resolve`, `static_binop`, any analyzer internal -- arrive
    # unseen, which is the module-level blind spot the symbol ledger exists to close.
    #
    # This is a BAN on the spelling, not a reconciliation of the ledger, because the previous attempt to catch
    # it by reconciliation was dead code: it ran after an assertion that had already guaranteed its condition
    # false, and a bare import added ALONGSIDE the symbol imports left the owner's symbols present anyway. The
    # package is internally all-relative, so the ban costs nothing.
    source = ast.parse(Path(_module_source(_EMITTER)).read_text(encoding="utf-8"))
    bare = sorted(
        alias.name
        for node in ast.walk(source)
        if isinstance(node, ast.Import)
        for alias in node.names
        if alias.name.startswith("holoso._frontend")
    )
    assert not bare, (
        f"emission imports {bare} as whole modules, which hides every symbol reached through them from the "
        "debt ledger; import the symbols by name instead"
    )


def test_the_ledger_still_measures_where_emission_lives() -> None:
    # The ledger reads ONE file, so a thin `_emit.py` re-exporting `lower_fir` from a sibling would drop it
    # from 99 names to two while every decision import moved next door -- and the ratchet would invite
    # deleting the entries. Pin that emission is still defined here; if it genuinely moves, this fails and the
    # ledger's root has to move with it, deliberately, rather than the debt appearing to evaporate.
    source = ast.parse(Path(_module_source(_EMITTER)).read_text(encoding="utf-8"))
    # Module level only: `ast.walk` would accept a homonym nested inside a facade function.
    defined = {node.name for node in source.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    assert {"lower_fir", "_Emitter"} <= defined, (
        "lower_fir and _Emitter are no longer defined in the module the debt ledger measures; point _EMITTER "
        "at wherever emission now lives and re-record the ledger there"
    )


def test_every_recorded_owner_is_a_real_module() -> None:
    # The ledger deliberately records SYMBOLS as well as modules -- 91 of its names are symbols (81 classes
    # and functions, 10 type aliases), which is the whole point of measuring at symbol level. What must
    # still resolve is every OWNER key. A
    # typo'd owner is already caught by the ratchet's `removed` arm, so this is a backstop that names the cause
    # directly rather than the sole catcher.
    unresolved = sorted(owner for owner in _EMITTER_FRONTEND_DEBT if not _module_source(owner))
    assert not unresolved, f"recorded owners that are not modules: {unresolved}"


def test_the_guard_root_fails_loudly_when_it_does_not_resolve() -> None:
    # A typo in the root would otherwise yield an empty import set and a permanently green guard.
    with pytest.raises(ValueError):
        direct_imports("holoso._frontend._fir._emit_that_does_not_exist")


def test_emission_rejection_sites_only_shrink() -> None:
    # M5 retires EmissionRejection: every refusal moves upstream, diagnostic-identical. The count is the debt,
    # and the corpus pins the messages, so this only has to stop the number from drifting either way.
    #
    # FIVE numbers, exactly, because each alone is defeatable and a ceiling is not a ratchet:
    #   - `raise` statements alone: hoisting them into a `_reject(...)` helper drops it to zero with
    #     byte-identical diagnostics and nothing moved upstream;
    #   - constructions alone: a helper that constructs internally lets a NEW refusal be added while the
    #     construction count stays put -- measured, 42 constructions and 41 direct raises;
    #   - both of those together: a `@classmethod` factory changes BOTH shapes at once, since `cls(...)` is not
    #     a Name and `raise EmissionRejection.make(...)` is a Raise of an Attribute call, so three new refusals
    #     hid behind one while the pair still read (42, 42). Such factories are ordinary style in this codebase.
    #   - `<=` on any of them would permit 42 -> 41 -> 42 regrowth.
    # Every NAME occurrence is therefore counted too, which no rewrite of the call shape can move without
    # moving the number. A genuine upstream move changes all five and updates them here in the same commit.
    # Counted over the PACKAGE, not `_emit.py` alone: the file is ~1500 lines against a ~2000 soft limit, so
    # splitting it is a plausible refactor, and a file-scoped count would read the move as M5 progress.
    constructed = 0
    raised = 0
    for module in sorted(Path(_module_source(_EMITTER)).parent.glob("*.py")):
        source = ast.parse(module.read_text(encoding="utf-8"))
        constructed += sum(
            1
            for node in ast.walk(source)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "EmissionRejection"
        )
        raised += sum(
            1
            for node in ast.walk(source)
            if isinstance(node, ast.Raise)
            and isinstance(node.exc, ast.Call)
            and isinstance(node.exc.func, ast.Name)
            and node.exc.func.id == "EmissionRejection"
        )
    named = sum(
        1
        for module in sorted(Path(_module_source(_EMITTER)).parent.glob("*.py"))
        for node in ast.walk(ast.parse(module.read_text(encoding="utf-8")))
        if (isinstance(node, ast.Name) and node.id == "EmissionRejection")
        or (isinstance(node, ast.Attribute) and node.attr == "EmissionRejection")
    )
    # Every `raise` in the emitter, whatever it raises and whatever the class is called. This is the
    # rename-proof and class-proof number: renaming EmissionRejection, or refusing with a different exception
    # type, moves the three counts above to zero or leaves them flat while the refusals are all still there.
    # It counts the emitter's `raise AssertionError` sites too, so converting one to a plain `assert` -- a
    # cleanup this project's style would welcome -- fires this with a message about refusals. That is the
    # intended trade: M5's end state is an emitter that does not raise.
    emitter = ast.parse(Path(_module_source(_EMITTER)).read_text(encoding="utf-8"))
    # These last two are scoped to EMISSION'S MODULE, deliberately unlike the three above. `EmissionRejection`
    # is emission's own exception wherever it is written, so counting it package-wide is right; a bare `raise`
    # in `_analyze.py` is the analyzer's business and package-scoping these would measure 312 instead of 48.
    # The cost is that moving a raising helper to a sibling reads as progress here -- which the guard pinning
    # where `lower_fir` is defined turns into a loud failure rather than a silent one.
    raises_here = sum(1 for node in ast.walk(emitter) if isinstance(node, ast.Raise))
    # Counting refusal SYNTAX cannot see a refusal PATH. The emitter has 21 functions that raise and 10 of them
    # are already called from several places, so one more call to `_carrier_float` or to any hoisted helper is
    # a brand-new refusal with every syntax count unmoved -- measured. Worse, the TIDY hoist is free: add a
    # helper and convert exactly one site, and the helper's own raise replaces the converted one, after which
    # every further call is invisible. So the call sites into raising functions are counted as well, which both
    # of those shapes do move. None of this proves a refusal moved UPSTREAM; the frozen rejection corpus does,
    # by pinning the public class and message. These numbers only make a change impossible to miss.
    # TRANSITIVE, and that is the whole point: computed non-transitively, hoisting a function's SOLE raise into
    # a helper drops the still-refusing host out of the set, after which every one of its call sites -- which
    # still refuse, through the helper -- contributes nothing in either direction. Measured on `_emit_cast`:
    # all five numbers flat, and 8 of the 21 raising functions can be emptied out this way one at a time.
    bodies = {node.name: node for node in ast.walk(emitter) if isinstance(node, ast.FunctionDef)}
    raising = {name for name, node in bodies.items() if any(isinstance(x, ast.Raise) for x in ast.walk(node))}
    while True:
        callers = {
            name
            for name, node in bodies.items()
            if name not in raising
            and any(
                (isinstance(call.func, ast.Name) and call.func.id in raising)
                or (isinstance(call.func, ast.Attribute) and call.func.attr in raising)
                for call in ast.walk(node)
                if isinstance(call, ast.Call)
            )
        }
        if not callers:
            break
        raising |= callers
    into_raising = sum(
        1
        for node in ast.walk(emitter)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id in raising)
            or (isinstance(node.func, ast.Attribute) and node.func.attr in raising)
        )
    )
    measured = (constructed, raised, named, raises_here, into_raising)
    assert measured == (42, 42, 42, 48, 146), (
        f"emission's refusal shape is {measured}, recorded (42, 42, 42, 48, 146) as "
        "(constructions, direct raises, name occurrences, raise statements, calls into raising functions). "
        "A DROP IS NOT SELF-EVIDENT PROGRESS: hoisting into a helper, renaming the class, or refusing with "
        "another type all lower a count while every refusal stays. What proves a refusal moved upstream is the "
        "frozen rejection corpus, which pins the public class and message; update these only alongside it"
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


def test_lower_fir_runs_the_plan_verifier_BEFORE_emission_walks(monkeypatch: pytest.MonkeyPatch) -> None:
    # The verifier cannot fail for any result today (see its docstring), so nothing else in the suite would
    # notice if the call were deleted -- and it exists to be in place BEFORE M1 rewrites recording. Both facts
    # have to be pinned: an earlier version of this test asserted only that the verifier was CALLED, and moving
    # the call after `_Emitter(...).emit()` passed it while destroying the whole point, since emission reaches
    # `call_plans` with a bare subscript and would raise the unlocated KeyError first.
    import holoso._frontend._fir._emit as emit_module

    trace: list[str] = []
    monkeypatch.setattr(emit_module, "verify_plan_totality", lambda result: trace.append("verify"))
    original_emit = emit_module._Emitter.emit

    def traced_emit(self: object) -> object:
        trace.append("emit")
        return original_emit(self)  # type: ignore[arg-type]

    monkeypatch.setattr(emit_module._Emitter, "emit", traced_emit)

    class Kernel:
        def __init__(self) -> None:
            self.s = 0.0

        def step(self, x: float) -> float:
            self.s = x + 1.0
            return self.s

    emit_module.lower_fir(Kernel().step)
    # The pin is on the ORDER of the two, and on the call existing at all. It does NOT pin the seam as tightly
    # as that reads: moving the call into `_Emitter.__init__` still passes, since construction precedes the
    # traced `emit`. Nor does it check that the verifier is handed the result actually emitted -- the stub
    # ignores its argument. What it does catch, measured, is the call moved after emission or deleted.
    assert trace == ["verify", "emit"], f"the verifier must run at the lower_fir seam, before emission, got {trace}"


def test_the_plan_verifier_catches_both_block_set_divergences() -> None:
    # Recovered from a review transcript rather than reasoned about: a walked block missing from
    # `executable_blocks`, and one missing from `block_in`, both escape `_check_reachability_settled` when the
    # block is a SINK, because that gate catches the direction only through a block's own out-edges. Left
    # unchecked the first reaches emission and dies with an unlocated "block N was not sealed with a
    # terminator". M1 rewrites recording, which is exactly when these become producible, so both are pinned.
    from holoso._frontend._fir._analyze import Analyzer, verify_plan_totality

    class Kernel:
        def step(self, x: float) -> float:
            if False:  # a statically dead arm, so the exit is reached by one edge and is a walk sink
                return x + 1.0
            return x

    stable = Analyzer(Kernel().step).fixpoint()
    verify_plan_totality(stable)  # the real analysis is total

    unmarked = Analyzer(Kernel().step).fixpoint()
    unmarked.executable_blocks.discard(unmarked.unit.exit)
    with pytest.raises(AssertionError, match="did not mark executable"):
        verify_plan_totality(unmarked)

    envless = Analyzer(Kernel().step).fixpoint()
    envless.block_in.pop(envless.unit.exit, None)
    with pytest.raises(AssertionError, match="no recorded environment"):
        verify_plan_totality(envless)


def test_the_plan_verifier_catches_a_severed_jump_edge() -> None:
    # Recovered from a review transcript, and the worst shape of the three: dropping a jump edge whose target
    # keeps another predecessor leaves every block walked and every table total, so the block-level arms see
    # nothing -- and emission then produces DIFFERENT HIR with no error at all. Measured before the arm existed.
    from holoso._frontend._fir._analyze import Analyzer, verify_plan_totality
    from holoso._frontend._fir._ir import BlockId, Jump

    class Kernel:
        def __init__(self) -> None:
            self.s = 0.0

        def step(self, x: float, flag: bool) -> float:
            if flag:
                y = x + 1.0
            else:
                y = x + 2.0
            self.s = y
            return y

    result = Analyzer(Kernel().step).fixpoint()
    verify_plan_totality(result)  # the real analysis is consistent

    predecessors: dict[BlockId, int] = {}
    for _, target in result.executable_edges:
        predecessors[target] = predecessors.get(target, 0) + 1
    severable = [
        (source, target)
        for source, target in sorted(result.executable_edges, key=lambda e: (e[0].index, e[1].index))
        if isinstance(result.unit.blocks[source].terminator, Jump) and predecessors[target] > 1
    ]
    assert severable, "the kernel must produce a jump into a merge for this test to mean anything"
    result.executable_edges.remove(severable[0])
    with pytest.raises(AssertionError, match="jump edge is missing"):
        verify_plan_totality(result)


def test_the_plan_verifier_catches_a_severed_residual_branch_arm() -> None:
    # The jump arm was not enough: a branch whose condition never settles takes BOTH arms, so severing one
    # leaves every block walked and every table total and dies inside emission with "phi N in block M has arms
    # for predecessors []". A branch whose condition folded keeps one arm legitimately and must not fire this.
    from holoso._frontend._fir._analyze import Analyzer, verify_plan_totality
    from holoso._frontend._fir._ir import Branch

    class Kernel:
        def __init__(self) -> None:
            self.s = 0.0

        def step(self, x: float, flag: bool) -> float:
            if flag:  # runtime condition, so both arms are obligatory
                self.s = self.s + 1.0
            return self.s + x

    result = Analyzer(Kernel().step).fixpoint()
    verify_plan_totality(result)

    arms = [
        (source, target)
        for source, target in sorted(result.executable_edges, key=lambda e: (e[0].index, e[1].index))
        if isinstance(result.unit.blocks[source].terminator, Branch)
    ]
    assert len(arms) == 2, "the kernel must keep a residual two-armed branch for this test to mean anything"
    result.executable_edges.remove(arms[-1])
    with pytest.raises(AssertionError, match="edge is missing"):
        verify_plan_totality(result)


def test_the_plan_verifier_catches_a_missing_binding_fact() -> None:
    # M1 rewrites fact recording, so a dropped fact is exactly the regression to expect from it. Emission reads
    # one for every destination it materializes and fails deep in the walk with a named assert; this names the
    # block and the destination before the walk starts.
    from holoso._frontend._fir._analyze import Analyzer, verify_plan_totality

    class Kernel:
        def __init__(self) -> None:
            self.s = 0.0

        def step(self, x: float) -> float:
            self.s = x * 2.0 + 1.0
            return self.s

    result = Analyzer(Kernel().step).fixpoint()
    verify_plan_totality(result)

    assert result.binding_facts, "the kernel must record binding facts for this test to mean anything"
    result.binding_facts.pop(next(iter(result.binding_facts)))
    with pytest.raises(AssertionError, match="no recorded fact"):
        verify_plan_totality(result)


def test_the_plan_verifier_catches_a_missing_parameter_fact() -> None:
    # Recovered from a review transcript. Parameters are not op destinations, so the binding-fact arm cannot
    # see them -- and emission types the module's INPUT PORTS from their entry-environment facts. Dropping one
    # passed every other arm and silently emitted a float input where the kernel declares a bool: an ABI
    # divergence with no error, which is the worst class here and squarely inside what M1 rewrites.
    from holoso._frontend._fir._analyze import Analyzer, verify_plan_totality
    from holoso._frontend._fir._ir import Local

    def kernel(flag: bool) -> float:
        return 1.0

    result = Analyzer(kernel).fixpoint()
    verify_plan_totality(result)

    parameter = result.unit.params[0]
    result.block_in[result.unit.entry].facts.pop(Local(parameter))
    with pytest.raises(AssertionError, match="no recorded fact for emission to type"):
        verify_plan_totality(result)
