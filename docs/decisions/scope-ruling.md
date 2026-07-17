# Scope ruling — the trimmed Python subset

Status: maintainer-ruled (planning session, 2026-07-17); Codex consult X1 recorded below; landing steps refer to
`docs/campaign.md`. Line numbers at `dev` = 54bfe78.

The utility oracle: the 24 kernels in `examples/` plus the natural spelling of common numerical / DSP / linalg /
control / estimation idioms. No cybersec framing: a guard that exists only to withstand a hostile construction is
itself a trim candidate; graceful refusal of the honestly-weird is sufficient.

## 1. Trims (each lands as a documented located rejection + fail-before/pass-after regression + DESIGN.md edit)

| # | Feature | Sites | Owned tests | Lands |
|---|---------|-------|-------------|-------|
| T1 | `getattr` call support (`setattr` never existed — already rejects) | `_analyze.py:2155-2167` | `test_frontend_calls.py:672`, `:693` (+ comment `:611`) | S2.7 |
| T2 | Elementwise array-comparison masks | `_analyze.py:1237-1272`, `_emit.py:838-866` (incl. `mask_operand`/`mask_broadcast`) | `test_matrix.py:1362`; AUDIT: `tests/_fuzz.py` if its generator can emit array comparisons | S2.8 |
| T3 | 0-d array support (reject at creation; today: scalarize at ~6 sites + reject navigation at ~6) | unwraps `_analyze.py:1139/:1243/:1292/:2350`, `_emit.py:492/:762/:849/:1167`; guards `_analyze.py:960/:1488/:1834/:2023/:2269/:2284` | `test_frontend_aggregates.py:1046`, `test_matrix.py:1028`, `:1678` | S2.8 |
| T4 | `isinstance`, both arms | scalar `_analyze.py:2314-2326`; record-subject `:2217-2244`; classinfo `_fold.py:75-134` | `test_frontend_calls.py:826/:937/:1063/:1461/:1482/:1499` | S2.9 |
| T5 | Enum member provenance / LOST taint (IntEnum constants keep folding to base value) | `_value.py:43-55/:85-100/:280-287/:418-439`, `_analysis_support.py:341-388`, `_fold.py:295-309` | `test_fir_foundations.py` enum rows, `test_frontend_aggregates.py` enum fields, `test_frontend_calls.py:1063/:1277` | S2.9 |
| T6 | str value methods (str constants stay) | `_analyze.py:1737-1783` value-method minting; `_value_methods` table `:322` | `test_fir_analyze.py:791`, `test_frontend_calls.py:1277` | S2.9 |
| T7 | Dataclass forensics (plain-dataclass validation + argument-to-field mapping errors stay; descriptor-backed fields become one plain refusal) | `_fold.py:182-264` (bytecode `__post_init__` scan, `__defaults__` identity forensics, descriptor taxonomy) | `test_fir_foundations.py:497` (dies), `test_frontend_aggregates.py:1398` (simplifies), `:1564` (mapping part stays) | S2.9 |
| T8 | Hook-guard thinning: one plain located refusal of components with custom `__setattr__`/descriptors (the refusal STAYS) | `_analysis_support.py:544-571`, `_analyze.py:1694-1716` | `test_fir_analyze.py:763` (stays as the refusal's test), `test_language_features.py:472/:523` (simplify) | S2.9 |
| T9 | Starred assignment targets `a, *rest = seq` (plain unpacking stays) | `_build.py:473-529` (star window machinery) | `test_frontend_aggregates.py:135` (partial rewrite), `:1906`; `test_matrix.py:343` REWRITE to explicit indexing (incidental use) | S2.10 |
| T10 | H2: starred element in a list display `[x, *rest]` — was never supported; becomes a *documented* located rejection | `_build.py:699` display arm (no `ast.Starred` case → default reject today) | `test_verify.py:454` (skip → enabled rejection test) | S2.10 |
| T11 | Misc adversarial-only micro-guards (§3, Codex-ratified) | see §3 | see §3 | S2.10 |

Draft rejection messages (final wording at landing; written for the honest user):
- T1: `getattr is not supported in a kernel; spell the attribute access directly (x.name)`
- T2: `elementwise array comparison is not supported`
- T3: `a 0-dimensional array is not supported; use the scalar directly`
- T4: `isinstance is not supported in a kernel: values are statically typed`
- T6: `str methods are not supported in a kernel; strings are inert constants here`
- T9/T10: `a starred element is not supported in an assignment target / list or tuple display`

## 2. Keeps (explicit maintainer rulings)

- `np.trace` / `np.outer` / `np.dot`: KEEP — examples will be added later. Consequence: defect C5 (integer trace
  folds to float, `_lib/_linalg.py:64`) is FIXED in S2.5, not deleted; E4's diagnostic gate stays and is fixed.
- `divmod`, `sum`: KEEP — "do not touch arithmetics"; divmod will be used soon.
- Kept despite zero example use (cheap + natural): `break`/`continue`, `assert`, `pass`, `del`, walrus, chained
  comparisons, list comprehensions, `list()`/`tuple()` conversion, starred CALL arguments, runtime bitwise/shift
  containment at MIR, oversized-range/fuel resource bounds (they protect honest large constants).
- H3 (`bool(2.0**-200)` folds on host float64): ruled FASTMATH POLICY — documented in the new DESIGN.md
  "Fastmath policy" section (S2.10); enabled test asserts the documented behavior.

## 3. Misc guard enumeration (T11) — for Codex ratification

| Guard | Site | Disposition |
|---|---|---|
| Hostile-metaclass admission survival | `_value.py` admission; `test_fir_foundations.py:497` | REMOVE (pure hostile-construction) |
| `__class__`-override refusal in record isinstance | `_analyze.py:2232-2236` | dies with T4 |
| classinfo planted-dunder validation | `_fold.py:84-90` | dies with T4 |
| `str.format`/`format_map` blocking | `_analyze.py:1768` | dies with T6 |
| Construction-schema decoration-vs-live divergence checks | `_fold.py:199-263` | dies with T7 |
| Admission cycle/depth refusal | `_value.py:232-233/:257` | KEEP (honest recursive/deep structures deserve graceful refusal; depth is a resource bound) |
| Record truth `__bool__`/`__len__` override refusal | `_analysis_support.py:536` | KEEP (a dataclass with `__bool__` is plausible honest code; refusal prevents silent divergence) |
| Oversized-range receiver/argument rejection | `_fold.py` admission | KEEP (resource bound serving honest code) |
| Pre-bound-builtin (live mutable receiver) never-folds rule | admission harness | KEEP pending Codex view (an honestly captured bound method is plausible) |
| Metaclass-property shadow handling in component reads | `test_language_features.py:523` territory | simplify with T8 |

## 4. Defect fates (the 18-row register + TODO extras)

| Defect | Fate | Lands |
|---|---|---|
| A1 dedent reparse | FIX | S2.3 |
| A2 int-array comparison in float | DELETED by T2 | S2.8 |
| A3 empty matmul trailing dim | FIX | S2.5 |
| B1 store-type policy | ENFORCE fixed storage schema (HANDOFF spec) | S2.12 |
| C1 0-d bool cast misclassify | DELETED by T3 | S2.8 |
| C2 bool/non-bool compare late refusal | FIX (move to analysis, located) | S2.5 |
| C3 inherited record fields | FIX | S2.4 |
| C4 `dims` false-positive | FIX | S2.4 |
| C5 integer trace folds float | FIX (trace kept by ruling) | S2.5 |
| C6 unroll-cache first-visit freeze | FIX | S2.6 |
| D1 nested None contract assert | FIX | S2.4 |
| D2 annotation NameError escape | FIX | S2.3 |
| E1 stub-line attributions / empty `.location` | FIX observable semantics (E1-lite); datatype migration in Stage 4 | S2.11 / M6 |
| E2 f-string raise degradation | FIX | S2.3 |
| E3 unlocated join message | FIX (re-confirm reproduction first) | S2.5 |
| E4 wrong reduction diagnostic | FIX (re-confirm reproduction first) | S2.5 |
| F1 pre-fuel materialization | FIX | S2.5 |
| G1 predication never fires | FIX | S2.13 |
| TODO: shared-live-out bare AssertionError | FIX (located SynthesisError) | S2.11 |
| TODO: EmissionRejection locations | FIX | S2.11 |

## 5. Codex consult X1 record

Session `019f70a0-9b3a-7040-bef3-d53366734dc6`, gpt-5.6-sol, ultra effort, 2026-07-17. Full log retained in the
session workspace. Verdict: the §4 fate table is ratified in full; every trim T1-T10 is ratified; §3 ratified
with two contests, both accepted below. No maintainer utility ruling is reopened.

Amendments accepted into this dossier (all verified against the tree by Codex with file:line evidence):

- T1: keep the `_analyze.py:2155` arm as the located-refusal site (specific guidance message) rather than
  falling through to the generic call rejection. Two additional incidental uses to rewrite to dotted access:
  `test_frontend_aggregates.py:1691`, `:1803`.
- T2: rationale corrected — the "mask has no consumer" claim is FALSE (masks are indexed, cast, and used in
  control flow per `test_matrix.py:1362`); the trim stands on utility grounds. The dispatch site
  `_analyze.py:899` added to the deletion list. `tests/_fuzz.py` cannot emit array comparisons (scalar-only
  pools: `_fuzz.py:201/:433/:882`) — no fuzz change needed.
- T3: blast radius extended — reject 0-d at EVERY creation/admission boundary: empty-shape `ArrayLayout`
  (`_fact.py:68`), global loading (`_build.py:737`), state resets (`_analyze.py:446`), reshape targets
  (`:1205`), concrete-call normalization (`:2367`), 0-d emission indexing (`_emit.py:950`). Additional owned
  tests: `test_frontend_aggregates.py:1192`, `test_matrix.py:1283/:1431/:1690`.
- T4: include the fold-time whitelist entry (`_fold.py:137`); additional owned tests
  `test_frontend_calls.py:1135/:1307`, `test_frontend_aggregates.py:1539`, `test_matrix.py:1726`.
- T5: additional provenance sites `_value.py:122/:418`, `_analyze.py:1634/:2367`; both IntEnum AND StrEnum fold
  to base values; numpy scalar-kind provenance (np.bool_/np.int64/np.float64 carriers) REMAINS.
- T6: do NOT delete the shared minted-method machinery (`_analyze.py:318`, `_fold.py:267`) — range/integer/
  numpy receivers keep it; refuse `StaticStr` receivers early. Additional owned tests
  `test_frontend_calls.py:1000/:1167/:1277`.
- T7: boundary narrowed — remove ONLY decoration-vs-live forensics (bytecode `__post_init__` scan,
  `__defaults__` identity checks, `_fold.py:239` area); RETAIN refusals for custom hooks, `default_factory`,
  descriptor-backed fields, `InitVar`, `init=False` (`_fold.py:204/:223/:249`) — those are ordinary-dataclass
  validation serving honest users. `test_fir_foundations.py:497` belongs to T11, not T7.
- T8: read-side hook refusal (`_analysis_support.py:532/:544`) KEEPS its semantics (honest lazy/accessor
  objects would silently diverge); thinning = diagnostic consolidation only. Metaclass-property shadow handling
  keeps its behavior (raw MRO lookup is already the simplest correct form); only diagnostics consolidate.
- T9: reject at the shared assignment-target star branch (`_build.py:466` area). `test_matrix.py:343` rewrite
  must use `first = v[0]; rest = [v[1], v[2]]` — NOT `rest = v[1:]`, which would make `rest` an ndarray and
  silently change `rest + rest` from concatenation to elementwise addition.
- T10: covers BOTH list and tuple displays (`_build.py:694`, `:710`). Fourth hollow test discovered:
  `test_frontend_aggregates.py:115` passes via `match="unpack"` matching the test's own qualified name while
  the real diagnostic is `expression Starred is not supported` — strengthen at landing (S2.2/S2.10).
- Guards kept in addition to §3: the oversized-argument guard for surviving minted methods (`_fold.py:279`);
  the aggregate element-count budget (`_value.py:24`) files under the cycle/depth KEEP row.
- Defect-fix precision for Stage 2: D1 retains top-level `None` returns and rejects only nested `None` early
  (`_signature.py:172`, `_emit.py:283`). E1-lite requires `Origin` to carry filename data first (`_ir.py:19`,
  `holoso/_errors.py`) — sequence inside S2.11. E2 must distinguish statically-foldable f-strings (fold them)
  from runtime interpolation (existing boundary tests `test_frontend_control.py:859/:902`). E3 reproduced
  (`.location is None`; sites `_analyze.py:424/:1041`). G1's repair must PRESERVE or safely hoist guard-block
  operations under a budget — fusion deletes the guard block (`_hir/_if_convert.py:151/:216`), so merely
  relaxing the `g.operations` predicate would delete live computation.
- B1 note: tests currently asserting ACCEPTANCE of type-changing rebinding reverse polarity at S2.12; record
  them in the h1/regression ledger as deliberate reversals, not regressions.

Example-impact sweep (Codex): no example uses T1-T6/T8-T10; T7 keeps the plain construction
`finite_set_current_controller.py:71` depends on; T9/T10 leave starred CALL arguments (`ekf1_stateful.py:43`)
untouched.
