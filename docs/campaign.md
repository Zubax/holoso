# Holoso frontend campaign: trim → stabilize+freeze → architecture gate → restructure

## Context

The `main..dev` frontend rewrite (FIR pipeline: `_build.py` AST→FIR 832 LOC, `_analyze.py` SCCP/W-D abstract
interpretation 2,498 LOC, `_emit.py` FIR→HIR 1,495 LOC) landed as a monumental first draft. The deep review
(HANDOFF.md, committed in S0) confirmed 18 defects, hygiene rot (87 stale skip decorators, DESIGN/TODO
contradictions, 3 hollow tests), and an oversized Python subset. The maintainer has now ruled on scope (below),
and the architecture question he posed — a bytecode-like IR sublayer split — has been evaluated hard by two
adversarial design agents plus code-level verification. Campaign: prune → stabilize+freeze → architecture gate
(spike) → restructure.

## The architecture question — what the evaluation established

- The literal "Python → bytecode-like IR first" sublayer ALREADY EXISTS: `_build.py` emits an 18-op/5-terminator
  non-SSA CFG over mutable Places, purely syntax-directed (chained comparisons, and/or, ifexp, aug-assign,
  unpacking, slices, comprehensions all desugared there). Every remaining pre-analysis delta is negative
  (StaticFor is richer than FOR_ITER), impossible without facts (kwargs binding, attribute meaning, starred
  flattening, inlining frames, getattr), or cosmetic. Pre-analysis bytecode is dead; both agents concur with the
  prior panels by direct code contact.
- The real boundary defect is DOWNSTREAM: emission re-decides semantics (41 unlocated `EmissionRejection`s,
  `_fact_sem`/`_leaf_is_int` re-classification, re-derived routings `_emit.py:621-641`/`:954-964`) — the
  A2/B1/C1/C2 habitat. The cure is post-stabilization resolution totality: every decision made once, in the
  analyzer/residualizer, emission consuming a closed typed plan surface (no Fact, no registry, no Py* op, no
  rejections, no kind decisions; Braun SSA and the `_exit_identity` port dedup remain legitimate value-level
  emitter work).
- B1's fixed storage schema is a PRECONDITION: today a fact's kind can differ from the kind its cell carries
  (`_emit.py:756-759`), so no resolved plan can be typed by analysis until stores are schema-checked.
- Raw byte-identity cannot survive the single landing that swaps the emitter (HIR node-id assignment order is
  byte-visible downstream, `_mir/_lower.py:370`); the maintainer PRE-AUTHORIZED a canonical-form gate for that
  one landing: HIR equality after `renumber` + cosim bit-exactness on all 25 cases + schedule metrics equality +
  full diagnostic corpus, then re-freeze. All other landings stay raw-byte-gated.
- Remaining open decision (morph in place vs staged transplant to a materialized resolved IR) is settled at the
  Stage-3 gate by a time-boxed spike with pre-agreed criteria; failure default = morph.

## Maintainer rulings (this session, final)

TRIM (each = documented located rejection + fail-before/pass-after regression + DESIGN.md edit in same commit):
1. getattr call support (`_analyze.py:2155-2167`; setattr never existed — already rejects).
2. Elementwise array-comparison masks (`_analyze.py:1237-1272`, `_emit.py:838-866`) — deletes defect A2.
3. 0-d array support: reject at creation; delete ~12 scalarize/guard sites — deletes defect C1.
4. isinstance, both arms (`_analyze.py:2217-2244`, `:2314-2326`) incl. its `__class__`/metaclass guards.
5. Enum member provenance / LOST taint (~120 LOC across `_value.py`, `_analysis_support.py:341-388`,
   `_fold.py:295-309`); IntEnum constants keep folding to their base value.
6. str value methods (`_analyze.py:1737-1783`); str constants stay.
7. Dataclass forensics (`_fold.py:182-264`): bytecode `__post_init__` scan, `__defaults__` identity forensics,
   descriptor-field taxonomy; plain-dataclass validation and argument-to-field mapping errors stay.
8. Hook-guard thinning (`_analysis_support.py:544-571`): one plain located refusal of components with custom
   `__setattr__`/descriptors; the refusal itself stays (dropping it would miscompile honest validating classes).
9. Starred assignment targets `a, *rest = seq` (`_build.py:473-529`); plain unpacking stays.
10. H2: starred element in a list display `[x, *rest]` = documented located rejection (was a lying skip).
11. Misc adversarial-only micro-guards: enumerated in S1.1, Codex-ratified, removed in S2.10.

KEEP (explicit rulings): np.trace/outer/dot (examples to be added later → C5 must be FIXED, not deleted);
divmod and sum ("do not touch arithmetics"; divmod will be used soon). Also keep (cheap + natural, zero-use):
break/continue, assert, pass, del, walrus, chained comparisons, comprehensions, list()/tuple() conversion,
starred CALL arguments, runtime bitwise/shift containment at MIR.

H3 ruling: the `bool(2.0**-200)` float64-truth fold is FASTMATH POLICY. Create a consolidated "Fastmath policy"
section in DESIGN.md gathering all scattered fastmath decisions (host-precision constant folding, constant-truth
on float64, ideal-result transcendental folds, x/x==1, FMA rounding, zero/infinity rewrites); home H3 there;
replace the skip at `test_verify.py:1501` with an enabled test asserting the documented behavior.

B1 ruling stands as written in HANDOFF.md (fixed storage schema; located rejection at the type-changing store).

## Non-negotiables (every stage)

- No cybersec framing; findings requiring hostile constructions are rejected at review consolidation.
- Codex = co-designer: `codex exec -m gpt-5.6-sol` at ultra effort, stdin from /dev/null, session ids logged;
  a stuck/guardrailed session is RESUMED (`codex resume`), never restarted; transient errors retried.
- `review-loop` skill after EVERY step (terse unbiased prompts; Claude ultrathink full-spectrum + Codex ultra
  correctness; reviewers read-only, no suite re-runs; first clean round ends).
- One step = one commit+push+CI-green unit. CI fires on push to every branch (5 jobs; 4 on self-hosted
  `ci-runner`); md-only commits fire NO CI (paths-ignore) — previous green carries; fold doc edits into the
  code commit they describe. Throwaway-branch pushes sanctioned for tentative CI verdicts.
- Heavy suites never block locally: CI runs them per push; `ssh user@cy-hans-reiser.local` /
  `ssh user@ci-terry-davis.local` (24 cores/96 GB) for pre-push certification and debugging only.
- Local gate per step: `HOLOSO_REGALLOC_EFFORT=10 nox -s tests typecheck black` (baseline-bearing tests pin
  their knobs internally; golden suite must pin via the `test_metrics.py:112-126` fixture pattern).
- Speed discipline (maintainer-directed): two-tier local runs — mid-step iterations run only the frontend-fast
  corpus (`test_fir_*`, `test_frontend_*`, `test_determinism`, `test_language_features`) at `-n 8` (~1 min; the
  ⅔-core cap exists to bound P&R memory, which the light suite has none of), and the full 18-min session runs
  exactly once per step. The full gate launches IN PARALLEL with the review round (reviewers read the diff, not
  the gate); a reviewer-invalidated gate run is cheap. Exception: steps that can change emitted Verilog (trims,
  S2.11 port order, S2.12 B1, S2.13 G1) also run `test_latency_freeze` + `test_metrics` in the fast tier —
  that guard class is what catches RTL-visible hazards pre-commit. Trims delete `match=` pins suite-wide, which
  the fast tier cannot see; the once-per-step full session retains that authority.
- Every defect fix / trim: regression test verified to fail before, pass after (observed failure quoted in the
  commit message). DESIGN.md high-level updates ride the same commit. No compatibility shims, clean breaks.

## Stage 0 — Bootstrap (1 step)

S0: commit HANDOFF.md (currently untracked), this plan as `docs/campaign.md`, `docs/decisions/` convention note.
Md-only; no review-loop needed.

## Stage 1 — Scope closeout (2 steps)

S1.1: finalize the scope dossier for the RATIFIED list: per trim — exact sites, owned tests (which of the 87
skips / 377 `match=` pins), rejection message drafts, DESIGN.md sections, landing step; plus the enumerated
misc-corner-guard list (ruling 11). `docs/decisions/scope-ruling.md`.
S1.2: Codex consult X1 (dossier + defect register): its per-defect scope ruling HANDOFF mandates, per-trim
blast-radius check, misc-guard ratification. Disagreements recorded in writing; anything that would reopen a
maintainer ruling goes back to the maintainer, otherwise proceed.

## Stage 2 — Semantic stabilization, then freeze (15 steps)

Ordering laws: determinism before triage; triage before fixes (enabling ~48 tests upgrades the net); all
Verilog-visible changes (G1, E1-lite port order, B1 rejections, trims) before golden capture; all diagnostic-
shape changes before the rejection corpus; trims before B1 (fewer schema rules; `_analyze.py` line-count relief).

- S2.1 Diagnostic determinism: structural total order at `_analyze.py:228` (`join_with` set-union iteration),
  `:616` (`executable_blocks` set), `:643` (state union). Re-enable the seed oracle `test_determinism.py:111`
  (stale skip — CordicSinCos tuple returns shipped). New regression: kernel with two competing rejections,
  identical exception text across PYTHONHASHSEED 0/3/31337 (subprocess).
- S2.2 H1 triage + enablement + H5: procedure per §H1; 87-row ledger `docs/decisions/h1-ledger.md`; enable
  passers of retained features; fix ~12 stale list-vs-array return-contract expectations; relabel remaining
  markers with their fate+landing step (token kept until landed). Repair hollow tests: `test_fir_analyze.py:209`
  (assert the state leaf, not the parameter), `test_fir_builder.py:58` (assert order/value, not count),
  `test_frontend_control.py:47` (live logger). Trial-push enablement wave on a throwaway branch first.
- S2.3 Build-layer fixes: A1 (`_build.py:148` dedent reparse), D2 (`:162` PEP-649 NameError), E2 (`:535`
  f-string raise message). 3 regressions.
- S2.4 Signature fixes: C3 (`_signature.py:148` inherited fields), C4 (`:89` dims false-positive), D1 (nested
  None contract assertion). 3 regressions.
- S2.5 Analysis/lib fixes: E3 (`_analyze.py:1041` unlocated join msg), E4 (`:2086` wrong reduction diagnostic),
  F1 (hoist fuel check above per-trip materialization `:1836`), C2 (bool/non-bool comparison refusal moved from
  `_emit.py:871` into analysis, located, message preserved), A3 + C5 (`_lib/_linalg.py` empty-product trailing
  dim; integer trace folds float — real fix since trace stays). Re-confirm E3/E4 agent reproductions first.
- S2.6 C6: unroll cache (`_analyze.py:1808`) keyed/revalidated against the stabilized iterable fact (first-visit
  freeze is the bug); Codex ping if the graph-identity constraint bites. Regression: conditional-rebind kernel.
- S2.7 Trim: getattr (the directive's first cut). Trim pattern: delete machinery; located rejection written for
  the honest user; rejection regression; convert/enable owned H1 tests; DESIGN.md edit; `match=` pin updates.
- S2.8 Trims: comparison masks (deletes A2) + 0-d arrays (deletes C1). Same pattern; one commit (shared array
  doctrine surface).
- S2.9 Trims: isinstance; enum provenance (keep base-value folding); str value methods; dataclass forensics +
  hook-guard thinning. Same pattern.
- S2.10 Trims/rulings closeout: starred assignment targets; H2 located rejection (kills the lying skip at
  `test_verify.py:454`); misc corner guards per X1 ratification; the new DESIGN.md "Fastmath policy" section
  consolidating scattered decisions, H3 homed there + enabled documented-behavior test replacing
  `test_verify.py:1501`.
- S2.11 E1-lite + located emission rejections: primary location = user call site (`origin[-1]`, fixing
  `matmul_:47:8`-class attributions from `_analysis_support.py:100-107`); `SynthesisError.location` populated on
  Build/Analysis/Emission rejections (today message-embedded only, `holoso/_errors.py:25-29`); deterministic
  state-port order under inlining (first-store key at `_analyze.py:624` compares positions across source
  functions — Verilog-visible); origins threaded to surviving EmissionRejection sites (TODO.md:28-30). Also fix
  the backend shared-live-out bare AssertionError (TODO.md:21-26) into a located SynthesisError (natural kernel
  triggers it). Regressions incl. enabling `test_matrix.py:2035` (stub misuse attributed to user call site).
- S2.12 B1 fixed storage schema — exactly the HANDOFF spec with its three traps: separate monotone schema flow
  (never a fact-kind transfer check); first definition establishes schema, independent first defs join with
  int→float promotion; bool←bool, int←int, float←float|int (conversion on the store edge); all else located
  rejection AT THE STORE; store-origin obligations resolved post-W/D/schema stabilization; explicit StorePlace
  roles exempting ifexp merge sinks (`_build.py:641`), comprehension accumulators (`:762`), ReturnPlace;
  merges/widening untouched. Deletes `_leaf_kind` (`_analyze.py:1071`), the towers (`:1044-1066`, `:528-604`),
  live-in state typing/dtype rebuild (`:572`), `_leaf_is_int` (`_emit.py:574`). Regression matrix = the
  four-outcome table + `x=0; x=input_float` rejects / `x=int(v) if c else v` legal float phi. DESIGN.md:585
  loses only "state-leaf join". Gate: `_analyze.py` line count ≤ pre-step (helpers to `_analysis_support.py`).
  Codex consult X2 precedes (B1 design note).
- S2.13 G1: `_hir/_if_convert.py:162` — guarded-region predication fires for the natural `if A: if B: S`
  spelling (currently requires an empty guard block). Verilog-visible → pre-freeze. HIR-shape regression;
  deliberate `_FROZEN_SCHEDULE`/`BASELINE` updates in same commit.
- S2.14 Freeze prep: DESIGN.md H4 reconciliation in full (`:345` array ports live; `:377` matmul shipped —
  story told straight both ways: in-kernel arrays live, matmul-on-array-parameter-PORTS the open gap
  (`test_matrix.py:2035` area, imu); `:271-283` contract restated to reality; `:451`; `:526` now true; `:583-585`
  per B1; frontend prose made maintainable). TODO.md: stale aggregate-gap paragraphs (32-47) removed, real gaps
  homed. `test_metrics.py:235` imu skip relabeled truthfully. octave_index guards re-keyed per (name, format):
  pin e6m18=(14,38) AND measured e8m36. H1 exit: delete registry + `parity_marks` + vacuous
  `test_parity_registry_is_empty` + mark plumbing in all 9 consumers; resolve non-decorator token sites
  (`_fuzz.py:875/1018/1570`, `_synth_targets.py:97`, docstrings, comments). Exit grep:
  `git grep -n FIR_PARITY_PENDING -- ':!HANDOFF.md' ':!docs/'` → zero.
- S2.15 THE FREEZE (Codex X3 precedes: corpus design, esp. public-class pinning): commit `tests/golden/`
  per §F + `tests/test_golden.py` + `tools/refreeze_golden.py` + shared dumper `tests/_hirdump.py` (factored
  from `test_determinism.py:89-108`, both consumers switched) + `.gitignore` re-include `!tests/golden/**`.
  Pre-commit: capture ×3 seeds byte-identical. Gate: full CI green at this commit; `verify-rtl-report-
  equivalence` skill across examples; optional overnight scaled fuzz on a runner VM; tag `freeze-1`.

Stage-2 exit gate: exit grep clean; every register row A1-G1 fixed-with-regression or trim-deleted-with-
regression; 25/25 cases green in cosim; `freeze-1` tagged; stage-level review-loop round clean.

### §F — Freeze corpus

```
tests/golden/
  README.md                      # update protocol
  verilog/<name>-e<we>m<wm>.v    # full RTL text; 25 files (24 specs; octave_index twice)
  hir/<name>.txt                 # PRE-optimize frontend HIR dump via tests/_hirdump.py; 24 files
  ports/<name>-e<we>m<wm>.txt    # port names/types + module_name + initiation_interval; 25 files
  rejection_kernels.py           # APPEND-ONLY rejected kernels (line numbers never shift; black from birth)
  rejections.txt                 # public class | rendered location | exact str(exc)
```
- Pin the PUBLIC exception class (what users catch), so Stage-4 can retire internal EmissionRejection
  corpus-identically. Rejection corpus covers: every trim, B1's four outcomes, C2/C6/F1-fuel, D1/D2/C3/C4, E2,
  representative pre-existing shapes (recursion, record iteration, 0-d navigation, stub shape mismatch,
  beyond-carrier constant, power-chain).
- `test_golden.py`: unmarked (light `tests` session + CI core job); autouse regalloc-knob pinning fixture
  copied from `test_metrics.py:112-126`; spec×format parametrization as `test_cosim_examples.py:44-46`;
  mismatch prints diff head + "deliberate change → tools/refreeze_golden.py; commit golden+code+DESIGN together".
  MANDATORY design constraint (maintainer): the golden suite must SHARE the per-(example, format) syntheses that
  `test_latency_freeze`/`test_metrics` already perform, not add 25 more — otherwise the light suite permanently
  ~2×es. Under xdist a session-scoped fixture is per-worker, so naive sharing still duplicates: use an on-disk
  build memo keyed by (example, format, knob set) with file locking, or `--dist loadgroup` grouping the three
  suites' cases by example. Settle the mechanism in the Codex X3 consult.
- Update protocol: goldens change ONLY in the commit carrying the causing code+DESIGN change. During Stage 4 any
  golden diff is prima facie violation except the newly-discovered-defect protocol: pause, land regression+fix+
  golden diff as its own baseline-update commit on dev, rebase branch, resume. CI's unset PYTHONHASHSEED makes
  every run a free seed trial of the corpus.
- Later trace/outer/dot examples (maintainer adding) extend the corpus as ordinary corpus-extension commits.

### §H1 — Triage procedure

1. Scratch worktree; strip the 87 skip decorators by parsing decorator extents (one is multi-line:
   `test_verify.py:1501`); never commit the strip.
2. Run marker-bearing files: `.nox/tests/bin/python -m pytest -p no:enabler -m "not cosim and not fuzz and not
   synth" -n <2/3 cores> <the 17 files>`. Expected ballpark 727 pass / 35 fail; deltas are findings.
3. Classify each of 87: ENABLE (passes, feature retained) | RELABEL→trim step (passes, feature trimmed —
   becomes the trim's rejection test) | FIX-EXPECTATION+ENABLE (stale ~12 return-contract rows) |
   RELABEL→fixing step (blocked by a Stage-2 defect) | CONVERT to located-rejection test (H2 class) |
   DOCUMENTED-DEVIATION with DESIGN/TODO home (H3 class).
4. Ledger: `docs/decisions/h1-ledger.md`, 87 rows: test → strip result → fate → landing step.
5. Exit grep at S2.14 (do NOT trust `test_parity_registry_is_empty` — vacuous; it gets deleted).

## Stage 3 — Architecture gate (4 steps)

- S3.1 Evaluation memo consolidating both design-agent reports (advocate's RIR schema: one Def form with three
  RHS kinds over leaf cells, cell-write lists not explicit phis, `join_kinds` table, state/port/const tables,
  closure+definedness+kind verifier, printer; skeptic's leak audit L1-L10 and cost accounting; the agreed
  redefinition of "mechanical"; the canonical-form gate). Spike criteria filled in. Markdown source committed;
  HTML render delivered to maintainer as a local file. Maintainer approves criteria BEFORE the spike.
- S3.2 Codex consult X4 on the memo (+ HANDOFF verdict). Iterate; record disagreement.
- S3.3 Spike on worktree branch `spike/resolved-ir` (throwaway pushes for CI verdicts). Timebox: 1 session,
  hard cap 2. Content: `_rir.py` schema as real dataclasses + prototype residualizer for the hard-construct
  subset {ekf1_stateful, finite_set_current_controller, iir1_hpf, + one branch-join kernel + one
  return-onto-state-port-dedup case} + the complete mechanical emitter. Criteria (pre-agreed):
  SC1 Verilog+HIR byte-identical to `freeze-1` for the subset (fallback: HIR canonical-identical after renumber
  + schedule metrics equal); SC2 emitter imports no Fact/registry/Py* (grep-checked); SC3 zero escapes from the
  closed op set (boundary-table additions allowed, decision-carrying node kinds not); SC4 residualizer ≤ ~1,200
  LOC for the subset (extrapolation ≤ ~1,800). Decision table: all pass at byte level → transplant with byte
  gate; pass at canonical level → transplant with the pre-authorized canonical gate for the swap landing;
  semantic divergence in >1 construct family, or SC4 blown, or timebox expiry → MORPH (default).
- S3.4 Codex X5 on spike diff + comparison results (resume X4 session); maintainer ruling; commit
  `docs/decisions/arch-ruling.md` (question, evidence, spike SHA, Codex position, ruling). Spike branch deleted,
  SHA recorded.

## Stage 4 — Restructure per ruling

Common: every landing byte-identical (golden verilog/hir/ports) and diagnostic-identical (rejections + 377
`match=` pins) to `freeze-1` — sole exception: the emitter-swap landing under the canonical gate, which
re-freezes in the same commit. Newly-discovered-defect protocol per §F. review-loop per step;
`verify-rtl-report-equivalence` at stage checkpoints.

### Variant MORPH (on dev directly; ~9 steps, each independently green)

- M0 Guards first: emitter Fact-import-ban test + plan-totality validator scaffold (a missing plan is a
  verifier error, not a walk-time surprise).
- M1 Evidence-atomic recording: facts/plans/routings recorded at every visit with overwrite (monotone facts ⇒
  last write = final); `_finalize` (`_analyze.py:606-646`) stops replaying `_transfer` — host calls cannot
  repeat. Small, de-risks everything.
- M2 Routing algebra (Codex X6a on the schema doc BEFORE adoption code): one typed record "result cell i :=
  source cell π(i) | const", keyed by plan site/cell NEVER dst (`StorePlace`/`PyStoreAttr` have no dst,
  `_ir.py:393-398`); winners recorded only at final stabilization (final edge set). Then adoption (the four
  id-keyed tables `:307-310`, `subscript_plans`+`route_plans`, both routing CallLowering variants, the two
  emission re-derivations) and same-commit deletion of the five copy routines + four clones (`_emit.py:621-745`).
- M3 Typed dispatch rows: `_expand_call` arms + PyAttr ladder (both much smaller post-trim) → ordered frozen
  first-match row tables; identity comparison (unhashable shadow targets `:2161`); messages verbatim.
- M4 Doctrine single-siting: surviving zero-dim guards, `contains_record` ×5, bool-arith ×4, elementwise
  skeleton ×3 → use-specific consumption ops (post-trim reality).
- M5 EmissionRejection retirement (locations exist from S2.11; corpus pins public classes) — refusals move to
  analysis/residualization, diagnostic-identical.
- M6 Structured origins datatype migration (observable semantics already settled in S2.11) — mechanically
  behavior-identical, last.
- M7 (optional, gate-deferred): if the totalized tables have become a de-facto RIR, materialize the spine as a
  small byte-gated commit with schema learned from contact — decide then with Codex, not now.

### Variant TRANSPLANT (branch `restructure/rebuild`; ~8 milestones)

- R0 Scaffold: `holoso/_frontend/_fir2/` with `_rir.py` datatypes (Codex X6b on the schema first) + A/B
  differential harness (lowers kernels through both frontends; compares `_hirdump` output, diagnostics, and
  full-pipeline Verilog bytes; note `test_fir_differential.py` is a value oracle, NOT this — new machinery).
  Transfer-function bodies lifted near-verbatim from `_analyze.py`; analyzer skeleton rebuilt (evidence-atomic
  recording + dispatch rows from day one).
- R1-R5 Milestones by construct family (scalar straight-line → branches/loops/unroll → state+W/D →
  aggregates/records → arrays/linalg/ports), each gated differential-green for its example subset + frontend
  unit tests via fixture switch.
- R6 Full-corpus differential: 25 golden cases + rejection corpus + extended fuzz A/B on the runner VMs.
- R7 Cutover: flip `holoso/_frontend/__init__.py`, DELETE `_fir` in the same commit (clean break); canonical
  gate + re-freeze; merge to dev; tag `restructure-done`. Branch abandonable at zero cost to dev until R7.
- dev merges into the branch after every baseline-update commit; adjudication: old-frontend-wrong → baseline-
  update protocol on dev; new-frontend-wrong → fix on branch.

## Codex touchpoints

X1 scope dossier (S1.2) · X2 B1 design (before S2.12) · X3 freeze design (before S2.15) · X4 architecture memo
(S3.2) · X5 spike results (S3.4) · X6a/X6b routing/RIR schema (before M2/R0) · X7 standing: the review-loop
correctness reviewer on every step's diff.

## Verification

- Per step: regression-first discipline; `HOLOSO_REGALLOC_EFFORT=10 nox -s tests typecheck black`; push; five
  CI jobs green (core, cosim_examples, fuzz, synth, synth_examples); review-loop clean round.
- Stage gates: Stage 2 exit checklist above; `freeze-1` certified by full CI + rtl-report-equivalence + 3-seed
  byte check; Stage 4 landings gated on goldens + diagnostic pins; cutover additionally on cosim 25/25 +
  schedule metrics + canonical HIR equality.
- End-to-end: `nox -s run_examples` and the example reference/cosim/interpreter oracles remain the behavioral
  ground truth; the numerical-model-vs-Python oracle (`test_example_reference.py`) is the semantic authority.

## Risks (top, with mitigations)

- Freeze captures a wrong behavior → goldens are evidence, not truth; baseline-update protocol is a first-class
  move; Stage 2 concentrates behavior changes before capture.
- CI/runner-VM contention (same self-hosted pool) → one commit per step; throwaway pushes only when a verdict is
  needed; direct ssh runs reserved for certification/debug.
- H1 enablement wave destabilizes CI → determinism first (S2.1); trial push on throwaway branch.
- `_analyze.py` (2,498) grows past ~2,000 during B1 → trims land first; B1 deletes more than it adds; per-commit
  wc gate; helpers to `_analysis_support.py`; never another file split.
- Spike goalpost drift / timebox blowout → criteria + cap maintainer-approved in advance; expiry = morph.
- Review-loop finding floods → skill's own noise rules; architecture re-litigation parked into decision records.
- Codex unavailability → resume discipline; non-gated steps proceed; gated decisions wait (non-negotiable).
- Rebuild divergence tail → milestone family gates localize; morph stays executable as fallback off `freeze-1`.

## Out of scope

LIR/backend beyond the named G1 + shared-live-out fixes; the integer wiring milestone (TODO.md); matmul-on-
array-parameter-ports capability (documented gap, homed in TODO.md at S2.14); restoring `_fuzz.py` tuple lanes
(queued as post-campaign coverage work, noted in S2.14); new trace/outer/dot examples (maintainer, later).

## Critical files

`holoso/_frontend/_fir/_analyze.py` (determinism :228/:616/:643, B1 towers, C6 :1808, F1 :1836, E1 origins,
side tables :304-310, dispatch :1997-2498) · `_emit.py` (C2 :871, B1 deletions :528-604/:574, copy routines
:621-745, EmissionRejection :165-171) · `_build.py` (:148, :162, :535, starred targets :473-529) ·
`_signature.py` (:89, :148, :179) · `_analysis_support.py` (:100-107, :341-388, :544-571) · `_fold.py`
(:182-264) · `_lib/_linalg.py` · `_value.py` · `holoso/_hir/_if_convert.py` (:162) · `holoso/_errors.py` ·
`tests/_examples.py` (registry deletion) · `tests/test_determinism.py` · `tests/test_metrics.py` ·
`tests/test_latency_freeze.py` · new: `tests/golden/`, `tests/test_golden.py`, `tests/_hirdump.py`,
`tools/refreeze_golden.py`, `docs/campaign.md`, `docs/decisions/`.
