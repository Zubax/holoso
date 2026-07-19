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
- Codex reviews run against a PINNED commit in a detached `git worktree`, never the live checkout: the user-level
  Codex config forces a danger-full-access sandbox (`-s read-only` does not take), and a live-tree review once
  reverted a legitimate concurrent amend it mistook for an intruder (recovered via reflog). Corollary: never
  amend a branch a live-tree reviewer is reading.
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
- Speed discipline (maintainer-directed): two-tier verification — mid-step iterations run only the frontend-fast
  corpus (`test_fir_*`, `test_frontend_*`, `test_determinism`, `test_language_features`) at `-n 8` locally (~1
  min; the ⅔-core cap exists to bound P&R memory, which the light suite has none of). The LOCAL full session is
  retired: for low-risk steps (test-only, docs, message-only diagnostics) the throwaway-branch CI push IS the
  full verification; for source-touching or RTL-visible steps the once-per-step full session runs on a runner VM
  over ssh (`user@ci-hans-reiser.local` / `user@ci-terry-davis.local`, 24 cores, ~6-8 min; clone at
  `~/holoso-gate`, run `nox -s tests typecheck black` at the trial commit) — note the HANDOFF hostnames' `cy-`
  spelling is a typo. Everything launches IN PARALLEL with the review round; a reviewer-invalidated run is
  cheap. Exception: steps that can change emitted Verilog (trims, S2.11 port order, S2.12 B1, S2.13 G1) also
  run `test_latency_freeze` + `test_metrics` in the local fast tier — that guard class catches RTL-visible
  hazards pre-commit. Trims delete `match=` pins suite-wide, which the fast tier cannot see; the trial-CI core
  job retains that authority.
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
- R6 Full-corpus differential: 35 golden cases + rejection corpus + extended fuzz A/B on the runner VMs.
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

## Progress log (live; update at every step boundary)

STANDING DIRECTIVE (maintainer): drive the plan to completion without stopping; consult Codex
(gpt-5.6-sol, ultra, resume-not-restart, pinned-worktree reviews) whenever stuck.

Landed on dev, CI-certified: S0 7af064b (bootstrap) · S1 c4e310f (scope ruling + X1) · S2.1 0c4afb6
(deterministic rejections; 2 witnessed fixes + property lock) · S2.2 574c7bf (84-skip triage: 46+2 enabled, 28
re-pinned, 8 relabeled; 4 hollow tests repaired) · S2.3 6badaf9 (A1 wrap-parse, D2 consumption-site annotation
resolution, E2 Fail message parts; 2 review rounds) · S2.4 ea3510e (C3 MRO merge, C4 structural detection, D1
component-position contracts; incident: live-tree Codex reverted an amend → reviews now run on PINNED worktrees).

In flight as a 4-commit stack on trial/s27-getattr-trim awaiting its trial + S2.5's Codex verdict, then dev
advances: S2.5 c91dd33 (E3 store-located state rejections, E4 per-stub gate + reduction arity, F1 threshold
hoist ~29s→0.07s, C2 residual-compare rejections located, A3 flat+reshape matmul, C5 int trace) · S2.6 7bdc727
(C6: unroll reseed; origin-keyed after both reviewers convergently found the block-id livelock; inlined +
nested regression shapes) · S2.7 cb96369 (T1 getattr trim; 8-spelling probe battery clean) · consolidation
058f83e→amended (all three rounds' accepted findings; Codex's classmethod(getattr) corner declined as hostile
construction).

S2.8-S2.10 COMMITTED locally (dev-remote at ea3510e; the targeted poll advances dev to 05e4675 on the
queued run's green, then pushes the trims stack as trial/s2-trims-stack — force the true tip over its 887f7b1
push): a66e4e8 S2.8 (T2 masks kill A2, T3 0-d doors kill C1, S2.5-Codex consolidation) · f3d7dd3 S2.9 (T4-T8,
net -543 lines, S2.8-pair consolidation) · 887f7b1 S2.10 (T9-T11, Fastmath policy section, H3 True-fold pin,
None-__setattr__ presence fix) · 5e1d833 S2.9-round F1 (aliased slot descriptors refuse; both remaining
corners declined on record). S2.10 pair reviewing at 5e1d833. Remaining skips: the three S2.11-blocked rows
only. gh API lesson: never `gh run watch` (quota exhaustion); 5-minute single-call polls.

NEXT: S2.11 (delegated; consolidate its return with the S2.10 pair's): Origin gains file data; primary =
origin[-1]; SynthesisError.location populated; port-order tie fix (TwoChildren reproducer in task #13);
EmissionRejection threading; backend shared-live-out assert -> located; enable the three blocked skips.
Then S2.12 B1 per docs/decisions/b1-design.md (X2 banked), S2.13 G1 (preserve/hoist, budgeted), S2.14 hygiene
(+ the S2.5-deferred rewordings; exit grep), S2.15 freeze (X3; build-sharing constraint), S3 gate, S4.

SESSION-LIMIT CHECKPOINT (resets 6:50am Europe/Tallinn): Claude subagents terminated mid-flight — the S2.10
Claude reviewer (Codex half COMPLETED: one finding, starred-inside-subscript-tuple `v[*[0], :]` draws the
generic message instead of T10's; safe+located; queued small fix in _build's multi-axis subscript arm; all
else clean — byte-identical unpacking, slots shapes match Python, H3 non-vacuous) and the S2.11 implementer
(NO edits made; the full seven-item brief lives in the session task #13 + this file's NEXT paragraph — re-run
it verbatim). The 05e4675 dev-advance poll exhausted on nulls (superseded-run backlog); RESOLUTION: push the
CURRENT tip as one fresh trial run and gate the whole dev advance (ea3510e -> tip) on that single green, then
re-run the S2.10 Claude reviewer post-reset. Codex remains available meanwhile.

QUEUED S2.10-round minors (post-S2.11-agent, same files): DESIGN.md transcendental-fold wording at the Fastmath
section AND ~:598 ("host libm at binary64", not "ideal infinite-precision"); add the __objclass__-in-MRO
conjunct to the component descriptor exemptions (_analysis_support write guard + _analyze read ladder) so a
name-matching foreign member descriptor refuses located instead of crashing raw (Python-identical either way);
single-site the duplicated display-rejection string via the shared starred check. S2.10 round then fully
consolidated (Codex + Claude halves both clean on correctness).

S2.11 COMMITTED 97eeca6 (E1-lite complete; the queued S2.10-round minors landed in the same commit — S2.10
round now fully consolidated, string-dedup declined as cosmetic). All seven items in: Origin carries the file;
origin[-1] primary attribution with "in callee():" context via shared render_rejection; .location populated on
all four rejection classes (linecache source line); all 41 EmissionRejection sites located (state slots via
ResidualUnit.store_origins, return contracts via earliest return store); deterministic state-port order keyed
by (line,column) over the reversed origin stack (TwoChildren fail-before observed: state_second before
state_first; 3-seed stable; zero frozen/metric row changes); backend shared-live-out bare assert is now a
SynthesisError naming both slots; the three blocked skips enabled. Full light suite 1757/3 green, mypy/black
clean, zero pin flips of 388 surveyed. Trial: trial/s2-e1-lite at 97eeca6; survivor poll advances dev on green.
Review pair running on pinned worktree review-s211. NEXT: S2.12 B1 per docs/decisions/b1-design.md (X2 banked:
7-step order; del keeps schema; non-SemType stores fact-only; side tables only; report order = first executable
store in CFG preorder; reversal inventory incl. test_frontend_state.py loop-counter rebinds), overlapping the
S2.11 review round; then S2.13 G1, S2.14 hygiene+exit-grep, S2.15 freeze (X3), S3, S4.

S2.11 round, Claude half returned: NO functional defects; four LOWs ACCEPTED and QUEUED behind the B1 agent
(shared files): (1) render_rejection strips the trailing underscore from ANY callee name, so an honest user
helper named `scale_` (PEP8 shadow-avoidance) renders as `in scale():` — strip only for registry stubs
(discriminate at graft time or by frame file) + regression; (2) TODO.md deferred-gaps heading's
FIR_PARITY_PENDING parenthetical is false (registry is empty and asserted so) and the aggregate-return
paragraph is stale — tuple/list returns emit through full Verilog (probe-verified with ports out_0/out_1[/2]);
(3) SynthesisError.__init__'s location-rendering branch is dead (all four rejection classes set .location
post-init) and SourceLocation.__str__ prints col+1 while rendered messages use 0-based columns — remove the
dead branch and unify the column base after checking pins/consumers; (4) the four byte-identical five-line
rejection __init__ bodies collapse into a shared mixin next to render_rejection. Verified non-findings banked:
origin grafting never mutates templates; port order is seed-stable and source-ordered incl. same-line call
sites; the live-out guard is exact on five lookalikes and closes the former -O silent-shadowing hole; both
__objclass__ ladders intercept cross-class aliases while honest/inherited/nested slots pass. Codex half still
running; B1 agent still running; trial/s2-e1-lite CI + dev-advance poll running.

S2.11 round, Codex half returned (6 findings) — round NOT clean: (1) SIGNIFICANT/ABI: tied origin keys fall
back to block-id order and the unroller allocates clones in reverse iterable order, so `for child in
(self.first, self.second): child.value = x` inside an inlined helper yields [state_second, state_first]
(_analyze.py ~:711) — fix the tie-break (CFG-preorder-like, not raw block id) + regression; this is the
TwoChildren fix's remaining hole. (2) NaN constants and NaN state resets reach _hir/_const.py:24 via frontend
_emit.py :653/:638 and raise a raw UNLOCATED UnsupportedConstruct — wrap located at emission + regression.
(3) backend shared-live-out SynthesisError unlocated — DECLINED as designed: origins are gone by LIR, the
message names both slots, homed in TODO.md with the install-copy gap. (4) the live-out guard groups slots by
final-tap register identity, so time-multiplexed reuse falsely blames an innocent slot (probe: _c ends 5, a/b
end 20, all RegRef(3); message names all three) — discriminate by install step/value identity + regression.
(5) the trailing-underscore strip corrupts `__call__` into "in Power.__call_():" — converges with Claude (1);
one fix covering scale_/__call__/matmul_. (6) a never-returning inlined helper attributes "the function never
returns on any path" to the root def line instead of the actionable site — better origin + regression.
Fix development dispatched to a separate detached worktree (fixes-s211) at 97eeca6 to overlap the B1 agent;
sequencing: B1 verifies+commits first, then the round-fix commit cherry-picks on top, then one combined trial
run + round-2 review pair over the stack.

S2.11-round fixes COMPLETE in worktree fixes-s211: commit 8772297 (parent 97eeca6), 16 files +298/-64, all 8
accepted findings fixed with 9 fail-before-observed regressions; latency/metrics ZERO row changes; mypy/black
clean; targeted batches 311+463 green. Key mechanics: F1 first-store key gains (source_position,
execution_rank[block]) with execution_rank = reverse postorder over executable edges (pure, in
_analysis_support) — only genuine ties can move; F2 NaN refused located inside _carrier_float (both const and
state-reset paths); F3 live-out partnership now requires boundary sampling (not s.needs_copy or
float_state_install_is_boundary), reproducer premise self-asserted; F4 graft rebrands registry-stub frames to
Library.display_name (strip gone from render_rejection; scale_/__call__ render verbatim; .origin now carries
"matmul" not "matmul_" — nothing pinned the raw form); F5 never-returns uses deepest reachable terminator
origin (root-level while True now attributes to the back-edge line — no pin flips); F6 TODO heading honest +
false aggregate-returns paragraph deleted (probe: tuple/list returns emit ~11 kB RTL); F7
SynthesisError.__init__ message-only + SourceLocation.__str__ unified 0-based (no consumers pinned the old);
F8 LocatedRejection MI base in _ir (library rejections stay non-UnsupportedConstruct). DISCOVERY (recorded):
exec-compiled kernels fail lower() with SourceUnavailable — randomized searches must use real files.
INTEGRATION PLAN unchanged: B1 verify+commit first, cherry-pick 8772297, combined gate, one trial push, round-2
pair over the stack, dev advance on green.

S2.12 COMMITTED 4bc6863 (B1 fixed storage schema per the design note; ONE ratified deviation recorded in the
note's addendum: aggregate schemas enforce on persistent state only, locals stay scalar-kind + fact-only for
aggregates — full local enforcement rejected the examples contract. _analyze.py 2531->2448. Full light suite
1781/3 green pre-integration; latency/metrics rows unchanged x3). S2.11-round fixes CHERRY-PICKED as 722d2fa
(one trivial import-list conflict). Combined gate at 722d2fa: 577/1 green over the 12-file union incl.
latency/metrics; mypy 197 files clean; black clean. Trial: trial/s2-b1-stack at 722d2fa; poll advances dev on
green. Round-2 pair (S2.11 round 2 + B1 round 1 merged) reviewing 61077f7..722d2fa on pinned worktree
review-s212. QUEUED for S2.14: test_metrics.py module docstring still claims finite_set_current_controller
"cannot yet lower" (stale; explains its missing frozen guard row). NOTED: B1 double-error corner — a doomed
kernel with a state-store violation plus a downstream rejection can surface the secondary message first; the
int-slot<-float carry rule kills the only suite-exercised class. NEXT after round-2 clean + CI green: S2.13 G1.

Round-2, Claude half returned (4 findings, none high): (1) ACCEPTED/code: B1 regresses mixed-kind static
iterables order-asymmetrically — `for c in (1, 2.5)` rejects ("variable 'c' is an int...") while `(2.5, 1)`
accepts; unratified (X2's Q4 never answered mixed iterables), untested, and the remedy text is unactionable for
a loop target. FIX: establishing-JOIN treatment for loop-target trip stores (same binding site joins kinds with
int->float promotion; both orders accept; c becomes float; bool mixes still reject) — extends the note's
independent-first-def join to same-site trip stores; ratify in b1-design.md addendum + pin both orders + the
comprehension spelling. (2) ACCEPTED/doc: DESIGN.md store-edge-conversion sentence over-promises for locals
(dead `x=1.5; x=2**7000` accepted vs the identical state store rejecting beyond-carrier) — scope the sentence
to state stores; local behavior is the more Python-faithful and stays. (3) ACCEPTED/simplification: four
hand-rolled deterministic walkers — then-arm-first preorder in _reject_executable_fails duplicated by
enforce_storage_schemas (docstring literally pins "matching the Fail walk"), and execution_rank duplicating
_emit's _reverse_postorder — consolidate to shared helpers so lockstep holds by construction. (4) DECLINED:
0-based diagnostic columns — established end-to-end convention, all pins encode it, 1-based flip pre-freeze =
mass churn for a click-through nit; the freeze pins 0-based. Verified-clean list banked (NaN surface closed
incl. folds/resets/globals; obligation id(op) keying safe; violating carries stabilize; F1/F3/F5/F6/F8 all
hold; no orphaned deleted names; StoreRole on all 12 sites). Codex half + G1 agent still running; round-2 fix
batch will go to a side worktree after Codex returns.

Round-2, Codex half returned (5 findings) — round NOT clean; TWO logged stances CORRECTED: (C1, HIGHEST)
float<-int LOCAL stores are admitted with NO store-edge conversion — `current = value; current = 2**53+1;
return current - 2**53` yields 1.0 while the float() spelling yields 0.0 (silent divergence); OVERTURNS the
Claude-half stance "local behavior stays" — the ratified spec demands conversion; fix mirrors
conform_state_store exactly (exact ints convert, non-representable reject on the carrier rule; dead
`x=1.5; x=2**7000` flips to reject — consistent with state). (C2) comprehension-target schemas leak across
separate executions (the schema walk ignores compiler-generated scope-reset UnbindPlace at _build.py:840):
`[float(item) for item in s]` inside an unrolled loop over ((1,),(2.0,)) rejects order-asymmetrically;
UNIFIES with Claude finding 1 (mixed static iterables) under per-execution-scope freshness — schema clears at
unchecked UnbindPlace and loop-target trip stores get per-trip freshness (per-instantiation helper-param
precedent; more Python-faithful than the previously logged establishing-join, which is SUPERSEDED); user del
retention (X2) must be preserved via the checked/unchecked discrimination (verify the flag exists). (C3) an
illegal int-slot<-float store propagates the violating float fact and a downstream secondary rejection
("<< requires integer operands") preempts the causal store diagnostic — schema violations must take causal
priority at resolution (seed-stable, CFG-preorder-first); the B1 agent's noted corner, now live. (C4) HIR
constant folding manufactures FloatConst(NaN) from `math.inf + -math.inf` — raw unlocated crash; fix: the HIR
fold defers NaN-producing folds to runtime (fastmath-doctrine-consistent; RTL adder legitimately yields runtime
NaN; zero blast radius — previously such kernels crashed). (C5) np.dot misuse renders "in matmul():" — the
graft rebrand uses per-Library display_name; thread the SPELLED alias per call ("in dot():"). Batch R1-R7
(these five + walker consolidation + DESIGN truth-up post-R1) dispatched to side worktree fixes-r2 at 722d2fa;
0-based columns decline stands.

S2.13 COMMITTED 6207eb7 (G1: guard blocks fuse via the shared speculatable-within-budget criterion; ops hoist
into the predecessor ahead of band(A,B); bypass-equality/faulting/budget refusals hold; nested spelling now
bit-identical to the hand-written `and`). Frozen rows: ZERO changes VERIFIED by corpus-wide A/B stash
measurement — the corpus contains no fusible natural nested guard (uart_tx reconverges late, uart_rx fails
bypass equality), so the plan's "deliberate row updates" would have been fabrication and were correctly not
made. Full light suite 1797/3 green; mypy/black clean. NOTED for S2.15 re-freeze: pre-existing baseline ceiling
slack (e.g. madd min_ii measured 14 vs frozen 15) exists identically at acf1ba8 — owned by the freeze step, not
G1. G1's review round merges into the next stack round (G1 + round-2 batch) once fixes-r2 returns.

Round-2 batch INTEGRATED: 76d7837 cherry-picked clean as 2bcd878 onto G1. Deviations adjudicated: (a) R4's
model-NaN pin is impossible — the numerical model is ZKF-bit-faithful and ZKF has no NaN (inf + -inf yields
bits 0x0); the regression pins the ZKF-defined result instead, fold-deferral as mandated. (b) R1's exact-or-
reject for locals exposed the state edge still silently ROUNDING (conform_state_store via _float_promoted;
docstring falsely claimed "exactly like the local rule") — RESOLVED toward symmetry as my own integration
commit 5c3d3c5: both scalar slots and aggregate cells now convert exact-or-reject through subject-parameterized
_binary64_store_image; the rounding promotion is merge-only; fail-before observed (both probes ACCEPTED with
9007199254740992.0); the round-1 fold-coherence pin inverts honestly with rounded-fold coverage kept via the
float() spelling. (c) b1-design amended: the schema lattice rides the SCCP env (conversion needs it at the
store) while verdicts stay post-stabilization — ratified. Left-noted: mid-flow non-schema rejections can still
preempt purely LOCAL schema violations (state obligations have causal priority; locals would need non-monotone
mid-flow recording — acceptable, documented). Full light suite over the stack (6207eb7+2bcd878+5c3d3c5) running
in background; on green: trial/s2-g1-stack push at 5c3d3c5, poll, round-3 pair on pinned worktree review-s213.

G1+round-2 stack VERIFIED whole: full light suite 1810/3 green (pipefail-armed rerun after the pipe trap bit a
THIRD time — my own `pytest | tail` chain masked 2 failures AND truncated the failure list; recovered the
second failure's identity from pytest's lastfailed cache; rule hardened: pipefail or file-redirect on every
gate chain). The 2 failures were honest R4 fallout — two old pins asserting rejection of `1e400 - 1e400` fold
EXPRESSIONS, whose "rejection" was the raw unlocated HIR-fold crash R4 removed; reversed as ea83691 (literal
NaN data keeps refusing; fold expressions synthesize; runtime semantics pinned by the R4 regression). Stack for
round 3: 6207eb7 (G1) + 2bcd878 (round-2 batch R1-R6) + 5c3d3c5 (state-edge symmetry) + ea83691 (NaN pin
reversals). Trial: trial/s2-g1-stack at ea83691; poll advances dev on green; round-3 pair on pinned worktree
review-s213.

Round-3, Claude half returned: ONE P1 + two P2s, all one root cause, stack otherwise clean (guard fusion
adversarially verified incl. state-write/leak refusals and a fused-vs-unfused bit-differential; the suspected
1-ulp fusion bug was the ZKF adder's own chartered rounding). (1) P1: _binary64_store_image raises MID-TRANSFER
on transient pre-join facts — swapped branch arms flip accept/reject for `n = 0 / n = 2**53+1; acc = n`
(local AND state twins reproduce); contradicts DESIGN "the verdict is never per-visit", the
enforce_storage_schemas docstring, and the adjacent code comment. (2) P2: DESIGN's causal-priority sentence is
unqualified but implemented for state obligations only — the local twin of R3 reports the secondary shift
error. (3) P2: the mid-flight raise also preempts an earlier-in-preorder violation, breaking the
first-in-preorder contract (control probe confirms the collected path orders correctly). UNIFIED FIX RATIFIED:
binary64 conversion failures never raise in _transfer — they record obligations (inexact carries the rounded
Known image, beyond-carrier carries Residual(FLOAT); both fixpoint-stable) and the post-stabilization walk
resolves ALL schema violations (local rebinds + state obligations + conversion failures) first-in-CFG-preorder,
outranking every deferred rejection — DESIGN's unqualified claims become true rather than scoped down. P4s
queued: dedupe the two first-violation scans into one helper; the 136-col DESIGN line. Codex half still
running; fix batch goes to a side worktree at ea83691 after it returns (S2.14 agent owns the live tree).

Round-3, Codex half returned (5: four P1 + one P2): (K1/P1) the local int->float store-edge conversion is
analyzed but NOT emitted — `current = int(value)` into a float-schema local keeps the int cell kind, downstream
truth selects IntToBool and rejects "integer operator 'float_to_int' is not yet lowerable" while the explicit
float() spelling synthesizes; the emitter must insert the real conversion op at the store edge (state twin to be
checked). (K2/P1) an all-integer phi bypasses exact-or-reject: `2**53+1 if flag else 1` joins to Residual(INT),
stores into float schema unchecked, and strength reduction rounds the constant arms -> silent 9007199254740992.0;
RULING: the boundary is static-vs-runtime — statically Known stores are exact-or-reject, genuinely-runtime ints
convert with hardware round-to-nearest semantics (the NaN precedent: static refuses, runtime defers); DESIGN
documents the boundary, pins cover both spellings, and K1's emitted conversion makes the runtime path real.
(K3/P1) = Claude's P1 (transient-fact mid-transfer raise; swapped-arm flip; else-arm-first state errors) —
convergent. (K4/P1) strength reduction's x/x->1.0 rewrite undoes the NaN-fold deferral for CONSTANT inf/inf
(interning makes operands identical): constant kernel returns 1.0, runtime divider returns ZKF 0.0; guard the
identity rewrites against zero/non-finite constant operands (0.0/0.0 same class). (K5/P2) causal-priority
ranking at a mid-round abort walks the INCOMPLETE graph (LIFO discovers else first): with a secondary shift the
else store reports, without it the then store — ranking must wait for the stabilized graph. UNIFIED FIX BATCH
(X1-X5) dispatched to side worktree fixes-r3 at ea83691: X1 full verdict deferral (no store-edge raises in
_transfer; inexact carries rounded Known, overflow carries Residual(FLOAT); schema-provoked secondaries defer;
rounds stabilize; ONE post-stabilization first-in-preorder resolution over locals+state+conversions outranking
deferred rejections; seed-stable) covering Claude 1/3 + K3/K5; X2 emitted store-edge conversions (K1, locals +
state); X3 the static/runtime boundary docs + pins (K2); X4 identity-rewrite guards (K4); X5 the P4s (dedupe
first-violation scans; 136-col DESIGN line). S2.14 agent still owns the live tree; cherry-pick order decided on
returns.

S2.14 COMMITTED 6b4a421 (hygiene closeout; exit grep ZERO). MAJOR DISCOVERY, probe-proven: the
matmul-on-array-parameter-ports gap NO LONGER EXISTS — imu lowers, LIR matches its frozen metrics row
byte-for-byte, synthesizes 19279 B at II=42, model matches numpy; the imu skip is DELETED and its BASELINE row
ENFORCED; DESIGN affirms the capability (the plan's mandated "open gap" wording would have been false). Guard
table re-keyed per (name, format): 15 rows value-unchanged (uart at true e4m8), octave_index pinned e6m18=(14,38)
AND new-measured e8m36=(14,47) — the only new frozen value. H1 machinery fully deleted (registry, parity_marks,
vacuous test, 9 consumers, SynthTarget.example + tools/synth_compare.py, all token sites). S2.5-deferred
rewordings landed with pins (list numpy-isms -> np.array guidance; runtime-subscript split index/scalar). FSCC
metrics row omitted honestly (needs fsort/fsincos beyond shared default_ops; docstring records it). Suite
1811/2, mypy 197, black clean. INTEGRATION NOTE: DESIGN's causal-priority sentence is deliberately scoped to
state-only in this commit; the X-batch cherry-pick MUST restore the unqualified wording (its X1 makes it true)
— expect a DESIGN.md conflict there and resolve accordingly. Remaining in flight: X1-X5 batch (fixes-r3),
Codex X3 freeze consult. NEXT after X-batch integration + round-4 clean: S2.15 freeze.

Consult X3 returned: five positions, ALL CHANGE-recommended, ALL ADOPTED — ratified in
docs/decisions/freeze-design.md, which supersedes section F where they differ. Headlines: structured JSONL
diagnostics (payload incl. location line text + origin frames + competing-error precedence); append-only
DIRECTORY of immutable per-case rejection modules; ONE typed GoldenCase catalogue absorbing the latency/metrics
example rows (the (example, format) key alone is WRONG — ekf1 ships two reset variants; xdist sharing becomes
automatic); seed-MATRIX determinism certification in the refreeze tool (8+ full-corpus seeds, 0-63 targeted,
same-seed process repeats, abort-before-write); and the missing load-bearing contracts — complete versioned HIR
serializer, a real block-and-value alpha-canonicalizer for the canonical landing (renumber() compacts blocks
only) plus independent Python-reference/MIR-interpreter gates there (cosim's model derives from the same LIR),
exact per-case metrics, structural-only cases for FSCC/vector-polar/imu/shipped-EKF, deliberate format policy,
full ABI manifest, version-token canonicalization + holoso_support.v freeze, provenance capture, and a
bijection-checked corpus index. S2.15 implements per that doc. Still in flight: X1-X5 batch (fixes-r3).

X-batch INTEGRATED: 96eac73 cherry-picked clean as 19639f5 onto S2.14; the flagged DESIGN semantic conflict
materialized exactly as predicted (S2.14's scoped causal-priority sentence survived textually) and is unified
as 8c60797 — the unqualified claim now covers local rebinds, state obligations, and conversion failures alike,
outranking every deferred rejection including user raises (the X-batch's justified extension; schema walk runs
before the Fail walk). X-batch highlights banked: terminator-level deferral (adversarial probe: range()
trip-count rejection was preempting the causal store); X2's state twin did NOT reproduce (state stores already
coerce via _slot_kind) — pinned anyway; X4 audit proved x*0 and 0/y ZKF-identical (zero kills unconditionally,
keep folding) while x/x and neg-cancellation defer on known zero/non-finite operands. Combined full light suite
at 8c60797: 1825/2 green (pipefail-armed), mypy 197 clean, black clean. Trial: trial/s2-prefreeze-stack at
8c60797; poll advances dev on green; round-4 pair on pinned worktree review-s214 over 6b4a421+19639f5+8c60797.
A clean round 4 opens S2.15 per docs/decisions/freeze-design.md.

Round-4, Codex half returned (4 findings, all deferral-net refinement gaps in 19639f5): (1) successor-env JOIN
rejections escape the deferral handlers — "irreconcilable kinds merge here" preempts a pending state-store
violation (the pre-X-batch worklist-wide handler got this right); (2) the net catches AnalysisRejection only —
LibraryAnalysisRejection is a SIBLING via the MI mixin, so a math.gamma() rejection preempts the causal store;
catch the shared LocatedRejection mixin (Build/Emission variants cannot fire there); (3) the verdict walk
re-derives local exactness from TRANSIENT round-one facts at the W/D-abort path — a Known-int that stabilizes
to runtime falsely rejects; stable-fact re-derivation only on stabilized rounds, abort paths rank recorded
obligations alone; (4) state obligations are wholesale-cleared between W/D rounds, losing the causal store to a
round-two induced shift error — persist per-op with clean-revisit clearing (the C6/S2.11 pattern). Doctrine to
hold in the fix: verdicts from stable facts only; recorded violations outrank provoked errors from ANY layer
(op, terminator, join, library); obligations persist across rounds; seed-stable. Claude half still running;
consolidated round-4 fix batch goes to a side worktree at 8c60797 (the S2.15 agent owns the live tree; the
freeze capture regenerates after these fixes land — one tool run).

Round-4, Claude half returned (2): (1) NEW — conform_state_store's AGGREGATE arm classifies a
deferral-cascade Unbound as "a scalar" ("persists an aggregate; a scalar cannot be stored into it" on innocent
code), a false verdict that then outranks the genuine causal violation; the scalar arm has the correct
neither-establish-nor-violate escape, the aggregate arm lacks it. (2) CONVERGES with Codex (1): successor-env
join rejections escape the net; with Codex's evidence that the pre-batch handler prioritized correctly, the fix
is DEFER THE JOIN (DESIGN stays unqualified), not a re-scope. Clean list banked: X4 bit-exact vs ZKF across
inf-inf/0*inf/(-inf)/(-inf)/x-x arms (the add-cancellation guard is load-bearing for SUB-as-add+neg); 2**70 phi
arm converts bit-exact; 10**400 phi arm rejects gracefully; _UnrollRestart not swallowed; id(op) keying stable;
6b4a421 hygiene verified through and through. CONSOLIDATED ROUND-4 BATCH (Y1-Y5) to side worktree fixes-r4 at
8c60797: Y1 joins enter the deferral net; Y2 the net catches the LocatedRejection mixin (sibling
LibraryAnalysisRejection escaped); Y3 verdict re-derivation from stable facts only (abort paths rank recorded
obligations alone); Y4 obligations persist across W/D rounds with per-op clean-revisit clearing (the C6
pattern); Y5 the aggregate-arm Unbound escape. SEQUENCING: commit the freeze infrastructure when its agent
returns, cherry-pick Y-batch on top, REGENERATE the capture (one tool run; expected diff = diagnostics corpus
only), then trial + round-5 + CI + seed matrix + tag freeze-1.

S2.15 FREEZE COMMITTED 557fc69 (160 files, +43575): 35-case GoldenCase catalogue (25 inherited spec-format rows
derived mechanically; 5 structural-only incl. FSCC/vector-polar/imu/shipped-EKF — shipped vs cosim EKF provably
distinct artifacts 127/127 vs 125/125; format probe e6m18/e8m24/e8m36/e11m53; one deeply staged config);
complete schema-versioned HIR serializer with a 14-dataclass field-set completeness guard; tests/golden/ 148
files 1.88 MB (canonicalized version token; exact 7-metric ABI manifests; 32 immutable rejection modules; 7
JSONL diagnostic families; bijection-checked index; provenance); test_golden gates it in the light suite; all
17 frozen-schedule and 18 baseline rows migrated bit-for-bit (nothing vanished — full disposition in the agent
report); refreeze tool with temp-tree generation + --write + --check-determinism. SEED MATRIX CERTIFIED: 8
full-corpus seeds + 3 fresh-process repeats + 8 witnesses x seeds 0-63, ALL byte-identical (270 s); regenerated
tree matches the capture 148/148. Suite 1859/2; mypy 201; black clean. Left-noted: C6/B1-accepting outcomes
live in their behavior tests (a rejection corpus cannot pin accepts); container_digest null by design. AWAITING
Y-batch (fixes-r4) -> cherry-pick -> capture REGENERATION (expected diff: diagnostics rows the Y-fixes change)
-> trial push -> round-5 pair -> CI green -> tag freeze-1.

Y-batch INTEGRATED: cbd1ec0 cherry-picked clean as 793a82b onto the freeze. Y-batch highlights: Y3 resolved by
ELIMINATING pre-stabilization aborts (a failing state join freezes that leaf's live-in at its last joinable
value; rounds always stabilize; the failure reports DeferredRejection-least on the stable branch after the
resolution walk) — the only shape keeping "verdicts from stable facts only" AND the unqualified causal-priority
sentence; Y4 two-layer keying (per-op-id within a round, origin-keyed across boundaries — clones of one store
share an origin, so pure origin keying would let a clean trip erase a sibling's violation); Y2 catches the
LocatedRejection mixin with the callee-BuildRejection-rewraps understanding asserted; six schema regressions +
two seed-parametrized determinism locks, all fail-before-observed. REFREEZE VERIFIED: regenerated corpus
byte-identical 148/148 — the Y-fixes change only multi-error shapes absent from the corpus kernels; no
regeneration commit needed. Full light suite over the integrated tree running; on green: trial/s2-freeze-stack
push at 793a82b, poll, round-5 pair (focus: freeze infrastructure code + Y-batch, not generated corpus bytes),
CI green -> tag freeze-1 -> Stage 2 CLOSED -> S3 architecture gate opens.

Round-5, Claude half returned (7): (1) MEDIUM analyzer — Y4's origin-keyed carryover pops on ANY store at the
origin, so a conforming unroll clone erases a violating sibling's obligation mid-round and a provoked rejection
lands in the window (`for v in range(2): self.n = x if v == 1 else 0` + a provoked shift → reports the shift;
single-clone/violate-first/no-provocation controls all correct; stably wrong across seeds); fix at the round
boundary (drop a carryover only if its origin saw NO still-pending violation across the whole round) +
regression. (2) LOW-MED freeze — the serializer completeness guard omits FloatMulPow2/Float+IntRelational and
the three Type classes (a future field on them vanishes silently, contradicting the guard's promise); add all
six to _SERIALIZED_FIELDS. (3) LOW-MED freeze — fround/ffma configured in NO golden case (null in all 35 ABIs);
FMA contraction and round() lowering are config-dependent codegen with zero byte coverage; add one probe case
with both configured. (4) LOW — HOLOSO_IFCONV_MAX_OPS unpinned by the gate (false-fail only; refreeze children
scrub it); pin it alongside the regalloc knobs. (5) LOW — replace_corpus is rmtree-then-copytree (a mid-copy
failure can lose both trees); copy-aside-and-rename. (6) LOW — the two new Y-batch determinism witnesses are
absent from the refreeze WITNESS_ENTRIES 64-seed sweep; append. (7) LOW — README promises CI fills
container_digest but no fill mechanism exists; align. Corpus verified clean otherwise (rows bit-for-bit;
recompute byte-identical; EKF pair materially distinct; identity/bijection guards hold). Codex half still
running; consolidated fix batch after it returns.

Consult X4 returned (6 positions: 5 CHANGE + 1 AGREE-with-refinement) — ALL ADOPTED: (1) memo factual
corrections (42 not 43 raise sites; StoreRole fixed at CONSTRUCTION incl. analyzer grafts; corrected routing
extents :631-651/:941-951/:968-972; "byte identity is NOT GUARANTEED" not "cannot survive"; the memo's SC1
line still said after-renumber contradicting its own X3 amendment; e3d5f18 is NOT seed-matrix-certified — the
matrix belongs to 557fc69, post-Y has only the 148/148 refreeze; DESIGN :150-153/:285-294 overclaim
analyzer-owned decisions; R6 case count fixed 25->35 in this commit). (2) VERDICT REFINED: morph the stabilized
analyzer but make the resolved spine MATERIAL — materialized-RIR boundary MORE justified, analyzer transplant
LESS (B1/store_conversions prove in-place migration works); the spike's substantive question is whether the
spine closes without Fact backchannels. (3) exact witness identities (ekf1_stateful-e8m36 [+shipped optional],
finite_set_current_controller-e8m36, iir1_hpf-e8m36, format_probe-e8m36 for branch/merge, iir1_lpf-e8m36 for
return-onto-state-port dedup) + a transpose-routing witness (imu-e8m36 or routed-diamond microkernel) + an
accepted store-conversion witness + a NEGATIVE policy packet (legacy_power_chain, legacy_beyond_carrier,
legacy_shared_live_out, one return-contract mismatch — success-only subsets cannot prove refusals moved
upstream) + a witness-to-surface matrix. (4) executable SC1-SC4 (SC1 both-entry-point golden comparisons +
real block-and-value canonicalizer with permutation tests + independent gates wired to the PROTOTYPE path;
SC2 transitive dependency closure; SC3 vocabulary frozen BEFORE the timer + recursive closure verifier + any
new decision-bearing variant = objective fail; SC4 manifest-based nonblank LOC + prelisted inventory +
declared formula). (5) input-topology-aware EXHAUSTIVE decision table, MORPH default: adapter-over-ResidualUnit
passing -> MORPH with MANDATORY materialized spine (M7 mandatory; M0 import ban becomes a ratcheting
allowlist); independent-residualizer passing byte -> TRANSPLANT byte; independent passing
canonical+independent-semantics -> TRANSPLANT canonical; anything else/expiry -> MORPH. (6) mandatory
append-only evidence ledger as a spike artifact (schema growth, backchannel audit, coverage matrix, mismatch
classification, canonicalizer cost, LOC by role, replay counts, diagnostic parity, runtime, provenance).
Doc agent dispatched: memo corrections + DESIGN overclaim fix + docs/decisions/arch-spike.md (the amended
spike spec); spike launches on its worktree branch after that lands. Codex round-5 half still running.

Round-5, Codex half returned (8) — consolidated with Claude's 7: ANALYZER CLUSTER (Codex 1-4 + Claude 1, one
family): the Y4 carryover pops on Unbound "stores" (vs the neither-establish-nor-violate doctrine), raises
stale round-one messages un-revalidated on the stable branch (dual of Y3), collides unroll clones (convergent
with Claude's MEDIUM), and lets current-round violations outrank source-earlier carried ones (checked before,
rank-less). REDESIGN RATIFIED — carryover as PENDING-BRIDGE ONLY: per-op violation statuses fold into the
bridge at ROUND BOUNDARIES (never popped mid-round; Unbound execution is not conforming); the bridge only
keeps the deferral net closed; verdicts come EXCLUSIVELY from the stable round (walk re-derivation + obligations
re-recorded by stores actually executing in the stable graph); carried entries are DISCARDED at stabilization
(dead/obsolete violations vanish legitimately; the deferred secondary then surfaces). All five reproducers
become regressions. FREEZE HARDENINGS: gate compares the RETURNED verilog_output.support_files (not a separate
generator call); build_artifacts calls make_ops() ONCE (ABI/RTL identity split); _hirdump formats big IntConst
digit-limit-safe (hex) + adds the 6 missing classes to the completeness guard (serializer schema version
bumps -> all HIR dumps regenerate, one refreeze run); GoldenCase/gate pins HOLOSO_IFCONV_MAX_OPS alongside the
regalloc knobs (Codex demonstrated different RTL hashes under budget 8 vs 0 with identical identity); fround/
ffma probe case added; replace_corpus becomes copy-aside-and-rename; Y-batch witnesses appended to
WITNESS_ENTRIES; README provenance promise aligned. Batch dispatched to side worktree fixes-r5 at 793a82b.

S3.2 COMMITTED 4f1dd4c (memo amended per X4 — counts/extents/byte-identity/SC1/provenance; DESIGN overclaims
reworded; docs/decisions/arch-spike.md carries the executable spec with the frozen-vocabulary draft — every
emitter decision family homed or excluded, every ResidualUnit field dispositioned; deviations recorded incl.
exact ids, the both-prototypes precedence sentence, and pre-baselined microkernels for the arms the corpus
cannot pin). Amended HTML delivered to the maintainer. SPIKE LAUNCHED (S3.3) on branch spike/resolved-ir at
4f1dd4c in its own worktree: primary adapter-shape prototype (ResidualUnit -> materialized RIR -> mechanical
emitter), schema frozen as the first commit, evidence ledger append-only, SC1-SC4 executable, witness order
iir1_lpf first; the adapter can reach only the MORPH rows of the table — transplant would need the optional
independent residualizer. In flight: spike + the round-5 fix batch (fixes-r5; carryover pending-bridge redesign
+ 8 freeze hardenings + capture regen). Tag freeze-1 after round-5 lands and round-6 is clean.

WORKFLOW DIRECTIVE (maintainer, 2026-07-18): no longer time-constrained — prefer SEQUENTIAL execution over
parallel tracks to reduce management complexity. Effective immediately: the two in-flight agents (round-5 fix
batch on fixes-r5; the S3.3 spike on spike/resolved-ir) run to completion, but no new work launches alongside
them. From here on, one thing at a time: implement -> integrate -> verify -> trial -> review round (the
Claude+Codex pair within a round stays, per the review-loop skill) -> consolidate -> only then the next step.
No more overlapping the next implementation with an open review round; consults run standalone. Order after
the in-flight agents return: (1) round-5 integration -> round-6 review -> consolidate -> tag freeze-1;
(2) spike results -> X5 consult -> ruling -> arch-ruling.md; (3) S4 per ruling, strictly stepwise.

Round-5 batch INTEGRATED: 3767e22 cherry-picked clean as f1f977f (targeted gate 157 green; mypy 201; black
clean; delta from the batch's fully-verified tree is docs-only). Batch highlights: pending-bridge exactly per
the ratified design (StoreVerdict per-op, bound executions only; bridge reconciles at round boundaries, never
popped mid-round, never a verdict source; re-attached at stabilization only to stores that executed without a
bound verdict, at their own preorder rank); two forced deltas documented (transfer-deferral pick is now
executable-preorder — the old lexicographic min compared location-prefixed strings; the StaticFor->Jump splice
moved after the rejection walks to preserve deferral key identity); all five reproducers now report the causal
diagnostic with fail-befores observed via stash. Freeze hardenings B1-B8 all in; fma_round_probe-e8m36 case
added (holoso_ffma/holoso_fround instantiated, contraction verified); refreeze diff surgically verified: 35
hir header-line-only changes + the new case, ZERO drift in verilog/abi/diagnostics/support. Full suite at the
batch tree 1877/2; heavy seed matrix deferred to the freeze-1 certification run. Trial: trial/s2-r5-stack at
f1f977f; round-6 pair follows; tag freeze-1 on clean+green. Spike results HELD for X5 until this closes
(sequential directive).

Round-6, Codex half returned (5): (1) a stale bridge verdict is SELF-FULFILLING — its pendency defers the real
stable rejection, keeps its own store Unbound, then the stabilization re-attach restores the stale verdict
(implicit reports "not exactly representable" where the stable truth is the range() rejection); (2) same-origin
clone verdicts collapse into one message reattached at the first clone's rank (aggregate-arm message overwrites
the earlier float-to-int one); (3) "last BOUND execution" per CFG op hides a later violating execution of the
SAME op in a while loop (conforming first visit retains StoreVerdict(None)); (4) _UnrollRestart reconciles a
PARTIAL round — a conformed clone clears the shared bridge before its violating sibling re-records, and the
restarted round reports the secondary shift; (5) refreeze replace retry deletes BOTH recovery trees before the
new copy lands (interrupted-retry leaves no corpus at all). RESOLUTION DIRECTION (to round-7 for
adjudication): drop re-attach entirely — the bridge becomes a mid-flight net-keeper only (populated by bound
violating executions; reconciled at TRUE round boundaries, restarts carry it unchanged; never a verdict
source); the stabilization walk re-derives STATE-store verdicts from stable envs exactly as it does locals
(state-store sites evaluated on the fixpoint env fact — joins make same-op multi-execution deterministic);
non-re-derivable entries expire and the deferred rejection surfaces preorder-first. CONSEQUENCE: the round-4
Cascaded expectation honestly REVERSES (producer rejection reports first — sequential error reporting; the
store violation surfaces after the user fixes the producer) — flagged for the round-7 pair. Claude half still
running; one consolidated batch after it returns.

Round-6, Claude half returned (5) — consolidation of both halves complete, REFINED RULE replacing the earlier
logged direction: at stabilization, a bridge origin ABSENT from every executable block's stores (STRANDED —
its own violation's cascade removed the block) REPORTS (ranked after in-graph violations, before deferrals,
lexicographic among themselves); an origin whose store executed ONLY-UNBOUND in a live block EXPIRES (the
deferral that caused unboundness is the true stable rejection); bound executions re-derive normally. This
separates Codex-1 (stale, expire -> range()) from Claude-2 (stranded, report the store — the parent behavior).
Full batch: boundary reconcile pops an origin only if every execution was bound-and-conforming AND none was
unbound (Claude 1/Codex 3 — unbound executions block the pop; violating-wins-within-op for same-op
multi-execution); _UnrollRestart carries the bridge UNCHANGED (only true round boundaries reconcile — Codex 4);
stranded-entry messages fold earliest-first (Codex 2); round-end rejection selection UNIFIES to executable
preorder everywhere if pin flips are few (Claude 3 — min-by-str vs preorder inconsistency; else document
honestly); _render_fail gets the digit-cap-safe spelling (Claude 4); the ifconv pin becomes a named catalogue
constant shared with refreeze (Claude 5); retry-safe corpus replacement (Codex 5). Round-7 adjudicates BOTH
reversals (Cascaded; any preorder unification flips). Batch to worktree fixes-r6 at f1f977f.

Round-6 batch INTEGRATED: 5f44e4e cherry-picked clean as 1273f7f (targeted 244 green; mypy 201; black clean;
batch's own full suite 1889/2; refreeze 151/0/0/0 — zero corpus drift). Batch deltas of note: A4 implemented as
last-bound-wins + unbound-origin exemption (literal violating-wins PROVABLY collided with the arm-order charter
— ElseArm would reject where the pinned behavior is compile-with-conversion); A6 took path (b) — the
always-defer unification experiment was disqualified in kind (2 invariant crashes + 1 silent acceptance among
4 flips), dual selection paths kept and documented truthfully + a new seed-stability witness. THREE reversals
flagged for round-7 adjudication: (1) Cascaded now reports the preorder-first PROVOKED SHIFT (not even the
float() call); (2) the round-5 stale-exactness kernel REVERSED BACK — its store strands behind the dead loop
head so the transient exactness message resurfaces and the Implicit/Explicit equivalence splits (the
stranded-vs-stale graph proxy trades this corner for Claude-2's — TAINT/provocation tracking is the true
discriminator both misses); (3) source-earlier-wins flips where surviving in-graph violations outrank expired
stale entries. CONSOLIDATION QUESTION queued for after round-7: whether to accept diagnostic selection among
multiple real errors in doomed kernels as explicitly best-effort (guarantees: some located rejection always;
seed-stable; single-error kernels exact; provoked secondaries never outrank an in-graph violation) rather than
chase perfect causal attribution — three rounds of corner-trading suggest diminishing returns. _analyze.py at
2631 lines (soft limit exceeded) — split queued. Trial: trial/s2-r6-stack at 1273f7f; round-7 pair follows;
tag freeze-1 on clean+green.

Round-7, Codex half returned — ADJUDICATION AGAINST the graph-shape proxy, three P1 + one P2: (1) P1 SILENT
ACCEPTANCE: _expand_call removes the PyCall from the CFG before argument validation; a transient violation
defers the arity error, last-bound-wins clears the violation, the error is now keyed to a REMOVED op and the
ownerless-discard drops it — an ordinary extra-argument typo COMPILES with the call absent from hardware; fix:
argument binding completes BEFORE destructive grafting, and live ownerless entries are never dropped (re-key to
the call origin). (2) P1: the stale-expiry rule blames the innocent shift in Cascaded — `self.count << 1` is
VALID per the int reset; only the violation's propagated float breaks it; the documented provocation doctrine
is violated. (3) P1: the stranded rule resurrects a FALSE transient verdict — the range() rejection is
INDEPENDENT (depends only on k) and the store line is INNOCENT under stable facts. (4) P2: the new Fail
renderer changes valid messages and is non-recursively digit-unsafe. RESOLUTION RATIFIED — TAINT tracking, the
discriminator all seven adjudicated shapes agree on: a violating store taints its carried fact; taint
propagates through transfer/joins/live-ins; deferred rejections record their taint; at stabilization the
report is preorder-first among {stable in-graph violations} ∪ {stranded/stale violations with at least one
taint-victim}; else preorder-first UNTAINTED deferral; else any remaining. Cascaded -> store (tainted shift
loses); stale-exactness -> range() (untainted independent wins; the transient verdict has no stable support);
the &-kernel -> store (stranded WITH taint-victims); Y4 -> store; the arity typo -> the untainted arity error.
The queued best-effort framing is OFF — Codex proved the corners are the serious innocent-line class. One
batch after Claude's half: taint + pre-graft validation/ownerless re-keying + recursive digit-safe renderer.

Round-7, Claude half returned — ADJUDICATION SPLIT RESOLVED IN CLAUDE'S FAVOR; the taint ratification in the
previous entry is SUPERSEDED (no taint machinery). Claude's standard: "innocent" is judged against
REFERENCE-EXECUTION reality — and under it both disputed shapes are truthful: Cascaded's shift line is where
plain Python dies on transaction 2 (0.0 << 1 -> TypeError at that exact line; the message names the poisoned
attribute); the stale-exactness store carries a REAL transaction-1 precision loss at k=0 and the two reports
form a principled two-step staircase (fix exactness -> surface the range refusal). All three reversals
ACCEPTED; the epistemic ladder (fresh stable testimony > bridged-only-where-no-fresh-possible > provoked
deferrals) stands; the misleading-line class is empty under six-direction adversarial probing; remaining warts
are seed-stable selection among true errors, witness-locked. SURVIVING WORK (one batch, fixes-r7 at 1273f7f):
F1 (Codex P1, REAL — a mechanism Claude's stranding probe could not reach): _expand_call validates arguments
BEFORE destructive grafting and live ownerless deferrals are never dropped (re-key to the call origin) — the
extra-arg typo must reject in both spellings (fail-before: implicit ACCEPTED, call absent from hardware).
F2 (Codex P2): _render_fail must not alter messages that format fine (decimal when possible, hex only above
the digit cap) and must be digit-safe RECURSIVELY (nested containers). F3 (Claude, pre-freeze): the
stranded-sibling sort key flips to (source_position, message) — the sole position-non-primary rule; corpus not
yet frozen so now is the moment. Round-8 after; tag freeze-1 on clean+green.

Round-7 batch INTEGRATED: c3d92ea cherry-picked clean as 45ea961 (targeted 318 green; mypy 201; black clean;
batch's own full suite 1894/2; refreeze 151/0/0/0 twice). F1 restructures _expand_call to validate the full
binding BEFORE any CFG mutation (per-param source plan consumed by the graft; serial/op allocation preserved —
accepted kernels byte-identical), re-keys graft-destroyed deferrals to the continuation, and replaces the blind
ownerless discard with a graph-anchored assert; fail-before: the extra-arg typo COMPILED (10255 B of Verilog,
call absent) behind "discarding 1 ownerless transfer deferral(s)". F2 render_interpolation: decimal up to a
conservative 4000-digit bound (bit-length estimate erring toward hex), hex above, recursive through
lists/tuples/ranges/slices; 65-bit stays decimal, nested wide ints locate instead of crashing. F3 stranded
siblings report in source order (the sole position-non-primary rule is gone); no existing pin needed flipping
(only docstring+DESIGN stated the old order). r6 stack certified on CI meanwhile (dev at b21da6f). Trial:
trial/s2-r7-stack at 45ea961; round-8 pair follows; tag freeze-1 on clean+green.

Round-8 returned (both halves) — F1's restructure VERIFIED CLEAN (14-kernel battery outcome- and
location-identical vs a validated parent replica; accepted-kernel dumps byte-identical; every key-destruction
path of the anchored assert covered). Four corner items consolidate into the final hardening batch (fixes-r8
at 45ea961): (1) P2 convergent — the dataclass-record renderer arm (as_python reconstructs records; format()
falls through to the generated repr and the raw digit-cap ValueError escapes) + regression; (2) P2 Claude,
PRE-EXISTING but now a supported flow — a deferred-then-grafted call leaves the OLD terminator's edges in
executable_edges (phantom edge into the stable result): accepted kernel crashes emission ("read of an
undefined place ... escaped analysis") or surfaces a nonsense located rejection at an innocent line, and the
no-violation control REJECTS while the violation variant accepts-then-crashes — retract the grafted block's
stale out-edges at graft time + both probe kernels as regressions + a DESIGN note; (3) P3 — derive the decimal
threshold from sys.get_int_max_str_digits() (a runtime-lowered cap resurfaces the ValueError); (4) P4 — delete
the observationally-inert nine-line re-key block for a bare pop (the caller's pop already guarantees the
claimed property; the anchored assert stays). After the batch: one light round-9 pair on the small diff,
then tag freeze-1 on clean+green.
