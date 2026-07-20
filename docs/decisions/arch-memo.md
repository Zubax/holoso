# S3.1 — Architecture evaluation memo: the resolved-plan boundary and the Stage-4 gate

Status: Stage-3 deliverable per `docs/campaign.md` (S3.1), amended per Codex consult X4 (S3.2, all positions
adopted; see the campaign log). Consolidates the pre-campaign evaluation (two adversarial design agents plus
code-level verification; recorded in `docs/campaign.md` and `HANDOFF.md`), refreshed against the
post-stabilization tree at dev = e3d5f18. Freeze provenance, precisely: the freeze corpus is committed (S2.15)
and seed-matrix-certified at the freeze commit 557fc69; the post-Y tip e3d5f18 has passed only the
byte-identical 148/148 refreeze check (the seed matrix has not been rerun there), and CI certification with the
`freeze-1` tag is still pending — the tip is not seed-matrix-certified. The X4-amended executable spike
specification is `docs/decisions/arch-spike.md`; it supersedes section 4 where they differ. Line numbers refer
to e3d5f18.

## 1. The question, and what the evaluation established

The question as originally posed: should the frontend gain a bytecode-like IR sublayer — Python lowered to a
bytecode-like form first, analysis after — to replace the per-construct lowering?

The verdict, reached independently by both design agents through direct code contact and consistent with the two
prior panels (HANDOFF "The architecture verdict"):

- The pre-analysis bytecode layer already exists. `_build.py` emits an 18-op/5-terminator non-SSA CFG over
  mutable Places, purely syntax-directed: chained comparisons, and/or, ifexp, aug-assign, unpacking, slices, and
  comprehensions all desugar there. Verified current: the `Op` union at `_ir.py:324-343` counts exactly 18 ops,
  the `Terminator` union at `_ir.py:405` exactly 5. Every remaining delta from real CPython bytecode is negative
  (StaticFor is richer than FOR_ITER), impossible without analysis facts (kwargs binding, attribute meaning,
  starred flattening, inlining frames), or cosmetic. A literal pre-analysis bytecode layer is dead; nobody
  advocates it.
- The real boundary defect is downstream: emission re-deciding semantics. At evaluation time: 41 unlocated
  EmissionRejections, `_fact_sem`/`_leaf_is_int` kind re-classification, re-derived routings
  (then `_emit.py:621-641`/`:954-964`). Three of the four surviving major defects — the A2/B1/C1/C2 habitat —
  shared this single structural cause, which is the whole argument for acting at all.
- The cure is post-stabilization resolution totality: every decision made once, in the analyzer/residualizer,
  with emission consuming a closed typed plan surface. The agreed redefinition of "mechanical": the emitter
  imports no Fact, no registry, no Py* op; raises no rejection; makes no kind decision. Braun sealed-block SSA
  and the `_exit_identity` port dedup remain legitimate value-level emitter work — SSA construction is value
  bookkeeping, not semantics.
- HANDOFF's one-sentence verdict stands: the semantic distinctions are essential in cardinality; their
  cross-layer duplication is incidental in multiplicity. The pipeline
  (AST → generic FIR → SCCP/W-D → typed resolved plan → mechanical HIR) is not replaced; the new element is the
  middle box.

The one open decision — morph the plan surface in place, or transplant onto a materialized resolved IR — is what
the Stage-3 gate settles by spike (section 4). Failure default is morph.

## 2. Where the tree stands now — the refresh

The evaluation predates Stage 2; a substantial part of resolution totality has since landed. Each item below is
verified in the current code, not quoted from the plan.

Already moved:

- The plan surface exists and is typed. `ResidualUnit` (`_analyze.py:315-336`) is the analyzer's single output:
  binding_facts, call_plans, subscript_plans, route_plans, store_order, runtime_state, state_livein,
  state_resets, provenance, store_origins, store_conversions. Its docstring pins the contract: emission consumes
  only this — it never re-derives a fold, never resolves the library registry, never replays the transfer.
- S2.11: `ResidualUnit.store_origins` threads every state leaf's first-store origin to emission
  (recorded `_analyze.py:724-731`, consumed `_emit.py:552-553`), and every emission refusal is located:
  EmissionRejection mixes LocatedRejection (`_emit.py:170`), the emitter keeps an attribution cursor
  (`_emit.py:337-340`, advanced per op at `:819`), and all 42 raise sites raise located (was: 41, all
  unlocated; grep counts 43 `EmissionRejection` hits — the 43rd is the class declaration at `:170`).
- S2.12/B1 killed the fact-kind/cell-kind incoherence: stores are schema-checked. StoreRole is fixed at
  construction (`_ir.py:171-190`) — construction, not build: the analyzer's call grafting also constructs
  SOURCE stores (`_analyze.py:2540`, `:2578`); the schema algebra lives in `_analysis_support.py`
  (ScalarSchema/ContradictorySchema `:497-511`, `join_schemas` `:529`, `conform_local_store` `:586`,
  `conform_state_store` `:626`, and the one resolution site `enforce_storage_schemas` `:707-758`). The emitter
  kind towers are gone — `_leaf_kind`, `_leaf_is_int`, `_admit_state_store`, and the live-in state typing /
  stored-value dtype rebuild are all grep-absent — and `_slot_kind` (`_emit.py:582-587`) reads the reset
  snapshot whose kind the schema pinned.
- The X-batch's store_conversions is a working example of exactly the target pattern: the post-stabilization
  schema walk returns a typed plan — the set of store ops whose value converts int→float on the store edge
  (`_analysis_support.py:707-758`) — and the emitter consumes it blind (`_emit.py:840-844`). One decision, made
  once, at the right layer, consumed as data.
- The deferral net centralizes verdict ordering: schema violations (local rebind, state obligation, conversion
  failure alike) are recorded as obligations that persist across W/D rounds, resolve post-stabilization
  first-in-CFG-preorder, and outrank every deferred rejection from any layer — operator transfers, library
  refusals, environment joins, the state live-in join (`_analyze.py:374-382`, `:391-509`; doctrine in DESIGN.md
  "Storage typing"). Verdicts derive from stabilized facts only; no verdict is per-visit.
- A proto-totality validator exists: `_validate` (`_analyze.py:339-344`) asserts no unexpanded call and no
  surviving loop template survive analysis, and `_call_plan` asserts every surviving call is classified — the
  seed of M0's plan-totality scaffold.

Still emission-owned — the remaining decision-carrying surfaces, with current extents:

- The five copy routines plus inline clones: `_emit_concat` `:631-651`, `_emit_conversion` `:668-711`,
  `_copy_leaves` `:713-724`, `_project` `:726-738`, `_install` `:740-755`; the same leaf-walk skeleton recurs
  inline in the PySubscript window walk (`:920-935`), the PyAttr state-aggregate arm (`:973-984`), and the
  CONSTRUCTION default fill (`:1026-1038`).
- Routing re-derivations: `_emit_concat` re-walks layouts to compute cell offsets (`:631-651`); the PySubscript
  fallback re-runs `operator.index` and `child_slice` (`:941-951`); the PyAttr record projection re-derives the
  field's cell window (`:968-972`). `subscript_plans`/`route_plans` cover only slice/gather selections and
  transposes; positional projections and concats are re-derived at emission. Analyzer-side, the routing evidence
  still lives in id-keyed side tables reset each round (`_analyze.py:358-364`, reset `:500-517`) and is
  re-collected by `_finalize`'s one replay of `_transfer` over the stabilized graph (`:702-752`) — non-mutating,
  but M1's evidence-atomic recording remains undone.
- Kind re-classification: `_is_bool_fact` `:321-328`, `_sem_of` `:474-480`, `_fact_port_type` `:482-489`,
  `_fact_sem` `:491-499`; the typed materializer (`:762-794`) decides promotion-versus-rejection per operand
  position. The PyCompare arm re-decides the bool/non-bool doctrine (`:849-869`) — now a duplicate of the
  analysis-side C2 check (`_analyze.py:1085-1093`) — and PySelect re-derives its select kind (`:876-902`).
- Emission-time lowering policy: `_emit_cast` decides identity-versus-conversion from the final facts
  (`:1173-1194`; deliberate today, per `_call_plan`'s comment); `_emit_intrinsic` dispatches result rules from
  operand facts (`:1042-1059`); `_emit_power` owns the whole power policy, including two refusal classes
  (`:1223-1283`); the return-contract tower enforces at exit (`:226-318`, `:1363-1391`, `:1410-1500`);
  `_slot_name` collision/unanchored-provenance refusals (`:522-550`); `_carrier_float` NaN/overflow policy
  (`:183-196`). In total 42 EmissionRejection raise sites — located now, but still emission-decided (M5's
  target).
- Open-coded dispatch in the analyzer, the M3/R0 row-table target: `_expand_call` `_analyze.py:2091-2583`
  (~490 lines, ~14 identity-dispatch arms, of which two are now trim-refusal arms: getattr `:2277`, isinstance
  `:2287`), and the attribute ladder `_attribute` `_analyze.py:1750-1880` (~130 lines).
- Doctrine still multi-sited: `contains_record` ×4 in `_analyze.py` (`:1577`, `:1613`, `:1645`, `:1777`) plus 2
  in `_fold.py`; the elementwise skeleton ×3 in the emitter (`_emit_elementwise` `:1126-1158`, the unary arm
  `:1289-1304`, the conversion coercion `:668-711`).

Honest assessment. The analyzer-side half of totality is done: the schema flow, the deferral net with causal
priority, located origins end to end, and a typed plan that already carries stores (conversions), calls
(classification), state (resets, origins, order), and selections/routes. The emitter-side consumption discipline
is not: emission still derives kinds from Facts at every operand, re-derives positional routing, owns
cast/intrinsic/power lowering policy and contract enforcement, and raises 42 refusals. Grep-checked today, the
live emitter fails SC2 outright — it imports Fact, Known, Residual, AggregateFact, the Py* ops, and reaches into
`_lib` for Intrinsic. That is precisely the gap Stage 4 closes under either shape; nothing found in the refresh
weakens the evaluation's verdict, and B1's precondition (no resolved plan can be typed until stores are
schema-checked) is now satisfied.

## 3. The two Stage-4 shapes

MORPH (campaign M0-M7; on dev directly, ~9 steps, each independently green and raw-byte-gated): M0 guards first
(emitter Fact-import-ban test + plan-totality validator scaffold); M1 evidence-atomic recording (`_finalize`
stops replaying `_transfer`); M2 the routing algebra — one typed record "result cell i ← source cell π(i), or a
constant", keyed by plan site/cell, never by dst (`StorePlace`/`PyStoreAttr` have no dst, `_ir.py:186-292`),
winners recorded only at final stabilization — then adoption and same-commit deletion of the copy routines and
clones; M3 typed dispatch rows (ordered frozen first-match tables, identity comparison, messages verbatim); M4
doctrine single-siting as use-specific consumption ops; M5 EmissionRejection retirement (locations exist from
S2.11; the corpus pins public classes); M6 structured-origins datatype migration; M7 optional and gate-deferred —
if the totalized tables have become a de-facto RIR, materialize the spine then, with schema learned from contact.

TRANSPLANT (campaign R0-R7; branch `restructure/rebuild`, ~8 milestones): R0 scaffold `_fir2/` with the `_rir.py`
datatypes (Codex X6b on the schema first) plus an A/B differential harness — new machinery; the existing
`test_fir_differential.py` is a value oracle, not this — with transfer-function bodies lifted near-verbatim and
the analyzer skeleton rebuilt around evidence-atomic recording and dispatch rows from day one; R1-R5 milestones
by construct family (scalar straight-line → branches/loops/unroll → state+W/D → aggregates/records →
arrays/linalg/ports), each gated differential-green; R6 full-corpus differential (36 golden cases + rejection
corpus + extended A/B fuzz); R7 cutover — flip the frontend, delete `_fir` in the same commit, canonical gate +
re-freeze. The branch is abandonable at zero cost to dev until R7.

The advocate's RIR sketch (the transplant's target schema, and where M7 would converge): one Def form; three RHS
kinds over leaf cells; cell-write lists rather than explicit phis (phi arms must come from the final edge set, so
Braun SSA stays emitter-side); a `join_kinds` table; state/port/const side tables; a closure+definedness+kind
verifier; a printer.

The skeptic's cost accounting (as recorded in the campaign): materializing the IR buys little the totalized
tables would not; the transplant pays for a new differential harness, an analyzer-skeleton rebuild, and a
divergence tail; and raw byte-identity is not guaranteed across the one landing that swaps the emitter — HIR
value-id assignment order is byte-visible downstream (`_mir/_lower.py:366-372` lowers constants and state reads
in sorted value-id order), so bytes hold only if the new emitter reproduces the old allocation order exactly:
achievable, since the dependence runs through id order alone, but not free — which is why the swap landing keeps
the maintainer's pre-authorized canonical gate available. The morph keeps every landing raw-byte-gated on dev
and retains M7 as the materialization option once schemas are learned from contact.

## 4. The spike (S3.3): plan, criteria, decision table

Restated from the maintainer-approved campaign plan (`docs/campaign.md` S3.3) for the record. The criteria are
pre-agreed; goalposts do not move mid-spike. As adopted at X4/S3.2, the executable spike specification —
amended witness set, executable SC1-SC4, input-topology-aware decision table, and the evidence ledger — is
`docs/decisions/arch-spike.md`; it supersedes this section where they differ.

Plan: spike on worktree branch `spike/resolved-ir` (throwaway pushes for CI verdicts). Timebox: 1 session, hard
cap 2. Content: `_rir.py` schema as real dataclasses + prototype residualizer for the hard-construct subset
{ekf1_stateful, finite_set_current_controller, iir1_hpf, + one branch-join kernel + one
return-onto-state-port-dedup case} + the complete mechanical emitter.

Criteria (pre-agreed):

- SC1: Verilog+HIR byte-identical to `freeze-1` for the subset (fallback per the X3 amendment below: HIR
  alpha-canonical under the real block-and-value canonicalizer, with the independent semantic gates green and
  schedule metrics equal — `renumber()` alone does not qualify).
- SC2: emitter imports no Fact/registry/Py* (grep-checked).
- SC3: zero escapes from the closed op set (boundary-table additions allowed, decision-carrying node kinds not).
- SC4: residualizer ≤ ~1,200 LOC for the subset (extrapolation ≤ ~1,800).

Decision table: all pass at byte level → transplant with byte gate; pass at canonical level → transplant with
the pre-authorized canonical gate for the swap landing; semantic divergence in >1 construct family, or SC4
blown, or timebox expiry → MORPH (default).

The comparison baseline the criteria reference now exists: the freeze corpus (S2.15, commit 557fc69) — 35 golden
cases with full RTL, schema-versioned pre-optimize HIR dumps, ABI manifests, exact schedule metrics, structured
JSONL diagnostics, and immutable rejection modules — seed-matrix-certified at the freeze commit 557fc69. The
post-Y tip e3d5f18 has passed only the byte-identical 148/148 refreeze check; the seed matrix has not been rerun
there, and CI certification with the `freeze-1` tag is still pending. X3 amendment
(`docs/decisions/freeze-design.md`, adopted): the canonical landing additionally requires a real block-and-value
alpha-canonicalizer — `renumber()` compacts block ids only — itself tested on deliberately permuted equivalent
HIRs, plus the independent semantic gates (the Python-reference differential and the MIR-interpreter oracle),
because cosim's numerical model derives from the same LIR it certifies. SC1's canonical fallback is read per
that amendment: canonical-identical means alpha-canonical under the real canonicalizer with the independent
gates green, not merely renumber-equal.

## 5. What would change the answer

Nothing is expected to: the decision table is exhaustive over the outcomes and its default is safe. The residual
risks worth naming — signals to read during the spike, not grounds to re-litigate after it:

- Schema growth beyond the sketch. If `_rir.py` cannot hold at one Def form + three RHS kinds — if a fourth
  decision-carrying RHS kind, or per-construct node kinds, creep in to keep the emitter mechanical — that is SC3
  failing in spirit even where the grep passes, and it argues morph: the closed op set is the transplant's
  premise.
- The emitter needing facts it cannot get from a plan. Any point where the mechanical emitter must consult a
  Fact rather than a plan row to pick an operator or kind is a totality leak in the residualizer. One instance
  is a bug to fix inside the timebox; a pattern across construct families is the >1-family divergence row of the
  table.
- Canonicalizer cost. If the block-and-value alpha-canonicalizer plus its permutation tests grow into a project
  of their own, the canonical row's price rises and byte-level SC1 — or morph — should win. The canonical gate
  exists to be affordable, not to become a second campaign.

Timebox expiry remains the hard override: expiry = morph, no extensions beyond the pre-approved hard cap.
