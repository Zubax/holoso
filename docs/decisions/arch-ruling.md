# Stage-3 architecture ruling: MORPH, with M7 optional and gate-deferred

Status: FINAL. Closes the architecture gate opened by `docs/decisions/arch-memo.md` and executed by
`docs/decisions/arch-spike.md`. Stage 4 proceeds as variant MORPH per `docs/campaign.md`.

## The question

Does the frontend's resolution-totality restructure proceed by morphing the existing analyzer/emitter in place,
or by transplanting onto an independently built resolved IR — and if it morphs, is materializing the resolved
spine (M7) mandatory or gate-deferred?

## Evidence

The spike ran on branch `spike/resolved-ir`, base `4f1dd4c`, tip `dc76fbf`; the branch ref is deleted. Its
append-only evidence ledger (entries E1-E10) is preserved verbatim at `docs/decisions/spike-ledger.md`, copied
onto `dev` rather than summarized away: the primary evidence is what makes the corrections below findable, and
three of them were found by reading it against the prototype.

It built the PRIMARY (adapter) topology only: today's `ResidualUnit` residualized into a materialized RIR, plus
a complete mechanical emitter. No independent residualizer was built, so NO TRANSPLANT row of the decision
table was reachable from this spike by construction — a limit declared at E1, before the timer, not discovered
afterwards.

| criterion | result |
| --- | --- |
| SC1 byte identity | PASS on all 8 witnesses (pre-optimize HIR, Verilog, ABI, exact schedule metrics, through both entry points); the alpha-canonicalizer was never needed and cost 0 LOC |
| SC2 mechanical emitter | FAILS, on the closure criterion AND on an emitter-side kind decision — see below |
| SC3 closed vocabulary | PASS historically: the RIR schema file is byte-identical at spike start and end, zero variants/fields/op members added after the freeze |
| SC4 size | FAIL: residualizer 1275 LOC against the 1200 subset bound, after three behavior-preserving size passes (1584 → 1373 → 1281 → 1275). Extrapolation passed either way (1403 ≤ 1800 as declared; 1275 ≤ 1800 under E9's own correction of N_total from 11 to 10) |
| negative policy packet | PASS 5/5: every refusal fires in the residualizer or backend, never in the mechanical emitter |

## Ruling

PLAIN MORPH (the default row). M7 — materializing the resolved spine — stays OPTIONAL and gate-deferred, and
if taken, lands LAST, after M1-M6.

The decision table is exhaustive and mechanical, and SC4 blown fires its default row explicitly. The
mandatory-materialized-spine row requires the adapter prototype to pass ALL criteria; it does not.

Two arguments were put to Codex against that reading, and both were rejected:

1. That the consult's own X4 position had refined the verdict in advance to "morph the analyzer but make the
   resolved spine MATERIAL". Codex's answer: that was an architectural prior, not authority to override a
   pre-agreed falsifier after seeing the result.
2. That ~330 of the residualizer's 1012 non-schema LOC are duplication forced by the spike's module boundary
   (the project bars importing underscore names across sibling modules) — lines that in a real morph MOVE out
   of `_emit.py` rather than being added, which would put the residualizer near 945. Codex's answer: legitimate
   migration-cost evidence, invalid gate arithmetic. Those lines remain in the production residualizer; only
   their `_emit.py` copies disappear, and `_emit.py` was never counted in SC4. SC4 measures total residualizer
   design complexity, not net repository growth. The ledger already uses the figure correctly, in its projected
   net-change calculation.

Re-reading a failed criterion into a pass after seeing the result is the goalpost drift the campaign's risk
register warns about. The criterion was not redefined.

The ruling is over-determined: SC2's failure below fires the same default row independently of SC4.

## Corrections to the ledger's own claims

Recorded here because the ledger cannot be rewritten and its branch does not survive. All three were raised by
the X5 consult and reproduced independently before being accepted.

SC2 FAILS as literally specified, and the ledger's PASS is wrong. The criterion bans user-facing rejection
construction from the emitter's TRANSITIVE dependency closure. That closure contains `holoso._errors`, reached
through `holoso._hir` → `_hir._const`, which constructs `UnsupportedConstruct` for a beyond-carrier constant.
The prototype's check missed it because its banned set enumerated frontend modules and omitted `_errors`.
Verified by recomputing the closure directly.

The criterion is, as written, unsatisfiable by any emitter that emits HIR: `_hir._const` constructs the
rejection type (for NaN — beyond-carrier admission is the frontend's `_carrier_float`, a separate policy), so
every HIR-building module reaches it. That much is a defect in the criterion as much as in the prototype.

SC2 fails a second time, and this one is substantive. The criterion also forbids emitter-side semantic
kind choices and requires promotions to arrive as explicit conversion rows. The prototype's J6 disclosed two
coercions as "kind-driven value bookkeeping", and they are real decisions: on a mixed int/float join and on
`def f(x: int) -> float: return x`, the RIR carries ZERO explicit int-to-float rows while the emitter inserts
an `IntToFloat` — chosen by inspecting the type of the HIR node it just generated. A promotion decided from
emitted nodes is exactly the re-deciding the boundary exists to abolish.

So the properties that DO hold are narrower than a "substantive half": the emitter's closure reaches no Fact,
no registry, no `Py*` op, no callback, and no frontend decision module; its own source contains zero `raise`
statements and zero lambdas; and all five refusals in the negative packet fire upstream with diagnostic
parity. That last is the packet's five, not "every refusal" — the ledger's own coverage matrix lists unforced
refusal sub-arms.

Two consequences for Stage 4. M0's guard must be written to the frontend-decision-layer property rather than
to a transitive `_errors` ban that nothing can satisfy. And an import-and-raise guard is NOT sufficient on its
own: it cannot see the J6 class, since inserting a conversion node reaches no banned module. The plan-totality
validator has to carry that weight — every kind promotion consumed from an explicit plan row, never derived by
inspecting emitted nodes.

The prototype's closure walker also dropped submodules named by `from . import x`, hiding 8 HIR pass modules
(closure 11 modules as measured, 19 when corrected). None of the hidden modules were banned, so the verdict
does not move, but the closure was smaller than it looked. The production helper `tests/_importguard.py` does
NOT share this gap — it resolves `from package import submodule` correctly — so M0 builds on it rather than on
the prototype's copy.

SC1's byte identity is established against the spike's base `4f1dd4c`, NOT against `freeze-1`, which is 38
commits later and carries material analyzer changes and a bumped HIR serializer schema. E1 recorded the tag
certification as pending and the claim was never re-established afterwards. The ruling does not depend on it —
SC1 passing more strongly would not change a table row that SC4 and SC2 both drive to the default — but any
future M7 must re-establish byte identity against the tag rather than inherit this result.

SC3's historical zero-growth claim is sound; its executable fingerprint is not complete, pinning field names
and enum cardinality but not enum vocabulary, signatures, annotations, unions, defaults, or optionality. The
coverage matrix is family-complete (10 of 10 emitter surfaces forced) but the ledger's own uncovered sub-arm
inventory is substantial, and the extrapolation inherits that caveat.

## What the spike established that survives it

The materialized resolved spine very nearly closed. Every semantic decision across all 8 witnesses fit the
frozen one-`Def`/three-RHS/`Route` vocabulary with ZERO schema growth after the freeze — save the two J6
coercions above, which the emitter still decides for itself — and a 351-line mechanical emitter reproduced
production HIR and Verilog byte-for-byte from resolved data alone. The residualizer performs zero transfer
replays, zero live host reads, and zero registry resolutions: it consumes `ResidualUnit` fields only, so
recording is evidence-atomic from the adapter down.

Projected net production change for the emitter half is +539 LOC under the ledger's own SC4 counting rule
(nonblank, noncomment): 1275 + 351 + 232 against `_emit.py`'s 1319, with its 42 emission-decided refusal sites
reduced to zero in the mechanical half. The ledger's "+358" mixed rules — it subtracted `_emit.py`'s 1500
PHYSICAL lines from three rule-counted terms. On physical lines throughout the figure is +614.

That is the strongest evidence available that M7 is achievable. It is not evidence that M7 is cheap, and it is
explicitly not a criterion pass.

## Why M7 lands last if it lands at all

Every intermediate Stage-4 landing must stay byte-identical to `freeze-1`, so the spine cutover is the one
landing that consumes the pre-authorized canonical gate. More decisively: the prototype INHERITS the analyzer's
state wholesale, and FOUR of those structures are corrupted by the deferral seam, not two. `executable_edges`
and `block_in` carry the stale reachability; `runtime_state` carries a leaf whose store a later round proved
unreachable, on a final CFG that is otherwise correct; and `state_livein` carries a live-in driven residual by
an arm that a later round pruned, on a stable graph that is impeccable. The residualizer consumes all four. A
spine materialized over them would bake every documented miscompile into the new boundary instead of
dissolving it, and the last two would arrive with nothing wrong in the graph to notice. The resolved spine must
RECOMPUTE reachability and typing from the stabilized facts, or derive equivalent state afresh. An early
cutover would materialize the known-defective seam; the schema, the verifier, and a dormant A/B harness may
land earlier, but the functional cutover follows M1-M6.

## Consult record

X5 ran as a resume of the X4 session (`019f769b-48c0-7411-97e0-e3b8883b363a`, `gpt-5.6-sol` at ultra effort)
against a pinned detached worktree at `dc76fbf`. Its ruling was plain MORPH with M7 gate-deferred, reached
independently of the table's mechanics and then confirmed by them. There was no disagreement to adjudicate:
the consult ruled against the position the question was drafted to test, and against its own earlier
preference. Its three ledger corrections were reproduced before acceptance and are recorded above.
