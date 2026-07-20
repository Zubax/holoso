# S3.3 spike evidence ledger — spike/resolved-ir

Append-only. One entry per event, never rewritten. Contract: `docs/decisions/arch-spike.md`.

## E1 — spike start, provenance

Worktree branch `spike/resolved-ir` at base 4f1dd4c (dev). Freeze provenance carried from the memo: the freeze
corpus is committed (S2.15) and seed-matrix-certified at freeze commit 557fc69; the tip lineage through e3d5f18
passed the byte-identical 148/148 refreeze check; CI `freeze-1` tag certification pending. The golden corpus in
this worktree (`tests/golden/`) is the SC1 comparison surface for the six golden witnesses; the two microkernel
witnesses are pre-baselined through the OLD pipeline at spike start (entry E4) before any prototype code exists.

Input topology: PRIMARY (adapter) prototype only — residualize today's `ResidualUnit` into RIR, plus the
complete mechanical emitter. No secondary prototype is planned within the timebox; consequently NO TRANSPLANT
row of the decision table is reachable from this spike, only MORPH-with-mandatory-spine or the default.

## E2 — the frozen RIR schema (SC3 freeze)

The schema is `holoso/_frontend/_fir/_rir.py` in the SAME commit as this ledger entry; that commit is the
freeze. Everything below is a judgement call of the spec's section 8 resolved AT the freeze, before the timer;
per SC3, any decision-bearing variant or field added in any LATER commit is an objective SC3 failure recorded
here as it happens.

Frozen vocabulary (spec section 8 verbatim, with the resolved calls J1-J11):

- Identifiers: `CellId(root, ordinal)`; `ConstId | BlockId | PortId | StateSlotId` as single-int frozen
  wrappers. J1: typed wrappers, not bare ints — `RhsRead(PortId | StateSlotId)` requires discriminable types;
  the information content is exactly the spec's ints.
- `Kind` = {BOOL, INT, FLOAT}.
- Statements: `Def(dst, rhs)` with `Rhs = RhsConst(ConstId) | RhsRead(PortId | StateSlotId) |
  RhsApply(op, args: (CellId | ConstId)*, kind)`; `Route(moves: (dst, src: CellId | ConstId)*)`.
- Terminators: `TJump | TBranch | TExit`; executable-only CFG, dense RPO block ids, entry = block 0, exactly
  one TExit block; folded branches arrive as jumps; never-returns refused by the residualizer.
- Side tables: consts (interned; floats as binary64 bit patterns; carrier-admissible only), ports (ordered per
  signature: name, kind, defining cell), state (ABI order = first-store source order: rendered slot name, kind,
  typed reset const, live-in cell, live-out cell, public flag), join_kinds ((block, cell) -> kind), returns
  (ordered rows: port name, source cell-or-const, kind; empty = none), origins ((block, statement index) ->
  interned frame chain; printer/debug only), root_names (printer only).
- Op vocabulary `RirOp`: one closed enum member per frontend-emittable HIR operator, each with a frozen
  operand/result kind signature row (`OP_SIGNATURES`).

Resolved judgement calls, all in the freeze commit:

- J2: relational operators are folded into per-relation members (FREL_EQ..FREL_GE, IREL_EQ..IREL_GE) so the op
  vocabulary carries no parameter fields. `FloatMulPow2` is excluded: it is introduced by HIR strength
  reduction only and is unreachable from the frontend emitter.
- J3: the state row carries `public: bool`. Production derives observability from the leading-underscore
  attribute-name policy inside `_finish_exit`; that policy is a residualizer decision now.
- J4: a state row's `live_in` cell is defined-at-entry by the schema (the verifier treats it so); the emitter
  materializes the interned `StateRead` node on first demand — a Braun entry-chase or an explicit
  `RhsRead(slot)` statement. The residualizer places `RhsRead` statements exactly where production eagerly
  registers a slot read (`PyStoreAttr` sites), preserving node-creation order byte-for-byte.
- J5: returns rows carry `source: CellId | ConstId`. A Known return leaf is a ConstId source because
  production materializes return constants interleaved with return-cell reads at the exit; a cells-only
  encoding would displace node creation and break byte identity. Empty rows = void; one row named `out_0` =
  scalar; n rows = the flattened aggregate in canonical leaf order. No extra variant is needed.
- J6: two coercions remain in the emitter as KIND-DRIVEN value bookkeeping, sanctioned under the agreed
  definition of mechanical (Braun SSA is emitter-side): (a) a phi whose join_kinds row says FLOAT promotes an
  INT arm with IntToFloat on the arm's own edge — phi arms come from the final edge set, which only the
  emitter sees; (b) a returns row whose kind is FLOAT promotes an INT source likewise. Both consume only
  resolved RIR kinds plus HIR types — no Fact, no re-derivation; anything not int->float is an internal assert
  (the residualizer already refused every genuine mismatch). Every OTHER promotion is an explicit convert
  Apply emitted by the residualizer.
- J7: each block carries `preds`, the final edge set in production's arm order (sorted by pre-renumbering FIR
  block index). Phi arm order is byte-visible downstream, so the order is data, not a derivation.
- J8: origins are keyed (BlockId, statement index) into an interned frame-chain table; explicitly
  non-decision-bearing, printer/ledger only.
- J9: `RhsRead(PortId)` is reserved-but-unused: the emitter prologue defines port cells eagerly from the ports
  table in order, matching production's parameter loop.
- J10: const-table admission implements production's `_carrier_float` policy (overflow/NaN refusals) at
  residualization; the residualizer decides IntConst vs FloatConst vs BoolConst per use context.
- J11: join_kinds enumerates the ADMISSIBLE phi sites: the emitter creates phis only at enumerated sites
  (asserted) and not necessarily at all of them (Braun is demand-driven; trivial phis collapse). The
  residualizer enumerates every (multi-predecessor executable block, known cell) pair.

## E3 — SC4 manifest, LOC rule, and extrapolation formula (declared before measurement)

LOC rule: physical lines excluding blank lines and lines whose first non-whitespace character is `#`.
Docstrings count as code. Counter: `tools/spike_loc.py`.

File manifest and roles:

| file | role (SC4 bucket) |
| --- | --- |
| `holoso/_frontend/_fir/_rir.py`, section "schema" (up to the verifier marker) | residualizer (schema + tables it uses) |
| `holoso/_frontend/_fir/_rir.py`, section "verifier" | verifier/printer |
| `holoso/_frontend/_fir/_rir.py`, section "printer" | verifier/printer |
| `holoso/_frontend/_fir/_residualize.py` | residualizer |
| `holoso/_frontend/_fir/_emit_rir.py` | emitter |
| `tests/test_spike_rir.py` | harness (SC1 comparisons, SC2 closure checker, negative packet) |
| `tools/spike_baseline.py` | harness (old-pipeline pre-baselining, no prototype imports) |
| `tools/spike_loc.py` | harness |

No canonicalizer is planned: SC1 targets byte identity; if bytes diverge the divergence is recorded and
classified, and absent an affordable canonicalizer that witness FAILS SC1 (default row). The criteria are not
redefined.

Extrapolation formula (declared now, before any measurement): the spec's section-8 "emitter surface -> home"
table has 11 rows with a residualizer/RIR home (the three EXCLUDED rows do not count). Let C be the number of
those rows exercised by at least one witness per the coverage matrix (E5, finalized at the end), U = 11 - C.
Then L_extrap = L_meas x 11 / C, where L_meas is the residualizer-bucket LOC. SC4 passes iff L_meas <= 1200
and L_extrap <= 1800; if the coverage matrix cannot be completed, SC4 is unresolved (default row).

## E4 — pre-baselined microkernel witnesses (old pipeline only; before any prototype code)

Captured by `tools/spike_baseline.py` into `tests/spike_golden/`, by the exact `build_artifacts` machinery of
the golden corpus (same regalloc pinning, version canonicalization, ABI/metrics shapes):

- `store_conversion` (positive): `x = value; x = 3; return x / 2` per `tests/test_frontend_schema.py:167` —
  the accepted local int->float store-edge conversion; kernel module `tests/spike_golden/kernels/store_conversion.py`.
- `routed_diamond` (positive): transpose + gather + branch microkernel, the sanctioned fallback for the
  transpose-routing witness should `imu_frame_transform-e8m36` prove too heavy; kernel module
  `tests/spike_golden/kernels/routed_diamond.py`. Baselined now because the fallback must predate prototype code.
- `return_mismatch` (negative): annotated `-> None` returning a value; the declared-versus-actual
  return-contract arm the frozen corpus does not pin; diagnostic row `tests/spike_golden/return_mismatch.diag.json`.

Both positive baselines carry hir/verilog/abi/metrics; the negative carries the full diagnostic row in the
golden JSONL shape. Build: e8m36, `default_ops`, fetch_stages=3, shipped regalloc knobs — identical to the
golden matrix conventions.

## E5 — SC1 per-witness verdicts (final)

Harness: `tests/test_spike_rir.py::test_sc1_witness`, 8/8 PASSED. For every witness, both entry points were
built through the identical downstream pipeline and compared on all four surfaces — schema-versioned
pre-optimize HIR dump, version-canonicalized full Verilog bytes, ABI manifest, exact schedule metrics — and
the OLD pipeline was additionally byte-compared against the frozen corpus row (golden cases) or the E4
pre-baseline (microkernels). Every comparison is BYTE-IDENTICAL; the canonical fallback was never needed and
no canonicalizer was built.

| witness | HIR dump | Verilog | ABI | metrics |
| --- | --- | --- | --- | --- |
| iir1_lpf-e8m36 | identical | identical | identical | identical |
| ekf1_stateful-e8m36 | identical | identical | identical | identical |
| finite_set_current_controller-e8m36 | identical | identical | identical | identical |
| iir1_hpf-e8m36 | identical | identical | identical | identical |
| format_probe-e8m36 | identical | identical | identical | identical |
| imu_frame_transform-e8m36 | identical | identical | identical | identical |
| store_conversion-e8m36 (pre-baselined) | identical | identical | identical | identical |
| routed_diamond-e8m36 (pre-baselined) | identical | identical | identical | identical |

`imu_frame_transform-e8m36` was carried directly; the routed-diamond fallback microkernel was baselined and
kept as an additional witness rather than a substitute. Mismatch log: ZERO observed mismatches across every
harness run of the final code (the first complete 17-test run passed; the two later size-reduction passes were
each re-verified green). Three byte-divergence hazards were identified analytically during design and
neutralized by construction before they could manifest, all of class ordering-only (HIR value-id allocation
order): (a) SUB's left operand is materialized by production before the right operand while the Apply tree
reads it at the second node — pinned by a touch Route; (b) a runtime power's exponent is materialized before
the base — same pin; (c) a select condition is materialized ahead of arm conversions — same pin. A fourth
hazard, production's eager slot-read registration at every state store, is pinned by the RhsRead statements
(ledger J4). The draft residualizer's eager slot registration at state READS (a true ordering bug against
production's lazy `_state_read`) was caught in self-review before the first harness run and removed.

## E6 — negative policy packet (final)

Harness: `test_negative_packet_*`, 5/5 PASSED, full diagnostic parity on class, rendered message, location
(file/line/column/source text), and origin frames, against the frozen JSONL rows (golden) and the E4
pre-baseline (return_mismatch):

- legacy_power_chain — UnsupportedConstruct, fires in the RESIDUALIZER (power policy moved upstream);
- legacy_beyond_carrier_constant — UnsupportedConstruct, fires in the residualizer's const-table admission;
- legacy_never_returns — UnsupportedConstruct, fires in the residualizer's CFG check;
- return_mismatch (pre-baselined) — UnsupportedConstruct, fires in the residualizer's returns resolution;
- legacy_shared_live_out — SynthesisError, fires in the Verilog backend on the prototype-emitted HIR exactly
  as on production's (location null, origin empty, class included in the parity check).

The harness additionally asserts per case that no traceback frame lies in `_emit_rir.py` (the mechanical
emitter never rejects) and none in the production `_fir/_emit.py` (the production emitter never ran on the
prototype path).

## E7 — SC2: emitter closure and the backchannel audit

- Stated-import closure of `_emit_rir` = {`holoso._hir` and its submodules, `holoso._util`, `._rir`}; the
  banned set (Fact domain, analyzer, FIR ops, builder, fold/resolve/opsem/value/signature, production
  emitter, residualizer, library registry, ast_support) is unreachable — `test_sc2_emitter_import_closure`.
  Recorded caveat: Python executes ancestor-package `__init__` modules (`holoso._frontend/__init__` imports
  the production frontend) for ANY module inside the package; the check measures the STATED-import closure,
  which is the dependency structure the emitter would carry after a real cutover. `._rir` itself imports no
  holoso module at all (asserted).
- Content: exhaustive dispatch over resolved variants + Braun sealed-block SSA + the sanctioned exit-identity
  dedup; zero `raise` statements and zero lambdas (AST-checked); asserts are the only failure mode.
- Object-graph half: `verify_rir`'s recursive closure walk (no Fact, no Py* op, no callable, no registry
  handle — only RIR types, scalars, tuples, dicts) runs on every witness RIR inside `lower_via_rir` plus the
  dedicated `test_sc2_object_graph_closure_on_a_witness`.
- Kind decisions inside the emitter: none. The two J6 coercions (phi-arm and returns-row int->float) consume
  only the resolved join_kinds/returns kinds plus HIR types; anything not int->float asserts.
- Attempted escape hatches, including reverted ones: (1) the first draft of the emitter's operator table
  mapped RirOp.INOT to IntXor() as a knowing placeholder — a lying table row; replaced with IntNot() before
  any commit. (2) No Fact, plan, or registry consultation was ever added to the emitter; no other hatch was
  attempted. (3) The emitter renders state observability ports as "state_" + slot name — a fixed mechanical
  prefix mirroring production's state_port_name; the OBSERVABILITY decision itself is the resolved `public`
  flag (J3). Recorded as spelling, not policy.

## E8 — SC3: schema stability verdict

`git diff e9716a8 -- holoso/_frontend/_fir/_rir.py` is EMPTY at the end of the spike: the schema file is
byte-identical to the freeze commit. Zero variants, zero fields, zero op-vocabulary members were added,
removed, or modified after the freeze; the harness fingerprint (`test_sc3_schema_fingerprint`: every
dataclass's field list plus the 61-member op count) pins it executably. The freeze contained everything the
witnesses needed — including the E2 judgement calls (preds order J7, public flag J3, RhsRead-at-store J4,
const-or-cell return sources J5) each of which proved load-bearing. SC3: PASSED.

## E9 — SC4: sizes, formula, coverage matrix, verdict

LOC (rule per E3; `tools/spike_loc.py`):

| role | LOC |
| --- | --- |
| residualizer = `_rir.py#schema` 263 + `_residualize.py` 1012 | 1275 |
| emitter (`_emit_rir.py`) | 351 |
| verifier + printer (`_rir.py` sections) | 232 |
| harness (`test_spike_rir.py` 293 + `spike_baseline.py` 103 + `spike_loc.py` 41) | 437 |
| canonicalizer | 0 (not built; SC1 held at byte level) |

Subset bound: 1275 > 1200 — SC4 residualizer budget FAILED (by 6.3%), after three size passes
(1584 -> 1373 -> 1281 -> 1275) of spelling-only compression, each re-verified byte-identical by the full
harness. The criterion was not redefined; further reduction within the timebox was judged achievable only by
either style vandalism or restructuring the production-mirroring logic whose fidelity is the spike's core
evidence.

Formula correction, recorded as an event: E3 declared N_total = 11 for the section-8 table; the table
actually holds 13 rows of which 3 are EXCLUDED, so the true N_total is 10. Both readings are reported; the
as-declared reading is the more conservative (larger) extrapolation.

Coverage matrix — every non-EXCLUDED emitter-surface row against the witnesses that force it (plan-kind
instrumentation, this session):

| emitter surface | forced by |
| --- | --- |
| Five copy routines + inline clones -> Route | all eight positives (ekf1 carried vectors, imu 2142 moves, fsc 1347) |
| Routing re-derivations -> resolved offsets | routed_diamond (positional t[1]); ekf1/fsc/routed subscript windows; imu/routed transpose route_plans |
| Kind re-classification -> kind fields + converts | all positives (iir1_lpf bool state, format_probe compares, fsc B2F, store_conversion) |
| Cast policy -> Route alias / convert Apply | iir1_hpf (identity float(x)), fsc (16 CAST plans) |
| Intrinsic result rules -> pre-lowered Applys | fsc (9 INTRINSIC plans: FABS/FMAX/FSIN/FCOS/FREL chains) |
| Power policy -> pre-lowered chains + upstream refusals | legacy_power_chain (refusal arm); positive chain arms UNFORCED |
| Return-contract tower + never-returns -> returns table | all positives; return_mismatch; legacy_never_returns |
| _slot_name refusals -> resolved names, verifier uniqueness | iir1_hpf (nested lpf__y), ekf1 (aggregate x_0..P_urt_5); collision/unanchored refusal arms UNFORCED |
| _carrier_float policy -> const-table admission | legacy_beyond_carrier_constant; every float const row |
| Doctrine multi-siting -> residualizer-internal helpers | ekf1/imu/routed elementwise; fsc records + CONSTRUCTION |

C = 10 of 10 rows forced at family level. Uncovered SUB-ARM inventory (feeds the extrapolation caveat):
positive power chains (int/float/negative/runtime-exponent), cast conversion arms other than identity
(F2I/B2F/F2B/B2I/I2B appear in fsc casts only partially), slot-name collision and unanchored-provenance
refusals, sequence concat/repeat offsets, ALWAYS_INT/INT_OVERLOAD integer intrinsic arms, int/bool binary
datapaths, MaybeUnbound reads, structural-flavor returns, record/list/variadic return-contract arms.

Extrapolation: as declared (N=11): 1275 x 11/10 = 1403 <= 1800 — PASSED. Corrected (N=10): 1275 <= 1800 —
PASSED. SC4 overall: FAILED on the subset bound, extrapolation within budget.

## E10 — runtime, tables, replay counts, duplication accounting, and the outcome

Residualizer wall time / peak traced allocation / RIR magnitudes per witness (Analyzer excluded; single run):
iir1_lpf 24 ms / 20 kB (19 statements, 5 blocks); iir1_hpf 11 ms / 30 kB; format_probe 3 ms / 19 kB;
store_conversion 1 ms / 5 kB; routed_diamond 4 ms / 58 kB (96 statements, 94 join rows); ekf1_stateful 55 ms
/ 448 kB (702 statements, 9 slots); imu_frame_transform 350 ms / 1.2 MB (2247 statements, 304 blocks);
finite_set_current_controller 360 ms / 3.4 MB (1473 statements, 241 blocks, 11521 join_kinds rows — the J11
superset enumeration is the dominant table and the obvious first optimization for a production spine).

Host-call/transfer replay counts: the residualizer performs ZERO transfer replays, ZERO live host reads, and
ZERO registry resolutions — it consumes ResidualUnit fields only (binding_facts, call_plans incl. resolved
Intrinsic rows, subscript/route plans, store_order/state_livein/state_resets/store_origins/store_conversions,
provenance, block_in, executable blocks/edges). Recording is evidence-atomic FROM THE ADAPTER DOWN; the one
`_finalize` transfer replay that builds ResidualUnit itself is upstream in the shared analyzer and unchanged
(M1's concern is untouched by this spike).

Duplicated versus reused logic: the adapter duplicates, rather than imports, the production emitter's
decision helpers (carrier policy, fact-kind classification, port-path spelling, contract-leaf-kind walk, the
return-layout validation tower, power/cast/intrinsic lowering logic) — approximately 330 LOC of the 1012 —
because the project convention bars importing underscore names across sibling modules. In a real morph these
lines MOVE out of `_emit.py` (1500 LOC) rather than being added beside it; projected net production change for
the emitter half: -1500 (delete `_emit.py`) + 1275 (residualizer bucket) + 351 (mechanical emitter) + 232
(verifier/printer) = +358 LOC gross, with the RIR spine materialized and 42 emission-decided refusal sites
reduced to zero in the mechanical half.

Freeze provenance at close: base 4f1dd4c; freeze commit e9716a8 (schema + ledger); baselines 5657973;
prototype 5aa2742; size passes follow. The golden corpus in-tree matched the OLD pipeline byte-for-byte on
every witness in every run (148-file refreeze state carried from e3d5f18 lineage).

OUTCOME per the pre-agreed decision table: SC1 byte-identical on all witnesses and surfaces; SC2 PASSED; SC3
PASSED; negative packet PASSED; SC4 FAILED on the subset bound (1275 > 1200). "ANY other outcome ... SC4
blown ..." fires: the ruling is MORPH (default). The adapter-pass row (MORPH with MANDATORY materialized
spine) is NOT claimable, by 75 lines of residualizer size. The evidence nonetheless bears directly on M7:
the materialized spine closed — every semantic decision fit the frozen one-Def/three-RHS/Route vocabulary
with zero schema growth, the mechanical emitter reproduced production byte-for-byte from data alone, and
every emitter-owned refusal moved upstream with diagnostic parity. No TRANSPLANT row was reachable (primary
prototype only, per E1).
