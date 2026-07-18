# S3.3 — Spike specification: the resolved-spine gate, as amended by consult X4

Status: the executable specification for the Stage-3 spike, incorporating every X4 position (S3.2, all adopted;
recorded in the campaign log at dev = 5d3029b). Supersedes `docs/campaign.md` S3.3 and `arch-memo.md` section 4
where they differ. Criteria are pre-agreed; goalposts do not move mid-spike. Line numbers refer to e3d5f18
(`holoso/` is unchanged through 5d3029b).

## 1. Purpose

The substantive question is no longer morph-versus-transplant at large: it is whether a materialized resolved
spine (RIR) can be closed — every semantic decision resolved into typed data before emission — without Fact
backchannels or policy-bearing escape hatches. The X4-refreshed verdict: morph the stabilized analyzer and make
the spine material (B1 and store_conversions prove in-place migration works, so the materialized-RIR boundary is
more justified while the analyzer transplant is less); the independent-transplant path remains unproven. The
spike exists to discriminate between these outcomes with pre-committed evidence, not to relitigate them.

## 2. Input topology

Declared up front, because it bounds what the spike can prove:

- PRIMARY prototype: consumes today's `ResidualUnit` (`_analyze.py:315-336`) through an adapter that
  residualizes it into RIR — evidence about the morph shape (can the existing analyzer's output close into a
  material spine).
- OPTIONAL SECONDARY prototype: residualizes generic FIR into RIR independently of the Analyzer and
  `ResidualUnit` — evidence about the transplant shape.

Which prototype actually ran determines which decision-table rows are reachable: without the secondary, no
TRANSPLANT row can fire; the primary alone can conclude only MORPH-with-mandatory-spine or the default.

## 3. Witness set

Positive witnesses, exact GoldenCase ids (frozen baselines in `tests/golden/`):

- `ekf1_stateful-e8m36` — state + W/D over carried vectors (optionally also `ekf1_stateful_shipped-e8m36`, the
  distinct shipped-reset artifact);
- `finite_set_current_controller-e8m36` — the most composite compiling kernel;
- `iir1_hpf-e8m36` — hierarchical component with a stateful child;
- `format_probe-e8m36` — real branch/merge diamond;
- `iir1_lpf-e8m36` — return-onto-state-port dedup;
- transpose-routing witness: `imu_frame_transform-e8m36`, or, if it proves too heavy for the timebox, a
  routed-diamond microkernel (transpose + gather + branch) pre-baselined through the old pipeline at spike start
  before any prototype code runs;
- an accepted local int→float store-conversion witness kernel (shape per
  `tests/test_frontend_schema.py:167`: `x = value; x = 3; return x / 2`), pre-baselined at spike start — the
  freeze corpus cannot pin accepts, so no golden case isolates this surface.

Negative policy packet — success-only subsets cannot prove refusals moved upstream. Golden rejection corpus ids
(`tests/_golden_cases.py` REJECTIONS; parity target `tests/golden/diagnostics/`):

- `legacy_power_chain` — the power policy refusal;
- `legacy_beyond_carrier_constant` — carrier admission;
- `legacy_shared_live_out` — shared-live-out policy (note: public class SynthesisError, not
  UnsupportedConstruct; parity includes the class);
- return contract: `legacy_never_returns`, the corpus's return-contract row. The frozen corpus carries no
  declared-versus-actual return-mismatch row, so the spike pre-baselines one mismatch microkernel (e.g.
  annotated `-> None` returning a value) at spike start by the same mechanism as the routed-diamond fallback;
  absent that, the coverage matrix must list the mismatch arm (`_emit.py:226-318`, `:1410-1500`) as uncovered.

Required artifact: a witness-to-surface coverage matrix mapping every RIR variant and every remaining emitter
policy family (the enumeration in `arch-memo.md` section 2) to the witnesses that force it. Unforced rows are
the uncovered-surface inventory that SC4's extrapolation and the ledger consume.

## 4. Success criteria (executable)

SC1 — fidelity. For each witness, build through BOTH entry points (old pipeline vs prototype) and compare:
schema-versioned pre-optimize HIR dumps (the freeze serializer), full Verilog bytes (version-canonicalized),
ABI manifests, and exact schedule metrics — byte-for-byte against each other and against the frozen corpus rows
where the witness is a golden case. Byte fallback: real block-and-value alpha-canonical equality — the
canonicalizer itself tested on deliberately permuted equivalent HIRs — PLUS the independent semantic gates
(Python-reference differential and MIR-interpreter oracle) wired to the PROTOTYPE path, not production
lowering. Per the X3 amendment, `renumber()` compacts block ids only and does not qualify.

SC2 — mechanical emitter. The emitter's TRANSITIVE dependency closure (module imports and the object graph it
receives) contains no Fact, no registry, no Py* ops, no callbacks, no user-facing rejection construction, and
no semantic operator/kind/routing/contract choice. Its content is exhaustive dispatch over resolved RIR
variants plus internal invariant asserts, nothing else.

SC3 — closed vocabulary. The complete RIR vocabulary — variants, fields, optional fields, table vocabularies —
is frozen in this document (section 8, finalized) BEFORE the timebox starts. Boundary rows contain only closed
serializable RIR values; a recursive closure verifier enforces it. ANY new decision-bearing variant or field
added during the spike is an objective SC3 failure; recording the need is the ledger's job, not grounds for an
exception.

SC4 — size. LOC counted as nonblank noncomment physical lines over a file manifest published in the ledger
before the timer starts. Residualizer ≤ 1200 for the subset, where "residualizer" includes every
decision-bearing helper, schema, and table it uses; emitter, verifier/printer, harness, and canonicalizer are
counted and reported separately. The ≤ 1800 whole-surface extrapolation must derive from the prelisted
uncovered-surface inventory and a formula declared before measurement; otherwise SC4 is unresolved — which is
not a pass and lands in the default decision row.

## 5. Decision table

Exhaustive over outcomes; default MORPH. If both prototypes ran and passed, the independent-prototype rows take
precedence (they carry strictly more evidence).

| Outcome | Ruling |
| --- | --- |
| The adapter (PRIMARY) prototype passes all criteria | MORPH with MANDATORY materialized RIR spine: M7 becomes mandatory, and M0's emitter import ban becomes a ratcheting allowlist until the spine cutover |
| The independent (SECONDARY) prototype passes byte-identical plus all other criteria | TRANSPLANT with the byte gate |
| The independent prototype passes alpha-canonical + the independent semantic gates plus all other criteria | TRANSPLANT with the pre-authorized canonical gate |
| ANY other outcome: SC2 or SC3 failure, any unresolved construct-family divergence, canonicalizer unavailable or over budget, incomplete witness coverage, SC4 blown or unresolved, timebox expiry | MORPH (default) |

## 6. The evidence ledger

A REQUIRED append-only artifact on the spike branch, one entry per event, never rewritten. It must record:

- initial and final RIR schemas, with the construct that forced each addition;
- the fact/backchannel audit of the RIR object graph and the emitter dependency closure, including every
  attempted escape hatch (reverted ones included);
- per-witness variant/policy coverage and the uncovered surfaces (the coverage matrix);
- every HIR/Verilog mismatch, classified: semantic / block-alpha / value-alpha / ordering-only / ABI /
  scheduling;
- canonicalizer LOC, wall time, algorithm, and the permutation corpus;
- LOC by role and file; duplicated versus reused transfer logic; projected net production change;
- host-call/transfer replay counts — whether recording is actually evidence-atomic;
- negative diagnostic parity: class, message, location, origin frames, precedence;
- residualizer runtime/memory and RIR node/table counts, with the extrapolation inputs;
- freeze provenance: SHA/tag, case identities, seed-matrix result, certification state.

## 7. Timebox and mechanics

One session, hard cap two; expiry → MORPH, no extensions. Branch `spike/resolved-ir` on a worktree; throwaway
trial pushes are allowed for CI verdicts. The branch is deleted after S3.4 with its SHA recorded.

## 8. Draft RIR vocabulary — to be frozen at spike start

The first cut, derived from the memo's advocate sketch (one Def form; three RHS kinds; cell-write lists, not
explicit phis; a join_kinds table; state/port/const side tables; a closure+definedness+kind verifier; a
printer), refined against the actual `ResidualUnit` surface and the emitter's remaining decision families.
Freezing this section — resolving every remaining judgement call below, without structural growth — is the
spike's first act, before the timer starts; after that, SC3 applies verbatim.

Identifiers and scalars: `CellId` = (root token, leaf ordinal) in the canonical flat leaf order (scalar roots
are ordinal 0) — the same cells the emitter's Braun SSA keys on today; `Kind` = {BOOL, INT, FLOAT};
`ConstId`/`BlockId`/`PortId`/`StateSlotId` = ints into the side tables. Root tokens are opaque interned ints
with a name side table for the printer only.

Statements, per executable block, ordered:

- `Def(dst: CellId, rhs: Rhs)` — the one Def form. `Rhs` has exactly three kinds:
  - `RhsConst(const: ConstId)`;
  - `RhsRead(source: PortId | StateSlotId)` — port or state live-in read;
  - `RhsApply(op, args: list[CellId | ConstId], kind: Kind)` — one resolved operator application over leaf
    cells. `op` is drawn from a closed resolved-operator vocabulary aligned with HIR's operator set (float
    arith/compare, integer arith/compare/shift, boolean ops, int→float convert, select), each with a frozen
    operand/result kind signature row; select's kind is a resolved field, never re-derived.
- `Route(moves: list[(dst: CellId, src: CellId | ConstId)])` — the M2 routing record, "result cell i 🠄 source
  cell π(i) or a constant": pure cell steering executed as emitter value bookkeeping, emitting no HIR nodes.
  Routing is deliberately a statement, not an RHS kind, so steering can never carry semantics; its moves admit
  only cell/const references (closed by construction).

Terminators: `TJump(target)`, `TBranch(cond: CellId, then, else)`, `TExit()`. The CFG is executable-only, in
reverse post-order; folded branches arrive as jumps; a kernel that never reaches the exit is refused by the
residualizer (never-returns parity).

Side tables on the closed root `RirUnit`:

- consts: interned `BoolConst | IntConst | FloatConst` (floats as binary64 bit patterns); carrier admission
  (`_carrier_float` policy) happens in the residualizer — the table holds only admissible values;
- ports: ordered per signature — name, kind, defining cell;
- state: ABI order = first-store source order — final rendered slot name (unique by construction), kind, typed
  reset const, live-in cell, live-out cell; only datapath runtime state reaches the table;
- join_kinds: (block, cell) → kind — the only join metadata; the emitter phis exactly the enumerated cells,
  with Braun sealed-block SSA remaining emitter-side (phi arms must come from the final edge set);
- returns: the resolved return contract — none | scalar | ordered cell list, with per-cell kinds, declared
  arity already validated;
- origins: statement → interned origin-frame chain — printer/debug metadata, explicitly non-decision-bearing.

Verifier, recursive over the object graph: closure (every reachable object is an RIR type or a plain scalar;
no Fact, no Py* op, no callable, no registry handle — SC3's enforcement and SC2's object-graph half);
definedness (every read cell written on every path, join cells enumerated in join_kinds, state live-outs and
return cells defined); kinds (every Apply matches its signature row, Route moves are kind-preserving,
join_kinds agree across incoming edges, table rows are internally consistent); ABI (port order matches the
signature, slot names unique). Printer: deterministic text form under the freeze HIR-dump discipline, feeding
the ledger and the canonicalizer's permutation tests.

Every remaining emitter decision family (memo section 2) has a home or an explicit exclusion:

| Emitter surface | Home |
| --- | --- |
| Five copy routines + inline clones (`:631-651`, `:668-711`, `:713-724`, `:726-738`, `:740-755`, `:920-935`, `:973-984`, `:1026-1038`) | Route |
| Routing re-derivations: concat offsets `:631-651`, positional subscript `:941-951`, record projection `:968-972` | Route, offsets resolved by the residualizer |
| Kind re-classification (`_is_bool_fact` `:321-328`, `_sem_of` `:474-480`, `_fact_port_type` `:482-489`, `_fact_sem` `:491-499`, materializer `:762-794`, PyCompare `:849-869`, PySelect `:876-902`) | Kind fields on Apply/join_kinds/ports/state/returns; promotions are explicit convert Applys; promotion-versus-rejection decided by the residualizer |
| Cast policy (`_emit_cast` `:1173-1194`) | identity → Route alias; conversion → convert Apply |
| Intrinsic result rules (`_emit_intrinsic` `:1042-1059`) | pre-lowered Apply chains; the registry is consulted only residualizer-side |
| Power policy (`_emit_power` `:1223-1283`) | pre-lowered Apply chains; both refusal classes move residualizer-side |
| Return-contract tower (`:226-318`, `:1363-1391`, `:1410-1500`) and never-returns (`:365`) | the returns table, resolved; every mismatch/never-returns refusal residualizer-side |
| `_slot_name` refusals (`:522-550`) | final names in the state table; collisions refused upstream; verifier asserts uniqueness |
| `_carrier_float` policy (`:183-196`) | const-table admission at the residualizer |
| Doctrine multi-siting (contains_record ×6, elementwise skeleton ×3) | no RIR construct: residualizer-internal single-sited helpers; the emitter sees only per-cell Defs and Routes |
| Braun SSA and `_exit_identity` port dedup | EXCLUDED: sanctioned emitter value bookkeeping per the agreed definition of "mechanical" |
| Phi nodes | EXCLUDED: join_kinds + emitter Braun SSA |
| Diagnostics in the IR | EXCLUDED: RIR is accept-only; every refusal originates residualizer-side, with parity proven by the negative packet |

`ResidualUnit` surface → RIR (adapter reachability): binding_facts resolve into Def kinds and never cross;
call_plans pre-lower (folded → consts, cast → Route or convert, intrinsic → Apply chains, re-flavoring
conversion → Route + converts, construction → Route + defaults); subscript_plans/route_plans → Route;
store_order/state_livein/state_resets/store_origins → the state table and origins; runtime_state → state-table
membership; store_conversions → explicit convert Applys on the store edge; provenance → finalized slot names
plus origins; executable blocks/edges → the RPO CFG; block_in environments are consumed during residualization
and do not cross.
