# B1 design note — the fixed storage schema (for Codex consult X2, lands at S2.12)

Ruling (HANDOFF.md, maintainer-final): variables are strongly typed, so a store that changes a variable's type
is a located rejection at the store site. Widening at merges is untouched: the C-style int→float promotion at
phi/select arms, comparison operands, explicit casts, return conversion, and mixed arithmetic all stay.
`x = 0; x = input_float` rejects; `x = int(v) if c else v` stays a legal float phi.

## The schema flow

A separate MONOTONE flow beside the fact fixpoint — never a fact-kind check inside `_transfer` (resolving
against the first fact seen falsely rejects a legal int/float phi; resolving against only the final fact misses
an illegal loop-carried type change).

- Storage schema per root place: `ScalarSchema(kind)` for scalars; for aggregates, flavor + geometry + per-leaf
  kinds (the reset already fixes state schemas; a source variable's first definition establishes its schema).
- Establishment: an unbound place's first store establishes; independent first definitions on different paths
  JOIN, int→float promoting (schemas descend the same lattice direction facts do — establishment is monotone).
- Acceptance per store once established: bool ← bool; int ← int; float ← float | int (the conversion rides the
  store edge, exactly like today's float-slot store promotion, DESIGN.md:583). Everything else is a rejection
  AT THE STORE, located by the store op's origin (the E3 machinery from S2.5 — `_store_origins` — generalizes:
  local stores get their op origin directly).
- Obligations: because SCCP discovers executable predecessors late (`_analyze.py:760-773` area), a store
  observed before the schema stabilizes records a (place, kind, origin) OBLIGATION rather than raising;
  obligations resolve after W/D + schema stabilization, in deterministic order (block index, op index). A store
  that widened the schema itself (the establishing join) is not an obligation.

## Store roles (the trap HANDOFF flags)

`StorePlace` also implements compiler-internal sinks that MUST NOT enforce rebinding:
- conditional-expression merge sinks (`_build.py` ifexp desugar; synthetic `ifexp@{line}` bindings),
- comprehension accumulators (`_build.py` comprehension frame),
- `ReturnPlace` (validated by the return contract instead).
Give stores an explicit role at BUILD time (an enum field on StorePlace, or a synthetic-place predicate — the
builder knows which places it minted). Only SOURCE-VARIABLE stores and STATE stores enforce. Known-vs-Residual
and Python-vs-NumPy provenance are NOT types — the schema sees SemType kinds only.

## What it deletes (bank before building)

`_leaf_kind()` (analyzer), the live-in-derived state typing and stored-value-driven array dtype rebuilding
(state-store tower), `_leaf_is_int()` (emitter), the separate scalar/aggregate state-store policy towers.
W/D survives, tracking reachability and residuality only — never changing a slot's type. DESIGN.md:585 loses
only "state-leaf join"; :583 stays.

## Regression matrix (each verified to fail before, pass after)

1. int slot ← float, float reaches the exit: was silently accepted with the slot changing type → rejects at the
   store, located.
2. int slot ← float, int restored before exit: was an unlocated emission refusal → rejects at the store, located.
3. bool slot ← float: was an unlocated W/D-join message (`K.u:0:0`) → rejects at the store, located.
4. Locals, incl. runtime float→bool rebinding: was silently accepted → rejects at the store, located.
5. Calibration keeps: `x = int(v) if c else v` (legal float phi); float slot ← int (store-edge conversion);
   independent first definitions int/float joining to float; state resets establishing aggregate schemas;
   the B1-proof stubs (`trace_`'s data-seeded accumulator) still lower.
6. Determinism: obligations resolve in one order; the competing-rejection seed test extends with a two-bad-store
   kernel.

## Sizing guards

`_analyze.py` line count must not exceed its pre-step count (the deletions offset the flow); schema helpers live
in `_analysis_support.py`; no new file. The trims (S2.7-S2.10) land first so the schema never covers deleted
constructs (masks, 0-d, isinstance, enum provenance, str methods).

## Questions for Codex X2

1. Obligation resolution: is post-stabilization batch resolution sound against unroll-reseed restarts (seeds
   change executable shapes between rounds — obligations must reset per round like the other side tables)?
2. Aggregate schema representation: reuse the existing layout algebra (flavor + geometry + leaf kinds) or a
   reduced per-leaf `SemType` tuple? The reset-driven state schema suggests the latter suffices.
3. The establishing join across an if/else where BOTH arms first-define with different aggregate flavors (tuple
   vs list): today's join degrades to StructuralLayout — does the schema adopt the degraded flavor or reject?
4. Any store-role we missed beyond ifexp sinks, comprehension accumulators, ReturnPlace (walrus targets are
   source variables; unpack projections write source variables; loop counters?).

## Codex X2 record (consulted ahead of schedule; session in the campaign workspace)

Verdict: design sound with amendments. Implementation order adopted (7 steps, each with its regression): (1)
StoreRole tagging at every constructor; (2) pure schema algebra in _analysis_support; (3) the separate CFG
schema solver with root-parameter seeding; (4) site-keyed obligations with round reset and deterministic
resolution; (5) finalized per-store coercion plans, then cut local emission over (facts and SSA cell types must
not diverge); (6) state schemas derived from the reset, then delete _admit_state_store/_leaf_kind/_leaf_is_int/
dtype rebuilding; (7) calibration + deletion gate (grep empty, LOC gate).

Decisions resolved (mine, from X2's missing-decision list):
- del does NOT erase a schema: variables are strongly typed for the function's lifetime, so
  `x = 0; del x; x = 1.0` still rejects.
- Non-SemType stores (references, None, strings, ranges, slices) are fact-only: they neither establish nor
  violate a schema — the schema covers datapath kinds only.
- No op mutation for plans: side tables reset per round (root instantiation shallow-copies op lists).
- Obligation/violation report order: the first executable store in CFG preorder (the S2.3 Fail-walk precedent),
  not lowest block index.

Test-reversal inventory beyond the note (X2): scalar annotation does not override an integer reset
(test_fir_differential.py:403-420); loop-counter tests deliberately rebinding int counters to floats
(test_frontend_state.py:1515-1564) need type-preserving rewrites that keep their stale-fact coverage;
the int-reset-array ← float acceptance at test_matrix.py:1716-1734 inverts.
