# Frontend cleanup

You are driving a cleanup round on the Holoso frontend (`holoso/_frontend/_fir/`). A deep multi-agent review of the
`main..dev` rewrite just concluded: 18 confirmed defects, test/doc rot, and an architecture verdict on whether the
per-construct "whack-a-mole" support can be replaced by something holistic. This document is the plan plus the
carry-over evidence.

## The goal

Cut frontend complexity on two axes at once, without losing genuinely useful capability:

1. **Trim the subset.** The frontend attempts more than it can handle. Some supported Python features carry high
   implementation cost and near-zero observable utility for the kernels this compiler exists to serve.
2. **Consolidate the redundancy.** One semantic concept is currently implemented in three to five synchronized places.
   That duplication is already producing defects, not merely offending taste.

They interact, and that fixes the order: every feature trimmed is a rule that never has to be ported during
consolidation. **Scope first, stabilize second, restructure third.**

## Non-negotiables

**No cybersec framing. None.** This compiler serves trusted, well-meaning users; it is not a security boundary. Corner
cases come from honest mistakes only — a typo, a misunderstood API, an unusual but legitimate pattern. Never harden
against, and never retain a feature or guard because of, adversarial input: planted dunders, hostile metaclasses,
sandbox escapes, mangled namespaces, resource-exhaustion "attacks". This framing has already cost this project
significant meaningless complexity. Take the warning seriously — the review that produced this document *initially
carried 14 findings that failed exactly this filter* and had to be struck at consolidation. The pull toward it is
strong and it survives careful reviewers. Two consequences:

- A defect counts only if a well-meaning user could plausibly trigger it. A kernel that raises on every plain-Python
  invocation, or that nobody would write, is not evidence of anything.
- Invert it into a trim rule: **a guard that exists only to withstand a hostile construction is itself a trim
  candidate.**

**Codex is your co-designer, not a rubber stamp.** Consult it at every significant decision — the subset, the plan,
each stage's design, each review round. Most advanced model, ultra effort. It found defects the review's Claude agents
missed, its architecture consult was the sharpest input received, and it overturned two of this plan's own calls.
Disagree in writing when you disagree; do not skip it.

**Run the `review-loop` skill after every stage.** Not at the end.

**Commit and push after every stage; CI green before the next begins.** Stages are the unit of recovery.

**Test execution.** Delegate the heavy suites (cosim, fuzz, synth, example matrices) to the runner VMs —
`ssh user@cy-hans-reiser.local`, `ssh user@ci-terry-davis.local`, 24 cores / 96 GB each, key-based auth. Light suites
run locally. Never block on a heavy suite locally.

## The plan

Current `dev` is a historical reference, **not** a byte oracle: it is knowingly incorrect and its diagnostics are
nondeterministic. The oracle gets created at the end of Stage 2, not before.

### Stage 1 — Scope

Critically review the frontend for simplification: which supported features cost real complexity and serve no
realistic numerical / DSP / linalg / control / estimator kernel? Decide the subset. Implement nothing yet.

- **The maintainer's directive:** the first candidate for removal is the support for `getattr`/`setattr` calls. That is
  where to begin, not the extent of the round.
- **The utility oracle is concrete**: the 24 kernels in `examples/`, plus what a competent engineer would plausibly
  write in this domain.
- Several defects in the register below are **entangled with this decision** — a trim resolves them by deletion, and
  repairing them first would burn the work twice and hand the restructuring more code to port. Settle their fate here,
  as part of scope, rather than fixing them in advance. Ask Codex explicitly for its per-defect scope ruling; it has
  strong, specific views on which of the 18 should never be repaired.
- Output: a proposal with evidence — cost in concrete sites, utility evidence, blast radius. **The maintainer rules on
  utility, not you.** Discuss and refine the plan with him beforehand.

### Stage 2 — Semantic stabilization, then freeze

Land everything that changes observable behavior, in one stage, so the baseline is honest exactly once.

- Implement every accepted trim as a **documented, located rejection** with an enabled regression test. A trim is never
  a silent behavior change.
- Fix every retained defect, each with a regression test verified to fail before and pass after.
- Reconcile the hygiene register: the parity-skip rot, the stale `DESIGN.md` claims, the three hollow tests.
- **Fix the diagnostic nondeterminism before freezing** (see Traps) and re-enable the seed oracle at
  `tests/test_determinism.py:111`.
- **Then freeze the baseline**: cross-revision Verilog/HIR/port outputs for all 24 examples plus `octave_index`'s
  second format, together with exact rejection class/message/location and the schedule/latency guards.

Exit gate, stricter than CI green: every retained defect has a verified regression; every trim has an enabled
located-rejection test; **no generic `FIR_PARITY_PENDING` marker survives** — each is removed, converted to an enabled
test, or documented as a deviation with a real home. Grep the codebase for `FIR_PARITY_PENDING`.

### Stage 3 — Restructure

The steps below, against the frozen baseline. Every landing is **byte-identical and diagnostic-identical** to Stage 2's
freeze. Trimming shrinks these steps but obviates none of them — the surviving example corpus exercises all of them
(array routing/matmul/transpose in `imu_frame_transform`; records, reshape, reductions and dispatch in
`finite_set_current_controller`; aggregate state, slicing and starred calls in `ekf1_stateful`; inlining in
`iir1_hpf`). A newly discovered semantic defect becomes a **separate baseline-update commit** before restructuring
resumes — never a silent drift inside a refactor.

## Defect register

Reproduced end-to-end unless marked *(agent)* — those were agent-reproduced and adversarially verified; re-confirm
before acting.

| #   | Defect                                                                                                                             | Where                                          | Observable                                     |
| --- | --------                                                                                                                           | -------                                        | ------------                                   |
| A1  | Dedent reparse corrupts multiline string literals in indented kernels; also a bare `IndentationError` on a column-0 docstring line | `_build.py:148`                                | `python=26.0`, `holoso=18.0`                   |
| A2  | Integer-array comparison lowered in **float**; the scalar spelling honestly rejects                                                | `_analyze.py:1237` → `_emit.py:847`            | `2**53+1 == 2**53` → `True`                    |
| A3  | Empty matrix product loses its trailing dimension, so folded shape queries lie                                                     | `_lib/_linalg.py:55`                           | `ndim` `2.0` → `1.0`                           |
| B1  | **Non-type-preserving rebinding is admitted; the store-type policy is unenforced and incoherent**                                  | `_analyze.py:1044-1066`, `:812`                | see below                                      |
| C1  | `_fact_sem` lacks the 0-d unwrap, so 0-d bool casts/compares misclassify and reject                                                | `_emit.py:492`, `:1205`, `:871`                | valid `bool(z)` refused                        |
| C2  | Bool/non-bool comparison admitted by analysis, refused at emission, unlocated                                                      | `_emit.py:871`                                 | `flag == 1` refused                            |
| C3  | Record ports reject inherited dataclass fields (`annotationlib` is own-class-only)                                                 | `_signature.py:148`                            | works inside the kernel, fails at the boundary |
| C4  | `is_array_annotation` false-positives on any class with a class-level `dims`                                                       | `_signature.py:89`                             | adding a field default flips a working kernel  |
| C5  | `np.trace` of an integer matrix folds to float                                                                                     | `_lib/_linalg.py:64`                           | breaks `range(np.trace(M))`                    |
| C6  | First-visit unroll freeze rejects a conditionally rebound loop iterable                                                            | `_analyze.py:1808`                             | ordinary Python refused                        |
| D1  | Nested `None` return contract hits an internal assertion                                                                           | `_signature.py:179`, `_emit.py:262`            | `AssertionError`                               |
| D2  | Typo'd annotation escapes as a raw `NameError` under PEP 649                                                                       | `_build.py:162`                                | raw traceback                                  |
| E1  | Stub rejections name compiler-internal lines; `SynthesisError.location` never populated                                            | `_analyze.py:2101`, `_analysis_support.py:100` | `matmul_:47:8`                                 |
| E2  | F-string `raise` messages degrade to a bare `'raise'`                                                                              | `_build.py:535`                                | message lost                                   |
| E3  | Inadmissible state reset → unlocated generic join message *(agent)*                                                                | `_analyze.py:1041`                             | `C.step:0:0:`                                  |
| E4  | Reduction-stub misuse reports the matrix-product diagnostic *(agent)*                                                              | `_analyze.py:2086`                             | `np.max(m, 0)`                                 |
| F1  | StaticFor materializes trip children before the fuel check, quadratically *(agent)*                                                | `_analyze.py:1836`                             | ~30 s on a 32k table                           |
| G1  | Guarded-region predication never fires for the natural spelling *(agent)*                                                          | `_hir/_if_convert.py:162`                      | two chained muxes                              |

### B1 in detail — the ruling, and the trap

Variables are strongly typed, so **a store that changes a variable's type must be a located rejection at the store
site**. Nothing enforces this. The state-store transfer checks only the bool/non-bool boundary and the int→float
widening; `StorePlace` for locals is a bare `env.set` with no kind comparison at all. The codebase is known to contain
unnecessary handling for variable-changing rebindings that needs to go. One policy question, four outcomes:

| case                                         | today                                            |
| ---                                          | ---                                              |
| `int` slot ← float, float reaches the exit   | **silently accepted; the slot changes type**     |
| `int` slot ← float, int restored before exit | rejected at *emission*, unlocated                |
| `bool` slot ← float                          | rejected at the *W/D join*, unlocated, `K.u:0:0` |
| locals, incl. runtime `float`→`bool`         | silently accepted                                |

An earlier review round misread the emission refusal as a spurious rejection. It is *correct* — merely late, unlocated,
and contradicted by the sibling case that accepts.

**The expectation, calibrated.** Enforcing this materially simplifies storage/state typing, but only moderately
simplifies the frontend overall — treat the win as real but bounded, and establish what it actually deletes before
banking on it. The shape that pays: a **fixed storage schema** (scalar kind, or aggregate flavor/geometry plus per-leaf
kinds). Known-vs-Residual and Python-vs-NumPy provenance are *not* types. An unbound source variable's first definition
establishes its schema; independent first definitions may join, including int→float promotion. Once established: bool
accepts bool, int accepts int, float accepts float or int (conversion on the store edge). Everything else rejects at
that store. That deletes `_leaf_kind()` (`_analyze.py:1071`), live-in-derived state typing and stored-value-driven
array dtype rebuilding (`:572`), `_leaf_is_int()` (`_emit.py:574`), and the separate scalar/aggregate state-store
policy towers (`:528-604`, `:1022-1066`). W/D survives, tracking reachability and residuality rather than changing slot
type.

**Three ways to get this wrong:**

- **Do not apply it to every `StorePlace`.** That op also implements conditional-expression merge sinks
  (`_build.py:641`), comprehension accumulators (`:762`), and `ReturnPlace`. Give stores explicit roles; only
  source-variable and state stores enforce rebinding.
- **Do not implement it as another fact-kind check in the transfer.** It cannot resolve against the *first* fact seen
  (that falsely rejects a legal int/float phi) nor against only the *final* value fact (that misses an illegal
  loop-carried type change). Use a separate monotone schema flow, record store-origin mismatch obligations, and resolve
  them after W/D/schema stabilization — otherwise predecessor discovery (`:760`) makes the result order-dependent.
- **Do not cross into merges.** This is about *rebinding*, not the documented C-style int→float promotion at phi/select
  arms and comparison operands, nor expression-level mixing. Widening preserves a variable's type; it does not change
  it. Keep `_float_promoted`/`join_facts`, the phi/select/comparison coercions, explicit casts, return conversion and
  mixed arithmetic. `x = 0; x = input_float` rejects; `x = int(v) if c else v` stays a legal float phi. In
  `DESIGN.md:585` remove only "state-leaf join"; retain float-slot store promotion at line 583.

## Hygiene register

| #   | Item                                                                                                                                                                                                                                                                                                                                                                                                                                |
| --- | ------                                                                                                                                                                                                                                                                                                                                                                                                                              |
| H1  | ~85 stale `FIR_PARITY_PENDING` skip markers behind a registry pinned empty. Measured: the affected files run *679 passed / 83 skipped*; strip the markers and they run *727 passed / 35 failed* — **48 of 83 pass the moment they are enabled**. Of the 35, most are stale expectations (12 alone are the now-strict list-vs-array return contract), but they conceal E1, H2 and H3.                                                |
| H2  | A real capability gap hides behind a skip reason naming already-shipped features: `test_verify.py:454` blames "aggregate indexing/slicing and list return"; the actual blocker is a starred element in a list *display* (`[v[2], *head]`), which `DESIGN.md` never states is unsupported.                                                                                                                                           |
| H3  | A deliberately parked deviation filed under the same token: `bool(2.0**-200)` folds `True` on float64 where the ZKF-encoded datapath says `False` (`test_verify.py:1501`). The skip reason is honest about the trade-off — this is **not** a stealth regression — but it needs a ruling and a home in `TODO.md`/`DESIGN.md`, not a skip marker under a token `DESIGN.md:451` declares empty.                                        |
| H4  | `DESIGN.md` contradicts the implementation and itself: `:345` says array parameter ports "await the ndarray stages" (live, and contradicted at `:419-425`); `:377` says `@` awaits the matmul stage (shipped); `:271-283` misstates the analyzer/emitter contract; `:526` presents G1 as working. The front-end section is also near-unmaintainable prose — 15-line sentences, a "deferred" subsection describing shipped features. |
| H5  | Three tests assert less than they claim: `test_fir_analyze.py:209` (satisfied by the parameter, not the state leaf it names), `test_fir_builder.py:58` (order-blind; a first-write-wins regression passes), `test_frontend_control.py:47` (`caplog` targets the deleted `_lower` logger; passes only via a pytest side effect).                                                                                                     |

## The architecture verdict

The question — can the per-construct whack-a-mole be replaced with something holistic and extensible? — went to two
independent panels (four Claude proposers plus a judge; two Codex consults plus a referee). They converged, and **every
candidate replacement was rejected by the agent assigned to advocate it**: tracing dies on residual control flow (13 of
the 24 examples branch; tracing both arms rebuilds the abstract interpreter you already have); a unified
capability-gated lattice provably reconverges onto what `_fir` already is; bytecode replaces `_build.py` — the part
that works — and adds a per-release opcode tax. The verdict:

> The semantic distinctions are essential in cardinality; their current cross-layer duplication is incidental in
> multiplicity.

So: **no replacement.** Keep `AST → generic FIR → SCCP/W-D → typed resolved unit → mechanical HIR`. The one new idea is
the middle box: a post-stabilization, aggregate-aware, origin-carrying typed plan where every leaf is exactly one of
`FactOnly | Const | Copy/Routing | ScalarOp | Select | Cast`, and emission becomes mechanical — importing no `Fact`, no
registry, no `Py*` op, making no user-facing decision. That invariant makes the A2/B1/C1/C2 *class* of defect
unrepresentable rather than merely fixed. Do not skip that framing: three of the four surviving majors are the same
structural cause, which is the whole argument for doing this at all.

## Restructuring steps (Stage 3)

1. **Routing algebra.** One concept — "result cell *i* := source cell π(*i*), or a constant" — is spelled nine ways:
   four id-keyed analyzer side tables reset each W/D round (`_analyze.py:307-310`), two `CallLowering` variants,
   `subscript_plans` + `route_plans` (`:274-275`), five near-identical emission copy routines (`_emit.py:621-745`) and
   four inline clones. Two routings are not recorded at all and emission re-derives them (`_emit_concat` re-walks
   layouts; the `PySubscript` fallback re-runs `operator.index` at `_emit.py:954-964`) — the scan/lowering duality the
   rewrite was built to kill, surviving at op granularity. Collapse to one typed record and one consumer.
   **Correction:** do *not* key it universally by `dst` — `StorePlace`/`PyStoreAttr` have no destination binding
   (`_ir.py:393-398`) and merges work on leaf cells. Key by plan site/cell. Specify the full schema before populating
   the first slice.
2. **Typed dispatch rows.** Convert `_expand_call`'s 16 `target is X` arms (~390 lines) and the 140-line `PyAttr`
   ladder into ordered frozen row tables in `_fir`, messages moved verbatim. Fixed tuples with identity comparison, not
   a dict — unhashable shadow call targets are supported (`:2161`). **First-match priority, not global
   matcher-uniqueness**: the rules intentionally overlap. Mandatory correction: record-subject `isinstance`
   (`:2217-2244`) deliberately *precedes* and bypasses generic `admit_call`; running admission first would reject the
   record leaves that arm exists to ignore. `_fold.py`'s own docstring (lines 12-14) already promises this refinement.
3. **Single-site the doctrines.** Zero-dim guards ×6 (three carrying the *identical* message), `contains_record` ×5,
   bool-arithmetic ×4, and one elementwise skeleton written three times. Not a global rejection predicate but a
   *use-specific* consumption operation — e.g. 0-d arrays reject for `len`/indexing/iteration while deliberately
   scalarizing for binary ops, comparison and casts. That missing distinction is C1.
4. **Post-W/D residualizer.** `_finalize()` today *replays* `_transfer()` (`:606-636`), potentially repeating host
   calls. Record evidence atomically with the fact; never residualize after an optimistic inner visit; `CallPlan` is
   erased, not elaborated. Only then retire the 41 user-facing `EmissionRejection` sites — right endpoint, wrong first
   move.
5. **Structured origins.** `Origin` carries function/line/column only; inlining stores frames inner-to-outer while
   rendering reads `origin[0]` — that is E1, and it also lets state first-store ordering (`:615-640`) reverse ports
   when a root call inlines a callee defined later. Primary location is `origin[-1]`; keep the inner chain as context.
   Settle the *observable* origin semantics in Stage 2 (user call site primary, `SynthesisError.location` populated,
   deterministic state-port order); this step is then a mechanically behavior-identical datatype migration and may
   safely come last.

Explicitly rejected: fusing `Known` and `Reference` (the identity-keyed non-data firewall at `_fact.py:221-240` is
load-bearing — refine `Reference` with closed tags instead); the token/proxy substrate; probe-derived routings (defer
until more structural constructs are actually scheduled).

## Traps

- **"Behavior-neutral by construction" is false today.** SCCP decisions change as executable predecessors join
  (`:760-773`); cast reclassification is acknowledged in-code (`:648-651`). **A first-visit routing or row winner is
  unsound.** Last-visit evidence cannot build phis either: a new executable edge whose joined environment is unchanged
  need not requeue the successor, so phi arms must come from the *final* edge set and `executable_edges`.
- **Determinism, precisely.** Emitted Verilog *is* byte-stable — verified across three `PYTHONHASHSEED` values on six
  examples including `ekf1_stateful` and `uart_rx` — so byte-gating is sound. What is *not* stable is diagnostics:
  `_Env.join_with()` iterates an unordered place union (`:228`), and `_finalize()` iterates `executable_blocks` as a
  set (`:616`) with a state union at `:643`. A kernel with two competing rejections can report either. That matters
  because ~377 pinned `match=` assertions and the frozen rejection messages are part of your gate. Do not conclude the
  compiler is nondeterministic; do fix the ordering before the freeze.
- **The examples are the contract.** 24 `ExampleSpec` rows, but 25 spec/format cases — `octave_index` carries two
  formats (`tests/_examples.py:517-527`). Gate on all of them.
- **`_analyze.py` is 2 498 lines**, over the ~2 000 soft limit even after the support split. The trims are the cheapest
  way back under it; do not solve this by splitting the file again.
- **"No example uses it" is not the whole test.** Necessary evidence, not sufficient — ask also whether the feature is
  the natural spelling of a common idiom in this domain. Where the two disagree, the maintainer rules. And separate a
  feature from its expensive part before cutting: the useful spelling and the machinery that makes it faithful in every
  corner are often different things, and frequently only the second needs to go.
- **What the review did not cover:** cosim/fuzz/synth were not re-run (green in CI at `8a37a82`, expensive); LIR and the
  backend got only a diff-level pass; the restructuring recommendation is a design judgment from two panels, not a
  prototype.
