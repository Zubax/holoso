# TODO

## Integer support adjacent (the integer wiring milestone)

Scalar-family policy: `PortConditioner` is a closed `FloatSignControl | BoolInversion` union enforced on every
MIR port (`_operators.py`); add an int conditioner (likely identity/no-op) + a scalar-family table owning
conditioner/bank/coercion/reset/lowering hooks.

The oracles store wide values as `FloatValue` (`numerical.py`, `_mir/_interpret.py`); introduce a
`FloatValue | IntValue` wide-value union, a typed `lir.wide_consts` pool (constants are float-encoded across
microcode/emit/html/model today), and one shared scalar port codec (cocotb + model duplicate it).

Strength reduction is float-keyed (`cval: dict[ValueId, float]`); add a sibling int reduction + a typed-constant
cache when int lands.

Also queued for this milestone (see DESIGN.md "Integers"): collapse the `pow`/`np.power` spellings onto `**` with an
integer `np.power` overload, and give `np.sign` an integer lowering in place of its analyzer rejection.

## Known defects needing resolution

Empty contractions diverge from numpy in the linalg stubs: `(n, 0) @ (0, m)` and an empty vector dot reject
(stub-internal index error) where numpy returns zeros, and an empty product's dtype collapses to float64.
Revisit together with the planned trace/outer/dot examples; the stubs' guards otherwise keep the reachable
domain faithful.

Call deferral can falsely reject a synthesizable kernel (the analyzer's optimistic-SCCP fixpoint meets
destructive mid-round call grafting). When a call cannot resolve on a visit because a store-schema violation is
transiently pending in scope -- typically an int/float merge into a float state slot whose Known-int arm is
momentarily inexact before the fixpoint promotes it -- its result is momentarily `Unbound` while the fixpoint
continues around it. The block's terminator still publishes out-edges, so the unbound result reaches the
successors; the graft-time retraction that unwinds those edges reaches only one edge deep, so a transitive
successor keeps the phantom-unbound environment and a later join reports `local '...' may be unbound here` at
an innocent line. The one-edge-deep retraction runs at a graft, but the false rejection does not need one: an
`np.array`-only kernel, whose call is a conversion that never grafts, produces the identical rejection. Any
deferred call reaches this.

A second mechanism was fixed rather than documented, because its residue included a SILENT MISCOMPILE. A
condition is evaluated on a fact that later becomes more precise, so BOTH arms are marked executable; marks are
add-only, so the arm the stabilized facts later prove dead stays open and is emitted. It has (at least) TWO
producers, which is why no check on the condition's operand can close it: `_truth_fact` mapping an unbound
operand to a runtime bool, and -- with every fact legitimately bound throughout -- a state read whose live-in
join settles `Residual -> Known` across visits. The first accounts for the large majority of observed cases and
for the miscompile; the second is invisible to any unbound-operand guard. A store on that arm promotes an
attribute from a read-only constant -- folded at binary64 -- into a runtime state slot whose reset is
materialized in the narrower target carrier, so a guard
reading it can flip: a kernel returning 10.0 in Python returned 20.0 in hardware, no error raised, and only in
carriers too narrow to hold the reset exactly (the measured flip point for that witness is E8M31). The stale
marking also reached the plan replay as a bare `KeyError`, `rank` being built from edge-REACHABLE blocks while
the replay iterates the add-only MARKED set. A companion defect surfaced as a raw `RuntimeError` out of HIR
emission: there the graft's mark removal leaves the orphaned block's own out-edges standing, so a successor
keeps a predecessor that never runs and its phi has no arm for it. This mechanism needs only a DEFERRED call,
not a graftable one: `np.array` is a conversion and never grafts, yet drives the witnesses.

The fix is a POST-STABILIZATION GATE and nothing else: the fixpoint's speculation is left exactly as it was,
and the unsound RESULT of it is refused once the facts are stable. Three contradictions between recorded
reachability and the stabilized facts are checked -- a branch whose settled condition disagrees with its own
recorded out-edges, a block marked executable that no executable edge chain reaches, and an edge out of a block
left unexecutable. Each becomes a located refusal instead of a wrong answer or a bare crash. Checking the
RESULT rather than any producer is what makes it complete: it catches the state-live-in producer above, which
an operand-level guard cannot see.

Two more invasive fixes were built and measured, and BOTH were rejected on evidence. Refusing at the producer
(the truth of an unavailable value defers, and an unbound `Branch` opens no edge) fixes the miscompile and
prunes the dead arm properly -- the strictly better outcome where it applies -- but it starves the fixed point:
a `Branch` inside a loop body sits BEFORE the body's trailing back-edge `Jump`, so deferring it stops the loop
re-flowing at all. That is the same failure round-10's edge withholding produced, reached by a different route,
and it is why `tools/deferral_seam_sweep.py` exists: the seam cannot be judged by argument, only by moving
counts. What it reports, and NEITHER design dominates:

| family | pre-gate | producer fix | narrowed gate | gate (shipped) |
| --- | --- | --- | --- | --- |
| dead_arm (54) | 42 accept, 8 crashes | 54 accept, 0 | 36 accept, 0 | 28 accept, 0 |
| loop (30) | 21 accept | 21 accept | 21 accept | 21 accept |
| loop_inner (6) | 5 accept | 4 accept | 5 accept | 5 accept |
| value oracle (5) | all miscompile | all correct | 2 MISCOMPILE | all refused |

The producer fix compiles 26 more dead-arm kernels than the shipped gate, and compiles them CORRECTLY rather
than refusing them -- it prunes the dead arm instead of detecting it afterwards, which is the better outcome
wherever it applies. It was rejected because its one loss is a starved fixed point: a kernel that lowered
before it stops lowering, and a regression of working code is worse to carry into a freeze than a refusal,
the same reasoning that reverted round-10. The narrowed gate bought back 8 accepts and paid with two silent
miscompiles, which is why the shipped rule is unconditional.

Two families exist because the obvious corpus could not see either regression. `loop_inner` puts the deferring
call INSIDE the loop, where withholding an edge actually costs something; the plain `loop` family reports
identical numbers for the producer fix and the baseline. The value oracle compares accepts against Python,
because the accept/refuse/crash alphabet tallies a wrong answer as a good accept -- which is exactly how the
narrowing regressions passed a green sweep. Note a refusal proves nothing about values, so the tool says how
many oracle kernels were refused rather than counting them as passes.
Retracting the stale mark is not local either:
destructive environment joins mean removing an edge requires recomputing downstream environments, schemas,
reachability, W/D discoveries, and phis. The gate costs accepts, and the refusal is deliberately
UNCONDITIONAL anyway: any settled branch contradicting its own recorded edges refuses, with no attempt to
judge the speculated arm harmless. Two narrowings were tried to recover those accepts and BOTH reintroduced
silent miscompiles. Testing only arms that store misses two shapes: an inert arm that poisons the merge phi so
a DOWNSTREAM guard's store promotes, and an arm where no store exists anywhere and the phi ALONE rounds an
inexact constant. Scoping the test to the arm's exclusive region additionally makes the check vacuous wherever
the branch reconverges through a loop back-edge, since the dead arm is then within the live arm's reach.
Deciding which arm is harmless needs exactly the reachability this gate exists because the analyzer got wrong,
so it does not try. All three witnesses are pinned, and the sweep's value oracle covers them.

A THIRD ROUTE ran through the accumulated runtime-state set rather than the graph. `W` grows monotonically, so
a leaf discovered on a round whose store a LATER round proves unreachable stays runtime state even though the
stabilized graph is entirely correct -- and its reset then materializes in the carrier instead of folding at
binary64. Every per-graph check passes; nothing is wrong with the final CFG. The gate now additionally requires
each retained runtime leaf to have a store in the final executable graph.

A FOURTH ROUTE runs through the OTHER half of the same accumulator. `W` is now guarded; `D`, the live-in map,
is not. A round-1 speculated arm drives a live-in down to residual, round 2 prunes that arm so the stable graph
is impeccable, and a trailing store keeps the leaf in `first_store` so the runtime-state check also passes. The
guard then reads the poisoned live-in and takes a branch Python never takes. Mirroring the `W` check does not
close it: `W` staleness leaves a residue to detect, while the poisoned `D` at stabilization is byte-identical to
what the final round would derive fresh. Pinned as `test_live_in_poisoning_miscompile_is_still_open`.

The check for route (3) is WEAKER THAN IT LOOKS: it requires every retained runtime leaf to have a store in the
final executable graph, and a trivial `self.s = self.s` satisfies that. Delete the self-assignment and the
kernel is refused; keep it and the same miscompile goes through. So (3) is closed only in the spelling where no
store survives, and the route is open in general -- pinned as `test_self_assignment_defeats_the_runtime_state_check`.

FOUR ROUTES, FOUND BY FOUR DIFFERENT ATTACK ANGLES, EACH NEW ANGLE FINDING ONE. That is the honest prior: the
defect density here is bounded by the attacks tried, not by the code, and no claim of soundness should be made
about this seam. Only ONE route is FULLY refused (the settled-branch one). The other three REMAIN OPEN AND
SILENTLY MISCOMPILE: the phantom-environment one, the live-in poisoning one, and the runtime-state one, whose
check refuses it as first written but which one line of ordinary Python reopens. All three are pinned as tests
asserting the wrong value they currently produce, and `tools/deferral_seam_sweep.py` carries a value-oracle
entry for each. Two review halves
agreeing has repeatedly meant less than it appeared to, each time because they shared an attack surface: the
within-round fuzz could not express the cross-round route at all. Named untested surfaces, for whoever comes
next: `_unroll_seeds` and `_pending_bridge`.

No further gate checks should be added. The gate went from one check to four, three of them reactive, and each
addition was followed by a new route through a dimension it did not model. The fix is the restructure.

SILENT MISCOMPILES REMAIN, and they are the most serious open defect in the compiler. This one the gate cannot
structurally see. When the phantom environment keeps a stale state fact alive, the condition
that reads it settles as a RUNTIME bool rather than a constant, so the arm that is dead in Python is genuinely
live as far as the analyzer can tell: there is no contradiction between recorded reachability and the settled
facts, and nothing for the gate to detect. That arm's store still promotes the attribute to runtime state, and
the reset still rounds in the carrier. `test_phantom_environment_miscompile_is_still_open` pins a witness
returning 12.0 in Python and 22.0 in E8M23 hardware, correct in E11M52.

The gate closes the routes where the recorded reachability visibly contradicts the settled facts, which is
where the majority of observed cases and all three raw-crash modes lived -- a bare `KeyError` out of the plan
replay, a bare `AssertionError` (which, vanishing under `-O`, crashed the debug build where the optimized one
explained), and a raw `RuntimeError` out of HIR emission. It does not close the class: on the
surviving route the condition settles too, but as a RUNTIME bool, so there is no contradiction to see.
Detecting that locally is not possible in principle -- the analyzer cannot know a fact "should" have been more
precise without the correct environment, which is exactly what the phantom edge denies it.

Besides that, what remains is FALSE REJECTIONS in two flavours: the phantom-environment ones
(`may be unbound here`), and the ones the gate itself produces on kernels whose speculated arm it cannot prove
harmless. Those are honest located diagnostics.

Witness shapes are pinned executably in `tests/test_frontend_state.py`
(`test_graftable_call_deferral_false_rejection_witnesses` and its siblings), covering the open shapes and the
gate's refusals -- the miscompile, the rank-walk shape that used to raise a bare `KeyError`, and the
side-effect shape that used to raise a raw `RuntimeError` out of HIR emission. They live in code rather than in
prose here because prose transcriptions of them rotted --
dropping the both-arms read or the wide-int feed makes a shape silently vanish, which is exactly what happened
to two of the four kernels this section used to carry.

Patching the seam in place has been tried and abandoned. Rounds 6-11 (`docs/campaign.md`) each traded one corner
for another, and the round-10 attempt -- withholding a deferred graftable call's terminator edges -- regressed
valid code: it starved the outer state fixed point when the withheld edge was a loop body's only successor,
turning a kernel that synthesizes into a false "not exactly representable" refusal. That attempt was reverted,
so the baseline is regression-free relative to the state before it; the starvation kernel is pinned as a
passing acceptance test (`test_deferred_graftable_call_does_not_starve_the_state_fixpoint`) so the trap cannot
be re-entered. The gate above is the deliberate exception to that stop-rule: a silent wrong answer is a
different category from a false rejection and could not be left standing. It earns the exception by changing
NOTHING about the fixpoint -- it only refuses an already-computed result whose reachability contradicts its own
facts -- which is why it cannot starve anything. The two fixes that did touch the fixpoint were both built,
measured, and rejected; that history is above, and it is the reason this one deliberately does not.

The class is the one the post-stabilization resolution-totality restructure (`docs/decisions/arch-memo.md`, the
resolved-IR spike) dissolves by making residualization a total pass after the fixpoint rather than interleaving
it with the fixpoint; when that lands, the witness tests flip to asserting synthesis and are gated
byte-for-byte against `freeze-1`. THE SURVIVING MISCOMPILE IS A FIRST-CLASS ACCEPTANCE CRITERION FOR THAT WORK:
the restructure is not done while `test_phantom_environment_miscompile_is_still_open` still reports 22.0.
Two further obligations on it, both load-bearing: it must be checked
against BOTH mechanisms, since residualizing after the fixpoint removes phantom-unbound environments directly
while stale executable marks are a separate concern; and its resolved spine must RECOMPUTE stable reachability
and typing from the stabilized facts rather than inherit today's executable sets and `block_in`, because a
spine built over a stale executable set would reintroduce the dead-arm behavior -- including the live register
write into a public port -- underneath a gate that is checking the old representation.

## Deferred capability gaps

Two public state slots sharing a live-out refuse at Verilog emission when the schedule has reused the
boundary-installing slot's register mid-transaction (e.g. `self.a = x + self.a` twice, then `self.b = self.a`;
the front-end, HIR, MIR, and LIR all accept it). The refusal is an honest `SynthesisError` naming both slots;
lifting it needs an install-copy capability -- an extra boundary-adjacent copy step (or a reserved home for a
shared live-out) so one value can install into several slot registers.

Tuple-valued state attributes reject at the reset ("state attribute ... has an unsupported reset type"):
aggregate persistent state covers flat lists of scalars and nonempty 1-D/2-D plain ndarrays only, so the
delay-line idiom must be spelled with a list -- `self.window = [self.window[1], x]` lowers where the tuple
spelling refuses.

## Test-coverage debt

The differential fuzzer sums multiple result lanes into one float return even though tuple returns lower;
restoring tuple-return lanes would perturb the tuned campaign seed streams, so it is queued as separate
coverage work together with re-tuning the seeds.
