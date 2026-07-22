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

The deferral/grafting seam is CLOSED as of the fresh-resolution restructure (M5). What follows is the record of
what it was, because the reasoning is the reason the fix took the shape it did.

The analyzer's optimistic-SCCP fixpoint met destructive mid-round call grafting. When a call could not resolve on
a visit because a store-schema violation was transiently pending in scope, its result was momentarily `Unbound`
while the fixpoint continued around it -- and the block's terminator still published out-edges, so the unbound
result reached the successors. Graft-time retraction unwound only one edge, so a transitive successor kept the
phantom-unbound environment and a later join reported `local '...' may be unbound here` at an innocent line. A
condition computed from such a result read as a runtime bool, so BOTH arms were marked executable; marks were
add-only, so an arm the stabilized facts later proved dead stayed open and was emitted. That was unsound rather
than wasteful: a store on the dead arm promoted an attribute from a read-only constant -- folded at binary64 --
into a runtime slot whose reset materializes in the narrower target carrier, so a guard reading it flipped. A
kernel returning 10.0 in Python returned 20.0 in hardware with no error raised, and only in carriers too narrow
to hold the reset exactly (the measured flip point for that witness is E8M31).

FOUR ROUTES WERE FOUND BY FOUR DIFFERENT ATTACK ANGLES, each new angle finding one -- the honest prior being that
the defect density was bounded by the attacks tried, not by the code. The post-stabilization gate that preceded
M5 fully closed exactly one of them, the settled-branch route where recorded reachability visibly contradicts the
settled facts. The other three were beyond any local check: the phantom-environment route settles its condition
as a genuine runtime bool, so no contradiction exists to detect; the runtime-state route's check asked only
whether SOME store survived, which one line of ordinary Python (`self.s = self.s`) satisfies; and a live-in
driven residual by a speculated arm that a later round prunes is, at stabilization, byte-identical to what a
fresh derivation would produce, leaving no residue at all.

All three are closed by construction rather than by a fifth check. The analysis now restarts its whole state
descent -- W, D and the store-obligation bridge -- whenever a round changes the graph, so the round that
stabilizes has re-derived reachability, environments, binding facts, schemas, W and D from entry facts and reset
state alone; and W is derived from state EFFECT (does the transaction leave the leaf holding something other than
its snapshot?) rather than by counting stores, so an identity write promotes nothing. A visit that cannot
complete publishes no environment, which removes the phantom edges at their source. The gate's four rules are now
invariants the resolved pass establishes, asserted rather than diagnosed.

`tools/deferral_seam_sweep.py` remains the standing check. Its `_KNOWN_OPEN` table is EMPTY: every value-oracle
kernel is now checked against Python with nothing tolerated, so any regression fails the tool outright. Two
review halves agreeing has repeatedly meant less than it appeared to, each time because they shared an attack
surface: the within-round fuzz could not express the cross-round route at all. Named untested surfaces, for
whoever comes next: `_unroll_seeds` and `_pending_bridge`.

No further gate checks should be added. The gate went from one check to four, three of them reactive, and each
addition was followed by a new route through a dimension it did not model. The fix is the restructure.

THE SILENT MISCOMPILES ARE CLOSED, and each witness now asserts the Python answer:
`test_phantom_environment_no_longer_keeps_a_stale_gate` (12.0 in both carriers),
`test_live_in_poisoning_does_not_survive_a_pruned_arm` (10.0 in both), and
`test_self_assignment_does_not_fabricate_runtime_state` (10.0, with no `state_s` port). Two further routes closed
with them: `test_runtime_state_discovered_on_a_dead_round_does_not_survive_it`, and the loop-carried acceptance
kernel that used to be the guard against over-eager fixes, which measurement showed was itself emitting
9007199254740992.0 against Python's 9007199254740994.0 on its PRIMARY output and is now a located refusal at the
store that cannot be represented.

The milder wrong-LINE residual of the same seam is closed too: a state verdict raised before its leaf had a
promotion-latch entry took its location from whatever stores the worklist had reached, speculated arms included,
so a refusal that was itself correct could name a line Python never runs. Dead arms are not walked now, so
`test_a_mid_round_verdict_anchors_on_a_store_that_runs` and
`test_a_verdict_names_the_live_store_not_the_raise_guarded_one` both name the store that executes.

What the restructure had to satisfy, and did, is worth keeping: it was not enough to remove the phantom-unbound
environments. Stale executable marks were a separate concern and the two accumulator routes a third and fourth,
invisible on a final graph that is entirely correct -- so the resolved pass had to RECOMPUTE all four inherited
structures, `executable_blocks`/`executable_edges`, `block_in`, `runtime_state` and `state_livein`, and to
recompute state EFFECTS rather than merely edges: replaying "any executable store promotes W" over freshly
computed reachability would still have promoted the self-assignment leaf, because that store genuinely is
reachable.

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
