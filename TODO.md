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

Graftable-call deferral can falsely reject a synthesizable kernel (the analyzer's optimistic-SCCP fixpoint
meets destructive mid-round call grafting). When a graftable call (a linalg composite like `np.dot`/`np.matmul`,
or an inlined user callable) cannot resolve on a visit because a store-schema violation is transiently pending
in scope -- typically an int/float merge into a float state slot whose Known-int arm is momentarily inexact
before the fixpoint promotes it -- the block's terminator still publishes out-edges, so the call's
not-yet-computed result reaches the successors as `Unbound`. The graft-time retraction that unwinds those edges
reaches only one edge deep, so a transitive successor keeps the phantom-unbound environment and a later join
reports `local '...' may be unbound here` at an innocent line.

The class is bounded to false rejections: the reviewers established that no kernel which should reject is
silently accepted (`Unbound` biases joins toward rejection, never toward a spurious bind), that the analysis
does not deadlock, and that emission never sees a phantom path. It is never a miscompile.

Three witness shapes are pinned executably in `tests/test_frontend_state.py`
(`test_graftable_call_deferral_false_rejection_witnesses`): a single graftable call whose result is read on both
arms of a following branch; two graftable calls where a later one grafts while the first is still deferred; and
starred call arguments, whose validation refuses before the call can graft. They live in code rather than in
prose here because prose transcriptions of them rotted -- dropping the both-arms read or the wide-int feed makes
a shape silently vanish, which is exactly what happened to two of the four kernels this section used to carry.

Patching the seam in place has been tried and abandoned. Rounds 6-11 (`docs/campaign.md`) each traded one corner
for another, and the round-10 attempt -- withholding a deferred graftable call's terminator edges -- regressed
valid code: it starved the outer state fixed point when the withheld edge was a loop body's only successor,
turning a kernel that synthesizes into a false "not exactly representable" refusal. That attempt was reverted, so
the class is now uniformly open with no regression against it; the starvation kernel is pinned as a passing
acceptance test (`test_deferred_graftable_call_does_not_starve_the_state_fixpoint`) so the trap cannot be
re-entered. The class is the one the post-stabilization resolution-totality restructure
(`docs/decisions/arch-memo.md`, the resolved-IR spike) dissolves by making residualization a total pass after the
fixpoint rather than interleaving it with the fixpoint; when that lands, the witness test flips to asserting
synthesis and is gated byte-for-byte against `freeze-1`.

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
