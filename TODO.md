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
before the fixpoint promotes it -- the call's not-yet-computed result can reach a following branch as `Unbound`
and a later join reports `local '...' may be unbound here` at an innocent line. The single-call/both-arms shape
is addressed by edge withholding; two open shapes remain: a block with two graftable calls where the first
defers and a later one grafts (the terminator-identity guard treats the later graft as resolution), and starred
call arguments whose validation precedes the graftability mark. Edge withholding also has a soundness cost: it
can starve a downstream `StaticFor` unroll fixed point when the withheld edge is the block's only successor,
turning a valid kernel into a false store-schema rejection. This whole class lives in the deferral-net x
grafting seam and resists in-place patching (each fix has traded or added a corner); it is the class the
post-stabilization resolution-totality restructure (docs/decisions/arch-memo.md, the resolved-IR spike) is
designed to dissolve by making residualization a total pass rather than interleaving it with the fixpoint.
Reproducers: `y = np.dot(a,a); z = np.dot(b,b); return y+z` behind a conditional inexact int/float state store;
the single-call variant with a branch after the call; a graftable call whose sole successor is a static loop.

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
