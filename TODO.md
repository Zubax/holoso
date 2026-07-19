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

The `docs/campaign.md` round 6-11 log entries hold the deeper record: the source anchors (the withholding guard
and the graftability mark in `_analyze.py`, the render arm in `_analysis_support.py`), the round-by-round
history of how each fix traded one corner for the next, and the bounding negative results the reviewers proved --
no kernel that should reject silently accepts, no analysis deadlock, and single-call diagnostic selection is
strictly improved -- which together scope the defect to false-rejection-only (never a miscompile).

The four minimal reproducers below were reduced and verified by the round 9-11 review pair (Claude ultrathink +
Codex ultra); each synthesizes with `default_ops(FloatFormat(8, 23))`. All but the last "should synthesize"; the
last "should keep synthesizing" and is a regression the round-10 edge-withholding fix introduced. When the
restructure lands, these become acceptance tests (all four synthesize) gated byte-for-byte against freeze-1.

```python
# Shape 1 -- single graftable call, result read on BOTH arms of a following branch.
# PROBE falsely rejects "local 'y' may be unbound here"; the CONTROL (single return) synthesizes (12332 B).
# The ONLY difference is the both-arms branch; the np.dot feed is clean (the control proves it).
class BothArms:
    def __init__(self): self.t = 0.0
    def step(self, x: float, flag: bool, pick: bool) -> float:
        if flag: u = 1.0; q = 1.0
        else:    u = 2**53 + 1; q = 2**64      # inexact int -> float slot: transient (promotes at the fixpoint)
        self.t = u                             # the transient store violation that opens the deferral net
        a = np.array([q, x]); y = np.dot(a, a) # graftable call defers behind it, grafts on revisit
        if pick: return y + 1.0
        return y + 2.0                          # CONTROL is this class minus `pick`, single `return y + 1.0`

# Shape 2 -- two graftable calls in one block; the first defers, a later one grafts (ordinary numeric code).
# Falsely rejects "local 'y' may be unbound"; Python oracle is well-defined (13.0 / 3.4e38). Single-dot,
# clean-dot-first, and no-wide-int variants all synthesize. Generalizes to np.matmul / user calls / loops.
class TwoDots:
    def __init__(self): self.t = 0.0
    def step(self, x: float, flag: bool) -> float:
        if flag: u = 1.0; q = 1.0
        else:    u = 2**53 + 1; q = 2**64
        self.t = u
        a = np.array([q, x]); y = np.dot(a, a)
        b = np.array([x, x]); z = np.dot(b, b)
        return y + z

# Shape 3 -- starred call arguments (validation precedes the graftability mark, so phantom edges publish first).
# Native execution returns 3.5; lowering falsely rejects "local 'y' may be unbound".
def helper(a: float, b: float) -> float: return a * b
class StarArgs:
    def __init__(self): self.t = 0.0
    def step(self, x: float, flag: bool) -> float:
        if flag: u = 1.0; q = 1.0
        else:    u = 2**53 + 1; q = 2**64
        self.t = u
        args = np.array([q, x]); y = helper(*args)
        return y + 1.0

# Shape 4 -- REGRESSION from round-10 edge-withholding: withholding the block's SOLE out-edge starves the
# downstream StaticFor unroll fixed point, so a kernel that lowered at round-8 now falsely rejects
# "state attribute 't' ... not exactly representable". Adding an independent bypass path around the loop
# makes the same datapath synthesize.
class UnrollStarve:
    def __init__(self): self.k = 0; self.t = 0.0
    def step(self, x: float, n: int) -> float:
        u = self.k + (2**53 + 1); q = self.k + (2**64 + 1)
        self.t = u
        a = np.array([q, x]); y = np.dot(a, a)
        for _ in range(1): self.k = n
        return y
```

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
