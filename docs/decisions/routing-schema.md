# Routing algebra: schema for M2

Status: REVISION 7 -- written against measurements AND against a review of what those measurements did not cover, and
the first to
correct a claim that four rounds of design review had left standing. Revisions 1-5 were each refused: the
record could not express routing at all, then the site set was not closed, then the predicate arms were listed
without being decided, then contradictions remained. Companion to `arch-ruling.md` (which ordered plain MORPH)
and `arch-memo.md`.

## What this replaces

Cell routing today is a bare `tuple[int, ...]` plus a convention that absence means identity. There is no
`Route`, `CellRef`, or comparable datatype anywhere in `_fir`, so M2 introduces a genuinely new type rather
than renaming an existing one.

The surfaces, re-anchored against the working tree because every line number the campaign quotes has drifted:

Four `id()`-keyed analyzer tables, cleared per W/D round in `_reset_round`:

| Table | Key | Value |
| --- | --- | --- |
| `_conversion_calls` | `id(PyCall)` | membership only -- a classification flag, not a route |
| `_subscript_selections` | `id(PySubscript)` | source ordinal per result cell |
| `_conversion_routes` | `id(PyCall)` | permuted source ordinals (transpose only) |
| `_construction_calls` | `id(PyCall)` | per-field source binding, `None` meaning default-filled |

Two result tables, `subscript_plans` and `route_plans`, both `dict[BindingId, tuple[int, ...]]` -- structurally
the same relation, separated only by which op kind produced them. Both are written in `_finalize` under an
`id(op) in ...` guard and read in `_emit.py` exactly once each, with `.get()`, where `None` is meaningful:
absent means positional projection for one and identity route for the other.

Two `CallLowering` variants carry routing, and they carry it inconsistently: `CONSTRUCTION`'s routing rides
inline on `CallPlan.construction`, while `CONVERSION`'s sits off to the side in `route_plans` keyed by dst.

Five copy routines in `_emit.py` (`_emit_concat`, `_emit_conversion`, `_copy_leaves`, `_project`, `_install`),
of which THREE are the same relation at three fixed offsets: `_copy_leaves` is the `k = 0` case generalized
over target places, and `_install` is `_project` with source and target exchanged. Those three collapse to
`pi(i) = i + k`; the other two do NOT, and revision 3 left the blanket affine claim standing here while
refuting it below. The identical five-line leaf-walk skeleton appears in four of the
five, and a second time inside `_install` for the scalar case.

Three further inline clones of that skeleton: the `PySubscript` gather walk, the `PyAttr`
aggregate-component-state arm, and the `CONSTRUCTION` default-field fill.

Emission re-derives routing the analyzer already knew, in three places with no recorded plan at all -- concat
and repeat offsets, the positional-subscript fallback (which re-imports `operator`, re-materializes the index
fact and re-runs `operator.index`, a second independent evaluation of a key the analyzer already folded), and
the record-field projection -- plus two more `child_slice` recomputations that have a partial plan naming
source bindings but not offsets (`BuildTuple`/`BuildList`, and `CONSTRUCTION`).

The accounting, at its third revision, because the first two were both wrong. `HANDOFF.md` says four inline
clones; revision 1 of this document said three, matching `arch-memo.md`. X6a recounted and ruled FOUR, and it
is right: the fourth is the aggregate `PyStoreAttr` walk, which is not a textual clone of the exact skeleton
but is a stronger inline route walk carrying promotion and state-slot registration. Counting the truncated
construction loop while excluding `PyStoreAttr` is not a defensible boundary for a step about routing.

Likewise the offset derivations are SIX, not the five revision 1 claimed: four wholly unplanned branches
(concat prefix, repeat base, positional-subscript window, record-field window) plus two partially planned ones
(`BuildTuple`/`BuildList` window, construction-field window). "Five" holds only if the two `_emit_concat`
branches are collapsed into their containing method, which is a location count, not a branch count. `child_slice`
appears exactly four times in `_emit.py`.

M2 is budgeted against four and six. The campaign said four and two, revision 1 said three and five, and the
truth is four and six -- worth recording as a caution, since each revision was stated with confidence and two
of the three were wrong.

## What revision 1 got wrong

Revision 1 addressed a source cell as `OperandCell(operand, ordinal)` -- an index into the routing op's operand
list. That list has no authoritative meaning. `_op_reads(PyCall)` yields the callee before the arguments, so the
conversion source is not operand 0; `PyStoreAttr` puts `src` at operand 1; `_op_reads(LoadPlace)` is empty, so a
`LoadPlace` route could name nothing at all; the aggregate-component `PyAttr` arm reads `StateLeaf` cells rather
than cells of its scalar `obj` operand; and construction takes positional and keyword sources with no defined
numbering between them. `StorePlace` and `PyStoreAttr` have no result layout, so they need an explicit target
as well as an explicit source.

The concrete failure this would have caused: analysis accepts `3 * seq` as well as `seq * 3`, so the sequence is
not always operand 0. With a one-cell sequence, a bounds check on the operand index still passes while the route
reads the wrong operand -- a well-formed wrong answer with no diagnostic, which is precisely the failure class
this campaign exists to stop adding. The compiler is correct here today; the proposed schema would have
regressed it. That case is now pinned in `tests/test_frontend_routing.py`.

Revision 1 also claimed every route is affine, `pi(i) = i + k`. False: transpose and gathers are arbitrary
selections, concat switches operands mid-result, and repetition is periodic. The per-target-row form expresses
all of them, but the affine framing does not.

## The record

Cells are addressed by `Place` and ordinal, never by operand index, which removes every ambiguity above. The
existing `_LeafPlace` already establishes that a scalar root is ordinal 0, so the scalar cases in `_project`
and `_install` need no variant of their own.

```python
@dataclass(frozen=True, slots=True)
class CellRef:
    place: Place
    ordinal: int

class CellTransfer(enum.Enum):
    IDENTITY = enum.auto()
    INT_TO_FLOAT = enum.auto()
    BOOL_TO_FLOAT = enum.auto()

@dataclass(frozen=True, slots=True)
class CopyCell:
    source: CellRef
    transfer: CellTransfer

@dataclass(frozen=True, slots=True)
class ConstantCell:
    value: StaticValue
    kind: SemType

@dataclass(frozen=True, slots=True)
class NoCell:
    pass

type CellAction = CopyCell | ConstantCell | NoCell

@dataclass(frozen=True, slots=True)
class RoutePlan:
    target: Place
    actions: tuple[CellAction, ...]   # index is the target's LOGICAL leaf ordinal
```

`NoCell` is mandatory, not tidiness, and its meaning is SITE-RELATIVE: "this site emits no datapath definition
for this logical target ordinal". Revision 2 defined it as "the fact is a `Reference`", which is wrong and
would have changed emitted output. A `Reference` leaf is indeed one case -- an unadmitted construction default
such as `None` becomes one, already a supported black-box case, and it is neither a source cell nor a
`StaticValue` constant. But a datapath-capable `Known` is NOT always materialized: a fully static construction
emits nothing at its call site, and all-known projections are gated the same way. Executing a `ConstantCell`
row unconditionally would introduce dead constants and could move pre-optimization HIR allocation and order,
which is exactly the byte-identity the corpus gates on.

So analysis chooses, per site and per ordinal:

- `ConstantCell` when THIS SITE materializes the `Known` today;
- `NoCell` when the `Known` stays fact-only at this site;
- `NoCell` for a `Reference` or a non-datapath `Known` ONLY where current successful emission skips it.

That last qualification is load-bearing and revision 3 had it as an unconditional rule, which would have
suppressed a real diagnostic. Scalar `PyStoreAttr` has no skip: it always materializes. A scalar non-datapath
`Known` passes storage conformance and then meets a LOCATED materialization rejection. Encoding it as `NoCell`
would delete that rejection and silently retain the previous state instead -- a silent wrong-state miscompile
manufactured by the very step meant to remove silent-absence bugs. A non-materializable scalar state store
keeps its rejection; it does not become a plan row.

The construction default rule follows, and revision 4 stated it wrongly as "an admitted default is a
`ConstantCell`". Admission covers strings, ranges, slices and records, while emission materializes only
boolean and numeric datapath `Known`s. The rule is:

- an INACTIVE (fully static) construction is `NoCell` at every ordinal;
- active, with a datapath `Known` -> `ConstantCell`;
- active, with a `Reference` or a non-datapath `Known` (a string default, say) -> `NoCell`;
- active, with a residual source -> `CopyCell`.

`ConstantCell.value` is the TARGET-SIDE, POST-TRANSFER value. This survived a challenge and is worth
recording, because the challenge was well argued and wrong: an implementation spike stores the UNCONFORMED
value beside a target-side kind, and reported that the document should follow it. It should not. The spike
NEVER VERIFIED CONSTANTS OR TRANSFERS AT ALL -- its zero-disagreement result covers cell sets and source
reads only -- and its own `StorePlace` producer contradicts itself, changing the kind to float while retaining
the integer value. An unvalidated behaviour is not evidence just because it sits inside something that was
validated for a different property.

A dst-less `StorePlace` has no final destination binding fact, so producer and verifier must each replay the
stable storage-schema conformance walk INDEPENDENTLY to derive the post-store target value and kind, the
required transfer, and the logical width. That walk is the authority; neither side may read the other's. A known
integer stored into a float slot begins as an integer `Known` and storage conformance produces a float one, so
"the exact semantic value" is not single-valued until the direction is fixed. It is the conformed value, and
values compare with the codebase's existing bit-faithful equality rather than Python `==`.

Emission must not rediscover any of that. `ConstantCell.kind` is explicit for the same reason the transfer is.

`CellTransfer` is J6 landing inside M2 for routing sites, per the ruling. An expected result kind alone is not
enough: `FLOAT` would still leave the emitter inspecting the source to choose between an integer and a boolean
promotion, which is the J6 violation restated. `_emit_conversion` may only EXECUTE the recorded transfer; it
must retain no fact-based coercion choice of its own.

The vocabulary is closed at three ONLY for M2's scope. `FLOAT_TO_INT`, the boolean/integer conversions and the
truth conversions live in scalar casts and other non-routing lowering. If a later step absorbs scalar
`CallLowering.CAST` into `RoutePlan`, three values stop being sufficient immediately -- recorded here so that
step does not discover it by producing a wrong promotion. This obligation extends
M2 past `_emit_conversion` to the aggregate `PyStoreAttr` promotion and the scalar `store_conversions` plan, or
M2 would delete route walks while leaving a second kind authority standing beside them. Phi-arm promotion and
return-contract promotion are NOT routing and stay in M3.

Every routing today is one instance of this shape:

| Site | Row |
| --- | --- |
| identity conversion | `CopyCell(CellRef(src, i), IDENTITY)` |
| transpose, gather | `CopyCell(CellRef(src, pi(i)), ...)` -- an arbitrary selection, not an offset |
| positional and record projection | `CopyCell(CellRef(src, start + i), ...)` |
| concat | the source `place` switches at the split, not merely the ordinal |
| repeat | `CellRef(seq, i % W)`, where `W` is the SOURCE LEAF WIDTH, not the repetition count |
| `BuildTuple` / `BuildList` | each item place at its own window |
| construction | per-field windows; defaults follow the four-way rule below -- a string default is `NoCell` |
| component `PyAttr` | `CellRef(StateLeaf(...), i)` -- a state root, unreachable from any operand index |
| `StorePlace`, `PyStoreAttr` | explicit `target`, since neither has a result layout |

## Totality is the point

The row shape matters less than the property it buys. Because a `RoutePlan` names every result cell, absence
stops being meaningful. That kills the two `.get()`-means-something conventions in `_emit.py`, and it closes a
hole `verify_plan_totality` currently names in its own docstring and declines to check:

> `subscript_plans` and `route_plans` are read with `.get()`, where absence legitimately means positional
> projection and identity route, so their omissions are indistinguishable from intent and are NOT checked;
> verifying those needs the typed explicit variants M2 introduces

The verifier criterion, stated precisely, because revision 2 left revision 1's text here and it referred to a
"result layout" and to `OperandCell` -- neither of which exists for a `StorePlace` or a `PyStoreAttr`, and the
latter not at all any more. The verifier must, independently of the producer:

- derive the EXPECTED set of route sites from the op and its final facts, by the predicate in "The site set"
  below, and compare `dict[PlanSite, RoutePlan]` key sets EXACTLY -- rejecting surplus plans as well as
  missing ones;
- derive the expected target `Place` per op, rather than reading it back from the plan;
- derive the target's LOGICAL width: zero for an empty aggregate, one for a scalar, the aggregate leaf count
  otherwise -- and for a state target, from the reset-fixed state schema rather than from the store's source;
- resolve every source `Place` in the PRE-OP environment and bounds-check the ordinal there;
- check every `ConstantCell` kind and every `CellTransfer` for legality against the source and target kinds.

And a bound is not a permutation. An in-range but WRONG permutation passes every structural check above, so
the behavioural witnesses in `tests/test_frontend_routing.py` carry that weight and the verifier does not
replace them.

This was claimed to remove the silent-absence failure mode outright -- "a recorder that stops writing produces
a verifier error" -- and A SHADOW IMPLEMENTATION MEASURED THAT CLAIM FALSE AS THE DOCUMENT ORIGINALLY MEANT IT.
Mutating a producer to emit `NoCell` where a `CopyCell` belongs (M6: exactly the silent-absence archetype) was
killed 0 times out of 58 kernels when the verifier's source check was AVAILABILITY. It survived every one.

What actually kills it is INDEPENDENT DISPOSITION DERIVATION: the verifier computing, from the target fact's
leaves, which ordinals must be a copy, which a constant, and which nothing -- and comparing that against the
plan. With that check the same mutant dies 58 out of 58, along with missing plans, surplus plans, out-of-range
ordinals and wrong targets at 58/58 each.

So the priority is the reverse of how revisions 1-5 presented it. DISPOSITION IS THE LOAD-BEARING CHECK.
Availability is supporting: it caught a wrong source place 19 times of 58 on its own and objected falsely
zero times, which makes it worth keeping and useless alone.

## The site set

Totality is meaningless without a closed set of sites to be total over, and revisions 1 and 2 both left it
open. The table of routings above is NOT the site set.

It is closed now, and provably: `_write` is the SOLE mutator of the cell map `_definitions`, every other
reference to it is a read, and it has thirty call sites. Anything not among those thirty cannot define a cell.
That is the authority the verifier's predicate derives from -- not from the producer, which must never be its
own authority on which sites exist, or a surplus plan and a missing one become indistinguishable from a
disagreement about the set.

The predicate is over (op kind, final facts) and yields a target `Place`, a logical width, and a class:
ROUTING (cells move from an existing place), COMPUTATION (a cell is defined by fresh HIR), JOIN (phi), or
CONTRACT (entry parameters). Only ROUTING sites carry a `RoutePlan`.

Two sites revisions 1 and 2 both missed, and why they were missed. `LoadPlace` routes in BOTH forms -- the
aggregate one through `_copy_leaves`, the scalar one directly -- and neither appears in any routing table
today. A `PySelect` whose condition is compile-time known RE-CHOOSES its source during emission from the
condition fact; it is a seventh routing re-derivation, and it escaped the offset recount because it derives no
offset at all. Counting offsets cannot find a site that permutes nothing and merely picks the wrong operand.

### The arms the predicate must settle

Counting `_write` proves nothing else mutates the cell map. It does NOT say which sites need a key, because a
required route can execute ZERO writes -- a zero-cell conversion, a fully static construction, an all-known
projection. Presence of a plan and presence of a write are independent, and revision 3 conflated them. These
arms are decided as follows, with the reasoning, because listing a tension is not settling it:

- `PyBin` is decided by the LAYOUT, not the op kind. A sequence aggregate ROUTES -- concat and repeat emit no
  HIR at all -- and takes a key. An array aggregate COMPUTES elementwise and takes none. A scalar computes.
- `PyAttr`: EVERY `PyAttr` takes a key, not only the record ones, including namespace, method and other
  fact-only attribute accesses. Revision 6 said "record `PyAttr`" while the shadow classified all of them, and
  the write-based closure measurement CANNOT distinguish the two policies because such sites write nothing --
  producer and verifier simply agreed with each other. Choosing "every" makes the predicate decidable from the
  op kind alone.
- `StorePlace` and `BuildTuple`/`BuildList` take keys, and `StorePlace`'s logical width comes from the
  POST-STORE source fact, NOT the pre-op target fact: a local aggregate may legally change arity across a
  store, so reading the width from the target is wrong wherever it does.
- `PySubscript` produces an ALL-`NoCell` PLAN rather than no key. The `needs_cells` gate
  then stops being an independent decision and becomes a consequence of the plan's dispositions, which is the
  document's whole thesis applied to itself: absence must never be the carrier of meaning. It costs nothing at
  emission, since a plan of `NoCell` rows emits exactly what the gate emits today -- nothing -- so it cannot
  disturb HIR node order.
- `PySelect` takes a key only when BOTH hold: the condition fact is `Known`, AND the result is neither `Known`
  nor `Reference`. The second condition is a separate bypass in emission and revision 3 recorded only the
  first.
- `PyCall` is classified from `CallPlan.lowering`, never from op kind plus destination fact, because a folded
  call can also produce an aggregate. A key exists for `CONVERSION` with an aggregate destination, and for
  EVERY `CONSTRUCTION` -- a fully static one gets an all-`NoCell` plan rather than no key, for the same reason
  projection does, and because revision 4 called it a required zero-write route in two places while granting
  it a key only when a leaf was `Residual`. `FOLDED`, `CAST` and `INTRINSIC` take none.
- Component-state source resolution falls back explicitly: pre-op environment first, then the state live-in,
  then the reset fact. Without the fallback the predicate fails on the FIRST attribute access, which is the
  one that installs the leaf during the very op that reads it.

### What the verifier must check beyond the above

Revision 3's criterion was directionally right and incomplete. It must also require that the action count
equals the logical width; that the DISPOSITION expected at each ordinal matches (not merely that some action
is present); that a `ConstantCell` carries the exact target-side value, not just a legal kind; that a
`CopyCell` source HOLDS A DATAPATH VALUE THIS PLAN MAY COPY -- narrower than "is defined", see below -- and that the
source place is the SITE-DETERMINED one, since an arbitrary in-range place must not pass.

That last point resolves a contradiction revision 3 contained: the `PySelect` trap said the verifier must
reproduce the AND/OR polarity, while the criterion only checked the producer-named source for existence and
bounds -- under which a wrong equal-width arm passes. The verifier derives the expected source place or arm.
Arbitrary gather and transpose permutations remain the behavioural witnesses' job, because no structural check
can distinguish an in-range wrong permutation from a right one.

Source-place verification applies ONLY to REPRESENTED `CopyCell` actions. A zero-width route encodes no source
at all -- a false aggregate `AND` selects the empty operand, so target width and action count are both zero --
and arm identity there is semantically absent rather than merely unverified. Do not add site-level arm
metadata to make unobservable information checkable.

### Two responsibilities the plan does not carry

Not everything the old emission path did belongs in a `RoutePlan`, and a cutover that assumes otherwise loses
a diagnostic or a port silently.

The scalar non-datapath `PyStoreAttr` rejection was assigned an owner on a FALSE PREMISE, and the premise is
withdrawn. The ordinary case does not reach emission at all: storing a string into a float-reset slot rejects
during ANALYSIS with "values of irreconcilable kinds merge here", raised from the analysis support module --
verified by probing the public API and reading the traceback, not by reading the emitter. So removing the old
emission path orphans nothing here, and no reassignment is needed for it.

What remains is narrower and must be established rather than assumed: whether any non-datapath scalar store
is reachable at EMISSION materialization at all. If one is, its rejection is PRECOMPUTED during plan
production and ATTACHED TO ITS `PlanSite`, not raised globally by that pass -- raising it globally would
reorder it ahead of an earlier diagnostic that a kernel hits first, turning a correct message into the wrong
one. If no such case is reachable, the obligation disappears with the code.

Note also that the routing witness pinning this rejection passes for a weaker reason than its name suggests:
it asserts a generic located rejection, and the analysis-phase diagnostic satisfies it. The real coverage for
the storage-conformance path is in the schema suite.

STATE-SLOT REGISTRATION is not plan-verifiable, and revision 4 wrongly listed it as a verifier arm. Emission
performs it as a side effect beside each state write, so a verifier running BEFORE emission cannot prove a
future executor will do it. It becomes an EXECUTOR INVARIANT instead: executing a state-target plan registers
every target ordinal. The verifier checks what it can see -- that the target is the expected state place and
its width comes from the reset-fixed schema -- and the write-only-state behavioural witness checks the ports
that result.

### Traps the predicate must encode, each measured

Ten places where a uniform rule is wrong, and where a verifier written from the obvious model would either
reject valid output or accept a missing plan.

- `LoadPlace` is ASYMMETRIC. A scalar `Known` destination emits nothing at all, while an aggregate whose
  leaves are all `Known` emits constants for the datapath ones. A verifier modelling the two uniformly is
  wrong in one direction or the other, whichever way it picks.
- THREE HELPER SCALAR ARMS, TWO POLICIES, and revisions 1-5 named only one of them. (Direct scalar
  `StorePlace` and `PyStoreAttr` are further materializing routing arms; the count is of the helpers.) `LoadPlace`'s
  scalar arm and
  `_project`'s scalar arm both SKIP a `Known` destination; `_install`'s scalar arm MATERIALIZES it as a
  constant. The `_project` half is an equally silent wrong-output trap and was found only by building a shadow
  producer, which mis-handled it and surfaced as 24 cell-set mismatches against the real emitter.
- A fully static record construction emits NOTHING at its call site -- not even constants for its
  datapath-`Known` leaves. This is the case that makes `NoCell` site-relative rather than fact-relative.
- The leaf-completeness policy DIVERGES between the arithmetic and routing paths. The elementwise and unary
  aggregate paths skip `Known` result leaves entirely with no constant; `_copy_leaves`, `_project` and
  `_install` materialize them. Any verifier assuming "every aggregate site defines every datapath leaf" is
  wrong for half the sites.
- Scalar `PyStoreAttr` has NO `Known`/`Reference` skip: a constant store does define a cell, unlike almost
  every other scalar site.
- A known-condition `PySelect` inverts with its mode -- `SelectMode.AND` takes the RIGHT operand when the
  condition is true. An independent verifier must reproduce that polarity exactly or it will name the wrong
  source and pass.
- Routing does not mean "the same value id". A kind promotion inserts an HIR node between source and target,
  which is why the row carries an explicit transfer rather than an equality claim.
- Conversely, SOME COMPUTATION sites degrade to aliases -- a same-kind cast, certain integer intrinsics, an
  identity integer implementation, a unary plus -- and write the source's own value id. Not every integer
  intrinsic does. They are indistinguishable from routing at the HIR level and must be classified by op,
  never by inspecting the result.
- `_write` is last-wins per (block, cell). One site deliberately overwrites another during phi construction,
  so "a cell is written twice" is not by itself an error.
- The exit owns no op-site datapath definition. Stated as "the exit writes no cells" this is not literal: its
  reads can create cached SSA definitions and phis. Return-contract promotion is NOT a routing site either way
  and stays in M3, consistent with the ruling.

Two shapes were checked rather than assumed, both raised as uncertainties by the enumeration. An
empty-sequence repeat writes nothing and is CORRECT to do so, because its result is genuinely empty. A
`Reference` leaf inside an aggregate stored to component state cannot reach emission's unguarded path: analysis
refuses it first with a located public rejection.

### What implementation contact settled

The question four consult rounds converged on was whether SOURCE AVAILABILITY can be independently
reconstructed from `block_in`, the final binding facts and an intra-block walk without reusing the producer's
decisions. Measured: YES -- zero unsound verdicts over 3,461 sites and 3,927 represented copies, with no false
objection. (An earlier draft quoted 940,977 "cells"; that is repeated PROGRAM-POINT OBSERVATIONS, not unique
logical cells, and quoting it as a coverage figure overstated the sweep by two orders of magnitude.) And that answer
turned out to matter far less than the question implied,
because availability alone catches almost nothing (above). The consult should have been asking about
DISPOSITION derivability. Four rounds of review refined a question that was not the load-bearing one.

Two corrections to the inputs. The verifier also requires `runtime_state`, which none of the five revisions
named: without it every state-leaf source is undecidable (14,777 program-point observations), because a promoted leaf is
unconditionally available while a non-promoted one is compile-time configuration with no cell, and only
`runtime_state` separates them. And "datapath-available" is NARROWER than "the cell is defined": 14,340 swept
observations are of cells defined as materialized constants while correctly reporting no datapath value. The
predicate is
"holds a datapath value this plan may COPY", not "has a definition".

ONE CASE DEFEATS AN INDEPENDENT VERIFIER, and it is the only one found. A `CONSTRUCTION`'s field-to-binding
mapping lives solely in `CallPlan.construction`, a producer record M2 absorbs. Swapping two field sources
yields a plan with identical target, width, dispositions, places and in-range ordinals -- the verifier accepts
both. Measured on a two-field record: the correct plan computes -47.0 and the swapped one -25.0, against a
Python reference of -47.0. So revision 5's "an arbitrary in-range place must not pass" is FALSE for
construction as written.

RULED: the verifier RE-DERIVES the field binding from the dataclass schema and the call's positional and
keyword structure. That is Python's own semantics, not a producer decision, so re-deriving it is what a
verifier is for and does not create the second routing authority M2 exists to remove -- whereas reading
`CallPlan.construction` back would make the check vacuous by construction.

## The key

Not `dst`: `StorePlace`, `PyStoreAttr` and `UnbindPlace` have none (`op_dst` returns `None` for exactly those
three). `StorePlace` copies through `_copy_leaves`; `PyStoreAttr` does NOT -- it has its own inline loop,
which revision 3 stated wrongly. Keying by dst is what forces both outside the plan today.

Recommendation: key by `(BlockId, op index)` over the stabilized graph, assigned in `_finalize`.

The analyzer keeps recording `id(op)`-keyed per round exactly as M1 established, and finalization translates
into position keys, which is the same record-at-the-visit, translate-at-finalization shape M1 landed and both
differential harnesses validated. The key is then stable over precisely the window where it is consumed --
finalization, the verifier, emission -- and typed, unlike `id()`.

The premise that window is frozen was measured, not assumed: instrumenting `verify_plan_totality` and
comparing op identity and order across every block at that point against the state at the end of emission
found them unchanged (48 ops across 3 blocks on a kernel exercising tuple projection, array reshape,
transpose and state). Emission contains no mutation of `unit.blocks` or of any block's `ops`.

RULED (X6a): `(BlockId, op index)`, wrapped in a typed `PlanSite`. No stamped `OpId`. This is a PHASE-LOCAL
contract, not a durable identity, and that is sufficient; a future pass that mutates the finalized FIR must
rebuild or invalidate its plans rather than expect them to survive.

One argument revision 1 made for this was wrong and is withdrawn: that `id(op)` "cannot be serialized into the
golden HIR dump". The dump serializes the resulting `Hir`, not `ResidualUnit` or its plans, so serialization
does not bear on the key at all. The decision stands on the other grounds.

The verifier's duty is stronger than revision 1 stated. It must independently derive the EXPECTED set of route
sites over `executable_rpo` and compare key sets exactly, rejecting surplus plans as well as missing ones, then
validate targets, `NoCell` rows, constants, source cells and transfer legality. And a bound is not a
permutation check: an in-range but WRONG permutation passes every structural test there is, so behavioural
route witnesses remain indispensable and the verifier does not replace them.

Alternatives considered and why they lose. Keying by `id(op)` matches what the analyzer does internally and
would work, since `ResidualUnit` holds the graph and every recorded op stays alive; but it is untypeable, and
M1 already showed that reasoning about id-keyed table lifetime costs a review round per table. Stamping a
stable `OpId` on every op at construction is the cleanest
in principle, but `_remap_op` clones ops with `replace()` during unrolling, which would duplicate the stamp
unless every clone site re-stamps explicitly -- a new invariant with no verifier behind it, introduced in the
step whose whole purpose is to make routing verifiable.

## Scope: what M2 absorbs and deletes in the same commit

Absorb: `_subscript_selections`, `_conversion_routes` and `_construction_calls`; `subscript_plans` and
`route_plans`; `CallPlan.construction`; the four wholly unplanned offset derivations and the two partially
planned ones; the aggregate `PyStoreAttr` promotion and the scalar `store_conversions` plan.

`_conversion_calls` leaves routing ENTIRELY (ruled): `CallLowering.CONVERSION` is recorded directly in
`CallPlan` at the visit. Classification must never be inferred from whether a route exists, because an identity
conversion and a zero-cell conversion are both still conversions.

`UnbindPlace` is excluded from route plans (ruled): it moves no value. Totality means total over a CLOSED SET
of route-producing sites, not that every FIR op owns a route. It must NOT be given an empty `RoutePlan` --
an empty tuple, list or array has a legitimate zero-cell route, so conflating "not a route" with "a route with
zero rows" would reintroduce exactly the absence-versus-intent ambiguity this step exists to remove. If every
op must carry a disposition, that is an explicit `NoRoute`, not an empty plan.

Delete: `_project` and `_install` collapse into the offset row; `_copy_leaves` becomes the identity row;
`_emit_conversion` keeps only the mechanical execution of a recorded transfer; `_emit_concat` loses its offset
arithmetic entirely. One leaf-walk remains where five routines plus four inline walks stand today.

## Sequencing (ruled)

One production commit for absorb-plus-delete. A dual-authority intermediate is useful only as an uncommitted
development state; it must not land. The order: author black-box route witnesses and verifier-mutation tests
against the baseline FIRST, build producer and verifier in shadow locally, check exact site sets, row counts,
source and target bounds, no-write leaves and transfers, then cut every consumer over and delete the old
tables, fallbacks, helpers and inline walks in the SAME commit, then run the byte gates, the behavioural value
tests and the review loop.

Raw-byte corpus identity is necessary but NOT sufficient, and revision 1 leaned on it too heavily. A route
error is ordinarily a semantic value miscompile rather than an ABI change: the manifest records ports and
metrics, not which value drives each port. The witnesses carry the weight the corpus cannot.

Still to write before cutover, per the ruling: reversed repetition (`3 * [x]`, now pinned); missing, surplus,
zero-cell and no-write plan mutations; place and state sources; and an in-range wrong permutation, which is
the case no structural verification can reach.

## Resolved questions

The five questions revision 1 raised are ruled above: `(BlockId, op index)` in a typed `PlanSite`; J6 folds
into M2 for routing sites only; `_conversion_calls` leaves routing entirely; `UnbindPlace` is excluded rather
than given an empty plan; one atomic commit with the tests written first.

## Risk

The corpus must regenerate byte-identically. Routing determines which cell reaches which port, so an error
here is an ABI change, and `store_order` is the part of the plan the campaign has already identified as being
the port ABI. No test names `subscript_plans` or `route_plans`, and none should: public synthesis and value
behaviour is the better contract. Coverage of both is behavioural only, which is correct rather than a defect.

The campaign says four examples exercise routing -- `routed_diamond`, `ekf1`, `fsc`, `imu` -- and this document
repeated it. `routed_diamond` DOES NOT EXIST. It was a spike artifact at `tests/spike_golden/kernels/` on the
branch that was deleted when Stage 3 closed, and it survives only in `spike-ledger.md`, which is a verbatim
record of that branch rather than of this tree. So there are three, not four, plus `polar` and `signal_window`
which the campaign's list omits -- one more miscount in this document's own subject matter, alongside the clone
and offset counts above, every one of them found by re-deriving rather than reading.

The safety net was then measured rather than described. `tests/test_frontend_routing.py` was added first, one
swap-sensitive kernel per routing construct, and four routing mutants were run against it and against the
pre-existing suites: a perturbed transpose route, a rotated repeat, a swap inside each repeated unit, and a
rotated aligned copy. Every mutant the new module caught, the example-driven matrix and aggregate tests caught
as well -- so the net at whole-suite level was NOT as thin as the inventory implied, and the new module's value
is localization and a per-construct invariant rather than newly closed holes. One route is inherently
untestable and M2 should not try: `seq * n` yields identical copies, so permuting whole repetitions maps
identical content onto identical content.
