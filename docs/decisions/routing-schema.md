# Routing algebra: schema for M2

Status: REVISION 3. Revision 1 was NOT APPROVED (its record could not express routing at all); revision 2 was
NOT APPROVED either (the record was fixed, but the site set was not closed, the verifier criterion was still
revision-1 text, and the Known-versus-no-write rule was wrong). Companion to `arch-ruling.md` (which
ordered plain MORPH) and `arch-memo.md`.

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
of which the last three are the same relation at three fixed offsets: `_copy_leaves` is the `k = 0` case
generalized over target places, and `_install` is `_project` with source and target exchanged. All of them
collapse to one row shape, `pi(i) = i + k`. The identical five-line leaf-walk skeleton appears in four of the
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
- `NoCell` for a `Reference`, and for a non-datapath `Known`.

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
| construction | per-field windows from several places; unadmitted defaults as `NoCell`, admitted as `ConstantCell` |
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

This also removes the silent-absence failure mode that this campaign has been bitten by repeatedly: a recorder
that stops writing currently produces a plausible identity route, which is a wrong answer that looks like a
default. Under a total plan it produces a verifier error.

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

### Traps the predicate must encode, each measured

These are the places where a uniform rule is wrong, and where a verifier written from the obvious model would
either reject valid output or accept a missing plan.

- `LoadPlace` is ASYMMETRIC. A scalar `Known` destination emits nothing at all, while an aggregate whose
  leaves are all `Known` emits constants for the datapath ones. A verifier modelling the two uniformly is
  wrong in one direction or the other, whichever way it picks.
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
- Conversely, several COMPUTATION sites degrade to aliases -- a same-kind cast, an all-integer intrinsic, an
  identity integer implementation, a unary plus -- and write the source's own value id. They are
  indistinguishable from routing at the HIR level and must be classified by op, never by inspecting the
  result.
- `_write` is last-wins per (block, cell). One site deliberately overwrites another during phi construction,
  so "a cell is written twice" is not by itself an error.
- The exit path writes no cells at all; it only reads. Return-contract promotion is therefore NOT a routing
  site and stays in M3, consistent with the ruling.

Two shapes were checked rather than assumed, both raised as uncertainties by the enumeration. An
empty-sequence repeat writes nothing and is CORRECT to do so, because its result is genuinely empty. A
`Reference` leaf inside an aggregate stored to component state cannot reach emission's unguarded path: analysis
refuses it first with a located public rejection.

## The key

Not `dst`: `StorePlace`, `PyStoreAttr` and `UnbindPlace` have none (`op_dst` returns `None` for exactly those
three), and the two emission arms that copy cells for a dst-less op both route through `_copy_leaves`. Keying
by dst is what forces those sites to stay outside the plan today.

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
