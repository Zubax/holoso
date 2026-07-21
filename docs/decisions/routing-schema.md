# Routing algebra: schema for M2

Status: DRAFT, submitted to Codex consult X6a. No adoption code exists or may be written until X6a rules.
Companion to `arch-ruling.md` (which ordered plain MORPH) and `arch-memo.md`.

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

Two corrections to the campaign's own accounting, both found by re-deriving rather than reading. `HANDOFF.md`
says four inline clones; only three exist that match the skeleton, and `arch-memo.md` independently says three.
And "two emission re-derivations" undercounts: three have no plan, and five places in total recompute a cell
offset from a layout. M2 should be budgeted against three and five, not four and two.

## The record

One row per result cell, total over the cells of every routing site:

```python
type CellSource = OperandCell | ConstantCell

@dataclass(frozen=True, slots=True)
class OperandCell:
    operand: int   # which operand of the routing op
    ordinal: int   # which flat cell of that operand

@dataclass(frozen=True, slots=True)
class ConstantCell:
    value: StaticValue

@dataclass(frozen=True, slots=True)
class RoutePlan:
    cells: tuple[CellSource, ...]   # index is the RESULT cell ordinal
```

`operand` indexes the routing op's operand list rather than naming a `BindingId`, so the row stays meaningful
for the dst-less ops (`StorePlace`, `PyStoreAttr`) that forced the key decision below, and so a route survives
a rebinding of the operand.

Every routing today is one instance of this shape:

| Site | Row |
| --- | --- |
| identity conversion | `cells[i] = OperandCell(0, i)` |
| transpose | `cells[i] = OperandCell(0, pi(i))` |
| subscript gather | `cells[i] = OperandCell(0, selection[i])` |
| positional projection, record projection | `cells[i] = OperandCell(0, start + i)` |
| concat | `OperandCell(0, i)` below the split, `OperandCell(1, i - n)` above |
| repeat | `cells[i] = OperandCell(0, i % count)` |
| `BuildTuple` / `BuildList` | each item operand's cells at its own window |
| construction | per-field windows from several operands; defaults as `ConstantCell` |

`ConstantCell` is exactly the `:= const` half the campaign's phrasing calls for, and its only present consumer
is the `CONSTRUCTION` default-field fill.

## Totality is the point

The row shape matters less than the property it buys. Because a `RoutePlan` names every result cell, absence
stops being meaningful. That kills the two `.get()`-means-something conventions in `_emit.py`, and it closes a
hole `verify_plan_totality` currently names in its own docstring and declines to check:

> `subscript_plans` and `route_plans` are read with `.get()`, where absence legitimately means positional
> projection and identity route, so their omissions are indistinguishable from intent and are NOT checked;
> verifying those needs the typed explicit variants M2 introduces

With a total plan, the verifier gains a real arm: every routing op in the stabilized graph has a `RoutePlan`,
its length equals the result layout's leaf count, and every `OperandCell` resolves to an in-range cell of an
operand that exists. That arm is the acceptance criterion for M2, and it is checkable rather than asserted.

This also removes the silent-absence failure mode that this campaign has been bitten by repeatedly: a recorder
that stops writing currently produces a plausible identity route, which is a wrong answer that looks like a
default. Under a total plan it produces a verifier error.

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

Alternatives considered and why they lose. Keying by `id(op)` matches what the analyzer does internally and
would work, since `ResidualUnit` holds the graph and every recorded op stays alive; but it is untypeable, it
cannot be serialized into the golden HIR dump, and M1 already showed that reasoning about id-keyed table
lifetime costs a review round per table. Stamping a stable `OpId` on every op at construction is the cleanest
in principle, but `_remap_op` clones ops with `replace()` during unrolling, which would duplicate the stamp
unless every clone site re-stamps explicitly -- a new invariant with no verifier behind it, introduced in the
step whose whole purpose is to make routing verifiable.

## Scope: what M2 absorbs and deletes in the same commit

Absorb: the four id-keyed tables (three routes plus the `_conversion_calls` classification flag, which is not
a route and should become a `CallLowering` discriminator rather than be forced into the routing type);
`subscript_plans` and `route_plans`; `CallPlan.construction`; the three unplanned re-derivations; and the two
partially-planned `child_slice` recomputations.

Delete: `_project` and `_install` collapse into the offset row; `_copy_leaves` becomes the identity row;
`_emit_conversion` keeps only its kind-coercion duty; `_emit_concat` loses its offset arithmetic entirely. One
leaf-walk remains where five plus three clones stand today.

## Open questions for X6a

1. Is `(BlockId, op index)` the right key, or does the serialization requirement (the golden corpus dumps
   pre-optimize frontend HIR through a schema-versioned serializer) argue for a stamped `OpId` after all?
   The measurement says positions are stable; the question is whether they are stable enough to be a
   contract rather than an observation.
2. J6 from the X5 ruling requires every kind promotion to be consumed from an explicit plan row rather than
   derived by inspecting emitted nodes. `_emit_conversion` performs kind coercion today (a `Residual(BOOL)`
   source under a `FLOAT` result leaf goes through `BoolToFloat`). Should `OperandCell` carry the promotion,
   folding J6 into M2 -- or does that overload a step already touching eight surfaces, and J6 belongs to M3?
3. `_conversion_calls` is a set, not a map. Grouping it with the three routing tables is accurate only in that
   all four are `id()`-keyed and round-reset. Confirm it should leave the routing type entirely.
4. The three dst-less ops include `UnbindPlace`, which routes nothing. Should it be excluded from the plan by
   construction, or carry an empty `RoutePlan` so the verifier's totality arm needs no exception?
5. Sequencing: is one commit right for absorb-plus-delete, given the corpus must stay byte-identical and the
   deletions are what prove the absorption complete? The alternative is absorb-then-delete across two commits,
   where the intermediate state has two sources of truth for the same routing.

## Risk

The corpus must regenerate byte-identically. Routing determines which cell reaches which port, so an error
here is an ABI change, and `store_order` is the part of the plan the campaign has already identified as being
the port ABI. No test names `subscript_plans` or `route_plans`; coverage of both is behavioural only.

The campaign says four examples exercise routing -- `routed_diamond`, `ekf1`, `fsc`, `imu` -- and this document
repeated it. `routed_diamond` DOES NOT EXIST. It was a spike artifact at `tests/spike_golden/kernels/` on the
branch that was deleted when Stage 3 closed, and it survives only in `spike-ledger.md`, which is a verbatim
record of that branch rather than of this tree. So there are three, not four, plus `polar` and `signal_window`
which the campaign's list omits. That is the second miscount found in this document's own subject matter, both
by re-deriving rather than reading.

The safety net was then measured rather than described. `tests/test_frontend_routing.py` was added first, one
swap-sensitive kernel per routing construct, and four routing mutants were run against it and against the
pre-existing suites: a perturbed transpose route, a rotated repeat, a swap inside each repeated unit, and a
rotated aligned copy. Every mutant the new module caught, the example-driven matrix and aggregate tests caught
as well -- so the net at whole-suite level was NOT as thin as the inventory implied, and the new module's value
is localization and a per-construct invariant rather than newly closed holes. One route is inherently
untestable and M2 should not try: `seq * n` yields identical copies, so permuting whole repetitions maps
identical content onto identical content.
