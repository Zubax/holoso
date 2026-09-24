# TODO

## Frontend limitations

An empty array slice (`v[:0]`) is refused where it is taken rather than where it is used, so even `len(v[:0])` fails;
an empty sequence slice is fine. An empty array carries no leaves, so the leaf-type and shape checks cannot run --
which is what must reject `-boolflags[:0]` and `a[:0,:] + b[:0,:]`, both of which CPython rejects too. Accepting the
valid empty-float case needs an empty-but-typed array in the value model.

A counted `for` above the unroll threshold turns its target into a runtime integer, so a body that indexes an
aggregate with it (`v[i]`) is refused at the subscript, and one that converts it needs the conversion operator the
unrolled form folds away; forcing the unroll with `for i in list(range(...))` is the rewrite. Its body is a
data-dependent region, so installing a whole aggregate into a state attribute there is refused as it is in a
`while` -- storing elements is the rewrite. A range returned across a helper's own data-dependent region is refused
at the call, and an unconditional-exit body (`for i in range(n): break`) keeps the existing no-back-edge refusal.

A data-dependent loop carries the syntactic set of names its body assigns, fixed before the body is interpreted
because the header phis must exist first. A leaked `for` counter assigned only on a statically-dead path
(`if False: i = ...`), or by a loop that rebinds nothing (`for i in []`), is carried anyway and stops being a
compile-time integer, so a later `v[i]` is refused as a non-static index. Only the compile-time index is lost; the
loop computes correctly. Exactness needs a fold-aware carried set, which the loop setup cannot have without
interpreting the body first.

Re-installing a tensor derivation into the state attribute it came from (`self.P = self.P.T`) is refused by the
state-disjointness rule even though the slot is fully overwritten, because the derivation shares its source's storage.
The diagnostic names the fix (`np.array(self.P.T)`); lifting it needs the install check to see that the source and the
destination are the same slot.

Also refused, each naming the construct and each with a plain rewrite: a walrus that reads the name it binds
(`b = (a := a + x)`); a `list[...]` return annotation, in favour of `tuple[...]`, though a returned list literal stays
legal; a value that is an array in one arm and a sequence in the other, since aggregates join a branch only when every
arm agrees in kind; a comprehension `if` filter or a comprehension with more than one `for` clause; and a starred
unpack target (`first, *rest = v`).

`while True:` with a data-dependent `break` exhausts the graph expansion budget rather than residualizing: a literal
`True` header is decidable on every trip, so the loop unrolls until the budget stops it, while a runtime `break` opens
an exit lane without ever closing the fall path. The exit condition belongs in the header. This is deliberate --
every cheap detector for the shape misfires on the legitimate counter-spelled loop.

A state attribute's shape and type come from the reset snapshot, so a field annotation contradicting it
(`P: Float64[np.ndarray, "2 2"]` on an instance holding a 3x3) is documentation rather than a checked declaration.
Parameter and return annotations are checked, so the module boundary is judged while the state boundary is not.

## HIR

### Jump chains left by a settled branch, and trivial loop phis

When pruning settles a branch the front end could not (a guard only an identity decides, such as `if x*0.0 > 1.0`),
the surviving path is left as a chain of blocks joined by jumps, `P -> T -> M`, each single-predecessor. Nothing fuses
them, and every block boundary drains, so the guard costs latency though not hardware: a straight-line kernel with a
settled guard took 19 cycles against 18 without it, and inside a 100-trip loop 2409 against 2109 (24 against 21 per
trip). The fix fuses a block into its sole predecessor when that predecessor jumps to it -- append its operations, take
its terminator, retarget successor phi arms as if-conversion's splice already does. It cannot form the branch on a phi
with an arm from its own block that LIR refuses, since fusion follows only jump edges and the emitter never points a
branch at a loop entry. Once it exists, the emitter's single-return-site special case becomes removable.

The same pass should fold a phi whose arms other than itself all name one value: pruning substitutes only single-armed
merges, and `rebuild` defers every loop-header phi past strength reduction, so `acc = 2.0; while ...: acc *= 1.0`
keeps a header phi that stops `acc * x` from becoming a scaling. The substitution is dominance-safe (the other value
reaches the merge on every first arrival) and must iterate with pruning, since a substituted boolean merge can settle
another branch. A prototype of both took about 64 lines in `_prune.py`, left all bundled examples unchanged (none has
either shape) and matched the model on about 1000 generated programs. They are optimizations rather than
simplifications, which is why the middle-layer cleanup left them out.

## LIR

### Pruning state slots nothing reads

A state slot whose live-in nothing reads is unobservable -- a public attribute's `state_<attr>` port reads the live-out
as an ordinary output -- yet it keeps its register, its install and its whole computation cone: `self._last = y * 7.0`,
never read, costs a register and a multiply, and fir keeps the shifted-out `_line_0` (8 registers where 7 suffice). The
fix belongs in HIR dead-code elimination: root the slots through their state reads instead of unconditionally, so a
slot survives only while its live-in is read, and drop the others. That subsumes LIR's
`_drop_redundant_state_slots`, whose aliases are exactly the unread members. It is deferred because about 15 LIR
witness tests build their shapes out of write-only slots (`test_aliased_state_slots_merge_onto_one_register`,
`test_cfg_write_only_state_slot_is_reserved`, `test_chained_slot_live_in_blocks_early_install`,
`test_two_slots_ending_on_one_value_hold_it_once_and_copy_once`, the `shared_live_out` steering witnesses, the
boundary-install and gap-tenant verification tests and their cosimulation twins) and must be rebuilt on slots that are
read, while the write-only slot reservation in `_bankalloc.py` becomes unreachable and goes with them.

### Blocks that only install settled phi-arm copies

A branch arm whose block holds nothing but phi-arm copies of values already settled (constants, inputs, state reads,
landed results) still drains and takes its own PCs, about four cycles on that path: finite_set_current_controller's
boolean constant arms (160 -> 156 on its shortest path), foc and imu_fusion arm blocks, image_agc_streamed,
octave_index, remainder. The narrowest sound fix hands the copies to a predecessor that spills nothing, landing them on
its terminator so the arm block becomes empty and takes no PC; the phi register then interferes with the other arm's
live values, which can cost a register under pressure. Arms whose predecessor spills (pid, foc's and imu_fusion's
first arms) or is the empty entry (flux_observer) gain nothing without draining overlaps into merges. Threading the
empty arm straight into the merge trades latency between the paths instead (a probe moved `x / y if c else 0.0` from
6/18 to 4/20 cycles), so a general version needs a schedule-aware acceptance test and was sized above medium.

### List-scheduler priority

`schedule_ops` issues ready firings by latency-weighted height to a sink, which ignores instance contention. A
randomized slack-based priority, 300 trials per block under an independent timing model, found shorter blocks the
height order misses: rigid_body_scalar 126 -> 124 cycles, imu_fusion's entry and one arm block one cycle each. Other
large kernels (both EKFs, cordic_sincos, foc) showed no gain, so the benefit is uneven; a contention-aware priority, or
a few perturbed orders per block keeping the shortest, are the candidates.
