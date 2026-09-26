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

### Copy-only arms behind an overlapping predecessor

The LIR build threads an arm block holding nothing but phi-arm installs into its predecessor wherever that shortens
the arm's path and lengthens none. An arm whose predecessor overlaps keeps its frame, about four cycles on that path:
pid's first arm, foc's and imu_fusion's first arms, image_agc_streamed's and majority_voter's block 2. Threading them
makes the predecessor drain into the merge, trading latency between the paths (a probe moved `x / y if c else 0.0`
from 6/18 to 4/20 cycles); flux_observer's arm sits behind the empty entry, whose frame would grow from 2 to 4 PCs.
Gaining here needs overlap across a multi-predecessor edge.

### Arm-threading trial cost

Arm threading judges each candidate by rescheduling and reconverging the whole graph, so build time grows with the
candidate count times the graph size: negligible on every example, but a generated 4,896-block kernel with 1,693
threadable arms spends about 13 minutes in it at regalloc effort 0 (0.45 s per candidate). A candidate's effect is
local to its predecessor's frame and the merge phi's interference, so an incremental judgement is possible; it is
deferred until a real kernel needs it.

### List-scheduler priority

`schedule_ops` issues ready firings by latency-weighted height to a sink, which ignores instance contention. A
randomized slack-based priority, 300 trials per block under an independent timing model, found shorter blocks the
height order misses: rigid_body_scalar 126 -> 124 cycles, imu_fusion's entry and one arm block one cycle each. Other
large kernels (both EKFs, cordic_sincos, foc) showed no gain, so the benefit is uneven; a contention-aware priority, or
a few perturbed orders per block keeping the shortest, are the candidates.
