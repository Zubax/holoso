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

## LIR

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
