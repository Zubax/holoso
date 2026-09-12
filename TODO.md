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

### Scheduler tie-break for multi-instance co-issue

A firing is one activation of a pooled operator, an instance is one physical
copy of it, and a mux arm is one input of a register-file multiplexer. When a pooled operator has several instances, the
list scheduler issues ready firings in critical-path order and binds each to the first free instance. Two firings that
share an operand and co-issue on two instances force that value onto a read port of each instance, one mux arm per
instance, and the register allocator cannot undo it: both firings are needed that cycle, so neither can move off. The
scheduler decides this before the allocator sees anything.

Proposed: break ties among ready firings of equal height (the sort key is `(-height, leader)`) in favor of the firing
sharing no operand with a same-class firing already issued this cycle, and then bind that firing to the instance whose
ports already read its operands (value-level affinity: the instances' operand sets so far). The tie-break is
latency-neutral by construction, since only ties are reordered, and the value-id tail keeps the order deterministic.

Measured once as a prototype, scoring each example kernel by the allocator's objective (its mux arms plus two per
register) and summing over the kernels: unchanged with one instance per operator; with two multipliers the sum fell from
1334 to 1327, and with two instances of every pooled operator from 1342 to 1332, while each EKF kernel gained one to
three arms and one kernel's transaction shortened by a step. The difference is not significant: the annealer's result
already varies by that much between random seeds, the tie-break changes the allocation the annealer starts from, and the
allocator's instance-binding moves already recover most of the duplicated arms. Considerations for the next attempt:
measure at the shipped allocator tuning, confirming that the latency figures the metrics test freezes stay put, and read
the hardware on the two-multiplier EKF under Vivado; prefer an affinity that survives allocation (the allocator
re-labels instances, so the scheduler's choice matters only through which firings co-issue, not which label they take);
consider one scheduler-allocator round trip instead, re-scheduling with the allocation's port affinities as the
tie-break; and keep the per-cycle ready set small enough that the extra ranking stays linear.

### Schedule once, outside the install fixpoint

The install fixpoint is the loop in `_converge_layout` that decides which blocks end with install copies
and repeats the block layout until that set is stable. Each round calls `layout_and_coalesce`,
which reschedules every block, rebuilds the scheduler's first-free instance binding and the instance counts, and
reconstructs the constant pool, although only three things can change between rounds: the block makespans including
their installs, the terminator offsets (the cycle at which a block hands control to its successor) of the blocks that do
not overlap their successors, and the coalescing. A block leaves the install set only by coalescing its phi arms away,
and its successor is then still multi-predecessor, so which blocks may overlap their successors, and the results still
in flight across each such hand-over, are the same in every round (the argument `_converge_layout` states). Splitting
`schedule_with_overlap` into a schedule pass run once after `_prepare` and a per-round offset derivation would make that
invariance structural instead of argued.

### One source for the handshake-gated write arms

A register's write multiplexer has arms selected by the microcode opcode and, for a few registers, arms selected by the
handshake instead: the input load and the persistent-state installs. `write_arms` in `holoso/_lir/_sources.py` restates
by hand which registers get such a handshake-gated arm, and the build's assertion that the register allocator's arm
count equals the emitted write-multiplexer arm count is only as strong as that restatement: a handshake arm added to the
emitter alone would go uncounted. A typed list of those arms per register, built once in `_sources.py`, rendered by the
emitter's clocked block and counted by `write_arms`, would close the gap.
