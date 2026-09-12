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

### Boundary-install drain charge on an empty Ret block

A slot installing at the accepted-output edge charges the Ret block's drain to the boundary step of its makespan,
which is a no-op whenever the block computes anything (its own landings reach that step) and costs two to three
cycles when it does not: a delay line, an input latched straight into a slot, a boolean-only slot. The charge looks
like a pure tightness cost, since an empty Ret block's install sources are resident by construction, and the
numerical model agreed with Python on every affected kernel with it forced off (delay lines from three cycles to
one, two merge-into-Ret shapes from nine to six). Dropping it is behavior-changing: it re-freezes the chained-copy
latency rows, needs the liveness diagnostic taught that in a one-cycle transaction the read-first boundary write
coincides with the live-in's own landing, and wants the RTL cosimulation of those kernels before it lands.

## Convex write-select cost in the register allocator

The register allocator prices every multiplexer arm alike, so a register whose write select (the multiplexer in front
of a register's data input) grows from four to seven inputs costs the same three arms as three separate two-input
selects. On the ECP5 fabric each extra pair of inputs is another logic level on every path that ends in that
register. The synthesis matrix showed the effect once the allocator started merging registers at the register price:
the pid Diamond row's critical path moved from inside the adder into the write select of a register that went from
four to seven inputs (its fmul row now carries a pack stage for it), and flux_observer (three to six) and imu_fusion
(nine to eleven) widened the same way without failing. The refinement DESIGN.md anticipates is a convex per-endpoint
cost: an arm beyond the fourth input of a select priced above one, so the allocator stops widening a select where the
fabric would add a level. It re-freezes every metrics row and needs the synthesis matrix to validate, so it waits for
a change that is measured end to end.
