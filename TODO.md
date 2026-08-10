# TODO

## Integer support adjacent

The integer kernels in `tests/_eel_corpus.py` (UART, CRC/LFSR, NCO, PWM, debouncer, priority encoder) are the
acceptance set, covered through the MIR interpreter and the numerical model, with an RTL cosimulation subset in
`tests/test_cosim_int.py`; porting `examples/uart.py` off its float-carried counters remains open.

The generated bench asserts `err_pc == 0` on every vector, so a transaction whose defined answer includes an
asserted error sideband -- an input-fed `x // 0`, a float division by zero -- cannot be cosimulated end to end;
that behavior is covered at the operator-bench and model levels instead. Letting an explicit vector declare its
expected `err_pc` would close this for both families.

A `for` that cannot unroll -- a dynamic trip count, or a static one above the unroll threshold -- is rejected rather
than lowered to a counted back-edge loop, which needs a runtime integer counter; with runtime integers now carried
through the whole pipeline, the counted back-edge loop is the natural follow-on.

## White-box test promotion

The `whitebox` marker is a promotion queue, not a category: it marks a test that reaches past the public API where a
black-box spelling is now possible. The integer selection and RTL-shape suites carry it; each should become an
ordinary kernel driven through `synthesize`, and the marker should go with the rewrite.

Predating the marker is a large body of tests that reach into `lower_to_mir`, `build` and the allocation tables
directly -- the schedule, install-landing, const-install, overlap and microcode suites among them. Sweep them the
same way and promote whatever the public API can now express. That sweep is its own piece of work.

## Frontend limitations

Valid kernels that are conservatively rejected. None is a wrong answer; each is a located refusal with a rewrite.

An empty array slice (`v[:0]`) is refused where it is taken rather than where it is used, so even `len(v[:0])` fails;
an empty sequence slice is fine. An empty array carries no leaves, so the leaf-type and shape checks cannot run --
which is what must reject `-boolflags[:0]` and `a[:0,:] + b[:0,:]`, both of which CPython rejects too. Accepting the
valid empty-float case needs an empty-but-typed array in the value model.

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

A tuple swap of two state attributes (`self.a, self.b = self.b, self.a + x`) is refused because an unpack target must
be a plain name. It is the one refusal here with no one-line rewrite -- the swap needs a temporary.

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
