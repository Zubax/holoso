# TODO

## Integer support adjacent

The native integer width is `max(wint_min, ffmt.width)`, so a kernel that builds no float operator at all still
inherits the default float format's width: the UART's byte lane emerges as a 24-bit port for a 0..255 value, and the
LFSR's genuine `wint_min=17` is inert because the default format is already wider. Sizing an integer design
therefore runs through a float format it will never instantiate, and the
only way to tighten the word today is to shrink a format nothing reads. The floor exists so one register can hold
either family, but when no float operator is configured no float ever enters the register file. Letting `ffmt` go
absent in that case, so the word comes from `wint_min` alone, would end the ritual; it would also give
`ExampleSpec.formats` a meaning for a float-free kernel, where two formats today differ only in register width.

Integers are signed and saturating with no modular flavour, so every wrapping accumulator carries an explicit mask
-- and the mask cannot rescue the operation it guards, because saturation happens inside the add before the mask
applies. A 32-bit DDS phase accumulator therefore forces a 34-bit machine word (one carry bit, one sign bit), and
since the width is global those two bits are paid on every register and port in the design to serve one add. A
wrapping add, or an unsigned integer type, would let the accumulator be exactly as wide as it is and delete the mask.
One potential approach is to *infer* wrapping/unsigned operations from the adjacent static mask applications.

Saturation is invisible in the numerical model, which is where the wrong answers appear first: an accumulator that
clamps instead of wrapping surfaces only as wrong numbers, with nothing naming the operation that clamped. A
model-level saturation flag, or an opt-in assertion, would name that defect immediately -- worthwhile for a datapath
whose headline property is saturation. The sharpest case is silent: a mask whose modulus the word can hold but whose
sum it cannot is accepted without complaint, because the refusal fires on the mask literal and `0xFFFFFFFF` is
exactly the maximum of a 33-bit word. Tying a masked addition's modulus to the width its operands
need would catch at compile time exactly the mistake the mask idiom invites -- and would also give `wint_min`, today
a declaration the compiler never checks against the kernel, something to be checked against.

The sincos core is turn-native but only radians are exposed, so `sin(x)` lowers as `fsincos(x * (1 / (2 * pi)))`.
A DDS phase is already in turns, so it multiplies by `tau / 2**WPHASE` purely so the compiler can multiply the
factor back out: two multiplies and two roundings to compute an identity, and that round trip dominates the output
error of the I/Q oscillator, exceeding the core's own. A turn-native intrinsic would delete both multiplies, both
roundings, and the scaler the kernel otherwise needs -- direct digital synthesis being the natural application of
the hardware that already exists.

There is no population count or XOR reduction: the parity of a byte, one operator in hardware, is written as an
eight-trip loop of `&`/`>>`/`^` that folds correctly but cannot say what it means. `int.bit_count()` is refused
(cleanly, naming the attribute). Recognizing it, or the unrolled reduction it expands into, would close the gap.
The second witness is `examples/majority_voter.py`, whose 3-of-5 vote is ten explicit three-way conjunctions where a
popcount says `>= 3` -- a voter over more channels grows as C(n, k) terms and stops being writable at all. Both that
example and `examples/uart.py`'s parity loop should be rewritten over the operator once it exists, since each is
today a hand-expansion of the very reduction the hardware would do in one step.

The generated bench asserts `err_pc == 0` on every vector, so a transaction whose defined answer includes an
asserted error sideband -- an input-fed `x // 0`, a float division by zero -- cannot be cosimulated end to end;
that behavior is covered at the operator-bench and model levels instead. Letting an explicit vector declare its
expected `err_pc` would close this for both families. The bench also caps a transaction at 2^20 cycles, so a valid
loop that legitimately runs longer -- an input-fed counter, a counted loop over a huge static range -- is reported
as a runaway; the numerical model carries its own, far higher ceiling.

## Frontend limitations

Valid kernels that are conservatively rejected. None is a wrong answer; each is a located refusal with a rewrite.

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

A tuple swap of two state attributes (`self.a, self.b = self.b, self.a + x`) is refused because an unpack target must
be a plain name. It is the one refusal here with no one-line rewrite -- the swap needs a temporary.

State is owned by exactly one object, the receiver of the traced entry method, so a component cannot hold a
sub-component that keeps a register of its own: an oscillator built around an accumulator instance is refused with
"an attribute store is only supported on the entry method's receiver", and inheriting the accumulator instead moves
the refusal to "a helper method cannot write attributes of the receiver; only the entry method may". Both name the
child's own store. Composition of behaviour is available -- a pure method on a snapshotted object inlines, so a child
that owns the width while the register stays with whoever ticks it compiles to byte-identical hardware -- but the
spelling a hardware engineer reaches for first, a component instantiating a stateful component, is the one that
fails. Flattening every register into the entry object is the rewrite, and it is the reason `examples/nco.py` and
`examples/iq_oscillator.py` each carry their own phase accumulator instead of the latter reusing the former.

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
