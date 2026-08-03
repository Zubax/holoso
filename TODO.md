# TODO

## Integer support adjacent

The front end carries integers end to end -- Eel has a width-less int type with C promotion, HIR has the int
vocabulary -- and everything below HIR refuses them at `_reject_integers` (`_mir/_lower.py:101`). Lifting that gate is
what this section is about.

The oracles store wide values as `FloatValue` (`numerical.py`, `_mir/_interpret.py`); they need a
`FloatValue | IntValue` union, a typed `lir.wide_consts` pool (constants are float-encoded across
microcode/emit/html/model today), and one shared scalar port codec (cocotb and the model duplicate it).

Strength reduction is float-keyed (`cval: dict[ValueId, float]`) and needs an int sibling with a typed-constant cache.

`abs`, `min`, `max` and `np.sign` of a runtime integer answer a float: registry dispatch is by callee identity alone,
so each entry names one operator with the operand type playing no part. Nothing is silent -- the conversion consumes
an integer-typed value the gate refuses, and the float result is loud one step later (`abs(a) % 3` reports that `%` is
integer-only). The fix is an entry carrying the integer form alongside the float one, chosen by operand type as the
arithmetic operators already are: `IntAbs` exists and would serve `abs`, `np.sign` follows from `IntRelational`.
`min`/`max` will be implemented as library functions because there is no dedicated integer sort hardware operator.

The integer kernels in `tests/_eel_corpus.py` (UART, CRC/LFSR, NCO, PWM, debouncer, priority encoder) are the
acceptance set: each is oracle-verified against CPython through HIR and records its MIR refusal, so lifting the gate
turns those refusals into end-to-end coverage. `examples/uart.py` carries its counters as floats until then.

## Frontend limitations -- to be addressed by the next frontend redesign

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
an exit lane without ever closing the fall path. The exit condition belongs in the header. This is the ratified
position -- every cheap detector for the shape misfires on the legitimate counter-spelled loop.

A state attribute's shape and type come from the reset snapshot, so a field annotation contradicting it
(`P: Float64[np.ndarray, "2 2"]` on an instance holding a 3x3) is documentation rather than a checked declaration.
Parameter and return annotations are checked, so the module boundary is judged while the state boundary is not.
