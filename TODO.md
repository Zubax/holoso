# TODO

## Integer support adjacent

The front end carries integers end to end -- Eel has a width-less int type with C promotion, HIR has the int
vocabulary -- and everything below HIR refuses them at `_reject_integers` (`_mir/_lower.py`). Lifting that gate is
what this section is about.

The oracles store wide values as `FloatValue` (`numerical.py`, `_mir/_interpret.py`); they need a
`FloatValue | IntValue` union, a typed payload for the `lir.wide_consts` pool (constants are float-encoded across
microcode/emit/html/model today), and one shared scalar port codec (cocotb and the model duplicate it).

`IntType` still lacks the value mapping its float sibling has: `value_class` has no `IntValue` to return, so
`SelectOperator(IntType(...))` is constructible and well-typed but its `evaluate` assertion refuses every payload
(and under `-O`, where that assertion is gone, it would pass integers through unchecked). Not reachable regardless:
`_reject_integers` refuses the kernel before lowering even begins.

The conditioner mapping is already in place -- an integer port carries `IntIdentity` and nothing else, because
integer sign conditioning cannot ride a port sideband the way `holoso_fsgnop` does (`holoso_iabss` is a pooled module
with a latency and a saturation output). What that did NOT reach is the float sign wiring in the microcode packer and
the Verilog emitter: the tapped-result sign field and its `_sgnop` wrapper binding are keyed on `is_wide` rather than
on the port's scalar type, and the per-operand pair is keyed on nothing at all -- it runs unconditionally over the
operator's arity. An integer pooled operator has no such ports, so all four must be keyed on `FloatType` when the
integer backend lands; they are left alone for now because both arms would be dead and untestable until an integer
operator exists to take the other one. Relatedly, `_Bank` in `_lir/_bankalloc.py` is generic over a
CONSTRAINED type variable, which cannot express `MirFloatStateSlot | MirIntStateSlot` for one wide bank; lifting it
means giving `MirFloatStateSlot.sign` and `MirBoolStateSlot.inversion` a common `conditioner` field.

`scalar_type_of` (`_lir/_ir.py`) recovers the scalar family from the LIR node class, so it still answers
`FloatType(fmt)` for the bank-named `WideInputLoad`/`WideOutputWire` carriers. When integer ports land the
discriminator must be re-established by giving those nodes a `scalar_type` field, not by reintroducing nominal
`Float*`/`Int*` sibling classes. Fixing `scalar_type_of` alone is not enough, because several consumers decide the
same thing independently rather than routing through it: `_coerce_inputs` and `NumericalSimulator.__init__` in
`_backend/numerical.py` and `_emit_consts` in the Verilog backend all read a wide carrier as float directly. Input
ports would at least fail loudly there (`_coerce_input` rejects a non-float), but wide constants and state would be
encoded as floats without a word. They must be converged onto the one dispatch first.

Strength reduction is float-keyed (`cval: dict[ValueId, float]`) and needs an int sibling with a typed-constant cache.

A kernel that never names an integer can still raise one and hit the gate: the roundings answer an integer over a
float, and `abs`/`min`/`max`/`np.sign` keep one they are handed. Only the adjacent `IntToFloat(FloatToInt(x))`
reduces today, so `float(math.floor(x))` and `abs(float(int(x)))` build while `math.floor(x) > 3`,
`float(math.floor(x) + 1)`, `float(abs(int(x)))`, `float(max(int(x), 0))`, an integer phi merging `floor` with
`ceil`, and any integer state slot do not. These lowerings are right -- the alternative silently computes over a
float image, and retyped an integer slot to a float register that stops counting at the mantissa width. Sinking
`IntToFloat` through a select and through the exactly-representable int arithmetic would recover most of them;
lifting the gate recovers all of them and is the real fix.

A speculated shift can carry an out-of-domain count, where `evaluate` names no number but the RTL defines a result;
the same is already true of float `mul`/`add` and the casts. Landing the backend means reconciling fold, MIR model
and RTL on what an out-of-domain operand answers.

The integer kernels in `tests/_eel_corpus.py` (UART, CRC/LFSR, NCO, PWM, debouncer, priority encoder) are the
acceptance set: each is oracle-verified against CPython through HIR and records its MIR refusal, so lifting the gate
turns those refusals into end-to-end coverage. `examples/uart.py` carries its counters as floats until then.

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

## HIR operator mnemonics

The operator class names are uniform (`Float*`, `Int*`, `Bool*`); the mnemonics they carry are not, in three ways.

Twenty-five float mnemonics carry no family marker where every int and bool one does -- `add`, `mul`, `select`
against `iadd`, `band` -- so float reads as the namespace default. The float comparisons contradict even that, being
marked (`flt`, `feq`), so one family is spelled two ways. Marking the rest is the floatism to remove, at the cost of
churning residual Eel dumps, HIR printouts, HTML reports and the test assertions over them.

The muxes name their family with a word, `bool_select` and `int_select`, where the rest of the vocabulary uses a
letter.

The integer bitwise operators drop the `Bw` their class names carry; extend `iand`, `ior`, `ixor` and `inot` to
`ibwand`, `ibwor`, `ibwxor` and `ibwnot`.
