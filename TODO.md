# TODO

## Integer support adjacent

The front end carries integers end to end -- Eel has a width-less int type with C promotion, HIR has the int
vocabulary -- and everything below HIR refuses them at `_reject_integers` (`_mir/_lower.py`). Lifting that gate is
what this section is about.

The pooled integer operators exist (`iadds`, `isubs`, `imuls`, `idivs`, `iabss`, `ishift`, `icmp`), each owning its
closed-form latency, its RTL parameters and its port names. Nothing selects them: MIR still refuses every integer
node. Two carry more than a lowering needs, so the lowering chooses: `ishift` emits both the raw shift and its
saturating reading, `idivs` the quotient and the remainder together. There is no negation module -- `ineg` is
`isubs(0, x)`, which saturates `-MIN` correctly. The saturation sideband every module raises is left unconnected,
because HIR marks the saturating operations speculatable and an if-converted arm must not raise the error flag; were
a kernel ever to want the flag it belongs on the operator as an ordinary boolean result lane, never in `error_ports`.

The operators still missing all have their RTL already; what is absent is the Python model. The int/float boundary
needs one for `holoso_ffromint`/`holoso_ftoint`, zkf-backed and float-parameterized rather than integer-family. The
inline integer operators need one each for the bitwise ops, the bool casts, and the constant shift `holoso_ishiftc`;
the mux already answers for integers, needing only its lowering.

The oracles store wide values as `FloatValue` (`numerical.py`, `_mir/_interpret.py`); they need a
`FloatValue | IntValue` union, a typed payload for the `lir.wide_consts` pool (constants are float-encoded across
microcode/emit/html/model today), and one shared scalar port codec (cocotb and the model duplicate it). The value
half is in place: `IntValue` is the saturating dual of `FloatValue` and `value_class` answers for `IntType`.

The conditioner mapping is already in place -- an integer port carries `IntIdentity` and nothing else, because
integer sign conditioning cannot ride a port sideband the way `holoso_fsgnop` does (`holoso_iabss` is a pooled module
with a latency and a saturation output). What that did NOT reach is the float sign wiring in the microcode packer and
the Verilog emitter: the tapped-result sign field and its `_sgnop` wrapper binding are keyed on `is_wide` rather than
on the port's scalar type, and the per-operand pair is keyed on nothing at all -- it runs unconditionally over the
operator's arity. An integer pooled operator has no such ports, so all four must be keyed on `FloatType` when the
integer backend lands; they are left alone for now because both arms would be dead and untestable until a lowering
selects an integer operator. Underneath them the families disagree on what `module_name` denotes: a float operator
names a Holoso wrapper that already carries the sign-conditioning ports, an integer one names the bare core, so the
wrapper the emitter assumes is not there on that side. Either the integer cores gain wrappers or the emitter learns
the difference, and that choice belongs with the four sign sites rather than before them.
`_Bank` in `_lir/_bankalloc.py` is generic over a CONSTRAINED type variable,
which cannot express `MirFloatStateSlot | MirIntStateSlot` for one wide bank; lifting it
means giving `MirFloatStateSlot.sign` and `MirBoolStateSlot.inversion` a common `conditioner` field.

`scalar_type_of` (`_lir/_ir.py`) recovers the scalar family from the LIR node class, so it still answers
`FloatType(fmt)` for the bank-named `WideInputLoad`/`WideOutputWire` carriers. When integer ports land the
discriminator must be re-established by giving those nodes a `scalar_type` field, not by reintroducing nominal
`Float*`/`Int*` sibling classes. Fixing `scalar_type_of` alone is not enough, because several consumers decide the
same thing independently rather than routing through it: `_coerce_inputs` and `NumericalSimulator.__init__` in
`_backend/numerical.py` and `_emit_consts` in the Verilog backend all read a wide carrier as float directly. Input
ports would at least fail loudly there (`_coerce_input` rejects a non-float), but wide constants and state would be
encoded as floats without a word. They must be converged onto the one dispatch first.

Strength reduction folds integer constants and applies the algebra each operator declares, so `x*0`, `x*1`, `x+0`,
`x&0`, `x|-1`, `x&-1`, `x|0` and `x^0` reduce, and the mux identities cover `iselect` beside its float and bool
siblings. Deferred is everything past that. The non-commutative integer operators declare no identity deliberately,
because the shared algebra drops an identity operand wherever it sits and so would rewrite `0 - x` to `x`; `x-0`,
`x//1`, `x%1`, `x<<0` and `x>>0` therefore survive and need rules of their own. So do the integer counterparts of the
float-specific rewrites -- `x*-1` to `ineg`, multiplication and floor division by a power of two to a shift, and the
value-equality folds `x-x`, `x^x` and `x&x`. Each of those must answer for saturation at the format extremes, a
question the float rules never faced.

A kernel that never names an integer can still raise one and hit the gate: the roundings answer an integer over a
float, and `abs`/`min`/`max`/`np.sign` keep one they are handed. Only the adjacent `IntToFloat(FloatToInt(x))`
reduces today, so `float(math.floor(x))` and `abs(float(int(x)))` build while `math.floor(x) > 3`,
`float(math.floor(x) + 1)`, `float(abs(int(x)))`, `float(max(int(x), 0))`, an integer phi merging `floor` with
`ceil`, and any integer state slot do not. These lowerings are right -- the alternative silently computes over a
float image, and retyped an integer slot to a float register that stops counting at the mantissa width. Sinking
`IntToFloat` through a select and through the exactly-representable int arithmetic would recover most of them;
lifting the gate recovers all of them and is the real fix.

A speculated shift can carry an out-of-domain count, where the HIR fold names no number but the RTL defines a result;
the same is already true of float `fmul`/`fadd` and the casts. The hardware side has since picked its answer -- the
shift operator is total over every representable count, saturating the amount at the word, as its module does -- so
what remains is reconciling the fold with the model and the RTL, which now agree.

The integer kernels in `tests/_eel_corpus.py` (UART, CRC/LFSR, NCO, PWM, debouncer, priority encoder) are the
acceptance set: each is oracle-verified against CPython through HIR, so lifting the gate extends that to end-to-end
coverage by adding them to the MIR-lowering parametrization. `examples/uart.py` carries its counters as floats
until then.

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
