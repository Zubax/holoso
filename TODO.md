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

Two obligations those operators decline. HIR shift counts are width-less while the constant shift serves only counts
that are shifts at all -- one reaching the word answers a constant or a sign fill, and zero the identity -- so folding
and clamping are the lowering's job. And `FloatToInt` over a `FloatRound` is NOT unconditionally the one `ftoint`
carrying that mode: where the float rounding itself overflows, the rounded value is an infinity that saturates to a
rail while the direct conversion answers the integer. In `ZkfFormat(2, 4)`, `3.5` rounds to `+inf` and thence to
`INT_MAX` where a direct nearest-even conversion gives `4`. It needs a format whose largest finite value is
non-integral, so most never show it, and `TRUNC` never mismatches because truncation cannot increase magnitude.
The fastmath charter applies here.

The oracles store wide values as `FloatValue` (`numerical.py`, `_mir/_interpret.py`); they need a
`FloatValue | IntValue` union and one shared scalar port codec (cocotb and the model duplicate it). The
`lir.wide_consts` pool already carries that union, though it still dedups on the source float rather than on the
encoding, so two literals that encode identically get two `const_N` wires.

The conditioner mapping is already in place -- an integer port carries `IntIdentity` and nothing else, because
integer sign conditioning cannot ride a port sideband the way `holoso_fsgnop` does (`holoso_iabss` is a pooled module
with a latency and a saturation output). What that did NOT reach is the wide datapath below MIR, which assumes float
wherever it keys on `is_wide` or on nothing at all, where it must key on `FloatType`. None of it is reachable until a
lowering selects an integer operator, so it is left alone rather than fixed blind, but the sites do not fail alike and
the order of the eventual fix matters. Three refuse loudly: `_wide_source_net` and `_render_inline` assert a wide
conditioner is a `FloatSignControl`, where an integer carries `IntIdentity`; the operator instantiation binds a
`_sgnop` per operand and per wide result unconditionally, which `ffromint`'s integer operand and `ftoint`'s integer
result do not have; and the microcode packer allocates and asserts the matching sign fields on both sides. One would
MISCOMPILE in silence, and is today masked only by the loud ones: `_emit_declarations` sizes every pooled operand
read-mux register and every wide result wire at `WFLT` rather than at the port's own width, so an integer port wider
than the float silently loses its top bits. Fix that one first, or fixing the loud ones uncovers it.

`scalar_type_of` (`_lir/_ir.py`) recovers the scalar family from the LIR node class, so it still answers
`FloatType(fmt)` for the bank-named `WideInputLoad`/`WideOutputWire` carriers. When integer ports land the
discriminator must be re-established by giving those nodes a `scalar_type` field, not by reintroducing nominal
`Float*`/`Int*` sibling classes. Fixing `scalar_type_of` alone is not enough, because `_coerce_inputs` and the slot
reset snapshot in `_backend/numerical.py` decide the same thing independently rather than routing through it. Input
ports would at least fail loudly there (`_coerce_input` rejects a non-float), but wide state would be encoded as a
float without a word. They must be converged onto the one dispatch first.

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

The emitted RTL requires an empirical study of the optimal way to bit-extend float results into the register file
when WREG>WFLT. The default treatment of `x <= y` where x is wider than y is to zero-fill the higher bits,
which may potentially require a wider mux and extra wires while we don't care about the value of those bits.
We must explore the possible alternatives, such as marking them explicitly as don't-care,
e.g., `x <= {{X{1'bx}}, result8}` for X extra bits, or a similar solution.
The winner will be chosen based on the synthesis metrics across Diamond, Vivado, and Yosys.

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
