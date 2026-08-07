# TODO

## Integer support adjacent

An integer now travels from the front end to LIR: Eel has a width-less int type with C promotion, HIR has the int
vocabulary and folds it exactly, MIR selects hardware for it, and LIR schedules, binds and allocates it. What is
left is the backends. `synthesize` refuses a built LIR that still names an integer (`_refuse_integer_lir`,
`_api.py`), listing every surviving port, constant, state slot and operator; lifting THAT gate is what this section
is about.

The lowering taps the two multi-output integer operators where a shift and a division mean it: `ishift` port 0, the
raw reading that lets a left shift drop what leaves the word, and `idivs` ports 0 and 1, whose quotient and
remainder fuse into one firing. There is no negation module -- `ineg` is `isubs(0, x)`, which saturates `-MIN`
correctly -- and a right shift borrows the same subtractor to negate its count for the left-positive shifter, at
runtime, even where the count is constant. The constant-shift fold will remove that subtractor along with the
`ishift` itself; until then a constant count too wide for the machine format is refused not as a shift but by the
generic literal range check every integer constant meets, so `x << 100000` and `x + 100000` fail alike. A constant
NEGATIVE count is refused in the lowering instead, because CPython has no such shift and the left-positive shifter
would otherwise read it as the opposite direction; a runtime negative count remains the hardware's total answer.
The saturation sideband every module raises is left unconnected, because HIR marks the
saturating operations speculatable and an if-converted arm must not raise the error flag; were a kernel ever to want
the flag it belongs on the operator as an ordinary boolean result lane, never in `error_ports`.

One obligation the operators decline is still open. `FloatToInt` over a `FloatRound` is NOT unconditionally the one
`ftoint` carrying that mode: where the float rounding itself overflows, the rounded value is an infinity that
saturates to a rail while the direct conversion answers the integer. In `ZkfFormat(2, 4)`, `3.5` rounds to `+inf`
and thence to `INT_MAX` where a direct nearest-even conversion gives `4`. It needs a format whose largest finite
value is non-integral, so most never show it, and `TRUNC` never mismatches because truncation cannot increase
magnitude. The fastmath charter applies here, so the fusion is owed regardless.

The numerical model stores every wide value as a float and says so with asserts in `_read` and `_write`, while the
slot reset snapshot leans on `FloatValue.from_float`'s own type check; the cocotb codec duplicates the assumption.
Both need a `FloatValue | IntValue` union and one shared scalar port codec. The model's `_coerce_inputs` and its
slot reset decide the family independently of the port metadata rather than routing through it, which the typed
`scalar_type` every wide carrier now names makes possible; converge them onto that one dispatch first, or an integer
input port fails loudly in one place and an integer slot is encoded as a float without a word in the other.
The `lir.wide_consts` pool carries both families, though its float half still dedups on the source Python float
rather than on the encoding, so two float literals that encode identically get two `const_N` wires.

What the conditioner mapping did NOT reach is the wide datapath in the Verilog backend, which assumes float wherever
it keys on `is_wide` or on nothing at all, where it must key on `FloatType`. None of it is reachable while the LIR
refusal stands, so it is left alone rather than fixed blind, but the sites do not fail alike and the order of the
eventual fix matters. Three refuse loudly: `_wide_source_net` and `_render_inline` assert a wide conditioner is a
`FloatSignControl`, where an integer carries `IntIdentity`; the operator instantiation binds a `_sgnop` per operand
and per wide result unconditionally, which `ffromint`'s integer operand and `ftoint`'s integer result do not have;
and the microcode packer allocates and asserts the matching sign fields on both sides. One would MISCOMPILE in
silence, and is today masked only by the loud ones: `_emit_declarations` sizes every pooled operand read-mux
register and every wide result wire at `WFLT` rather than at the port's own width, so an integer port wider than the
float silently loses its top bits. Fix that one first, or fixing the loud ones uncovers it.

Strength reduction folds integer constants and applies the algebra each operator declares, so `x*0`, `x*1`, `x+0`,
`x&0`, `x|-1`, `x&-1`, `x|0` and `x^0` reduce, and the mux identities cover `iselect` beside its float and bool
siblings. Deferred is everything past that. The non-commutative integer operators declare no identity deliberately,
because the shared algebra drops an identity operand wherever it sits and so would rewrite `0 - x` to `x`; `x-0`,
`x//1`, `x%1`, `x<<0` and `x>>0` therefore survive and need rules of their own. So do the integer counterparts of the
float-specific rewrites -- `x*-1` to `ineg`, multiplication and floor division by a power of two to a shift, and the
value-equality folds `x-x`, `x^x` and `x&x`. That power-of-two rewrite must tap `ishift`'s saturating `prod` reading
rather than the raw `shft` one the `<<` lowering takes, or it changes the answer at the rails. Each of those must
answer for saturation at the format extremes, a question the float rules never faced. Below MIR, a constant shift
count still selects the runtime `ishift` and a runtime negation of the count, where `IntShiftConstOperator` would
cost no module at all.

A kernel that never names an integer can still raise one and reach the LIR refusal: the roundings answer an integer
over a float, and `abs`/`min`/`max`/`np.sign` keep one they are handed. Only the adjacent `IntToFloat(FloatToInt(x))`
reduces before that, so `float(math.floor(x))` and `abs(float(int(x)))` still synthesize whole while
`math.floor(x) > 3`, `float(math.floor(x) + 1)`, `float(abs(int(x)))`, `float(max(int(x), 0))`, an integer phi
merging `floor` with `ceil`, and any integer state slot now build to LIR and stop there. Sinking `IntToFloat`
through a select and through the exactly-representable int arithmetic would let more of them synthesize sooner;
carrying integers through the backends recovers all of them and is the real fix.

The emitted RTL requires an empirical study of the optimal way to bit-extend float results into the register file
when WREG>WFLT. The default treatment of `x <= y` where x is wider than y is to zero-fill the higher bits,
which may potentially require a wider mux and extra wires while we don't care about the value of those bits.
We must explore the possible alternatives, such as marking them explicitly as don't-care,
e.g., `x <= {{X{1'bx}}, result8}` for X extra bits, or a similar solution.
The winner will be chosen based on the synthesis metrics across Diamond, Vivado, and Yosys.

The integer kernels in `tests/_eel_corpus.py` (UART, CRC/LFSR, NCO, PWM, debouncer, priority encoder) are the
acceptance set. Each is oracle-verified against CPython through HIR and then re-run through the MIR interpreter and
built to LIR, so selection and carriage are covered end to end; what the gate still withholds is the cosimulation
against the emitted RTL. `examples/uart.py` carries its counters as floats until then.

## White-box test promotion

The `whitebox` marker is a promotion queue, not a category: it marks a test that reaches past the public API only
because a gate blocks the path. The integer tests carry it because `synthesize` still refuses a built LIR naming an
integer; when the gate lifts, each should become an ordinary kernel and the marker should go.

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
