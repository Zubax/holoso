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
(`b = (a := a + x)`); a `list[...]` return annotation at the module boundary, in favour of `tuple[...]`, though a
returned list literal stays legal; a value that is an array in one arm and a sequence in the other, since aggregates
join a branch only when every arm agrees in kind; a comprehension `if` filter or a comprehension with more than one
`for` clause; and a starred unpack target (`first, *rest = v`).

`while True:` with a data-dependent `break` exhausts the graph expansion budget rather than residualizing: a literal
`True` header is decidable on every trip, so the loop unrolls until the budget stops it, while a runtime `break` opens
an exit lane without ever closing the fall path. The exit condition belongs in the header. This is deliberate --
every cheap detector for the shape misfires on the legitimate counter-spelled loop.

A state attribute's shape and type come from the reset snapshot, so a field annotation contradicting it
(`P: Float64[np.ndarray, "2 2"]` on an instance holding a 3x3) is documentation rather than a checked declaration.
Parameter and return annotations are checked, so the module boundary is judged while the state boundary is not.

## LIR

### The number of fetch stages is still not configurable

The groundwork has been laid to enable configurable `fetch_stages`, but the feature remains incomplete.
This is needed to avoid unnecessary fetch stages (slowing down branching due to pipeline flush)
in small kernels where the microcode memory is compact and can be read in two cycles instead of three.

### Copy-only arms behind an overlapping predecessor

The LIR build threads an arm block holding nothing but phi-arm installs into its predecessor wherever that shortens
the arm's path and lengthens none. An arm whose predecessor overlaps keeps its frame, about four cycles on that path:
pid's first arm, foc's and imu_fusion's first arms, image_agc_streamed's and majority_voter's block 2. Threading them
makes the predecessor drain into the merge, trading latency between the paths (a probe took two cycles off the short
path of `x / y if c else 0.0` and added two to the long one); flux_observer's arm sits behind the empty entry, whose
frame would grow from 2 to 4 PCs.
Gaining here needs overlap across a multi-predecessor edge.

### Interference that keeps a copy-only arm

A copy-only arm, such as an empty `continue` path, keeps its frame when its phi-arm copy cannot coalesce, and the
allocator has no live-range splitting to make it coalesce. In three nested `while` loops whose middle-loop value stays
live across the innermost loop, each of the middle loop's two `continue` arms pays a frame, 4 cycles per middle-loop
trip (80, 652 and 1307 cycles at n = 1, 3 and 4). A variant with an unchanged innermost carry (`b = b * 1.0`) folds to
this kernel and pays the same, although the carry's phi, were it kept, would hold the two ranges apart and save those
cycles (76, 616 and 1243).

### Arm-threading trial cost

Arm threading judges each candidate by rescheduling and reconverging the whole graph, so build time grows with the
candidate count times the graph size: negligible on every example, but a generated 4,896-block kernel with 1,693
threadable arms spends about 13 minutes in it at regalloc effort 0 (0.45 s per candidate). A candidate's effect is
local to its predecessor's frame and the merge phi's interference, so an incremental judgement is possible; it is
deferred until a real kernel needs it.

### Primitive-aware binding

The register allocator binds firings to instances without regard to their primitives, but an instance renders its
operator's block for the primitives bound to it alone, as a CORDIC that only rotates sheds the vectoring datapath and a
divider that only divides sheds the square root. With two or more instances, steering rotations and vectorings, or
divisions and roots, onto separate instances where the schedule allows would specialize more of them and so make them
smaller. The binding objective counts only registers and steering arms today, so this needs an area term for each
instance's set of primitives.

### List-scheduler priority

`schedule_ops` issues ready firings by latency-weighted height to a sink, which ignores instance contention. A
randomized slack-based priority, 300 trials per block under an independent timing model, found shorter blocks the
height order misses: rigid_body_scalar by two cycles, imu_fusion's entry and one arm block by one cycle each. Other
large kernels (both EKFs, cordic_sincos, foc) showed no gain, so the benefit is uneven; a contention-aware priority, or
a few perturbed orders per block keeping the shortest, are the candidates.

## MIR

### Sign tests against zero and a branchless absolute value

A comparison against a constant zero occupies a comparator firing even where it only reads its operand's sign bit.
We need inline primitives and lower comparisons against zero through them:

- `iisneg`, `iisnonneg` evaluating the sign bit only.
- `iispos`, `iisnonpos` evaluating the sign bit and the integer value. This may turn out expensive, subject to test.
- `fisneg`, `fisnonneg`, `fispos`, `fisnonpos` evaluating the sign bit and the exponent of the float.
  Exponent is needed always to handle `-0.0` correctly; it is cheap because always narrow.

The integer absolute value then needs no operator of its own: it lowers to the saturating `0 + x` and `0 - x` and an
inline select between them on the sign test, which removes `holoso_iabss`.

### An early product output on the fused multiply-add

`zkf_fma` could present the rounded product `a*b` on a second output at the multiplier's latency, enabled at
elaboration since its packer costs fabric. A product sharing a block with an `fma(a, b, c)` over the same two factors
would then tap that firing instead of taking a multiplier, as a comparison taps a subtraction's flags.

- A drive carries one timing, which all its primitives share, and the results of an instance leave it in issue order;
  this needs a latency per primitive of a drive, and the overtaking that brings. The schedule is already kept per
  value; the firing record (`PooledScheduledOp.commit_cycle` and its readers) commits every tap together.
- A fused `c - a*b` yields the product negated, which folds into its readers as any sign does; a site taking the
  product's magnitude cannot host the tap.

Running plain multiplications on the same output would let a kernel with `ffma` do without `fmul`,
and enable kernels with both to achieve greater issue parallelism.

### Sign-insensitive operands as an HIR identity

A few float primitives cannot observe their operand's sign (`isfinite`, `bool(x)`, `ilog2`) and declare it
(`unconditioned_operands`), so that MIR normalizes the operand's sign away before interning and LIR construction erases
a pooled constant's sign by the same declaration. An HIR rewrite stripping a negation or an absolute value under these
operators (`isfinite(-x)` to `isfinite(x)`) would carry the same knowledge where the other algebraic identities live and
delete the declaration's plumbing through MIR and LIR.

### A promotion re-run keeps the discarded pass's ownership events

A data-dependent loop whose int carry meets a float re-runs its body with the carry promoted, rewinding outputs,
elisions and the budget but not allocation states, which are monotone facts. A store the first pass admitted can then
be refused (`a[0] += x` beside `total += np.sum(a)` with `total = 0`, while `total = 0.0` builds; a plain read of the
stored element, an alias, or a helper argument after the store does the same). Stricter, never wrong; rewinding the
states would need the pass to record every allocation it moves.

## Operators

### IDivs fast mode

`IDivs` needs `fast` where truth uses a radix 8 divider, which needs to be implemented first.

### Per-instance stage configurations

An operator has one set of stage knobs for all its instances, while an instance specialized to fewer primitives can have
fewer product register positions than the operator's own block: CORDIC's vectoring-only product at e6m18 on 18-bit
multipliers has 3 against 4, and a build without the magnitude at e8m36 on 18-bit multipliers has 4 against 5. A count
the kernel's other instances take can therefore be refused for a specialized one, and that refusal depends on how the
allocator binds the primitives. Ideally an operator could build its instances with different stages when they run
different primitives. Two candidates, neither adopted: several configurations per operator where it has several
instances, or ZKF's automatic staging (Zubax/zkf#18: the user names the chip, speed grade and target frequency, and ZKF
ships predefined per-operation staging). Not a priority; the solution is unclear.

## Backend

### An error on the first step

`err_pc` reports the step of the last error, and 0 means no error. Only a combinational firing can commit on step 0, and
an operator with an error output is asserted to have a register stage under every primitive, which every such operator
has today. Should one ever be combinational, either `err_pc` takes all ones for no error, which costs an extra bit
wherever the ROM depth is a power of two, or a separate error flag accompanies it.

### Retire the inline support header

`holoso_support_inline.vh` is spliced into every kernel module. Its helpers are all very simple expressions, and each
inline primitive could render its own through `verilog_expr()`, as some already do, retiring the header.

- State seeding scans every readable method of a component's class, reached or not, so an unreached setter makes its
  attribute a runtime value and can refuse a kernel that needs it as a constant; a write in a function a decorator
  forwards to is not scanned at all and is refused as unseeded. Seeding from reached writes alone, re-running when a
  run reaches a new one, would remove both and would also free the held audit items that only fail through seeding.
