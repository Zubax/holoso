# Holoso design

Holoso lowers a small subset of Python (numerical control/DSP kernels) into vendor-neutral, synthesizable Verilog.
See `README.md` for scope and `PRIOR_ART.md` for why existing tools don't fit.

THIS IS NOT A SPECIFICATION. It records the architecture we are building toward, capturing design intent rather than
implementation detail -- the code is the low-level reference. Many of the trade-offs here won't survive contact with
reality, and we discard and redesign freely. Do not pollute this document with exact code references or
verification-suite mechanics. Read the representative examples under `examples/` to understand the motivation.

## Direction

Build our own compiler. The differentiating work is the front/mid-end: partial evaluation of Python, shape inference,
and operator scheduling for a resource-shared FSM. No external HLS gives us this for Python, and most would force a
pipeline-oriented optimizer we don't want. We delegate only to lightweight Python tools where it clearly pays: SymPy
(fold/CSE/simplify), Cocotb for testbenches, ILP solvers and function minimization (SciPy) for scheduling/regalloc.
Other lightweight dependencies may be introduced as needed.

The target is a specialized program, not a pipeline. We synthesize a sequential FSM (a zero-instruction-set computer,
ZISC) that time-multiplexes a few shared operators over a register file. We do not pursue a constant or near-1
initiation interval like a streaming pipeline: the II is whatever the scheduled program costs -- for a fixed control
path an exact, statically known cycle count from the per-operator latency model, varying across programs and branch
paths. This is a compiler problem more than a circuit-design one.

We encourage departure from IEEE 754 where it makes sense for numerical control/DSP (e.g., drop NaN/subnormals).

Compilation is deterministic and reproducible for fixed inputs and dependency versions: identical input produces
byte-identical output (except diagnostics and reports, which may carry timestamps and the like), achieved by sorted
iteration over name-keyed merge points and a fixed seed for every stochastic optimization pass.

## Pipeline

```mermaid
flowchart LR
    Python[Python] -->|front-end| HIR[HIR]
    HIR -->|optimize| HIRO["HIR (optimized)"]
    HIRO -->|lower| MIR[MIR]
    MIR -->|schedule / bind / regalloc| LIR[LIR]

    LIR -->|backend| Verilog[Verilog]
    LIR -->|backend| Testbench[Testbench]
    LIR -->|backend| Report[Report]
    LIR -->|backend| Model[Model]
```

HIR -- "what to compute": SSA dataflow inside a control-flow graph with real branches. Target-independent and semantic;
it does not know how an operation is implemented.

MIR -- "which hardware to use": selected hardware operators over typed nodes, still unscheduled. This is the first stage
allowed to inspect hardware operator configs or operand numerical limits.

LIR -- "the microprogram": the scheduled, bound, register-allocated op stream for the synthesized machine, over typed
storage (a shared wide data register file and a separate 1-bit boolean bank). LIR owns scheduling, binding, and register
allocation, and is RTL-controller-agnostic -- the seam where a second controller backend can be added later.

Backends -- Verilog, testbench, HTML report, numerical model, and possibly other HDLs later. The numerical-model backend
(see Backend) gives bit-exact, cycle-exact emulation of the emitted HDL, so the synthesis logic can be stabilized down
to LIR before the slow HDL-emission/simulation iteration begins.

## Glossary

- Issue / commit / landing -- the three cycles of a result. An op issues when its operands are sampled, commits its
  result `issue + latency` later, and that result lands (first becomes readable) a further fixed latency later.
  A consumer reads at the landing, not the commit.

- Pooled operator -- a latency-bearing arithmetic operator (fadd, fdiv, etc.) time-multiplexed across all its uses.

- Inline operator -- a combinational, simple zero-latency op (boolean logic, select, type cast, etc.)
  emitted as a single HDL expression rather than a pooled operator instance.

- Spill -- NOT a register spill to memory. A value whose landing extends past its block's terminator
  into a single-predecessor successor: the cross-block software-pipelining overlap.

- Install -- a copy that writes a value into a persistent-state slot or a merged-phi register at a block boundary,
  used when coalescing could not make the write free.

- Coalesce -- merging a phi and its identity-arm predecessors onto one register (union-find) so the install becomes a
  no-op.

- Slot (state slot) -- a register holding persistent state across transactions (e.g. `self.x`), committed in place.

- Drain -- the cycles a block's terminator waits past its last commit for in-frame writebacks to land.

- Fetch lag -- the depth by which the control-fetch pipeline leads the executing step;
  every operand is sampled at `issue + fetch lag`.

- Read-first -- within a cycle a register read returns the OLD value, before any same-cycle write; this is the origin of
  the +1 dependency edge between producer and consumer. Aka "read-before-write", "write-after-read" (WAR).

- Dwell -- the PC stalling at one of its hold points: pc 0 (accept, awaiting `in_valid`) or LASTPC (present,
  awaiting the result being taken before restarting).

- Makespan / II -- a block's schedule length in cycles; the initiation interval (II) is the whole executed path's exact
  cycle count.

- ZISC -- zero-instruction-set computer: the VLIW microcode-driven sequential FSM that Holoso synthesizes.

## Python API

`synthesize` is the main entry point; it returns an in-memory result and touches the filesystem only on an explicit
write.

Passing the live object (not a file) is more ergonomic and strictly more capable: it carries the runtime environment the
binding-time front-end needs -- `__globals__`, closure cells, default args, and the result of running `__init__` --
which is what evaluates compile-time tables and follows/inlines imported callables. The object is the compile root; the
boundary ("what to ignore") falls out of reachability + binding-time analysis, not manual enumeration.

A plain function synthesizes to a stateless module. A stateful module is requested by passing a bound method of a
constructed instance, e.g. `synthesize(filt.update, ops)`: the bound instance's attribute snapshot seeds the reset
state, and its method is the analyzed body. The constructor runs in plain Python, so its arguments are ordinary values
frozen into the build -- no separate parameter marshalling.

The root package re-exports only the supported public API. A future second mode -- several methods sharing one state,
selected by passing the class plus a method list and a runtime selector port -- is deferred: it needs a backend selector
and per-method schedules over shared state.

## Types

Runtime values are only:

- `float` -- one ZKF format, `WEXP`/`WMAN` fixed per build. FPGA-friendly formats usually set WMAN to a multiple of the
  native DSP tile width (commonly 18); e.g. WEXP=8 WMAN=36 for precision, WEXP=6 WMAN=18 for simpler targets.
- `bool` -- 1 bit.
- A separate fixed-width `int` type may appear eventually. The LIR wide data register file is already neutral storage,
  so future non-boolean scalars can share the bank when their physical width matches the build, at minimal waste.

Compile-time ints/shapes/structure are resolved in the front-end and never reach HIR.

## Operators

HIR carries pure semantic operations from a HIR-local operator hierarchy; an operation is one operator applied to
operand value IDs. Concrete hardware operators are frozen dataclasses whose fields are Holoso-exposed parameters;
float ones delegate to the external ZKF library.
Each hardware operator owns its signature and a compact HDL-safe identity stem, so the fully specified operator instance
is itself the resource-sharing key and equal operators time-share one module. Per-node-parameterized operators are
factories that instantiate a concrete operator (e.g. multiply-by-constant-power-of-two differs by exponent).

Operators are chosen by a single `OpConfig`, constructed explicitly by the user and passed into `synthesize`; there is
no implicit default. Its float format is verified consistent across the configured operators and drives HIR-to-MIR
lowering; thereafter the format is derived from selected MIR. Latency-tuning knobs are named after the HDL parameters.
Some operators are optional (default `None`).

An operator may declare per-firing small microcode-driven immediate inputs.

Each operator declares a per-instance initiation interval; most operators have II=1, i.e., fully pipelined.

## Front-end

A rewrite is in progress under `holoso/_frontend/_fir/` (shadow-only until cutover; nothing in production consumes
it). Landed so far, terse facts: name classification comes from the live code object (CPython's own; `del`-locality
and PEP 709 targets correct by construction) with comprehension-only targets carved out via the AST; non-locals
resolve closure-cell before global before builtin, values read at resolve time. Static values form a closed
whitelisted domain with provenance as identity: exact Python ints (MetaInt) are distinct from numpy 64-bit scalars
(NpInt/NpFloat, numpy's own mixed-operand semantics; narrower numpy widths are not static); arrays admit as read-only
snapshots; sequences keep list/tuple flavor; dataclasses admit as records only when reconstructible from fields;
containers bound depth and refuse cycles. Fixed-point equality is tagged and bitwise (True is not 1, signed zero
distinct, NaN stable). Static execution runs real Python/numpy on the domain's own objects; zero-division/invalid
defer to runtime, float overflow folds to infinity per the charter, numpy wraparound folds faithfully, NaN never
folds, and integer powers fold only under a result-width bound.

The FIR itself is a private non-SSA CFG over mutable Places (Local/StateLeaf/ReturnPlace), built syntax-directed
with no analysis: ANF write-once temporaries in exact Python evaluation order; ordered stores; one canonical exit
per FunctionUnit (return = store + jump); eager and/or/chained-compares via selects (pinned semantics); ifexp as
real branches; for-loops and comprehensions as StaticFor templates (comprehension targets in their own frame, the
outermost iterable evaluated in the enclosing scope); lazy PyCall; raise/assert/missing-name as Fail terminators
(dead branches stay dead); del as UnbindPlace; origin stacks on every node. Walrus is supported except in
short-circuit positions; nested def/class/lambda, imports, subscript stores, and variadic parameters are located
rejections. A structural verifier and a deterministic printer close the stage.

The production front-end below remains authoritative until the staged cutover.

Abstract interpretation over the Python AST/CFG with a binding-time lattice (static vs. dynamic). Static values (shapes,
`__init__`-derived constants, compile-time tables) are evaluated concretely -- real Python/NumPy runs at synthesis time.
Dynamic values (input ports, persistent state) become SSA handles that accumulate HIR. A `for` over a static `range` or
an aggregate is unrolled (unless the count exceeds the unroll threshold); a list comprehension unrolls the same way
(its generator count is likewise bounded, so the unroller cannot recurse without limit),
yielding a Python list, with every `if` filter folded statically and its targets confined to the comprehension -- which,
being its own scope in Python, also has them shadow a same-named module constant while bound; a `while` lowers to a real
back-edge loop; an `if` on a static test takes one branch, on a dynamic test emits a real branch. Static evaluation
(used for branch/loop reachability and compile-time indices) resolves only the operands that both the reachability
scan and lowering reconstruct identically: numeric literals, module-level numeric/array constants,
read-only instance attributes, and loop counters -- including numpy navigation (indexing, slicing, transpose,
`.flatten()`) of a module-level array constant or read-only array attribute.

Shape queries. An aggregate's shape is a static value, so `len`, `.ndim`, and `.shape[k]` evaluate to compile-time
integers usable wherever one is expected, there being no runtime integer type to carry one into a value position.
`len` follows Python and counts the items of any aggregate; `.ndim` and `.shape` are numpy-only and rejected on
a list/tuple, whatever expression the receiver is spelled as.
A scalar is rank zero, as a numpy scalar is and a bare Python float is not, which is how the matrix product
rejects a scalar operand by asking its rank; consistently, an empty-tuple key `x[()]` yields the scalar itself, as it
does for a numpy scalar, while `x[0]` or a slice is rejected. The resolver sees a closed set of receivers --
a bound name, a static subscript of one, a state attribute, a module constant,
a transpose of those -- so a shape query on anything else is rejected rather than guessed at.

Rejecting from within the subset. Lowering descends only the arms the static fold selects, so a `raise` it reaches sits
on a path taken unconditionally within its own function: it becomes a synthesis error carrying the exception's own
message (a literal, or an f-string interpolating compile-time integers), located at the raising line. Guarded by a
data-dependent test it would have to become a runtime exception, which the hardware cannot signal, so it is rejected.
Branch depth restarts per inlined function -- whether a `raise` is data-dependent is a property of the function that
writes it, not of a call site that happens to sit in a branch arm. This is what lets a library stub validate its own
operand shapes while remaining ordinary, executable Python that raises the very same error.

Persistent state. A synthesized method's `self` is not a port: each instance attribute the method writes becomes a
persistent register (a loop-carried value, the back-edge of the initiation loop), and each attribute it only reads is a
frozen constant folded from the `__init__` snapshot. Within the method `self.attr` is an ordinary SSA variable, so reads
and writes interleave freely. Public attributes additionally drive a `state_<attr>` output port, so a method need not
return anything (and a returned value that is by dataflow a public attribute is deduped onto that state port);
underscore-prefixed attributes stay internal. A vector-valued attribute (list, tuple, or 1-D numpy array) decomposes
into one scalar register per element (`attr_0`, `attr_1`, ...), a matrix-valued one (2-D numpy array) row-major into
`attr_0_0`, `attr_0_1`, ...; a scalar keeps its bare name. Reassigning `self` itself is rejected: attributes resolve
against the fixed original instance, so a rebinding would silently miscompile.

Whether an attribute is state follows reachability, and the scan that decides it runs before the body is lowered, so
it cannot always fold what lowering folds -- a branch on the shape of a local, or a `for` over an aggregate whose trip
count it does not know. Where it cannot see a trip count it must fold less, not guess: it walks such a body once with
the loop target and every counter the body rebinds demoted, and restores that same context afterwards, since an empty
aggregate runs the body zero times and an unrolled one runs it many. The contract is therefore one-sided: the scan
yields a conservative superset, trimmed back to the truth once lowering has run, since an attribute lowering never
touched keeps its reset value for good and is no more state than a read-only one. An attribute lowering only read is
the other half of that trim: its reads are already in the graph, so it stays a slot whose live-out is its live-in -- a
register that holds, and, if the attribute is public, a port. Dead code can therefore cost a register and a port an
exact scan would have folded away, and can make an otherwise read-only attribute look written, costing it the constant
folding a frozen attribute enjoys -- but never a wrong value. The opposite direction is forbidden and asserted
against, a write lowering reaches but the scan missed having nowhere to land. The duality is a wart, not a design: see
DEFERRED.

State lives in the float registers, so an integer reset the format cannot represent exactly is rejected rather than
silently rounded: it would read back as a different number than the source compares against, and there is no integer
type to fall back on yet -- see DEFERRED.

Matrices/vectors are statically shaped and unrolled to scalar operations; arrays never exist as hardware aggregates,
only as compile-time bookkeeping over scalar registers. That bookkeeping is a front-end value -- either a scalar wire or
an ordered aggregate -- supporting list/tuple literals and comprehensions, numpy-style indexing and constant slicing
(`m[i, j]`, `m[:, j]`, chained `m[i][j]`), iteration, unpacking, transpose (`.T`), and the array factories
`np.array`/`asarray`/`asanyarray`; only scalar leaves reach HIR, so the supported source is executable numpy.
Module-level numeric and ndarray globals fold into constants.

List/tuple vs. array. A front-end aggregate carries its Python flavor. The guiding principle is to follow Python
semantics where sensible and otherwise reject a construct rather than silently reinterpret it; in particular a Python
list/tuple is never given numpy semantics. A Python list/tuple (a list/tuple literal, the `list()`/`tuple()` builtins,
or a starred-unpack remainder) has sequence semantics: indexing, constant slicing, unpacking, splatting, building, and
returning all work on it. A numpy array -- a shaped parameter, an ndarray module constant or state attribute,
`np.array`/`asarray`/`asanyarray(...)`, or the result of any array operation -- additionally carries numpy semantics.
The numpy-only operations (elementwise arithmetic, the matrix product, transpose, `.flatten()`, and multi-axis
indexing) apply to arrays and are rejected on a list/tuple: on a Python sequence they mean something else or nothing
(list `+`/`*` are concatenation/repetition, list `-`/`@`/`.T` are undefined), none of which is the array operation
intended.

Array arithmetic. The elementwise arithmetic operators `+ - * /` apply leafwise to same-shape arrays, and a scalar
operand broadcasts over the other side's leaves; this is deliberately narrower than numpy broadcasting -- a mixed-rank
pair (vector + matrix) is rejected rather than silently aligned along a different axis than numpy would pick, and `**`
stays scalar-only (base two with a runtime exponent lowers to exp2; otherwise the exponent must be a compile-time
integer). Augmented assignment to any aggregate (`+=`, `@=`) is rejected: an array would be mutated in place by
numpy while the front-end rebinds (diverging for an alias), and a list `+=` is concatenation, which is unsupported.

Linear algebra is ordinary library code. The matrix product `@` and the transpose `.T` lower by resolving `np.matmul`
and `np.transpose` in the same registry a spelled call goes through, so an operator and its call are necessarily
one implementation. The stubs follow numpy's shape rules for 1-D and 2-D operands and expand into scalar
multiply/add chains. Each dot product is a left fold, enabling FMA contraction in the MIR.
Also provided are `np.dot` (the matrix product, since the two agree on this domain), `np.trace`,
and `np.outer`. Because the stubs expand through the same comprehensions any kernel uses, a matrix dimension is
bounded by the unroll threshold exactly as a loop trip count is -- an operand axis wider than the threshold is
rejected rather than unrolled into thousands of scalar operations.

Inlining. A pure function reachable through `__globals__` is inlined -- its body lowered in a fresh scope, its return
consumed as an aggregate -- so kernels compose. A method call on the synthesized instance (`self.helper(...)`) is
inlined with the instance context kept, so the callee's own `self.<attr>` reads resolve (the method is found through the
class MRO; `@staticmethod` and `@property` getters are supported). A called method may read `self` but not write it --
only the entry method owns the state-slot analysis. Name resolution follows Python.

Math library. A call dispatches through a registry on the object identity its callee resolves to, not the spelled name,
so an alias resolves and a shadow does not. The registry maps that object to its lowering: an intrinsic stub (1:1 onto
an HIR float operator) or a composite stub (built from the intrinsics and inlined). A composite is ordinary Python in
the supported subset -- shape queries, comprehensions, and `raise` let the linear-algebra stubs be shape-polymorphic and
self-validating -- so each stub is its own numerical reference, and a rejection inside one is re-attributed to the
user's call site under the spelling they wrote.

Parameters. Positional and keyword-only parameters become input ports and require an explicit annotation:
`float`-annotated scalars are floating-point ports, `bool`-annotated ones are 1-bit boolean ports, and a jaxtyping
array annotation with fixed 1-D/2-D dimensions and a floating dtype (e.g. `Float64[np.ndarray, "3 3"]`) decomposes
row-major into one float port per element (a vector's are `name_0, name_1, ...`, a matrix's `name_0_0, name_0_1, ...`).
The jaxtyping types are detected structurally, so jaxtyping stays a dependency of the user's code only.
An aggregate attribute's shape is read from its reset value, optionally validated against an explicit jaxtyping
annotation; interior shapes are inferred.

Return type. The return annotation is likewise mandatory and validated against the inferred output leaves: `float`,
`bool`, a fixed-shape jaxtyping array, an arbitrarily nested `tuple[...]`/`tuple[X, ...]`/`list[X]` of them, or
`None` for a method that returns nothing. A missing annotation or any shape, arity, or scalar-type divergence rejects
the kernel.

## HIR

HIR is a real CFG of basic blocks -- entry first, a single `Ret` exit -- carrying an SSA value DAG. Values are input
ports (one per parameter), constants, state reads (persistent state at block entry), phis (SSA merges),
and pure semantic operations; terminators are `jump`, `branch`, and `ret` (which commits state live-outs and
output ports). The pure operations cover float arithmetic, scalar float classification, relational comparisons and
boolean logic yielding `bool`, float<->bool casts, and `select` (a data mux produced by if-conversion, distinct from
control flow).

`bool` is implemented alongside `float` throughout (constants, input ports, state reads, phis), and a state slot's reset
is a typed constant, so a boolean state register carries a boolean snapshot. Node names stay explicit (`FloatConst`,
`FloatAdd`, ...) so int nodes can be added later without overloading float semantics. Negation and absolute value are
ordinary semantic float operations here, not hardware details until selection.

Interning is block-scoped for operations and global for entry-dominating pure values (constants, state reads); inputs
are never interned (each parameter is a distinct ordered port). CSE'ing an operation only within its own block is the
point: an identical expression in two sibling `if` arms must stay two distinct values, because a globally interned DAG
would illegally share a value across non-dominating arms. Merges emit one phi per diverging scalar leaf.

Operators split structurally into POOLED -- a physical streaming module the scheduler contends for --
and INLINE -- a pure expression folded into a register write, like boolean logic.
This split is load-bearing for scheduling and emission.
A comparison taps one of the comparator's order flags with an optional inversion,
so one physical comparator serves every relation (the ZKF ordering is total: lt/gt/eq directly, le/ge/ne by inversion)
and several relations over one operand pair share a firing.
The sorter is the wide-output analogue: it emits the smaller and larger operand on two ports,
so a `min` and a `max` over one pair share a firing.
Boolean `and`/`or` are inline gates that always evaluate both operands (they are pure booleans);
a chained comparison `a < b < c` desugars to `band(a<b,b<c)`.
`not` never materializes hardware: NOT chains fold into the consumer's sideband (an operand/output/state/phi-arm
inversion, or a branch-target swap), so one comparator tap and one register serve both polarities. The casts
(`bool(x)` = `x != 0.0`, `float(cond)` = `1.0`/`0.0`) are inline writebacks that confine the ZKF bit layout to a single
shared header, cross-checked against the bit-exact model at build time.

Branch vs. select is the core control-flow decision:

- A real `if`/`else` lowers to a `branch` terminator + a `phi` at the merge. Only one side executes; the merge is
  resolved at register allocation -- no runtime mux, the untaken arm never computed, no spurious error recorded.
  Branches are the default.
- `select` (a mux, both inputs live) implements data multiplexing. The if-conversion peephole collapses a small, pure,
  cheap branch diamond into per-phi muxes, making the region straight-line (so it pipelines and reuses registers).
  Because both arms then execute, conversion is gated: every arm operation must be SPECULATABLE (division is not -- a
  speculated div-by-zero would assert the error flag on an untaken path) and each arm must fit a configurable op budget.
  A boolean-phi merge converts to a first-class `bool_select` (the 1-bit dual), reusing the float select's
  scheduling and emission paths. Running both arms can RAISE the static lower-bound II while LOWERING the realized
  per-transaction latency, which is the goal -- the regression guard is the realized-latency test, not the static lower
  bound. Arm sign chains fold into the mux's conditioners, so `x if c else -x` costs one comparison and one mux.

A conditional expression `x if c else y` lowers exactly like an `if` lifted into expression position (a compile-time
test selects one arm with no branch). A walrus `(name := expr)` is supported only where evaluated unconditionally and
rejected where the binding could be short-circuited (inside `and`/`or`, a chained comparison, a conditional-expression
arm, or a `while` condition).

An `assert` statement is accepted and ignored wholesale: its test is never lowered, mirroring Python under `-O`, so an
assertion has no hardware effect (each reached assert is logged). Any effect the test would have had when executed is
dropped along with it; as under `-O`, an assert must be side-effect-free.

A nested `if` with no `else` on either level folds to a single combined-`and` branch (`if A: if B: S` becomes
`if A and B: S`), emitting one branch instead of two. This is exact because a boolean test here is a pure combinational
value; the fold is disabled the moment the outer `if` carries an `else` (then the `and` would mis-route the `else`).

Loops. A `for` over a static trip count fully unrolls below the unroll threshold: the counter is a compile-time integer,
so each trip lowers the body once with the counter bound. Reassigning the counter to a runtime value demotes it,
matching Python. Rotation-mode CORDIC sin/cos illustrates this -- its per-iteration `2**-i` shift forces unrolling and
its sign test is a per-iteration branch.

A `while` lowers to a real back-edge loop: preheader -> header -> body -> back-edge to the header. The header carries a
phi for each scalar or persistent attribute the body reassigns. Blocks lay out with each body below its header and the
single `Ret` last, so a back-edge is just a jump to a lower address the sequencer already handles; the loop header is
multi-predecessor, so the body fully drains before jumping back and no overlap crosses the back-edge. The static II
weights the back-edge as not-taken (a true lower bound); the numerical model is the authority on the realized count. A
Newton-Raphson reciprocal iterated to a tolerance illustrates this, on its convergent domain.

### HIR optimization

Holoso is intentionally very liberal when it comes to expression optimization.
Bit-exactness, numerical determinism, or strict IEEE 754 compliance are anti-goals.

HIR optimization is hardware-agnostic and ordered so each pass sees final costs: const-fold -> strength reduction
(trivial fast-math identities, powers of two to shifts, `x/c` to `x*(1/c)`) -> diamond if-conversion (after folding, so
arm costs are final; before DCE, which then sweeps a converted diamond's now-dead condition cone) -> a second
const-fold/strength-reduction pass for the muxes created by if-conversion -> merge threading -> DCE. Constant folding
is typed, so bool/int constants need no float-specific rebuilding.

Merge threading eliminates an empty pass-through merge block -- the shape a non-convertible diamond leaves when its
merge feeds a following control structure -- by retargeting each predecessor's jump onto the merge's successor and
composing the phi arms. A merge phi reached any other way (e.g. a loop-invariant value the header carries on its
back-edge arm) keeps its real branch -- deferred (see LIR DEFERRED).

FP math is non-associative, so some of these optimizations may produce non-bit-exact results -- accepted, analogous to
fast-math in C/C++ compilers. Division identities may also rewrite zero/infinity special cases and drop sidebands when
they remove an error-bearing op; sign/identity folds may preserve or expose a zero sign through raw sign conditioning.
NaN constants are rejected because ZKF has no NaN, including float64 constant folds of ZKF-defined infinity cases;
infinities are ordinary float values. The transcendental folds likewise take the ideal (infinite-precision) result, so
a folded constant can differ from the datapath's own value.

### DEFERRED

Variable-trip `for` loops: a `for` above the unroll threshold is rejected, not lowered to a counted back-edge loop (that
needs a runtime integer counter).

Early return from a loop body.

Static folding of a constant that reaches a static position (a branch/loop condition or a compile-time index) only
through a local alias or an inline-built value, e.g. `g = CONST; if g[0] > 0:` or `if np.asarray([...])[0] > 0:`. Such
a value still lowers correctly but is treated as dynamic, because folding it would require the reachability scan to
track arbitrary local bindings it does not build. Only module-level constants, read-only instance attributes, and loop
counters fold in static positions.

The scan/lowering duality itself. The reachability scan runs before the body it describes, so it cannot see the local
bindings lowering will have, and shape queries on locals therefore fold at lowering but not in the scan. Today the
mismatch is contained by making the contract one-sided -- the scan over-approximates, the truth is trimmed out
afterwards, and the opposite direction is asserted against -- so the cost is a stray register or port, an attribute
that stops folding as a constant, or a conservative rejection, never a wrong value. It remains a duality, and every
new static-evaluation source has to be checked against it by hand. The way out is to give the scan a real
binding-time environment over locals (shapes at least, values where cheap) so that scan and lowering fold identically
and neither the trim nor the assertion is needed; short of that, folding could be confined to bindings both phases
reconstruct. Either is a redesign of the scan, not a patch to it.

Integer operands: typed int operands/constants/operators sharing the wide register bank when their width matches the
build. Until then every integer enters the float datapath, so one that binary64 cannot represent exactly is rejected
rather than silently rounded -- otherwise a stored integer reads back as a different number than the source compares
against. The check is against binary64, not against the build's own float format: the front end does not know that
format, and a narrow one rounds every constant, integer or not. An integer attribute in a narrow build can therefore
still lose its exact value, which typed integers, not a wider check here, are what fix.

## MIR

HIR-to-MIR lowering selects concrete hardware. The float lowerer maps each semantic float operator to its configured
hardware operator and collapses semantic negation/absolute-value chains into MIR sign-control sidebands on operands,
results, or output wires. Multiply-by-power-of-two selects the constant-shift operator when the float format supports
that exponent; an out-of-range exponent is rejected, since the equivalent constant would overflow or underflow the
format anyway. The four rounding operators map to one shared `fround` distinguished by its `round_mode` immediate.

Some operator lowerings are context-sensitive, where the final lowering depends on the nearby operations.
Examples include computing min/max in a single pooled comparison operator transaction,
sin/cos being simultaneously computed by the sincos hardware operator, etc.
Another example is the FMA contraction, where a single-use `a*b+c` is lowered into one fused multiply-add.
Boolean infinity predicates adjacent to a zero sign-test, such as `isinf(x) and x > 0`, lower directly to the
corresponding directional infinity classifier.
The matching is done at the MIR level because this is the first layer that is aware of the hardware semantics.

Some semantic operators are lowered into a combination of hardware operators depending on the available hardware
operators and the context. This is only done for operators for which a possibility of specialized hardware
handling exists (e.g., hypotenuse calculation can be done using the fatan2 operator, but it only makes sense
if arctan is also computed simultaneously). Such composite lowerings may use inline muxes to sanitize operands fed into
their internal primitives so that semantically valid edge cases do not raise avoidable primitive-side errors, while
invalid source inputs still reach the error-bearing primitive.

The MIR builder has no global scalar type, so mixed-type expressions share one value namespace, but carries the
configured float format explicitly so float-less modules still elaborate with a known scalar width. The CFG is carried
through MIR as typed per-resource-family views (a float view and a boolean view sharing the block skeleton), then
scheduled and register-allocated per block.

## LIR

LIR is the scheduled, bound, register-allocated microprogram. Its resources are the bound operator instances (each a
fully-specified pooled hardware operator), the float format, the storage banks (a wide data register file and a separate
1-bit boolean bank), a pool of nonnegative float constants (the sign rides the consumer's sideband), and the typed input
loads and output wires. Each scheduled firing -- pooled or inline -- carries its operands and conditioners, its register
writes, and an issue cycle; the makespan is the last commit cycle, and the observable input-to-output latency follows
from it and the fetch timing of the datapath. LIR exposes a minimal API plus shared analysis helpers (per-cycle
grouping, liveness, read/writer sets) so backends do not each re-derive them.

Storage is a sparse register file synthesized per kernel: each operand's read mux spans only the sources it reads,
each register's write mux only the sources it takes (see Backend for the encoding). A CPU-conventional full-reach
crossbar was tried first and abandoned -- its read/write port multiplexors imposed untenable timing.

### Scheduling

The LIR scheduler runs software-pipelined list scheduling over each block of the selected MIR. Operator latencies are
fully static and data-independent (most throughput-1, zero-bubble), so the whole schedule is computed at compile time:
each op gets an issue cycle and a bound instance, and the backend just replays it with a cycle counter -- no scoreboard.
This makes the latency model load-bearing rather than advisory: the backend commits each result at `issue + latency`
without watching `out_valid`, the generated RTL passes that latency into each operator wrapper's mandatory `LATENCY`
parameter, and any Python/RTL drift fails at elaboration. An inaccurate latency is a correctness bug, not a bad
estimate.

Each op issues on the earliest cycle its operands are ready and a free instance exists, with no barrier, so a
cross-domain chain (`float(x>0)*k`) schedules tightly. The commit-to-issue spacing a dependence requires is not one
constant but is derived pairwise from a single cycle-accurate timing model built from a few named primitives (a global
fetch lag and a read-first edge), never per-case constants. Both register banks sample an operand at
issue + fetch lag through a combinational read mux. Every result -- pooled or inline, on
either bank -- writes the register array combinationally and becomes readable a fixed fetch-lag-plus-read-first edge
after its commit. (Earlier designs latched pooled wide results in a writeback latch and presented the wide bank's read
address a step early in a read latch; both were dropped as inconsistent -- the writeback latch caught only float
operators while installs, control flow, and inline writes went direct, and it needlessly delayed short installs --
so every result now writes direct and both banks read alike.) Because the two banks and the pooled/inline classes
are uniform instances of that one model rather than hand-coded cases,
boolean-logic and cast chains schedule back-to-back, which shortens logic-dense kernels. Block-resident operands
(inputs, state reads, phis) are available from the block's first control word, so an op can issue from there.

Each cycle, the ready ops (every operand committed) are issued in critical-path order
onto free instances. Instances are pooled by the fully specified hardware operator itself (equal-by-value): all
`fadd`/`fmul`/`fdiv` of a config share instances; constant-shift operators of different exponent are distinct modules. A
configurable per-class budget (default 1) caps instances per distinct operator value, serializing co-issues beyond the
budget.

### Register allocation

Register allocation is reach-aware over the whole CFG: whether two values may share a register is decided on a
hardware-frame interference graph from per-block liveness, two values interfering when their residences overlap under
the read-first rule (the older value's last read must precede the newer's landing). Path-awareness is free: the two arms
of an `if` are live in no common block, so their temporaries reuse the same registers, which keeps a heavily-branched
kernel (e.g. a 12-iteration CORDIC) to a handful of wide registers.

The primary objective is to minimize per-port read-set and per-register writer-set fan-in -- the FPGA steering cost that
matters, not flip-flop count. Register count is a bounded secondary objective. There is no spill. The coloring is a
port-affinity greedy seed refined by simulated annealing over the same objective, and colors both banks.

Phi-arm coalescing eliminates most install copies: before coloring, each phi and its register-backed, identity-arm
predecessors merge by union-find whenever the two sides do not interfere, so the arm value flows straight into the
merged register with no copy (a diamond's mutually-exclusive arms always coalesce away). The pass is pure post-schedule
register reassignment -- it changes only register and copy counts, never behavior; the surviving copy set feeds the
per-block drain/push classification, so PC layout may shift with it.

Commutative port assignment, after allocation, orients each commutative firing's two operands across its read ports to
minimize total read-set size (a register read in both positions would otherwise sit in both ports' muxes). Commutation
is a pure relabelling -- an input swap with an induced output-port permutation -- so the value stays bit-identical at no
hardware or latency cost. Minimizing read-set size over orientations is graph bipartisation, which a local search
cannot reliably optimize, so it is solved exactly per instance as a small MILP, with the local search as a fallback.

Persistent state slots. Both banks commit state in place: a live-out is written directly into its slot register,
read-first, so a same-frame self-update (`self.x = self.x | y`, an accumulator) reads the old value and writes the new
one with no copy. A conditional or loop update whose "unchanged" arm is the slot live-in coalesces onto the slot
register through the same phi-arm union-find. When it cannot commit in place (a genuine overlap, a folded sign, or a
chained copy `self.a = self.b`) the live-out keeps its own register and a microcode-driven copy installs it as early as
the old live-in is read. Two slots that always hold the same value collapse onto one register (a public attribute and
its private alias), dropping the duplicate register and its install copy. State
registers are the one datapath exception that reset reaches (each loaded with its snapshot); pure datapath state stays
out of the reset cone.

### Control flow

`branch` is the real control transfer: the PC jumps, untaken ops never run, and the II is whatever the executed path
costs (each path's count exact). Blocks lay out in reverse-postorder with the canonical `Ret` forced last as the
out_valid boundary, so a back-edge is a jump to a lower address; each block's terminator redirects the fetch PC via a
small `case(pc)` that, for a branch, reads the condition's 1-bit register.

A block's terminator offset is the latest cycle a value still lands in its frame -- it must cover every landing the
block does not forward to a successor. A block whose boundary values are all already resident in predecessors pays none
(the drain-only `Ret`s of loop/diamond kernels, whose body produces every output the `Ret` reads combinationally).
A tail install lands at the block's work makespan, read-first at the boundary, and costs an extra terminator cycle only
when its source is an operator result committing at the block's makespan (its own last work, or conservatively any
operator result from outside the block); a resident source (a constant, input, state read, or phi result) or an
earlier-committing computed source lands at the makespan and recovers cycles in every downstream block. A source register
that a sibling install writes is always a phi's, hence resident, hence fired at the makespan -- strictly before any
sibling's write lands (same-step installs read all sources before any write lands; a pushed sibling lands later
still) -- the invariant that keeps cross-referencing loop-carried phis (a swap) correct.

Cross-block software pipelining then shrinks the terminator offset down to the issue-side envelope -- the latest PC at
which the block still drives a control word -- whenever every successor is single-predecessor, so a spill cannot reach a
wrong path. In-flight results land past the terminator in the uniquely-reached successor frame, carried there by the
fetch pipeline with no replicated microcode; the successor inherits the predecessor's per-instance busy residue and each
spilled value's landing cycle. A multi-predecessor successor (merge, loop header, `Ret`) never receives a spill, so the
carry converges in one reverse-postorder pass and no overlap crosses a back-edge.

Compile-time-known branch conditions fold to a single arm so the other is never lowered (no spurious state from an
unreachable write). This shared reachability predicate is deliberately narrower than the complete HIR constant folder --
a constant condition buried under a shape it does not inspect stays a runtime branch, at worst an unused state register,
never a miscompile. Unifying the two is tracked future work.

### DEFERRED

Aggressive cross-block overlap: the landed pipelining shrinks a terminator only to its issue-side envelope, so
write-opcode words stay in-block and no microcode is replicated. Pushing further -- letting the write-opcode words spill
past the terminator -- would shave the remaining per-block tail but needs the commit-side control fields replicated into
every successor arm, policed by the single-writer microcode validator (already in place). Overlap also stays off across
any multi-predecessor edge.

Empty merge-block elimination (the HIR merge-threading pass above) leaves two cases a real branch: an empty `else`-arm
block (threading would create a forbidden branch-block phi arm) and a merge phi read outside a successor phi arm (a
loop-invariant value used in the loop body, which would need rematerialization as a self-referential loop-header phi --
unproven against the emitter, not worth the niche benefit).

## Backend (VLIW/ZISC)

The Verilog backend is mechanical from LIR: an inline flop bank plus the 1-bit boolean bank, one module per pooled
operator instance, and one continuous assignment per pooled constant. Either bank is emitted only when used (a
purely-boolean kernel carries no wide register file). There is no general multiport register-file module; storage is
the sparse, schedule-specific fabric below. The controller is a microcode ROM -- one pre-decoded VLIW
control word per step, written as a synchronous `case` over the fetch PC. That `case`-over-address is the inferable-ROM
form every backend recognizes, so each maps it to an appropriate ROM (LUT logic or block RAM), rather than the
array-plus-`initial` form, which some backends do not recognize as a ROM at all (flattening it to logic) and others
force into a slow block RAM even when tiny. It is read through a short multi-stage fetch (a PC
latch, the ROM read register, and a routing register) so the controller is short register-to-register paths rather than
a wide combinational `case(cyc)` cone, for a fast clock-to-out. The ROM `case` occupies its own clocked block -- the
sole sanctioned second `always @(posedge clk)` -- since that dedicated form is what triggers the memory inference. The
fetch leads the executing step, which under static scheduling only adds to the makespan/II; the depth is currently fixed
but may be made disableable for faster chips.

The schedule replays step by step: at PC 0 the machine accepts and parallel-loads inputs into the low registers of each
bank in one cycle (gated by `in_valid`); the PC advances every clock; at the last PC it asserts `out_valid` while
outputs drive combinationally from their registers by fixed index. Both banks read combinationally, so each operand's
read opcode rides its issue step (a combinational read mux samples the operand a fetch lag later) and each register's
write opcode its source's executing step. The PC holds only at the two I/O boundaries; bubble steps carry an explicit
NOP. While the PC dwells it re-fetches that word each cycle, but `transacting` (high only while a transaction is in
flight) forces every operator's `in_valid` and every register's write opcode to the inert NOP code, so the idle
re-fetch commits nothing and the entry word can carry real work.

The control word stores selectors and value routing is uniform across two dual endpoints: a per-operand READ opcode
selects that port's source, and a per-register WRITE opcode selects that register's next value (code 0 == NOP hold).
An operator output, an inline expression (boolean logic, a cast, a select), and a phi-arm/constant/state
move are all just sources one write opcode picks, so PC gates no datapath read or write -- it is left to control flow
alone. A control field constant across the whole program -- a sign control the program never varies, say -- is driven
by a constant net and lifted out of the ROM, so synthesis prunes what it feeds; the Python ROM packer and the module's
bit-slice offsets are produced together so they cannot drift. A gated field (an operator issue strobe or a write
opcode) ANDs `transacting` into its decode -- a width-generic mask to the NOP code -- so a dwelling re-fetch is inert.

Sparse storage. Each operand's read mux is a `case` over its read codebook -- the registers it reads plus each distinct
constant it reads (a constant's magnitude, its sign riding the operand-sign field into the wrapper) -- and each
register's write is a `case` over its write codebook, both indexed by the endpoint's dense opcode. The `case` form
deliberately avoids an indexed part-select into a packed gather bus, whose variable offset is a multiply that, at a
non-power-of-two word width, makes Lattice Diamond's LSE infer a DSP per operand; a `case` has no offset arithmetic and
measures smaller and faster on every flow. A single-source read port drives its lone source directly and needs no
opcode field (a single-source register still carries a 1-bit write opcode -- its folded write-enable/NOP); every opcode
is sized to its own codebook, never the file-wide index, so the ROM word stays narrow.

For error-bearing operators that survive optimization, errors are non-fatal and informative: each flag (`div0`,
`domain_error`, etc.) ORs into a global `err` gated by whether some destination register's write opcode selects that
instance's output this step (its commit window), and an `err_pc` latch records the executing step of the last error
(reset at every accept).

Reset covers the control registers and the persistent state registers: the reset arm loads each state register with its
snapshot while the non-reset arm applies that register's opcode-selected update and its boundary install (a
handshake-gated arm at `out_valid && out_ready`), the two segregated as the arms of one `rst` condition. The fetch
registers and the rest of the datapath are reset-unconditional (so they pack into the BRAM output register) and settle
to the first word under reset. The control word and datapath skeleton are the only ZISC-specific part --
LIR itself is controller-agnostic.

Why read-first plus a +1 dependency cycle, not write-through forwarding? Write-through would erase the +1 but its
forwarding muxes cost `O(NRD*NWR)`, and we need many ports -- unsustainable. Read-first plus the +1, hidden under
pipelined overlap, is the better trade. Constant operands are kept as immediate `const_N` nets the read opcode selects,
never stored in the ROM word; folding them into the register file or emitting constant-load micro-instructions are noted
alternatives for when this becomes a constraint.

Each operator instance carries its own Holoso-exposed parameters and float format, fixed at construction from the
`OpConfig`. Every instantiation lists every hardware parameter explicitly, so it is self-describing and turns a
param-name mismatch into a loud elaboration error.

Support library. The auxiliary HDL shipped with a module is a single self-contained `holoso_support.v`, assembled in
memory from hand-written operator catalogues plus included external RTL sources, so the end application introduces
all RTL dependencies by adding one large file to the synthesis input.

### Numerical model

The numerical model gives bit-exact, cycle-exact emulation of the emitted HDL without HDL emission or simulation, so the
synthesis logic can be verified through LIR during heavy refactors. It is bit-exact because it replaces native float
operators with the ZKF package's bit-exact software implementation of the selected float format, and
cycle-exact because it mirrors the RTL's fetch PC, register files, and sequencer.

It splits into a serializable handle and a runtime machine: the handle is a trivially-picklable wrapper carrying only
the LIR (kept private, so the LIR never enters the public API) -- the artifact a generated testbench embeds -- and
elaborating it builds the runnable per-clock state machine. The split keeps the serializable artifact pure data and
frees the simulator from fighting pickle. Both expose the kernel's logical signature as read-only metadata (each port a
logical name paired with a scalar type), so a driver decides a port's encoding by matching its type and the signature
stays honest as scalar types are added.

A tick advances exactly one `posedge clk` with the same sequencer the Verilog emits (reset, out_valid, in_ready,
terminator redirect, back-pressure). The only mutable state beyond the register files is a small in-flight buffer -- the
stand-in for the operator pipeline: a result is computed when its operands are sampled but written to the register file
only at its landing PC, exactly as the hardware does. It therefore stays correct when blocks
overlap and runs an arbitrarily deep loop in bounded memory. The persistent state is just the slot registers, carried
across transactions.

Generated RTL testbenches (Cocotb today) run the RTL simulator in cycle-by-cycle lockstep with the elaborated numerical
simulator: each cycle ticks both with the same handshake and asserts that `out_valid` agrees (the data-dependent latency
check) and that the output bits match when valid, back-pressure included. End-to-end verification of the original Python
against the numerical model is left to the user, since it requires knowledge of the source semantics.

The RTL-versus-model cosimulation is structurally blind to one miscompile class: a scheduling, instance-binding,
register-allocation, or cross-block-overlap fault in the LIR is shared by both the RTL and the numerical model (the RTL
is emitted from the very LIR the model replays), so a wrong-but-consistent LIR passes the cosim. A schedule-independent
oracle closes this gap: a MIR interpreter evaluates the unscheduled MIR dataflow graph directly through the operators'
own bit-exact `evaluate`, owning no registers, no schedule, and no overlap machinery, and deliberately importing nothing
from the LIR. Since it shares the front/mid-end and the operators with the numerical model but none of the LIR, the
differential `interpreter == model` isolates exactly the LIR layer.

The HTML report must give humans an EXACT representation of the generated core behavior -- the tool for understanding
and debugging what the compiler did -- not a simplified or approximated view.

## Fabric-area exploration

The synthesized fabric is dominated by the per-operand read multiplexers: on a register-pressure-heavy kernel (an EKF
update) they are roughly 60-65% of the LUTs. The read-set sizes sit at the interference floor -- the values a port reads
are largely simultaneously live -- so the muxes encode real liveness rather than allocation slack, which bounds most
levers. Results below were measured end-to-end across Yosys+nextpnr-ECP5, Lattice Diamond, and Vivado, and are recorded
so the dead ends are not re-explored.

Adopted (lossless, f_max-neutral):

- Read and write muxes as a `case` over the endpoint's dense opcode (the read codebook folds in the constants a port
  reads; the write codebook, the sources a register takes) rather than an indexed part-select into a packed gather bus,
  or the nested-ternary const-pool selector it replaced: smallest and fastest of the encodings tried. Nested-ternary
  muxes are catastrophic.
- Commutative operand port assignment, solved exactly as a MILP: a few percent LUT on the EKF across all three tools, at
  zero hardware or latency cost. Based on Chen & Cong.
- A per-register write opcode and a grouped input load: read/write symmetry (read selects a source per port, write
  selects a source per register) that folds every write-enable, write-address, const-pool selector, and boolean
  inversion into one tiny opcode, at modest ROM cost. A signed-constant install folds its sign into a compile-time
  constant; a register-source sign stays a runtime `holoso_fsgnop`.

Explored and rejected for register-pressure-bound kernels:

- LUTRAM register file: a multi-write workload needs a live-value table costing as many LUTs as the FF+mux it replaces;
  banking helps only when access sets partition cleanly.
- Register-file size cap via pressure-limited scheduling: `nreg` floors at peak liveness, so it trades large latency and
  f_max for a couple percent.
- Operator replication and FMA fusion: both raise read-operand traffic (more, or wider, read ports), enlarging total mux
  area despite fewer ops or a shorter makespan. This is why the FMA contraction is opt-in (only when `ffma` is
  configured): it is a numerical feature (single- vs double-rounding), not an area lever, so a pressure-bound kernel
  should leave `ffma` unconfigured.
- Operand collectors (copy/move ops off the worst-reach ports): a copy relocates fan-in rather than removing it -- a net
  gain needs a value moved onto a co-reachable but not co-live target, which the interference floor denies, and copies
  on the shared operator also cost cycles.

Latency-for-area trades (set aside -- latency is a real cost and the area gain did not justify it):

- Distributed/banked register file with scheduled inter-bank copies (Cong's RDR): banking narrows each port's mux but
  serializes the schedule. On the EKF, whose muxes are already at the interference floor, not worthwhile.
- Shared read bus / vertical microcode (one operator per cycle, two shared operand buses): a modest total-LUT saving for
  a large latency cost, f_max-safe -- not pursued.

## References

- L. Chen, J. Cong. Register Binding and Port Assignment for Multiplexer Optimization. ASP-DAC 2004. Basis for the
  commutative operand port-assignment pass.
- J. Cong, Y. Fan, et al. Architecture and Synthesis for Multi-Cycle Communication (the Regular Distributed Register
  microarchitecture). ISPD 2003. Banking plus scheduled inter-bank copies.
- A. Terechko, et al. Inter-cluster Communication Models for Clustered VLIW Processors. HPCA 2003. Producer-side
  placement preferred over after-the-fact copies.
- M. Gebhart, et al. A Compile-Time Managed Multi-Level Register File Hierarchy. MICRO 2011; S. Asghari Esfeden, et al.
  CORF: Coalescing Operand Register File for GPUs. ASPLOS 2019. Compiler-staged operand near-files (target access
  energy).
- A. W. Appel, K. J. Supowit. Generalizations of the Sethi-Ullman algorithm for register allocation, 1987. Why a copy
  relocates steering cost unless it collapses a fan-in cone.
- AMD UG949, Vivado Design Methodology, "When and Where to Use a Reset." Intel Hyperflex Architecture Handbook,
  "Synchronous Resets Summary" and "Reset Strategies."
