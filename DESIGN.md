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

The front-end is the FIR pipeline under `holoso/_frontend/_fir/`: build (AST -> FIR) -> analyze (facts + plan) ->
emit (FIR -> HIR). The layering exists because a single-pass lowerer forces a reachability scan to run before the
folds it must agree with -- the scan/lowering duality that plagued the previous front-end; here every semantic
decision is the analyzer's, made once, over the same graph the emitter walks.

Name classification comes from the live code object (CPython's own; `del`-locality and PEP 709 targets correct by
construction) with comprehension-only targets carved out via the AST; non-locals resolve closure-cell before global
before builtin, values read at resolve time. Static values form a closed whitelisted domain with provenance as
identity: exact Python ints (MetaInt) are distinct from numpy 64-bit scalars (NpInt/NpFloat, numpy's own
mixed-operand semantics; narrower numpy widths are not static); arrays admit as read-only snapshots; sequences keep
list/tuple flavor; dataclasses admit as records only when reconstructible from fields; containers bound depth and
refuse cycles. Fixed-point equality is tagged and bitwise (True is not 1, signed zero distinct, NaN stable). Static
execution runs real Python/numpy on the domain's own objects; zero-division/invalid defer to runtime, float overflow
folds to infinity per the charter, numpy wraparound folds faithfully, NaN never folds, and integer powers fold only
under a result-width bound.

The FIR itself is a private non-SSA CFG over mutable Places (Local/StateLeaf/ReturnPlace), built syntax-directed
with no analysis: ANF write-once temporaries in exact Python evaluation order; ordered stores; one canonical exit
per FunctionUnit (return = store + jump); eager and/or/chained-compares via selects (pinned semantics); ifexp as
real branches; for-loops and comprehensions as StaticFor templates (comprehension targets in their own frame, the
outermost iterable evaluated in the enclosing scope); lazy PyCall; raise/assert/missing-name as Fail terminators
(dead branches stay dead); del as UnbindPlace; origin stacks on every node. Walrus is supported except in
short-circuit positions; nested def/class/lambda, imports, subscript stores, and variadic parameters are located
rejections. A structural verifier and a deterministic printer close the stage.

The analyzer is SCCP-style optimistic executable-edge abstract interpretation over the FIR with flow-sensitive
per-edge environments (strong updates, joins only over executable in-edges). Facts: Unbound | Known(StaticValue) |
Residual(type) | fact-level sequences (static shape, runtime leaves) | MaybeUnbound (a read of one is a located
rejection: Python may raise). An int/float join promotes the integer side to float, C-style (see Integers). Folding
is Python-exact on Knowns; runtime-typed values never fold (width rule); a Known Bool always drives edge selection.
StaticFor unrolls by cloning the body per trip once the iterable is Known; calls expand on demand by grafting the
callee template (defaults/kwargs bound, member __call__ dispatch, recursion rejected by function+receiver ancestry,
origins re-attributed to the user call site). State: the W/D fixed point -- W accumulates executable
exit-co-reachable store leaves (typing by reset value), D live-ins start at Known(reset) and join executable exit
live-outs, descending only; each round rebuilds the working graph from immutable templates. Executable Fail
terminators are located rejections (data-dependent raise included) -- which is also what lets a library stub
validate its own operand shapes in ordinary Python: a `raise` behind a statically-false shape check folds dead.
Fuel bounds cover visits and blocks; exhaustion is a located rejection, never a truncated fixed point.

On stabilization the analyzer finalizes its result into the emission plan: the authoritative fact per binding
(temporaries are write-once, so one replay of the transfer records each), a typed plan per surviving call --
folded | identity | conversion | intrinsic, keyed by the call's destination binding -- and the state leaves in
first-store source order (the established port ABI orders ports by source text, not CFG shape). This plan is the
whole analyzer/emitter contract: emission never re-derives a fold, never resolves the library registry, and never
replays the transfer function, so the two layers cannot disagree about a value's meaning.

Emission lowers the stabilized residual graph to HIR: executable blocks/edges only, in reverse post-order, with
value numbering by Braun sealed-block SSA over Places (named locals, state leaves, the hidden return place) and
write-once ANF temps unified into the same layer. One typed materializer serves every operand position: a Known
value becomes a constant of the expected kind (a Known integer stays an IntConst in an integer context and rounds
into a float constant in a float context), a residual value is its SSA read, coerced only where the coercion is a
genuine int->float promotion on its own edge (phi arms, select arms, state stores, comparison operands) and a
located rejection otherwise. Folded branches lower to jumps; a loop header reads its init arm by recursing through
already-emitted predecessors and closes its latch arm at sealing (phi insertion saves/restores the builder
position; a write-once temp never needs a phi). Module and class attribute access is a plain namespace lookup, not
component state. State slots carry the reset snapshot and canonical-exit live-out; out ports, public `state_<slot>`
ports, and live-out dedup follow the established ABI. The differential harness's oracle is the kernel's own Python
float64 evaluation on quantized inputs.

Rejecting from within the subset. Analysis descends only executable edges, so a `raise` it reaches sits on a path
taken unconditionally (or under a residual guard, which the hardware cannot signal either way): it becomes a
synthesis error carrying the exception's own message, located at the raising line. A rejection inside an inlined
library stub is re-attributed to the user's call site under the spelling they wrote.

Persistent state. A synthesized method's `self` is not a port: each instance attribute the method writes on an
executable exit-co-reachable path becomes a persistent register (a loop-carried value, the back-edge of the
initiation loop), and each attribute it only reads is a frozen constant folded from the `__init__` snapshot. Within
the method `self.attr` is an ordinary Place, so reads and writes interleave freely. Public attributes additionally
drive a `state_<attr>` output port, so a method need not return anything (and a returned value that is by dataflow
a public attribute is deduped onto that state port); underscore-prefixed attributes stay internal. Reassigning
`self` itself is rejected: attributes resolve against the fixed original instance, so a rebinding would silently
miscompile.

Inlining. A pure function reachable through `__globals__` is inlined -- its body grafted with a fresh activation
frame, its return remapped to the caller -- so kernels compose. A method call on the synthesized instance
(`self.helper(...)`) is inlined with the instance context kept, so the callee's own `self.<attr>` reads resolve
(found through the class MRO; `@staticmethod` and `@property` getters are supported -- a property getter desugars
to a bound zero-argument call and is inlined like any method, so it recomputes from the current state on each
read). A called member method may read AND write `self`: the state fixed point spans the whole expansion, so a
write in an inlined method promotes and carries its slot exactly as a write in the entry method does.

Hierarchical components. A component may hold other components as members (`self.lpf = IIR1LPF()`) and call them;
each child's persistent attributes become nested state slots, so several stateful owners coexist in one kernel. An
owner is identified by object identity, and its slot name is its canonical member path from the root -- the fewest
segments, lexicographically least, so an aliased child (two member names for one object) resolves to one slot --
joined to the leaf attribute by a double underscore. A root attribute keeps its bare name (`m`, port `state_m`,
preserving the flat ABI); a child's becomes `lpf__m`. The encoding is injective except when an attribute name
literally spans a `__` boundary, which is a located slot-name collision rejection. Public/private visibility is
read from the leaf attribute itself, not the owning alias. A stateful component reached only through an unanchored
reference (a global, not a member of the root) is rejected at its state access, and rebinding a component member --
a per-transaction topology change -- is a located rejection.

Math library. A call dispatches through a registry on the object identity its callee resolves to, not the spelled
name, so an alias resolves and a shadow does not. The registry maps that object to its lowering: an intrinsic stub
(1:1 onto an HIR operator, result kind per the three-rule scheme in Integers) or a composite stub (built from the
intrinsics and inlined). A composite is ordinary Python in the supported subset, so each stub is its own numerical
reference, and a rejection inside one is re-attributed to the user's call site.

Parameters and return. Positional and keyword-only parameters become input ports and require an explicit
annotation: `float` scalars are floating-point ports, `bool` are 1-bit boolean ports, `int` are typed integer ports
(contained at MIR until the integer backend). The return annotation is likewise mandatory and validated against the
emitted kind: `float`, `bool`, `int`, or `None` for a method that returns nothing (an `X | None` union unwraps its
None arm for early-return kernels).

An `assert` statement is accepted and ignored wholesale: its test is never lowered, mirroring Python under `-O`, so
an assertion has no hardware effect. Any effect the test would have had when executed is dropped along with it; as
under `-O`, an assert must be side-effect-free.

### Deferred: the aggregate contract (tracked by FIR_PARITY_PENDING; stage 10 asserts the registry empty)

The previous front-end supported statically-shaped aggregates end-to-end; the FIR pipeline does not yet, and every
disabled test carries the greppable marker. The contract the aggregate stages restore (and extend with records,
reductions, and the bounded gather):

Matrices/vectors are statically shaped and unrolled to scalar operations; arrays never exist as hardware
aggregates, only as compile-time bookkeeping over scalar leaves -- list/tuple literals and comprehensions,
numpy-style indexing and constant slicing, iteration, unpacking, transpose, and the array factories
`np.array`/`asarray`/`asanyarray`. A Python list/tuple is never given numpy semantics: elementwise arithmetic, the
matrix product, transpose, `.flatten()`, and multi-axis indexing apply to arrays and are rejected on a list/tuple.
Elementwise `+ - * /` apply leafwise to same-shape arrays with scalar broadcast only (mixed-rank pairs reject
rather than silently aligning). Shape queries (`len`, `.ndim`, `.shape[k]`) evaluate to compile-time integers.
Linear algebra is ordinary library code: `@` and `np.matmul` resolve through the same registry a spelled call uses,
expanding into scalar multiply/add chains with left-fold dot products (enabling FMA contraction); operand axes are
bounded by the unroll threshold. A jaxtyping array annotation with fixed 1-D/2-D dimensions and a floating dtype
decomposes row-major into one float port per element (`name_0, name_1, ...`; matrices `name_0_0, ...`), detected
structurally so jaxtyping stays a dependency of the user's code only. Aggregate-valued state decomposes into one
scalar slot per leaf (`attr_0`, ...; matrices row-major `attr_0_0`, ...). Aggregate returns flatten to ordered
`out_<k>` ports, validated against an arbitrarily nested `tuple[...]`/`list[X]`/array return annotation.

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

A nested `if` with no `else` (`if A: if B: S`) is predicated to a single combined-`and` branch by guarded-region
if-conversion in HIR: the two-branch region reconverging at one merge collapses to `select(A and B, ...)`, emitting one
mux instead of the nested pair a bottom-up diamond collapse would leave. It fires only when every value the merge
observes on the inner-false path equals its outer-false peer, so the two bypasses are interchangeable -- an assignment,
walrus, or state write on the inner path makes some value differ and disables it, as does a faulting or stateful
operation in the guard block (which leaves it non-empty) or an `else` on either level. It combines existing boolean SSA
values only, so eager-`and` evaluation is unchanged; the guard `B` must already dominate the outer branch, so nothing
is newly speculated.

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

Integers. HIR carries a typed integer vocabulary -- `IntType`/`IntConst` and signed operators (saturating
add/sub/mul/neg/abs, floor-coupled `//`/`%`, dynamic shifts, bitwise, relational, int-select, and the
int<->float/int<->bool conversions). All exact integer folding is the front end's (`MetaInt`, arbitrary precision);
HIR performs no integer constant arithmetic, only the conversion folds and the identity/elision peepholes. Genuinely
pure integer expressions stay integer end-to-end and stop at MIR: the integer BACKEND (typed MIR views sharing the
wide register bank) is a later milestone, so any integer node -- operator, constant, input, or state slot -- reaching
MIR is a clean located "not yet lowerable" rejection.

Everything mixed promotes to float, C-style. An `IntToFloat` sits on each integer edge feeding a float operation, a
`return`, or a state store into a float slot; `/` and int+float promote while `//`/`%` stay integer and floor-couple.
A control-flow merge (phi, conditional select, state-leaf join) of an integer path with a float path promotes the
integer side on its own edge and yields a plain float -- Python instead keeps each path's runtime kind, which is the
documented C-style deviation, and the same rule covers an int/float comparison, which promotes and compares in float.
The precision loss is accepted under the fastmath charter: a Known integer materializing in a float position rounds
into the binary64 carrier (only a value beyond that carrier entirely, e.g. 10**400, is a located rejection), and the
selected target format then rounds again like any constant. Static folding remains Python-exact: a fully-Known
int/float comparison folds exactly; only runtime values promote.

Library intrinsics are typed by a three-rule declarative registry: SIGNATURE (the operator's own result; integer
operands promote -- `math.fabs`/`np.fabs`, `np.rint`, the transcendentals), ALWAYS_INT (the int-returning
`math.floor`/`ceil`/`trunc` and one-argument `round`: identity on an integer, float-op-then-`FloatToInt` on a float),
and INT_OVERLOAD (`abs`/`np.abs`, `np.floor`/`ceil`/`trunc`/`round`, every `min`/`max` spelling: all-integer operands
use the integer implementation, contained at MIR; any float operand promotes and runs the float operator). A builtin
`min`/`max` mixing an integer and a float therefore returns float, not the winning operand's own type. A
classification (`isfinite`/`isinf`/...) of an integer folds ideally to its constant (integers saturate, hence stay
finite). Bitwise and shift operators are bit-true and require two integers (or two booleans for `&`/`|`/`^`, which
stay in the boolean bank); a boolean shift or a mixed bool/int operand is rejected, as is a compile-time-known
negative shift count, while a runtime negative count is the hardware's documented reverse-shift deviation.

Power: a compile-time integer exponent expands to a bounded multiply chain in the base's own kind (an integer base
with a nonnegative exponent stays an exact integer chain, contained at MIR; a negative exponent reciprocates in
float). A runtime exponent computes as the direct fastmath identity `exp2(e * log2(b))` -- base two skips the log2 --
with a constant base's log2 folding statically; a negative base is a log2 domain error, as in C. The `pow`/`np.power`
spellings still route through the guarded composite stub, which honors negative bases on small integer exponents;
collapsing them onto `**` and giving `np.power` an integer overload is deferred to the integer wiring milestone, as
is an integer lowering for `np.sign` (an integer operand is a located rejection meanwhile).

Two conversion round-trips canonicalize in HIR under the charter: `f2i(i2f(n)) -> n` (promotion precision loss
deliberately ignored), and `i2f(f2i(x)) -> FloatTrunc(x)` (collapsing to `x` when `x` is already integer-valued),
which keeps a `float(int(x))` truncation inside the float datapath.

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
