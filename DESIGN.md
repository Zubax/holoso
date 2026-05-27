# Holoso design

Holoso lowers a small subset of Python (numerical control/DSP kernels) into vendor-neutral, synthesizable Verilog.
See `README.md` for scope and `PRIOR_ART.md` for why existing tools don't fit. This document records the architecture
we are building toward; it is expected to change frequently, and often may not be up to date.
Initial exploratory notes live in `DESIGN.draft.md` (outdated, superseded by this document).

One must read the representative use-case examples under the `examples/` directory to understand the motivation.

## Direction

- Build our own compiler. The differentiating work is the front/mid-end: partial evaluation of Python, shape
  inference, and operator scheduling for a resource-shared FSM. No external HLS gives us this for Python, and every one
  would force us to drop the Zubax Kulibin float (ZKF) library and adopt a pipeline-oriented optimizer we don't want.

- Delegate only to lightweight Python tools where it clearly pays: SymPy (fold/CSE/simplify), optionally
  Veriloggen's AST for emission, Cocotb for testbenches, optionally an ILP solver for an exact scheduling mode.
  Other lightweight dependencies may be freely introduced as needed.

- Bambu/XLS/CIRCT are not backends. Bambu is kept as a verification oracle and as inspiration only.

- The target is a specialized program, not a pipeline. We synthesize a sequential FSM (a zero-instruction-set
  computer, ZISC) that time-multiplexes a few shared operators over a register file.
  We do not pursue a constant II or II~1 like a streaming pipeline: the initiation interval is whatever the scheduled
  program costs. For a fixed control path it is an exact, statically known cycle count derived from the per-operator
  latency model (data-independent in v0); it varies across programs and, later, across branch paths.
  This is a compiler problem more than a circuit-design one.

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

- HIR -- "what to compute": SSA dataflow inside a control-flow graph with real branches. Target-independent and
  semantic; it does not know how an operation is implemented. The `holoso._hir` subpackage owns the IR, semantic
  operators, and hardware-agnostic optimization passes.

- MIR -- "which hardware to use": selected hardware operators, with typed input/constant/operation/output nodes,
  still unscheduled. The current implementation has float-specific subclasses carrying folded sign controls. The
  `holoso._mir` subpackage owns the selected IR and HIR-to-MIR lowerer; this is the first stage allowed to inspect
  `OpConfig` or float-format limits.

- LIR -- "the microprogram": the scheduled, bound, register-allocated op stream for the synthesized machine.
  It has generic resource/operation base classes plus typed resource families such as the current float register file.
  The `holoso._lir` subpackage owns the IR, MIR-to-LIR construction, scheduling, binding, and register allocation.
  Controller-agnostic; this is the seam where a second controller backend can be added later.

- Backends -- Verilog, testbench, HTML report, numerical model

Mental model: HIR is the source-level compiler IR; MIR is selected machine-independent hardware dataflow; LIR is the
instruction stream of a tiny specialized processor; the Verilog backend is its assembler and datapath generator. The
backend stage is a family of independent backends --
the Verilog module, an HTML report, a Cocotb testbench, and a bit-exact numerical model -- each consuming the LIR,
and possibly some additional inputs, such as outputs of another backend.

The numerical backend is helpful during development and heavy refactors: it allows early verification of the synthesis
logic without involving the actual HDL emission and simulation steps, which are slow to iterate on.
Thus, the normal policy during development is to stabilize the synthesis logic down to the LIR using the numerical
model for verification, and once that is proven, move on to the actual HDL generation and testbenches.

## Python API

`synthesize` takes the object -- a function or class, not a source file or path -- and returns an in-memory result;
nothing touches the filesystem unless the caller asks.

```python
def synthesize(target, *, ops: OpConfig, parameters: Mapping[str, object] | None = None,
               entry: str = "__call__", name: str | None = None,
               operator_instances: Mapping[type[HardwareOperator], int] | None = None) -> SynthesisResult: ...

@dataclass(frozen=True)
class SynthesisResult:
    module_name: str
    interface:      ModuleInterface   # typed ports -- the composition contract
    verilog_output: VerilogOutput     # generated module text + support_files (the shared holoso_support .v/.vh)
    numerical_model: NumericalModel   # bit-exact, picklable pure-Python model of the module (flat in -> flat tuple out)
    cocotb_output:  CocotbOutput      # self-contained testbench: embeds the model, checks the DUT bit-for-bit
    html_output:    HtmlOutput        # self-contained single-page report
```

The root package re-exports only the supported public API, keeping the API surface to the minimum.
Private implementation modules may still expose unprefixed package-internal entrypoints at subsystem boundaries;
this is fine because they are shielded by the `__init__.py` selective re-export policy (not visible from outside).
Purely module-local helpers and type aliases inside those private modules are underscore-prefixed.
Same applies to nested subpackages: their internals are private to the subpackage, each has its own API.
Private module-local classes use unprefixed attributes for fields that sibling helpers need to access; the class name
itself provides the module-local privacy boundary. Underscore-prefixed attributes are reserved for state accessed only
by the owning class or its descendants.

The library should not contain entities that are only used in the unit test suite; those belong in the suite.

Passing the object is more ergonomic and strictly more capable than a file: it carries the runtime environment the
binding-time front-end needs -- `__globals__`, closure cells, default args, and the result of running `__init__` --
which is what evaluates compile-time tables and follows/inlines imported callables. The object is the compile root; the
boundary ("what to ignore") falls out of reachability + binding-time analysis, not manual enumeration. Source is read
via `inspect.getsource` + `ast`; when unavailable (REPL/`exec`/notebook-defined, some lambdas) synthesis fails with an
explicit error. For a class, `__init__` runs with `parameters` (overriding the kw-only defaults that otherwise map to
Verilog parameters), attributes written by `entry` become state registers, and `entry` (default `__call__`) is analysed
with the ports dynamic; a plain function is analysed directly. `result.write(out_dir)` is the only operation that
touches the filesystem.

## Front-end

Abstract interpretation over the Python AST/CFG with a binding-time lattice (static vs. dynamic), not tracing:

- Static values (shapes, `__init__`-derived constants, compile-time tables) are evaluated concretely -- real
  Python/NumPy runs at synthesis time.
- Dynamic values (input ports, persistent state) become SSA handles that accumulate HIR.
- `for`/`while` with a static trip count is unrolled; a dynamic trip count is rejected for now (the only case that needs
  a genuine variable-length loop -- a future feature).
- `if` on a static test takes one branch; `if` on a dynamic test emits a real branch (see HIR below).

Matrices/vectors are statically shaped and unrolled to scalar operations at synthesis time (as in the SymPy-CSE'd
`ekf1` example); arrays never exist as hardware aggregates, only as compile-time bookkeeping over scalar registers.
Reductions (`max`, `argmax`, `mean`, `@`) lower to compare/select trees and multiply chains. Input shapes are declared
with jaxtyping (`Float64[np.ndarray, "4 4"]`, concrete dims only); interior shapes are inferred.

## Types

Runtime values are only:

- `float` -- one ZKF format, `WEXP`/`WMAN` fixed per build.
Typical FPGA-friendly formats: WEXP=8 WMAN=36 (44 bits) for precision; WEXP=6 WMAN=18 (24 bits) for simpler targets.
Generated top-level modules are not parameterizable by `WEXP`/`WMAN`: port widths are hardcoded and the selected float
format is recorded by the typed float register-file resource and as internal localparams. Changing the float format
requires re-running synthesis because operator latencies, the static schedule, and register widths are all tied to that
choice.
- `bool` -- 1 bit.

HIR types live in `holoso._hir` as format-free `Type` values; today that family contains `FloatType`. Concrete scalar
types live in `holoso._type`: `ScalarType` is the width-bearing MIR/interface/resource type family, today containing
`FloatType`, whose `FloatFormat` describes the ZKF encoding. A data port carries its scalar type and derives its bit
width from it; control ports carry explicit bit widths. Today all data ports are the same scalar `FloatType`, but this
is an implementation detail.

Compile-time ints/shapes/structure are resolved in the front-end and never reach HIR. A dynamic integer only ever
appears as an index into a static table; it is lowered to a one-hot bool vector + mux, never materialized as an int.

A FloPoCo backend may be introduced later on if makes sense, but it is likely to be mostly shielded behind the
`holoso_support.v` wrapper, so the effect on the codegen is minimal.

## HIR

```
# values
in_port(name, type)               # module input; concrete scalar type is assigned at HIR-to-MIR lowering
float_const(value)
state_read(slot)                  # persistent state at block entry
phi([(pred_block, value)])        # SSA merge

# pure semantic operations (generic; selected into concrete hardware by a later pass)
operation(operator, operands)      # float_add, float_mul, float_div, float_neg, float_abs, float_mul_pow2, ...
relational(op, a, b) -> bool      # lt, le, eq, ...
boolean(op, ...)     -> bool      # and, or, not, xor
select(cond, a, b)                # DATA mux (not control flow)
cast(a, to_ty)                    # bool <-> float
intrinsic(kind, args)             # sqrt, sincos, exp, ...   -> operator module, else hard error

# sinks
state_write(slot, value)
out_port(name, value)
```

Terminators: `jump(target)`, `branch(cond_bool, t, f)`, `ret` (commit state-writes + outputs, raise `done`).

State. Persistent state = class attributes; `__init__` gives initial values (folded, or kw-only params -> Verilog
`parameter`s). An unwritten persistent register holds its value. Reset reaches only state regs that are live-in at reset
before any dominating write (in practice the boolean control flags); pure datapath state stays out of the reset cone.
Registers that hold values assigned in `__init__` are explicitly assigned initial values at module reset.

Branch vs. select (the core control-flow decision):

- A real `if`/`else` lowers to a `branch` terminator + a `phi` at the merge. Only one side executes; the merge is
  resolved at register allocation by coalescing both definitions onto one register -- no runtime mux, the untaken
  arm is never computed, and no spurious error is recorded. Branches are the default.
- `select` (a mux, both inputs live) is reserved for data multiplexing (one-hot lookup, `where`-style picks) and for an
  optional if-conversion peephole that collapses a tiny, pure, cheap diamond. Conservative by default.

The implemented v0 HIR value set is float-only, but the node names are explicit (`FloatConst`, `FloatAdd`, etc.) so
bool/int nodes can be added later without overloading float semantics. Negation and absolute value are ordinary
semantic float operations in HIR. They are not represented as hardware details until selection.
HIR operators expose a HIR-local `Signature`, and the builder rejects operands whose HIR types do not match.

## HIR optimization and lowering

HIR optimization is hardware-agnostic: const-fold + algebraic simplify (SymPy-assisted) - CSE - strength reduction
(`x*2^k`, `x/2^k` -> semantic `float_mul_pow2`; `x/c` -> `x*(1/c)` for finite non-power-of-two constants; `x**n` ->
multiply chain) - optional if-conversion - DCE. Constant folding is typed: an operator receives constant nodes and
returns a folded `Const` node, not an untyped Python value. The HIR builder can re-intern an arbitrary `Const` node
with `const_node()`, so future bool/int constants do not need float-specific rebuilding in shared passes.

HIR-to-MIR lowering lives in `holoso._mir` and is implemented by a lowering context that owns the HIR tree, `OpConfig`,
MIR builder, and value remap. The context delegates domain-specific nodes to private lowerers; today the only domain
lowerer is float. The float lowerer maps each semantic float operator to its configured `FloatHardwareOperator` from the
single root-level hardware-operator config and collapses semantic `float_neg`/`float_abs` chains into selected-float MIR
`FloatSignControl` values on operator operands/results or output wires. Semantic `float_mul_pow2(k)` selects
`fmul_ilog2_const` when the configured float format supports that exponent; otherwise it falls back to ordinary multiply
by the constant `2^k`.
`MirBuilder` is a single graph builder with typed construction methods; it does not own a global scalar type, so future
mixed-type expressions can share one value namespace and add typed constructors for bool/int values. Hardware operators
expose a concrete `ScalarSignature`, and MIR construction validates operands against the selected operator's signature.
HIR-to-MIR lowering rejects semantic domains that do not yet have a selected MIR representation instead of silently
treating them as floats.

Typed MIR subclasses validate their local invariants at construction time. For example, selected-float MIR nodes verify
that their scalar types/operators/sign controls are float-domain objects and that operation operand/sign sidebands match
the selected operator arity. Cross-node operand type checks still belong in `MirBuilder`, where the referenced values
are available.

Note: it is understood that FP math is non-associative and some of these optimizations may result in non-bit-exact
results, which is accepted.

## LIR

```
resources:
  float_instances: [inst(operator), ...]    # each inst binds a fully-specified FloatHardwareOperator
  float_regfile: fmt + N float regs         # FF bank, multiport -- parallel reads are free
  float_consts: [fconst(value), ...]
  float_inputs: [input_load(name, dst_reg), ...]
  float_outputs: [output_wire(name, source, sign), ...]

scheduled float op:
  (inst, operands+sign_controls, dst_reg, issue_cycle)  # commits at issue_cycle + latency
makespan: the last commit cycle (the in_valid->out_valid latency is makespan + 1)
```

LIR exposes only a minimal API surface, following the design policies.
The top-level `Lir` fields are typed explicitly as `float_instances`, `float_regfile`, `float_inputs`, `float_ops`,
and `float_outputs` because the current machine has only float data resources; future bool/int resources should add
sibling fields instead of overloading these. LIR construction fails explicitly if selected MIR contains a non-float
domain before that domain has a register/constant/output resource family.

The `holoso._lir` package exports the LIR consumer contract: the LIR dataclasses backends need, plus `build()` and
`interface_of()`. Its private `_build.py` module orchestrates selected MIR validation, scheduling, binding, float
register allocation, constant-pool construction, and interface derivation. Its private `_schedule.py` module contains
the list-scheduling algorithm and schedule result type. Its private `_regalloc.py` module contains the current
float-register allocator.

- Reads are cheap (multiport FF), so binding is constrained only by operator-instance count and writes.
- Register allocation = liveness + phi-coalescing; widen `N` rather than spill at these sizes.
- `branch` is the real control transfer: the microprogram counter jumps, untaken ops never run, and the II is whatever
  the executed path costs (each path's count is itself exact).

## Operators

HIR semantic float operations are value instances of the HIR-local `Operator` hierarchy: `FloatAdd`, `FloatMul`,
`FloatDiv`, `FloatNeg`, `FloatAbs`, and strength-reduced `FloatMulPow2`. A HIR `Operation` is an occurrence of one
semantic operator applied to operand value IDs. There are no global semantic-operator singletons; frontend lowering and
HIR passes construct operator values ad hoc and distinguish operation kinds by class-pattern matching.

Concrete hardware operators live at the root in `holoso._operators` and are instances of the `HardwareOperator`
hierarchy. A concrete `HardwareOperator` is a frozen dataclass whose fields are its parameters. Float operators use the
`FloatHardwareOperator` subclass, which owns its `FloatFormat` and typed `evaluate(*float) -> float` reference semantics.
Each hardware operator owns its own timing, notation, concrete `ScalarSignature`, instantiation params, and
`instance_stem`: a lowercase Verilog-safe compact physical identity stem used for HDL names. The visible prefix is the
normalized mnemonic; the suffix is a hex stable hash of the canonical hardware parameters. For float
operators, that hash covers the float format and all sorted HDL params -- always including zero-valued stage params.
Examples look like `fadd_326215ea` or `fmul_ilog2_const_7296114c` rather than spelling every parameter into the name.

Generic, per-node-parameterized hardware operators are factories: a standalone `ParameterizedHardwareOperator`
carrying only its config-time knobs whose `instantiate(k)` returns a concrete `HardwareOperator`. The fully specified
hardware operator instance is itself the resource-sharing key (equal operators time-share one module).

Operators are chosen by a single `OpConfig`, constructed explicitly by the caller and passed into `synthesize`; today
its fields are all floating-point operators, and future bool/int operators should be added to the same config. Its
`float_format` property verifies the configured float operators agree and is used by HIR-to-MIR lowering. After that,
schedule construction derives the format from selected MIR instead of accepting a separate format argument. There is no
implicit default configuration.
Pipeline-stage knobs are named after the HDL parameters in lowercase, such as `stage_decode`, `stage_align`,
`stage_product`, `stage_input`, and `stage_output`.

## Scheduler

The private `holoso._lir._schedule` module implements software-pipelined (zero-bubble) list scheduling over selected
single-block MIR. Operators are fully pipelined (throughput 1) and their latencies are static and data-independent, so
the entire schedule is computed at compile time: each op is assigned an issue cycle and a bound instance, and the backend
just replays it with a cycle counter. The scheduler itself is domain-agnostic; LIR construction partitions the scheduled
result into the typed resource families that exist today. There is no global operator-class registry and no global
operator ordering: physical instance indices are local to one equal-by-value hardware operator, while HDL names use
`instance_stem` to distinguish different operator values of the same class.

The per-operator latency model is therefore exact and load-bearing: the backend commits each result on
`issue + op.latency` without watching `out_valid`, so each operator's `latency` property (mirroring its RTL
wrapper) must match the hardware cycle-for-cycle. An inaccurate latency is a *correctness* bug, not a bad estimate --
the consumer would read a stale register. The resulting cycle count is exact, never an estimate.

We issue each op on the earliest cycle its operands are ready and a free instance exists -- without waiting for
unrelated ops (no barrier), so a fast `fmul` no longer idles behind a co-scheduled `fdiv`. The register file is
read-first (`RWPASS=0`): a result written on cycle `c+L` lands in the flop on the next edge and is readable from
`c+L+1`, so a data-dependent consumer is held one cycle past the producer's latency.

```
for cycle = 1, 2, ...:                         # cycle 0 accepts/loads inputs; they read back from cycle 1
    ready = unscheduled ops whose every operator-operand has committed (issue + L + 1 <= cycle)
    for op in ready by critical_path desc:
        if an instance of op's concrete hardware operator is free this cycle (and ports permit):
            bind op to that instance; issue_cycle[op] = cycle
regalloc: linear scan over commit cycles; share a register when last_use <= def (sound under read-first); no spill
```

- Instances are pooled by the fully specified hardware operator itself (a frozen, equal-by-value `HardwareOperator`):
  ops that elaborate to the same hardware are equal and share instances. E.g., all `fadd`/`fmul`/`fdiv` of a given
  config are equal; `fmul_ilog2_const` differs by its exponent `K`, so same-`K` ops pool while different `K` are
  distinct modules. Each distinct operator value numbers its physical copies from zero, so different `K` values may both
  have instance index `0`; their `instance_stem` keeps emitted names unique.
  The configurable per-class budget (default 1) caps instances of each distinct operator value of that class: equal ops
  time-share, serializing when more than the budget would co-issue.

- Read/write ports are dedicated, so the controller word carries only addresses (no operand or write-data crossbar):
  one read port per operator operand (`nrd` = sum of instance arities) and one write port per operator instance
  (`nwr` = instance count), independent of I/O width. Inputs preload through the register file's immediate `load` port
  on step 0 instead of write ports. Outputs are tapped from the register file's passive `view` bus by fixed register
  index instead of read ports.

- The `load` port folds into each low register's write-data OR (one masked term, no address comparator), so a
  single-cycle preload of registers `0..nload-1` costs far less than the per-register comparators that one write
  port per input would add. `nload` spans the input block (the highest input register index plus one).

- Selected-float MIR `FloatSignControl` value objects fold into operand/result sign-control sidebands, and `fconst` is an
  immediate on the input mux; both are free in the schedule. Sign controls are constructed ad hoc rather than
  represented by global sentinel instances.

Why not write-through forwarding? A write-through register file (`RWPASS=1`) would erase the +1 dependency cycle, but
its forwarding muxes cost O(NRD*NWR) and we need many ports -- unsustainable. Read-first plus the +1, hidden under
pipelined overlap, is the better trade.

## Backend (ZISC)

Mechanical from LIR: a `holoso_regfile` flop bank, one operator instance per `FloatOperatorInstance`, and one continuous
assignment per pooled constant -- its ZKF bit pattern precomputed in Python by `FloatFormat.encode`. The controller is a
microcode ROM: one pre-decoded VLIW control word per step, stored in a (BRAM-inferable) ROM read through two cascaded
registers, so the second packs into the BRAM's dedicated output register (DP16KD OUTREG / Xilinx DO*_REG) -- a fast
clock-to-out instead of the slow array-access clock-to-out. Registering the word is the point: it splits what used to
be one wide combinational `case(cyc)` cone (`cyc -> read-address mux -> regfile read -> operand mux -> operator`) into
short register-to-register paths (`pc -> ROM -> ucode_q -> ucode_word` and `ucode_word -> datapath`). The two-stage
fetch costs +1 cycle of read latency, so the executing step lags the fetch `pc` by one: `pc` runs `0..makespan+2` and
`out_valid` is asserted at `makespan+2`. Under fully static scheduling that cycle is essentially free (it only adds one
to the makespan/II). The extra stage mainly helps tools that infer BRAM with a no-output-register read (e.g.
Yosys+nextpnr); flows that already register the control store in fabric (e.g. Diamond) do not need it -- a future
per-target build knob should let those flows drop the stage and reclaim the cycle.

The schedule is replayed step by step: `pc==0` accepts and parallel-loads the inputs through the register file's `load`
port (registers `0..nload-1` in one cycle, gated by `in_valid`); `pc` advances every clock through the compute steps
`1..makespan`; and `pc==makespan+2` asserts `out_valid` (the executing step lags `pc` by one) while the outputs are
driven combinationally from the register file's `view` bus (wired by fixed register index). The PC holds only at these
two I/O boundaries; bubble steps with no
issue/commit carry an explicit NOP word and the PC keeps advancing -- trading a little ROM for a trivial, fast sequencer.
No scoreboard is needed because latencies are static.

The control word stores selectors and addresses, never data: each operator operand has a dedicated register-file read
port (the word carries only its address, so there is no operand crossbar), and each operator instance has a dedicated
write port (its result wires straight in; the word carries the write enable and destination). Constant operands keep
using the `const_<i>` immediate wires through a small per-operand select. A control field that is constant across the
whole program -- very common for sign controls -- is driven by a constant net and lifted out of the ROM, so synthesis
prunes the logic it feeds (e.g. an unused sign conditioner); the Python packer that builds the ROM and the bit-slice
offsets the module reads are produced together, so they cannot drift.

Errors are non-fatal and informative: `err` ORs each error-bearing operator's flag gated by that instance's write-enable
(the step it commits), and the control block  latches `err_pc <= pc` whenever `err`, so `err_pc` is 0 if the run hit
no errors (reset at every accept; `|err_pc` means "any error"), else the last step one occurred.

Reset covers only the control registers (`pc`, `err_pc`); the ROM word register follows `pc` (`next_pc==0` under
reset). The control word and datapath skeleton are the only ZISC-specific part -- LIR itself is controller-agnostic.

Each operator instance carries its own parameters and float format, fixed at construction from the `OpConfig` threaded
through `synthesize`. The wrappers' instantiation params come from `operator.hdl_params()` and the scheduler's timing
from `operator.latency` -- both read the same hardware operator instance, so the emitted RTL and the static schedule
cannot drift. `hdl_params()` always lists every parameter explicitly (including zero-valued `STAGE_*`), so the
instantiation is self-describing, survives changes to wrapper defaults, and turns a param-name mismatch into a loud
elaboration error. The wrapper instance name is `u_{operator.instance_stem}_{index}`, where `index` is local to that
concrete operator value; the stem uses a short stable hash of the operator's canonical hardware parameters.

Constant operands are kept as immediates on the input mux. Two alternatives are noted in the backend for when this turns
into a constraint: folding constants into the register file (preloaded like inputs, so every operand is a uniform
register read), or emitting explicit constant-load micro-instructions (uniform operand path with lighter register
pressure, at the cost of scheduling/allocation complexity).

## Decisions

1. Phi merges are resolved by register coalescing, not materialized selects.
2. Split float and bool register banks.
3. If-conversion is conservative -- trivial pure diamonds only; real branches otherwise.
4. SymPy-assisted algebra (fold/CSE/simplify); simple HIR strength reduction in-house.
5. The per-operator latency model is exact and must match the RTL wrappers cycle-for-cycle: the static schedule commits
   each result on `issue + latency` without watching `out_valid`, so an inaccurate latency is a correctness bug, not a
   bad estimate. (Module I/O still uses a valid/ready handshake; `div0` is the only data-dependent runtime signal.)
6. Software-pipelined (zero-bubble) static list scheduling over a read-first register file; the controller is a
   microcode ROM replaying the schedule step by step (no runtime scoreboard), since v0 operator latencies are
   data-independent.
7. API takes the function/class object (not source files); synthesis is in-memory, returning `SynthesisResult`; disk
   I/O is an opt-in helper.

## Example (`iir1_lpf`): state + branch + coalescing

```
entry:  f = state_read(first); y_in = state_read(y)
        branch(f, b_init, b_run)
b_init: ya = in_a;                            jump(exit)        # y = x
b_run:  d  = sub(in_a, y_in)                                    # x - y
        m  = fmul_ilog2_const(d, -16)                           # 2^-16 * (x - y)
        yb = add(y_in, m);                    jump(exit)
exit:   y_out = phi[(b_init, ya), (b_run, yb)]
        state_write(y, y_out); state_write(first, const false)
        out_port(out_0, y_out); ret
```

`ya`/`yb` coalesce to the `y` register; the `phi` is free; only one arm runs; `first` resets to True; `y` is unreset.

## First delivery (v0)

Minimal end-to-end slice -- front-end -> HIR -> MIR -> scheduler -> LIR -> Verilog + testbench + report + model --
on a single basic block: combinational, scalar-only, operators `fadd`/`fmul`/`fdiv`/`fmul_ilog2_const` plus semantic
`FloatNeg`/`FloatAbs` folding and `FloatConst`
(`fdiv` and its wrapper already exist in ZKF). No state, control flow, arrays, or bools (`M = 0`); intrinsics
(`sqrt`, `sincos`, ...) raise the "implement this operator" error, pending ZKF support. State, branches, and arrays
follow in later milestones.

## Deferred

Operator-pool auto-sizing, optional ILP mode, dynamic-trip loops, second controller backend, FloPoCo backend,
intrinsics (`sqrt`, `sincos`, `exp`, ... -- pending ZKF support).

A fused multiply-add operator wrapping kulibin `zkf_fma` is planned: a `holoso_ffma` wrapper plus an `FFmaOperator`
(with the selection pass fusing `a*b + c` chains). It should cut operator count, register pressure, and latency, and
help timing closure -- to be added later.
