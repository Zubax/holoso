# Holoso -- simple high-level synthesis of numerical RTL modules

This is a Python library that provides a simple way of synthesizing numerical Verilog FSM modules for applications involving control systems and DSP from high-level Python models. The idea is that the user would model/verify the system in Python, and then use Python code very similar to the original model (if not the same) to generate the final production-ready RTL.

Holoso uses a single floating point data type for all scalars with configurable exponent and significand width (WEXP and WMAN), plus booleans (mostly for flow control purposes). The total floating point scalar bit width is referred to as WFULL=WEXP+WMAN (the sign bit is added, the most significant bit of the significand is hidden). The floatint point format used by Holoso does not necessarily have to be standards-compliant; one option is the Zubax Kulibin float (ZKF), which is IEEE 754-like but without NaN, subnormals, or signed zero, which results in a more efficient FPGA resource usage with marginal loss of utility.

The generated Verilog is an FSM implementing a kind of zero-instruction-set computer. The generated module instantiates basic arithmetic and logic operator modules that it needs to evaluate the original function, a multi-port register file with the minimum required number of registers (or two register files: one for floats (WFULL-wide words), one for booleans (1-bit words), remains to be seen), and defines a narrowly specialized custom scheduler/FSM circuit that connects each arithmetic/logic operator module to the correct register at the correct time. We are expressly NOT building a more traditional zero-bubble pipelined circuit.

Given an exemplar Python function like this:

```python
def norm2(a, b):
    return np.sqrt(a**2 + b**2)
```

Holoso will generate a Verilog module with an interface like this, assuming that it was configured with WEXP=8 WMAN=36
(so 44 bits total per scalar):

```verilog
module norm2(
    input wire clk,
    input wire rst,

    // Inputs.
    // The inputs are sampled once when in_valid is asserted, and are not required to persist otherwise.
    input  wire        in_valid,    // pulsed by the user to commence computation
    output wire        in_ready,    // held high when idle and ready to commence new computation; low when busy
    input  wire [43:0] in_a,
    input  wire [43:0] in_b,

    // Outputs. This function has only a single output.
    // The output values are undefined (may change arbitrarily at runtime) unless out_valid is set.
    output wire        out_valid,   // pulsed by the module when results are ready
    input  wire        out_ready,   // output backpressure; held low to delay out_valid if not ready to accept results
    output wire [43:0] out_0,       // out_1, out_2, ... depending on the implemented function.

    // Diagnostics (optional).
    // The diag_error will be nonzero in a cycle when a numerical error has occurred (like division by zero,
    // domain error, etc). Computation still continues regardless of errors.
    // This interface requires revision, suggestions welcome.
    output wire [2:0] diag_error   // sized per maximum number of possible errors
);
```

See more representative examples with further details under the `examples/` directory.

The only native data types are floating-point scalars of a fixed format defined at generation time and booleans.

A restricted set of operations is targeted:

- Basic arithmetic operators: addition, multiplication, division, etc.
- Sign manipulation (performed via trivial combinational circuits, not dedicated operator modules): negation/absolute. This enables subtraction via addition.
- Relational operators.
- Conditional branching and loops.
- Some logic operators that can be sensibly mapped to reals; e.g., `x<<y` and `x>>y` are treated as `x*(2**y)` and `x*(2**-y)`, respectively; zero is boolean false and nonzero is boolean true; etc.
- Simple algebra on booleans, and also conversion between booleans and floats if needed.

The generated Verilog ships with a small fixed support file `holoso_support.v`/`.vh` containing the primitives that may be instantiated by the generated module. The same support file may be shared by multiple generated modules in the project.

Standard numerical operators such as abs, trigonometry, isinf, etc. are inferred from predefined forms. For Python, we must support the builtin `abs()`, the NumPy alternative, etc. Initially, the holoso_support module ships only a very limited set of basic operators, so we will start with only the basics, but eventually we will add support for more operators (sqrt, exp, log, sin/cos are the first order targets). Until they are in place, synthesis should fail with an explicit error suggesting implementing the missing operators.

Contrary to a conventional pipelined module generation, Holoso essentially operates in the compiler problem domain.

- The front-end analyzes the input source code (at the moment we only support Python, but in the future other front-ends will be added, in particular SymPy with CSE) and generates an internal AST.

- Initial optimization passes, CSE, constant folding, etc. are performed while lowering AST into the next-level representation. At some stage around this point, multiplication/division by a power of 2 must be identified (e.g., like the `x<<y` --> `x*(2**y)` mapping above, or an explicit `x*16` literal form) because those can be represented using extremely efficient `holoso_fmul_ilog2_const` instead of the true general `holoso_fmul`/`fdiv`, saving many cycles and fabric area. Where possible, division must be replaced with multiplication (true float divider takes an enormous amount of fabric area and is much slower than any other basic operator).

- The set of constants is determined (1.0, pi, etc) and `holoso_fconst` instantiations are produced. Constants can be treated as special read-only registers; better ideas are welcome.

- The set of operator modules needed is determined (`holoso_fmul`, etc.), along with the cycle latency of each (depends on their instantiation parameters, such as the floating point exponent/significand width).

- Optimal computation schedule is defined per FSM step that maximizes the operator module utilization and reduces the total FSM step count.

- The final circuit is generated and converted into Verilog.

The source function may return either a scalar (the simplest case), a matrix/vector represented as nested lists/tuples (later on we will also add support for NumPy scalars but their dimensions obviously will have to be statically inferrable), or a dataclass. Ideally, arbitrary nesting should be allowed. Matrices n by m map to outputs like `out_0_0`...`out_<n>_<m>`, etc. Dataclass fields are mapped according to their names prefixed with `out_`, nesting levels are flattened so the following:

```python
@dataclass
class Foo:
    bar: float

@dataclass
class Baz:
    foo: Foo

def my_function(...) -> tuple[Baz, float]: ...
```

Translates into `out_0_foo_bar, out_1`.

The output ports are connected (assigned) directly to the corresponding registers in the register file so that we don't have to waste cycles transferring the results. This implies that outputs will change unpredictably while computation is in progress, which is fine because we don't guarantee output stability until the computation is done.

If we later add a C front-end (can't imagine why at this point but for the sake of illustration), dataclasses would be mapped to structs.

There is a possibility to replace the traditional FSM state / instruction pointer with a shift register selecting which state to activate; whether this makes sense remains to be seen.

It is desirable to maximize each operator module utilization, but if we commence only one operation per engaged operator per cycle, utilization will drop when a fast and a slow operators are involved in a cycle (e.g., multiplication is very fast but division is very slow; engaging both simultaneously will drop the multiplication throughput to that of the divider). This may be acceptable for the first revision, but ideally we should aim to structure the circuit where within one cycle, completion of one fast operation may trigger the next operation within the same cycle while the slower one is still in progress. Correct implementation provided, this may enable sequences like (multiply r1 r2, store result into r1, immediately multiply r1 r3, etc) running within one cycle.

It would be interesting to consider, perhaps as an option, the use of conventional function minimization algorithms (e.g., annealing) for the search of the optimal scheduler. I'm not sure if it makes sense at this stage though.

Note on the reset policy: routing reset to every register is known to adversely affect timings. Thus, only the control registers are reset (such as the `stage` etc), while the register file is left uninitialized.

Key parameters defined per module: number of cycles from `in_ready` to `out_ready`, which is also the *initiation interval* (II) (the equivalence exists only for FSM modules but not pipelined ones; ours is an FSM). This may not be a constant beause it depends on the instantiation parameters of the operator modules (e.g., number of stages in the multiplier, etc), so the generator should produce a model that is a function of input-output cycle latency of all involved operators.

It is important to provide an opportunity for the user to review the structure of the generated FSM. Due to its construction, doing so by just looking at the generated Verilog will not be practical because of the heavy functional transformations involved. The solution is to make the compiler generate a rich, human-friendly, colorful single-page HTML report describing the structure of the generated FSM. The report will contain a table, where rows correspond to the scheduler states (so there will be `N_STAGES` rows), the height of each is proportional to its expected cycle count; and columns correspond to registers. Each stage modifies the register file, which is reflected in the table in the space between the input/output register file stage; affected cells are color-coded (e.g., r3 and r47 are input to `holoso_fmul` with signops such and such, the result is stored into r31, shown with arrows or colors or whatever makes sense). To the right of the table we place a vertical swimline diagram with one line per operator module; the color/shape of the line indicates which stages engage the corresponding operator module (the length of the indication should be normalized to the total expected stage duration; e.g., division takes a very long time, so if it happens concurrently with multiplication, then the division line will cover the whole stage height while the multiplication will only cover a small fraction of it, illustrating the underutilization of the multiplier).

That said, the above is just one possible approach. We could also consider a more direct approach to building FSM that does not attempt to mimic a NISC/ZISC processor. In this approach we encode the state machine state roughly similar to an instruction pointer, and define a huge case statement that directly engages each operator module and waits for completion. If implemented directly, however, we will face the same issue of module underutilization when a fast and a slow operators are engaged simultaneously and the state is not advanced until all are done. One of the possible approaches is to allow asynchronous operations, where state can be advanced while some of the operators are still running. Alternatively, the problem could be inverted by introducing sub-states that split a full state into a number of smaller micro-states, where each schedules a complete operator module instruction and asynchrony is confined to a single full state (but not micro-states). I am not sure which approach makes more sense here.

If we go with the direct FSM construction approach, then the natural visualization format is just an ordinary state transition graph diagram, augmented with the register file state transitions.

One important matter is the interoperability/composability with other code, whether Python or Verilog. The simplest way to leverage a synthesized module is to just manually add it into the final Verilog design, but these manual activities should ideally be minimized. The problem has two sides:

- Invocation of Python code from Python.
- Invocation of Verilog modules from Python.

The first one is straightforward: when a Python source code invokes a Python function (or a callable class) that is visible to the synthesizer, the invocation target is simply inlined (if it's a class, its state is merged with the state of the invoker), so that at the end we obtain a single module.

The second one requires some analogy of a *calling convention* for Verilog modules. Obviously modules generated by Holoso should be compatible with each other. We know that all modules have a clearly defined compact set of standard signals: `clk, rst, in_valid, in_ready, out_valid, out_ready`,  plus `in_*`/`out_*` signals whose names are directly derived from Python names. This implies that Holoso does not necessarily have to see the invoked module in order to generate its valid instantiation in the generated Verilog, as long as the module follows the convention.

Crucially, the second option allows one to plug hand-written Verilog modules into Python-generated modules by merely providing Python prototypes; e.g. (this is based on an example from the `examples/` directory):

```
class TrapezoidalLeakyStreamingIntegrator:
    def __init__(self, *, K: float) -> None: ... # K translates into real parameter `.K(...)` at instantiation
    def __call__(self, x: float, /) -> float: ... # Parameter names are well-defined!
```

That said, the second composition mode -- *invocation of Verilog modules from Python* -- is not the primary design objective by any margin, so it can be safely shelved until further notice. This is definitely not something that will make it into the eventual v1.0.

Generated modules must be accompanied by an extensive Cocotb testbench. Not sure yet how to construct the verification suite itself; one idea is to inject a large number of random vectors and compare against the original, but we're working with floats that are imprecise by definition, and when the implementation float uses a different format compared to Python, we obviously cannot expect exact matching, and in degenerate cases the divergence may be huge while not being indicative of an actual problem that would affect the application. So it is possible that we may need to perform some static analysis and derive sensible test vectors and interval bounds from that. The objective is explicitly NOT to verify numerical accuracy (that is up to the user) but rather to verify the logical equivalence between the original Python code and the generated FSM.

Aside from the verification suite, we should also produce some minimal scaffolding for out-of-context (OOC) synthesis in Yosys against specified target platforms. That part will be bolted-on at a later stage and so right now it does not require much attention.

There exists a number of interesting projects around the Python HDL/digital design ecosystem; this includes PyCDE (a Python adapter for CIRCT), Veriloggen, and a few others. Those may be leveraged to avoid reinventing the wheel where possible.

Bambu could be looked at as an inspiration for FSM construction -- see `/mnt/storage/test/hls-experiments/work/bambu/`. Bambu by itself is a very strong option already, its main limitation is that it only accepts C as the source input language, not Python. Perhaps we could consider transpilation from Python (relevant subset thereof) into C, then Bambu picks it up from there, but frankly this is likely to get unwieldy quickly.
