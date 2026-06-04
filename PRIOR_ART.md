# Prior work on Python-to-HDL synthesis

We evaluated all major existing tools for high-level circuit synthesis from Python source using several representative
kernels (stored in the `examples/` directory) and scored them against Holoso's target profile: configurable-precision
floating point stateful FSM modules with a simple low-level valid/ready interface,
generated from near-unmodified Python, ideally portable between various target chips and process flows.

The reader is assumed to understand the main goals of Holoso to get the full picture.
Contributions to this document are welcome.

## Comparison matrix

| Trait                           | Polyphony                   | PyLog                    | Allo+Vitis                 | Allo+XLS                            | Veriloggen                   | Bambu (C)                          |
| ---                             | ---                         | ---                      | ---                        | ---                                 | ---                          | ---                                |
| Verilog interface/integration   | ✅ scalar ports + FSM        | ❌ AXI master + AXI-Lite  | ⚠️ `ap_ctrl` + BRAM/scalar | ✅ plain Verilog                     | ✅ scalar ports + valid       | ✅ start/done + scalar              |
| Setup                           | ✅ Python → Verilog          | ❌ needs Vitis HLS        | ❌ ~41 GB MLIR + Vitis HLS  | ❌ ~41 GB MLIR + XLS + build `xlscc` | ✅ `pip install` to Verilog   | ✅ 1.4 GB AppImage (self-contained) |
| Portability (vendor-neutrality) | ✅ plain Verilog             | ❌ Xilinx/Vitis           | ❌ Xilinx/Vitis             | ✅ vendor-neutral                    | ✅ vendor-neutral             | ✅ vendor-neutral                   |
| Float (arbitrary-precision)     | ❌ no float                  | ⚠️ `float`/`double` only | ⚠️ limited                 | ❌ none — int/FixP only              | ❌ none — int/FixP only       | ✅ `--fp-format` e/m configurable   |
| Stateful modules                | ✅ `@module` + workers       | ❌ function-only          | ✅ `@Stateful`              | ❌ xlscc rejects statefulness        | ✅ FSM regs (recurrent)       | ✅ `static` (recurrent)             |
| Automatic scheduling/pipelining | ❌ none — combinational blob | ✅ via Vitis              | ✅ via Vitis                | ✅ via XLS                           | ❌ control-flow FSM only      | ✅ SDC, latency-aware               |
| Math operators                  | ❌ poor (integer only)       | ✅ solid                  | ✅ solid                    | ⚠️ int/fixed only (no float)        | ⚠️ int/fixed only (no float) | ✅ solid (float/int/fixed)          |
| Code-change extent              | ❌ large                     | ✅ minimal (array I/O)    | ⚠️ type annotations        | ❌ annotations + no float            | ❌ HW-idiom heavy rewrite     | ❌ C, not Python                    |
| Verification                    | ⚠️ RTL sim only             | ❌ none built in          | ✅ bit-exact CPU JIT        | ✅ bit-exact CPU JIT                 | ✅ built-in iverilog sim      | ✅ C/RTL co-simulation              |

Polyphony is vendor-free and end-to-end (does not depend on external frameworks like XLS or Vitis) and offers a
clean Verilog interface, but is integer-only with zero datapath scheduling.
The lack of scheduling/pipelining renders it effectively unusable beyond trivial examples.

PyLog and Allo+Vitis support float math with near-zero code changes, and real timing-driven scheduling,
but only by handing the C++-to-RTL step to Vitis HLS, which brings AXI/BRAM interfaces
(hard to couple with external logic), a heavy vendor toolchain, and Xilinx vendor lock-in.

Allo+Vitis supports f16/bf16/f32/f64; Vitis HLS has an `ap_float<W,E>` (arbitrary exp/significand),
but it is not exposed by either Python frontend, is device-gated (UltraScale+/Versal only, not 7-series),
and Allo's C++ emitter truncates float constants (e.g., 2^-16 comes out as 0.000015, ~1.7 % off).
So custom float is reachable only by hand-editing generated C++ and only on a supported part.

Allo+XLS — xlscc rejects floating point entirely (int/FixP only).
This is the only functional vendor-neutral path, but stateless and integer-only.

Veriloggen compiles imperative Python (using its "threads") straight to a vendor-neutral Verilog FSM,
with thread-local variables becoming persistent registers.
The stateful IIR therefore synthesizes with genuine recurrent feedback and a clean clk/rst+scalars+valid interface,
simulated and verified without any vendor tools (it ships an Icarus-Verilog harness).
Its critical gaps are the lack of floating point and that its scheduling is control-flow-level —
each statement's datapath is emitted combinationally inside one FSM state,
with no operator-latency-aware scheduling or pipelining
(so multi-cycle float operators would not be scheduled across cycles automatically;
in particular it is unclear how division would work, for example).
One also writes in Veriloggen's hardware idiom that is very far from the original model code.

Bambu (PandA) is C-based, not Python, but it is by far the strongest contender.
It is an open-source, fully self-contained HLS (a single AppImage; no vendor tools) that compiles C to
vendor-neutral Verilog with real operator-latency-aware (SDC) scheduling.
The stateful IIR example (below) maps one-to-one from the Python class —
`static` locals become genuine persistent registers —
and synthesizes to a clean clock/reset+start/done+scalar interface (no AXI etc).
Option `--fp-format=<fn>*e<exp>m<frac>b<bias>...` provides arbitrary-precision floating point with
selectable rounding/exception/specialization modes (the last could even drop NaN/subnormals).
What Bambu does not offer is a Python frontend.

## Single-pole IIR example

Take a stateful single-pole IIR low-pass. Original Python, as written:

```python
class IIR1LPF:
    def __init__(self, *, ALPHA: float = 2**-16):
        self.alpha = ALPHA; self.y = 0.0; self._first = True
    def __call__(self, x: float) -> float:
        if self._first:
            self._first = False
            self.y = x
        else:
            self.y += self.alpha * (x - self.y)
        return self.y
```

Polyphony won't take it at all: no float type, requires heavy code rework.

PyLog has no per-call state and no class support, so the filter must be re-shaped into a block over an array
(statefulness is lost; alpha becomes an inline constant):

```python
@pylog(mode='cgen')
def iir1lpf(x, y):                       # whole block; "state" is just a loop variable
    acc = x[0]; y[0] = acc
    for i in range(1, 1024):
        acc = acc + 1.52587890625e-05 * (x[i] - acc)
        y[i] = acc
    return 0
```

...and it generates an AXI master kernel — **~107 (sic!) bus signals**, unusable in hand-crafted RTL:

```verilog
module iir1lpf(ap_clk, ap_rst_n,
   m_axi_data0_AWVALID, m_axi_data0_AWADDR, m_axi_data0_WDATA, m_axi_data0_RDATA, /* ... x[] */
   m_axi_data1_AWADDR,  m_axi_data1_WDATA,  /* ... y[] */
   s_axi_control_AWADDR, s_axi_control_WDATA, /* ... */ );          // ~107 AXI ports
```

Allo+Vitis: the `@Stateful` annotation keeps the per-sample, stateful shape (close to the original),
and emits a clean `ap_ctrl`+scalar interface resembling the conventional valid(/ready) semantics:

```python
def iir_step(x: float32) -> float32:
    y:     float32 @ Stateful = 0.0          # persists across calls (a true state register)
    first: int32   @ Stateful = 1
    if first == 1: first = 0; y = x
    else:          y = y + 1.52587890625e-05 * (x - y)
    return y
```
```verilog
module iir_step(input ap_clk, ap_rst, ap_start, output ap_done, ap_idle, ap_ready,
                input  [31:0] v0,                          // x
                output [31:0] v1, output v1_ap_vld);       // y + valid
```

The elephant in the room is the `@Stateful` notation presented as a type annotation obviously renders the code
incompatible with direct execution in Python, which implies that one cannot simply translate an existing numerical
Python model into hardware-ready RTL code, requiring a manual rewrite, which is error-prone and increases the
verification effort and the iteration time.

Allo+XLS (integer/fixed only): the same IIR, after manual adaptation to fixed-point math,
run through xlscc on Allo's own emitted C++, yields clean Verilog:

```verilog
module iir_step(input [31:0] v0, v1, output [31:0] out);   // v0=x, v1=y_prev
  // out = y_prev + ((x - y_prev) >>> 16)   (XLS emits the sign-extended bit-slice)
  assign out = v1 + {{15{sub[32]}}, sub[32:16]};   // sub = sext(x) - sext(y_prev)
endmodule
```

...which is not useful as-is because it supports neither statefulness nor floating point math.

Veriloggen (integer/fixed only): the IIR is written as a "thread"; the thread-local `acc`/`first` become persistent
registers (recurrent state) and the generated interface is very clean (no AXI):

```python
def filt():
    acc = 0; first = 1
    while True:
        if en:
            if first: acc = x; first = 0
            else:     acc = acc + ((x - acc) >> 16)   # recurrent feedback update
            y.value = acc; valid.value = 1
        else:
            valid.value = 0
vthread.Thread(m, 'th_iir', clk, rst, filt).start()
```
```verilog
module iir1lpf(input CLK, RST, input signed [31:0] x, input en,
               output reg signed [31:0] y, output reg valid);
  // persistent state register, updated by the feedback recurrence (signed arithmetic shift):
  //   _th_iir_acc_0 <= _th_iir_acc_0 + (x - _th_iir_acc_0 >>> 16);
```

Its only shortfalls are floating point, operator-latency-aware scheduling, and the need for a heavy rewrite.

Bambu (C, not Python): the class maps one-to-one to C with `static` state;
Bambu keeps it stateful and emits a clean, AXI-free interface:

```c
float iir_step(float x) {              // synth with --fp-format=iir_step*e8m35b127nih for custom float
    static float y = 0.0f;             // persistent recurrent state
    static int first = 1;
    if (first) { first = 0; y = x; }
    else { y = y + 1.52587890625e-05f * (x - y); }   // alpha = 2**-16
    return y;
}
```
```verilog
module iir_step(clock, reset, start_port, x, done_port, return_port);  // start/done + scalar; no AXI/mem
```

## Verdict

Bambu is the strongest option that delivers stateful, arbitrary-precision floating point, vendor-neutral Verilog,
real operator-latency-aware scheduling, and a clean Verilog module interface.
Its main downside is that it only supports C as the source input.

## Non-alternnatives

Python HDLs such as [MyHDL](https://www.myhdl.org/), [Amaranth](https://github.com/amaranth-lang/amaranth),
[Migen](https://m-labs.hk/migen/), [PyRTL](https://ucsbarchlab.github.io/PyRTL/), et al
describe RTL in Python rather than synthesizing it from algorithmic code --
completely different tools for a different task, mentioned only for completeness.
