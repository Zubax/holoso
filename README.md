<div align="center">

<img src="docs/holoso-logo-128.png" width="60px">

<h1>Holoso</h1>

_Simple high-level synthesis of portable Verilog from idiomatic Python_

[![Try online](https://img.shields.io/badge/try_online-holoso.digital-black?color=ff0000)](https://holoso.digital/)
[![Forum](https://img.shields.io/discourse/https/forum.zubax.com/users.svg?logo=discourse&color=ff0000)](https://forum.zubax.com)

</div>

-----

Holoso converts a subset of idiomatic Python into synthesizable and verifiable Verilog.
It is primarily designed for heavy numerical code which is abundant in control systems and DSP
where manual RTL coding is inefficient and error-prone.

Holoso focuses on Python because this is a popular language in modeling, system design, and verification domains;
ability to generate production HDL directly from the original model allows the designer to work with much simpler
harnesses and iterate faster.

See [PRIOR_ART.md](PRIOR_ART.md) for a detailed review of existing alternatives,
and why none are good enough for practical use.

Holoso is under active development and as such it has no burden of backward compatibility.
Breaking changes will occur regularly without notice until v1.0 is out.
Many critical features are missing which may limit applicability beyond applications that we are immediately involved with.
Contributions of any kind are emphatically welcome!

<img src="docs/hero.png" width="900px">

## Design

Holoso implements essentially a separate programming language whose syntax is a strict subset of Python,
and whose semantics is largely equivalent to Python with minor deviations that make sense in chip design context.
Save for the minor differences in semantics, Holoso ensures that one can execute the original Python code
and run the generated circuit (RTL) side by side, and obtain equivalent results (bit-exact unless floating points
are used, in which case small errors may creep up, due to the inherent limitations of floating points).

Unlike most (all known to us) HLS engines out there, Holoso does not generate a straight-line II=1 pipeline
because this is rarely what you actually need in practice; instead, it designs a narrowly specialized
computing core (a zero-instruction-set processor) with custom microcode, and statically schedules a program for the
designed core. Being in control of both the core synthesis and the program compilation, Holoso tends to generate
extremely efficient designs in terms of cycle latency and chip area utilization compared to the state of the art.

Holoso outputs a purely portable and vendor-agnostic Verilog that can be fed into thid-party synthesis tools as-is,
along with its support library implementing various arithmetic operators. So far it has been tested at least with
Yosys (ECP5), Diamond (ECP5), and Vivado (Artix-7).

By default, Holoso is tuned for the minimum cycle latency and minimum $f_\max$.
If timing closure fails, one needs to locate the critical path and enable the staging knob that inserts a
register stage into the offending path; then re-synthesize and repeat until timings close.

Holoso has its own efficient floating point engine that is a subset of IEEE-754, omitting support for subnormals and NaN.
Arbitrary exponent and significand bit widths are supported (the IEEE-754 defaults map poorly onto FPGA DSP tiles).

Along with the synthesized Verilog, Holoso produces a Cocotb co-simulation testbench and a detailed and beautiful HTML
report that provides a human-friendly view of the processor and the microcode sequence constructed by the synthesizer.

>*You can SEE the pipeline — every cycle, every landing, every spill. It's gorgeous. People love it.*
>*They come up to me with tears in their eyes, they say sir, that schedule report, it's the most beautiful report we have ever seen.*

For a detailed review of the design and trade-offs, please refer to `DESIGN.md`.

### Semantics

Holoso follows Python with minimal deviations where it makes sense for hardware synthesis.

- Static typing only.
- No implicit type conversions.
- Boolean short-circuiting is not supported, all operands evaluated eagerly.
- Floating-point precision depends on the selected floating-point format. See below for more info about floats.
- No exceptions: division by zero, domain errors, etc. produce the closest meaningful result and assert the error flag.

### Floating point

The floating point engine is based on [Zubax Kulibin Float (ZKF)](https://github.com/Zubax/kulibin).

Differences from IEEE 754: no NaN, no subnormals (exponent 0 always encodes +0; finite magnitudes in `(0, min_normal/2)`
round to +0; magnitudes in `[min_normal/2, min_normal)` round to signed min_normal), no exceptions, overflow produces ±∞.
Canonical representations do not include negative zero (it is not an error to pass negative zero though).

Floating-point optimizations are fast-math style, assuming commutativity and associativity,
allowing non-bit-exact rewrites.

Infinity cases that would be NaN in IEEE 754:

| Expression          | Result                         |
|---------------------|--------------------------------|
| +∞ + −∞             | +0                             |
| 0⋅±∞                | +0                             |
| 0 ÷ 0               | +0                             |
| ±∞ ÷ ±∞             | +0                             |

Non-NaN infinity cases (same intent as IEEE 754):

| Expression          | Result                         |
|---------------------|--------------------------------|
| finite≠0 ÷ 0        | ±∞  (sign = sign of dividend)  |
| ±∞ ÷ 0              | ±∞  (sign = sign of dividend)  |
| finite ÷ ±∞         | +0                             |
| ±∞⋅±∞               | ±∞  (sign = signs XOR)         |
| finite≠0⋅±∞         | ±∞  (sign = signs XOR)         |

WEXP can be chosen freely depending on the required range, while WMAN is sensitive to the chip's DSP capabilities
and thus requires careful selection to achieve best resource utilization.

| WMAN | ≈ε (interval) | Description                                                                         |
|------|---------------|-------------------------------------------------------------------------------------|
| 16   | 3.052e-05     | DSP tiles in Lattice iCE40 and similar                                              |
| 18   | 7.629e-06     | Classic FPGA DSP width, very common: ECP5, PolarFire, Trion, many Intel modes, etc. |
| 24   | 1.192e-07     | IEEE 754 binary32; also fits Versal DSP58's 27x24 asymmetric multiplier side        |
| 27   | 1.490e-08     | Intel/Altera variable-precision DSPs                                                |
| 32   | 4.657e-10     | 2x16                                                                                |
| 36   | 2.910e-11     | 2x18 (very common) or native Intel/Altera 36x36-style variable-precision mode       |
| 48   | 7.105e-15     | 2x24 or 3x16; with an 8-bit exponent amounts to 7 bytes exactly                     |
| 53   | 2.220e-16     | IEEE 754 binary64                                                                   |

Narrower WMAN is rarely practical for computation due to low precision and fast error accumulation,
although they can still be useful for storage/exchange. One notable exception is neural networks though.

## Usage

Unlike most tools in this domain, Holoso is trivial to set up and get started with; it is not a framework.

Install: `pip install holoso`.

```python
# Select the floating-point format you wish to use.
# Ideally, wman (mantissa width) should be a multiple of DSP tile operand width.
float_format = holoso.FloatFormat(wexp=6, wman=18)

# Define the numerical operators. This is where you can configure additional stages to close timings.
ops = holoso.OpConfig(
    holoso.FAddOperator(float_format),
    holoso.FMulOperator(float_format),
    holoso.FDivOperator(float_format),
    holoso.FMulILog2OperatorFamily(float_format),
    holoso.FCmpOperator(float_format),
)

# Run Holoso -- construct the processor and the microcode.
# The results are returned in-memory; you can write them to disk where you want.
# They include the generated Verilog module, the fixed holoso_support.v/.vh, testbench, and the reports.
result = holoso.synthesize(your_function_or_method_here)

# Write the files -- this is usually what you want.
out = result.write(Path(__file__).resolve().parent)

# Show what's been written.
for filename, path in out.items():
    print(f"{filename}: {path}")
```

See the `examples/` directory for self-contained usage examples.

## Verification

Just say `nox`. Read the `noxfile.py` and `DESIGN.md` for details.

You may find the [zubax-fpga-toolchain](https://github.com/Zubax/fpga-toolchain-docker/pkgs/container/zubax-fpga-toolchain)
container useful as it comes with all of the required tools out of the box.
