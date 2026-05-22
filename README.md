# Holoso -- simple high-level synthesis of Python into Verilog for numerical code

Holoso converts a small subset of Python functions and expressions into synthesizable and verifiable Verilog.
It does not attempt to be a general-purpose high-level synthesizer (HLS); instead, it is focused at a very narrow set
of applications involving heavy numerical code which is abundant in control systems and DSP pipelines.

Holoso focuses on Python because this is a popular language in modeling, system design, and verification domains;
ability to generate production HDL directly from the original model allows the designer to work with much simpler
harnesses and iterate faster.

Holoso is an experimental project and as such it has no burden of backward compatibility.
Breaking changes will occur regularly without notice until v1.0 is out.

## Usage

Unlike pretty much any other tool in this domain, Holoso is trivial to set up and get started with. This is not a framework.

Clone the repo and install: `cd holoso && pip install -e .`

## Verification

Just say `nox`.

You may find the [zubax-fpga-toolchain](https://github.com/Zubax/fpga-toolchain-docker/pkgs/container/zubax-fpga-toolchain)
container useful as it comes with all of the required tools out of the box.

## Prior art

Projects in the same domain (Python-source HLS aimed at numerical/algorithmic kernels):

- [PyLog](https://github.com/hst10/pylog) -- algorithm-centric Python-to-FPGA flow that decorates a function and emits HLS C, also targets numerics.
- [Polyphony](https://github.com/polyphony-dev/polyphony) -- Python-based HLS compiler that translates a restricted Python subset directly to Verilog.
- [Allo](https://github.com/cornell-zhang/allo) (successor to [HeteroCL](https://github.com/cornell-zhang/heterocl)) -- MLIR-based Python-embedded DSL focused on composable ML accelerators; adjacent rather than overlapping.

See [PRIOR_ART.md](PRIOR_ART.md) for a detailed review.

Industrial-grade HLS toolchains -- [Vitis HLS](https://www.xilinx.com/products/design-tools/vitis/vitis-hls.html), [Intel HLS Compiler](https://www.intel.com/content/www/us/en/software/programmable/quartus-prime/hls-compiler.html), [Bambu](https://github.com/ferrandi/PandA-bambu), [Google XLS](https://github.com/google/xls) -- live in a different weight class with a much steeper adoption threshold and, to the best of our knowledge, do not support Python front-end. Holoso instead aims to be a narrowly specialized and easy-to-adopt tool rather than a framework.

### Other Python-based HDL tools

Projects that share superficial similarity but are actually aimed at a completely different problem:

- [PyMTL3](https://github.com/pymtl/pymtl3) -- a Python framework for RTL modeling and simulation, not algorithmic synthesis.
- [hwtHls](https://github.com/Nic30/hwtHls) -- LLVM-based HLS library for the HWToolkit ecosystem, oriented at general-purpose hardware development rather than numerical kernels.

Python HDLs such as [MyHDL](https://www.myhdl.org/), [Amaranth](https://github.com/amaranth-lang/amaranth), [Migen](https://m-labs.hk/migen/), [PyRTL](https://ucsbarchlab.github.io/PyRTL/), [Veriloggen](https://github.com/PyHDI/veriloggen) et al describe RTL in Python rather than synthesizing it from algorithmic code -- completely different tools for a different task, listed only for completeness.
