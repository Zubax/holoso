# Holoso -- simple high-level synthesis of Python into Verilog for numerical code

Holoso converts a small subset of Python functions and expressions into synthesizable and verifiable Verilog.
It does not attempt to be a general-purpose high-level synthesizer (HLS); instead, it is focused at a very narrow set
of applications involving heavy numerical code which is abundant in control systems and DSP pipelines.

Holoso focuses on Python because this is a popular language in modeling, system design, and verification domains;
ability to generate production HDL directly from the original model allows the designer to work with much simpler
harnesses and iterate faster.

Holoso is an experimental project and as such it has no burden of backward compatibility.
Breaking changes will occur regularly without notice until v1.0 is out.

See [PRIOR_ART.md](PRIOR_ART.md) for a detailed review of existing alternatives, and why none are good enough.

## Usage

Unlike most tools in this domain, Holoso is trivial to set up and get started with; it is not a framework.

Clone the repo and install: `cd holoso && pip install -e .`

Come back later.

## Verification

Just say `nox`. Read the `noxfile.py` for details.

You may find the [zubax-fpga-toolchain](https://github.com/Zubax/fpga-toolchain-docker/pkgs/container/zubax-fpga-toolchain)
container useful as it comes with all of the required tools out of the box.
