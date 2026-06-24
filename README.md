<div align="center">

<img src="docs/holoso-logo-128.png" width="60px">

<h1>Holoso</h1>

_Simple high-level synthesis of portable Verilog from idiomatic Python_

[![Try online](https://img.shields.io/badge/try_online-holoso.digital-black?color=ff0000)](https://holoso.digital/)
[![Forum](https://img.shields.io/discourse/https/forum.zubax.com/users.svg?logo=discourse&color=ff0000)](https://forum.zubax.com)

</div>

-----

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

Synthesizing a kernel with `holoso.synthesize(...).write(out_dir)` produces a self-contained set of files: the generated
module, a single `holoso_support.v` support library together with its `holoso_support.vh` header, a cocotb testbench,
and an HTML schedule report. The support library bundles the upstream Zubax Kulibin float primitives, so nothing else
has to be fetched to simulate or synthesize the result. Maintainers refresh those vendored primitives with
`tools/update_support_rtl.py`; each source's provenance and license live in `holoso/_backend/verilog/rtl/<source>/README.md`
(e.g. `rtl/kulibin/README.md`).

Come back later.

## Verification

Just say `nox`. Read the `noxfile.py` for details.

You may find the [zubax-fpga-toolchain](https://github.com/Zubax/fpga-toolchain-docker/pkgs/container/zubax-fpga-toolchain)
container useful as it comes with all of the required tools out of the box.
