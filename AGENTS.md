# Holoso -- simple high-level synthesis of Python into Verilog for numerical code

Holoso converts a small subset of Python functions and expressions into synthesizable and verifiable Verilog.
Read the `README.md`.

Whenever introducing changes, update `DESIGN.md` as well to keep it reasonably up-to-date and non-conflicting with the implementation.

## Conventions

### Reset strategy

Use synchronous active-high reset for stream control only: validity flags, state-machine state, and other control
registers that define whether an output transaction is meaningful.
Avoid resetting pure datapath registers whose contents are ignored while their associated valid flag is deasserted.
This keeps high-fanout reset nets out of wide payload cones, reduces control-set pressure,
and gives synthesis/place-and-route more freedom to retime and optimize pipeline registers.

Do not write the datapath assignment only in the reset-else branch, as it still makes data depend on reset because
the register is held during reset. A better strategy is to make datapath manipulation reset-unconditional
and only keep the control signals under reset/else.

References:

- AMD UG949, "When and Where to Use a Reset":
  <https://docs.amd.com/r/en-US/ug949-vivado-design-methodology/When-and-Where-to-Use-a-Reset>
- Intel Hyperflex Architecture High-Performance Design Handbook, "Synchronous Resets Summary":
  <https://docs.altera.com/r/docs/683353/25.1.1/hyperflex-architecture-high-performance-design-handbook/synchronous-resets-summary?contentId=vgtR8yUs_Z5DH0ApHJFiTQ>
- Intel Hyperflex Architecture High-Performance Design Handbook, "Reset Strategies":
  <https://docs.altera.com/r/docs/683353/25.1.1/hyperflex-architecture-high-performance-design-handbook/reset-strategies?contentId=gzd92HdsL40qZGHurB0ezg>

### Python

Follow PEP8 with one exception: the maximum line length is 120 columns. This is already configured in Black.
Comment block lines should utilize the 120 column limit well, avoiding overly short lines.

Use strongly typed primitives. Instead of int constants, prefer enums; instead of dicts, prefer dataclasses;
instead of existence/vaidity flags, prefer optional type or unions, etc.

If a design calls for a leaky abstraction, discard it and redesign from scratch, even if it involves breaking changes.
An acceptable design will not involve special-casing. Do not bypass existing abstractions to get the job done.
Work will not be accepted unless architected cleanly.

The Python version to target is 3.14 and newer. No need to ensure compatibility with older versions.

Do not use `from __future__ import annotations`.
Prefer `list` over `tuple[X, ...]` for homogeneous sequences.
Do not use `Protocol`, prefer `ABC`/`@abstractmethod` instead for interfaces and abstract base classes.
Prefer explicit `from X import Y as Y` instead of using `__all__` in `__init__.py` files.

Public APIs can only include items that are required to use the API and nothing else.

If a docstring comment doesn't fit on one line, add an initial line break like this:

```python
"""
This is a very long
comment string.
"""
```

Instead of this:

```python
"""This is a very long
comment string.
"""
```

### Verilog

Verilog style: 4-space indentation, concise names, snake_case files and directories, uppercase `parameter`/`localparam`.
Keep line length at or below 120 columns. Comment block lines should utilize the 120 column limit well, avoiding overly short lines.

Testbenches are written in Python using Cocotb or similar tools.

Functions can be used in synthesizable Verilog but only if avoiding them is unduly burdensome.
In synthesizable code, prefer `case` statements over nested ternary operators unless there are contraindications.

In complex modules, it is best to avoid a large number of named nets that are only used once; this does not help readability but rather the opposite.

Leave unused module outputs unconnected, like `.out_foo()`, instead of creating unused wires.

### Other

Keep in-code documentation brief. Long-form belongs in design docs and other non-code files.

In all source code and documentation, units of measure are given in the correct SI-compliant capitalization
regardless of any coding style. For example, `target_frequency_MHz` is correct as a lowercase snake_case name
despite having capital letters because conventional unit spelling requires so; `_mhz` would be incorrect
because it reads as millihertz. Same goes for `DELAY_us` instead of `DELAY_US` etc.

In Markdown, it is best to avoid bold `**` and italics `*` for emphasis; prefer plain text instead.
Prefer prose over lists, and avoid excessive formatting in general.
These are not hard rules but rather soft suggestions.

Generated reports must be written in rich and colorful human-friendly HTML format, not Markdown.

No need to add tests nor update the design docs for report-generation-related changes.

## Verification

Entirely driven by `nox`; read the `noxfile.py` for details.
