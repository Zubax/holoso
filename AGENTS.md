# Holoso -- simple high-level synthesis of Python into Verilog for numerical code

Holoso converts a small subset of Python functions and expressions into synthesizable and verifiable Verilog.
Read the `README.md`.

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

### Language

Python and Verilog style: 4-space indentation, concise names, snake_case files and directories, uppercase Verilog `parameter`/`localparam`.
Keep line length at or below 120 columns. Comment block lines should utilize the 120 column limit well, avoiding overly short lines.

Testbenches are written in Python using Cocotb or similar tools.

The following constructs are banned in synthesizable Verilog (fine in reference models or testbenches):

- Blocking register assignment.
- Functions.

In synthesizable code, prefer `case` statements over nested ternary operators unless there are contraindications.

In complex modules, it is best to avoid a large number of named nets that are only used once; this does not help readability but rather the opposite.

### Other

Generated reports must be written in rich and colorful human-friendly HTML format, not Markdown.

## Verification

Entirely driven by `nox`; read the `noxfile.py` for details.
