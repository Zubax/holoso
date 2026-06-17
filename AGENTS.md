# Holoso -- simple high-level synthesis of Python into Verilog for numerical code

Holoso converts a small subset of Python functions and expressions into synthesizable and verifiable Verilog.
Read the `README.md`.

Whenever introducing nontrivial changes, update `DESIGN.md` as well to keep it fully up-to-date and non-conflicting
with the implementation. However, do not attempt to capture minor implementation minutiae there, keep it high-level.

Do not commit anything unless asked explicitly to do so.

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
This means that even implementing a seemingly minor change may warrant a deep refactor if the change doesn't fit
cleanly into the existing architecture.
An acceptable design will not involve special-casing. Do not bypass existing abstractions to get the job done.
Work will not be accepted unless architected cleanly.

When editing code, do not ever leave any compatibility shims behind. Always do a clean break with bridges burned.
API compatibility is not a concern.

The Python version to target is 3.14 and newer. No need to ensure compatibility with older versions.

Do not use `from __future__ import annotations`.
Prefer `list` over `tuple[X, ...]` for homogeneous sequences (unless immutability is required).
Do not use `Protocol`, prefer `ABC`/`@abstractmethod` instead for interfaces and abstract base classes.
Prefer explicit `from X import Y as Y` instead of using `__all__` in `__init__.py` files.

Public APIs can only include items that are required to use the API and nothing else.
All non-public items are hidden in underscore-prefixed submodules.
Minimize the public API surface.

Importing anything from a package or subpackage is only allowed as long as it doesn't involve referencing
underscore-prefixed names. Exceptions apply for importing from parent modules with the dot notation, and for unit tests.
Accessing underscore-prefixed names from outside a class (or its descendants) is not allowed;
all externally accessble entities must be non-underscore-prefixed.

Avoid files longer than about ~3000 lines (this is a soft limit).
If a file grows beyond that, consider refactoring into smaller modules.

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

It is encouraged to use rich formatting and emojis in the output of command-line utilities.

### Verilog

Verilog style: 4-space indentation, concise names, snake_case files and directories, uppercase `parameter`/`localparam`.
Keep line length at or below 120 columns. Comment block lines should utilize the 120 column limit well, avoiding overly short lines.

Testbenches are written in Python using Cocotb or similar tools.

Functions can be used in synthesizable Verilog but only if avoiding them is unduly burdensome.
In synthesizable code, prefer `case` statements over nested ternary operators unless there are contraindications.

In complex modules, it is best to avoid a large number of named nets that are only used once; this does not help readability but rather the opposite.

Leave unused module outputs unconnected, like `.out_foo()`, instead of creating unused wires.

It is best to keep at most one `always @(posedge clk)` per module, unless there are strong reasons to do otherwise.
This rule should be followed, in particular, in Verilog emitted by the compiler.

The same register can be assigned multiple times only as long as the assignments reside in different branches that
cannot be active at the same time and are explicitly segregated with a single condition that is explicit to the
synthesizer. For example:

```verilog
reg [31:0] foo;
// COMPLIANT:
if (a) foo <= bar;
else   foo <= baz;
// BANNED even if it is known that a and b are mutually exclusive:
if (a) foo <= bar;
if (b) foo <= baz;
```

### Other

Keep in-code documentation brief. Long-form belongs in design docs and other non-code files.

When altering code behavior, do not comment on the changes; e.g., avoid constructs like "this used to be such and such"
or "this is done like this -- no longer like so". Document only the current state.

Do not add comments that add no new information, such as describing code behavior that is already clear from the code itself.
Only add comments that explain the rationale or any non-obvious implications or considerations.

In all source code and documentation, units of measure are given in the correct SI-compliant capitalization
regardless of any coding style. For example, `target_frequency_MHz` is correct as a lowercase snake_case name
despite having capital letters because conventional unit spelling requires so; `_mhz` would be incorrect
because it reads as millihertz. Same goes for `DELAY_us` instead of `DELAY_US` etc.

In Markdown, it is best to avoid bold `**` and italics `*` for emphasis; prefer plain text instead.
Prefer prose over lists, and avoid excessive formatting in general.
These are not hard rules but rather soft suggestions.

Don't hesitate to use rich Unicode where appropriate; e.g., in HTML reports, prefer `×` over `*` for multiplication,
`µs` over `us` for microseconds, `🠄` instead of `=`/`<=`/`:=` for assignment, `≤` instead of `<=` for less-or-equal,
and so on. Sensible use of emojis is also encouraged, especially in command-line output.

Use logging extensively. Ensure that every significant action/decision/condition is logged.

Generated reports must be written in rich and colorful human-friendly HTML format, not Markdown.

No need to add tests nor update the design docs for report-generation-related changes (e.g., HTML backend).

## Timing closure

Timing closure is an iterative process of hunting the next bottleneck and adding registers to break combinational paths:

- Set the desired frequency and synthesize the design.
- If f_max > f_target, exit.
- Locate the critical path and break it with a new register stage, e.g. configure `fadd.stage_decode=1`.
- Repeat.

This process works regardless of whether the failure to meet timings is caused by too many logic levels or long routing.
Special things to look out for:

- DSP tiles must begin and end with a register stage. If retiming has moved a register away from a DSP tile,
  it means that the adjacent hop is starving and needs a new register there, even if it's not on the critical path.
- Splitting multiplication into parallel halves (e.g., `STAGE_PRODUCT=1`) is almost never a good idea unless the
  multiplicand bitwidth exceeds the DSP slice input width.
- Retiming is sneaky: a moved stage may cause a different path to become critical, so with retiming enabled one needs
  to evaluate the adjacent stages as well.

More pipeline stages do not necessarily improve f_max, and can cost both timing and area. Every optional stage spreads
that operator's flip-flops across more slices; on a wide, register-pressure-heavy datapath this adds routing congestion,
so a congestion-bound design gets slower as stages are added even though no logic path got longer. When the critical
path is routing-dominated -- most of the delay is wire across only a few logic levels -- and adding a stage near it
makes things worse, the design is over-pipelined, not under-pipelined: strategically removing stages can raise f_max and
free flip-flops at the same time.

A robust closure procedure that accounts for this starts lean and adds back one stage at a time:

- Disable every optional stage. Mind the load-bearing exceptions above: the DSP product keeps `STAGE_PRODUCT` once the
  multiplicand exceeds the slice input width, and DSP tiles keep their bracketing registers.
- Read the critical path and judge, from the logic and the physics, whether it is the true bottleneck or merely a
  retiming casualty -- a path that only looks critical because a register was retimed away from the real cone. A true
  bottleneck is a recognizable deep operation (a wide barrel shift, a long carry chain, a DSP cascade); a casualty is an
  incidental cone that a stage added elsewhere will relieve.
- Add exactly one stage, at the boundary that splits the true bottleneck, and re-measure. Adding stages one at a time
  this way logic-balances a routing-dominated design without over-populating it with flip-flops.
- Repeat until f_max clears the target. If a newly added stage lowers f_max it was relieving congestion, not logic
  depth: back it out and split a different boundary.

## Verification

Entirely driven by `nox`; read the `noxfile.py` for details and follow its recommendations.
Tests may take a long time to run; if there is no output, assume they are still running, not stuck,

Treat all code as suspect and likely defective until proven otherwise through testing. A passing build, a clean type
check, a green-looking review, or code that merely reads as correct is not evidence of correctness -- only a test that
exercises the behavior and could have failed is. When in doubt, assume the path is untested and the behavior is wrong
until a kernel demonstrates otherwise.

Prefer API-level black-box kernel tests over intrusive tests: a specific kernel exercising the target path,
driven through the public API (e.g., `holoso.synthesize(fn, ops).numerical_model.elaborate()` etc),
asserting mostly on publicly observable behavior -- output values against a reference,
persistent state across transactions, typed-port metadata, error diagnostics -- not on internal structures.
Such black-box tests are more likely to survive a deep refactoring.
White-box tests remain valuable only where pure black-box tests are impractical.

Whenever a defect is found (whether by a review agent or reported by a user), you MUST add a regression test that
is verified to crash with the defect in place, and pass once the fix is implemented.

Use your best judgement as to which features do not need test coverage. For example, the following should be avoided:
- parameter validation in developer-only features;
- HTML layout correctness;
- white-box tests of implementation details rather than behaviors;
- rejection of invalid inputs where an exception is raised.

## Review team

After every change or milestone, or when explicitly prompted, dispatch several fresh-context review agents set to the
MAXIMUM THINKING EFFORT to review your work:

- A subagent focusing on the FUNCTIONAL CORRECTNESS and ROBUSTNESS of the implementation.
- A subagent focusing on the ARCHITECTURAL CLEANLINESS, DESIGN PRACTICES, and CODE QUALITY.
- Distinct tools -- Codex, Antigravity (`agy`), Claude (check what's available, exclude yourself) --
  focusing on CORRECTNESS only.

It is important that we use all available distinct tools to maximize the diversity of perspectives
and minimize blind spots. When all are done, review and consolidate their findings and act accordingly.
If behavioral defects are found, ensure extensive regression tests are introduced.

Repeat the review/refine loop until the agents return only trivial feedback (or none) for three (sic!) consecutive turns.
Here, "trivial feedback" means stylistic/inconsequential issues such as wording, formatting, trivial parameter
validation, or anything else that does not materially affect the correctness or maintainability of the codebase.
Iteration until no feedback has been attempted in the past but it is not practical because in the absence of significant
issues the review agents tend to degrade to nitpicking.
Hence, we stop iteration earlier, as soon as the feedback ceases to contain significant findings.

The requirement of multiple consecutive reviews with no significant findings is intended to improve the coverage.
We have seen in the past how a single review turn would come up blank while the next round (with zero code changes in
between) would dig up a critical defect. Hence, we repeat turns generously across distinct agents for maximum assurance.

When working around the low-level compiler components around regalloc, scheduler, etc, consider using the
`audit-schedule-quality` skill in an extra subagent.

Review agents in maximum thinking mode may go silent for a long time.
Set a generous timeout of about 1 hour or so, use your best judgement.
Some agents expect input from stdin when launched headless and may get hung if no input is given;
in those cases consider redirecting from `/dev/null` or something like that; read the docs to figure out usage.
