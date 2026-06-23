---
name: verify-rtl-report-equivalence
description: >-
  Cross-check that the emitted Verilog RTL and the HTML schedule report describe the SAME schedule for
  each bundled example: every operator issue and landing present in one artifact must be present in the
  other, at a consistent clock and on the same registers and operands. Two independent max-thinking agents
  verify in opposite directions (RTL against HTML, and HTML against RTL). The check reads ONLY the two
  emitted text artifacts and NEVER uses any compiler internals, so a divergence between the two backends
  cannot hide behind shared code. Use during the review/refine loop.
---

# Verify RTL ⇄ HTML-report equivalence

Both artifacts are emitted from the same internal schedule. This check confirms the two BACKENDS agree
with each other by reading only their output, as a human would.

## What this catches — and what it does not

- Catches: a divergence between the RTL emitter and the HTML renderer — one artifact describing a
  different operation, clock, register, or operand than the other.
- Does NOT catch: a bug in the shared upstream schedule that both backends render faithfully.

## Independence — the rule that makes this trustworthy

If the verification borrowed any compiler code to read or align the artifacts, a bug in that code would
corrupt both sides identically and the check would only confirm "the emitter agrees with the emitter."
The value is an oracle that shares NO code with the thing under test.

## Regenerate the artifacts first

Rebuild every bundled example so the artifacts reflect the current code: run each script in the examples
directory; each one synthesizes its design(s) through the public API and writes the `.v` + `.html`. A
single script may emit several designs. This build is the ONLY place the compiler runs; once the files
exist, the verification agents touch nothing but the files.

## The invariant to verify

For each design, the two artifacts must describe the same set of operations. Match operations across the
two by their CONTENT, not their position:

- operator kind (select, bor, bxor, fadd, fmul, fcmp, …);
- destination register;
- source operands — including any sign/negation/inversion decoration and any constants.

For every matched operation, the two artifacts must agree on its timing: both when it is issued and when
it lands (commits its result).

Clock numbering is a convention, not behavior. The two artifacts may label clocks on different frames, and
the RTL may place different classes of operation on different timing surfaces. So do NOT compare absolute
clock numbers blindly, and do NOT assume any fixed offset (the offset is an internal detail that drifts —
discover it, never presume it):

- Establish the clock correspondence empirically from the artifacts themselves; each is self-documenting
  about how its displayed clocks relate to the steps the RTL switches on.
- Require that correspondence to be a single constant shift across all operations on a given surface. A
  uniform shift is fine — it relabels nothing real. A NON-constant shift, or an issue→landing latency that
  differs between the two artifacts, is a genuine finding: those survive any relabeling.

A finding is any operation present in one artifact but absent from the other, or any matched operation
whose kind, destination, operands, issue, or landing disagree.

## Two independent agents, opposite directions

For each design, dispatch — at MAXIMUM THINKING EFFORT — two fresh-context agents that do NOT share notes:

- RTL → HTML: enumerate every operation in the RTL and confirm each appears in the report, consistent in
  kind, destination, operands, issue, and landing.
- HTML → RTL: enumerate every operation in the report and confirm each appears in the RTL, the same way.

Illustrative only, not a checklist — concrete numbers go stale: in one example several inline operations
issue together on one clock and land together on the next — a boolean-or writing a bool register and a few
selects each writing a register from a condition. The check confirms the RTL contains those same writes,
guarded on a single matching step, reading the same condition and the same source/constant operands, with
the same one-clock issue→landing relationship the report shows.

## Report

Per design, list every discrepancy: which artifact held the operation, what the other artifact said (or
that it was absent), and which attribute disagreed (kind / destination / operand / issue / landing /
non-constant shift). A clean design reports "no divergence." Aggregate across all examples. Every confirmed
discrepancy is a backend defect: fix it and add a regression test, per the project's verification policy.
