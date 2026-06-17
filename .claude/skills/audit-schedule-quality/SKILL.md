---
name: audit-schedule-quality
description: >-
  Assess the QUALITY (tightness/optimality, not correctness) of a generated Holoso schedule and write a
  crisp defect report: wasted cycles such as an op/select/install/branch scheduled later than its operands
  or condition allow, or a cascade of such single-cycle delays. The analysis is done strictly from first
  principles over the schedule's final state, NEVER by calling the compiler's own timing helpers (which
  would make the audit blind to bugs in that very timing). Use during the review/refine loop.
---

# Auditing schedule quality

Judge the schedule for tightness, not correctness. A defect is a wasted cycle: an operation, select,
install, or branch placed later than the datapath physics permits, while its operands (or condition) were
already available and the hardware it needs was free. Every credible finding pairs a wasted cycle with
the reason the delay is NOT justified.

## Independence — the rule that makes this trustworthy

Read ONLY the schedule's settled final state. Do NOT call the compiler's timing/scheduling helpers or its
derived analyses (landing/read/boundary/install helpers, liveness, cycle-grouping, etc.). Reusing them
makes the audit inherit the scheduler's own model of time, so a defect living in that model shows zero
slack and hides — the audit would only confirm "the scheduler did what the scheduler thinks." The value
is an oracle that does not share the scheduler's assumptions.

Fair game (facts the schedule committed to): each op's issue cycle, its operator's pipeline latency and
reuse interval, its operand source registers and result destination registers; the grouping into basic
blocks with each block's base offset and terminator; the installs and their source/destination. Recompute
everything else — availability, residence, earliest-feasible issue, block extent — yourself.

## Get the schedule

Synthesize the kernel in memory through the public entry point; reach the low-level IR (it may be private)
from the result and work in a throwaway temp script. Discover current attribute/class names from the live
IR module — treat any name here as illustrative.

## Timing model — re-derive it, then PIN it by calibration

DESIGN.md documents the datapath timing from first principles; re-derive the model from that prose, do not
copy constants out of the compiler. The constants are perishable and subtler than the prose — an edge can
depend not only on a value's bank but on the producer's and the consumer's kind — so trust no hardcoded
number; pin every one by calibration (below). The model has this shape:

- An op issued at T commits at T + latency (read latency from the operator; an inline op's latency is 0).
- A result becomes readable a fixed offset after its producer commits; a consumer samples a fixed offset
  from its own issue. Their difference is the minimum producer→consumer distance — the "edge".
- An op has an earliest-issue floor (a pooled op one cycle later than an inline op). Operands resident
  from the block start (constants, inputs, prior-block values) impose only the floor.

## Audit

Per block, in block-local cycles; report in the absolute frame (base + local).

1. Time-aware producer map. Registers are reused within a block, so an operand's producer is the most
   recent in-block writer of that register issued BEFORE the consumer — not simply "the writer of that
   register." Getting this wrong manufactures large fake slack.

2. Earliest-feasible issue = max(floor, producer_commit + edge) over the register operands.

3. Slack = issue − earliest-feasible. Built on producers' ACTUAL commits, so upstream lateness raises a
   consumer's earliest-feasible in lockstep: slack is intrinsic. A cascade's ROOT shows positive slack;
   the inheritors merely waiting on it show zero. Report roots; name inheritors as consequences.

4. Negative ("impossible") slack means the model is too tight or a value crossed a block boundary — never
   a finding, always a signal to fix a constant or treat the block as overlap-affected.

A positive-slack op is a defect only once the delay is shown unjustified:

- Resource. Inline ops do not contend; a pooled op may be late because its instance was busy — keep the
  finding only if that instance was idle somewhere in the gap.

- Register pressure. Pulling the op up lands its result earlier; keep it only if the destination register
  is free across that window. This rules out the checkable obstruction; it does not prove optimality.

Where a finding's justification or a calibration disagreement is a genuine judgment call,
get an independent adversarial second opinion from an advisor agent before committing to it;
a wrong "confirmed defect" costs more than the extra step.

## Calibrate first — the make-or-break step

An uncalibrated model fails quietly, and a handful of known defects cannot pin it. So before trusting any
finding, run the full FILTERED pipeline on RIGID kernels — kernels whose dataflow has no scheduling
freedom — and require zero SURVIVING findings (not zero raw slack: a rigid kernel sharing one instance
shows raw slack from pure serialization, which the resource filter correctly removes). A survivor on a
kernel you believe rigid is a model bug; fix the constant and re-run. Cover one rigid kernel per timing
primitive (e.g. a serial arithmetic chain, a serial boolean chain, a select chain, a single-branch
kernel); a primitive with no clean rigid kernel is a guessed constant — say so.

Calibration underwrites the method: the scheduler is tight on the rigid cases, which is exactly what lets
the loose cases stand out — so the audit detects looseness relative to the scheduler's demonstrated best
case, not absolute optimality. If the model instead DISAGREES on a rigid kernel, the model is wrong or the
physics itself is non-minimal — stop and surface it (grounding the latter needs RTL co-simulation and is
out of scope).

## Boundaries — installs, branches, block extent (needs-confirmation tier)

Late installs, late branches, empty trailing cycles, and over-long blocks are real defect classes, but
their timing leans on the boundary-drain and cross-block-pipelining rules — the hard part. The standard
writeback drain legitimately pushes a terminator past the last commit, so a naive "terminator past the
last activity" test fires on nearly every block; the defect is only the EXCESS drain (a block whose
boundary values are all boolean or already resident pays no wide drain, so a wide-sized tail is wasted).
Treat any finding that leans on boundary/drain or cross-block residence as needs-confirmation; the
within-block op slack above is the high-confidence tier. (The elaborated simulator is bit/cycle-exact but
rides the same helpers, so use it only to confirm a proposed tightening doesn't change results — never as
an optimality oracle.)

## Recipe (schematic — map to current names)

```
for each op:  efi = max(floor, max over register-operands of producer.commit + edge(...))
              # producer = most-recent in-block writer of that reg BEFORE this op  (time-aware)
              slack = op.issue - efi
              slack > 0 & instance-idle-in-gap & dest-free  -> ROOT;  slack < 0 -> model/overlap signal
```

## Examples (illustrative reasoning, not a checklist — concrete numbers go stale)

- Cascade root. A select `r ← b ? c1 : c2` issues one cycle after its condition `b` became readable, so
  it carries slack 1; its consumer `r2 ← c + r` reads the select a fixed edge later and shows zero slack —
  a pure inheritor, reported as a consequence, not a second defect. Repeated in an unrolled body, the
  one-cycle lag accumulates down the block. The instances are idle through the gap and the destination is
  free, so the delay is unjustified: move the root up and the whole tail follows.

- Boundary classes (needs-confirmation). An install sitting well after its source was ready and pushing
  the block tail; a branch taken a cycle past the last real work while its condition was long available;
  a block whose only activity is at the very end, leaving cycles idle beyond the legitimate drain.

## Report

One entry per finding: kind; block and operation (in the report's register/label vocabulary); issue
cycle; computed earliest-feasible; slack; justifications ruled out; for a cascade, the root and the
inheritors it explains; and a confidence tier. Lead with the calibration outcome — which rigid kernels
passed, which constants were pinned, any primitive left as a guess — so the reader can weigh the findings.
