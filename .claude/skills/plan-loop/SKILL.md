---
name: plan-loop
description: >-
  Adversarial review of a PLAN -- before execution begins, and again before each remaining step of a long-horizon work.
  Critique it with an ultrathink Claude agent plus Codex, apply what survives project policies,
  repeat until significant findings cease.
  This is NOT intended for reviewing work that is already completed.
---

# Plan review loop

Think the work through before starting it, then let two dissimilar reviewers attack the thinking.
Applies whether or not plan mode is in use.

## The loop

1. Draft the plan -- or take the one you were handed -- and write it to a scratch file, along with the
   original request it serves, any decisions the user has made since, the reasons earlier findings were
   rejected, and -- once execution is under way -- what has actually happened so far.
   Reviewers know nothing beyond what that file says, and will resurrect settled objections it omits.

2. Dispatch both reviewers in parallel, pointing each at that file:
   - an *ultrathink* Claude agent;
   - Codex on its most advanced model at maximum reasoning effort.

   Every round starts from fresh reviewer contexts.
   A reviewer carrying its own earlier critique defends that critique instead of reading the revised plan on its merits.

   Keep the prompts extremely terse -- goal, plan path, "critique it"; detail breeds tunnel vision.
   A prompt longer than 2 sentences is a bad prompt; 3 sentences is the absolute limit, anything larger is unacceptable.
   State the remit explicitly: critique the PLAN's reasoning -- sequencing, assumptions, omissions, wrong direction --
   NOT the current state of the worktree.
   Reviewers are read-only; they may read anything to test the plan against reality, but they change nothing
   (creation of their own disposable copies of the worktree is allowed).

3. Vet every finding against the project's own policy docs and the original request.
   Apply the survivors, discard the rest with a reason; a finding rejected on policy does not keep the loop alive.

4. If anything of substance changed, go back to 2 with the revised plan.

Stop as soon as a round yields nothing that survives vetting, or only minor findings.
With nothing real left to say, reviewers degrade into nitpicking,
so further rounds burn iterations while changing nothing. A nitpick is not a mandate.

In plan mode the loop runs before `ExitPlanMode`, so what reaches the user for approval is the converged
plan rather than the first draft.
A later revision that changes the scope goes back to the user too.

## Re-review before each remaining step

Step here means a material milestone, not every edit.
Step one is rooted in a known initial state and rarely needs this.
Every later step rests on predictions that execution has since confirmed or invalidated -- a plan is a map,
and its worth lies in what it omits. So before beginning each subsequent step, run the same loop on
the larger question: given where we actually are now, is the remaining plan still the right direction?
Corrections grow the further along you are, and that is the method working rather than the plan failing --
but a round that confirms the plan unchanged is just as successful.

## Operational notes

Hand the plan over by file path, not on stdin: headless Codex hangs unless its stdin is `< /dev/null`.
Max-effort reviewers go silent for long stretches -- set generous timeouts and do not read quiet as stuck.
Retry reviewers that die on transient errors.
