---
name: comment-cleanup
description: >-
  Prune low-value comments and docstrings across a codebase while preserving genuine rationale and
  enforcing consistency. Use when asked to remove verbose or redundant comments, tighten documentation,
  or make comment style uniform. Aggressive downsizing, but never erase valuable information.
---

# Comment cleanup

Remove comments and docstrings that restate what the code or the type system already says; keep the ones
carrying information a competent reader could not reconstruct from names, signatures, and code in a few seconds.

## Keep vs. remove

REMOVE: name-echo describers (a docstring that just expands the identifier), narration of obvious control flow,
section-divider banners, value-trace and "what the next line does" comments, type-obvious prose.

KEEP: design rationale (the WHY), gotchas and edge-cases, contracts and invariants, cross-references,
standards or paper citations, hardware and timing facts, and the provenance of a regression test.

When uncertain whether something is rationale or restatement, keep it. The expensive mistake is erasing valuable
information, not leaving one extra obvious line. This guardrail outranks any size target.

## Never touch

Functional comments are not prose: type-checker and linter directives, formatter on/off switches, coverage
pragmas, shebangs, and encoding lines. Preserve every one. Never delete a docstring that is a block's only body
— it breaks syntax; reduce it to a terse line or a bare ellipsis instead.

## Calibrate with the user first

A blanket percentage target often misfits the codebase: well-documented code may be mostly rationale, so an honest
pass removes far less than a naive estimate. Sample the real comments, and if the honest reduction is far from what
was asked, offer a concrete choice — conservative, moderate, aggressive — with real before/after examples, and let
the user pick. Apply ONE threshold uniformly so the same kind of comment gets the same decision everywhere.

## Prove you changed nothing but comments

This safety net is what lets you cut confidently. Parse each file before and after, strip the docstrings from both,
and compare the syntax trees: equality proves only comments and docstrings changed, since comments never appear in
the tree. Treat an empty body, a bare ellipsis, and a lone pass as equivalent, so converting a sole-body docstring
passes. Run it after auto-formatting. The type checker is the separate guard that no functional directive was lost.

## Scale with subagents, then audit

Partition by file ownership so agents never share a file, and never let an agent run a history-rewriting or revert
command — it destroys a sibling's concurrent work; undo mistakes by hand-editing. Keep the verification script out
of their writable set; they only run it. Fresh subagents apply a given policy more willingly than ones asked to
reverse an earlier, more conservative instruction.

When the cutting is done, audit every deletion once more for erased rationale and restore the genuine losses. Then
format, type-check, and run the full test suite before committing. A differential fuzzer or a simulation pass, if
the project has one, is strong independent proof that behavior is unchanged.
