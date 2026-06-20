"""
Saved fuzz regressions: self-contained replayable differential divergences found by the campaign.

Each ``*.py`` here holds a generated kernel's source plus a ``META`` dict (the failing input vectors as exact ZKF bits,
the op-config, the regalloc effort, and which differential check failed). ``test_fuzz_regressions.py`` globs them and
re-asserts the previously-failing check so a fixed bug can never silently regress.
"""
