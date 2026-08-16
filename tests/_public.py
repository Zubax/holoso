"""
Helpers over public synthesis artifacts only (`frontend_ir` text and emitted Verilog), stdlib-implemented, for
the promoted tests that must not reach into compiler internals.
"""

import re

_LOCATION_SUFFIX = re.compile(r"  # \S+:\d+(?: via \S+)?$", re.MULTILINE)

_PRELUDE_BEGIN = "// BEGIN holoso_support_inline.vh"
_PRELUDE_END = "// END of holoso_support_inline.vh"


def strip_locations(text: str) -> str:
    """
    Drop the per-statement location suffix (`  # file.py:NN` plus an optional ` via callee,...` chain) from a
    `frontend_ir` dump, yielding the canonical location-free Eel text. The pattern is anchored at end of line
    because bare substring assertions on the raw dump collide with the suffixes (basenames and line numbers).
    """
    return _LOCATION_SUFFIX.sub("", text)


def strip_inline_prelude(verilog: str) -> str:
    """
    Remove the spliced support-function prelude from an emitted module, so that every remaining `holoso_<name>(`
    occurrence is a genuine call site; the prelude nests calls of its own, so raw-text counts over-count.
    """
    lines = verilog.splitlines(keepends=True)
    begins = [i for i, line in enumerate(lines) if _PRELUDE_BEGIN in line]
    ends = [i for i, line in enumerate(lines) if _PRELUDE_END in line]
    assert len(begins) == 1 and len(ends) == 1 and begins[0] < ends[0]
    return "".join(lines[: begins[0]] + lines[ends[0] + 1 :])
