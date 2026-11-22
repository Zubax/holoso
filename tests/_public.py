"""
Helpers over public synthesis artifacts only (`frontend_ir` text and emitted Verilog), stdlib-implemented, for
the promoted tests that must not reach into compiler internals.
"""

import re

_LOCATION_SUFFIX = re.compile(r"  # (?:\S+:\d+ via [^,\n]+, )*\S+:\d+$", re.MULTILINE)

_PRELUDE_BEGIN = "// BEGIN holoso_support_inline.vh"
_PRELUDE_END = "// END of holoso_support_inline.vh"


def strip_locations(text: str) -> str:
    """
    Drop the per-statement location suffix (`  # file.py:NN`, preceded by a `file.py:NN via callee, ` hop per call
    site it expanded through) from a `frontend_ir` dump, yielding the canonical location-free Eel text. The pattern
    is anchored at end of line because bare substring assertions on the raw dump collide with the suffixes
    (basenames and line numbers).
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


def module_of(verilog: str, instance: str) -> str:
    """The module the kernel instantiates as the named instance (`u_fadd_0`, ...)."""
    modules: list[str] = re.findall(rf"^(\S+) {instance} \($", verilog, re.MULTILINE)
    (module,) = modules
    return module


def knob_table(verilog: str, instance: str) -> list[str]:
    """
    The configuration lines (`stage_core: CoreStages.DECODE`, `timing: Timing(latency=2, initiation_interval=1)`, ...)
    of the comment table that heads the module of the named instance.
    """
    module = module_of(verilog, instance)
    (table,) = re.findall(rf"((?:^//   .*\n)+)module {re.escape(module)}\(", verilog, re.MULTILINE)
    return [line.removeprefix("//   ") for line in table.splitlines()]


def instance_count(verilog: str, operator: str) -> int:
    """The number of instances of the named Holoso operator (`fadd`, `fdivsqrt`, ...) the kernel module drives."""
    return len(re.findall(rf"^\S+ u_{operator}_\d+ \($", verilog, re.MULTILINE))


def divsqrt_builds(verilog: str) -> list[int]:
    """
    What each `fdivsqrt` instance of an emitted design is built for, read off its module's ports: 0 divides only (no
    operation select), 1 takes roots only (no divisor either), 2 does both. One operator serves both operations, so its
    mere presence says nothing of which a kernel kept.
    """
    builds: list[int] = []
    for module in re.findall(r"^(\S+) u_fdivsqrt_\d+ \($", verilog, re.MULTILINE):
        (ports,) = re.findall(rf"^module {re.escape(module)}\(([^)]*)\);$", verilog, re.MULTILINE)
        names = set(ports.split(", "))
        assert {"a", "y"} <= names
        builds.append(2 if "op_sqrt" in names else 0 if "b" in names else 1)
    return builds
