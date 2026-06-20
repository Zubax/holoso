"""
End-to-end blackbox differential fuzzing of the Holoso compiler.

The campaign generates small kernels as Python *source text* -- rendered to real importable modules under ``build/`` so
the frontend's ``inspect.getsourcelines`` retrieval succeeds -- and drives each through the public-ish compiler pipeline
twice: once into the numerical model (downstream of LIR scheduling/binding/regalloc/overlap) and once into the MIR
interpreter (the schedule-independent oracle, upstream of the LIR layer). The two share the front/mid-end and
``operator.evaluate`` but NOT the LIR, so the primary check ``interpreter == model`` (bit-exact, exception-free on both
ZKF sides) isolates exactly the LIR layer -- the miscompile class the RTL-versus-model cosimulation is blind to.

The generator emits the *danger shapes by construction* -- real un-if-convertible diamonds (kept branchy by an
unspeculatable division with a structurally-nonzero divisor), the dead-arm-spill family, wide register-pressure chains,
nested ifs, const-branch-via-division, diamond-then-loop, bounded while loops, cross-domain bool/float logic, selects,
unrolled reductions, and stateful classes with private chained slots -- because these are the shapes a naive
straight-line fuzzer misses. Every kernel carries a typed shape descriptor used both for the campaign histogram and to
assert the danger shape *survived* compilation (a diamond silently degrading to a select must fail loudly, else the
kernel tests nothing).

A secondary, best-effort check compares the model against a float64 reference. In EXACT-arithmetic mode every float
output is exact in the format (selects, gates, integer counters, power-of-two scaling, Sterbenz-exact subtraction), so
this is bit-exact and IS a sound full-pipeline oracle; in CONTINUOUS mode (general +,*,/) only the interpreter==model
check is bit-exact and the float64 check is a gross net that is skipped wherever float64 legitimately diverges (a raise,
inf/nan, a ZKF saturation, or a loop trip-count split near a threshold). A CONTINUOUS finite out-of-tolerance miss is
deliberately NOT failed: the per-operation tolerance is a heuristic, not a sound bound on ZKF-vs-float64 divergence
(catastrophic cancellation makes the relative error unbounded), so a hard failure there would flag precision drift as a
miscompile and -- because divergences are saved as replayable regressions that run in the normal test session -- could
wedge that session red on a non-bug. Continuous out-of-tolerance vectors are instead counted and surfaced for
visibility; operator rounding correctness is covered by the dedicated arithmetic tests, not by this differential net.
"""

import linecache
import math
import types
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any

import numpy as np

from holoso._backend.numerical import NumericalSimulator, generate
from holoso._frontend import lower as lower_frontend
from holoso._hir import optimize
from holoso._lir import Lir, build
from holoso._mir import Mir, MirBranch, MirInterpreter, MirJump, MirTerminator
from holoso._mir import lower as lower_to_mir
from holoso._operators import OpConfig
from holoso._type import BoolType, FloatFormat
from holoso._value import FloatValue

from ._modelref import (
    Vector,
    default_ops,
    default_tolerance,
    flatten_value,
    format_edge_bits,
    show_value,
    staged_ops,
    within,
)

# The directory the generated kernel modules are rendered into. ``build/`` is gitignored, so the fuzz corpus never
# pollutes the tree; the directory is created on demand.
_REPO = Path(__file__).resolve().parent.parent
FUZZ_TMP = _REPO / "build" / "fuzz_tmp"

# The operator-config catalogue swept per kernel, keyed by a stable label so a saved reproducer can name its config and
# the regression replayer can rebuild it. Two points: the minimum-latency default and a deeply-pipelined config (the
# model's timing -- but never the interpreter's -- changes with latency, so this cross-checks the LIR layer at two
# depths against the one fixed reference).
OP_CONFIGS: dict[str, Callable[[FloatFormat], OpConfig]] = {
    "default": default_ops,
    "staged": staged_ops,
}


class Shape(Enum):
    """
    A structural feature a generated kernel exercises. A kernel carries the set of shapes it realizes; the set drives
    the campaign histogram and -- for :data:`Shape.BRANCH` -- the post-compile survival assertion.
    """

    BRANCH = auto()  # a real (un-if-converted) diamond: the compiled MIR must keep more than one block
    DEAD_ARM_SPILL = auto()  # the dead-arm-spill soundness family (a value live in one arm, a wide chain spilling both)
    WIDE_CHAIN = auto()  # a long register-pressure multiply-add chain
    NESTED_IF = auto()  # a branch nested inside a branch
    CONST_BRANCH = auto()  # a constant-true inner condition formed by division (an empty const-branch block)
    LOOP = auto()  # a bounded back-edge ``while`` loop
    DIAMOND_THEN_LOOP = auto()  # a real diamond whose merge feeds a loop header (a three-arm header phi)
    SELECT = auto()  # a ternary ``a if c else b`` select
    BOOL_BANK = auto()  # boolean inputs and/or boolean connectives (the cross-domain bool/float bank)
    REDUCTION = auto()  # an unrolled ``for`` reduction over a literal range
    STATE = auto()  # persistent private slots (a stateful class), including the chained-slot pattern


class Mode(Enum):
    """The numerical regime of a kernel's float outputs -- it decides the secondary (float64) check's strictness."""

    EXACT = auto()  # every float output is exact in the format: the float64 reference must match bit-for-bit
    CONTINUOUS = auto()  # general +,*,/ : only interpreter==model is exact; the float64 check is a gross net


@dataclass(frozen=True, slots=True)
class GeneratedKernel:
    """
    One rendered kernel ready to drive: its module-qualified callable, its source text and file, the shapes it
    realizes, its numerical mode, its input-port names (in order), and the seed it was drawn from (for provenance).
    """

    name: str
    source: str
    filename: str
    callable: Callable[..., object]
    shapes: frozenset[Shape]
    mode: Mode
    input_names: list[str]
    bool_inputs: frozenset[str]
    seed: int
    index: int
    is_stateful: bool
    # A factory for a *fresh* reference instance (stateful kernels only) so the float64 reference is stepped on its own
    # instance, independent of the one the model was compiled from. ``None`` for a stateless kernel.
    fresh_reference: Callable[[], Callable[..., object]] | None = None


# --------------------------------------------------------------------------------------------------------------------
# Source rendering: a tiny templated emitter producing real importable .py modules.
# --------------------------------------------------------------------------------------------------------------------


def _render_module(name: str, source: str) -> types.ModuleType:
    """
    Write ``source`` to a real file under :data:`FUZZ_TMP`, register it with ``linecache`` (so ``inspect.getsource``
    works), compile it, and exec it into a fresh module. The frontend retrieves the kernel's source via
    ``inspect.getsourcelines``, which a REPL/exec-only function cannot satisfy -- hence the real, linecache-backed file.
    """
    FUZZ_TMP.mkdir(parents=True, exist_ok=True)
    path = str((FUZZ_TMP / f"{name}.py").resolve())
    Path(path).write_text(source, encoding="utf-8")
    linecache.cache[path] = (len(source), None, source.splitlines(keepends=True), path)
    module = types.ModuleType(name)
    module.__file__ = path
    exec(compile(source, path, "exec"), module.__dict__)  # noqa: S102 -- generated, sandboxed, gitignored source
    return module


# --------------------------------------------------------------------------------------------------------------------
# Expression builders: bounded, structurally-controlled fragments.
# --------------------------------------------------------------------------------------------------------------------


class _Emitter:
    """
    A per-kernel source builder. It tracks the live float/bool variable pools it may read from, the running set of
    realized shapes, and the numerical mode, accumulating body lines under a fixed indentation. Every fragment is
    bounded and structurally controlled so the resulting kernel compiles (stays branchy where intended), terminates
    (loop-carried values strictly approach the exit), and -- in EXACT mode -- stays bit-exact against float64.
    """

    def __init__(self, rng: np.random.Generator) -> None:
        self._rng = rng
        self._floats: list[str] = []
        self._bools: list[str] = []
        self._counter = 0
        self._lines: list[tuple[int, str]] = []
        self.shapes: set[Shape] = set()
        self.mode = Mode.EXACT  # demoted to CONTINUOUS the first time a non-exact float fragment is emitted
        self.return_line = ""  # set by the assembly step before rendering

    def then_value(self) -> str:
        """
        The value assigned in a diamond's then arm: a verbatim select, a power-of-two scale, or a continuous
        expression. (The kernel is already CONTINUOUS in any case, since a diamond's else arm is a rounding division.)
        """
        if self.chance(0.4):
            return self.scaled_value()
        if self.chance(0.4) and len(self.floats) >= 2:
            return self.exact_select_value(_emit_condition(self))
        return self.continuous_expr(2)

    def else_division(self) -> str:
        """
        The unspeculatable division that keeps a diamond a real branch: a numerator over a structurally-nonzero
        divisor. A division rounds, so its arm is continuous; since every real diamond routes through here, emitting
        one demotes the whole kernel to CONTINUOUS mode (a phi whose arm is a rounding division is not exact).
        """
        self.go_continuous()
        return f"({self.pick_float()} {self.choice(['+', '*'])} {self.pick_float()}) / {self.nonzero_divisor()}"

    def fresh(self, prefix: str = "t") -> str:
        self._counter += 1
        return f"_{prefix}{self._counter}"

    def add_float(self, name: str) -> None:
        self._floats.append(name)

    def add_bool(self, name: str) -> None:
        self._bools.append(name)

    @property
    def floats(self) -> list[str]:
        return self._floats

    @property
    def bools(self) -> list[str]:
        return self._bools

    def pick_float(self) -> str:
        return self._rng.choice(self._floats)  # type: ignore[return-value]

    def pick_floats(self, n: int) -> list[str]:
        """
        ``n`` operand names, DISTINCT when the pool is large enough -- so a relation like ``a < b`` stays
        input-dependent. A self-comparison (``a < a``) is constant and makes one branch arm UNREACHABLE, which would let
        a forced dead-arm kernel pass a real clobber regression vacuously (the clobbered value is never read on the dead
        path). Falls back to with-replacement only for a pool too small to supply ``n`` distinct names.
        """
        if len(self._floats) >= n:
            return [str(name) for name in self._rng.choice(self._floats, size=n, replace=False)]
        return [self.pick_float() for _ in range(n)]

    def has_bool(self) -> bool:
        return bool(self._bools)

    def pick_bool(self) -> str:
        return self._rng.choice(self._bools)  # type: ignore[return-value]

    def chance(self, p: float) -> bool:
        return bool(self._rng.random() < p)

    def choice(self, options: list[str]) -> str:
        return self._rng.choice(options)  # type: ignore[return-value]

    def randint(self, lo: int, hi: int) -> int:
        return int(self._rng.integers(lo, hi + 1))

    def go_continuous(self) -> None:
        self.mode = Mode.CONTINUOUS

    # -- bounded numeric fragments ----------------------------------------------------------------------------------

    def small_literal(self) -> str:
        """A small exact power-of-two-friendly literal (exact in any reasonable format)."""
        return self.choice(["1.0", "2.0", "0.5", "0.25", "4.0", "-1.0", "-2.0", "-0.5"])

    def nonneg_expr(self) -> str:
        v = self.pick_float()
        return f"({v} * {v})"

    def nonzero_divisor(self) -> str:
        """A structurally-positive divisor ``(<nonneg>) + c`` with ``c`` a positive literal."""
        c = self.choice(["1.0", "0.5", "2.0"])
        return f"(({self.nonneg_expr()}) + {c})"

    def continuous_expr(self, depth: int) -> str:
        """A bounded continuous-arithmetic expression of ``+``, ``*`` over the live float pool (demotes the mode)."""
        self.go_continuous()
        if depth <= 0 or not self.chance(0.7):
            return self.pick_float()
        op = self.choice(["+", "*"])
        left = self.continuous_expr(depth - 1)
        right = self.continuous_expr(depth - 1)
        return f"({left} {op} {right})"

    def wide_chain(self, width: int) -> str:
        """
        A wide Horner-style multiply-add chain ``(((a*z + b)*z + b)...)`` of the requested width. This commits late and
        creates register pressure; reused as the spilled value in the dead-arm family. Continuous arithmetic.
        """
        self.go_continuous()
        self.shapes.add(Shape.WIDE_CHAIN)
        z = self.pick_float()
        b = self.pick_float()
        acc = self.pick_float()
        for _ in range(width):
            acc = f"(({acc} * {z}) + {b})"
        return acc

    def exact_select_value(self, cond: str) -> str:
        """
        A select whose arms are *operands verbatim* (no arithmetic), so the result is exact in the format. Includes the
        ``x if c else -x`` sign-flip shape (negation is exact). Adds :data:`Shape.SELECT`.
        """
        self.shapes.add(Shape.SELECT)
        a, b = self.pick_float(), self.pick_float()
        if self.chance(0.4):
            return f"({a} if {cond} else -{a})"
        return f"({a} if {cond} else {b})"

    def scaled_value(self) -> str:
        """
        A power-of-two scaling ``x * 2.0`` / ``x * 0.5``. Exact within a safe exponent window, but it can overflow at
        the largest-finite edge bit or underflow a subnormal, so it conservatively demotes to CONTINUOUS rather than
        risk a spurious EXACT-mode mismatch at the format extremes.
        """
        self.go_continuous()
        return f"({self.pick_float()} * {self.choice(['2.0', '0.5'])})"

    def exact_clamp_value(self) -> str:
        """
        Emit a clamp of one live float into ``[-1, 1]`` via two selects (each arm an operand verbatim) and return the
        result variable name. A clamp routes an operand or a literal verbatim, so it is exact in the format.
        """
        x = self.pick_float()
        lo, hi = "-1.0", "1.0"
        below = self.fresh("clmp")
        clamped = self.fresh("clamped")
        self.emit(f"{below} = ({lo} if {x} < {lo} else {x})")
        self.emit(f"{clamped} = ({hi} if {below} > {hi} else {below})")
        self.shapes.add(Shape.SELECT)
        self.add_float(clamped)
        return clamped

    # -- line accumulation ------------------------------------------------------------------------------------------

    def emit(self, line: str, indent: int = 1) -> None:
        self._lines.append((indent, line))

    def reset_lines(self) -> None:
        self._lines = []

    def render_body(self, base_indent: int = 0) -> str:
        """
        Render the accumulated body, shifting every line by ``base_indent`` levels. A top-level function uses 0 (its
        body indent of 1 == 4 spaces); a class method uses 1 (its body lives one level deeper, at 8 spaces).
        """
        return "\n".join("    " * (indent + base_indent) + line for indent, line in self._lines)


# --------------------------------------------------------------------------------------------------------------------
# Kernel templates. Each returns the body lines (via the emitter) and the name of the returned float variable.
# --------------------------------------------------------------------------------------------------------------------


def _emit_diamond(em: _Emitter, *, nested: bool) -> str:
    """
    A real (un-if-convertible) diamond: a runtime comparison branches into two arms, the else arm uses an
    unspeculatable division with a structurally-nonzero divisor (so if-conversion refuses), each arm assigns the result
    variable, and the kernel reads it after the merge. Optionally nests a second diamond inside the then arm.
    Returns the result-variable name.
    """
    em.shapes.add(Shape.BRANCH)
    cond = _emit_condition(em)
    r = em.fresh("r")
    em.emit(f"if {cond}:")
    if nested and len(em.floats) >= 2:
        em.shapes.add(Shape.NESTED_IF)
        inner = _emit_condition(em, indent=2)
        em.emit(f"if {inner}:", indent=2)
        em.emit(f"{r} = {em.then_value()}", indent=3)
        em.emit("else:", indent=2)
        em.emit(f"{r} = {em.then_value()}", indent=3)
    else:
        em.emit(f"{r} = {em.then_value()}", indent=2)
    em.emit("else:")
    em.emit(f"{r} = {em.else_division()}", indent=2)
    em.add_float(r)
    return r


def _emit_condition(em: _Emitter, indent: int = 1, *, balanced: bool = False) -> str:
    """
    A runtime boolean condition: a float relation, or (when bool inputs exist) a boolean connective of them. With
    ``balanced=True`` the float relation is restricted to inequalities (``<``, ``>``, ``<=``, ``>=``), excluding
    ``==``/``!=`` -- whose two arms are almost never BOTH hit by independent random float draws -- so a branch whose
    hazard lives in one arm (the forced dead-arm shape) actually exercises that arm over the vector sequence.
    """
    if em.has_bool() and em.chance(0.4):
        em.shapes.add(Shape.BOOL_BANK)
        a = em.pick_bool()
        if em.chance(0.5) and len(em.bools) >= 2:
            return f"({a} {em.choice(['and', 'or'])} {em.pick_bool()})"
        if em.chance(0.5):
            return f"(not {a})"
        return a
    a, b = em.pick_floats(2)
    relations = ["<", ">", "<=", ">="] if balanced else ["<", ">", "<=", ">=", "==", "!="]
    return f"({a} {em.choice(relations)} {b})"


def _emit_dead_arm_spill(em: _Emitter) -> str:
    """
    The dead-arm-spill soundness shape, generalized from ``overlap_dead_arm_spill_kernel``: several simultaneously-live
    values ``v`` computed in the entry block and read in ONLY one arm, plus a wide chain ``w`` that commits late and
    spills into BOTH arms but is DEAD in the arm that reads the ``v`` values. Two levers arm the register-reuse hazard
    the shape probes: a long (5-9 deep) chain so ``w`` commits late and reliably SPILLS past the overlap-shrunk
    terminator into both arms; and several live ``v`` siblings so register PRESSURE forces the allocator to reuse the
    spilled chain's register in the dead arm -- where a dropped inflight reservation would clobber a still-read value.
    Randomizes the chain width, the pressure, and which arm is the dead (``v``-reading) one. Returns the result name.
    """
    em.shapes.add(Shape.BRANCH)
    em.shapes.add(Shape.DEAD_ARM_SPILL)
    live = [em.fresh("v") for _ in range(em.randint(3, 6))]
    for v in live:  # a fan of entry values, all live into the dead arm only (never pooled, so only that arm reads them)
        em.emit(f"{v} = {em.pick_float()} * {em.pick_float()} + {em.pick_float()}")
    em.go_continuous()
    w = em.fresh("w")
    # ``w`` is NOT added to the readable float pool: it is referenced by literal name only in the LIVE arm, so it stays
    # live (spilling into both arms) yet DEAD in the v-reading arm -- the exact register-reuse hazard.
    em.emit(f"{w} = {em.wide_chain(em.randint(5, 9))}")
    cond = _emit_condition(em, balanced=True)  # inequality (not ==/!=) so the dead arm is actually hit over the vectors
    r = em.fresh("r")
    dead_then = em.chance(0.5)
    a = em.fresh("a")
    dead_body = (f"{a} = ({' + '.join(live)})", f"{r} = {a} / {em.nonzero_divisor()}")  # reads the live v fan, NOT w
    live_body = (f"{r} = {w} * {em.choice(['2.0', '0.5'])}",)  # the only reader of w
    em.emit(f"if {cond}:")
    for line in dead_body if dead_then else live_body:
        em.emit(line, indent=2)
    em.emit("else:")
    for line in live_body if dead_then else dead_body:
        em.emit(line, indent=2)
    em.add_float(r)
    return r


def _emit_const_branch(em: _Emitter) -> str:
    """
    A const-branch-via-division shape: an outer runtime diamond whose then arm contains an inner ``if (1.0/5.0) > 0.0:``
    -- constant-true but formed by division, which escapes the AST-level reachability fold, leaving an empty
    const-branch block. Adds :data:`Shape.CONST_BRANCH` and :data:`Shape.BRANCH`.
    """
    em.shapes.add(Shape.BRANCH)
    em.shapes.add(Shape.CONST_BRANCH)
    em.shapes.add(Shape.NESTED_IF)
    cond = _emit_condition(em)
    r = em.fresh("r")
    base = em.pick_float()
    em.emit(f"{r} = {base}")
    em.emit(f"if {cond}:")
    em.emit("if (1.0 / 5.0) > 0.0:", indent=2)
    em.emit(f"{r} = {base} + 1.0", indent=3)
    em.emit("else:", indent=2)
    em.emit(f"{r} = {base} + 2.0", indent=3)
    em.add_float(r)
    em.go_continuous()  # the +1.0/+2.0 round-trips through the format only if base is integer-ish; treat as continuous
    return r


def _emit_bounded_loop(em: _Emitter) -> str:
    """
    A bounded back-edge ``while`` loop: a carried scalar strictly decreasing by a positive literal toward the exit
    comparison, with a bounded initial value so the trip count stays small. The model and the float64 reference have NO
    step bound, so the loop MUST terminate by construction -- this is the only loop shape emitted.
    Returns the carried-variable name.
    """
    em.shapes.add(Shape.LOOP)
    r = em.fresh("loop")
    # Start from a non-negative, bounded quantity (a square, clamped via min(...) is unavailable, so use a select).
    seed = em.pick_float()
    capped = em.fresh("cap")
    # cap the loop seed to a small magnitude so the trip count is bounded regardless of the input
    em.emit(f"{capped} = ({seed} if {seed} < 8.0 else 8.0)")
    em.emit(f"{r} = ({capped} if {capped} > 1.0 else 1.0)")
    em.emit(f"while {r} > 1.0:")
    em.emit(f"{r} = {r} - 1.0", indent=2)
    em.add_float(r)
    em.go_continuous()  # the subtraction is exact (Sterbenz) but the seed select makes the trip count input-dependent
    return r


def _emit_diamond_then_loop(em: _Emitter) -> str:
    """
    A real diamond whose merge feeds a ``while`` loop (a three-arm loop-header phi), generalized from
    ``diamond_then_loop_kernel``. The diamond stays a real branch (unspeculatable division), and the loop strictly
    decreases a bounded carried value. Returns the carried-variable name.
    """
    em.shapes.add(Shape.BRANCH)
    em.shapes.add(Shape.DIAMOND_THEN_LOOP)
    em.shapes.add(Shape.LOOP)
    cond = _emit_condition(em)
    r = em.fresh("dl")
    a = em.pick_float()
    em.emit(f"if {cond}:")
    em.emit(f"{r} = ({a} if {a} < 6.0 else 6.0)", indent=2)
    em.emit("else:")
    em.emit(f"{r} = {a} / {em.nonzero_divisor()}", indent=2)
    # bound the carried value into [1.0, 6.0] before the loop so it terminates and stays small
    bounded_name = em.fresh("dlb")
    em.emit(f"{bounded_name} = ({r} if {r} > 1.0 else 1.0)")
    em.emit(f"{r} = ({bounded_name} if {bounded_name} < 6.0 else 6.0)")
    em.emit(f"while {r} > 1.0:")
    em.emit(f"{r} = {r} - 1.0", indent=2)
    em.add_float(r)
    em.go_continuous()
    return r


def _emit_reduction(em: _Emitter) -> str:
    """An unrolled ``for`` reduction over a literal range (under the unroll threshold). Continuous arithmetic."""
    em.shapes.add(Shape.REDUCTION)
    em.go_continuous()
    n = em.randint(2, 6)
    acc = em.fresh("acc")
    em.emit(f"{acc} = {em.pick_float()}")
    em.emit(f"for _i in range({n}):")
    em.emit(f"{acc} = ({acc} + {em.pick_float()}) * 0.5", indent=2)
    em.add_float(acc)
    return acc


def _emit_bool_logic(em: _Emitter) -> str:
    """
    Cross-domain bool/float logic: build a boolean from a float relation, combine bools with ``and``/``or``/``not``/
    ``^``, cast back to float with ``float(cond)``, and (optionally) ``bool(x)`` a float. Returns a float-variable name
    (the ``float(...)`` of the boolean). Exact (the result is 0.0 or 1.0).
    """
    em.shapes.add(Shape.BOOL_BANK)
    a, b = em.pick_floats(2)
    c1 = em.fresh("c")
    em.emit(f"{c1} = {a} < {b}")
    em.add_bool(c1)
    if em.has_bool() and em.chance(0.6):
        c2 = em.fresh("c")
        connective = em.choice(["and", "or", "^"])
        other = em.pick_bool()
        if connective == "^":
            em.emit(f"{c2} = {c1} ^ {other}")
        else:
            em.emit(f"{c2} = ({c1} {connective} {other})")
        em.add_bool(c2)
        c1 = c2
    out = em.fresh("bf")
    em.emit(f"{out} = float({c1})")
    em.add_float(out)
    return out


def _emit_exact_select(em: _Emitter) -> str:
    """
    A straight-line ternary select whose arms are operands *verbatim* (no arithmetic): ``r = a if c else b`` or the
    sign-flip ``a if c else -a`` (negation is exact). A select routes an operand bit-for-bit with no rounding and no
    underflow, so it is exact in the format at EVERY input including the edge bits -- a genuine full-pipeline float64
    oracle over the select path, with zero false-failure risk. Stays in EXACT mode (it never demotes). Returns the
    routed-value name. The condition is a float relation (or a boolean connective when bool inputs exist), so this is
    an if-converted *select*, not a branch -- complementary to the real diamonds, exercising the select path directly.
    """
    cond = _emit_condition(em)
    return em.exact_select_value(cond)


def _emit_exact_clamp(em: _Emitter) -> str:
    """A straight-line clamp into ``[-1, 1]`` via two verbatim-arm selects: exact in the format at every input."""
    return em.exact_clamp_value()


_STATELESS_TEMPLATES: list[Callable[[_Emitter], str]] = [
    lambda em: _emit_diamond(em, nested=False),
    lambda em: _emit_diamond(em, nested=True),
    _emit_dead_arm_spill,
    _emit_const_branch,
    _emit_bounded_loop,
    _emit_diamond_then_loop,
    _emit_reduction,
    _emit_bool_logic,
    _emit_exact_select,
    _emit_exact_clamp,
]


# --------------------------------------------------------------------------------------------------------------------
# Top-level kernel assembly.
# --------------------------------------------------------------------------------------------------------------------


def _seed_rng(master_seed: int, index: int) -> np.random.Generator:
    """An independent generator for kernel ``index`` of campaign ``master_seed`` -- reproducible regardless of order."""
    return np.random.default_rng(np.random.SeedSequence([master_seed, index]))


def _make_emitter(rng: np.random.Generator, params: list[str], bool_params: set[str]) -> _Emitter:
    em = _Emitter(rng)
    em.reset_lines()
    for p in params:
        if p in bool_params:
            em.add_bool(p)
        else:
            em.add_float(p)
    return em


def generate_stateless_kernel(name: str, master_seed: int, index: int) -> GeneratedKernel:
    """
    Render one stateless kernel: 2-4 float (and occasionally bool) inputs, a sequence of 1-3 danger-shape fragments, and
    a return of a combination of the live values. The shapes realized and the numerical mode are read off the emitter.
    """
    rng = _seed_rng(master_seed, index)
    n_float = int(rng.integers(2, 5))
    use_bool = bool(rng.random() < 0.35)
    float_params = [chr(ord("a") + i) for i in range(n_float)]
    bool_params = [f"c{i}" for i in range(int(rng.integers(1, 3)))] if use_bool else []
    params = float_params + bool_params
    bool_set = set(bool_params)
    em = _make_emitter(rng, params, bool_set)

    n_fragments = int(rng.integers(1, 4))
    produced: list[str] = []
    for _ in range(n_fragments):
        template = rng.choice(_STATELESS_TEMPLATES)  # type: ignore[arg-type]
        produced.append(template(em))

    # The return value combines the produced fragments (and possibly other live floats) into one or a small tuple of
    # floats. Combining via ``+`` keeps every produced value LIVE so DCE cannot drop the diamonds/divisions feeding it.
    _emit_return(em, produced)

    source = _assemble_function(name, params, bool_set, em.render_body(), em.return_line)
    module = _render_module(name, source)
    kernel = getattr(module, name)
    return GeneratedKernel(
        name=name,
        source=source,
        filename=module.__file__ or "",
        callable=kernel,
        shapes=frozenset(em.shapes),
        mode=em.mode,
        input_names=params,
        bool_inputs=frozenset(bool_set),
        seed=master_seed,
        index=index,
        is_stateful=False,
    )


def generate_dead_arm_kernel(name: str, master_seed: int, index: int) -> GeneratedKernel:
    """
    A kernel consisting SOLELY of the dead-arm-spill shape, so no other fragment dilutes its block structure and the
    wide chain reliably spills past the overlapped entry terminator into the dead arm. ``_armed_dead_arm_kernel``
    re-rolls this until the spill is observed, guaranteeing armed overlap coverage independent of the random draw.
    """
    em = _make_emitter(_seed_rng(master_seed, index), ["a", "b", "c"], set())
    result = _emit_dead_arm_spill(em)
    em.return_line = f"return {result}"
    source = _assemble_function(name, ["a", "b", "c"], set(), em.render_body(), em.return_line)
    module = _render_module(name, source)
    return GeneratedKernel(
        name=name,
        source=source,
        filename=module.__file__ or "",
        callable=getattr(module, name),
        shapes=frozenset(em.shapes),
        mode=em.mode,
        input_names=["a", "b", "c"],
        bool_inputs=frozenset(),
        seed=master_seed,
        index=index,
        is_stateful=False,
    )


def _emit_return(em: _Emitter, produced: list[str]) -> None:
    """
    Build the return line, combining every produced fragment so each stays live (DCE cannot drop a diamond feeding the
    output). A single fragment is returned verbatim (exactness-preserving); multiple fragments are returned either as a
    TUPLE of verbatim lanes -- each an output port, still exact -- or as a SUM, which rounds and so demotes the kernel
    to CONTINUOUS mode. A summed return of one fragment does not round (it is the fragment itself).
    """
    live = produced or [em.pick_float()]
    if len(live) == 1:
        em.return_line = f"return {live[0]}"
    elif em.chance(0.5):
        # A small tuple: each lane is one produced value verbatim (kept live), so multiple output ports are exercised
        # without introducing a rounding add -- an EXACT-mode kernel stays exact end to end.
        lanes = ", ".join(live[:3])
        em.return_line = f"return ({lanes},)"
    else:
        em.go_continuous()  # the summing add rounds, so the float64 reference can only be matched within a tolerance
        em.return_line = f"return {' + '.join(live)}"


def _assemble_function(name: str, params: list[str], bool_set: set[str], body: str, return_line: str) -> str:
    """Assemble the full module source for a stateless kernel from its signature, body, and return line."""
    sig = ", ".join(f"{p}: bool" if p in bool_set else p for p in params)
    docstring = '    """A generated fuzz kernel."""'
    return f"def {name}({sig}):\n{docstring}\n{body}\n    {return_line}\n"


# --------------------------------------------------------------------------------------------------------------------
# Stateful kernel assembly: a class with PRIVATE chained slots.
# --------------------------------------------------------------------------------------------------------------------


def generate_stateful_kernel(name: str, master_seed: int, index: int) -> GeneratedKernel:
    """
    Render one stateful kernel: a class with private (underscore) float slots, a ``__call__`` taking 1-2 float inputs
    that advances the slots -- including the chained-slot pattern (``self._a = self._b; self._b = <expr>``), capturing
    one slot's OLD value while another advances -- and returns a scalar combining the live-in slot values with the
    inputs (a structurally-nonzero divisor keeps it well-defined). The slots are PRIVATE so no ``state_*`` output port
    is created (a public attribute would break the float64 comparison).
    """
    rng = _seed_rng(master_seed, index)
    n_inputs = int(rng.integers(1, 3))
    inputs = [chr(ord("x") + i) for i in range(n_inputs)]
    n_slots = int(rng.integers(2, 4))
    slots = [f"_s{i}" for i in range(n_slots)]
    resets = [float(rng.choice([0.0, 1.0, -1.0, 0.5, 2.0])) for _ in slots]

    em = _Emitter(rng)
    em.reset_lines()
    for p in inputs:
        em.add_float(p)
    for s in slots:
        em.add_float(f"self.{s}")  # the slot's CURRENT (live-in) value is a readable float

    # Capture every slot's OLD value first (so updates that reference each other use the live-in, like the chained
    # pattern) -- the frontend's parallel slot semantics make this faithful, but binding locals keeps the source clear.
    olds = {s: em.fresh("old") for s in slots}
    for s in slots:
        em.emit(f"{olds[s]} = self.{s}")
        em.add_float(olds[s])

    # The chained-slot pattern: slot i captures slot (i+1)'s old value; the last slot advances from an input.
    for i, s in enumerate(slots):
        if i + 1 < n_slots and em.chance(0.6):
            em.emit(f"self.{s} = {olds[slots[i + 1]]}")  # chained: ``self._a = old of self._b``
        else:
            em.emit(f"self.{s} = {em.pick_float()} + {em.small_literal()}")
            em.go_continuous()

    # Optionally fold a real diamond into the update so a stateful kernel also exercises branchy scheduling. Its result
    # is threaded into the return so the diamond stays LIVE -- otherwise DCE could drop it and erase the branch.
    diamond_result = _emit_diamond(em, nested=False) if em.chance(0.5) else None

    head = diamond_result if diamond_result is not None else em.pick_float()
    em.return_line = f"return ({head}) * 2.0 + ({em.pick_float()} * 1.5) / {em.nonzero_divisor()}"
    em.shapes.add(Shape.STATE)

    source = _assemble_class(name, inputs, slots, resets, em.render_body(base_indent=1), em.return_line)
    module = _render_module(name, source)
    cls = getattr(module, name)

    return GeneratedKernel(
        name=name,
        source=source,
        filename=module.__file__ or "",
        callable=cls().__call__,
        shapes=frozenset(em.shapes),
        mode=em.mode,
        input_names=inputs,
        bool_inputs=frozenset(),
        seed=master_seed,
        index=index,
        is_stateful=True,
        fresh_reference=lambda: cls().__call__,
    )


def _assemble_class(
    name: str, inputs: list[str], slots: list[str], resets: list[float], body: str, return_line: str
) -> str:
    """Assemble the full module source for a stateful kernel: a class with literal-initialized private slots."""
    init_lines = "\n".join(f"        self.{slot} = {reset!r}" for slot, reset in zip(slots, resets))
    sig = ", ".join(inputs)
    return (
        f"class {name}:\n"
        f'    """A generated stateful fuzz kernel with private chained slots."""\n\n'
        f"    def __init__(self):\n"
        f"{init_lines}\n\n"
        f"    def __call__(self, {sig}):\n"
        f"{body}\n"
        f"        {return_line}\n"
    )


def generate_kernel(master_seed: int, index: int) -> GeneratedKernel:
    """
    Render kernel ``index`` of a campaign: stateful ~30% of the time, stateless otherwise. Reproducible by ``(seed,
    index)`` regardless of order.
    """
    name = f"fuzz_k_{master_seed:x}_{index}"
    rng = np.random.default_rng(np.random.SeedSequence([master_seed, index, 0x5A5A]))
    if rng.random() < 0.3:
        return generate_stateful_kernel(name, master_seed, index)
    return generate_stateless_kernel(name, master_seed, index)


# --------------------------------------------------------------------------------------------------------------------
# The differential runner.
# --------------------------------------------------------------------------------------------------------------------


class CheckKind(Enum):
    """Which differential check fired on a divergence (recorded into the saved reproducer)."""

    INTERP_VS_MODEL = "interp_vs_model"  # bit-exact primary oracle; a failure indicts the LIR layer
    MODEL_VS_FLOAT64 = "model_vs_float64"  # gross-net secondary; a failure may indict the front/mid-end or operators


@dataclass(frozen=True, slots=True)
class Divergence:
    """A captured differential failure, enough to reconstruct and replay it."""

    kernel: GeneratedKernel
    op_label: str
    effort: str
    check: CheckKind
    vectors: list[Vector]  # the full ordered vector sequence up to and including the failing one
    detail: str


class DangerShapeLost(AssertionError):
    """A kernel intended to branch compiled to a single block: if-conversion silently degraded a diamond to a select."""


@dataclass
class CampaignStats:
    """Running totals over a campaign, including the shape histogram (kernel counts per realized shape)."""

    kernels: int = 0
    vectors: int = 0
    stateful: int = 0
    exact_mode: int = 0
    secondary_checked: int = 0
    secondary_skipped: int = 0
    # CONTINUOUS finite out-of-tolerance vectors: surfaced for visibility but never failed (the float64 tolerance is not
    # a sound bound on continuous ZKF arithmetic). A large count would hint at a too-tight tolerance or a real bug.
    continuous_drift: int = 0
    dead_arm_forced: int = 0  # ARMED dead-arm-only kernels run (the overlap-hazard guarantee; see run_campaign)
    shape_counts: dict[Shape, int] = field(default_factory=lambda: {shape: 0 for shape in Shape})
    divergences: list[Divergence] = field(default_factory=list)

    def record_kernel(self, kernel: GeneratedKernel) -> None:
        self.kernels += 1
        if kernel.is_stateful:
            self.stateful += 1
        if kernel.mode is Mode.EXACT:
            self.exact_mode += 1
        for shape in kernel.shapes:
            self.shape_counts[shape] += 1


def _build_all(fn: Callable[..., object], ops: OpConfig, name: str) -> tuple[Mir, NumericalSimulator, MirInterpreter]:
    """
    Lower the kernel ONCE to MIR, then build both the numerical model (downstream of ``build``) and the interpreter
    (off the same MIR), returning the MIR too so the caller can assert on its block structure. Building both from the
    same ``mir`` is what makes the survival assert sound without double-lowering or touching internals. Shared by the
    campaign runner and the regression replayer, so both drive the identical build path.
    """
    mir = lower_to_mir(optimize(lower_frontend(fn)), ops)
    model = generate(build(mir, name)).elaborate()
    interpreter = MirInterpreter(mir)
    return mir, model, interpreter


def _has_overlap_spill(lir: Lir) -> bool:
    """
    Whether any value computed in a block spills past that block's (overlap-shrunk) terminator into a successor frame --
    the observable signature that the cross-block overlap machinery engaged. For a dead-arm kernel this means the wide
    chain landed in the arms, so a dropped inflight reservation could clobber a co-allocated value; a kernel with no
    such spill generated the shape WITHOUT arming the hazard.
    """
    for block in lir.blocks:
        term_pc = lir.term_pc(block)
        for op in (*block.ops, *block.inline_ops):
            for write in op.writes:
                if any(pc > term_pc for pc in lir.write_landing_pcs(block, write.dst, op.commit_cycle)):
                    return True
    return False


def _branch_successors(terminator: MirTerminator) -> tuple[int, ...]:
    match terminator:
        case MirBranch(if_true=if_true, if_false=if_false):
            return (if_true, if_false)
        case MirJump(target=target):
            return (target,)
        case _:
            return ()


def surviving_forward_branches(mir: Mir) -> int:
    """
    The number of FORWARD (non-loop-header) ``MirBranch`` terminators. A loop header is the target of a BACK EDGE -- an
    edge ``b -> s`` where ``s`` DOMINATES ``b`` (every path from the entry to ``b`` runs through ``s``) -- so its branch
    is the loop test, not a diamond, and is excluded. Using DOMINATORS (not mere reachability, which flags every block
    in a cycle, nor block-layout position, which is not guaranteed reverse-postorder) makes a degraded diamond drop the
    count even when a co-occurring loop -- or a diamond NESTED inside a loop -- keeps a branch terminator alive, without
    mis-excluding any forward branch.
    """
    succ = {block.id: _branch_successors(block.terminator) for block in mir.blocks}
    preds: dict[int, list[int]] = {block.id: [] for block in mir.blocks}
    for block in mir.blocks:
        for target in succ[block.id]:
            preds[target].append(block.id)
    ids = [block.id for block in mir.blocks]
    dom = {i: set(ids) for i in ids}
    dom[mir.entry] = {mir.entry}
    changed = True
    while changed:  # classic iterative dominators; the small generated CFGs converge in a couple of passes
        changed = False
        for i in ids:
            if i == mir.entry:
                continue
            doms = {i} | (set.intersection(*(dom[p] for p in preds[i])) if preds[i] else set())
            if doms != dom[i]:
                dom[i] = doms
                changed = True
    loop_headers = {target for block in mir.blocks for target in succ[block.id] if target in dom[block.id]}
    return sum(1 for block in mir.blocks if isinstance(block.terminator, MirBranch) and block.id not in loop_headers)


def _assert_danger_survived(kernel: GeneratedKernel, mir: Mir, op_label: str) -> None:
    """
    A branchy kernel must keep a real FORWARD (diamond) branch after compilation. The generated diamonds are built
    un-if-convertible (an unspeculatable division with a structurally-nonzero divisor in one arm), so a silent
    degradation to a straight-line ``select`` should not happen -- this is a loud safety net against an unexpected
    if-conversion change. Counting only FORWARD branches (excluding loop-header back-edge branches) is robust to a
    co-occurring loop, which would otherwise keep a branch terminator alive even if every diamond degraded away.
    """
    if Shape.BRANCH not in kernel.shapes:
        return
    if surviving_forward_branches(mir) == 0:
        raise DangerShapeLost(
            f"{kernel.name} [{op_label}] was generated as branchy (shapes={sorted(s.name for s in kernel.shapes)}) "
            f"but compiled with NO forward branch across {len(mir.blocks)} block(s): every diamond degraded to a "
            f"select/straight-line, so it exercises no branch or overlap hazard"
        )


def _port_vector(model: NumericalSimulator, fmt: FloatFormat, values: dict[str, float | bool]) -> Vector:
    """Build a positional input vector in the model's port order from a name->value mapping."""
    vector: Vector = []
    for port in model.inputs:
        raw = values[port.name]
        if isinstance(port.scalar_type, BoolType):
            vector.append(bool(raw))
        else:
            vector.append(FloatValue.from_float(fmt, float(raw)))
    return vector


def _vector_from_bits(model: NumericalSimulator, fmt: FloatFormat, bits: dict[str, int]) -> Vector:
    """Build a positional input vector from a name->ZKF-bits mapping (the saved-reproducer encoding)."""
    vector: Vector = []
    for port in model.inputs:
        raw = bits[port.name]
        if isinstance(port.scalar_type, BoolType):
            vector.append(bool(raw))
        else:
            vector.append(FloatValue.from_bits(fmt, raw))
    return vector


def _draw_vectors(
    kernel: GeneratedKernel, model: NumericalSimulator, fmt: FloatFormat, rng: np.random.Generator, count: int
) -> list[Vector]:
    """
    The vector sequence for one (kernel, op-config) run: bounded random draws (small magnitudes so loop trip counts
    stay small and most operations stay in range) plus a handful of format-edge-derived exact patterns. Stateful
    kernels consume the sequence in order on one persistent model + interpreter.
    """
    edges = format_edge_bits(fmt)
    vectors: list[Vector] = []
    # A dead-arm kernel branches on an inequality of two distinct float inputs (or a bool); over few random vectors the
    # hazardous arm might never be taken, hiding a clobber. Two DIRECTED vectors -- floats ASCENDING by port order with
    # bools True, and floats DESCENDING with bools False -- take OPPOSITE arms of any such condition, guaranteeing the
    # dead arm is exercised regardless of the vector budget.
    if Shape.DEAD_ARM_SPILL in kernel.shapes:
        float_ports = [p.name for p in model.inputs if not isinstance(p.scalar_type, BoolType)]
        for ascending in (True, False):
            magnitudes = range(1, len(float_ports) + 1) if ascending else range(len(float_ports), 0, -1)
            directed: dict[str, float | bool] = {
                p.name: ascending for p in model.inputs if isinstance(p.scalar_type, BoolType)
            }
            directed.update({name: float(mag) for name, mag in zip(float_ports, magnitudes)})
            vectors.append(_port_vector(model, fmt, directed))
    for _ in range(count):
        values: dict[str, float | bool] = {}
        for port in model.inputs:
            if isinstance(port.scalar_type, BoolType):
                values[port.name] = bool(rng.integers(0, 2))
            else:
                values[port.name] = float(rng.uniform(-3.0, 3.0))
        vectors.append(_port_vector(model, fmt, values))
    # A few edge-bit vectors: each float port gets a random legal edge magnitude; bool ports a random bit. These stress
    # interp==model at the format extremes (where the secondary float64 check legitimately skips).
    for _ in range(max(2, count // 4)):
        bits: dict[str, int] = {}
        for port in model.inputs:
            if isinstance(port.scalar_type, BoolType):
                bits[port.name] = int(rng.integers(0, 2))
            else:
                bits[port.name] = int(rng.choice(edges))
        vectors.append(_vector_from_bits(model, fmt, bits))
    return vectors


def _reference_outputs(
    fn: Callable[..., object], fmt: FloatFormat, input_names: list[str], bool_inputs: frozenset[str], vector: Vector
) -> list[float | bool] | None:
    """
    Evaluate the float64 reference for one vector, or ``None`` if it raises (ZeroDivisionError/OverflowError/ValueError)
    -- in which case the secondary check skips this vector. Float inputs are decoded from their exact ZKF bits so the
    reference sees the same value the DUT received.
    """
    kwargs: dict[str, float | bool] = {}
    for name, value in zip(input_names, vector, strict=True):
        if name in bool_inputs:
            assert isinstance(value, bool)
            kwargs[name] = value
        else:
            assert isinstance(value, FloatValue)
            kwargs[name] = float(value)
    try:
        result = fn(**kwargs)
    except ZeroDivisionError, OverflowError, ValueError:
        return None
    return [value for _, value in flatten_value(result)]


class _SecondaryResult(Enum):
    """The outcome of comparing one vector's model output against the float64 reference, leaf by leaf."""

    PASS = auto()  # every lane finite and within tolerance (EXACT: bit-exact)
    SKIP_NONFINITE = auto()  # a lane was inf/nan on either side (a datapath property, not a miscompile) -> not compared
    FAIL = auto()  # a finite FLOAT lane fell outside tolerance: precision drift in CONTINUOUS mode, a real bug in EXACT
    STRUCTURAL_FAIL = (
        auto()
    )  # arity / lane type / a bool lane disagreed: a structural miscompile, NEVER precision drift


def _secondary_ok(
    mode: Mode,
    model_out: list[FloatValue | bool],
    reference: list[float | bool],
    fmt: FloatFormat,
    op_count: int,
) -> tuple[_SecondaryResult, str]:
    """
    Compare the model output against the float64 reference, leaf by leaf: a bool lane must match exactly, a float lane
    within a format tolerance (EXACT mode: rtol=atol=0). A non-finite reference or model leaf is a datapath property
    (a ZKF saturation to infinity that float64 keeps finite, say), so the vector is reported ``SKIP_NONFINITE`` rather
    than checked. The tolerance is widened by the operation count.
    """
    if len(model_out) != len(reference):
        return _SecondaryResult.STRUCTURAL_FAIL, f"output arity differs: model {len(model_out)} vs ref {len(reference)}"
    if mode is Mode.EXACT:
        # EXACT-mode bit-exactness (rtol=atol=0, a finite mismatch is a real bug) is sound ONLY if every finite ZKF
        # value is exactly representable in float64 -- i.e. the format fits float64's 53-bit mantissa and 11-bit
        # exponent. A wider format would round in float64 and turn legitimate precision drift into a spurious
        # "miscompile" that wedges the campaign and the saved regressions red; guard the assumption so widening the
        # fuzz format cannot silently break the EXACT oracle.
        assert fmt.wman <= 53 and fmt.wexp <= 11, f"EXACT mode requires a float64-representable format, got {fmt}"
        rtol = atol = 0.0
    else:
        rtol, atol = default_tolerance(fmt, op_count, magnitude=64.0)
    nonfinite = False
    tolerance_fail: str | None = None  # remembered, not returned early, so a later STRUCTURAL failure takes precedence
    for lane, (m, r) in enumerate(zip(model_out, reference, strict=True)):
        if isinstance(m, bool) != isinstance(r, bool):
            # A lane's type must match the source's: a bool output lowered as a float (or vice versa) is a real
            # miscompile that coercing both sides with ``bool(...)`` would mask (model True vs reference 1.0 compare ==).
            return (
                _SecondaryResult.STRUCTURAL_FAIL,
                f"lane {lane}: type mismatch -- model {type(m).__name__} vs reference {type(r).__name__}",
            )
        if isinstance(m, bool):
            if m != bool(r):
                return _SecondaryResult.STRUCTURAL_FAIL, f"bool lane {lane}: model {m} vs reference {r}"
            continue
        mf = float(m)
        if math.isinf(mf) or math.isnan(mf) or math.isinf(r) or math.isnan(r):
            # Not comparable; defer the verdict. A non-finite lane -- INCLUDING the model saturating to inf where the
            # float64 reference stays finite -- is SKIPPED, not failed: ZKF's narrower range legitimately saturates on
            # overflow, including INTERMEDIATE overflow (``a*b`` exceeds the format yet the true result fits), so a
            # model-inf/finite-reference lane cannot be distinguished from a real shared inf-miscompile without false
            # positives. That narrow shared-upstream class is backstopped by the operator unit tests and the example
            # suite (same casts/operators vs Python), not by this best-effort float64 net.
            nonfinite = True
            continue
        if tolerance_fail is None and not within(mf, r, rtol, atol):
            # Remember the first float tolerance miss but KEEP scanning -- a STRUCTURAL failure on a LATER lane must take
            # precedence, since in CONTINUOUS mode a tolerance FAIL is suppressed as drift and would otherwise mask it.
            tolerance_fail = f"float lane {lane}: model {mf!r} vs reference {r!r} (rtol={rtol:g}, atol={atol:g})"
    if nonfinite:
        # A non-finite lane voids the whole vector BEFORE a tolerance miss is reported: a finite lane derived from a
        # saturated intermediate (e.g. ``1/inf``) can drift spuriously, so a saturation must SKIP, not FAIL. (STRUCTURAL
        # failures already returned above -- they are real miscompiles independent of any saturation.)
        return _SecondaryResult.SKIP_NONFINITE, ""
    if tolerance_fail is not None:
        return _SecondaryResult.FAIL, tolerance_fail
    return _SecondaryResult.PASS, ""


@dataclass(frozen=True, slots=True)
class _SecondaryOutcome:
    """The result of the secondary float64 check for one vector: whether it ran, and any finite EXACT-mode mismatch."""

    checked: bool  # True if the float64 reference was finite and the lanes were compared
    latch_off: bool  # True if a stateful kernel's float64 reference has drifted -> stop the secondary for the sequence
    exact_failure: str | None  # a finite EXACT-mode divergence detail (a real bug), else None
    continuous_drift: bool = False  # a CONTINUOUS finite out-of-tolerance miss: surfaced for visibility, never failed


class _Differential:
    """
    The per-vector differential engine shared by the campaign runner and the regression replayer: one model + one
    interpreter built from the same MIR, plus the float64 reference function and the kernel's numerical mode.
    ``primary`` returns the bit-exact interp-vs-model detail (or None), and ``secondary`` runs the best-effort float64
    net with the stateful-latch rule -- so both call sites drive the identical oracle and cannot drift.
    """

    def __init__(
        self,
        model: NumericalSimulator,
        interpreter: MirInterpreter,
        reference_fn: Callable[..., object],
        mode: Mode,
        is_stateful: bool,
        input_names: list[str],
        bool_inputs: frozenset[str],
        fmt: FloatFormat,
        op_count: int,
    ) -> None:
        self.model = model
        self.interpreter = interpreter
        self._reference_fn = reference_fn
        self._mode = mode
        self._is_stateful = is_stateful
        self._input_names = input_names
        self._bool_inputs = bool_inputs
        self._fmt = fmt
        self._op_count = op_count

    def primary(self, vector: Vector) -> tuple[list[FloatValue | bool], str | None]:
        """Run both engines and return the model output plus the interp-vs-model mismatch detail (None when equal)."""
        model_out = self.model.run(*vector)
        interp_out = self.interpreter.run(*vector)
        if model_out == interp_out:
            return model_out, None
        detail = (
            f"inputs {[show_value(v) for v in vector]}: "
            f"model {[show_value(v) for v in model_out]} vs interpreter {[show_value(v) for v in interp_out]}"
        )
        return model_out, detail

    def secondary(self, vector: Vector, model_out: list[FloatValue | bool]) -> _SecondaryOutcome:
        """
        Run the best-effort float64 check for one vector, applying the stateful latch and EXACT-mode failure rule. A
        CONTINUOUS finite out-of-tolerance mismatch is NOT reported as a divergence: ``default_tolerance`` is a linear
        heuristic, not a sound bound on ZKF-vs-float64 divergence (catastrophic cancellation makes the relative error
        unbounded), so a finite continuous miss is treated as expected precision drift, not a miscompile -- the EXACT
        mode and the bit-exact interp==model check are the sound oracles. Any skip (a raise or a non-finite lane) that
        could desync a stateful kernel's reference state latches the secondary off for the rest of its sequence.
        """
        reference = _reference_outputs(self._reference_fn, self._fmt, self._input_names, self._bool_inputs, vector)
        if reference is None:
            # The float64 reference raised; a stateful kernel's reference state is now out of step, so latch off.
            return _SecondaryOutcome(checked=False, latch_off=self._is_stateful, exact_failure=None)
        result, detail = _secondary_ok(self._mode, model_out, reference, self._fmt, self._op_count)
        match result:
            case _SecondaryResult.PASS:
                return _SecondaryOutcome(checked=True, latch_off=False, exact_failure=None)
            case _SecondaryResult.SKIP_NONFINITE:
                # A non-finite lane (ZKF saturation) was not compared; a stateful reference may now have drifted.
                return _SecondaryOutcome(checked=False, latch_off=self._is_stateful, exact_failure=None)
            case _SecondaryResult.STRUCTURAL_FAIL:
                # Arity / lane-type / bool-lane disagreement is a STRUCTURAL miscompile, never precision drift, so it is
                # ALWAYS a reported divergence -- including in CONTINUOUS mode, where a finite float miss is suppressed.
                return _SecondaryOutcome(checked=False, latch_off=False, exact_failure=detail)
            case _SecondaryResult.FAIL:
                # A finite EXACT-mode float mismatch is a real bug; a CONTINUOUS finite float miss is expected precision
                # drift that only latches the secondary off for a stateful kernel (never a reported divergence).
                if self._mode is Mode.EXACT:
                    return _SecondaryOutcome(checked=False, latch_off=False, exact_failure=detail)
                return _SecondaryOutcome(
                    checked=False, latch_off=self._is_stateful, exact_failure=None, continuous_drift=True
                )


def run_kernel(
    kernel: GeneratedKernel,
    op_label: str,
    ops: OpConfig,
    fmt: FloatFormat,
    effort: str,
    n_vectors: int,
    stats: CampaignStats,
    expect_armed: bool = False,
) -> Divergence | None:
    """
    Drive one (kernel, op-config) through the differential runner over an ordered vector sequence on ONE model + ONE
    interpreter (reset state shared). The model is built ONCE here and the vectors are drawn from it (so the campaign
    compiles the kernel exactly once per op-config). Returns the first :class:`Divergence` found (its vector prefix
    captured), or ``None`` if every check passed. The primary interp==model check is unconditional and never skipped;
    the secondary float64 check is best-effort and latches off permanently for a stateful kernel once the float64
    reference diverges.
    """
    name = f"{kernel.name}__{op_label}"
    mir, model, interpreter = _build_all(kernel.callable, ops, name)
    _assert_danger_survived(kernel, mir, op_label)
    if expect_armed:
        assert _has_overlap_spill(
            model._lir
        ), f"{name}: dead-arm kernel did not spill -- the overlap hazard is not armed"

    assert [p.name for p in model.inputs] == [p.name for p in interpreter.inputs], f"{name}: input ports differ"
    # The float64 reference binds by PARAMETER NAME against a vector drawn in model-port order, so a frontend bug that
    # swapped the param->port mapping and the port order together would feed model, interpreter, AND reference the same
    # wrong name->value mapping and pass vacuously. Pinning the port order to the source parameter order closes that.
    assert [p.name for p in model.inputs] == list(
        kernel.input_names
    ), f"{name}: model input ports {[p.name for p in model.inputs]} differ from kernel params {list(kernel.input_names)}"
    assert [p.name for p in model.outputs] == [p.name for p in interpreter.outputs], f"{name}: output ports differ"
    # The vector sequence is keyed by (master_seed, index) so it is identical across op-configs and reproducible; drawn
    # from the just-built model's port order, so no throwaway compile is needed.
    vectors = _draw_vectors(kernel, model, fmt, _seed_rng(kernel.seed, kernel.index), n_vectors)

    reference_fn = kernel.fresh_reference() if kernel.fresh_reference is not None else kernel.callable
    diff = _Differential(
        model,
        interpreter,
        reference_fn,
        kernel.mode,
        kernel.is_stateful,
        kernel.input_names,
        kernel.bool_inputs,
        fmt,
        len(mir.nodes),
    )
    secondary_live = True  # for a stateful kernel a float64 divergence is permanent -> stop the secondary check

    for position, vector in enumerate(vectors):
        prefix = vectors[: position + 1]
        stats.vectors += 1
        model_out, primary_detail = diff.primary(vector)
        if primary_detail is not None:
            return Divergence(kernel, op_label, effort, CheckKind.INTERP_VS_MODEL, prefix, primary_detail)
        if not secondary_live:
            continue
        outcome = diff.secondary(vector, model_out)
        if outcome.exact_failure is not None:
            return Divergence(kernel, op_label, effort, CheckKind.MODEL_VS_FLOAT64, prefix, outcome.exact_failure)
        stats.secondary_checked += 1 if outcome.checked else 0
        stats.secondary_skipped += 0 if outcome.checked else 1
        stats.continuous_drift += 1 if outcome.continuous_drift else 0
        if outcome.latch_off:
            secondary_live = False
    return None


_ARM_RETRIES = 8  # dead-arm-only kernels arm ~100%, so this re-roll is a guard, not a hot path


def _armed_dead_arm_kernel(master_seed: int, index: int, fmt: FloatFormat) -> GeneratedKernel:
    """
    A dead-arm-only kernel re-rolled until its wide chain actually spills past the overlapped terminator (arming the
    register-reuse hazard). The shape arms reliably, so a re-roll is rare; failing after ``_ARM_RETRIES`` is a loud
    signal that the generator or the overlap machinery regressed.
    """
    for attempt in range(_ARM_RETRIES):
        kernel = generate_dead_arm_kernel(f"deadarm_{index}_{attempt}", master_seed, index * 64 + attempt)
        # Verify the chain spills under EVERY op-config the forced batch asserts (``expect_armed``), not just the
        # default -- so a future config under which the chain happens not to spill cannot false-fail the campaign.
        builds = (_build_all(kernel.callable, make_ops(fmt), kernel.name)[1] for make_ops in OP_CONFIGS.values())
        if all(_has_overlap_spill(model._lir) for model in builds):
            return kernel
    raise DangerShapeLost(f"dead-arm generator failed to arm a spill in {_ARM_RETRIES} tries (index {index})")


def run_campaign(
    n_kernels: int,
    n_vectors: int,
    master_seed: int,
    effort: str,
    fmt: FloatFormat,
    on_divergence: Callable[[Divergence], None],
) -> CampaignStats:
    """
    Run a full campaign: generate ``n_kernels`` kernels, sweep each across :data:`OP_CONFIGS`, and drive ``n_vectors``
    vectors through each. ``on_divergence`` is invoked with the FIRST divergence per (kernel, op-config) -- the caller
    saves a reproducer and decides whether to fail. Returns the accumulated stats.
    """
    stats = CampaignStats()
    # A dedicated batch of ARMED dead-arm-only kernels exercises the overlap register-reuse hazard every campaign,
    # independent of the random draw -- each re-rolled until its wide chain actually spills, then run with a hard arming
    # assert, so a future change that silently stops the chain spilling fails loudly.
    for j in range(max(4, n_kernels // 7)):
        forced = _armed_dead_arm_kernel(master_seed, j, fmt)
        stats.dead_arm_forced += 1
        stats.record_kernel(forced)
        for op_label, make_ops in OP_CONFIGS.items():
            divergence = run_kernel(forced, op_label, make_ops(fmt), fmt, effort, n_vectors, stats, expect_armed=True)
            if divergence is not None:
                stats.divergences.append(divergence)
                on_divergence(divergence)
    for index in range(n_kernels):
        kernel = generate_kernel(master_seed, index)
        stats.record_kernel(kernel)
        for op_label, make_ops in OP_CONFIGS.items():
            divergence = run_kernel(kernel, op_label, make_ops(fmt), fmt, effort, n_vectors, stats)
            if divergence is not None:
                stats.divergences.append(divergence)
                on_divergence(divergence)
    return stats


# --------------------------------------------------------------------------------------------------------------------
# Reproducer persistence (a self-contained replayable case under tests/fuzz_regressions/).
# --------------------------------------------------------------------------------------------------------------------

REGRESSIONS_DIR = Path(__file__).resolve().parent / "fuzz_regressions"


@dataclass(frozen=True, slots=True)
class ReproMeta:
    """
    The typed metadata of a saved reproducer: everything needed to rebuild and replay the failing case. It serializes
    to a plain ``dict`` literal (so the saved module imports nothing) via :meth:`to_dict`, and parses back at a single
    boundary via :meth:`from_dict` -- so the rest of the code works with this strongly-typed record, not a stringly
    -keyed dict.
    """

    kernel_name: str
    op_label: str
    effort: str
    check: CheckKind
    mode: Mode
    is_stateful: bool
    input_names: list[str]
    bool_inputs: frozenset[str]
    fmt: FloatFormat
    seed: int
    index: int
    shapes: list[str]
    detail: str
    vectors_bits: list[dict[str, int]]

    def to_dict(self) -> dict[str, object]:
        """The serializable dict literal embedded in the saved reproducer (enums as names/values, the format split)."""
        return {
            "kernel_name": self.kernel_name,
            "op_label": self.op_label,
            "effort": self.effort,
            "check": self.check.value,
            "mode": self.mode.name,
            "is_stateful": self.is_stateful,
            "input_names": self.input_names,
            "bool_inputs": sorted(self.bool_inputs),
            "wexp": self.fmt.wexp,
            "wman": self.fmt.wman,
            "seed": self.seed,
            "index": self.index,
            "shapes": self.shapes,
            "detail": self.detail,
            "vectors_bits": self.vectors_bits,
        }

    @classmethod
    def from_dict(cls, meta: Mapping[str, Any]) -> "ReproMeta":
        """Parse a saved ``META`` dict into the typed record -- the single place the on-disk format is decoded."""
        return cls(
            kernel_name=str(meta["kernel_name"]),
            op_label=str(meta["op_label"]),
            effort=str(meta["effort"]),
            check=CheckKind(str(meta["check"])),
            mode=Mode[str(meta["mode"])],
            is_stateful=bool(meta["is_stateful"]),
            input_names=[str(name) for name in meta["input_names"]],
            bool_inputs=frozenset(str(name) for name in meta["bool_inputs"]),
            fmt=FloatFormat(int(meta["wexp"]), int(meta["wman"])),
            seed=int(meta["seed"]),
            index=int(meta["index"]),
            shapes=[str(name) for name in meta["shapes"]],
            vectors_bits=[{str(k): int(v) for k, v in row.items()} for row in meta["vectors_bits"]],
            detail=str(meta["detail"]),
        )


def save_reproducer(divergence: Divergence, fmt: FloatFormat) -> Path:
    """
    Write a self-contained reproducer for a divergence into :data:`REGRESSIONS_DIR`: the rendered kernel source, the
    failing input vectors as exact ZKF bits, the op-config label, the triggering effort, the format, and which check
    failed. The replayer (``test_fuzz_regressions``) globs these and re-asserts the previously-failing check.
    """
    REGRESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    kernel = divergence.kernel
    meta = ReproMeta(
        kernel_name=kernel.name,
        op_label=divergence.op_label,
        effort=divergence.effort,
        check=divergence.check,
        mode=kernel.mode,
        is_stateful=kernel.is_stateful,
        input_names=kernel.input_names,
        bool_inputs=kernel.bool_inputs,
        fmt=fmt,
        seed=kernel.seed,
        index=kernel.index,
        shapes=sorted(s.name for s in kernel.shapes),
        detail=divergence.detail,
        vectors_bits=[_vector_to_bits(kernel, vector) for vector in divergence.vectors],
    )
    path = REGRESSIONS_DIR / f"{kernel.name}__{divergence.op_label}__{divergence.check.value}.py"
    path.write_text(_render_reproducer(kernel.source, meta.to_dict()), encoding="utf-8")
    return path


def _vector_to_bits(kernel: GeneratedKernel, vector: Vector) -> dict[str, int]:
    """
    Map a positional vector to a name->exact-bits mapping with NO float round-trip: a float port keeps its raw ZKF
    ``bits`` (a ``float()`` decode of a largest-finite value would overflow), a bool port keeps ``int(value)``.
    """
    bits: dict[str, int] = {}
    for name, value in zip(kernel.input_names, vector, strict=True):
        bits[name] = int(value) if isinstance(value, bool) else value.bits
    return bits


def replay_case(kernel_callable: Callable[..., object], meta: dict[str, object]) -> tuple[bool, str]:
    """
    Replay a saved reproducer: rebuild the model + interpreter at the saved op-config (via the same ``_build_all`` path
    the campaign uses), drive the saved bit-vectors in order, and re-check the previously-failing differential check
    through the same ``_Differential`` engine. Returns ``(passes_now, detail)`` -- a fixed bug means ``passes_now is
    True``. The regression replayer calls this in a subprocess pinned to the saved regalloc effort (which is import
    -frozen), so the caller is responsible for having set the right effort before importing this module.
    """
    repro = ReproMeta.from_dict(meta)
    ops = OP_CONFIGS[repro.op_label](repro.fmt)
    mir, model, interpreter = _build_all(kernel_callable, ops, f"{repro.kernel_name}__replay")
    if [p.name for p in model.inputs] != list(repro.input_names):  # the reference binds by name in port order; pin it
        return (
            False,
            f"input port order changed on replay: {[p.name for p in model.inputs]} vs {list(repro.input_names)}",
        )
    # Re-assert the danger shape survived: a saved branchy case that now if-converts to a single block would pass the
    # differential replay vacuously while no longer testing the branch -- the same loud-failure rule as the campaign.
    if "BRANCH" in repro.shapes and surviving_forward_branches(mir) == 0:
        return (
            False,
            f"branch survival lost on replay: no forward branch across {len(mir.blocks)} block(s) (was branchy)",
        )
    diff = _Differential(
        model,
        interpreter,
        kernel_callable,
        repro.mode,
        repro.is_stateful,
        repro.input_names,
        repro.bool_inputs,
        repro.fmt,
        len(mir.nodes),
    )
    secondary_live = True
    for bit_row in repro.vectors_bits:
        vector = _vector_from_bits(model, repro.fmt, bit_row)
        model_out, primary_detail = diff.primary(vector)
        if repro.check is CheckKind.INTERP_VS_MODEL:
            if primary_detail is not None:
                return False, primary_detail
            continue
        if primary_detail is not None:
            # The saved failure was the secondary check, but the primary now disagrees too -- still a (worse) failure.
            return False, primary_detail
        if not secondary_live:
            continue
        outcome = diff.secondary(vector, model_out)
        if outcome.exact_failure is not None:
            return False, outcome.exact_failure
        if outcome.latch_off:
            secondary_live = False
    return True, ""


def _render_reproducer(source: str, meta: dict[str, object]) -> str:
    """
    Render a saved reproducer module: the kernel source verbatim plus a ``META`` dict literal. Float inputs are stored
    as exact ZKF bits so replay is bit-faithful regardless of the float64 round-trip. The replayer imports ``META`` and
    the kernel symbol by name.
    """
    import pprint

    header = (
        '"""\n'
        "Auto-saved fuzz regression. A previously-found differential divergence, captured as a self-contained\n"
        "replayable case: the kernel source, the failing input vectors as exact ZKF bits, the op-config, the effort,\n"
        "and which differential check failed. Replayed by tests/test_fuzz_regressions.py.\n"
        '"""\n\n'
    )
    return f"{header}{source}\n\nMETA = {pprint.pformat(meta, width=120)}\n"
