"""
End-to-end blackbox differential fuzzing of the Holoso compiler.

The campaign generates small kernels as Python *source text* -- rendered to real importable modules under ``build/`` so
the frontend's ``inspect.getsourcelines`` retrieval succeeds -- and drives each through the public-ish compiler pipeline
twice: once into the numerical model (downstream of LIR scheduling/binding/regalloc/overlap) and once into the MIR
interpreter (the schedule-independent oracle, upstream of the LIR layer). The two share the front/mid-end and
``operator.evaluate`` but NOT the LIR, so the primary check ``interpreter == model`` (bit-exact, exception-free on both
ZKF sides) isolates exactly the LIR layer -- the miscompile class the RTL-versus-model cosimulation is blind to.

The generator emits the *danger shapes by construction* -- real un-if-convertible diamonds (kept branchy by either an
unspeculatable division or an over-budget exact arm), the dead-arm-spill family, wide register-pressure chains, nested
ifs, const-branch-via-division, diamond-then-loop, bounded while loops, cross-domain bool/float logic, selects,
unrolled reductions, exact non-commutative wiring checks, and stateful classes with private chained slots -- because
these are the shapes a naive straight-line fuzzer misses. Every kernel carries a typed shape descriptor used both for
the campaign histogram and to assert the danger shape *survived* compilation (a diamond silently degrading to a select
must fail loudly, else the kernel tests nothing).

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
from holoso._hir import _if_convert as if_convert_pass
from holoso._hir import optimize
from holoso._lir import Branch, InlineScheduledOp, Lir, RegRef, ScheduledOp, build, inline_landing_cycle
from holoso._lir import operand_read_cycle, result_landing_cycle
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
    OVERBUDGET_BRANCH = auto()  # a branch kept alive by exceeding the if-conversion arm operation budget
    DEAD_ARM_SPILL = auto()  # the dead-arm-spill soundness family (a value live in one arm, a wide chain spilling both)
    WIDE_CHAIN = auto()  # a long register-pressure multiply-add chain
    NESTED_IF = auto()  # a branch nested inside a branch
    CONST_BRANCH = auto()  # a constant-true inner condition formed by division (an empty const-branch block)
    LOOP = auto()  # a bounded back-edge ``while`` loop
    DIAMOND_THEN_LOOP = auto()  # a real diamond whose merge feeds a loop header (a three-arm header phi)
    SELECT = auto()  # a ternary ``a if c else b`` select
    BOOL_BANK = auto()  # boolean inputs and/or boolean connectives (the cross-domain bool/float bank)
    RELATION_PAIR = auto()  # two comparator taps over the same ordered operand pair
    EXACT_WIRING = auto()  # exact non-commutative arithmetic that exposes operand transposition
    REDUCTION = auto()  # an unrolled ``for`` reduction over a literal range
    STATE = auto()  # persistent private slots (a stateful class), including the chained-slot pattern


class Mode(Enum):
    """The numerical regime of a kernel's float outputs -- it decides the secondary (float64) check's strictness."""

    EXACT = auto()  # every float output is exact in the format: the float64 reference must match bit-for-bit
    CONTINUOUS = auto()  # general +,*,/ : only interpreter==model is exact; the float64 check is a gross net


@dataclass(frozen=True, slots=True)
class GeneratedKernel:
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
    dead_arm_chain_depth: int | None = None


@dataclass(frozen=True, slots=True)
class _Fragment:
    value: str
    mode: Mode = Mode.EXACT
    shapes: frozenset[Shape] = frozenset()
    dead_arm_chain_depth: int | None = None


type _Body = list[tuple[int, str]]
_OVERBUDGET_ARM_OPS = max(1, if_convert_pass._IFCONV_MAX_OPS + 1)  # noqa: SLF001 -- tests track the active knob.
_DEFAULT_EXACT_FMT = FloatFormat(6, 18)


def _fragment(
    value: str,
    mode: Mode = Mode.EXACT,
    shapes: frozenset[Shape] = frozenset(),
    dead_arm_chain_depth: int | None = None,
) -> _Fragment:
    return _Fragment(value, mode, shapes, dead_arm_chain_depth)


def _combine_mode(fragments: list[_Fragment], extra: Mode = Mode.EXACT) -> Mode:
    if extra is Mode.CONTINUOUS or any(fragment.mode is Mode.CONTINUOUS for fragment in fragments):
        return Mode.CONTINUOUS
    return Mode.EXACT


def _combine_shapes(fragments: list[_Fragment], extra: frozenset[Shape] = frozenset()) -> frozenset[Shape]:
    shapes = set(extra)
    for fragment in fragments:
        shapes.update(fragment.shapes)
    return frozenset(shapes)


def _combine_dead_arm_chain_depth(fragments: list[_Fragment]) -> int | None:
    depths = [fragment.dead_arm_chain_depth for fragment in fragments if fragment.dead_arm_chain_depth is not None]
    return max(depths) if depths else None


def _min_normal_exp(fmt: FloatFormat) -> int:
    return 2 - (1 << (fmt.wexp - 1))


def _max_normal_exp(fmt: FloatFormat) -> int:
    return (1 << (fmt.wexp - 1)) - 1


def _power2_literal(exp: int) -> str:
    return repr(math.ldexp(1.0, exp))


def _binary_literal(significand: float, exp: int) -> str:
    assert significand in {1.25, 1.5}
    return repr(math.ldexp(significand, exp))


def _render_module(name: str, source: str) -> types.ModuleType:
    """
    The frontend retrieves the kernel's source via ``inspect.getsourcelines``, which a REPL/exec-only function cannot
    satisfy -- hence the real, linecache-backed file rather than a bare ``exec``.
    """
    FUZZ_TMP.mkdir(parents=True, exist_ok=True)
    path = str((FUZZ_TMP / f"{name}.py").resolve())
    Path(path).write_text(source, encoding="utf-8")
    linecache.cache[path] = (len(source), None, source.splitlines(keepends=True), path)
    module = types.ModuleType(name)
    module.__file__ = path
    exec(compile(source, path, "exec"), module.__dict__)  # noqa: S102 -- generated, sandboxed, gitignored source
    return module


class _Emitter:
    """
    A per-kernel source builder. It tracks the live float/bool variable pools it may read from and accumulates body
    lines under a fixed indentation. Every fragment is bounded and structurally controlled so the resulting kernel
    compiles, terminates, and -- when the returned fragment metadata says EXACT -- stays bit-exact against float64.
    """

    def __init__(self, rng: np.random.Generator, fmt: FloatFormat = _DEFAULT_EXACT_FMT) -> None:
        self._rng = rng
        self._fmt = fmt
        self._floats: list[str] = []
        self._bools: list[str] = []
        self._counter = 0
        self._lines: list[tuple[int, str]] = []
        self.return_line = ""  # set by the assembly step before rendering

    def then_value(self) -> _Fragment:
        if self.chance(0.4):
            return self.scaled_value()
        if self.chance(0.4) and len(self.floats) >= 2:
            return self.exact_select_value(_emit_condition(self))
        return _fragment(self.continuous_expr(2), Mode.CONTINUOUS)

    def else_division(self) -> _Fragment:
        """
        The unspeculatable division that keeps a diamond a real branch: a numerator over a structurally-nonzero
        divisor. This is a general division, so the returned metadata marks the fragment continuous.
        """
        return _fragment(
            f"({self.pick_float()} {self.choice(['+', '*'])} {self.pick_float()}) / {self.nonzero_divisor()}",
            Mode.CONTINUOUS,
        )

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
        return str(self._rng.choice(self._floats))

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
        return str(self._rng.choice(self._bools))

    def chance(self, p: float) -> bool:
        return bool(self._rng.random() < p)

    def choice(self, options: list[str]) -> str:
        return str(self._rng.choice(options))

    def randint(self, lo: int, hi: int) -> int:
        return int(self._rng.integers(lo, hi + 1))

    def exact_scale_shift(self) -> int:
        """
        A nonzero power-of-two scale whose exponent shift can be absorbed by at least one normal operand bin in this
        format without crossing the finite normal exponent range. The scale literal itself is also kept normal, because
        generated constants are represented in the same ZKF format as the datapath.
        """
        min_exp = _min_normal_exp(self._fmt)
        max_exp = _max_normal_exp(self._fmt)
        span = max_exp - min_exp
        max_abs_shift = min(3, span)
        assert max_abs_shift >= 1
        lo = max(-max_abs_shift, min_exp)
        hi = min(max_abs_shift, max_exp)
        shifts = [shift for shift in range(lo, hi + 1) if shift != 0]
        assert shifts
        return int(self._rng.choice(shifts))

    def exact_division_scale_exp(self) -> int:
        return self.randint(1, min(3, _max_normal_exp(self._fmt)))

    def exact_division_denominator_hi_exp(self, scale_exp: int) -> int:
        """The largest denominator exponent whose product with ``2**scale_exp`` stays finite with guard margin."""
        assert 1.5 < 2.0 - (2.0 ** (1 - self._fmt.wman))
        hi_exp = _max_normal_exp(self._fmt) - scale_exp
        assert hi_exp >= 0
        return hi_exp

    def sterbenz_hi_lower_literal(self) -> str:
        """A lower bound for ``hi`` that keeps ``hi / 2.0`` normal while preserving Sterbenz exactness."""
        return _power2_literal(max(_min_normal_exp(self._fmt) + 1, 1))

    def small_literal(self) -> str:
        return self.choice(["1.0", "2.0", "0.5", "0.25", "4.0", "-1.0", "-2.0", "-0.5"])

    def nonneg_expr(self) -> str:
        v = self.pick_float()
        return f"({v} * {v})"

    def nonzero_divisor(self) -> str:
        """A structurally-positive divisor ``(<nonneg>) + c`` with ``c`` a positive literal."""
        c = self.choice(["1.0", "0.5", "2.0"])
        return f"(({self.nonneg_expr()}) + {c})"

    def continuous_expr(self, depth: int) -> str:
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
        z = self.pick_float()
        b = self.pick_float()
        acc = self.pick_float()
        for _ in range(width):
            acc = f"(({acc} * {z}) + {b})"
        return acc

    def exact_select_value(self, cond: _Fragment) -> _Fragment:
        """
        A select whose arms are *operands verbatim* (no arithmetic), so the result is exact in the format. Includes the
        ``x if c else -x`` sign-flip shape (negation is exact).
        """
        a, b = self.pick_float(), self.pick_float()
        if self.chance(0.4):
            value = f"({a} if {cond.value} else -{a})"
        else:
            value = f"({a} if {cond.value} else {b})"
        return _fragment(value, cond.mode, _combine_shapes([cond], frozenset({Shape.SELECT})))

    def scaled_value(self) -> _Fragment:
        """
        A power-of-two scaling bounded away from overflow and underflow, so it is exact at every tested input edge.
        The selected operand magnitude is clamped into an exponent window ``[lo, hi]`` such that adding the shift keeps
        the product within the format's finite normal exponent range.
        """
        x = self.pick_float()
        shift = self.exact_scale_shift()
        lo_exp = max(_min_normal_exp(self._fmt), _min_normal_exp(self._fmt) - shift)
        hi_exp = min(_max_normal_exp(self._fmt), _max_normal_exp(self._fmt) - shift)
        assert lo_exp <= hi_exp
        mag = f"({x} if {x} > 0.0 else -{x})"
        hi = _power2_literal(hi_exp)
        lo = _power2_literal(lo_exp)
        below_hi = f"({mag} if {mag} < {hi} else {hi})"
        bounded = f"({below_hi} if {below_hi} > {lo} else {lo})"
        return _fragment(f"({bounded} * {_power2_literal(shift)})")

    def exact_clamp_value(self) -> _Fragment:
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
        self.add_float(clamped)
        return _fragment(clamped, shapes=frozenset({Shape.SELECT}))

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


def _body(line: str, indent: int = 0) -> tuple[int, str]:
    return indent, line


def _assign_body(target: str, fragment: _Fragment) -> _Body:
    return [_body(f"{target} = {fragment.value}")]


def _if_else_body(cond: str, then_body: _Body, else_body: _Body) -> _Body:
    lines = [_body(f"if {cond}:")]
    lines.extend(_body(line, indent + 1) for indent, line in then_body)
    lines.append(_body("else:"))
    lines.extend(_body(line, indent + 1) for indent, line in else_body)
    return lines


def _emit_if_else(em: _Emitter, cond: str, then_body: _Body, else_body: _Body, indent: int = 1) -> None:
    for extra, line in _if_else_body(cond, then_body, else_body):
        em.emit(line, indent + extra)


def _merge_fragment(value: str, fragments: list[_Fragment], shapes: frozenset[Shape] = frozenset()) -> _Fragment:
    return _fragment(
        value,
        _combine_mode(fragments),
        _combine_shapes(fragments, shapes),
        _combine_dead_arm_chain_depth(fragments),
    )


def _emit_condition(em: _Emitter, *, balanced: bool = False) -> _Fragment:
    """
    A runtime boolean condition: a float relation, or (when bool inputs exist) a boolean connective of them. With
    ``balanced=True`` the float relation is restricted to inequalities, so a branch whose hazard lives in one arm is
    exercised over the vector sequence.
    """
    if em.has_bool() and em.chance(0.4):
        a = em.pick_bool()
        if em.chance(0.5) and len(em.bools) >= 2:
            value = f"({a} {em.choice(['and', 'or'])} {em.pick_bool()})"
        elif em.chance(0.5):
            value = f"(not {a})"
        else:
            value = a
        return _fragment(value, shapes=frozenset({Shape.BOOL_BANK}))
    a, b = em.pick_floats(2)
    relations = ["<", ">", "<=", ">="] if balanced else ["<", ">", "<=", ">=", "==", "!="]
    return _fragment(f"({a} {em.choice(relations)} {b})")


def _overbudget_arm(em: _Emitter, target: str) -> tuple[_Body, _Fragment]:
    """
    An if-conversion barrier independent of division: one arm exceeds the operation budget using exact identity selects,
    so the diamond remains branchy even though every operation in the arm is speculatable.
    """
    seed = em.pick_float()
    current = em.fresh("budget")
    lines = [_body(f"{current} = ({seed} if {seed} > 1.0 else 1.0)")]
    for _ in range(_OVERBUDGET_ARM_OPS):
        nxt = em.fresh("budget")
        lines.append(_body(f"{nxt} = ({current} if {current} >= 1.0 else 1.0)"))
        current = nxt
    lines.append(_body(f"{target} = {current}"))
    return lines, _fragment(target, shapes=frozenset({Shape.OVERBUDGET_BRANCH}))


def _division_arm(em: _Emitter, target: str) -> tuple[_Body, _Fragment]:
    fragment = em.else_division()
    return _assign_body(target, fragment), _fragment(target, fragment.mode, fragment.shapes)


def _survival_arm(em: _Emitter, target: str) -> tuple[_Body, _Fragment]:
    if em.chance(0.5):
        return _division_arm(em, target)
    return _overbudget_arm(em, target)


def _emit_diamond(em: _Emitter, *, nested: bool) -> _Fragment:
    """
    A real diamond: a runtime comparison branches into two arms, one arm carries either an unspeculatable division or
    an over-budget exact chain, each arm assigns the result variable, and the kernel reads it after the merge. When
    nested, the inner branch gets its own barrier arm so the :data:`Shape.NESTED_IF` tag means a surviving nested
    branch, not merely a generated one.
    """
    cond = _emit_condition(em, balanced=True)
    r = em.fresh("r")
    fragments = [cond]
    if nested and len(em.floats) >= 2:
        inner = _emit_condition(em)
        inner_then = em.then_value()
        inner_else_body, inner_else_fragment = _survival_arm(em, r)
        then_body = _if_else_body(inner.value, _assign_body(r, inner_then), inner_else_body)
        fragments.extend([inner, inner_then, inner_else_fragment])
        shapes = frozenset({Shape.BRANCH, Shape.NESTED_IF})
    else:
        then_value = em.then_value()
        then_body = _assign_body(r, then_value)
        fragments.append(then_value)
        shapes = frozenset({Shape.BRANCH})
    else_body, else_fragment = _survival_arm(em, r)
    fragments.append(else_fragment)
    _emit_if_else(em, cond.value, then_body, else_body)
    em.add_float(r)
    return _merge_fragment(r, fragments, shapes)


def _emit_dead_arm_spill(em: _Emitter) -> _Fragment:
    """
    The dead-arm-spill soundness shape: several simultaneously-live values ``v`` computed in the entry block and read
    in only one arm, plus a wide chain ``w`` that commits late and spills into both arms but is dead in the arm that
    reads the ``v`` values.
    """
    live = [em.fresh("v") for _ in range(em.randint(3, 6))]
    for v in live:
        em.emit(f"{v} = {em.pick_float()} * {em.pick_float()} + {em.pick_float()}")
    w = em.fresh("w")
    chain_depth = em.randint(5, 9)
    em.emit(f"{w} = {em.wide_chain(chain_depth)}")
    cond = _emit_condition(em, balanced=True)
    r = em.fresh("r")
    dead_then = em.chance(0.5)
    a = em.fresh("a")
    dead_body = [_body(f"{a} = ({' + '.join(live)})"), _body(f"{r} = {a} / {em.nonzero_divisor()}")]
    live_body = [_body(f"{r} = {w} * {em.choice(['2.0', '0.5'])}")]
    _emit_if_else(em, cond.value, dead_body if dead_then else live_body, live_body if dead_then else dead_body)
    em.add_float(r)
    return _fragment(
        r,
        Mode.CONTINUOUS,
        _combine_shapes([cond], frozenset({Shape.BRANCH, Shape.DEAD_ARM_SPILL, Shape.WIDE_CHAIN})),
        dead_arm_chain_depth=chain_depth,
    )


def _emit_const_branch(em: _Emitter) -> _Fragment:
    """
    A const-branch-via-division shape: an outer runtime diamond whose then arm contains an inner constant-true
    division condition, leaving an empty const-branch block. The inner false arm is over-budget so the
    :data:`Shape.CONST_BRANCH` and :data:`Shape.NESTED_IF` tags denote a surviving inner branch.
    """
    cond = _emit_condition(em, balanced=True)
    r = em.fresh("r")
    base = em.pick_float()
    em.emit(f"{r} = {base}")
    inner_then = _fragment(f"{base} + 1.0", Mode.CONTINUOUS)
    inner_else_body, inner_else_fragment = _overbudget_arm(em, r)
    inner = _if_else_body(
        "(1.0 / 5.0) > 0.0",
        _assign_body(r, inner_then),
        inner_else_body,
    )
    else_body, else_fragment = _overbudget_arm(em, r)
    _emit_if_else(em, cond.value, inner, else_body)
    em.add_float(r)
    return _merge_fragment(
        r,
        [cond, inner_then, inner_else_fragment, else_fragment],
        frozenset({Shape.BRANCH, Shape.CONST_BRANCH, Shape.NESTED_IF}),
    )


def _emit_bounded_loop(em: _Emitter) -> _Fragment:
    r = em.fresh("loop")
    seed = em.pick_float()
    capped = em.fresh("cap")
    em.emit(f"{capped} = ({seed} if {seed} < 8.0 else 8.0)")
    em.emit(f"{r} = ({capped} if {capped} > 1.0 else 1.0)")
    em.emit(f"while {r} > 1.0:")
    em.emit(f"{r} = {r} - 1.0", indent=2)
    em.add_float(r)
    return _fragment(r, Mode.CONTINUOUS, frozenset({Shape.LOOP}))


def _emit_diamond_then_loop(em: _Emitter) -> _Fragment:
    """
    A real diamond whose merge feeds a ``while`` loop (a three-arm loop-header phi), generalized from
    ``diamond_then_loop_kernel``.
    """
    cond = _emit_condition(em, balanced=True)
    r = em.fresh("dl")
    a = em.pick_float()
    then_fragment = _fragment(f"({a} if {a} < 6.0 else 6.0)")
    else_body, else_fragment = _division_arm(em, r)
    _emit_if_else(em, cond.value, _assign_body(r, then_fragment), else_body)
    bounded_name = em.fresh("dlb")
    em.emit(f"{bounded_name} = ({r} if {r} > 1.0 else 1.0)")
    em.emit(f"{r} = ({bounded_name} if {bounded_name} < 6.0 else 6.0)")
    em.emit(f"while {r} > 1.0:")
    em.emit(f"{r} = {r} - 1.0", indent=2)
    em.add_float(r)
    return _merge_fragment(
        r, [cond, then_fragment, else_fragment], frozenset({Shape.BRANCH, Shape.DIAMOND_THEN_LOOP, Shape.LOOP})
    )


def _emit_reduction(em: _Emitter) -> _Fragment:
    n = em.randint(2, 6)
    acc = em.fresh("acc")
    em.emit(f"{acc} = {em.pick_float()}")
    em.emit(f"for _i in range({n}):")
    em.emit(f"{acc} = ({acc} + {em.pick_float()}) * 0.5", indent=2)
    em.add_float(acc)
    return _fragment(acc, Mode.CONTINUOUS, frozenset({Shape.REDUCTION}))


def _emit_bool_logic(em: _Emitter) -> _Fragment:
    """
    Cross-domain bool/float logic: build a boolean from a float relation, combine bools with ``and``/``or``/``not``/
    ``^``, and cast back to float with ``float(cond)``. Exact (the result is 0.0 or 1.0).
    """
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
    return _fragment(out, shapes=frozenset({Shape.BOOL_BANK}))


def _emit_relation_pair(em: _Emitter) -> _Fragment:
    """Two relations over the same ordered pair, forcing a fused multi-output comparator use."""
    a, b = em.floats[:2]
    lt = em.fresh("rel")
    eq = em.fresh("rel")
    out = em.fresh("relout")
    em.emit(f"{lt} = {a} < {b}")
    em.emit(f"{eq} = {a} == {b}")
    em.emit(f"{out} = (1.0 if {lt} else (0.0 if {eq} else -1.0))")
    em.add_bool(lt)
    em.add_bool(eq)
    em.add_float(out)
    return _fragment(out, shapes=frozenset({Shape.BOOL_BANK, Shape.RELATION_PAIR, Shape.SELECT}))


def _emit_exact_division_wiring(em: _Emitter) -> _Fragment:
    """An exact non-commutative division: ``(k*d)/d == k`` with a bounded non-one denominator."""
    k_exp = em.exact_division_scale_exp()
    den_hi_exp = em.exact_division_denominator_hi_exp(k_exp)
    k = _power2_literal(k_exp)
    den_lo = "1.25"
    den_hi = _binary_literal(1.5, den_hi_exp)
    d = em.pick_float()
    mag = em.fresh("divmag")
    capped_hi = em.fresh("divhi")
    den = em.fresh("divden")
    num = em.fresh("divnum")
    out = em.fresh("divout")
    checked = em.fresh("divchk")
    em.emit(f"{mag} = ({d} if {d} > 0.0 else -{d})")
    em.emit(f"{capped_hi} = ({mag} if {mag} < {den_hi} else {den_hi})")
    em.emit(f"{den} = ({capped_hi} if {capped_hi} > {den_lo} else {den_lo})")
    em.emit(f"{num} = {den} * {k}")
    em.emit(f"{out} = {num} / {den}")
    em.emit(f"{checked} = ({out} if {out} == {k} else 0.0)")
    em.add_float(checked)
    return _fragment(checked, shapes=frozenset({Shape.EXACT_WIRING, Shape.SELECT}))


def _emit_sterbenz_subtract(em: _Emitter) -> _Fragment:
    """An exact non-commutative subtraction with observable sign if the operands are transposed."""
    hi_lo = em.sterbenz_hi_lower_literal()
    a = em.pick_float()
    mag = em.fresh("submag")
    hi = em.fresh("subhi")
    lo = em.fresh("sublo")
    out = em.fresh("subout")
    checked = em.fresh("subchk")
    em.emit(f"{mag} = ({a} if {a} > 0.0 else -{a})")
    em.emit(f"{hi} = ({mag} if {mag} > {hi_lo} else {hi_lo})")
    em.emit(f"{lo} = {hi} / 2.0")
    em.emit(f"{out} = {hi} - {lo}")
    em.emit(f"{checked} = ({out} if {out} == {lo} else 0.0)")
    em.add_float(checked)
    return _fragment(checked, shapes=frozenset({Shape.EXACT_WIRING, Shape.SELECT}))


def _emit_forced_overbudget_branch(em: _Emitter) -> _Fragment:
    """A deterministic exact branch kept alive solely by the if-conversion operation budget."""
    cond = _fragment(f"({em.floats[0]} < {em.floats[1]})")
    r = em.fresh("ob")
    then_value = _fragment(em.floats[0])
    else_body, else_fragment = _overbudget_arm(em, r)
    _emit_if_else(em, cond.value, _assign_body(r, then_value), else_body)
    em.add_float(r)
    return _merge_fragment(r, [cond, then_value, else_fragment], frozenset({Shape.BRANCH}))


def _emit_exact_select(em: _Emitter) -> _Fragment:
    cond = _emit_condition(em)
    return em.exact_select_value(cond)


def _emit_exact_clamp(em: _Emitter) -> _Fragment:
    return em.exact_clamp_value()


_STATELESS_TEMPLATES: list[Callable[[_Emitter], _Fragment]] = [
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

_DIRECTED_TEMPLATES: list[Callable[[_Emitter], _Fragment]] = [
    _emit_relation_pair,
    _emit_exact_division_wiring,
    _emit_sterbenz_subtract,
    _emit_forced_overbudget_branch,
]


def _seed_rng(master_seed: int, index: int) -> np.random.Generator:
    """An independent generator for kernel ``index`` of campaign ``master_seed`` -- reproducible regardless of order."""
    return np.random.default_rng(np.random.SeedSequence([master_seed, index]))


def _make_emitter(
    rng: np.random.Generator,
    params: list[str],
    bool_params: set[str],
    fmt: FloatFormat = _DEFAULT_EXACT_FMT,
) -> _Emitter:
    em = _Emitter(rng, fmt)
    em.reset_lines()
    for p in params:
        if p in bool_params:
            em.add_bool(p)
        else:
            em.add_float(p)
    return em


def _finish_function_kernel(
    name: str,
    master_seed: int,
    index: int,
    params: list[str],
    bool_set: set[str],
    em: _Emitter,
    shapes: frozenset[Shape],
    mode: Mode,
    dead_arm_chain_depth: int | None = None,
) -> GeneratedKernel:
    source = _assemble_function(name, params, bool_set, em.render_body(), em.return_line)
    module = _render_module(name, source)
    return GeneratedKernel(
        name=name,
        source=source,
        filename=module.__file__ or "",
        callable=getattr(module, name),
        shapes=shapes,
        mode=mode,
        input_names=params,
        bool_inputs=frozenset(bool_set),
        seed=master_seed,
        index=index,
        is_stateful=False,
        dead_arm_chain_depth=dead_arm_chain_depth,
    )


def generate_stateless_kernel(
    name: str, master_seed: int, index: int, fmt: FloatFormat = _DEFAULT_EXACT_FMT
) -> GeneratedKernel:
    rng = _seed_rng(master_seed, index)
    n_float = int(rng.integers(2, 5))
    use_bool = bool(rng.random() < 0.35)
    float_params = [chr(ord("a") + i) for i in range(n_float)]
    bool_params = [f"c{i}" for i in range(int(rng.integers(1, 3)))] if use_bool else []
    params = float_params + bool_params
    bool_set = set(bool_params)
    em = _make_emitter(rng, params, bool_set, fmt)

    produced: list[_Fragment] = []
    n_fragments = int(rng.integers(1, 4))
    for _ in range(n_fragments):
        template = rng.choice(_STATELESS_TEMPLATES)  # type: ignore[arg-type]
        produced.append(template(em))

    # Combining the produced fragments keeps every one LIVE so DCE cannot drop the diamonds/divisions feeding output.
    return_mode = _emit_return(em, produced)

    return _finish_function_kernel(
        name,
        master_seed,
        index,
        params,
        bool_set,
        em,
        _combine_shapes(produced),
        _combine_mode(produced, return_mode),
        _combine_dead_arm_chain_depth(produced),
    )


def generate_directed_kernel(
    name: str,
    master_seed: int,
    index: int,
    template: Callable[[_Emitter], _Fragment],
    fmt: FloatFormat = _DEFAULT_EXACT_FMT,
) -> GeneratedKernel:
    params = ["a", "b", "c"]
    em = _make_emitter(_seed_rng(master_seed, index), params, set(), fmt)
    payload = template(em)
    em.return_line = f"return {payload.value}"
    return _finish_function_kernel(
        name=name,
        master_seed=master_seed,
        index=index,
        params=params,
        bool_set=set(),
        em=em,
        shapes=payload.shapes,
        mode=payload.mode,
        dead_arm_chain_depth=payload.dead_arm_chain_depth,
    )


def generate_dead_arm_kernel(
    name: str, master_seed: int, index: int, fmt: FloatFormat = _DEFAULT_EXACT_FMT
) -> GeneratedKernel:
    """
    A kernel consisting SOLELY of the dead-arm-spill shape, so no other fragment dilutes its block structure and the
    wide chain reliably spills past the overlapped entry terminator into the dead arm. ``_armed_dead_arm_kernel``
    re-rolls this until the spill is observed, guaranteeing armed overlap coverage independent of the random draw.
    """
    em = _make_emitter(_seed_rng(master_seed, index), ["a", "b", "c"], set(), fmt)
    fragment = _emit_dead_arm_spill(em)
    em.return_line = f"return {fragment.value}"
    return _finish_function_kernel(
        name=name,
        master_seed=master_seed,
        index=index,
        params=["a", "b", "c"],
        bool_set=set(),
        em=em,
        shapes=fragment.shapes,
        mode=fragment.mode,
        dead_arm_chain_depth=fragment.dead_arm_chain_depth,
    )


def _emit_return(em: _Emitter, produced: list[_Fragment]) -> Mode:
    """
    Build the return line, combining every produced fragment so each stays live (DCE cannot drop a diamond feeding the
    output). A single fragment is returned verbatim (exactness-preserving); multiple fragments are returned either as a
    TUPLE of verbatim lanes -- each an output port, still exact -- or as a SUM, which rounds and so demotes the kernel
    to CONTINUOUS mode. A summed return of one fragment does not round (it is the fragment itself).
    """
    live = [fragment.value for fragment in produced] or [em.pick_float()]
    if len(live) == 1:
        em.return_line = f"return {live[0]}"
        return Mode.EXACT
    elif em.chance(0.5):
        lanes = ", ".join(live[:3])
        em.return_line = f"return ({lanes},)"
        return Mode.EXACT
    else:
        em.return_line = f"return {' + '.join(live)}"
        return Mode.CONTINUOUS


def _assemble_function(name: str, params: list[str], bool_set: set[str], body: str, return_line: str) -> str:
    sig = ", ".join(f"{p}: bool" if p in bool_set else p for p in params)
    docstring = '    """A generated fuzz kernel."""'
    return f"def {name}({sig}):\n{docstring}\n{body}\n    {return_line}\n"


def generate_stateful_kernel(
    name: str, master_seed: int, index: int, fmt: FloatFormat = _DEFAULT_EXACT_FMT
) -> GeneratedKernel:
    """
    Render one stateful kernel exercising the chained-slot pattern (capturing one slot's OLD value while another
    advances). The slots are PRIVATE so no ``state_*`` output port is created -- a public attribute would break the
    float64 comparison.
    """
    rng = _seed_rng(master_seed, index)
    n_inputs = int(rng.integers(1, 3))
    inputs = [chr(ord("x") + i) for i in range(n_inputs)]
    n_slots = int(rng.integers(2, 4))
    slots = [f"_s{i}" for i in range(n_slots)]
    resets = [float(rng.choice([0.0, 1.0, -1.0, 0.5, 2.0])) for _ in slots]

    em = _Emitter(rng, fmt)
    em.reset_lines()
    fragments: list[_Fragment] = []
    for p in inputs:
        em.add_float(p)
    for s in slots:
        em.add_float(f"self.{s}")

    # Capture every slot's OLD value first (so updates that reference each other use the live-in, like the chained
    # pattern) -- the frontend's parallel slot semantics make this faithful, but binding locals keeps the source clear.
    olds = {s: em.fresh("old") for s in slots}
    for s in slots:
        em.emit(f"{olds[s]} = self.{s}")
        em.add_float(olds[s])

    # The chained-slot pattern: slot i captures slot (i+1)'s old value; the last slot advances from an input.
    for i, s in enumerate(slots):
        if i + 1 < n_slots and em.chance(0.6):
            em.emit(f"self.{s} = {olds[slots[i + 1]]}")
        else:
            em.emit(f"self.{s} = {em.pick_float()} + {em.small_literal()}")

    # Optionally fold a real diamond into the update so a stateful kernel also exercises branchy scheduling. Its result
    # is threaded into the return so the diamond stays LIVE -- otherwise DCE could drop it and erase the branch.
    diamond_result = _emit_diamond(em, nested=False) if em.chance(0.5) else None
    if diamond_result is not None:
        fragments.append(diamond_result)

    head = diamond_result.value if diamond_result is not None else em.pick_float()
    em.return_line = f"return ({head}) * 2.0 + ({em.pick_float()} * 1.5) / {em.nonzero_divisor()}"

    source = _assemble_class(name, inputs, slots, resets, em.render_body(base_indent=1), em.return_line)
    module = _render_module(name, source)
    cls = getattr(module, name)

    return GeneratedKernel(
        name=name,
        source=source,
        filename=module.__file__ or "",
        callable=cls().__call__,
        shapes=_combine_shapes(fragments, frozenset({Shape.STATE})),
        mode=Mode.CONTINUOUS,
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


def generate_kernel(master_seed: int, index: int, fmt: FloatFormat = _DEFAULT_EXACT_FMT) -> GeneratedKernel:
    """Reproducible by ``(seed, index)`` regardless of draw order, so any kernel replays from its provenance alone."""
    name = f"fuzz_k_{master_seed:x}_{index}"
    rng = np.random.default_rng(np.random.SeedSequence([master_seed, index, 0x5A5A]))
    if rng.random() < 0.3:
        return generate_stateful_kernel(name, master_seed, index, fmt)
    return generate_stateless_kernel(name, master_seed, index, fmt)


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


def _build_with_lir(
    fn: Callable[..., object], ops: OpConfig, name: str
) -> tuple[Mir, Lir, NumericalSimulator, MirInterpreter]:
    """
    Lower the kernel ONCE to MIR/LIR, then build both the numerical model (from that LIR) and the interpreter (from the
    same MIR). Returning the MIR/LIR lets callers assert on block structure and schedule facts without double-lowering
    or touching simulator internals. Shared by the campaign runner and the regression replayer, so both drive the
    identical build path.
    """
    mir = lower_to_mir(optimize(lower_frontend(fn)), ops)
    lir = build(mir, name)
    model = generate(lir).elaborate()
    interpreter = MirInterpreter(mir)
    return mir, lir, model, interpreter


def _build_all(fn: Callable[..., object], ops: OpConfig, name: str) -> tuple[Mir, NumericalSimulator, MirInterpreter]:
    mir, _, model, interpreter = _build_with_lir(fn, ops, name)
    return mir, model, interpreter


def _has_overlap_spill(lir: Lir) -> bool:
    """
    Whether a wide-bank write computed in a branch block spills past that block's overlap-shrunk terminator into its
    successor arms. For a dead-arm kernel this is the real clobber hazard: the wide chain lands inside both arms, so a
    dropped inflight reservation could reuse its register before the dead-arm write arrives.
    """
    return _overlap_spill_depth(lir) is not None


def _has_overlap_spill_at_depth(lir: Lir, min_depth: int) -> bool:
    """
    Whether the spilling write is deep enough to witness the generated dead-arm wide chain, not a shallow incidental
    value that happens to land past the branch terminator.
    """
    depth = _overlap_spill_depth(lir)
    return depth is not None and depth >= min_depth


def _depth_at(history: list[tuple[int, int]], read_cycle: int) -> int:
    depth = 0
    latest_landing = -1
    for landing, candidate in history:
        if landing <= read_cycle and landing >= latest_landing:
            latest_landing = landing
            depth = candidate
    return depth


def _overlap_spill_depth(lir: Lir) -> int | None:
    best: int | None = None
    for block in lir.blocks:
        if not isinstance(block.terminator, Branch):
            continue
        reg_depth: dict[RegRef, list[tuple[int, int]]] = {
            load.dst: [(0, 0)] for load in lir.inputs if isinstance(load.dst, RegRef)
        }
        block_ops: list[ScheduledOp] = sorted(
            [*block.ops, *block.inline_ops], key=lambda op: (op.issue_cycle, op.commit_cycle)
        )
        for op in block_ops:
            read_cycle = operand_read_cycle(op.operator, op.issue_cycle)
            operands = [
                _depth_at(reg_depth.get(operand.source, []), read_cycle)
                for operand in op.operands
                if isinstance(operand.source, RegRef)
            ]
            op_depth = max(operands, default=0) + 1
            for write in op.writes:
                if not isinstance(write.dst, RegRef):
                    continue
                landing = (
                    inline_landing_cycle(op.commit_cycle)
                    if isinstance(op, InlineScheduledOp)
                    else result_landing_cycle(write.dst, op.commit_cycle)
                )
                reg_depth.setdefault(write.dst, []).append((landing, op_depth))
                if landing > block.term_offset:
                    best = op_depth if best is None else max(best, op_depth)
    return best


def _has_fused_relation_pair(lir: Lir) -> bool:
    """
    Whether the LIR contains one comparator firing with multiple tapped order-flag outputs. The relation-pair directed
    kernel emits ``a < b`` and ``a == b`` over the same ordered pair; if scheduler firing fusion regresses to two
    independent comparator firings, each ``fcmp`` scheduled op will have only one write and this witness fails.
    """
    for block in lir.blocks:
        for op in block.ops:
            if op.operator.mnemonic == "fcmp" and len(op.writes) > 1:
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


def _required_forward_branches(shapes: frozenset[Shape]) -> int:
    if Shape.BRANCH not in shapes:
        return 0
    if Shape.NESTED_IF in shapes or Shape.CONST_BRANCH in shapes:
        return 2
    return 1


def _assert_danger_survived(kernel: GeneratedKernel, mir: Mir, op_label: str) -> None:
    """
    A branchy kernel must keep a real FORWARD (diamond) branch after compilation. The generated diamonds are built
    un-if-convertible via an unspeculatable division or an over-budget exact arm, so a silent degradation to a
    straight-line ``select`` should not happen -- this is a loud safety net against an unexpected if-conversion change.
    Counting only FORWARD branches (excluding loop-header back-edge branches) is robust to a co-occurring loop, which
    would otherwise keep a branch terminator alive even if every diamond degraded away.
    """
    required = _required_forward_branches(kernel.shapes)
    if required == 0:
        return
    survived = surviving_forward_branches(mir)
    if survived < required:
        raise DangerShapeLost(
            f"{kernel.name} [{op_label}] was generated as branchy (shapes={sorted(s.name for s in kernel.shapes)}) "
            f"but compiled with {survived} forward branch(es), below the required {required}, across "
            f"{len(mir.blocks)} block(s): a generated diamond degraded to a select/straight-line, so it exercises "
            f"less branch or overlap hazard than claimed"
        )


def _port_vector(model: NumericalSimulator, fmt: FloatFormat, values: dict[str, float | bool]) -> Vector:
    vector: Vector = []
    for port in model.inputs:
        raw = values[port.name]
        if isinstance(port.scalar_type, BoolType):
            vector.append(bool(raw))
        else:
            vector.append(FloatValue.from_float(fmt, float(raw)))
    return vector


def _vector_from_bits(model: NumericalSimulator, fmt: FloatFormat, bits: dict[str, int]) -> Vector:
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
    # Dead-arm and over-budget kernels branch on a balanced condition; over few random vectors the hazardous arm might
    # never be taken. Two DIRECTED vectors -- floats ASCENDING by port order with bools True, and floats DESCENDING with
    # bools False -- take OPPOSITE arms of any such condition, guaranteeing both arms are exercised.
    if Shape.DEAD_ARM_SPILL in kernel.shapes or Shape.OVERBUDGET_BRANCH in kernel.shapes:
        float_ports = [p.name for p in model.inputs if not isinstance(p.scalar_type, BoolType)]
        for ascending in (True, False):
            magnitudes = range(1, len(float_ports) + 1) if ascending else range(len(float_ports), 0, -1)
            directed: dict[str, float | bool] = {
                p.name: ascending for p in model.inputs if isinstance(p.scalar_type, BoolType)
            }
            directed.update({name: float(mag) for name, mag in zip(float_ports, magnitudes)})
            vectors.append(_port_vector(model, fmt, directed))
    if Shape.RELATION_PAIR in kernel.shapes:
        float_ports = [p.name for p in model.inputs if not isinstance(p.scalar_type, BoolType)]
        assert len(float_ports) >= 2
        for a_value, b_value in ((1.0, 2.0), (2.0, 1.0), (1.0, 1.0)):
            directed = {p.name: False for p in model.inputs if isinstance(p.scalar_type, BoolType)}
            directed.update({name: 1.0 for name in float_ports})
            directed[float_ports[0]] = a_value
            directed[float_ports[1]] = b_value
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
    except (ZeroDivisionError, OverflowError, ValueError):
        return None
    return [value for _, value in flatten_value(result)]


class _SecondaryResult(Enum):
    """The outcome of comparing one vector's model output against the float64 reference, leaf by leaf."""

    PASS = auto()  # every lane finite and within tolerance (EXACT: bit-exact)
    SKIP_NONFINITE = auto()  # a CONTINUOUS lane was inf/nan on either side, so the float64 net cannot compare it
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
    within a format tolerance (EXACT mode: rtol=atol=0). A non-finite EXACT leaf is a hard failure because every
    EXACT-mode probe is required to stay finite in the configured ZKF format; only CONTINUOUS non-finite leaves are
    reported ``SKIP_NONFINITE``. The tolerance is widened by the operation count.
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
    exact_nonfinite_fail: str | None = None
    for lane, (m, r) in enumerate(zip(model_out, reference, strict=True)):
        if isinstance(m, bool) != isinstance(r, bool):
            # A lane's type must match the source's: a bool output lowered as a float (or vice versa) is a real
            # miscompile that coercing both sides with ``bool(...)`` would mask (model True vs reference 1.0 compare
            # ==).
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
            if mode is Mode.EXACT and exact_nonfinite_fail is None:
                exact_nonfinite_fail = f"EXACT float lane {lane} is non-finite: model {mf!r} vs reference {r!r}"
                continue
            # Not comparable; defer the verdict. A non-finite lane -- INCLUDING the model saturating to inf where the
            # float64 reference stays finite -- is SKIPPED, not failed: ZKF's narrower range legitimately saturates on
            # overflow, including INTERMEDIATE overflow (``a*b`` exceeds the format yet the true result fits), so a
            # model-inf/finite-reference lane cannot be distinguished from a real shared inf-miscompile without false
            # positives. That narrow shared-upstream class is backstopped by the operator unit tests and the example
            # suite (same casts/operators vs Python), not by this best-effort float64 net.
            nonfinite = True
            continue
        if tolerance_fail is None and not within(mf, r, rtol, atol):
            # Remember the first float tolerance miss but KEEP scanning -- a STRUCTURAL failure on a LATER lane must
            # take precedence, since in CONTINUOUS mode a tolerance FAIL is suppressed as drift and would otherwise
            # mask it.
            tolerance_fail = f"float lane {lane}: model {mf!r} vs reference {r!r} (rtol={rtol:g}, atol={atol:g})"
    if exact_nonfinite_fail is not None:
        return _SecondaryResult.FAIL, exact_nonfinite_fail
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
            if self._mode is Mode.EXACT:
                detail = f"EXACT-mode float64 reference raised for inputs {[show_value(v) for v in vector]}"
                return _SecondaryOutcome(checked=False, latch_off=False, exact_failure=detail)
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
    mir, lir, model, interpreter = _build_with_lir(kernel.callable, ops, name)
    _assert_danger_survived(kernel, mir, op_label)
    if Shape.RELATION_PAIR in kernel.shapes:
        assert _has_fused_relation_pair(lir), f"{name}: relation-pair kernel did not fuse comparator order-flag taps"
    if expect_armed:
        assert kernel.dead_arm_chain_depth is not None, f"{name}: forced dead-arm kernel lacks chain-depth metadata"
        assert _has_overlap_spill_at_depth(
            lir, kernel.dead_arm_chain_depth
        ), f"{name}: dead-arm kernel did not spill a depth-{kernel.dead_arm_chain_depth} chain -- hazard not armed"

    assert model.inputs == interpreter.inputs, f"{name}: input ports differ (name or type)"
    # The float64 reference binds by PARAMETER NAME against a vector drawn in model-port order, so a frontend bug that
    # swapped the param->port mapping and the port order together would feed model, interpreter, AND reference the same
    # wrong name->value mapping and pass vacuously. Pinning the port order to the source parameter order closes that.
    assert [p.name for p in model.inputs] == list(
        kernel.input_names
    ), f"{name}: model input ports {[p.name for p in model.inputs]} differ from params {list(kernel.input_names)}"
    assert model.outputs == interpreter.outputs, f"{name}: output ports differ (name or type)"
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
        kernel = generate_dead_arm_kernel(f"deadarm_{index}_{attempt}", master_seed, index * 64 + attempt, fmt)
        assert kernel.dead_arm_chain_depth is not None
        # Verify the chain spills under EVERY op-config the forced batch asserts (``expect_armed``), not just the
        # default -- so a future config under which the chain happens not to spill cannot false-fail the campaign.
        lirs = (_build_with_lir(kernel.callable, make_ops(fmt), kernel.name)[1] for make_ops in OP_CONFIGS.values())
        if all(_has_overlap_spill_at_depth(lir, kernel.dead_arm_chain_depth) for lir in lirs):
            return kernel
    raise DangerShapeLost(f"dead-arm generator failed to arm a spill in {_ARM_RETRIES} tries (index {index})")


def _run_campaign_kernel(
    kernel: GeneratedKernel,
    stats: CampaignStats,
    fmt: FloatFormat,
    effort: str,
    n_vectors: int,
    on_divergence: Callable[[Divergence], None],
    *,
    expect_armed: bool = False,
) -> None:
    stats.record_kernel(kernel)
    for op_label, make_ops in OP_CONFIGS.items():
        divergence = run_kernel(kernel, op_label, make_ops(fmt), fmt, effort, n_vectors, stats, expect_armed)
        if divergence is not None:
            stats.divergences.append(divergence)
            on_divergence(divergence)


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
        _run_campaign_kernel(forced, stats, fmt, effort, n_vectors, on_divergence, expect_armed=True)
    for j, template in enumerate(_DIRECTED_TEMPLATES):
        directed = generate_directed_kernel(f"directed_{j}", master_seed, 0xD1EC7ED + j, template, fmt)
        _run_campaign_kernel(directed, stats, fmt, effort, n_vectors, on_divergence)
    for index in range(n_kernels):
        kernel = generate_kernel(master_seed, index, fmt)
        _run_campaign_kernel(kernel, stats, fmt, effort, n_vectors, on_divergence)
    return stats


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
    Write a self-contained reproducer into :data:`REGRESSIONS_DIR`. The replayer (``test_fuzz_regressions``) globs
    these and re-asserts the previously-failing check.
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
    repro_shapes = frozenset(Shape[name] for name in repro.shapes)
    required = _required_forward_branches(repro_shapes)
    survived = surviving_forward_branches(mir)
    if survived < required:
        return (
            False,
            f"branch survival lost on replay: {survived} forward branch(es), required {required}, across "
            f"{len(mir.blocks)} block(s) (shapes={sorted(repro.shapes)})",
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
