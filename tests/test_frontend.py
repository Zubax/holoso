"""Unit tests for the Python-to-HIR frontend."""

import dataclasses
import logging
import math
import sys
import textwrap
import types
from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import pytest

import holoso
from holoso import FloatFormat, SourceUnavailable, UnsupportedConstruct, UnsupportedLibraryFunction
from holoso._frontend import lower
from holoso._frontend._ast_support import port_name
from holoso._hir import (
    BoolAnd,
    BoolConst,
    BoolNot,
    BoolOr,
    BoolSelect,
    BoolToFloat,
    BoolType,
    Branch,
    FloatAbs,
    FloatAdd,
    FloatConst,
    FloatCos,
    FloatDiv,
    FloatExp2,
    FloatMul,
    FloatNeg,
    FloatRelational,
    FloatSin,
    FloatToBool,
    IntType,
    FloatType,
    Hir,
    InPort,
    Jump,
    Operation,
    optimize,
    Phi,
    Ret,
    Select,
    StateRead,
)

from ._modelref import arith_count as _arith_count, default_ops, flatten_value, output_names


def test_scalar_is_output_zero() -> None:
    assert output_names(3.14) == ["out_0"]


def test_flat_sequence_is_positional() -> None:
    assert output_names((1.0, 2.0, 3.0)) == ["out_0", "out_1", "out_2"]


def test_nested_list_row_major_like_ekf1_stateless() -> None:
    # ekf1_stateless's update_x_P returns a 9x1 nested list -> out_0_0 .. out_8_0
    matrix = [[float(i)] for i in range(9)]
    assert output_names(matrix) == [f"out_{i}_0" for i in range(9)]


def test_matrix_n_by_m() -> None:
    matrix = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    assert output_names(matrix) == ["out_0_0", "out_0_1", "out_0_2", "out_1_0", "out_1_1", "out_1_2"]


def test_dataclass_fields_and_nesting() -> None:
    @dataclasses.dataclass
    class Foo:
        bar: float

    @dataclasses.dataclass
    class Baz:
        foo: Foo

    assert output_names((Baz(Foo(1.0)), 2.0)) == ["out_0_foo_bar", "out_1"]


def test_bare_dataclass_uses_field_names() -> None:
    @dataclasses.dataclass
    class Out:
        x: float
        y: float

    assert output_names(Out(1.0, 2.0)) == ["out_x", "out_y"]


def test_port_name_paths() -> None:
    assert port_name([0]) == "out_0"
    assert port_name([0, "foo", "bar"]) == "out_0_foo_bar"
    assert port_name([3, 1]) == "out_3_1"


def test_flatten_value_returns_leaves() -> None:
    leaves = flatten_value([[1.5], [2.5]])
    assert [value for _, value in leaves] == [1.5, 2.5]


def test_small_kernel_inputs_outputs_and_ops() -> None:
    def kernel(a: float, b: float) -> float:
        return (a - b) * 0.25 + a * b

    hir = lower(kernel)
    assert hir.input_names() == ["a", "b"]
    assert [o.name for o in hir.outputs] == ["out_0"]
    assert _arith_count(hir, FloatMul) == 2
    assert _arith_count(hir, FloatAdd) == 2  # subtraction (add+neg) and the final add
    assert _arith_count(hir, FloatNeg) == 1  # the negation introduced by subtraction


def test_bool_parameter_annotation_becomes_bool_input() -> None:
    def passthrough(flag: bool) -> bool:
        return flag

    hir = lower(passthrough)
    assert hir.input_names() == ["flag"]
    node = hir.nodes[hir.input_ids[0]]
    assert isinstance(node, InPort)
    assert isinstance(node.type, BoolType)
    assert [o.name for o in hir.outputs] == ["out_0"]


def test_float_parameter_annotation_remains_float_input() -> None:
    def passthrough(value: float) -> float:
        return value

    hir = lower(passthrough)
    node = hir.nodes[hir.input_ids[0]]
    assert isinstance(node, InPort)
    assert isinstance(node.type, FloatType)


def test_unsupported_scalar_parameter_annotation_is_rejected() -> None:
    def passthrough(value: str) -> float:  # int is now a supported scalar; str/bytes/complex remain rejected
        return float(len(value))

    with pytest.raises(UnsupportedConstruct, match="parameter annotation"):
        lower(passthrough)


def test_missing_parameter_annotation_is_rejected() -> None:
    # An unannotated parameter is rejected: there is no implicit float default.
    def passthrough(value):  # type: ignore[no-untyped-def]
        return value

    with pytest.raises(UnsupportedConstruct, match="requires an explicit type annotation"):
        lower(passthrough)


def test_assert_is_ignored_with_info_message(caplog: pytest.LogCaptureFixture) -> None:
    # The whole test subtree -- the comparison and its nested call -- is dropped, so neither op reaches HIR.
    def with_assert(x: float) -> float:
        assert abs(x) > 0.0
        return x + 1.0

    with caplog.at_level(logging.INFO, logger="holoso._frontend._lower"):
        hir = lower(with_assert)
    assert [o.name for o in hir.outputs] == ["out_0"]
    assert _arith_count(hir, FloatRelational) == 0 and _arith_count(hir, FloatAbs) == 0
    assert "has no effect in Holoso" in caplog.text


def test_walrus_in_assert_is_ignored_not_rejected() -> None:
    # A walrus even inside an and/or in an assert is ignored, not caught by the short-circuit-walrus pre-pass: the
    # assert is dropped and the unused binding simply vanishes, as under -O.
    def unused_walrus(x: float) -> float:
        assert (ok := x > 0.0) and True
        return x + 1.0

    assert [o.name for o in lower(unused_walrus).outputs] == ["out_0"]


def test_dropped_assert_walrus_used_later_is_unknown_name() -> None:
    # The dropped assert never binds its walrus, yet the target stays a function local (a walrus is scoped
    # syntactically), so a later use is a clean unknown-name error that shadows the same-named global rather than
    # silently reading it -- mirroring Python -O, which keeps the local and raises UnboundLocalError. The injected
    # ``y`` global is what makes this discriminating: were the target NOT kept local, the read would resolve to it.
    def used_later(x: float) -> float:
        assert (y := x * 2.0) > 0.0
        return y

    with pytest.raises(UnsupportedConstruct, match="unbound"):
        lower(_rebind_globals(used_later, y=5.0))


def test_pow_expands_to_multiply_chain() -> None:
    def cube(a: float) -> float:
        return a**3

    hir = lower(cube)
    assert _arith_count(hir, FloatMul) == 2


def test_abs_lowers_to_semantic_operation() -> None:
    def f(a: float) -> float:
        return abs(a)

    hir = lower(f)
    abs_ops = [n for n in hir.nodes.values() if isinstance(n, Operation) and type(n.operator) is FloatAbs]
    assert len(abs_ops) == 1


def test_division_lowers_to_div() -> None:
    def f(a: float, b: float) -> float:
        return a / b

    hir = lower(f)
    assert _arith_count(hir, FloatDiv) == 1
    divs = [n for n in hir.nodes.values() if isinstance(n, Operation) and type(n.operator) is FloatDiv]
    assert len(divs) == 1


def test_ekf1_stateless_structure() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    import ekf1_stateless

    hir = lower(ekf1_stateless.update_x_P)
    assert len(hir.input_ids) == 17
    assert [o.name for o in hir.outputs] == [f"out_{i}_0" for i in range(9)]
    assert _arith_count(hir, FloatDiv) == 1  # only x22 = 1 / x21


def test_static_for_loop_unrolls() -> None:
    def f(a: float) -> float:
        x = a
        for _ in range(3):
            x = x + a
        return x

    hir = lower(f)
    assert _arith_count(hir, FloatAdd) == 3


def test_for_loop_counter_is_a_compile_time_constant() -> None:
    # The counter indexes a constant table and sets a power-of-two shift exponent; both fold per unrolled trip.
    def f(a: float) -> float:
        table = (1.0, 2.0, 4.0)
        y = a
        for i in range(3):
            y = y + table[i] * (2.0**-i)
        return y

    lower(f)  # lowers without error: table[i] and 2**-i are resolved at compile time for each i


def test_dead_arm_attr_write_does_not_block_readonly_fold() -> None:
    # Regression (Codex): a write to a read-only boolean attribute inside a statically-dead `if False:` arm must not
    # mark it as assigned -- otherwise the attribute is wrongly treated as runtime and a later guard on it is not
    # folded, spuriously rejecting a return that the fold would have made unreachable.
    class DeadFlagGuard:
        def __init__(self) -> None:
            self._flag = False
            self.y = 0.0

        def __call__(self, x: float) -> float:
            if self._flag:  # _flag is read-only False -> folds away; the return arm is dead
                return x
            if False:
                self._flag = True  # noqa -- dead arm: must not count as assigning _flag
            self.y = x
            return self.y

    hir = lower(DeadFlagGuard().__call__)  # must not raise (the read-only fold removes the return-in-branch)
    assert [slot.name for slot in hir.state_slots] == ["y"]  # only y is state; _flag stays a read-only constant


def test_static_comparison_dead_arm_does_not_block_readonly_fold() -> None:
    # Regression (Codex): a write under a statically-false COMPARISON guard (not just a literal bool) must not mark the
    # attribute as assigned -- the read-only scan folds any attribute-free statically-known condition, as lowering does.
    class StaticCmpDeadFlag:
        def __init__(self) -> None:
            self._flag = False
            self.y = 0.0

        def __call__(self, x: float) -> float:
            if 1.0 < 0.0:
                self._flag = True  # noqa -- dead arm (statically-false comparison): must not assign _flag
            if self._flag:
                return x
            self.y = x
            return self.y

    hir = lower(StaticCmpDeadFlag().__call__)  # must not raise; _flag stays read-only so the return arm folds away
    assert [slot.name for slot in hir.state_slots] == ["y"]


def test_over_threshold_for_in_statically_dead_arm_is_skipped() -> None:
    # Regression (Codex): an over-threshold for-loop in a statically-dead branch arm (here the else of a read-only
    # True attribute guard) is unreachable; the fold removes the dead arm, so the loop is neither unrolled nor rejected.
    class DeadOverThreshold:
        def __init__(self) -> None:
            self.flag = True
            self.y = 0.0

        def __call__(self, x: float) -> float:
            if self.flag:
                y = x
            else:
                for _ in range(1000):  # dead (flag read-only True): not unrolled, not rejected
                    y = x
            return y

    hir = lower(DeadOverThreshold().__call__)
    assert len(optimize(hir).blocks) == 1  # the dead else arm and its over-threshold loop are folded away


def test_zero_trip_for_write_does_not_mark_attribute_assigned() -> None:
    # Regression (user): a write inside `for _ in range(0)` never executes, so the read-only scan must not count it as
    # an assignment -- otherwise a later guard on the attribute becomes a runtime branch and a return in the (actually
    # dead) arm is wrongly rejected. The scan mirrors the static trip count, as lowering and the state scan do.
    class ZeroForFlag:
        def __init__(self) -> None:
            self._flag = False
            self.y = 0.0

        def __call__(self, x: float) -> float:
            for _ in range(0):
                self._flag = True  # noqa -- zero-trip loop: never runs
            if self._flag:
                return x
            self.y = x
            return self.y

    hir = lower(ZeroForFlag().__call__)
    assert [slot.name for slot in hir.state_slots] == ["y"]  # _flag stays a read-only constant; only y is state


def test_zero_trip_self_attr_range_write_does_not_mark_attribute_assigned() -> None:
    # A read-only integer attribute used as a static range bound must be visible to the read-only assignment scan too.
    # Otherwise the scan treats the zero-trip loop body as reachable and later fails to fold the read-only flag guard.
    class ZeroSelfRangeFlag:
        def __init__(self) -> None:
            self.iterations = 0
            self._flag = False
            self.y = 0.0

        def __call__(self, x: float) -> float:
            for _ in range(self.iterations):
                self._flag = True  # noqa -- zero-trip loop: never runs
            if self._flag:
                return x
            self.y = x
            return self.y

    hir = lower(ZeroSelfRangeFlag().__call__)
    assert [slot.name for slot in hir.state_slots] == ["y"]


def test_nested_function_definition_is_rejected() -> None:
    # A nested function or class definition inside a kernel is unsupported -- even a dead one after a return. The
    # original scope-shadowing concern (the nested ``np`` leaking to the outer scope) cannot arise, because the nested
    # def is rejected outright at build time before any name resolution.
    def kernel(x: float) -> float:
        y = np.asarray([x])
        return y[0]  # type: ignore[no-any-return]

        def helper() -> int:  # noqa -- dead nested scope; its ``np`` is not the outer's
            np = 1
            return np

    with pytest.raises(UnsupportedConstruct, match="nested function"):
        lower(kernel)


def test_globally_shadowed_range_is_rejected(tmp_path: Path) -> None:
    # Regression (Codex): Python resolves a module global before the builtin, so a shadowed `range` is not the
    # unrollable builtin and must be rejected, not silently unrolled. The frontend needs real source, so the kernel
    # lives in a temp module that shadows `range` at module scope.
    import importlib.util

    source = textwrap.dedent("""
        range = lambda n: [0, 0, 0]

        def kernel(a: float) -> float:
            y = a
            for _ in range(3):
                y = y + 1.0
            return y
        """)
    module_path = tmp_path / "_shadowed_range_mod.py"
    module_path.write_text(source)
    spec = importlib.util.spec_from_file_location("_shadowed_range_mod", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(UnsupportedConstruct, match="only plain functions can be kernels"):
        lower(module.kernel)


def _lower_generated_kernel(tmp_path: Path, name: str, source: str):  # type: ignore[no-untyped-def]
    import importlib.util

    module_path = tmp_path / f"_{name}.py"
    module_path.write_text(source)
    spec = importlib.util.spec_from_file_location(f"_{name}", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_comprehension_with_too_many_generators_is_a_located_rejection(tmp_path: Path) -> None:
    # Regression (Codex): each generator adds one `_expand_comprehension` frame, so a comprehension with hundreds of
    # generators used to leak a bare RecursionError. It now rejects with a located error; CPython runs it normally.
    clauses = " ".join(f"for a{i} in [{float(i)!r}]" for i in range(200))
    module = _lower_generated_kernel(
        tmp_path, "many_generators", f"def kernel(x: float) -> float:\n    return [x {clauses}][0]\n"
    )
    assert module.kernel(6.5) == 6.5  # valid, runnable Python
    with pytest.raises(UnsupportedConstruct, match="comprehension nesting expands"):
        lower(module.kernel)


def test_deeply_nested_comprehensions_are_a_located_rejection(tmp_path: Path) -> None:
    # Regression (Codex): a single comprehension's generators are bounded, but nested comprehensions accumulate
    # expansion frames across levels, so deep nesting used to leak a bare RecursionError; it is now a located error.
    inner = "x"
    for depth in range(15):
        clauses = " ".join(f"for a{depth}_{i} in [0.0]" for i in range(64))
        inner = f"[{inner} {clauses}][0]"
    module = _lower_generated_kernel(
        tmp_path, "nested_comprehensions", f"def kernel(x: float) -> float:\n    return {inner}\n"
    )
    assert module.kernel(3.25) == 3.25  # valid, runnable Python (below CPython's own compiler limit)
    with pytest.raises(UnsupportedConstruct, match="comprehension nesting expands"):
        lower(module.kernel)


def test_constant_boolean_attribute_branch_folds() -> None:
    # Regression (Codex): a branch on a read-only boolean attribute has a compile-time-known condition; only the taken
    # arm lowers, so a write in the dead arm does not become spurious persistent state.
    class Disabled:
        def __init__(self) -> None:
            self.flag = False
            self.y = 0.0

        def __call__(self, x: float) -> float:
            if self.flag:
                self.y = x
            return self.y

    hir = lower(Disabled().__call__)
    assert [slot.name for slot in hir.state_slots] == []  # folded: y never written, no state, no branch
    assert len(optimize(hir).blocks) == 1


def test_numpy_boolean_attribute_branch_folds() -> None:
    # Regression (Codex): a read-only np.bool_ attribute must fold like a Python bool (it is exposed as boolean state
    # elsewhere), so the disabled arm's write does not become spurious state.
    class NpDisabled:
        def __init__(self) -> None:
            self.flag = np.bool_(False)
            self.y = 0.0

        def __call__(self, x: float) -> float:
            if self.flag:
                self.y = x
            return self.y

    hir = lower(NpDisabled().__call__)
    assert [slot.name for slot in hir.state_slots] == []
    assert len(optimize(hir).blocks) == 1


def test_static_integer_comparison_branch_folds() -> None:
    # Regression (Codex): a comparison of static integers (an unrolled loop counter against a bound) is known at
    # compile time and folds to one arm; a write gated by a statically-false guard must not become spurious state, and
    # no dynamic branch is emitted (integers are exact in any ZKF format, so the fold matches the comparator).
    class GuardAlwaysFalse:
        def __init__(self) -> None:
            self.x = 0.0

        def __call__(self, v: float) -> float:
            for i in range(3):
                if i > 5:  # never true over range(3): x must not become state
                    self.x = v
            return self.x

    folded = lower(GuardAlwaysFalse().__call__)
    assert [slot.name for slot in folded.state_slots] == []
    assert len(optimize(folded).blocks) == 1

    class GuardReal:
        def __init__(self) -> None:
            self.acc = 0.0

        def __call__(self, v: float) -> float:
            for i in range(3):
                if i > 0:  # true for i in {1, 2}: acc genuinely accumulates
                    self.acc = self.acc + v
            return self.acc

    real = lower(GuardReal().__call__)
    assert [slot.name for slot in real.state_slots] == ["acc"]
    assert _arith_count(real, FloatAdd) == 2  # one accumulate per folded-true trip (i=1, i=2), none for i=0


def test_static_float_comparison_branch_folds() -> None:
    # Regression (Codex finding 1, fast-math): a comparison of compile-time floats (a literal, a read-only float
    # attribute, or arithmetic of these) folds to one arm so a guarded write under a statically-false condition does
    # not become spurious state. Folding is float64 (fast-math, accepted per DESIGN.md); model and RTL follow the same
    # arm regardless.
    class ConfigGate:
        def __init__(self) -> None:
            self.threshold = 0.0  # read-only config: 0.0 > 1.0 is statically false
            self.gain = 2.0
            self.y = 0.0

        def __call__(self, x: float) -> float:
            if self.threshold > 1.0:
                self.y = x
            if self.gain * 3.0 > 10.0:  # 6.0 > 10.0 is statically false
                self.y = x
            return self.y

    folded = lower(ConfigGate().__call__)
    assert [slot.name for slot in folded.state_slots] == []
    assert len(optimize(folded).blocks) == 1

    class ConfigEnabled:
        def __init__(self) -> None:
            self.threshold = 5.0
            self.y = 0.0

        def __call__(self, x: float) -> float:
            if self.threshold > 1.0:  # 5.0 > 1.0 statically true: the write is taken
                self.y = x
            return self.y

    enabled = lower(ConfigEnabled().__call__)
    assert [slot.name for slot in enabled.state_slots] == ["y"]
    assert len(optimize(enabled).blocks) == 1


def test_dead_assignment_after_return_does_not_suppress_fold() -> None:
    # Regression (Codex finding 3): the read-only-attribute scan stops at a return (like lowering), so an assignment
    # in dead code after a return does not mask the attribute's read-only-ness and the branch on it still folds.
    class DeadAfterReturn:
        def __init__(self) -> None:
            self.flag = False
            self.y = 0.0

        def __call__(self, x: float) -> float:
            if self.flag:  # flag is read-only -> folds to the (empty) else arm
                self.y = x
            result = self.y
            return result
            self.flag = False  # noqa -- dead code: must not count as an assignment of flag

    hir = lower(DeadAfterReturn().__call__)
    assert [slot.name for slot in hir.state_slots] == []
    assert len(optimize(hir).blocks) == 1


def test_boolean_ordering_and_mixed_comparison_are_rejected() -> None:
    # Booleans compare only with == and != (which lower to xnor/xor; see tests/test_language_features.py). Ordering on
    # booleans, and a mixed boolean/float comparison, remain rejected with a clear UnsupportedConstruct.
    class BoolOrdering:
        def __call__(self, a: bool, b: bool) -> bool:
            return a < b

    with pytest.raises(UnsupportedConstruct, match="only == and != are defined between boolean"):
        lower(BoolOrdering().__call__)

    class MixedComparison:
        def __call__(self, flag: bool, x: float) -> bool:
            return flag == x

    with pytest.raises(UnsupportedConstruct, match="mixes a boolean and a non-boolean"):
        lower(MixedComparison().__call__)


def test_public_boolean_state_attribute_is_output() -> None:
    class PublicBool:
        def __init__(self) -> None:
            self.flag = False
            self.y = 0.0

        def __call__(self, a: float, b: float) -> float:
            self.flag = a < b
            self.y = a
            return self.y

    hir = lower(PublicBool().__call__)
    assert [o.name for o in hir.outputs] == ["state_flag", "state_y"]
    assert {slot.name for slot in hir.state_slots} == {"flag", "y"}


def test_while_loop_lowers_to_back_edge() -> None:
    # A while loop lowers to preheader -> header(loop phi + exit branch) -> body(back-edge jump) -> exit(ret).
    def f(a: float) -> float:
        x = a
        while x < 10.0:
            x = x + 1.0
        return x

    hir = optimize(lower(f))
    assert len(hir.blocks) == 4
    header = next(b for b in hir.blocks if b.phis and isinstance(b.terminator, Branch))
    assert len(header.phis) == 1
    # the body closes the loop with a back-edge to the (lower-id) header from below
    body = next(
        b
        for b in hir.blocks
        if isinstance(b.terminator, Jump) and b.terminator.target == header.id and b.id > header.id
    )
    assert body.id > header.id


def test_while_loop_with_else_is_unsupported() -> None:
    def f(a: float) -> float:
        x = a
        while x < 10.0:
            x = x + 1.0
        else:
            x = x + 100.0
        return x

    with pytest.raises(UnsupportedConstruct, match="else"):
        lower(f)


def test_loop_else_write_keeps_attribute_assigned_so_the_loop_else_is_rejected() -> None:
    # Regression (Codex review): a ``for``/``while`` ``else`` clause runs when the loop completes, so a self-attr write
    # there is a real assignment the read-only-attribute scan must see. If it did not, the attribute would fold to
    # read-only, the guard gating the (lowering-unsupported) loop-else would fold it away as dead, and the program would
    # be silently COMPILED instead of rejected -- a behavior change, not a diagnostic shift. The scan must descend the
    # loop ``else`` so ``self.flag`` stays assigned, ``if self.flag`` stays a runtime branch, and lowering reaches and
    # rejects the loop-else.
    class K:
        def __init__(self) -> None:
            self.flag = True
            self.y = 0.0

        def __call__(self, x: float) -> float:
            if self.flag:
                self.y = x
            else:
                for _ in range(1):
                    pass
                else:
                    self.flag = False  # the loop-else assigns flag; it must not be treated as a read-only constant
            return self.y

    with pytest.raises(UnsupportedConstruct, match="else"):
        lower(K().__call__)


def test_while_else_write_keeps_attribute_assigned_so_the_while_else_is_rejected() -> None:
    # The ``while`` companion to the ``for`` else regression above. A ``while`` else likewise runs on a reachable path
    # (a runtime guard keeps it from being statically dead), so the read-only scan must descend it; otherwise the same
    # silent-compile-instead-of-reject hazard applies to the ``while`` branch of the unified driver.
    class K:
        def __init__(self) -> None:
            self.flag = True
            self.y = 0.0

        def __call__(self, x: float) -> float:
            if self.flag:
                self.y = x
            else:
                while x < 0.0:
                    x = x + 1.0
                else:
                    self.flag = False  # the while-else assigns flag; it must not be treated as a read-only constant
            return self.y

    with pytest.raises(UnsupportedConstruct, match="else"):
        lower(K().__call__)


def test_return_inside_while_lowers_with_the_early_return_preserved() -> None:
    # The new frontend supports an early return inside a while body: both arms here return ``x``, so the lowered
    # kernel returns its input for every path (verified against the numerical model).
    def f(a: float) -> float:
        x = a
        while x < 10.0:
            return x
        return x

    assert [o.name for o in lower(f).outputs] == ["out_0"]


def _helper_with_return_in_while(a: float) -> float:
    while a > 0.0:
        return a
    return a + 1.0


def test_return_inside_inlined_callee_while_lowers_correctly() -> None:
    # The new frontend inlines a callee whose while body has an early return: ``_helper_with_return_in_while`` returns
    # ``a`` when ``a > 0`` else ``a + 1``, so the inlined kernel lowers to that same conditional (verified against the
    # numerical model: 3 -> 3, -2 -> -1, 0 -> 1), with the single ``a + 1`` on the fall-through path.
    def kernel(x: float) -> float:
        return _helper_with_return_in_while(x)

    hir = lower(kernel)
    assert [o.name for o in hir.outputs] == ["out_0"]
    assert _arith_count(hir, FloatAdd) == 1


def test_statically_false_while_is_skipped() -> None:
    # Regression (Codex): a statically-false while never runs, so its body is not lowered -- no spurious persistent
    # state from a body write, and a return in the dead body does not reach the single-exit rejection (it is skipped).
    class DeadWhileWrite:
        def __init__(self) -> None:
            self.s = 0.0

        def __call__(self, x: float) -> float:
            while False:
                self.s = 1.0
            return x

    hir = lower(DeadWhileWrite().__call__)
    assert [slot.name for slot in hir.state_slots] == []  # the dead body's write is not state
    assert [o.name for o in hir.outputs] == ["out_0"]
    assert len(optimize(hir).blocks) == 1  # the loop is skipped, no back-edge emitted

    def dead_while_return(a: float) -> float:
        while False:
            return a
        return a + 1.0

    lowered = lower(dead_while_return)  # the dead body's return is skipped, not rejected
    assert _arith_count(lowered, FloatAdd) == 1


def test_for_loop_over_unroll_threshold_is_unsupported() -> None:
    def f(a: float) -> float:
        x = a
        for _ in range(1000):
            x = x + a
        return x

    with pytest.raises(UnsupportedConstruct, match="unroll threshold"):
        lower(f)


def test_enormous_range_is_rejected_not_crashed() -> None:
    # Regression (Codex F10): a trip count beyond a C ssize_t must be rejected cleanly, not crash with OverflowError
    # from len(range(...)) (the unroll threshold is checked with a big-integer trip count).
    def f(a: float) -> float:
        x = a
        for _ in range(100000000000000000000000000000000000000):
            x = x + a
        return x

    with pytest.raises(UnsupportedConstruct, match="unroll threshold"):
        lower(f)


def test_range_zero_step_is_rejected() -> None:
    def f(a: float) -> float:
        x = a
        for _ in range(0, 10, 0):
            x = x + a
        return x

    with pytest.raises(UnsupportedConstruct, match="must not be zero"):
        lower(f)


@pytest.mark.skip(reason="FIR_PARITY_PENDING: runtime subscript/indexing — stage 9")
def test_divergent_loop_counter_as_static_index_is_rejected() -> None:
    # A counter left differing by the two branch arms must not leak as a trusted compile-time index: using it to index
    # a table afterwards is path-dependent and must be rejected, not silently compiled to one arm's value.
    def f(a: float) -> float:
        table = (10.0, 20.0)
        if a > 0.0:
            for i in range(1):  # leaves i == 0
                pass
        else:
            for i in range(2):  # leaves i == 1
                pass
        return table[i]

    with pytest.raises(UnsupportedConstruct, match="compile-time integer"):
        lower(f)


def test_agreeing_loop_counter_as_static_index_after_branch() -> None:
    # When both arms leave the same counter value, it stays a usable compile-time index past the merge.
    def f(a: float) -> float:
        table = (10.0, 20.0)
        if a > 0.0:
            for i in range(1):
                pass
        else:
            for i in range(1):
                pass
        return table[i]

    lower(f)  # both arms leave i == 0, so table[i] resolves at compile time


def test_attribute_written_only_in_while_is_not_read_only() -> None:
    # Regression: the read-only-attribute scan (_collect_assigned) must descend `while` bodies, not just `if`/`for`.
    # An attribute written only inside a while loop is genuinely runtime-varying state; if the scan misses the write it
    # is misclassified as read-only and a later branch on it folds against the (stale) reset snapshot -- a SILENT
    # MISCOMPILATION that takes a fixed arm for every input. Here ``acc`` becomes 3*x at runtime (reset 0.0), so the
    # guard ``acc > 1.0`` is genuinely dynamic and must emit a real branch, not fold to the reset's (false) arm.
    class WhileWrittenGuard:
        def __init__(self) -> None:
            self.acc = 0.0

        def __call__(self, x: float) -> float:
            c = 3.0
            while c > 0.0:
                self.acc = self.acc + x
                c = c - 1.0
            if self.acc > 1.0:  # acc is runtime state, NOT the read-only reset 0.0: must stay a real branch
                r = 100.0
            else:
                r = -100.0
            return r

    hir = lower(WhileWrittenGuard().__call__)
    assert [slot.name for slot in hir.state_slots] == ["acc"]  # acc is persistent state
    # The acc-guard must be a real runtime branch (plus the while's own exit branch): two branches, not one folded away.
    assert sum(1 for b in hir.blocks if isinstance(b.terminator, Branch)) == 2


def test_loop_carried_attr_in_statically_dead_arm_does_not_crash() -> None:
    # Regression: _loop_assigned must be fold-aware, mirroring lowering. When an attribute's only write inside a while
    # body sits in a statically-dead (constant-folded-away) `if` arm, that write is never reachable, so the attribute
    # is not persistent state. A fold-unaware scan would still list it as loop-carried and crash _lower_while with a
    # KeyError opening a header phi for a value that is not loaded as state.
    class DeadArmCarry:
        def __init__(self) -> None:
            self.acc = 0.0

        def __call__(self, x: float) -> float:
            c = 2.0
            while c > 0.0:
                if 0.5 > 1.0:  # statically false: the only write of acc is unreachable
                    self.acc = self.acc + x
                c = c - 1.0
            return x

    hir = lower(DeadArmCarry().__call__)
    assert [slot.name for slot in hir.state_slots] == []  # the dead write makes acc no state at all


def test_loop_carried_attr_written_only_in_live_folded_arm() -> None:
    # Companion to the above: when the live (folded-true) arm carries the only write, the attribute IS state and the
    # loop lowers without a spurious self-referential header phi for an unwritten value.
    class LiveArmCarry:
        def __init__(self) -> None:
            self.b = 0.5

        def __call__(self, x: float) -> float:
            h = 1.0
            while h > 0.0:
                if 1.5 <= 2.0:  # statically true: this arm's write is the reachable one
                    self.b = self.b + x
                else:
                    self.b = x  # folded-away arm: must not create a phantom carried value
                h = h - 1.0
            return self.b

    hir = lower(LiveArmCarry().__call__)
    assert [slot.name for slot in hir.state_slots] == ["b"]


def test_unknown_global_is_unsupported() -> None:
    def f(a: float) -> float:
        return a + UNDEFINED_GLOBAL  # type: ignore[name-defined, no-any-return]  # noqa: F821

    with pytest.raises(UnsupportedConstruct):
        lower(f)


def test_tan_lowers_to_sin_cos_division() -> None:
    def f(a: float) -> float:
        return math.tan(a)

    hir = lower(f)
    assert _arith_count(hir, FloatSin) == 1
    assert _arith_count(hir, FloatCos) == 1
    assert _arith_count(hir, FloatDiv) == 1


def test_unsupported_library_function_message() -> None:
    def f(a: float) -> float:
        return math.erf(a)

    with pytest.raises(UnsupportedLibraryFunction, match="erf"):
        lower(f)


def test_unsupported_library_function_covers_unregistered_ufuncs() -> None:
    # np.spacing is a ufunc with no fast-math float equivalent (it reads the format's ULP), so it stays unregistered
    # and reports an unimplemented library function rather than a generic unsupported-call.
    def f(a: float) -> float:
        return np.spacing(a)  # type: ignore[no-any-return]

    with pytest.raises(UnsupportedLibraryFunction, match="spacing"):
        lower(f)


@pytest.mark.skip(reason="FIR_PARITY_PENDING: sum() of a runtime aggregate — stage 9")
def test_non_operator_numpy_call_stays_unsupported() -> None:
    def f(a: float) -> float:
        return np.sum(a)  # type: ignore[no-any-return]

    with pytest.raises(UnsupportedConstruct, match="unsupported call to 'sum'"):
        lower(f)


def test_pow_static_integer_exponent_stays_multiplication() -> None:
    # The static-integer path precedes the base-2 exp2 path, so ``2 ** 3`` still unrolls to multiplies.
    def f(x: float) -> float:
        return x * (2**3)

    hir = lower(f)
    assert _arith_count(hir, FloatExp2) == 0


def _integrator_class() -> type:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    from trapezoidal_leaky_streaming_integrator import TrapezoidalLeakyStreamingIntegrator

    return TrapezoidalLeakyStreamingIntegrator


def test_stateful_method_state_slots_and_dedup() -> None:
    integrator = _integrator_class()(k=2**-22)
    hir = lower(integrator.__call__)
    assert hir.input_names() == ["x", "dt"]  # self is dropped; remaining parameters become inputs
    # `return self.y` is deduped onto the public state port state_y; the private _x_prev gets no port, so the output
    # list alone distinguishes public from private. Both slots reset to 0.
    assert [o.name for o in hir.outputs] == ["state_y"]
    slots = {s.name: s for s in hir.state_slots}
    assert set(slots) == {"y", "_x_prev"}
    assert (
        cast(FloatConst, slots["y"].reset_value).value == 0.0
        and cast(FloatConst, slots["_x_prev"].reset_value).value == 0.0
    )
    assert {n.slot for n in hir.nodes.values() if isinstance(n, StateRead)} == {"y", "_x_prev"}


def test_returned_public_state_alias_is_deduped() -> None:
    # The dedup is by dataflow, not spelling: returning a public attribute through an alias must still collapse onto its
    # state_<attr> port rather than emitting a second positional output for the same value.
    class Aliased:
        def __init__(self) -> None:
            self.y = 0.0

        def __call__(self, x: float) -> float:
            self.y = x
            y = self.y
            return y

    hir = lower(Aliased().__call__)
    assert [o.name for o in hir.outputs] == ["state_y"]


def test_mixed_return_dedupes_public_alias_keeps_distinct_leaf() -> None:
    class Mixed:
        def __init__(self) -> None:
            self.y = 0.0

        def __call__(self, x: float) -> tuple[float, float]:
            self.y = x * 2.0
            a = self.y
            return (a, x)  # a aliases public self.y (deduped to state_y); x is distinct (keeps its positional out_1)

    hir = lower(Mixed().__call__)
    assert [o.name for o in hir.outputs] == ["out_1", "state_y"]


def test_return_value_equal_to_public_state_is_deduped_even_without_aliasing() -> None:
    # Dedup keys on the value, not provenance: returning x while x is also a public slot's live-out collapses onto that
    # slot's port even though the return never names the attribute. This is safe -- state_last carries the very same
    # wire, so the value stays observable; a separate out_0 would only duplicate it.
    class Passthrough:
        def __init__(self) -> None:
            self.last = 0.0

        def __call__(self, x: float) -> float:
            self.last = x
            return x

    hir = lower(Passthrough().__call__)
    assert [o.name for o in hir.outputs] == ["state_last"]


def test_unreachable_state_write_is_ignored() -> None:
    # A state write after the return is unreachable and never lowered; collecting it must not be attempted (it used to
    # crash with a KeyError). The method synthesizes as if the dead line were not there.
    class Dead:
        def __init__(self) -> None:
            self.y = 0.0

        def __call__(self, x: float) -> float:
            return x
            self.y = x  # unreachable

    hir = lower(Dead().__call__)
    assert [o.name for o in hir.outputs] == ["out_0"]
    assert hir.state_slots == []


def test_attribute_written_only_in_dead_code_reads_as_constant() -> None:
    # An attribute whose only assignment is unreachable is not state: a reachable read of it folds to its snapshot
    # constant, so it gets no slot and no out_<attr> port (whether it is state depends on its write being reachable).
    class Stale:
        def __init__(self) -> None:
            self.y = 5.0

        def __call__(self, x: float) -> float:
            r = x + self.y  # y folds to its snapshot 5.0 -- its only write is dead
            return r
            self.y = x  # unreachable

    hir = lower(Stale().__call__)
    assert hir.state_slots == []
    assert [o.name for o in hir.outputs] == ["out_0"]
    assert all(not (isinstance(n, StateRead) and n.slot == "y") for n in hir.nodes.values())


def test_stateful_readonly_attribute_is_folded_constant() -> None:
    integrator = _integrator_class()(k=2**-22)
    hir = optimize(lower(integrator.__call__))
    assert "k" not in {s.name for s in hir.state_slots}
    assert all(not (isinstance(n, StateRead) and n.slot == "k") for n in hir.nodes.values())


def test_stateful_reset_state_is_the_instance_snapshot() -> None:
    # The reset value is whatever the instance holds at synthesis time, including post-construction mutation.
    integrator = _integrator_class()(k=2**-22)
    integrator.y = 1.5
    slots = {s.name: s for s in lower(integrator.__call__).state_slots}
    assert cast(FloatConst, slots["y"].reset_value).value == 1.5


def test_init_method_target_is_lowered_as_a_state_writer() -> None:
    # An __init__ is just a method that assigns self attributes; the frontend lowers it, treating those attributes as
    # the state it writes (public ones are exposed as state ports, private ones stay internal).
    integrator = _integrator_class()(k=2**-22)
    hir = lower(integrator.__init__)
    assert {slot.name for slot in hir.state_slots} == {"k", "y", "_x_prev"}
    assert [o.name for o in hir.outputs] == ["state_k", "state_y"]


def test_class_object_target_is_rejected() -> None:
    with pytest.raises(SourceUnavailable, match="bound method"):
        lower(_integrator_class())


def test_method_without_return_exposes_public_state() -> None:
    class Accumulator:
        def __init__(self) -> None:
            self.total = 0.0

        def update(self, x: float) -> None:
            self.total = self.total + x

    hir = lower(Accumulator().update)
    assert [o.name for o in hir.outputs] == ["state_total"]
    assert {s.name for s in hir.state_slots} == {"total"}


def test_assigning_uninitialized_attribute_is_rejected() -> None:
    class Bad:
        def __init__(self) -> None:
            self.y = 0.0

        def __call__(self, x: float) -> float:
            self.scratch = x
            return self.y

    with pytest.raises(UnsupportedConstruct, match="does not exist on the component"):
        lower(Bad().__call__)


def test_read_only_self_attribute_real_part_folds_through() -> None:
    # ``self.y.real`` on a read-only float attribute is just ``self.y`` (a float is its own real part); the frontend
    # reads it permissively and folds the access, lowering to ``x + self.y``.
    class ReadsReal:
        def __init__(self) -> None:
            self.y = 0.0

        def __call__(self, x: float) -> float:
            return x + self.y.real

    hir = lower(ReadsReal().__call__)
    assert [o.name for o in hir.outputs] == ["out_0"]


@pytest.mark.skip(reason="FIR_PARITY_PENDING: runtime aggregate value — stage 9")
def test_tuple_build_and_index() -> None:
    def f(a: float, b: float) -> list[float]:
        z = a, b
        return [z[1], z[0]]

    assert [o.name for o in lower(f).outputs] == ["out_0", "out_1"]


@pytest.mark.skip(reason="FIR_PARITY_PENDING: aggregate slicing — stage 9")
def test_list_slice() -> None:
    def f(a: float, b: float, c: float) -> list[float]:
        v = [a, b, c]
        return v[1:3]

    assert [o.name for o in lower(f).outputs] == ["out_0", "out_1"]


@pytest.mark.skip(reason="FIR_PARITY_PENDING: np.array/asarray of runtime values — stage 9")
def test_vector_scalar_broadcast() -> None:
    def f(a: float, b: float) -> list[float]:
        v = np.array([a, b])
        return v * 0.5  # type: ignore[return-value]

    hir = lower(f)
    assert _arith_count(hir, FloatMul) == 2
    assert [o.name for o in hir.outputs] == ["out_0", "out_1"]


@pytest.mark.skip(reason="FIR_PARITY_PENDING: np.array/asarray of runtime values — stage 9")
def test_flatten_collapses_nesting() -> None:
    def f(a: float, b: float) -> list[float]:
        m = np.array([[a], [b]])
        return m.flatten()  # type: ignore[return-value]

    assert [o.name for o in lower(f).outputs] == ["out_0", "out_1"]


def test_index_out_of_range_is_rejected() -> None:
    def f(a: float) -> float:
        v = [a]
        return v[3]

    with pytest.raises(UnsupportedConstruct, match="out of range"):
        lower(f)


@pytest.mark.skip(reason="FIR_PARITY_PENDING: runtime subscript/indexing — stage 9")
def test_indexing_a_scalar_is_rejected() -> None:
    def f(a: float) -> float:
        return a[0]  # type: ignore[no-any-return, index]

    with pytest.raises(UnsupportedConstruct, match="index or slice a scalar"):
        lower(f)


def test_star_unpacking_a_scalar_is_rejected() -> None:
    def f(a: float) -> float:
        return [*a]  # type: ignore[misc, return-value]

    with pytest.raises(UnsupportedConstruct, match="unpack"):
        lower(f)


def test_tuple_unpacking_routes_values() -> None:
    # The right-hand side is built once before any binding, so a swap reads both sources first (no clobber).
    def swap(a: float, b: float) -> list[float]:
        x, y = b, a
        return [x, y]

    hir = lower(swap)
    assert hir.input_names() == ["a", "b"]
    assert [o.value for o in hir.outputs] == [hir.input_ids[1], hir.input_ids[0]]


@pytest.mark.skip(reason="FIR_PARITY_PENDING: starred unpacking targets — stage 9")
def test_starred_and_nested_unpacking_route_values() -> None:
    def f(a: float, b: float, c: float) -> list[float]:
        first, *rest = [a, b, c]
        r0, r1 = rest
        return [first, r0, r1]

    hir = lower(f)
    assert [o.value for o in hir.outputs] == list(hir.input_ids)


def test_chained_assignment_binds_every_target() -> None:
    def f(a: float) -> list[float]:
        x = y = a + a
        return [x, y]

    hir = lower(f)
    out = [o.value for o in hir.outputs]
    assert out[0] == out[1]


def test_unpacking_a_scalar_source_is_rejected() -> None:
    def f(a: float) -> float:
        x, y = a  # type: ignore[misc]
        return x + y  # type: ignore[no-any-return, has-type]

    with pytest.raises(UnsupportedConstruct, match="length of a runtime value"):
        lower(f)


def test_unpacking_arity_mismatch_is_rejected() -> None:
    def f(a: float, b: float, c: float) -> float:
        x, y = [a, b, c]  # type: ignore[misc]
        return x + y  # type: ignore[no-any-return, has-type]

    with pytest.raises(UnsupportedConstruct, match="cannot unpack: expected 2 values"):
        lower(f)


def test_stateful_tuple_assignment_to_attributes() -> None:
    # Unpacking into self attributes must register both as persistent state; the swap reads the live-ins first.
    class Rotate:
        def __init__(self) -> None:
            self.x = 1.0
            self.y = 2.0

        def step(self, k: float) -> float:
            self.x, self.y = self.y, self.x + k
            return self.x

    hir = lower(Rotate().step)
    assert {s.name for s in hir.state_slots} == {"x", "y"}
    assert "state_x" in {o.name for o in hir.outputs}


def test_unpacked_name_shadows_global_callable() -> None:
    # A name bound only via tuple unpacking is local, so a same-named global function is not inlined at a call site;
    # this exercises _collect_local_names descending into unpacking targets.
    def f(a: float) -> float:
        _addmul, b = a, a  # _addmul is now a local value (Python would raise 'float not callable' when called)
        return _addmul(b)  # type: ignore[no-any-return, operator]

    with pytest.raises(UnsupportedConstruct, match="call target is not resolvable"):
        lower(f)


def _addmul(p: float, q: float) -> list[float]:
    return [p + q, p * q]


@pytest.mark.skip(reason="FIR_PARITY_PENDING: runtime aggregate value — stage 9")
def test_inlined_global_function() -> None:
    def f(a: float, b: float) -> list[float]:
        return _addmul(a, b)

    hir = lower(f)
    assert [o.name for o in hir.outputs] == ["out_0", "out_1"]
    assert _arith_count(hir, FloatAdd) == 1 and _arith_count(hir, FloatMul) == 1


@pytest.mark.skip(reason="FIR_PARITY_PENDING: argument unpacking in calls — stage 9")
def test_inlined_global_with_star_args() -> None:
    def f(a: float, b: float) -> list[float]:
        v = [a, b]
        return _addmul(*v)

    hir = lower(f)
    assert _arith_count(hir, FloatAdd) == 1 and _arith_count(hir, FloatMul) == 1


def test_inline_arity_mismatch_is_rejected() -> None:
    def f(a: float) -> float:
        return _addmul(a)  # type: ignore[call-arg, return-value]

    with pytest.raises(UnsupportedConstruct, match="missing argument 'q'"):
        lower(f)


def cbrt(x: float) -> float:
    return x * x  # a user-defined global whose name collides with the same-named intrinsic placeholder


def test_user_global_function_shadows_intrinsic_name() -> None:
    # A module-level def named like an intrinsic is the caller's own function; Python would call it, so it is inlined.
    def f(a: float) -> float:
        return cbrt(a)

    assert _arith_count(lower(f), FloatMul) == 1  # the inlined x * x, not an UnsupportedLibraryFunction rejection


def test_local_name_shadows_global_callable() -> None:
    # A parameter named like a global function refers to the parameter (a value), which is not callable.
    def f(_addmul: float, a: float) -> float:
        return _addmul(a)  # type: ignore[no-any-return, operator]

    with pytest.raises(UnsupportedConstruct, match="call target is not resolvable"):
        lower(f)


@pytest.mark.skip(reason="FIR_PARITY_PENDING: shape query (.ndim/.shape/.T/.flatten) — stage 9")
def test_flatten_on_a_scalar_is_rejected() -> None:
    def f(a: float) -> float:
        return a.flatten()  # type: ignore[no-any-return, attr-defined]

    with pytest.raises(UnsupportedConstruct, match="aggregate"):
        lower(f)


def test_boolean_in_float_arithmetic_is_rejected() -> None:
    # Boolean literals are supported as values (branch conditions, boolean state), but arithmetic on them is not:
    # negating a boolean fails the float operator's operand-type check.
    def f() -> float:
        return -True

    with pytest.raises((UnsupportedConstruct, ValueError)):
        lower(f)


@pytest.mark.skip(reason="FIR_PARITY_PENDING: argument unpacking in calls — stage 9")
def test_abs_accepts_a_star_unpacked_argument() -> None:
    def f(a: float) -> float:
        v = [a]
        return abs(*v)

    assert _arith_count(lower(f), FloatAbs) == 1


@pytest.mark.skip(reason="FIR_PARITY_PENDING: np.array/asarray of runtime values — stage 9")
def test_unary_plus_and_minus_apply_elementwise_to_aggregates() -> None:
    def scalar_ok(a: float) -> float:
        return +a

    assert [o.name for o in lower(scalar_ok).outputs] == ["out_0"]

    def aggregate_ok(a: float, b: float) -> list[float]:
        v = np.array([a, b])
        return +v  # type: ignore[return-value]

    assert [o.name for o in lower(aggregate_ok).outputs] == ["out_0", "out_1"]

    def negated(a: float, b: float) -> list[float]:
        v = np.array([a, b])
        return -v  # type: ignore[return-value]

    hir = lower(negated)
    assert [o.name for o in hir.outputs] == ["out_0", "out_1"]
    assert _arith_count(hir, FloatNeg) == 2


def test_method_style_abs_call_is_rejected() -> None:
    # Only a bare-name abs(...) is the builtin; a method-style a.abs(b) must not be silently treated as it (which would
    # drop the receiver) -- there is no supported scalar method, so it is an unsupported call.
    def f(a: float, b: float) -> float:
        return a.abs(b)  # type: ignore[no-any-return, attr-defined]

    with pytest.raises(UnsupportedConstruct, match="abs"):
        lower(f)


def _rebind_globals(fn: Callable[..., object], **overrides: object) -> Callable[..., object]:
    """A copy of ``fn`` whose module globals carry ``overrides`` (its source stays retrievable via the shared code)."""
    assert isinstance(fn, types.FunctionType)
    copy = types.FunctionType(
        fn.__code__, {**fn.__globals__, **overrides}, fn.__name__, fn.__defaults__, fn.__closure__
    )
    copy.__annotations__ = dict(fn.__annotations__)  # FunctionType does not copy these; the entry point reads them
    return copy


def test_noncallable_global_shadowing_builtin_is_rejected() -> None:
    # A non-callable global shadows the built-in (Python raises TypeError on the call), so the name is not the builtin
    # it spells; holoso must reject rather than silently emitting FloatAbs / the list-tuple identity.
    def use_abs(a: float) -> float:
        return abs(a)

    def use_list(a: float) -> float:
        return list((a, a))  # type: ignore[return-value]

    def use_tuple(a: float) -> float:
        return tuple((a, a))  # type: ignore[return-value]

    # ``None`` shadows too -- it is present-but-non-callable, distinct from an absent global (the _ABSENT sentinel).
    shadows = ((use_abs, {"abs": 5}), (use_abs, {"abs": None}), (use_list, {"list": 5}), (use_tuple, {"tuple": 5}))
    for fn, shadow in shadows:
        with pytest.raises(UnsupportedConstruct, match=r"not resolvable|runtime argument|is not supported in a kernel"):
            lower(_rebind_globals(fn, **shadow))


def test_callable_global_shadowing_abs_is_inlined_not_floatabs() -> None:
    # A callable global named ``abs`` is the caller's own function; Python would call it, so holoso inlines it instead
    # of emitting the FloatAbs builtin -- the non-callable guard must not disturb this legitimate shadow.
    def use_abs(a: float) -> float:
        return abs(a)

    hir = lower(_rebind_globals(use_abs, abs=cbrt))
    assert _arith_count(hir, FloatAbs) == 0 and _arith_count(hir, FloatMul) == 1


def test_unhashable_global_shadowing_registered_name_is_rejected() -> None:
    # A registry lookup on an unhashable shadow must not crash the compiler; the shadow simply misses and gets the
    # standard non-callable diagnostic instead. Various unhashable shapes are covered.
    def use_abs(a: float) -> float:
        return abs(a)

    for shadow in (np.zeros(3), (1.0, [2.0]), {1: 2}, {1, 2}):
        with pytest.raises(UnsupportedConstruct, match=r"not resolvable|runtime argument|is not supported in a kernel"):
            lower(_rebind_globals(use_abs, abs=shadow))


def test_closure_freevar_shadowing_a_registered_name_resolves_to_the_captured_object() -> None:
    # A freevar (enclosing-scope binding) shadows the name Python would call, so holoso resolves it to the captured
    # object -- never the stub/operator it merely spells. A callable freevar is inlined: the user 'pow' computes a - b,
    # not the pow stub's value (regression: it used to lower to the stub, 256 instead of 0). A non-callable freevar is
    # rejected, as Python would raise.
    def make_pow(pow: Callable[[float, float], float]) -> Callable[[float], float]:  # noqa: A002 -- closure shadow
        def kernel(x: float) -> float:
            return pow(x, x)

        return kernel

    def user_pow(a: float, b: float) -> float:
        return a - b

    model = holoso.synthesize(
        make_pow(user_pow), default_ops(FloatFormat(11, 52)), name="freevar_pow"
    ).numerical_model.elaborate()
    for x in (2.0, 5.0):
        assert float(model.run(x)[0]) == user_pow(x, x)  # x - x = 0: the captured function, not the pow stub

    def make_abs(abs: float) -> Callable[[float], float]:  # noqa: A002 -- a non-callable closure shadow
        def kernel(x: float) -> float:
            return abs(x)  # type: ignore[operator, no-any-return]

        return kernel

    with pytest.raises(UnsupportedConstruct, match="not resolvable"):
        lower(make_abs(3.0))


def test_closure_freevar_bound_to_a_library_function_still_dispatches() -> None:
    # The fix must not over-reject: a freevar capturing an actual library function dispatches by identity as usual.
    def make() -> Callable[[float], float]:
        s = math.sin

        def kernel(x: float) -> float:
            return s(x)

        return kernel

    assert _arith_count(lower(make()), FloatSin) == 1


@pytest.mark.skip(reason="FIR_PARITY_PENDING: np.array/asarray of runtime values — stage 9")
def test_call_dispatch_is_by_identity_not_spelling() -> None:
    # Dispatch resolves the callee object, so an aliased import lowers exactly like the canonical spelling -- the numpy
    # array factories and the cast/sequence builtins are matched by identity, not by the name written at the call.
    def use_asarray(a: float, b: float) -> list[float]:
        return aa([a, b])  # type: ignore[name-defined, no-any-return]  # noqa: F821 -- 'aa' is np.asarray, injected

    assert [o.name for o in lower(_rebind_globals(use_asarray, aa=np.asarray)).outputs] == ["out_0", "out_1"]

    def use_float(a: bool) -> float:
        return f(a)  # type: ignore[name-defined, no-any-return]  # noqa: F821 -- 'f' is the builtin float, injected

    assert _arith_count(lower(_rebind_globals(use_float, f=float)), BoolToFloat) == 1


def _module_scoped_helper(a: float) -> float:  # a module global used by the freevar-shadowing test below
    return a + 100.0


def test_freevar_shadowing_a_global_function_is_not_inlined_as_the_global() -> None:
    # A freevar shadows the same-named module global Python would otherwise call. Dispatch is freevar-aware (resolved),
    # so the inline path must not lower the module global in its place -- the captured user function is a closure
    # callable, which is rejected, never silently swapped for the wrong global.
    def outer(helper: Callable[[float], float]) -> Callable[[float], float]:  # noqa: A002 -- shadows the global name
        def kernel(x: float) -> float:
            return helper(x)  # 'helper' is the freevar

        return kernel

    def captured(a: float) -> float:
        return a * 3.0

    kernel = _rebind_globals(outer(captured), helper=_module_scoped_helper)  # freevar helper + a same-named global
    model = holoso.synthesize(
        kernel, default_ops(FloatFormat(11, 52)), name="freevar_helper"
    ).numerical_model.elaborate()
    for x in (2.0, 4.0):
        assert float(model.run(x)[0]) == captured(x)  # the freevar (a*3) is inlined, not the same-named global


@pytest.mark.skip(reason="FIR_PARITY_PENDING: runtime aggregate value — stage 9")
def test_library_stub_error_is_attributed_to_the_call_site() -> None:
    def f(a: float) -> float:
        return math.tan((a, a))  # type: ignore[arg-type]

    with pytest.raises(UnsupportedConstruct, match=r"in tan\(\):") as excinfo:
        lower(f)
    location = excinfo.value.location
    assert location is not None
    assert location.filename == __file__
    assert location.line is not None and "math.tan((a, a))" in location.line


def test_stub_calling_an_unimplemented_library_function_is_reattributed(monkeypatch: pytest.MonkeyPatch) -> None:
    # A stub body can itself call an unimplemented library function, raising UnsupportedLibraryFunction -- a sibling of
    # UnsupportedConstruct under SynthesisError. Re-attribution must catch it too (not just UnsupportedConstruct), so
    # the error points at the user's call site with the concrete type preserved, never the stub-internal location.
    from holoso._frontend._lib import Library
    from holoso._frontend._lib._registry import _REGISTRY

    def sentinel(x: float) -> float:  # a stand-in external callable, mapped into the registry for this test
        return x

    def bad_stub(x: float) -> float:  # a composite whose body calls an unimplemented library function
        return math.erf(x)

    monkeypatch.setitem(_REGISTRY, sentinel, Library(bad_stub))  # type: ignore[arg-type]

    def kernel(x: float) -> float:
        return sentinel(x)

    with pytest.raises(UnsupportedLibraryFunction, match="erf") as excinfo:
        lower(kernel)
    assert "not implemented" in excinfo.value.message  # the concrete unimplemented-function diagnostic is preserved


@pytest.mark.skip(reason="FIR_PARITY_PENDING: np.array-valued state — stage 9")
def test_numpy_array_state_decomposes_like_a_list() -> None:
    import numpy.typing as npt

    @dataclasses.dataclass
    class Filt:
        v: npt.NDArray[np.float64]  # shape-less annotation: holoso infers the length from the reset value

        def step(self, a: float) -> None:
            self.v = self.v * a

    hir = lower(Filt(np.array([1.0, 2.0, 3.0])).step)
    assert {s.name for s in hir.state_slots} == {"v_0", "v_1", "v_2"}
    assert [o.name for o in hir.outputs] == ["state_v_0", "state_v_1", "state_v_2"]


@pytest.mark.skip(reason="FIR_PARITY_PENDING: np.array-valued state — stage 9")
def test_jaxtyping_array_field_lowers_and_is_validated() -> None:
    from jaxtyping import Float64

    @dataclasses.dataclass
    class Filt:
        v: Float64[np.ndarray, "3"]

        def step(self, a: float) -> None:
            self.v = self.v * a

    assert {s.name for s in lower(Filt(np.array([1.0, 2.0, 3.0])).step).state_slots} == {"v_0", "v_1", "v_2"}
    with pytest.raises(UnsupportedConstruct, match="declared array type"):
        lower(Filt(np.array([1.0, 2.0, 3.0, 4.0])).step)  # value shape (4,) violates the declared "3"


@pytest.mark.skip(reason="FIR_PARITY_PENDING: np.array-valued state — stage 9")
def test_numpy_integer_array_values_coerce_to_real() -> None:
    @dataclasses.dataclass
    class Filt:
        v: np.ndarray

        def step(self, a: float) -> None:
            self.v = self.v * a

    assert {s.name for s in lower(Filt(np.array([2, 3])).step).state_slots} == {"v_0", "v_1"}


@pytest.mark.skip(reason="FIR_PARITY_PENDING: np.array/asarray of runtime values — stage 9")
def test_numpy_asarray_is_identity_on_an_aggregate() -> None:
    def f(a: float, b: float) -> list[float]:
        return np.asarray([a, b]).flatten()  # type: ignore[return-value]  # identity in this compile-time model

    assert [o.name for o in lower(f).outputs] == ["out_0", "out_1"]


@pytest.mark.skip(reason="FIR_PARITY_PENDING: aggregate slicing — stage 9")
def test_list_is_identity_on_an_aggregate() -> None:
    def f(a: float, b: float, c: float) -> list[float]:
        v = [a, b, c]
        return list(v[0:2])

    assert [o.name for o in lower(f).outputs] == ["out_0", "out_1"]


def test_list_of_a_scalar_is_rejected() -> None:
    def f(a: float) -> float:
        return list(a)  # type: ignore[no-any-return, call-overload]  # list(scalar) is a Python TypeError

    with pytest.raises(UnsupportedConstruct, match="list"):
        lower(f)


@pytest.mark.skip(reason="FIR_PARITY_PENDING: aggregate slicing — stage 9")
def test_tuple_is_identity_on_an_aggregate() -> None:
    def f(a: float, b: float, c: float) -> list[float]:
        v = [a, b, c]
        return tuple(v[0:2])  # type: ignore[return-value]

    assert [o.name for o in lower(f).outputs] == ["out_0", "out_1"]


def test_numpy_alias_shadowed_by_a_local_is_not_numpy() -> None:
    # ``np`` is rebound to a local value, so ``np.asarray`` is a method call on that value, not the numpy function.
    def f(a: float) -> float:
        np = [a]
        return np.asarray([a])  # type: ignore[no-any-return, attr-defined]

    # The shadowing local is a list, so the read is the (unsupported) list attribute -- a more specific message
    # than the generic runtime-attribute rejection, but the same refusal.
    with pytest.raises(UnsupportedConstruct, match="list method 'asarray'"):
        lower(f)


def test_name_assigned_later_is_local_before_its_assignment() -> None:
    # A name assigned anywhere in a function is local throughout (Python's rule); using it as a global/builtin/numpy
    # before that assignment is invalid Python (UnboundLocalError), so holoso rejects it rather than seeing the global.
    def shadows_numpy(a: float) -> float:
        y = np.asarray([a])  # type: ignore[used-before-def]
        np = [a]  # noqa: F841  # makes np local for the whole body
        return y  # type: ignore[return-value]

    with pytest.raises(UnsupportedConstruct):
        lower(shadows_numpy)

    def shadows_builtin(a: float) -> float:
        y = abs(a)  # type: ignore[used-before-def]
        abs = [a]  # noqa: F841  # makes abs local for the whole body
        return y

    with pytest.raises(UnsupportedConstruct, match="may be unbound here"):
        lower(shadows_builtin)


@pytest.mark.skip(reason="FIR_PARITY_PENDING: np.array-valued state — stage 9")
def test_matrix_state_decomposes_row_major() -> None:
    @dataclasses.dataclass
    class Filt:
        m: np.ndarray

        def step(self, a: float) -> None:
            self.m = self.m * a

    hir = lower(Filt(np.array([[1.0, 2.0], [3.0, 4.0]])).step)
    assert [s.name for s in hir.state_slots] == ["m_0_0", "m_0_1", "m_1_0", "m_1_1"]
    assert [cast(FloatConst, s.reset_value).value for s in hir.state_slots] == [1.0, 2.0, 3.0, 4.0]
    assert [o.name for o in hir.outputs] == ["state_m_0_0", "state_m_0_1", "state_m_1_0", "state_m_1_1"]


@pytest.mark.skip(reason="FIR_PARITY_PENDING: np.array-valued state — stage 9")
def test_three_dimensional_array_state_is_rejected() -> None:
    @dataclasses.dataclass
    class Filt:
        m: np.ndarray

        def step(self, a: float) -> None:
            self.m = self.m * a

    with pytest.raises(UnsupportedConstruct, match="3-D"):
        lower(Filt(np.zeros((2, 2, 2))).step)


@pytest.mark.skip(reason="FIR_PARITY_PENDING: argument unpacking in calls — stage 9")
def test_ekf1_stateful_structure() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    import ekf1_stateful

    filt = ekf1_stateful.Ekf1(
        x=[0.0, 0.0, 0.0], P_urt=[1.0, 0.0, 0.0, 1.0, 0.0, 1.0], R_diag=[1.0, 1.0], Q_diag=np.array([1.0, 1.0, 1.0])
    )
    hir = lower(filt.update)
    assert hir.input_names() == ["dt", "u_shunt", "di_dt"]  # self dropped; keyword-only params become inputs
    assert [o.name for o in hir.outputs] == ["state_x_0", "state_x_1", "state_x_2"] + [
        f"state_P_urt_{i}" for i in range(6)
    ]
    assert {s.name for s in hir.state_slots} == {f"x_{i}" for i in range(3)} | {f"P_urt_{i}" for i in range(6)}
    assert _arith_count(hir, FloatDiv) == 1  # the inlined kernel's single 1/x21


@pytest.mark.skip(reason="FIR_PARITY_PENDING: vector/array-valued state — stage 9")
def test_vector_state_decomposes_to_per_element_slots() -> None:
    class Vec:
        def __init__(self) -> None:
            self.v = [1.0, 2.0, 3.0]

        def update(self, a: float) -> None:
            self.v = [self.v[0] + a, self.v[1], self.v[2]]

    hir = lower(Vec().update)
    assert {s.name: cast(FloatConst, s.reset_value).value for s in hir.state_slots} == {
        "v_0": 1.0,
        "v_1": 2.0,
        "v_2": 3.0,
    }
    assert [o.name for o in hir.outputs] == ["state_v_0", "state_v_1", "state_v_2"]


@pytest.mark.skip(reason="FIR_PARITY_PENDING: vector/array-valued state — stage 9")
def test_vector_state_shape_mismatch_is_rejected() -> None:
    class Vec:
        def __init__(self) -> None:
            self.v = [0.0, 0.0]

        def update(self, a: float) -> None:
            self.v = [a]

    with pytest.raises(UnsupportedConstruct, match="2-element vector"):
        lower(Vec().update)


@pytest.mark.skip(reason="FIR_PARITY_PENDING: vector/array-valued state — stage 9")
def test_vector_state_nested_shape_is_rejected() -> None:
    # A nested aggregate has the right leaf count (2) but the wrong shape: the slot layout is a flat 2-vector, so the
    # next transaction would reconstruct a flat shape that disagrees with the one written this transaction.
    class Vec:
        def __init__(self) -> None:
            self.v = [0.0, 0.0]

        def update(self, a: float, b: float) -> None:
            self.v = [[a, b]]  # type: ignore[list-item]

    with pytest.raises(UnsupportedConstruct, match="incompatible shape"):
        lower(Vec().update)


@pytest.mark.skip(reason="FIR_PARITY_PENDING: vector/array-valued state — stage 9")
def test_vector_state_slot_name_collision_is_rejected() -> None:
    # The vector ``v`` decomposes into slot ``v_0``, which would alias the distinct scalar attribute ``v_0``.
    class Vec:
        def __init__(self) -> None:
            self.v = [1.0]
            self.v_0 = 2.0

        def update(self, a: float) -> None:
            self.v = [a]
            self.v_0 = a + 1.0

    with pytest.raises(UnsupportedConstruct, match="aliasing collision"):
        lower(Vec().update)


def test_keyword_only_params_become_inputs() -> None:
    def f(a: float, *, b: float, c: float) -> float:
        return a + b + c

    assert lower(f).input_names() == ["a", "b", "c"]


def test_dataclass_instance_is_stateful() -> None:
    @dataclasses.dataclass
    class Acc:
        total: float
        gain: list  # type: ignore[type-arg]

        def step(self, x: float) -> None:
            self.total = self.total + x * self.gain[0]

    hir = lower(Acc(0.0, [2.0]).step)
    assert {s.name for s in hir.state_slots} == {"total"}  # gain is read-only config, not state
    assert [o.name for o in hir.outputs] == ["state_total"]


def test_first_sample_branch_if_converts_to_one_block() -> None:
    # examples/iir1_lpf.py: a boolean first-sample state and an if/else that both write self.y. The diamond merges a
    # float phi (y) AND a boolean phi (_first), so it if-converts (the float phi to a select, the boolean phi to a
    # bool_select reduced by strength reduction) into a single straight-line block -- no branch survives.
    class Iir:
        def __init__(self) -> None:
            self.alpha = 2**-16
            self.y = 0.0
            self._first = True

        def __call__(self, x: float) -> float:
            if self._first:
                self._first = False
                self.y = x
            else:
                self.y += self.alpha * (x - self.y)
            return self.y

    hir = optimize(lower(Iir().__call__))
    assert len(hir.blocks) == 1
    assert isinstance(hir.blocks[0].terminator, Ret)
    assert not any(isinstance(b.terminator, Branch) for b in hir.blocks)
    assert any(isinstance(n, Operation) and isinstance(n.operator, Select) for n in hir.nodes.values())
    slots = {s.name: s for s in hir.state_slots}
    assert isinstance(slots["_first"].reset_value, BoolConst) and slots["_first"].reset_value.value is True
    assert isinstance(slots["y"].reset_value, FloatConst) and slots["y"].reset_value.value == 0.0
    assert [o.name for o in hir.outputs] == ["state_y"]  # return self.y dedups onto the public state port


def test_nested_if_lowers_through_optimize() -> None:
    # Regression: block visitation must be topological -- an inner if's merge feeds the outer merge phi. The conditions
    # are dynamic comparisons (a read-only boolean attribute would fold to one arm and emit no branch).
    class C:
        def __init__(self) -> None:
            self.y = 0.0

        def __call__(self, x: float, w: float) -> float:
            if x > 0.0:
                if w > 0.0:
                    self.y = x
            return self.y

    raw = lower(C().__call__)
    assert any(isinstance(b.terminator, Branch) for b in raw.blocks)
    hir = optimize(raw)
    # Both nested diamonds are small and pure, so if-conversion collapses them: no branch survives and the merges
    # became selects -- the optimized pipeline must still carry the slot through.
    assert not any(isinstance(b.terminator, Branch) for b in hir.blocks)
    assert any(isinstance(n, Operation) and isinstance(n.operator, Select) for n in hir.nodes.values())
    assert {s.name for s in hir.state_slots} == {"y"}


def test_attribute_written_on_one_arm_becomes_a_phi() -> None:
    # The update lives in only one arm (anti-windup style); its live-out is a phi against the live-in. The condition is
    # a dynamic comparison so a real branch is emitted (a read-only boolean attribute would fold the branch away).
    class Clamp:
        def __init__(self) -> None:
            self.acc = 0.0

        def __call__(self, x: float) -> float:
            if x > 0.0:
                self.acc = x
            return self.acc

    raw = lower(Clamp().__call__)
    slots = {s.name: s for s in raw.state_slots}
    assert isinstance(raw.nodes[slots["acc"].live_out], Phi)
    # The empty-else diamond then if-converts: the merge becomes select(cond, written, live_in) -- a data mux.
    hir = optimize(raw)
    slots = {s.name: s for s in hir.state_slots}
    live_out = hir.nodes[slots["acc"].live_out]
    assert isinstance(live_out, Operation) and isinstance(live_out.operator, Select)


def _op_count(hir: Hir, op_type: type) -> int:
    return sum(1 for n in hir.nodes.values() if isinstance(n, Operation) and type(n.operator) is op_type)


def test_boolean_and_in_condition_lowers_to_combinational_bool_and() -> None:
    def f(x: float, a: float, b: float) -> float:
        return 1.0 if (x > a and x < b) else 0.0

    hir = lower(f)
    assert _op_count(hir, BoolSelect) == 1  # short-circuit ``and`` lowers to one combinational boolean select
    assert _op_count(hir, FloatRelational) == 2


def test_boolean_or_lowers_to_combinational_bool_or() -> None:
    def f(x: float, a: float, b: float) -> float:
        return 1.0 if (x < a or x > b) else 0.0

    hir = lower(f)
    assert _op_count(hir, BoolSelect) == 1  # short-circuit ``or`` lowers to one combinational boolean select
    assert _op_count(hir, FloatRelational) == 2


def test_not_lowers_to_combinational_bool_not() -> None:
    def f(x: float) -> float:
        return -1.0 if not (x > 0.0) else 1.0

    hir = lower(f)
    assert _op_count(hir, BoolNot) == 1
    assert _op_count(hir, FloatRelational) == 1


def test_chained_comparison_lowers_to_two_comparisons_and_one_and() -> None:
    def f(x: float, lo: float, hi: float) -> float:
        return 0.0 if lo < x < hi else x

    hir = lower(f)
    assert _op_count(hir, FloatRelational) == 2
    assert _op_count(hir, BoolSelect) == 1  # the chained comparison's implicit ``and`` is one combinational select


def test_chained_comparison_evaluates_each_operand_once() -> None:
    # The shared middle operand ``x`` feeds both comparisons but is evaluated once: only one Sub (x - 0.5) is built.
    def f(a: float) -> float:
        x = a - 0.5
        return 0.0 if 0.0 < x < 1.0 else x

    hir = lower(f)
    assert _op_count(hir, FloatAdd) == 1  # subtraction lowers to add(+neg); only one, so x was built once


def _branch_count(hir: Hir) -> int:
    return sum(1 for block in hir.blocks if isinstance(block.terminator, Branch))


def test_nested_if_without_else_compiles_like_the_hand_written_and() -> None:
    # ``if A: (if B: S)`` with no ``else`` on either is exactly ``if (A and B): S``. If-conversion turns both the nested
    # form and the hand-written ``and`` into branchless combinational logic (no data-dependent control flow survives),
    # and the two agree on every input. Regression: the nested form must compile to the same behavior as the ``and``.
    def nested(x: float, lo: float, hi: float) -> float:
        r = 0.0
        if x > lo:
            if x < hi:
                r = 1.0
        return r

    def manual(x: float, lo: float, hi: float) -> float:
        r = 0.0
        if x > lo and x < hi:
            r = 1.0
        return r

    def triple(x: float, lo: float, hi: float) -> float:
        r = 0.0
        if x > lo:
            if x < hi:
                if x != 0.0:
                    r = 1.0
        return r

    assert _branch_count(optimize(lower(nested))) == 0  # if-converted: no data-dependent branch survives
    assert _branch_count(optimize(lower(manual))) == 0
    assert _branch_count(optimize(lower(triple))) == 0
    ops = default_ops(FloatFormat(11, 52))
    nested_model = holoso.synthesize(nested, ops, name="nested").numerical_model.elaborate()
    manual_model = holoso.synthesize(manual, ops, name="manual").numerical_model.elaborate()
    for x, lo, hi in ((1.0, 0.0, 2.0), (5.0, 0.0, 2.0), (-1.0, 0.0, 2.0), (1.5, 1.0, 2.0), (0.0, 0.0, 2.0)):
        assert nested_model.run(x, lo, hi) == manual_model.run(x, lo, hi)  # identical behavior to the hand-written and


def test_nested_if_with_outer_else_does_not_fold() -> None:
    # The fold must NOT trigger when the outer ``if`` has an ``else``: ``if A: (if B: S) else: T`` is not
    # ``if (A and B): S``, because T runs whenever ``not A``, whereas ``not (A and B)`` also covers A-and-not-B. Both
    # branches must survive and no spurious conjunction is synthesized.
    def f(x: float, lo: float, hi: float) -> float:
        r = 0.0
        if x > lo:
            if x < hi:
                r = 1.0
        else:
            r = 2.0
        return r

    assert _branch_count(lower(f)) == 2
    assert _op_count(lower(f), BoolAnd) == 0


def test_nested_if_fold_is_suppressed_when_the_inner_test_has_a_walrus() -> None:
    # The fold must NOT absorb an inner test that carries a walrus: nested, the walrus binds only when the outer test
    # holds, but ``A and B`` evaluates B unconditionally -- folding would over-bind it. The two branches must survive.
    def f(x: float) -> float:
        r = 0.0
        if x > 0.0:
            if (t := x * 3.0) < 100.0:
                r = t
        return r

    assert _branch_count(lower(f)) == 2


def test_walrus_in_conditional_expression_arm_is_rejected() -> None:
    # A ternary arm is evaluated only when selected; a walrus binding there cannot leak across arms, so it is rejected.
    def f(x: float) -> float:
        return (t := 1.0) if x > 0.0 else (t := 2.0)

    with pytest.raises(UnsupportedConstruct, match="walrus"):
        lower(f)


def test_walrus_in_and_or_operand_is_rejected() -> None:
    # An ``and``/``or`` operand may be short-circuited (statically dropped by the connective fold, or unevaluated in
    # Python), so whether its walrus binds cannot be reconciled between the scans and lowering -- rejected. (Regression:
    # the scan invalidated the target unconditionally while lowering short-circuited past it, desyncing the two.)
    def f(x: float, y: float) -> float:
        if x > 0.0 and (t := y) > 0.0:
            return t
        return 0.0

    with pytest.raises(UnsupportedConstruct, match="walrus"):
        lower(f)


def test_walrus_in_chained_comparison_is_rejected() -> None:
    def f(x: float) -> float:
        if x < 0.0 < (t := 5.0):
            return t
        return 0.0

    with pytest.raises(UnsupportedConstruct, match="walrus"):
        lower(f)


_DEAD_WALRUS_GLOBAL = 3  # a module global a dead-code walrus shadows below


def test_walrus_in_dead_or_unsupported_statement_still_scopes_the_name_local() -> None:
    # Local-name collection is syntactic, as in Python: a walrus target is a function local throughout the body even in
    # dead or out-of-subset code, so it shadows a same-named global. Here the earlier ``range(_DEAD_WALRUS_GLOBAL)``
    # must see the runtime local (rejected), NOT silently fold the module int 3 from the global.
    def f(x: float) -> float:
        v = x
        for _ in range(_DEAD_WALRUS_GLOBAL):  # type: ignore[used-before-def]
            v = v + 1.0
        return v
        assert (_DEAD_WALRUS_GLOBAL := 2)  # noqa -- unreachable, but makes the name a local for the whole function

    with pytest.raises(UnsupportedConstruct):
        lower(f)


def test_walrus_in_a_nested_scope_default_scopes_the_name_in_the_enclosing_function() -> None:
    # A nested def/lambda is a separate scope, but its default-argument expressions execute in the ENCLOSING scope, so a
    # walrus there binds an enclosing local (as in Python). The earlier ``range(_DEAD_WALRUS_GLOBAL)`` must therefore
    # see the runtime local, not the module int -- even though the lambda is dead code that lowering never reaches.
    def f(x: float) -> float:
        v = x
        for _ in range(_DEAD_WALRUS_GLOBAL):
            v = v + 1.0
        return v
        h = lambda y=(_DEAD_WALRUS_GLOBAL := 1): y  # noqa: E731 -- dead, but its default's walrus is an enclosing local

    with pytest.raises(UnsupportedConstruct):
        lower(f)


def test_walrus_in_while_condition_is_lowered() -> None:
    # A while-condition walrus rebinds every iteration; the frontend lowers the loop and its post-test exit value
    # matches Python for every input.
    def f(x: float) -> float:
        while (x := x - 1.0) > 0.0:
            pass
        return x

    model = holoso.synthesize(f, default_ops(FloatFormat(11, 52)), name="walrus_while").numerical_model.elaborate()
    for start in (3.0, 3.5, 0.5, -1.0):
        x = start
        while (x := x - 1.0) > 0.0:
            pass
        assert float(model.run(start)[0]) == x


_WALRUS_SHADOWED_INT = 3  # a module global an inner walrus shadows in the test below


def test_walrus_target_shadowing_a_global_int_is_a_runtime_local() -> None:
    # Python makes a walrus target a function local for the whole body, shadowing a same-named module global. Using it
    # as a static range bound must therefore see the runtime local (rejected), NOT silently fold the global's value.
    def f(x: float) -> float:
        v = x
        if (_WALRUS_SHADOWED_INT := x) > 0.0:
            for _ in range(_WALRUS_SHADOWED_INT):  # type: ignore[call-overload]  # local float, not module int 3
                v = v + 1.0
        return v

    with pytest.raises(UnsupportedConstruct):
        lower(f)


def test_reassigning_the_instance_parameter_self_is_rejected() -> None:
    # ``self`` is the fixed instance the attributes resolve against, not a value: ``self.x`` keeps reading the original
    # instance regardless of any later ``self = ...``, so rebinding it (any form) would silently miscompile -- rejected.
    class _PlainAssign:
        def __init__(self) -> None:
            self.a = 1.0

        def __call__(self, x: float) -> float:
            self = x  # type: ignore[assignment]
            return self.a

    class _Walrus:
        def __init__(self) -> None:
            self.a = 1.0

        def __call__(self, x: float) -> float:
            y = (self := x)  # type: ignore[assignment]
            return self.a + y

    class _Augmented:
        def __init__(self) -> None:
            self.a = 1.0

        def __call__(self, x: float) -> float:
            self += x  # type: ignore[operator, assignment]
            return self.a

    class _ForCounter:
        def __init__(self) -> None:
            self.a = 1.0

        def __call__(self, x: float) -> float:
            for self in range(2):  # type: ignore[assignment]
                pass
            return self.a

    class _Unpack:
        def __init__(self) -> None:
            self.a = 1.0

        def __call__(self, x: float) -> float:
            self, y = x, x  # type: ignore[assignment]
            return self.a + y

    for ctor in (_PlainAssign, _Walrus, _Augmented, _ForCounter, _Unpack):
        with pytest.raises(UnsupportedConstruct, match="instance parameter"):
            lower(ctor().__call__)


def test_writing_a_self_attribute_and_a_plain_local_named_self_are_accepted() -> None:
    # The rejection must not touch a legitimate attribute write (persistent state) or a plain (non-method) function
    # whose local happens to be named ``self`` -- there is no instance there, so ``self`` is an ordinary local.
    class _StateWrite:
        def __init__(self) -> None:
            self.a = 1.0

        def __call__(self, x: float) -> float:
            self.a = self.a + x
            return self.a

    lower(_StateWrite().__call__)  # no exception

    def plain(x: float) -> float:
        self = x  # an ordinary local in a plain function (no instance)
        return self + 1.0

    lower(plain)  # no exception


def test_conditional_expression_lowers_to_branch_and_phi() -> None:
    def f(x: float, y: float, c: float) -> float:
        return x if c > 0.0 else y

    hir = lower(f)
    assert any(isinstance(node.terminator, Branch) for node in hir.blocks)
    assert any(isinstance(n, Phi) for n in hir.nodes.values())


def test_nested_conditional_expression_clamp_lowers() -> None:
    def f(x: float, lo: float, hi: float) -> float:
        return hi if x > hi else (lo if x < lo else x)

    hir = lower(f)
    assert _op_count(hir, FloatRelational) == 2
    assert sum(1 for n in hir.nodes.values() if isinstance(n, Phi)) >= 2


def test_statically_true_connective_in_condition_does_not_branch() -> None:
    def f(x: float) -> float:
        return 1.0 if (1.0 < 2.0 and 3.0 > 2.0) else 0.0

    hir = lower(f)
    assert len(optimize(hir).blocks) == 1
    assert _op_count(optimize(hir), FloatRelational) == 0
    assert _op_count(hir, BoolAnd) == 0


def test_statically_true_connective_operand_is_dropped() -> None:
    def f(x: float) -> float:
        return 1.0 if (True and x > 0.0) else 0.0

    hir = lower(f)
    assert _op_count(hir, BoolAnd) == 0  # the identity True is dropped; the AND collapses to the single comparison
    assert _op_count(hir, FloatRelational) == 1


def test_float_truthiness_in_a_connective_is_lowered() -> None:
    # ``x and y`` over floats follows Python truthiness: the result is truthy iff both operands are nonzero.
    def f(x: float, y: float) -> float:
        return 1.0 if (x and y) else 0.0

    model = holoso.synthesize(f, default_ops(FloatFormat(11, 52)), name="float_and").numerical_model.elaborate()
    for x, y in ((1.0, 2.0), (0.0, 2.0), (3.0, 0.0), (0.0, 0.0), (-1.0, 5.0)):
        assert float(model.run(x, y)[0]) == (1.0 if (x and y) else 0.0)


def test_chained_comparison_with_boolean_operand_is_rejected() -> None:
    class BoolMid:
        def __init__(self) -> None:
            self.flag = False
            self.y = 0.0

        def __call__(self, x: float) -> float:
            return 1.0 if 0.0 < self.flag < 1.0 else self.y  # noqa -- exercising the rejection

    with pytest.raises(UnsupportedConstruct, match="mixes a boolean and a non-boolean"):
        lower(BoolMid().__call__)


def test_not_of_a_float_is_lowered() -> None:
    # ``not x`` over a float follows Python truthiness: True iff x is zero.
    def f(x: float) -> float:
        return 1.0 if not x else 0.0

    model = holoso.synthesize(f, default_ops(FloatFormat(11, 52)), name="float_not").numerical_model.elaborate()
    for x in (0.0, 1.0, -2.0):
        assert float(model.run(x)[0]) == (1.0 if not x else 0.0)


def test_bool_cast_lowers_to_float_to_bool() -> None:
    def f(x: float, y: float) -> float:
        return 1.0 if bool(x) else y

    hir = lower(f)
    assert _op_count(hir, FloatToBool) == 1


def test_bool_of_a_boolean_is_identity() -> None:
    def f(x: float, a: float) -> float:
        return 1.0 if bool(x > a) else 0.0

    hir = lower(f)
    assert _op_count(hir, FloatToBool) == 0  # bool(<bool>) is identity; only the comparison remains
    assert _op_count(hir, FloatRelational) == 1


def test_bool_cast_rejects_aggregate_argument() -> None:
    def f(x: float, y: float) -> float:
        return 1.0 if bool((x, y)) else 0.0

    with pytest.raises(UnsupportedConstruct, match="runtime arguments"):
        lower(f)


def test_bool_cast_rejects_multiple_arguments() -> None:
    def f(x: float, y: float) -> float:
        return 1.0 if bool(x, y) else 0.0  # type: ignore[call-arg]

    with pytest.raises(UnsupportedConstruct, match="runtime arguments"):
        lower(f)


def test_float_cast_of_bool_lowers_to_bool_to_float() -> None:
    def f(x: float) -> float:
        return float(x > 0.0)

    hir = lower(f)
    assert _op_count(hir, BoolToFloat) == 1
    assert _op_count(hir, FloatRelational) == 1


def test_float_cast_of_float_is_identity() -> None:
    def f(x: float) -> float:
        return float(x) + 1.0

    hir = lower(f)
    assert _op_count(hir, BoolToFloat) == 0  # float(<float>) is identity; no cast op
    assert _op_count(hir, FloatAdd) == 1


def test_cross_domain_cast_chain_lowers() -> None:
    def f(x: float, k: float) -> float:
        return float(x > 0.0) * k

    hir = lower(f)
    assert _op_count(hir, FloatRelational) == 1
    assert _op_count(hir, BoolToFloat) == 1
    assert _op_count(hir, FloatMul) == 1


def test_float_cast_rejects_aggregate_argument() -> None:
    def f(x: float, y: float) -> float:
        return float((x, y))[0]  # type: ignore[no-any-return, index, arg-type]

    with pytest.raises(UnsupportedConstruct, match="runtime arguments"):
        lower(f)


def test_non_boolean_or_operand_before_absorbing_constant_is_rejected() -> None:
    # Regression (Codex): ``x or True`` with a float x must be rejected, not folded to constant True. Python evaluates
    # x first and returns it when falsy, so a non-boolean operand reached before the absorbing constant cannot be
    # silently folded away -- it must be lowered and type-checked.
    def f(x: float) -> float:
        return 1.0 if (x or True) else 0.0

    with pytest.raises(UnsupportedConstruct, match="irreconcilable kinds"):
        lower(f)


def _fn_with_globals(name: str, src: str, extra_globals: dict[str, object]) -> object:
    import linecache

    filename = f"<shadow_{name}>"
    linecache.cache[filename] = (len(src), None, [line + "\n" for line in src.splitlines()], filename)
    namespace = {**extra_globals}
    exec(compile(src, filename, "exec"), namespace)
    return namespace[name]


def test_callable_global_shadowing_bool_is_inlined_not_the_builtin() -> None:
    # Regression (Codex): a callable global named ``bool`` (a callable instance) is what Python would call, so the
    # bare-name ``bool(x)`` is inlined as that call -- NOT the builtin float->bool cast. Here it always returns False,
    # so the kernel is the constant 0.0.
    class AlwaysFalse:
        def __call__(self, x: float) -> bool:
            return False

    f = _fn_with_globals(
        "f", "def f(x: float) -> float:\n    return 1.0 if bool(x) else 0.0\n", {"bool": AlwaysFalse()}
    )
    model = holoso.synthesize(
        cast("Callable[..., object]", f), default_ops(FloatFormat(11, 52)), name="callable_bool"
    ).numerical_model.elaborate()
    for x in (1.0, 5.0, 0.0, -2.0):
        assert float(model.run(x)[0]) == 0.0


def test_static_bool_sees_through_bool_cast_so_return_in_branch_folds() -> None:
    # Regression (Codex round 2): the static evaluator must see through ``bool(<static bool>)`` so a guard like
    # ``if bool(True):`` folds to the taken arm with no branch -- otherwise the return in the dead arm is wrongly
    # rejected as a return-inside-a-branch.
    def f(x: float) -> float:
        if bool(True):
            return 1.0
        return x  # unreachable; must not force a branch nor a return-in-branch rejection

    hir = lower(f)
    assert len(optimize(hir).blocks) == 1


def test_static_bool_cast_short_circuits_a_dead_non_boolean_operand() -> None:
    # ``bool(False) and x`` short-circuits to False exactly like ``False and x``; the dead float operand is not
    # evaluated and must not be rejected (the static evaluator folds the bool() cast of a static bool).
    def f(x: float) -> float:
        return 1.0 if (bool(False) and x) else 0.0

    hir = lower(f)
    assert len(optimize(hir).blocks) == 1
    assert _op_count(hir, FloatToBool) == 0


def test_static_float_sees_through_float_cast_of_a_bool() -> None:
    # ``float(True) > 0.5`` folds: float(<static bool>) is 1.0/0.0, so the comparison and the ternary fold statically.
    def f(x: float) -> float:
        return x if float(True) > 0.5 else 0.0

    hir = lower(f)
    assert len(optimize(hir).blocks) == 1
    assert _op_count(hir, BoolToFloat) == 0


def test_or_true_in_a_condition_folds_and_permits_a_return() -> None:
    # Regression (user): ``X or True`` (X a valid boolean) is the constant True, so the guard must fold to its taken
    # arm with no branch -- including allowing the return in that arm, which a runtime branch would reject.
    def f(x: float) -> float:
        if x > 0.0 or True:
            return 1.0
        return x  # unreachable

    # ``x > 0.0 or True`` is always taken, so the return in that arm is permitted (a runtime branch would reject it)
    # and the function is the constant 1.0 for every input.
    model = holoso.synthesize(f, default_ops(FloatFormat(11, 52)), name="or_true").numerical_model.elaborate()
    for x in (5.0, -3.0, 0.0):
        assert float(model.run(x)[0]) == 1.0


def test_and_false_in_a_condition_folds_to_the_else_arm() -> None:
    def f(x: float) -> float:
        if x > 0.0 and False:
            y = 1.0
        else:
            y = 2.0
        return y

    hir = lower(f)
    assert len(optimize(hir).blocks) == 1
    assert _op_count(hir, BoolAnd) == 0


def test_chained_comparison_with_a_static_true_link_collapses_the_dead_and() -> None:
    # ``0.0 < 1.0 < x`` is ``(0 < 1) and (1 < x)``; the static-true link folds, and the constant folder's identity
    # element collapses ``True and (1 < x)`` to just ``1 < x`` -- no residual dead AND.
    def f(x: float) -> float:
        return 1.0 if 0.0 < 1.0 < x else 0.0

    hir = optimize(lower(f))
    assert _op_count(hir, BoolAnd) == 0
    assert _op_count(hir, FloatRelational) == 1


def test_statically_false_while_still_type_checks_its_condition() -> None:
    # Regression (review): a statically-false ``while`` is skipped, but its condition must still be type-checked --
    # ``while x and False:`` with a non-boolean x must be rejected (symmetric with ``if x and False:``), not silently
    # accepted because the loop never runs.
    def f(x: float) -> float:
        while x and False:
            x = x + 1.0
        return x

    with pytest.raises(UnsupportedConstruct, match="irreconcilable kinds merge here"):
        lower(f)


def test_statically_false_while_with_a_boolean_condition_is_skipped() -> None:
    def f(x: float) -> float:
        while x > 0.0 and False:  # a valid boolean condition that is statically false: the loop never runs
            x = x + 1.0
        return x

    # The statically-false guard means the body never executes, so the input passes through unchanged.
    model = holoso.synthesize(
        f, default_ops(FloatFormat(11, 52)), name="static_false_while"
    ).numerical_model.elaborate()
    for x in (5.0, -3.0, 0.0):
        assert float(model.run(x)[0]) == x


def test_reachability_folds_through_a_bool_cast_of_a_connective() -> None:
    # ``bool(X or True)`` carries the truthiness of ``X or True`` (= True), so the guard folds and the return is
    # allowed.
    def f(x: float) -> float:
        if bool(x > 0.0 or True):
            return 1.0
        return x

    assert len(optimize(lower(f)).blocks) == 1  # the folded guard leaves a trivial jump chain that pruning merges


def test_ternary_condition_with_equal_arms_folds() -> None:
    # ``True if x > 0.0 else True`` is True regardless of the (runtime) test, so the enclosing guard folds and the
    # return is not rejected as branch-nested (the inner ternary still lowers, but the outer ``if`` takes no branch).
    def f(x: float) -> float:
        if True if x > 0.0 else True:
            return 1.0
        return x

    lower(f)  # must not raise (the return is reachable, not inside a branch)


def test_readonly_scan_stops_at_a_returning_folded_arm() -> None:
    # Regression (review #1): a folded ``if`` whose taken arm returns makes the rest unreachable; the read-only scan
    # must stop there, so an attribute assigned only afterwards is not wrongly counted as written. Here ``gate`` is
    # read-only, so the first guard folds and its return is permitted -- which fails if ``gate`` is mismarked.
    class K:
        def __init__(self) -> None:
            self.gate = True
            self.y = 0.0

        def __call__(self, u: float) -> float:
            if self.gate:
                return u + 1.0
            self.y = u
            if True:
                return self.y
            self.gate = False  # unreachable; must not mark ``gate`` assigned

    assert lower(K().__call__).state_slots == []


def test_float_cast_connective_comparison_condition_folds_without_spurious_state() -> None:
    # Regression (review #2): ``float(X or True) > 0.5`` is the constant True; the guard must fold so the dead else-arm
    # write does NOT become a persistent-state slot (and output port).
    class K:
        def __init__(self) -> None:
            self.y = 0.0
            self.z = 0.0

        def __call__(self, u: float) -> float:
            if float(u > 0.0 or True) > 0.5:
                self.y = u
            else:
                self.z = u  # unreachable
            return self.y

    hir = optimize(lower(K().__call__))
    assert [slot.name for slot in hir.state_slots] == ["y"]
    assert len(hir.blocks) == 1


def test_absorbing_attribute_connective_keeps_a_dead_arm_attribute_read_only() -> None:
    # Regression (review #3): ``self.flag or True`` folds in the read-only scan (attribute opaque, absorbing operand
    # decides it), so ``self.other`` -- written only in the dead else -- stays read-only, and the later guard on it
    # folds rather than leaking ``self.z`` as state.
    class K:
        def __init__(self) -> None:
            self.flag = True
            self.other = True
            self.y = 0.0
            self.z = 0.0

        def __call__(self, u: float) -> float:
            if self.flag or True:
                pass
            else:
                self.other = False  # unreachable
            if self.other:
                self.y = u
            else:
                self.z = u  # unreachable
            return self.y

    hir = lower(K().__call__)
    assert [slot.name for slot in hir.state_slots] == ["y"]


def test_equal_arm_ternary_condition_leaves_no_dead_branch() -> None:
    # Regression (review #4): a ternary whose arms agree is that value with no branch, so a statically-false loop
    # guarded by one is skipped cleanly (no dead diamond left in the CFG).
    def f(x: float) -> float:
        while False if x > 0.0 else False:
            x = x + 1.0
        return x

    assert _branch_count(optimize(lower(f))) == 0  # the equal-arm ternary folds, leaving no dead loop diamond


def test_equal_arm_ternary_value_fold_does_not_bypass_operand_type_checks() -> None:
    # Regression (review, miscompile): the equal-arm ternary VALUE fold must use the strict static evaluator, not the
    # reachability one. ``(float(x or True) > 0.5) if c else (float(x or True) > 0.5)`` has equal arms, but folding it
    # without lowering would skip type-checking ``x or True`` -- accepting a non-boolean x and miscompiling. It must
    # be rejected, exactly as the un-wrapped ``float(x or True) > 0.5`` is.
    def f(x: float, c: float) -> float:
        return 1.0 if ((float(x or True) > 0.5) if c > 0.0 else (float(x or True) > 0.5)) else 0.0

    with pytest.raises(UnsupportedConstruct, match="irreconcilable kinds merge here"):
        lower(f)


def test_read_only_scan_does_not_misfold_a_reassigned_for_counter() -> None:
    # Regression (review, miscompile): the read-only scan must not bind a static ``for`` counter and then fold a
    # counter-dependent condition against a STALE value -- which would drop ``_flag`` from the assigned set, wrongly
    # treat it as read-only, and fold the later ``if self._flag:`` to a fixed arm, diverging from lowering. The scan
    # leaves the counter unbound (conservative), so the body's writes are recorded and ``_flag`` stays state.
    class K:
        def __init__(self) -> None:
            self._flag = False
            self.y = 0.0
            self.z = 0.0

        def __call__(self, u: float) -> float:
            for i in range(1):
                i = u  # type: ignore[assignment]  # the loop counter is reassigned to a runtime value
                if i > 0.0:
                    self._flag = True
            if self._flag:
                self.z = u
            else:
                self.y = u
            return self.y

    slots = {slot.name for slot in lower(K().__call__).state_slots}
    assert "_flag" in slots and "z" in slots


def test_ternary_with_mismatched_scalar_arm_types_is_cleanly_rejected() -> None:
    # Regression (review): a conditional whose arms have different scalar types (a boolean and a float) is out of
    # subset; it must be rejected with a clear UnsupportedConstruct, not leak an internal phi type-mismatch error.
    def f(x: float, c: float) -> float:
        return 1.0 if (False if c > 0.0 else x) else 0.0

    with pytest.raises(UnsupportedConstruct, match="irreconcilable kinds merge here"):
        lower(f)


def test_missing_return_annotation_is_rejected() -> None:
    def f(a: float):  # type: ignore[no-untyped-def]
        return a + 1.0

    with pytest.raises(UnsupportedConstruct, match="return type must be explicitly annotated"):
        lower(f)


def test_return_annotation_scalar_type_mismatch_is_rejected() -> None:
    def f(a: float) -> bool:
        return a + 1.0  # type: ignore[return-value]

    with pytest.raises(UnsupportedConstruct, match="return type mismatch"):
        lower(f)


def test_return_annotation_bool_declared_float_inferred_is_rejected() -> None:
    def f(a: float) -> float:
        return a > 0.0

    with pytest.raises(UnsupportedConstruct, match="return type mismatch"):
        lower(f)


def test_unsupported_return_annotation_is_rejected() -> None:
    def f(a: float) -> int:
        return a  # type: ignore[return-value]  # int is now valid, but a float value cannot match a declared int return

    with pytest.raises(UnsupportedConstruct, match="return type mismatch"):
        lower(f)


def test_scalar_declared_but_tuple_returned_is_rejected() -> None:
    def f(a: float) -> float:
        return a, a  # type: ignore[return-value]

    with pytest.raises(UnsupportedConstruct, match="declared float, returns an aggregate"):
        lower(f)


def test_return_tuple_arity_mismatch_is_rejected() -> None:
    def f(a: float) -> tuple[float, float, float]:
        return a, a  # type: ignore[return-value]

    with pytest.raises(UnsupportedConstruct, match="arity mismatch"):
        lower(f)


def test_return_none_declared_but_value_returned_is_rejected() -> None:
    # The return annotation is validated, per the design contract: a ``-> None`` kernel that returns a value is a
    # located mismatch, never silently lowered against its own signature.
    def f(a: float) -> None:
        return a  # type: ignore[return-value]

    with pytest.raises(UnsupportedConstruct, match="-> None"):
        lower(f)


def test_return_value_declared_but_method_returns_nothing_is_rejected() -> None:
    class Acc:
        def __init__(self) -> None:
            self._acc = 0.0

        def update(self, x: float) -> float:  # type: ignore[return]
            self._acc = self._acc + x

    with pytest.raises(UnsupportedConstruct, match="returns nothing"):
        lower(Acc().update)


def test_tuple_return_annotation_accepted() -> None:
    def f(a: float, b: float) -> tuple[float, bool]:
        return a + b, a > b

    assert [port.name for port in lower(f).outputs] == ["out_0", "out_1"]


def test_variadic_tuple_return_annotation_accepted() -> None:
    def f(a: bool, b: bool) -> tuple[bool, ...]:
        return a, b, a and b

    assert [port.name for port in lower(f).outputs] == ["out_0", "out_1", "out_2"]


def test_list_return_annotation_accepted() -> None:
    def f(a: float, b: float) -> list[float]:
        return [a, b]

    assert [port.name for port in lower(f).outputs] == ["out_0", "out_1"]


def test_none_return_annotation_accepted_for_stateful_method() -> None:
    class Acc:
        def __init__(self) -> None:
            self._acc = 0.0

        def update(self, x: float) -> None:
            self._acc = self._acc + x

    lower(Acc().update)


def test_scalar_returned_but_tuple_declared_is_rejected() -> None:
    # The return annotation is validated, per the design contract: an aggregate annotation demands an aggregate
    # value, so a scalar return under ``tuple[...]`` is a located mismatch.
    def f(a: float) -> tuple[float, float]:
        return a  # type: ignore[return-value]

    with pytest.raises(UnsupportedConstruct, match="aggregate return"):
        lower(f)


_STATIC_PAIR = np.array([1.0, 2.0])


def test_shaped_array_ports_are_honest_contract_rejections() -> None:
    # A fixed-shape jaxtyping parameter or return parses as a real contract and rejects honestly pending the
    # aggregate stages -- never a silent scalar seed that later surfaces as a nonsense diagnostic ("@ is not
    # defined for scalars" on a matrix kernel).
    from jaxtyping import Float64

    def array_parameter(v: Float64[np.ndarray, "3"]) -> float:
        return v[0]  # type: ignore[no-any-return]

    def array_return(x: float) -> Float64[np.ndarray, "2"]:
        return _STATIC_PAIR

    with pytest.raises(UnsupportedConstruct, match="array ports are not lowerable yet"):
        lower(array_parameter)
    with pytest.raises(UnsupportedConstruct, match="array returns are not lowerable yet"):
        lower(array_return)


def test_malformed_list_return_annotation_is_rejected() -> None:
    def f(a: float) -> list[float, float]:  # type: ignore[type-arg]
        return [a, a]

    with pytest.raises(UnsupportedConstruct, match="exactly one element type"):
        lower(f)


def test_nested_tuple_return_annotation_accepted() -> None:
    def f(a: float, b: float) -> tuple[tuple[float, float], bool]:
        return (a, b), a > b

    assert [port.name for port in lower(f).outputs] == ["out_0_0", "out_0_1", "out_1"]


def test_nested_tuple_return_shape_mismatch_is_rejected() -> None:
    def f(a: float, b: float) -> tuple[tuple[float, float], float]:
        return a, a + b  # type: ignore[return-value]

    with pytest.raises(UnsupportedConstruct, match="declared a tuple, returns a scalar"):
        lower(f)


def test_explicit_return_none_is_accepted_for_stateful_method() -> None:
    class Acc:
        def __init__(self) -> None:
            self._acc = 0.0

        def update(self, x: float) -> None:
            self._acc = self._acc + x
            return None

    lower(Acc().update)


# ---------------------------------------------------------------- compile-time shape queries


@pytest.mark.skip(reason="FIR_PARITY_PENDING: runtime subscript/indexing — stage 9")
def test_static_shape_queries_in_index_range_and_branch_positions() -> None:
    from jaxtyping import Float64

    def kernel(v: Float64[np.ndarray, "3"], m: Float64[np.ndarray, "2 3"]) -> float:
        acc = v[0]
        for i in range(1, len(v)):  # len() bounds an unrolled range
            acc = acc + v[i]
        acc = acc + m[m.ndim - 2][m.shape[-1] - 3]  # .ndim and .shape[k] are compile-time integers
        if v.ndim == 1:  # a shape comparison folds, so only the taken arm is lowered
            acc = acc * 2.0
        return acc  # type: ignore[no-any-return]

    hir = lower(kernel)
    assert [o.name for o in hir.outputs] == ["out_0"]
    assert _arith_count(hir, FloatAdd) == 3  # v[0]+v[1]+v[2] then +m[0][0]; the ndim branch adds no add
    assert _arith_count(hir, FloatMul) == 1


@pytest.mark.skip(reason="FIR_PARITY_PENDING: len() of a runtime aggregate — stage 9")
def test_len_follows_python_and_accepts_a_ragged_list() -> None:
    # len() is a Python builtin, not a numpy one, so it counts the items of any aggregate; only .ndim/.shape are
    # numpy-only and rejected on a sequence.
    def ragged(a: float, b: float) -> float:
        rows = [[a, b], [a]]
        acc = 0.0
        for i in range(len(rows)):
            for j in range(len(rows[i])):
                acc = acc + rows[i][j]
        return acc

    assert _arith_count(lower(ragged), FloatAdd) == 3


@pytest.mark.skip(reason="FIR_PARITY_PENDING: shape query (.ndim/.shape/.T/.flatten) — stage 9")
def test_shape_queries_are_rejected_outside_a_static_position() -> None:
    from jaxtyping import Float64

    def ndim_as_value(v: Float64[np.ndarray, "3"]) -> float:
        return float(v.ndim)

    def shape_as_value(v: Float64[np.ndarray, "3"]) -> float:
        return float(v.shape[0])

    def len_as_value(v: Float64[np.ndarray, "3"]) -> float:
        return float(len(v))

    for kernel in (ndim_as_value, shape_as_value, len_as_value):
        with pytest.raises(UnsupportedConstruct, match="compile-time integer"):
            lower(kernel)

    def ndim_of_list(a: float, b: float) -> float:
        rows = [a, b]
        return a if rows.ndim == 1 else b  # type: ignore[attr-defined]

    with pytest.raises(UnsupportedConstruct, match="Python list/tuple"):
        lower(ndim_of_list)

    def len_of_scalar(a: float) -> float:
        acc = 0.0
        for _ in range(len(a)):  # type: ignore[arg-type]
            acc = acc + a
        return acc

    with pytest.raises(UnsupportedConstruct, match="len\\(\\) of a scalar"):
        lower(len_of_scalar)

    def bad_axis(v: Float64[np.ndarray, "3"]) -> float:
        return v[v.shape[2]]  # type: ignore[return-value]

    with pytest.raises(UnsupportedConstruct, match="axis 2 is out of range"):
        lower(bad_axis)


@pytest.mark.skip(reason="FIR_PARITY_PENDING: shape query (.ndim/.shape/.T/.flatten) — stage 9")
def test_numpy_only_shape_queries_are_rejected_on_a_sequence_however_it_is_spelled() -> None:
    # A shape query never lowers its receiver, so the list/tuple rejection has to walk the receiver expression itself.
    # Reaching a list through a subscript, a transpose, or a state attribute must not hand it .ndim/.shape/.T, none of
    # which Python gives a list -- otherwise Holoso would accept a kernel that is not runnable Python.
    def ndim_through_a_subscript(a: float, b: float) -> float:
        rows = [[a, b], [b, a]]
        return a if rows[0].ndim == 1 else b  # type: ignore[attr-defined]

    with pytest.raises(UnsupportedConstruct, match="ndim on a Python list/tuple"):
        lower(ndim_through_a_subscript)

    def transpose_of_a_sequence_in_a_static_position(a: float, b: float) -> float:
        rows = [a, b]
        acc = 0.0
        for i in range(len(rows.T)):  # type: ignore[attr-defined]
            acc = acc + rows[i]
        return acc

    with pytest.raises(UnsupportedConstruct, match="transpose on a Python list/tuple"):
        lower(transpose_of_a_sequence_in_a_static_position)

    class ListState:
        def __init__(self) -> None:
            self.vec = [1.0, 2.0]

        def __call__(self, a: float) -> float:
            return a if self.vec.ndim == 1 else -a  # type: ignore[attr-defined]

    with pytest.raises(UnsupportedConstruct, match="ndim on a Python list/tuple"):
        lower(ListState().__call__)


@pytest.mark.skip(reason="FIR_PARITY_PENDING: aggregate slicing — stage 9")
def test_empty_slice_has_no_shape_but_still_has_a_length() -> None:
    # An empty aggregate is not an array (array_shape says so), so the static shape probe must not report a zero-length
    # axis that lowering can never produce. Iteration and len() still follow Python, which give an empty slice a length.
    from jaxtyping import Float64

    def iterate_an_empty_slice(v: Float64[np.ndarray, "5"]) -> float:
        acc = v[0]
        for x in v[2:2]:
            acc = acc + x
        return acc  # type: ignore[no-any-return]

    assert iterate_an_empty_slice(np.arange(5.0)) == 0.0  # the kernel is runnable Python, and the loop is empty
    assert _arith_count(lower(iterate_an_empty_slice), FloatAdd) == 0

    def ndim_of_an_empty_slice(v: Float64[np.ndarray, "5"]) -> float:
        return v[0] if v[2:2].ndim == 1 else v[1]  # type: ignore[no-any-return]

    with pytest.raises(UnsupportedConstruct, match="rectangular"):
        lower(ndim_of_an_empty_slice)

    def matmul_of_an_empty_slice(v: Float64[np.ndarray, "5"], w: Float64[np.ndarray, "5"]) -> float:
        return v[2:2] @ w  # type: ignore[no-any-return]

    # The stub's own shape guard surfaces the same diagnostic through the operator, at the user's call site.
    with pytest.raises(UnsupportedConstruct, match="rectangular"):
        lower(matmul_of_an_empty_slice)


@pytest.mark.skip(reason="FIR_PARITY_PENDING: runtime subscript/indexing — stage 9")
def test_shape_queries_still_reach_arrays_through_the_same_spellings() -> None:
    # The complement of the rejection above: an array receiver keeps .ndim/.shape/.T through a subscript or a state
    # attribute, and len() keeps working on a list attribute, where Python does give it a length.
    class Mixed:
        def __init__(self) -> None:
            self.m = np.array([[1.0, 2.0], [3.0, 4.0]])
            self.v = [1.0, 2.0]

        def __call__(self, a: float) -> float:
            acc = a
            for i in range(len(self.v)):
                acc = acc + self.v[i]
            for i in range(self.m.ndim):
                acc = acc + self.m[i][i]
            return acc + self.m.T[0][1]  # type: ignore[no-any-return]

    assert [o.name for o in lower(Mixed().__call__).outputs] == ["out_0"]

    from jaxtyping import Float64

    def array_row_has_ndim(m: Float64[np.ndarray, "2 3"]) -> float:
        return m[0][0] if m[0].ndim == 1 else m[1][0]  # type: ignore[no-any-return]

    assert [o.name for o in lower(array_row_has_ndim).outputs] == ["out_0"]


# ---------------------------------------------------------------- comprehensions and aggregate iteration


@pytest.mark.skip(reason="FIR_PARITY_PENDING: len() of a runtime aggregate — stage 9")
def test_list_comprehension_unrolls_and_scopes_its_target() -> None:
    from jaxtyping import Float64

    def scaled(v: Float64[np.ndarray, "3"], s: float) -> Float64[np.ndarray, "3"]:
        return np.array([v[i] * s for i in range(len(v))])

    hir = lower(scaled)
    assert [o.name for o in hir.outputs] == ["out_0", "out_1", "out_2"]
    assert _arith_count(hir, FloatMul) == 3

    def nested(m: Float64[np.ndarray, "2 3"]) -> Float64[np.ndarray, "3 2"]:
        return np.array([[m[i][j] for i in range(2)] for j in range(3)])

    assert [o.name for o in lower(nested).outputs] == [f"out_{i}_{j}" for i in range(3) for j in range(2)]

    def filtered(m: Float64[np.ndarray, "3 3"]) -> Float64[np.ndarray, "3"]:
        return np.array([m[i][j] for i in range(3) for j in range(3) if i < j])

    assert [o.name for o in lower(filtered).outputs] == ["out_0", "out_1", "out_2"]

    def target_does_not_leak(v: Float64[np.ndarray, "3"]) -> float:
        rows = [v[k] for k in range(3)]
        return rows[0] + k  # type: ignore[name-defined, no-any-return]  # noqa: F821

    # Unlike a ``for`` counter, a comprehension target is confined to the comprehension, exactly as in Python.
    with pytest.raises(UnsupportedConstruct, match="unknown name 'k'"):
        lower(target_does_not_leak)


@pytest.mark.skip(reason="FIR_PARITY_PENDING: runtime subscript/indexing — stage 9")
def test_comprehension_yields_a_python_list_not_an_array() -> None:
    from jaxtyping import Float64

    def arithmetic_on_a_comprehension(v: Float64[np.ndarray, "2"]) -> float:
        rows = [v[i] for i in range(2)]
        return (rows * 2.0)[0]  # type: ignore[no-any-return, operator]

    # A comprehension is a Python list, so numpy-only operations need np.array(...) around it, as in Python.
    with pytest.raises(UnsupportedConstruct, match="Python list/tuple"):
        lower(arithmetic_on_a_comprehension)


@pytest.mark.skip(reason="FIR_PARITY_PENDING: runtime subscript/indexing — stage 9")
def test_comprehension_rejections() -> None:
    from jaxtyping import Float64

    def dynamic_filter(v: Float64[np.ndarray, "3"]) -> Float64[np.ndarray, "3"]:
        return np.array([v[i] for i in range(3) if v[i] > 0.0])

    with pytest.raises(UnsupportedConstruct, match="compile-time condition"):
        lower(dynamic_filter)

    def walrus_inside(v: Float64[np.ndarray, "2"]) -> Float64[np.ndarray, "2"]:
        return np.array([(t := v[i]) + t for i in range(2)])

    with pytest.raises(UnsupportedConstruct, match="walrus"):
        lower(walrus_inside)

    def tuple_target(v: Float64[np.ndarray, "2"]) -> float:
        pairs = [[v[0], v[1]]]
        return [a + b for a, b in pairs][0]  # type: ignore[no-any-return]

    with pytest.raises(UnsupportedConstruct, match="comprehension target must be a plain name"):
        lower(tuple_target)

    def over_threshold(a: float) -> float:
        return [a for _ in range(1000)][0]

    with pytest.raises(UnsupportedConstruct, match="unroll threshold"):
        lower(over_threshold)


@pytest.mark.skip(reason="FIR_PARITY_PENDING: iteration over an aggregate — stage 9")
def test_for_loop_iterates_an_aggregate() -> None:
    from jaxtyping import Float64

    def sum_rows(m: Float64[np.ndarray, "2 3"]) -> float:
        acc = 0.0
        for row in m:
            for x in row:
                acc = acc + x
        return acc

    hir = lower(sum_rows)
    assert _arith_count(hir, FloatAdd) == 6  # the 0.0 seed folds away only later, in the optimizer
    assert [o.name for o in hir.outputs] == ["out_0"]

    def iterate_a_scalar(a: float) -> float:
        acc = 0.0
        for x in a:  # type: ignore[attr-defined]
            acc = acc + x
        return acc

    with pytest.raises(UnsupportedConstruct, match="range|aggregate"):
        lower(iterate_a_scalar)


# ---------------------------------------------------------------- statically reachable raise


@pytest.mark.skip(reason="FIR_PARITY_PENDING: shape query (.ndim/.shape/.T/.flatten) — stage 9")
def test_raise_on_a_statically_taken_path_is_a_located_synthesis_error() -> None:
    from jaxtyping import Float64

    def rejects(m: Float64[np.ndarray, "2 3"]) -> float:
        if m.ndim != 1:
            raise ValueError(f"expected 1-D, got {m.ndim}-D with {len(m)} rows")
        return m[0]  # type: ignore[no-any-return]

    with pytest.raises(UnsupportedConstruct, match="expected 1-D, got 2-D with 2 rows") as excinfo:
        lower(rejects)
    assert excinfo.value.location is not None and "raise ValueError" in (excinfo.value.location.line or "")

    def accepts(v: Float64[np.ndarray, "3"]) -> float:
        if v.ndim != 1:
            raise ValueError("expected 1-D")  # the fold never takes this arm, so it never lowers
        return v[0]  # type: ignore[no-any-return]

    assert [o.name for o in lower(accepts).outputs] == ["out_0"]


def test_raise_rejections() -> None:
    def data_dependent(a: float) -> float:
        if a > 0.0:
            raise ValueError("positive")
        return a

    # Hardware cannot signal a runtime exception, so a raise whose reachability depends on data is rejected as such.
    with pytest.raises(UnsupportedConstruct, match="positive"):
        lower(data_dependent)

    def bare(a: float) -> float:
        raise  # noqa: PLE0704

    with pytest.raises(UnsupportedConstruct, match="raise"):
        lower(bare)

    def not_an_exception(a: float) -> float:
        raise a  # type: ignore[misc]

    with pytest.raises(UnsupportedConstruct, match="raise"):
        lower(not_an_exception)

    def runtime_interpolation(a: float) -> float:
        raise ValueError(f"bad {a}")

    with pytest.raises(UnsupportedConstruct, match="raise"):
        lower(runtime_interpolation)


def _raises_under_a_dynamic_guard(a: float) -> float:
    if a > 0.0:
        raise ValueError("positive")
    return a


def test_branch_depth_restarts_per_inlined_function() -> None:
    # Whether a raise is data-dependent is a property of the function that WRITES it, not of a call site that happens to
    # sit in a branch arm. So a callee's own dynamic guard is rejected even from a straight-line caller, and a stub's
    # static guard is still a compile-time rejection even when the call site is inside a dynamic arm.
    def dynamic_guard_in_callee(a: float, b: float) -> float:
        r = b
        if b > 0.0:
            r = _raises_under_a_dynamic_guard(a)
        return r

    with pytest.raises(UnsupportedConstruct, match="positive"):
        lower(dynamic_guard_in_callee)

    def static_guard_in_stub(c: bool, x: float) -> float:
        r = 0.0
        if c:
            r = math.sqrt(-1.0) + x  # a static domain error, though the arm it sits in is data-dependent
        return r

    with pytest.raises(UnsupportedConstruct, match="nonnegative input"):
        lower(static_guard_in_stub)


# ---------------------------------------------------------------- reachability scan vs lowering


@pytest.mark.skip(reason="FIR_PARITY_PENDING: len() of a runtime aggregate — stage 9")
def test_state_write_only_on_a_folded_away_shape_branch_is_not_state() -> None:
    # The scan runs before the body is lowered, so it cannot fold a shape query and descends both arms, registering the
    # write. Lowering folds the branch away and never touches the attribute, which therefore keeps its reset value for
    # good and is not state. Before ``_prune_untouched_state`` this crashed with a raw KeyError from slot registration.
    from jaxtyping import Float64

    class DeadShapeBranch:
        def __init__(self) -> None:
            self.s = 0.0

        def step(self, x: Float64[np.ndarray, "3"]) -> float:
            if len(x) == 4:  # statically false for a declared 3-vector
                self.s = x[0]
            return x[0]  # type: ignore[no-any-return]

    hir = lower(DeadShapeBranch().step)
    assert [slot.name for slot in hir.state_slots] == []
    assert [o.name for o in hir.outputs] == ["out_0"]

    class DeadAggregateLoop:
        def __init__(self) -> None:
            self.a = 0.0

        def step(self, x: float) -> float:
            for _ in []:  # zero trips, so the write is unreachable
                self.a = x
            return x

    assert [slot.name for slot in lower(DeadAggregateLoop().step).state_slots] == []

    class AlsoRead:
        # When the attribute is also READ on a live path, lowering must emit the read before it can know the write is
        # unreachable, so the conservative classification stands and the slot survives as a register that holds its
        # reset value. Correct -- reads see the reset, as in Python -- at the cost of one register and one port.
        def __init__(self) -> None:
            self.s = 0.25

        def step(self, x: Float64[np.ndarray, "3"]) -> float:
            if len(x) == 4:
                self.s = x[0]
            return self.s + x[0]  # type: ignore[no-any-return]

    hir = lower(AlsoRead().step)
    assert [slot.name for slot in hir.state_slots] == ["s"]
    sim = holoso.synthesize(AlsoRead().step, default_ops(FloatFormat(11, 52)), name="alsoread").numerical_model
    simulator = sim.elaborate()
    reference = AlsoRead()
    inputs = np.array([1.5, 0.0, 0.0])
    for _ in range(3):
        returned, state = simulator.run(*inputs.tolist())
        assert float(returned) == pytest.approx(reference.step(inputs))
        assert float(state) == pytest.approx(0.25)  # never written, so the register holds its reset forever


@pytest.mark.skip(reason="FIR_PARITY_PENDING: iteration over an aggregate — stage 9")
def test_state_write_under_an_aggregate_for_is_not_dropped_by_a_stale_counter() -> None:
    # ``for i in <aggregate>`` binds a runtime value, so the target's compile-time binding must be demoted in the
    # reachability scan exactly as lowering demotes it. Otherwise the scan folds ``i == 2.0`` on the leaked counter of
    # the preceding range loop, walks only one arm, misses the write, and the state slot silently disappears -- the
    # module would return the reset constant forever. ``_assign_attr`` now asserts against that direction outright.
    from jaxtyping import Float64

    class StaleCounter:
        def __init__(self) -> None:
            self.s = 0.0

        def step(self, x: Float64[np.ndarray, "2"]) -> float:
            for i in range(3):
                pass
            for i in x:  # i is demoted here; the scan must not keep the leaked value 2
                if i == 2.0:
                    pass
                else:
                    self.s = i
            return self.s

    hir = lower(StaleCounter().step)
    assert [slot.name for slot in hir.state_slots] == ["s"]

    sim = holoso.synthesize(StaleCounter().step, default_ops(FloatFormat(11, 52)), name="stale").numerical_model
    simulator = sim.elaborate()
    inputs = np.array([5.0, 7.0])
    assert float(simulator.run(*inputs.tolist())[0]) == pytest.approx(StaleCounter().step(inputs))


_COMPREHENSION_SHADOW = 1  # a module-level integer constant a comprehension target below deliberately shadows


@pytest.mark.skip(reason="FIR_PARITY_PENDING: iteration over an aggregate — stage 9")
def test_comprehension_target_shadows_a_same_named_module_constant() -> None:
    # A comprehension is its own scope in Python, so its target shadows a same-named global while it is bound. If the
    # target were not registered as a local for its extent, the static evaluators would resolve the global integer
    # behind the binding and fold the comparison to a compile-time answer -- a silent miscompile: the kernel would
    # return all ones instead of a one-hot vector.
    from jaxtyping import Float64

    def one_hot(v: Float64[np.ndarray, "3"]) -> Float64[np.ndarray, "3"]:
        return np.array([1.0 if _COMPREHENSION_SHADOW == 1 else 0.0 for _COMPREHENSION_SHADOW in v])

    hir = lower(one_hot)
    assert _arith_count(hir, FloatRelational) == 3  # three runtime comparisons, not a folded constant

    inputs = np.array([5.0, 1.0, 9.0])
    sim = holoso.synthesize(one_hot, default_ops(FloatFormat(11, 52)), name="onehot").numerical_model.elaborate()
    assert [float(x) for x in sim.run(*inputs.tolist())] == pytest.approx(list(np.asarray(one_hot(inputs))))

    def indexed(v: Float64[np.ndarray, "3"]) -> Float64[np.ndarray, "3"]:
        # The complement: a range-bound target IS a compile-time integer, so its comparison still folds away.
        return np.array([v[i] * 2.0 if i == 1 else v[i] for i in range(3)])

    assert _arith_count(lower(indexed), FloatRelational) == 0 and _arith_count(lower(indexed), FloatMul) == 1


_COMPREHENSION_BOUND = 2  # a module-level integer the outermost generator below reads from the enclosing scope


@pytest.mark.skip(reason="FIR_PARITY_PENDING: np.array/asarray of runtime values — stage 9")
def test_comprehension_scoping_follows_python_exactly() -> None:
    # Python evaluates the OUTERMOST iterable in the enclosing scope, before the comprehension's scope exists, so the
    # range bound below is the module constant, not the (as yet unbound) target that shadows it.
    from jaxtyping import Float64

    def outermost_iterable_reads_the_enclosing_scope(x: float) -> Float64[np.ndarray, "2"]:
        return np.array([x for _COMPREHENSION_BOUND in range(_COMPREHENSION_BOUND)])

    assert [o.name for o in lower(outermost_iterable_reads_the_enclosing_scope).outputs] == ["out_0", "out_1"]
    assert len(outermost_iterable_reads_the_enclosing_scope(1.0)) == 2  # and it is runnable Python

    def inner_generator_sees_the_unbound_comprehension_local(v: Float64[np.ndarray, "2"]) -> float:
        y = v
        return [y for x in range(1) for y in y][0]  # type: ignore[no-any-return]  # Python: UnboundLocalError on the inner y

    with pytest.raises(UnboundLocalError):
        inner_generator_sees_the_unbound_comprehension_local(np.array([1.0, 2.0]))
    with pytest.raises(UnsupportedConstruct, match="unknown name 'y'"):
        lower(inner_generator_sees_the_unbound_comprehension_local)


def test_comprehension_filter_is_lowered_before_it_is_folded() -> None:
    # The filter decides which items exist, so it must fold -- but it is lowered first, exactly as an ``if`` test is,
    # so its operands are type-checked. A fold that never looked at the condition would wave an unsupported operand
    # through whenever the other side of an ``or`` happened to be statically true.
    def unsupported_operand(x: float) -> float:
        return [x for i in range(1) if _returns_a_dict(x) or True][0]

    # The rejection surfaces from BUILDING the filter's callee (its dict literal is unsupported), which is the
    # point: the filter was lowered and expanded rather than being folded away by the statically-true ``or`` arm.
    with pytest.raises(UnsupportedConstruct, match="Dict is not supported"):
        lower(unsupported_operand)


def _returns_a_dict(v: object) -> object:
    return {"not": v}


def test_state_write_after_a_raise_does_not_poison_the_read_only_scan() -> None:
    # A raise ends the block, so the assignment below it is unreachable and must not mark ``flag`` as written --
    # otherwise ``flag`` stops being a read-only constant, the guard becomes a runtime branch, and the raise in the
    # statically-dead else arm is misreported as sitting on a data-dependent path.
    class GuardedByAReadOnlyFlag:
        def __init__(self) -> None:
            self.flag = True
            self.y = 0.0

        def step(self, a: float) -> float:
            if self.flag:
                self.y = a
            else:
                raise ValueError("flag must be set")
                self.flag = False  # noqa: F841  # unreachable: the raise ends the block

            return self.y

    hir = lower(GuardedByAReadOnlyFlag().step)
    assert [slot.name for slot in hir.state_slots] == ["y"]  # flag stays a read-only constant


@pytest.mark.skip(reason="FIR_PARITY_PENDING: len() of a runtime aggregate — stage 9")
def test_an_untouched_state_attribute_is_not_resurrected_by_an_unrelated_branch() -> None:
    # _merge_state must not load the live-in of an attribute NEITHER arm touched: both arms start from the same
    # pre-branch state, so doing so would conjure a register (and a public port) out of a branch that never
    # mentions the attribute, undoing _prune_untouched_state.
    from jaxtyping import Float64

    class DeadWritePlusUnrelatedBranch:
        def __init__(self) -> None:
            self.s = 0.0

        def step(self, x: Float64[np.ndarray, "3"]) -> float:
            if len(x) == 4:  # dead: the write never happens
                self.s = x[0]
            r = x[0]
            if x[1] > 0.0:  # an unrelated dynamic branch, which merges state
                r = x[2]
            return r  # type: ignore[no-any-return]

    hir = lower(DeadWritePlusUnrelatedBranch().step)
    assert [slot.name for slot in hir.state_slots] == []
    assert [o.name for o in hir.outputs] == ["out_0"]


@pytest.mark.skip(reason="FIR_PARITY_PENDING: iteration over an aggregate — stage 9")
def test_a_scan_never_folds_a_shape_query_against_an_environment_lowering_will_not_have() -> None:
    # The loop-carried scan walks a while body BEFORE its phis exist, so the environment it sees is the preheader's.
    # Were it allowed to resolve a name there, it would fold ``i.ndim == 1`` against the scalar ``i`` bound before the
    # loop, miss the state write in the arm it skipped, open no loop phi, and discard the accumulation entirely --
    # a silent miscompile returning the reset value forever.
    from jaxtyping import Float64

    class AccumulateOverRows:
        def __init__(self) -> None:
            self.s = 0.0

        def step(self, c: bool, m: Float64[np.ndarray, "2 3"]) -> float:
            i = 0.0
            while c:
                for i in m:
                    if i.ndim == 1:
                        self.s = self.s + i[0]
                i = 0.0
                c = False
            return self.s

    rows = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert AccumulateOverRows().step(True, rows) == 5.0  # the kernel is runnable Python
    assert [slot.name for slot in lower(AccumulateOverRows().step).state_slots] == ["s"]

    sim = holoso.synthesize(
        AccumulateOverRows().step, default_ops(FloatFormat(11, 52)), name="accumulate_rows"
    ).numerical_model
    assert float(sim.elaborate().run(True, *rows.flatten().tolist())[0]) == pytest.approx(5.0)


def test_a_state_attribute_read_only_inside_a_while_loop_keeps_its_slot() -> None:
    # A while loop restores the pre-loop state environment on exit, dropping whatever its body loaded, so membership
    # there cannot decide whether an attribute was touched. Pruning on it would drop a slot whose StateRead is still
    # in the HIR, leaving the register allocator to trip over an undeclared slot.
    class ReadOnlyInsideLoop:
        def __init__(self) -> None:
            self.gain = 2.0

        def update(self, x: float) -> float:
            v = [1.0, 2.0]
            if len(v) == 3:  # dead: the scan cannot fold it, so ``gain`` is over-registered as state
                self.gain = x
            acc = 0.0
            while acc < x:
                acc = acc + self.gain
            return acc

    assert ReadOnlyInsideLoop().update(5.0) == 6.0
    hir = lower(ReadOnlyInsideLoop().update)
    slots = {slot.name for slot in hir.state_slots}
    reads = {node.slot for node in hir.nodes.values() if isinstance(node, StateRead)}
    assert reads <= slots  # every StateRead names a declared slot

    sim = holoso.synthesize(
        ReadOnlyInsideLoop().update, default_ops(FloatFormat(11, 52)), name="read_only_in_loop"
    ).numerical_model
    assert float(sim.elaborate().run(5.0)[0]) == pytest.approx(6.0)


@pytest.mark.skip(reason="FIR_PARITY_PENDING: shape query (.ndim/.shape/.T/.flatten) — stage 9")
def test_a_shape_query_cannot_slip_past_a_rejection_the_stub_makes() -> None:
    # ``.T`` is rejected on a scalar, so asking for the shape of one must be rejected identically -- otherwise a
    # static position would quietly accept an expression a value position rejects, and neither is runnable Python.
    def transpose_of_a_scalar_as_a_value(a: float) -> float:
        return a.T  # type: ignore[attr-defined, no-any-return]

    def transpose_of_a_scalar_in_a_shape_query(a: float) -> float:
        return a if a.T.ndim == 0 else -a  # type: ignore[attr-defined]

    for kernel in (transpose_of_a_scalar_as_a_value, transpose_of_a_scalar_in_a_shape_query):
        with pytest.raises(UnsupportedConstruct, match="transpose a scalar"):
            lower(kernel)


@pytest.mark.skip(reason="FIR_PARITY_PENDING: shape query (.ndim/.shape/.T/.flatten) — stage 9")
def test_only_a_write_lowering_reaches_is_validated() -> None:
    # The scan walks paths lowering folds away, so it validates nothing: a write it cannot turn into state is passed
    # over, and the rejection happens at the write itself, if and when lowering gets there. A dead branch assigning an
    # attribute the instance never had is dead code, exactly as it is in Python.
    from jaxtyping import Float64

    class DeadWriteToAnUninitializedAttribute:
        def __init__(self) -> None:
            self.ok = 0.0

        def step(self, v: Float64[np.ndarray, "2"]) -> float:
            if v.ndim == 2:  # statically false for a vector
                self.never_initialized = 1.0
            return v[0]  # type: ignore[no-any-return]

    assert DeadWriteToAnUninitializedAttribute().step(np.array([3.0, 4.0])) == 3.0  # runnable Python
    assert [slot.name for slot in lower(DeadWriteToAnUninitializedAttribute().step).state_slots] == []

    class ReachableWriteToAnUninitializedAttribute:
        def __init__(self) -> None:
            self.ok = 0.0

        def step(self, x: float) -> float:
            if x > 0.0:  # a runtime arm is lowered, so the write is reached
                self.never_initialized = x
            return x

    with pytest.raises(UnsupportedConstruct, match="assigned but not initialized"):
        lower(ReachableWriteToAnUninitializedAttribute().step)


def test_an_all_integer_state_selector_stays_a_typed_integer_slot() -> None:
    # An integer reset with only integer stores keeps a typed integer slot: the exact 2**53 + 1 never enters the
    # float bank, the guard compares integer-to-integer, and the kernel is contained at the MIR integer boundary.
    inexact = 2**53 + 1  # the first integer float64 cannot represent

    class Selector:
        def __init__(self) -> None:
            self.selector = inexact
            self.total = 0.0

        def step(self, x: float) -> float:
            if x > 100.0:  # a runtime guard, so `selector` really is persistent state
                self.selector = 0
            if self.selector == 2**53:  # False in Python: the integer slot compares exactly, never rounded
                self.total = self.total + 100.0 * x
            else:
                self.total = self.total + x
            return self.total

    assert Selector().step(1.0) == 1.0  # Python compares the integer exactly
    hir = lower(Selector().step)
    slot = next(s for s in hir.state_slots if s.name == "selector")
    assert isinstance(hir.nodes[slot.live_out].type, IntType)


@pytest.mark.skip(reason="FIR_PARITY_PENDING: vector/array-valued state — stage 9")
def test_an_integer_vector_state_reset_keeps_exact_per_element_slots() -> None:
    class ExactVector:
        # 2**53 itself round-trips into the float bank exactly, and so does any small integer.
        def __init__(self) -> None:
            self.taps = [1, 2**53, -3]
            self.y = 0.0

        def step(self, x: float) -> float:
            self.taps = [self.taps[0], self.taps[1], self.taps[2]]  # written, so the vector really is state
            self.y = self.y + self.taps[1] * x
            return self.y

    assert [slot.name for slot in lower(ExactVector().step).state_slots] == ["taps_0", "taps_1", "taps_2", "y"]


@pytest.mark.skip(reason="FIR_PARITY_PENDING: shape query (.ndim/.shape/.T/.flatten) — stage 9")
def test_state_slot_names_only_collide_among_the_attributes_lowering_keeps() -> None:
    # The scan over-registers `v_0` from a write lowering folds away, and `v_0` is also the first slot of the vector
    # `v`. Checking for the collision before the prune would reject a kernel whose colliding attribute is dead code.
    from jaxtyping import Float64

    class DeadCollider:
        def __init__(self) -> None:
            self.v = np.array([1.0, 2.0])
            self.v_0 = 100.0

        def step(self, x: Float64[np.ndarray, "1"]) -> float:
            if x.ndim == 2:  # dead: the scan cannot fold it, so `v_0` is over-registered
                self.v_0 = x[0]
            self.v = self.v + x[0]
            return self.v[0]  # type: ignore[no-any-return]

    assert [slot.name for slot in lower(DeadCollider().step).state_slots] == ["v_0", "v_1"]

    class LiveCollider:
        def __init__(self) -> None:
            self.v = np.array([1.0, 2.0])
            self.v_0 = 100.0

        def step(self, x: float) -> float:
            self.v_0 = x  # reached, so both attributes really do claim the slot name `v_0`
            self.v = self.v + x
            return self.v[0]  # type: ignore[no-any-return]

    with pytest.raises(UnsupportedConstruct, match="aliasing collision"):
        lower(LiveCollider().step)


@pytest.mark.skip(reason="FIR_PARITY_PENDING: aggregate slicing — stage 9")
def test_iteration_and_shape_queries_reach_every_aggregate_spelling() -> None:
    # A `for` iterates whatever Python iterates: a slice, a transpose, a flattened matrix, a comprehension. The shape
    # queries reach the same values. Each kernel is runnable Python, so a construct Holoso accepts but Python rejects
    # would fail here rather than pass as a spurious positive.
    from jaxtyping import Float64

    def over_a_slice(m: Float64[np.ndarray, "2 3"]) -> float:
        acc = 0.0
        for e in m[0][1:]:
            acc = acc + e
        return acc

    def over_a_transpose(m: Float64[np.ndarray, "2 3"]) -> float:
        acc = 0.0
        for row in m.T:
            acc = acc + row[0]
        return acc

    def over_a_flatten(m: Float64[np.ndarray, "2 2"]) -> float:
        acc = 0.0
        for e in m.flatten():
            acc = acc + e
        return acc

    def over_a_comprehension(v: Float64[np.ndarray, "3"]) -> float:
        acc = 0.0
        for e in [v[i] * 2.0 for i in range(3)]:
            acc = acc + e
        return acc

    def negative_axis_on_a_vector(v: Float64[np.ndarray, "3"]) -> float:
        return v[v.shape[-1] - 1]  # type: ignore[no-any-return]

    m23, m22, v3 = np.arange(6.0).reshape(2, 3), np.arange(4.0).reshape(2, 2), np.arange(3.0)
    for kernel, args in (
        (over_a_slice, (m23,)),
        (over_a_transpose, (m23,)),
        (over_a_flatten, (m22,)),
        (over_a_comprehension, (v3,)),
        (negative_axis_on_a_vector, (v3,)),
    ):
        assert [o.name for o in lower(kernel).outputs] == ["out_0"]
        kernel(*args)  # runnable Python


@pytest.mark.skip(reason="FIR_PARITY_PENDING: shape query (.ndim/.shape/.T/.flatten) — stage 9")
def test_a_raise_message_may_interpolate_a_shape_and_a_counter() -> None:
    from jaxtyping import Float64

    def guard(m: Float64[np.ndarray, "2 3"]) -> float:
        if m.shape[1] == 3:
            raise ValueError(f"width {m.shape[1]} of a {m.ndim}-D value is not allowed")
        return m[0][0]  # type: ignore[no-any-return]

    with pytest.raises(UnsupportedConstruct, match="width 3 of a 2-D value is not allowed"):
        lower(guard)

    def dead_elif_chain(v: Float64[np.ndarray, "3"]) -> float:
        if v.ndim == 3:
            raise ValueError("three")
        elif v.ndim == 2:
            raise ValueError("two")
        return v[0]  # type: ignore[no-any-return]

    assert dead_elif_chain(np.arange(3.0)) == 0.0  # runnable Python: neither arm is taken
    assert [o.name for o in lower(dead_elif_chain).outputs] == ["out_0"]


@pytest.mark.skip(reason="FIR_PARITY_PENDING: runtime subscript/indexing — stage 9")
def test_a_loop_carries_only_attributes_that_are_really_state() -> None:
    # The scan collects self-attribute writes syntactically, so a write it cannot turn into state must not open a
    # loop-header phi for it; otherwise the phi's live-in lookup fails with a bare KeyError.
    from jaxtyping import Float64

    class DeadWriteInLoop:
        def __init__(self) -> None:
            self.ok = 0.0

        def step(self, run: bool, v: Float64[np.ndarray, "3"]) -> float:
            while run:
                if v.ndim == 2:  # dead
                    self.never_initialized = v[1]
                run = False
            return v[0]  # type: ignore[no-any-return]

    assert DeadWriteInLoop().step(True, np.array([2.0, 3.0, 5.0])) == 2.0
    assert [slot.name for slot in lower(DeadWriteInLoop().step).state_slots] == []


@pytest.mark.skip(reason="FIR_PARITY_PENDING: shape query (.ndim/.shape/.T/.flatten) — stage 9")
def test_a_shape_query_reads_the_reset_value_not_the_state_decomposition() -> None:
    # `.ndim` of a read-only 3-D array attribute is a compile-time integer; only STATE is restricted to 1-D and 2-D.
    from jaxtyping import Float64

    class Cube:
        def __init__(self) -> None:
            self.cube = np.zeros((2, 2, 2))

        def step(self, v: Float64[np.ndarray, "2"]) -> float:
            if v.ndim == 2:
                if self.cube.ndim == 3:
                    return v[1]  # type: ignore[no-any-return]
            return v[0]  # type: ignore[no-any-return]

    assert Cube().step(np.array([2.0, 9.0])) == 2.0
    assert [o.name for o in lower(Cube().step).outputs] == ["out_0"]


@pytest.mark.skip(reason="FIR_PARITY_PENDING: aggregate slicing — stage 9")
def test_multi_axis_indexing_validates_its_axes_against_the_shape() -> None:
    from jaxtyping import Float64

    def out_of_range_behind_an_empty_slice(m: Float64[np.ndarray, "2 2"]) -> float:
        if len(m[:0, 99]) == 0:  # Python: IndexError, axis 1 has size 2
            return 17.0
        return -17.0

    with pytest.raises(IndexError):
        out_of_range_behind_an_empty_slice(np.array([[1.0, 2.0], [3.0, 4.0]]))

    # An empty leading slice selects no item, so a per-item bounds check never fires: the axes need an up-front probe.
    with pytest.raises(UnsupportedConstruct, match="invalid index"):
        lower(out_of_range_behind_an_empty_slice)

    def too_many_axes_behind_an_empty_slice(m: Float64[np.ndarray, "2 2"]) -> float:
        if len(m[:0, 0, 0]) == 0:  # Python: IndexError, too many indices
            return 17.0
        return -17.0

    with pytest.raises(IndexError):
        too_many_axes_behind_an_empty_slice(np.array([[1.0, 2.0], [3.0, 4.0]]))
    with pytest.raises(UnsupportedConstruct, match="too many indices"):
        lower(too_many_axes_behind_an_empty_slice)

    def multi_axis_on_an_empty_slice(m: Float64[np.ndarray, "2 2"]) -> float:
        acc = 2.0
        for x in m[0:0, :][:, 0]:
            acc = acc + x
        return acc

    # An empty aggregate is not an array, so it has no axes to index; this must be a located error, not an assertion.
    with pytest.raises(UnsupportedConstruct, match="rectangular"):
        lower(multi_axis_on_an_empty_slice)


@pytest.mark.skip(reason="FIR_PARITY_PENDING: aggregate slicing — stage 9")
def test_a_sequence_stays_a_sequence_through_a_subscript() -> None:
    class ListState:
        def __init__(self) -> None:
            self.values = [1.0, 2.0]

        def step(self, x: float) -> float:
            return x if self.values[0:1].ndim == 1 else -x  # type: ignore[attr-defined]

    with pytest.raises(AttributeError):
        ListState().step(3.0)  # a Python list slice has no .ndim
    with pytest.raises(UnsupportedConstruct, match="ndim on a Python list/tuple"):
        lower(ListState().step)


@pytest.mark.skip(reason="FIR_PARITY_PENDING: runtime subscript/indexing — stage 9")
def test_a_write_is_validated_only_where_lowering_reaches_it() -> None:
    # The scan walks paths lowering folds away, so it cannot validate. Each attribute below is unrepresentable as
    # state; a dead write to it is dead code, a reachable one is an error. Both halves must hold, in a loop body too.
    from jaxtyping import Float64

    class Descriptor:
        def __init__(self) -> None:
            self.__dict__["p"] = 1.0
            self.y = 0.0

        @property
        def p(self) -> float:
            return 2.0

        @p.setter
        def p(self, value: float) -> None:
            pass

    class DeadDescriptorWriteInLoop(Descriptor):
        def step(self, run: bool, v: Float64[np.ndarray, "2"]) -> float:
            while run:
                if v.ndim == 2:  # dead
                    self.p = v[0]
                run = False
            return v[0]  # type: ignore[no-any-return]

    class LiveDescriptorWriteInLoop(Descriptor):
        def step(self, run: bool, v: Float64[np.ndarray, "2"]) -> float:
            while run:
                self.p = v[0]
                run = False
            return v[0]  # type: ignore[no-any-return]

    class DeadCubeWrite:
        def __init__(self) -> None:
            self.cube = np.zeros((2, 2, 2))
            self.ok = 0.0

        def step(self, v: Float64[np.ndarray, "2"]) -> float:
            if v.ndim == 2:  # dead: a 3-D attribute cannot be state
                self.cube = v
            return v[0]  # type: ignore[no-any-return]

    v = np.array([1.0, 2.0])
    assert DeadDescriptorWriteInLoop().step(True, v) == 1.0
    assert [slot.name for slot in lower(DeadDescriptorWriteInLoop().step).state_slots] == []
    assert [slot.name for slot in lower(DeadCubeWrite().step).state_slots] == []

    with pytest.raises(UnsupportedConstruct, match="descriptor"):
        lower(LiveDescriptorWriteInLoop().step)


def test_an_integer_the_float_datapath_cannot_hold_never_enters_it() -> None:
    # An integer that rounds would read back as another number, so a comparison against the source literal flips.
    # The guard is on the value entering the datapath, not on the spelling: a reset, a literal, or a module constant.
    inexact, colliding = 2**53 + 1, 2**53

    class WrittenFromAModuleConstant:
        def __init__(self) -> None:
            self.selector = 0
            self.total = 0.0

        def step(self, x: float) -> float:
            if x > 100.0:
                self.selector = _INEXACT_INTEGER
            self.total = self.total + (100.0 * x if self.selector == colliding else x)
            return self.total

    reference = WrittenFromAModuleConstant()
    assert [reference.step(v) for v in (1.0, 101.0)] == [1.0, 102.0]  # the integer never equals 2**53 in Python
    hir = lower(WrittenFromAModuleConstant().step)  # the inexact integer stays a typed integer, never a rounded float
    selector = next(slot for slot in hir.state_slots if slot.name == "selector")
    assert isinstance(hir.nodes[selector.live_out].type, IntType)

    class HugeReset:
        def __init__(self) -> None:
            self.counter = 10**400  # beyond the float range entirely

        def step(self, x: float) -> float:
            if x > 0.0:
                self.counter = 0
            return x

    hir = lower(HugeReset().step)  # the huge integer is kept exact as an integer, not overflowed into a float
    (counter,) = hir.state_slots
    assert isinstance(hir.nodes[counter.live_out].type, IntType)

    class ReadOnlyInexact:
        # An inexact integer attribute in a float-add position promotes and rounds, exactly as Python's `int + float`
        # promotes -- accepted fastmath precision loss, not a rejection.
        def __init__(self) -> None:
            self.offset = inexact

        def step(self, x: float) -> float:
            return self.offset + x

    rounded = lower(ReadOnlyInexact().step)
    assert float(2**53) in [n.value for n in rounded.nodes.values() if isinstance(n, FloatConst)]


_INEXACT_INTEGER = 2**53 + 1
_BIG_A = 2**53
_BIG_B = 1
_INT_TABLE = np.array([[2**53 + 1, 3]], dtype=np.int64)
_BIG_F = float(2**53)


def test_mixed_int_float_static_comparison_folds_exactly() -> None:
    # Regression (TODO): a static comparison mixing an integer expression with a float must compare exactly, as
    # Python does; a float64 fold of the integer side rounds 2**53 + 1 onto 2**53 and takes the wrong arm silently.
    class WrongArmGuard:
        def __init__(self) -> None:
            self.x = 0.0

        def step(self, v: float) -> float:
            if _BIG_A + _BIG_B == _BIG_F:  # False in Python: the integer sum compares exactly
                self.x = v
            return self.x

    assert WrongArmGuard().step(1.0) == 0.0
    hir = lower(WrongArmGuard().step)
    assert [slot.name for slot in hir.state_slots] == []
    assert len(optimize(hir).blocks) == 1

    class RightArmGuard:
        def __init__(self) -> None:
            self.x = 0.0

        def step(self, v: float) -> float:
            if _BIG_A + _BIG_B > _BIG_F:  # True in Python: the fold must still take the arm, not reject
                self.x = v
            return self.x

    assert RightArmGuard().step(1.0) == 1.0
    assert [slot.name for slot in lower(RightArmGuard().step).state_slots] == ["x"]


def test_read_only_inexact_int_attribute_comparison_folds_exactly() -> None:
    # Regression (TODO): a read-only integer attribute keeps its exact value in a static comparison; the float64
    # fold of the attribute would round it onto the comparand and take the wrong arm silently.
    class Selector:
        def __init__(self) -> None:
            self._sel = 2**53 + 1
            self.y = 0.0

        def step(self, v: float) -> float:
            if self._sel == _BIG_F:  # False in Python
                self.y = v
            return self.y

    assert Selector().step(1.0) == 0.0
    hir = lower(Selector().step)
    assert [slot.name for slot in hir.state_slots] == []
    assert len(optimize(hir).blocks) == 1


def test_np_int_array_element_comparison_follows_numpy_semantics() -> None:
    # Companion pin to the Python-int exactness fix: a numpy scalar operand must NOT be folded exactly, because
    # numpy itself converts an np.int64 to float64 in a comparison -- np.int64(2**53 + 1) == float(2**53) is True in
    # numpy -- so each operand folds under its own source semantics.
    class Selector:
        def __init__(self) -> None:
            self.y = 0.0

        def step(self, v: float) -> float:
            if _INT_TABLE[0, 0] == _BIG_F:  # True under numpy: the element converts to float64 and rounds
                self.y = v
            return self.y

    assert Selector().step(1.0) == 1.0
    hir = lower(Selector().step)
    assert [slot.name for slot in hir.state_slots] == ["y"]
    assert len(optimize(hir).blocks) == 1


def test_equal_inexact_int_ternary_arms_round_like_the_literal() -> None:
    # The equal-arm ternary folds to the one integer, which then promotes into the float add and rounds under
    # fastmath -- the same accepted rounding a plain literal read gets.
    def kernel(x: float, c: bool) -> float:
        return x + (_INEXACT_INTEGER if c else _INEXACT_INTEGER)

    hir = lower(kernel)
    assert float(2**53) in [n.value for n in hir.nodes.values() if isinstance(n, FloatConst)]


def test_literal_exponent_expands_to_a_multiply_chain() -> None:
    # ``x**66`` expands to a chain of multiplies; the frontend lowers it and the result matches Python.
    def kernel(x: float) -> float:
        return x**66

    model = holoso.synthesize(kernel, default_ops(FloatFormat(11, 52)), name="x66").numerical_model.elaborate()
    for x in (1.1, 0.5):
        assert float(model.run(x)[0]) == pytest.approx(x**66, rel=1e-9)


def test_read_only_object_attribute_ndim_folds_as_a_constant() -> None:
    # ``self.config.ndim`` reads the stored object's class attribute (1) as a compile-time constant, so ``ndim == 0``
    # folds to False and the kernel is ``-x`` -- matching Python.
    class Config:
        ndim = 1

    class Kernel:
        def __init__(self) -> None:
            self.config = Config()

        def step(self, x: float) -> float:
            return x if self.config.ndim == 0 else -x  # Python: -x, because Config.ndim is 1

    assert Kernel().step(3.0) == -3.0
    model = holoso.synthesize(
        Kernel().step, default_ops(FloatFormat(11, 52)), name="ndim_fold"
    ).numerical_model.elaborate()
    for x in (3.0, -2.0, 0.0):
        assert float(model.run(x)[0]) == -x


@pytest.mark.skip(reason="FIR_PARITY_PENDING: aggregate slicing — stage 9")
def test_an_empty_aggregate_makes_no_check_vacuous() -> None:
    # An empty aggregate has no leaves, so a per-leaf type check and a per-item shape check both prove nothing.
    from jaxtyping import Float64

    def negate_an_empty_boolean_slice(c: bool) -> float:
        flags = np.array([c])
        invalid = -flags[:0]  # Python: TypeError, numpy cannot negate booleans
        return 17.0 if len(invalid) == 0 else -17.0

    with pytest.raises(TypeError):
        negate_an_empty_boolean_slice(True)
    with pytest.raises(UnsupportedConstruct, match="empty aggregate"):
        lower(negate_an_empty_boolean_slice)

    def add_empty_slices_of_different_widths(a: Float64[np.ndarray, "2 3"], b: Float64[np.ndarray, "2 2"]) -> float:
        invalid = a[:0, :] + b[:0, :]  # Python: ValueError, shapes (0,3) and (0,2)
        return 17.0 if len(invalid) == 0 else -17.0

    with pytest.raises(ValueError):
        add_empty_slices_of_different_widths(np.zeros((2, 3)), np.zeros((2, 2)))
    with pytest.raises(UnsupportedConstruct, match="empty aggregate"):
        lower(add_empty_slices_of_different_widths)


@pytest.mark.skip(reason="FIR_PARITY_PENDING: shape query (.ndim/.shape/.T/.flatten) — stage 9")
def test_indexing_a_sequence_of_arrays_yields_an_array() -> None:
    from jaxtyping import Float64

    def row_of_a_list(a: Float64[np.ndarray, "2"], x: float) -> float:
        rows = [a]
        return x if rows[0].ndim == 1 else -x  # the element is the ndarray, not a list

    assert row_of_a_list(np.zeros(2), 3.0) == 3.0
    assert [o.name for o in lower(row_of_a_list).outputs] == ["out_0"]


@pytest.mark.skip(reason="FIR_PARITY_PENDING: shape query (.ndim/.shape/.T/.flatten) — stage 9")
def test_a_scan_never_rejects_an_arm_lowering_folds_away() -> None:
    # The scan descends both arms of a shape-dependent branch, so it must not validate what it finds there.
    from jaxtyping import Float64

    class DeadInvalidShapeQuery:
        def __init__(self) -> None:
            self.values = [1.0, 2.0]
            self.total = 0.0

        def step(self, v: Float64[np.ndarray, "2"], x: float) -> float:
            if v.ndim == 2:  # dead
                if self.values.ndim == 1:  # type: ignore[attr-defined]  # a list has no .ndim
                    self.total = x
            return self.total

    assert DeadInvalidShapeQuery().step(np.zeros(2), 1.0) == 0.0
    # `total` is read on a live path, so it keeps a register holding its reset; the point is that it LOWERS at all.
    assert [slot.name for slot in lower(DeadInvalidShapeQuery().step).state_slots] == ["total"]


def _assert_shape_kernel_matches_python(fn: Callable[..., float], v: np.ndarray) -> None:
    sim = holoso.synthesize(fn, default_ops(FloatFormat(11, 52)), name=fn.__qualname__.split(".")[0]).numerical_model
    assert [float(x) for x in sim.elaborate().run(*v.tolist())] == pytest.approx([fn(v)])


@pytest.mark.skip(reason="FIR_PARITY_PENDING: runtime subscript/indexing — stage 9")
def test_a_nested_reset_sequence_is_shaped_like_the_aggregate_it_denotes() -> None:
    # `len(self.nested[0])` is 3 in Python, so the snapshot's shape must describe every axis, not just the outermost.
    from jaxtyping import Float64

    class NestedRows:
        def __init__(self) -> None:
            self.nested = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

        def step(self, v: Float64[np.ndarray, "3"]) -> float:
            acc = 0.0
            for i in range(len(self.nested[0])):
                acc = acc + v[i]
            return acc

    v = np.array([10.0, 20.0, 30.0])
    assert NestedRows().step(v) == 60.0
    _assert_shape_kernel_matches_python(NestedRows().step, v)


@pytest.mark.skip(reason="FIR_PARITY_PENDING: runtime subscript/indexing — stage 9")
def test_a_ragged_or_empty_reset_sequence_still_has_a_length() -> None:
    from jaxtyping import Float64

    class RaggedRows:
        def __init__(self) -> None:
            self.ragged = [[1.0], [2.0, 3.0]]
            self.empty: list[float] = []

        def step(self, v: Float64[np.ndarray, "3"]) -> float:
            return v[len(self.ragged)] + v[len(self.ragged[1])] + v[len(self.empty)]  # type: ignore[no-any-return]

    v = np.array([10.0, 20.0, 30.0])
    assert RaggedRows().step(v) == 70.0
    _assert_shape_kernel_matches_python(RaggedRows().step, v)


@pytest.mark.skip(reason="FIR_PARITY_PENDING: runtime subscript/indexing — stage 9")
def test_indexing_a_reset_sequence_of_arrays_yields_an_array() -> None:
    from jaxtyping import Float64

    class ArrayRows:
        def __init__(self) -> None:
            self.rows = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]

        def step(self, v: Float64[np.ndarray, "3"]) -> float:
            return v[self.rows[0].ndim]  # type: ignore[no-any-return]  # the element is the ndarray, not a list

    v = np.array([10.0, 20.0, 30.0])
    assert ArrayRows().step(v) == 20.0
    _assert_shape_kernel_matches_python(ArrayRows().step, v)


@pytest.mark.skip(reason="FIR_PARITY_PENDING: shape query on a list — stage 9")
def test_a_shape_query_on_a_nested_reset_sequence_element_is_rejected() -> None:
    from jaxtyping import Float64

    class NestedNdim:
        def __init__(self) -> None:
            self.nested = [[1.0, 2.0]]

        def step(self, v: Float64[np.ndarray, "3"]) -> float:
            return v[self.nested[0].ndim]  # type: ignore[attr-defined, return-value]  # a list has no .ndim

    with pytest.raises(UnsupportedConstruct, match="ndim on a Python list/tuple"):
        lower(NestedNdim().step)


def test_subscripting_a_non_container_reset_attribute_is_a_located_rejection() -> None:
    # Navigating the reset snapshot must not index whatever `__getitem__` a stored object happens to carry.
    from jaxtyping import Float64

    class ForeignAttr:
        def __init__(self) -> None:
            self.lookup = {"a": 1}

        def step(self, v: Float64[np.ndarray, "3"]) -> float:
            return v[len(self.lookup[0])]  # type: ignore[arg-type, index, no-any-return]  # a KeyError in Python

    with pytest.raises(UnsupportedConstruct):
        lower(ForeignAttr().step)


def test_a_scan_must_not_fold_a_counter_an_empty_aggregate_never_rebinds() -> None:
    # Lowering runs an empty aggregate's body zero times, so `i` keeps its outer value. A scan that walks the body once
    # and adopts the inner counter would fold `i == 1` away and never see the state write it guards.
    class EmptyAggregateCounter:
        def __init__(self) -> None:
            self.s = 0.0

        def step(self, a: float) -> float:
            for i in range(2):
                pass
            for _unused in []:  # type: ignore[var-annotated]
                for i in range(5):  # noqa: B007  # never runs; must not leak i == 4 into the scan
                    pass
            if i == 1:
                self.s = a
            return self.s

    reference = EmptyAggregateCounter()
    assert reference.step(7.0) == 7.0 and reference.s == 7.0
    assert [slot.name for slot in lower(EmptyAggregateCounter().step).state_slots] == ["s"]
    sim = holoso.synthesize(
        EmptyAggregateCounter().step, default_ops(FloatFormat(11, 52)), name="empty_aggregate_counter"
    ).numerical_model.elaborate()
    assert dict(zip([p.name for p in sim.outputs], [float(x) for x in sim.run(7.0)], strict=True))["state_s"] == 7.0


@pytest.mark.skip(reason="FIR_PARITY_PENDING: iteration over an aggregate — stage 9")
def test_a_scan_must_not_fold_a_branch_on_a_counter_the_loop_body_rebinds() -> None:
    # The aggregate loop's first trip rebinds `i` to a runtime value, so lowering takes the else arm on the second
    # trip. A scan that keeps `i == 0` static walks only the then arm and misses `self.s`, whose write then has
    # nowhere to land.
    from jaxtyping import Float64

    class RebindingCounter:
        def __init__(self) -> None:
            self.s = 0.0

        def step(self, v: Float64[np.ndarray, "2"]) -> float:
            for i in range(1):
                pass
            for x in v:
                if i == 0:
                    i = x  # noqa: PLW2901
                else:
                    self.s = x
            return self.s

    reference = RebindingCounter()
    assert reference.step(np.array([5.0, 7.0])) == 7.0
    sim = holoso.synthesize(
        RebindingCounter().step, default_ops(FloatFormat(11, 52)), name="rebinding_counter"
    ).numerical_model.elaborate()
    assert (
        dict(zip([p.name for p in sim.outputs], [float(x) for x in sim.run(5.0, 7.0)], strict=True))["state_s"] == 7.0
    )


@pytest.mark.skip(reason="FIR_PARITY_PENDING: iteration over an aggregate — stage 9")
def test_a_scan_demotes_the_aggregate_target_before_discovering_body_rebinds() -> None:
    # The aggregate loop's target `x` leaks a stale value from an earlier same-named range loop. Discovering what the
    # body rebinds must happen with `x` already demoted, or the fold of `if x != 0` hides the `j = x` rebind, `j` is
    # restored stale, and the guarded state write is missed -- tripping `assert attr in self._state_order`.
    class LeakedAggregateTarget:
        def __init__(self) -> None:
            self.s = 0.0

        def step(self, a: float) -> float:
            for x in range(1):  # noqa: B007  # leaks x == 0
                pass
            for j in range(1):  # noqa: B007  # leaks j == 0
                pass
            for x in [a]:  # type: ignore[assignment]  # noqa: B007  # aggregate: target demoted, body rebinds j
                if x != 0:
                    j = x  # noqa: PLW2901
            if j != 0:
                self.s = j
            return self.s

    reference = LeakedAggregateTarget()
    assert reference.step(5.0) == 5.0 and reference.s == 5.0
    sim = holoso.synthesize(
        LeakedAggregateTarget().step, default_ops(FloatFormat(11, 52)), name="leaked_aggregate_target"
    ).numerical_model.elaborate()
    assert dict(zip([p.name for p in sim.outputs], [float(x) for x in sim.run(5.0)], strict=True))["state_s"] == 5.0


def test_a_tuple_index_of_a_list_reset_attribute_is_a_located_rejection() -> None:
    # `self.rows[0,]` indexes a Python list with a one-tuple, which CPython rejects; the reset-state navigator must
    # not silently reinterpret it as the numpy-style `self.rows[0]`.
    from jaxtyping import Float64

    class TupleIndexedRows:
        def __init__(self) -> None:
            self.rows = [[1.0, 2.0]]

        def step(self, v: Float64[np.ndarray, "3"]) -> float:
            return v[len(self.rows[0,])]  # type: ignore[call-overload,no-any-return]  # a TypeError in Python

    with pytest.raises(UnsupportedConstruct):
        lower(TupleIndexedRows().step)


_INEXACT_COUNTER_START = 2**53 + 1  # a compile-time range bound whose counter no float holds exactly


def test_an_inexact_integer_loop_counter_is_rejected_not_silently_rounded() -> None:
    # A counter the float register cannot hold exactly would round in value position, flipping a comparison against a
    # runtime float. The range binds the single counter 2**53+1, which rounds to 2**53.
    def counter_rounds(x: float) -> float:
        return (  # type: ignore[no-any-return]
            x
            + np.array(
                [1.0 if i == float(2**53) else 0.0 for i in range(_INEXACT_COUNTER_START, _INEXACT_COUNTER_START + 1)]
            )[0]
        )

    assert counter_rounds(0.0) == 0.0  # (2**53+1) == 2.0**53 is False in Python, so the element is 0.0
    # The integer counter is now compared EXACTLY as a MetaInt (Python-faithful), not silently rounded into float64:
    # (2**53+1) != 2**53, so the element folds to 0.0 and the kernel lowers to the correct passthrough ``x``. No
    # datapath materialization of the inexact integer occurs, so there is nothing to reject.
    hir = lower(counter_rounds)
    assert [o.name for o in hir.outputs] == ["out_0"]


@pytest.mark.skip(reason="FIR_PARITY_PENDING: runtime subscript/indexing — stage 9")
def test_a_scalar_takes_an_empty_tuple_key_as_identity_like_a_numpy_scalar() -> None:
    # A Holoso scalar is rank zero, as a numpy scalar is, so `x[()]` yields the scalar itself -- while `x[0]` and a
    # slice, which a numpy scalar also rejects, stay rejected.
    from jaxtyping import Float64

    def element_full_index(v: Float64[np.ndarray, "3"]) -> float:
        return v[0][()]  # type: ignore[no-any-return]  # numpy: v[0], a 0-D identity

    assert element_full_index(np.array([2.0, 4.0, 6.0])) == 2.0
    _assert_shape_kernel_matches_python(element_full_index, np.array([2.0, 4.0, 6.0]))

    def index_a_scalar(v: Float64[np.ndarray, "3"]) -> float:
        return v[0][0]  # type: ignore[no-any-return]  # numpy: IndexError, too many indices for a scalar

    with pytest.raises(IndexError):
        index_a_scalar(np.array([2.0, 4.0, 6.0]))
    with pytest.raises(UnsupportedConstruct, match="cannot index or slice a scalar"):
        lower(index_a_scalar)


def test_an_aggregate_operand_to_an_intrinsic_is_a_located_rejection() -> None:
    # Review round 2: a tuple fed to a scalar intrinsic (valid NumPy, an honest porting mistake) must be a located
    # rejection at analysis, not an internal assertion crash during emission.
    def in_sqrt(x: float) -> float:
        return float(np.sqrt((x, 1.0))[0])

    def in_isfinite(x: float) -> float:
        return 1.0 if math.isfinite((x, 1.0)) else 0.0  # type: ignore[arg-type]

    for kernel in (in_sqrt, in_isfinite):
        with pytest.raises(UnsupportedConstruct, match="non-numeric operand"):
            lower(kernel)


def test_a_runtime_aggregate_local_lowers_and_computes() -> None:
    # The structural spine: a tuple of runtime leaves bound to a NAMED local (previously "a runtime aggregate in a
    # local is not supported yet") flows leafwise through SSA and indexes back out.
    def kernel(x: float, y: float) -> float:
        pair = (x * 2.0, y + 1.0)
        return pair[0] + pair[1]

    model = holoso.synthesize(kernel, default_ops(FloatFormat(11, 52)), name="agg_local").numerical_model.elaborate()
    assert float(model.run(3.0, 4.0)[0]) == 11.0


def test_an_aggregate_local_joins_across_a_diamond_per_leaf() -> None:
    # A diamond whose arms bind different tuples to one local merges leafwise: the differing leaf gets a phi, the
    # equal Known leaf stays a constant, and a Known-int leaf merging with a float leaf promotes C-style.
    def kernel(c: bool, x: float) -> float:
        if c:
            pair = (x, 1.0)
        else:
            pair = (2, 1.0)  # the first leaf is a Known INTEGER on this arm: the merge promotes it
        return pair[0] * 10.0 + pair[1]

    model = holoso.synthesize(kernel, default_ops(FloatFormat(11, 52)), name="agg_diamond").numerical_model.elaborate()
    assert float(model.run(True, 3.5)[0]) == 36.0
    assert float(model.run(False, 3.5)[0]) == 21.0


def test_an_aggregate_conditional_selection_selects_per_leaf() -> None:
    # ``t if c else u`` over tuples emits one typed select per differing residual leaf (previously rejected).
    def kernel(c: bool, x: float, y: float) -> float:
        chosen = (x, y) if c else (y, x)
        return chosen[0] - chosen[1]

    model = holoso.synthesize(kernel, default_ops(FloatFormat(11, 52)), name="agg_select").numerical_model.elaborate()
    assert float(model.run(True, 7.0, 2.0)[0]) == 5.0
    assert float(model.run(False, 7.0, 2.0)[0]) == -5.0


def test_aggregate_concat_and_repeat_of_runtime_leaves_compute() -> None:
    # ``+`` and ``*`` on tuple/list values route leaves without hardware; the elements still compute.
    def kernel(x: float, y: float) -> float:
        joined = (x,) + (y,)
        tripled = [x] * 3
        return joined[1] * 100.0 + tripled[2]

    model = holoso.synthesize(kernel, default_ops(FloatFormat(11, 52)), name="agg_concat").numerical_model.elaborate()
    assert float(model.run(3.0, 4.0)[0]) == 403.0


def test_a_maybe_unbound_aggregate_read_is_a_located_rejection() -> None:
    # Boundness stays at the aggregate ROOT: a tuple bound on one arm only is maybe-unbound at the join and its
    # read rejects exactly as a scalar's would (Python would raise UnboundLocalError).
    def kernel(c: bool, x: float) -> float:
        if c:
            pair = (x, x)
        return pair[0]  # noqa: F821  # deliberately maybe-unbound: the kernel under test

    with pytest.raises(UnsupportedConstruct, match="may be unbound"):
        lower(kernel)


def test_static_string_and_record_locals_lower_because_every_use_folds() -> None:
    # Review round 2: a fully-static string or record bound to a NAMED local never reaches the datapath (every use
    # folds), so the store must not try to materialize it.
    def string_mode(x: float) -> float:
        mode = "fast"
        return x * 2.0 if mode == "fast" else x

    @dataclasses.dataclass(frozen=True)
    class Params:
        gain: float

    def record_local(x: float) -> float:
        p = Params(gain=2.0)
        return x * p.gain

    for kernel, argument, expected in ((string_mode, 3.0, 6.0), (record_local, 3.0, 6.0)):
        hir = optimize(lower(kernel))
        assert all(not isinstance(b.terminator, Branch) for b in hir.blocks)  # the static guard folded away
        model = holoso.synthesize(kernel, default_ops(FloatFormat(11, 52)), name=kernel.__name__)
        assert float(model.numerical_model.elaborate().run(argument)[0]) == expected == kernel(argument)


def test_a_negative_inexact_integer_literal_promotes_and_rounds() -> None:
    # A negative inexact counter compared with a runtime float promotes into the float datapath and rounds onto
    # -2**53 (Python compares the integer exactly -- the documented C-style deviation).
    def negative_counter_rounds(x: float) -> float:
        result = 0.0
        for i in [-9007199254740993]:  # -(2**53+1), which no float holds exactly
            if i == x:
                result = 1.0
        return result

    assert negative_counter_rounds(float(-(2**53))) == 0.0  # -(2**53+1) != -2.0**53 in Python
    hir = lower(negative_counter_rounds)
    assert float(2**53) in [abs(n.value) for n in hir.nodes.values() if isinstance(n, FloatConst)]


# ---------------------------------------- spine review round (round 5) ----------------------------------------


def test_a_flavor_degraded_join_computes_per_leaf() -> None:
    # Review round 5: a tuple-meets-list diamond degrades to a flavor-erased structural layout; the per-leaf SSA
    # cells must stay aligned across the arms' differing container flavors (crashed: "read of an undefined place").
    def flat(c: bool, x: float) -> float:
        if c:
            pair = (x, 1.0)
        else:
            pair = [2.0, x]  # type: ignore[assignment]
        return pair[0] + pair[1]

    def nested(c: bool, x: float) -> float:
        if c:
            v = ((x,), 1.0)
        else:
            v = ([2.0], x)  # type: ignore[assignment]
        return v[0][0] + v[1]

    def looped(x: float) -> float:
        acc = (x, 0.0)
        while acc[0] < 10.0:
            acc = [acc[0] * 2.0, acc[1] + 1.0]  # type: ignore[assignment]
        return acc[0] + acc[1]

    for kernel, arguments in (
        (flat, (True, 3.0)),
        (flat, (False, 3.0)),
        (nested, (True, 3.0)),
        (nested, (False, 3.0)),
        (looped, (0.5,)),
        (looped, (11.0,)),
    ):
        model = holoso.synthesize(kernel, default_ops(FloatFormat(11, 52)), name=kernel.__name__)
        assert float(model.numerical_model.elaborate().run(*arguments)[0]) == kernel(*arguments)


def test_a_residual_record_field_projection_computes() -> None:
    # Review round 5: a record with residual leaves (a select of two records) projects a field through the leaf
    # cells (crashed: PyAttr emission assumed a component ObjectRef). Nested case: an aggregate-valued field.
    @dataclasses.dataclass(frozen=True)
    class Params:
        gain: float
        offset: float

    def scalar_field(c: bool, x: float) -> float:
        p = Params(2.0, 1.0) if c else Params(3.0, -1.0)
        return x * p.gain + p.offset

    @dataclasses.dataclass(frozen=True)
    class Gains:
        pair: tuple[float, float]
        scale: float

    def aggregate_field(c: bool, x: float) -> float:
        g = Gains((1.5, 2.0), 3.0) if c else Gains((2.5, 4.0), 5.0)
        return g.pair[0] * g.scale + g.pair[1] * x

    for kernel in (scalar_field, aggregate_field):
        model = holoso.synthesize(kernel, default_ops(FloatFormat(11, 52)), name=kernel.__name__)
        elaborated = model.numerical_model.elaborate()
        for c in (True, False):
            assert float(elaborated.run(c, 3.0)[0]) == kernel(c, 3.0)


def test_a_known_condition_selection_of_aggregates_routes_leaves() -> None:
    # Review round 5: ``and``/``or`` picking an aggregate under a Known condition must route leaves, not
    # materialize the aggregate as a scalar (crashed: "an aggregate value reaches a scalar operand position").
    def kernel(x: float, y: float) -> float:
        pair = True and (x, 1.0)
        other = (y, 2.0) or (x, 0.0)
        return pair[0] + other[0]

    model = holoso.synthesize(kernel, default_ops(FloatFormat(11, 52)), name="known_cond_agg").numerical_model
    assert float(model.elaborate().run(3.0, 4.0)[0]) == 7.0


def test_aggregate_static_leaves_outside_the_datapath_fold_at_every_use() -> None:
    # Review round 5: a str/function leaf inside an aggregate must stay fact-only (its every use folds); eager
    # materialization rejected kernels whose non-datapath leaves never reach hardware.
    def tag_guard(x: float) -> float:
        pair = ("gain", x)
        return pair[1] if pair[0] == "gain" else 0.0

    def names(x: float) -> float:
        mode = ("fast", "slow")
        return x * 2.0 if mode[0] == "fast" else x

    @dataclasses.dataclass(frozen=True)
    class Named:
        label: str
        value: float

    def record_with_str(x: float) -> float:
        n = Named("boost", 2.0)
        return x * n.value

    def helper(v: float) -> float:
        return v + 1.0

    def dispatch(x: float) -> float:
        table = (helper, helper)
        return table[0](x)

    for kernel, expected in ((tag_guard, 3.0), (names, 6.0), (record_with_str, 6.0), (dispatch, 4.0)):
        model = holoso.synthesize(kernel, default_ops(FloatFormat(11, 52)), name=kernel.__name__)
        assert float(model.numerical_model.elaborate().run(3.0)[0]) == expected == kernel(3.0)


def test_component_aggregate_attributes_normalize_at_admission() -> None:
    # Review round 5: attribute-sourced aggregates must enter the fact domain normalized (Known(StaticSeq) is
    # banned), so concat and selects treat them exactly like locally-built sequences.
    class Comp:
        def __init__(self) -> None:
            self.gains = (1.0, 2.0)

        def __call__(self, c: bool, x: float) -> float:
            extended = self.gains + (3.0, 4.0)
            chosen = self.gains if c else (5.0, 6.0)
            return x * extended[3] + chosen[0]

    model = holoso.synthesize(Comp().__call__, default_ops(FloatFormat(11, 52)), name="attr_agg").numerical_model
    elaborated = model.elaborate()
    assert float(elaborated.run(True, 2.0)[0]) == 9.0
    assert float(elaborated.run(False, 2.0)[0]) == 13.0


def test_list_mutation_through_a_component_attribute_is_a_located_rejection() -> None:
    # Review round 5 MISCOMPILE: ``.append`` on an attribute-sourced list mutated a disposable reconstruction of
    # the snapshot (returned 4.0 where Python returns 5.0); ``+=`` lost its dedicated in-place message.
    class Appender:
        def __init__(self) -> None:
            self.config = [1.0]

        def __call__(self, x: float) -> float:
            self.config.append(2.0)
            return x + float(len(self.config))

    class Augmenter:
        def __init__(self) -> None:
            self.buf = [0.0]

        def __call__(self, x: float) -> float:
            self.buf += [x]
            return self.buf[0]

    with pytest.raises(UnsupportedConstruct, match="list method 'append'"):
        lower(Appender().__call__)
    with pytest.raises(UnsupportedConstruct, match="in-place list mutation"):
        lower(Augmenter().__call__)


def test_record_truth_is_python_object_truth() -> None:
    # Review round 5 MISCOMPILE: a zero-field record is truthy in Python; arity-based truth chose the else arm.
    @dataclasses.dataclass(frozen=True)
    class Marker:
        pass

    marker = Marker()

    def kernel(x: float) -> float:
        return x if marker else 0.0

    model = holoso.synthesize(kernel, default_ops(FloatFormat(11, 52)), name="record_truth").numerical_model
    assert float(model.elaborate().run(3.0)[0]) == 3.0


def test_flavor_erased_truth_with_an_array_side_stays_ambiguous() -> None:
    # Review round 5: a join carrying the ARRAY flavor cannot fold truth by arity (numpy raises on multi-element
    # truth), so it inherits the array ambiguity rejection.
    def kernel(c: bool, x: float) -> float:
        v = np.array([1.0, 2.0]) if c else (1.0, 2.0)
        return x if v else 0.0

    with pytest.raises(UnsupportedConstruct, match="truth value of an array"):
        lower(kernel)


def test_a_record_is_not_positionally_subscriptable() -> None:
    # Review round 5 MISCOMPILE: positional subscript of a record projected field 0 where Python raises TypeError.
    @dataclasses.dataclass(frozen=True)
    class Params:
        gain: float

    p = Params(2.0)

    def global_record(x: float) -> float:
        return x + p[0]  # type: ignore[index,no-any-return]

    def local_record(x: float) -> float:
        q = Params(3.0)
        return x + q[0]  # type: ignore[index,no-any-return]

    for kernel in (global_record, local_record):
        with pytest.raises(UnsupportedConstruct, match="record is not subscriptable"):
            lower(kernel)


def test_a_boolean_index_into_an_array_is_a_located_rejection() -> None:
    # Review round 5 MISCOMPILE: ``table[True]`` applied operator.index (True == 1) where numpy prepends an axis
    # (boolean advanced indexing). Python's tuple semantics genuinely accept booleans as indices, so only the
    # array flavor rejects.
    table = np.array([10.0, 20.0])

    def array_bool_index(x: float) -> float:
        return x + table[True]  # type: ignore[no-any-return]

    def tuple_bool_index(x: float) -> float:
        pair = (x, x * 2.0)
        return pair[True]

    with pytest.raises(UnsupportedConstruct, match="boolean index"):
        lower(array_bool_index)
    model = holoso.synthesize(tuple_bool_index, default_ops(FloatFormat(11, 52)), name="tuple_bool").numerical_model
    assert float(model.elaborate().run(3.0)[0]) == 6.0


def test_an_all_known_aggregate_folds_through_an_intrinsic() -> None:
    # Review round 5: an all-Known aggregate operand must fold concretely through an intrinsic exactly as a Known
    # scalar does (rejected: "a non-numeric operand reaches a numeric intrinsic").
    def kernel(x: float) -> float:
        return x + float(np.sqrt((1.0, 4.0))[1])

    model = holoso.synthesize(kernel, default_ops(FloatFormat(11, 52)), name="known_agg_fold").numerical_model
    assert float(model.elaborate().run(3.0)[0]) == 5.0


def test_a_zero_dimensional_array_folds_and_rejects_navigation() -> None:
    # Review rounds 5+6: a 0-d ndarray crashed structural navigation with a raw IndexError; scalarizing it at
    # admission was tried and reverted (it broke static type identity: isinstance(z, np.ndarray) folded False).
    # It stays an array -- concrete folds work through materialization, navigation is a located rejection.
    z = np.array(3.0)

    def scales(x: float) -> float:
        return x * float(z)

    def type_sensitive(x: float) -> float:
        return x if isinstance(z, np.ndarray) else 0.0

    def subscripted(x: float) -> float:
        return x * z[0]  # type: ignore[no-any-return]  # numpy raises IndexError: too many indices

    def sized(x: float) -> float:
        return x + float(len(z))  # numpy raises TypeError: len() of unsized object

    fmt = FloatFormat(11, 52)
    assert float(holoso.synthesize(scales, default_ops(fmt), name="p").numerical_model.elaborate().run(2.0)[0]) == 6.0
    model = holoso.synthesize(type_sensitive, default_ops(fmt), name="p").numerical_model
    assert float(model.elaborate().run(3.0)[0]) == 3.0
    with pytest.raises(UnsupportedConstruct, match="0-dimensional array cannot be indexed"):
        lower(subscripted)
    with pytest.raises(UnsupportedConstruct, match=r"len\(\) of"):  # the concrete fold surfaces numpy's own message
        lower(sized)


# ---------------------------------------- spine review round 6 ----------------------------------------


def test_record_truth_with_a_custom_bool_is_a_located_rejection() -> None:
    # Review rounds 6+7: a record class defining __bool__ folded truthy by default (round 6 MISCOMPILE); folding
    # the override concretely was then found unsound too -- it runs on a value-faithful but not TYPE-faithful
    # reconstruction (np.bool_ admits as plain bool, flipping ``a + b == 2``), so any class-dictionary
    # __bool__/__len__ entry rejects, ``__bool__ = None`` included (Python raises TypeError on its truth).
    @dataclasses.dataclass(frozen=True)
    class Disabled:
        def __bool__(self) -> bool:
            return False

    @dataclasses.dataclass(frozen=True)
    class Sized:
        def __len__(self) -> int:
            return 0

    @dataclasses.dataclass(frozen=True)
    class Unbooled:
        __bool__ = None

    for instance in (Disabled(), Sized(), Unbooled()):

        def kernel(x: float) -> float:
            return x if instance else 0.0  # noqa: B023

        with pytest.raises(UnsupportedConstruct, match="custom __bool__"):
            lower(kernel)

    @dataclasses.dataclass(frozen=True)
    class Gated:
        level: float

        def __bool__(self) -> bool:
            return self.level > 0.0

    def residual(c: bool, x: float) -> float:
        g = Gated(1.0) if c else Gated(-1.0)
        return x if g else 0.0

    with pytest.raises(UnsupportedConstruct, match="custom __bool__"):
        lower(residual)


def test_a_flavor_divergent_return_is_a_contract_rejection() -> None:
    # Review round 6: one arm returns a tuple, the other a list; strict contracts refuse the flavor-erased join
    # instead of blessing the diverging path with the declared flavor.
    def diverges(flag: bool, x: float) -> tuple[float, float]:
        if flag:
            return x, 1.0
        return [x, 2.0]  # type: ignore[return-value]

    def joined_local(flag: bool, x: float) -> list[float]:
        v = (x, 1.0) if flag else [x, 2.0]
        return v  # type: ignore[return-value]

    for kernel in (diverges, joined_local):
        with pytest.raises(UnsupportedConstruct, match="container flavor diverges across paths"):
            lower(kernel)


def test_a_returned_leaf_equal_to_a_public_live_out_rides_the_state_port_across_branches() -> None:
    # Review rounds 6+7: the return leaf and the state live-out read through DIFFERENT places, so a value merged
    # across branches arrives as two distinct-but-identical exit phis (the value must flow through a LOCAL with
    # per-arm stores -- returning self.y itself shares one phi through the Braun cache and never exercised the
    # fix). Dedup keys on the phi's structure. Round 7 extended the identity through pure conversion wrappers:
    # a store-side and a return-side IntToFloat live in different blocks, which the per-block interner keeps
    # distinct.
    class Latch:
        def __init__(self) -> None:
            self.y = 0.0

        def step(self, flag: bool, x: float) -> tuple[float]:
            if flag:
                r = x
                self.y = r
            else:
                r = -x
                self.y = r
            return (r,)

    result = holoso.synthesize(Latch().step, default_ops(FloatFormat(11, 52)), name="dedup_phi")
    assert [p.name for p in result.output_ports] == ["state_y"]
    elaborated = result.numerical_model.elaborate()
    assert float(elaborated.run(True, 3.0)[0]) == 3.0
    assert float(elaborated.run(False, 5.0)[0]) == -5.0

    class Wrapped:
        def __init__(self) -> None:
            self.y = 0.0

        def step(self, flag: bool) -> float:
            n = 1 if flag else 2
            self.y = n
            return n

    wrapped = holoso.synthesize(Wrapped().step, default_ops(FloatFormat(11, 52)), name="dedup_wrap")
    assert [p.name for p in wrapped.output_ports] == ["state_y"]
    assert float(wrapped.numerical_model.elaborate().run(True)[0]) == 1.0


def test_concretely_folded_aggregate_subscripts_need_no_cells() -> None:
    # Review round 6: a tuple key or a slice OBJECT folds concretely at analysis; emission must not enter the
    # positional projection with a non-integer key (crashed: AssertionError / TypeError at operator.index).
    def tuple_key(x: float) -> float:
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        row = a[(1,)]
        return float(row[1] + x)

    def slice_object(x: float) -> float:
        t = (1.0, 2.0, 3.0)
        s = slice(0, 2)
        u = t[s]
        return u[0] + u[1] + x  # type: ignore[index,no-any-return]  # mypy picks the int overload for t[s]

    fmt = FloatFormat(11, 52)
    for kernel, expected in ((tuple_key, 5.0), (slice_object, 4.0)):
        model = holoso.synthesize(kernel, default_ops(fmt), name=kernel.__name__).numerical_model
        assert float(model.elaborate().run(1.0)[0]) == expected == kernel(1.0)


def test_sequence_repeat_counts_follow_python() -> None:
    # Review rounds 6-8: a negative count yields the empty sequence and a plain-bool count repeats 0/1 times,
    # exactly as Python -- sound since the NpBool split keeps np.bool_ provenance, so the np spelling (which
    # numpy 2 stripped of __index__, a Python TypeError) rejects instead of miscompiling. A count beyond the
    # ssize_t index range (Python: OverflowError) rejects rather than clamping.
    def repeat_negative(x: float) -> float:
        rest = [x] * -1
        return x + float(len(rest))

    def repeat_python_bool(x: float) -> float:
        pair = (x,) * True
        return pair[0]

    fmt = FloatFormat(11, 52)
    for kernel in (repeat_negative, repeat_python_bool):
        model = holoso.synthesize(kernel, default_ops(fmt), name=kernel.__name__).numerical_model
        assert float(model.elaborate().run(3.0)[0]) == 3.0 == kernel(3.0)

    numpy_true = np.bool_(True)

    def repeat_numpy_bool(x: float) -> float:
        pair = (x,) * numpy_true  # type: ignore[operator]  # numpy 2: TypeError, np.bool_ has no __index__
        return pair[0]  # type: ignore[no-any-return]

    def repeat_beyond_index_range(x: float) -> float:
        rest = (x,) * -(1 << 100)
        return x + float(len(rest))

    for kernel in (repeat_numpy_bool, repeat_beyond_index_range):
        with pytest.raises(UnsupportedConstruct, match="arithmetic on an aggregate value"):
            lower(kernel)


def test_a_non_datapath_leaf_under_a_scalar_contract_names_the_leaf() -> None:
    # Review round 6: the str leaf hit the generic "cannot materialize" message; the leaf-kind check now runs
    # first, so the rejection names the diverging leaf and the declared kind.
    def kernel(x: float) -> tuple[float, float]:
        return ("tag", x)  # type: ignore[return-value]

    with pytest.raises(
        UnsupportedConstruct, match=r"return type mismatch at leaf \[0\]: declared float, returns a str"
    ):
        lower(kernel)


# ---------------------------------------- spine review round 7 ----------------------------------------


def test_concretely_folded_array_attributes_need_no_cells() -> None:
    # Review round 7: attribute navigation of an all-Known array (.shape, .T, a 0-d array's .real) folds at
    # analysis; emission asserted the aggregate source was a record and crashed. Same invariant as the folded
    # subscript: an all-Known destination needs no cells.
    zero_d = np.array(3.0)
    vector = np.array([1.0, 2.0, 3.0])
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]])

    def real_of_zero_d(x: float) -> float:
        return x + float(zero_d.real)

    def shape_len(x: float) -> float:
        return x + float(len(zero_d.shape))

    def shape_element(x: float) -> float:
        return x + float(vector.shape[0])

    def transposed_element(x: float) -> float:
        return x + float(matrix.T[0][1])

    fmt = FloatFormat(11, 52)
    for kernel in (real_of_zero_d, shape_len, shape_element, transposed_element):
        model = holoso.synthesize(kernel, default_ops(fmt), name=kernel.__name__).numerical_model
        assert float(model.elaborate().run(2.0)[0]) == kernel(2.0)


def test_an_index_able_aggregate_key_projects_like_a_known_integer() -> None:
    # Review round 7: a 0-d integer array is a valid Python sequence index (operator.index accepts it); the
    # analyzer projected the child but emission asserted the key fact was Known and crashed on the aggregate.
    zero_d_index = np.array(0)

    def kernel(x: float) -> float:
        t = (x, 2.0)
        return t[zero_d_index] * 4.0

    model = holoso.synthesize(kernel, default_ops(FloatFormat(11, 52)), name="idx_agg").numerical_model
    assert float(model.elaborate().run(3.0)[0]) == 12.0 == kernel(3.0)


# ---------------------------------------- spine review round 8 ----------------------------------------


def test_numpy_bool_provenance_survives_admission() -> None:
    # Review round 8: np.bool_ admitted as a plain StaticBool, so reconstructions and index/repeat semantics
    # silently took the Python-bool meaning. The NpBool variant keeps provenance exactly like NpInt/NpFloat: the
    # np spelling of a subscript index rejects (numpy 2 removed __index__) while the plain bool keeps Python's
    # bool-as-int semantics, and a concrete operator.index(np.True_) surfaces numpy's own TypeError.
    numpy_true = np.bool_(True)

    def np_key(x: float) -> float:
        return (x, x + 1.0)[numpy_true]  # type: ignore[call-overload]

    import operator

    index_of = operator.index

    def np_index_call(x: float) -> float:
        return ((x,) * index_of(numpy_true))[0]  # type: ignore[arg-type,no-any-return]

    with pytest.raises(UnsupportedConstruct, match="np.bool_ subscript index"):
        lower(np_key)
    with pytest.raises(UnsupportedConstruct, match="cannot be interpreted as an integer"):
        lower(np_index_call)

    def py_key(x: float) -> float:
        pair = (x, x * 2.0)
        return pair[True]

    model = holoso.synthesize(py_key, default_ops(FloatFormat(11, 52)), name="py_key").numerical_model
    assert float(model.elaborate().run(3.0)[0]) == 6.0 == py_key(3.0)


def test_identity_dependent_array_attributes_are_a_located_rejection() -> None:
    # Review round 8 MISCOMPILE: .base observed the admitted snapshot's storage, not the user's view (returned the
    # backing array's element). Only the value-determined navigation set folds.
    backing = np.array([1.0, 2.0, 3.0])
    view = backing[1:]

    def kernel(x: float) -> float:
        return x + float(view.base[0])  # type: ignore[index]

    with pytest.raises(UnsupportedConstruct, match="array attribute 'base' is not supported"):
        lower(kernel)


def test_record_dunders_never_run_on_reconstructions() -> None:
    # Review round 8 MISCOMPILEs: a record __index__ used as a subscript key, bool()/len() of a record with a
    # truth override, and a non-field record attribute (a property) all executed user code on the reconstruction;
    # each is a located rejection now.
    @dataclasses.dataclass(frozen=True)
    class Key:
        a: bool

        def __index__(self) -> int:
            return int(self.a)

    key = Key(True)

    def record_key(x: float) -> float:
        return (x, -x)[key]

    @dataclasses.dataclass(frozen=True)
    class Gate:
        a: bool

        def __bool__(self) -> bool:
            return self.a

    gate = Gate(True)

    def bool_call(x: float) -> float:
        return x if bool(gate) else 0.0

    @dataclasses.dataclass(frozen=True)
    class Sized:
        def __len__(self) -> int:
            return 2

    sized = Sized()

    def len_call(x: float) -> float:
        return x + float(len(sized))

    @dataclasses.dataclass(frozen=True)
    class WithProperty:
        raw: float

        @property
        def doubled(self) -> float:
            return self.raw * 2.0

    prop = WithProperty(2.0)

    def property_read(x: float) -> float:
        return x * prop.doubled

    with pytest.raises(UnsupportedConstruct, match="record subscript index"):
        lower(record_key)
    for kernel in (bool_call, len_call):
        # Rounds 9-10 generalized the callee-keyed bool()/len() guard into the argument-shaped concrete-call
        # guard, which round 10 extended to every record (the dataclass-generated __repr__ is itself not
        # reconstruction-safe: an enum field prints as its base value).
        with pytest.raises(UnsupportedConstruct, match="record cannot cross into a concrete call"):
            lower(kernel)
    with pytest.raises(UnsupportedConstruct, match="record attribute 'doubled'"):
        lower(property_read)


def test_exit_identity_is_depth_bounded_and_catches_nested_merges() -> None:
    # Review round 8: the recursive dedup identity walked the whole exit DAG -- exponential on shared operands
    # (repeated squaring) and a RecursionError on x**1024 (an explicitly permitted power chain). The walk is
    # depth-capped and cycle-guarded, and recursing through phi ARMS dedups nested merges that differ only by
    # inner phi ids.
    def power_chain(x: float) -> float:
        return x**1024

    def repeated_squaring(x: float) -> float:
        for _ in range(30):
            x = x * x
        return x

    for kernel in (power_chain, repeated_squaring):
        lower(kernel)  # the pin is completion itself (previously RecursionError / effectively non-terminating)

    class Nested:
        def __init__(self) -> None:
            self.y = 0.0

        def step(self, a: bool, b: bool, x: float) -> float:
            if a:
                if b:
                    r = x
                    self.y = r
                else:
                    r = -x
                    self.y = r
            else:
                r = x + x
                self.y = r
            return r

    result = holoso.synthesize(Nested().step, default_ops(FloatFormat(11, 52)), name="nested_dedup")
    assert [p.name for p in result.output_ports] == ["state_y"]
    elaborated = result.numerical_model.elaborate()
    assert float(elaborated.run(True, True, 3.0)[0]) == 3.0
    assert float(elaborated.run(True, False, 3.0)[0]) == -3.0
    assert float(elaborated.run(False, True, 3.0)[0]) == 6.0


# ---------------------------------------- spine review round 9 ----------------------------------------


def test_derived_numpy_booleans_keep_provenance() -> None:
    # Review round 9: comparisons, bitops, and numpy classifier folds produced plain StaticBool even with numpy
    # operands, laundering the provenance the NpBool split introduced -- the derived value then passed the
    # bool-as-int gates that a spelled np.bool_ fails. Boolean-producing folds now keep numpy provenance.
    derived_key = np.True_ == np.True_
    derived_count = np.int64(1) == np.int64(1)

    def np_compare_key(x: float) -> float:
        return (x, -x)[derived_key]

    def np_compare_count(x: float) -> float:
        return ((x,) * derived_count)[0]  # type: ignore[no-any-return]

    def np_bitop_count(x: float) -> float:
        return ((x,) * (np.True_ & np.True_))[0]  # type: ignore[operator,no-any-return]

    def np_classifier_count(x: float) -> float:
        return ((x,) * np.isfinite(1))[0]  # type: ignore[no-any-return]

    with pytest.raises(UnsupportedConstruct, match="np.bool_ subscript index"):
        lower(np_compare_key)
    for kernel in (np_compare_count, np_bitop_count, np_classifier_count):
        with pytest.raises(UnsupportedConstruct, match="arithmetic on an aggregate value"):
            lower(kernel)

    def python_compare_key(x: float) -> float:
        pair = (x, x * 2.0)
        return pair[1 == 1]

    model = holoso.synthesize(python_compare_key, default_ops(FloatFormat(11, 52)), name="py_cmp").numerical_model
    assert float(model.elaborate().run(3.0)[0]) == 6.0 == python_compare_key(3.0)


def test_records_with_user_behavior_never_cross_concrete_calls() -> None:
    # Review round 9 MISCOMPILEs: float(record)/str(record)/operator.index(record) executed user dunders on the
    # type-unfaithful reconstruction, getattr() reached attributes the direct spelling rejects, range()[record]
    # bypassed the aggregate-only key guard, and iterating a record drove __len__/__getitem__ on the rebuild
    # (a demonstrated compiler hang). All are located rejections now.
    import enum

    class Mode(enum.IntEnum):
        A = 1
        B = 2

    @dataclasses.dataclass(frozen=True)
    class Rec:
        mode: Mode
        v: float

        def __float__(self) -> float:
            return self.v * 2.0 if isinstance(self.mode, Mode) else self.v * 3.0

    rec = Rec(Mode.A, 3.0)

    def float_call(x: float) -> float:
        return x + float(rec)

    @dataclasses.dataclass(frozen=True)
    class Key:
        pick: Mode

        def __index__(self) -> int:
            return 1 if self.pick is Mode.B else 0

    key = Key(Mode.B)

    def range_key(x: float) -> float:
        return x + float(range(2)[key])

    @dataclasses.dataclass(frozen=True)
    class It:
        a: float
        b: float

        def __len__(self) -> int:
            return 2

        def __getitem__(self, i: int) -> float:
            return (self.a, self.b)[i]

    it_rec = It(1.0, 2.0)

    def iterate(x: float) -> float:
        acc = x
        for v in it_rec:  # type: ignore[attr-defined]
            acc = acc + v
        return acc

    with pytest.raises(UnsupportedConstruct, match="record cannot cross into a concrete call"):
        lower(float_call)
    with pytest.raises(UnsupportedConstruct, match="record subscript index"):
        lower(range_key)
    with pytest.raises(UnsupportedConstruct, match="iteration over a record"):
        lower(iterate)


def test_getattr_routes_through_the_attribute_guards() -> None:
    # Review round 9 MISCOMPILE: getattr(view, "base") reached the generic concrete-call path and observed the
    # admitted snapshot's storage; it now routes through the attribute transfer, so the whitelist applies.
    backing = np.array([1.0, 2.0, 3.0])
    view = backing[1:]

    def bypass(x: float) -> float:
        return x + float(getattr(view, "base")[0])

    def legitimate(x: float) -> float:
        return x + float(getattr(view, "ndim"))

    with pytest.raises(UnsupportedConstruct, match="array attribute 'base'"):
        lower(bypass)
    model = holoso.synthesize(legitimate, default_ops(FloatFormat(11, 52)), name="getattr_ok").numerical_model
    assert float(model.elaborate().run(3.0)[0]) == 4.0 == legitimate(3.0)


def test_flatten_is_no_longer_navigable() -> None:
    # Review round 9: flatten(order="K"/"A") observes the memory layout the C-contiguous snapshot discarded (a
    # demonstrated wrong value on a Fortran-ordered original), so flatten leaves the whitelist entirely.
    fortran = np.asfortranarray([[1.0, 2.0], [3.0, 4.0]])

    def kernel(x: float) -> float:
        return x + float(fortran.flatten(order="K")[1])

    with pytest.raises(UnsupportedConstruct, match="array attribute 'flatten'"):
        lower(kernel)


def test_exit_dedup_handles_deep_and_shared_exit_graphs() -> None:
    # Review round 9: the depth-4 identity cap missed dedup on deeper per-arm-store nesting; the memoized walk
    # shares subtree identities (repeated squaring stays linear) and dedups arbitrarily nested merges, while the
    # depth bound only guards genuinely chain-shaped graphs (x**1024 still lowers).
    class Deep:
        def __init__(self) -> None:
            self.y = 0.0

        def step(self, a: bool, b: bool, c: bool, x: float) -> float:
            r = x
            if a:
                r = -r
            self.y = r
            if b:
                r = -r
                self.y = r
            if c:
                r = -r
                self.y = r
            return r

    result = holoso.synthesize(Deep().step, default_ops(FloatFormat(11, 52)), name="deep_dedup")
    assert [p.name for p in result.output_ports] == ["state_y"]
    elaborated = result.numerical_model.elaborate()
    for a in (True, False):
        for b in (True, False):
            for c in (True, False):
                assert float(elaborated.run(a, b, c, 3.0)[0]) == Deep().step(a, b, c, 3.0)


# ---------------------------------------- spine review round 10 ----------------------------------------


def test_getattr_is_attribute_access_in_every_admitted_shape() -> None:
    # Review round 10: getattr rewrites into the attribute op, so state reads, record fields with residual
    # leaves, and the array whitelist all behave exactly as the dotted spelling (the round-9 routing crashed on
    # residual results and the 3-argument spelling bypassed interception entirely -- a state read folded to its
    # reset snapshot, [1.0, 1.0] where Python steps [3.0, 5.0]).
    class Accumulator:
        def __init__(self) -> None:
            self.g = 1.0

        def step(self, x: float) -> float:
            self.g = self.g + x
            return getattr(self, "g")  # type: ignore[no-any-return]

    result = holoso.synthesize(Accumulator().step, default_ops(FloatFormat(11, 52)), name="getattr_state")
    elaborated = result.numerical_model.elaborate()
    assert [float(elaborated.run(2.0)[0]), float(elaborated.run(2.0)[0])] == [3.0, 5.0]

    @dataclasses.dataclass(frozen=True)
    class Plain:
        v: float

    def residual_field(c: bool, x: float) -> float:
        p = Plain(2.0) if c else Plain(3.0)
        return x * getattr(p, "v")  # type: ignore[no-any-return]

    model = holoso.synthesize(residual_field, default_ops(FloatFormat(11, 52)), name="getattr_field").numerical_model
    elaborated = model.elaborate()
    assert float(elaborated.run(True, 3.0)[0]) == 6.0
    assert float(elaborated.run(False, 3.0)[0]) == 9.0

    class WithDefault:
        def __init__(self) -> None:
            self.g = 1.0

        def step(self, x: float) -> float:
            self.g = self.g + x
            return getattr(self, "g", 0.0)

    with pytest.raises(UnsupportedConstruct, match="static attribute name and no default"):
        lower(WithDefault().step)


def test_attrgetter_objects_are_a_located_rejection() -> None:
    # Review round 10 MISCOMPILE: operator.attrgetter("strides") reached the snapshot's internals through an
    # opaque callable the attribute guards cannot see into (folded the C-contiguous snapshot's strides).
    import operator

    get_strides = operator.attrgetter("strides")
    fortran = np.asfortranarray([[1.0, 2.0], [3.0, 4.0]])

    def kernel(x: float) -> float:
        return x + float(get_strides(fortran)[0])

    with pytest.raises(UnsupportedConstruct, match="is not supported in a kernel"):
        lower(kernel)


def test_a_record_nested_in_an_argument_cannot_cross_a_concrete_call() -> None:
    # Review round 10 MISCOMPILE: the record guard checked top-level argument facts only, so str((record,))
    # folded the dataclass-generated __repr__ on the reconstruction -- where an enum field prints as its base
    # value ("R1(mode=1)" instead of "R1(mode=<Mode.A: 1>)").
    import enum

    class Mode(enum.IntEnum):
        A = 1

    @dataclasses.dataclass(frozen=True)
    class R1:
        mode: Mode

    r1 = R1(Mode.A)

    def nested_in_whitelisted_call(x: float) -> float:
        return x + float(sum((r1,)))  # type: ignore[arg-type]

    def non_whitelisted_callee(x: float) -> float:
        return x if str((r1,)) == "(R1(mode=1),)" else -x

    with pytest.raises(UnsupportedConstruct, match="record cannot cross into a concrete call"):
        lower(nested_in_whitelisted_call)
    with pytest.raises(UnsupportedConstruct, match="is not supported in a kernel"):
        lower(non_whitelisted_callee)


def test_snapshot_observing_spellings_are_located_rejections() -> None:
    # Review round 10: the unbound-method spelling (np.ndarray.flatten(a, order="K")) and unregistered numpy
    # callables (np.ravel) reached the C-contiguous snapshot through the generic concrete path, observing a
    # memory order the admission discarded; np.array construction stays vetted and folds.
    fortran = np.asfortranarray([[1.0, 2.0], [3.0, 4.0]])

    def unbound_flatten(x: float) -> float:
        return x + float(np.ndarray.flatten(fortran, order="K")[1])

    def unregistered_ravel(x: float) -> float:
        return x + float(np.ravel(fortran, order="K")[1])

    for kernel in (unbound_flatten, unregistered_ravel):
        with pytest.raises(UnsupportedConstruct, match="is not supported in a kernel"):
            lower(kernel)

    def vetted_array(x: float) -> float:
        m = np.array([[1.0, 2.0], [3.0, 4.0]])
        return x + float(m[(1,)][1])

    model = holoso.synthesize(vetted_array, default_ops(FloatFormat(11, 52)), name="np_array").numerical_model
    assert float(model.elaborate().run(1.0)[0]) == 5.0 == vetted_array(1.0)


def test_a_record_nested_in_a_subscript_key_is_a_located_rejection() -> None:
    # Review round 10 MISCOMPILE: the record-key guard was root-only, so a tuple key containing an __index__
    # record reached numpy's fancy indexing and ran the dunder on the rebuild (enum field as its base value).
    import enum

    class Mode(enum.IntEnum):
        A = 1

    @dataclasses.dataclass(frozen=True)
    class Key:
        mode: Mode

        def __index__(self) -> int:
            return 1 if isinstance(self.mode, Mode) else 0

    table = np.array([10.0, 20.0])
    key_tuple = (Key(Mode.A),)

    def kernel(x: float) -> float:
        return x + float(table[key_tuple])

    with pytest.raises(UnsupportedConstruct, match="record subscript index"):
        lower(kernel)


def test_isinstance_against_an_enum_type_is_a_located_rejection() -> None:
    # Review round 10 MISCOMPILE: enum members normalize to their base value at admission (the sanctioned
    # erasure), so isinstance(member, EnumType) folded False where Python sees the member -- reachable directly
    # or through any inlined method reading an enum-typed field. Non-enum class queries stay decidable.
    import enum

    class Mode(enum.IntEnum):
        A = 1

    mode = Mode.A

    def direct(x: float) -> float:
        return x if isinstance(mode, Mode) else -x

    def tuple_classinfo(x: float) -> float:
        return x if isinstance(mode, (float, Mode)) else -x

    for kernel in (direct, tuple_classinfo):
        with pytest.raises(UnsupportedConstruct, match="not decidable|enum-free class"):
            lower(kernel)

    def plain_class(x: float) -> float:
        return x * 2.0 if isinstance(1.0, float) else x

    model = holoso.synthesize(plain_class, default_ops(FloatFormat(11, 52)), name="plain_isinstance").numerical_model
    assert float(model.elaborate().run(3.0)[0]) == 6.0 == plain_class(3.0)


# ---------------------------------------- spine review round 11 ----------------------------------------


def test_concrete_evaluation_is_a_closed_whitelist() -> None:
    # Review round 11: every blacklist guard kept resurfacing under new spellings (functools.partial(getattr),
    # vars(self), bound tuple dunders, partial(np.ravel), repr/type/issubclass over erased enums). Concrete
    # evaluation now admits only vetted value-determined callables; everything else is a located rejection.
    import functools

    wrapped_getattr = functools.partial(getattr)

    class Acc:
        def __init__(self) -> None:
            self.g = 1.0

        def step(self, x: float) -> float:
            self.g = self.g + x
            return wrapped_getattr(self, "g")  # type: ignore[no-any-return]

    class Vars:
        def __init__(self) -> None:
            self.g = 1.0

        def step(self, x: float) -> float:
            self.g = self.g + x
            return vars(self)["g"]  # type: ignore[no-any-return]

    fortran = np.asfortranarray([[1.0, 2.0], [3.0, 4.0]])
    ravel_k = functools.partial(np.ravel, order="K")

    def partial_ravel(x: float) -> float:
        return x + float(ravel_k(fortran)[1])

    import enum

    class Mode(enum.IntEnum):
        A = 1

    mode = Mode.A
    expected = repr(Mode.A)

    def repr_of_enum(x: float) -> float:
        return x if repr(mode) == expected else -x

    def issubclass_of_type(x: float) -> float:
        return x if issubclass(type(mode), Mode) else -x

    for kernel in (Acc().step, Vars().step, partial_ravel, repr_of_enum, issubclass_of_type):
        with pytest.raises(UnsupportedConstruct, match="is not supported in a kernel"):
            lower(kernel)


def test_object_references_never_cross_concrete_calls() -> None:
    # Review round 11 MISCOMPILE: a stateful component's dunder ran on the live reset-time object while the
    # kernel's writes existed only as state facts -- float(self) stepped [1.0, 1.0] where Python steps
    # [3.0, 5.0]; len(self) reached the same hazard through the unpack transfer.
    class FloatSelf:
        def __init__(self) -> None:
            self.g = 1.0

        def __float__(self) -> float:
            return self.g

        def step(self, x: float) -> float:
            self.g = self.g + x
            return float(self)

    class LenSelf:
        def __init__(self) -> None:
            self.g = 1.0

        def __len__(self) -> int:
            return int(self.g)

        def step(self, x: float) -> float:
            self.g = self.g + x
            return x + float(len(self))

    for kernel in (FloatSelf().step, LenSelf().step):
        with pytest.raises(UnsupportedConstruct, match="object reference cannot cross|len\\(\\) of an object"):
            lower(kernel)


def test_isinstance_classinfo_must_resolve_completely() -> None:
    # Review round 11: a precomputed tuple or union classinfo was one opaque ObjectRef, slipping enum members
    # past the round-10 guard and folding on the erased value; classinfo must now resolve member by member.
    import enum

    class Mode(enum.IntEnum):
        A = 1

    mode = Mode.A
    tuple_classinfo = (float, Mode)
    union_classinfo = str | Mode

    def opaque_tuple(x: float) -> float:
        return x if isinstance(mode, tuple_classinfo) else -x

    def opaque_union(x: float) -> float:
        return x if isinstance(mode, union_classinfo) else -x

    for kernel in (opaque_tuple, opaque_union):
        with pytest.raises(UnsupportedConstruct, match="not decidable|enum-free class"):
            lower(kernel)

    plain_union = float | int

    def resolvable_union(x: float) -> float:
        return x * 2.0 if isinstance(1.0, plain_union) else x

    model = holoso.synthesize(resolvable_union, default_ops(FloatFormat(11, 52)), name="iu").numerical_model
    assert float(model.elaborate().run(3.0)[0]) == 6.0 == resolvable_union(3.0)


def test_bound_dunders_of_values_are_a_located_rejection() -> None:
    # Review round 11 MISCOMPILE: T.__repr__() bound off a record-carrying tuple ran the generated __repr__ on
    # the reconstruction (an enum field prints as its base value); dunder binding and record-carrying receivers
    # both reject, while plain value methods keep folding.
    import enum

    class Mode(enum.IntEnum):
        A = 1

    @dataclasses.dataclass(frozen=True)
    class R:
        mode: Mode

    record_tuple = (R(Mode.A),)
    expected = repr(record_tuple)

    def bound_repr(x: float) -> float:
        return x if record_tuple.__repr__() == expected else -x

    def plain_dunder(x: float) -> float:
        pair = (2.0, 1.0)
        return pair.__len__() * x

    with pytest.raises(UnsupportedConstruct, match="record-carrying sequence"):
        lower(bound_repr)
    with pytest.raises(UnsupportedConstruct, match="dunder attribute"):
        lower(plain_dunder)


# ---------------------------------------- spine review round 12 ----------------------------------------


def test_whitelist_members_are_value_determined() -> None:
    # Review round 12: several whitelist members were not value-determined for some argument shape -- a dataclass
    # __post_init__ observing erased enum provenance, str.format's !r conversion, tuple.count's identity shortcut
    # (a NaN element matches itself in Python, never after a rebuild), and a PRE-BOUND builtin whose live mutable
    # receiver was emptied at compile time. Construction now requires the generated __init__ with no
    # __post_init__; sequence methods and str.format reject; only bind-site-minted value methods are admitted.
    import enum

    class Mode(enum.IntEnum):
        A = 1

    @dataclasses.dataclass
    class Gain:
        mode: Mode
        scale: float = 0.0

        def __post_init__(self) -> None:
            self.scale = 10.0 if isinstance(self.mode, Mode) else 20.0

    mode = Mode.A

    def post_init_construction(x: float) -> float:
        return x * Gain(mode).scale

    expected = repr(Mode.A)

    def str_format(x: float) -> float:
        return x if "{!r}".format(mode) == expected else -x

    nan = np.float64(np.nan)
    nan_tuple = (nan, 1.0)

    def tuple_count(x: float) -> float:
        return x if nan_tuple.count(nan) == 1 else -x

    live = [9.0, 7.0]
    captured_pop = live.pop

    def prebound_pop(x: float) -> float:
        return x + captured_pop()

    for kernel, match in (
        (post_init_construction, "is not supported in a kernel"),
        (str_format, "str.format is not supported"),
        (tuple_count, "sequence method 'count'"),
        (prebound_pop, "is not supported in a kernel"),
    ):
        with pytest.raises(UnsupportedConstruct, match=match):
            lower(kernel)
    assert live == [9.0, 7.0]  # compilation must never mutate the user's live objects

    @dataclasses.dataclass(frozen=True)
    class Plain:
        v: float

    def generated_init_construction(x: float) -> float:
        p = Plain(2.0)
        return x * p.v

    model = holoso.synthesize(generated_init_construction, default_ops(FloatFormat(11, 52)), name="p").numerical_model
    assert float(model.elaborate().run(3.0)[0]) == 6.0


def test_isinstance_subjects_fold_through_retained_members_and_reject_on_lost_provenance() -> None:
    # Review round 12 established that a mixin base and an ABC register() distinguish a live member from its
    # erased base value. Admission now RETAINS the member as the scalar's source, so an isinstance subject
    # reconstructs faithfully: the mixin-base query folds to exactly Python's verdict. The ABC classinfo still
    # rejects (its instance check is not type's own), and a subject whose source was dropped by a join (the
    # runtime value may be a member the fact no longer names) still rejects as undecidable.
    import abc
    import enum

    class Marker:
        pass

    class Mode(Marker, enum.IntEnum):
        A = 1
        B = 2

    class AbcMarker(abc.ABC):
        pass

    AbcMarker.register(Mode)
    mixed = Mode.A

    def mixin(x: float) -> float:
        return x if isinstance(mixed, Marker) else -x

    model = holoso.synthesize(mixin, default_ops(FloatFormat(11, 52)), name="mixin").numerical_model
    assert float(model.elaborate().run(3.0)[0]) == mixin(3.0) == 3.0

    def value_query(x: float) -> float:
        return x if isinstance(mixed, int) and not isinstance(mixed, str) else -x

    model = holoso.synthesize(value_query, default_ops(FloatFormat(11, 52)), name="vq").numerical_model
    assert float(model.elaborate().run(3.0)[0]) == value_query(3.0) == 3.0

    def registered(x: float) -> float:
        return x if isinstance(mixed, AbcMarker) else -x

    with pytest.raises(UnsupportedConstruct, match="enum-free class"):
        lower(registered)

    def lost_source(x: float, pick: bool) -> float:
        y = Mode.A if pick else 1
        return x if isinstance(y, Marker) else -x

    with pytest.raises(UnsupportedConstruct, match="not decidable"):
        lower(lost_source)


def test_object_references_reject_at_any_nesting_depth() -> None:
    # Review round 12 MISCOMPILE: sum((self,), 0.0) handed the callable the live reset-time component through the
    # rebuilt tuple (stepping [1.0, 1.0] where Python steps [3.0, 5.0]); object-reference leaves now reject
    # inside aggregate arguments, and an oversized static range rejects instead of burning unbounded compile time.
    class RAdd:
        def __init__(self) -> None:
            self.g = 1.0

        def __radd__(self, other: float) -> float:
            return other + self.g

        def step(self, x: float) -> float:
            self.g = self.g + x
            return sum((self,), 0.0)  # type: ignore[type-var,return-value]

    def oversized_range(x: float) -> float:
        return x + float(sum(range(10**12)))

    with pytest.raises(UnsupportedConstruct, match="object reference cannot cross"):
        lower(RAdd().step)
    with pytest.raises(UnsupportedConstruct, match="oversized range"):
        lower(oversized_range)


# ---------------------------------------- spine review round 13 ----------------------------------------


def test_dataclass_construction_admits_only_the_generated_machinery() -> None:
    # Review round 13: a user __new__ and a field default_factory are the remaining compile-time code hooks in
    # construction -- a factory popped the user's live list once per analysis round and baked the replay's draw
    # into the emitted model. Construction admits only generated-__init__ classes with no hooks or factories.
    import enum
    import itertools

    class Mode(enum.IntEnum):
        A = 1

    @dataclasses.dataclass
    class NewBox:
        mode: Mode
        scale: float = dataclasses.field(init=False)

        def __new__(cls, mode: Mode) -> "NewBox":
            obj = super().__new__(cls)
            obj.scale = 10.0 if isinstance(mode, Mode) else 20.0
            return obj

    mode = Mode.A

    def dunder_new(x: float) -> float:
        return x * NewBox(mode).scale

    live = [9.0, 7.0, 5.0]

    @dataclasses.dataclass
    class FactoryBox:
        v: float
        junk: float = dataclasses.field(default_factory=live.pop)

    def factory(x: float) -> float:
        return x * FactoryBox(2.0).v

    counter = itertools.count(10)

    @dataclasses.dataclass
    class CounterBox:
        v: float = dataclasses.field(default_factory=lambda: float(next(counter)))

    def drawing_factory(x: float) -> float:
        return x + CounterBox().v

    for kernel in (dunder_new, factory, drawing_factory):
        with pytest.raises(UnsupportedConstruct, match="is not supported in a kernel"):
            lower(kernel)
    assert live == [9.0, 7.0, 5.0]  # compilation must never mutate the user's live objects


def test_inline_classinfo_tuples_fold() -> None:
    # Review round 13 REGRESSION (introduced in round 12): the aggregate ObjectRef-leaf guard fired on the inline
    # isinstance classinfo tuple before the classinfo exemption could apply, so the inline and precomputed
    # spellings of the same valid Python diverged.
    gain = 2.0

    def kernel(x: float) -> float:
        return x * 2.0 if isinstance(gain, (float, str)) else x

    model = holoso.synthesize(kernel, default_ops(FloatFormat(11, 52)), name="inline_ci").numerical_model
    assert float(model.elaborate().run(3.0)[0]) == 6.0 == kernel(3.0)


def test_oversized_ranges_reject_in_every_position() -> None:
    # Review round 13: the round-12 guard covered only top-level argument facts; a range RECEIVER
    # (range(10**12).count(0.5) iterates linearly for non-int arguments) and a range nested in an aggregate
    # argument both burned unbounded compile time.
    def receiver(x: float) -> float:
        return x + float(range(10**12).count(0.5))  # type: ignore[arg-type]

    def nested(x: float) -> float:
        return x + float(np.array([range(10**9)])[0][0])

    with pytest.raises(UnsupportedConstruct, match="oversized range"):
        lower(receiver)
    with pytest.raises(UnsupportedConstruct, match="oversized range"):
        lower(nested)


# ------------------------------ architecture migration phase 0 ------------------------------


def test_live_object_protocols_never_run_at_compile_time() -> None:
    # Architecture-review probes (honest code): a component's ordinary __getitem__/__iter__ reached host
    # protocols through the subscript and iteration transfers, reading reset-time state where the kernel's
    # writes exist only as state facts; an inherited __setattr__ ran during generated dataclass construction on
    # an erasure-reconstructed argument; an oversized integer argument to a minted value method allocated
    # gigabytes at compile time. All are located rejections now.
    import enum

    class GetItem:
        def __init__(self) -> None:
            self.g = 1.0

        def __getitem__(self, i: int) -> float:
            return self.g

        def step(self, x: float) -> float:
            self.g = self.g + x
            return self[0]

    class Iter:
        def __init__(self) -> None:
            self.g = 1.0

        def __iter__(self):  # type: ignore[no-untyped-def]
            return iter((self.g,))

        def step(self, x: float) -> float:
            self.g = self.g + x
            acc = 0.0
            for v in self:
                acc = acc + v
            return acc

    class Mode(enum.IntEnum):
        A = 1

    class SetattrBase:
        def __setattr__(self, name: str, value: object) -> None:
            scaled = float(int(value)) * 20.0 if isinstance(value, Mode) else value
            object.__setattr__(self, name, scaled)

    @dataclasses.dataclass
    class Hooked(SetattrBase):
        mode: object

    mode = Mode.A

    def inherited_setattr(x: float) -> float:
        return x * float(Hooked(mode).mode)  # type: ignore[arg-type]

    def oversized_method_argument(x: float) -> float:
        return x + float(len("x".ljust(10**12)))

    for kernel, match in (
        (GetItem().step, "subscript of an object"),
        (Iter().step, "iteration over an object"),
        (inherited_setattr, "is not supported in a kernel"),
        (oversized_method_argument, "oversized integer argument"),
    ):
        with pytest.raises(UnsupportedConstruct, match=match):
            lower(kernel)


# ------------------------------ architecture migration phase 1 ------------------------------


def test_concrete_evaluation_has_one_admission_door() -> None:
    # Phase-1 invariant: the generic concrete-evaluation site in the analyzer sits behind the fold admission
    # harness, and the vetted set lives only in _fold. A second admission door, or a vetted set re-grown inside
    # the analyzer, is a regression to distributed guard prose.
    import holoso._frontend._fir._analyze as analyze_module
    import holoso._frontend._fir._fold as fold_module

    analyze_source = Path(analyze_module.__file__).read_text()
    fold_source = Path(fold_module.__file__).read_text()
    assert analyze_source.count("admit_call(") == 1
    assert analyze_source.count("concrete = target(") == 1
    assert analyze_source.index("admit_call(") < analyze_source.index("concrete = target(")
    assert "_vetted_concrete_target" not in analyze_source
    assert "_vetted_concrete_target" in fold_source


def test_namespace_attribute_reads_are_snapshot_once() -> None:
    # A module-level __getattr__ (PEP 562: lazy imports, deprecation shims) is honest code that executes per
    # getattr. The analyzer's fixpoint visits a PyAttr transfer many times; without the first-read snapshot each
    # visit would re-run the hook, observing drift (a fresh object per call breaks reference-identity joins, a
    # counting hook shows the re-execution directly).
    module = types.ModuleType("lazy_ns")
    calls = {"n": 0}

    def module_getattr(name: str) -> float:
        if name != "gain":
            raise AttributeError(name)
        calls["n"] += 1
        return 2.5

    module.__getattr__ = module_getattr  # type: ignore[method-assign]

    def kernel(x: float) -> float:
        return module.gain * x  # type: ignore[no-any-return]

    kernel.__globals__["module"] = module
    try:
        unit = lower(kernel)
    finally:
        kernel.__globals__.pop("module", None)
    assert calls["n"] == 1, f"the live namespace was read {calls['n']} times; the snapshot admits exactly one read"
    del unit


def test_value_methods_on_enum_members_bind_base_type_only() -> None:
    # A retained StrEnum member reconstructs faithfully at identity-sensitive queries, but its VALUE METHODS
    # must come from the base type: an honest custom method on the enum class is still arbitrary user code that
    # must never run at compile time, so the receiver is stripped to its base value before binding.
    import enum

    class Tag(enum.StrEnum):
        A = "ab"

        def describe(self) -> str:
            raise RuntimeError("user code ran at compile time")

    tag = Tag.A

    def base_method(x: float) -> float:
        return x * float(len(tag.upper()))

    model = holoso.synthesize(base_method, default_ops(FloatFormat(11, 52)), name="bm").numerical_model
    assert float(model.elaborate().run(3.0)[0]) == base_method(3.0) == 6.0

    def custom_method(x: float) -> float:
        return x * float(len(tag.describe()))

    with pytest.raises(UnsupportedConstruct, match="enum member attribute 'describe' is not supported"):
        lower(custom_method)


# ------------------------------ retention review round (Claude ultrathink) ------------------------------


def test_cross_enum_value_equal_members_degrade_to_lost_not_one_arms_provenance() -> None:
    # REGRESSION (miscompile): IntEnum/StrEnum members compare equal ACROSS enums by base value, so a
    # source-including dataclass __eq__ let a cross-enum join keep the FIRST arm's member and isinstance folded
    # a constant that was wrong on the other path (compiled 9.0 where Python gives 6.0). Source equality is
    # identity-keyed now; the join degrades to LOST and the query refuses, soundly, on both paths.
    import enum

    from holoso._frontend._fir._value import MetaInt, same

    class Marker:
        pass

    class Mode(Marker, enum.IntEnum):
        A = 1

    class Other(enum.IntEnum):
        X = 1

    assert not same(MetaInt(1, source=Mode.A), MetaInt(1, source=Other.X))

    def int_kernel(x: float, p: bool) -> float:
        m = Mode.A if p else Other.X
        return x * 2.0 if isinstance(m, Marker) else x * 3.0

    with pytest.raises(UnsupportedConstruct, match="not decidable"):
        lower(int_kernel)

    class TagMarker:
        pass

    class Tag(TagMarker, enum.StrEnum):
        A = "ab"

    class OtherTag(enum.StrEnum):
        Z = "ab"

    def str_kernel(x: float, p: bool) -> float:
        t = Tag.A if p else OtherTag.Z
        return x * 2.0 if isinstance(t, TagMarker) else x * 3.0

    with pytest.raises(UnsupportedConstruct, match="not decidable"):
        lower(str_kernel)


def test_object_subscript_keys_reject_instead_of_running_live_index() -> None:
    # MISCOMPILE: a referenced key resolved through the LIVE object's __index__ at compile time (per analysis
    # visit, in the replay, and again at emission), reading reset-time state the kernel's writes never touch:
    # t[self] selected index 0 (reset g=0.0) where Python selects 1 (g=1.0). Both the aggregate-subject and the
    # concrete-subject key paths refuse now; slice objects keep working as proper values.
    class SelfIndexed:
        def __init__(self) -> None:
            self.g = 0.0

        def __index__(self) -> int:
            return int(self.g)

        def step(self, x: float) -> float:
            self.g = 1.0
            t = (x, -x)
            return t[self]

    with pytest.raises(UnsupportedConstruct, match="an object subscript index is not supported"):
        lower(SelfIndexed().step)

    selector = SelfIndexed()

    def concrete_subject(x: float) -> float:
        return x * float(range(5)[selector])

    with pytest.raises(UnsupportedConstruct, match="an object subscript index is not supported"):
        lower(concrete_subject)


def test_oversized_range_unpacking_is_a_located_rejection() -> None:
    # len() of range(10**30) raises OverflowError (past ssize_t), which escaped raw through the builder's
    # unpacking arity check; it is a located rejection now, like the unroll path's.
    def kernel(x: float) -> float:
        a, b = range(10**30)
        return x * float(a + b)

    with pytest.raises(UnsupportedConstruct, match="oversized range"):
        lower(kernel)


def test_reference_returns_reject_with_named_contracts_not_assertions() -> None:
    # A reference leaf inside a returned aggregate crashed emission with a raw AssertionError ("read of an
    # undefined place"); a root object return under a value contract claimed "returns nothing". Both are named
    # contract mismatches now.
    def ref_leaf(x: float) -> tuple[float, float]:
        return (x, math)  # type: ignore[return-value]

    with pytest.raises(UnsupportedConstruct, match="returns an object"):
        lower(ref_leaf)

    def ref_root(x: float) -> float:
        return math  # type: ignore[return-value]

    with pytest.raises(UnsupportedConstruct, match="returns an object"):
        lower(ref_root)


def test_conditional_none_names_itself_at_the_join() -> None:
    # The X | None early-return contract unwraps the annotation, but a LIVE None path is not lowerable; the
    # join used to reject with the generic "irreconcilable kinds", naming neither None nor the return.
    def kernel(x: float) -> float | None:
        if x > 0.0:
            return None
        return x

    with pytest.raises(UnsupportedConstruct, match="None merges with a value"):
        lower(kernel)
