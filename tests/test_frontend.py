"""Unit tests for the Python-to-HIR frontend."""

import dataclasses
import math
import sys
import textwrap
import types
from pathlib import Path

import numpy as np
import pytest

from holoso import MissingIntrinsic, UnsupportedConstruct
from holoso._frontend import lower
from holoso._frontend._lower import _port_name
from holoso._hir import (
    BoolAnd,
    BoolConst,
    BoolNot,
    BoolOr,
    BoolType,
    Branch,
    FloatAbs,
    FloatAdd,
    FloatConst,
    FloatDiv,
    FloatMul,
    BoolToFloat,
    FloatNeg,
    FloatRelational,
    FloatToBool,
    Jump,
    Operation,
    Phi,
    Ret,
    StateRead,
    optimize,
)

from ._modelref import flatten_value, output_names


def _arith_count(hir, op_type):  # type: ignore[no-untyped-def]
    return sum(1 for n in hir.nodes.values() if isinstance(n, Operation) and type(n.operator) is op_type)


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
    assert _port_name([0]) == "out_0"
    assert _port_name([0, "foo", "bar"]) == "out_0_foo_bar"
    assert _port_name([3, 1]) == "out_3_1"


def test_flatten_value_returns_leaves() -> None:
    leaves = flatten_value([[1.5], [2.5]])
    assert [value for _, value in leaves] == [1.5, 2.5]


def test_small_kernel_inputs_outputs_and_ops() -> None:
    def kernel(a, b):  # type: ignore[no-untyped-def]
        return (a - b) * 0.25 + a * b

    hir = lower(kernel)
    assert hir.input_names() == ["a", "b"]
    assert [o.name for o in hir.outputs] == ["out_0"]
    assert _arith_count(hir, FloatMul) == 2  # (a-b)*0.25 and a*b
    assert _arith_count(hir, FloatAdd) == 2  # subtraction (add+neg) and the final add
    assert _arith_count(hir, FloatNeg) == 1  # the negation introduced by subtraction


def test_pow_expands_to_multiply_chain() -> None:
    def cube(a):  # type: ignore[no-untyped-def]
        return a**3

    hir = lower(cube)
    assert _arith_count(hir, FloatMul) == 2  # a*a*a


def test_abs_lowers_to_semantic_operation() -> None:
    def f(a):  # type: ignore[no-untyped-def]
        return abs(a)

    hir = lower(f)
    abs_ops = [n for n in hir.nodes.values() if isinstance(n, Operation) and type(n.operator) is FloatAbs]
    assert len(abs_ops) == 1


def test_division_lowers_to_div() -> None:
    def f(a, b):  # type: ignore[no-untyped-def]
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
    def f(a):  # type: ignore[no-untyped-def]
        x = a
        for _ in range(3):
            x = x + a
        return x

    hir = lower(f)
    assert _arith_count(hir, FloatAdd) == 3  # one add per unrolled trip


def test_for_loop_counter_is_a_compile_time_constant() -> None:
    # The counter indexes a constant table and sets a power-of-two shift exponent; both fold per unrolled trip.
    def f(a):  # type: ignore[no-untyped-def]
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

        def __call__(self, x):  # type: ignore[no-untyped-def]
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

        def __call__(self, x):  # type: ignore[no-untyped-def]
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

        def __call__(self, x):  # type: ignore[no-untyped-def]
            if self.flag:
                y = x
            else:
                for _ in range(1000):  # dead (flag read-only True): not unrolled, not rejected
                    y = x
            return y

    hir = lower(DeadOverThreshold().__call__)
    assert len(hir.blocks) == 1  # the dead else arm and its over-threshold loop are folded away


def test_zero_trip_for_write_does_not_mark_attribute_assigned() -> None:
    # Regression (user): a write inside `for _ in range(0)` never executes, so the read-only scan must not count it as
    # an assignment -- otherwise a later guard on the attribute becomes a runtime branch and a return in the (actually
    # dead) arm is wrongly rejected. The scan mirrors the static trip count, as lowering and the state scan do.
    class ZeroForFlag:
        def __init__(self) -> None:
            self._flag = False
            self.y = 0.0

        def __call__(self, x):  # type: ignore[no-untyped-def]
            for _ in range(0):
                self._flag = True  # noqa -- zero-trip loop: never runs
            if self._flag:
                return x
            self.y = x
            return self.y

    hir = lower(ZeroForFlag().__call__)
    assert [slot.name for slot in hir.state_slots] == ["y"]  # _flag stays a read-only constant; only y is state


def test_nested_function_scope_does_not_shadow_global() -> None:
    # Regression (Codex): a name bound in a nested (here dead) def is a separate scope; it must not make the OUTER
    # function treat that name as local, which would shadow the numpy alias (or a builtin) at an earlier use.
    def kernel(x):  # type: ignore[no-untyped-def]
        y = np.asarray([x])
        return y[0]

        def helper():  # type: ignore[no-untyped-def]  # noqa -- dead nested scope; its ``np`` is not the outer's
            np = 1
            return np

    lower(kernel)  # must not raise: the nested ``np`` does not shadow the module-level numpy alias here


def test_globally_shadowed_range_is_rejected(tmp_path: Path) -> None:
    # Regression (Codex): Python resolves a module global before the builtin, so a shadowed `range` is not the
    # unrollable builtin and must be rejected, not silently unrolled. The frontend needs real source, so the kernel
    # lives in a temp module that shadows `range` at module scope.
    import importlib.util

    source = textwrap.dedent("""
        range = lambda n: [0, 0, 0]

        def kernel(a):
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
    with pytest.raises(UnsupportedConstruct, match="range"):
        lower(module.kernel)


def test_constant_boolean_attribute_branch_folds() -> None:
    # Regression (Codex): a branch on a read-only boolean attribute has a compile-time-known condition; only the taken
    # arm lowers, so a write in the dead arm does not become spurious persistent state.
    class Disabled:
        def __init__(self) -> None:
            self.flag = False
            self.y = 0.0

        def __call__(self, x):  # type: ignore[no-untyped-def]
            if self.flag:
                self.y = x
            return self.y

    hir = lower(Disabled().__call__)
    assert [slot.name for slot in hir.state_slots] == []  # folded: y never written, no state, no branch
    assert len(hir.blocks) == 1


def test_numpy_boolean_attribute_branch_folds() -> None:
    # Regression (Codex): a read-only np.bool_ attribute must fold like a Python bool (it is exposed as boolean state
    # elsewhere), so the disabled arm's write does not become spurious state.
    class NpDisabled:
        def __init__(self) -> None:
            self.flag = np.bool_(False)
            self.y = 0.0

        def __call__(self, x):  # type: ignore[no-untyped-def]
            if self.flag:
                self.y = x
            return self.y

    hir = lower(NpDisabled().__call__)
    assert [slot.name for slot in hir.state_slots] == []
    assert len(hir.blocks) == 1


def test_static_integer_comparison_branch_folds() -> None:
    # Regression (Codex): a comparison of static integers (an unrolled loop counter against a bound) is known at
    # compile time and folds to one arm; a write gated by a statically-false guard must not become spurious state, and
    # no dynamic branch is emitted (integers are exact in any ZKF format, so the fold matches the comparator).
    class GuardAlwaysFalse:
        def __init__(self) -> None:
            self.x = 0.0

        def __call__(self, v):  # type: ignore[no-untyped-def]
            for i in range(3):
                if i > 5:  # never true over range(3): x must not become state
                    self.x = v
            return self.x

    folded = lower(GuardAlwaysFalse().__call__)
    assert [slot.name for slot in folded.state_slots] == []  # statically-dead write, no spurious state
    assert len(folded.blocks) == 1  # every guard folded, no branch emitted

    class GuardReal:
        def __init__(self) -> None:
            self.acc = 0.0

        def __call__(self, v):  # type: ignore[no-untyped-def]
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

        def __call__(self, x):  # type: ignore[no-untyped-def]
            if self.threshold > 1.0:
                self.y = x
            if self.gain * 3.0 > 10.0:  # 6.0 > 10.0 is statically false
                self.y = x
            return self.y

    folded = lower(ConfigGate().__call__)
    assert [slot.name for slot in folded.state_slots] == []  # both float guards fold false: no spurious state
    assert len(folded.blocks) == 1  # no runtime branch emitted

    class ConfigEnabled:
        def __init__(self) -> None:
            self.threshold = 5.0
            self.y = 0.0

        def __call__(self, x):  # type: ignore[no-untyped-def]
            if self.threshold > 1.0:  # 5.0 > 1.0 statically true: the write is taken
                self.y = x
            return self.y

    enabled = lower(ConfigEnabled().__call__)
    assert [slot.name for slot in enabled.state_slots] == ["y"]
    assert len(enabled.blocks) == 1


def test_dead_assignment_after_return_does_not_suppress_fold() -> None:
    # Regression (Codex finding 3): the read-only-attribute scan stops at a return (like lowering), so an assignment
    # in dead code after a return does not mask the attribute's read-only-ness and the branch on it still folds.
    class DeadAfterReturn:
        def __init__(self) -> None:
            self.flag = False
            self.y = 0.0

        def __call__(self, x):  # type: ignore[no-untyped-def]
            if self.flag:  # flag is read-only -> folds to the (empty) else arm
                self.y = x
            result = self.y
            return result
            self.flag = False  # noqa -- dead code: must not count as an assignment of flag

    hir = lower(DeadAfterReturn().__call__)
    assert [slot.name for slot in hir.state_slots] == []  # folded: y never written, no spurious state
    assert len(hir.blocks) == 1  # no runtime branch emitted


def test_boolean_comparison_operand_is_rejected() -> None:
    # Regression (Codex): comparing booleans (e.g. `self.flag == True`) must raise a clear UnsupportedConstruct, not an
    # internal error from feeding a boolean into the float comparator.
    class BoolCompare:
        def __init__(self) -> None:
            self.flag = False
            self.y = 0.0

        def __call__(self, x):  # type: ignore[no-untyped-def]
            if self.flag == True:  # noqa: E712 -- exercising the rejection
                self.y = x
            return self.y

    with pytest.raises(UnsupportedConstruct, match="floating-point"):
        lower(BoolCompare().__call__)


def test_public_boolean_state_attribute_is_rejected() -> None:
    # Regression (Codex): a written public boolean attribute would be exposed as a boolean output port (unsupported);
    # the frontend must reject it clearly up front rather than failing cryptically in MIR lowering.
    class PublicBool:
        def __init__(self) -> None:
            self.flag = False
            self.y = 0.0

        def __call__(self, a, b):  # type: ignore[no-untyped-def]
            self.flag = a < b
            self.y = a
            return self.y

    with pytest.raises(UnsupportedConstruct, match="public boolean attribute"):
        lower(PublicBool().__call__)


def test_while_loop_lowers_to_back_edge() -> None:
    # A while loop lowers to preheader -> header(loop phi + exit branch) -> body(back-edge jump) -> exit(ret).
    def f(a):  # type: ignore[no-untyped-def]
        x = a
        while x < 10.0:
            x = x + 1.0
        return x

    hir = lower(f)
    assert len(hir.blocks) == 4
    header = hir.blocks[1]
    assert len(header.phis) == 1  # x is the single loop-carried value
    assert isinstance(header.terminator, Branch)
    body = hir.blocks[2]
    assert isinstance(body.terminator, Jump)
    assert body.terminator.target == header.id and body.id > header.id  # the back-edge to the (lower) header


def test_while_loop_with_else_is_unsupported() -> None:
    def f(a):  # type: ignore[no-untyped-def]
        x = a
        while x < 10.0:
            x = x + 1.0
        else:
            x = x + 100.0
        return x

    with pytest.raises(UnsupportedConstruct, match="else"):
        lower(f)


def test_return_inside_while_is_unsupported() -> None:
    def f(a):  # type: ignore[no-untyped-def]
        x = a
        while x < 10.0:
            return x
        return x

    with pytest.raises(UnsupportedConstruct, match="return"):
        lower(f)


def _helper_with_return_in_while(a):  # type: ignore[no-untyped-def]
    while a > 0.0:
        return a
    return a + 1.0


def test_return_inside_inlined_callee_while_is_unsupported() -> None:
    # Regression (user): a return inside an inlined callee's while body is exempt from _reject_nested_return (the
    # callee's own return is consumed locally), so _lower_while must reject it on the body's lowered-a-return result --
    # otherwise the back-edge is emitted and the early return is silently dropped (the model would not reach Ret).
    def kernel(x):  # type: ignore[no-untyped-def]
        return _helper_with_return_in_while(x)

    with pytest.raises(UnsupportedConstruct, match="return"):
        lower(kernel)


def test_statically_false_while_is_skipped() -> None:
    # Regression (Codex): a statically-false while never runs, so its body is not lowered -- no spurious persistent
    # state from a body write, and a return in the dead body does not reach the single-exit rejection (it is skipped).
    class DeadWhileWrite:
        def __init__(self) -> None:
            self.s = 0.0

        def __call__(self, x):  # type: ignore[no-untyped-def]
            while False:
                self.s = 1.0
            return x

    hir = lower(DeadWhileWrite().__call__)
    assert [slot.name for slot in hir.state_slots] == []  # the dead body's write is not state
    assert [o.name for o in hir.outputs] == ["out_0"]
    assert len(hir.blocks) == 1  # the loop is skipped, no back-edge emitted

    def dead_while_return(a):  # type: ignore[no-untyped-def]
        while False:
            return a
        return a + 1.0

    lowered = lower(dead_while_return)  # the dead body's return is skipped, not rejected
    assert _arith_count(lowered, FloatAdd) == 1


def test_for_loop_over_unroll_threshold_is_unsupported() -> None:
    def f(a):  # type: ignore[no-untyped-def]
        x = a
        for _ in range(1000):
            x = x + a
        return x

    with pytest.raises(UnsupportedConstruct, match="unroll threshold"):
        lower(f)


def test_enormous_range_is_rejected_not_crashed() -> None:
    # Regression (Codex F10): a trip count beyond a C ssize_t must be rejected cleanly, not crash with OverflowError
    # from len(range(...)) (the unroll threshold is checked with a big-integer trip count).
    def f(a):  # type: ignore[no-untyped-def]
        x = a
        for _ in range(100000000000000000000000000000000000000):
            x = x + a
        return x

    with pytest.raises(UnsupportedConstruct, match="unroll threshold"):
        lower(f)


def test_range_zero_step_is_rejected() -> None:
    def f(a):  # type: ignore[no-untyped-def]
        x = a
        for _ in range(0, 10, 0):
            x = x + a
        return x

    with pytest.raises(UnsupportedConstruct, match="invalid range"):
        lower(f)


def test_divergent_loop_counter_as_static_index_is_rejected() -> None:
    # A counter left differing by the two branch arms must not leak as a trusted compile-time index: using it to index
    # a table afterwards is path-dependent and must be rejected, not silently compiled to one arm's value.
    def f(a):  # type: ignore[no-untyped-def]
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
    def f(a):  # type: ignore[no-untyped-def]
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

        def __call__(self, x):  # type: ignore[no-untyped-def]
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

        def __call__(self, x):  # type: ignore[no-untyped-def]
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

        def __call__(self, x):  # type: ignore[no-untyped-def]
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
    def f(a):  # type: ignore[no-untyped-def]
        return a + UNDEFINED_GLOBAL  # type: ignore[name-defined]  # noqa: F821

    with pytest.raises(UnsupportedConstruct):
        lower(f)


def test_missing_intrinsic_message() -> None:
    def f(a):  # type: ignore[no-untyped-def]
        return math.sqrt(a)

    with pytest.raises(MissingIntrinsic, match="sqrt"):
        lower(f)


def _integrator_class():  # type: ignore[no-untyped-def]
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    from trapezoidal_leaky_streaming_integrator import TrapezoidalLeakyStreamingIntegrator

    return TrapezoidalLeakyStreamingIntegrator


def test_stateful_method_state_slots_and_dedup() -> None:
    integrator = _integrator_class()(k=2**-22)
    hir = lower(integrator.__call__)
    assert hir.input_names() == ["x"]  # self is dropped; remaining parameters become inputs
    # `return self.y` is deduped onto the public state port state_y; the private _x_prev gets no port, so the output
    # list alone distinguishes public from private. Both slots reset to 0.
    assert [o.name for o in hir.outputs] == ["state_y"]
    slots = {s.name: s for s in hir.state_slots}
    assert set(slots) == {"y", "_x_prev"}
    assert slots["y"].reset_value.value == 0.0 and slots["_x_prev"].reset_value.value == 0.0
    assert {n.slot for n in hir.nodes.values() if isinstance(n, StateRead)} == {"y", "_x_prev"}


def test_returned_public_state_alias_is_deduped() -> None:
    # The dedup is by dataflow, not spelling: returning a public attribute through an alias must still collapse onto its
    # state_<attr> port rather than emitting a second positional output for the same value.
    class Aliased:
        def __init__(self) -> None:
            self.y = 0.0

        def __call__(self, x):  # type: ignore[no-untyped-def]
            self.y = x
            y = self.y
            return y

    hir = lower(Aliased().__call__)
    assert [o.name for o in hir.outputs] == ["state_y"]


def test_mixed_return_dedupes_public_alias_keeps_distinct_leaf() -> None:
    class Mixed:
        def __init__(self) -> None:
            self.y = 0.0

        def __call__(self, x):  # type: ignore[no-untyped-def]
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

        def __call__(self, x):  # type: ignore[no-untyped-def]
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

        def __call__(self, x):  # type: ignore[no-untyped-def]
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

        def __call__(self, x):  # type: ignore[no-untyped-def]
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
    # k is only read, so it is a folded constant, not a persistent slot or a state read.
    assert "k" not in {s.name for s in hir.state_slots}
    assert all(not (isinstance(n, StateRead) and n.slot == "k") for n in hir.nodes.values())


def test_stateful_reset_state_is_the_instance_snapshot() -> None:
    # The reset value is whatever the instance holds at synthesis time, including post-construction mutation.
    integrator = _integrator_class()(k=2**-22)
    integrator.y = 1.5  # type: ignore[attr-defined]
    slots = {s.name: s for s in lower(integrator.__call__).state_slots}
    assert slots["y"].reset_value.value == 1.5


def test_init_method_target_is_rejected() -> None:
    integrator = _integrator_class()(k=2**-22)
    with pytest.raises(UnsupportedConstruct, match="__init__"):
        lower(integrator.__init__)


def test_class_object_target_is_rejected() -> None:
    with pytest.raises(UnsupportedConstruct, match="bound method"):
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
            self.scratch = x  # never initialized on the instance
            return self.y

    with pytest.raises(UnsupportedConstruct, match="not initialized"):
        lower(Bad().__call__)


def test_nested_attribute_access_is_rejected() -> None:
    class Bad:
        def __init__(self) -> None:
            self.y = 0.0

        def __call__(self, x: float) -> float:
            return x + self.y.real  # nested attribute access on self.y

    with pytest.raises(UnsupportedConstruct, match="direct self"):
        lower(Bad().__call__)


# --- Compile-time aggregates -----------------------------------------------------------------------------------------


def test_tuple_build_and_index() -> None:
    def f(a, b):  # type: ignore[no-untyped-def]
        z = a, b
        return [z[1], z[0]]  # swapped

    assert [o.name for o in lower(f).outputs] == ["out_0", "out_1"]


def test_list_slice() -> None:
    def f(a, b, c):  # type: ignore[no-untyped-def]
        v = [a, b, c]
        return v[1:3]

    assert [o.name for o in lower(f).outputs] == ["out_0", "out_1"]


def test_vector_scalar_broadcast() -> None:
    def f(a, b):  # type: ignore[no-untyped-def]
        v = [a, b]
        return v * 0.5  # elementwise: one multiply per leaf

    hir = lower(f)
    assert _arith_count(hir, FloatMul) == 2
    assert [o.name for o in hir.outputs] == ["out_0", "out_1"]


def test_flatten_collapses_nesting() -> None:
    def f(a, b):  # type: ignore[no-untyped-def]
        m = [[a], [b]]
        return m.flatten()

    assert [o.name for o in lower(f).outputs] == ["out_0", "out_1"]


def test_index_out_of_range_is_rejected() -> None:
    def f(a):  # type: ignore[no-untyped-def]
        v = [a]
        return v[3]

    with pytest.raises(UnsupportedConstruct, match="out of range"):
        lower(f)


def test_indexing_a_scalar_is_rejected() -> None:
    def f(a):  # type: ignore[no-untyped-def]
        return a[0]

    with pytest.raises(UnsupportedConstruct, match="index or slice a scalar"):
        lower(f)


def test_star_unpacking_a_scalar_is_rejected() -> None:
    def f(a):  # type: ignore[no-untyped-def]
        return [*a]

    with pytest.raises(UnsupportedConstruct, match="unpack"):
        lower(f)


# --- Tuple-unpacking assignment --------------------------------------------------------------------------------------


def test_tuple_unpacking_routes_values() -> None:
    # The right-hand side is built once before any binding, so a swap reads both sources first (no clobber).
    def swap(a, b):  # type: ignore[no-untyped-def]
        x, y = b, a
        return [x, y]

    hir = lower(swap)
    assert hir.input_names() == ["a", "b"]
    assert [o.value for o in hir.outputs] == [hir.input_ids[1], hir.input_ids[0]]


def test_starred_and_nested_unpacking_route_values() -> None:
    def f(a, b, c):  # type: ignore[no-untyped-def]
        first, *rest = [a, b, c]  # rest binds the surplus as an aggregate
        r0, r1 = rest  # nested unpacking of that aggregate
        return [first, r0, r1]

    hir = lower(f)
    assert [o.value for o in hir.outputs] == list(hir.input_ids)


def test_chained_assignment_binds_every_target() -> None:
    def f(a):  # type: ignore[no-untyped-def]
        x = y = a + a
        return [x, y]

    hir = lower(f)
    out = [o.value for o in hir.outputs]
    assert out[0] == out[1]  # both targets name the same single value


def test_unpacking_a_scalar_source_is_rejected() -> None:
    def f(a):  # type: ignore[no-untyped-def]
        x, y = a
        return x + y

    with pytest.raises(UnsupportedConstruct, match="unpack a scalar"):
        lower(f)


def test_unpacking_arity_mismatch_is_rejected() -> None:
    def f(a, b, c):  # type: ignore[no-untyped-def]
        x, y = [a, b, c]
        return x + y

    with pytest.raises(UnsupportedConstruct, match="unpack 3 values into 2"):
        lower(f)


def test_stateful_tuple_assignment_to_attributes() -> None:
    # Unpacking into self attributes must register both as persistent state; the swap reads the live-ins first.
    class Rotate:
        def __init__(self) -> None:
            self.x = 1.0
            self.y = 2.0

        def step(self, k):  # type: ignore[no-untyped-def]
            self.x, self.y = self.y, self.x + k
            return self.x

    hir = lower(Rotate().step)
    assert {s.name for s in hir.state_slots} == {"x", "y"}
    assert "state_x" in {o.name for o in hir.outputs}


def test_unpacked_name_shadows_global_callable() -> None:
    # A name bound only via tuple unpacking is local, so a same-named global function is not inlined at a call site;
    # this exercises _collect_local_names descending into unpacking targets.
    def f(a):  # type: ignore[no-untyped-def]
        _addmul, b = a, a  # _addmul is now a local value (Python would raise 'float not callable' when called)
        return _addmul(b)

    with pytest.raises(UnsupportedConstruct, match="not a callable"):
        lower(f)


# --- Importing and inlining a pure function --------------------------------------------------------------------------


def _addmul(p, q):  # type: ignore[no-untyped-def]
    return [p + q, p * q]


def test_inlined_global_function() -> None:
    def f(a, b):  # type: ignore[no-untyped-def]
        return _addmul(a, b)

    hir = lower(f)
    assert [o.name for o in hir.outputs] == ["out_0", "out_1"]
    assert _arith_count(hir, FloatAdd) == 1 and _arith_count(hir, FloatMul) == 1


def test_inlined_global_with_star_args() -> None:
    def f(a, b):  # type: ignore[no-untyped-def]
        v = [a, b]
        return _addmul(*v)

    hir = lower(f)
    assert _arith_count(hir, FloatAdd) == 1 and _arith_count(hir, FloatMul) == 1


def test_inline_arity_mismatch_is_rejected() -> None:
    def f(a):  # type: ignore[no-untyped-def]
        return _addmul(a)  # _addmul takes two positional arguments

    with pytest.raises(UnsupportedConstruct, match="positional arguments"):
        lower(f)


def cbrt(x):  # type: ignore[no-untyped-def]
    return x * x  # a user-defined global whose name collides with the same-named intrinsic placeholder


def test_user_global_function_shadows_intrinsic_name() -> None:
    # A module-level def named like an intrinsic is the caller's own function; Python would call it, so it is inlined.
    def f(a):  # type: ignore[no-untyped-def]
        return cbrt(a)

    assert _arith_count(lower(f), FloatMul) == 1  # the inlined x * x, not a MissingIntrinsic rejection


def test_local_name_shadows_global_callable() -> None:
    # A parameter named like a global function refers to the parameter (a value), which is not callable.
    def f(_addmul, a):  # type: ignore[no-untyped-def]
        return _addmul(a)

    with pytest.raises(UnsupportedConstruct, match="not a callable"):
        lower(f)


def test_flatten_on_a_scalar_is_rejected() -> None:
    def f(a):  # type: ignore[no-untyped-def]
        return a.flatten()

    with pytest.raises(UnsupportedConstruct, match="aggregate"):
        lower(f)


def test_boolean_in_float_arithmetic_is_rejected() -> None:
    # Boolean literals are supported as values (branch conditions, boolean state), but arithmetic on them is not:
    # negating a boolean fails the float operator's operand-type check.
    def f():  # type: ignore[no-untyped-def]
        return -True

    with pytest.raises((UnsupportedConstruct, ValueError)):
        lower(f)


def test_abs_accepts_a_star_unpacked_argument() -> None:
    # Call-argument unpacking applies uniformly: abs(*v) on a one-element aggregate is abs of that single element.
    def f(a):  # type: ignore[no-untyped-def]
        v = [a]
        return abs(*v)

    assert _arith_count(lower(f), FloatAbs) == 1


def test_unary_plus_is_scalar_identity_and_rejects_aggregates() -> None:
    def scalar_ok(a):  # type: ignore[no-untyped-def]
        return +a  # identity on a scalar

    assert [o.name for o in lower(scalar_ok).outputs] == ["out_0"]

    def aggregate_bad(a, b):  # type: ignore[no-untyped-def]
        v = [a, b]
        return +v  # unary plus does not apply to an aggregate

    with pytest.raises(UnsupportedConstruct, match="scalar"):
        lower(aggregate_bad)


def test_method_style_abs_call_is_rejected() -> None:
    # Only a bare-name abs(...) is the builtin; a method-style a.abs(b) must not be silently treated as it (which would
    # drop the receiver) -- there is no supported scalar method, so it is an unsupported call.
    def f(a, b):  # type: ignore[no-untyped-def]
        return a.abs(b)

    with pytest.raises(UnsupportedConstruct, match="abs"):
        lower(f)


def _rebind_globals(fn, **overrides):  # type: ignore[no-untyped-def]
    """A copy of ``fn`` whose module globals carry ``overrides`` (its source stays retrievable via the shared code)."""
    return types.FunctionType(
        fn.__code__, {**fn.__globals__, **overrides}, fn.__name__, fn.__defaults__, fn.__closure__
    )


def test_noncallable_global_shadowing_builtin_is_rejected() -> None:
    # A non-callable global shadows the built-in (Python raises TypeError on the call), so the name is not the builtin
    # it spells; holoso must reject rather than silently emitting FloatAbs / the list-tuple identity.
    def use_abs(a):  # type: ignore[no-untyped-def]
        return abs(a)

    def use_list(a):  # type: ignore[no-untyped-def]
        return list((a, a))

    def use_tuple(a):  # type: ignore[no-untyped-def]
        return tuple((a, a))

    # ``None`` shadows too -- it is present-but-non-callable, distinct from an absent global (the _ABSENT sentinel).
    shadows = ((use_abs, {"abs": 5}), (use_abs, {"abs": None}), (use_list, {"list": 5}), (use_tuple, {"tuple": 5}))
    for fn, shadow in shadows:
        with pytest.raises(UnsupportedConstruct, match="non-callable"):
            lower(_rebind_globals(fn, **shadow))


def test_callable_global_shadowing_abs_is_inlined_not_floatabs() -> None:
    # A callable global named ``abs`` is the caller's own function; Python would call it, so holoso inlines it instead
    # of emitting the FloatAbs builtin -- the non-callable guard must not disturb this legitimate shadow.
    def use_abs(a):  # type: ignore[no-untyped-def]
        return abs(a)

    hir = lower(_rebind_globals(use_abs, abs=cbrt))  # cbrt is a module-level def returning x * x
    assert _arith_count(hir, FloatAbs) == 0 and _arith_count(hir, FloatMul) == 1


# --- numpy-array aggregates and executable-numpy interop --------------------------------------------------------------


def test_numpy_array_state_decomposes_like_a_list() -> None:
    import numpy.typing as npt

    @dataclasses.dataclass
    class Filt:
        v: npt.NDArray[np.float64]  # shape-less annotation: holoso infers the length from the reset value

        def step(self, a):  # type: ignore[no-untyped-def]
            self.v = self.v * a

    hir = lower(Filt(np.array([1.0, 2.0, 3.0])).step)
    assert {s.name for s in hir.state_slots} == {"v_0", "v_1", "v_2"}
    assert [o.name for o in hir.outputs] == ["state_v_0", "state_v_1", "state_v_2"]


def test_jaxtyping_array_field_lowers_and_is_validated() -> None:
    from jaxtyping import Float64

    @dataclasses.dataclass
    class Filt:
        v: Float64[np.ndarray, "3"]

        def step(self, a):  # type: ignore[no-untyped-def]
            self.v = self.v * a

    assert {s.name for s in lower(Filt(np.array([1.0, 2.0, 3.0])).step).state_slots} == {"v_0", "v_1", "v_2"}
    with pytest.raises(UnsupportedConstruct, match="declared array type"):
        lower(Filt(np.array([1.0, 2.0, 3.0, 4.0])).step)  # value shape (4,) violates the declared "3"


def test_numpy_integer_array_values_coerce_to_real() -> None:
    @dataclasses.dataclass
    class Filt:
        v: np.ndarray  # type: ignore[type-arg]

        def step(self, a):  # type: ignore[no-untyped-def]
            self.v = self.v * a

    assert {s.name for s in lower(Filt(np.array([2, 3])).step).state_slots} == {"v_0", "v_1"}


def test_numpy_asarray_is_identity_on_an_aggregate() -> None:
    def f(a, b):  # type: ignore[no-untyped-def]
        return np.asarray([a, b]).flatten()  # asarray of an array-like is identity in this compile-time model

    assert [o.name for o in lower(f).outputs] == ["out_0", "out_1"]


def test_list_is_identity_on_an_aggregate() -> None:
    def f(a, b, c):  # type: ignore[no-untyped-def]
        v = [a, b, c]
        return list(v[0:2])  # list() of a slice carries the same elements -- identity here

    assert [o.name for o in lower(f).outputs] == ["out_0", "out_1"]


def test_list_of_a_scalar_is_rejected() -> None:
    def f(a):  # type: ignore[no-untyped-def]
        return list(a)  # Python: list(scalar) is a TypeError -- a scalar is not iterable

    with pytest.raises(UnsupportedConstruct, match="list"):
        lower(f)


def test_tuple_is_identity_on_an_aggregate() -> None:
    def f(a, b, c):  # type: ignore[no-untyped-def]
        v = [a, b, c]
        return tuple(v[0:2])  # tuple() of a slice is identity here, co-equal with list()

    assert [o.name for o in lower(f).outputs] == ["out_0", "out_1"]


def test_numpy_alias_shadowed_by_a_local_is_not_numpy() -> None:
    # ``np`` is rebound to a local value, so ``np.asarray`` is a method call on that value, not the numpy function.
    def f(a):  # type: ignore[no-untyped-def]
        np = [a]
        return np.asarray([a])

    with pytest.raises(UnsupportedConstruct, match="asarray"):
        lower(f)


def test_name_assigned_later_is_local_before_its_assignment() -> None:
    # A name assigned anywhere in a function is local throughout (Python's rule); using it as a global/builtin/numpy
    # before that assignment is invalid Python (UnboundLocalError), so holoso rejects it rather than seeing the global.
    def shadows_numpy(a):  # type: ignore[no-untyped-def]
        y = np.asarray([a])
        np = [a]  # noqa: F841  # makes np local for the whole body
        return y

    with pytest.raises(UnsupportedConstruct):
        lower(shadows_numpy)

    def shadows_builtin(a):  # type: ignore[no-untyped-def]
        y = abs(a)
        abs = [a]  # noqa: F841  # makes abs local for the whole body
        return y

    with pytest.raises(UnsupportedConstruct, match="local name"):
        lower(shadows_builtin)


def test_multidimensional_array_state_is_rejected() -> None:
    @dataclasses.dataclass
    class Filt:
        m: np.ndarray  # type: ignore[type-arg]

        def step(self, a):  # type: ignore[no-untyped-def]
            self.m = self.m * a

    with pytest.raises(UnsupportedConstruct, match="1-D"):
        lower(Filt(np.array([[1.0, 2.0], [3.0, 4.0]])).step)


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


# --- Vector-valued state and keyword-only inputs ---------------------------------------------------------------------


def test_vector_state_decomposes_to_per_element_slots() -> None:
    class Vec:
        def __init__(self) -> None:
            self.v = [1.0, 2.0, 3.0]

        def update(self, a):  # type: ignore[no-untyped-def]
            self.v = [self.v[0] + a, self.v[1], self.v[2]]

    hir = lower(Vec().update)
    assert {s.name: s.reset_value.value for s in hir.state_slots} == {"v_0": 1.0, "v_1": 2.0, "v_2": 3.0}
    assert [o.name for o in hir.outputs] == ["state_v_0", "state_v_1", "state_v_2"]


def test_vector_state_shape_mismatch_is_rejected() -> None:
    class Vec:
        def __init__(self) -> None:
            self.v = [0.0, 0.0]

        def update(self, a):  # type: ignore[no-untyped-def]
            self.v = [a]  # the slot holds two scalars, but one is assigned

    with pytest.raises(UnsupportedConstruct, match="2-element vector"):
        lower(Vec().update)


def test_vector_state_nested_shape_is_rejected() -> None:
    # A nested aggregate has the right leaf count (2) but the wrong shape: the slot layout is a flat 2-vector, so the
    # next transaction would reconstruct a flat shape that disagrees with the one written this transaction.
    class Vec:
        def __init__(self) -> None:
            self.v = [0.0, 0.0]

        def update(self, a, b):  # type: ignore[no-untyped-def]
            self.v = [[a, b]]

    with pytest.raises(UnsupportedConstruct, match="incompatible shape"):
        lower(Vec().update)


def test_vector_state_slot_name_collision_is_rejected() -> None:
    # The vector ``v`` decomposes into slot ``v_0``, which would alias the distinct scalar attribute ``v_0``.
    class Vec:
        def __init__(self) -> None:
            self.v = [1.0]
            self.v_0 = 2.0

        def update(self, a):  # type: ignore[no-untyped-def]
            self.v = [a]
            self.v_0 = a + 1.0

    with pytest.raises(UnsupportedConstruct, match="aliasing collision"):
        lower(Vec().update)


def test_keyword_only_params_become_inputs() -> None:
    def f(a, *, b, c):  # type: ignore[no-untyped-def]
        return a + b + c

    assert lower(f).input_names() == ["a", "b", "c"]


def test_dataclass_instance_is_stateful() -> None:
    @dataclasses.dataclass
    class Acc:
        total: float
        gain: list  # type: ignore[type-arg]

        def step(self, x):  # type: ignore[no-untyped-def]
            self.total = self.total + x * self.gain[0]

    hir = lower(Acc(0.0, [2.0]).step)
    assert {s.name for s in hir.state_slots} == {"total"}  # gain is read-only config, not state
    assert [o.name for o in hir.outputs] == ["state_total"]


def test_first_sample_branch_lowers_to_branch_and_phis() -> None:
    # examples/iir1_lpf.py: a boolean first-sample state and an if/else that both write self.y.
    class Iir:
        def __init__(self):  # type: ignore[no-untyped-def]
            self.alpha = 2**-16
            self.y = 0.0
            self._first = True

        def __call__(self, x):  # type: ignore[no-untyped-def]
            if self._first:
                self._first = False
                self.y = x
            else:
                self.y += self.alpha * (x - self.y)
            return self.y

    hir = optimize(lower(Iir().__call__))
    # Four blocks: entry (branch on _first), then, else, merge (ret).
    assert len(hir.blocks) == 4
    assert isinstance(hir.blocks[0].terminator, Branch)
    cond = hir.blocks[0].terminator.cond
    assert isinstance(hir.nodes[cond], StateRead) and isinstance(hir.nodes[cond].type, BoolType)
    assert isinstance(hir.blocks[-1].terminator, Ret)
    merge_phis = [hir.nodes[p] for p in hir.blocks[-1].phis]
    assert len(merge_phis) == 2 and all(isinstance(p, Phi) for p in merge_phis)
    slots = {s.name: s for s in hir.state_slots}
    assert isinstance(slots["_first"].reset_value, BoolConst) and slots["_first"].reset_value.value is True
    assert isinstance(slots["y"].reset_value, FloatConst) and slots["y"].reset_value.value == 0.0
    assert [o.name for o in hir.outputs] == ["state_y"]  # return self.y dedups onto the public state port


def test_nested_if_lowers_through_optimize() -> None:
    # Regression: block visitation must be topological -- an inner if's merge feeds the outer merge phi. The conditions
    # are dynamic comparisons (a read-only boolean attribute would fold to one arm and emit no branch).
    class C:
        def __init__(self):  # type: ignore[no-untyped-def]
            self.y = 0.0

        def __call__(self, x, w):  # type: ignore[no-untyped-def]
            if x > 0.0:
                if w > 0.0:
                    self.y = x
            return self.y

    hir = optimize(lower(C().__call__))
    assert any(isinstance(b.terminator, Branch) for b in hir.blocks)
    assert {s.name for s in hir.state_slots} == {"y"}


def test_attribute_written_on_one_arm_becomes_a_phi() -> None:
    # The update lives in only one arm (anti-windup style); its live-out is a phi against the live-in. The condition is
    # a dynamic comparison so a real branch is emitted (a read-only boolean attribute would fold the branch away).
    class Clamp:
        def __init__(self):  # type: ignore[no-untyped-def]
            self.acc = 0.0

        def __call__(self, x):  # type: ignore[no-untyped-def]
            if x > 0.0:
                self.acc = x
            return self.acc

    hir = optimize(lower(Clamp().__call__))
    slots = {s.name: s for s in hir.state_slots}
    assert isinstance(hir.nodes[slots["acc"].live_out], Phi)  # merged: written value on one path, live-in on the other


def _op_count(hir, op_type):  # type: ignore[no-untyped-def]
    return sum(1 for n in hir.nodes.values() if isinstance(n, Operation) and type(n.operator) is op_type)


def test_boolean_and_in_condition_lowers_to_combinational_bool_and() -> None:
    def f(x, a, b):  # type: ignore[no-untyped-def]
        return 1.0 if (x > a and x < b) else 0.0

    hir = lower(f)
    assert _op_count(hir, BoolAnd) == 1
    assert _op_count(hir, FloatRelational) == 2  # the two comparisons feeding the AND


def test_boolean_or_lowers_to_combinational_bool_or() -> None:
    def f(x, a, b):  # type: ignore[no-untyped-def]
        return 1.0 if (x < a or x > b) else 0.0

    hir = lower(f)
    assert _op_count(hir, BoolOr) == 1
    assert _op_count(hir, FloatRelational) == 2


def test_not_lowers_to_combinational_bool_not() -> None:
    def f(x):  # type: ignore[no-untyped-def]
        return -1.0 if not (x > 0.0) else 1.0

    hir = lower(f)
    assert _op_count(hir, BoolNot) == 1
    assert _op_count(hir, FloatRelational) == 1


def test_chained_comparison_lowers_to_two_comparisons_and_one_and() -> None:
    def f(x, lo, hi):  # type: ignore[no-untyped-def]
        return 0.0 if lo < x < hi else x

    hir = lower(f)
    assert _op_count(hir, FloatRelational) == 2
    assert _op_count(hir, BoolAnd) == 1


def test_chained_comparison_evaluates_each_operand_once() -> None:
    # The shared middle operand ``x`` feeds both comparisons but is evaluated once: only one Sub (x - 0.5) is built.
    def f(a):  # type: ignore[no-untyped-def]
        x = a - 0.5
        return 0.0 if 0.0 < x < 1.0 else x

    hir = lower(f)
    assert _op_count(hir, FloatAdd) == 1  # subtraction lowers to add(+neg); only one, so x was built once


def _branch_count(hir) -> int:  # type: ignore[no-untyped-def]
    return sum(1 for block in hir.blocks if isinstance(block.terminator, Branch))


def test_nested_if_without_else_folds_into_one_and_branch() -> None:
    # ``if A: (if B: S)`` with no ``else`` on either is exactly ``if (A and B): S``: the frontend folds it to a single
    # branch (one combinational ``and``), NOT two nested jumps. Folded repeatedly, ``if A: if B: if C: S`` collapses to
    # one ``A and B and C`` branch. Regression: the nested form must compile identically to the hand-written ``and``.
    def nested(x, lo, hi):  # type: ignore[no-untyped-def]
        r = 0.0
        if x > lo:
            if x < hi:
                r = 1.0
        return r

    def manual(x, lo, hi):  # type: ignore[no-untyped-def]
        r = 0.0
        if x > lo and x < hi:
            r = 1.0
        return r

    def triple(x, lo, hi):  # type: ignore[no-untyped-def]
        r = 0.0
        if x > lo:
            if x < hi:
                if x != 0.0:
                    r = 1.0
        return r

    assert _branch_count(lower(nested)) == 1  # one branch, not two
    assert _branch_count(lower(nested)) == _branch_count(lower(manual))
    assert _op_count(lower(nested), BoolAnd) == 1  # the conjunction the fold synthesized
    assert _branch_count(lower(triple)) == 1  # still a single branch
    assert _op_count(lower(triple), BoolAnd) == 2  # A and B and C -> two binary ANDs


def test_nested_if_with_outer_else_does_not_fold() -> None:
    # The fold must NOT trigger when the outer ``if`` has an ``else``: ``if A: (if B: S) else: T`` is not
    # ``if (A and B): S``, because T runs whenever ``not A``, whereas ``not (A and B)`` also covers A-and-not-B. Both
    # branches must survive and no spurious conjunction is synthesized.
    def f(x, lo, hi):  # type: ignore[no-untyped-def]
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
    def f(x):  # type: ignore[no-untyped-def]
        r = 0.0
        if x > 0.0:
            if (t := x * 3.0) < 100.0:
                r = t
        return r

    assert _branch_count(lower(f)) == 2  # not folded into a single ``and`` branch


def test_walrus_in_conditional_expression_arm_is_rejected() -> None:
    # A ternary arm is evaluated only when selected; a walrus binding there cannot leak across arms, so it is rejected.
    def f(x):  # type: ignore[no-untyped-def]
        return (t := 1.0) if x > 0.0 else (t := 2.0)

    with pytest.raises(UnsupportedConstruct, match="walrus"):
        lower(f)


def test_walrus_in_and_or_operand_is_rejected() -> None:
    # An ``and``/``or`` operand may be short-circuited (statically dropped by the connective fold, or unevaluated in
    # Python), so whether its walrus binds cannot be reconciled between the scans and lowering -- rejected. (Regression:
    # the scan invalidated the target unconditionally while lowering short-circuited past it, desyncing the two.)
    def f(x, y):  # type: ignore[no-untyped-def]
        if x > 0.0 and (t := y) > 0.0:
            return t
        return 0.0

    with pytest.raises(UnsupportedConstruct, match="walrus"):
        lower(f)


def test_walrus_in_chained_comparison_is_rejected() -> None:
    def f(x):  # type: ignore[no-untyped-def]
        if x < 0.0 < (t := 5.0):
            return t
        return 0.0

    with pytest.raises(UnsupportedConstruct, match="walrus"):
        lower(f)


_DEAD_WALRUS_GLOBAL = 3  # a module global a dead-code walrus shadows below


def test_walrus_in_dead_or_unsupported_statement_still_scopes_the_name_local() -> None:
    # Local-name collection is syntactic, as in Python: a walrus target is a function local throughout the body even in
    # dead or out-of-subset code, so it shadows a same-named global. Here the earlier ``range(_DEAD_WALRUS_GLOBAL)`` must
    # see the runtime local (rejected), NOT silently fold the module int 3 from the global.
    def f(x):  # type: ignore[no-untyped-def]
        v = x
        for _ in range(_DEAD_WALRUS_GLOBAL):
            v = v + 1.0
        return v
        assert (_DEAD_WALRUS_GLOBAL := 2)  # noqa -- unreachable, but makes the name a local for the whole function

    with pytest.raises(UnsupportedConstruct):
        lower(f)


def test_walrus_in_a_nested_scope_default_scopes_the_name_in_the_enclosing_function() -> None:
    # A nested def/lambda is a separate scope, but its default-argument expressions execute in the ENCLOSING scope, so a
    # walrus there binds an enclosing local (as in Python). The earlier ``range(_DEAD_WALRUS_GLOBAL)`` must therefore see
    # the runtime local, not the module int -- even though the lambda is dead code that lowering never reaches.
    def f(x):  # type: ignore[no-untyped-def]
        v = x
        for _ in range(_DEAD_WALRUS_GLOBAL):
            v = v + 1.0
        return v
        h = lambda y=(_DEAD_WALRUS_GLOBAL := 1): y  # noqa: E731 -- dead, but its default's walrus is an enclosing local

    with pytest.raises(UnsupportedConstruct):
        lower(f)


def test_walrus_in_while_condition_is_rejected() -> None:
    # A while-condition walrus rebinds every iteration and its post-test value is the loop-exit value, which the header
    # phi does not capture; rejected rather than miscompiled.
    def f(x):  # type: ignore[no-untyped-def]
        while (x := x - 1.0) > 0.0:
            pass
        return x

    with pytest.raises(UnsupportedConstruct, match="walrus"):
        lower(f)


_WALRUS_SHADOWED_INT = 3  # a module global an inner walrus shadows in the test below


def test_walrus_target_shadowing_a_global_int_is_a_runtime_local() -> None:
    # Python makes a walrus target a function local for the whole body, shadowing a same-named module global. Using it
    # as a static range bound must therefore see the runtime local (rejected), NOT silently fold the global's value.
    def f(x):  # type: ignore[no-untyped-def]
        v = x
        if (_WALRUS_SHADOWED_INT := x) > 0.0:
            for _ in range(_WALRUS_SHADOWED_INT):  # the local float, not the module int 3
                v = v + 1.0
        return v

    with pytest.raises(UnsupportedConstruct):
        lower(f)


def test_conditional_expression_lowers_to_branch_and_phi() -> None:
    def f(x, y, c):  # type: ignore[no-untyped-def]
        return x if c > 0.0 else y

    hir = lower(f)
    assert any(isinstance(node.terminator, Branch) for node in hir.blocks)
    assert any(isinstance(n, Phi) for n in hir.nodes.values())


def test_nested_conditional_expression_clamp_lowers() -> None:
    def f(x, lo, hi):  # type: ignore[no-untyped-def]
        return hi if x > hi else (lo if x < lo else x)

    hir = lower(f)
    assert _op_count(hir, FloatRelational) == 2
    assert sum(1 for n in hir.nodes.values() if isinstance(n, Phi)) >= 2  # one phi per ternary merge


def test_statically_true_connective_in_condition_does_not_branch() -> None:
    def f(x):  # type: ignore[no-untyped-def]
        return 1.0 if (1.0 < 2.0 and 3.0 > 2.0) else 0.0

    hir = lower(f)
    assert len(hir.blocks) == 1  # the whole guard folds to True: no branch, no comparison
    assert _op_count(hir, FloatRelational) == 0
    assert _op_count(hir, BoolAnd) == 0


def test_statically_true_connective_operand_is_dropped() -> None:
    def f(x):  # type: ignore[no-untyped-def]
        return 1.0 if (True and x > 0.0) else 0.0

    hir = lower(f)
    assert _op_count(hir, BoolAnd) == 0  # the identity True is dropped; the AND collapses to the single comparison
    assert _op_count(hir, FloatRelational) == 1


def test_non_boolean_connective_operand_is_rejected() -> None:
    # Python's value-returning ``and``/``or`` over non-booleans is out of subset: a float operand in a boolean
    # position must raise rather than silently feed a non-boolean into the logic.
    def f(x, y):  # type: ignore[no-untyped-def]
        return 1.0 if (x and y) else 0.0

    with pytest.raises(UnsupportedConstruct, match="boolean"):
        lower(f)


def test_chained_comparison_with_boolean_operand_is_rejected() -> None:
    class BoolMid:
        def __init__(self) -> None:
            self.flag = False
            self.y = 0.0

        def __call__(self, x):  # type: ignore[no-untyped-def]
            return 1.0 if 0.0 < self.flag < 1.0 else self.y  # noqa -- exercising the rejection

    with pytest.raises(UnsupportedConstruct, match="floating-point"):
        lower(BoolMid().__call__)


def test_not_of_non_boolean_is_rejected() -> None:
    def f(x):  # type: ignore[no-untyped-def]
        return 1.0 if not x else 0.0

    with pytest.raises(UnsupportedConstruct, match="boolean"):
        lower(f)


def test_bool_cast_lowers_to_float_to_bool() -> None:
    def f(x, y):  # type: ignore[no-untyped-def]
        return 1.0 if bool(x) else y

    hir = lower(f)
    assert _op_count(hir, FloatToBool) == 1


def test_bool_of_a_boolean_is_identity() -> None:
    def f(x, a):  # type: ignore[no-untyped-def]
        return 1.0 if bool(x > a) else 0.0

    hir = lower(f)
    assert _op_count(hir, FloatToBool) == 0  # bool(<bool>) is identity; only the comparison remains
    assert _op_count(hir, FloatRelational) == 1


def test_bool_cast_rejects_aggregate_argument() -> None:
    def f(x, y):  # type: ignore[no-untyped-def]
        return 1.0 if bool((x, y)) else 0.0

    with pytest.raises(UnsupportedConstruct, match="scalar"):
        lower(f)


def test_bool_cast_rejects_multiple_arguments() -> None:
    def f(x, y):  # type: ignore[no-untyped-def]
        return 1.0 if bool(x, y) else 0.0  # type: ignore[call-arg, arg-type]

    with pytest.raises(UnsupportedConstruct, match="single scalar"):
        lower(f)


def test_float_cast_of_bool_lowers_to_bool_to_float() -> None:
    def f(x):  # type: ignore[no-untyped-def]
        return float(x > 0.0)

    hir = lower(f)
    assert _op_count(hir, BoolToFloat) == 1
    assert _op_count(hir, FloatRelational) == 1


def test_float_cast_of_float_is_identity() -> None:
    def f(x):  # type: ignore[no-untyped-def]
        return float(x) + 1.0

    hir = lower(f)
    assert _op_count(hir, BoolToFloat) == 0  # float(<float>) is identity; no cast op
    assert _op_count(hir, FloatAdd) == 1


def test_cross_domain_cast_chain_lowers() -> None:
    # The keystone: float -> bool (comparison) -> float (cast) -> float (multiply) all in one straight-line block.
    def f(x, k):  # type: ignore[no-untyped-def]
        return float(x > 0.0) * k

    hir = lower(f)
    assert _op_count(hir, FloatRelational) == 1
    assert _op_count(hir, BoolToFloat) == 1
    assert _op_count(hir, FloatMul) == 1


def test_float_cast_rejects_aggregate_argument() -> None:
    def f(x, y):  # type: ignore[no-untyped-def]
        return float((x, y))[0]  # type: ignore[index]

    with pytest.raises(UnsupportedConstruct, match="scalar"):
        lower(f)


def test_non_boolean_or_operand_before_absorbing_constant_is_rejected() -> None:
    # Regression (Codex): ``x or True`` with a float x must be rejected, not folded to constant True. Python evaluates
    # x first and returns it when falsy, so a non-boolean operand reached before the absorbing constant cannot be
    # silently folded away -- it must be lowered and type-checked.
    def f(x):  # type: ignore[no-untyped-def]
        return 1.0 if (x or True) else 0.0

    with pytest.raises(UnsupportedConstruct, match="boolean"):
        lower(f)


def _fn_with_globals(name, src, extra_globals):  # type: ignore[no-untyped-def]
    import linecache

    filename = f"<shadow_{name}>"
    linecache.cache[filename] = (len(src), None, [line + "\n" for line in src.splitlines()], filename)
    namespace = {**extra_globals}
    exec(compile(src, filename, "exec"), namespace)
    return namespace[name]


def test_callable_global_shadowing_bool_is_not_treated_as_the_builtin() -> None:
    # Regression (Codex): a callable global named ``bool`` (e.g. a callable instance) is what Python would call, so the
    # bare-name ``bool(x)`` must NOT be lowered as the builtin float->bool cast; it is rejected as an unsupported call.
    class AlwaysFalse:
        def __call__(self, x):  # type: ignore[no-untyped-def]
            return False

    f = _fn_with_globals("f", "def f(x):\n    return 1.0 if bool(x) else 0.0\n", {"bool": AlwaysFalse()})
    with pytest.raises(UnsupportedConstruct, match="bool"):
        lower(f)


def test_static_bool_sees_through_bool_cast_so_return_in_branch_folds() -> None:
    # Regression (Codex round 2): the static evaluator must see through ``bool(<static bool>)`` so a guard like
    # ``if bool(True):`` folds to the taken arm with no branch -- otherwise the return in the dead arm is wrongly
    # rejected as a return-inside-a-branch.
    def f(x):  # type: ignore[no-untyped-def]
        if bool(True):
            return 1.0
        return x  # unreachable; must not force a branch nor a return-in-branch rejection

    hir = lower(f)
    assert len(hir.blocks) == 1  # folded: no branch


def test_static_bool_cast_short_circuits_a_dead_non_boolean_operand() -> None:
    # ``bool(False) and x`` short-circuits to False exactly like ``False and x``; the dead float operand is not
    # evaluated and must not be rejected (the static evaluator folds the bool() cast of a static bool).
    def f(x):  # type: ignore[no-untyped-def]
        return 1.0 if (bool(False) and x) else 0.0

    hir = lower(f)
    assert len(hir.blocks) == 1
    assert _op_count(hir, FloatToBool) == 0  # the whole guard folds to False; no cast survives


def test_static_float_sees_through_float_cast_of_a_bool() -> None:
    # ``float(True) > 0.5`` folds: float(<static bool>) is 1.0/0.0, so the comparison and the ternary fold statically.
    def f(x):  # type: ignore[no-untyped-def]
        return x if float(True) > 0.5 else 0.0

    hir = lower(f)
    assert len(hir.blocks) == 1  # the ternary's static test selects one arm with no branch
    assert _op_count(hir, BoolToFloat) == 0


def test_or_true_in_a_condition_folds_and_permits_a_return() -> None:
    # Regression (user): ``X or True`` (X a valid boolean) is the constant True, so the guard must fold to its taken
    # arm with no branch -- including allowing the return in that arm, which a runtime branch would reject.
    def f(x):  # type: ignore[no-untyped-def]
        if x > 0.0 or True:
            return 1.0
        return x  # unreachable

    hir = lower(f)
    assert len(hir.blocks) == 1  # the guard folded; no branch, so the return is not inside a branch
    assert _op_count(optimize(hir), FloatRelational) == 0  # the dead ``x > 0.0`` is dead-code-eliminated


def test_and_false_in_a_condition_folds_to_the_else_arm() -> None:
    def f(x):  # type: ignore[no-untyped-def]
        if x > 0.0 and False:
            y = 1.0
        else:
            y = 2.0
        return y

    hir = lower(f)
    assert len(hir.blocks) == 1
    assert _op_count(hir, BoolAnd) == 0


def test_chained_comparison_with_a_static_true_link_collapses_the_dead_and() -> None:
    # ``0.0 < 1.0 < x`` is ``(0 < 1) and (1 < x)``; the static-true link folds, and the constant folder's identity
    # element collapses ``True and (1 < x)`` to just ``1 < x`` -- no residual dead AND.
    def f(x):  # type: ignore[no-untyped-def]
        return 1.0 if 0.0 < 1.0 < x else 0.0

    hir = optimize(lower(f))
    assert _op_count(hir, BoolAnd) == 0
    assert _op_count(hir, FloatRelational) == 1  # only the dynamic ``1.0 < x`` survives


def test_statically_false_while_still_type_checks_its_condition() -> None:
    # Regression (review): a statically-false ``while`` is skipped, but its condition must still be type-checked --
    # ``while x and False:`` with a non-boolean x must be rejected (symmetric with ``if x and False:``), not silently
    # accepted because the loop never runs.
    def f(x):  # type: ignore[no-untyped-def]
        while x and False:
            x = x + 1.0
        return x

    with pytest.raises(UnsupportedConstruct, match="boolean"):
        lower(f)


def test_statically_false_while_with_a_boolean_condition_is_skipped() -> None:
    def f(x):  # type: ignore[no-untyped-def]
        while x > 0.0 and False:  # a valid boolean condition that is statically false: the loop never runs
            x = x + 1.0
        return x

    assert len(lower(f).blocks) == 1  # the loop body is not lowered


def test_reachability_folds_through_a_bool_cast_of_a_connective() -> None:
    # ``bool(X or True)`` carries the truthiness of ``X or True`` (= True), so the guard folds and the return is allowed.
    def f(x):  # type: ignore[no-untyped-def]
        if bool(x > 0.0 or True):
            return 1.0
        return x

    assert len(lower(f).blocks) == 1  # folded; no branch, return permitted


def test_ternary_condition_with_equal_arms_folds() -> None:
    # ``True if x > 0.0 else True`` is True regardless of the (runtime) test, so the enclosing guard folds and the
    # return is not rejected as branch-nested (the inner ternary still lowers, but the outer ``if`` takes no branch).
    def f(x):  # type: ignore[no-untyped-def]
        if True if x > 0.0 else True:
            return 1.0
        return x

    lower(f)  # must not raise (the return is reachable, not inside a branch)


def test_collect_assigned_stops_at_a_returning_folded_arm() -> None:
    # Regression (review #1): a folded ``if`` whose taken arm returns makes the rest unreachable; the read-only scan
    # must stop there, so an attribute assigned only afterwards is not wrongly counted as written. Here ``gate`` is
    # read-only, so the first guard folds and its return is permitted -- which fails if ``gate`` is mismarked.
    class K:
        def __init__(self):
            self.gate = True
            self.y = 0.0

        def __call__(self, u):  # type: ignore[no-untyped-def]
            if self.gate:
                return u + 1.0
            self.y = u
            if True:
                return self.y
            self.gate = False  # unreachable; must not mark ``gate`` assigned

    assert lower(K().__call__).state_slots == []  # gate read-only -> everything folds; no rejection, no state


def test_float_cast_connective_comparison_condition_folds_without_spurious_state() -> None:
    # Regression (review #2): ``float(X or True) > 0.5`` is the constant True; the guard must fold so the dead else-arm
    # write does NOT become a persistent-state slot (and output port).
    class K:
        def __init__(self):
            self.y = 0.0
            self.z = 0.0

        def __call__(self, u):  # type: ignore[no-untyped-def]
            if float(u > 0.0 or True) > 0.5:
                self.y = u
            else:
                self.z = u  # unreachable
            return self.y

    hir = lower(K().__call__)
    assert [slot.name for slot in hir.state_slots] == ["y"]  # z is not spurious state
    assert len(hir.blocks) == 1


def test_absorbing_attribute_connective_keeps_a_dead_arm_attribute_read_only() -> None:
    # Regression (review #3): ``self.flag or True`` folds in the read-only scan (attribute opaque, absorbing operand
    # decides it), so ``self.other`` -- written only in the dead else -- stays read-only, and the later guard on it
    # folds rather than leaking ``self.z`` as state.
    class K:
        def __init__(self):
            self.flag = True
            self.other = True
            self.y = 0.0
            self.z = 0.0

        def __call__(self, u):  # type: ignore[no-untyped-def]
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
    assert [slot.name for slot in hir.state_slots] == ["y"]  # z is not spurious state


def test_equal_arm_ternary_condition_leaves_no_dead_branch() -> None:
    # Regression (review #4): a ternary whose arms agree is that value with no branch, so a statically-false loop
    # guarded by one is skipped cleanly (no dead diamond left in the CFG).
    def f(x):  # type: ignore[no-untyped-def]
        while False if x > 0.0 else False:
            x = x + 1.0
        return x

    assert len(lower(f).blocks) == 1


def test_equal_arm_ternary_value_fold_does_not_bypass_operand_type_checks() -> None:
    # Regression (review, miscompile): the equal-arm ternary VALUE fold must use the strict static evaluator, not the
    # reachability one. ``(float(x or True) > 0.5) if c else (float(x or True) > 0.5)`` has equal arms, but folding it
    # without lowering would skip type-checking ``x or True`` -- accepting a non-boolean x and miscompiling. It must
    # be rejected, exactly as the un-wrapped ``float(x or True) > 0.5`` is.
    def f(x, c):  # type: ignore[no-untyped-def]
        return 1.0 if ((float(x or True) > 0.5) if c > 0.0 else (float(x or True) > 0.5)) else 0.0

    with pytest.raises(UnsupportedConstruct, match="boolean"):
        lower(f)


def test_read_only_scan_does_not_misfold_a_reassigned_for_counter() -> None:
    # Regression (review, miscompile): the read-only scan must not bind a static ``for`` counter and then fold a
    # counter-dependent condition against a STALE value -- which would drop ``_flag`` from the assigned set, wrongly
    # treat it as read-only, and fold the later ``if self._flag:`` to a fixed arm, diverging from lowering. The scan
    # leaves the counter unbound (conservative), so the body's writes are recorded and ``_flag`` stays state.
    class K:
        def __init__(self):
            self._flag = False
            self.y = 0.0
            self.z = 0.0

        def __call__(self, u):  # type: ignore[no-untyped-def]
            for i in range(1):
                i = u  # the loop counter is reassigned to a runtime value
                if i > 0.0:
                    self._flag = True
            if self._flag:
                self.z = u
            else:
                self.y = u
            return self.y

    slots = {slot.name for slot in lower(K().__call__).state_slots}
    assert "_flag" in slots and "z" in slots  # the guard stays dynamic; neither arm is wrongly dropped


def test_ternary_with_mismatched_scalar_arm_types_is_cleanly_rejected() -> None:
    # Regression (review): a conditional whose arms have different scalar types (a boolean and a float) is out of
    # subset; it must be rejected with a clear UnsupportedConstruct, not leak an internal phi type-mismatch error.
    def f(x, c):  # type: ignore[no-untyped-def]
        return 1.0 if (False if c > 0.0 else x) else 0.0

    with pytest.raises(UnsupportedConstruct, match="different scalar types"):
        lower(f)
