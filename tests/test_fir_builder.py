"""Black-box tests for the FIR builder: golden structure, evaluation-order pins, and located rejections."""

import numpy as np
import pytest

from holoso._frontend._fir._build import BuildRejection, build_unit
from holoso._frontend._fir._value import MetaInt
from holoso._frontend._fir._ir import (
    Branch,
    Fail,
    FunctionUnit,
    LoadConst,
    LoadPlace,
    LoadRef,
    Local,
    PySelect,
    ReturnPlace,
    PyStoreAttr,
    PySubscript,
    StaticFor,
    StorePlace,
    UnitExit,
    pretty,
)

_GAIN = 2.5
_TABLE = np.array([1.0, 2.0])


def _ops(unit: FunctionUnit) -> list[object]:
    return [op for block in unit.blocks.values() for op in block.ops]


def _terminators(unit: FunctionUnit) -> list[object]:
    return [block.terminator for block in unit.blocks.values()]


def test_swap_reads_all_sources_before_any_write() -> None:
    def fn(a: float, b: float) -> float:
        a, b = b, a
        return a - b

    printed = pretty(build_unit(fn))
    first_write = printed.index("store a.")
    assert printed.index("py.tuple") < first_write  # the RHS bundle is fully built before either target is written
    assert printed.index("py.subscript") < first_write

    def chained(x: float) -> float:
        y = z = x + 1.0  # noqa: F841
        return y

    chained_print = pretty(build_unit(chained))
    assert chained_print.index("store y.") < chained_print.index("store z.")  # chained targets go left to right


def test_ordered_stores_leave_the_last_write() -> None:
    def fn(x: float) -> float:
        x, x = x + 1.0, x + 2.0  # noqa: B020
        return x

    unit = build_unit(fn)
    ops = _ops(unit)
    projections = {op.dst: op for op in ops if isinstance(op, PySubscript)}
    constants = {op.dst: op.value for op in ops if isinstance(op, LoadConst)}
    x_stores = [op for op in ops if isinstance(op, StorePlace) and isinstance(op.place, Local)]
    element_indices: list[int] = []
    for store in x_stores:
        index = constants[projections[store.src].index]
        assert isinstance(index, MetaInt)
        element_indices.append(index.value)
    # Both writes land left to right onto ONE binding, and the return reads that same binding — so the surviving
    # value of x is the tuple's element 1 (x + 2.0), and a builder that mints a second same-named local (leaving
    # the return on the stale binding) fails here rather than passing on names alone.
    assert element_indices == [0, 1]
    assert len({store.place for store in x_stores}) == 1
    loads = {op.dst: op for op in ops if isinstance(op, LoadPlace)}
    return_reads = [
        loads[op.src].place
        for op in ops
        if isinstance(op, StorePlace) and isinstance(op.place, ReturnPlace) and op.src in loads
    ]
    assert return_reads == [x_stores[0].place]


def test_eager_boolean_operators_lower_to_selects() -> None:
    def fn(a: float, b: float) -> float:
        return a and b

    unit = build_unit(fn)
    selects = [op for op in _ops(unit) if isinstance(op, PySelect)]
    assert len(selects) == 1  # both operands evaluated, combined by select: pinned eager semantics

    def chain(a: float, b: float, c: float) -> bool:
        return a < b < c

    chain_unit = build_unit(chain)
    assert sum(isinstance(op, PySelect) for op in _ops(chain_unit)) == 1  # two links, one AND-combine


def test_walrus_is_rejected_only_in_short_circuit_positions() -> None:
    def ok(x: float) -> float:
        total = 0.0
        while (n := total) < x:
            total = n + 1.0
        return total

    assert isinstance(build_unit(ok), FunctionUnit)  # walrus in a while condition is supported

    def bad_and(x: float) -> float:
        return x and (y := x)  # noqa: F841

    with pytest.raises(BuildRejection, match="short-circuit position"):
        build_unit(bad_and)

    def bad_ifexp(x: float) -> float:
        return (y := x) if x > 0.0 else 0.0  # noqa: F841

    with pytest.raises(BuildRejection, match="conditional-expression arm"):
        build_unit(bad_ifexp)

    def bad_chain(x: float) -> bool:
        return 0.0 < x < (y := x + 1.0)  # noqa: F841

    with pytest.raises(BuildRejection, match="chained-comparison tail"):
        build_unit(bad_chain)


def test_conditional_expression_lowers_to_real_branches() -> None:
    def fn(x: float) -> float:
        return x if x > 0.0 else -x

    unit = build_unit(fn)
    assert sum(isinstance(t, Branch) for t in _terminators(unit)) == 1
    assert not any(isinstance(op, PySelect) for op in _ops(unit))


def test_comprehension_target_is_isolated_and_iterable_is_enclosing() -> None:
    def fn(x: float) -> float:
        values = [x + k for k in (1.0, 2.0)]
        return values[0] + x  # this x must be the parameter, not any comprehension binding

    unit = build_unit(fn)
    printed = pretty(unit)
    assert sum(isinstance(t, StaticFor) for t in _terminators(unit)) == 1
    assert printed.count("load x.0") == 2  # the element expression and the final read both see the parameter

    def shadowing(x: float) -> float:
        inner = [x * x for x in (1.0, 2.0)]
        return inner[0] + x

    shadow_print = pretty(build_unit(shadowing))
    assert "static_for x." in shadow_print
    assert "load x.0" in shadow_print  # the trailing read is the parameter again: the target never leaks


def test_nested_comprehension_generators_share_the_scope() -> None:
    def fn() -> list[float]:
        return [a + b for a in (1.0, 2.0) for b in (a, 3.0)]

    unit = build_unit(fn)
    assert sum(isinstance(t, StaticFor) for t in _terminators(unit)) == 2


def test_early_and_implicit_returns_share_the_canonical_exit() -> None:
    def fn(x: float) -> float | None:  # type: ignore[return]  # the implicit fall-off IS the test
        if x > 0.0:
            return x
        x = x + 1.0  # noqa: F841

    unit = build_unit(fn)
    assert sum(isinstance(t, UnitExit) for t in _terminators(unit)) == 1
    return_stores = [op for op in _ops(unit) if isinstance(op, StorePlace) and str(op.place) == "return"]
    assert len(return_stores) == 2  # the early return and the implicit None both write the hidden place


def test_state_attribute_writes_lower_to_store_attr() -> None:
    class Component:
        def __init__(self) -> None:
            self.state = 0.0

        def step(self, x: float) -> float:
            self.state = self.state + x
            return self.state

    component = Component()
    unit = build_unit(component.step)
    assert unit.bound_self is component
    assert sum(isinstance(op, PyStoreAttr) for op in _ops(unit)) == 1


def test_module_constants_resolve_to_admitted_values_at_build_time() -> None:
    def fn(x: float) -> float:
        return x * _GAIN + float(_TABLE[0])

    unit = build_unit(fn)
    consts = [op.value for op in _ops(unit) if isinstance(op, LoadConst)]
    rendered = [str(value) for value in consts]
    assert any("2.5" in text for text in rendered)  # the global's value is snapshot into the IR
    referents = [op.obj for op in _ops(unit) if isinstance(op, LoadRef)]
    assert any(obj is float for obj in referents)  # the builtin resolves too, as a reference


def test_missing_name_fails_lazily_not_at_build_time() -> None:
    def fn(x: float) -> float:
        if x > 0.0:
            return UNDEFINED_TUNING  # type: ignore[name-defined,no-any-return]  # noqa: F821
        return x

    unit = build_unit(fn)  # builds fine: the dead-branch doctrine
    fails = [t for t in _terminators(unit) if isinstance(t, Fail)]
    assert len(fails) == 1 and "UNDEFINED_TUNING" in fails[0].message


def test_break_and_continue_target_the_loop_blocks() -> None:
    def fn(x: float) -> float:
        total = 0.0
        while total < x:
            total = total + 1.0
            if total > 10.0:
                break
            if total > 5.0:
                continue
            total = total + 0.5
        return total

    assert isinstance(build_unit(fn), FunctionUnit)


def test_rejections_are_located() -> None:
    def nested(x: float) -> float:
        def helper(v: float) -> float:
            return v

        return helper(x)

    with pytest.raises(BuildRejection, match="nested function") as info:
        build_unit(nested)
    assert info.value.origin[0].line > 0 and "nested" in info.value.origin[0].function

    def subscript_store(x: float) -> float:
        table = [1.0, 2.0]
        table[0] = x
        return table[0]

    with pytest.raises(BuildRejection, match="immutable"):
        build_unit(subscript_store)

    def identity_compare(x: float) -> bool:
        return x is None

    with pytest.raises(BuildRejection, match="Is"):
        build_unit(identity_compare)

    def loop_else(x: float) -> float:
        while x > 0.0:
            x = x - 1.0
        else:
            x = 0.0
        return x

    with pytest.raises(BuildRejection, match="while-else"):
        build_unit(loop_else)

    def lambda_kernel(x: float) -> float:
        f = lambda v: v + 1.0  # noqa: E731
        return f(x)  # type: ignore[no-any-return,no-untyped-call]

    with pytest.raises(BuildRejection, match="lambda"):
        build_unit(lambda_kernel)


def test_variadic_parameters_are_rejected() -> None:
    def fn(*values: float) -> float:
        return values[0]

    with pytest.raises(BuildRejection, match="variadic parameters"):
        build_unit(fn)


def test_printer_is_deterministic() -> None:
    def fn(x: float) -> float:
        return x + 1.0

    assert pretty(build_unit(fn)) == pretty(build_unit(fn))


def test_private_attribute_and_local_spellings_mangle() -> None:
    class Filter:
        def __init__(self) -> None:
            self.__state = 0.0

        def step(self, x: float) -> float:
            self.__state = self.__state + x
            __t = self.__state
            if x > 0.0:
                __t = __t + 1.0
            return __t

    printed = pretty(build_unit(Filter().step))
    assert "_Filter__state" in printed and ".__state" not in printed  # the live object holds only the mangled key
    assert "_Filter__t." in printed and " __t." not in printed  # loads and stores agree on one binding spelling


def test_wraps_decorated_kernels_build_the_wrapped_function() -> None:
    import functools
    from collections.abc import Callable

    def logged(fn: Callable[[float], float]) -> Callable[[float], float]:
        @functools.wraps(fn)
        def wrapper(*args: float) -> float:
            return fn(*args)

        return wrapper

    @logged
    def kernel(x: float) -> float:
        t = x + 1.0
        return t * 2.0

    # A wrapper may add behavior, so it is never silently unwrapped: the variadic wrapper rejects on its own
    # parameters with a located message instead of building either function's semantics wrongly.
    with pytest.raises(BuildRejection, match="variadic parameters"):
        build_unit(kernel)


def test_unpack_arity_is_guarded() -> None:
    def fn(x: float) -> float:
        a, b = x + 1.0, x + 2.0, x + 3.0  # type: ignore[misc]  # the honest arity mistake Python refuses
        return a + b  # type: ignore[has-type,no-any-return]

    unit = build_unit(fn)
    fails = [t for t in _terminators(unit) if isinstance(t, Fail)]
    assert fails and "unpack" in fails[0].message  # the guard folds at analysis and fires exactly when Python would


def test_later_comprehension_targets_scope_the_whole_comprehension() -> None:
    def fn(bound: float) -> list[float]:
        return [bound for sample in (1.0,) if bound > 0.0 for bound in (2.0, 3.0)]  # noqa: F821

    printed = pretty(build_unit(fn))
    assert "load bound.0" not in printed  # Python scopes ALL targets comprehension-wide: the filter's read is the
    # comprehension's own binding (unbound at that point, as Python's UnboundLocalError attests), never the parameter


def test_augmented_attribute_assignment_evaluates_the_object_once() -> None:
    class Node:
        def __init__(self) -> None:
            self.value = 0.0

        def bump(self, x: float) -> float:
            self.value += x
            return self.value

    printed = pretty(build_unit(Node().bump))
    assert printed.count("load self.0") == 2  # once for the augmented store, once for the return read


def _helper(v: float) -> float:
    return v


def test_printer_renders_object_refs_address_free() -> None:
    def fn(x: float) -> float:
        return _helper(x)

    printed = pretty(build_unit(fn))
    assert "0x" not in printed  # a memory address would change per process, breaking cross-process golden tests
    assert "ref _helper" in printed


def test_keyword_only_parameters_are_ordinary_parameters() -> None:
    def fn(x: float, *, dt: float) -> float:  # the bundled Ekf1 example uses this exact shape
        return x * dt

    unit = build_unit(fn)
    assert [str(p) for p in unit.params] == ["x.0", "dt.1"]


def test_state_leaf_identity_semantics() -> None:
    import dataclasses

    from holoso._frontend._fir._ir import StateLeaf

    @dataclasses.dataclass
    class Component:
        gain: float = 1.0

    a, b = Component(), Component()
    assert StateLeaf(a, ("gain",)) != StateLeaf(b, ("gain",))  # equal-valued distinct owners stay distinct
    assert StateLeaf(a, ("gain",)) == StateLeaf(a, ("gain",))
    assert len({StateLeaf(a, ("gain",)), StateLeaf(a, ("gain",)), StateLeaf(b, ("gain",))}) == 2  # hash is total


def test_bare_annotation_evaluates_its_receiver() -> None:
    def fn(x: float) -> float:
        UNBOUND.component: float  # type: ignore[misc,name-defined]  # noqa: F821  # Python raises NameError here
        return x

    unit = build_unit(fn)
    fails = [t for t in _terminators(unit) if isinstance(t, Fail)]
    assert fails and "UNBOUND" in fails[0].message


def test_unbound_closure_cells_fail_lazily() -> None:
    def outer() -> object:
        if False:
            calibration = 1.0  # noqa: F841

        def kernel(x: float) -> float:
            if x < 0.0:
                return calibration * x  # type: ignore[has-type,no-any-return]
            return x

        return kernel

    unit = build_unit(outer())  # builds: Python only raises if the branch executes
    fails = [t for t in _terminators(unit) if isinstance(t, Fail)]
    assert fails and "calibration" in fails[0].message


def test_augmented_assignment_is_marked_in_place() -> None:
    def aug(a: float, b: float) -> float:
        a += b
        return a

    def rebind(a: float, b: float) -> float:
        a = a + b
        return a

    assert "py.bin[+=]" in pretty(build_unit(aug))  # += mutates aliases of a mutable aggregate; rebinding does not
    assert "py.bin[+=]" not in pretty(build_unit(rebind))


def test_comprehension_targets_clear_on_each_scope_entry() -> None:
    from holoso._frontend._fir._ir import UnbindPlace

    def fn(x: float) -> list[list[float]]:
        rows: list[list[float]] = []
        for _ in (1.0, 2.0):
            rows = rows + [[v * x for v in (1.0, 2.0)]]
        return rows

    unit = build_unit(fn)
    clears = [op for op in _ops(unit) if isinstance(op, UnbindPlace) and not op.checked]
    assert clears and any("v." in str(op.place) for op in clears)  # each execution starts its targets unbound
