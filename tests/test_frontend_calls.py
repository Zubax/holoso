"""Frontend tests: call expansion and admission -- dispatch, library/registry, intrinsics, folds, getattr/isinstance."""

import dataclasses
import math
import textwrap
import types
from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import pytest

import holoso
from holoso import FloatFormat, UnsupportedConstruct, UnsupportedLibraryFunction
from holoso._frontend import lower
from holoso._hir import (
    BoolToFloat,
    FloatAbs,
    FloatAdd,
    FloatCos,
    FloatDiv,
    FloatExp2,
    FloatMul,
    FloatRelational,
    FloatSin,
    FloatToBool,
    Operation,
)

from ._frontend_common import _rebind_globals as _rebind_globals, _op_count as _op_count
from ._modelref import arith_count as _arith_count, default_ops


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


def test_non_operator_numpy_call_stays_unsupported() -> None:
    def f(a: float) -> float:
        return np.sum(a)  # type: ignore[no-any-return]

    with pytest.raises(UnsupportedConstruct, match="call to 'sum' is not supported in a kernel"):
        lower(f)


def test_pow_static_integer_exponent_stays_multiplication() -> None:
    # The static-integer path precedes the base-2 exp2 path, so ``2 ** 3`` still unrolls to multiplies.
    def f(x: float) -> float:
        return x * (2**3)

    hir = lower(f)
    assert _arith_count(hir, FloatExp2) == 0


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


def test_inlined_global_function() -> None:
    def f(a: float, b: float) -> list[float]:
        return _addmul(a, b)

    hir = lower(f)
    assert [o.name for o in hir.outputs] == ["out_0", "out_1"]
    assert _arith_count(hir, FloatAdd) == 1 and _arith_count(hir, FloatMul) == 1


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


def test_abs_accepts_a_star_unpacked_argument() -> None:
    def f(a: float) -> float:
        v = [a]
        return abs(*v)

    assert _arith_count(lower(f), FloatAbs) == 1


def test_method_style_abs_call_is_rejected() -> None:
    # Only a bare-name abs(...) is the builtin; a method-style a.abs(b) must not be silently treated as it (which would
    # drop the receiver) -- there is no supported scalar method, so it is an unsupported call.
    def f(a: float, b: float) -> float:
        return a.abs(b)  # type: ignore[no-any-return, attr-defined]

    with pytest.raises(UnsupportedConstruct, match="abs"):
        lower(f)


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


def test_call_dispatch_is_by_identity_not_spelling() -> None:
    # Dispatch resolves the callee object, so an aliased import lowers exactly like the canonical spelling -- the numpy
    # array factories and the cast/sequence builtins are matched by identity, not by the name written at the call.
    from jaxtyping import Float64

    def use_asarray(a: float, b: float) -> Float64[np.ndarray, "2"]:
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


def test_literal_exponent_expands_to_a_multiply_chain() -> None:
    # ``x**66`` expands to a chain of multiplies; the frontend lowers it and the result matches Python.
    def kernel(x: float) -> float:
        return x**66

    model = holoso.synthesize(kernel, default_ops(FloatFormat(11, 52)), name="x66").numerical_model.elaborate()
    for x in (1.1, 0.5):
        assert float(model.run(x)[0]) == pytest.approx(x**66, rel=1e-9)


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


def test_an_all_known_aggregate_folds_through_an_intrinsic() -> None:
    # Review round 5: an all-Known aggregate operand must fold concretely through an intrinsic exactly as a Known
    # scalar does (rejected: "a non-numeric operand reaches a numeric intrinsic").
    def kernel(x: float) -> float:
        return x + float(np.sqrt((1.0, 4.0))[1])

    model = holoso.synthesize(kernel, default_ops(FloatFormat(11, 52)), name="known_agg_fold").numerical_model
    assert float(model.elaborate().run(3.0)[0]) == 5.0


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


def test_getattr_is_a_located_rejection() -> None:
    # Trim T1 (docs/decisions/scope-ruling.md): getattr's static-name requirement made it pure spelling
    # redundancy over the dotted access, and its concrete-path history was a recurring miscompile habitat
    # (review rounds 9-10). Every admitted shape now rejects with guidance, located at the call.
    class Accumulator:
        def __init__(self) -> None:
            self.g = 1.0

        def step(self, x: float) -> float:
            self.g = self.g + x
            return getattr(self, "g")  # type: ignore[no-any-return]

    with pytest.raises(UnsupportedConstruct, match=r"step:\d+:\d+: getattr is not supported in a kernel"):
        lower(Accumulator().step)

    def with_default(x: float) -> float:
        return getattr(x, "real", 0.0)

    with pytest.raises(UnsupportedConstruct, match="spell the attribute access directly"):
        lower(with_default)


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


def test_isinstance_is_a_located_rejection() -> None:
    # Trim T4 (docs/decisions/scope-ruling.md): values are statically typed, so an honest isinstance query
    # answers itself at authoring time, while a faithful compile-time verdict demanded real machinery (member
    # provenance, classinfo resolution, record-layout folds) with a demonstrated miscompile history. Every
    # spelling -- scalar subject, record subject, inline or precomputed tuple classinfo -- refuses at the
    # dispatch with one located message.
    def scalar_subject(x: float) -> float:
        return x * 2.0 if isinstance(1.0, float) else x

    gain = 2.0

    def inline_tuple_classinfo(x: float) -> float:
        return x * 2.0 if isinstance(gain, (float, str)) else x

    precomputed = (float, int)

    def precomputed_classinfo(x: float) -> float:
        return x * 2.0 if isinstance(gain, precomputed) else x

    @dataclasses.dataclass(frozen=True)
    class Tagged:
        v: float

    def record_subject(x: float) -> float:
        t = Tagged(x * 2.0)
        return t.v if isinstance(t, Tagged) else -t.v

    for kernel in (scalar_subject, inline_tuple_classinfo, precomputed_classinfo, record_subject):
        with pytest.raises(
            UnsupportedConstruct, match=r":\d+:\d+: isinstance is not supported in a kernel: values are statically"
        ):
            lower(kernel)


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
    # __post_init__ running user code at construction, tuple.count's identity shortcut (a NaN element matches
    # itself in Python, never after a rebuild), and a PRE-BOUND builtin whose live mutable receiver was emptied
    # at compile time. Construction requires the generated __init__ with no __post_init__; sequence and str
    # methods reject; only bind-site-minted value methods are admitted.
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
        (str_format, "str methods are not supported"),
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


def test_str_methods_are_a_located_rejection() -> None:
    # Trim T6 (docs/decisions/scope-ruling.md): a str constant stays an inert value (equality, len, and
    # concatenation all fold), but its methods are host machinery a kernel does not need -- every honest use
    # precomputes the constant. The refusal is at the attribute fetch, so even the bare bound-method spelling
    # rejects; minted value methods survive for range and integer receivers.
    def method_call(x: float) -> float:
        return x * float(len("ab".upper()))

    def bound_fetch(x: float) -> float:
        pad = "ab".ljust
        return x + float(len(pad(4)))

    for kernel in (method_call, bound_fetch):
        with pytest.raises(UnsupportedConstruct, match=r":\d+:\d+: str methods are not supported in a kernel"):
            lower(kernel)

    def inert_constant(x: float) -> float:
        tag = "ab" + "c"
        return x * float(len(tag)) if tag == "abc" else -x

    model = holoso.synthesize(inert_constant, default_ops(FloatFormat(11, 52)), name="inert_str").numerical_model
    assert float(model.elaborate().run(2.0)[0]) == inert_constant(2.0) == 6.0


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
        return x + float(len((1).to_bytes(10**12)))

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


def test_enum_members_fold_to_their_base_value() -> None:
    # Trim T5 (docs/decisions/scope-ruling.md): an IntEnum/StrEnum member admits as its plain base value, full
    # stop -- no member identity survives into the domain. A kernel reading a member global computes with the
    # int value; member-specific attributes (.value, .name, an enum-defined method) no longer resolve and
    # reject with Python's own AttributeError text, located.
    import enum

    class Mode(enum.IntEnum):
        FAST = 3

        def describe(self) -> str:
            raise RuntimeError("user code ran at compile time")

    mode = Mode.FAST

    def computes(x: float) -> float:
        return x * float(mode)

    model = holoso.synthesize(computes, default_ops(FloatFormat(11, 52)), name="enum_base").numerical_model
    assert float(model.elaborate().run(2.0)[0]) == computes(2.0) == 6.0

    def member_attribute(x: float) -> float:
        return x * float(mode.value)

    def member_method(x: float) -> float:
        return x * float(len(mode.describe()))

    for kernel, missing in ((member_attribute, "value"), (member_method, "describe")):
        with pytest.raises(UnsupportedConstruct, match=f"'int' object has no attribute '{missing}'"):
            lower(kernel)


# ------------------------------ retention review round (Claude ultrathink) ------------------------------


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


# ------------------------------ retention review round (Codex gpt-5.6-sol ultra) ------------------------------


def test_attribute_snapshot_admits_once_so_referent_mutation_cannot_move_facts() -> None:
    # MISCOMPILE: the snapshot memo pinned the live LIST OBJECT but re-admitted it per consultation, so a
    # permitted module hook mutating the list mid-analysis moved the folded constant (compiled 18.0 where
    # Python computes 2.0). The memo now caches the first read's ADMITTED value.
    module = types.ModuleType("lazy_mut")
    module.table = [1.0]  # type: ignore[attr-defined]

    def module_getattr(name: str) -> float:
        if name != "trigger":
            raise AttributeError(name)
        module.table[0] = 9.0
        return 0.0

    module.__getattr__ = module_getattr  # type: ignore[method-assign]

    def kernel(x: float) -> float:
        coefficient = module.table[0]
        ignored = module.trigger
        return coefficient * x + ignored  # type: ignore[no-any-return]

    kernel.__globals__["module"] = module
    try:
        model = holoso.synthesize(kernel, default_ops(FloatFormat(11, 52)), name="snap_admit").numerical_model
        assert float(model.elaborate().run(2.0)[0]) == 2.0
    finally:
        kernel.__globals__.pop("module", None)


def test_call_argument_unpacking_flattens_static_containers() -> None:
    # f(*t) flattens before any dispatch: the starred container's children become ordinary arguments through
    # synthesized projections, so template inlining, vetted folds, and intrinsics all see a plain call. The
    # star may mix with leading/trailing positionals; a container of runtime values rides its leaves.
    def helper(a: float, b: float, c: float) -> float:
        return a + b * 10.0 + c * 100.0

    def star_call(x: float, y: float) -> float:
        t = (y, x + 1.0)
        return helper(x, *t)

    def star_leading(x: float) -> float:
        t = (x, 2.0)
        return helper(*t, 3.0)

    def star_concrete(x: float) -> float:
        return x * float(len(*(("ab",))))

    for kernel, argsets in (
        (star_call, [(1.0, 2.0)]),
        (star_leading, [(4.0,)]),
        (star_concrete, [(2.0,)]),
    ):
        model = holoso.synthesize(kernel, default_ops(FloatFormat(11, 52)), name=kernel.__name__).numerical_model
        elaborated = model.elaborate()
        for argset in argsets:
            assert float(elaborated.run(*argset)[0]) == kernel(*argset)

    def star_scalar(x: float) -> float:
        return helper(*x, 1.0, 2.0)  # type: ignore[misc,call-arg]

    with pytest.raises(UnsupportedConstruct, match="argument unpacking requires a tuple, list, or array"):
        lower(star_scalar)

    def double_star(x: float) -> float:
        return helper(**dict(a=x), b=1.0, c=2.0)

    with pytest.raises(UnsupportedConstruct, match="dictionary argument unpacking"):
        lower(double_star)


def test_reduction_stub_misuse_names_the_reduction_not_the_matrix_product() -> None:
    # Regression (E4): np.max(m, axis) reported the matrix-product diagnostic from the shared operand gate.
    from jaxtyping import Float64

    def with_axis(m: Float64[np.ndarray, "2 3"]) -> float:
        return float(np.max(m, 0))

    with pytest.raises(UnsupportedConstruct, match="default axis") as excinfo:
        lower(with_axis)
    assert "matrix product" not in str(excinfo.value)

    def scalar_operand(x: float) -> float:
        return float(np.mean(x))

    with pytest.raises(UnsupportedConstruct, match="np.mean requires array operands"):
        lower(scalar_operand)


_ZERO_D_DEFAULT = np.array(0.5)


def _helper_with_zero_d_default(a: float, b: object = _ZERO_D_DEFAULT) -> float:
    return a + 1.0


def test_a_zero_d_helper_default_is_rejected_at_the_binding() -> None:
    # S2.8 review (both agents): an inlined helper's OMITTED 0-d default bound as a silent reference (or, when
    # observed, a wrong-reason crossing message); the default binding is a creation door like every other.
    def kernel(x: float) -> float:
        return x + _helper_with_zero_d_default(x)

    with pytest.raises(UnsupportedConstruct, match=r":\d+:\d+: a 0-dimensional array is not supported"):
        lower(kernel)


def test_binary_linalg_stubs_take_exact_positional_arguments() -> None:
    # S2.8 review: matmul kwargs lowered where numpy raises (ufunc positional-only), np.outer kwargs leaked the
    # stub's internal parameter names, and np.trace(m, offset) drew the array-semantics message for the offset.
    from jaxtyping import Float64

    def matmul_kwargs(m: Float64[np.ndarray, "2 2"]) -> float:
        return float(np.matmul(a=m, b=m)[0][0])  # type: ignore[call-overload]

    def trace_offset(m: Float64[np.ndarray, "2 2"]) -> float:
        return float(np.trace(m, 1))

    for kernel in (matmul_kwargs, trace_offset):
        with pytest.raises(UnsupportedConstruct, match=r"takes exactly \d positional array argument"):
            lower(kernel)
