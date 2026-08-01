"""
The rejection suite: one located diagnostic pin per banned family — the specification's teeth. Every case
asserts the diagnostic class, a message fragment, and the presence of a source location.
"""

import functools
import inspect
import types
from collections.abc import Callable

import pytest

from holoso._eel._desugar import desugar
from holoso._errors import UnsupportedConstruct


def _reject(target: object, fragment: str) -> None:
    fn = target.__func__ if inspect.ismethod(target) else target
    assert isinstance(fn, types.FunctionType)
    with pytest.raises(UnsupportedConstruct, match=fragment) as info:
        desugar(fn)
    assert info.value.location is not None
    assert info.value.location.lineno > 0


def _k_del(x: float) -> float:
    del x
    return 0.0


def _k_global() -> None:
    global _MODULE_LEVEL  # noqa: PLW0603
    _MODULE_LEVEL = 1.0


_MODULE_LEVEL = 0.0


def _k_nonlocal() -> Callable[[], float]:
    v = 1.0

    def inner() -> float:
        nonlocal v
        return v

    return inner


def _k_import(x: float) -> float:
    import math  # noqa: F401

    return x


def _k_lambda(x: float) -> float:
    f: Callable[[float], float] = lambda v: v * 2.0  # noqa: E731
    return f(x)


def _k_nested_def(x: float) -> float:
    def helper(v: float) -> float:
        return v

    return helper(x)


def _k_nested_class(x: float) -> float:
    class C:
        pass

    return x


def _k_dict(x: float) -> float:
    d = {"a": x}
    return d["a"]


def _k_set(x: float) -> float:
    s = {x}
    return x if x in s else 0.0


def _k_dict_comp(vs: tuple[float, ...]) -> float:
    return {v: v for v in vs}[vs[0]]


def _k_set_comp(vs: tuple[float, ...]) -> float:
    return len({v for v in vs}) * 1.0


def _k_genexp(vs: tuple[float, ...]) -> float:
    return sum(v for v in vs)


def _k_comp_filter(vs: tuple[float, ...]) -> list[float]:
    return [v for v in vs if v > 0.0]


def _k_comp_multi(vss: tuple[tuple[float, ...], ...]) -> list[float]:
    return [v for vs in vss for v in vs]


def _k_comp_unpack(ps: tuple[tuple[float, float], ...]) -> list[float]:
    return [a + b for a, b in ps]


def _k_yield(x: float) -> object:
    yield x


async def _k_async(x: float) -> float:
    return x


def _k_try(x: float) -> float:
    try:
        return x
    except ValueError:
        return 0.0


def _k_with(x: float) -> float:
    with open("/dev/null") as f:  # noqa: F841
        return x


def _k_match(x: float) -> float:
    match x:
        case _:
            return x


def _k_type_alias(x: float) -> float:
    type Alias = float  # noqa: F841
    return x


def _k_while_else(x: float) -> float:
    while x > 0.0:
        x = x - 1.0
    else:
        x = -1.0
    return x


def _k_for_else(vs: tuple[float, ...]) -> float:
    acc = 0.0
    for v in vs:
        acc = acc + v
    else:
        acc = acc + 1.0
    return acc


def _k_starred_target(vs: tuple[float, ...]) -> float:
    a, *rest = vs  # noqa: F841
    return a


def _k_slice_store(t: list[float]) -> None:
    t[0:2] = [1.0, 2.0]


def _k_slice_step(t: list[float]) -> list[float]:
    return t[::2]


def _k_tuple_slice_store(m: object, v: float) -> None:
    m[0, 0:2] = v  # type: ignore[index]


_SPLAT_KWARGS = {"v": 1.0}


def _k_double_splat(x: float) -> float:
    return abs(**_SPLAT_KWARGS) + x  # type: ignore[call-arg]


def _k_varargs(*xs: float) -> float:
    return xs[0]


def _k_kwargs(**kw: float) -> float:
    return kw["x"]


def _k_for_unpack(ps: tuple[tuple[float, float], ...]) -> float:
    acc = 0.0
    for a, b in ps:
        acc = acc + a + b
    return acc


def _k_is(x: object) -> bool:
    return x is None


def _k_in(x: float, vs: tuple[float, ...]) -> bool:
    return x in vs


def _k_mangled_local(x: float) -> float:
    __y = x
    return __y


def _k_mangled_trailing(x: float) -> float:
    __y_ = x
    return __y_


class _Mangled:
    def step(self, x: float) -> float:
        return self.__v * x  # type: ignore[attr-defined, no-any-return]


def _k_string(x: float) -> str:
    s = "text"
    return s


def _k_none(x: float) -> object:
    v = None
    return v


def _k_fstring(x: float) -> str:
    return f"{x}"


def _k_bare_call(x: float) -> float:
    print(x)
    return x


def _k_bare_raise(x: float) -> float:
    raise


def _k_raise_from(x: float) -> float:
    raise ValueError("v") from None


def _k_raise_no_call(x: float) -> float:
    raise ValueError


def _k_raise_conversion(x: float) -> float:
    raise ValueError(f"{x!r}")


def _k_raise_spec(x: float) -> float:
    raise ValueError(f"{x:.2f}")


def _k_walrus_ifexp(c: bool, x: float) -> float:
    return (w := x) if c else 0.0  # noqa: F841


def _k_walrus_gate(a: bool, x: float) -> bool:
    return a and (w := x > 0.0)  # noqa: F841


def _k_walrus_chain(a: float, b: float) -> bool:
    return a < b < (w := a + b)  # noqa: F841


def _k_walrus_comp(vs: tuple[float, ...]) -> list[float]:
    return [(w := v) * 2.0 for v in vs]  # noqa: F841


def _k_walrus_read(t: float) -> float:
    return t + (t := 2.0)


def _k_walrus_aug(x: float) -> float:
    x += (x := 2.0)
    return x


def _k_walrus_index_store(t: list[float], i: int) -> None:
    t[i] = (i := 0)


def _k_walrus_index_aug(t: list[float], i: int) -> None:
    t[i] += (i := 0)


def _k_walrus_while(x: float) -> float:
    d = 1.0
    while (d := d + x) < 10.0 and d > 0.0:
        pass
    return d


def _wrapping(fn: Callable[[float], float]) -> Callable[[float], float]:
    @functools.wraps(fn)
    def wrapper(x: float) -> float:
        return fn(x) + 1.0

    return wrapper


@_wrapping
def _k_wrapped(x: float) -> float:
    return x * 2.0


def _k_assert_yield() -> object:
    assert (yield 1)  # a yield anywhere, even in an ignored assert, makes this a generator in CPython
    return 2


def _k_generic[T](x: float) -> float:
    return x


def _k_bare_ann_subscript(t: list[float], i: int) -> float:
    t[i]: float  # type: ignore[misc]  # CPython evaluates the target expression; dropping it would diverge
    return t[i]


class _MangledParam:
    def kernel(self, __gain: float) -> float:
        return __gain


_k_lambda_target = lambda x: x * 2.0  # noqa: E731


class _MangledRaise:
    def kernel(self, x: float) -> float:
        if x < 0.0:
            raise __Error("boom")  # type: ignore[name-defined]  # noqa: F821
        return x


def _k_stub_body(x: float) -> float: ...  # type: ignore[empty-body]


_CASES: list[tuple[object, str]] = [
    (_k_del, "unsupported statement: Delete"),
    (_k_global, "unsupported statement: Global"),
    (_k_nonlocal(), "unsupported statement: Nonlocal"),
    (_k_import, "unsupported statement: Import"),
    (_k_lambda, "lambdas are not supported"),
    (_k_nested_def, "unsupported statement: FunctionDef"),
    (_k_nested_class, "unsupported statement: ClassDef"),
    (_k_dict, "dictionary displays"),
    (_k_set, "set displays"),
    (_k_dict_comp, "set/dict comprehensions"),
    (_k_set_comp, "set/dict comprehensions"),
    (_k_genexp, "generator expressions"),
    (_k_comp_filter, "comprehension `if` filters"),
    (_k_comp_multi, "multiple `for` clauses"),
    (_k_comp_unpack, "comprehension target unpacking"),
    (_k_yield, "generators are not supported"),
    (_k_async, "async functions are not supported"),
    (_k_try, "unsupported statement: Try"),
    (_k_with, "unsupported statement: With"),
    (_k_match, "unsupported statement: Match"),
    (_k_type_alias, "unsupported statement: TypeAlias"),
    (_k_while_else, "`while ... else` is not supported"),
    (_k_for_else, "`for ... else` is not supported"),
    (_k_starred_target, "starred assignment targets"),
    (_k_slice_store, "slice assignment is not supported"),
    (_k_slice_step, "slice steps are not supported"),
    (_k_tuple_slice_store, "slice assignment is not supported"),
    (_k_double_splat, r"`\*\*` call arguments"),
    (_k_varargs, r"\*args parameters"),
    (_k_kwargs, r"\*\*kwargs parameters"),
    (_k_for_unpack, "for-loop target unpacking"),
    (_k_is, "comparison operator `is`"),
    (_k_in, "comparison operator `in`"),
    (_k_mangled_local, "name mangling"),
    (_k_mangled_trailing, "name mangling"),
    (_Mangled().step, "name mangling"),
    (_k_string, "string literals are only supported as raise messages"),
    (_k_none, "`None` is only supported as a bare return value"),
    (_k_fstring, "f-strings are only supported as raise messages"),
    (_k_bare_call, "expression statement result is unused"),
    (_k_bare_raise, "bare `raise` is not supported"),
    (_k_raise_from, "`raise ... from` is not supported"),
    (_k_raise_no_call, "`raise` requires"),
    (_k_raise_conversion, "conversions and format specs"),
    (_k_raise_spec, "conversions and format specs"),
    (_k_walrus_ifexp, "conditional-expression arm"),
    (_k_walrus_gate, "operand after the first"),
    (_k_walrus_chain, "comparator after the first"),
    (_k_walrus_comp, "inside a comprehension"),
    (_k_walrus_read, "may not also be read in the same statement"),
    (_k_walrus_aug, "may not also be read in the same statement"),
    (_k_walrus_index_store, "may not also be read in the same statement"),
    (_k_walrus_index_aug, "may not also be read in the same statement"),
    (_k_walrus_while, "update the name inside the loop body"),
    (_k_wrapped, "wrapped by a decorator"),
    (_k_assert_yield, "generators are not supported"),
    (_k_generic, "generic type parameters"),
    (_k_bare_ann_subscript, "bare annotation"),
    (_MangledParam().kernel, "name mangling"),
    (_k_lambda_target, "lambdas are not supported"),
    (_MangledRaise().kernel, "name mangling"),
    (_k_stub_body, "stub body"),
]


def test_lambda_rejection_points_at_the_lambda_token() -> None:
    with pytest.raises(UnsupportedConstruct) as info:
        desugar(_k_lambda_target)  # type: ignore[arg-type]
    location = info.value.location
    assert location is not None and location.line is not None
    assert location.line[location.col :].startswith("lambda")


@pytest.mark.parametrize("fn,fragment", _CASES, ids=[getattr(fn, "__name__", "?") for fn, _ in _CASES])
def test_rejection(fn: object, fragment: str) -> None:
    _reject(fn, fragment)
