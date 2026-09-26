"""
The rejection suite: one located refusal per banned family — the specification's teeth. Every case asserts the
diagnostic class and the presence of a source location.
"""

import functools
from collections.abc import Callable

import pytest

import holoso
from holoso import UnsupportedConstruct
from holoso._eel import lower

from ._modelref import DEFAULT_UNROLL_MAX_TRIPS

_OPTIONS = holoso.Options(holoso.OperatorOptions())


def _reject(target: object) -> None:
    assert callable(target)
    with pytest.raises(UnsupportedConstruct) as info:
        lower(target, DEFAULT_UNROLL_MAX_TRIPS)
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


def _k_is(x: object, y: object) -> bool:
    return x is y


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


_CASES: list[object] = [
    _k_del,
    _k_global,
    _k_nonlocal(),
    _k_import,
    _k_lambda,
    _k_nested_def,
    _k_nested_class,
    _k_dict,
    _k_set,
    _k_dict_comp,
    _k_set_comp,
    _k_genexp,
    _k_comp_filter,
    _k_comp_multi,
    _k_comp_unpack,
    _k_yield,
    _k_async,
    _k_try,
    _k_with,
    _k_match,
    _k_type_alias,
    _k_while_else,
    _k_for_else,
    _k_starred_target,
    _k_slice_store,
    _k_slice_step,
    _k_tuple_slice_store,
    _k_double_splat,
    _k_varargs,
    _k_kwargs,
    _k_is,
    _k_in,
    _k_mangled_local,
    _k_mangled_trailing,
    _Mangled().step,
    _k_string,
    _k_none,
    _k_fstring,
    _k_bare_call,  # a bare call desugars; the callee is what refuses
    _k_bare_raise,
    _k_raise_from,
    _k_raise_no_call,
    _k_raise_conversion,
    _k_raise_spec,
    _k_walrus_ifexp,
    _k_walrus_gate,
    _k_walrus_chain,
    _k_walrus_comp,
    _k_walrus_read,
    _k_walrus_aug,
    _k_walrus_index_store,
    _k_walrus_index_aug,
    _k_walrus_while,
    _k_wrapped,
    _k_assert_yield,
    _k_generic,
    _k_bare_ann_subscript,
    _MangledParam().kernel,
    _k_lambda_target,
    _MangledRaise().kernel,
    _k_stub_body,
]


# ------------------------------------------------------------------ partial-evaluation-level families


def _k_zip(a: float, b: float) -> float:
    acc = 0.0
    for p in zip((a, b), (b, a)):
        acc = acc + p[0]
    return acc


def _k_dynamic_index_read(i: int, x: float) -> float:
    t = (x, x + 1.0)
    return t[i]


def _k_dynamic_index_store(i: int, x: float) -> float:
    t = [0.0, 0.0]
    t[i] = x
    return t[0]


def _k_aggregate_truthiness(a: float, b: float) -> float:
    t = (a, b)
    if t:
        return 1.0
    return 0.0


def _k_aggregate_equality(a: float, b: float) -> bool:
    t = (a, b)
    u = (b, a)
    return t == u


def _k_bool_arithmetic(a: bool, x: float) -> float:
    return a * x


_PE_CASES: list[object] = [
    _k_zip,
    _k_dynamic_index_read,
    _k_dynamic_index_store,
    _k_aggregate_truthiness,
    _k_aggregate_equality,
    _k_bool_arithmetic,
]


@pytest.mark.parametrize("fn", _PE_CASES, ids=[getattr(fn, "__name__", "?") for fn in _PE_CASES])
def test_pe_rejection(fn: object) -> None:
    _reject(fn)


def test_lambda_rejection_points_at_the_lambda_token() -> None:
    with pytest.raises(UnsupportedConstruct) as info:
        holoso.synthesize(_k_lambda_target, _OPTIONS, name="k")
    location = info.value.location
    assert location is not None and location.line is not None
    assert location.line[location.col :].startswith("lambda")


@pytest.mark.parametrize("fn", _CASES, ids=[getattr(fn, "__name__", "?") for fn in _CASES])
def test_rejection(fn: object) -> None:
    _reject(fn)


def _returns_when_positive(x: float) -> float:
    if x > 0.0:
        return x
    raise ValueError("negative")


def _k_raise_behind_a_helper_return(x: float) -> float:
    return _returns_when_positive(x)


def _k_raise_behind_a_break(x: float) -> float:
    for i in range(3):
        if x > float(i):
            break
        raise ValueError("unreached when x > 0")
    return x


@pytest.mark.parametrize("fn", [_k_raise_behind_a_helper_return, _k_raise_behind_a_break])
def test_a_raise_behind_a_pending_exit_is_data_dependent(fn: Callable[..., object]) -> None:
    """It runs only where an earlier return or break did not fire, so it is no compile-time diagnostic."""
    with pytest.raises(UnsupportedConstruct, match="data-dependent path"):
        lower(fn, DEFAULT_UNROLL_MAX_TRIPS)
