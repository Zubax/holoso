"""
The corpus differential in two layers. The HIR oracle (CPython against ``HirEvaluator``) runs the float kernels --
whose binary64-exact comparison has no black-box spelling -- and ONLY the five UART integer cases, whose written FSM
slots are private: those rows check exact private-slot changes, parameter/port mirroring, output-name uniqueness and
private-slot non-exposure, none of which a public port can observe. The remaining integer corpus is accepted
publicly: ``synthesize`` and the numerical model against a fresh CPython reference, mapped by port NAME -- never
positionally, because an assign-and-return leaf may be elided onto its ``state_*`` port -- with a typed
``(kind, value)`` comparison, since Python's ``==`` conflates ``True``, ``1`` and ``1.0``.
"""

import inspect
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

import holoso
from holoso import BoolType, FloatType, IntType
from holoso._eel import lower
from holoso._value import FloatValue, IntValue

from ._modelref import DEFAULT_UNROLL_MAX_TRIPS, flatten_value, port_name
from ._eel_corpus import INT_CASES, band_scan, convergence_steps, int_corpus_options, rows
from ._eeloracle import InputRow, assert_hir_matches_reference, instance_leaves

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from biquad import Biquad  # noqa: E402
from fir import Fir4  # noqa: E402

_UART_CASES = [case for case in INT_CASES if case[0].startswith("int_uart")]
assert len(_UART_CASES) == 5

_FLOAT_CASES: list[tuple[str, Callable[[], Callable[..., object]], list[InputRow]]] = [
    ("fir4", lambda: Fir4().__call__, rows("x", [1.0, 2.0, -1.0, 0.5, 3.0, -2.5, 0.0])),
    ("biquad", lambda: Biquad().__call__, rows("x", [1.0, 0.0, 0.0, 2.0, -1.0, 0.5, 0.0])),
    (
        "convergence_steps",
        lambda: convergence_steps,
        [{"x": 100.0, "tol": 1.0}, {"x": 3.0, "tol": 8.0}, {"x": 500.0, "tol": 0.0}, {"x": -1.0, "tol": 0.5}],
    ),
    (
        "band_scan",
        lambda: band_scan,
        [{"x": 20.0, "floor": 0.5}, {"x": 1.5, "floor": 0.0}, {"x": 0.25, "floor": 1.0}, {"x": 2.0, "floor": -1.0}],
    ),
]

_ORACLE_CASES = _UART_CASES + _FLOAT_CASES


@pytest.mark.parametrize("name,make,vectors", _ORACLE_CASES, ids=[name for name, _, _ in _ORACLE_CASES])
def test_corpus_oracle(name: str, make: Callable[[], Callable[..., object]], vectors: list[InputRow]) -> None:
    target = make()
    compared = assert_hir_matches_reference(lower(target, DEFAULT_UNROLL_MAX_TRIPS).hir, target, vectors, label=name)
    assert compared == len(vectors)


def _ops() -> holoso.Options:
    fmt = holoso.FloatFormat(wexp=8, wman=23)
    return holoso.Options(
        holoso.OperatorOptions(
            fadd=holoso.FAddOptions(),
            fmul=holoso.FMulOptions(),
            fdiv=holoso.FDivOptions(),
            fmul_ilog2=holoso.FMulILog2Options(),
            fcmp=holoso.FCmpOptions(),
        ),
        ffmt=fmt,
    )


@pytest.mark.parametrize("name,make,vectors", _FLOAT_CASES, ids=[name for name, _, _ in _FLOAT_CASES])
def test_float_corpus_synthesizes(
    name: str, make: Callable[[], Callable[..., object]], vectors: list[InputRow]
) -> None:
    holoso.synthesize(make(), _ops(), name=f"float_{name}")


def _typed(value: object) -> tuple[type, float | bool | int]:
    """Python's ``==`` conflates ``True``, ``1`` and ``1.0``, blinding a bare comparison to family substitution."""
    match value:
        case bool():
            return bool, value
        case IntValue():
            return int, int(value)
        case FloatValue():
            return float, float(value)
        case int():
            return int, value
        case float():
            return float, value
    raise AssertionError(value)


def _family(scalar_type: object) -> type:
    match scalar_type:
        case BoolType():
            return bool
        case IntType():
            return int
        case FloatType():
            return float
    raise AssertionError(scalar_type)


@pytest.mark.parametrize("name,make,vectors", INT_CASES, ids=[name for name, _, _ in INT_CASES])
def test_int_corpus_model_matches_python(
    name: str, make: Callable[[], Callable[..., object]], vectors: list[InputRow]
) -> None:
    result = holoso.synthesize(make(), int_corpus_options(), name=f"corpus_{name}")
    model = result.numerical_model.elaborate()
    reference = make()
    instance = reference.__self__ if inspect.ismethod(reference) else None
    assert instance is not None
    input_names = [port.name for port in model.inputs]
    port_families = {port.name: _family(port.scalar_type) for port in result.output_ports}
    assert list(port_families) == [port.name for port in model.outputs]
    for index, row in enumerate(vectors):
        arguments = [row[input_name] for input_name in input_names]
        returned = reference(*arguments)
        leaves = {} if returned is None else {port_name(path): leaf for path, leaf in flatten_value(returned)}
        slots = instance_leaves(instance)
        produced = dict(zip([port.name for port in model.outputs], model.run(*arguments), strict=True))
        for out_name, actual in produced.items():
            expected = slots[out_name[len("state_") :]] if out_name.startswith("state_") else leaves[out_name]
            assert _typed(actual) == _typed(expected), f"{name}[{index}] {out_name}"
            assert port_families[out_name] is _typed(expected)[0], f"{name}[{index}] {out_name} family"
        for leaf_name, leaf in leaves.items():  # an elided return leaf must be observable through a state port
            if leaf_name not in produced:
                held = [slots[out_name[len("state_") :]] for out_name in produced if out_name.startswith("state_")]
                assert any(_typed(leaf) == _typed(value) for value in held), f"{name}[{index}] {leaf_name}"
