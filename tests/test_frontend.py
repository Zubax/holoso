"""Unit tests for the Python-to-HIR frontend (holoso._frontend.lower)."""

import math
import sys
from pathlib import Path

import pytest

from holoso import FloatFormat, MissingIntrinsic, UnsupportedConstruct
from holoso._frontend import lower
from holoso._hir import ABS, ADD, DIV, MUL, Arith, SignFix

FMT = FloatFormat(6, 18)


def _arith_count(hir, op):  # type: ignore[no-untyped-def]
    return sum(1 for n in hir.nodes.values() if isinstance(n, Arith) and n.op is op)


def _count(hir, cls):  # type: ignore[no-untyped-def]
    return sum(1 for n in hir.nodes.values() if isinstance(n, cls))


def test_small_kernel_inputs_outputs_and_ops() -> None:
    def kernel(a, b):  # type: ignore[no-untyped-def]
        return (a - b) * 0.25 + a * b

    hir = lower(kernel, FMT)
    assert hir.input_names() == ["a", "b"]
    assert [o.name for o in hir.outputs] == ["out_0"]
    assert _arith_count(hir, MUL) == 2  # (a-b)*0.25 and a*b
    assert _arith_count(hir, ADD) == 2  # subtraction (add+neg) and the final add
    assert _count(hir, SignFix) == 1  # the negation introduced by subtraction


def test_pow_expands_to_multiply_chain() -> None:
    def cube(a):  # type: ignore[no-untyped-def]
        return a**3

    hir = lower(cube, FMT)
    assert _arith_count(hir, MUL) == 2  # a*a*a


def test_abs_lowers_to_signfix_abs() -> None:
    def f(a):  # type: ignore[no-untyped-def]
        return abs(a)

    hir = lower(f, FMT)
    signfixes = [n for n in hir.nodes.values() if isinstance(n, SignFix)]
    assert len(signfixes) == 1
    assert signfixes[0].op == ABS


def test_division_lowers_to_div() -> None:
    def f(a, b):  # type: ignore[no-untyped-def]
        return a / b

    hir = lower(f, FMT)
    assert _arith_count(hir, DIV) == 1
    divs = [n for n in hir.nodes.values() if isinstance(n, Arith) and n.op == DIV]
    assert len(divs) == 1


def test_ekf1_structure() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    import ekf1

    hir = lower(ekf1.update_x_P, FMT)
    assert len(hir.input_ids) == 17
    assert [o.name for o in hir.outputs] == [f"out_{i}_0" for i in range(9)]
    assert _arith_count(hir, DIV) == 1  # only x22 = 1 / x21


def test_for_loop_is_unsupported() -> None:
    def f(a):  # type: ignore[no-untyped-def]
        x = a
        for _ in range(3):
            x = x + a
        return x

    with pytest.raises(UnsupportedConstruct):
        lower(f, FMT)


def test_unknown_global_is_unsupported() -> None:
    def f(a):  # type: ignore[no-untyped-def]
        return a + UNDEFINED_GLOBAL  # type: ignore[name-defined]  # noqa: F821

    with pytest.raises(UnsupportedConstruct):
        lower(f, FMT)


def test_missing_intrinsic_message() -> None:
    def f(a):  # type: ignore[no-untyped-def]
        return math.sqrt(a)

    with pytest.raises(MissingIntrinsic, match="sqrt"):
        lower(f, FMT)
