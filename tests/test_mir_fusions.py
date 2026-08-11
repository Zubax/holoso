"""
Directional-infinity composite recognition (``isinf(x) and x > 0`` -> one classifier), asserted on public
artifacts: classifier call sites in the emitted Verilog with the support-function prelude stripped, pooled
``fcmp`` presence/absence, and model values against CPython. The post-fusion ``band`` facts keep one white-box
MIR sentinel because a boolean AND renders as an inline expression with no call-site spelling.
"""

import math
from collections.abc import Callable

import holoso
from holoso import FCmpOptions, FloatFormat, OperatorOptions, Options
from holoso._eel import lower
from holoso._mir import MirOperation
from holoso._mir import lower as lower_to_mir

from ._modelref import build_ops, DEFAULT_IFCONV_MAX_OPS, DEFAULT_UNROLL_MAX_TRIPS
from ._public import strip_inline_prelude

_OPTIONS = Options(OperatorOptions(fcmp=FCmpOptions()), ffmt=FloatFormat(6, 18))


def _directional(a: float, b: float, c: float, d: float) -> tuple[bool, ...]:
    return (
        math.isinf(a) and a > 0.0,
        b < 0.0 and math.isinf(b),
        math.isinf(-c) and -c > 0.0,
        0.0 > d and math.isinf(d),
    )


def _reused(x: float) -> tuple[bool, ...]:
    inf = math.isinf(x)
    pos = x > 0.0
    return inf and pos, inf, pos


def _shared(x: float) -> tuple[bool, ...]:
    inf = math.isinf(x)
    return inf and x > 0.0, inf and x < 0.0


def _classified(kernel: Callable[..., tuple[bool, ...]], name: str) -> tuple[holoso.SynthesisResult, dict[str, int]]:
    result = holoso.synthesize(kernel, _OPTIONS, name=name)
    body = strip_inline_prelude(result.verilog_output.verilog)
    return result, {n: body.count(f"holoso_{n}(") for n in ("fisposinf", "fisneginf", "fisfinite")}


def _sweep(result: holoso.SynthesisResult, kernel: Callable[..., tuple[bool, ...]], values: tuple[float, ...]) -> None:
    model = result.numerical_model.elaborate()
    arity = len(result.input_ports)
    for x in values:
        args = [x] * arity
        assert model.run(*args) == list(kernel(*args)), f"x={x}"


def test_directional_inf_composites_lower_to_one_classifier() -> None:
    result, classifiers = _classified(_directional, "directional_inf")
    assert classifiers == {"fisposinf": 2, "fisneginf": 2, "fisfinite": 0}
    assert "holoso_fcmp #(" not in result.verilog_output.verilog
    _sweep(result, _directional, (math.inf, -math.inf, 1.0, -1.0, 0.0))


def test_directional_inf_fusion_preserves_reused_predicates() -> None:
    result, classifiers = _classified(_reused, "directional_inf_reused")
    assert classifiers == {"fisposinf": 0, "fisneginf": 0, "fisfinite": 1}
    assert "holoso_fcmp #(" in result.verilog_output.verilog
    _sweep(result, _reused, (math.inf, -math.inf, 1.0, -1.0))


def test_directional_inf_fusion_suppresses_predicate_shared_only_by_fused_ands() -> None:
    result, classifiers = _classified(_shared, "directional_inf_shared")
    assert classifiers == {"fisposinf": 1, "fisneginf": 1, "fisfinite": 0}
    assert "holoso_fcmp #(" not in result.verilog_output.verilog
    _sweep(result, _shared, (math.inf, -math.inf, 1.0, -1.0))


def test_band_survives_only_where_fusion_leaves_a_conjunction() -> None:
    def band_count(kernel: Callable[..., object]) -> int:
        mir = lower_to_mir(lower(kernel, DEFAULT_UNROLL_MAX_TRIPS).hir, build_ops(_OPTIONS), DEFAULT_IFCONV_MAX_OPS)
        return sum(1 for n in mir.nodes.values() if isinstance(n, MirOperation) and n.operator.mnemonic == "band")

    assert band_count(_directional) == 0
    assert band_count(_reused) == 1
    assert band_count(_shared) == 0
