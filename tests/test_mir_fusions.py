"""Tests for MIR-level composite operator recognition."""

import math
from collections import Counter
from collections.abc import Callable

from holoso import (
    FAddOptions,
    FCmpOptions,
    FDivOptions,
    FMulILog2Options,
    FMulOptions,
    FloatFormat,
    OperatorOptions,
    Options,
)
from holoso._eel import lower
from holoso._hir import optimize
from ._modelref import build_lir, build_ops
from holoso._operators import OpConfig
from holoso._mir import Mir, MirOperation
from holoso._mir import lower as lower_to_mir

from ._modelref import build_model

FMT = FloatFormat(6, 18)
OPS = build_ops(
    Options(
        OperatorOptions(
            fadd=FAddOptions(),
            fmul=FMulOptions(),
            fdiv=FDivOptions(),
            fmul_ilog2=FMulILog2Options(),
            fcmp=FCmpOptions(),
        ),
        ffmt=FMT,
    )
)


def _run(target: Callable[..., object], ops: OpConfig = OPS) -> Mir:
    return lower_to_mir(optimize(lower(target).hir), ops)


def _mir_operation_counts(mir: Mir) -> Counter[str]:
    return Counter(n.operator.mnemonic for n in mir.nodes.values() if isinstance(n, MirOperation))


def test_directional_inf_composites_lower_to_one_classifier() -> None:
    def kernel(a: float, b: float, c: float, d: float) -> tuple[bool, ...]:
        return (
            math.isinf(a) and a > 0.0,
            b < 0.0 and math.isinf(b),
            math.isinf(-c) and -c > 0.0,
            0.0 > d and math.isinf(d),
        )

    mir = _run(kernel)
    counts = _mir_operation_counts(mir)
    assert counts.get("fisposinf") == 2 and counts.get("fisneginf") == 2
    assert "band" not in counts and "fcmp" not in counts and "fisfinite" not in counts

    model = build_model(build_lir(mir, "directional_inf"))
    for x in (float("inf"), float("-inf"), 1.0, -1.0, 0.0):
        got = model.run(x, x, x, x)
        want = [
            math.isinf(x) and x > 0.0,
            x < 0.0 and math.isinf(x),
            math.isinf(-x) and -x > 0.0,
            0.0 > x and math.isinf(x),
        ]
        assert got == want, f"x={x}: {got} vs {want}"


def test_directional_inf_fusion_preserves_reused_predicates() -> None:
    def kernel(x: float) -> tuple[bool, ...]:
        inf = math.isinf(x)
        pos = x > 0.0
        return inf and pos, inf, pos

    mir = _run(kernel)
    counts = _mir_operation_counts(mir)
    assert "fisposinf" not in counts
    assert counts.get("fisfinite") == 1
    assert counts.get("fcmp") == 1
    assert counts.get("band") == 1

    model = build_model(build_lir(mir, "directional_inf_reused"))
    for x in (float("inf"), float("-inf"), 1.0, -1.0):
        inf = math.isinf(x)
        pos = x > 0.0
        assert model.run(x) == [inf and pos, inf, pos]


def test_directional_inf_fusion_suppresses_predicate_shared_only_by_fused_ands() -> None:
    def kernel(x: float) -> tuple[bool, ...]:
        inf = math.isinf(x)
        return inf and x > 0.0, inf and x < 0.0

    mir = _run(kernel)
    counts = _mir_operation_counts(mir)
    assert counts.get("fisposinf") == 1
    assert counts.get("fisneginf") == 1
    assert "fisfinite" not in counts and "fcmp" not in counts and "band" not in counts

    model = build_model(build_lir(mir, "directional_inf_shared"))
    for x in (float("inf"), float("-inf"), 1.0, -1.0):
        inf = math.isinf(x)
        assert model.run(x) == [inf and x > 0.0, inf and x < 0.0]
