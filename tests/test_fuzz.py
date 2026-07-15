"""
End-to-end blackbox differential fuzzing entry point.

The marked campaign (``pytest -m fuzz``) generates many kernels and drives them through the differential runner; it is
slow and excluded from the normal ``tests`` session. A tiny UNMARKED smoke campaign runs in the normal session so the
fuzzer cannot bit-rot. Both read their budget from the environment (with sane defaults), so a CI job can scale coverage
without editing code. The regalloc effort is whatever ``HOLOSO_REGALLOC_EFFORT`` was at process import (it is frozen
once in ``holoso._lir._regalloc`` and cannot be changed in-process), and any divergence saves a self-contained
reproducer under ``tests/fuzz_regressions/`` before failing.
"""

import os
from collections.abc import Callable

import numpy as np
import pytest

from holoso._type import FloatFormat

from . import _fuzz as fuzz_impl
from ._fuzz import CheckKind, Divergence, run_campaign, save_reproducer

# The campaign datapath: a shallow format keeps the per-kernel build fast while still exercising rounding, branches,
# and the bool bank. The differential oracle is format-agnostic, so one well-chosen format suffices.
_FMT = FloatFormat(6, 18)

# The effort frozen at import time in the regalloc; recorded into every reproducer so a regression replays at the same
# effort. Reading it here (not mutating it) matches how the compiler reads it.
_EFFORT = os.environ.get("HOLOSO_REGALLOC_EFFORT", "")


def _budget(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw, 0) if raw else default  # base 0: accept decimal or a 0x-prefixed hex value (e.g. a hex seed)


def _ansi(text: str, code: str) -> str:
    return f"\x1b[{code}m{text}\x1b[0m"


def _print_summary(stats: object) -> None:
    from ._fuzz import CampaignStats, Shape  # local import keeps the module import light

    assert isinstance(stats, CampaignStats)
    title = _ansi("  HOLOSO FUZZ CAMPAIGN  ", "1;97;44")
    print(f"\n{title}")
    print(
        f"  {_ansi('kernels', '96')}={stats.kernels}  {_ansi('vectors', '96')}={stats.vectors}  "
        f"{_ansi('stateful', '96')}={stats.stateful}  {_ansi('exact-mode', '96')}={stats.exact_mode}  "
        f"{_ansi('secondary', '96')}={stats.secondary_checked}✓/{stats.secondary_skipped}⤳  "
        f"{_ansi('cont-drift', '96')}={stats.continuous_drift}  "
        f"{_ansi('armed-deadarm', '96')}={stats.dead_arm_forced}  "
        f"{_ansi('deferred-skip', '96')}={stats.deferred_skipped}"
    )
    print(f"  {_ansi('shape histogram (kernels realizing each shape)', '93')}")
    peak = max((stats.shape_counts[s] for s in Shape), default=1) or 1
    for shape in Shape:
        count = stats.shape_counts[shape]
        bar = _ansi("█" * (count * 32 // peak), "92" if count else "90")
        print(f"    {shape.name:18s} {count:4d} {bar}")
    if stats.divergences:
        print(f"  {_ansi('DIVERGENCES', '1;91')} ({len(stats.divergences)})")
        for div in stats.divergences:
            tag = "91" if div.check is CheckKind.INTERP_VS_MODEL else "33"
            print(f"    {_ansi(div.check.value, tag)}  {div.kernel.name} [{div.op_label}]  {div.detail[:120]}")
    else:
        print(f"  {_ansi('no divergences ✔', '1;92')}")


def _run_and_assert(n_kernels: int, n_vectors: int, seed: int) -> None:
    """
    Run a campaign, save a reproducer for every divergence, print the summary, and fail if any divergence occurred. An
    ``interp_vs_model`` divergence is a genuine LIR-layer miscompile; a ``model_vs_float64`` divergence (only ever
    reported in EXACT mode) is a front/mid-end or operator discrepancy. Either is a real bug -- the saved reproducer is
    a permanent regression.
    """
    saved: list[tuple[Divergence, str]] = []

    def on_divergence(divergence: Divergence) -> None:
        path = save_reproducer(divergence, _FMT)
        saved.append((divergence, str(path)))

    stats = run_campaign(n_kernels, n_vectors, seed, _EFFORT, _FMT, on_divergence)
    _print_summary(stats)

    if stats.divergences:
        lines = [
            f"{div.check.value}: {div.kernel.name} [{div.op_label}] effort={div.effort or '<default>'}\n"
            f"    {div.detail}\n    reproducer: {path}"
            for div, path in saved
        ]
        pytest.fail(
            f"fuzz campaign found {len(stats.divergences)} differential divergence(s):\n" + "\n".join(lines),
            pytrace=False,
        )

    # A campaign that produced no branchy kernels at all would silently test nothing; guard against a timid generator.
    from ._fuzz import Shape

    assert stats.shape_counts[Shape.BRANCH] > 0, "no branchy kernels generated -- the fuzzer is degenerate"
    assert stats.shape_counts[Shape.OVERBUDGET_BRANCH] > 0, "no over-budget branch kernels generated"
    assert stats.shape_counts[Shape.RELATION_PAIR] > 0, "no relation-pair kernels generated"
    assert stats.shape_counts[Shape.EXACT_WIRING] >= 2, "exact wiring kernels were not both generated"


def _surviving_forward_branches_for_probe(name: str, emit: Callable[[fuzz_impl._Emitter], fuzz_impl._Fragment]) -> int:
    em = fuzz_impl._make_emitter(np.random.default_rng(0xC0FFEE), ["a", "b", "c"], set())
    fragment = emit(em)
    em.return_line = f"return {fragment.value}"
    kernel = fuzz_impl._finish_function_kernel(
        name,
        0xC0FFEE,
        0,
        ["a", "b", "c"],
        set(),
        em,
        fragment.shapes,
        fragment.mode,
    )
    mir, _lir, _model, _interpreter = fuzz_impl._build_with_lir(
        kernel.callable, fuzz_impl.OP_CONFIGS["default"](_FMT), name
    )
    return fuzz_impl.surviving_forward_branches(mir)


def test_branch_claiming_inner_shapes_survive_compilation() -> None:
    """
    A data-dependent nested diamond must keep BOTH branches (an inner shape must not pass merely because the outer
    survived); a const-guarded inner branch, by contrast, folds away and leaves only the outer runtime diamond.
    """
    assert (
        _surviving_forward_branches_for_probe("nested_probe", lambda em: fuzz_impl._emit_diamond(em, nested=True)) >= 2
    )
    assert _surviving_forward_branches_for_probe("const_probe", fuzz_impl._emit_const_branch) == 1


@pytest.mark.fuzz
@pytest.mark.parametrize("shard", range(_budget("HOLOSO_FUZZ_SHARDS", 1)))
def test_fuzz_campaign(shard: int) -> None:
    """
    HOLOSO_FUZZ_SHARDS=K splits the kernel budget across K parametrized cases with disjoint derived seeds, so a
    many-core machine parallelizes the campaign under pytest-xdist (the campaign is embarrassingly parallel by
    seed). The default of 1 keeps today's exact single-sequence behavior, so CI reproducibility is untouched.
    """
    shards = _budget("HOLOSO_FUZZ_SHARDS", 1)
    n_kernels = _budget("HOLOSO_FUZZ_KERNELS", 200)
    n_vectors = _budget("HOLOSO_FUZZ_ITERS", 32)
    seed = _budget("HOLOSO_FUZZ_SEED", 0xF0007A11)
    per_shard = -(-n_kernels // shards)  # ceil: the total meets or slightly exceeds the requested budget
    _run_and_assert(per_shard, n_vectors, seed ^ (shard * 0x9E3779B9))


def test_fuzz_smoke() -> None:
    """
    A tiny fixed-budget campaign that runs in the NORMAL (unmarked) ``tests`` session so the fuzzer can never bit-rot.
    It exercises the full generator + runner end to end on a handful of kernels and asserts the differential oracle.
    Deliberately UNMARKED, so it is collected by ``-m "not cosim and not fuzz"``.
    """
    _run_and_assert(n_kernels=8, n_vectors=12, seed=0x5A1ED)
