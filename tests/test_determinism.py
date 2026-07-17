"""
Cross-process determinism: identical input must produce identical output regardless of PYTHONHASHSEED.

The compiler had two hash-order-sensitive value-numbering points, both in the frontend: the branch-arm merge
(iterating a set intersection of the arms' bound names; the merge order decides phi creation order, hence HIR value
ids, hence every vid-keyed tie-break downstream) and the carried-state live-in materialization in while-loop headers
(StateRead creation order). Both now iterate sorted. Since the hash seed is fixed per interpreter at startup, this
suite spawns SUBPROCESSES under explicitly different seeds and checks the property end to end: byte-identical Verilog
for a branch-heavy kernel, and an identical HIR node table for the loop-carried-state shape (whose vid permutation is
masked downstream on small kernels, so RTL bytes alone would under-test the numbering claim).

The subprocess entry points are the plain functions below (imported from this module by the child), so the kernels
and operator configs are ordinary, type-checked Python rather than templated source strings.
"""

import os
import subprocess
import sys
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent


class TwoCarried:
    """
    Two persistent attributes first WRITTEN inside a ``while``: materializing their live-ins creates StateRead nodes
    whose creation order is decided by the carried-attribute iteration -- the second historical hash-order leak.
    """

    def __init__(self) -> None:
        self.s1 = -1.0
        self._s2 = 2.0

    def step(self, a: float) -> float:
        w = 2.0
        while w > 0.0:
            if a > w:
                self.s1 = a
            else:
                self._s2 = a
            w = w - 1.0
        return self.s1 + self._s2


def coalesce_conflict(x: float, b: float, cc: float) -> float:
    """
    The phi-coalescing residual-install hazard: ``a`` coalesces onto ``x`` while ``x`` is still live as ``z``'s arm,
    so the soundness fixpoint must de-coalesce. The fixpoint's de-coalescing is set-driven, so this exercises that its
    iteration order -- and the resulting register assignment -- is seed-independent. The division keeps the diamond a
    real branch (un-if-converted), which is what creates the phi merge. The three merged values are summed into one
    scalar output (the new front-end does not emit aggregate returns yet) -- all three phis stay live, so the coalescing
    conflict is preserved.
    """
    if b < cc:
        a = x
        z = 1.0
        d = b
    else:
        a = -(x + 1.0)
        z = x
        d = x / b
    return a + z + d


def emit_coalesce_conflict() -> None:
    from holoso import FloatFormat, synthesize

    from ._modelref import default_ops

    result = synthesize(coalesce_conflict, default_ops(FloatFormat(6, 18)))
    sys.stdout.write(result.verilog_output.verilog)


def competing_rejections(c: bool, x: float) -> float:
    """
    Two places whose facts are BOTH inadmissible at the same branch merge, with DIFFERENT rejection messages:
    ``a`` merges a float with a bool (irreconcilable kinds) while ``b`` merges a value with None. Which of the
    two surfaces must not depend on the environment-join iteration order, i.e. on PYTHONHASHSEED.
    """
    if c:
        a = 1.0
        b = None
    else:
        a = True
        b = 1.0
    return x


def competing_fails(c: bool, x: float) -> float:
    """Two executable Fail terminators: which raise is reported must not depend on block-set iteration order."""
    if c:
        raise ValueError("left arm rejects")
    else:
        raise ValueError("right arm rejects")
    return x


def _emit_rejection(kernel: "Callable[..., float]") -> None:
    from holoso import FloatFormat, SynthesisError, synthesize

    from ._modelref import default_ops

    try:
        synthesize(kernel, default_ops(FloatFormat(6, 18)))
    except SynthesisError as error:
        sys.stdout.write(f"{type(error).__name__}: {error}")
    else:
        raise AssertionError("the kernel must reject")


def emit_competing_rejection() -> None:
    _emit_rejection(competing_rejections)


def emit_competing_fail() -> None:
    _emit_rejection(competing_fails)


class BadResets:
    """
    Two state leaves whose reset joins are BOTH inadmissible, with different messages: the W/D loop iterates a
    StateLeaf set (address-keyed hashes), so which leaf's rejection surfaces must be made order-independent.
    """

    def __init__(self) -> None:
        self.a: object = None
        self.b: object = object()

    def step(self, x: float) -> float:
        self.a = x
        self.b = x
        return x


def emit_competing_reset_rejection() -> None:
    _emit_rejection(BadResets().step)


def emit_cordic() -> None:
    sys.path.insert(0, str(_REPO / "examples"))
    from cordic_sincos import CordicSinCos

    from holoso import FloatFormat, synthesize

    from ._modelref import default_ops

    result = synthesize(CordicSinCos().__call__, default_ops(FloatFormat(6, 18)))
    sys.stdout.write(result.verilog_output.verilog)


def dump_two_carried_hir() -> None:
    from holoso._frontend import lower

    hir = lower(TwoCarried().step)
    for vid in sorted(hir.nodes):
        print(vid, repr(hir.nodes[vid]))


@lru_cache(maxsize=None)
def _entry_output_under_seed(entry: str, seed: str) -> str:
    bootstrap = f"import sys; sys.path.insert(0, {str(_REPO)!r}); from {__name__} import {entry}; {entry}()"
    proc = subprocess.run(
        [sys.executable, "-c", bootstrap],
        capture_output=True,
        text=True,
        timeout=600,
        env={**os.environ, "PYTHONHASHSEED": seed},
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


@pytest.mark.parametrize("other_seed", ["3", "31337"])
def test_verilog_is_byte_identical_across_hash_seeds(other_seed: str) -> None:
    assert _entry_output_under_seed("emit_cordic", "0") == _entry_output_under_seed("emit_cordic", other_seed)


@pytest.mark.parametrize("other_seed", ["3", "31337"])
def test_phi_coalescing_de_coalescing_is_byte_identical_across_hash_seeds(other_seed: str) -> None:
    # The coalescing soundness fixpoint forbids whole conflicting classes via a set; the byte-identical Verilog across
    # seeds proves its de-coalescing decisions (and the register coloring that follows) are order-independent.
    assert _entry_output_under_seed("emit_coalesce_conflict", "0") == _entry_output_under_seed(
        "emit_coalesce_conflict", other_seed
    )


@pytest.mark.parametrize("other_seed", ["1", "3", "31337"])
def test_competing_rejections_report_identically_across_hash_seeds(other_seed: str) -> None:
    # Regression: _Env.join_with iterated an unordered place union (and _finalize iterated block/state-leaf
    # sets), so which of two simultaneous rejections surfaced depended on PYTHONHASHSEED. Seed 1 is in the
    # matrix because it was observed to flip the reported message against seed 0 before the fix.
    assert _entry_output_under_seed("emit_competing_rejection", "0") == _entry_output_under_seed(
        "emit_competing_rejection", other_seed
    )


@pytest.mark.parametrize("other_seed", ["1", "2", "3", "31337"])
def test_competing_state_reset_rejections_report_identically_across_hash_seeds(other_seed: str) -> None:
    # Regression (Codex review): the W/D fixpoint's state-join loop iterated the runtime-leaf set and raised at
    # the first inadmissible reset, so the reported leaf depended on PYTHONHASHSEED.
    assert _entry_output_under_seed("emit_competing_reset_rejection", "0") == _entry_output_under_seed(
        "emit_competing_reset_rejection", other_seed
    )


@pytest.mark.parametrize("other_seed", ["1", "3", "31337"])
def test_competing_fail_terminators_report_identically_across_hash_seeds(other_seed: str) -> None:
    # Property lock, not a regression: BlockId hashes on an int (PYTHONHASHSEED-independent), so the pre-sort
    # iteration happened to be stable. The _validate sort canonicalizes the choice to the lowest block index so
    # it stays stable if block identity or hashing ever changes.
    assert _entry_output_under_seed("emit_competing_fail", "0") == _entry_output_under_seed(
        "emit_competing_fail", other_seed
    )


@pytest.mark.parametrize("other_seed", ["3", "31337"])
def test_loop_carried_state_numbering_is_identical_across_hash_seeds(other_seed: str) -> None:
    # Regression (review): carried state attributes were iterated in set hash order when materializing their
    # live-ins, permuting StateRead value ids (and everything downstream) whenever a while loop first-writes two
    # or more persistent attributes.
    reference = _entry_output_under_seed("dump_two_carried_hir", "0")
    assert reference == _entry_output_under_seed("dump_two_carried_hir", other_seed)
