"""
Byte-exact goldens of the canonical printed Eel at both seams -- post-desugar for a cross-section of kernels,
and the final residual program for the kernels the partial evaluator fully lowers. The strictness is
load-bearing against canonicalization drift, nondeterminism, and semantic creep: every deliberate change
regenerates these files (set HOLOSO_EEL_REGEN=1, which rewrites them and then FAILS the run so CI can never
silently refreeze).
"""

import os
import sys
import types
from collections.abc import Callable
from pathlib import Path

import pytest

from holoso._eel._desugar import desugar
from holoso._eel._pe import partial_evaluate
from holoso._eel._print import print_eel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import ekf1_stateless  # noqa: E402
import equal_temperament  # noqa: E402
import madd  # noqa: E402
import octave_index  # noqa: E402
import polar  # noqa: E402
import poly3  # noqa: E402
import signal_window  # noqa: E402
from pid import PID  # noqa: E402
from recip_newton import NewtonReciprocal  # noqa: E402
from schmitt_trigger import SchmittTrigger  # noqa: E402
from uart import UartRx  # noqa: E402

from ._eel_corpus import PriorityEncoder

_GOLDENS: list[tuple[str, Callable[[], object]]] = [
    ("madd", lambda: madd.madd),
    ("poly3", lambda: poly3.poly3),
    ("pid", lambda: PID().__call__),
    ("uart_rx", lambda: UartRx(parity=False).__call__),
    ("recip_newton", lambda: NewtonReciprocal().__call__),
    ("ekf1_stateless", lambda: ekf1_stateless.update_x_P),
]

_RESIDUAL_GOLDENS: list[tuple[str, Callable[[], object]]] = [
    ("madd", lambda: madd.madd),
    ("poly3", lambda: poly3.poly3),
    ("signal_window", lambda: signal_window.signal_window),
    # Pins the pow_ stub-inline shape: its runtime guard survives residually and log2(2.0) folds away.
    ("equal_temperament", lambda: equal_temperament.equal_temperament),
    # Pins the array seams: the decomposed leaf parameters and the fully dissolved aggregate structure.
    ("to_polar", lambda: polar.to_polar),
    ("ekf1_stateless", lambda: ekf1_stateless.update_x_P),
    # Pins the residual-while shape: the phi rows, the header re-evaluated per test, and the back edge.
    ("octave_index", lambda: octave_index.octave_index),
    # Pins the persistent-state shape: slot declarations with resets, the eager entry reads, the per-leaf
    # SlotWrite commits before the return, and the elision of a returned public slot.
    ("pid", lambda: PID().__call__),
    ("schmitt_trigger", lambda: SchmittTrigger().__call__),
    # Pins the loop-exit shape: break lanes hold their branches open, the trips nest into the surviving
    # arms, and the loop-end join seals the exit value through pairwise join chains.
    ("priority_encoder", lambda: PriorityEncoder().step),
]

_DIRECTORY = Path(__file__).resolve().parent / "eel_goldens"


def _target(make: Callable[[], object]) -> tuple[types.FunctionType, object | None]:
    target = make()
    if isinstance(target, types.MethodType):
        fn = target.__func__
        assert isinstance(fn, types.FunctionType)
        return fn, target.__self__
    assert isinstance(target, types.FunctionType)
    return target, None


def _check(path: Path, text: str) -> None:
    if os.environ.get("HOLOSO_EEL_REGEN"):
        _DIRECTORY.mkdir(exist_ok=True)
        path.write_text(text, encoding="utf-8")
        pytest.fail(f"golden {path.name} regenerated; rerun without HOLOSO_EEL_REGEN")
    assert text == path.read_text(encoding="utf-8"), f"golden mismatch: {path.name} (HOLOSO_EEL_REGEN=1 to refresh)"


@pytest.mark.parametrize("name,make", _GOLDENS, ids=[name for name, _ in _GOLDENS])
def test_desugar_golden(name: str, make: Callable[[], object]) -> None:
    fn, _ = _target(make)
    _check(_DIRECTORY / f"{name}.desugar.eel", print_eel(desugar(fn)))


@pytest.mark.parametrize("name,make", _RESIDUAL_GOLDENS, ids=[name for name, _ in _RESIDUAL_GOLDENS])
def test_residual_golden(name: str, make: Callable[[], object]) -> None:
    fn, instance = _target(make)
    _check(_DIRECTORY / f"{name}.residual.eel", print_eel(partial_evaluate(desugar(fn), fn, instance)))
