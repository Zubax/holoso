"""
Byte-exact goldens of the canonical printed Eel at the post-desugar seam, for a cross-section of kernels.
The strictness is load-bearing against canonicalization drift, nondeterminism, and semantic creep into the
desugarer: every deliberate desugar change regenerates these files (set HOLOSO_EEL_REGEN=1, which rewrites them
and then FAILS the run so CI can never silently refreeze).
"""

import os
import sys
import types
from collections.abc import Callable
from pathlib import Path

import pytest

from holoso._eel._desugar import desugar
from holoso._eel._print import print_eel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import ekf1_stateless  # noqa: E402
import madd  # noqa: E402
import poly3  # noqa: E402
from pid import PID  # noqa: E402
from recip_newton import NewtonReciprocal  # noqa: E402
from uart import UartRx  # noqa: E402

_GOLDENS: list[tuple[str, Callable[[], object]]] = [
    ("madd", lambda: madd.madd),
    ("poly3", lambda: poly3.poly3),
    ("pid", lambda: PID().__call__),
    ("uart_rx", lambda: UartRx(parity=False).__call__),
    ("recip_newton", lambda: NewtonReciprocal().__call__),
    ("ekf1_stateless", lambda: ekf1_stateless.update_x_P),
]

_DIRECTORY = Path(__file__).resolve().parent / "eel_goldens"


@pytest.mark.parametrize("name,make", _GOLDENS, ids=[name for name, _ in _GOLDENS])
def test_desugar_golden(name: str, make: Callable[[], object]) -> None:
    target = make()
    fn = target.__func__ if isinstance(target, types.MethodType) else target
    assert isinstance(fn, types.FunctionType)
    text = print_eel(desugar(fn))
    path = _DIRECTORY / f"{name}.desugar.eel"
    if os.environ.get("HOLOSO_EEL_REGEN"):
        _DIRECTORY.mkdir(exist_ok=True)
        path.write_text(text, encoding="utf-8")
        pytest.fail(f"golden {path.name} regenerated; rerun without HOLOSO_EEL_REGEN")
    assert text == path.read_text(encoding="utf-8"), f"golden mismatch: {path.name} (HOLOSO_EEL_REGEN=1 to refresh)"
