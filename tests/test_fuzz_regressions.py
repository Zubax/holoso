"""
Replay of saved fuzz regressions (UNMARKED -- runs in the normal ``tests`` session).

Globs ``tests/fuzz_regressions/*.py`` and replays each previously-found differential divergence, asserting the
previously-failing check now passes. Each case was saved at a specific ``HOLOSO_REGALLOC_EFFORT``; since that effort is
read once at import in ``holoso._lir._regalloc`` and cannot be changed in-process, every case is replayed in a
SUBPROCESS pinned to its saved effort -- mirroring the subprocess-per-seed pattern in ``test_determinism.py``. A case is
skipped
(with a reason) only if its saved op-config is unknown to the current code.

The corpus is empty until the campaign finds and saves a divergence; an empty corpus collects nothing, which is the
intended steady state (a green campaign leaves no regressions to replay).
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from ._fuzz import OP_CONFIGS, REGRESSIONS_DIR

_REPO = Path(__file__).resolve().parent.parent
_CASES = sorted(p for p in REGRESSIONS_DIR.glob("*.py") if p.name != "__init__.py")


def _replay_entry() -> None:
    """
    Subprocess entry: load the saved reproducer module named by ``argv[1]``, pull its kernel callable and ``META``, and
    replay the saved check. Exits 0 on PASS; prints the failure detail and exits 1 on FAIL. The kernel symbol shares the
    reproducer's filename (``META['kernel_name']``): a function for a stateless kernel, a class for a stateful one.
    """
    import importlib.util

    sys.path.insert(0, str(_REPO))
    from tests._fuzz import replay_case  # imported here so the child sets the effort env before this module loads

    case_path = Path(sys.argv[1])
    spec = importlib.util.spec_from_file_location(case_path.stem, case_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    meta = module.META
    symbol = getattr(module, meta["kernel_name"])
    kernel_callable = symbol().__call__ if meta["is_stateful"] else symbol

    passes_now, detail = replay_case(kernel_callable, meta)
    if passes_now:
        print(f"PASS {meta['kernel_name']} [{meta['op_label']}] {meta['check']}")
        sys.exit(0)
    print(f"FAIL {meta['kernel_name']} [{meta['op_label']}] {meta['check']}: {detail}")
    sys.exit(1)


def _read_meta(path: Path) -> dict[str, object]:
    """Load just the ``META`` dict from a saved reproducer (to read its op-config and effort without replaying)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return dict(module.META)


@pytest.mark.skipif(not _CASES, reason="no saved fuzz regressions to replay")
@pytest.mark.parametrize("case", _CASES, ids=[p.stem for p in _CASES])
def test_fuzz_regression(case: Path) -> None:
    """Replay one saved case in a subprocess pinned to its saved regalloc effort; assert its check now passes."""
    meta = _read_meta(case)
    if meta["op_label"] not in OP_CONFIGS:
        pytest.skip(f"unknown op-config {meta['op_label']!r} (saved by a newer/older generator)")
    effort = str(meta.get("effort") or "")
    env = {**os.environ}
    if effort:
        env["HOLOSO_REGALLOC_EFFORT"] = effort
    else:
        env.pop("HOLOSO_REGALLOC_EFFORT", None)
    bootstrap = (
        f"import sys; sys.path.insert(0, {str(_REPO)!r}); "
        f"from tests.test_fuzz_regressions import _replay_entry; _replay_entry()"
    )
    proc = subprocess.run(
        [sys.executable, "-c", bootstrap, str(case)],
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )
    assert (
        proc.returncode == 0
    ), f"{case.name} regressed at effort={effort or '<default>'}:\n{proc.stdout}\n{proc.stderr}"
