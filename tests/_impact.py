"""
Verilog-hash test-impact cache for the example-driven cosim/synth matrices.

Generation is deterministic (``test_determinism.py`` proves byte-identical Verilog across hash seeds), so the
generated Verilog is a sound impact oracle: if a change leaves an example's emitted Verilog byte-identical, the
row's previous verdict still holds and the expensive simulation/synthesis can be skipped. Digesting costs one
``synthesize()`` pass per row (seconds); the simulations it replaces cost minutes each.

Opt-in and local-only: rows consult the cache only under ``HOLOSO_IMPACT_CACHE=1``, CI never sets it (the full
uncached matrix remains the authoritative backstop), and the manifest lives in a gitignored directory of one
atomically-replaced file per row, so parallel pytest-xdist workers never contend on a shared file. Only PASSING
verdicts are recorded; a failure always reruns.

The one theoretical gap -- a backend change mapping two different LIRs onto one Verilog while the numerical model
diverges -- is exactly what the uncached CI backstop exists for.
"""

import hashlib
import json
import os
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

import holoso

_CACHE_DIR = Path(__file__).resolve().parents[1] / ".impact_cache"


def enabled() -> bool:
    return os.environ.get("HOLOSO_IMPACT_CACHE", "") == "1"


def verilog_digest(kernel: "Callable[..., object]", ops: holoso.OpConfig, name: str) -> str:
    result = holoso.synthesize(kernel, ops, name=name)
    hasher = hashlib.sha256()
    hasher.update(result.verilog_output.verilog.encode())
    for filename in sorted(result.verilog_output.support_files):
        hasher.update(filename.encode())
        hasher.update(result.verilog_output.support_files[filename].encode())
    return hasher.hexdigest()


def content_digest(*parts: str) -> str:
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(part.encode())
        hasher.update(b"\x00")
    return hasher.hexdigest()


def _row_path(row: str) -> Path:
    return _CACHE_DIR / (hashlib.sha256(row.encode()).hexdigest()[:32] + ".json")


def cached_pass(row: str, digest: str) -> str | None:
    """The HEAD sha under which this row last passed with this exact Verilog, or None."""
    try:
        entry = json.loads(_row_path(row).read_text())
    except (OSError, ValueError):
        return None
    if entry.get("digest") == digest and entry.get("verdict") == "pass":
        head = entry.get("head", "unknown")
        assert isinstance(head, str)
        return head
    return None


def record_pass(row: str, digest: str) -> None:
    _CACHE_DIR.mkdir(exist_ok=True)
    head = _git_head()
    entry = {"row": row, "digest": digest, "verdict": "pass", "head": head, "time": time.time()}
    scratch = _row_path(row).with_suffix(".tmp")
    scratch.write_text(json.dumps(entry))
    os.replace(scratch, _row_path(row))


def _git_head() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"
