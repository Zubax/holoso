"""Access to the shared ``holoso_support`` HDL that generated modules instantiate."""

from __future__ import annotations

from functools import cache
from pathlib import Path

_HDL_DIR = Path(__file__).resolve().parents[1] / "hdl"


@cache
def support_verilog() -> str:
    """Return the contents of ``holoso_support.v`` (the operator wrappers and register file)."""
    path = _HDL_DIR / "holoso_support.v"
    return path.read_text(encoding="utf-8")


@cache
def support_header() -> str:
    """Return the contents of ``holoso_support.vh`` (lane-select and sign-op macros)."""
    path = _HDL_DIR / "holoso_support.vh"
    return path.read_text(encoding="utf-8")
