"""Vendored ZKF (Zubax Kulibin float) bit-exact model and RTL."""

from functools import cache
from importlib import resources
from importlib.resources.abc import Traversable

from ._model import Zkf as Zkf, ZkfFormat as ZkfFormat

_RTL_DIR = "rtl"


@cache
def get_rtl() -> dict[str, str]:
    """``{path: verilog_text}``, each path relative to the vendored ``rtl/`` root."""
    return dict(_iter_rtl(resources.files(__package__).joinpath(_RTL_DIR)))


def _iter_rtl(node: Traversable, prefix: str = "") -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for child in node.iterdir():
        rel = f"{prefix}{child.name}"
        if child.is_dir():
            out += _iter_rtl(child, f"{rel}/")
        elif child.name.endswith(".v"):
            out.append((rel, child.read_text(encoding="utf-8")))
    return out
