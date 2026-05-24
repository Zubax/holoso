"""Bundled demo kernels for Holoso.

Each demo is a *function-form* kernel -- a single top-level ``def`` of scalar float arguments, straight-line, no
state or control flow -- which is the shape the v0 frontend accepts. The kernels live as ordinary modules in this
package, so they are the single source of truth: the CLI examples import them, the tests exercise them, and the web
UI lists them by reading their source text via ``importlib.resources`` (the same source ships inside the wheel, so an
in-browser Pyodide install needs no separate corpus).

``load_demos()`` returns them ready for a picker: id, human label, and verbatim source.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources


@dataclass(frozen=True, slots=True)
class Demo:
    """A demo kernel for display: ``id`` (module stem), human-readable ``label``, and verbatim ``source``."""

    id: str
    label: str
    source: str


# Ordered for display, trivial -> advanced. The first column is the module file; the second is the picker label.
_MANIFEST: tuple[tuple[str, str], ...] = (
    ("dot2.py", "dot2 — 2-vector dot product"),
    ("madd.py", "madd — subtract, power-of-2 scale, fused multiply-add"),
    ("cube.py", "cube — a**3 expands to a multiply chain"),
    ("poly3.py", "poly3 — Horner-form cubic polynomial"),
    ("blend.py", "blend — division + scaled add"),
    ("ekf_update.py", "ekf_update — EKF covariance + state update (advanced)"),
)


def load_demos() -> list[Demo]:
    """Return the bundled demo kernels in display order, reading each module's source text from the package."""
    root = resources.files(__package__)
    return [
        Demo(id=filename.removesuffix(".py"), label=label, source=root.joinpath(filename).read_text(encoding="utf-8"))
        for filename, label in _MANIFEST
    ]
