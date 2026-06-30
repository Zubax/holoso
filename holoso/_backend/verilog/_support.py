"""
Assemble the shared support library that every Holoso-generated module instantiates.
"""

import logging
from functools import cache
from importlib import resources
from importlib.resources.abc import Traversable
from ..._legal import output_header

_logger = logging.getLogger(__name__)

_TEMPLATE_FILE = "holoso_support_template.v"
_INLINE_FILE = "holoso_support_inline.vh"
_MEGAFILE = "holoso_support.v"
_MEGAFILE_COMPONENT_SOURCE_DIR = "rtl"
_MANIFEST = "README.md"
_SEPARATOR = "// " + "=" * 117


def _iter_rtl(node: Traversable, prefix: str = "") -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for child in node.iterdir():
        rel = f"{prefix}{child.name}"
        if child.is_dir():
            out += _iter_rtl(child, f"{rel}/")
        elif child.name.endswith(".v"):
            out.append((rel, child.read_text(encoding="utf-8")))
    return out


def _megafile_header() -> str:
    lines = output_header("// ").splitlines() + [
        "//",
        "// A Holoso-synthesized design needs only this file. It serves every Holoso-generated module in a design.",
    ]
    return "\n".join(lines)


def _build_megafile(pkg: Traversable) -> str:
    rtl = pkg.joinpath(_MEGAFILE_COMPONENT_SOURCE_DIR)
    # The hand-written wrappers lead; exclude the template from the glob so it is not also embedded as a module.
    modules = [(rel, text) for rel, text in _iter_rtl(rtl) if rel != _TEMPLATE_FILE]
    assert modules, "no .v files found under rtl/"
    modules.sort(key=lambda rc: (rc[0].rsplit("/", 1)[-1].startswith("_"), rc[0]))
    blocks = [_megafile_header(), rtl.joinpath(_TEMPLATE_FILE).read_text(encoding="utf-8").strip()]
    for rel, content in modules:
        blocks.append(f"{_SEPARATOR}\n// EMBEDDED FILE BEGIN: {rel}")
        blocks.append(content.strip())
        blocks.append(f"// EMBEDDED FILE END: {rel}")
    return "\n\n".join(blocks) + "\n"


@cache
def inline_support() -> str:
    """The helper functions for the emitter to splice into each generated module."""
    pkg = resources.files(__package__)
    return pkg.joinpath(_MEGAFILE_COMPONENT_SOURCE_DIR).joinpath(_INLINE_FILE).read_text(encoding="utf-8").strip()


@cache
def support_files() -> dict[str, str]:
    """The shipped support library ``{filename: content}`` -- just the module library."""
    files = {_MEGAFILE: _build_megafile(resources.files(__package__))}
    _logger.info("Assembled support library: %s", ", ".join(f"{n} ({len(t.encode())} B)" for n, t in files.items()))
    return files
