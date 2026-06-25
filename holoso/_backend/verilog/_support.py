"""
Assemble the shared support library that every Holoso-generated module instantiates.

It ships as two files: the single self-contained module library ``holoso_support.v`` and the function header
``holoso_support.vh`` that generated modules ``include``. Both live under ``rtl/`` together with every primitive they
bundle: the hand-written wrappers (``holoso_support_template.v``) and the header sit at the top, while each vendored
source occupies its own subdirectory (refreshed by ``tools/update_support_rtl.py``), alongside any locally-maintained
subdirectories. The module library is assembled in memory and is invariant to the generated module, so one file serves
a whole design.
"""

import logging
from functools import cache
from importlib import resources
from importlib.resources.abc import Traversable

_logger = logging.getLogger(__name__)

_TEMPLATE_FILE = "holoso_support_template.v"
_HEADER_FILE = "holoso_support.vh"
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


def _megafile_header(rtl: Traversable) -> str:
    lines = [
        _SEPARATOR,
        "// HOLOSO SUPPORT LIBRARY -- AUTO-GENERATED, DO NOT EDIT.",
        "//",
        f"// A Holoso-synthesized design needs only this file plus {_HEADER_FILE}.",
        "// The same set of support files serves every Holoso-generated module in a design.",
    ]
    for name in sorted(child.name for child in rtl.iterdir() if child.is_dir()):
        readme = rtl.joinpath(name).joinpath(_MANIFEST)
        if readme.is_file():
            lines += ["//", _SEPARATOR, "//"]
            lines += [f"// {line}".rstrip() for line in readme.read_text(encoding="utf-8").strip().splitlines()]
    lines += ["//", _SEPARATOR]
    return "\n".join(lines)


def _build_megafile(pkg: Traversable) -> str:
    rtl = pkg.joinpath(_MEGAFILE_COMPONENT_SOURCE_DIR)
    # The hand-written wrappers lead; exclude the template from the glob so it is not also embedded as a module.
    modules = [(rel, text) for rel, text in _iter_rtl(rtl) if rel != _TEMPLATE_FILE]
    assert modules, "no .v files found under rtl/"
    modules.sort(key=lambda rc: (rc[0].rsplit("/", 1)[-1].startswith("_"), rc[0]))
    blocks = [_megafile_header(rtl), rtl.joinpath(_TEMPLATE_FILE).read_text(encoding="utf-8").strip()]
    for rel, content in modules:
        blocks.append(f"{_SEPARATOR}\n// EMBEDDED FILE BEGIN: {rel}")
        blocks.append(content.strip())
        blocks.append(f"// EMBEDDED FILE END: {rel}")
    return "\n\n".join(blocks) + "\n"


@cache
def support_files() -> dict[str, str]:
    """The shared support library that generated modules instantiate ``{filename: content}``; invariant, so cached."""
    pkg = resources.files(__package__)
    files = {
        _MEGAFILE: _build_megafile(pkg),
        _HEADER_FILE: pkg.joinpath(_MEGAFILE_COMPONENT_SOURCE_DIR).joinpath(_HEADER_FILE).read_text(encoding="utf-8"),
    }
    _logger.info("Assembled support library: %s", ", ".join(f"{n} ({len(t.encode())} B)" for n, t in files.items()))
    return files
