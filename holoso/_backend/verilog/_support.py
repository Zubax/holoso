"""
Assemble the shared support library that every Holoso-generated module instantiates.
"""

import logging
from functools import cache
from importlib import resources

import zkf

from ..._legal import output_header

_logger = logging.getLogger(__name__)

_TEMPLATE_FILES = ["holoso_int.v", "holoso_float.v"]
_INLINE_FILE = "holoso_support_inline.vh"
_MEGAFILE = "holoso_support.v"
_SEPARATOR = "// " + "=" * 117


def _megafile_header() -> str:
    lines = output_header("// ").splitlines() + [
        "//",
        "// A Holoso-synthesized design needs only this file. It serves every Holoso-generated module in a design.",
    ]
    return "\n".join(lines)


def _build_megafile() -> str:
    # Public modules sort before internal ones for readability of the assembled file.
    modules = sorted(zkf.get_rtl().items(), key=lambda rc: (rc[0].rsplit("/", 1)[-1].startswith("_"), rc[0]))
    assert modules, "no .v files in the ZKF RTL package"
    package = resources.files(__package__)
    blocks = [
        _megafile_header(),
        *(package.joinpath(name).read_text(encoding="utf-8").strip() for name in _TEMPLATE_FILES),
    ]
    for rel, content in modules:
        blocks.append(f"{_SEPARATOR}\n// EMBEDDED FILE BEGIN: {rel}")
        blocks.append(content.strip())
        blocks.append(f"// EMBEDDED FILE END: {rel}")
    return "\n\n".join(blocks) + "\n"


@cache
def inline_support() -> str:
    """The helper functions for the emitter to splice into each generated module."""
    return resources.files(__package__).joinpath(_INLINE_FILE).read_text(encoding="utf-8").strip()


@cache
def support_files() -> dict[str, str]:
    """The shipped support library ``{filename: content}`` -- just the module library."""
    files = {_MEGAFILE: _build_megafile()}
    _logger.info("Assembled support library: %s", ", ".join(f"{n} ({len(t.encode())} B)" for n, t in files.items()))
    return files
